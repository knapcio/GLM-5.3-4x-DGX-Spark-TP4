# SPDX-License-Identifier: Apache-2.0
"""Fixed-budget DSA prefill with the stock scorer and exact stock selector.

A single 256 MiB returned logits allocation has constant bytes at every
prefix. Query/key bucket dimensions trade places; causal bounds skip padded
keys. Constructor banks are shared by target/MTP on one worker stream.
"""
import ast
import functools
import os
from pathlib import Path

Q_TILE = 2048
K_TILE = 4096  # reader/interpreter fixture geometry
LOGITS_BYTES = 256 << 20
TOPK = 2048


def capacities(pool_tokens=1573 * 64, max_model_len=100288):
    if pool_tokens < 1573 * 64 or pool_tokens % 64 or not 64 <= max_model_len <= pool_tokens - 384:
        raise ValueError('FP4x prefill capacity must match a packed pool and its six-block margin')
    keys = max(524288, 1 << (4 * max_model_len - 1).bit_length())
    return keys, pool_tokens, max(keys * 132, pool_tokens * 576 * 2)


K_CAPACITY, MLA_ROWS, WORKSPACE_BYTES = capacities(
    int(os.environ.get('GLM_FP4_POOL_BLOCKS', '1573')) * 64,
    int(os.environ.get('GLM_FP4_MAX_MODEL_LEN', '100288')))
_BANKS = {}


def geometry(prefix):
    if not 0 < prefix <= K_CAPACITY:
        raise RuntimeError('FP4x prefill exceeds fixed four-slot key capacity')
    keys = max(32768, 1 << (prefix - 1).bit_length())
    return LOGITS_BYTES // (4 * keys), keys


def scratch_bytes(heads=64, pool_tokens=None, max_model_len=None):
    _, rows, workspace = capacities(pool_tokens if pool_tokens is not None else MLA_ROWS,
                                    max_model_len if max_model_len is not None else
                                    int(os.environ.get('GLM_FP4_MAX_MODEL_LEN', '100288')))
    return (workspace + rows * 4 + Q_TILE * (heads * 132 + 8)
            + 4096 * 512 * 2 + LOGITS_BYTES)


def bank_bytes(bank):
    # Keys/scales are views into the attention expansion arena. Count storage
    # once, rather than charging aliases as independent allocations.
    return sum(s.nbytes() for s in {
        v.untyped_storage().data_ptr(): v.untyped_storage()
        for v in bank.values()}.values())


def reserve(device, heads):
    import torch
    key = str(device)
    if key in _BANKS:
        if _BANKS[key]['q'].shape[1] != heads:
            raise RuntimeError('FP4x indexer head count changed')
        return _BANKS[key]
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)
    workspace = empty((WORKSPACE_BYTES,), torch.uint8)
    bank = dict(workspace=workspace,
                selected=empty((MLA_ROWS,), torch.int32),
                q=empty((Q_TILE, heads, 128), torch.float8_e4m3fn),
                weights=empty((Q_TILE, heads), torch.float32),
                keys=workspace[:K_CAPACITY * 128].view(torch.float8_e4m3fn).view(K_CAPACITY, 128),
                scales=workspace[K_CAPACITY * 128:K_CAPACITY * 132].view(torch.float32),
                starts=empty((Q_TILE,), torch.int32), ends=empty((Q_TILE,), torch.int32),
                fresh=empty((4096, 512), torch.bfloat16))
    _BANKS[key] = bank
    return bank


def bank_for(device):
    try:
        return _BANKS[str(device)]
    except KeyError:
        raise RuntimeError('FP4x prefill bank missing: model must reserve at boot') from None


def qdq(x, recent=False):
    from glm_fp4_kv_kernel import qdq_prefill
    bank = bank_for(x.device)
    if x.shape[0] > bank['fresh'].shape[0]:
        raise RuntimeError('FP4x fresh prefill exceeds constructor chunk capacity')
    return qdq_prefill(x, bank['fresh'][:x.shape[0]], recent=recent)


def tiled_prefill(cache, q, q_scale, weights, chunks, out, logits_fn, topk_fn=None):
    import torch
    from glm_fp4_prefill_kernel import gather_index_tile, prepare_index_queries
    if topk_fn is None:
        from vllm import _custom_ops as ops
        topk_fn = ops.top_k_per_row_prefill
    if q_scale is not None or cache.shape[-1] != 132 or cache.dtype != torch.uint8:
        raise RuntimeError('FP4x prefill requires the stock DSA E4M3/FP32 ABI')
    b = bank_for(q.device)
    if (q.shape[1:] != b['q'].shape[1:] or q.dtype != b['q'].dtype
            or not q.is_contiguous() or not weights.is_contiguous()
            or weights.dtype != torch.float32 or weights.shape != q.shape[:2]):
        raise RuntimeError('FP4x prefill query ABI drift')
    for chunk in chunks:
        if not chunk.local_total_seq_lens:
            out[chunk.token_start:chunk.token_end].fill_(-1)
            continue
        qt, kt = geometry(chunk.local_total_seq_lens)
        keys, scales = b['keys'][:kt], b['scales'][:kt]
        # Never reuse a prior chunk based on skip_kv_gather: its request slice
        # may differ. One bounded gather also initializes the padded K tail.
        gather_index_tile(cache, chunk.block_table, chunk.local_cu_seq_lens,
                          keys, scales, 0)
        for qoff in range(0, chunk.token_end - chunk.token_start, qt):
            begin = chunk.token_start + qoff
            rows = min(qt, chunk.token_end - begin)
            qs, ws = b['q'][:qt], b['weights'][:qt]
            ks, ke = b['starts'][:qt], b['ends'][:qt]
            prepare_index_queries(q, weights, chunk.cu_seqlen_ks,
                                  chunk.cu_seqlen_ke, qs, ws, ks, ke,
                                  begin, qoff, rows)
            # clean_logits=False is stock: the stock selector reads only
            # [ks,ke). Padded queries have empty intervals and zero weights.
            logits = logits_fn((qs, None), (keys, scales), ws, ks, ke,
                               clean_logits=False)
            if logits.numel() * logits.element_size() != LOGITS_BYTES:
                raise RuntimeError('FP4x DeepGEMM fixed logits ABI drift')
            topk_fn(logits, ks, ke, out[begin:begin + rows], rows,
                    logits.stride(0), logits.stride(1), TOPK)
            # Same allocation size on every iteration/layer/forward. Release
            # before the next scoring call; the worker stream orders reuse.
            del logits


def transform_indexer(source):
    """Keep stock insert, short bypass, mixed decode and metadata contracts."""
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sparse_attn_indexer')
    lines = source.splitlines(keepends=True)
    raw = ''.join(lines[fn.lineno-1:fn.end_lineno])
    raw = raw.replace('max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024',
                      'max_logits_elems = 256 * 1024 * 1024')
    profile_start = raw.index('        # Reserve workspace for indexer during profiling run')
    profile_end = raw.index('        # Dummy allocation', profile_start)
    raw = raw[:profile_start] + raw[profile_end:]
    start = raw.index('    if has_prefill:\n')
    end = raw.index('    if has_decode:\n', start)
    return raw[:start] + '''    if has_prefill:
        if use_fp4_cache or use_pcp or dcp_world_size != 1 or head_dim != 128 or topk_tokens != 2048:
            raise RuntimeError("FP4x tiled prefill requires release DSA/DCP1")
        from glm_fp4_prefill import tiled_prefill
        tiled_prefill(kv_cache, q_quant, q_scale, weights,
                      attn_metadata_narrowed.prefill.chunks,
                      topk_indices_buffer, fp8_fp4_mqa_logits, ops.top_k_per_row_prefill)

''' + raw[end:]


def install_indexer(mod):
    import torch
    ns = {}
    exec(compile(transform_indexer(Path(mod.__file__).read_text()), mod.__file__, 'exec'), mod.__dict__, ns)
    tiled = ns['sparse_attn_indexer']
    cls = mod.SparseAttnIndexer
    original_init, original_forward = cls.__init__, cls.forward_cuda

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Fixed four-slot capacity; profiling uses these same constructor banks.
        self.max_total_seq_len = K_CAPACITY
        config = mod.get_current_vllm_config()
        reserve(self.topk_indices_buffer.device, config.model_config.hf_config.index_n_heads)

    @functools.wraps(original_forward)
    def forward(self, hidden_states, q_quant, k, weights):
        context = mod.get_forward_context()
        md = context.attn_metadata
        layer_md = md.get(self.k_cache.prefix) if isinstance(md, dict) else None
        if isinstance(md, dict) and (layer_md is None or layer_md.num_prefills == 0):
            return original_forward(self, hidden_states, q_quant, k, weights)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('FP4x tiled prefill must be eager')
        q, scale = q_quant if isinstance(q_quant, tuple) else (q_quant, None)
        return tiled(hidden_states, self.k_cache.prefix, self.k_cache.kv_cache,
                     q, scale, k, weights, self.quant_block_size, self.scale_fmt,
                     self.topk_tokens, self.head_dim, self.max_model_len,
                     self.max_total_seq_len, self.topk_indices_buffer,
                     self.skip_k_cache_insert, self.use_pcp, self.dense_mha_metadata_layer_name,
                     self.use_fp4_cache, self.dcp_rank, self.dcp_world_size,
                     self.cp_kv_cache_interleave_size)

    cls.__init__, cls.forward_cuda = init, forward


def transform_metadata(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prefill_chunk_metadata')
    raw = ''.join(source.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    old = 'token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)'
    if raw.count(old) != 1 or raw.count('_BUILD_PREFILL_CHUNK_METADATA_KERNEL(') != 1:
        raise RuntimeError('FP4x indexer metadata source anchor drift')
    raw = raw.replace(old, 'token_to_seq = torch.empty(0, dtype=torch.int32, device=device)')
    return raw.replace('_BUILD_PREFILL_CHUNK_METADATA_KERNEL(', '_glm_fp4x_build_bounds(')


def install_metadata(mod):
    def bounds(*args, **kwargs):
        from glm_fp4_prefill_kernel import index_prefill_bounds
        return index_prefill_bounds(*args, **kwargs)
    mod.__dict__['_glm_fp4x_build_bounds'] = bounds
    ns = {}
    exec(compile(transform_metadata(Path(mod.__file__).read_text()), mod.__file__, 'exec'), mod.__dict__, ns)
    mod.build_prefill_chunk_metadata = ns['build_prefill_chunk_metadata']


def install_sparse_metadata(mod):
    cls = mod.SparseMLACommonMetadataBuilder
    original = cls._build_chunked_context_fields

    @functools.wraps(original)
    def fields(self, common_attn_metadata, num_decodes, num_prefills, prefill_query_lens_cpu):
        # Quantized release KV disables masked MHA. Above index_topk the
        # central layer sends the entire batch through sparse MQA, so context
        # gather metadata (including a prefix-sized token map) is dead data.
        if num_prefills and prefill_query_lens_cpu is not None and self.topk_mask_workspace is None:
            seq = common_attn_metadata.seq_lens_cpu_upper_bound[num_decodes:num_decodes + num_prefills]
            if int(seq.max()) > self.model_config.hf_config.index_topk:
                return None
        return original(self, common_attn_metadata, num_decodes, num_prefills, prefill_query_lens_cpu)

    cls._build_chunked_context_fields = fields
