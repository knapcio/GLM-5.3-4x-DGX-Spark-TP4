# SPDX-License-Identifier: Apache-2.0
"""Short-context DSA shortcut for full GLM-5.3. Inert unless GLM_INDEXER_SHORTCUT=1.

For decode/verify rows whose context is at most 2048 tokens, the pinned
persistent_topk takes its all-selected branch: it returns 0..L-1 in order and
never reads the indexer logits. This module skips the indexer query GEMM,
query RoPE/quantisation and the paged MQA logits for exactly those rows and
keeps every stock key/cache operation. No new GPU kernel.

Eligibility is decided on the CPU before graph dispatch (native MTP K2:
target verify M3/M12, draft passes M1/M4, one or four requests); eligible
batches replay separately captured FULL graphs. Everything else, including
prefill, mixed widths and contexts above 2048, uses the stock path.
Register before vLLM imports; every transformed module is source-pinned.
"""
import contextlib
from contextvars import ContextVar
import dataclasses
import functools
import hashlib
import importlib.abc
import importlib.util
import os
import sys

MODEL = 'vllm.model_executor.models.deepseek_v2'
SPARSE = 'vllm.model_executor.layers.sparse_attn_indexer'
AUTO = 'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator'
CG = 'vllm.v1.worker.gpu.cudagraph_utils'
RUN = 'vllm.v1.worker.gpu.model_runner'
PINS = {
    MODEL: '58d8916458de7c6f73b40bfef9d2f57bdd6fa0fb79be9e269331af6e66149fe2',
    SPARSE: '22d1d98bd475b1dc22de85d0e414d0490860de50c8bbd26371175e5ea6411116',
    AUTO: '575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1',
    CG: 'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f',
    RUN: 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415',
}
ENABLED = False
SHORT = ContextVar('full_glm_dsa_short', default=False)
COUNTS = dict(eligible=0, fallback=0, short_dispatch=0, stock_dispatch=0)
CAPTURE_HITS = {}


@contextlib.contextmanager
def context(value):
    token = SHORT.set(bool(value))
    try:
        yield
    finally:
        SHORT.reset(token)


def active():
    return SHORT.get() and ENABLED


def selected(desc):
    # Also covers need_eager/profile, which bypass manager.dispatch entirely.
    SHORT.set(bool(getattr(desc, 'short_context', False)))


def capture_hit(stage, rows):
    # Python runs during eager/warmup/capture, not FULL graph replay. These
    # prove source reachability; dispatch counts prove execution.
    key = f'{stage}:M{rows}'
    CAPTURE_HITS[key] = CAPTURE_HITS.get(key, 0) + 1


def supported(config, parallel=None):
    # This implementation is the non-pooled full target, not GLM5next kpool.
    ok = (getattr(config, 'architectures', None) == ['GlmMoeDsaForCausalLM']
          and getattr(config, 'index_topk', None) == 2048
          and getattr(config, 'index_head_dim', None) == 128
          and getattr(config, 'index_n_heads', None) == 32
          and getattr(config, 'qk_rope_head_dim', None) == 64
          and getattr(config, 'index_kpool', 1) == 1)
    return ok and (parallel is None or (
        parallel.tensor_parallel_size == 4
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1))


def eligible(schedule, batch, computed, uniform, supported_model=True):
    """Same pre-step CPU upper bounds used by prepare_inputs, no GPU sync.

    Actual GPU rejection can shorten a verifier batch. Scheduled P+4 is still
    an upper bound on all causal rows. Never substitute positions for the
    native metadata lengths. Reject padding, prefill, resume and mixed widths.
    """
    if not supported_model or batch is None or uniform != 3 or batch.has_prefill:
        return False
    ns = schedule.num_scheduled_tokens
    if len(ns) not in (1, 4) or any(n != 3 for n in ns.values()):
        return False
    if (schedule.scheduled_new_reqs
            or getattr(schedule, 'has_structured_output_requests', False)
            or getattr(schedule, 'scheduled_encoder_inputs', {})
            or getattr(schedule.scheduled_cached_reqs, 'resumed_req_ids', set())):
        return False
    if (set(batch.req_ids) != set(ns) or len(batch.req_ids) != len(ns)
            or batch.num_tokens != 3 * len(ns)
            or schedule.total_num_scheduled_tokens != batch.num_tokens
            or any(int(n) != 3 for n in batch.num_scheduled_tokens)):
        return False
    if any(len(schedule.scheduled_spec_decode_tokens.get(r, ())) != 2 for r in ns):
        return False
    return (len(computed) == len(ns)
            and all(0 < int(p) and int(p) + 3 <= 2048 for p in computed))


def draft_eligible(lengths, num_reqs, num_tokens, uniform, step, has_prefill=False, dummy=False):
    # Same native CPU upper bounds consumed by _build_draft_attn_metadata;
    # no positions/rejection guesses, and no device-to-host synchronization.
    return (not dummy and not has_prefill and num_reqs in (1,4)
            and uniform in (1,3) and num_tokens == num_reqs*uniform
            and step in (0,1) and len(lengths)==num_reqs
            and all(0 < int(n) and int(n)+step <= 2048 for n in lengths))

def select_draft(lengths, num_reqs, num_tokens, uniform, step, **kwargs):
    SHORT.set(draft_eligible(lengths,num_reqs,num_tokens,uniform,step,**kwargs))

def transform_auto(source):
    source=once(source, '        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(\n',
        '        from glm_dsa_short import select_draft as _dsa_select_draft\n'
        '        _dsa_select_draft(input_batch.seq_lens_cpu_upper_bound[:num_reqs].tolist(),\n'
        '            num_reqs, num_tokens_padded, uniform_token_count, 0,\n'
        '            has_prefill=input_batch.has_prefill, dummy=dummy_run or is_profile)\n'
        '        prefill_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(\n')
    source=once(source, '        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(\n',
        '        _dsa_select_draft(input_batch.seq_lens_cpu_upper_bound[:num_reqs].tolist(),\n'
        '            num_reqs, num_reqs, 1, self.num_speculative_steps-1,\n'
        '            has_prefill=input_batch.has_prefill, dummy=dummy_run or is_profile)\n'
        '        decode_batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(\n')
    return source

def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('DSA source anchor drift: ' + old[:90])
    return source.replace(old, new)


def transform_model(source):
    if 'from glm_dsa_short import' in source:
        raise ValueError('DSA model already transformed')
    # Select the exact Indexer.forward anchor, not main attention's query.
    anchor = '''    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
'''
    return once(source, anchor, anchor + '''        from glm_dsa_short import active as _dsa_short_active
        if _dsa_short_active():
            if (not current_platform.is_cuda() or not self.use_fused_indexer_q
                    or hidden_states.dtype != torch.bfloat16 or qr.dtype != torch.bfloat16):
                raise RuntimeError("short DSA requires stock BF16 fused CUDA query path")
            from glm_dsa_short import capture_hit as _dsa_capture_hit
            _dsa_capture_hit("query-bypass", hidden_states.shape[0])
            # Exact stock fused-path K operations, in the same order. Keep the
            # whole wk_weights GEMM: slicing its weights would change arithmetic.
            kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]
            k = self.k_norm(k)
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe = k_pe.unsqueeze(1)
            q_dummy = torch.empty_like(k_pe)
            _, k_pe = rotary_emb(positions, q_dummy, k_pe)
            k_pe = k_pe.reshape(-1, self.rope_dim)
            k = torch.cat([k_pe, k_nope], dim=-1)
            q_fp8 = torch.empty((hidden_states.shape[0], self.n_head, self.head_dim),
                               device=hidden_states.device, dtype=torch.float8_e4m3fn)
            return self.indexer_op(hidden_states, q_fp8, k, weights)
''')


def transform_sparse(source):
    if 'from glm_dsa_short import' in source:
        raise ValueError('DSA sparse op already transformed')
    anchor = '    slot_mapping = attn_metadata_narrowed.slot_mapping\n'
    source = once(source, anchor, '''    from glm_dsa_short import active as _dsa_short_active
    _dsa_short = _dsa_short_active()
    if _dsa_short and (
        forward_context.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
        or attn_metadata_narrowed.num_prefills != 0
        or attn_metadata_narrowed.num_decodes == 0
        or attn_metadata_narrowed.decode.requires_padding
        or use_fp4_cache or use_pcp or dcp_world_size != 1
        or topk_tokens != 2048 or head_dim != 128
    ):
        raise RuntimeError("short DSA received unsupported metadata")
''' + anchor)
    anchor = '''        else:
            logits = fp8_fp4_paged_mqa_logits(
'''
    return once(source, anchor, '''        elif _dsa_short:
            from glm_dsa_short import capture_hit as _dsa_capture_hit
            _dsa_capture_hit("logits-bypass", num_padded_tokens)
            # Same shape/strides/scalar bound as stock DeepGEMM output.
            # persistent_topk never dereferences these bytes for admitted rows.
            logits = torch.empty((num_padded_tokens, max_model_len),
                                 device=hidden_states.device, dtype=torch.float32)
        else:
            logits = fp8_fp4_paged_mqa_logits(
''')


def transform_cg(source):
    if 'short_context: bool' in source:
        raise ValueError('DSA descriptor already transformed')
    return once(source, '    num_active_loras: int = 0\n',
                '    num_active_loras: int = 0\n    short_context: bool = False\n')


def transform_runner(source):
    if 'from glm_dsa_short import' in source:
        raise ValueError('DSA runner already transformed')
    return once(source, '        if batch_desc.num_tokens == 0:\n',
                '        from glm_dsa_short import selected as _dsa_short_selected\n'
                '        _dsa_short_selected(batch_desc)\n'
                '        if batch_desc.num_tokens == 0:\n')


def install_runner(mod):
    cls = mod.GPUModelRunner
    load, execute, gather = cls.load_model, cls.execute_model, cls.gather_batch_req_state

    @functools.wraps(load)
    def loaded(self, *a, **kw):
        sp = self.speculative_config
        from vllm.config.compilation import CUDAGraphMode
        if (sp is None or sp.method != 'mtp' or self.num_speculative_steps != 2
                or self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or self.lora_config is not None):
            raise RuntimeError('GLM_INDEXER_SHORTCUT requires native MTP K2, FULL_DECODE_ONLY, no LoRA')
        result = load(self, *a, **kw)
        self._dsa_short_supported = supported(self.model_config.hf_config, self.parallel_config)
        if not self._dsa_short_supported:
            raise RuntimeError('GLM_INDEXER_SHORTCUT requires full GLM non-pooled TP4/PCP1/DCP1')
        sys.stderr.write('glm-dsa-short: armed (native MTP K2, context <= 2048)\n')
        return result

    @functools.wraps(gather)
    def gathered(self, schedule, dummy_run):
        batch, uniform = gather(self, schedule, dummy_run)
        # Runs after update_requests and before graph dispatch. CPU state here
        # is exactly the state used in prepare_inputs' seq_lens upper bounds.
        if not dummy_run:
            computed = [self.req_states.num_computed_tokens_np[int(i)]
                        for i in batch.idx_mapping_np] if batch is not None else []
            ok = eligible(schedule, batch, computed, uniform,
                          getattr(self, '_dsa_short_supported', False))
            SHORT.set(ok)
            COUNTS['eligible' if ok else 'fallback'] += 1
        return batch, uniform

    @functools.wraps(execute)
    def executed(self, *a, **kw):
        # Reset eligibility across requests, dummy/profile, exceptions and draft.
        with context(False):
            return execute(self, *a, **kw)

    cls.load_model, cls.gather_batch_req_state, cls.execute_model = loaded, gathered, executed


def install_cg(mod):
    cls = mod.CudaGraphManager
    init, capture, dispatch = cls.__init__, cls.capture, cls.dispatch

    @functools.wraps(init)
    def initialized(self, *a, **kw):
        init(self, *a, **kw)
        if isinstance(self, mod.CudaGraphManager):
            self._capture_descs = {mode: [d for old in descs for d in (
                (old, dataclasses.replace(old, short_context=True))
                if old.uniform_token_count in (1,3)
                and old.num_tokens == old.uniform_token_count * old.num_reqs
                and old.num_reqs in (1, 4) and old.num_active_loras == 0
                else (old,))] for mode, descs in self._capture_descs.items()}

    @functools.wraps(capture)
    def captured(self, factory, *a, **kw):
        def create(desc, warmup):
            fn = factory(desc, warmup)
            def run(mode):
                with context(desc.short_context):
                    return fn(mode)
            return run
        return capture(self, create, *a, **kw)

    @functools.wraps(dispatch)
    def dispatched(self, num_reqs, num_tokens, uniform_token_count,
                   num_active_loras, max_query_len=None):
        desc = dispatch(self, num_reqs, num_tokens, uniform_token_count,
                        num_active_loras, max_query_len)
        if (isinstance(self, mod.CudaGraphManager) and active()
                and desc.cg_mode == mod.CUDAGraphMode.FULL
                and uniform_token_count in (1,3) and num_active_loras == 0
                and desc.num_tokens == num_tokens == uniform_token_count * num_reqs
                and desc.num_reqs == num_reqs and num_reqs in (1, 4)
                and desc.uniform_token_count == uniform_token_count):
            desc = dataclasses.replace(desc, short_context=True)
        if (desc.cg_mode == mod.CUDAGraphMode.FULL and self._graphs_captured
                and desc not in self.graphs):
            raise RuntimeError('short DSA descriptor missing from captured graphs')
        if isinstance(self, mod.ModelCudaGraphManager):
            key = 'short_dispatch' if desc.short_context else 'stock_dispatch'
            COUNTS[key] += 1
            if COUNTS[key] == 1 and self._graphs_captured:
                sys.stderr.write('glm-dsa-short: first ' + key + ' ' + repr(
                    (desc.cg_mode.name, desc.num_tokens, desc.num_reqs)) + '\n')
        return desc

    cls.__init__, cls.capture, cls.dispatch = initialized, captured, dispatched


def capture_summary(mod):
    # One stderr line per graph manager after capture: short graphs present.
    cls = mod.CudaGraphManager
    capture = cls.capture

    @functools.wraps(capture)
    def captured(self, *a, **kw):
        result = capture(self, *a, **kw)
        if self._graphs_captured and self.graphs:
            short = sorted((d.num_tokens, d.num_reqs) for d in self.graphs if d.short_context)
            sys.stderr.write('glm-dsa-short: ' + type(self).__name__ + ' captured '
                             + str(len(self.graphs)) + ' FULL graphs, short ' + repr(short) + '\n')
        return result

    cls.capture = captured


TRANSFORMS = {MODEL: transform_model, SPARSE: transform_sparse, AUTO: transform_auto,
              CG: transform_cg, RUN: transform_runner}


def _install_cg(mod):
    install_cg(mod)
    capture_summary(mod)


INSTALL = {CG: _install_cg, RUN: install_runner}


def transform(name, source):
    got = hashlib.sha256(source.encode()).hexdigest()
    if got != PINS[name]:
        raise RuntimeError('short DSA source drift: ' + name + ' ' + got)
    return TRANSFORMS[name](source)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.done = set()

    def find_spec(self, name, path=None, target=None):
        if name not in TRANSFORMS or name in self.done:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or not hasattr(spec.loader, 'get_source'):
            raise RuntimeError('short DSA needs a source loader: ' + name)
        code = compile(transform(name, spec.loader.get_source(name)), spec.origin, 'exec',
                       dont_inherit=True)
        spec.loader.get_code = lambda fullname: code
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            if name in INSTALL:
                INSTALL[name](module)
            self.done.add(name)

        spec.loader.exec_module = execute
        return spec


def flag(env):
    value = env.get('GLM_INDEXER_SHORTCUT', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_INDEXER_SHORTCUT must be 0 or 1')
    return value == '1'


def register(env=None):
    global ENABLED
    env = os.environ if env is None else env
    if not flag(env):
        return False
    loaded = [name for name in TRANSFORMS if name in sys.modules]
    if loaded:
        raise RuntimeError('GLM_INDEXER_SHORTCUT must register before vLLM imports: ' + repr(loaded))
    ENABLED = True
    if not any(isinstance(h, Finder) for h in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    return True
