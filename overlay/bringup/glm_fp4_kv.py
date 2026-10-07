# SPDX-License-Identifier: Apache-2.0
"""Boot-fixed, source-pinned FP4x KV overlay; fp8 installs no hooks.

Only MLA storage changes. Global vLLM cache dtype remains fp8_e4m3 for
DSA index keys, whose separate E4M3+FP32-scale ABI is unchanged.
"""
import ast
from dataclasses import replace
import functools
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
import textwrap

ABI = 'fp4x_v1_e2m1_s16_e4m3_pow2_r64'
COMMON = 'vllm.model_executor.layers.attention.mla_attention'
BACKEND = 'vllm.v1.attention.backend'
SPARSE = 'vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90'
SPEC = 'vllm.v1.kv_cache_interface'
PINS = {
    COMMON: 'a309ca752b01e85360b268fba56245b816ff245340e9e361166ee96b99d93a94',
    BACKEND: '301c76c90d5f26cdecfedfb385f8f17453154106d2decf763a31cb978f1a5d99',
    SPARSE: '4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136',
    SPEC: '54a761dd60907945c8f3bfc450b1264033013b83b204320581961f887b03e5b0',
    'vllm.model_executor.layers.attention.sparse_mla_attention': '0e784684bfe73bfc57ecc75464ba98c0163572e3961ca9a40c31df7480a1d0a9',
    'vllm.model_executor.models.deepseek_v2': '58d8916458de7c6f73b40bfef9d2f57bdd6fa0fb79be9e269331af6e66149fe2',
    'vllm.model_executor.layers.sparse_attn_indexer': '22d1d98bd475b1dc22de85d0e414d0490860de50c8bbd26371175e5ea6411116',
    'vllm.v1.attention.backends.mla.indexer': 'fe1b106466008c21d0194a59909b19832e127fad255965611c4c7f7a971422a2',
    'vllm.v1.worker.gpu.attn_utils': '1dd3dd2826a2cc73005e7baecb71c26de8d56285b35d780716ab11ffe0f8495b',
}
_FORMAT = None


def kv_format(env):
    mode = env.get('GLM_KV_FORMAT', 'fp8')
    if mode not in ('fp8', 'fp4x'):
        raise ValueError('GLM_KV_FORMAT must be fp8 or fp4x (boot-time only)')
    if mode == 'fp4x':
        if env.get('GLM_FULL_MLA') != 'triton' or env.get('VLLM_USE_V2_MODEL_RUNNER') != '1':
            raise ValueError('FP4x requires Triton MLA and the V2 runner')
        if env.get('GLM_MLA_SPLIT_K') != '32' or env.get('GLM_MLA_SPLIT_MAX_ROWS', '36') != '36':
            raise ValueError('FP4x release requires split32 and the 36-row dispatch cap')
        for key in ('GLM_FP4_PROBE_SIM', 'GLM_FP4_PROBE_CAPTURE', 'GLM_DSA_SWA_POOL',
                    'GLM_GLUE_DSA_IDX_CACHE', 'GLM_SPEC_SAMPLE', 'GLM_TRITON_MLA_PREFILL'):
            if env.get(key, '0') not in ('', '0'):
                raise ValueError('FP4x does not compose with ' + key)
    return mode


def active():
    return _FORMAT == 'fp4x'


def check_source(mod):
    got = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    if got != PINS[mod.__name__]:
        raise RuntimeError('FP4x source drift: ' + mod.__name__ + ': ' + got)


def transform_forward(raw):
    """Compile only the pinned method, leaving all other module hooks intact."""
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLAAttention')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_impl')
    lines = raw.splitlines(keepends=True)
    source = textwrap.dedent(''.join(lines[method.lineno-1:method.end_lineno]))
    old = 'if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":'
    if source.count(old) != 1:
        raise RuntimeError('FP4x central reinterpretation anchor drift')
    source = source.replace(old, 'if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla" and not self._glm_fp4x:')
    anchor = 'k_c_normed[num_mqa_tokens:],'
    if source.count(anchor) != 1:
        raise RuntimeError('FP4x fresh MHA anchor drift')
    return source.replace(anchor, '_glm_fp4x_qdq(k_c_normed[num_mqa_tokens:], recent=getattr(self, "_glm_recent_target", False)),')


def install_spec(mod):
    cls = mod.MLAAttentionSpec
    stock = cls.real_page_size_bytes.fget

    def page(self):
        if self.cache_dtype_str == ABI:
            if self.head_size != 576 or self.num_kv_heads != 1 or self.compress_ratio != 1:
                raise RuntimeError('FP4x spec requires semantic D576, one latent, no compression')
            return self.block_size * 368
        return stock(self)

    cls.real_page_size_bytes = property(page)
    # Dataclass equality/merge already include cache_dtype_str and dtype. No
    # subclass: retain UniformTypeKVCacheSpecs with the independent indexers.


def install_writer(mod):
    cls = mod.MLAAttentionImpl

    def update(self, x, rope, cache, slots, dtype, scale):
        if dtype not in ('fp8', 'fp8_e4m3') or self.dcp_world_size != 1:
            raise RuntimeError('FP4x writer requires original E4M3 config, DCP1')
        from glm_fp4_kv_kernel import pack_store
        return pack_store(x, rope, cache, slots, scale,
                          recent=getattr(self, '_glm_recent_bank', None))

    update._glm_fp4x = True
    cls.do_kv_cache_update = update


def install_gather(ops):
    if getattr(ops.gather_and_maybe_dequant_cache, '_glm_fp4x', False):
        return
    stock = ops.gather_and_maybe_dequant_cache

    @functools.wraps(stock)
    def gather(*args, **kwargs):
        cache = kwargs.get('src_cache', args[0] if args else None)
        if cache.shape[-1] == 368:
            from glm_fp4_kv_kernel import gather as packed_gather
            return packed_gather(*args, **kwargs)
        return stock(*args, **kwargs)

    gather._glm_fp4x = True
    ops.gather_and_maybe_dequant_cache = gather
    # FP8 projection workspaces are unsupported for packed context; fail closed
    # rather than let a direct FP8 gather reinterpret scales as activations.
    for name in ('cp_gather_cache', 'cp_gather_and_upconvert_fp8_kv_cache'):
        original = getattr(ops, name)

        def guarded(*args, _original=original, **kwargs):
            cache = kwargs.get('src_cache', args[0] if args else None)
            if cache.shape[-1] == 368:
                raise RuntimeError('FP4x requires BF16 context gather workspace')
            return _original(*args, **kwargs)

        setattr(ops, name, guarded)


def install_common(mod):
    import torch
    cls = mod.MLAAttention
    cls._glm_fp4x = True
    from glm_recent_kv import reserve
    original_init = cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        reserve(self)

    cls.__init__ = initialize
    # Preserve dependency dummy/custom-op ordering: the writer is replaced at
    # its pinned implementation dispatch, never the unified update operator.
    raw = Path(mod.__file__).read_text()
    from glm_fp4_prefill import qdq
    mod.__dict__['_glm_fp4x_qdq'] = qdq
    namespace = {}
    tree = ast.parse(transform_forward(raw))
    owner = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'MLAAttention')
    line = next(n.lineno for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_impl')
    ast.increment_lineno(tree, line - 1)
    exec(compile(tree, mod.__file__, 'exec'), mod.__dict__, namespace)
    stock_forward = namespace['forward_impl']
    stock_forward.__qualname__ = 'MLAAttention.forward_impl'
    stock_spec = cls.get_kv_cache_spec

    def spec(self, config):
        if (self.head_size != 576 or self.kv_lora_rank != 512 or self.use_pcp or
                self.impl.dcp_world_size != 1 or self.sliding_window is not None or
                self.kv_cache_dtype not in ('fp8', 'fp8_e4m3') or
                self.attn_backend.get_name() != 'FLASHINFER_MLA_SPARSE_SM90'):
            raise RuntimeError('FP4x requires release D512/R64 sparse MLA, DCP1/no PCP/no SWA')
        if not getattr(self.impl.do_kv_cache_update, '_glm_fp4x', False):
            raise RuntimeError('FP4x writer missing or overridden')
        s = stock_spec(self, config)
        return replace(s, dtype=torch.uint8, cache_dtype_str=ABI)

    @functools.wraps(stock_forward)
    def forward(self, q, x, rope, cache, md, *args, **kwargs):
        if getattr(self, '_glm_recent_target', False) and float(self._k_scale_float or 1.0)!=1.0:
            raise RuntimeError('recent KV requires released unit FP8 reader scale')
        if cache.numel() and (cache.dtype != torch.uint8 or cache.shape[-1] != 368):
            raise RuntimeError('FP4x central attention received a nonpacked cache')
        return stock_forward(self, q, x, rope, cache, md, *args, **kwargs)

    cls.get_kv_cache_spec = spec
    cls.forward_impl = forward
    install_gather(mod.ops)


def install_sparse(mod):
    backend = mod.FlashInferMLASparseSM90Backend
    if (not getattr(mod.FlashInferMLASparseSM90Impl.forward_mqa, '_glm_full_mla', False) or
            getattr(mod.FlashInferMLASparseSM90Impl.forward_mqa, '_glm_kv_format', None) != 'fp4x'):
        raise RuntimeError('FP4x requires the release Triton reader hook')

    def shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str='auto'):
        if head_size != 576 or num_kv_heads != 1:
            raise RuntimeError('FP4x physical shape requires semantic D576 and one latent')
        return (num_blocks, block_size, 368)

    backend.get_kv_cache_shape = staticmethod(shape)
    # Keep the planner for K-stop's host payload. No stock wrapper may read
    # packed bytes, regardless of whether its plan was refreshed.
    state_cls = mod._SM90State
    stock_init = state_cls.__init__

    def init(self, *a, **kw):
        stock_init(self, *a, **kw)

        def refused(*a, **kw):
            raise RuntimeError('stock FlashInfer MLA run forbidden for FP4x storage')

        self.wrapper.run = refused

    state_cls.__init__ = init
    if mod._SM90_STATE is not None:
        raise RuntimeError('FP4x installed after MLA state construction; restart required')


def install(mod):
    check_source(mod)
    if getattr(mod, '_glm_fp4x_installed', False):
        return
    if mod.__name__ == SPEC:
        install_spec(mod)
    elif mod.__name__ == BACKEND:
        install_writer(mod)
    elif mod.__name__ == COMMON:
        install_common(mod)
    elif mod.__name__ == SPARSE:
        install_sparse(mod)
    elif mod.__name__.endswith('sparse_attn_indexer'):
        from glm_fp4_prefill import install_indexer
        install_indexer(mod)
    elif mod.__name__.endswith('mla.indexer'):
        from glm_fp4_prefill import install_metadata
        install_metadata(mod)
    elif mod.__name__.endswith('sparse_mla_attention'):
        install_gather(mod.ops)
        from glm_fp4_prefill import install_sparse_metadata
        install_sparse_metadata(mod)
    # Other entries are ABI preservation pins (model and allocator).
    mod._glm_fp4x_installed = True
    sys.stderr.write('FP4x pinned: ' + mod.__name__ + '\n')


class Hooks(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in PINS or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None:
            raise ImportError(name)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            install(module)

        spec.loader.exec_module = execute
        return spec


def register(env=None):
    global _FORMAT
    mode = kv_format(os.environ if env is None else env)
    if _FORMAT is not None:
        if mode != _FORMAT:
            raise RuntimeError('KV format is fixed for the process; restart to switch')
        return mode == 'fp4x'
    _FORMAT = mode
    if mode == 'fp8':
        return False
    sys.meta_path.insert(0, Hooks())
    for name in PINS:
        if name in sys.modules:
            install(sys.modules[name])
    sys.stderr.write('FP4x KV ARMED ABI=' + ABI + '; target+MTP; E4M3 index retained\n')
    return True
