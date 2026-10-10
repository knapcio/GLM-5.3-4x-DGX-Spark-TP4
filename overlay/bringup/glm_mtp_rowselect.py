# SPDX-License-Identifier: Apache-2.0
"""Select draft pass-1 rows AFTER native attention/KV, BEFORE o_proj.

Only loaded draft instances receive methods. Every target class/method and
every full-row attention descriptor stays native. No runtime switch exists.
"""
import contextlib
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import types

from glm_mtp_rowselect_config import options, descriptor_rows

RUNNER = 'vllm.v1.worker.gpu.model_runner'
PINS = {
    RUNNER: 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415',
    'vllm.model_executor.models.deepseek_mtp': '88521bcf3bfec6c773998dbefe30448504640907a4ac98f4e822d0ed38e2436f',
    'vllm.model_executor.models.deepseek_v2': '58d8916458de7c6f73b40bfef9d2f57bdd6fa0fb79be9e269331af6e66149fe2',
    'vllm.model_executor.layers.mla': '936b06c4671d52fce52ae85bb24b986db17855f32740bf55b226053fc385c4b4',
    'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator': '575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1',
    'vllm.v1.worker.gpu.spec_decode.mtp.speculator': '1fcffbf5e5a85e4c901bd71c65a73da814e7273627276dbdfebf367720a0bc1a',
    'vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils': '13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5',
    'vllm.v1.worker.gpu.cudagraph_utils': 'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f',
    'vllm.model_executor.layers.attention.mla_attention': 'a309ca752b01e85360b268fba56245b816ff245340e9e361166ee96b99d93a94',
}


def gather(module, tensor):
    indices = module._glm_rowselect_indices
    return tensor if indices is None else tensor.index_select(0, indices)


def rewrite(kind, source):
    """Small, count-checked edits of source-pinned native methods."""
    edits = {
        'mla': [('        return self.o_proj(attn_out)[0]',
                 '        attn_out = _rowselect_gather(self, attn_out)\n'
                 '        return self.o_proj(attn_out)[0]')],
        'block': [('        # Fully Connected\n',
                   '        residual = _rowselect_gather(self, residual)\n'
                   '        # Fully Connected\n')],
        'prefill': [('        sample_hidden_states = last_hidden_states[last_token_indices]',
                     '        sample_hidden_states = last_hidden_states'),
                    ('            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]',
                     '            self.hidden_states[:num_reqs] = hidden_states')],
    }
    for old, new in edits[kind]:
        if source.count(old) != 1:
            raise RuntimeError('rowselect anchor drift: ' + kind)
        source = source.replace(old, new, 1)
    return source


def clone_method(method, kind):
    raw = textwrap.dedent(inspect.getsource(method))
    # Dedented method body retains native globals, annotations and arithmetic.
    rewritten = rewrite(kind, textwrap.indent(raw, '    '))
    namespace = dict(method.__globals__, _rowselect_gather=gather)
    exec(compile('from __future__ import annotations\n' + textwrap.dedent(rewritten),
                 __file__, 'exec'), namespace)
    return namespace[method.__name__]


def prepare(sp, target):
    """Build all methods before an all-rank vote; attachment happens later."""
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    if type(sp) is not MTPSpeculator or type(sp.model) is not DeepSeekMTP or sp.model is target:
        raise RuntimeError('rowselect requires the separate native MTP instance')
    layers = list(sp.model.model.layers.values())
    if len(layers) != 1 or getattr(sp, '_glm_rowselect', None) is not None:
        raise RuntimeError('rowselect requires one unmodified MTP layer')
    block = layers[0].mtp_block
    mla = block.self_attn.mla_attn
    if (type(block) is not DeepseekV2DecoderLayer or
            type(mla) is not MultiHeadLatentAttentionWrapper or block.use_mha or
            block.use_sequence_parallel_moe or mla.dcp_q_replicate or
            sp.vllm_config.compilation_config.mode != 0):
        raise RuntimeError('rowselect requires native MLA, no SP/DCP or compilation')
    # Class identity alone does not exclude instance overrides or target aliases.
    target_ids = {id(m) for m in target.modules()}
    if any(id(m) in target_ids for m in (layers[0], block, mla, mla.o_proj, block.mlp)):
        raise RuntimeError('rowselect draft tail aliases target')
    for obj, method in ((block, DeepseekV2DecoderLayer.forward),
                        (mla, MultiHeadLatentAttentionWrapper.forward)):
        if obj.__dict__.get(method.__name__) is not None or getattr(obj, method.__name__).__func__ is not method:
            raise RuntimeError('rowselect native method was overridden')
    # Router capture may already own the outer prefill wrapper. Its inner
    # method must still be the pinned native body, and attachment keeps it outer.
    prefill = sp._prefill
    native_prefill = getattr(prefill, '_glm_router_prefill', prefill)
    if (native_prefill.__func__ is not AutoRegressiveSpeculator._prefill or
            native_prefill.__self__ is not sp or
            (native_prefill is prefill and sp.__dict__.get('_prefill') is not None)):
        raise RuntimeError('rowselect native method was overridden')
    return dict(block=block, mla=mla, native_block=block.forward, native_mla=mla.forward,
        native_prefill=native_prefill,
        block_fn=clone_method(DeepseekV2DecoderLayer.forward, 'block'),
        mla_fn=clone_method(inspect.unwrap(MultiHeadLatentAttentionWrapper.forward), 'mla'),
        prefill_fn=clone_method(AutoRegressiveSpeculator._prefill, 'prefill'))


def attach(sp, prepared):
    router_capture = getattr(getattr(sp, '_prefill', None), '_glm_router_capture', None)
    block, mla = prepared['block'], prepared['mla']
    block._glm_rowselect_indices = mla._glm_rowselect_indices = None
    block.forward = types.MethodType(prepared['block_fn'], block)
    mla.forward = types.MethodType(prepared['mla_fn'], mla)
    native = prepared['prefill_fn']

    @functools.wraps(native)
    def prefill(self, num_reqs, num_tokens, *args, **kwargs):
        if (getattr(self, '_glm_rowselect_reference', False) or
                getattr(self, '_glm_rowselect', {}).get('disabled', False)):
            return prepared['native_prefill'](num_reqs, num_tokens, *args, **kwargs)
        if block._glm_rowselect_indices is not None or mla._glm_rowselect_indices is not None:
            raise RuntimeError('reentrant rowselect prefill')
        # Device values are never read by Python. The view has a static shape
        # per native descriptor and follows last_token_indices across replay.
        indices = self.last_token_indices[:num_reqs]
        block._glm_rowselect_indices = mla._glm_rowselect_indices = indices
        try:
            return native(self, num_reqs, num_tokens, *args, **kwargs)
        finally:
            block._glm_rowselect_indices = mla._glm_rowselect_indices = None

    sp._prefill = types.MethodType(prefill, sp)
    if not hasattr(sp, "model"):
        return
    sp._glm_rowselect_native = (block, mla, block.forward, mla.forward, sp._prefill,
        prepared['native_block'], prepared['native_mla'], prepared['native_prefill'])
    sp._glm_rowselect = dict(boot_only=True, ready=False, descriptors=[])
    if router_capture is not None:
        from glm_router_capture import wrap_prefill
        wrap_prefill(sp, router_capture)


def capture_logits(sp):
    """Fixed FP32 observation buffer, captured with each native graph."""
    if hasattr(sp, '_glm_rowselect_logits'):
        return
    import torch
    vocab = sp.model.config.vocab_size
    sp._glm_rowselect_logits = torch.zeros((sp.max_num_reqs, vocab),
        device=sp.hidden_states.device, dtype=torch.float32)
    compute = sp.model.compute_logits
    def logits(self, hidden, *args, **kwargs):
        out = compute(hidden, *args, **kwargs)
        sp._glm_rowselect_logits[:len(out)].copy_(out)
        return out
    sp.model.compute_logits = types.MethodType(logits, sp.model)


@contextlib.contextmanager
def full_reference(sp):
    block, mla = sp._glm_rowselect_native[:2]
    native_block, native_mla = sp._glm_rowselect_native[5:7]
    # Snapshot the live methods: other instance wrappers can attach after us.
    entry = block.forward, mla.forward, sp._prefill
    reference = getattr(sp, '_glm_rowselect_reference', False)
    block.forward, mla.forward = native_block, native_mla
    sp._glm_rowselect_reference = True
    try:
        yield
    finally:
        block.forward, mla.forward, sp._prefill = entry
        sp._glm_rowselect_reference = reference


def kv_scratch(sp):
    """Only reserved null block 0; INIT precedes request admission."""
    import torch
    caches = {}
    for module in sp.model.modules():
        if getattr(getattr(module, 'impl', None), '_glm_recent_bank', None) is not None:
            raise RuntimeError('rowselect INIT does not support recent KV banks')
        cache = getattr(module, 'kv_cache', None)
        if isinstance(cache, torch.Tensor) and cache.numel():
            if cache.ndim < 2 or cache.shape[0] < 2 or cache.shape[1] < 16:
                raise RuntimeError('rowselect requires reserved null KV block with >=16 rows')
            caches.setdefault(cache.data_ptr(), cache[0])
    if not caches:
        raise RuntimeError('rowselect missing native KV cache scratch block')
    return list(caches.values())


def disable(sp, refusal):
    """Boot fallback only, after synchronization and draft graph release.

    Keep the outer prefill wrapper (including router capture), but dispatch its
    inner call to the pinned native full-M method on every future capture/run.
    """
    block, mla = sp._glm_rowselect_native[:2]
    block.forward, mla.forward = sp._glm_rowselect_native[5:7]
    block._glm_rowselect_indices = mla._glm_rowselect_indices = None
    sp._glm_rowselect.update(ready=False, disabled=True, descriptors=[],
                             init_failure=refusal.receipt)
    capture = getattr(sp._prefill, '_glm_router_capture', None)
    if capture is not None:
        # Native full-M routing uses token spans, not selected request rows.
        capture.mtp_rowselect = False
        capture.manifest()


def qualification_valid(report):
    row = (report or {}).get('rowselect', {})
    return (row.get('full_m_selected') is True and row.get('changed_positions') is True
            and row.get('bit_exact') is True and row.get('det_align') is True
            and row.get('logits_confidence_feedback_kv') is True and bool(row.get('cases')))


def account(sp):
    rows = []
    for stage, manager in (('pass1', sp.prefill_cudagraph_manager),
                           ('later', sp.decode_cudagraph_manager)):
        if manager is None or not manager.graphs:
            raise RuntimeError('rowselect missing native draft graphs')
        for d in manager.graphs:
            if str(d.cg_mode).split('.')[-1] != 'FULL':
                raise RuntimeError('rowselect requires native FULL draft graphs')
            row = descriptor_rows(d.num_tokens, d.num_reqs, sp.max_num_reqs)
            if stage == 'later':
                row.update(tail_rows=d.num_tokens, discarded_tail_rows=0)
            rows.append(dict(stage=stage, descriptor=repr(d), **row))
    return rows


def install_runner(mod):
    cls = mod.GPUModelRunner
    load, capture = cls.load_model, cls.capture_model

    @functools.wraps(load)
    def load_model(self, *args, **kwargs):
        load(self, *args, **kwargs)
        import glm_draft_head as dh
        prepared = None
        error = None
        try:
            prepared = prepare(self.speculator, self.model)
        except Exception as exc:
            error = exc
        dh.agree({'rowselect': 1, 'pins': PINS, 'boundary': 'post-attention/pre-o-proj', 'capture_layout': 'reuse'}, error is None)
        if error is not None:
            raise RuntimeError('rowselect preparation refused') from error
        attach(self.speculator, prepared)

    @functools.wraps(capture)
    def capture_model(self, *args, **kwargs):
        sp = self.speculator
        sp._glm_rowselect['ready'] = False
        import glm_draft_head as dh
        self._draft_head_ready = False
        error = None
        try:
            capture_logits(sp)
        except Exception as exc:
            error = exc
        dh.agree({'rowselect_logits_capture': True}, error is None)
        if error is not None:
            raise RuntimeError('rowselect logits buffer preparation refused') from error
        try:
            result = capture(self, *args, **kwargs)
        except BaseException:
            self._draft_head_ready = False
            raise
        import glm_draft_head as dh
        sp = self.speculator
        # The inner DH wrapper already completed a collective OFF recapture.
        # Do not reinterpret that expected refusal as an outer INIT failure.
        if getattr(self, '_draft_head_init_failure', None) is not None:
            if (not sp._glm_rowselect.get('disabled') or self._draft_head.on):
                raise RuntimeError('rowselect INIT fallback did not restore OFF arm')
            return result
        rows, error = [], None
        try:
            rows = account(sp)
            if not self._draft_head_ready or not qualification_valid(self._draft_head_qualification):
                raise RuntimeError('rowselect requires native INIT eager/replay qualification')
        except Exception as exc:
            error = exc
        try:
            dh.agree({'rowselect_descriptors': rows}, error is None)
        except dh.VoteRefused as exc:
            dh.initial_fallback(self, exc)
            return result
        except BaseException:
            self._draft_head_ready = False
            raise
        if error is not None:
            raise RuntimeError('rowselect capture refused') from error
        sp._glm_rowselect.update(ready=True, descriptors=rows)
        print('GLM_MTP_ROWSELECT ' + json.dumps(sp._glm_rowselect, sort_keys=True), flush=True)
        return result

    cls.load_model, cls.capture_model = load_model, capture_model


def register(env=None):
    env = os.environ if env is None else env
    if not options(env):
        return False
    import importlib.util
    import sys
    if RUNNER in sys.modules:
        raise RuntimeError('rowselect must register before runner import')
    root = Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    for name, expected in PINS.items():
        path = root / (name.removeprefix('vllm.').replace('.', '/') + '.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('rowselect source drift: ' + name)
    from glm_skip_mla_plan import Hooks
    hooks = Hooks()
    sys.meta_path.insert(0, hooks)
    hooks.after_import(RUNNER, install_runner)
    return True
