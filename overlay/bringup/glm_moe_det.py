# SPDX-License-Identifier: Apache-2.0
"""Boot-only deterministic Marlin align; inert unless GLM_MOE_DET_ALIGN=1.

Uses the same pinned target/MTP import sites and admission checks as ce09046.
"""
import functools
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

ALIGN = 'vllm.model_executor.layers.fused_moe.moe_align_block_size'
MARLIN = 'vllm.model_executor.layers.fused_moe.experts.marlin_moe'
MTP = 'glm_nvfp4_mtp'
RUNNER = 'vllm.v1.worker.gpu.model_runner'
PINS = {
    ALIGN: 'c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b',
    MTP: '825b7ecb15b771c158812d6f89e04d2c4ea90f6db80d2fe415fab8586e47c8a5',
}
PARAMETERS = ('topk_ids', 'block_size', 'num_experts', 'expert_map',
              'pad_sorted_ids', 'ignore_invalid_experts')
STATS = dict(installs=0, sites=[], mtp_checked=False, logged=False)
_REGISTERED = False


def deterministic_wrapper(align):
    if tuple(inspect.signature(align).parameters) != PARAMETERS:
        raise ValueError('moe-det: unsupported align variant (batched/LoRA)')
    if getattr(align, '_glm_moe_det', False):
        return align
    from det_align.runtime import deterministic_align
    @functools.wraps(align)
    def ordered(topk_ids, block_size, num_experts, expert_map=None,
                pad_sorted_ids=False, ignore_invalid_experts=False):
        return deterministic_align(topk_ids, block_size, num_experts, expert_map,
                                   pad_sorted_ids, ignore_invalid_experts)
    ordered._glm_moe_det = True
    ordered._glm_moe_det_original = align
    return ordered


def _check_source(module):
    from glm_glue_lite import check_pin
    if module.__name__ in (MARLIN, RUNNER):
        check_pin(module.__name__, module.__file__)
    elif hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PINS[module.__name__]:
        raise RuntimeError('moe-det: source drift ' + module.__name__)


def _refuse_variant(*args, **kwargs):
    raise ValueError('moe-det: batched/LoRA Marlin is not qualified')


def install(module, strict=True):
    if strict:
        _check_source(module)
    name = module.__name__
    if name in (ALIGN, MARLIN):
        original = getattr(module, 'moe_align_block_size', None)
        if original is None:
            raise RuntimeError('moe-det: missing align call site ' + name)
        module.moe_align_block_size = deterministic_wrapper(original)
        if name not in STATS['sites']:
            STATS['sites'].append(name)
            STATS['installs'] += 1
        if name == MARLIN:
            # Refuse unsupported variants during expert installation, before any
            # warmup/capture. Also guard the standalone batched entry point.
            for attr in ('BatchedMarlinExperts', 'MarlinExperts', 'fused_marlin_moe'):
                if not hasattr(module, attr):
                    raise RuntimeError('moe-det: missing Marlin site ' + attr)
            module.BatchedMarlinExperts.__init__ = _refuse_variant
            module.batched_fused_marlin_moe = _refuse_variant
            module.MarlinExperts.set_lora_context = _refuse_variant
    elif name == MTP:
        if tuple(inspect.signature(module.apply_experts).parameters) != (
                'layer', 'x', 'topk_weights', 'topk_ids', 'ops', 'align', 'scalar'):
            raise RuntimeError('moe-det: missing MTP default-align call site')
        # The pinned apply_experts lazily reads ALIGN.moe_align_block_size.
        STATS['mtp_checked'] = True
    elif name == RUNNER:
        cls = module.GPUModelRunner
        if getattr(cls.load_model, '_glm_moe_det', False):
            return
        load = cls.load_model

        @functools.wraps(load)
        def loaded(self, *args, **kwargs):
            if getattr(self.vllm_config, 'lora_config', None) is not None:
                raise ValueError('moe-det: LoRA align is not qualified')
            out = load(self, *args, **kwargs)
            verify_installs(strict)
            from det_align.runtime import prepare_stream, prepare_capture_stream
            prepare_stream()  # device is initialized; before profile/capture
            prepare_capture_stream()  # FULL manager omits stream= in torch.cuda.graph
            return out

        loaded._glm_moe_det = True
        cls.load_model = loaded


def verify_installs(strict=True):
    missing = [name for name in (ALIGN, MARLIN)
               if name not in STATS['sites'] or not getattr(
                   getattr(sys.modules.get(name), 'moe_align_block_size', None), '_glm_moe_det', False)]
    if not STATS['mtp_checked'] or not callable(
            getattr(sys.modules.get(MTP), 'apply_experts', None)):
        missing.append(MTP)
    if strict and missing:
        raise RuntimeError('moe-det: missing expected call sites: ' + ', '.join(missing))
    if not STATS['logged']:
        sys.stderr.write(f"glm-moe-det: installs={STATS['installs']} target={MARLIN} "
                         f"mtp_default={ALIGN} mtp_checked={int(STATS['mtp_checked'])} missing={missing}\n")
        STATS['logged'] = True
    return not missing


def register(env=None):
    global _REGISTERED
    env = os.environ if env is None else env
    value = env.get('GLM_MOE_DET_ALIGN', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_MOE_DET_ALIGN must be 0 or 1')
    if value == '0':
        return False
    if env.get('GLM_MOE_CANON_ALIGN', '0') != '0':
        raise ValueError('moe-det: canonical and deterministic switches are mutually exclusive')
    if _REGISTERED:
        return True
    strict = env.get('GLM_MOE_DET_STRICT', '1') != '0'
    from glm_skip_mla_plan import Hooks
    from glm_glue_lite import check_pin
    # Fail before loading weights on source drift or absent expected modules.
    if strict:
        root = Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
        for name in (ALIGN, MARLIN, RUNNER):
            path = root / (name.removeprefix('vllm.').replace('.', '/') + '.py')
            if name in (MARLIN, RUNNER):
                check_pin(name, path)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
                raise RuntimeError('moe-det: source drift ' + name)
        spec = importlib.util.find_spec(MTP)
        if spec is None or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != PINS[MTP]:
            raise RuntimeError('moe-det: absent/drifted native MTP source')
    from det_align.runtime import prepare_library
    prepare_library()  # boot only, before model load/capture
    hooks = Hooks()
    sys.meta_path.insert(0, hooks)
    for name in (ALIGN, MARLIN, MTP, RUNNER):
        hooks.after_import(name, lambda module: install(module, strict))
    _REGISTERED = True
    return True
