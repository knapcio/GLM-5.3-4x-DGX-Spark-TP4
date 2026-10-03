# SPDX-License-Identifier: Apache-2.0
"""GLM_DISPRAM_KV: put the tail of the KV pool in the GB10 display carveout (vLLM 487ecf187, V2 runner).

Inert unless GLM_DISPRAM_KV is 1/require or auto. Source-pinned; refuses drift.

1. Worker.determine_available_memory: after the stock answer (the pinned --kv-cache-memory-bytes),
   every rank claims, maps and verifies the same carveout size (glm_carveout.negotiate, CPU-group MIN
   collectives over size, success and clean teardown) and reports pinned + agreed bytes. vLLM's own cross-worker MIN then yields
   identical block counts.
2. attn_utils._allocate_kv_cache, only inside Worker.initialize_from_config: the per-layer KV tensors
   are carved out of one VMM range [ordinary head | carveout tail], each with its own storage (so
   copy_kv_cache_blocks_inplace and the zeroer keep per-layer semantics). Zeroed by an SM kernel.
3. After initialize_from_config the pool must exist whenever a lease exists; otherwise the extra
   bytes went to ordinary memory and the worker aborts.

Refused configurations: no pinned pool, V1 runner, KV transfer/offload, sleep mode.
GLM_DISPRAM_KV_GUARD=1 (smoke/qualification): Tensor.copy_/clone touching the pool raise.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

import glm_carveout as gc

TARGETS = {
    'vllm.v1.worker.gpu.attn_utils': '1dd3dd2826a2cc73005e7baecb71c26de8d56285b35d780716ab11ffe0f8495b',
    'vllm.v1.worker.gpu_worker': 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1',
}
STATE = dict(lease=None, base=None, armed=False, decided=False)


class _Coll:
    def __init__(self):
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_world_group
        self.g = get_world_group().cpu_group
        self.dist = dist
        self.rank = dist.get_rank(self.g)

    def min(self, v):
        import torch
        t = torch.tensor([int(v)], dtype=torch.int64)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MIN, group=self.g)
        return int(t.item())


def _device():
    import torch
    torch.cuda.synchronize()
    return torch.cuda.current_device()


def _make_coll():
    return _Coll()


def refuse_config(worker, env=os.environ):
    vc = worker.vllm_config
    bad = []
    if worker.cache_config.kv_cache_memory_bytes is None:
        bad.append('--kv-cache-memory-bytes must pin the ordinary-memory part of the pool')
    if env.get('VLLM_USE_V2_MODEL_RUNNER') != '1':
        bad.append('only the V2 runner allocation path is hooked')
    if getattr(vc, 'kv_transfer_config', None) is not None:
        bad.append('KV transfer copies KV with the copy engine')
    if getattr(worker.cache_config, 'kv_offloading_size', None):
        bad.append('KV offload copies KV with the copy engine')
    if getattr(worker.model_config, 'enable_sleep_mode', False):
        bad.append('sleep mode backs up and restores KV with memcpy')
    if bad:
        raise gc.CarveoutError('GLM_DISPRAM_KV refused: ' + '; '.join(bad))


def determine(orig, self, *a, **kw):
    base = orig(self, *a, **kw)
    if STATE['decided']:
        raise gc.CarveoutError('determine_available_memory called twice; lease is decided once per process')
    STATE['decided'] = True
    refuse_config(self)
    lease = gc.negotiate(_make_coll(), _device())
    STATE['lease'], STATE['base'] = lease, int(base)
    extra = lease.size if lease else 0
    gc.log('budget', pinned=int(base), carveout=extra, reported=int(base) + extra)
    return int(base) + extra


def initialize(orig, self, *a, **kw):
    STATE['armed'] = True
    try:
        out = orig(self, *a, **kw)
    finally:
        STATE['armed'] = False
    lease = STATE['lease']
    if lease is not None and lease.pool is None:
        raise gc.CarveoutError('lease held but the KV pool was not built by the hook: '
                               'extra KV would sit in ordinary memory')
    return out


def allocate(orig, kv_cache_config, shared_layers, device):
    lease = STATE['lease']
    if not STATE['armed'] or lease is None:
        return orig(kv_cache_config, shared_layers, device)
    keys, sizes, owner = [], [], {}
    for t in kv_cache_config.kv_cache_tensors:
        key = 'packed' if t.block_stride > 0 else id(t)
        if key not in owner:
            owner[key] = len(sizes)
            keys.append(key)
            sizes.append(t.size)
        elif sizes[owner[key]] != t.size:
            raise gc.CarveoutError('packed KV tensors disagree on size')
    offs, total, head = gc.plan_regions(sizes, lease.size, lease.gran, STATE['base'])
    rb, ra = lease.drv.release_cached()  # fix7: the stock torch.zeros path would have reused these cached blocks
    gc.log('torch-cache-released', reserved_before=rb, reserved_after=ra)
    pool = lease.build_pool(total, head, lease.gran)
    whole = pool.tensor(0, total)
    whole.zero_()  # one-time init; elementwise fill kernel (smoke S6 checks it adds no memset/memcpy)
    del whole
    tensors = [pool.tensor(o, s) for o, s in zip(offs, sizes)]
    raw = {}
    for t in kv_cache_config.kv_cache_tensors:
        key = 'packed' if t.block_stride > 0 else id(t)
        for name in t.shared_by:
            raw[name] = tensors[owner[key]]
    names = {n for g in kv_cache_config.kv_cache_groups for n in g.layer_names}
    if names != (raw.keys() | shared_layers.keys()):
        raise gc.CarveoutError('Some layers are not correctly initialized')
    seam = next((i for i, (o, s) in enumerate(zip(offs, sizes)) if o + s > head), len(sizes))
    gc.log('pool', total=total, head=head, tail=lease.size, regions=len(sizes), first_region_touching_tail=seam,
           num_blocks=kv_cache_config.num_blocks, va=hex(pool.va))
    if not gc.guard_arm(pool) and os.environ.get('DISPRAM_GUARD_REQUIRED') == '1':
        raise gc.CarveoutError('copy guard required but not loaded in the worker')
    if os.environ.get('GLM_DISPRAM_KV_GUARD') == '1':
        install_copy_guard(pool)
    lease.drv.synchronize()
    return raw


def install_copy_guard(pool, on_device=lambda t: t.is_cuda):
    """Trip-wire for Python-level copy_/clone on the pool (cudaMemcpy runs at 0.49x on the carveout)."""
    import torch

    def touches(t):
        return isinstance(t, torch.Tensor) and on_device(t) and t.numel() > 0 and \
            pool.overlaps(t.data_ptr(), t.numel() * t.element_size())

    copy_, clone = torch.Tensor.copy_, torch.Tensor.clone

    @functools.wraps(copy_)
    def guarded_copy(self, src, *a, **kw):
        if touches(self) or touches(src):
            raise gc.CarveoutError('copy_ on the carveout KV pool (copy-engine path forbidden)')
        return copy_(self, src, *a, **kw)

    @functools.wraps(clone)
    def guarded_clone(self, *a, **kw):
        if touches(self):
            raise gc.CarveoutError('clone of the carveout KV pool (copy-engine path forbidden)')
        return clone(self, *a, **kw)

    torch.Tensor.copy_, torch.Tensor.clone = guarded_copy, guarded_clone
    gc.log('copy-guard', armed=True)
    return copy_, clone


def install(mod):
    got = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    if got != TARGETS[mod.__name__]:
        raise RuntimeError(f'glm-dispram-kv: source drift {mod.__name__}: {got}')
    if mod.__name__.endswith('attn_utils'):
        orig = mod._allocate_kv_cache

        @functools.wraps(orig)
        def _allocate_kv_cache(kv_cache_config, shared_layers, device):
            return allocate(orig, kv_cache_config, shared_layers, device)
        mod._allocate_kv_cache = _allocate_kv_cache
    else:
        W = mod.Worker
        od, oi = W.determine_available_memory, W.initialize_from_config

        @functools.wraps(od)
        def determine_available_memory(self, *a, **kw):
            return determine(od, self, *a, **kw)

        @functools.wraps(oi)
        def initialize_from_config(self, *a, **kw):
            return initialize(oi, self, *a, **kw)
        W.determine_available_memory = determine_available_memory
        W.initialize_from_config = initialize_from_config
    gc.log('installed', module=mod.__name__)


def guard_loaded():
    import ctypes
    try:
        ctypes.CDLL(None).dispram_guard_set
        return True
    except (AttributeError, OSError):
        return False


def register(env=None):
    env = os.environ if env is None else env
    if gc.parse_mode(env) == '0':
        return False
    # qualification boots: the LD_PRELOAD copy guard must be in THIS process; ld.so only warns and
    # continues when a preload is missing, so a missing guard is made fatal here (sitecustomize exits 78)
    if env.get('DISPRAM_GUARD_REQUIRED') == '1' and not guard_loaded():
        raise gc.CarveoutError(f'DISPRAM_GUARD_REQUIRED=1 but the copy guard is not loaded in pid {os.getpid()} '
                               f'(LD_PRELOAD={env.get("LD_PRELOAD")!r})')

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            original_exec = spec.loader.exec_module

            def execute(module):
                original_exec(module)
                install(module)
            spec.loader.exec_module = execute
            return spec

    for name in TARGETS:
        if name in sys.modules:
            install(sys.modules[name])
    sys.meta_path.insert(0, Finder())
    return True
