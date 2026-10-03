# SPDX-License-Identifier: Apache-2.0
"""GLM_SKIP_MLA_PLAN: do not plan the FlashInfer SM90 sparse MLA wrapper when nothing runs it.

With GLM_FULL_MLA=triton the MLA overlay replaces `forward_mqa`, and the replacement never
calls `_SM90_STATE.wrapper.run`, the plan's only consumer. The stock metadata builder still
plans on every build, and under async scheduling `_kv_lens_host` reads the exact positions
with a blocking `positions[:n].cpu()`: one read before the second MTP pass and one before
the next target launch, on every cycle. This wraps `FlashInferMLASparseSM90Builder.build`
so that, while the overlay forward is armed and the switch is on, it returns the parent
metadata (the same call the stock build makes before planning) without the plan and
without the two reads.

Modes (env GLM_SKIP_MLA_PLAN, default off):
  unset / 0  nothing is installed; the stock builder plans on every build.
  1          skip from boot.
  ab         in-boot A/B: the switch starts at GLM_SKIP_MLA_PLAN_AB_INIT (0|1, default 0) and
             changes only through the worker RPCs `skip_mla_plan_set` / `skip_mla_plan_status`
             over /collective_rpc (VLLM_SERVER_DEV_MODE=1 on the private loopback API). All
             ranks apply a collective RPC at the same position of the worker message stream.
             Never a serving mode.

Fail closed:
  * register refuses unless GLM_FULL_MLA=triton; install refuses unless the overlay
    forward is already armed on the class when the module loads.
  * the decision is re-read at every build from the class attribute; when `forward_mqa`
    is not the overlay's, the build plans exactly as stock (the plan is rebuilt lazily).
  * `wrapper.run` is guarded: it raises unless the last build planned, so the stock
    forward never runs on a stale or missing plan. Once it has run in this process the
    skip is off for good: a captured graph may then hold stock kernels that read the plan
    buffers on replay, where no Python guard runs.
  * the decision reads only the env mode, the switch, the class attribute and that latch;
    never positions, memory, timing or the rank.

With native MTP K-stop (GLM_MTP_KSTOP=1): K-stop carries its decision probabilities and its
guard flag inside this planner's host copy (`_kv_lens_host` -> `kstop_runtime.packed_host`).
A build that would skip while K-stop has such a payload waiting plans as stock instead, so the
payload rides the same copy as without this switch (counted in `counts['kstop']`, not in
`planned` or `fallback`). The payload state is set by K-stop identically on every rank. With
K-stop on, this switch therefore removes only the plans and reads that carry no K-stop payload.
"""
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

TARGET = 'vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90'
PIN = '4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136'  # = glm_full_mla.PIN
WORKER = 'vllm.v1.worker.gpu_worker'
WORKER_PIN = 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1'
MODES = ('0', '1', 'ab')
MODE = '0'
# plan: 'none' (never planned), 'fresh' (the last build planned), 'skipped' (the last build did not).
STATE = dict(on=False, plan='none', run_seen=False, switches=0)
COUNTS = dict(planned=0, skipped=0, fallback=0, kstop=0)
_mod = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_armed(mod):
    return getattr(mod.FlashInferMLASparseSM90Impl.forward_mqa, '_glm_full_mla', False) is True


def skipping(mod):
    """The whole skip decision. Rank-invariant inputs only."""
    return MODE != '0' and STATE['on'] and not STATE['run_seen'] and forward_armed(mod)


def kstop_payload():
    """True while native MTP K-stop waits for this planner's host copy (decision or guard).

    K-stop sets and clears both fields at the same host points on every rank."""
    runtime = sys.modules.get('kstop_runtime')
    state = getattr(runtime, 'STATE', None) if runtime is not None else None
    if not state:
        return False
    r = state['runtime']
    return r.pending is not None or r.guard_pending is not None


def guard_wrapper(state):
    """Refuse a stock `run` on a plan that is stale or was never built."""
    run = state.wrapper.run
    if getattr(run, '_glm_skip_mla_plan', False):
        return

    def guarded(*args, **kwargs):
        if STATE['plan'] != 'fresh':
            raise RuntimeError(f"glm-skip-mla-plan: stock FlashInfer MLA run on a {STATE['plan']} plan")
        STATE['run_seen'] = True
        return run(*args, **kwargs)

    guarded._glm_skip_mla_plan = True
    state.wrapper.run = guarded


def install(mod):
    global _mod
    if sha(mod.__file__) != PIN:
        raise RuntimeError('glm-skip-mla-plan: source drift ' + mod.__file__)
    builder = mod.FlashInferMLASparseSM90Builder
    if getattr(builder.build, '_glm_skip_mla_plan', False):
        return
    if not forward_armed(mod):
        raise RuntimeError('glm-skip-mla-plan: the MLA overlay forward is not armed (GLM_FULL_MLA=triton installs first)')
    stock = builder.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if skipping(mod) and kstop_payload():
            # Plan as stock so K-stop's payload rides the planner copy it was built for.
            COUNTS['kstop'] += 1
            metadata = stock(self, common_prefix_len, common_attn_metadata, fast_build)
            if mod._SM90_STATE is not None:
                STATE['plan'] = 'fresh'
            return metadata
        if skipping(mod):
            COUNTS['skipped'] += 1
            STATE['plan'] = 'skipped'
            return super(builder, self).build(common_prefix_len, common_attn_metadata, fast_build)
        if MODE != '0' and STATE['on']:
            COUNTS['fallback'] += 1
        COUNTS['planned'] += 1
        metadata = stock(self, common_prefix_len, common_attn_metadata, fast_build)
        if mod._SM90_STATE is not None:
            STATE['plan'] = 'fresh'
        return metadata

    build._glm_skip_mla_plan = True
    build._stock = stock
    builder.build = build

    cls = mod._SM90State
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        guard_wrapper(self)

    cls.__init__ = __init__
    if mod._SM90_STATE is not None:
        guard_wrapper(mod._SM90_STATE)
    _mod = mod
    sys.stderr.write(f"glm-skip-mla-plan: ARMED mode={MODE} on={int(STATE['on'])}\n")


def status():
    return dict(mode=MODE, installed=_mod is not None, forward_armed=_mod is not None and forward_armed(_mod),
                on=STATE['on'], skipping=_mod is not None and skipping(_mod), plan=STATE['plan'],
                run_seen=STATE['run_seen'], switches=STATE['switches'], counts=dict(COUNTS),
                pin=PIN, worker_pin=WORKER_PIN)


def set_switch(on):
    if MODE != 'ab':
        raise RuntimeError('glm-skip-mla-plan: the switch exists only in ab mode')
    if on not in (0, 1) or isinstance(on, float):
        raise ValueError('glm-skip-mla-plan: switch value must be 0 or 1')
    if _mod is None:
        raise RuntimeError('glm-skip-mla-plan: not installed')
    on = bool(on)
    if on and STATE['run_seen']:
        raise RuntimeError('glm-skip-mla-plan: a stock FlashInfer MLA run happened; the skip stays off')
    if on and not forward_armed(_mod):
        raise RuntimeError('glm-skip-mla-plan: the MLA overlay forward is not armed')
    at = dict(COUNTS)
    STATE['on'] = on
    STATE['switches'] += 1
    return dict(at_counts=at, **status())


def _rank():
    from vllm.distributed.parallel_state import get_tp_group
    return get_tp_group().rank_in_group


def install_worker(mod):
    if sha(mod.__file__) != WORKER_PIN:
        raise RuntimeError('glm-skip-mla-plan: source drift ' + mod.__file__)

    def skip_mla_plan_set(self, on):
        # /collective_rpc passes JSON args through; accept 0/1 and "0"/"1".
        return dict(rank=_rank(), **set_switch({'0': 0, '1': 1}.get(on, on) if isinstance(on, str) else on))

    def skip_mla_plan_status(self):
        return dict(rank=_rank(), **status())

    mod.Worker.skip_mla_plan_set = skip_mla_plan_set
    mod.Worker.skip_mla_plan_status = skip_mla_plan_status


class Hooks(importlib.abc.MetaPathFinder):
    """After-import callbacks; composes with the other startup finders."""

    def __init__(self):
        self.callbacks = {}
        self.busy = set()

    def after_import(self, name, fn):
        if name in sys.modules:
            fn(sys.modules[name])
            return
        self.callbacks.setdefault(name, []).append(fn)

    def find_spec(self, name, path=None, target=None):
        if name not in self.callbacks or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None:
            raise ImportError(name)
        original = spec.loader.exec_module
        callbacks = self.callbacks.pop(name)

        def exec_module(module):
            original(module)
            for fn in callbacks:
                fn(module)
        spec.loader.exec_module = exec_module
        return spec


def register(env=None):
    global MODE
    e = os.environ if env is None else env
    mode = e.get('GLM_SKIP_MLA_PLAN', '0') or '0'
    if mode not in MODES:
        raise ValueError('GLM_SKIP_MLA_PLAN must be 0, 1 or ab')
    if mode == '0':
        return False
    if e.get('GLM_FULL_MLA') != 'triton':
        raise ValueError('GLM_SKIP_MLA_PLAN needs GLM_FULL_MLA=triton')
    on = True
    if mode == 'ab':
        if e.get('VLLM_SERVER_DEV_MODE') != '1':
            raise ValueError('GLM_SKIP_MLA_PLAN=ab needs VLLM_SERVER_DEV_MODE=1 on the private loopback API')
        init = e.get('GLM_SKIP_MLA_PLAN_AB_INIT', '0')
        if init not in ('0', '1'):
            raise ValueError('GLM_SKIP_MLA_PLAN_AB_INIT must be 0 or 1')
        on = init == '1'
    MODE = mode
    STATE['on'] = on
    hooks = Hooks()
    sys.meta_path.insert(0, hooks)
    hooks.after_import(TARGET, install)
    if mode == 'ab':
        hooks.after_import(WORKER, install_worker)
    return True
