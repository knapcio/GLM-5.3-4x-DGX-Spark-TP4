"""Monkeypatches that wire ``GlmRoceAllReduce`` into the image's vLLM.

The four hook points are the ones local-inference-lab/vllm#597 edits in the vLLM
source (and tonyd2wild bind-mounted as whole files into the v11 tree); here they
are wrappers applied at import time, so no vLLM file in the image changes and the
image behaves exactly like its base when ``GLM_ROCE_ALLREDUCE`` is unset.

1. ``CudaCommunicator.__init__``: for the ``tp`` group only, build the adapter into
   ``self.glm_roce_comm`` after the stock backends (PyNccl etc.) exist.
2. ``CudaCommunicator.all_reduce`` / ``all_gather``: eligible inputs go to the adapter
   first, everything else to the stock dispatch (NCCL on this fleet); ``destroy``
   closes the runtime before the stock teardown.
3. ``GroupCoordinator.graph_capture``: enter the adapter's ``capture()`` around the
   stock context, so launchers are compiled and the eager ordering event is dropped
   before any capture.  Both vLLM runners (V1 ``capture_model`` and V2
   ``CudaGraphManager.capture``) capture inside ``graph_capture()``.
4. ``Worker.execute_model`` / ``sample_tokens``: the fail-stop health check once the
   step's output is on the host (async outputs are wrapped so the check follows
   ``get_output()``), exactly where #597 puts it.

``verify_targets`` checks that the image's vLLM still has these shapes; the image
build runs it so an upstream drift fails the build, not a boot.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional

logger = logging.getLogger("vllm.glm_roce")

MOD_CUDA_COMM = "vllm.distributed.device_communicators.cuda_communicator"
MOD_PARALLEL_STATE = "vllm.distributed.parallel_state"
MOD_GPU_WORKER = "vllm.v1.worker.gpu_worker"

_PATCHED = "_glm_roce_patched"


def _is_tp_group(unique_name: str) -> bool:
    """vLLM names coordinator groups "<kind>:<index>"; only the TP group is routed."""
    return str(unique_name or "").split(":", 1)[0] == "tp"


def roce_comm_of(device_communicator: Any):
    """The active adapter of a device communicator, or None."""
    comm = getattr(device_communicator, "glm_roce_comm", None)
    if comm is None or getattr(comm, "disabled", True):
        return None
    return comm


# -- 1 + 2: the device communicator ---------------------------------------------------


def patch_cuda_communicator(module, adapter_factory: Optional[Callable[..., Any]] = None) -> None:
    cls = module.CudaCommunicator
    if getattr(cls, _PATCHED, False):
        return
    orig_init = cls.__init__
    orig_all_reduce = cls.all_reduce
    orig_all_gather = cls.all_gather
    orig_destroy = cls.destroy

    def make_adapter(**kwargs):
        if adapter_factory is not None:
            return adapter_factory(**kwargs)
        from glm_roce.adapter import GlmRoceAllReduce

        return GlmRoceAllReduce(**kwargs)

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.glm_roce_comm = None
        if not _is_tp_group(getattr(self, "unique_name", "")) or self.world_size <= 1:
            return
        existing = getattr(self, "b12x_ar_comm", None)
        if existing is not None and not getattr(existing, "disabled", True):
            # An image that already carries #597 routes through its own slot; two
            # runtimes on one QP set would deadlock.  Same image on every rank, so
            # every rank takes this branch.
            logger.warning(
                "GLM_ROCE: the image already routes through b12x_ar_comm (%s); the shim stays off",
                getattr(existing, "backend_name", type(existing).__name__),
            )
            return
        self.glm_roce_comm = make_adapter(
            group=self.cpu_group, device_group=self.device_group, device=self.device
        )

    @functools.wraps(orig_all_reduce)
    def all_reduce(self, input_, *args, **kwargs):
        comm = roce_comm_of(self)
        if comm is not None and not args and not kwargs and comm.should_custom_ar(input_):
            out = comm.custom_all_reduce(input_)
            if out is not None:
                return out
        return orig_all_reduce(self, input_, *args, **kwargs)

    @functools.wraps(orig_all_gather)
    def all_gather(self, input_, dim: int = -1):
        comm = roce_comm_of(self)
        if comm is not None:
            d = dim + input_.dim() if dim < 0 else dim
            if comm.should_all_gather(input_, d):
                return comm.all_gather(input_, d)
        return orig_all_gather(self, input_, dim)

    @functools.wraps(orig_destroy)
    def destroy(self, *args, **kwargs):
        comm = getattr(self, "glm_roce_comm", None)
        if comm is not None:
            comm.close()
            self.glm_roce_comm = None
        return orig_destroy(self, *args, **kwargs)

    cls.__init__ = __init__
    cls.all_reduce = all_reduce
    cls.all_gather = all_gather
    cls.destroy = destroy
    setattr(cls, _PATCHED, True)
    logger.debug("GLM_ROCE patched %s.CudaCommunicator", module.__name__)


# -- 3: graph capture -------------------------------------------------------------------


def patch_parallel_state(module) -> None:
    cls = module.GroupCoordinator
    if getattr(cls, _PATCHED, False):
        return
    orig_graph_capture = cls.graph_capture

    @contextmanager
    def graph_capture(self, graph_capture_context=None):
        comm = roce_comm_of(getattr(self, "device_communicator", None))
        outer = comm.capture() if comm is not None else nullcontext()
        with outer:
            with orig_graph_capture(self, graph_capture_context) as context:
                yield context

    functools.update_wrapper(graph_capture, orig_graph_capture)
    cls.graph_capture = graph_capture
    setattr(cls, _PATCHED, True)
    logger.debug("GLM_ROCE patched %s.GroupCoordinator.graph_capture", module.__name__)


# -- 4: the worker's fail-stop health check ---------------------------------------------


def _tp_health_check() -> Optional[Callable[[], None]]:
    from vllm.distributed.parallel_state import get_tp_group

    try:
        group = get_tp_group()
    except Exception:  # noqa: BLE001 - no TP group yet (early calls)
        return None
    comm = roce_comm_of(getattr(group, "device_communicator", None))
    return comm.check_health if comm is not None else None


def make_checked_async_output_class(base):
    """Subclass of vLLM's ``AsyncModelRunnerOutput`` whose ``get_output`` runs the check
    after the copy to host completes (#597), forwarding other attributes (tonyd2wild)."""

    class GlmRoceCheckedAsyncOutput(base):
        def __init__(self, inner, check):
            self._inner = inner
            self._check = check

        def get_output(self):
            output = self._inner.get_output()
            self._check()
            return output

        def __getattr__(self, name):
            if name in ("_inner", "_check"):
                raise AttributeError(name)
            return getattr(self._inner, name)

    return GlmRoceCheckedAsyncOutput


def make_guard(async_base, passthrough_types: tuple, check_lookup=_tp_health_check):
    wrapper_cls = make_checked_async_output_class(async_base)

    def guard(output):
        if passthrough_types and isinstance(output, passthrough_types):
            return output  # pipeline-parallel intermediate tensors: not a step result
        check = check_lookup()
        if check is None:
            return output
        if isinstance(output, async_base):
            return wrapper_cls(output, check)
        check()
        return output

    guard.wrapper_cls = wrapper_cls
    return guard


def patch_gpu_worker(module, check_lookup=_tp_health_check) -> None:
    cls = module.Worker
    if getattr(cls, _PATCHED, False):
        return
    from vllm.v1.outputs import AsyncModelRunnerOutput

    try:
        from vllm.sequence import IntermediateTensors

        passthrough = (IntermediateTensors,)
    except Exception:  # noqa: BLE001
        passthrough = ()
    guard = make_guard(AsyncModelRunnerOutput, passthrough, check_lookup)
    _wrap_worker(cls, guard)
    logger.debug("GLM_ROCE patched %s.Worker", module.__name__)


def _wrap_worker(cls, guard) -> None:
    orig_execute = cls.execute_model
    orig_sample = cls.sample_tokens

    @functools.wraps(orig_execute)
    def execute_model(self, *args, **kwargs):
        return guard(orig_execute(self, *args, **kwargs))

    @functools.wraps(orig_sample)
    def sample_tokens(self, *args, **kwargs):
        return guard(orig_sample(self, *args, **kwargs))

    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    cls._glm_roce_guard = staticmethod(guard)
    setattr(cls, _PATCHED, True)


PATCHES = {
    MOD_CUDA_COMM: patch_cuda_communicator,
    MOD_PARALLEL_STATE: patch_parallel_state,
    MOD_GPU_WORKER: patch_gpu_worker,
}


# -- build-time check against the real image -------------------------------------------


def _params(fn) -> list[str]:
    return list(inspect.signature(inspect.unwrap(fn)).parameters)


def verify_targets() -> list[str]:
    """Import the three vLLM modules (CPU is enough) and return every mismatch found."""
    import importlib

    problems: list[str] = []

    def need(cond: bool, what: str) -> None:
        if not cond:
            problems.append(what)

    cc = importlib.import_module(MOD_CUDA_COMM)
    cls = getattr(cc, "CudaCommunicator", None)
    need(cls is not None, "CudaCommunicator missing")
    if cls is not None:
        p = _params(cls.__init__)
        for name in ("cpu_group", "device", "device_group", "unique_name"):
            need(name in p, f"CudaCommunicator.__init__ lacks {name}: {p}")
        need(_params(cls.all_reduce)[:2] == ["self", "input_"], f"all_reduce{_params(cls.all_reduce)}")
        need(_params(cls.all_gather)[:3] == ["self", "input_", "dim"], f"all_gather{_params(cls.all_gather)}")
        need(callable(getattr(cls, "destroy", None)), "CudaCommunicator.destroy missing")
        # The wrapper reads these after the stock __init__; they are set by the
        # base class (cpu_group, device_group, unique_name, world_size, device).
        src = "".join(
            inspect.getsource(inspect.unwrap(k.__init__))
            for k in cls.__mro__
            if "__init__" in vars(k) and k is not object
        )
        for attr in ("cpu_group", "device_group", "unique_name", "world_size", "device"):
            need(f"self.{attr} = " in src, f"no CudaCommunicator base sets self.{attr}")
    ps = importlib.import_module(MOD_PARALLEL_STATE)
    gc = getattr(getattr(ps, "GroupCoordinator", None), "graph_capture", None)
    need(gc is not None, "GroupCoordinator.graph_capture missing")
    if gc is not None:
        need(_params(gc)[:2] == ["self", "graph_capture_context"], f"graph_capture{_params(gc)}")
    need(callable(getattr(ps, "in_the_same_node_as", None)), "in_the_same_node_as missing")
    need(callable(getattr(ps, "get_tp_group", None)), "get_tp_group missing")
    need(callable(getattr(ps, "graph_capture", None)), "module graph_capture() missing")
    gw = importlib.import_module(MOD_GPU_WORKER)
    worker = getattr(gw, "Worker", None)
    need(worker is not None, "gpu_worker.Worker missing")
    if worker is not None:
        need(callable(getattr(worker, "execute_model", None)), "Worker.execute_model missing")
        need(callable(getattr(worker, "sample_tokens", None)), "Worker.sample_tokens missing")
    outputs = importlib.import_module("vllm.v1.outputs")
    need(hasattr(outputs, "AsyncModelRunnerOutput"), "vllm.v1.outputs.AsyncModelRunnerOutput missing")
    return problems
