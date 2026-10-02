"""``GlmRoceAllReduce``: route eligible TP collectives to ``b12x.comm.roce`` (RoCEnante).

Port of ``vllm/distributed/device_communicators/b12x_roce_all_reduce.py`` from
local-inference-lab/vllm#597 (Jason Cook, @original-el8; merged 2026-09-03 as
ba233a68), by way of tonyd2wild's port to the ``sm121-v11-dflash2`` tree.
Runtime: RoCEnante, local-inference-lab/b12x#295 (Luke Alonso, Jason Cook).

Changes against #597:
- Environment names are ``GLM_ROCE_*`` and nothing is added to ``vllm.envs``;
  ``_parse_byte_size`` is inlined (the image has no ``b12x_pcie_all_reduce.py``).
- ``GLM_ROCE_GATHER_MAX_SIZE=0`` keeps all-gathers on NCCL (tonyd2wild's
  ``VLLM_ROCE_ALLGATHER_ENABLE``) and also shrinks the pinned slots to the
  all-reduce size.  It travels with the vote, so every rank agrees.
- ``GLM_ROCE_REQUIRE`` (default 1): if the vote disables the backend, every rank
  raises at start-up instead of silently serving on NCCL.  The verdict is shared,
  so all ranks raise together.  Set 0 for #597's log-and-continue behaviour.
- Launchers and the alignment scratch are prepared right after construction,
  followed by a barrier, so a rank with a cold compile cache cannot make its
  peers spin in a RoCE wait during the eager profile run.
- One READY line per rank and one route line per rank (not rank 0 only): the
  four ranks run in four containers on four hosts, each with its own log.

Contract with the runtime (unchanged from #597): capability and limits are voted
over the gloo group before the collective constructor; dispatch depends only on
dtype, shape, contiguity and size (rank-invariant); failures are fail-stop,
never a fallback (``check_health`` after each step's host synchronization).
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# A child of the "vllm" logger, so vLLM's handlers print it.
logger = logging.getLogger("vllm.glm_roce")

REQUIRED_B12X_ROCE_API_VERSION = 1
BACKEND_NAME = "GLM_ROCENANTE"

ENV_MAX_SIZE = "GLM_ROCE_MAX_SIZE"
ENV_GATHER_MAX_SIZE = "GLM_ROCE_GATHER_MAX_SIZE"
ENV_REQUIRE = "GLM_ROCE_REQUIRE"
# Sizing (GLM-5.3-Flash, hidden 4096, bf16): a TP all-reduce is 8 KiB per token, so c16 x k7
# (128 verify rows) is 1 MiB and c32 x k7 (256 rows) 2 MiB; 4 MiB leaves room for fp32
# all-reduces and bigger mixed batches.  Prefill chunks (4096 tokens = 32 MiB) stay on NCCL.
# The logits all-gather shard is rows x 38,720 x 2 B (bf16 head): 128 rows = 9.9 MB fit 16 MiB;
# 256 rows (19.8 MB) or an fp32 head fall back to NCCL by the size gate, on every rank alike.
# Pinned host memory per rank = 10 x max(all-reduce, gather) at TP4 = 160 MiB with these values.
DEFAULT_MAX_SIZE = "4MiB"
DEFAULT_GATHER_MAX_SIZE = "16MiB"
PACK_BYTES = 16


def _parse_byte_size(value: str) -> int:
    """Parse "2MB", "84KB", "16MiB", "4096" into bytes (binary multiples).

    Copied from local-inference-lab/vllm
    ``vllm/distributed/device_communicators/b12x_pcie_all_reduce.py`` (via
    tonyd2wild's port).
    """
    match = re.fullmatch(r"\s*([+-]?\d+)\s*([kmgt]?i?b?)?\s*", str(value).lower())
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    amount = int(match.group(1))
    suffix = match.group(2) or ""
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1 << 10,
        "kb": 1 << 10,
        "kib": 1 << 10,
        "m": 1 << 20,
        "mb": 1 << 20,
        "mib": 1 << 20,
        "g": 1 << 30,
        "gb": 1 << 30,
        "gib": 1 << 30,
        "t": 1 << 40,
        "tb": 1 << 40,
        "tib": 1 << 40,
    }
    try:
        return amount * multipliers[suffix]
    except KeyError as exc:
        raise ValueError(f"invalid byte-size suffix: {suffix!r}") from exc


def read_limits(environ: Optional[dict] = None) -> tuple[int, int]:
    """``(all-reduce max bytes, all-gather max bytes per rank)`` from the environment.

    Raises ``ValueError`` on an unparsable or unusable value; the caller reports it
    through the vote so every rank sees the same verdict.
    """
    env = os.environ if environ is None else environ
    max_size = _parse_byte_size(env.get(ENV_MAX_SIZE, DEFAULT_MAX_SIZE))
    max_gather = _parse_byte_size(env.get(ENV_GATHER_MAX_SIZE, DEFAULT_GATHER_MAX_SIZE))
    if max_size < PACK_BYTES or max_size % PACK_BYTES:
        raise ValueError(f"{ENV_MAX_SIZE}={max_size} must be a positive multiple of 16 bytes")
    if max_gather < 0 or max_gather % PACK_BYTES:
        raise ValueError(f"{ENV_GATHER_MAX_SIZE}={max_gather} must be 0 or a multiple of 16 bytes")
    return max_size, max_gather


def require_enabled(environ: Optional[dict] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(ENV_REQUIRE, "1").strip() != "0"


def _in_the_same_node_as(group: ProcessGroup, source_rank: int = 0) -> list[bool]:
    """vLLM's collective same-node probe (indirection so CPU tests can replace it)."""
    from vllm.distributed.parallel_state import in_the_same_node_as

    return in_the_same_node_as(group, source_rank=source_rank)


def _import_roce():
    from b12x.comm import roce

    return roce


class GlmRoceAllReduce:
    """Route eligible tensor-parallel all-reduces and all-gathers to ``b12x.comm.roce``."""

    backend_name = BACKEND_NAME

    def __init__(
        self,
        group: ProcessGroup,
        device_group: Optional[ProcessGroup],
        device: torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_size = 0
        self.max_gather = 0
        self._runtime = None
        self._announced = False
        self._announced_gather = False
        self._health_logged = False
        self.verdict: Optional[str] = None

        if device_group is None:
            self._give_up("RoCEnante requires a CUDA process group")
            return
        if all(_in_the_same_node_as(group, source_rank=0)):
            # Single-node group: nothing to do, and not an error.
            logger.info("GLM_ROCE skipped: the TP group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take part
        # disables the backend on every rank instead of stranding its peers in the
        # runtime's setup exchange.  The parsed limits travel with the vote.
        reason, limits = self._local_capability()
        verdict = self._exchange_vote(reason, limits)
        if verdict is not None:
            self._give_up(f"disabled on every rank: {verdict}")
            return
        max_size, max_gather = limits
        roce = _import_roce()
        try:
            # Setup exchange over the CPU (gloo) group: the torch NCCL group would
            # build a second NCCL communicator (~3.4 GB of unified memory per rank
            # on Spark, measured by #597's author).
            self._runtime = roce.AllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                max_size=max_size,
                max_gather_bytes=max_gather,
            )
            # Compile launchers and allocate scratch now (the runtime refuses both
            # inside a capture), then line the ranks up again: a cold compile cache
            # on one node must not leave its peers spinning in a RoCE wait.
            self._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
            error = None
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = f"rank {self.rank}: {exc}"
        errors = [e for e in self._gather_objects(error) if e]
        if errors:
            # The runtime coordinates its own failures; the prepare step may fail
            # on one rank only, so agree before deciding.
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self._give_up("initialization failed: " + "; ".join(errors))
            return
        self.max_size, self.max_gather = max_size, max_gather
        self.disabled = False
        stats = self._runtime.stats() if hasattr(self._runtime, "stats") else {}
        logger.info(
            "GLM_ROCE_READY rank=%d world=%d hcas=%s all_reduce_max=%d all_gather_max=%d "
            "spin_limit=%s %s",
            self.rank,
            self.world_size,
            ",".join(getattr(self._runtime, "hca_names", ()) or ()),
            max_size,
            max_gather,
            stats.get("spin_limit", "?"),
            json.dumps(stats, sort_keys=True, default=str),
        )

    # -- setup ---------------------------------------------------------------

    def _give_up(self, why: str) -> None:
        self.verdict = why
        self.disabled = True
        if require_enabled():
            raise RuntimeError(
                f"GLM_ROCE_ALLREDUCE=1 but RoCEnante is {why} "
                "(set GLM_ROCE_REQUIRE=0 to serve on NCCL instead)"
            )
        logger.warning("GLM_ROCE %s; collectives stay on NCCL.", why)

    def _local_capability(self) -> tuple[Optional[str], Optional[tuple[int, int]]]:
        """This rank's reason for not taking part (None when it can) and its parsed limits."""
        try:
            roce = _import_roce()
        except Exception as exc:  # noqa: BLE001 - missing package or broken build
            return f"b12x.comm.roce is not importable: {exc}", None
        api = getattr(roce, "API_VERSION", None)
        if api != REQUIRED_B12X_ROCE_API_VERSION:
            return (
                f"b12x.comm.roce API version {api}, adapter needs "
                f"{REQUIRED_B12X_ROCE_API_VERSION}"
            ), None
        try:
            supported = roce.is_supported(self.device)
        except Exception as exc:  # noqa: BLE001
            return f"b12x.comm.roce.is_supported raised: {exc}", None
        if not supported:
            return "needs an integrated GPU with an active RDMA device", None
        try:
            limits = read_limits()
        except Exception as exc:  # noqa: BLE001
            return f"invalid RoCEnante size limit: {exc}", None
        return None, limits

    def _gather_objects(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj, group=self.group)
        return out

    def _exchange_vote(
        self, reason: Optional[str], limits: Optional[tuple[int, int]]
    ) -> Optional[str]:
        """None when every rank can proceed with identical limits, else why not."""
        votes = self._gather_objects((reason, limits))
        failures = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if failures:
            return "; ".join(failures)
        reference = votes[0][1]
        differing = [f"rank {i}: {lim}" for i, (_, lim) in enumerate(votes) if lim != reference]
        if differing:
            return (
                f"size limits differ across ranks (rank 0: {reference}; "
                + "; ".join(differing)
                + ")"
            )
        return None

    # -- health ----------------------------------------------------------------

    def check_health(self) -> None:
        """Fail-stop check: raises when a RoCEnante wait timed out or the proxy died."""
        if self.disabled or self._runtime is None:
            return
        self._runtime.check_health()
        if not self._health_logged:
            self._health_logged = True
            logger.info("GLM_ROCE_HEALTH rank=%d first guarded step ok", self.rank)

    def stats(self) -> dict:
        if self._runtime is None:
            return {}
        return self._runtime.stats()

    # -- all-reduce ----------------------------------------------------------------

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            logger.info(
                "GLM_ROCE_ROUTE all_reduce rank=%d first=%d bytes (%s); NCCL above %d bytes",
                self.rank,
                inp.numel() * inp.element_size(),
                str(inp.dtype).replace("torch.", ""),
                self.max_size,
            )
        return self._runtime.all_reduce(inp)

    # -- all-gather ----------------------------------------------------------------

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return (
            not self.disabled
            and self.max_gather > 0
            and self._runtime.should_all_gather(inp, dim)
        )

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate along ``dim`` (0 or last), written directly in the output layout."""
        if not self._announced_gather:
            self._announced_gather = True
            logger.info(
                "GLM_ROCE_ROUTE all_gather rank=%d first shard=%s %s dim=%d",
                self.rank,
                tuple(inp.shape),
                str(inp.dtype).replace("torch.", ""),
                dim,
            )
        return self._runtime.all_gather(inp, dim=dim)

    # -- capture / lifecycle ---------------------------------------------------------

    @contextmanager
    def capture(self, stream: Optional[torch.cuda.Stream] = None):
        """Enter around vLLM's graph capture: re-prepare (a no-op when warm) and let the
        runtime drop its eager cross-stream ordering event for the capture."""
        if self.disabled:
            yield
            return
        self._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
        with self._runtime.capture(stream=stream):
            yield

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True
