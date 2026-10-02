"""RoCEnante for the GLM-5.3-Flash vLLM TP4 image (opt-in: GLM_ROCE_ALLREDUCE=1).

Routes tensor-parallel all-reduces and all-gathers at decode sizes to the b12x
one-shot RoCE collectives; NCCL keeps everything else.  Nothing here runs unless
GLM_ROCE_ALLREDUCE=1 is set in the container environment.

Credits:
- RoCEnante (``b12x.comm.roce``): Local Inference Lab, Luke Alonso (@lukealonso)
  and Jason Cook (@original-el8), local-inference-lab/b12x#295 (Apache-2.0).
- vLLM shim this package ports: Jason Cook (@original-el8),
  local-inference-lab/vllm#597 (``B12xRoceAllReduce``, the vote, the worker
  health check).
- Port of #597 to the tonyd2wild ``sm121-v11-dflash2`` vLLM tree (the image this
  repository runs), including the all-gather switch and the async-output
  attribute forwarding: tonyd2wild,
  GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark ``speed-night-2026-09-18/roce``.
- SGLang TP8 overlay that showed the pattern on DeepSeek: rhys101 (SG17).

Layout:
- ``adapter``: ``GlmRoceAllReduce``, the per-TP-group runtime wrapper.
- ``install``: the monkeypatches into vLLM (communicator, graph capture, worker).
- ``boot``: env-gated post-import hook, loaded by ``glm_roce.pth``.
"""

__all__ = ["ENV_ENABLE", "enabled"]

ENV_ENABLE = "GLM_ROCE_ALLREDUCE"


def enabled() -> bool:
    """True when the container asked for RoCEnante (``GLM_ROCE_ALLREDUCE=1``)."""
    import os

    return os.environ.get(ENV_ENABLE, "0").strip() == "1"
