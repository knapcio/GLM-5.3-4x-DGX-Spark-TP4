# Exact probabilistic native MTP experiment

`GLM_SPEC_SAMPLE=0` (default) preserves the current greedy draft mode. Boot with
`GLM_SPEC_SAMPLE=1` for native MTP probabilistic drafts on T>0 requests and an
independent residual random stream. T=0 rows use stock no-noise argmax semantics.
The flag is forwarded to every rank, with no CLI/config argument delta. No draft
top-p or tau knob is added. This is experimental and not GPU-qualified.

The startup overlay pins vLLM 487ecf187 sources from base image 4def0ef6. It sets
`SpeculativeConfig.draft_sample_method` before the original post-init validation,
and replaces only `_resample_kernel`; the copied helper adds a conditional seed
salt. All residual arithmetic is preserved. Other source modules are checked
on import. Source drift and unsupported proposer/rejection modes stop startup.

Accepted token mass is `min(p(x), q(x))`; independently sampling the normalized
residual `max(p-q, 0)` supplies the remaining target mass. The conditional seed
salt separates residual noise from draft noise. Kernel distributions and token
identity still require GPU qualification. Combining this option with a separate
draft early-stop overlay requires independent integration and qualification.

Validation scripts:

* `scripts/compare_spec_sample_dry.py`: no-contact launch-vector A/B.
* `tests/run_spec_sample_cpu.sh`: guarded offline pinned-image Torch Monte Carlo,
  exactness identity, shared-noise negative control, structural kernel comparison,
  pins, cold imports and driver-free sm_121 compile matrix.
* `tests/gpu/spec_sample_distribution.py`: later one-GPU real-kernel distributions
  and committed T=0 token/length identity. It is not a TP4 serving test.

Run the CPU suite before the single-GPU gate, then compare serving output token
IDs for T=0 and mixed T=0/T>0 traffic under identical prompts, seeds and launch
arguments. Host Torch tests alone do not qualify CUDA execution or serving
token identity.
