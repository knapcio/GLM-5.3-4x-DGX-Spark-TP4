# Draft-only NVFP4 LM head

On by default: `GLM_DRAFT_HEAD=nvfp4`, `GLM_DRAFT_HEAD_INIT=1` (`profiles/current.env`). Source:
`overlay/bringup/glm_draft_head.py`, `glm_draft_head_config.py`, `glm_draft_head_qual.py`. To boot without it,
set both `GLM_DRAFT_HEAD=0` and `GLM_DRAFT_HEAD_INIT=0`.

## What it does

GLM-5.3's native MTP layer shares the target's BF16 LM head (vocabulary 154,880 x 6,144, one 38,720-row shard per
rank). Every MTP draft step reads that whole BF16 shard.
The draft-only head gives the MTP draft its own **NVFP4 copy** of the head shard, run by vLLM's Marlin W4A16
kernel. It costs 133.8 MB per rank (118.9 MB packed weights, 14.9 MB scales).

- The **target keeps the BF16 head** and its logits processor; neither is patched. The unchanged target model,
  whose LM head remains BF16, verifies drafts through standard speculative verification, so a draft token the
  quantized head gets wrong is rejected. This does not establish bitwise-identical output across runs.
- The quantized head can change draft proposals. Committed tokens per cycle were consistent with unchanged on the
  measured panels; the gain comes from a 4-5 % shorter decode cycle.
- No new CUDA kernel. The bank is built at load from the BF16 head; the draft CUDA graphs are captured with it.

## INIT qualification at boot

`GLM_DRAFT_HEAD_INIT=1` captures the draft graphs with the head ON and qualifies them on all four ranks before
the engine reports ready. Every draft graph descriptor is replayed against an eager run on synthetic inputs:

- **Write coverage:** every active draft token and every prefill hidden element must overwrite a sentinel value,
  on every descriptor and every execution.
- **Exact draft tokens:** eager and graph executions must produce identical draft tokens.
- **Bounded floats:** hidden feedback within 16 BF16 steps with at most 1/64 of elements differing, confidence
  within 1e-5, logits within 1/16; all values finite.
- The target head is checked bit-exact, and memory stays above the live floor.

All ranks vote. If any rank refuses, **all ranks fall back** to the BF16 draft head and the boot continues with
the recipe otherwise unchanged; a refusal is logged with its receipts. Runtime head switching is unsupported in the
qualified serving procedure.

On the release boot all four ranks passed: 341 qualification rows per rank, 0 failed, 16 cases, head ON and ready,
target bit-exact.

## Why the floats are bounded, not exact

Earlier, stricter criteria (bit-exact eager vs graph) refused the head on some boots: the first execution at a
new input value sometimes differed by a few BF16 ULP in the draft hidden feedback, while draft tokens stayed
identical. Instrumented boots localised it:

- Everything before the routed experts (EH projection, attention, router, shared experts) was bit-stable.
- The first varying stage was the MTP layer's **routed-expert (Marlin MoE) output**.
- vLLM's `moe_align_block_size` writes token ids into each expert segment with `atomicAdd`, so the **order of tokens
  within an expert varies between calls** (the token sets are always identical).
- The Marlin MoE kernel partitions work by token block and reduces in fp32; a token landing in a different block
  is summed in a different order, giving a **few-ULP difference**. Each GEMM was bit-exact for a fixed order.
- The head itself is downstream of the variation, and turning off the L2 discard (`GLM_DIRTY_L2=0`) did not remove
  it. The same two components are used by the target's routed experts; that they also explain the known run-to-run
  variation of full GLM-5.3 at temperature 0 is an untested hypothesis.

Every order is a valid summation, and the target verifies every draft token through standard speculative
verification; the variation affects draft proposals, not the verification rule.
The INIT criterion therefore requires exact draft tokens and full write coverage, and bounds the float drift
instead of requiring bit equality. Making the expert token order canonical is expected to remove this source of
variation; GPU validation remains outstanding, and it is not part of this release.

## Measured

- Same-boot ON/OFF confirmation (October 7 boot, fresh 12-prompt panels): prose **+5.61 %** [+3.94, +7.47],
  code+structured **+4.43 %** [+3.17, +5.71]; committed tokens per cycle flat, cycle 3.9-4.7 % shorter.
- Release boot D (October 8): every gate passed; tables in the [README](../README.md) and
  [GATE-RESULT](../GATE-RESULT.md).

## Tests

`tests/test_draft_head.py`, `test_draft_head_init.py`, `test_draft_head_replay_criterion.py` and
`test_draft_head_leakgate.py` run on CPU with the pinned vLLM source. `tests/draft_head_*_gloo.py` run the
collective vote and fallback on four CPU Gloo ranks; `draft_head_init_native_cpu.py`, `draft_head_init_import.py`
and `draft_head_native_prepare.py` exercise the native capture, prefill and head-scheme paths.
`tests/diagnose_draft_head_init.py` extracts a saved per-rank refusal from boot logs.
