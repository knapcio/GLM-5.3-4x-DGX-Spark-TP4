# Temperature-0 reproducibility: deterministic MoE alignment

On by default: `GLM_MOE_DET_ALIGN=1` (`profiles/current.env`). Source: `overlay/bringup/glm_moe_det.py` and
`overlay/bringup/det_align/` (CUDA kernel, launcher, test-only reference). To boot without it, set
`GLM_MOE_DET_ALIGN=0`.

## Problem

Before this change the full GLM-5.3 stack was not reproducible at temperature 0. With 30 fixed prompts, each sent
three times as byte-identical requests with the prefix cache reset before every attempt, only **4 of 30** prompts
produced identical output in all three repeats; 26 diverged, 15 of them already at the first generated token.

## Root cause

- vLLM's `moe_align_block_size` groups routed tokens by expert for the Marlin MoE kernel. Its CUDA kernel places each
  token with `atomicAdd` on the expert's counter, so the **order of tokens inside an expert segment** is the order in
  which threads happen to arrive. The set of tokens per expert is always the same; their order is not.
- The Marlin MoE kernel splits each expert's tokens into blocks (8 rows for the MTP layer) and reduces partial results
  in fp32. A token that lands in a different block is summed in a different order, which changes the result by a few
  ULP. In the instrumented runs each Marlin GEMM was bit-exact for a fixed token order.
- Instrumented boots showed every stage before the routed experts bit-stable, and the routed-expert output as the
  first stage that varied; calling `moe_align_block_size` twice on identical routing gave a different token order in
  16-20 of 20 calls.
- The target's routed experts and the native MTP layer both use this path. A few-ULP difference can flip a near-tie
  token at temperature 0; replacing the align step removed the observed divergence in the tested panel.

## Fix

`GLM_MOE_DET_ALIGN=1` replaces the align step, for the target and the native MTP Marlin MoE, with a counting-sort
kernel written for this recipe (`det_align/kernel.cu`). It returns exactly the stock layout (padded expert segments,
block labels, padded token count, sentinel padding) with **token ids ascending inside each expert segment**:

- an entry's position is the number of smaller flat indices routed to the same expert, computed with warp
  `match_any` peer masks and CUB block scans; no placement atomics, sorting or floating-point arithmetic;
- one CTA for decode sizes (up to 128 routed entries), one compact CTA up to 1024, and a tiled
  histogram / prefix / scatter path above that;
- every output slot is rewritten on every call; scratch is a fixed 688,004-byte arena per device and stream, so CUDA
  graph replays allocate nothing.

The kernel is compiled once per boot with the image's `nvcc` (`-O3 -std=c++17 -arch=sm_121`) and loaded through a
small C ABI. The hook is source-pinned to the image's `moe_align_block_size`, Marlin MoE and model-runner files and
refuses batched/LoRA Marlin. Each rank logs a `glm-moe-det: installs=2 ...` receipt at load.

Single-GPU microbenchmark on GB10, graph mode: 2.1-2.6 µs per call at decode sizes M1-M16 against 4.4-4.5 µs for
the stock align; 12.0 µs against 6.2 µs at M512 (prefill only). 181 layout, expert-parallel-map and narrowing
checks against the stock layout passed on the GPU; that benchmark's own speed gate failed because of the M512 cell.

## Measured (October 9, 2026)

| Check | Result |
|---|---|
| Temperature-0 probe, 30 prompts x 3 sequential repeats, 64 tokens, prefix cache reset before each (qualification boot, same code and configuration as boot E) | **30 / 30 identical**, 0 first-token flips (stock align, October 8 boot: 4 / 30, 15 first-token flips) |
| qeval x3, boot E | 73 / 73 / 73, the same two tasks failing in every run |
| Paired cycle, boot E vs the previous configuration on other boots the same night | prose 70.0 ms vs 70.7-70.8 ms; code 78.1 ms vs 78.5-79.2 ms |

30 of 30 tested prompts produced identical outputs across three sequential repeats at temperature 0, with
`max_tokens` 64 and prefix-cache resets; requests were byte-identical within each prompt's repeats. This supports the
diagnosed MoE-order explanation. Reproducibility with concurrent requests and across boots remains unqualified.

## How to verify

`bench/t0_probe.py` (standard library only) sends a fixed 30-prompt panel three times at temperature 0, resets the
prefix cache before every attempt and compares the outputs. Run it on the rank-0 host against an idle server; the
cache reset needs `VLLM_SERVER_DEV_MODE=1`, which the draft head already enables.

```bash
python3 bench/t0_probe.py --base-url http://127.0.0.1:8095/v1 --det-align ON --boot-label my-boot \
    --apc-cold --out-dir results/t0-on
```

It writes `RESULT.md` and `result.json` (prompts, request hashes, outputs, first divergence, prefix-cache hit deltas).
For the ON / cold command above, exit code 0 means 30 / 30 identical rows with no prefix-cache hits (OFF, warm and
dry-run modes also return 0 when they complete). `--dry-run` prints the plan without any HTTP call;
`python3 tests/test_t0_probe.py` tests the probe against a fake server. The `--det-align` value is only a label.

## Tests

`tests/test_moe_det.py` (CPU, pinned vLLM source) proves the ranking and one-writer ownership against an independent
reference of the stock layout, runs the pinned stock wrapper with a reference op, checks the source pins, the hook
installation on target and MTP sites, the launcher switch, and the pinned CUDA graph manager's capture streams.
