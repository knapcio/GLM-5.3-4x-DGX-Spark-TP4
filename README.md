# Full GLM-5.3 on 4x NVIDIA DGX Spark

## sparkDash (thinking off)

Decode tok/s, aggregate **[per stream]**, 256 output tokens, temperature 0.
Report medians across every scored run: c1 five, c2/c4 three, c8 two; discard only named warmups.
Measurements belong to the exact release configuration and its October 10 gen-12 boot. This was one complete
262,144-context sweep with no failed jobs and no best-of selection; see [the gate result](GATE-RESULT.md).

| Prompt type | c1 | c2 | c4 | c8* |
|---|---:|---:|---:|---:|
| prose | 39.38 | 50.90 [25.97] | 65.89 [17.05] | 61.31 [18.09] |
| code | 45.74 | 54.39 [27.19] | 72.94 [18.38] | 62.78 [19.26] |
| structured | 48.66 | 67.04 [33.52] | 92.22 [24.20] | 72.64 [24.32] |
| json | 45.54 | 55.26 [28.40] | 78.98 [20.22] | 70.69 [21.34] |

*Four serving slots: c8 queues four requests. Per-stream decode covers active generation;
aggregate includes batch wall time. sparkDash repeats one prompt across streams; memory stress uses distinct prompts.

Cold prefill, median of three, with prefix-cache state recorded separately:

| Input | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 977 | 882 | 862 | 881 |
| time to first token (s) | 4.22 | 9.32 | 19.04 | 37.23 |

## Current release configuration

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B), tensor parallel across four DGX Sparks,
with native MTP speculative decoding, FP4x KV, NVFP4 weight sidecars and **262,144-token context**.
One shared **264,640-token KV pool (4,135 blocks)** spans the ranks; four simultaneous maximum-context requests
do not fit. The ordinary KV head is 6,318,718,976 bytes per rank, plus the display carveout.
The recent-FP8 bank is disabled. The fast loader is the default.

Deterministic MoE alignment, the draft-only NVFP4 LM head, **FP8 draft eh_proj**, and **glue-lite F1+F2** are on.
Native confidence stopping retains K3 at c1 and K2 in multi-request batches. Decode/prefill time-slicing remains
4096/N40. Row selection is off, F3 index caching is off, and cache trim and async v2 are absent from this export.
Split32 MLA and the existing rotary storage retain the public release layout.
This configuration passed the owner-approved reduced release gate on October 10, 2026 and is serving under the
watchdog. The exact scope, carried evidence and deliberately skipped full-gate phases are recorded in
[GATE-RESULT](GATE-RESULT.md).

## Draft projection and launch overhead

The native MTP draft's replicated 6144x12288 `eh_proj` uses FP8 W8A16 Marlin with per-output-channel BF16 scales.
The target model stays unchanged and verifies drafts using vLLM speculative verification.
Combined INIT checks qualify the projection and draft head on all four ranks before readiness; a collective refusal
restores the native BF16 projection and disables the draft head, which fails this candidate's acceptance gate.
[eh_proj details](docs/eh_proj.md).

Glue-lite F1 feeds BF16 router logits directly into vLLM's widening grouped-topk path; F2 reuses the Marlin MoE
workspace after the kernel's existing lock reset. The glue defaults are `GLM_GLUE_ROUTER_BF16=1`,
`GLM_GLUE_MOE_WS=1`, `GLM_GLUE_DSA_IDX_CACHE=0`, `GLM_GLUE_IDX_EXPECT=0`, and
`GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1`; the zero index expectation matches F3 being off.
[Glue-lite details and exactness boundaries](docs/glue-lite.md).

## Reproducibility and responsiveness

Deterministic MoE alignment keeps token ids ascending within each expert, avoiding atomic-arrival-dependent
Marlin reductions. Single-request temperature-0 repeatability was 30/30 across three sequential runs on the
release boot. Concurrent repeatability remains unqualified.
[Determinism](docs/determinism.md), `bench/t0_probe.py`.

Decode/prefill time-slicing runs 40 pure-decode steps between mixed prefill chunks capped at 4096 tokens,
keeping a decoding request responsive during another request's long prefill. A lone prefill retains adaptive chunks:
2048 below 16,384 total prompt tokens, otherwise 4096. This trades the second request's TTFT for decode responsiveness.
The optional `GLM_DECODE_FAIR_CONTROL` sidecar supports runtime changes; the default boot needs no sidecar.
[Time-slicing](docs/time-slicing.md) explains boot options and emergency control.

## Quality and capacity

The reduced gate ran boot/admission, receipt and INIT checks, temperature-0 repeats, full sparkDash, RigMark,
stress, a short mixed-load soak, one 128K needle, collection and handover/verification. Qpanel and qeval
evidence, 16K needles and glue exactness were carried from the source runs named below. The time-slicing
scenario, qeval x3, 250K needles and long soak were not run on this release boot.
[Receipt paths and SHA256 provenance](docs/release-1010-gate.md#recorded-reduced-gate-and-cell-provenance).

| Check | Release result |
|---|---|
| Quality admission; qeval status | Carried qpanel/B (glm53full-cand3win7-w4ehproj-6): 110/116, 2 truncations; decision PASS. carried w4-ehproj-g5: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision KILL; carried w4-ehproj-g6: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision INFORMATIONAL (release_decision KILL). Qeval x3 not run on release boot. |
| Needle 16K, carried twice | Carried w4-ehproj-g6: 8/10, 8/10; not rerun on release boot. |
| Needle 128K, one run | 1 run, 9/10; missed `matching_count`; TTFT 170.86024899967015 s; PASS under the reduced rule. |
| Needle 250K | Not run; no carried 250K receipt in the reduced scope. |
| c4 shared-pool and single maximum-context stress | PASS; c4 65024+1024 each, single 261120+1024; rank 0/1/2/3 minima 5.808849334716797/6.767314910888672/7.456321716308594/7.352359771728516 GiB; swap maxima 0/0/0/0 KiB; preemptions 0.0. |
| Prefix-cache first/repeat and retained-cache stress | 8K cold/replay prefill 904.667/16114.311 tok/s; stress APC hits/queries 64896.0/65024.0 tokens; retained-cache quiet minima 5.89/6.836/7.515/7.398 GiB (ranks 0/1/2/3). |
| Mixed-load soak | PASS; requested 15.0 min, observed 1071.0947126080282 s; 27 requests, 0 errors, 0.0 preemptions; quiet drift -0.022/0.006/-0.024/0.017 GiB (ranks 0/1/2/3). |
| Draft-head + eh_proj INIT | 4/4 ranks ON/ready, combined FP8 eh_proj qualified; 16 cases and 342/342/342/342 rows per rank; 0 failed; target head bit-exact on every rank. |
| Temperature-0 sequential repeats on this boot | 30/30 prompts identical across 3 repeats; 30/30 native-ID equality to carried stack-g2/glue-t0. |
| RigMark 1.0.0, thinking on, effort low | prose 30.051 tok/s; code 42.372 tok/s; structured 47.338 tok/s; code c4 aggregate 67.904 tok/s; 8K cold prefill 904.667 tok/s; all workload gates passed. |

These are operational screens; multi-day stability remains unqualified.

## Memory safety

On every rank, the floors are **4.5 GiB live and release stress**, **6.0 GiB capture**,
**6.5 GiB admission for 60 s**, and **7.5 GiB before capture**.
**knapcio selected the 4.5 GiB stress criterion on October 10, 2026**, matching the live floor.
This changes only stress acceptance; capture, admission, swap and preemption checks remain in force.
Stress must report the observed minimum on each rank, rather than substituting the threshold as a result.
The watchdog applies the live floor, sustained swap/error guards, 600 s busy-without-progress allowance
with idle reset, and an 1800 s boot deadline. APC stays on. [Memory layout](docs/memory-1006.md).

## Levers and credit

| Piece | What it does | Switch | Credit |
|---|---|---|---|
| Native MTP with K-stop | Confidence stop at c1; K2 batches; graphs [1,4,12,16] | `GLM_MTP_KSTOP=1` | Z.ai; vLLM; SpecDec++/DISCO ideas; DeepSeek verify-cap idea; keys/@u1tra_instinct TensorFold results; knapcio implementation |
| Display carveout KV | 6,318,718,976 B ordinary head plus 2046 MiB external carveout | `RECIPE_DISPRAM=require` | [kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram), external unmodified AGPL-3.0; NVIDIA; vLLM |
| FP4x latent KV | Packed latent rows; RoPE and index keys remain FP8 | `GLM_KV_FORMAT=fp4x` | Mia (MiaAI-Lab), FP4 KV idea only, no code copied; knapcio implementation; NVIDIA/Triton/vLLM |
| Adaptive prefill | 2048/4096 chunk budget selected from total prompt context | `GLM_PREFILL_CHUNK_ADAPTIVE=1` | vLLM scheduler contributors; knapcio |
| Deterministic MoE align | Counting-sort replacement for `moe_align_block_size` on target and MTP Marlin MoE; token ids ascending per expert; single-request temperature-0 output reproducible in the tested panel | `GLM_MOE_DET_ALIGN=1` | vLLM `moe_align_block_size` authors (layout contract) and Marlin MoE via vLLM; NVIDIA CUDA/CUB; knapcio kernel |
| Draft-only NVFP4 LM head | MTP draft uses an NVFP4 Marlin copy of the LM head; BF16 target head verifies; boot-time all-rank INIT qualification with BF16 fallback | `GLM_DRAFT_HEAD=nvfp4`, `GLM_DRAFT_HEAD_INIT=1` | Marlin W4A16 kernels (IST-DASLab) and NVIDIA NVFP4 format via vLLM; vLLM contributors (MTP speculator, CUDA graphs, Marlin integration, `moe_align_block_size`); Z.ai native MTP; knapcio |
| Decode/prefill time-slicing | While a request decodes: prefill capped at 4096 tokens per step, then 40 pure-decode steps | `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40` | vLLM chunked-prefill scheduler contributors; Sarathi-Serve stall-free batching (Agrawal et al., prior work, no code used); knapcio |
| NVFP4 sidecars | 385 attention +225 shared +6 dense +776 MTP matrices, Marlin W4A16 | `GLM_ATTN_WEIGHTS=nvfp4`, `GLM_NVFP4_GROUPS=attn,shared,dense,mtp` | Marlin kernels via vLLM; NVIDIA NVFP4 format; vLLM quantization contributors; Tech2wild/tonyd2wild checkpoint; Z.ai; knapcio converters |
| Prefix caching | Reuses full KV blocks of repeated prefixes | `--enable-prefix-caching` | vLLM |
| Short-context DSA | All-selected short contexts skip unnecessary indexer work | `GLM_INDEXER_SHORTCUT=1` | Guess-Verify-Refine idea; DeepSeek DSA; vLLM |
| Pad hygiene and c2 reuse | Padded rows reuse live experts; c2 replays existing exact graph | `GLM_PAD_HYGIENE=1`, `GLM_MTP_KSTOP_CAPTURE_LAYOUT=reuse` | vLLM MoE/graph contributors; knapcio |
| FP8 draft eh_proj | Replicated draft-only projection; combined INIT qualification | `GLM_DRAFT_EHPROJ=fp8`, `GLM_DRAFT_EHPROJ_INIT=1` | knapcio; vLLM FP8 Marlin contributors; NVIDIA FP8; PyTorch |
| Glue-lite F1+F2 | Remove router cast and repeated MoE workspace reset; token-identical panel | `GLM_GLUE_ROUTER_BF16=1`, `GLM_GLUE_MOE_WS=1`; F3 off | knapcio; vLLM grouped-topk and Marlin contributors |
| Sparse MLA and dirty L2 | GB10-compatible split32 MLA; discard consumed partials | `GLM_FULL_MLA=triton`, `GLM_DIRTY_L2=discard` | CosmicRaisins; Matt Mastracci ideas; NVIDIA PTX; Triton |
| Switched RoCE | RDMA TP collectives on both rails | `GLM_ROCE_ALLREDUCE=1` | b12x/RoCEnante; Luke Alonso; Jason Cook; tonyd2wild; rhys101 |
| Fast loading and shard selection | Bounded copies; MTP reads its five shards | `GLM_FAST_LOAD=1`, `GLM_MTP_ONLY_LOAD=1` | Willian-Zhang; vLLM; knapcio |
| CPU boot preflight | Cached-header check, default on only with a cache; warns and continues on failure | `RECIPE_BOOT_PREFLIGHT=1` | vLLM constructors/loader hooks; knapcio |
| Coalesced loader, opt-in | Bounded staging and destination-write memory credit | `GLM_LOADER=coalesced` (default fast) | Allan Clark / ajclark, Apache-2.0 modified port; original NOTICE retained |
| Recent-FP8 bank, opt-in | Reservation only when explicitly requested, default zero | `GLM_FP4_RECENT_WINDOW=0` | NVIDIA FP8; vLLM; knapcio |

[CREDITS](CREDITS.md), [NOTICE](NOTICE), [licence boundaries](LICENSES/README.md).
No weights, dispramd binary, or container image is distributed here.

## Install and run

Four ARM64 DGX Sparks, switched ConnectX-7 fabric configured with NVIDIA Sync, management access,
the pinned v11 ARM64 image, local checkpoint and sidecars on every rank, and NCCL 2.30.7 are required.
Fill the four hosts, fabric addresses, image IDs, paths and NCCL hashes before launching.
[Installation](docs/install.md), [sidecar conversion and hashes](docs/NVFP4-SIDECARS.md),
[service and recovery](service/README.md). Gate instructions and table-filling notes are in
[docs/release-1010-gate.md](docs/release-1010-gate.md).

```bash
cp .env.example .env
bash overlay/guard/build_guard.sh
DRY=1 ./start.sh serve
./start.sh preflight
./start.sh serve
./start.sh stop
```

Keep management recovery access working; no public router port forwarding is needed.
Thinking off uses `chat_template_kwargs: {"enable_thinking": false}`; RigMark uses effort low.
Tool parser `glm47`, reasoning parser `glm45`. The coalesced loader is opt-in (`GLM_LOADER=coalesced`).
