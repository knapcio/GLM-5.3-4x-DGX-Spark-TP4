# Full GLM-5.3 on 4x NVIDIA DGX Spark

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B), tensor parallel across four DGX Sparks,
with native MTP speculative decoding, FP4x KV, NVFP4 weight sidecars and **262,144-token context**.
The ranks implement one shared **264,640-token KV pool (4,135 blocks)**; four simultaneous maximum-context
requests do not fit. The ordinary KV head is 6,318,718,976 bytes per rank, plus the display carveout.
The recent-FP8 bank is disabled by default. The native MTP draft uses its own NVFP4 copy of the LM head, and
decode/prefill time-slicing keeps a decoding request responsive while another request prefills a long prompt.
Results are from one boot, D, measured October 8, 2026, unless a row says otherwise.
[Gate result](GATE-RESULT.md).

## sparkDash (thinking off)

Decode tok/s, aggregate **[per stream]**, 256 output tokens, temperature 0, boot D, better of two sparkDash runs on one boot.
Cells are medians: c1 five runs, c2/c4 three runs, c8 two runs. All requests completed without failures.
The whole table is from the first of two full sparkDash runs on boot D; it had the higher geometric mean over the
16 decode cells (aggregate 58.11 vs 58.09, per stream 26.05 vs 25.91) and over prefill. Cells are not mixed across runs.

| Prompt type | c1 | c2 | c4 | c8* |
|---|---:|---:|---:|---:|
| prose | **36.25** | 48.99 [24.61] | 65.04 [16.52] | 59.45 [17.48] |
| code | 44.10 | 52.62 [26.66] | 67.35 [17.77] | 62.58 [19.14] |
| structured | 47.31 | 62.01 [31.33] | 90.65 [23.70] | 74.12 [24.61] |
| json | 43.29 | 54.48 [27.96] | 80.15 [20.31] | 67.49 [20.50] |

*Four serving slots: at c8 four requests decode and the rest queue. When a slot frees, the next queued prompt
prefills while the remaining streams decode, which is the mix time-slicing paces. Per-stream decode is measured during
active generation; aggregate includes the batch wall time. sparkDash repeats one prompt across concurrent streams;
the separate memory stress uses four distinct prompts.

### Prefill

Cold sparkDash prompts, boot D, better of two sparkDash runs on one boot, median of three:

| Input | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 976 | 933 | 880 | 876 |
| time to first token (s) | 4.23 | 8.81 | 18.66 | 37.46 |

Long retrieval prompts, boot D, first cold request at each length (a different workload from sparkDash):

| Needle target | Actual input tokens | Cold TTFT (s) |
|---|---:|---:|
| 16K | 16,154 | 19.6 |
| 128K | 127,814 | 168.4 |
| 250K | 249,837 | 382.5 |

Adaptive chunks select 2048 below 16,384 total prompt tokens, otherwise 4096, including prefix-cache hits.
Prefix caching remains on; cold prompts and warm replay are reported separately.

## Draft-only NVFP4 LM head

The MTP draft reads its own NVFP4 (Marlin W4A16) copy of the LM head, 133.8 MB per rank. The unchanged target model,
whose LM head remains BF16, verifies drafts through standard speculative verification; this does not establish
bitwise-identical output across runs. Committed tokens per cycle were consistent with unchanged on the measured
panels; the decode cycle is about 4-5 % shorter. Same-boot ON/OFF confirmation (October 7 boot, fresh 12-prompt panels): prose **+5.61 %**
[+3.94, +7.47], code+structured **+4.43 %** [+3.17, +5.71]. At boot, `GLM_DRAFT_HEAD_INIT=1` qualifies the draft
graphs on all four ranks (write coverage, exact draft tokens, bounded float drift) and falls back to the BF16 draft
head on all ranks if any rank refuses; boot D passed on all ranks. The float bound exists because vLLM's MoE token
alignment orders tokens with atomics, so the Marlin MoE fp32 reduction order, and the draft hidden state at the few-ULP
level, can vary between runs. [Details](docs/draft-head.md).

## Decoding while another agent prefills

Agent A is decoding (a 60,043-token prefix-cached prompt) when agent B sends a cold ~100K-token prompt.
Without time-slicing every scheduler step mixes a 4096-token prefill chunk with A's decode, about 5.2 s per step.
With [time-slicing](docs/time-slicing.md) (prefill capped at 4096 tokens per step, then 40 pure-decode steps):

| Two runs each | A decode during B's prefill | B time to first token |
|---|---:|---:|
| time-slicing on (default), boot D | **12.59 tok/s** (12.06 / 13.12) | 209.1 s (208.2 / 210.0) |
| policy off, October 7 same-boot window | 0.44 tok/s (0.45 / 0.42) | 130.9 s (131.7 / 130.1) |

The cost is B's TTFT, **+60 %** against the October 7 policy-off boot (cross-boot). A returns to its normal rate after B's first token, and a lone prefill runs at full
speed. c1 decode is consistent with unchanged: same-boot ON/OFF ABBA on the time-slicing release boot (October 8),
48 pairs, **+0.83 %** [-0.36, +2.05]. The policy-off
row comes from a single October 7 boot that switched the policy with the runtime sidecar; with the policy on, that
boot measured 13.03 tok/s and 212.4 s.

Emergency off without a reboot, if the boot used the runtime sidecar (`GLM_DECODE_FAIR_CONTROL`, see
[time-slicing](docs/time-slicing.md#runtime-control-without-a-reboot)); run on the rank-0 host:

```bash
SEQ=1   # one more than the sequence currently in the file
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 0 --decode-steps 0 --sequence "$SEQ"
```

Without the sidecar, boot with `GLM_DECODE_FAIR=0`.

## RigMark 1.0.0 (thinking on, effort low)

Boot D, c1 decode medians of five; all workload gates passed.

| Workload | c1 decode tok/s | range |
|---|---:|---|
| prose | **29.32** | 28.72–29.67 |
| code | 41.63 | 41.03–41.73 |
| structured | 46.60 | 46.29–46.69 |

Concurrency, code workload, aggregate end-to-end tok/s **[per-stream decode]**, boot D:

| Workload | c1 | c2 | c4 |
|---|---:|---:|---:|
| code | 35.75 [38.41] | 47.23 [26.02] | 62.93 [17.74] |

| Prefill | Boot D tok/s |
|---|---:|
| 8K cold | 900 |
| 8K warm replay | 15,792 |

## Quality and capacity

| Check | Boot D result |
|---|---|
| qeval, 75 tasks, x3 | 72 / 74 / 73, **mean 73.0 passed tasks** (threshold 70.553); primary 52/54/53; no truncations or request failures |
| Needle 16K, twice | 9/10 / 8/10; PASS |
| Needle 128K, twice | 9/10 / 9/10; PASS |
| Needle 250K, twice | 8/10 / 8/10; PASS |
| Memory stress | c4 4x(65,024 prompt +1024 output), single 261,120+1024 PASS; rank-0 minimum **6.05 GiB** (criterion 5.8), swap 0, preemptions 0 |
| Prefix cache | APC repeat hit 64,896/65,024 prompt tokens; cache retained through stress |
| Mixed-load soak | 40.8 min, 65 requests (thinking prose, code, 16K-200K prompts, APC repeats, cancels), 0 errors, 0 preemptions; rank-0 minimum 5.72 GiB, quiet drift -0.028 GiB |
| Draft-head INIT | all four ranks ON and ready; 341 qualification rows per rank, 0 failed; target head bit-exact |

qeval failures: `code_camel_to_snake` (3 of 3 runs), `reason_r10` (2 of 3), `code_interval_intersect` (1 of 3).
Every needle response had all seven registry fields correct and finished with `stop`.
The retrieval probe checks latest revisions, aliases and an operator fix embedded in synthetic public text.
These are operational screens. The 50 tok/s prose-c1 goal and multi-day stability remain unqualified.
[Portable receipt summary](docs/results/release-1008b-summary.json).

## Memory safety

The floors are **4.5 GiB live**, **5.8 GiB stress criterion**, **6.0 GiB capture**,
**6.5 GiB admission for 60 s**, and **7.5 GiB before capture**, on every rank.
Boot D's rank-0 admission minimum was 6.71 GiB; the stress minimum was 6.05 GiB, the soak minimum 5.72 GiB and the
quiet level after the soak 5.84 GiB. The draft head's bank costs 133.8 MB per rank.

Two bounded, unpinned ballast runs on the October 7 release boot found **no observed harm down to 2.5 GiB under
load**: at a 2.59 GiB quiet level rank 0 reached 2.46 GiB, with paired prose/code cycle ratios 0.995/1.002, c1 TTFT
1.014, c4 wall 0.961 and 120K TTFT 0.977 relative to the unballasted baseline; PSI some avg10 stayed at 0.00 and
preemptions were 0. A second node at 2.58 GiB quiet reached 2.32 GiB with cycle ratios 0.976/0.981. Lower levels were
not tested. The live floor is the lowest harmless level plus 2.0 GiB; the 5.8 GiB stress criterion was set for the
draft head's bank. 2.5 GiB is evidence, not a serving floor.

## What is in the stack

Base checkpoint: [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9),
revision `206507bbb047d8223964a0414cd83230c59428f9`. Its files remain unchanged; conversion writes separate sidecars.

| Piece | What it does | Switch | Credit |
|---|---|---|---|
| Native MTP with K-stop | Confidence stop at c1; K2 batches; graphs [1,4,12,16] | `GLM_MTP_KSTOP=1` | Z.ai; vLLM; SpecDec++/DISCO ideas; DeepSeek verify-cap idea; keys/@u1tra_instinct TensorFold results |
| Display carveout KV | 6,318,718,976 B ordinary head plus 2046 MiB external carveout | `RECIPE_DISPRAM=require` | [kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram), external unmodified AGPL-3.0; NVIDIA; vLLM |
| FP4x latent KV | Packed latent rows; RoPE and index keys remain FP8 | `GLM_KV_FORMAT=fp4x` | Mia (MiaAI-Lab), FP4 KV idea only, no code copied; knapcio implementation; NVIDIA/Triton/vLLM |
| Adaptive prefill | 2048/4096 chunk budget selected from total prompt context | `GLM_PREFILL_CHUNK_ADAPTIVE=1` | vLLM scheduler contributors; knapcio |
| Draft-only NVFP4 LM head | MTP draft uses an NVFP4 Marlin copy of the LM head; BF16 target head verifies; boot-time all-rank INIT qualification with BF16 fallback | `GLM_DRAFT_HEAD=nvfp4`, `GLM_DRAFT_HEAD_INIT=1` | Marlin W4A16 kernels (IST-DASLab) and NVIDIA NVFP4 format via vLLM; vLLM contributors (MTP speculator, CUDA graphs, Marlin integration, `moe_align_block_size`); Z.ai native MTP; knapcio |
| Decode/prefill time-slicing | While a request decodes: prefill capped at 4096 tokens per step, then 40 pure-decode steps | `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40` | vLLM chunked-prefill scheduler contributors; Sarathi-Serve stall-free batching (Agrawal et al., prior work, no code used); knapcio |
| NVFP4 sidecars | 385 attention +225 shared +6 dense +776 MTP matrices, Marlin W4A16 | `GLM_ATTN_WEIGHTS=nvfp4`, `GLM_NVFP4_GROUPS=attn,shared,dense,mtp` | Marlin kernels via vLLM; NVIDIA NVFP4 format; vLLM quantization contributors; Tech2wild/tonyd2wild checkpoint; Z.ai; knapcio converters |
| Prefix caching | Reuses full KV blocks of repeated prefixes | `--enable-prefix-caching` | vLLM |
| Short-context DSA | All-selected short contexts skip unnecessary indexer work | `GLM_INDEXER_SHORTCUT=1` | Guess-Verify-Refine idea; DeepSeek DSA; vLLM |
| Pad hygiene and c2 reuse | Padded rows reuse live experts; c2 replays existing exact graph | `GLM_PAD_HYGIENE=1`, `GLM_MTP_KSTOP_CAPTURE_LAYOUT=reuse` | vLLM MoE/graph contributors; knapcio |
| Sparse MLA and dirty L2 | GB10-compatible split32 MLA; discard consumed partials | `GLM_FULL_MLA=triton`, `GLM_DIRTY_L2=discard` | CosmicRaisins; Matt Mastracci ideas; NVIDIA PTX; Triton |
| Switched RoCE | RDMA TP collectives on both rails | `GLM_ROCE_ALLREDUCE=1` | b12x/RoCEnante; Luke Alonso; Jason Cook; tonyd2wild; rhys101 |
| Fast loading and shard selection | Bounded copies; MTP reads its five shards | `GLM_FAST_LOAD=1`, `GLM_MTP_ONLY_LOAD=1` | Willian-Zhang; vLLM; knapcio |
| CPU boot preflight | Cached-header check, default on only with a cache; warns and continues on failure | `RECIPE_BOOT_PREFLIGHT=1` | vLLM constructors/loader hooks; knapcio |
| Coalesced loader, opt-in | Bounded staging and destination-write memory credit | `GLM_LOADER=coalesced` (default fast) | Allan Clark / ajclark, Apache-2.0 modified port; original NOTICE retained |
| Recent-FP8 bank, opt-in | Reservation only when explicitly requested, default zero | `GLM_FP4_RECENT_WINDOW=0` | NVIDIA FP8; vLLM; knapcio |

[CREDITS](CREDITS.md), [NOTICE](NOTICE), [licence boundaries](LICENSES/README.md).
No weights, dispramd binary, or container image is distributed here.

## Install and run

Four ARM64 DGX Sparks, switched ConnectX-7 fabric configured with NVIDIA Sync, management LAN/Tailscale,
the pinned v11 ARM64 image, local checkpoint and sidecars on every rank, and NCCL 2.30.7 are required.
[Installation](docs/install.md), [sidecar conversion commands, hashes and timing](docs/NVFP4-SIDECARS.md),
[service and recovery](service/README.md).

```bash
cp .env.example .env          # fill four hosts, fabric addresses, image IDs, paths and NCCL hashes
bash overlay/guard/build_guard.sh
DRY=1 ./start.sh serve
./start.sh preflight
./start.sh serve
./start.sh stop
```

Keep management recovery access working. The watchdog uses the 4.5 GiB live floor above, sustained swap/error guards,
a 600 s busy-without-progress allowance with idle reset, and an 1800 s boot deadline.
No public router port forwarding is needed.
Thinking off uses `chat_template_kwargs: {"enable_thinking": false}`; RigMark uses effort low.
Tool parser `glm47`, reasoning parser `glm45`. The [changelog](CHANGELOG.md) describes the changes against the
current public recipe; [GATE-RESULT.md](GATE-RESULT.md) records the completed qualification.

The default is the fast loader (`GLM_LOADER=fast`). The ajclark Apache-2.0 coalesced loader port is
**opt-in**, not default: boot B refused at load on rank 2 because of external memory pressure.
