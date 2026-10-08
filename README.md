# Full GLM-5.3 on 4x NVIDIA DGX Spark

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B), tensor parallel across four DGX Sparks,
with native MTP speculative decoding, FP4x KV, NVFP4 weight sidecars and **262,144-token context**.
The ranks implement one shared **264,640-token KV pool (4,135 blocks)**; four simultaneous maximum-context
requests do not fit. The ordinary KV head is 6,318,718,976 bytes per rank, plus the display carveout.
The recent-FP8 bank is disabled by default. Decode/prefill time-slicing keeps a decoding request responsive
while another request prefills a long prompt. Results are from one boot, T2, measured October 8, 2026, unless a row
says otherwise.
[Gate result](GATE-RESULT.md).

## sparkDash (thinking off)

Decode tok/s, aggregate **[per stream]**, 256 output tokens, temperature 0, boot T2.
Cells are medians: c1 five runs, c2/c4 three runs, c8 two runs. All requests completed without failures.

| Prompt type | c1 | c2 | c4 | c8* |
|---|---:|---:|---:|---:|
| prose | **33.99** | 48.06 [24.72] | 63.69 [16.40] | 58.71 [17.28] |
| code | 42.23 | 49.96 [26.11] | 70.53 [17.80] | 60.44 [18.63] |
| structured | 45.04 | 56.28 [28.83] | 85.81 [23.36] | 78.24 [24.43] |
| json | 41.83 | 51.34 [26.21] | 77.55 [20.27] | 68.61 [20.73] |

*Four serving slots: at c8 four requests decode and the rest queue. When a slot frees, the next queued prompt
prefills while the remaining streams decode, which is the mix time-slicing paces. Per-stream decode is measured during
active generation; aggregate includes the batch wall time. sparkDash repeats one prompt across concurrent streams;
the separate memory stress uses four distinct prompts.

### Prefill

Cold sparkDash prompts, boot T2, median of three:

| Input | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 977 | 930 | 895 | 888 |
| time to first token (s) | 4.22 | 8.84 | 18.34 | 36.92 |

Long retrieval prompts, boot T2, first cold request at each length (a different workload from sparkDash):

| Needle target | Actual input tokens | Cold TTFT (s) |
|---|---:|---:|
| 16K | 16,154 | 25.3 |
| 128K | 127,814 | 167.4 |
| 250K | 249,837 | 376.8 |

Adaptive chunks select 2048 below 16,384 total prompt tokens, otherwise 4096, including prefix-cache hits.
Prefix caching remains on; cold prompts and warm replay are reported separately.

## Decoding while another agent prefills

Agent A is decoding (a 60,043-token prefix-cached prompt) when agent B sends a cold ~100K-token prompt.
Without time-slicing every scheduler step mixes a 4096-token prefill chunk with A's decode, about 5.2 s per step.
With [time-slicing](docs/time-slicing.md) (prefill capped at 4096 tokens per step, then 40 pure-decode steps):

| Two runs each | A decode during B's prefill | B time to first token |
|---|---:|---:|
| time-slicing on (default), boot T2 | **12.46 tok/s** (14.15 / 10.77) | 212.0 s (216.1 / 208.0) |
| policy off, October 7 same-boot window | 0.44 tok/s (0.45 / 0.42) | 130.9 s (131.7 / 130.1) |

The cost is B's TTFT, **+62 %**. A returns to its normal rate after B's first token, and a lone prefill runs at full
speed. c1 decode is consistent with unchanged: T2 same-boot ON/OFF ABBA, 48 pairs, **+0.83 %** [-0.36, +2.05]. The policy-off
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

Boot T2, c1 decode medians of five; all workload gates passed.

| Workload | c1 decode tok/s | range |
|---|---:|---|
| prose | **28.67** | 27.53–29.62 |
| code | 40.07 | 38.62–40.65 |
| structured | 44.42 | 44.20–44.90 |

Concurrency, code workload, aggregate end-to-end tok/s **[per-stream decode]**, boot T2:

| Workload | c1 | c2 | c4 |
|---|---:|---:|---:|
| code | 35.05 [37.63] | 45.85 [24.86] | 65.62 [17.57] |

| Prefill | Boot T2 tok/s |
|---|---:|
| 8K cold | 913 |
| 8K warm replay | 15,589 |

## Quality and capacity

| Check | Boot T2 result |
|---|---|
| qeval, 75 tasks, x1 | **74 passed** (threshold 70.553); primary 54; failed `code_camel_to_snake`; no truncations or request failures |
| Needle 16K, twice | 8/10 / 8/10; PASS |
| Needle 128K, twice | 9/10 / 9/10; PASS |
| Needle 250K, twice | 9/10 / 9/10; PASS |
| Memory stress | c4 4x(65,024 prompt +1024 output), single 261,120+1024 PASS; rank-0 minimum **6.30 GiB**, swap 0, preemptions 0 |
| Prefix cache | APC repeat hit 64,896/65,024 prompt tokens; cache retained through stress |
| Mixed-load soak | 41.2 min, 71 requests (thinking prose, code, 16K-200K prompts, APC repeats, cancels), 0 errors, 0 preemptions; rank-0 minimum 6.18 GiB, quiet drift -0.063 GiB |

Every needle response had all seven registry fields correct and finished with `stop`.
The retrieval probe checks latest revisions, aliases and an operator fix embedded in synthetic public text.
These are operational screens. The 50 tok/s prose-c1 goal and multi-day stability remain unqualified.
[Portable receipt summary](docs/results/release-1008-summary.json).

## Memory safety

The floors are **4.5 GiB live**, **6.0 GiB stress criterion**, **6.0 GiB capture**,
**6.5 GiB admission for 60 s**, and **7.5 GiB before capture**, on every rank.
T2's rank-0 admission minimum was 7.06 GiB; the stress minimum was 6.30 GiB and the soak minimum 6.18 GiB.

A bounded, unpinned ballast test on the October 7 release boot found **no observed harm down to 3.5 GiB under load**:
rank 0 reached 3.50 GiB minimum at a 3.56 GiB quiet level. Paired prose/code cycle ratios were 1.011/0.990,
c1 TTFT 1.013, c4 wall 0.955 and 120K TTFT 1.000 relative to the unballasted baseline; preemptions were 0
and PSI some total was 2.6 ms. Rank 1 at a 3.57 GiB quiet level briefly reached 3.22 GiB, with paired cycle
ratios 1.001/0.981 and 0 preemptions. The live floor was lowered from 5.5 to 4.5 GiB on October 7 after real
multi-agent traffic tripped the 5.5 GiB floor; 4.5 GiB is 1.0 GiB above the lowest harmless level, and
3.5 GiB is evidence, not a serving floor.

## What is in the stack

Base checkpoint: [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9),
revision `206507bbb047d8223964a0414cd83230c59428f9`. Its files remain unchanged; conversion writes separate sidecars.

| Piece | What it does | Switch | Credit |
|---|---|---|---|
| Native MTP with K-stop | Confidence stop at c1; K2 batches; graphs [1,4,12,16] | `GLM_MTP_KSTOP=1` | Z.ai; vLLM; SpecDec++/DISCO ideas; DeepSeek verify-cap idea; keys/@u1tra_instinct TensorFold results |
| Display carveout KV | 6,318,718,976 B ordinary head plus 2046 MiB external carveout | `RECIPE_DISPRAM=require` | [kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram), external unmodified AGPL-3.0; NVIDIA; vLLM |
| FP4x latent KV | Packed latent rows; RoPE and index keys remain FP8 | `GLM_KV_FORMAT=fp4x` | Mia (MiaAI-Lab), FP4 KV idea only, no code copied; knapcio implementation; NVIDIA/Triton/vLLM |
| Adaptive prefill | 2048/4096 chunk budget selected from total prompt context | `GLM_PREFILL_CHUNK_ADAPTIVE=1` | vLLM scheduler contributors; knapcio |
| Decode/prefill time-slicing | While a request decodes: prefill capped at 4096 tokens per step, then 40 pure-decode steps | `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40` | vLLM chunked-prefill scheduler contributors; Sarathi-Serve stall-free batching (Agrawal et al., prior work, no code used); knapcio |
| NVFP4 sidecars | 385 attention +225 shared +6 dense +776 MTP matrices, Marlin W4A16 | `GLM_ATTN_WEIGHTS=nvfp4`, `GLM_NVFP4_GROUPS=attn,shared,dense,mtp` | NVIDIA Marlin; vLLM quantization contributors; Tech2wild/tonyd2wild checkpoint; Z.ai; knapcio converters |
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
