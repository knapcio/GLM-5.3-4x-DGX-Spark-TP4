# Full GLM-5.3 on 4x NVIDIA DGX Spark

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B), tensor parallel across four DGX Sparks,
with native MTP speculative decoding, FP4x KV, NVFP4 weight sidecars and **262,144-token context**.
The ranks implement one shared **264,640-token KV pool (4,135 blocks)**; four simultaneous maximum-context
requests do not fit. The ordinary KV head is 6,318,718,976 bytes per rank, plus the display carveout.
The recent-FP8 bank is disabled by default. Measured October 6–7, 2026: boot H below, except qeval on boot A
with identical per-rank parameter and buffer hashes. [Gate result](GATE-RESULT.md).

## sparkDash (thinking off)

Decode tok/s, aggregate **[per stream]**, 256 output tokens, temperature 0, boot H.
Cells are medians: c1 five runs, c2/c4 three runs, c8 two runs. All requests completed without failures.

| Prompt type | c1 | c2 | c4 | c8* |
|---|---:|---:|---:|---:|
| prose | **35.00** | 47.85 [23.92] | 62.76 [16.23] | 61.38 [16.01] |
| code | 41.96 | 49.39 [25.61] | 67.45 [17.70] | 61.92 [16.73] |
| structured | 45.36 | 54.93 [28.47] | 88.35 [23.26] | 82.11 [21.85] |
| json | 41.72 | 53.77 [27.15] | 75.23 [19.33] | 71.74 [19.06] |

*Four serving slots: at c8 four requests decode and the rest queue. Per-stream decode is measured during
active generation; aggregate includes the batch wall time. sparkDash repeats one prompt across concurrent streams;
the separate memory stress uses four distinct prompts.

### Prefill

Cold sparkDash prompts, boot H, median of three:

| Input | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| cold prefill tok/s | 973 | 930 | 880 | 864 |
| time to first token (s) | 4.24 | 8.85 | 18.66 | 37.95 |

Long retrieval prompts, boot H, first cold request at each length (a different workload from sparkDash):

| Needle target | Actual input tokens | Cold TTFT (s) |
|---|---:|---:|
| 16K | 16,154 | 25.6 |
| 128K | 127,814 | 169.1 |
| 250K | 249,837 | 382.8 |

Adaptive chunks select 2048 below 16,384 total prompt tokens, otherwise 4096, including prefix-cache hits.
Prefix caching remains on; cold prompts and warm replay are reported separately.

## RigMark 1.0.0 (thinking on, effort low)

Boot H, c1 decode medians of five; all workload gates passed.

| Workload | c1 decode tok/s | range |
|---|---:|---|
| prose | **28.42** | 27.86–29.16 |
| code | 39.02 | 38.60–39.71 |
| structured | 44.41 | 41.68–44.90 |

Concurrency, code workload, aggregate end-to-end tok/s **[per-stream decode]**, boot H:

| Workload | c1 | c2 | c4 |
|---|---:|---:|---:|
| code | 33.06 [35.29] | 44.94 [24.45] | 64.93 [17.63] |

| Prefill | Boot H tok/s |
|---|---:|
| 8K cold | 881 |
| 8K warm replay | 15,679 |

## Quality and capacity

| Check | Qualified 262,144 result |
|---|---|
| qeval, 75 tasks, x3 (A) | 73 / 71 / 71; **mean 71.67 passed tasks** (threshold 70.553); no truncations or request failures |
| Needle 16K (H), twice | 8/10 / 8/10; PASS |
| Needle 128K (H), twice | 9/10 / 9/10; PASS |
| Needle 250K (H), twice | 9/10 / 9/10; PASS |
| Memory stress (H) | c4 4x(65,024 prompt +1024 output), single 261,120+1024 PASS; rank-0 minimum **6.69 GiB**, swap 0, preemptions 0 |
| Prefix cache (H) | APC repeat hit 64,896/65,024 prompt tokens; cache retained through stress |
| Mixed-load soak (H) | 40.9 min, 59 requests, 0 errors, 0 preemptions; rank-0 minimum 6.28 GiB, quiet drift -0.127 GiB |

qeval was run on boot A, not repeated on H: the code and weight formats are identical and per-rank parameter
and buffer digests match. Primary scores were 54 / 51 / 51. Recurring failures were `code_word_wrap`,
`code_camel_to_snake` and `math_m9` (two of three runs each); all failures are retained in the receipts.
Every needle response had all seven registry fields correct and finished with `stop`.
The retrieval probe checks latest revisions, aliases and an operator fix embedded in synthetic public text.
These are operational screens. The 50 tok/s prose-c1 goal and multi-day stability remain unqualified.
[Portable receipt summary](docs/results/release-1006-summary.json).

## Memory safety

The qualified floors are **5.5 GiB live**, **6.0 GiB stress criterion**, **6.0 GiB capture**,
**6.5 GiB admission for 60 s**, and **7.5 GiB before capture**, on every rank.
H's sampled rank-0 admission minimum was 6.66 GiB; c4/single stress minima were 6.71/6.69 GiB.

A bounded, unpinned ballast test on H found **no observed harm down to 3.5 GiB under load**:
rank 0 reached 3.50 GiB minimum at a 3.56 GiB quiet level. Paired prose/code cycle ratios were 1.011/0.990,
c1 TTFT 1.013, c4 wall 0.955 and 120K TTFT 1.000 relative to the unballasted baseline; preemptions were 0
and PSI some total was 2.6 ms. Rank 1 at a 3.57 GiB quiet level briefly reached 3.22 GiB, with paired cycle
ratios 1.001/0.981 and 0 preemptions. The ballast's temporary 2.8 GiB watch floor was restored to 5.5 GiB.
The live floor adds 2.0 GiB to the lowest tested harmless quiet level; 3.5 GiB is evidence, not a serving floor.

## What is in the stack

Base checkpoint: [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9),
revision `206507bbb047d8223964a0414cd83230c59428f9`. Its files remain unchanged; conversion writes separate sidecars.

| Piece | What it does | Switch | Credit |
|---|---|---|---|
| Native MTP with K-stop | Confidence stop at c1; K2 batches; graphs [1,4,12,16] | `GLM_MTP_KSTOP=1` | Z.ai; vLLM; SpecDec++/DISCO ideas; DeepSeek verify-cap idea; keys/@u1tra_instinct TensorFold results |
| Display carveout KV | 6,318,718,976 B ordinary head plus 2046 MiB external carveout | `RECIPE_DISPRAM=require` | [kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram), external unmodified AGPL-3.0; NVIDIA; vLLM |
| FP4x latent KV | Packed latent rows; RoPE and index keys remain FP8 | `GLM_KV_FORMAT=fp4x` | Mia (MiaAI-Lab), FP4 KV idea only, no code copied; knapcio implementation; NVIDIA/Triton/vLLM |
| Adaptive prefill | 2048/4096 chunk budget selected from total prompt context | `GLM_PREFILL_CHUNK_ADAPTIVE=1` | vLLM scheduler contributors; knapcio |
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

Keep management recovery access working. The watchdog uses the 5.5 GiB live floor above, sustained swap/error guards,
a 600 s busy-without-progress allowance with idle reset, and an 1800 s boot deadline.
No public router port forwarding is needed.
Thinking off uses `chat_template_kwargs: {"enable_thinking": false}`; RigMark uses effort low.
Tool parser `glm47`, reasoning parser `glm45`. The [changelog](CHANGELOG.md) describes the changes against the
current public recipe; [GATE-RESULT.md](GATE-RESULT.md) records the completed qualification.

The default is the fast loader (`GLM_LOADER=fast`). The ajclark Apache-2.0 coalesced loader port is
**opt-in**, not default: boot B refused at load on rank 2 because of external memory pressure.
