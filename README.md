# Full GLM-5.3 on 4x NVIDIA DGX Spark — stack-1003

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), TP4 on four DGX Sparks,
with native MTP confidence stopping, short-context DSA, automatic prefix caching,
display-carveout KV and switched RoCE. Primary release gate: **PASS**; measured
**2026-10-03**, inference source **0f24383**, receipt set
**W4 v2 primary (pad=1, spec-sample=0)**.

Measurements, trial values and the configuration are recorded in
[`w4-v2-summary.json`](docs/results/w4-v2-summary.json). Every table below uses
the admitted primary v2 boot, except prose/code c1 and the separate prose c3
sweep: those use every scored trial across the frozen pair of identical boots.
The RigMark concurrency table labels its additional measurements separately.
The original primary trials remain available in the receipt above. Earlier R results remain archived
in [`w4-R-summary.json`](docs/results/w4-R-summary.json).

The frozen independent boot measured prose c1 **32.97 tok/s** (five runs);
the median of all ten trials across both identical boots is **32.71 tok/s**.
Combined code c1 is **37.85 tok/s** (ten runs) and prose c3 is
**53.355 aggregate [17.86 per stream] tok/s** (six runs). These supplemental
results retain every scored trial and do not replace the primary tables.
The additional c3 qeval screen scored **71/75**, but **failed** its no-truncation
condition: `code_parse_query` used all 420 tokens on reasoning without final code.
The cause of this c3 difference is unresolved. The primary serial gate remains
71/72/72 with no truncations. Full trial values and failures:
[`w4-v2-confirmation.json`](docs/results/w4-v2-confirmation.json).

## sparkDash — thinking off

Decode, aggregate tok/s **[per-stream tok/s]**. c8 submits eight requests to
four serving slots: excess requests queue, so this tests throughput under load
rather than eight simultaneously decoding streams. Similar c4/c8 aggregate rates
indicate saturation. Aggregate timing includes the complete batch; per-stream
decode timing excludes waiting before the first token. Workloads appear prose first.

| Prompt type | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| prose | 32.7 [32.7] | 44.9 [22.9] | 62.4 [16.2] | 62.0 [16.2] |
| code | 37.9 [37.9] | 47.3 [24.0] | 67.8 [17.1] | 61.8 [16.4] |
| structured | 42.1 [42.1] | 50.4 [26.0] | 80.7 [21.2] | 82.4 [21.9] |
| json | 38.6 [38.6] | 49.9 [25.6] | 77.2 [19.9] | 74.4 [19.5] |

The separate prose c3 sweep measured **53.4 aggregate [17.9 per stream] tok/s**,
median of all six runs across two boots, with descriptor-scoped pad hygiene v2 on.
Prose/code c1 are medians of all ten runs each across those same two boots;
other sparkDash cells, prefill and context sweeps use the primary boot.

### Prefill

Cold input tok/s; actual input length accompanies each nominal bucket. APC is on,
so use disjoint uncached prefixes for cold measurements and label warm replay
separately. A shared-prefix concurrency sweep can become warm within a wave.

| State | 4K | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|---:|
| cold tok/s | 948 | 903 | 879 | 854 | 801 |
| actual input tokens | 4126 | 8226 | 16414 | 32799 | 65567 |
| cold TTFT (s) | 4.36 | 9.11 | 18.67 | 38.42 | 81.90 |
| warm replay tok/s | not measured | not measured | not measured | not measured | not measured |

### Decode vs context length

Thinking-off c1 prose, temperature 0, 512 output tokens; medians of three
requests per length. This uses a different prompt and output budget from the
sparkDash 256-token prose benchmark above; compare the relative context cost
within this table. Nonzero contexts have distinct prefixes to avoid prefix reuse.
The zero-context question repeats unchanged; its cache state is unverified.
Reported cached-token fields were absent.

| Added context | Actual prompt tokens | Decode tok/s | Change vs 0 | TTFT (s) |
|---|---:|---:|---:|---:|
| 0 | 70 | 29.3 | — | 0.45 |
| 16K | 16,322 | 28.9 | -1.5% | 21.70 |
| 30K | 30,661 | 29.0 | -1.2% | 40.83 |
| 60K | 61,381 | 28.2 | -3.7% | 83.98 |

## RigMark — thinking on, effort low

Record version and source pin in the gate receipts. Keep this protocol separate
from sparkDash. Standalone decode uses a 4096-token budget and five runs;
concurrency uses a 256-token budget. Decode medians and ranges are tok/s.

| Workload | c1 median | c1 range |
|---|---:|---|
| prose | 25.4 | 24.7–26.5 |
| code | 36.4 | 35.7–36.7 |
| structured | 41.3 | 41.1–41.5 |

Concurrency, aggregate end-to-end tok/s **[per-stream decode tok/s]**; preserve
the distinction between request-wall aggregate and steady decode timing.

Code c1/c2/c4 come from the primary boot. Other concurrency cells were measured
on the independent boot with the same v2 configuration, three rounds each,
a 256-token budget and one discarded warmup per workload. RigMark 1.0.0
source is pinned at `d8353e93`; its stream, prompt and concurrency functions are
unchanged. Structured concurrency extends its CLI workload selection and is
labelled supplemental. c8 submits eight requests to four serving slots.
Fixed-length concurrency probes measure throughput, not output-quality passes.
[Trial values and source hashes](docs/results/w4-v2-rigmark-concurrency.json).

| Workload | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| prose | 23.9 [26.3] | 35.3 [19.4] | 50.3 [13.7] | 49.5 [13.5] |
| code | 32.8 [35.6] | 45.7 [24.8] | 75.8 [20.3] | 65.2 [17.6] |
| structured | 35.1 [41.2] | 53.4 [30.5] | 81.6 [23.0] | 93.2 [26.5] |

| Prefill state | 8K tok/s |
|---|---:|
| cold | 880 |
| warm replay | 16103 |

## What is in the stack

The target uses [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9),
revision `206507bbb047d8223964a0414cd83230c59428f9`, with the checkpoint's own MTP
layer. Weights are downloaded separately and kept unchanged. The pinned ARM64
image, source hashes and immutable manifests remain part of the recipe.

| Lever | Current release behavior | Switch / implementation | Credit |
|---|---|---|---|
| Native MTP with K-stop | At c1, confidence stopping chooses K1–3; cumulative draft confidence decides whether another proposal is made. Multi-request batches use uniform K2 for graph replay. | `GLM_MTP_FIX=1`, `GLM_MTP_KSTOP=1`, `GLM_MTP_KSTOP_UNIFORM_BATCH=k2`; `overlay/kstop/` | Z.ai; vLLM; SpecDec++ and DISCO authors (ideas); DeepSeek verify-cap idea; local integration; TensorFold results from keys/@u1tra_instinct |
| Short-context DSA | Skip indexer query/logits only when every token is selected; long, padded, mixed and prefill shapes retain the stock indexer. K-stop compatibility is implemented; record arming in the gate. | `GLM_INDEXER_SHORTCUT=1`; `overlay/bringup/glm_dsa_short.py` | Guess-Verify-Refine authors (idea); DeepSeek DSA; vLLM; local adapter |
| Automatic prefix caching (APC) | On; identical full prefix blocks reuse the shared KV pool. Cold/warm benchmark states are labelled separately. | `--enable-prefix-caching` | vLLM contributors |
| Display carveout as KV | Required release layout: ordinary KV head plus display carveout in one mapped range, about 64K context. Admitted context: 66112 tokens; KV blocks/rank: 1039. | `RECIPE_DISPRAM=require`, `RECIPE_MAX_MODEL_LEN=66112`; `glm_dispram_kv.py`, copy guard | kindling dispramd (unmodified external AGPL-3.0 tool); NVIDIA mapping interfaces; vLLM; local KV integration |
| Fast load | Slab/read-ahead loading and MTP-only shard selection; boot through health 283.8 s, through admission 343.9 s. | `GLM_FAST_LOAD=1`, `GLM_MTP_ONLY_LOAD=1`, `GLM_TARGET_SKIP_MTP=1` | Willian-Zhang's GB10 copy-cost report; vLLM loader contributors; local loader |
| c2 capture-layout reuse | Dispatch the existing captured verify rectangle for uniform K2 at c2; preserve c1/c4 captures and memory charge. | `GLM_MTP_KSTOP_CAPTURE_LAYOUT=reuse` | vLLM V2 graph manager contributors; local dispatch integration |
| Pad hygiene | On: descriptor-scoped v2 remaps routes only in graphs/forwards that can have padding; exact c1 graphs omit the remap. Live-token routes remain intact. | `GLM_PAD_HYGIENE`; `tests/test_pad_hygiene.py` | vLLM MoE/graph contributors; local remap and checks |
| Full-model sparse MLA | Split-layout Triton MLA keeps NoPE and RoPE separate on GB10. | `GLM_FULL_MLA=triton`, `GLM_MLA_SPLIT_K=32` | CosmicRaisins; Matt Mastracci (ideas only); Triton; local kernels |
| Dirty-L2 discard | Discard consumed split MLA partials after reduction, retaining reduction arithmetic. | `GLM_DIRTY_L2=discard` | NVIDIA PTX; Triton; local diagnosis and fix |
| Switched RoCE | Eligible TP collectives use both rails; unsupported collectives use NCCL. | `GLM_ROCE_ALLREDUCE=1`; `roce/` | b12x/RoCEnante; Luke Alonso; Jason Cook; tonyd2wild; rhys101 |
| Chunked prefill and guarded boot | Drained scheduler-cap switch, fixed constructor capacity, JIT preparation, hash records, health polling and all-rank memory admission. | `GLM_W2_PREFILL_CHUNK`; `scripts/cluster.py` | vLLM; FlashInfer; NVIDIA NCCL; local launcher |
| Spec-sample | Experimental, off. Probabilistic native drafts and an independent residual noise stream require separate distribution qualification. | `GLM_SPEC_SAMPLE=0`; [details](docs/spec-sample/README.md) | Leviathan et al.; Chen et al.; vLLM; local residual-noise integration |
| Red Hat DSpark alternative | Separate opt-in recipe with its own draft pool and protocol; outside the measurements above. | `RECIPE_PROFILE=dspark-k3` | Red Hat DSpark / Speculators; vLLM; local pool integration |

All source-pinned hooks refuse drift or unsupported layouts. See [runtime details](docs/runtime.md),
[CREDITS](CREDITS.md), [NOTICE](NOTICE) and [licence boundaries](LICENSES/README.md).
Ash Hart / ashhart and TensorFold retain MIT credit; Jay Leaton's Spark recipe
retains Apache-2.0 credit. Matt Mastracci is credited for ideas only.

## Quality and serving gate

| Check | Receipt value | Admission rule |
|---|---|---|
| Serving | primary v2 gate passed; restart count 0 (all four ranks); measured minimum available memory 8.96 GiB | Health, exact `OK`, every rank running, no OOM/restarts, levers armed and unchanged watchdog admission |
| qeval | 71 / 72 / 72; mean 71.7/75; threshold 70.553/75; truncations 0; no new recurring failures; PASS | Three serial runs, unchanged checker/task budgets and checker-matched threshold fixed before the gate |
| long64k-v2, repeat 1 | control 9/10, long 9/10; all seven long registry fields correct; finish stop | Both paired repeats satisfy the unchanged rule |
| long64k-v2, repeat 2 | control 9/10, long 9/10; all seven long registry fields correct; finish stop | Both paired repeats satisfy the unchanged rule |
| long64k-v2 verdict | PASS; receipt [w4-v2-long64k.json](docs/results/w4-v2-long64k.json) | Every transport completes; both paired repeats pass |
| APC repeat | PASS; cold/warm TTFT 2.58/0.51 s; reused tokens 2048 | Record a repeated-prefix hit and correct completion |

`long64k-v2` is the seeded synthetic retrieval probe in
[`bench/long_retrieval_v2.py`](bench/long_retrieval_v2.py). It pairs a 16K control
with a near-64K input, checks latest-revision registry retrieval, an alias chain,
an operator repair and an archive count, and repeats each length twice. **PASS:**
for each paired repeat, long score is at least control score minus one, all seven
long registry fields are correct, and every response finishes with `stop`.
Transport/protocol failures yield ERROR. Offline self-checks do not establish
served quality. The second repeat may have APC warmth; retain that fact.

The quality screen is bounded; full intelligence equivalence and multi-day
stability are unqualified. See [validation and receipt protocol](docs/validation.md).

## Install and run

Requirements: four DGX Sparks with an ARM64 GPU runtime, the pinned image,
complete weights on each rank, matched NCCL libraries, an existing switched RoCE
fabric and the compaction helper. Set deployment addresses and paths privately
in `.env`. [Installation](docs/install.md) includes the unmodified, pinned
dispramd setup; [persistent service and recovery](service/README.md) covers the
included service units. The release layout requires the compiled copy guard.

```bash
cp .env.example .env
# Configure private deployment fields; the example selects the qualified layout.
bash overlay/guard/build_guard.sh
DRY=1 ./start.sh serve
./start.sh preflight
./start.sh serve
```

`profiles/current.env` is the base profile; `.env.example` selects 66112 context,
required dispram, short DSA and pad hygiene v2 on, with spec-sample off. Stop with `./start.sh stop`; verified shutdown returns the KV
lease and leaves the lender running. Any failed lease/context check retains the
lock for recovery. Never force a daemon restart under live borrowers.

Requests use reasoning effort `low`, `high` or `max`; thinking off is
`chat_template_kwargs: {"enable_thinking": false}`. Tool parsing uses `glm47`
and reasoning parsing uses `glm45`. Use your configured authenticated endpoint.

```bash
SPARKDASH_API="$DASH_API" python3 bench/sparkdash.py full 66112
python3 bench/qeval.py run stack-1003-q1 --url "$CHAT_COMPLETIONS_URL"
python3 bench/long_retrieval_v2.py --execute --endpoint "$SERVING_URL" \
  --tokenizer-dir "$TOKENIZER_DIR" --out "$LONG_PROBE_RECEIPTS"
```

Repeat qeval serially with distinct receipt labels. The sparkDash collector's
standard prefill sweep ends at 32K; obtain the 64K prefill and context-length
cells separately under the [same gate protocol](docs/validation-release-RUN.md).
Every table cell must be backed by a completed receipt. Earlier measurements
are confined to [measurement history](docs/history.md) and its source receipts.
