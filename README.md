# Full GLM-5.3 on 4× NVIDIA DGX Spark

Full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B), tensor parallel across four DGX Sparks,
with native MTP speculative decoding, prefix caching and 64K context. Measured 2026-10-03,
GPU clocks capped at 2200 MHz. Raw results: [`docs/results/w4-v2-summary.json`](docs/results/w4-v2-summary.json).

## sparkDash (thinking off)

Decode tok/s, aggregate **[per stream]**. 256 output tokens, temperature 0.

| Prompt type | c1 | c2 | c3 | c4 | c8\* |
|---|---:|---:|---:|---:|---:|
| prose | **32.7** | 44.9 [22.9] | 53.4 [17.9] | 62.4 [16.2] | 62.0 [16.2] |
| code | 37.9 | 47.3 [24.0] | — | 67.8 [17.1] | 61.8 [16.4] |
| structured | 42.1 | 50.4 [26.0] | — | 80.7 [21.2] | 82.4 [21.9] |
| json | 38.6 | 49.9 [25.6] | — | 77.2 [19.9] | 74.4 [19.5] |

\* Four serving slots: at c8 four requests decode and the rest queue.

### Prefill

| Input | 4K | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|---:|
| cold prefill tok/s | 948 | 903 | 879 | 854 | 801 |
| time to first token (s) | 4.4 | 9.1 | 18.7 | 38.4 | 81.9 |

A repeated prompt is served from the prefix cache: 2K-token prompt TTFT 2.6 s → 0.5 s,
8K replay at 16,100 tok/s.

### Decode vs context length

Prose, c1, 512 output tokens, median of three. This uses a different prompt from the
sparkDash table, so compare within the table.

| Context | 0 | 16K | 30K | 60K |
|---|---:|---:|---:|---:|
| decode tok/s | 29.3 | 28.9 | 29.0 | 28.2 |
| change vs 0 | — | −1.5 % | −1.2 % | −3.7 % |

## RigMark 1.0.0 (thinking on, effort low)

| Workload | c1 decode tok/s | range |
|---|---:|---|
| prose | **25.4** | 24.7–26.5 |
| code | 36.4 | 35.7–36.7 |
| structured | 41.3 | 41.1–41.5 |

Concurrency, 256 output tokens, aggregate end-to-end tok/s **[per-stream decode]**:

| Workload | c1 | c2 | c4 | c8\* |
|---|---:|---:|---:|---:|
| prose | 23.9 [26.3] | 35.3 [19.4] | 50.3 [13.7] | 49.5 [13.5] |
| code | 32.8 [35.6] | 45.7 [24.8] | 75.8 [20.3] | 65.2 [17.6] |
| structured | 35.1 [41.2] | 53.4 [30.5] | 81.6 [23.0] | 93.2 [26.5] |

## Quality

| Check | Result |
|---|---|
| qeval (75 tasks, 3 runs) | 71 / 72 / 72, mean 71.7 (threshold 70.55) |
| 64K long-context retrieval ([`bench/long_retrieval_v2.py`](bench/long_retrieval_v2.py)) | PASS: 9/10 at 16K and at 63K, twice; all registry values correct |
| prefix cache repeat | PASS: identical answer, 2,048 cached tokens |

The probe hides latest-revision records, aliases and an operator fix in long synthetic text
and asks for short answers. Details: [validation](docs/validation.md).

## What is in the stack

Weights: [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9)
(revision `206507bb`), unchanged, with the checkpoint's own MTP layer.

| Piece | What it does | Switch | Credit |
|---|---|---|---|
| Native MTP with K-stop | At c1 a confidence stop drafts 1–3 tokens per step; batches draft 2 | `GLM_MTP_KSTOP=1` | Z.ai; vLLM; SpecDec++ and DISCO authors (ideas); DeepSeek verify-cap idea; TensorFold results from keys/@u1tra_instinct |
| Display carveout as KV | Uses the 2 GiB display reservation as KV cache: 66,112-token context | `RECIPE_DISPRAM=require` | kindling dispramd (unmodified external AGPL-3.0 tool); NVIDIA; vLLM |
| Prefix caching | Reuses KV of repeated prompt prefixes | `--enable-prefix-caching` | vLLM |
| Short-context DSA | Skips the sparse indexer while every token is selected anyway | `GLM_INDEXER_SHORTCUT=1` | Guess-Verify-Refine authors (idea); DeepSeek DSA; vLLM |
| Pad hygiene | Padded rows in CUDA graphs no longer pull extra MoE experts (c3 +20 %) | `GLM_PAD_HYGIENE=1` | vLLM MoE/graph contributors |
| c2 graph reuse | Two concurrent requests replay an exact-size graph | `GLM_MTP_KSTOP_CAPTURE_LAYOUT=reuse` | vLLM V2 graph manager contributors |
| Full-model sparse MLA | Triton MLA that fits GB10 shared memory | `GLM_FULL_MLA=triton` | CosmicRaisins; Matt Mastracci (ideas only); Triton |
| Dirty-L2 discard | Drops consumed MLA partials from L2 | `GLM_DIRTY_L2=discard` | NVIDIA PTX; Triton |
| Switched RoCE | TP collectives over RDMA on both rails | `GLM_ROCE_ALLREDUCE=1` | b12x/RoCEnante; Luke Alonso; Jason Cook; tonyd2wild; rhys101 |
| Fast load | Boot to healthy in ~285 s | `GLM_FAST_LOAD=1` | Willian-Zhang's GB10 copy-cost report; vLLM |
| Spec-sample (off) | Experimental probabilistic drafts for T>0 | `GLM_SPEC_SAMPLE=0`, [details](docs/spec-sample/README.md) | Leviathan et al.; Chen et al.; vLLM |
| Red Hat DSpark (alternative) | Separate opt-in drafter profile, not measured above | `RECIPE_PROFILE=dspark-k3` | Red Hat DSpark / Speculators; vLLM |

Ash Hart / ashhart and TensorFold retain MIT credit; Jay Leaton's Spark recipe retains
Apache-2.0 credit. See [CREDITS](CREDITS.md), [NOTICE](NOTICE),
[licence boundaries](LICENSES/README.md) and [runtime details](docs/runtime.md).

## Install and run

Needs four DGX Sparks on a switched RoCE fabric, the pinned ARM64 image and the weights on
every node. [Installation](docs/install.md) covers the pinned dispramd setup;
[service and recovery](service/README.md) covers the persistent units.

```bash
cp .env.example .env          # set your addresses and paths
bash overlay/guard/build_guard.sh
DRY=1 ./start.sh serve        # print the launch without starting
./start.sh preflight
./start.sh serve
./start.sh stop
```

Reasoning effort `low`, `high` or `max`; thinking off is
`chat_template_kwargs: {"enable_thinking": false}`. Tool parser `glm47`, reasoning parser `glm45`.

Benchmarks:

```bash
SPARKDASH_API="$DASH_API" python3 bench/sparkdash.py full 66112
python3 bench/qeval.py run my-run --url "$CHAT_COMPLETIONS_URL"
python3 bench/long_retrieval_v2.py --execute --endpoint "$SERVING_URL" \
  --tokenizer-dir "$TOKENIZER_DIR" --out "$LONG_PROBE_RECEIPTS"
```

Earlier versions and their numbers: [history](docs/history.md).
