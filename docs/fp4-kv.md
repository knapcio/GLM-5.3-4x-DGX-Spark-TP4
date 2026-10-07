# FP4x KV cache

The shipped native serving profile selects `GLM_KV_FORMAT=fp4x` and
`RECIPE_MAX_MODEL_LEN=98176`. This is boot g (`64269fe`), measured on
2026-10-06. FP8 remains selectable in this same profile with
`GLM_KV_FORMAT=fp8` and `RECIPE_MAX_MODEL_LEN=66112`. Format changes require
all requests to drain and all serving processes to restart; prefixes and
CUDA graphs cannot cross the format boundary.

The model weights, semantic latent dimensions, DSA index keys, native MTP,
K-stop, scheduler, prefix hashing and collectives retain their contracts.
The command-line cache dtype remains `fp8_e4m3`; the packed overlay changes
the target and MTP MLA storage and dispatch, rather than introducing a
second serving profile or a new command-line dtype.

## Format

ABI: `fp4x_v1_e2m1_s16_e4m3_pow2_r64`. Each physical MLA row is 368 bytes;
semantic dimensions remain latent512 + rotary64.

| Bytes | Field |
|---|---|
| 0–255 | Two signed E2M1 codes per byte; even element in low nibble |
| 256–287 | E4M3 block scale, one per 16 latent values |
| 288–291 | Little-endian FP32 power-of-two row scale |
| 292–303 | Deterministic zero padding |
| 304–367 | E4M3 RoPE, existing `rope / k_scale` writer convention |

Latent reconstruction is `(E2M1 * block_scale) * row_scale`, rounded once
to BF16. It does not apply MLA tensor `k_scale` again. The bit-exact
numerical oracle is retained in `tests/fixtures/fp4_probe_quant_dc5ad9e.py`.
DSA keys retain 128 E4M3 bytes plus a four-byte scale per token, with
page-separated codes and scales. Their writer and paged decode-logits path
remain source-pinned. MLA and index storage are separate.

Main-model and MTP rows use the fused pack/store writer. Packed sparse
readers reconstruct each tile for QK and PV. Split32 decode retains the
dirty-L2 discard reduce and the 36-row dispatch cap. Fresh dense prefill
uses matching quantize/dequantize; cached dense prefixes use bounded BF16
gathers. Packed storage is never passed to a reader that expects plain
FP8; unsupported direct prefix gathers and stock attention fail closed.

## Memory

The same per-rank pool combines a 1 GiB ordinary head with a
2,145,386,496-byte display carveout. There are 79 MLA and 22 index tensors,
64-token blocks and 512-byte tensor alignment.

| Format | MLA / index bytes per token | Block bytes | Pool blocks | Physical tokens | Aligned pool bytes | Selected max_model_len |
|---|---:|---:|---:|---:|---:|---:|
| FP8 selectable | 576 / 132 | 3,098,112 | 1,039 | 66,496 | 3,218,944,000 | 66,112 |
| FP4x default | 368 / 132 | 2,046,464 | 1,573 | 100,672 | 3,219,093,504 | 98,176 |

Physical capacity is not the advertised request limit: the null block,
MTP lookahead and admission margin consume part of it. The selected limit
is 98,176; no larger limit is qualified by this release.
Boot g recorded 1,573 blocks and a healthy lease on every rank.
Post-admission MemAvailable minima in rank order were
9.132 / 9.942 / 10.281 / 9.948 GiB, with no fatal log entries.

Long prefill has a constructor-bounded bank plus returned logits of
406,324,480 bytes per rank (387.501221 MiB). Its physical row cap is
100,672. The separate split32 graph scratch remains bounded to 36 rows;
target and MTP reuse it sequentially. No request-length-sized candidate
merge or unused `40 * max_model_len` workspace is reserved.

## Prefill design

Scoring uses a fixed 256 MiB FP32 logits budget: Q2048/K32768 through 32K,
Q1024/K65536 through 64K and Q512/K131072 through 128K. Larger combined
prefix buckets use Q256/K262144 and Q128/K524288. Keys are gathered into
fixed banks; fused preparation copies and pads queries, weights and causal
bounds. Each subchunk receives its own bounds, including subchunks that
start partway through a request. The stock top-2048 prefill selector is
used without a candidate-merge stage or host tensor reads.

After selection, prefill marks the union of selected physical KV rows,
expands each row once to exact BF16 and calls the unchanged plain-cache
Triton sparse reader. The gather bank and BF16 expansion share one arena.
Reuse is across query rows; packed reconstruction is no longer repeated
for every selected occurrence. Pure and mixed decode keep their packed
readers and dispatch threshold.

The stock histogram selector assigns cutoff ties through atomics, without
a guaranteed ID order. The checker requires every ID strictly above the
cutoff, bit-identical selected-score multisets and permits differing IDs
only at cutoff scores with identical bits. It also checks bounds, unique
IDs, counts and `-1` sentinels. Signed-zero and one-ULP swaps are rejected.
Neither selector's tie policy is changed.

## Tests and qualification

Boot g's GPU receipt for `64269fe` reports PASS for read/write/gather/replay
and fixed-budget stock-top-k score-bit checks. The CPU tests cover packed
encoding, page alignment, readers, fixed banks, query subchunk bounds,
cutoff ties, integration and preservation of the FP8 kernels. The offline
draft checks launch rendering and receipt transcription; it does not
repeat GPU qualification or start a serving process.

| Boot g check | Receipt |
|---|---|
| Context ladder | PASS at 16K, 32K, 64K and 98K; exact prompt tokens 16,336 / 32,735 / 65,511 / 98,073 |
| Ladder TTFT | 26.6 / 41.6 / 87.1 / 136.2 s; separate prompts from sparkDash prefill |
| sparkDash cold prefill | 946 / 907 / 883 / 861 / 822 tok/s at 4K / 8K / 16K / 32K / 64K |
| Decode vs context | 29.64 / 28.66 / 28.29 / 27.63 tok/s at 0 / 30K / 60K / 96K |
| Retrieval | PASS, 9/10 at 16K / 63K / 96K, twice; all registry fields correct, all responses stopped |
| qeval | 71 / 73 / 73; mean 72.3333 above threshold 70.5535; two `code_interval_intersect` truncations |
| Prefix cache | PASS, identical answer; 2,048 hits on the second 2,150-token prompt; TTFT 2.27 → 0.68 s |

The coordinator RESULT qualifies boot g. Its legacy qeval admission JSON
has `mean_passes_threshold=true` and `passed_w4_rule=false`; the two
truncations explain the latter. No new recurring failure task was reported.
Retrieval's matching-count field was wrong in all six responses; the
registry/alias/operator fields were correct. These receipts establish the
recorded checks, rather than broad intelligence equivalence. A McNemar
statistic, twenty-seed retrieval campaign and full APC eviction/abort/page-
hash coverage are not reported by this qualification.

sparkDash code c1 is about 6 % below the paired FP8 layout while per-step
time is unchanged. RigMark is from boot q (`5d62f77`, FP4x, same selected
context); boot g has none. Do not combine the two boots into one benchmark.
The [public summary](results/fp4x-release-summary.json) contains only
allowlisted receipt fields. Earlier implementation forecasts and release
comparisons belong in [history](history.md), rather than current results.

MiaAI-Lab GLM-5.3 TensorFold recipe is credited for the FP4 latent KV format
and McNemar-style quality-check ideas only. The packed ABI, encoder,
fused implementation and tests are local work. vLLM supplies cache/indexer/
MTP/APC infrastructure; Triton and NVIDIA supply compiler/datatype work.
The existing sparse layout retains CosmicRaisins and Matt Mastracci credit.
No source or FP4 kernel from the idea reference is copied.
