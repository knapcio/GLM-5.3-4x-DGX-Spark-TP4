# Acceptance recovery design — 2026-10-06

**Recommendation: target-only recent FP8 latent KV, W=2048, with FP4x backing retained.** Implemented locally on `perf/accept-recovery` from `perf/integ-1006` / `3316eb4`. The release defaults remain off. Keep NVFP4 attention weights, the 1573-block FP4x pool and the serving max_model_len=98176. No SSH, fleet calls, endpoint calls, image pulls or pushes were performed.

Planning forecast: **+2% relative committed/cycle in prose, +1% in code** versus FP4x + NVFP4. Plausible short-context gain +1..3% / +0.5..1.5%; null recovery remains possible. These are hypotheses, not measurements or statistical confidence intervals. They target the KV deficit; do not credit recovery of NVFP4's separate 2.6% pooled prose deficit. With cycle overhead <=1%, the central forecast gives about +1% prose tok/s and approximately flat code tok/s. Gate on tok/s, not acceptance alone.

## Evidence and what it establishes

Receipt roots under `receipts/`:

* `glm53-full-20261006-v2cycle/RESULT.json`: FP8 vs FP4x, prose 2.796 vs 2.704 committed/cycle (-3.29%), code 3.740 vs 3.669 (-1.90%). Cycle 76.51 vs 77.12 ms / 83.11 vs 83.76 ms. sparkDash prose c1 32.65 vs 30.65 tok/s (-6.13%). The difference between the throughput and committed ratios cannot all be assigned to quantization acceptance.
* `glm53-full-20261006-mtpfp8/window/compare-m.json`: MTP-layer-only FP8 vs all-FP4x e, pooled prose 0.9942, code 1.0047. Against FP8 C, prose 0.9856 [0.9709,0.9997], code 0.9949 [0.9805,1.0076]. No useful MTP-only precision recovery. Trajectories and small samples limit causal certainty; this is strong prioritization evidence for the target.
* `glm53-full-20261006-nvfp4a/window/RESULT.json`: NVFP4 vs INT8, both FP4x, pooled committed prose 0.97374, code 0.99812. Paired median cycle ratio 0.90922 / 0.92825. NVFP4 quiet rank-0 memory 11.181 GiB and post-admission minimum 10.711 GiB; INT8 quiet 9.567 GiB. This memory saving can fund a sidecar without shrinking the FP4x pool.
* `glm53-full-20261005-fp4kv/IMPL.md` and serving source at 3316eb4: FP4x rows contain 256 latent code bytes, 32 E4M3 scale/16 bytes, an **FP32** pow2 row scale, 12 padding bytes, and 64 E4M3 RoPE bytes. DSA index keys already use E4M3 FP8 plus FP32 scale. The draft uses the same FP4x writer/readers. NVFP4 selects target layers 1..77 attention projections; layer 0 and MTP78 retain their existing weight precision.

## Offline reconstruction and attention experiment

Inputs: original BF16 captures at
`glm53-full-20260929/day3/release-stack/stack-1002b/fp4probe-20261005-0855/normale/captures/*.pt`.
The reproducible evaluator is `bench/fp4_accept_eval.py`. Artifacts beside DESIGN.md: `offline.json`, `offline.log`, `summary.json`, `age-errors.csv`; the JSON contains SHA256 of all 273 input files.

Each case/layer/epoch is reconstructed in writer order. Physical-slot overwrites retain logical positions, including rejected draft tails. Query selections are the recorded actual sparse selections; no absent prefix is replaced with zeros. Ages are relative to the captured query's logical position. The evaluator uses the last query of each decode event and the last query of the last prefill event: 129 queries, 95 in target layers 0/39/77, 34 in MTP78; **zero missing-prefix queries**. Short prose/code epochs have 65/74 distinct target positions; long epochs have about 16,412..16,416 / 61,464..61,468 positions. Some short prompt events are marked decode by the stock short-attention dispatch.

Compare FP8 reconstruction under the recorded writer/reader scales to FP4x, keeping E4M3 RoPE common. Re-run FP32 softmax, latent PV and captured W_UV projection with exactly the same FP8 queries and top-k. For hybrid replay, only selected tokens of age < W use FP8. The table aggregates target decode samples by squared error/reference energy, not by averaging NRMSEs.

| Case | FP4x projected-output NRMSE vs FP8 | W=2K | W=4K | W=8K | 2K residual/error ratio |
|---|---:|---:|---:|---:|---:|
| prose | 9.755% | 0 | 0 | 0 | 0 |
| code | 9.610% | 0 | 0 | 0 | 0 |
| long16k | 8.039% | 4.695% | 4.696% | 4.694% | 0.584 |
| long60k | 8.199% | 4.715% | 4.715% | 4.678% | 0.575 |

Zero short-case error is expected because these entire captured contexts fit in 2K; it does **not** demonstrate zero error through all 78 target layers or exact acceptance recovery. At long contexts 2K removes 65.9% / 66.9% of projected error energy, or 41.6% / 42.5% of NRMSE. Extending to 8K changes the 60K residual NRMSE ratio by only 0.0045. This makes 2K the first falsifier, not 8K.

Attention mass alone does not support an assumption that most mass is recent:

| Target decode | age 0..2047 | 2048..4095 | 4096..8191 | >=8192 |
|---|---:|---:|---:|---:|
| long16k FP8 attention mass | 34.30% | 5.74% | 10.89% | 49.07% |
| long60k FP8 attention mass | 29.45% | 3.86% | 8.09% | 58.61% |
| long16k selected rows | 20.27% | 8.89% | 21.90% | 48.94% |
| long60k selected rows | 10.59% | 6.48% | 15.14% | 67.79% |

The reconstruction error itself is flat by age: at 16K, FP4-vs-BF16 NRMSE is 9.475% in recent 2K vs 9.485% at >=8K; at 60K, 9.492% vs 9.487%. FP8 reconstruction NRMSE is about 2.66% in all age bins. The benefit comes from how those rows affect the output, not larger quantization errors in recent rows.

The recent rows nevertheless dominate the error-bearing output contribution. Exact decomposition of `o4-o8` is `w4*(x4-x8)` (PV term) plus `(w4-w8)*x8` (QK term). At 16K, recent-2K PV vector SSE is 45.349 vs >=8K 14.499; QK SSE 4.250 vs 0.962. At 60K, corresponding PV 37.797 vs 13.410, QK 4.221 vs 1.097. Per-bin SSEs **do not add** because vectors cancel or reinforce across bins. `age-errors.csv` gives reconstruction error, attention mass and PV/QK contributions by age. Do not interpret top-2048 row count as attention mass.

Scale diagnosis (target decode; projected-output residual NRMSE / FP4x baseline):

| Variant | prose | code | 16K | 60K |
|---|---:|---:|---:|---:|
| non-pow2 FP32 row scale | 1.002 | 1.007 | 1.016 | 0.987 |
| BF16 direct scale per 16 | 0.971 | 0.983 | 1.004 | 1.004 |
| FP32 direct scale per 16 | 0.969 | 0.984 | 1.009 | 1.004 |
| E4M3 scale per 8 | 0.910 | 0.910 | 0.888 | 0.903 |
| one guarded least-squares scale/16 fit | 0.974 | 0.970 | 0.962 | 0.957 |

Across cases the original latent reconstruction NRMSE vs BF16 is approximately 9.43..9.48%; direct FP32 scales reduce it to 9.32..9.40%, scale/8 to 8.49..8.52%, and the fitted scale/16 to 9.05..9.14%. Most error comes from E2M1 value quantization/block width, not storing an FP32 row scale as pow2. FP32 scale precision cannot by itself remove the approximately 9% latent error. Removing pow2 has no consistent output benefit and is rejected as the first intervention.

Limitations: rank-0 heads, four layers, fixed FP8/INT8-run queries and top-k, FP32 diagnostic matmuls/softmax rather than compiled split32 BF16 probability arithmetic. No downstream hidden states, final vocabulary logits, draft logits or NVFP4-weight-run captures. Changing target KV also changes later queries, indexer selections, draft conditioning and trajectories. Therefore output error reduction is an engineering proxy; no conversion from NRMSE to committed/cycle is identified by these dumps. All recovery estimates below require a fleet falsifier.

## Ranked interventions and falsifiers

| Rank | Intervention | Acceptance forecast, relative | Memory / compute | Cheapest falsifier |
|---|---|---|---|---|
| 1 | Target recent FP8 latent, W=2K; keep old FP4x and NVFP4 weights | central prose +2%, code +1%; plausible +1..3% / +0.5..1.5%; zero possible | implementation +986.2 MiB/rank; one target-step staging launch; FP8 encoding fused into existing writer; no dequant launch; mixed reader | One candidate boot, device-backed 0/2K/0 switch, drained requests and APC reset, identical 20-prompt cycle panel + c1/c4 and long guards |
| 2 | Guarded least-squares fit of existing E4M3 scale/16 | approximately 0..0.2%; extrapolation from 2.6..4.3% output-error reduction | zero KV byte increase; unchanged reader ABI; second code assignment/reductions in writer and fresh QDQ | Device scalar in writer/fresh-QDQ, same-boot 0/1/0; kill if prose gain <0.2% or cycle cost outweighs it |
| 3 | E4M3 block scales per 8 | approximately 0..0.5%; extrapolation from 9..11% output-error reduction | 400-byte row after alignment, +32 bytes/target row, +239.6 MiB at 100672 slots; higher KV head needed to preserve blocks; more scale traffic | First same-boot QDQ simulator using BF16 eager probe, then real ABI only if it shows >=0.5% committed gain |
| 4 | BF16/FP32 direct scale per 16 | approximately 0..0.1%; long-output error did not improve | 384/448-byte aligned rows, +119.8/+599.1 MiB for 78 targets at fixed slots; different ABI, more reader bytes | Offline already weak; same-boot eager QDQ simulation before investing in a storage ABI |
| 5 | Selective target attention-weight precision at sensitivity outliers | unknown, 0..2.6% prose weight deficit is only a planning ceiling | reinstate INT8 for a few matrices, preserving most NVFP4; exact matrix timing needed, cap extra cycle cost at 0.5% | Same-boot device-controlled original/NVFP4 output blend for a bounded matrix list on frozen tokens; reject unless net tok/s wins |
| 6 | Draft calibration / matching quantizer | unknown; no evidence of positive gain | greedy temperature scaling costs ~0 but cannot change argmax; bias/hidden affine calibration needs paired target/draft logits; MTP NVFP4 would use distinct draft weights | Capture fixed-token target/draft top-2 margins under both target KV numerics, then test a tiny inference-only correction on held-out prompts; no fine-tuning |
| eliminated | FP8 indexer keys | 0 from switching to precision already in use | existing 128 E4M3 bytes + 4 FP32 bytes unchanged | Source audit is sufficient. BF16 keys do not reproduce FP8 selection exactly; upstream hidden-state drift remains |
| eliminated | non-pow2 row scale / MTP-only FP8 KV | no consistent benefit / no measured recovery | row-scale mode is byte-free but re-encodes; MTP FP8 adds 208 bytes/token for layer78 | Offline free-row results and the mtpfp8 fleet receipt already falsify prioritization |

Selective INT8 is a contingency for the **weight** deficit, not part of the implementation. It would spend some NVFP4 speed and must pass the explicit cycle budget. With the user's speed constraint, rank 1 is preferred. A useful extra diagnostic is per-layer last-token hidden/logit drift, which can distinguish target KV error from weight drift and indirect DSA selection changes. The existing frozen-token logprob receipt indicates a larger prose-pl weight effect (0.652 nat vs A/A 0.221), not a reason to widen all KV to 8K.

## Implemented storage and execution

Files: `glm_recent_kv.py`, `glm_recent_kv_kernel.py`, the fused FP4x writer, shared latent loader, split32/unsplit readers, prefix gather, union expansion, fresh dense QDQ, bootstrap and launcher wiring. MTP78 keeps its original FP4x path. Target RoPE and all DSA cache formats remain unchanged.

* Allocate 78 target sidecars in the attention constructor, before profiling/capture. Each contains FP8 **latent only**, physical-slot tags and absolute-position tags. The existing FP4x cache remains fully populated, so eviction, missing sidecar rows and the A control have a complete fallback.
* V2 `prepare_attn` / `prepare_dummy_attn` hook stages `InputBatch.idx_mapping` (stable request-state index, not batch order), positions and query bounds into fixed device buffers. Supports the existing four request slots and <=4096 query rows. Reorder does not change ring ownership. Out-of-range request indices fail via a device assertion; excessive batch sizes fail before launch.
* Ring capacity per request = W + 4096 + 36. The extra rows hold the full current prefill/verify chunk without concurrent modulo aliases. During a multi-token batch, the high-precision region starts W-1 rows before the **first** query; all current rows are FP8. Thus decode/verification can include up to W+35 rows and a 4096-row prefill can include up to W+4095. This is a deliberate, bounded precision guard, not an exact per-query cutoff at W. Exact W per query would require query-aware union expansion or repeated expansion and is not worth that cost in the first falsifier.
* The fused writer stores the original BF16 latent as E4M3 using the released writer k_scale. Unit host latent reader scale is required and checked. It also writes FP4x and updates reverse maps/tags. No extra per-layer launch. A wrapped/overwritten ring row fails its physical-slot tag and falls back to FP4x. Accepted replacement of a rejected tail writes new sidecar and backing bytes together.
* Split32, unsplit, cached-prefix gather and union expansion consume the same tagged loader. Hits mask out the FP4 latent reads and load 512 FP8 bytes; misses use the original FP4x reconstruction. RoPE always comes from its existing FP8 bytes. Dense fresh prefill uses the original BF16 latent when enabled, matching the stock FP8 fresh-MHA behavior; the control uses the existing FP4 QDQ.
* APC shares the unchanged FP4x pages. An intact sidecar entry for a shared physical row may be reused; absent entries safely use FP4x. **An APC-only prefix cannot be promoted to FP8 from FP4x bytes**, and bounds belong to its writing owner, so residency can be opportunistic across shared prefixes. Do not claim all APC recent rows are FP8. Drained, cache-reset fresh contexts are required for the precision A/B. Testing fresh vs APC must be separate.
* `GLM_FP4_RECENT_WINDOW=0` is the inert default. Opt-in `2048|4096|8192` reserves that maximum. AB mode uses `GLM_FP4_RECENT_AB=1`, `GLM_FP4_RECENT_INIT=0|2048|4096|8192` <= reserved maximum, and private `VLLM_SERVER_DEV_MODE=1`. Worker `recent_kv_set` changes an existing device int32 flag in every rank. Captured readers/QDQ read that scalar; no graph recapture or Python-only replay switch. Both arms continue shadow writes, so A/B time excludes allocation and different writer work. `recent_kv_status` reports the active/reserved size and bound caches. The overlay hash separates persistent compilation cache generations.

## Memory and compute ledger

Ideal format replacement is `78 * slots * W * (576-368)` bytes: W=2K/4K/8K costs **31.6875/63.375/126.75 MiB per request**, or 126.75/253.5/507 MiB for four slots. That assumes replacement storage, no duplicate backing, no prefill guard and excludes MTP. It is not the implementation's allocation.

Implemented sidecar bytes, with N=100672 physical slots:

`78 * [4 * (W+4096+36) * (512+4+4) + 4*N] + 4096*8 + 4*4 + 4`.

| Reserved W | Sidecar MiB/rank | Estimated quiet rank-0 GiB from NVFP4 B | Estimated observed-min-minus-allocation GiB |
|---|---:|---:|---:|
| 2K | 986.181 | 10.218 | 9.748 |
| 4K | 1303.056 | 9.908 | 9.438 |
| 8K | 1936.806 | 9.290 | 8.820 |

These are subtraction estimates, **not calibrated admission results**. Allocator rounding, graph/compiler resources and peak prefill memory are additional uncertainties. The existing 10-GiB pre-capture guard stays intact; the simple 4K/8K estimates already fall below it. Use 2K only for the first boot, retain >=8.5 GiB exact-layout lower bound and >=8.0 GiB live watchdog, and recalibrate instead of relying on the old expanded-context simulator. No page-cache credit or extra context expansion is claimed. Sidecars are ordinary CUDA allocations, not included in the dispram KV bytes; constructors make their cost visible to profiling.

Worst-case selected latent traffic increase is approximately `78 * 2048 * (512-292) = 35.1 MB/cycle/rank`, plus map/tag/bound loads; actual increase is smaller for old/missing rows. This is ~0.18..1.17 ms at an assumed effective 200..30 GB/s, before encoder and staging cost. Those bandwidths are sensitivity assumptions, not measured GB10 bandwidth. Compare to the ~75.6-ms NVFP4 prose cycle: an added 1 ms is 1.32% and may wipe out a small code acceptance win. Static register/spill/occupancy and real same-boot cycle measurements are mandatory before promotion. The NVFP4 matrices and Marlin dispatch remain unchanged.

## Cheapest fleet falsifier — prepared, not executed

Use one isolated candidate boot from the exact saved NVFP4-B vector, changing only the commit/overlay and these three selectors; keep FP4x, max_model_len=98176, 1573 blocks, 1-GiB ordinary KV head, dispram require, k2/reuse, adaptive prefill, graph descriptors, clocks and weights fixed:

```sh
export GLM_FP4_RECENT_WINDOW=2048
export GLM_FP4_RECENT_AB=1 GLM_FP4_RECENT_INIT=0
export VLLM_SERVER_DEV_MODE=1
```

1. Before any boot: complete real Triton interpreter and static compilation of the recent writer/loader in split32, unsplit, gather, union and fresh-QDQ shapes; check zero spills and resources against released kernels. Replay cases include request reorder/reuse, APC shared/evicted blocks, ordinary/dispram seam, modulo wrap, padding and rejected tails. Mac AST tests are not substitutes for these checks. Recalculate memory with the exact sidecar allocation. Do not weaken the capture/watchdog guards.
2. One candidate boot; warm all target/MTP descriptors and prefill paths with the reserved sidecar present. Require `recent_kv_status` on all four ranks to report W reserved=2048, active=0 and 78 bound target caches after warmup. Save memory, block count, hashes and source/graph identity. Qualify 16K/63K/96K retrieval and c4 shared-pool use at the current serving context, not expanded integ headroom.
3. At each drained arm boundary POST `/reset_prefix_cache`; require `{"success":true}`. Switch via private-loopback collective RPC:

   ```json
   {"method":"recent_kv_set","args":["2048"],"timeout":30}
   ```

   Use `"0"` for the control. Read `/collective_rpc` method `recent_kv_status`; all four active windows must agree. Discard warmups. Existing graphs must keep the same addresses. Never switch while requests are live; do not expose the dev API externally.
4. Run A(0)/B(2K)/A(0), then reverse order B/A/B if the result is near the threshold. Same existing 10 prose +10 code prompts, maxTokens=256, temperature=0, exact tokenizer/prompts and metric deltas. Also five sparkDash prose c1 repeats, three code c1 repeats, distinct-prompt c4; include at least two >8K prose/code contexts. Bootstrap paired committed and cycle deltas across prompts; preserve all outputs, not just medians. APC disabled by cache-reset for this acceptance experiment. Add a separate APC correctness/reuse guard.
5. **Kill:** short prose paired committed gain <=0.5% after repeated blocks, code regression >0.5%, candidate cycle ratio >1.01, or no net c1 tok/s gain; also kill on new recurrent quality failures, allocation/graph errors, memory-floor or swap violation. A/A drift comparable to A/B means inconclusive, not success. **Promote only** with prose >=1% committed gain, code nonregression, actual c1 tok/s gain, cycle <=1.01, existing qeval/needles/APC/c4 and memory checks passing.
6. The reserved A control shares writer/metadata overhead with B. Compare its cycle against the existing NVFP4 B receipt and, if promising, a baseline boot to measure that absolute overhead. A/B alone cannot prove preserving the original NVFP4 speed. A failed candidate returns to the untouched serving vector; runtime `0` disables the precision experiment but keeps allocations/writes, so it is not a complete performance rollback.

## Local verification

Mac CPU tests: **165 passed, zero skips** across recent KV (12), FP4 (35), launcher (59), recipe (24), persistent cache (10) and NVFP4 (25). Actual Triton AST bodies cover stable request reorder, ring wrap/stale reverse maps, padded graph request descriptors without idx_mapping overreads, rejected-tail rewrites, missing APC rows, device flags, constructor reservation, fused writer, union/prefix gather and split32/unsplit readers. Recent FP8 reconstruction matches 54 BF16 rows from 18 target groups; the existing FP4 suite checks 68 BF16 rows from 24 case/layer/stage groups against the original probe encoder. `verification.json` records counts, source pins, hashes, exact Python paths and the final commit.

Real interpreter/static CUDA compile/graph replay and fleet acceptance are **unverified**. The Mac Colima Docker socket returned permission denied before a container could be started. No GPU speed, spills, register counts or measured acceptance recovery is claimed.
