# Real NVFP4 attention weights

Implemented on `perf/nvfp4-attn`, from serving `84a38e06f16ee09e0521a653851fe574bbf57fae`, in `/srv/projects/glm53-full-nvfp4a`. Mac-only preparation: no SSH, fleet operation, endpoint requests, pushes, image pulls or weight downloads. The requested diagnostic `IMPL_REAL.md` was absent; the saved `PROBE.md`, qualified simulator `8d169af`, actual checkpoint index, and pinned image Python sources supplied the contract.

## Kernel decision and source evidence

Use the image's **Marlin NVFP4 W4A16**, with BF16 activations and FP32 reduction. No activation quantization, FP4 activation MMA, custom Triton kernel, new JIT, or persistent BF16 weight bank. Existing compiled `vllm._C` supplies the GEMM and repack. Actual sm121 FP4 instantiation and latency still require the GB10 checker below; Mac shape checks are not a CUDA execution claim.

Pinned base: `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`, vLLM source `487ecf187`. Local evidence root: `/srv/campaign/diagnostics/glm53-full-20260929/day3/release-dirtyl2/image-source/`. [Source hashes](../../overlay/overlay/nvfp4_source_pins.json) cover 13 Python source files and are enforced at enabled startup. Sources are not modified.

Relevant paths below are relative to that image root:

- `vllm/model_executor/kernels/linear/nvfp4/marlin.py:18`: `"NVFP4 weight-only GEMM via Marlin (W4A16)."` Calls `prepare_fp4_layer_for_marlin` and `apply_fp4_marlin_linear`.
- `vllm/model_executor/kernels/linear/__init__.py:1020`: `elif linear_backend == "auto" and use_a16:` then `force_kernel = MarlinNvFp4LinearKernel`. Our scheme passes `use_a16=True` and rejects another selected kernel. Batch-invariant CUTLASS/emulation or an incompatible forced backend fails startup.
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:39`: `current_platform.is_cuda() and current_platform.has_device_capability(75)`. This source-level capability predicate includes 12.1.
- Same file, `apply_fp4_marlin_linear:199`: with `input_dtype=None`, `inputs = reshaped_x`, `a_scales = None`; `ops.marlin_gemm(... b_q_type=scalar_types.float4_e2m1f, global_scale=weight_global_scale, ...)`. BF16 is passed directly. Repacking uses contiguous `[N,K/2]` bytes viewed as INT32, transposed, and `gptq_marlin_repack(...num_bits=4)`.
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:29`: `CompressedTensorsW4A4Fp4(use_a16=True)` creates `[N,K/2]` U8 `weight_packed`, `[N,K/16]` E4M3FN `weight_scale`, and one FP32 `weight_global_scale` per logical output partition; it creates **no input scale** in A16 mode.
- Stock CT `process_weights_after_loading` uses `layer.weight_global_scale.max()` and then takes its reciprocal. Our sidecar uses **multipliers** and our scheme bypasses both operations. q_a and kv_a have different full-tensor global scales. The fused runtime A projection executes two separate Marlin calls followed by concatenation, preserving both globals without another quantization. No shared-global approximation is permitted.
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py:221`: Marlin accepts `(N%64==0,K%128==0)` or `(N%128==0,K%64==0)`, otherwise pads. Every selected rank shape satisfies the first family, including kv_a N576. No weight padding is needed.
- `vllm/model_executor/kernels/linear/nvfp4/cutlass.py:56` calls `scaled_fp4_quant(x, ...)` before `cutlass_scaled_fp4_mm`. It changes activations and is excluded.
- `vllm/model_executor/layers/attention/mla_attention.py:1022` expands `kv_b_proj` through `get_and_maybe_dequant_weights` into BF16 `W_UK_T` and `W_UV`. The generic dequant helper in `quantization/utils/quant_utils.py:553` uses an identity GEMM when it does not recognize the scheme. This is load-time work, with M512 here. Decode UK/UV stays BF16; its 572,522,496 B/rank cycle reads are unchanged.

Marlin's special scale transform zeroes positive scales below 1/64 when normalized full-tensor scales use maximum448. The converter/loader reject such scales rather than silently alter QDQ. All nine available real matrices passed this check. E4M3 scales and FP4 products are representable in BF16 before the FP32 global multiplier; global placement and GEMM reduction/output rounding still differ from a CPU FP32 weight reconstruction. This is an existing W4A16 dequantization path, not a claim of bit-identical GPU GEMM arithmetic.

## Shapes, time forecast, and memory

Matrices are listed as **N output × K input**, so the requested o_proj K4096×N6144 is N6144×K4096 here; q_b K2048×N4096 is N4096×K2048. q_a/kv_a are replicated, q_b/kv_b split output heads across TP4, and o_proj splits K. Full-tensor global scales are determined **before** TP slicing.

| Projection | N×K/rank | INT8 bytes | NVFP4 bytes incl global | Stream lower bound INT8 → FP4, µs @255GB/s |
|---|---:|---:|---:|---:|
| q_a | 2048×6144 | 12,779,520 | 7,077,892 | 50.12 → 27.76 |
| kv_a | 576×6144 | 3,594,240 | 1,990,660 | 14.09 → 7.81 |
| q_b | 4096×2048 | 8,519,680 | 4,718,596 | 33.41 → 18.50 |
| o_proj | 6144×4096 | 25,559,040 | 14,155,780 | 100.23 → 55.51 |
| kv_b (load/prefill) | 7168×512 | 3,727,360 | 2,064,388 | 14.62 → 8.10 |
| UK/UV (decode) | 16 heads, UK192×512 / UV512×256 | 7,340,032 BF16 | unchanged | 28.78 unchanged |

These are one-stream lower bounds, not measured kernel latency. For M1–16 the stream bounds dominate at an assumed effective BF16100TFLOP/s. Marlin may reread scales/tiles and incur substantial dequant overhead. The saved M3 INT8 trace has qkv_a74.69µs, q_b42.98µs, o_proj128.51µs per layer; discounted FP4 estimates are qkv_a64.52µs (both calls and concat, allowing6.49µs extra/layer), q_b33.39µs, o_proj99.84µs. q_a/kv_a standalone INT8 latency was not independently measured.

[cost-model.json](cost-model.json) contains per-M roofline estimates at M1–16,32,64,128,256,512,1024,2048,4096 for every projection, generated by `scripts/nvfp4_attn_cost.py`. Equation: `max(B/255GB/s, 2*M*N*K/100TFLOP/s)`, equal BF16 compute assumptions for W8 and W4. At M4096, compute lower bounds are q_a1.031ms, kv_a0.290ms, q_b0.687ms, o_proj2.062ms, kv_b0.301ms **for either format**. No prefill speedup is promised; scale instructions and retiled traffic can make W4 slower. GPU benchmarks cover these M values and compare INT8 Marlin with NVFP4, including the two-call A projection.

Decode attention traffic drops **3,884,840,960 → 2,151,605,456 B/rank/cycle**, excluding kv_b's retained BF16 absorption. At255GB/s, halve the ideal byte saving for dequant/layout inefficiency and subtract0.5ms/cycle for the additional77 launches and concat operations: **2.90ms saved**. Applying the same half-discount to the saved18.973ms INT8 trace gives **3.73ms**. On the83ms prior, forecast **cycle-ms −3.49% to −4.50%, cycle rate +3.62% to +4.71%** at unchanged committed/cycle. This estimate has little slack against the −3% gate; actual FP4 kernel overhead can erase it. Recalculate using the normal FP4x arm's measured cycle time.

Across77 quantized target layers, steady packed attention residency including kv_b goes **4,171,847,680 → 2,310,563,332 B/rank**. Subtract14,784 B for77 extra48-SM INT32 workspaces: **1,861,269,564 B/rank freed =1.861GB =1.733GiB**. Layer0 and MTP78 stay original. Allocator bookkeeping, transient repack/identity-GEMM scratch, graphs and concatenation buffers need actual memory receipts; the byte ledger is not an admission result.

The display carveout remains2,145,386,496 B. If all predicted savings become ordinary KV, its head can rise **1,073,741,824 → 2,935,011,388 B/rank**. FP4x uses2,046,464 B per64-token block (79 MLA layers×368 plus22 indexer layers×132 B/token, 512B allocation alignment): **1,573 → 2,482 blocks**, +909 blocks,158,848 physical token slots. Six-block margin gives a theoretical158,464-token cap. Keep **the served98,176 cap and original1,573 blocks** in both A/B boots. This implementation does not alter the launcher's pinned KV pool or the carveout. Larger context needs later memory/metadata/admission and quality qualification.

## Exact conversion and loader contract

The exhaustive [tensors.txt](tensors.txt) lists all385 `.weight_packed` names, verified against the actual index. They are exactly:

`model.layers.{1..77}.self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}.weight_packed`.

Each has `.weight_scale` and `.weight_shape` source companions. Layer0 BF16, MTP78 attention/channel-INT8, shared experts, dense MLP, routers, indexer and head are excluded. There is no separately stored UK/UV checkpoint tensor.

The offline converter reads biased unsigned INT8 bytes packed four per INT32 word, subtracts128, and multiplies original BF16 group128 scales in FP32. It makes two64-row passes per logical matrix: full-tensor absmax, then block16 quantization. Global `t=FP32(amax/(6*448))`; E4M3FN block scale uses `(block_amax/6)/t` with RNE; signed E2M1 magnitudes `{0,.5,1,1.5,2,3,4,6}` use RNE. The low nibble is the earlier K element. Zero matrices use all-zero codes/scales/global. Finite/shape/nonnegative/range checks fail closed.

Sidecar files contain `weight_packed` U8[N,K/2], `weight_scale` E4M3FN[N,K/16], `weight_global_scale` FP32[1] **multiplier**. FP32 decoding uses `fp4 * (scale8*t)`, the simulator's exact operation order. This is bit-exact with its **FP4 QDQ stage**. The qualified simulator then recalculates BF16 group128 scales and re-encodes INT8; those served values cannot generally be represented exactly by a pure FP4 block16 tensor. We retain a test showing that difference. No promise of bit equality to the simulator's final served INT8 grid is made; the fresh real-kernel quality gate is required. Re-encoding the simulator's INT8 grid into FP4 would introduce a second quantization and is excluded.

Conversion writes a separate **new sidecar directory**, outside and disjoint from the source checkpoint. It never edits, hardlinks, copies back, or rewrites original shards/config/tokenizer. It reads local files only. A nonblocking process lock prevents concurrent conversion; data and progress manifest are atomically replaced. Source index/config and all82 referenced source shards are SHA256-bound. Resume rehashes source and completed outputs and rejects drift/corruption. Each matrix has a hash; inventory and final manifest appear in `SHA256SUMS`. Preserve an external hash of that checksum file in receipts. Interrupted uncommitted matrices are rebuilt; incomplete manifests cannot serve. `.convert.lock`, temporary files and `SHA256SUMS` are control artifacts, with the latter externally hashable.

Full conversion handles12.707B elements, writes about7.147GB of FP4 data and scans/hashes123,987,267,848 B of82 source shards, per node. Estimated **10–30min per Spark**, assuming10–40M quantized elements/s and0.5–2GB/s hashing/storage; allow45min including contention and reserve disk space. Run once against each node's existing checkpoint copy; all four output hashes must agree. This is an estimate, not a Spark measurement. Nine available real matrices converted on the Mac in8.36s; all296,484,864 reconstructed elements were bit-exact against the qualified simulator QDQ and source shard hashes remained unchanged; [receipt](mac-real-samples.json). Partial output is under `/private/tmp/glm53-nvfp4a-real-samples`, cannot serve, and is not a full checkpoint.

`GLM_ATTN_WEIGHTS=int8|nvfp4` defaults to int8. Default launch vectors have no added container env/mount. In NVFP4 mode, `GLM_ATTN_NVFP4_DIR` is a separate absolute host path mounted read-only at `/attn-nvfp4` on every rank. `/model` stays the original INT8 checkpoint. Enabled startup checks source pins and forces the custom attention-only scheme. The iterator hashes/validates the completed sidecar and checks original index/config before emitting replacement packed/scales/global companions and suppressing original shape/scales. It demands385 logical replacements per target load. Native draft loading bypasses the sidecar and retains INT8. Both fast and stock safetensors strategies compose with existing MTP selection. Combining real NVFP4 with `GLM_NVFP4_WSIM` is rejected. All preparation occurs before graph capture.

## Local verification

CPU tests cover random-weight bit-exact FP32 dequant vs the unchanged simulator reference, every rounding midpoint, zero/sign/nonfinite and Marlin scale range, row chunk invariance, conversion/resume/cross-shard reads/hash drift/source immutability, full inventory, default mode, loader companions/counts, MTP exclusion, actual pinned parameter TP slicing and CT parameter creation, independent fused globals, 120 pinned FP4 dispatch calls over the requested shapes/M grid, and all-rank dry mounts/env. Existing recipe/source tests and fast-loader byte/wiring tests are retained. The copied quality helpers retain strict mean-score/needle/teacher-forcing tests. The final [verification receipt](verification.json) records counts and command results.

No new Triton/JIT kernel exists, so a sm121 **JIT static compile check is not applicable**. The pinned Python dispatch and sources are checked on the Mac; the later GB10 tool resolves and executes the native repack/FP4 GEMM symbols, compares to QDQ BF16 matmul (NRMSE≤0.02, finite outputs), captures CUDA graphs and checks M1–16 and prefill through4096 against W8A16. Keep eager numerical checks separate from graph replay timings. GPU results are pending.

## Later fleet plan — not executed

Use one exclusive, coordinated window. Preserve the exact production boot manifest, dispram/lenders/guard integration, watchdog hold and restoration procedure. Both boots use the served FP4x KV, adaptive chunks, K-stop, graph descriptors, TP4, ordinary/dispram pools, dirty-L2 policy, tokenizer/checkpoint and98,176 cap. The serving external guard is required; do not silently substitute the stock32K profile. The simulator window used FP8 KV; its scores are quality evidence, not a substitute for this FP4x baseline.

1. Before stopping production, stage this commit locally in an experiment clone and review all-rank DRY vectors. Run offline conversion against each already installed checkpoint in a new sidecar directory. Verify `SHA256SUMS`, source hashes,385 matrices, same hashes on every node, enough disk and no writes to source. Preserve converter log and externally hash `SHA256SUMS`.

   ```sh
   # Later, on each Spark, with torch/numpy/safetensors already in the pinned image.
   python3 scripts/convert_attn_nvfp4.py /local/original/checkpoint /local/new/attn-nvfp4
   (cd /local/new/attn-nvfp4 && sha256sum -c SHA256SUMS)
   sha256sum /local/new/attn-nvfp4/SHA256SUMS
   ```

2. After coordinated drain/stop/compaction, run the GB10 kernel checker on one otherwise idle node, inside the existing pinned image with original checkpoint and sidecar read-only. No image pull/build or model boot is needed. Reject unavailable FP4 op, compilation/runtime failure, nonfinite output, excessive numerical error, or pathological decode/prefill latency. This checker uses warm repeated weight access, so its timings are not the full-model DRAM/cycle forecast.

   ```sh
   python3 bench/attn_nvfp4_gemm.py --checkpoint /model --sidecar /attn-nvfp4 \
     --layer 1 --rank 0 --out /receipts/gb10-kernels.json
   ```

3. **Boot A int8**, with sidecar switch off, both KV formats FP4x as served. Warm all graph descriptors and2K/16K/63K prefill. Freeze the next-token panel once. Require a valid A/A logprob panel, health, correct KV ABI/count, capture/replay and ordinary memory floor before proceeding. Run the probes below. Stop A through the coordinator.

4. **Boot B nvfp4**, same boot vector plus only weight env/mount. Verify all four target logs say `loaded 385 logical attention matrices`; draft MTP stays INT8. Record before/after RAM, model tensor bytes, transient load peak, cache/capture sizes and cycle shapes. Run identical probes. No extra boot for another configuration.

   ```sh
   # Two sequential boot slots, with coordinated stop between them.
   GLM_ATTN_WEIGHTS=int8 GLM_KV_FORMAT=fp4x "$STACK/start.sh" serve
   GLM_ATTN_WEIGHTS=nvfp4 GLM_ATTN_NVFP4_DIR=/local/new/attn-nvfp4 \
     GLM_KV_FORMAT=fp4x "$STACK/start.sh" serve
   # DRY=1 added to either command renders locally with no SSH or state access.
   ```

   These require the copied **served** profile/integration/cap rather than the repository's default example `.env`. Do not put both commands into an unattended sequential shell script: `serve` is a foreground monitor. Keep all performance/quality probes exclusive c1; no unrelated inference requests.

5. For each arm, discard one prose and one code warm-up; collect **20 scored requests per kind** with request-bound `gate_metrics`, using identical prompts/seeds, fixed256-token budgets and low reasoning effort. `nvfp4_attn_cycles.py` retains all Prometheus before/after scrapes and hashes and refuses incomplete/short/reset/mixed-traffic samples. Its cycle-ms is **request decode wall / draft opportunities**, not a CUDA-event hardware-cycle median. Retain ordinary GPU-event cycle profiling if the serving gate requires that measure; do not reinterpret this proxy as hardware time.

   ```sh
   python3 bench/nvfp4_attn_cycles.py run --base "$BASE" --metrics "$BASE/metrics" \
     --out "$R/$ARM/cycles" --arm "$ARM"
   # After both arms:
   python3 bench/nvfp4_attn_cycles.py compare "$R/A/cycles" "$R/B/cycles" --out "$R/cycle-gate.json"
   ```

   The metrics implementation is copied byte-for-byte from `day3/think10/round5-gpu/gate_metrics.py`; its request-bound limitations remain explicit. Pair prompt hashes, seed, K mode, target-M policy, metric identity and time source. Report raw rows, medians, pooled committed/cycle ratios and uncertainty per prose/code. Require median paired cycle ratio≤0.97 **and** committed/cycle ratio≥0.99 for **each kind**. No TPS-only substitute and no cherry-picked repetitions. Report acceptance, K-stop and expert unions to explain divergence. If noise straddles the gate, the two-boot window is inconclusive and no adoption follows.

6. sparkDash prose/code c1: one warm-up plus five scored runs of each, all streams valid. qeval×3, unchanged75-task budgets, report all-task mean and55 primary tasks. Needle16K=16,384;63K=63,488;96K=96,000 input tokens, each20 fixed seeds at2/25/50/75/98% depth. RigMark pinned source, three scored4096-token prose/code/structured runs after discarded warm-up, low effort, same metadata/source hash; no prefill/concurrency sweep. Frozen next-token probes use the same actual token IDs and teacher-forced logprobs across arms.

   ```sh
   # Freeze on A only; BASE is the direct existing model endpoint, DASH the existing sparkDash API.
   python3 bench/nvfp4_quality.py freeze --base "$BASE" --out "$R/frozen.json"
   python3 bench/nvfp4_quality.py collect --base "$BASE" --panel "$R/frozen.json" --out "$R/A/logprob-aa.json"
   # Per admitted arm, ARM=A then ARM=B; mkdir receipt/qeval directories beforehand.
   python3 bench/nvfp4_quality.py collect --base "$BASE" --panel "$R/frozen.json" --out "$R/$ARM/logprob.json"
   python3 bench/nvfp4_quality.py dash --base "$DASH" --out "$R/$ARM/sparkdash.json"
   (cd "$R/$ARM/qeval" && for i in 1 2 3; do
     python3 "$STACK/bench/qeval.py" run "nvfp4-attn-$ARM-$i" --url "$BASE/v1/chat/completions" --concurrency 1
   done)
   python3 bench/nvfp4_quality.py needles --base "$BASE" --out "$R/$ARM/needles.json"
   # After B:
   python3 bench/nvfp4_quality.py qgate "$R/A/qeval" "$R/B/qeval" --out "$R/qeval-gate.json"
   python3 bench/nvfp4_quality.py ngate "$R/A/needles.json" "$R/B/needles.json" --out "$R/needle-gate.json"
   python3 bench/nvfp4_quality.py drift "$R/A/logprob-aa.json" "$R/A/logprob.json" --out "$R/aa-drift.json"
   python3 bench/nvfp4_quality.py drift "$R/A/logprob.json" "$R/B/logprob.json" --out "$R/drift-gate.json"
   python3 "$RIGMARK" run --base-url "$BASE" --model GLM-5.3 \
     --label "nvfp4-attn-$ARM" --comparison-id nvfp4-attn \
     --metadata "$R/$ARM/rigmark-metadata.json" \
     --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
     --runs 3 --decode-tokens 4096 --skip-prefill --skip-concurrency --output "$R/$ARM/rigmark.json"
   ```

7. Accept quality only if qeval mean drop≤0.7 passed task versus same-day A, all needles pass with identical fixture hashes, max frozen-token drift≤0.3nat with A/A≤0.3, valid sparkDash/RigMark functional gates and no prose/code degeneration. Supplied simulator results (70/72/70→74/73/72;16K/63K passes; RigMark25.06→25.40 prose,36.46→36.54 code; MTP0.988/1.006; drift0.213 versus A/A0.298) are a prior. Require the real FP4x run to hold those quality gates, including new96K retrieval. Missing receipts, wrong tensor counts, OOM/capture/health/admission failure or any failed gate rejects B.

8. Stop the experiment and restore the exact saved normal INT8 stack, FP4x KV and98,176 cap through the coordinator; verify health, watchdog/heartbeat, original endpoints and returned dispram lease before releasing the hold. Keep normal serving unchanged unless the later operator explicitly promotes a fully passing result. Extra KV capacity is a separate later memory/context experiment, not a third boot in this window.
