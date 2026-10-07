# NVFP4 more — offline preparation, 2026-10-06

Branch `perf/nvfp4-more`, based on integration `3316eb4` (serving attention `3488797` included). Worktree `/srv/projects/glm53-full-nvfp4more`. This work is Mac-only: no SSH, fleet requests, model boots, pushes, downloads, or CUDA execution. The user's completed 385-matrix attention audit is the attention control; this extension does not reconvert or replace its serving sidecar.

## Delivered path

`GLM_ATTN_WEIGHTS=nvfp4` retains the real Marlin W4A16 activation/reduction contract. `GLM_NVFP4_GROUPS` defaults to **attn only**. An explicit `attn` has the same container flags and mount as the default; the Mac test compares DRY output against `perf/integ-1006`, with only the checkout path normalized and a fixed boot name. INT8/off defaults also remain unchanged. Unknown/empty/duplicate selectors fail.

Examples: `attn,shared`, `attn,shared,dense`, `attn,shared,mtp`, `attn,shared,mtp,indexer`. Canonical selector order is attn/shared/indexer/dense/mtp. Extra groups require a new absolute `GLM_NVFP4_MORE_DIR`, mounted read-only at `/more-nvfp4`. Original checkpoint `/model` and serving attention `/attn-nvfp4` remain separate. All four ranks receive identical switches. The real mode rejects `GLM_NVFP4_WSIM`.

| Group | Original logical matrices | Original format | Runtime handling |
|---|---:|---|---|
| attn | 385 | Group128 INT8, layers1–77 | Existing serving sidecar and scheme |
| shared | 225 | Group128 INT8, layers3–77 | CT linear scheme, two independently scaled gate/up Marlin calls, concat; down row-parallel |
| dense | 6 | Group128 INT8, layers1–2 | Same gate/up and down path |
| indexer | 63 | BF16: wq_b/wk/weights_proj, including layer0 | Replicated W4A16; independent WK/weights globals in fused runtime projection |
| mtp | 776 | Channel INT8, layer78 | Draft only: 5 attention, 3 shared, 256×3 routed; routers, BF16 extras, norms, head and indexer remain original |

The inventory matches the complete local checkpoint index and simulator classifier. Runtime MTP prefixes with or without `mtp_block` are accepted. MTP native target-skip/draft-only context is required. Original iterator names are counted by `glm_mtp_select` before replacement, preserving its source-header completeness audit. The extra iterator then independently requires every selected original companion and every replacement exactly once. Target extras never transform the draft; MTP never transforms the target.

`convert_more_nvfp4.py` uses the existing block16 encoder, FP32 full-tensor absmax before TP slicing, E4M3 and E2M1 RNE, original multiplier convention and FP32 `fp4 * (scale8 * global)` dequant association. Shared/dense decode group128 INT8; MTP decodes channel INT8; indexer reads BF16 directly. There is no INT8 re-encode. The default attn-only converter delegates to the unchanged serving converter, including its byte serialization. Extra manifests are schema2, contain exact group inventories, source index/config/shard hashes and output hashes. A lock, atomic output/manifest replacement, source/output rehash on resume and one metadata key preserve resumability and cross-process byte reproduction. Source and destination must be disjoint. Group set or source drift refuses resume; use a fresh directory to change a converted inventory. A superset extra sidecar can serve narrower group arms.

All admitted sidecars retain the Marlin positive-scale >=1/64 check. Fused globals are never maximized, inverted, or recalibrated after TP slicing. Added source pins bind the exact MTP/MoE implementation used by the extension.

Two pinned-source hazards required handling:

* Fused indexer WK/weights_proj is constructed with `quant_config=None`. A prefix-specific constructor hook supplies the same CT W4A16 linear method before parameter allocation. The stock FP8 WK loader would consume any `weight_scale`, including NVFP4 scales; a selected-prefix hook bypasses that helper only for the real packed/block/global companions. The 32-output weights projection is padded to N64 by existing Marlin code.
* Stock NVFP4 MoE discards up's distinct global and uses gate's global. Native MTP instead preserves gate/up/down globals separately and dispatches three existing `moe_wna16_marlin_gemm` calls, with BF16 activations, FP32 reduction, stock SiLU-and-mul and the existing top-k aligner. TP4/EP1, unclamped SiLU are enforced. No custom CUDA/Triton kernel, activation quantizer or persistent BF16 weight bank is introduced. The runner retains shared-expert ownership. Load-time gate/up preparation currently repacks down twice; only one down bank remains resident. That extra transient and startup cost must be recorded.

The simulator groups remain on `exp/nvfp4-wsim`: `GLM_NVFP4_WSIM=shared,indexer,dense,mtp`. They are useful historical evidence. They are not enabled or copied as serving kernels here. Its additional INT8 grid was about0.65% NRMSE from attention's real FP4 grid; channel MTP samples have a larger residual. **All qualification below is on real packed weights and real Marlin execution.**

## Mac evidence and indexer decision

[mac-real-samples.json](mac-real-samples.json): **23 real matrices, 551,878,656 elements, zero bit mismatches** against the independently pinned simulator's pre-INT8 QDQ reference. Shared3/indexer3/dense3/MTP14, including routed experts0/255 and MTP attention/shared. Original source hashes are unchanged. Partial sidecar `/private/tmp/glm53-nvfp4more-real-samples` cannot pass the full inventory loader and cannot serve. This is not an all-matrix non-attention audit or full-model quality result.

[indexer-sensitivity.json](indexer-sensitivity.json): isolated scoring with real layer1 BF16 weights, seeded synthetic activations, 4096 keys, 8 queries and topk2048 retained **93.927%** of selections on average. About6.1% changed. Weight NRMSE is about9.3%. The screen omits trained activation distributions, RoPE and the actual FP8 index-cache scoring path; it cannot predict production overlap. For a score perturbation bounded by delta, a sufficient top-k stability condition is `score[k]-score[k+1] > 2*delta`; the measured synthetic margins do not satisfy that condition. RMS weight error does not bound ranking changes.

**Indexer is opt-in only and is a high-risk W4A16 candidate.** Its BF16 weights choose retrieval indices, with discontinuous effects on selected KV entries. At <=2048 context all keys can be selected, so short prompts are insufficient. Before considering indexer, collect actual selected IDs and score-margin/overlap receipts from identical long-context activations (16384,63488,98048), then run the REAL-path model gates. If that capture is unavailable, retain BF16 indexer and leave its arm unqualified. The synthetic result is sufficient to reject automatic inclusion, not to prove failure of the model.

**MTP is measure-before-ship.** Quantization can change draft proposals, K-stop pass counts, accepted tokens and committed/cycle even when target outputs remain valid. Its memory saving is mostly resident inactive experts; its read/cycle numerator is only0.360GB under the K2 prior. No acceptance benefit is assumed.

## Cost and conversion

[cost-model.json](cost-model.json) and `scripts/nvfp4_more_cost.py` use TP4 shapes from `nvfp4w/PROBE.md`, 255GB/s, and **divide ideal savings by2**. The illustrative B-current cycle prior is75ms; replace it with the same-day per-kind `cycle.py` median using `--cycle-ms`. Historical trace forecasts and bandwidth forecasts are separate estimates, never summed. The split-call allowance is borrowed from attention's0.5ms/77 extra-call budget and is not measured for these groups.

| Added group | Discounted trace saving, before extra calls | After split allowance | Cycle-ms gain at75ms, after allowance → before | Packed bytes freed GB/rank |
|---|---:|---:|---:|---:|
| shared | 0.883ms | 0.396ms | 0.53% → 1.18% | 0.3207 |
| dense | 0.119ms | 0.106ms | 0.14% → 0.16% | 0.0513 |
| mtp | 0.130–0.410ms | 0.117–0.397ms | 0.16–0.53% → 0.17–0.55% | 1.0882 |
| indexer | 0.650ms | 0.514ms | 0.68% → 0.87% | 0.2806 |

Attention is already in B-current, so its incremental gain here is zero. Freed tensor bytes include scales/globals, indexer N32→64 padding and workspace allowances; they are not discounted physical tensor bytes. A separate half-credit memory column is supplied for conservative planning. Real allocator/graph/transient receipts are still required. KV pools, dispram carveout and served **98176 cap remain unchanged** in every arm. No new capacity admission is proposed.

Even shared is marginal; dense/MTP/indexer priors fail the >1% incremental gate. Extra GEMMs and concatenations can erase more saving than modeled. A bundled shared+dense pass cannot credit dense independently. Measure each optional addition or leave it disabled. Do not lower the gates to match these forecasts.

Extra conversion: **13,347,389,440 elements**, about**7.508GB** payload plus headers/manifests, **81 referenced shards** (shared75, dense2, indexer21, MTP4; overlapping). At10–40M elements/s plus hashing/storage, estimate **10–35min per Spark, reserve45min**. Approximate components: shared3–10min, dense0.5–2min, indexer1–4min, MTP5–20min; they overlap in source hashing and are not independent measured timers. Conversion on four nodes contends for shared memory/storage; run under the coordinator's chosen resource budget and record actual elapsed time. Reserve at least10GB extra disk per node plus temp output. All four `SHA256SUMS` file hashes must match. Metadata/header byte identity is checked, not only canonical tensor equality.

## Offline commands and prepared packages

```bash
cd /srv/projects/glm53-full-nvfp4more
PY=/srv/operator/.cache/uv/archive-v0/_WsxjzKnb0gWDXwh/bin/python
export GLM_IMAGE_SRC=/srv/campaign/diagnostics/glm53-full-20260929/day3/release-dirtyl2/image-source
"$PY" -m unittest discover -s tests -p 'test_nvfp4*.py'
"$PY" -m unittest discover -s tests -p test_recipe.py
"$PY" tests/test_glm_fast_load.py
"$PY" -m unittest discover -s tests -p 'test_coalesced*.py'
python3 scripts/nvfp4_more_cost.py --cycle-ms 75
# Fresh external directory; prepare renders DRY only and never calls SSH.
python3 scripts/nvfp4_more_prepare.py \
  --reference-clone /srv/campaign/diagnostics/glm53-full-20261006-integ/prepared-mac-final/restore0/clone \
  --out /path/to/fresh/prepared-window --tag WINDOWTAG
```

Prepared Mac artifacts are at `/srv/campaign/diagnostics/glm53-full-20261006-nvfp4more/prepared-mac/`: B,S,SD,M,I, each with a frozen clone, launcher, source guard, exact98,176 cap/ordinary1GiB/dispram geometry, `cycle.py`, `gate_metrics.py`, four-rank DRY commands and hashes. SD is optional and separately measured. Default M and I retain shared only; `--with-dense` inserts qualified dense into M/I. Preparation is no-clobber and never advances an arm. Refresh packages after changing code; hashes bind their actual contents. A freshly prepared B package is the restore candidate, but the coordinator must record the actual current watch/guard/container identity before using it.

## Later coordinator window — written only, not executed

Use the existing exclusive serving coordinator's lock/expiring hold, supervised units, external dispram guard and normal stop/postcheck/restore handoff. Preserve the current B deployment JSON, per-rank image IDs, watch drop-in, guard unit, `.env`, tokenizer/checkpoint/attention checksums, endpoints, graph/K-stop/prefill settings and cap. The integration's **KV stress** `integ_1006.py run` is not the weight ladder driver and must not be invoked for these arms. Do not run `start.sh stop` against a guessed clone or transfer an unknown fleet lock.

1. Stage the committed checkout and prepared packages through the established coordinator staging mechanism. Conversion is CPU-only and can precede the boot window subject to resource headroom. On **each node**, inside its already installed frozen serving image (no pulls), original checkpoint mounted read-only, new extra directory mounted writable:

   ```bash
   # NODE-LOCAL commands later; IMAGE is the frozen serving image ID for that rank.
   CODE=/srv/glm/glm-control/runs/nvfp4more-WINDOWTAG/code
   SOURCE=/srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix
   MORE=/srv/glm/models/GLM-5.3-more-nvfp4-20261006
   mkdir -p "$MORE"
   time docker run --rm --pull never --network none --cpus 1 \
     -e OMP_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES='' \
     -v "$CODE:/code:ro" -v "$SOURCE:/model:ro" -v "$MORE:/more:rw" \
     --entrypoint python3 "$IMAGE" /code/scripts/convert_more_nvfp4.py \
     /model /more --groups shared,indexer,dense,mtp
   (cd "$MORE" && sha256sum -c SHA256SUMS)
   sha256sum "$MORE/SHA256SUMS"
   # Repeating the same conversion resumes after source/output rehash; never change source/directory/group set.
   ```

   Preserve every converter log, elapsed time, source hash manifest and externally recorded checksum hash. Verify exact shared225/indexer63/dense6/mtp776, complete=true, safe scales, and matching file hashes on all ranks before any boot. The existing attention directory is retained unchanged. Run the full extra reference audit inside the same CPU-only image with code/source/sidecar read-only and a fresh writable receipt directory:

   ```bash
   python3 /code/scripts/nvfp4_more_audit.py /model /more --out /receipts/more-cpu-audit.json
   ```

2. At the coordinated idle GPU slot, run existing compiled Marlin checks on sm121 before model admission, inside that same image. No custom JIT compile is needed. For all ranks and each group, use the read-only full sidecar:

   ```bash
   for rank in 0 1 2 3; do
     for group in shared dense mtp indexer; do
       python3 /code/bench/more_nvfp4_gemm.py --sidecar /more --group "$group" \
         --rank "$rank" --out "/receipts/linear-$group-r$rank.json"
     done
     python3 /code/bench/mtp_nvfp4_gemm.py --sidecar /more --rank "$rank" \
       --out "/receipts/mtp-routed-r$rank.json"
   done
   ```

   Linear checker covers every selected matrix sequentially, M1–16 and prefill through4096, exact FP32 dequant oracle with TF32 off, finite outputs, NRMSE<=0.02 and graph replay. Routed checker covers real experts0/255, independent globals, topk2 and M1/3/16/512. It is a focused numerical screen; production topk8, all256 experts, routing, graph-tail behavior, fused indexer runtime and native draft acceptance are exercised by the full-model gates. No GPU checker has run on the Mac. Warm kernel timings do not establish full-model DRAM savings. Record weight/reference/repack/workspace peak shared memory and boot transient; MTP double-down repack is included.

3. Run sequential, supervised boot slots **B-current → S shared → optional SD dense → M MTP → I indexer**. There is a coordinated stop/postcheck between them. Copy the prepared packages under rank0 HOME. For each slot, after the coordinator has established the idle handoff, execute the concrete package launcher under a transient user unit:

   ```bash
   export WINDOW=/srv/glm/glm-control/runs/nvfp4more-WINDOWTAG/prepared
   export ARM=B                       # later S, optional SD, M, I; no automatic loop
   mkdir -p "$WINDOW/$ARM/run"
   systemd-run --user --unit="glm-nvfp4more-WINDOWTAG-$ARM" \
     --setenv=PATH=/srv/glm/glm-control/bin:/usr/local/bin:/usr/bin:/bin \
     --property=StandardOutput="append:$WINDOW/$ARM/run/serve.log" \
     --property=StandardError="append:$WINDOW/$ARM/run/serve.log" \
     /bin/bash "$WINDOW/$ARM/launcher.sh" serve
   # Attach the frozen current source memwatch.py under the coordinator's existing hold,
   # with this exact unit, serve.log and launcher.sh, at the existing8.05GiB all-rank floor.
   # After probes / on failure: coordinator detaches the monitor and terminates that unit,
   # then invokes the same package's stop/postcheck, preserving its containers and logs.
   "$WINDOW/$ARM/launcher.sh" stop
   ```

   These are operator-controlled slots, not an unattended sequential script. Keep hold/watchdog ownership and samplers active throughout; log every launch/stop and verify health, capture/replay, memory floor and all-rank replacement counts before probing. Counts: B target attn385/draft original; S target +shared225; SD +dense6; M draft +mtp776; I target +indexer63. All arms retain attention385. A missing companion or partial count invalidates the arm. Freeze B-current results in this window; historical attention or simulator receipts cannot be its performance control.

4. **Exclusive c1 probes on every admitted arm**. Prepare `BASE`, `DASH`, `RIGMARK` from the existing approved endpoint/client setup; do not create new tunnels/listeners. Retain raw requests, seeds, texts, request IDs, metadata and errors. Run serially:

   ```bash
   export REPO=/srv/glm/glm-control/runs/nvfp4more-WINDOWTAG/code
   export R="$WINDOW/$ARM/run"
   export BASE=http://127.0.0.1:18095
   export DASH=http://127.0.0.1:5555/api/sparks/rank0/llm
   python3 "$REPO/bench/cycle.py" --endpoint "$BASE" --out "$R/cycle" --boot "$ARM"
   python3 "$REPO/bench/nvfp4_more_gate.py" dash --base "$DASH" --out "$R/sparkdash-c1.json"
   mkdir -p "$R/qeval"
   (cd "$R/qeval" && for i in 1 2 3; do
     python3 "$REPO/bench/qeval.py" run "nvfp4more-$ARM-$i" \
       --url "$BASE/v1/chat/completions" --concurrency 1
   done)
   python3 "$REPO/bench/nvfp4_more_gate.py" qeval "$R/qeval" --out "$R/qeval-gate.json"
   python3 "$REPO/bench/nvfp4_more_gate.py" needles --base "$BASE" \
     --max-input 98048 --out "$R/needles.json"
   # Same pinned RigMark source/hash and metadata collector for all arms.
   python3 "$RIGMARK" run --base-url "$BASE" --model GLM-5.3 \
     --label "nvfp4more-$ARM-warmup" --comparison-id nvfp4more-warmup \
     --metadata "$R/rigmark-metadata.json" \
     --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
     --runs 1 --decode-tokens 256 --skip-prefill --skip-concurrency --output "$R/rigmark-warmup.json"
   python3 "$RIGMARK" run --base-url "$BASE" --model GLM-5.3 \
     --label "nvfp4more-$ARM" --comparison-id nvfp4more-WINDOWTAG \
     --metadata "$R/rigmark-metadata.json" \
     --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
     --runs 3 --decode-tokens 4096 --skip-prefill --skip-concurrency --output "$R/rigmark.json"
   ```

   `cycle.py` is the saved current serving cycle panel (ten English prose and ten code prompts, T0/thinking-off/max512, discarded warm-up, per-request engine counters). Only its GateMetrics import is made checkout-local. GateMetrics source is preserved. It reports both committed/cycle and cycle-ms, paired by exact prompt/seed/layout metadata, with failed/foreign/reset metrics retained and invalid. `sparkDash` helper discards one warm-up and retains five prose + five code jobs. RigMark also retains its structured decode result. Profiled GPU cycles/kernel breakdown are recorded separately from uninstrumented cycle requests; use them to diagnose sub-ms groups, not to replace acceptance.

   Needles: 20 fixed seeds × **16384,63488,98048 input tokens**, five depth strata,64 output tokens.98048 is the maximum whole64-token prompt grid used here under cap98176 with output/K3 reserve; it is not relabeled96K. No arm changes cap or needle length. Require every needle pass, exact token/prompt hashes matching B, valid finish and no transport error. Retain partial failures without retrying to erase them. For indexer, collect real selected-ID/margin overlap at these lengths; short-context results cannot qualify it.

5. Compare each addition to the **previous accepted arm**, and also report against B-current. Dense requires its own S→SD comparison. Default M compares S→M; I compares M→I. If dense qualifies and is retained, prepare with `--with-dense` so M/I both keep it. Use the matching predecessor's fresh results:

   ```bash
   export PREV=S                  # S uses B; SD uses S; M uses S or qualified SD; I uses M
   python3 "$REPO/bench/nvfp4_more_gate.py" cycle \
     "$WINDOW/$PREV/run/cycle/requests.jsonl" "$R/cycle/requests.jsonl" --out "$R/cycle-incremental-gate.json"
   python3 "$REPO/bench/nvfp4_more_gate.py" cycle \
     "$WINDOW/B/run/cycle/requests.jsonl" "$R/cycle/requests.jsonl" --out "$R/cycle-vs-B-gate.json"
   python3 "$REPO/bench/nvfp4_more_gate.py" ngate \
     "$WINDOW/B/run/needles.json" "$R/needles.json" --max-input 98048 --out "$R/needle-gate.json"
   ```

   Adoption requires **cycle-ms gain >1% per addition and committed/cycle drop <1%**, independently on English prose and code. Strict equality at1% fails. Cycle ratios use median matched per-request ratios; committed ratios use the ratio of sums of matched per-request committed/cycle. Also retain raw deltas/acceptance/K-stop/num-drafts, means, uncertainty and GPU cycle evidence. **Polish drift/acceptance is ignored for this task**; no Polish veto is added.

   qeval mean must be **>=70.553 passed tasks out of75 across exactly three complete c1 runs**, matching the existing serving receipt's units (not70.553 percent). Report primary55 tasks and per-task recurring losses separately. Missing/incomplete/transport-failed runs cannot pass. Needles, RigMark/sparkDash functional results, health/memory/capture and source/count receipts must all be valid. A failed B control qualifies nothing. On an addition's failure, restore the previous qualified configuration and leave that group disabled; do not advance by treating a combined gain as an independent group pass. Given priors, dense/MTP/indexer are expected to miss the speed gate.

6. End through the coordinator's saved normal serving restoration procedure on pass or failure. Restore **B-current attn only** unless a separately approved promotion decision records the passed extra groups; this preparation does not authorize promotion. Use the recorded watch drop-in/guard/heartbeat/lock handoff, verify endpoints and original admission, preserve stopped experiment containers and all receipts, then release only the owned hold/lock. No fleet action occurred in this preparation.

## Verification receipt

See [verification.json](verification.json) for Mac commands/counts, independent reference hash, source-pin checks, DRY/default identity, real-sample audit and static validation. CPU tests cover group inventory/switches, original BF16/channel INT8 handling, full-tensor globals, bit-exact QDQ, cross-shard reads, row chunk invariance, resume/source/output drift and multi-process byte identity, selected target/draft stream counts, indexer constructor/FP8-helper behavior, TP slicing, fused independent globals, routed dispatch and strict cycle/qeval gates. GPU execution, actual selected-index overlap, model quality, draft acceptance and cycle gains remain unmeasured.
