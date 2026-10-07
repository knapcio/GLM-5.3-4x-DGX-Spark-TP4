# One candidate boot: coalesced loader

Written offline on the Mac. No step below has been executed on the fleet.
Candidate: `perf/coalesced-loader`, loaderfix descendant of `44cf298`, base `84a38e0`.
The 2026-10-06 13:59 attempt failed at the obsolete 10 GiB absolute loader
floor on all ranks; this plan is for one separately authorized fixed-loader
boot. The earlier load has no final hashes and supplies no speed claim. Reference: the current
qualified `84a38e0` FP4x/adaptive-chunk launch, max length 98176, with its
existing dispram layout and caches. Keep NVFP4 attention sidecars off for this
first transport comparison. Do not change clocks, sysctls, image, checkpoint,
KV pool, prefill banks, NCCL settings, or the 60 s admission window.

## Before the window

1. Pin the candidate commit, image IDs, NCCL hashes and the qualified serving
   configuration. Run the Mac checks from the offline receipt. Stage an isolated
   candidate checkout/runtime and the unchanged dispram copy guard through the
   normal coordinator process. Preserve the current containers and restore
   configuration. Builds and checkpoint verification are outside the timed boot.
2. Select a **compatible normal-fast-loader final-weight baseline manifest for every rank**.
   Record `GLM_LOADER=fast` (or the equivalent existing fast default), and
   match checkpoint, TP/EP, quantization/sidecars and runtime source exactly.
   A baseline from the failed coalesced boot is unusable.
   Record its checkpoint, runtime/quantizer, module/parameter names, dtype,
   shape and alias contract. The saved prior fast-load hash-test directories are
   `diagnostics/glm53-full-20260929/day3/boot-time/hash-test/`:
   `run-20261002-180833/arm-A/manifests` and
   `run-20261002-183626/arm-B/manifests`. Their all-rank EQUAL receipt is in
   `WINDOW.md`; these manifests have 8,269 parameter entries and 518 buffers
   per rank. They are not a 3,142-entry current-launch baseline, and their older
   capture/scratch configuration must not be assumed compatible.
   If the running control already has hash RPCs attached, capture its manifests
   without reboot. Otherwise use a previously qualified matching baseline.
   If neither is available, record the candidate hashes but label the final
   comparison INCOMPLETE; obtaining a new control baseline is separate work.
   Never send unknown hash RPCs to an uninstrumented engine.
3. Candidate configuration differs only by `GLM_LOADER=coalesced` plus read-only
   hash instrumentation `GLM_PARAM_HASH=1`, `VLLM_SERVER_DEV_MODE=1`. Leave
   `GLM_SKIP_MLA_PLAN=0`. Keep `GLM_MTP_ONLY_LOAD=1`,
   `GLM_TARGET_SKIP_MTP=1`, `GLM_DRAFT_LOWMEM=0`; use lazy safetensors.
   Defaults: batch128 MiB, threads16, direct1, owned4096 MiB,
   emergency3072 MiB, margin1024 MiB. If overridden, render
   `GLM_COALESCED_EMERGENCY_MB` and `GLM_COALESCED_MARGIN_MB` identically
   on all four ranks; record their exact values. Those options are
   launcher-validated and identical on all ranks. On a staged checkout:

   ```bash
   DRY=1 GLM_LOADER=coalesced GLM_PARAM_HASH=1 VLLM_SERVER_DEV_MODE=1 ./start.sh serve > dry.txt
   ```

   Compare four rendered rank vectors to the qualified vector, allowing only
   these three env keys and candidate runtime/container names. Check the
   temporary CUDA/pinned release code is present. Do not give KV extra capacity.

## Window sequence

1. The coordinator acquires the existing owner-labelled fleet lock/serving hold,
   records container IDs/argv, and pauses the watch using the established
   control-plane procedure. Stop and preserve the current containers; confirm
   idle GPUs and existing preboot 110 GiB headroom and dispram postcheck.
   Start a 100 ms host sampler locally on **each** rank, before loading:

   ```bash
   python3 scripts/loader_memwatch.py --rank RANK --seconds 900 --out receipts/memory-rRANK.jsonl
   ```

   This process only reads `/proc/meminfo`; it sets no memory/cache policy.
   Retain the existing guard/watchdog samples as well. Require an initial
   sampler record before starting transport tests or the boot.
2. In the pinned CUDA image with no serving workers, run
   `python tests/gpu/coalesced_transport.py`. It requires direct-I/O, 132 exact
   tensor cases, a non-default consumer stream and at least one deliberately
   pending upload-event handoff. Any failure ends this candidate window and
   restores the recorded serving containers. This gate does not boot a model.
3. Start **one** candidate boot using the qualified coordinator launcher pattern
   with a new container prefix and the pinned candidate checkout. Record
   launch monotonic/epoch timestamps, all four full rank logs, launcher events,
   one-second health polling, and the host samples. No second tuning boot or
   sidecar-combined boot in this experiment. The loader bounds its own staging/owning storage relative to the start
   baseline, credits native destination writes after TP/EP slicing, and
   retains an independent 3 GiB emergency floor. All existing launch, capture
   and admission guards stay enabled. Before placement, a refusal on any rank
   makes all ranks fall back to normal fast loading; after placement, failure
   aborts the whole load through the bounded failure channel. Verify channel
   initialization and end-of-load rendezvous on all ranks. A fallback boot is
   recorded as FAST-FALLBACK and cannot qualify coalesced performance; do not
   run a second candidate boot in this window.
4. At health 200, record time immediately, before hash work. Complete the
   unchanged 60 s admission window, exact text/tool smoke and APC check. Require
   all-rank `glm-coalesced-load` target/draft receipts with `complete=true`,
   no `falling back to fast` messages,
   `cuda=true`, `direct=true`; native MTP checks must pass on all ranks. Require
   the same KV pool/block/layout readback as the control.
5. Drain requests. Verify `GLM_PARAM_HASH=1`, dev mode and the
   `glm-phash: worker methods attached` receipt on each rank before calling
   `glm_phash_status`. Use the existing guarded `phash_client.py hash` client
   (in the saved hash-test directory) to run `glm_phash_run` and collect
   `rank0.json` through `rank3.json`. It hashes logical parameter/buffer bytes,
   includes dtype/shape/name/aliases, and hashes in bounded 32 MiB chunks.
   Compare with the compatible baseline using that directory's `compare.py`,
   pointing `--module` to this candidate's `glm_param_hash.py`. Require EQUAL
   for **all four ranks**, exact inventory/counts and aliases, no missing
   entries. Scratch attr differences remain listed separately. Report actual
   tensor/parameter counts; never substitute the synthetic 3,142 cases for
   a full-checkpoint comparison. Hash time is outside boot-to-health timing.
6. Run the existing short c1 cycle probe and compare with the control receipt;
   no sustained regression is expected because kernels/weights are unchanged.
   Stop/preserve the candidate, collect logs/manifests/samples, restore the
   recorded qualified containers and serving watch/heartbeat, and verify
   health/smoke/dispram postcheck and lock handoff. Promotion is a later decision.

## Required result ledger and decision

Per-rank loader JSON records start/end wall time, source count, selected tensor
count, total seconds, read bytes/calls, uploads, reader wait, upload enqueue,
tile/event wait, consumer time, placement synchronization time, staging bytes,
peak owning bytes, start baseline, destination-write credit, relative budget,
loader transient live/accounted bytes, emergency floor/margin, and sampled
available-memory low/peak. Preserve failures and fallback reasons per rank. Producer/consumer
counters overlap and must not be added as exclusive boot phases.

Build the same exclusive ledger as round6-batch item 3: slowest-rank main
weights; MTP; other model construction/repacking; graph capture; remaining
profile/KV/warm-up; launch/distributed/API residual; health; then separate
admission duration. Use `Model loading took` and graph/profile timestamps
alongside the new loader receipts. Across ranks use the critical path, never
sum four parallel durations. Report intervals that logs cannot separate as
grouped/unknown rather than estimating individual phases.

Required four-row comparison table: rank, normal-fast manifest SHA-256,
candidate manifest SHA-256, actual tensor/parameter/buffer counts, alias
verdict and EQUAL/DIFFERENT/INCOMPLETE. No successful-boot claim without
all four final hashes equal to the compatible normal loader.

For every rank, restrict 100 ms memory samples to each loader's start/end
epoch. Report `MemAvailable` minimum **and maximum**, maximum
`MemTotal - MemAvailable`, dirty/writeback peaks, and swap growth. Retain
preboot, pre-capture and post-admission minima. Include an explicit four-row
table of target-load and draft-load `MemAvailable` minima (GiB), timestamps,
start baselines, destination credit and peak transient bytes. Below 10 GiB
during loading is no longer a loader failure by itself; the emergency floor
and unchanged later phase guards must still pass. Check transient CUDA caches
are released before KV sizing; no decrease in KV admission headroom is credited
as a speed win. Sampling establishes observed extrema, not a continuous bound
against allocations made by unrelated processes.

Forecast: main60 s, health153.68 s, admission213.68 s if other phases stay
fixed; range main45–90 s. Promote for a follow-up only with all-rank final
weight EQUAL, no safety/memory/layout failure, main <=90 s and health <=185 s
(against saved mean282.44 s). A hash/contract mismatch rejects the loader;
an absent compatible baseline is INCOMPLETE. Keep the measured GPU transport,
full-model hashes and fleet performance separate from the Mac receipt.
