# KV headroom: offline design and implementation, 2026-10-06

Owner-approved scope: Mac only; no SSH, fleet operations, deployment, pushes,
system tuning, or pinned-memory experiments. Branch `perf/kv-headroom` starts
at `perf/adaptive-chunk` `84a38e0`. This is an experimental plan and runnable
future window harness, not a qualification or a claim of unchanged tok/s.

## Decision

Keep the sim lower-bound gate at **8.5 GiB on every rank**, serving floor at
8.0 GiB on rank 0, and emergency stop at 6.0 GiB on any rank. Give cache flushing
**zero extra MemAvailable credit**. Test the flusher at the existing 1 GiB head
before considering a bigger pin. +1 GiB is a candidate for recalibration, not
admitted by the current conservative simulator. +2 and +3 GiB miss even a simple
observed 8 GiB serving-floor projection; they must not automatically advance.

The notes path supplied with the task did not exist. This plan creates it. The
referenced `day3/RULES.md` also did not exist; the campaign's parent `RULES.md`
contains the 8.5/8.0/7.5/6.0 rules. The new task's no-push restriction governs
this work even though that historical rules file contains a later push rule.

## 1. Receipt-backed memory ledger

Sources, relative to `day3/release-stack/stack-1002b/`:

- `fp4x-cal-20261005-2215/e2/{rank0..3-boot.log,receipts.json,memwatch.log,memwatch-minima.json}`
  is the 84a38e0 serving boot. Full `/proc/meminfo` and Torch counters are embedded
  in `glm-window-memory` lines, before graph capture.
- `fp4x-cal-20261005-2215/{g,e,r2}` supply loaded and stressed comparisons.
- `fp4x-cal-20261005-2215/q/hold-post-warm.json` is a 60 s hold; not an e2 hold.
- `v2-serving-20261005/hold-post-warm.json` and `memwatch-serve.log` are the older
  FP8 serving control. No boot-mem.json was present inside the fp4x-cal arms.
  `w4-20261003-0654/final/boot-mem.json` supplies the historical FP8 boot comparison.

`headroom_receipts.py` makes the adjacent source-hashed `fp4x-receipts.json` and
`v2-receipts.json`; no network or host actions. These are frozen observations,
not a live recheck. All units below are GiB, except where stated.

| e2 before capture | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---:|---:|---:|---:|
| Linux MemTotal | 121.690 | 121.690 | 121.690 | 121.690 |
| MemAvailable | 10.839 | 11.850 | 12.385 | 11.738 |
| MemFree | 2.717 | 4.422 | 2.077 | 1.965 |
| Cached | 10.220 | 8.959 | 11.856 | 11.727 |
| Shmem (included in Cached) | 0.962 | 0.957 | 0.954 | 0.955 |
| Cached minus Shmem: disk-cache upper estimate | 9.257 | 8.002 | 10.901 | 10.773 |
| Clean unmapped candidates, conservative estimate | 7.532 | 6.022 | 8.932 | 8.755 |
| AnonPages | 4.338 | 3.678 | 3.343 | 3.354 |
| SReclaimable | 0.400 | 0.525 | 0.455 | 0.512 |
| SUnreclaim | 2.091 | 2.120 | 2.089 | 2.092 |
| Mlocked / Unevictable | 0.022 | 0.023 | 0.022 | 0.025 |
| Torch allocated / reserved | 97.736 / 98.180 | same | same | same |

Clean unmapped candidates = max(0, Cached - Shmem - Mapped - Dirty - Writeback).
It deliberately subtracts overlapping mapped/shared fields conservatively;
it is not a promise of bytes that drop_caches will release. Mapped weight pages,
dirty pages, pinned driver pages and pages with active references can remain.
Slab reclaim requires a different policy; this proposal never writes 2 or 3.

The model loader reports **96.83 GiB** on every rank (target plus native MTP).
Torch pre-capture reserved includes those weights and workspaces; it must not
be added to the model-loading figure. Torch's reserved-minus-allocated is
0.444 GiB. After target capture it becomes allocated 97.928 / reserved 98.420
on all ranks. The separate dispram VMM pool is **1 GiB ordinary head + 2046 MiB
carveout**, 101 regions, 1573 blocks on every rank; this VMM allocation is outside
Torch's ordinary allocator counters. The carveout is a reserved display region,
not 2 GiB newly taken from MemAvailable when borrowed. Do not debit it twice.
The cache-released event freed 1.232 GiB of Torch reservations before this pool.

A coarse nonfree/noncache/nonslab/nonanon accounting residual is about
101.83–101.94 GiB per rank. It contains CUDA/driver and unclassified kernel
memory; it is NOT an independent CUDA measurement and is not reclaimable credit.
Ordinary KV, resident weights, graph pools, sampler/communication buffers and
CUDA pinned allocations cannot be freed by dropping file cache. Mlocked at
~23 MiB does **not** bound total NVIDIA pinned/driver memory on unified memory.

| Observed phase minima, rank order 0/1/2/3 | GiB |
|---|---|
| FP8 v2 60 s post-warm hold | 9.159 / 10.469 / 10.569 / 10.486 |
| FP4x q 60 s post-warm hold | 9.891 / 10.366 / 10.727 / 10.744 |
| FP4x g cold context ladder | 9.132 / 10.031 / 10.599 / 9.948 |
| Adaptive e cold prefill | 9.193 / 9.789 / 10.356 / 10.130 |
| Adaptive e cycle probe | 9.165 / 9.763 / 10.657 / 10.136 |
| Adaptive e2 APC check | 9.569 / 10.489 / 10.038 / 9.883 |

These minima can occur at different seconds; subtracting them is not a
simultaneous rank-difference measurement. e2's APC check has only 13 samples;
its lower rank-0 boot minimum was 9.337 GiB. g/e supply the longer stress evidence.
The memwatch stream records availability and swap, not under-load cache or PSS.

**Why the head is lower:** equal weight load, equal KV block count and equal
Torch allocation exclude a larger ordinary KV pin or larger recorded graph
bank on rank 0. Rank 0's host AnonPages exceed workers by 0.66–0.995 GiB. Its
boot log uniquely includes the HTTP APIServer; engine orchestration,
tokenizer/detokenizer and API-side Python allocations are the likely source.
Kernel and cache state explain additional variation. The rejection sampler warms
on **every rank**, so attributing the entire GiB to a head-only sampler would be
unsupported. Exact per-process API/sampler attribution still needs PSS and
worker allocation snapshots under load. The driver records process PSS where
permissions allow (10 s, off the 1 s counter path), plus full meminfo. Failure to
read PSS is recorded as unknown; it must not be replaced with summed RSS.

The older DeepSeek memory notes establish head-rank pressure and hang mechanisms,
not the exact allocator implementation of this vLLM stack. Here KV is explicitly
pinned and the carveout is negotiated by rank-invariant MIN collectives.
The limiting host rank must still constrain the **common** pin. Never give
workers a bigger block count independently or add a pinned side allocation.

## 2. Conditional page-cache experiment

Use a **local root systemd timer on each node**, outside the user manager.
This matches Mac-independent serving and needs neither a Mac loop nor password
sudo each minute. The lender has no reliable knowledge of checkpoint activity;
keep its lease/INVALID/DRM monitor untouched. The borrower launcher owns the
checkpoint/serving-state hooks; the root helper alone can write drop_caches.

Files: `service/page-cache/` and `scripts/page_cache_policy.py`. They are drafts,
not installed by this work. `RECIPE_PAGE_CACHE_POLICY=0` by default. Installation
is an operator-window change after coordinator review of the concrete files.

Every 60 s, flush only if ALL of these hold:

1. `/run/glm-kv-headroom/enabled` exists (off switch; absent after reboot).
2. A serving lease (`ready`) was refreshed less than 90 s ago and no `loading`
   marker exists. Launcher begins this gate before checkpoint verification and
   weight reads; it marks ready only after full warm admission, then refreshes
   it from its foreground monitor. A controller failure expires the lease.
3. Conservative clean unmapped file-cache estimate is **>4 GiB**, MemFree is
   <=4 GiB, MemAvailable >=8 GiB, and Dirty+Writeback <=64 MiB.
4. Hold the same local flock used by `begin/ready/end`, run `sync` with a 15 s
   userspace timeout, recheck all gates, then write **1** to drop_caches.

Checkpoint loaders and snapshot readers must participate in begin/end and must
not run concurrently with the serving monitor's ready refresh. A new independent
loader needs a separate exclusive loading window (stop the ready refresher
first); the file marker is not an automatic process detector. An unrelated
checkpoint reader that ignores the protocol invalidates the safety guarantee.
Never global-flush during boot; preserve `GLM_FAST_LOAD_DROP_CACHE=1`, which
already uses range POSIX_FADV_DONTNEED after consuming slabs. Do not change
swappiness, compact_memory, kernel knobs, or lender policy in this branch.

Root installation recipe, later on each node (not executed here): copy the
Python helper to `/usr/local/sbin/glm-page-cache-policy.py`, the two units to
`/etc/systemd/system/`, and the tmpfiles file to `/etc/tmpfiles.d/`; then run
`systemd-tmpfiles --create /etc/tmpfiles.d/glm-kv-headroom.conf` and
`systemctl daemon-reload`. Enable the timer, not the oneshot service. For the
experiment, create the enabled file only after checking the checkpoint protocol
and boot with RECIPE_PAGE_CACHE_POLICY=1. Group `knapcio` needs the run-state
write permissions in tmpfiles; root retains the proc write authority.

Off: remove `/run/glm-kv-headroom/enabled` and
`systemctl disable --now glm-page-cache.timer`; stop an active oneshot to prevent
a pending sync from proceeding. The helper rechecks the marker after sync.
Set the recipe knob to 0 on the next clone. A broken helper never prevents
container shutdown: stop still runs docker stop and dispram postcheck, retaining
the fleet lock if policy cleanup fails. State gates fail closed when missing.
Journal: `journalctl -u glm-page-cache.service` contains skip reason, full
before/after meminfo, duration and buddyinfo for a flush; systemd also records
failures/timeouts. Logs stay on the node and use existing journal rotation.

**Speed and memory expectation:** dropping cache can convert file pages to
MemFree and may improve immediate allocator conditions. It does not create an
extra 8–11 GiB of MemAvailable: that cache is already largely counted there.
It neither releases CUDA allocations nor guarantees high-order contiguous
pages. sync can stall behind I/O; cache eviction can cause later major faults
and weight/snapshot rereads. Fully resident decode often has no disk need, but
mmap-backed leftovers and host activity can refault, so unchanged speed is
unmeasured. A timer or timeout cannot prevent a kernel-level reclaim livelock.

Before raising KV, do same-boot A/A then OFF/ON/OFF at the existing head, on
fixed public prompts, all graph widths unchanged. Include probes immediately
before/after an actual flush and through at least five timer firings; record
cycle-ms, TTFT, pgmajfault, swap, MemFree/MemAvailable, buddy orders, I/O and
kernel messages. If the threshold never fires, report "flusher unexercised".
Require no >2% cycle regression and no new fault/I/O stalls before adopting it.
Retain it only with evidence of improved free-page conditions. A best-case
one-time cache conversion does not authorize a larger KV head.

Linux documents automatic cache reclaim and warns that repeated drop_caches
can add I/O and CPU work: [kernel VM documentation](https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches).
MemAvailable and buddyinfo have different meanings; see
[kernel proc documentation](https://docs.kernel.org/filesystems/proc.html).
The owner-supplied Tony 300K comparison is motivation, not a matched measurement
of this FP4x ABI, loader, process layout, graph banks, or concurrency capacity.

## 3. Common KV sizing and admission

Override in each fresh clone's `.env`, sourced after current.env:

```
export GLM_KV_FORMAT=fp4x
export RECIPE_DISPRAM=require
export RECIPE_KV_HEAD_BYTES=2147483648
export RECIPE_MAX_MODEL_LEN=131712
```

The new head knob requires FP4x + dispram require, is >=1 GiB and 2 MiB aligned.
The layout helper prices 79 MLA rows x368 B +22 index rows x132 B = **31,976 B
per token per rank** (31.976 decimal KB, 31.227 KiB), or 2,046,464 B per block.
It checks every region's 512 B alignment. Therefore one extra GiB adds ~33,580
tokens. Pin identical ordinary bytes, max length and graph/scheduler settings
on every rank. Existing MIN lease negotiation and vLLM block agreement remain.
The driver verifies emitted four-rank pins plus actual pool/lease log receipts.

The prefill expansion arena was fixed at 1573x64 rows. This branch extends its
constructor capacity with the physical KV pool, plus a power-of-two key bank
covering four logical max contexts. Extra blocks therefore cost both KV and
prefill workspace; the sim charges that bank, tables, decode logits and its
existing warm allowance. The 256 MiB logits allocation, kernels, quantization,
4096 constructor chunk, adaptive 2048/4096 policy and captured decode widths
are unchanged. Larger K buckets use smaller query tiles and need future CPU/
GPU shape qualification; preserving kernel source is not a speed guarantee.

| Extra head / rank | Ordinary head | Blocks | Total KV tokens | Expected max length, preserving 39-block serving slack | Geometric ceiling, 6-block slack |
|---|---:|---:|---:|---:|---:|
| baseline | 1 GiB | 1573 | 100672 | 98176 | 100288 |
| +1 GiB | 2 GiB | 2097 | 134208 | 131712 | 133824 |
| +2 GiB | 3 GiB | 2622 | 167808 | 165312 | 167424 |
| +3 GiB | 4 GiB | 3147 | 201408 | 198912 | 201024 |

These are token totals including output, not prompt plus an additional 1024.
For each long request, prompt=max length-1024, output=1024, with K3/null slack.

A simple projection from the observed ~9.13–9.57 GiB rank-0 stressed baseline
is ~8.1–8.5 after +1, ~7.1–7.5 after +2, ~6.0–6.5 after +3, before added
workspace and any longer-context transient. The added prefill arena costs
about 37/74/111 MiB for +1/+2/+3. No cache credit is justified. +1 is borderline
with the serving floor and sim gate; +2/+3 are not ready for a fleet step.

`steps.json` preserves the **legacy conservative** FP4 admission calculation:
rank-0 steady lower 8.705/7.664/6.622/5.581 GiB; after the old long-prefill
loss model 6.716/4.995/3.273/1.551 GiB. It refuses even the now-qualified 98K
baseline because its earlier ~50K FP4 loss predates the tiled fix. That is a
calibration mismatch, not evidence that the later baseline failed. Do not
silently remove the debit or lower the gate. Refit the fixed/adaptive path
against g/e receipts, including all warmed graphs and uncertainty, then obtain
new shape/stress receipts for each extra head. Keep unknown longer-prefix
losses conservative and write explicit per-lever costs. The driver has no
8.5 bypass; `admit` currently writes REFUSED receipts for the larger steps.

**Four max-length requests:** one pool is shared across four request slots.
The baseline log itself reports maximum concurrency **1.03x** at 98176.
Four distinct 98176 contexts require 6141 blocks (K3 + null), versus 1573.
The ordinary head would need roughly 9.707 GiB (2 MiB rounded), about **+8.707
GiB/rank** over baseline. This is already in the known dangerous allocation
range. +1/+2/+3 heads at their expanded max lengths require 8237/10337/12437
blocks for literal c4, also about four times the proposed pools. Queueing or
preempting four HTTP requests does not satisfy four resident max contexts.

The driver defaults to literal c4 and refuses it before boot when impossible.
An explicit `--c4-mode pool-quarter` screen instead runs one full-max c1 and
four distinct requests at the aggregate pool quarter, then labels its result
**SCREEN_ONLY**, never worst-case qualification. Per-request total lengths are
25088/33472/41856/50272 at steps 0/1/2/3 (including output). Another coherent
product policy is to advertise that smaller c4-guaranteed context; it needs a
new reviewed recipe and sim, not a hidden relaxation of the requested test.

## 4. Future step window and driver

`stress_step.py` has local-only `plan`, `prepare`, `admit` and an explicitly
activated `run`. The latter refuses Darwin even with --execute and requires
rank0 plus a systemd invocation. No command here has been run on the fleet.

Offline preparation, using existing qualified clone conventions:

```
python3 scripts/stress_step.py plan --sim-dir "$D3/release-stack" --out steps.json
python3 scripts/stress_step.py prepare \
  --reference-clone "$D3/release-stack/stack-1002b/clone-fp4x-e" \
  --out "$WINDOW/step1" --boot glm53full-kvhr-step1-UNIQUE \
  --extra-gib 1 --c4-mode pool-quarter \
  --cycle-probe "$D3/release-stack/stack-1002b/fp4x-cal-20261005-2215/cycle.py" \
  --gate-metrics "$D3/think10/round5-gpu/gate_metrics.py"
python3 scripts/stress_step.py admit --step "$WINDOW/step1" \
  --sim-dir "$D3/release-stack" --out "$WINDOW/step1/admission.json"
```

`D3` and `WINDOW` are operator paths, not automatically guessed. Prepare copies
tracked recipe sources, the reference .env and its existing copy-guard binary,
keeps unique CTN/runtime paths, and excludes state/logs/cache. It records source,
guard, helper and DRY hashes, and compares all four pin/maxlen pairs. No guard
binary is committed. The source wrapper uses `~/glm-control/bin/ssh` on rank0,
matching the local rank-0 shim and existing fabric aliases. No Mac watchdog.
After copying the prepared step to rank0 during a later authorized window,
repeat the DRY vector there and compare `dry.txt` byte-for-byte before running.
Prepared clone sources and environment must retain their hashes after copying.

A run admission receipt must contain exact `package_sha256`, `dry_sha256`,
`ordinary_bytes`, `max_model_len`, `blocks`, `stress_mode`, `sim_source_sha256`,
`page_cache_credit_GiB: 0`, and four finite `per_rank_lower_GiB >=8.5`. It cannot
be made green by editing `passes_8_5`. Preserve recalibration source, input
receipts and their hashes with it. `admit` uses the retained conservative model;
new calibration is a future review task, not performed by inventing a green row.

Future execution on rank0 only, once admitted:

```
systemd-run --user --unit=glm-kvhr-step1-UNIQUE \
  --working-directory="$WINDOW" \
  python3 "$RECIPE/scripts/stress_step.py" run --execute \
  --step "$WINDOW/step1" --admission "$WINDOW/step1/admission.json" \
  --baseline-cycle "$BASELINE/cycle-summary.json"
```

Procedure, one boot per step:

1. Exclusive coordinator window; preserve serving-watch drop-in/deployment and
   the exact restoration recipe. Document who owns the window and recovery.
   Follow MAC_INDEPENDENT_SERVING: hold + fleet lock, stop the user serving
   watch (SIGTERM detaches), docker stop the existing ranks, dispram postcheck,
   verify idle. Hand off the operator lock to the launcher's own no-clobber
   token. Do not run two launchers or two serving watches.
2. Baseline instrumentation at head +0 and the flusher OFF/ON/OFF experiment
   precede budget changes. Preserve a matched 20-prompt cycle baseline. Refit
   admission with lower bounds and exact costs before any expanded boot.
3. Prepare a fresh unique clone for +1, admit, verify remote DRY, then run as a
   local transient user unit. The existing launcher compacts before boot,
   verifies the 110 GiB preboot floor, lenders and copy guard, boots workers
   first, samples boot/capture guards and admits only after 60 s >=8 GiB on
   all ranks. Its watchdog continues throughout. No in-boot KV resize.
4. The driver starts four persistent 1 s samplers **before boot**. It records
   full meminfo, swap baseline/deltas, buddy order counts, page size, free bytes
   in blocks >=2 MiB and >=8 MiB, vmstat major faults/compaction/allocstall, and
   a continuously followed current-boot kernel journal. PSS is asynchronous
   every 10 s where readable. Kernel journal access is required; absent access
   or stale telemetry fails closed before launch. Store logs below HOME, not
   Spark /tmp (cleared on reboot). A 1 s sampler can still miss shorter peaks.
5. Verify 4/4 FP4x pool block counts, ordinary heads, leases and max length.
   Run matched cycle probes first, then one full-max c1 prompt +1024 decode.
   For the c4 test, send four **distinct** token-ID prompts together, cold-fill
   them (one output each), await all four prefills, then repeat with APC and
   decode 1024 each. Require near-complete cache-hit deltas, four simultaneous
   decoding streams observed, exact prompt/output usage, and no preemptions.
   Standard HTTP has no decode-start barrier; this two-phase test establishes
   retained prefills before the requested long decode phase. Record that
   limitation. Do not claim cold prefill/decode overlap is a barrier test.
6. Repeat an APC prefix and verify hit/query deltas. Run sparkDash prose c1
   (discarded warmup +3 scored, existing /bench convention, 256 output). Re-run
   the same cycle panel after long stress to detect pressure/refault changes.
7. Stop via the launcher's explicit PID and stop/postcheck path; containers
   are preserved. Save RESULT, per-rank minima and swap, kernel/raw buddy logs,
   actual block receipts, request usage/c4 overlap/APC and cycle comparisons.
   Recompute next-step sim from measured lower bounds. Repeat for +2/+3 only
   if newly admitted and the preceding step met every gate. No automatic loop.
8. Restore the best **qualified full GLM** recipe at session end, refresh its
   swap_base, verify exact OK/APC, install its watch drop-in/log symlink, start
   the rank0 watch, verify heartbeat touched(watch), then retire the hold
   and release the owned lock. No automatic restoration after a failed stop.

Stops: any rank <6.0 GiB; any new host OOM/hung-task/stall/fault message;
swap growth **>64 MiB from that boot's rank-wise baseline**; decode cycle median
**>1.02x** the matched baseline for prose or code; stale telemetry >3 s; runtime
failure, nonprogress or timeout. The existing launcher additionally stops
sustained swap >=64 MiB and capture failures. The driver's rank-0 serving floor
8.0 applies after admission; 6.0 is an emergency limit, not permission to serve
at 6.1. Screens do not silently use the historical 7.5 exception. A monitor trip
interrupts blocking HTTP and reaches shutdown without waiting for request
threads to finish. No pinned side tests, builds or snapshot jobs while serving.

The frozen cycle helper measures first-to-last output wall / speculative draft
opportunities, **not CUDA-event cycle time**. The recipe preserves physical
widths/K policy, and the helper requires exclusive request boundaries and 10
attributable prompts per kind. +2% is a conservative operational stop. A
no-tok/s-loss claim needs matched A/B/A rounds with uncertainty, c1/c4 outputs,
acceptance/committed-per-cycle and final sparkDash tok/s; a single passing median
is insufficient (campaign prose A/A tok/s floor ~12.8%). No speed qualification
is claimed by the offline tests.

Recovery: stop advancing immediately; preserve the owned lock if any stop or
lease postcheck fails. Try only bounded docker stop of the named ranks and
capture kernel logs on reachable nodes. Stop the remaining ranks when one node
is unreachable. Retain the node-local control plane/deadman; never keep a
Mac-only heartbeat alive to disguise a failed owner. Do not mark OOMKilled=false
as proof of no host OOM. A fresh OS boot before loading is the known mitigation
for persistent fragmentation; restarting a container is not proof of recovery.

If SSH has remained dead for **15+ minutes**, the owner-approved last resort is
the NUC dashboard smart socket, **plug-2**, which likely powers all four Sparks.
The memory note's NUC helper is:

```
ssh -i ~/.ssh/nuc_agent_ed25519 COORDINATOR_DESTINATION \
  'python3 ~/Infrastructure/ops/antoni_smart_home_control.py status plug-2'
# then the same helper with turn-off plug-2, wait 30 s, turn-on plug-2
```

This is documentation only: no helper is called by the driver or this task.
Verify plug identity/status before switching. The memory note says ~20 s off;
the campaign rules say 30 s, so use the more conservative 30 s. Restore current
qualified **full GLM**, not the memory note's historical DeepSeek command.
After reboot lenders/monitors/deadmen need recovery (they are not boot-persistent),
then preborrow, known-good clone, admission/OK/APC, swap-base refresh, watch and
heartbeat checks. Preserve original INVALID/OOM logs before any reset-invalid.

## Local validation and attribution

Mac-only verification is recorded in `VALIDATION.md`. No GPU kernel was added
or rewritten; expanded shapes and performance remain unqualified. New code is
Apache-2.0. Credit: kindlingai for dispram; Tech2wild/tonyd2wild for the model and
cache-flushing comparison; Linux kernel maintainers for memory interfaces;
the existing fp4x-cal, think10 GateMetrics and sparkDash campaign harnesses for
launcher, request-cycle and dashboard conventions. Existing source/licence
notices are retained; no upstream guard or benchmark helper binary is shipped.
