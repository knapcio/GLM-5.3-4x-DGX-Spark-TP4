# integ-1006 — offline integration and coordinator procedure

Mac-only integration on `perf/integ-1006`, starting at `3488797`, with merge
parents `593c791` (KV headroom) and `44cf298` (coalesced loader). No SSH,
fleet access, image builds/pulls, deployment, service installation or pushes
were performed. The serving configuration below is from saved receipts and
owner instructions, not a new live query.

## Serving B and preserved switches

B: `glm53full-nvfp4a-20261006-b`, original coordinator clone
`/srv/glm/glm-control/runs/nvfp4a-20261006/clone-a`.
NVFP4 sidecar: `/srv/glm/models/GLM-5.3-attn-nvfp4-87cf357`, read-only,
385 logical attention matrices, Marlin W4A16, native MTP weights still INT8.
FP4x KV, adaptive 2048/4096 prefill, max length 98176, 1573 equal blocks,
ordinary head 1 GiB/rank, carveout 2145386496 B/rank.

All default switches are preserved: attention defaults to INT8,
`GLM_LOADER` unset retains `GLM_FAST_LOAD`, ordinary head defaults to 1 GiB,
`RECIPE_PAGE_CACHE_POLICY=0`; the timer is not installed or enabled here.
NVFP4 registration precedes reader registration; the sidecar transform wraps
the selected reader, and the draft bypass remains intact. An explicit
`GLM_LOADER=coalesced` now selects coalesced transport with NVFP4 even when
`GLM_FAST_LOAD` is unset. Sidecar reads/repacking remain outside the reader's
own staging budget; total host-memory monitoring covers the combined path.
No kernels, clocks, graph widths, dispram carveout or scheduler slots change.

## Memory evidence and exact common geometry

Receipt root on the Mac:
`/srv/campaign/diagnostics/glm53-full-20261006-nvfp4a/window`.
`a/done-mem.json`, `b/done-mem.json` and `b/memwatch-minima.json` are bound
by SHA256 in the generated `ladder.json`.

Quiet post-warm medians, rank 0/1/2/3, GiB:

- INT8: 9.567 / 10.548 / 10.887 / 10.793.
- NVFP4 B: 11.181 / 12.264 / 12.562 / 12.445.
- Measured gain: 1.614 / 1.716 / 1.675 / 1.652.
- NVFP4 quiet minima (20 samples): 11.140 / 12.246 / 12.530 / 12.413.
- Minima across B's non-boot phases: 10.711 / 11.343 / 11.971 / 11.705.
  These rank minima occur at different times; they are not a simultaneous
  observation. Rank 0's 10.711 is the rigmark phase; the needle phase reached
  10.725. The saved long needle reaches 96000 prompt tokens, not the new shapes.

Use the exact ABI rather than the approximate 32.7 KB/token estimate:
79 MLA rows ×368 B +22 index rows ×132 B = **31976 B/token/rank**
(31.976 decimal KB, 31.227 KiB); block64 =2046464 B. Region512 alignment
is included. Carveout remains 2145386496 B, never credited as newly freed
host memory. Each rank gets the same ordinary bytes and block count; the
existing MIN lease negotiation remains the authority for usable tail memory.

| Step | Extra head GiB | Total ordinary GiB | Ordinary bytes/rank | Equal blocks | max_model_len | Added prefill arena MiB |
|---|---:|---:|---:|---:|---:|---:|
| B / control0 | 0 | 1 | 1073741824 | 1573 | 98176 | 0 |
| 1 | 1.0 | 2.0 | 2147483648 | 2097 | 131712 | 36.972 |
| 2 | 1.5 | 2.5 | 2684354560 | 2360 | 148544 | 55.528 |
| 3 | 2.0 | 3.0 | 3221225472 | 2622 | 165312 | 74.014 |
| 4 | 2.5 | 3.5 | 3758096384 | 2884 | 182080 | 92.500 |

Lengths preserve B's 39-block physical slack: 98176 +64×(blocks−1573).
The separate theoretical six-block capacity limit is higher and is not the
recipe length. Total length includes 1024 output tokens, not an additional
1024 beyond the advertised limit. Heads are 2 MiB aligned.

Subtracting head growth alone from observed rank-0 stressed minimum gives
9.711 / 9.211 / 8.711 / 8.211 GiB for steps1..4. These are **projections**, not
sim lower bounds: added arena, decode logits, block tables, graph/workspace
peaks, longer-context loss and measurement uncertainty still need charging.
Step4 has only 0.211 GiB above the live floor before these costs; step3 has
only 0.211 GiB above the sim floor before these costs. Start the proposal at
+1.0 and evaluate +1.5 only after matched stress data; +2.0/+2.5 are conditional
probes and may remain inadmissible. No cache-flush credit is used.

Admission: **all four finite sim lower bounds ≥8.5 GiB**, exact package,
ordinary bytes, blocks, max length, c4 mode and source bindings, zero cache
credit. The 60 s boot admission remains ≥8.0 GiB on every rank. Once admitted,
rank0 must remain **≥8.0 GiB live**; any rank <6.0 is an emergency stop, not
permission to serve below8.0. No historical7.5 exception or8.5 bypass.

The retained simulator still charges pre-tiled FP4 long-prefill loss and
refuses all expanded steps. Even adding the measured quiet gain as an
optimistic sensitivity leaves rank0 at 6.609 / 5.747 / 4.887 / 4.026 GiB.
This is a known calibration mismatch with the qualified B path. It is not
removed or made green here. Generated `admission.json` files are REFUSED.
Before any expanded boot, retain an independently reviewed recalibration
against NVFP4 B and adaptive-path stress receipts, all warmed graphs and exact
per-lever costs. Unknown longer-prefix/concurrent loss remains conservative.
The driver checks the source-hashed receipt; changing `passes_8_5` alone cannot
admit a step. `calibration_model` describes the model and
`calibration_sources` maps step-relative model/input files to SHA256; the
exact `sim_source_sha256` must occur in that map. The numeric receipt must be
produced by that retained reviewed calculation, not hand-edited. The offline
work does not claim this recalibration or fleet qualification has happened.

## Worst resident stress within one shared pool

`stress_step.py` now accepts half-GiB increments. `geometry(extra,
'pool-quarter')` supplies the exact c4 lengths below. Four independent
max_model_len contexts would need about four times the pool and are refused;
queueing or preemption would not qualify four resident streams.

| Step | Full c1 prompt | c1 output | c4 prompt per request | c4 output each | c4 total each | c4 required / pool blocks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 130688 | 1024 | 32448 | 1024 | 33472 | 2097 /2097 |
| 2 | 147520 | 1024 | 36608 | 1024 | 37632 | 2357 /2360 |
| 3 | 164288 | 1024 | 40832 | 1024 | 41856 | 2621 /2622 |
| 4 | 181056 | 1024 | 44992 | 1024 | 46016 | 2881 /2884 |

Each boot runs: exact OK/APC smoke; compatible full-weight hashes; matched
20-prompt c1 cycle panel (10 prose +10 code); one full-length c1 request;
four distinct concurrent long token-ID requests in the shared pool; APC
repeat; sparkDash prose c1 warmup +3 scored runs; the same cycle panel again.

The c4 test cold-fills four distinct prefixes (one output each), drains,
then submits all four retained prefixes together for 1024 output each.
Require near-complete prefix-hit deltas, four simultaneous decoding streams
observed, exact prompt/output usage and zero preemptions. A further APC
repeat checks query/hit deltas. Standard HTTP has no decode-start barrier;
this is a resident-prefix worst shared-pool screen, not a synchronized cold
prefill/decode barrier or four full advertised contexts. The underlying
headroom harness labels pool-quarter SCREEN_ONLY for literal full-max c4;
the coordinator's SHARED_POOL_PASS describes only the table above.

Frozen `cycle.py` imports the frozen `gate_metrics.py`. Compare prose and
code median cycle_ms to the normal control; either >1.02× stops the step.
Cycle timing is output wall per speculative opportunity, not CUDA-event
kernel timing. Preserve individual results, draft opportunities, acceptance
and committed tokens. This screen does not establish unchanged tok/s;
that needs matched A/B/A rounds and uncertainty.

## One driver and exact future procedure

Driver: `scripts/integ_1006.py` (plan / prepare / run). Mac `run --execute`
refuses before any process or network action. Execution is limited to
rank0 under its systemd user manager. No automatic next-step loop, promotion,
reboots or power control. Mac CPU runner: `tests/run_integ_mac.py`.

Offline preparation (executed locally with a validation tag; use a fresh tag
and output for the real future package):

```bash
RECIPE=/srv/projects/glm53-full-integ
D3=/srv/campaign/diagnostics/glm53-full-20260929/day3
RECEIPTS=/srv/campaign/diagnostics/glm53-full-20261006-nvfp4a/window
WINDOW=/srv/campaign/diagnostics/glm53-full-20261006-integ/prepared-WINDOWTAG
python3 "$RECIPE/scripts/integ_1006.py" prepare \
  --tag WINDOWTAG --out "$WINDOW" --receipts "$RECEIPTS" \
  --sim-dir "$D3/release-stack" \
  --reference-clone "$D3/release-stack/stack-1002b/clone-fp4x-nvfp4a" \
  --cycle-probe "$D3/release-stack/stack-1002b/fp4x-cal-20261005-2215/cycle.py" \
  --gate-metrics "$D3/think10/round5-gpu/gate_metrics.py"
```

This creates control0 (normal fast loader, head1GiB, 98176), step1..4
(coalesced +NVFP4, corresponding heads/lengths), and five fresh original-B
restore clones. Hash instrumentation `GLM_PARAM_HASH=1`,
`VLLM_SERVER_DEV_MODE=1` is enabled for control and candidates; all loader
budgets keep defaults128MiB/16threads/direct1/owned4096MiB. The page-cache
policy is explicitly0 for candidates. Restore clones retain original B
sources and disable coalesced/hash instrumentation. Guard binary is copied
from the reference clone into the private package; no binary is committed.
Each restore also freezes the existing node-local `memwatch.py` source from
the B receipts. The previous floor-guard unit is paused during handoff and
a fresh guard is attached to the restored serving watch at its original
8.05GiB/all-rank threshold, which is stricter than the requested rank0 floor.

Normal final-weight hashes must use **NVFP4 B**, not an old INT8 manifest.
Control0 obtains the compatible normal-loader manifests before step1. It
uses the current known capacity; it does not bypass a larger-head sim gate.
Its source/vector/image checks, launcher guards, sampler, smoke and full
hash checks still apply. No hash RPC is sent to an uninstrumented serving B.
If compatible hashes already exist, retain their origin, full inventory,
aliases, source/quantizer contract and checksums; the supplied driver takes
its own control0 so there is no silent substitution of the older8269-entry
manifest or a synthetic3142-case manifest.

The following is **for a separately authorized coordinator window**, not
executed by this Mac task:

1. Review source pins, source identity, copied original-B restore sources,
   four-rank DRY vectors and current serving B identity. Stage the complete
   prepared directory under
   `/srv/glm/glm-control/runs/integ1006-WINDOWTAG` on rank0 through the
   existing coordinator staging process. Keep all outputs under HOME, not
   fleet `/tmp`. Runtime directories and container names must be unused.
   Hashes/DRY are relocation independent; `run` rechecks them byte-for-byte.
   Do not start other clients, snapshots, loaders or GPU jobs in this window.
2. Execute the normal control with the same one driver, locally on rank0:

   ```bash
   W=/srv/glm/glm-control/runs/integ1006-WINDOWTAG
   systemd-run --user --unit=glm-integ1006-WINDOWTAG-control \
     --working-directory="$W" \
     --setenv=PATH=/srv/glm/glm-control/bin:/usr/local/bin:/usr/bin:/bin \
     /usr/bin/python3 "$W/step0/clone/scripts/integ_1006.py" run \
     --execute --window "$W" --index 0
   ```

3. Require `step0/run/RESULT.json` CONTROL_CAPTURED, restored_B=true;
   four normal manifests, matched cycle panel, `restore0/run/RESTORED-B.json`,
   watch active and heartbeat touched(watch). The driver updates
   `window.json` to the freshly restored B identity and binds control hashes.
   It preserves original B's stopped containers. Restore uses fresh names
   because the original launcher refuses reused container/runtime paths.
4. Produce source-bound recalibrated `step1/admission.json` as described above.
   If the numeric gate refuses, **stop here with B serving**. The driver
   checks admission and control hashes before touching the watch or fleet.
   For an admitted step, run:

   ```bash
   systemd-run --user --unit=glm-integ1006-WINDOWTAG-step1 \
     --working-directory="$W" \
     --setenv=PATH=/srv/glm/glm-control/bin:/usr/local/bin:/usr/bin:/bin \
     /usr/bin/python3 "$W/step1/clone/scripts/integ_1006.py" run \
     --execute --window "$W" --index 1
   ```

5. Before candidate launch the driver snapshots deployment, four docker
   inspect records and watch drop-in; requires a free coordinator lock or the exact serving deployment token,
   creates a no-clobber operator token when the lock is free, or changes the
   exact serving token to a fresh operator token under a local exclusive flock;
   checks candidate/restore image IDs against the actual current B containers;
   starts a local expiring hold; detaches the serving watch; uses B's own
   stop/postcheck path. Only after a verified idle handoff and owned-lock
   release may the candidate launcher take its own no-clobber lock.
6. Four persistent1s full meminfo/swap/buddy/vmstat/current-boot kernel/PSS
   samplers start before launch. Four additional100ms memory readers must
   each emit before launch; missing/stale streams fail the step. The first
   coalesced candidate runs the pinned CUDA direct-I/O transport gate on
   **all four idle nodes**,132 exact cases, non-default consumer stream and
   a deliberately pending upload event. The driver repeats this gate on
   each later step; test containers and staging files are preserved. No
   builds/pulls occur. Gate failure restores B after successful cleanup.
7. The unmodified launcher safety path verifies110GiB preboot floor,
   compaction, lenders/copy guard, workers first, capture safety and60s
   admission. After health200, record the first observed epoch (1s polling
   precision) before hashes. Verify exact pools/leases/lengths, NVFP4 matrices
   and native MTP receipt on every rank; all coalesced target/draft receipts
   must have complete/cuda/direct=true. Hash only after drain and explicit
   all-rank env/attach receipts, using32MiB chunks and8.5GiB hash floor.
   Require full parameter/buffer dtype/shape/name/alias inventories and
   bit-identical logical bytes on all four ranks. Scratch differences are
   reported separately. Missing/incompatible hashes fail the step.
8. Perform the table's stress and matched c1/cycle gates. Stop any rank<6GiB,
   rank0<8GiB after admission, kernel OOM/hung-task/stall/fault, swap growth
   >64MiB from each boot baseline, stale telemetry>3s, timeout/nonprogress,
   usage/APC/overlap/preemption/hash/layout failures or cycle median>1.02×.
9. The driver stops and preserves the candidate with explicit launcher PID
   termination and its own stop/postcheck. On pass **or failure**, after
   verified cleanup it boots the index-specific original-source restore
   clone with B's ordinary1GiB / FP4x / adaptive / NVFP4 /98176 configuration.
   Require60s admission, exact OK, APC. Detach the restore launcher, refresh
   rank-wise swap_base, write the serving-watch drop-in/log symlink, start
   watch and the source-bound original floor guard, retire the hold, verify
   both units active, fresh watchdog and touched(watch). The
   driver releases only the exact restored owner token after these checks. B configuration is
   restored; the original parked B containers stay preserved.
10. Require SHARED_POOL_PASS, restored_B=true and complete receipt ledgers.
    Refit the next-step bounds from that measured shape/stress result,
    retaining uncertainty and all costs. Run `--index 2`, then3, then4 with
    fresh source-bound admission each time and unique systemd units
    `glm-integ1006-WINDOWTAG-stepN`. Every step returns to B; no automatic
    advancement or new serving promotion occurs. A failed step ends the
    ladder. Retries need a fresh package/tag and may specify current restored
    B via prepare's `--serving-ctn` / `--serving-clone` / `--serving-guard-unit`; no same-directory retry.

If candidate stop, node access, postcheck, restore boot or handoff fails:
RESULT=RECOVERY_REQUIRED. The driver never releases an unknown/retained lock
or boots over incomplete cleanup. It attempts bounded stop of named transport
containers/candidate ranks and preserves logs. Hold renewal ends when the
controller exits, leaving the existing node-local heartbeat/deadman policy
in charge. No automatic reset-invalid/reboot/power action. Coordinator
recovery must prove all ranks idle and leases released before a fresh B
restore; an unhealthy/unreachable node can require the established OS/lender
recovery procedure. Do not describe a failed restore as restored_B=true.

## Required receipts and boot comparison

Per step: `run/{serve.log,rankR-boot.log,rankR-memory.jsonl,
rankR-loader-memory.jsonl,memory-summary.json,receipts.json,
boot-breakdown.json,weight-comparison.json,manifests/rankR.json,
cycle-before/,single-max, cold-prefill/,resident-decode/,c4-trace.json,
apc.json,sparkDash-c1.json,cycle-after/,stop.log,RESULT.json}`.
Restore verification is in `restoreN/run/RESTORED-B.json` and its raw logs.

`boot-breakdown.json` records exclusive main-weight time, MTP time, total
model-construction time and its remaining construction/repacking component,
graph time, health elapsed and separate admission time for each rank.
Profile/KV/warm and launch/API residuals remain grouped/unknown when log
boundaries cannot separate them. Take the slowest-rank critical path; never
sum four parallel durations or overlapping producer/consumer counters.
All timestamped source lines and loader receipts are retained.
`boot-vs-normal.json` compares the normal control and candidate records;
KV capacity differs, so this is not a loader-only causal speed benchmark.

For each coalesced target/draft start/end epoch, the driver clips100ms samples
and records observed min/max MemAvailable, max(MemTotal−MemAvailable),
Dirty/Writeback peaks and swap growth relative to preboot. NVFP4 sidecar CPU
read/native repack peaks are covered by the full streams and worker logs;
report them grouped with construction if no separate phase timing exists.
Sampling can miss sub100ms peaks; no continuous upper-bound claim.
Hash time is excluded from health timing. No synthetic fixture result is
substituted for full-checkpoint final weights or fleet boot measurements.

## Local validation

See `VALIDATION.md` and the requested Mac diagnostic directory. All CPU test
files were attempted. Source pins23+7+13, registered sidecar hash equality,
seven coordinator/geometry/recovery tests, half-GiB dry preparation and
syntax/whitespace checks pass. Full distributed/socket and image/interpreter
checks are explicitly limited by the Mac sandbox; no fleet speed, GPU
transport or full-model hash qualification is claimed.
