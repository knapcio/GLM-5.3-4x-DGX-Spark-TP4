# Pad hygiene v2

2026-10-03, Europe/Warsaw. Offline Mac work on `perf/pad-hygiene-v2`,
base `release/stack-1003` / `2538edfc307c8cf5f871491d6d6107968c457cf5`,
historical worktree `perf/pad-hygiene-v2`.

`GLM_PAD_HYGIENE=1` now includes the existing remap only in captured descriptors
that can receive fewer live tokens than their captured size. The analyzer reads
the real priority-ordered `_candidates` table and uses the pinned native
`_is_compatible` predicate. Capture analysis and dispatch share the same lookup-key
helper, including the m12/reuse policy. Short-DSA descriptors are exact-only by
native dispatch. Both warmup and capture forward functions carry a scoped boolean;
it is restored even when a forward fails. Potentially padded graphs include the
op at identity capture and continue to read the stable device map at replay.

Eager target and both draft roles set a host padding boolean from actual and padded
counts before forwarding. Exact eager forwards skip the padding remap. Existing
heterogeneous K-stop row remaps still run when their source map is nonidentity.
Draft reset clears the padding boolean along with the existing row map and mask.
The custom op and its live-row arithmetic are unchanged. Capture selection uses
only host shapes and the already agreed layout; no rank-local values, tensor reads,
new collective or synchronization. Boot/control digests and every existing prepare,
sampling and shortcut guard field remain intact. The flag still defaults to 0.

## Descriptor table

The test extracts `KSTOP_K2_CAPTURE_SIZES` from the actual launcher and executes
the pinned/composed native candidate builder, dispatcher and capture loop with CPU
recording substitutes for CUDA graph operations. Shipped K2 sizes are [1,4,12,16].
Target and first-MTP have the following identical table under both layouts:

| M | Requests | q | Context | Remap in graph | Reason |
|---:|---:|---:|---|---|---|
| 2 | 1 | 2 | stock / short | no | exact c1 |
| 3 | 1 | 3 | stock / short | no | exact c1 |
| 4 | 1 | 4 | stock / short | no | exact c1 |
| 4 | 2 | 2 | stock | no | exact c2/q2 |
| 6 | 2 | 3 | stock | no | exact c2 reuse; unselected in m12 layout |
| 12 | 4 | 3 | stock | yes | c3 9-to-12; also c2 6-to-12 in m12 layout |
| 12 | 4 | 3 | short | no | exact c4 |
| 12 | 3 | 4 | stock | yes | c2/q4 8-to-12, including synthetic paths |
| 16 | 4 | 4 | stock | no | exact warmup c4/q4; M12 serves smaller q4 inputs |

Later-MTP: stock M1/c1/q1 has no remap; stock M4/c4/q1 carries remap for
c2/c3; short M1/c1/q1 and M4/c4/q1 are exact-only and have none.
A c4 replay of the *shared stock* M12/q3 must retain remap because that descriptor
also serves padded c3. The separate exact c4 short graph excludes it.
For the older [1,4,16] plan, M16/q4 can receive padded c2/c3 and retains remap;
this decision follows the table rather than the token size alone.

The independent test oracle enumerates physical request/query shapes through the
actual composed dispatcher and matches every descriptor's recorded remap count.
Additional coverage checks five alternate banks, all 0/1/k2 policies and both
layouts, native varlen descriptors, exact/padded eager calls, scoped-state restoration,
changing replay tails at one stable source address, both draft preparation hooks,
flag-off field guards, named guard digests and live expert IDs/weights/output bit
identity using the pinned modular Marlin hook with CPU expert substitutes.

## Mac checks

Cached wheels only (torch 2.13.0); no installs or downloads. Source extraction:
`receipts/glm53-full-20260929/day3/apc-prep/source`.
Receipts and reproduction scripts: `tests/results/padv2/` in this worktree.

- Padding 14 PASS; composition 14 PASS; pure K-stop 12 PASS; all K-stop 29 PASS
  with single-rank transport substitutes (not real Gloo).
- Recipe 24 PASS; launcher 55 PASS; release gate 18 PASS; short-DSA Mac subset
  6 PASS; spec-sample 6 PASS; conditional distribution 1 PASS (FP32/BF16).
- Persistent cache 10 PASS; kernel CPU 1 PASS; dirty-L2 12 run / 4 image-dependent
  skips; glue-lite 35 run / 1 image-dependent skip; parameter hash 17 run / 2 skips.
- Source pins 30 PASS (23 main + 7 compatibility); all composed transforms compile
  and reject drift. All 16 frozen campaign adapters compile.
- Four-rank DRY comparison PASS for padding 0 and 1. Python/shell syntax and
  `git diff --check` PASS.

These are CPU/source results. Actual CUDA graph replay, Marlin GPU arithmetic,
real four-rank Gloo, full paired T=0 matrix and GPU timing remain for coordinator
or GPU execution. No Docker, SSH, fleet, push or network operation was executed.

## Old c1 kernel cost estimate

The saved profile expects `target:75,mtp:1` routed layers. The unchanged custom
op launches one CUDA remap kernel per routed layer per forward. One steady c1
verify/draft cycle therefore adds `75 + d` launches per rank, where `d` is the
number of executed MTP forwards (1..3): **76..78 launches/cycle/rank**. These
launches disappear in v2's exact c1 target, first-MTP and later-MTP graphs.
Across four ranks the count is 304..312, executed concurrently; it is not four
times the wall-time cost.

With measured per-launch cost tau, old overhead is `(75 + d) * tau` per cycle,
or `(75 + d) * tau / A` per emitted token when A tokens are emitted per cycle.
For each 1 microsecond of kernel cost this predicts 76..78 microseconds per
cycle. No remap kernel timing or accepted-tokens/cycle receipt is available, so
an absolute millisecond or percentage estimate from this count would be invented.

The user-supplied boot medians (31.4 vs 32.6 / 32.9 tokens/s) correspond to
3.68% / 4.56% lower throughput, or 1.17 / 1.45 ms extra per emitted token.
That is the observed whole-run difference, not isolated remap cost or a controlled
A/B measurement; it cannot establish that the removed launches explain all of it.

## Exact coordinator command

The campaign control-flow fixture is external to this repository. Provide its
`day3` directory (containing `mtp-kstop`) as `GLM_CAMPAIGN_DAY`. The helper
resolves this repository relative to its own path, uses the locally cached
digest-pinned ARM64 image without network or GPU access, and produces the real
CPU compatibility suite, Gloo guards and all 16 paired T=0 matrix receipts.

```bash
GLM_CAMPAIGN_DAY=/path/to/day3 bash tests/run_pad_hygiene_v2_coordinator.sh
```

Successful execution must produce `tests/results/padv2/coordinator/PASS.json`,
all 16 matrix receipts, exact token-ID equality for each padding off/on pair,
and collective refusal for divergent padding flags. It does not measure CUDA
or serving performance.
