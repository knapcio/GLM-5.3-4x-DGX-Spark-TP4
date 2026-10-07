# Local validation, 2026-10-06

Mac-only, Python 3.14.6. **109 selected CPU tests passed**:

| Suite | Tests | Scope |
|---|---:|---|
| test_kv_headroom | 10 | Cache gates/off switch, checkpoint exclusion, post-sync recheck, common pin, buffer capacity, stop despite helper failure, missing metrics/stale telemetry, Mac execution refusal |
| test_launcher | 56 | Existing launcher behavior/defaults, boot/stop/watch and graph shape wiring |
| test_recipe | 24 | Recipe/source drift, manifests and DRY behavior |
| test_adaptive_chunk | 14 | Existing centralized adaptive scheduler behavior retained |
| test_fp4_kv_admission | 2 | Retained prefill calibration and conservative admission boundary |
| selected test_fp4_kv_integration | 3 | Hook composition, original/new max geometry and VMM region alignment |

Logs are in `validation/` (trailing whitespace normalized for review). The three integration tests are the hook-order,
new-max/FP8 geometry, and dispram seam/alignment tests; the Torch-dependent
spec-class test was not included. Full tensor/GPU tests were not run.

All 23 base and 7 compatibility source pins passed against a local component
fixture assembled from existing extracts. `pinned-source-check.json` records
which source root/hash supplied each exact module and the two additional
scheduler broadcast dependencies. This verifies source bytes and CPU control
flow, not a running container or a single complete image provenance claim.
An older full extract failed the gumbel compatibility pin, and a newer extract
failed the KV-coordinator base pin; their bytes were not changed or accepted
as matching. The final fixture uses only independently matching source files.

Four fresh local clone DRY preparations (+0,+1,+2,+3) passed. `dry-steps.json`
records all-rank byte pins, constructor capacity env and hashes. Existing
qualified clone-e `.env` and its copy-guard binary were used as reference,
without altering that clone. No guard binary is committed. The region planner
returned exact ordinary heads 1/2/3/4 GiB with the common 2046 MiB carveout.

`stress_step.py plan` produced source-bound `steps.json`. Its larger-head
`admit` smoke returned REFUSED; there is no sim-gate bypass. A `run --execute`
unit test on Darwin refused before any subprocess or network call. Cache tests
injected fake meminfo/sync/drop callbacks and temporary state directories:
no write to /proc/sys/vm/drop_caches occurred.

Python compilation, sampler-program AST parsing, shell syntax and git diff
whitespace checks passed. Linux unit execution was not tested on the Mac.
The Colima Docker socket was inaccessible under the local sandbox; no Docker
container/image test was run. No dependency installation or network fallback
was used. The helper/kernel/runtime shape qualification is a future fleet
window, after admission recalibration. No SSH, GPU activity, host-memory side
allocation, service installation, sysctl change or push was performed.
