# Changelog

## 2026-10-02 (README results from one boot)

- README "Current results" filled from one fresh-clone boot of this release:
  sparkDash decode for prose/code/structured/json at c1/c2/c4/c8, cold prefill at 4k/8k/16k/32k (32,256-token
  prompt at the 32,768-token context), and RigMark 1.0.0 from the same boot, replacing the release-gate boot's
  RigMark rows (kept in `docs/history.md`).
- `bench/sparkdash.py` is now the sweep that produced the tables (`full` mode, about 19 minutes; `short` mode
  for the three earlier cells); it reads the sparkDash API base from `SPARKDASH_API`.
- Receipts: `docs/results/readme-1002-boot.json`, `readme-1002-rigmark.json`, `readme-1002-sparkdash.jsonl`.

## 2026-10-02 (README results layout)

- README results show only the current default: sparkDash first (prompt-type rows prose/code/structured/json,
  columns c1/c2/c4/c8, aggregate with per-stream in brackets; separate cold-prefill table by length), then
  RigMark (c1 decode per workload with run range, code-workload concurrency, prefill 8K). Cells the
  2026-10-02 gate boot did not measure are marked with a dash.
- The DSpark K3 alternative rows, the same-boot dirty-L2 A/B table and the boot-to-boot RigMark note moved to
  `docs/history.md`.
- Credits: the short-context DSA shortcut now credits the Guess-Verify-Refine authors for the short-sequence
  all-selected case; the dirty-L2 fix is marked as the recipe's own diagnosis and fix; the launcher JIT prep and
  weight-hash records have their own row in the stack table.

## 2026-10-02 (dirty-L2 fix, fresh-clone gate)

- Add the dirty-L2 fix (`GLM_DIRTY_L2=discard`) to the native MTP K2 profile: the split32 MLA reduce ends with
  a CTA barrier and `discard.global.L2` of the fp32 partial lines it consumed, so they are not written back
  to DRAM while the following INT8 o_proj streams its weights. Output bits unchanged; split 32 only; no
  buffer, graph or argument change. Off in the DSpark profile.
- Evidence: o_proj after the MLA 124.9 -> 111.5 us (M3) and 150.0 -> 111.7 us (M12) on one GB10, bit-equal
  in eager and graph replay; in-boot A/B on the released configuration -1.07 ms per cycle at c1 (95 % CI
  0.32-1.82) and -2.39 ms at c4 (CI 1.51-3.19).
- Launcher preflight keeps a per-host record of the last full weight hash (file size, times, inode, device)
  and skips the about 18 minute rehash when nothing changed and the record is less than seven days old;
  `VERIFY_WEIGHTS=full` always rehashes. The remote verification program now embeds the manifest once
  (58.6 KB instead of 110.7 KB, against Linux's 128 KiB single-argument limit).
- Launcher JIT prep: `./start.sh serve` now builds FlashInfer's sampling and batch MLA modules into the fresh
  per-deployment cache before the model containers start (one `docker run --rm` per node in parallel, same image
  and environment with `PYTHONPATH` empty, no GPU or network, 600 s timeout, fail-closed before compaction and the
  110 GiB preboot check). The first fresh-clone gate of this release stopped twice at the 8 GiB floor on rank 0:
  the cold sampling build (three nvcc processes, about 2-2.7 GiB of host memory for about 37 s) ran right after
  graph capture, in the released configuration as well. Same sources, flags and modules (ninja command hashes
  equal the fleet's in-boot build); the serving process finds them up to date. DRY prints the four prep commands
  before the four model commands. The 8 GiB floor is unchanged.
- Fresh-clone gate with the JIT prep step: health 200 after 467 s, qeval 73/72/73, RigMark code/prose/structured
  34.4/24.9/36.2 tok/s, code C4 aggregate 66.4, prefill 8K 954.4; sparkDash prose/code c1 30.71/34.03.
- Release rule: speed is compared with the current default in the same boot when the change has a runtime switch,
  otherwise on at least two boots per arm; no longer against a README row from another boot. The qeval band is
  unchanged. Same-boot RigMark A/B of this fix: prose no regression (+0.13 %, 95 % CI -0.48 to +0.75 %),
  structured +1.43 % (95 % CI +0.08 to +2.77 %).

## 2026-10-01 (short-context DSA, fresh-clone gate)

- Enable the short-context DSA shortcut (`GLM_INDEXER_SHORTCUT=1`) in the native MTP K2 profile: while a context
  is at most 2048 tokens the indexer query GEMM and logits are skipped; selection and key/index caches are unchanged.
  Off in the DSpark profile.
- First fresh-clone `./start.sh serve` on the fleet; fixed four launcher bugs it found: per-host NCCL hashes
  (the measured fleet has two 2.30.7 builds), fleet-marker check on the head only, RoCE readiness counters
  (`error_hca`) no longer stop a boot as worker errors, and an 8 GiB floor during graph capture instead of 10 GiB
  (native MTP K2 capture dips to about 9.5 GiB on the head; the 10 GiB pre-capture check is unchanged).
- Current results from that boot: RigMark code/prose/structured 33.8/25.3/35.9 tok/s, qeval 72/72/72.

## 2026-10-01

- Select native Z.ai MTP K2 async with packed-module mapping fix, FP8 KV block64 and no external drafter.
- Keep qualified Red Hat DSpark K3 as the selectable `RECIPE_PROFILE=dspark-k3` alternative.
- Make RigMark 1.0.0 thinking-on/effort-low results primary; retain named sparkDash results separately.
- Record exact async qeval qualification, snapshot boot evidence and offline checks; preserve attribution.

## 2026-09-30

- Initial full GLM-5.3 TP4 recipe from the selected Boot B RedHat DSpark K3 configuration.
- Included only the used split32/unsplit sparse-MLA kernels, independent drafter pool, fast-loading policy
  and switched RoCE transport, with image/module source pins.
- Added immutable target/drafter downloads with complete SHA256 manifests, guarded unique-name launch,
  foreground watchdog, CPU tests, benchmarks and four-rank dry-vector evidence.
- Removed diagnostic flags, passive receipt hooks, phase profiling and the superseded duplicate speculative argument.
- Documented current measured performance, unresolved quality checks and the limits of offline validation.
