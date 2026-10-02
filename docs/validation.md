# Validation scope

2026-10-02 dirty-L2 candidate (offline): CPU suite plus the Triton interpreter and sm_121 compile checks of
`tests/test_dirty_l2.py` in the pinned base image, and a startup import chain with `profiles/current.env`
in that image ([receipt](results/dirty-l2-cpu-tests.txt)). GPU exactness and timing evidence:
`results/dirty-l2-evidence.json`. The first fresh-clone fleet gate stopped twice at the launcher's 8 GiB floor
on rank 0 during the cold FlashInfer sampling JIT build right after graph capture (not the lever; the released
configuration has the same build in the same phase). The launcher now builds those FlashInfer modules before the
model containers start (JIT prep, `tests/test_launcher.py`; a no-GPU build of the rendered prep command in the
pinned base image and an unchanged `.ninja_log` on a second run: [receipt](results/jit-prep-check.txt)).

2026-10-02 fresh-clone gate (dirty-L2 fix, with JIT prep): cloned from the remote, configured from `.env.example`
with a fresh runtime path, launched with `./start.sh serve` (per-host weight-hash records reused, JIT prep,
compaction, foreground watchdog). It passed health, an exact reply, four-rank running/no-OOM, both levers armed
and dispatched on all ranks, 60-second all-rank 8 GiB admission after warm-up, qeval x3 73/72/73, RigMark 1.0.0
thinking-on with 15/15 completion gates and sparkDash ([receipt](results/dirty-l2-release-gate.json)). Speed was
decided by the same-boot RigMark A/B under the release rule below ([receipt](results/dirty-l2-rigmark-ab.json)).
The image was the existing per-node build of `Dockerfile.roce`, not rebuilt.

## Release rule

Adopted 2026-10-02. A change merges into `main` when all of the following hold:

1. **Fresh-clone gate.** A clone of the release branch from the remote, configured from `.env.example` and
   booted with `./start.sh serve`, passes health, an exact reply, four running ranks without OOM or restarts,
   the change armed and dispatched on all ranks, and the 60-second all-rank 8 GiB admission after warm-up.
2. **Qeval band.** Three serial qeval runs with the unchanged checker and budgets: mean at least 70.553/75
   (checker-matched baseline mean minus two baseline SD), zero truncation outside the known cap items and no new
   recurring failure among historically always-passing tasks.
3. **Speed, compared with the current default, never with a README row from another boot.** RigMark prose c1
   varies by about ±4 % between runs within one boot, more than a small change moves it.
   - **The change has a runtime switch:** a same-boot A/B. Interleave A1/B/A2 blocks with the same prompts in
     every block, change the switch only while the engine is idle and read it back on all ranks, and report the
     per-triplet difference B - mean(A1, A2) of block medians with its 95 % t interval. A cell regresses when
     the whole interval is below zero; a gain is reported only when the whole interval is above zero.
   - **No runtime switch:** at least two boots per arm, each running the identical warm-up and benchmark
     sequence, compared on the per-boot medians with every boot's value reported.

README rows are the fresh-clone gate boot's own measurements, labelled with its date; the A/B result is shown
next to them, and earlier rows move to `docs/history.md`.

2026-10-01 fresh-clone gate: the release branch was cloned from the remote, configured from `.env.example`
with a fresh runtime path and launched with `./start.sh serve` on all four nodes (full weight-hash preflight,
compaction, empty JIT cache, foreground watchdog). It passed health, an exact reply, four-rank running/no-OOM,
60-second all-rank 8 GiB admission after warm-up, qeval x3 72/72/72, RigMark 1.0.0 thinking-on and sparkDash
([receipt](results/short-dsa-release-gate.json)). The image was the existing per-node build of
`Dockerfile.roce`, not rebuilt. CPU suite for this release: `results/short-dsa-cpu-tests.txt`.

2026-10-01 native MTP: the default is native MTP K2 with async scheduling; its operational quality and
protocol receipts are `results/e5-qeval-admission.json` and `results/e5-comparison.json`.
The 2026-09-30 fresh-clone gate description below keeps its DSpark K3 runtime expectations; it does not
qualify native MTP. The fresh native launcher has offline source/dry/CPU evidence; cold-cache
fresh-clone fleet qualification and multi-day stability remain unknown.

Offline packaging was done on a workstation on 2026-09-30. No Spark container or GPU workload
was started or stopped. Read-only SSH retrieved small checkpoint files, the NCCL binary hash, the compaction
helper and host version information. The saved Boot B configuration and deployed-source archive are the
reference; current serving state was not inferred from those historical receipts.

## CPU and source checks

```bash
GLM_IMAGE_SRC=/path/to/image-source tests/run_cpu_tests.sh
```

The directory must contain `vllm/` from the pinned v11 image. The retained source used for this run was:

```
<extracted image source>
```

All 20 packaged source pins match this extraction. Source trees that carry the Flash prefix-cache patch
have a different `kv_cache_coordinator.py` and are deliberately refused; this is not a reason to relax
the pin. No source extraction is distributed here.

On an idle build host, a CPU-only inspection container can expose the image's source without starting it:

```bash
name="glm53full-source-$(date +%Y%m%d-%H%M%S)-$$"
docker create --name "$name" --entrypoint /bin/true glm53-roce:v11-b58f34ea
mkdir -p image-source
docker cp "$name":/usr/local/lib/python3.12/dist-packages/vllm image-source/vllm
GLM_IMAGE_SRC="$PWD/image-source" tests/run_cpu_tests.sh
```

The inspection container is preserved. The offline packaging run used the already extracted source and
performed no Docker operation on the fleet.

The CPU suite covers exact four-rank dry vectors and no-SSH dry execution; source drift refusal; unchanged
MLA kernel arithmetic bodies; checkpoint manifest integrity and corruption/path-escape refusal; the original
75-task checker; launcher ownership, memory/swap/error decisions and admission deadline; the pinned real
vLLM allocator's allocation/rejection/cancellation/preemption/refcount paths; a NumPy execution of the actual
transformed draft-input slot kernel; 324 loader tensors plus iterator wiring; and 18 RoCE checks including
four real Gloo processes. The Gloo stand-in does not exercise RDMA, CUDA or SM121 numerical attention.

`docs/results/cpu-tests.txt` records the pass output; `docs/results/packaging.json` records counts, commands
and the extracted image source. Eleven small checkpoint files were fetched from immutable public resolve
URLs and passed size/SHA256 checks (`docs/results/download-metadata.json`). Large weights were not downloaded
during packaging. Their hashes retain the original pinned manifests and saved all-four Boot B verification.

The qeval checker SHA256 is `54719522d26996198c870264dfe5a93e2dd23f33436626c2477f1ac71206ffd2`, the exact
checker recorded by Boot B. Only the runner's model alias/default URL/documentation were adapted. It scores
final content, preserving empty-final failures and original budgets. The current suite does not establish
intelligence equivalence by counting scores alone.

## Dry launch comparison

```bash
python3 scripts/compare_dry.py
```

Every explicit environment key and ordered serving argument matches all four saved rank vectors after
the documented instrumentation removals. The report includes the exact rendered commands. Source paths,
unique container names and caches are deployment-specific. The original launcher seeded a warm cache;
the fresh recipe begins with an empty private cache. Removing passive hooks and adding loader file pins
was checked on CPU; its real deployment behavior has not been measured.

## Fresh-clone fleet gate

A fleet qualification is valid only in an exclusive idle window with a working management/local-console
recovery path. It builds the pinned ARM64 image from a clean clone, checks source/NCCL/weight hashes on all
four nodes, verifies the Sync rails/GID/HCA mapping, acquires the fleet marker and boots with the recipe's
own unique names, empty runtime tree, successful compaction and active watchdog.

Its runtime receipt must show four running ranks with no OOM/restarts, health 200, the split32 MLA and SWA
hooks armed, actual TP4/RoCE dispatch, one fixed-K3 speculative config, the intended graphs and an engine KV
capacity consistent with the configured 2 GiB/rank. It records cold-cache boot time separately from the
seeded-cache measurements and verifies the 10 GiB capture / 8 GiB steady-admission floors, no growing swap,
stop behavior and owned-lock release.

Functional qualification includes capital/arithmetic/exact-counting/Polish-text checks, thinking-on and
thinking-off final-content semantics, tool round trips, cancellation and four concurrent short requests;
32k-context and mixed-context stress need their own bounded receipts. Quality assessment uses three original
qeval runs plus fresh same-boot repeated teacher-logprob references at c1/c4 and long contexts, preserving
any unstable or infinite comparison as HOLD. An A/A floor is not a permission to absorb unexplained drift.

Performance qualification repeats `bench/sparkdash.py` at the existing 2200 MHz cap on an idle endpoint,
checks all stream token/reasoning fields and reports medians per prompt type and concurrency with the same timing
definition (`full` mode: the README tables).
Changing clocks is outside this recipe. CPU, dry and saved-history passes cannot substitute for these
fresh-clone fleet receipts. No such gate was executed for this packaging release.

## Runnable fresh-clone release gate (2026-09-30)

`python3 scripts/release_gate.py` prints a plan without fleet access. The explicit `--execute` command
and prepared-image requirements are in [Running the fresh-clone release gate](validation-release-RUN.md).
It creates separate clean detached reference/candidate clones with `git clone --no-local`, then uses
**each clone's own `start.sh serve`**. Untracked `.env`, caches and deployment state are excluded.
Only allowlisted launch configuration is persisted. Docker images must be pinned by complete SHA256
IDs on every rank; full target/draft manifests and NCCL hashes are rechecked before each boot.
The launchers compact memory, create empty private runtime caches, and keep the watchdog active.
Cold here means a cold **runtime/JIT cache**, not an NVMe/page-cache flush, host reboot, or image build.

| Evidence | Protocol | Credit |
|---|---|---|
| Cold boot/runtime | Health + 60-second 8 GiB admission; all four image IDs, no restart/OOM; split32/SWA/source/capture/RoCE markers; served TP4/K3/2 GiB/context config and KV token count | knapcio, Tony, CosmicRaisins, Local Inference Lab |
| Correctness | Warsaw, arithmetic, exact counting, Polish; thinking on/off final semantics; forced tool + round trip; disconnect after a content token and independent completion recovery | knapcio Flash/DS gates |
| Decode sweep | 64 sparkDash cells: prose/code/structured/json × **every integer c1–c16**; discarded warmups, additional prose c1/c4 samples; fresh job identity + full 256-token stream validation | sparkDash DecodeBench contributors |
| Cold prefill | Exact token IDs at 4k/8k/16k/32k/64k/128k; nonce/salt, TTFT to first content token, full usage and cached-token validation; unsupported cells retained | knapcio Flash prefill protocol |
| Prefix scan / T>0 | 512 reproducible random-code records; early/middle/late lookup at client c4; three first/repeat pairs each at T=0 and T=0.8, finite token logprobs and exact JSON ground truth | knapcio Flash prefix_scan lineage |
| Qeval | Original 75 tasks/budgets, c1, three runs **per arm**, all final content/truncations/errors retained; paired grid and median primary score screen | knapcio Flash/DS qeval |
| Teacher KL | Six public prose/code/structured/JSON/math/Polish texts + fixed 8k/24k token panels; A1/A2 and B1/B2 at c1/c4 in the **same window**; token IDs, lengths and complete top20 supports required | knapcio Flash kld_probe and strict comparator |
| Memory/watchdog/stop | Preboot swap baseline; 6 GiB floor, three samples ≥64 MiB swap growth, CUDA/numeric/worker failures, busy GPU without real token progress 180 s, 900 s admission; exact owned stop + GPU idle + lock release | knapcio guarded launcher |

The production profile has four active slots. c5–c16 measures queueing and aggregate client load, not
actual batch16 graph coverage. The prefix test is a **shared-prefix correctness/sampling scan** with
caching disabled, and explicitly makes no warm-hit qualification. A separate enabled-cache profile
and hit-counter evidence would be required to qualify cache reuse. The public short panel and both
long panels are written once by the reference and reused exactly; private traffic is not a dependency.

The KL estimator folds the unobserved tail and uses the candidate's minimum reported logprob for
missing reference support, following the local Flash comparator. It is a **top20 folded-tail estimate**,
not full-vocabulary KL. Full token-grid validation prevents silent truncation or greedy-only fallback.
Default frozen A/A and A/B means for every panel item must each be ≤0.01 at c1 and c4; noisy, missing, non-finite or mismatched
panels produce HOLD. These are conservative initial screen limits, not a transfer of Flash's measured
0.03 threshold. Reports preserve per-panel lengths/positions/p99 and both repeatability measurements.
The 55 primary qeval tasks cannot resolve <1% intelligence equivalence. No sequence bit-equality gate
is imposed on independently generated greedy continuations.

The candidate is retained only if correctness, runtime, prefix/sampling, full prefill scope, qeval and
KL screens pass and prose c1 median reaches 50 tok/s. The current 32k profile cannot qualify 64k/128k;
its unsupported cells therefore force HOLD. A dedicated qualified long-context profile is still needed.
A timeout or implementation failure ends in HOLD with raw receipts; it is not a rejection of the change.
The gate does not change GPU clocks, networking, model bytes, kernel arithmetic or serving parameters.

Offline verification uses `tests/test_release_gate.py` plus the launcher/recipe/source regressions.
Tests run at `nice -n 10` with numerical libraries capped at two threads. Four network readers are used
only during owned fleet c4 panels/scans. CPU mocks do not exercise CUDA, RDMA, cold boot or fleet stop.
The offline checks produce no fleet result; fleet receipts come only from `--execute` runs (the dated
gates at the top of this page).

Offline verification on 2026-09-30: **74 CPU checks passed** (13 recipe, 9 launcher, 18 release,
13 allocator, 1 transformed-kernel, 2 loader, 18 RoCE), with 20 image source pins and 324 loader
tensors. The complete-suite receipt and final release/recipe reruns are retained in
`docs/results/release-gate-cpu.txt`. A complete-window CPU simulation reaches the
32k unsupported-scope HOLD and reference-recovery branch; no real endpoint or fleet stop was exercised.
