# Changelog

## 2026-10-10 — full-model public release

Det-align stays on; FP8 draft eh_proj and combined INIT qualification default on,
as do glue-lite F1 router BF16 and F2 workspace reuse. Row selection and F3 remain off.
The public glue defaults are `ROUTER_BF16=1`, `MOE_WS=1`, `DSA_IDX_CACHE=0`,
`IDX_EXPECT=0` (all prefixed `GLM_GLUE_`); `GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1`.
The index expectation now matches the frozen gate's disabled F3 instead of requiring 57 layers.
No cache trim or async v2 is included. The public K3/K2 policy and split32 layout remain.
knapcio selected a 4.5 GiB release stress criterion to match the live floor; other floors are unchanged.
F3, row selection, cache trim, async kstop v2 and K4 are off; tau is 0.74 and decode fairness is 4096/N40.
The exact gen-12 configuration passed the owner-approved reduced release gate and was handed to the serving
watchdog. sparkDash c1 prose/code/structured/json is 39.38/45.74/48.66/45.54 tok/s; RigMark
prose/code/structured is 30.051/42.372/47.338. Temperature-0 repeatability is 30/30 across three runs and byte/token
identical to stack-2. Stress and the 1,071-second soak passed with rank-0 minima 5.809 and 5.803 GiB respectively,
zero swap, errors and preemptions. The reduced scope carried the preregistered 116-item quality panel and paired
16K/128K needle evidence; it did not rerun qeval x3 or exact-config 16K/250K needles, and the documentation does not
claim those as fresh passes.
Handover to `glm-serving-watch` completed at approximately 18:18 Europe/Warsaw.
The public packaging also aligns the display-lender path with `/srv/glm-dispram`,
implements the documented monitor-adoption and serving deadman modes, and makes
the historical pad-hygiene coordinator helper repository-relative.
[eh_proj](docs/eh_proj.md), [glue-lite](docs/glue-lite.md), [gate notes](docs/release-1010-gate.md).


## 2026-10-09 — release/glm53-1009 (deterministic MoE align, temperature-0 reproducibility)

`GLM_MOE_DET_ALIGN=1` is on by default. It replaces vLLM's `moe_align_block_size` for the target and native MTP
Marlin MoE with an original counting-sort CUDA kernel that returns the stock layout with token ids ascending inside
each expert segment. The stock kernel places tokens with `atomicAdd`, so the within-expert order, and with it the
Marlin MoE fp32 reduction order, varied between calls. The kernel is compiled once per boot with the image's `nvcc`
and was faster than the stock align at decode sizes in a single-GPU microbenchmark (2.1-2.6 µs vs 4.4-4.5 µs at
M1-M16; slower at M512, 12.0 vs 6.2 µs, prefill only).
[Determinism](docs/determinism.md); `bench/t0_probe.py` verifies a boot.

Measured on one boot, E; all serving gates passed. Temperature-0 probe on the October 9 qualification boot (same code
and configuration; 30 prompts x 3 sequential repeats, prefix cache reset): **30/30 identical**, against 4/30 with the
stock align on an October 8 boot; concurrent and cross-boot reproducibility remain unqualified. qeval 73/73/73 with the
same two failures each run; cross-boot paired cycle the same night, prose 70.0 ms vs 70.7-70.8 ms. sparkDash c1
prose/code/structured/json 35.22/43.78/48.09/43.38 from one sweep without best-of selection (the previous table's
36.25 came from the selected first of two sweeps; the receipts do not isolate the cause of the difference). RigMark prose/code/structured
29.13/41.75/46.63, code c4 65.77; needles 8/10 twice at 16K, 9/10 twice at 128K, 8/10 twice at 250K; stress rank-0
minimum 5.96 GiB; 43-minute soak with 66 requests, 0 errors and 0 preemptions. Floors are unchanged.
[Gate result](GATE-RESULT.md), [receipt summary](docs/results/release-1009-summary.json).

## 2026-10-08 — release/glm53-1008b (draft-only NVFP4 LM head)

The native MTP draft uses its own NVFP4 (Marlin W4A16) copy of the LM head by default (`GLM_DRAFT_HEAD=nvfp4`,
`GLM_DRAFT_HEAD_INIT=1`), 133.8 MB per rank. The unchanged target model, whose LM head remains BF16, verifies drafts
through standard speculative verification (this does not establish bitwise-identical output across runs). Committed
tokens per cycle were consistent with unchanged on the measured panels; the decode cycle is 4-5 % shorter. At boot, all four ranks qualify
the draft graphs (write coverage, exact draft tokens, bounded float drift) and fall back to the BF16 draft head
together if any rank refuses. The float bound follows the root cause found during qualification: vLLM's
`moe_align_block_size` orders tokens within an expert by atomic arrival, so the Marlin MoE fp32 reduction order and
the draft hidden state can vary by a few ULP. [Draft head](docs/draft-head.md).

Measured on one boot, D; all gates passed. Same-boot ON/OFF confirmation on October 7: prose +5.61 % [+3.94, +7.47],
code+structured +4.43 % [+3.17, +5.71]. Boot D: sparkDash c1 prose/code/structured/json 36.25/44.10/47.31/43.29;
RigMark prose/code/structured 29.32/41.63/46.60; qeval 72/74/73 (mean 73.0); needles 9/10 and 8/10 at 16K,
9/10 twice at 128K, 8/10 twice at 250K; stress rank-0 minimum 6.05 GiB; 41-minute soak with 65 requests, 0 errors and
0 preemptions. Time-slicing stays on: a request decoding during another request's cold ~100K prefill runs at
12.59 tok/s; that prefill's TTFT is 209.1 s, +60 % against the October 7 policy-off boot (cross-boot).

The stress criterion is 5.8 GiB (was 6.0), set for the draft head's bank. The live floor stays 4.5 GiB, now
documented as the lowest harmless ballast level (2.5 GiB, second October 7 run) plus 2.0. README tables come from
boot D. [Gate result](GATE-RESULT.md), [receipt summary](docs/results/release-1008b-summary.json).

## 2026-10-08 — release/glm53-1008 (decode/prefill time-slicing)

Decode/prefill time-slicing is on by default (`GLM_DECODE_FAIR=1`, chunk 4096, 40 pure-decode steps).
While a request decodes, the central scheduler caps the total prefill per step at 4096 tokens, reserving
decode rows first, then runs 40 pure-decode steps before the next prefill chunk. Without a decoding request
the adaptive 2048/4096 prefill is unchanged. An optional schema-2 runtime sidecar (`GLM_DECODE_FAIR_CONTROL`,
`scripts/decode_fair_control.py`) changes the policy or switches it off without a reboot; the profile boots the
policy with no file dependency. [Time-slicing](docs/time-slicing.md).

Measured on one boot, T2; all gates passed, with one documented functional exception (a reasoning probe returned
empty reasoning text once, five of six re-probes were normal; single-request T=0 nondeterminism, policy inert).
A request decoding during another request's cold ~100K prefill runs at 12.46 tok/s; that prefill's TTFT is 212.0 s.
Against the policy switched off on an October 7 boot (0.44 tok/s, 130.9 s) that is +62 % TTFT.
c1 decode, T2 same-boot ON/OFF ABBA over 48 pairs: +0.83 % [-0.36, +2.05]. sparkDash c1 prose/code/structured/json 33.99/42.23/45.04/41.83;
RigMark prose/code/structured 28.67/40.07/44.42; qeval 74/75; needles 8/10 twice at 16K, 9/10 twice at 128K and at 250K;
stress rank-0 minimum 6.30 GiB; 41-minute soak with 71 requests, 0 errors and 0 preemptions.
The container command for this boot differs from the October 7 release only by the four `GLM_DECODE_FAIR*` keys.

The live floor is 4.5 GiB (was 5.5), in use since October 7 after real multi-agent traffic tripped the 5.5 GiB
floor. The ballast evidence shows no harm down to 3.5 GiB. Stress criterion 6.0, capture 6.0, admission 6.5 for
60 s and pre-capture 7.5 GiB are unchanged. README tables now come from boot T2.
[Gate result](GATE-RESULT.md), [receipt summary](docs/results/release-1008-summary.json).

## 2026-10-07 — release/glm53-1006 (262144 qualified)

Qualified 262,144-token context with FP4x latent KV, a 6,318,718,976-byte ordinary head per rank,
4,135 KV blocks /264,640 tokens, adaptive prefill and NVFP4 attention/shared/dense/native-MTP sidecars.
Native MTP, prefix caching, pad hygiene, graph reuse and switched RoCE remain enabled. Recent-FP8 reservation
is zero by default; cached-header CPU boot preflight is advisory. The fast loader remains default;
the ajclark Apache-2.0 coalesced loader port stays opt-in after boot B's external-memory-pressure refusal.

README reports only the qualified configuration: sparkDash first, prose first, c1/c2/c4/c8 aggregate
[per-stream] rates from H; cold 4K/8K/16K/32K prefill in its own table; needle cold TTFT 169.1 s at 128K
and 382.8 s at 250K; RigMark H prose/code/structured 28.42/39.02/44.41 decode tok/s and code concurrency.
qeval x3 on A is 73/71/71, mean 71.67, with no truncations; A and H have identical code and weight hashes.
H needles pass twice at 16K/128K/250K (8/8, 9/9, 9/9 out of 10); stress passes at a 6.69 GiB rank-0 minimum;
40.9-minute mixed-load soak has 0 errors and 0 preemptions.

Memory safety records the bounded ballast evidence: no observed harm down to 3.5 GiB under load.
Release floors are live 5.5, stress/capture 6.0, admission 6.5 for 60 s and pre-capture 7.5 GiB.
Per-lever credits, CREDITS, NOTICE and LICENSES preserve Mia's FP4 KV idea-only credit, kindling's linked
unmodified AGPL dispramd, ajclark's loader port, Tech2wild/tonyd2wild's checkpoint, NVIDIA Marlin,
vLLM contributors and Z.ai. [Gate result](GATE-RESULT.md), [receipt summary](docs/results/release-1006-summary.json),
[export audit](docs/publication-audit.md). Public publication remains owner-controlled; the sanitized squash
is prepared locally without a public push.

## 2026-10-02 (stack candidate 1002b, not released)

Release candidate assembled from `main` (`983ac38`), `boot/fast-load`, `release/stack-1002`, `perf/skip-mla-plan`
and `perf/spec-sample`. The release gate fills the README results; none are claimed here.

On by default:
- Prefix caching: `profiles/serve-args.json` carries `--enable-prefix-caching` in place of
  `--no-enable-prefix-caching`, the same single-token flip as the APC deployment run on the fleet on 2026-10-02.
  The release gate accepts either flag and records which one it booted; its cold prefill and teacher probes
  already carry a nonce or a fresh `cache_salt`. The DSpark K3 profile keeps prefix caching off.
- Native MTP K-stop (`GLM_MTP_KSTOP=1`) with the uniform-batch policy (`GLM_MTP_KSTOP_UNIFORM_BATCH=1`): K3,
  capture sizes `[1, 4, 16]`, four slots and the memory-admitted 544-block KV pool at 32,768 context. Uniform
  batches are on because with the policy off every c2-c4 cycle whose requests stopped at different lengths runs
  the target pass eager. The runtime and hook are the versions that booted and served at c1 on the fleet
  (warm-up treated as synthetic and checked clean afterwards; padded new or resumed requests verified at full
  width; the full GLM-5.3 target and `DeepSeekMTP` drafter bound explicitly), with the uniform-batch policy kept.
- `GLM_INDEXER_SHORTCUT=0`: the short-context DSA shortcut and K-stop transform the same pinned sources and are
  refused together; set it back to 1 together with `GLM_MTP_KSTOP=0`.
- Native MTP shard selection (`GLM_MTP_ONLY_LOAD=1`, `GLM_TARGET_SKIP_MTP=1`) and the one-second health poll
  from `boot/fast-load`. A per-rank hash of the final GPU weights matched the full-scan load on all four ranks.
  The persistent compile cache stays opt-in (`RECIPE_PERSISTENT_CACHE=0`) and the parameter-hash tool is in no
  profile.

Optional, off by default:
- MLA plan skip (`GLM_SKIP_MLA_PLAN=1`, or `ab` with `VLLM_SERVER_DEV_MODE=1` for an in-boot A/B). These keys are
  not in the profile; the launcher passes them only when set and validates them. With K-stop on, a build that
  would skip while K-stop has a decision or guard payload pending plans as stock, so the payload still rides the
  planner copy (counted as `kstop` in the switch status).
- Spec-sample (`GLM_SPEC_SAMPLE=1`); the launcher refuses it with K-stop on (K-stop runs greedy drafts only).
- Display-carveout KV hook (`RECIPE_DISPRAM`, `.env.example`, default `0`): the launcher's four call points for
  the dispram integration; the integration is not included, so any other value is refused.

Checks (offline): the CPU suite in the pinned image, the K-stop real control-flow test with the uniform policy
off and on, six startup-chain scenarios, the four-rank DRY comparison against the deployed APC launch vector and
the memory admission are listed in [docs/validation.md](docs/validation.md#stack-candidate-1002b-offline-checks).

The sections below describe the merged branches as they were written; where they say a lever is off, this
candidate's defaults above take precedence.

## 2026-10-02 (boot/fast-load, not released)

- Native MTP shard selection (`GLM_MTP_ONLY_LOAD=1`, `GLM_TARGET_SKIP_MTP=1`, on in the native profile): the
  K2 draft load reads the 2,343 MTP-layer tensors from shards 270-274 instead of re-reading all 282 shards, and
  the target load no longer reads them. Fail-closed header/index/stream/parameter checks; `audit` mode keeps the
  full scan with the same checks. Same weight-loader calls and bytes as the stock full scan on the real shards.
- Opt-in persistent compile cache (`RECIPE_PERSISTENT_CACHE=1`, `PERSISTENT_CACHE_DIR`), keyed by image ID,
  FlashInfer version, CUDA arch lists, overlay hash and cache layout; copied only inside containers. Per-file
  SHA-256 manifest verified before and after seeding (fail closed); the post-admission save runs beside the
  watchdog with a 630 s deadline and is stopped on abort.
- Boot watchdog polls `/health` every second and samples ranks every 5 seconds until admission (was 10 s).

## 2026-10-02 (probabilistic MTP experiment)

- Add boot-time `GLM_SPEC_SAMPLE=0|1` (default 0), enabling the pinned native MTP
  probabilistic draft path on T>0; temperature-zero rows keep no-noise argmax.
- Domain-separate rejected probabilistic draft resampling from draft Gumbel noise.
  Preserve residual arithmetic and bonus/placeholder/greedy paths behind source pins.
- Add million-draw Torch laws with shared-noise negative controls, real-kernel
  single-GPU distribution/T=0 gates, and sampling validation documentation.
- DRY vectors differ only by the flag. Pinned-image and GPU qualification remain
  pending.

## 2026-10-02 (README results from one boot)

- README "Current results" filled from one fresh-clone boot of `main` (`31b5412`, release commit `8ccac85`):
  sparkDash decode for prose/code/structured/json at c1/c2/c4/c8, cold prefill at 4k/8k/16k/32k (32,256-token
  prompt at the 32,768-token context), and RigMark 1.0.0 from the same boot, replacing the release-gate boot's
  RigMark rows (kept in `docs/history.md`).
- `bench/sparkdash.py` is now the sweep that produced the tables (`full` mode, about 19 minutes; `short` mode
  for the three earlier cells); it reads the sparkDash API base from `SPARKDASH_API`.
- Receipts: `docs/results/readme-1002-boot.json`, `readme-1002-rigmark.json`, `readme-1002-sparkdash.jsonl`.

## 2026-10-02 (stack release candidate, staged levers)

- Launcher: `RECIPE_*` launch-shape switches in `profiles/current.env`, read only by the launcher and never
  passed to the containers; their defaults reproduce the released launch vector, other values are refused and the
  DSpark profile resets them.
- `RECIPE_C4_PASS2_GRAPH=1` (default `0`): capture sizes `[1, 3, 4, 6, 12]`, so the later native MTP pass at four
  requests replays a FULL graph instead of running eager. Config only; target and first-MTP graphs unchanged.
  Expected 0.5-1.5 ms per c4 cycle from profiled traces; not measured on the fleet yet, so it stays off.
- Glue-lite (`overlay/bringup/glm_glue_lite.py`, default off): `GLM_GLUE_ROUTER_BF16`, `GLM_GLUE_MOE_WS` and
  `GLM_GLUE_DSA_IDX_CACHE` remove the router-logit FP32 cast, the per-call Marlin MoE workspace zeroing and the
  index conversion on the 57 skip-top-k layers; no arithmetic change, source-pinned, strict arming counts.
  About 0.39 ms per cycle from traces. CPU and pinned-image tests pass; the single-GPU bit-exactness gates
  (`tests/gpu/`) and a fleet A/B have not run, so it stays off.
- `RECIPE_MAX_MODEL_LEN=44288` (default `32768`): the released 2 GiB pool already holds 44,352 tokens (693
  blocks, one kept as vLLM's null block), so 44,288 needs no extra memory except the longer prefill's working
  memory (about 0.09 GiB on rank 0, estimated). `RECIPE_KV_PIN_L1=1` (default `0`) pins 821 blocks (2.369 GiB).
  The launcher refuses a context that does not fit the pin. Not booted yet, so both stay off.
- `RECIPE_NCCL_NO_LL128=1` (default `0`): `NCCL_PROTO=^LL128` in every container, so NCCL allocates no LL128
  buffers (about 0.44 GiB of pinned host memory per rank). Decode all-reduces use the RoCE path; whether any
  NCCL operation changes is unmeasured, so it stays off until shown bit-exact and not slower.
- Native MTP K-stop (`overlay/kstop/`, `GLM_MTP_KSTOP=1`, default `0`): native MTP with up to three drafts and
  a uniform confidence stop (pass 2 iff p1 >= 0.74, pass 3 iff p1·p2 >= 0.74), local rank-identical decisions
  checked by a 104-byte PyNccl guard before each decision. The launcher then selects K3, capture sizes
  `[1, 4, 16]` and a 544-block KV pool with 32,768 context (rank 0 predicted 8.78 GiB, memory-admitted), or with
  `RECIPE_MAX_MODEL_LEN=44224` and `RECIPE_NCCL_NO_LL128=1` the full 693-block pool (predicted 8.70 GiB, unmeasured);
  every other K-stop layout, including capture sizes 1..16 at 693 blocks (predicted 7.82 GiB), is refused, as is
  the switch with the short-context shortcut on. `GLM_MTP_KSTOP_UNIFORM_BATCH=1` (default `0`) makes batches of
  two or more requests draft all three tokens, so their verify passes replay graphs instead of running eager.
  The served path carries the overlay and its guard only, no gate, canary or A/B harness. Projected +2.6 to
  +7.6 % at c1 offline; the K-stop layout has never booted, so it stays off until a fleet screen and a
  qualification pass.

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
