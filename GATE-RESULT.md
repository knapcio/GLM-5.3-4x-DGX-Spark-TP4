# Release gate result: decode/prefill time-slicing (2026-10-08 02:23 to 05:56 Europe/Warsaw)

**Verdict: PASS. Time-slicing 4096/N40 is on by default and is the serving configuration.**

- Code: `release/glm53-1006` plus the time-slicing policy, its launcher wiring, defaults and tests.
  Context 262144, FP4x KV, 6,318,718,976 B head, 4135 blocks (264,640 tokens), live floor 4.5 GiB.
- The DRY render of all eight container commands (four JIT-prep, four serve) differs from the October 7 release
  only by `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40` and
  `GLM_DECODE_FAIR_CONTROL`. vLLM arguments, volumes, docker flags, KV layout and graph sizes [1,4,12,16] are
  identical; the floor header is identical because the October 7 serving boot already ran with a 4.5 GiB live floor
  (the public default then was 5.5 and is now 4.5). On every rank the graph capture receipts equal the release boot's; `GLM_DECODE_FAIR armed`
  appears on rank 0 only.
- Gate boots also set `GLM_PARAM_HASH=1` and `VLLM_SERVER_DEV_MODE=1` (hash RPC only).
- T2 booted with the runtime sidecar provisioned ON (schema 2, 4096/N40) so that the same-boot ABBA test could
  switch the policy off and on. T1 was the same code with the boot policy and no sidecar; it passed stress, needles,
  the scenario and the functional checks, and was stopped before sparkDash because it cannot switch in-boot.

## Gates on boot T2

| Gate | Result | Rule | Verdict |
|---|---|---|---|
| Admission | health 318.7 s, admission 378.7 s; pool 4135 x4, KV 264,640, served 262144; 0 fatal lines; admission minima 7.06/8.46/8.13/8.44 GiB | floors | PASS |
| c1 same-boot ABBA (ON 4096/N40 vs OFF, x2, 8 exact rank-0 acknowledgements, ends ON) | all 48 pairs **+0.83 %** [-0.36, +2.05]; prose +1.51 % [-0.63, +3.43]; code +0.29 % [-1.09, +1.64] | point >= -1 % and CI contains 0 | PASS |
| Scenario: A decodes during B's cold ~100K prefill | A 14.15 / 10.77 tok/s (mean 12.46); B TTFT 216.1 / 208.0 s (+62 % vs policy off, 130.9 s); 0 preemptions, 0 errors, no foreign traffic | A >= 10 tok/s, TTFT <= +80 % (235.6 s) | PASS |
| Memory stress (c4 4x65,024+1024, single 261,120+1024) | minima 6.30/7.77/7.95/7.73 GiB; swap 0; preemptions 0; APC hit 64,896/65,024 | 5.8 GiB applied (release criterion 6.0) | PASS |
| Needles 16K/128K/250K, twice | 8,8 / 9,9 / 9,9 of 10; registry correct; TTFT 25.3 / 167.4 / 376.8 s | long >= control - 1 | PASS |
| Functional (APC, exact copy/numbers, c1-c2-c1, cancel, 24.6K chunked prefill during decode, tools) | all PASS, 0 preemptions | | PASS |
| Functional reasoning item (17x23, thinking on) | answer 391 with `stop`, but empty reasoning text; re-probed six times on the same boot: 1 empty, 5 with reasoning. T1 passed it | | model-level T=0 nondeterminism on a single request, where the policy is inert; noted, not a blocker |
| sparkDash c1 x5, c2/c4 x3, c8 x2, prefill 4K-32K x3 | see README; 0 failed runs | | recorded |
| RigMark 1.0.0 | prose 28.67, code 40.07, structured 44.42; code c4 65.62; 8K cold prefill 913; all gates | workload gates | PASS |
| qeval x1 | 74/75 (failed `code_camel_to_snake`), primary 54, 0 truncations | 70.553 | PASS |
| Mixed-load soak, 40 min (c1-c4) | 71 requests, 0 errors, 0 preemptions; minima 6.18/7.16/7.45/7.72 GiB; rank-0 quiet drift -0.063 GiB; step ratio prose 1.003 / code 1.000 | 0 errors, drift <= 0.3 GiB, ratio <= 1.02 | PASS |
| Collect | 0 fatal lines x4, containers running, not OOM-killed, 0 restarts; post-soak quiet 6.31/7.51/7.97/7.81 GiB | | PASS |

Cross-boot checks against the October 7 release boot H, for drift only: paired cycle prose 0.991 [0.987, 0.998],
code 1.005 [0.996, 1.009]; sparkDash c1 prose -2.9 % (runs 33.2-35.9 vs 32.7-37.8), code +0.6 %, structured
-0.7 %, json +0.3 %; c2/c4 within -4.5..+4.6 %; prefill 4K-32K +0.0..+2.8 %. All four c8 cells were 2.4-4.7 %
lower (n = 2). At c8, a queued prompt prefills when a slot frees while the remaining streams decode, which is the
mix the policy paces; the receipts do not establish that this is the cause.

## Time-slicing arms (October 7, one boot, policy switched by sidecar)

Scenario as above; two repetitions per arm; c1 panel of 24 pairs per arm against OFF.

| Arm | A tok/s during B's prefill | B TTFT | c1 paired vs OFF |
|---|---:|---:|---|
| OFF | 0.435 | 130.9 s | — |
| 4096 / N20 | 7.07 | 169.9 s (+30 %) | -0.45 % [-2.3, +1.3] |
| **4096 / N40** | **13.03** | 212.4 s (+62 %) | -0.43 % [-2.0, +1.1] |
| 2048 / N10 | 6.60 | 182.4 s (+39 %) | -0.36 % [-2.2, +1.5] |

A's gaps are bimodal: median 0.08-0.09 s (its solo cadence) and one ~5.2 s gap per prefill chunk. The long gap
matches the mixed-step fit `0.443 + 0.001164 x C` seconds. 4096/N40 was chosen for multi-agent use; 2048/N10 is
dominated by N20.

# Previous gate: 262144 context (2026-10-06 19:32 to 2026-10-07 02:50 Europe/Warsaw)

**Verdict: 262144 PASSES with the floors from the ballast test (live 5.5 GiB, stress criterion 6.0 GiB).**
Boot H serves it now under `glm-serving-watch`. No public release was made.

- Release branch `release/glm53-1006`. The fleet clone was a fresh clone of `0394c04`. Later commits change only
  floors, tests and harness: `f364b13`, `71e8d27`, `f4e8dab` and `56a558e`. Weights and runtime are identical for
  every boot (per-rank parameter and buffer digests match on A, B2, F and H).
- Layout: FP4x KV with a 6,318,718,976 B head and 4135 blocks (264,640 tokens). NVFP4 attention, shared, dense and
  MTP sidecars are reused by hash. The recent-FP8 bank is gone and adaptive prefill is on. Gate boots also set
  `GLM_PARAM_HASH=1` and `VLLM_SERVER_DEV_MODE=1`, which add the hash RPC only.
- The DRY render differs from the previous server (g) only in these settings: max_model_len, KV bytes, pool blocks,
  `GLM_PRECAPTURE_FLOOR_GIB`, the recent-FP8 selectors, and an explicit `GLM_LOADER=fast` (equivalent to g's
  `GLM_FAST_LOAD=1`).
- Receipts and raw data are in `diagnostics/glm53-full-20261006-gate/` (RESULT.json, a/ b/ b2/ f/ h/ and memcliff2/).

## Boots

| Boot | Context | Loader | Health / admission | Outcome |
|---|---:|---|---:|---|
| A | 262144 | fast | 317.5 / 377.5 s | Full qualification passed. Soak passed under the one-sided step rule. |
| B | 262144 | coalesced | — | **Refused at load.** On rank 2: "coalesced loader detected external memory pressure". Coalesced loading stays opt-in. |
| B2 | 262144 | fast | 308.5 / 368.5 s | Reproducibility passed (hashes equal A). **Stress 6.33 GiB < 6.4** under the floors in force then. |
| F | 248320 | fast | 311.5 / 371.5 s | Fallback qualified (stress 6.65 GiB). Soak stopped at 13 min by re-plan (0 errors so far). |
| H | 262144 | fast | 318.6 / 378.6 s | Ballast test, then the full qualification under the new floors. **Serving.** |

Every boot was preceded by memory compaction on all four nodes. Boot A had the launcher's own compaction
(sudo journal 19:49:49-53); B, B2, F and H had an explicit compaction followed by the launcher's.

## Boot minima (MemAvailable GiB, ranks 0/1/2/3, sampled at 10 Hz)

| Phase | A | B2 | F (248320) | H |
|---|---|---|---|---|
| weights load | 12.05/12.73/13.14/13.46 | 12.50/12.56/13.09/13.45 | 11.96/13.04/12.85/12.78 | 12.43/12.54/13.41/13.31 |
| KV allocation + profile | 8.68/9.43/10.04/9.68 | 9.16/9.71/9.63/9.63 | 9.07/10.21/10.34/10.45 | 8.51/9.18/9.97/9.84 |
| graph capture | 8.10/8.37/8.90/9.25 | 8.44/8.60/8.92/8.71 | 8.28/9.10/9.38/9.71 | 8.34/8.46/8.90/8.57 |
| post-capture to health | 7.58/8.21/8.74/8.05 | 7.61/8.01/8.90/8.03 | 7.29/8.80/8.31/9.11 | 6.94/7.76/8.73/8.32 |
| admission (60 s) | 7.70/8.39/8.90/8.07 | 7.64/8.51/8.98/8.07 | 6.90/8.66/9.29/9.22 | 6.66/7.77/8.14/8.84 |

- The in-process value just before capture, on A, was 8.68/9.43/10.04/10.04 (floor 7.5).
- In F and H, rank 0 dips by about 1.3 GiB for about 25 s right after health 200.
- Memory pressure (PSI some, avg10) peaks at 8-34 % during the weight load and at 10 % or less during capture.
- Quiet levels after admission and after the soak, rank 0 first:
  - A: 6.52 → 6.46
  - H: 6.50 → 6.44
  - F: 7.13

## Ballast test on H (owner request 23:52)

The test used memcliff2: plain, unpinned ballast that releases itself at 2.5 GiB. The watch floor was lowered to
2.8 GiB for this phase only. Ratios below are paired against the unballasted base (s0).

| Step | Rank-0 quiet / minimum | Cycle prose / code | TTFT c1 | c4 wall | 120K TTFT | PSI some total | Direct reclaim / compact stalls | Preemptions |
|---|---|---|---|---|---|---|---|---|
| s0 base | 6.84 / 6.38 | 74.16 / 84.44 ms | 355 ms | 92.5 s | 153.9 s | 67 ms | 1111 / 663 | 0 |
| 5.0 | 5.06 / 4.97 | 0.999 / 0.992 | 0.993 | 0.962 | 0.999 | 0.5 ms | 0 / 0 | 0 |
| 4.0 | 4.05 / 3.99 | 0.991 / 0.992 | 1.002 | 0.953 | 1.000 | 1.5 ms | 513 / 4 | 0 |
| 3.5 | 3.56 / 3.50 | 1.011 / 0.990 | 1.013 | 0.955 | 1.000 | 2.6 ms | 927 / 8 | 0 |
| Rank 1 at 3.5 | 3.57 / 3.22 | 1.001 / 0.981 | 0.988 | 0.955 | — | 4.5 ms | 330 / 0 | 0 |

- No kernel lines appeared apart from the routine allocation messages, and the ballast never released itself.
- **No harm appeared down to 3.5 GiB.**
- Applying the floor rule (lowest harmless level + 2.0) gives **live 5.5**, and **stress criterion 6.0** (live + 0.5).
- Capture headroom (6.0), admission (6.5 for 60 s) and pre-capture (7.5) are unchanged.
- The watch floor went back to 5.5 afterwards; the watch environment was checked to confirm it.

## Stress (c4 = 4 x 65024 prompt + 1024 output, resident in a shared pool; single = 261120 + 1024)

| Boot | Rank-0 minimum: c4 / single | Minima, ranks 0/1/2/3 | Criterion | Result |
|---|---|---|---:|---|
| A | 6.565 / 6.706 | 6.56/6.84/8.10/7.32 | 6.4 | PASS |
| B2 | 6.348 / 6.331 | 6.33/7.12/8.03/7.01 | 6.4 | FAIL; the probe stopped B2 |
| F (248320: 61568 / 247296) | 6.646 / 7.130 | 6.65/7.60/8.12/8.42 | 6.4 | PASS |
| H | 6.712 / 6.694 | 6.69/7.69/7.48/7.66 | 6.0 | PASS |

All runs had 0 preemptions, retained the prefix cache (APC repeat hit 64896/65024), 0 swap and a clean monitor.
Single-request decode takes 27.5 s for 1024 tokens.

## Decode speed: paired cycle c1 x10 against g's receipts (165312, 17:32)

| Measurement | Prose cycle (ratio, 90 % CI) | Code cycle | Prose / code tok/s ratio |
|---|---|---|---|
| A, after stress and needles | 76.72 ms (1.025, 0.978-1.071) | 89.40 (1.049, 0.998-1.072) | 0.984 / 0.941 |
| A cycle2, quiet | 77.65 (1.030, 1.005-1.049) | 84.73 (0.995) | 0.927 / 0.973 |
| **B2, first measurement** | 73.99 (0.983, 0.974-0.998) | 83.32 (0.990) | 1.021 / 0.994 |
| F, first measurement | 74.34 (0.991) | 82.01 (0.984) | 1.019 / 0.992 |
| F, after stress | 74.48 (0.991) | 83.47 (0.993) | 1.004 / 0.994 |
| **H, first measurement** | 73.52 (0.986, 0.972-0.990) | 82.63 (0.982) | 1.021 / 1.000 |

A's prose penalty was specific to that boot: B2 and H at the same 262144 config are 1-2 % faster than g, and
RigMark on H is at or above the 98K e-boot. **Boot C (165312) was therefore not needed** (rule: only if B2 ≥ +2 %).

A is also noisier than other boots. About one request in six runs roughly 25 % slower (seen in cycle, sparkDash
and one soak probe).

## sparkDash (thinking off, 256 tokens, aggregate tok/s; per-stream rates in the receipts)

| Prompt type | H c1 | H c2 | H c3 | H c4 | H c8 | A c1 | F c1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose | **35.00** | 47.85 | 53.25 | 62.76 | 61.38 | 35.24 | 34.72 |
| code | 41.96 | 49.39 | — | 67.45 | 61.92 | 42.57 | 40.80 |
| structured | 45.36 | 54.93 | — | 88.35 | 82.11 | 44.13 | 45.41 |
| json | 41.72 | 53.77 | — | 75.23 | 71.74 | 36.93 | 41.87 |

Cold prefill on H, in tok/s with TTFT:

| Size | Prefill tok/s | TTFT |
|---|---:|---:|
| 4k | 973 | 4.2 s |
| 8k | 930 | 8.9 s |
| 16k | 880 | 18.7 s |
| 32k | 864 | 38.0 s |

A measured 975 / 770 / 753 / 723 tok/s; F measured 973 / 916 / 890 / 879 tok/s.

Long-prompt cold TTFT on H:

| Prompt | TTFT |
|---|---:|
| 16K | 25.6 s |
| 128K | 169.1 s |
| 250K | 382.8 s |

On A the same prompts took 27.2 / 195.3 / 440.1 s.

## RigMark 1.0.0 (thinking on, low; c1 decode tok/s, median of 5)

| Boot | Prose | Code | Structured | c4 code aggregate | 8K cold prefill | 8K warm replay | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| **H** | **28.42** (27.86-29.16) | 39.02 | 44.41 | 64.93 | 881 | 15679 | all True |
| A | 26.32 | 37.08 | 42.04 | 64.03 | 909 | 15895 | all True |
| e (98K, earlier) | 27.68 | 39.55 | 43.21 | 62.55 | 908 | — | all True |

## Quality

- qeval x3 on A (262144): 73 / 71 / 71, **mean 71.67** (threshold 70.553), primary 54/51/51, no truncations.
- Recurring failures are `code_word_wrap`, `code_camel_to_snake` and `math_m9` (two of three runs each). There was
  no request failure.
- qeval was not repeated on H, which has the same weights (equal hashes) and the same code.
- Needles (score out of 10, two repeats each; every run had all registry fields correct and finish=stop):

| Boot | 16K | 128K | ~250K | 240K |
|---|---|---|---|---|
| A | 9 / 8 | 9 / 9 | 8 / 8 | — |
| F | 8 / 8 | 9 / 9 | — | 8 / 8 |
| H | 8 / 8 | 9 / 9 | 9 / 9 | — |

## Soak (40 min, mixed load)

The mix ran at c1 to c4: thinking-on prose at T=0.7, code, long prompts from 16K to 200K, APC repeats and client
aborts.

| Boot | Duration / requests | Errors | Preemptions | Rank-0 minimum | Rank-0 quiet drift | Step ratio, last/first prose / code |
|---|---|---:|---:|---:|---:|---|
| A | 45.1 min / 51 | 0 | 0 | 6.32 | -0.042 | 0.965 / 0.999 |
| **H** | 40.9 min / 59 | 0 | 0 | 6.28 | -0.127 | 0.987 / 0.994 |
| F (stopped at 13 min) | 13 min / 19 | 0 | — | — | — | — |

- A's soak was recorded as **PASS** under the coordinator's ruling (22:15): its first verdict was a false negative
  from a two-sided step check.
- The check is now one-sided (`bench/soak.py`, `f364b13`): a soak fails only if the last 10 minutes are more than
  2 % slower than the first 10 minutes.
- On H, APC repeats reached a median TTFT of 20.3 s, and long-prompt TTFT had a median of 124 s (maximum 284 s,
  under load).

## Loader

The coalesced loader was refused for the second time today. All four ranks aborted about 10.5 s into the target
load:

- Rank 2 MemAvailable fell from 18.76 to 12.82 GiB.
- Credited destination commit was 1.24 GiB and accounted transient memory 3.80 GiB.
- That leaves about 1 GiB unexplained, against a 1 GiB margin.

Coalesced loading stays opt-in. The normal fast loader is reproducible: hashes on A, B2, F and H are equal, and
health takes 308-319 s.

Fix `71e8d27` makes the launcher report the real exception instead of the routine import_utils NCCL traceback.

## Open items

- README and CHANGELOG finalized on October 7 from H, with qeval explicitly labelled A (identical weights).
  Public publication is still pending owner review of the sanitized local snapshot.
- The rank-0 clone runs `0394c04` with a one-line patch to the cluster.py floor range, and the watch drop-in sets
  `RECIPE_LIVE_FLOOR_GIB=5.5`. A fresh clone of `56a558e` has the same defaults built in.
- Boot-to-boot spread at the same config is about 0.37 GiB in stress minima (rank 0: 6.33 to 6.71). The margin to
  the 6.0 criterion is therefore about 0.3 GiB at worst.
- The slow-request outliers on A (about +25 % step, roughly 1 in 6) are not explained.
- Failed boots leave `state/deployment.json` behind; it has to be renamed by hand.
