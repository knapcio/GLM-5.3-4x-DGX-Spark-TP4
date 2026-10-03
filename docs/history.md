# Measurement history

Historical measurements below retain their original release scope. Archived
launch/configuration receipts in `docs/results/` have deployment identities
redacted for public export; their source hashes refer to the original private
receipts, not a byte-identical copy of the redacted files. Literal deployment
addresses are template markers, and loopback endpoints use `localhost`.
Current stack-1003 values are only in the release placeholder tables.

2026-09-30: Boot A (first boot of the K3 configuration): prose c1 33.62, code c1 39.97, prose c4 aggregate 61.27 tok/s,
2200 MHz GPU cap, 2 GiB configured target KV per rank, 32k context/four slots. Boot wall time 336.4407 s
was measured with the workstation monotonic clock through health and the postboot snapshot.
The final restart passed a bounded functional gate; the selected speed numbers retain their original run provenance.

2026-09-30: Boot B retained the same fixed RedHat K3 serving arithmetic. Its unprofiled medians of three
were 30.18/38.60/62.68 tok/s; boot/admission wall time 335.2422 s. The engine reported 41,728 KV tokens.
Serial qeval was 71/72/72. The unchanged-reference A/A checks remained HOLD, including an infinite long-panel KL.
No newly measured numerical variant was qualified. Full intelligence equivalence was not established.

2026-09-30: Offline packaging removed passive tail/graph receipts and profiler arguments, reduced the two
speculative arguments to their effective final value, removed inactive code paths and kept current serving
settings. The checkpoint revision already contains the measured thinking-off template. No serving experiment
was executed for the package, no clock was changed and no fleet workload was interrupted.

2026-09-30: README switched to medians over both boots of the K3 configuration: prose c1 33.3 (runs 33.62/33.24/34.85
and 30.18/29.48/33.30), code c1 39.6 (39.29/40.28/39.97 and 40.52/38.60/35.68), prose c4 aggregate 62.0
(61.27 and 59.49/63.30/62.68).

2026-09-30 (third boot, 512-token chunks): historical native512 final medians prose/code-c1 31.49/40.02tok/s (five runs each), prose-c4 aggregate61.65 (three). The2048 candidate qeval70/72/73 was held by a per-run71 veto; review replaced that uncalibrated veto with a mean-score noise allowance and required chunk-exercising long-context coverage. These512 measurements are not included in current2048 medians.

2026-09-30 (2048-token chunks): native RedHat K3, split32/RoCE, drained2048 with4096 constructor capacity,2GiB/rank and32K context. All nine long-context panels12/12; strict qeval 72/72/70, mean71.333 against threshold70.553. Current medians across all scored served-config runs: prose-c1 31.02, code-c1 37.37, prose-c4 aggregate62.26/per-stream16.31tok/s; counts5/5/3. Cold16K request-wall input throughput 936.05tok/s, paired gain27.37% vs512 (three triplets). Live KV pool34,304tokens; historical41,728 is not current. Qualified deployment used v5 immutable prepared snapshots with exact readback, health116.44s/admission189.21s; the published native launcher has not been re-timed for this profile. V2/v4 80-round c1 CIs missed0.5ms precision, c4 full-wave harness mixed widths; Split16 partial fixed-width proxy+4.31ms remains parked for loaded oracle/timing qualification. No new decode kernel was adopted. Complete current receipts: `docs/results/w4-*`.

2026-09-30 (E1 three-boot refresh): same served native K3/2048 configuration, 256 output tokens, thinking off; five prose/code c1 runs and three prose c4 waves per boot, warmups discarded. Per-boot medians:

| Boot | Prose c1 | Code c1 | Prose c4 aggregate |
|---|---:|---:|---:|
| D2-W4 README boot | 31.02 | 37.37 | 62.26 |
| D2-W5 stock boot | 31.63 | 39.91 | 61.69 |
| E1 baseline arm | 31.85 | 39.66 | 62.22 |

Pooled all-scored medians: prose c1 31.63, code c1 39.66, prose c4 aggregate 62.22 tok/s. E1 baseline used native A with candidate banks resident; joined cycle timers were inactive. This refresh does not adopt a decode lever or infer a speed improvement from prose throughput. Receipts: `docs/results/e1-three-boots.json`.

2026-09-30–2026-10-01 (E2b/E4, E5 default selection): native Z.ai MTP K2 async versus Red Hat
DSpark K3 sync, both TP4/split32/RoCE, 2 GiB target KV per rank, 32K context, four slots and 2048
prefill chunks. The owner selects MTP for thinking/prose use; DSpark remains a selectable alternative.
No new GPU kernel or changed weights were adopted. TensorFold results from keys/@u1tra_instinct pointed
the investigation toward native MTP; Z.ai, vLLM, Red Hat, tonyd2wild and Tech2wild retain credit.

| RigMark 1.0.0, thinking on / effort low | Code c1 | Prose c1 | Structured c1 | Code C4 aggregate | Cold prefill 8K |
|---|---:|---:|---:|---:|---:|
| Native MTP K2 async (E4) | 33.316 | 24.815 | 35.291 | 70.607 | 959.091 |
| Red Hat DSpark K3 sync (E2b) | 35.708 | 23.097 | 42.123 | 63.504 | 950.005 |

Rates are tok/s; five complete decode runs per workload, 4096-token budget, 15/15 completion gates
per config. C4 uses three separate 256-token code waves and end-to-end aggregate timing; prefill uses
three 8192-token prompts and effective input throughput to first token. Warm replay prefill is
967.058 for MTP and 950.308 for DSpark; prefix caching was disabled. MTP prose is 7.4% faster here,
while DSpark code and structured are faster. These are historical protocol-matched receipts,
not a paired causal experiment or a whole-intelligence proof.

| sparkDash, thinking off, 256 tokens | Prose c1 | Code c1 | Prose C4 aggregate |
|---|---:|---:|---:|
| Native MTP K2 async (E4, 5/5/3 runs) | 29.11 | 32.78 | 59.43 |
| Red Hat DSpark K3 sync (pooled E1 reference, 15/15/9 runs) | 31.63 | 39.66 | 62.22 |

RigMark is the primary results table for the chosen workload. sparkDash remains a separate short
thinking-off protocol; its c4 rate excludes prefill/first-token latency and cannot be substituted for
RigMark C4. Exact settings and source receipt hashes: `docs/results/e5-comparison.json`.
E5 verified the existing async snapshot: health200/catalogGLM-5.3/OK/all4running/noOOM/restarts0,
265.70s overall boot, 60-second all-rank8GiB admission and actual async-K2 worker audit. Corrected async
qeval scored71/72/72,mean71.667>=70.553,zero truncation/no new recurring failures: PASS. The earlier
72/71/72 native MTP quality receipt was sync. Fresh-clone cold boot remains unknown.

2026-10-01 (short-context DSA shortcut, fresh-clone release gate): the shortcut was qualified on native MTP K2
async in the in-boot A/B harness before packaging. Same-boot RigMark 1.0.0 thinking-on, shortcut vs control:
code 34.16/33.79, prose 25.46/25.07, structured 36.29/35.49 tok/s; sparkDash prose c1 29.97/29.34, code c1
34.98/34.03. Paired fixed-width cycle time: c1 -1.88 ms, c4 -2.14 ms (CI 1.80-2.48). A first c4 aggregate
reading (68.35 vs 72.97) did not reproduce in an interleaved A1/B/A2 check with four distinct prompts per
round: -0.30 % (95 % CI -1.50 to +1.10), null A1/A2 -0.22 %. A four-rank component gate matched selector,
key-cache and index-cache bytes against the stock path at 2047/2048 tokens and the switch to stock at 2049.
Qeval in that window 71/72/71 with one `code_two_sum` 420-token cap hit (a known cap item in controls).

The release was then measured from a fresh clone with `./start.sh serve` (empty JIT cache, native loader,
health 518 s after launch): qeval 72/72/72; RigMark code/prose/structured 33.82/25.31/35.93, code C4
aggregate 66.08, cold prefill 8K 966.6 tok/s (15/15 completion gates); sparkDash prose c1 29.40, code c1
34.75, prose c4 aggregate 59.64. All-rank MemAvailable at least 9.04 GiB for 60 s after warm-up. The previous
default row (E4 exact-snapshot MTP K2 async) was RigMark 33.32/24.82/35.29, C4 70.61, prefill 959.1 and
sparkDash 29.11/32.78/59.43; the C4 difference is within the run-to-run spread seen in the interleaved check
above. This first fresh-clone boot also found four launcher defects (see CHANGELOG). Receipts:
`docs/results/short-dsa-*`.

2026-10-02 (RigMark C4 follow-up): an interleaved A1/B/A2 check that sends RigMark 1.0.0's exact C4 request
(the same code prompt four times, thinking on) found the shortcut +1.77 % faster than control (95 % CI -0.44 to
+4.19 %, 12 triplets, null A1/A2 +0.24 %); 98 % of the difference is decode after the last stream is admitted.
RigMark's C4 median depends on how the four streams are admitted: rounds where all four start together reach
69.5-75.6 tok/s, rounds where one stream starts first and three follow reach 65.4-70.6 tok/s, on both arms.
The lower fresh-clone C4 readings came from runs where two of three rounds started one-plus-three.

2026-10-02 (dirty-L2 fix, release candidate): on native MTP K2 async the INT8 o_proj after the split32 MLA ran
slower in the served graph than in isolation. The cause was the MLA's own scratch: the split-K kernel leaves
1 MiB of fp32 partials per row dirty in L2 after the reduce has consumed them, and their write-back to DRAM
competes with the o_proj weight stream. On one GB10 (serving stopped, two identical runs) a reduce that ends
with `discard.global.L2` of its consumed lines brought o_proj from 124.9 to 111.5 us at M3 and from 150.0 to
111.7 us at M12, within about 3 us of the clean isolated o_proj; the discard itself costs 0.1-0.3 us per
reduce. The output was bit-equal to the released kernel at split 32 in eager and graph replay at M3 and M12.
In one boot of the released configuration, switching the discard on and off between requests (A1/B/A2
rounds, 16 groups per width, cycle time from GPU events, maximum over ranks) measured -1.07 ms per cycle at c1
(78.41 -> 77.35 ms, 95 % CI 0.32-1.82; A/A floor +0.51, CI -0.06 to 1.03) and -2.39 ms at c4 (155.92 ->
153.59 ms, CI 1.51-3.19; A/A +0.06), with committed tokens per cycle unchanged within noise. The released
layout has no four-request later-MTP graph, so at c4 only cycles with the target and first MTP pass at
(12,4,3) and an eager later step were scored, identically for both arms. Fewer splits change fp32 summation
order and were not adopted; the repository ships only the static discard at split 32. Receipts:
`docs/results/dirty-l2-*`.

2026-10-02 (dirty-L2 fix, fresh-clone release gate): the first fresh-clone attempts stopped at the launcher's 8 GiB
capture floor on rank 0 (7.86 and 7.67 GiB) while FlashInfer's sampling module was compiled inside the serving
process right after graph capture; the launcher now builds it before the model containers start (see CHANGELOG).
The re-run from a fresh clone (health 467 s after launch, rank-0 minimum 9.23 GiB in the capture window) gave
qeval 73/72/73 (mean 72.67); RigMark 1.0.0 thinking-on code/prose/structured 34.41/24.93/36.18, code C4 aggregate
66.45, cold prefill 8K 954.4 tok/s (15/15 completion gates); sparkDash prose c1 30.71, code c1 34.03, prose c4
aggregate 63.40. RigMark prose runs in that boot spanned 24.06-25.92. The gate's rule at the time required each
RigMark cell to be at least the previous README row from the 2026-10-01 boot (33.8/25.3/35.9), so it read FAIL on
prose only (24.93 < 25.3).

A same-boot RigMark A/B then compared the fix with the previous default on one fresh deployment of the same
branch. A measurement-only switchable reduce selected discard off (A) or on (B), both at split 32 and bit-equal to
the released reduce on one GB10; the switch was changed only with the engine idle and read back on all four
ranks. Four A1/B/A2 triplets, each block prose #1-3 and structured #1-2 with RigMark's own prompts, nonces and
validators (T=0, effort low, 4096 tokens), 60/60 completion gates valid. Per-triplet B - mean(A1, A2) of block
medians: prose +0.034 tok/s, +0.13 % (95 % CI -0.48 to +0.75 %; block medians A 25.07, B 25.24); structured
+0.508 tok/s, +1.43 % (95 % CI +0.08 to +2.77 %; A 35.82, B 36.20). Prompt-matched per-run prose differences
agree (+0.093 tok/s, CI -0.196 to +0.381, n=12). Mean prose length and TTFT did not differ (968 vs 973 tokens,
0.642 vs 0.637 s). Five structured runs (four A, one B) produced a 687-token instead of a 437-token answer; on the
437-token runs alone B was +1.1 %. Code was not run in that window. The prose upper bound is below the +1.36 %
that the c1 cycle A/B implies, so prose is reported as no regression, not as a gain. The previous default's own
arm A gave 25.07 in this boot, also below its 2026-10-01 row.

After this result the release rule was changed (owner decision, 2026-10-02): speed is compared with the current
default in the same boot when a runtime switch exists, otherwise on at least two boots per arm; the qeval band is
unchanged (docs/validation.md, "Release rule"). The README rows were replaced by the 2026-10-02 fresh-clone gate.
The previous default's 2026-10-01 fresh-clone row was RigMark 33.8/25.3/35.9, C4 66.1, prefill 966.6 and sparkDash
29.40/34.75/59.64. Receipts: `docs/results/dirty-l2-release-gate.json`, `dirty-l2-rigmark.json`,
`dirty-l2-rigmark-ab.json`, `dirty-l2-qeval-admission.json`, `dirty-l2-sparkdash.jsonl`.

2026-10-02 (README results layout): the README now shows only the current default, sparkDash first and RigMark
second, with prompt-type rows and concurrency columns. Three comparisons moved here from it. The default's row
next to the Red Hat DSpark K3 alternative (DSpark: pooled three-boot K3/2048 sparkDash reference and the E2b
RigMark receipt, `docs/results/e5-comparison.json`; not a paired experiment; DSpark qeval 72/72/70 in
`docs/results/w4-qeval-admission.json`):

| RigMark 1.0.0, thinking on / effort low | Prose c1 | Code c1 | Structured c1 | Code C4 aggregate | Cold prefill 8K |
|---|---:|---:|---:|---:|---:|
| Native MTP K2 async + short-context DSA + dirty-L2 fix (fresh clone, 2026-10-02) | 24.9 | 34.4 | 36.2 | 66.4 | 954.4 |
| Red Hat DSpark K3 sync | 23.1 | 35.7 | 42.1 | 63.5 | 950.0 |

| sparkDash, thinking off, 256 tokens | Prose c1 | Code c1 | Prose C4 aggregate |
|---|---:|---:|---:|
| Native MTP K2 async + short-context DSA + dirty-L2 fix (5/5/3 runs) | 30.71 | 34.03 | 63.40 |
| Red Hat DSpark K3 sync (15/15/9 runs) | 31.63 | 39.66 | 62.22 |

MTP is the default for thinking/prose workloads; DSpark K3 was faster on short code and structured work. The
same-boot RigMark A/B of the dirty-L2 fix against the previous default (four A1/B/A2 triplets, B - mean(A1, A2),
95 % CI; `docs/results/dirty-l2-rigmark-ab.json`), described in the dirty-L2 entries above:

| Same-boot A/B, dirty-L2 fix vs previous default | Prose c1 | Structured c1 |
|---|---|---|
| RigMark 1.0.0 | no regression: +0.13 % [-0.48, +0.75] | +1.43 % [+0.08, +2.77] |

The note that RigMark c1 varies between runs within one boot (prose 24.1-25.9 tok/s in the 2026-10-02 gate), so
rows from different boots do not measure a change, moved here with it; the README keeps the per-workload run range.

2026-10-02 (README results boot): a second fresh clone of `main` (`31b5412`, release commit `8ccac85` plus
documentation) was booted with `./start.sh serve` to fill every README cell from one boot; its launch vector was
identical to the release-gate boot's after container-name and runtime-path normalisation. Health 200 after 470.3 s,
rank-0 minimum 9.67 GiB in the capture window, all ranks at or above 9.01 GiB for 60 s after warm-up and at or
above 8.65 GiB through the benchmarks; serving and arming checks passed on all four ranks; qeval was not repeated.
RigMark 1.0.0 thinking-on code/prose/structured 33.42/24.91/35.36 (15/15 completion gates; prose runs
24.11-25.68), code C1/C2/C4 aggregate 30.59/46.54/68.37, cold prefill 8K 950.9 tok/s; these replace the README
RigMark rows of the release-gate boot (34.41/24.93/36.18, C4 66.45, prefill 954.4), and the differences are within
the two boots' run ranges. The full sparkDash sweep (prose/code/structured/json at c1/c2/c4/c8 and cold prefill
4k-32k) is in the README. Receipts: `docs/results/readme-1002-boot.json`, `readme-1002-rigmark.json`,
`readme-1002-sparkdash.jsonl`.

2026-10-02 (stack candidate, staged levers): the stack release candidate adds levers that ship switched off until
each one has been measured on the fleet against the released configuration.
- c4 later-MTP graph (`RECIPE_C4_PASS2_GRAPH`), awaiting measurement. In the released capture list `[1, 3, 6, 12]`
  the later native MTP pass (one token per request) can only use M1 and M3, so at c4 it runs eager: 94 launches
  instead of 24. In profiled c4 cycles that pass spanned 6.24-6.27 ms per rank; one rank's host lag (1.70 ms GPU
  idle) held the other three about 1.0 ms each inside the pass's four collectives. A graph should bring the span
  to about 4.8 ms, about 1.45 ms per c4 cycle in the profiled traces and 0.5-1.5 ms (+0.3 to +0.9 % c4
  throughput) without the profiler; c1-c3 are unchanged. The pinned image's graph manager confirms that adding
  width 4 adds exactly the later-pass M4 graph. Earlier experimental layouts already captured
  `[1, 3, 4, 6, 12]`, but none was compared against the released list. Receipt: `docs/results/c4-pass2-graph.json`.
- Glue-lite (`GLM_GLUE_*`), awaiting measurement. Profiled rank-0 target graphs launch about 1,790 kernels of at
  most 11 us per cycle. Three groups can go without changing arithmetic: 75 router-logit FP32 casts (the fused
  top-k kernel widens BF16 itself; the cast mostly overlaps the shared expert, so about 0.05 ms of critical
  path), 75 Marlin lock-workspace zeroings (0.11 ms) and 57 index conversions plus their fills on the skip-top-k
  layers (0.23-0.25 ms): about 0.39 ms per cycle at c1 and c4 together. That is below the c1 A/A offset seen
  between graph-capture instances in earlier in-boot A/Bs (+0.52 ms), so a single-boot A/B may not resolve it.
  The module passed 35 CPU tests and a composed startup in the pinned image; its single-GPU gates and a fleet
  A/B have not run. Receipt: `docs/results/glue-lite-profile.json`.
- 44K context (`RECIPE_MAX_MODEL_LEN=44288`, optional `RECIPE_KV_PIN_L1=1`), awaiting measurement. A memory
  ledger of the released layout found the 2 GiB target pool already holds 44,352 tokens; with vLLM's null
  block, 44,288 is the largest context it serves. Rank 0 sat at 9.03 GiB after a full 32K warm-up, 0.53 GiB
  over the 8.5 GiB admission rule. The extra context should cost about 0.09 GiB of long-prefill working
  memory on rank 0 (about 8 KiB per token of maximum context, measured only up to 32K); the L1 pin (821 blocks,
  2.369 GiB) uses another 0.37 GiB of that margin. The next step is one boot with a 44K warm-up prefill that
  records the rank-0 low point. Receipt: `docs/results/context-44k.json`.
- NCCL without LL128 (`RECIPE_NCCL_NO_LL128`), awaiting measurement. The same ledger counted 0.58 GiB of pinned
  NCCL host buffers per rank: 96 connections of 6.19 MiB, 4.69 MiB of each for the LL128 protocol. Excluding
  LL128 should return about 0.44 GiB on every rank. Decode all-reduces go over the RoCE one-shot path; the
  NCCL operations that remain must be shown to give the same outputs and no slower cycles. There is no runtime
  switch, so the comparison needs at least two boots per arm. Receipt: `docs/results/nccl-no-ll128.json`.
- Native MTP K-stop (`GLM_MTP_KSTOP`), awaiting the fleet screen and qualification. Offline, on saved
  teacher-forced native MTP proposals, a uniform cumulative stop at tau 0.74 projected +2.6 % (hard), +2.8 %
  (easy), +7.6 % (thinking) and +6.0 % (code) at c1, from assumed costs (K2 79.6 ms, K3 91.2 ms, 11.75 ms
  per skipped position); tau should be refit to the measured skip cost. On the fleet, the guard canary passed
  five times (symmetric refusal, dead-rank self-abort, recovery; about 96 us per guard), and the native K2
  control layout booted, warmed and was admitted (native c1 cycle 80-81 ms). Its correctness gate passed once
  and failed twice on run-to-run numerics of the unchanged native arm, so the campaign moved to a screen mode
  where only hard safety checks block. The K3-stop layout itself has not booted, so there is no speed, memory
  or quality result. Memory: only a one-slot screen with the KV pool cut to 544 blocks (context at most 34,752)
  was admitted, at 8.78 GiB [8.68, 8.85] on rank 0; at the full 2 GiB pool the prediction is 8.35 [8.25, 8.42],
  under the 8.5 GiB pre-boot rule. The switch first launched capture sizes 1..16 at the full pool (28 graph
  descriptors, predicted 7.82 GiB, under the launcher's 8 GiB floor). It now launches four slots with capture
  sizes [1, 4, 16] (14 descriptors, every c1 shape and c4 q4 on a graph) at 544 blocks and 32,768 context, the
  same 8.78 GiB prediction as the admitted screen; the full pool with 44,224 context is allowed only with NCCL
  without LL128 (8.70 GiB predicted, unmeasured). In batches of two to four requests whose drafts differ in
  length the target pass runs eager; `GLM_MTP_KSTOP_UNIFORM_BATCH=1` (default off) drafts all three tokens in
  such batches so they replay graphs. A c4 timing screen with four distinct prompts decides between the two.
  Receipts: `docs/results/kstop-profile.json`, `docs/results/kstop-memory-admission.json`.

## README results of the native MTP K2 default (main `983ac38`)

The default profile (`profiles/current.env`), measured on 2026-10-02 from a fresh clone of `main` at `31b5412`
(release commit `8ccac85` plus documentation only) booted with `./start.sh serve` as documented below, on four DGX
Spark with the existing 2200 MHz GPU cap (the recipe does not change clocks). Every number below comes from that
one boot ([boot receipt](results/readme-1002-boot.json)). Its RigMark rows replace those of the `8ccac85`
release-gate boot earlier the same day (prose 24.9, code 34.4, structured 36.2; see [docs/history.md](history.md)),
so that sparkDash and RigMark come from the same boot.

#### sparkDash, thinking off

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| prose | **30.0** | 42.4 (22.2) | 61.1 (15.9) | 58.9 (15.4) |
| code | 34.1 | 49.3 (24.7) | 65.0 (16.7) | 62.2 (16.4) |
| structured | 36.6 | 56.9 (28.5) | 91.8 (23.2) | 89.3 (23.1) |
| json | 35.5 | 48.7 (25.6) | 76.9 (19.6) | 67.5 (17.9) |

**Prefill, cold, tok/s**

| 4k | 8k | 16k | 32k |
|---:|---:|---:|---:|
| 1000 | 942 | 890 | 839 |

The 32k cell uses a 32,256-token prompt: the 32,768-token context cannot hold a 32,768-token prompt plus output.

#### RigMark 1.0.0, thinking on, effort low

**Decode, c1, tok/s**

| workload | median | range |
|---|---:|---:|
| prose | **24.9** | 24.1-25.7 |
| code | 33.4 | 32.9-33.8 |
| structured | 35.4 | 34.6-36.6 |

**Concurrency, code workload, aggregate end-to-end tok/s (per-stream decode in brackets)**

| workload | C1 | C2 | C4 |
|---|---:|---:|---:|
| code | 30.6 (32.6) | 46.5 (25.5) | 68.4 (18.8) |

**Prefill 8K:** 951 tok/s cold (warm replay 843; prefix caching is off).

**How these were measured.** sparkDash 1.8.8 DecodeBench via [`bench/sparkdash.py`](../bench/sparkdash.py) `full`:
256 new tokens, temperature 0, thinking off, idle endpoint; one discarded c1 warm-up per prompt type and one
discarded prose c4 warm-up; each cell is the median of 5 runs at c1, 3 at c2 and c4 and 2 at c8 (aggregate and
per-stream medians are taken separately); decode excludes prefill and first-token latency. At c > 1 sparkDash
sends the same prompt to every stream, and at c8 four streams queue behind the server's four slots. The prefill
table is sparkDash's prefill bench: prompt tokens divided by time to first token, one discarded 4k warm-up, median
of three rounds per length; prefix caching is off, so every prompt is cold. RigMark 1.0.0 (pinned
`d8353e93b274`): 4096-token budget, five decode runs per workload with all 15 completion gates passed (range =
slowest to fastest run); the concurrency rows are RigMark's separate 256-token code workload, three rounds each,
timed end to end, so they are not comparable with the sparkDash cells; prefill is effective input throughput to
the first token, median of three 8192-token prompts. Raw output:
[sparkDash](results/readme-1002-sparkdash.jsonl), [RigMark](results/readme-1002-rigmark.json).

The selectable Red Hat DSpark K3 profile (`RECIPE_PROFILE=dspark-k3`) was not re-measured for this release; its
last numbers and the comparison that made native MTP the default are in [docs/history.md](history.md), as are
the earlier releases and same-boot A/B checks.

#### Serving and quality

The default is native MTP K2 async with the short-context DSA shortcut and the dirty-L2 fix on `GLM-5.3`,
loopback port 8095, 32,768-token context and four slots. On the fresh-clone boot qeval scored **73/72/73**, mean
**72.67/75**: **PASS**, zero truncation and no new recurring failures ([quality receipt](results/dirty-l2-qeval-admission.json)).
Health 200, an exact `OK` reply, four running ranks without OOM or restarts, the shortcut and the discard armed and
dispatched on all ranks, and all-rank MemAvailable at least 9.03 GiB for 60 s after warm-up (8.95 GiB minimum over
the whole gate) are in the [gate receipt](results/dirty-l2-release-gate.json). The results boot above (same launch vector
after container-name and runtime-path normalisation; qeval not repeated) passed the same serving and arming checks,
held all ranks at or above 9.01 GiB for 60 s after warm-up and stayed at or above 8.65 GiB (rank 0) through the
benchmarks ([boot receipt](results/readme-1002-boot.json)).

The operational screen uses the checker-matched mean threshold 70.553/75, zero truncation and no new
recurring failure among historically always-passing tasks; there is no per-run 71 veto. It is not a global
intelligence-equivalence proof. The ≥50 tok/s prose-c1 target and multi-day stability remain unqualified.
The fresh-clone launcher uses the native loader: health 200 came 467 s after launch (470 s on the results boot). Preflight hashes all 290
target files on every node (about 20 minutes) unless the node's record of its last full hash still matches every
file's size, times, inode and device and is less than seven days old (`VERIFY_WEIGHTS=full` always rehashes); the
2026-10-02 gate reused those records. The launcher starts from a fresh per-deployment cache built from the pinned
image before the model containers start: one short
`docker run --rm` per node (same image and environment, no GPU or network) compiles FlashInfer's sampling and
batch MLA modules into it, so no nvcc build runs inside the serving process after graph capture. It is not a
cache seed from an earlier boot and ships no binaries.


## Archived CHANGELOG.md before stack-1003

# Changelog

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
the memory admission are listed in [docs/validation.md](validation.md#stack-candidate-1002b-offline-checks).

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


## Archived docs/validation.md before stack-1003

# Validation scope

2026-10-02 stack candidate 1002b (offline): prefix caching, K-stop with uniform batches and shard selection on by
default; see [its offline checks](#stack-candidate-1002b-offline-checks). The paragraph below describes the earlier
`release/stack-1002` candidate, whose levers were all off.

2026-10-02 stack candidate (offline): five staged levers, all off by default, so the default launch is the
released dirty-L2 default's (DRY vectors differ only by the added off-switch keys). CPU suite on macOS, then in the
pinned base image `tests/test_dirty_l2.py`, `tests/test_glue_lite.py` and `tests/test_kstop.py`, and the real
startup chain with `profiles/current.env` in five scenarios: default; glue-lite on; K-stop on (shortcut off);
K-stop and glue-lite on; K-stop with the shortcut on, which must refuse (exit 78)
([receipt](results/stack-1002-cpu-tests.txt)). No lever has GPU or fleet evidence of its own yet; each needs
the comparison described in `docs/history.md` and the release rule below before it is switched on.

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

Owner decision, 2026-10-02. A change merges into `main` when all of the following hold:

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

2026-10-01 E5: the selected fleet default is native MTP K2 async; current exact-snapshot operational
quality and protocol receipts are `results/e5-qeval-admission.json` and `results/e5-comparison.json`.
The older fresh-clone release packet below retains its DSpark K3 runtime expectations; it does not
qualify native MTP. The fresh native launcher has offline source/dry/CPU evidence; cold-cache
fresh-clone fleet qualification and multi-day stability remain unknown.

Offline packaging was performed on macOS on 2026-09-30, Europe/Warsaw. No Spark container or GPU workload
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

All 20 packaged source pins match this extraction. The retained hostcp/inboot trees have a different
`kv_cache_coordinator.py` from the Flash prefix-cache patch and are deliberately refused; this is not a
reason to relax the pin. No source extraction is distributed here.

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
and prepared-image requirements are in [the operator RUN.md](validation-release-RUN.md).
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
A timeout/implementation failure is PARKED-IMPL or HOLD, with raw receipts; no idea is rejected by this
packet. It does not change GPU clocks, networking, model bytes, kernel arithmetic or serving parameters.

Offline verification uses `tests/test_release_gate.py` plus the launcher/recipe/source regressions.
Tests run at `nice -n 10` with numerical libraries capped at two threads. Four network readers are used
only during owned fleet c4 panels/scans. CPU mocks do not exercise CUDA, RDMA, cold boot or fleet stop.
All fresh fleet results remain **NOT_RUN** until the coordinator executes the packet.

Offline verification on 2026-09-30: **74 CPU checks passed** (13 recipe, 9 launcher, 18 release,
13 allocator, 1 transformed-kernel, 2 loader, 18 RoCE), with 20 image source pins and 324 loader
tensors. The complete-suite receipt and final release/recipe reruns are retained in
`docs/results/release-gate-cpu.txt`. The coordinator's complete-window CPU simulation reaches the
32k unsupported-scope HOLD and reference-recovery branch; no real endpoint or fleet stop was exercised.

## Stack candidate 1002b offline checks

Offline on the Mac and Colima on 2026-10-02 (Europe/Warsaw); no fleet contact. Not a qualification: the
release gate of this candidate decides.

- CPU suite in the pinned image `4def0ef6` (`--pull never --network none`, nice 10, empty `docker ps`): source
  pins, recipe, launcher (including prefix-flag placement, the optional MLA-plan-skip keys, the spec-sample
  refusal under K-stop and the dispram hook), persistent cache, SWA pool, short DSA, dirty L2, glue-lite, K-stop,
  MLA plan skip (including the K-stop payload deferral), kernel, fast loader, shard selection, parameter hash,
  spec-sample, RoCE and `scripts/compare_dry.py` all pass. `tests/test_release_gate.py` needs `git`, which the
  image does not have; it passes on the workstation. Log: [stack-1002b-cpu-tests.txt](results/stack-1002b-cpu-tests.txt).
- K-stop real control flow (the campaign's control-flow test: real engine core, async scheduler, transformed
  runner, speculator and MLA planner, warm-up and K-stop initialization on the meta-built full model; tensor math
  mocked; two slots) on this branch's runtime and hook, uniform-batch policy off and on: warm-up clean and every
  serving scenario passes in both; a request without a recorded draft is still refused on every rank; with the
  policy on, a two-request batch makes no stop decision. Receipt: [stack-1002b-kstop-controlflow.json](results/stack-1002b-kstop-controlflow.json).
- Startup chain in the pinned image, six scenarios: the default (K-stop, uniform batches, dirty L2, shard
  selection; shortcut off), the default with the MLA plan skip in `ab` and in `1` mode (both transforms present on
  the SM90 planner), native K2 rollback, native K2 with spec-sample, and K-stop with the shortcut (refused, exit 78).
- Launch vector: `DRY=1 ./start.sh serve` against the per-rank argv of the deployed APC launch
  (`glm53full-apcq-20261002-121736-b5`). After name and runtime-path normalisation, docker flags, image, mounts
  and the prefix-caching flag are identical on all four ranks; the differences are the same on every rank and
  each has a reason in [stack-1002b-dry-vs-b5.json](results/stack-1002b-dry-vs-b5.json): K-stop's KV pin, capture
  sizes and K3; `GLM_INDEXER_SHORTCUT=0`; K-stop, its control file and uniform batches; shard selection;
  `GLM_SPEC_SAMPLE=0`; and the five glue-lite keys of `release/stack-1002` (all off).
- Memory admission: the campaign sim (multi-slot version) admits the default layout on rank 0, plus a prefix-cache
  term from the APC window's on/off memory; the 10 GiB pre-capture check is predicted to clear with the 544-block
  pool. Receipt: [stack-1002b-memory-admission.json](results/stack-1002b-memory-admission.json).


## Archived docs/dry-vs-best.md before stack-1003

# Dry launch versus the selected control

The Docker commands below were rendered locally by `DRY=1 ./start.sh serve`: four short
`docker run --rm` JIT prep commands (same image and environment with PYTHONPATH empty, no GPU
or network; they build FlashInfer's sampling and batch MLA modules into the fresh per-deployment
cache before the model containers start), then the four model commands compared here.
No SSH, container, GPU, cache seed or fleet mutation occurs in dry mode.
`scripts/compare_dry.py` parses each command with shlex and compares every explicit
container environment key and ordered vLLM argument against the saved per-rank
Boot B BEST_FULL_GLM.json vectors (source SHA256 in results/best-launch.json).

Allowed removals: GLM_W4_TAIL, GLM_W6_GRAPH_RECEIPTS, and six profiler-config
arguments. PROFILE=1 belongs to the diagnostic wrapper, not the container env.
The earlier duplicate speculative-config is removed. D2-W4 changes constructor
capacity512->4096 and adds drained serving cap2048 (results/w4-profile.json).
The native MTP delta in results/e2b-profile.json selects method=mtp, K2, TP4 draft,
compressed-tensors, FP8 draft KV, block64, async scheduling and graphs1/3/6/12;
GLM_MTP_FIX=1 and the DSpark SWA/low-memory hooks are disabled. The /draft mount
is omitted. results/short-dsa-profile.json adds GLM_INDEXER_SHORTCUT=1 and
results/dirty-l2-profile.json adds GLM_DIRTY_L2=discard. results/boot-fast-profile.json adds GLM_MTP_ONLY_LOAD=1 and
GLM_TARGET_SKIP_MTP=1 (native MTP shard selection);
results/glue-lite-profile.json adds the five glue-lite keys with every switch off and
results/kstop-profile.json adds GLM_MTP_KSTOP, its control path and GLM_MTP_KSTOP_UNIFORM_BATCH.
results/stack-1002b-profile.json then turns prefix caching on (one flag token), turns K-stop and its
uniform-batch policy on with the launcher's K3 / [1, 4, 16] / 544-block layout, and sets
GLM_INDEXER_SHORTCUT=0. Greedy drafting and standard rejection remain explicit. The DSpark
alternative preserves the historical K3 vector separately.

Docker's inherited CUDA/base-image environment is not a launcher override and is
outside this comparison. Image IDs differ across the four saved builds; each is
recorded in results/best-launch.json. The recipe rebuild uses the same pinned
base and RoCE source and requires its own image-source and fleet validation.
Host source paths and unique names differ; /model, /draft, /overlay and /cache
container paths retain their meanings. The three startup entry files contain
only used hooks. Inactive diagnostic branches were removed from the MLA adapter;
the split32 and larger-row unsplit kernel bodies preserve deployed arithmetic.
The pre-capture 10 GiB floor remains; per-load phase logging is omitted.

## Environment / argument diff

```diff
# Empty: all 4 ranks match after the stated removals.
```

## Rendered commands

```bash
# K-stop layout: K3, capture sizes [1, 4, 16], 544 KV blocks, max_model_len 32768, NCCL LL128 on, uniform batch K on; sim-admitted, not booted
# jit-prep rank=0 host=NODE_1
docker run --rm --name glm53full-dry-verification-jitprep-r0 --network none --memory 16g --memory-swap 16g -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH= -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_1 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache --entrypoint timeout glm53-roce:v11-b58f34ea -k 10 600 python3 -c 'import torch; from flashinfer.jit.attention import gen_batch_mla_module; from flashinfer.jit.sampling import gen_sampling_module; specs = [gen_sampling_module(), gen_batch_mla_module('"'"'fa2'"'"', torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 64, False)]; assert [s.name for s in specs] == ['"'"'sampling'"'"', '"'"'batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64_profiler_False'"'"'], [s.name for s in specs]; [s.build(need_lock=True) for s in specs]; assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]; print(chr(10).join('"'"'JIT PREP %s %s'"'"' % (s.name, s.get_library_path()) for s in specs), flush=True)'
# jit-prep rank=1 host=NODE_2
docker run --rm --name glm53full-dry-verification-jitprep-r1 --network none --memory 16g --memory-swap 16g -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH= -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_2 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache --entrypoint timeout glm53-roce:v11-b58f34ea -k 10 600 python3 -c 'import torch; from flashinfer.jit.attention import gen_batch_mla_module; from flashinfer.jit.sampling import gen_sampling_module; specs = [gen_sampling_module(), gen_batch_mla_module('"'"'fa2'"'"', torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 64, False)]; assert [s.name for s in specs] == ['"'"'sampling'"'"', '"'"'batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64_profiler_False'"'"'], [s.name for s in specs]; [s.build(need_lock=True) for s in specs]; assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]; print(chr(10).join('"'"'JIT PREP %s %s'"'"' % (s.name, s.get_library_path()) for s in specs), flush=True)'
# jit-prep rank=2 host=NODE_3
docker run --rm --name glm53full-dry-verification-jitprep-r2 --network none --memory 16g --memory-swap 16g -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH= -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_3 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache --entrypoint timeout glm53-roce:v11-b58f34ea -k 10 600 python3 -c 'import torch; from flashinfer.jit.attention import gen_batch_mla_module; from flashinfer.jit.sampling import gen_sampling_module; specs = [gen_sampling_module(), gen_batch_mla_module('"'"'fa2'"'"', torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 64, False)]; assert [s.name for s in specs] == ['"'"'sampling'"'"', '"'"'batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64_profiler_False'"'"'], [s.name for s in specs]; [s.build(need_lock=True) for s in specs]; assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]; print(chr(10).join('"'"'JIT PREP %s %s'"'"' % (s.name, s.get_library_path()) for s in specs), flush=True)'
# jit-prep rank=3 host=NODE_4
docker run --rm --name glm53full-dry-verification-jitprep-r3 --network none --memory 16g --memory-swap 16g -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH= -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_4 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache --entrypoint timeout glm53-roce:v11-b58f34ea -k 10 600 python3 -c 'import torch; from flashinfer.jit.attention import gen_batch_mla_module; from flashinfer.jit.sampling import gen_sampling_module; specs = [gen_sampling_module(), gen_batch_mla_module('"'"'fa2'"'"', torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 512, 64, False)]; assert [s.name for s in specs] == ['"'"'sampling'"'"', '"'"'batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64_profiler_False'"'"'], [s.name for s in specs]; [s.build(need_lock=True) for s in specs]; assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]; print(chr(10).join('"'"'JIT PREP %s %s'"'"' % (s.name, s.get_library_path()) for s in specs), flush=True)'
# rank=3 host=NODE_4
docker run -d --restart no --name glm53full-dry-verification-r3 --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 --memory 112g --memory-swap 112g --entrypoint vllm -v /srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix:/model:ro -v /srv/glm/glm53-full-recipe:/overlay:ro -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH=/overlay/bringup:/overlay/overlay -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_4 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache glm53-roce:v11-b58f34ea serve /model --served-model-name GLM-5.3 --dtype bfloat16 --tensor-parallel-size 4 --nnodes 4 --node-rank 3 --master-addr DATA_ADDRESS_1 --master-port 29679 --distributed-executor-backend mp --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1685372928 --gpu-memory-utilization 0.80 --max-num-seqs 4 --max-num-batched-tokens 4096 --moe-backend marlin --enable-chunked-prefill --no-enable-flashinfer-autotune --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja --default-chat-template-kwargs '{"reasoning_effort":"high"}' --host localhost --port 8095 --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4, 16], "max_cudagraph_capture_size": 16}' --headless --enable-prefix-caching --attention-backend FLASHINFER_MLA_SPARSE_SM90 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "kv_cache_dtype": "fp8_e4m3", "draft_sample_method": "greedy", "rejection_sample_method": "standard", "attention_backend": "FLASHINFER_MLA_SPARSE_SM90", "quantization": "compressed-tensors"}' --block-size 64 --async-scheduling
# rank=2 host=NODE_3
docker run -d --restart no --name glm53full-dry-verification-r2 --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 --memory 112g --memory-swap 112g --entrypoint vllm -v /srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix:/model:ro -v /srv/glm/glm53-full-recipe:/overlay:ro -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH=/overlay/bringup:/overlay/overlay -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_3 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache glm53-roce:v11-b58f34ea serve /model --served-model-name GLM-5.3 --dtype bfloat16 --tensor-parallel-size 4 --nnodes 4 --node-rank 2 --master-addr DATA_ADDRESS_1 --master-port 29679 --distributed-executor-backend mp --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1685372928 --gpu-memory-utilization 0.80 --max-num-seqs 4 --max-num-batched-tokens 4096 --moe-backend marlin --enable-chunked-prefill --no-enable-flashinfer-autotune --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja --default-chat-template-kwargs '{"reasoning_effort":"high"}' --host localhost --port 8095 --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4, 16], "max_cudagraph_capture_size": 16}' --headless --enable-prefix-caching --attention-backend FLASHINFER_MLA_SPARSE_SM90 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "kv_cache_dtype": "fp8_e4m3", "draft_sample_method": "greedy", "rejection_sample_method": "standard", "attention_backend": "FLASHINFER_MLA_SPARSE_SM90", "quantization": "compressed-tensors"}' --block-size 64 --async-scheduling
# rank=1 host=NODE_2
docker run -d --restart no --name glm53full-dry-verification-r1 --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 --memory 112g --memory-swap 112g --entrypoint vllm -v /srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix:/model:ro -v /srv/glm/glm53-full-recipe:/overlay:ro -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH=/overlay/bringup:/overlay/overlay -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_2 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache glm53-roce:v11-b58f34ea serve /model --served-model-name GLM-5.3 --dtype bfloat16 --tensor-parallel-size 4 --nnodes 4 --node-rank 1 --master-addr DATA_ADDRESS_1 --master-port 29679 --distributed-executor-backend mp --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1685372928 --gpu-memory-utilization 0.80 --max-num-seqs 4 --max-num-batched-tokens 4096 --moe-backend marlin --enable-chunked-prefill --no-enable-flashinfer-autotune --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja --default-chat-template-kwargs '{"reasoning_effort":"high"}' --host localhost --port 8095 --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4, 16], "max_cudagraph_capture_size": 16}' --headless --enable-prefix-caching --attention-backend FLASHINFER_MLA_SPARSE_SM90 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "kv_cache_dtype": "fp8_e4m3", "draft_sample_method": "greedy", "rejection_sample_method": "standard", "attention_backend": "FLASHINFER_MLA_SPARSE_SM90", "quantization": "compressed-tensors"}' --block-size 64 --async-scheduling
# rank=0 host=NODE_1
docker run -d --restart no --name glm53full-dry-verification-r0 --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 --memory 112g --memory-swap 112g --entrypoint vllm -v /srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix:/model:ro -v /srv/glm/glm53-full-recipe:/overlay:ro -v /srv/glm/glm53-full-recipe/cache:/cache -v /srv/glm/nccl-2.30.7:/opt/nccl:ro -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e CUDA_CACHE_PATH=/cache/nv -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/fi -e GLM_DIRTY_L2=discard -e GLM_DRAFT_LOWMEM=0 -e GLM_DSA_DRAFT_FOLD=0 -e GLM_DSA_SWA_POOL=0 -e GLM_FAST_LOAD=1 -e GLM_FAST_LOAD_AHEAD_MB=256 -e GLM_FAST_LOAD_DROP_CACHE=1 -e GLM_FAST_LOAD_SLAB_MB=64 -e GLM_FAST_LOAD_THREADS=4 -e GLM_FAST_LOAD_VERIFY=2 -e GLM_FLASH_SITECUSTOMIZE=/overlay/swa-pool/sitecustomize.py -e GLM_FULL_MLA=triton -e GLM_GLUE_DSA_IDX_CACHE=0 -e GLM_GLUE_IDX_EXPECT=57 -e GLM_GLUE_MOE_WS=0 -e GLM_GLUE_ROUTER_BF16=0 -e GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1 -e GLM_INDEXER_SHORTCUT=0 -e GLM_MAMBA_ALIGN_FIX=0 -e GLM_MLA_SPLIT_K=32 -e GLM_MLA_SPLIT_MAX_ROWS=36 -e GLM_MTP_FIX=1 -e GLM_MTP_KSTOP=1 -e GLM_MTP_KSTOP_CONTROL=/overlay/kstop/control.json -e GLM_MTP_KSTOP_UNIFORM_BATCH=1 -e GLM_MTP_ONLY_LOAD=1 -e GLM_ROCE_ALLREDUCE=1 -e GLM_SPEC_SAMPLE=0 -e GLM_TARGET_SKIP_MTP=1 -e GLM_TRITON_MLA_PREFILL=0 -e GLM_W2_PREFILL_CHUNK=2048 -e GLM_W2_PREFILL_CONTROL=/cache/d2w2-prefill-control.json -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e HF_HOME=/cache/hf -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -e HF_HUB_OFFLINE=1 -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e NCCL_BUFFSIZE=1048576 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_MAX_NCHANNELS=8 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME==enp1s0f0np0 -e OMP_NUM_THREADS=1 -e PYTHONPATH=/overlay/bringup:/overlay/overlay -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCH_CUDA_ARCH_LIST=12.1a -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP=DATA_ADDRESS_1 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e XDG_CACHE_HOME=/cache glm53-roce:v11-b58f34ea serve /model --served-model-name GLM-5.3 --dtype bfloat16 --tensor-parallel-size 4 --nnodes 4 --node-rank 0 --master-addr DATA_ADDRESS_1 --master-port 29679 --distributed-executor-backend mp --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1685372928 --gpu-memory-utilization 0.80 --max-num-seqs 4 --max-num-batched-tokens 4096 --moe-backend marlin --enable-chunked-prefill --no-enable-flashinfer-autotune --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja --default-chat-template-kwargs '{"reasoning_effort":"high"}' --host localhost --port 8095 --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4, 16], "max_cudagraph_capture_size": 16}' --enable-prefix-caching --attention-backend FLASHINFER_MLA_SPARSE_SM90 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "kv_cache_dtype": "fp8_e4m3", "draft_sample_method": "greedy", "rejection_sample_method": "standard", "attention_backend": "FLASHINFER_MLA_SPARSE_SM90", "quantization": "compressed-tensors"}' --block-size 64 --async-scheduling
```
