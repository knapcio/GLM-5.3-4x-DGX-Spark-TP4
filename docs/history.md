# Measurement history

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

2026-09-30 (2048-token chunks): native RedHat K3, split32/RoCE, drained2048 with4096 constructor capacity,2GiB/rank and32K context. All nine long-context panels12/12; strict qeval 72/72/70, mean71.333 against threshold70.553. Current medians across all scored served-config runs: prose-c1 31.02, code-c1 37.37, prose-c4 aggregate62.26/per-stream16.31tok/s; counts5/5/3. Cold16K request-wall input throughput 936.05tok/s, paired gain27.37% vs512 (three triplets). Live KV pool34,304tokens; historical41,728 is not current. Qualified deployment used v5 immutable prepared snapshots with exact readback, health116.44s/admission189.21s; the published native launcher has not been re-timed for this profile. V2/v4 80-round c1 CIs missed0.5ms precision, c4 full-wave harness mixed widths; Split16 partial fixed-width proxy+4.31ms awaits loaded oracle/timing qualification. No new decode kernel was adopted. Complete current receipts: `docs/results/w4-*`.

2026-09-30 (E1 three-boot refresh): same served native K3/2048 configuration, 256 output tokens, thinking off; five prose/code c1 runs and three prose c4 waves per boot, warmups discarded. Per-boot medians:

| Boot | Prose c1 | Code c1 | Prose c4 aggregate |
|---|---:|---:|---:|
| W4 README boot | 31.02 | 37.37 | 62.26 |
| W5 stock boot | 31.63 | 39.91 | 61.69 |
| E1 baseline arm | 31.85 | 39.66 | 62.22 |

Pooled all-scored medians: prose c1 31.63, code c1 39.66, prose c4 aggregate 62.22 tok/s. E1 baseline used native A with candidate banks resident; joined cycle timers were inactive. This refresh does not adopt a decode lever or infer a speed improvement from prose throughput. Receipts: `docs/results/e1-three-boots.json`.

2026-09-30–2026-10-01 (E2b/E4, E5 default selection): native Z.ai MTP K2 async versus Red Hat
DSpark K3 sync, both TP4/split32/RoCE, 2 GiB target KV per rank, 32K context, four slots and 2048
prefill chunks. MTP is selected for thinking/prose use; DSpark remains a selectable alternative.
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

After this result the release rule was changed (2026-10-02): speed is compared with the current
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

2026-10-02 (README results boot): a second fresh clone of this release (same code, documentation-only
differences) was booted with `./start.sh serve` to fill every README cell from one boot; its launch vector was
identical to the release-gate boot's after container-name and runtime-path normalisation. Health 200 after 470.3 s,
rank-0 minimum 9.67 GiB in the capture window, all ranks at or above 9.01 GiB for 60 s after warm-up and at or
above 8.65 GiB through the benchmarks; serving and arming checks passed on all four ranks; qeval was not repeated.
RigMark 1.0.0 thinking-on code/prose/structured 33.42/24.91/35.36 (15/15 completion gates; prose runs
24.11-25.68), code C1/C2/C4 aggregate 30.59/46.54/68.37, cold prefill 8K 950.9 tok/s; these replace the README
RigMark rows of the release-gate boot (34.41/24.93/36.18, C4 66.45, prefill 954.4), and the differences are within
the two boots' run ranges. The full sparkDash sweep (prose/code/structured/json at c1/c2/c4/c8 and cold prefill
4k-32k) is in the README. Receipts: `docs/results/readme-1002-boot.json`, `readme-1002-rigmark.json`,
`readme-1002-sparkdash.jsonl`.

