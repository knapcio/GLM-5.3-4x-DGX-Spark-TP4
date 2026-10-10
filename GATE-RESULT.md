# Release gate result: eh_proj + glue-lite F1/F2 (2026-10-10, gen 12)

**Verdict: PASS under the owner-approved reduced release scope.** The exact configuration is serving as
`glm53full-cand3win7-w4stack-12` under `glm-serving-watch`: deterministic align, FP8 draft `eh_proj`, and glue-lite
F1/F2 on; F3, cache trim, async kstop v2 and K4 off; tau 0.74; decode fairness 4096/N40; 262,144-token context.
Measured private serving-source commit: `6f5235dc14549677704d10e6564277883686e60d`. The public branch is a portable
export whose effective release profile and DRY launch vectors were checked for equivalence; that private object is
not presented as the public branch commit.

The [full gate procedure](docs/release-1010-gate.md) remains the stricter reference protocol. At 15:05/15:07 Europe/Warsaw the owner approved
a reduced gate with fresh boot/receipts, T=0, sparkDash, RigMark, memory stress, 15-minute soak, one 128K needle,
collection, handover and verification. The 116-item quality panel, paired 16K/128K needles and glue exactness were
carried with hashes. qeval x3 and exact-config 16K/250K needles were not run; the carried g6 qeval diagnostic keeps
its original informational/KILL label and is not represented as a PASS.

| Check | Release result |
|---|---|
| Quality admission; qeval status | Carried qpanel/B (glm53full-cand3win7-w4ehproj-6): 110/116, 2 truncations; decision PASS. carried w4-ehproj-g5: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision KILL; carried w4-ehproj-g6: 71/75, primary 51/55, 1 truncation, 4 failed tasks; decision INFORMATIONAL (release_decision KILL). Qeval x3 not run on release boot. |
| Needle 16K, carried twice | Carried w4-ehproj-g6: 8/10, 8/10; not rerun on release boot. |
| Needle 128K, one run | 1 run, 9/10; missed `matching_count`; TTFT 170.86024899967015 s; PASS under the reduced rule. |
| Needle 250K | Not run; no carried 250K receipt in the reduced scope. |
| c4 shared-pool and single maximum-context stress | PASS; c4 65024+1024 each, single 261120+1024; rank 0/1/2/3 minima 5.808849334716797/6.767314910888672/7.456321716308594/7.352359771728516 GiB; swap maxima 0/0/0/0 KiB; preemptions 0.0. |
| Prefix-cache first/repeat and retained-cache stress | 8K cold/replay prefill 904.667/16114.311 tok/s; stress APC hits/queries 64896.0/65024.0 tokens; retained-cache quiet minima 5.89/6.836/7.515/7.398 GiB (ranks 0/1/2/3). |
| Mixed-load soak | PASS; requested 15.0 min, observed 1071.0947126080282 s; 27 requests, 0 errors, 0.0 preemptions; quiet drift -0.022/0.006/-0.024/0.017 GiB (ranks 0/1/2/3). |
| Draft-head + eh_proj INIT | 4/4 ranks ON/ready, combined FP8 eh_proj qualified; 16 cases and 342/342/342/342 rows per rank; 0 failed; target head bit-exact on every rank. |
| Temperature-0 sequential repeats on this boot | 30/30 prompts identical across 3 repeats; 30/30 native-ID equality to carried stack-g2/glue-t0. |
| RigMark 1.0.0, thinking on, effort low | prose 30.051 tok/s; code 42.372 tok/s; structured 47.338 tok/s; code c4 aggregate 67.904 tok/s; 8K cold prefill 904.667 tok/s; all workload gates passed. |

Portable source run: `diagnostics/glm53-full-20261009-night/relgate/run-1010d/`.
`COMPLETE.json` records `all_pass=true` for the reduced scope. Full receipt paths and SHA256s
are in [the gate provenance](docs/release-1010-gate.md#recorded-reduced-gate-and-cell-provenance).
