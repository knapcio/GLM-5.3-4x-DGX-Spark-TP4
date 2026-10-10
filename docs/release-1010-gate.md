# Frozen full-model release gate and README completion

## Completed reduced gate

The exact gen-12 configuration passed the owner-approved reduced gate on October 10, 2026 and was handed to
`glm-serving-watch`. The reduced scope ran fresh boot/receipts, T=0, full sparkDash, RigMark, memory stress,
15-minute soak, one 128K needle, collection and handover/verify. It carried hashed quality-panel, 16K/128K needle
and glue exactness evidence. It did not run qeval x3 or exact-config 16K/250K needles. See `GATE-RESULT.md` and
`docs/results/release-1010-summary.json`; the full procedure below remains the stricter reference protocol.

Offline preparation does not run this gate. Pin the candidate's commit, effective `.env` selectors,
image/source identities, target checkpoint, sidecars, NCCL and benchmark versions before boot.
Use the prepared release commit without later experiment merges. Verify the portable serving-source manifest
`docs/results/release-1010-source.sha256` from the repository root. Record DRY launch hashes for all ranks
and keep config/source hashes unchanged throughout measurement. Private machine addresses and receipts
stay outside the public snapshot; publish only a portable summary.

The serving source baseline is `release/glm53-1009` at `36373c6`, the dev det-align release corresponding
to public release `988413f`. The latter object is not present in the local dev object database.
This candidate adds the FP8 eh_proj implementation from `81e8863`; det-align retains the `6da108c` lineage.
Both glue-lite F1 and F2 are on. Required selectors are:

| Selector | Frozen value |
|---|---|
| `GLM_MOE_DET_ALIGN` | `1` |
| `GLM_DRAFT_HEAD`, `GLM_DRAFT_HEAD_INIT` | `nvfp4`, `1` |
| `GLM_DRAFT_EHPROJ`, `GLM_DRAFT_EHPROJ_INIT` | `fp8`, `1` |
| `GLM_GLUE_ROUTER_BF16`, `GLM_GLUE_MOE_WS`, `GLM_GLUE_DSA_IDX_CACHE` | `1`, `1`, `0` |
| `GLM_GLUE_ROUTER_EXPECT`, `GLM_GLUE_IDX_EXPECT` | `target:75,mtp:1`, `0` (F3 off) |
| `GLM_MTP_ROWSELECT` | `0` |
| `GLM_MTP_KSTOP`, uniform batch, capture layout | `1`, `k2`, `reuse` |
| `GLM_DECODE_FAIR`, chunk, decode steps | `1`, `4096`, `40` |
| MLA split / dispatch cap | `32` / `36` |
| Cache trim / async v2 | absent; no implementation or selectors included |
| KV format / context / ordinary head | `fp4x` / `262144` / `6318718976` bytes |
| Live / release stress floor | `4.5` / `4.5` GiB on every rank |
| Capture / admission / pre-capture | `6.0` / `6.5` for 60 s / `7.5` GiB |

knapcio selected the **4.5 GiB release stress criterion on October 10, 2026**, matching the live floor.
Only that acceptance criterion changes. Keep swap, request failures, preemptions, capture and admission
requirements unchanged. The ordinary public K3/K2 policy stays in place; no experimental confidence/width
controls, alternate split kernels or rotary compression are included.

### Public profile versus gate harness environment

`tests/fixtures/release-1010-gate-env.json` pins the gate's `config.env` from
`release-run.json` (source SHA256
`f9bd4b1e31d6dda93702842125374ab872d3b5707e93101c2be99f36b97129e3`).
`tests/test_public_gate_defaults.py` runs the real `DRY=1 ./start.sh serve` entry point
with only host/site paths and addresses, and compares every JIT and serving vector.
It substitutes only the presence check for the separately built copy-guard library.
For decode fairness it imports the overlay's actual `settings`, `boot_decode_steps` and
`control_policy` resolvers. Public vectors with the sidecar path empty or absent resolve to
enabled, chunk `4096`, decode steps `40`, equal to the gate vector with `/cache/decode-fair.json`
fed from `tests/fixtures/release-1010-decode-fair.json` (schema 2, `4096/40`, ON).
The historical launch comparison in `scripts/compare_dry.py` explicitly retains its
old index expectation of 57; it does not test today's public defaults.

These gate harness keys differ literally from the public vectors. They are excluded
from literal env equality, with boot equivalents checked separately; they are not
extra public feature defaults or instructions to enable the trial harness:

| Gate key / value | Public default and reason |
|---|---|
| `GLM_PARAM_HASH=1` | Absent; gate diagnostic hashing only. |
| `VLLM_SERVER_DEV_MODE=1` | Also `1`, enabled by draft INIT; excluded as a diagnostic API. |
| `RECIPE_LIVE_FLOOR_GIB=4.5` | Host-only floor, never forwarded to containers. |
| `GATE_CTN` | Host-only gate identity; absent from this pinned env. |
| `GLM_DECODE_FAIR_CONTROL=/cache/decode-fair.json` | Empty; optional runtime sidecar, same boot policy `4096/40` ([time-slicing](time-slicing.md)). |
| `GLM_SKIP_MLA_PLAN=ab`, `GLM_SKIP_MLA_PLAN_AB_INIT=0` | Absent; skip defaults off, equivalent to the gate's initial disabled A/B state ([runtime](runtime.md#optional-switches)). |
| `GLM_PREFILL_SOLO_CAPACITY=4096`, `GLM_PREFILL_SOLO_CHUNK=0` | Absent; gate launcher validation-only keys, not forwarded there either. Public constructor capacity is 4096 with ordinary adaptive prefill. |
| `GLM_KSTOP_ASYNC_PAYLOAD=0` | Absent; async v2 implementation is excluded, so the ordinary payload path remains. |
| `GLM_MTP_TAU_AB=1`, `GLM_MTP_TAU_INIT=0.74` | Absent; public K-stop reads `overlay/kstop/control.json` with tau `0.74`, without the trial harness ([runtime](runtime.md)). |
| `GLM_MTP_K4_CAPTURE=0`, `GLM_MTP_K4_GATE=0` | Absent; K4 implementation is excluded. Public maximum width is K3, with uniform K2 batches. |

These equivalences describe the frozen boot only. The public export does not reproduce
the gate harness's runtime trial APIs. The index expectation was the sole differing
glue default: it is now `0`, matching the gate with F3 disabled.

Boot with management recovery access and a single measurement process. All ranks must report det-align,
draft head and FP8 eh_proj ready, and expected glue arming counts; fallback is a failed candidate gate.
Record the target fixed-hidden head's bit-exact check and projection INIT qualification per rank.
Run shared-pool c4 stress (four distinct 65,024-token prompts plus 1024 outputs) and single maximum
stress (261,120+1024), retaining APC, all-rank observed minima, zero swap and zero preemptions.
The probe's default stress floor is 4.5; pass it explicitly when invoking outside the launcher environment.

Then run needle 16K/128K/250K twice, original 75-task qeval c1 x3 (threshold 70.553 mean passed tasks,
primary scores, zero truncations/failures), sequential temperature-0 repeats, prefix first/repeat,
mixed-load soak, and RigMark 1.0.0 with thinking on/effort low. Preserve every receipt and report failed
checks; a timeout or incomplete check is not a pass. Follow the established recovery procedure on failure.

For sparkDash, use one full thinking-off sweep with 256 output tokens and temperature 0. Keep every scored
run (c1 five, c2/c4 three, c8 two), discard only explicit warmups, and record failures. Cold prefill is the
median of three at 4K/8K/16K/32K. Do not select the best sweep or mix different configurations or boots.

The collector is `bench/sparkdash.py`. Run it only during the live gate, then use the offline formatter
`scripts/release_table.py` to produce replacement Markdown from its JSONL receipt. The formatter does
not contact the endpoint and refuses incomplete or failed sweeps. Review the receipt's commit/config
binding before replacing the 24 numeric placeholders. Fill the quality table's 10 cells from the other
gate receipts, and replace `GATE-RESULT.md` with the exact measured qualification and portable receipt path.

```bash
python3 scripts/release_1006_probe.py --help
python3 bench/sparkdash.py full 262144 > "$RECEIPTS/sparkdash.jsonl"
python3 scripts/release_table.py "$RECEIPTS/sparkdash.jsonl" > "$RECEIPTS/readme-tables.md"
```

Before export, rerun `scripts/public_export_audit.py`, the pattern review and `tests/run_cpu_tests.sh`.
Use an existing offline Python environment through `GLM_CPU_PYTHON`, with `GLM_IMAGE_SRC` pointing to
the extracted pinned image source. No downloads or fleet access are required for the CPU suite.

## Recorded reduced gate and cell provenance

Reduced gate **PASS**, boot `glm53full-cand3win7-w4stack-12`, serving `6f5235dc14549677704d10e6564277883686e60d`.
The public profile boot equivalences above bind this serving vector to the export.
Fresh phases: preboot, serve, receipts, smoke, bootmin, mem, initgate, t0, dash, rigmark, memstress, stress, soak, needle, collect, handover, verify.
Qpanel and qeval were carried; the time-slicing scenario and long soak were dropped.
Fresh soak used the reduced duration; fresh needle used only 128K x1.
16K x2 is carried from w4-ehproj-g6. 250K was neither run nor carried.
The qeval regression decisions remain KILL; reduced PASS does not upgrade them.

| README cells | Receipt references | Fields / interpretation |
|---|---|---|
| sparkDash: prose/code/structured/json x c1/c2/c4/c8 (16 cells), cold prefill and TTFT x 4K/8K/16K/32K (8 cells) | `sparkdash.jsonl`, `dash-vs-det-release.json` | Every scored run; `tables()` median/format rules. Comparison columns are not published. |
| Quality admission; qeval status | `qpanel_B`, `qeval_g5`, `qeval_g6`, `release-run.json` | passed/items/truncated; results[].pass/category; decision/release_decision; phases |
| Needle 16K, carried twice | `needles_g6`, `release-run.json` | runs[].grade.score_over_n; phases |
| Needle 128K, one run | `needle/summary.json` | runs; grade.score_over_n/fields; ttft_seconds; verdict.status |
| Needle 250K | `release-run.json`, `COMPLETE.json` | phases; carried.groups (no 250K measurement) |
| c4 shared-pool and single maximum-context stress | `done-stress.json`, `stress/memory-summary.json`, `stress/c4/c4-trace.json`, `stress/single.json`, `stress/rank0-memory.jsonl`, `stress/rank1-memory.jsonl`, `stress/rank2-memory.jsonl`, `stress/rank3-memory.jsonl` | result.geometry/single_prompt/output_tokens/min_GiB; problem; trace[].preemptions; usage; max(SwapTotal-SwapFree) |
| Prefix-cache first/repeat and retained-cache stress | `rigmark-summary.json`, `rigmark-receipt.json`, `stress/c4/apc.json`, `done-memstress.json` | rates.prefill_8k_cold/prefill_8k_warm_replay; hit_delta/query_delta; quiet_min_GiB |
| Mixed-load soak | `done-soak.json`, `soak/RESULT.json` | result.minutes/soak_s/requests/errors/preemptions/mem_drift_GiB |
| Draft-head + eh_proj INIT | `done-initgate1.json`, `init-qual-rank0.jsonl`, `init-qual-rank1.jsonl`, `init-qual-rank2.jsonl`, `init-qual-rank3.jsonl` | ranks[].on/ready/target_bit_exact/n_cases; terms; ehproj_captured.conditions.combined_init_qualified |
| Temperature-0 sequential repeats on this boot | `done-t0.json`, `release-t0/result.json`, `release-t0-comparison.json`, `release-t0-stack2-equality.json`, `glue_stack2` | determinism/native_equal; repeats/reference |
| RigMark 1.0.0, thinking on, effort low | `rigmark-summary.json`, `rigmark-receipt.json` | rates/gates; protocol.version; settings.extra_body |

Local receipt root: `diagnostics/glm53-full-20261009-night/relgate/run-1010d/`. Each short filename above resolves there.
Carried group references resolve to the complete file list below; hashes were verified before rendering.

| Carried group / source run | Recorded interpretation |
|---|---|
| `qpanel_B` | B 110/116; owner selected quality admission; DECISION PASS |
| `qeval_g5` | g5 KILL; g6 INFORMATIONAL with release_decision KILL; carried as diagnosis, not relabeled PASS |
| `qeval_g6` | g5 KILL; g6 INFORMATIONAL with release_decision KILL; carried as diagnosis, not relabeled PASS |
| `needles_g6` | 16K/128K x2 PASS; includes frozen panel and 16K scoring control |
| `sparkdash_g6` | full 262144, duration 1102.344 seconds; repeated fresh on release boot |
| `glue_stack2` | stack-2 B glue-t0: 30/30 hard native ID equality to e7 A-t0; fresh release repeats x3 against B |

| Receipt path | SHA256 |
|---|---|
| `diagnostics/glm53-full-20261009-night/dash-g6/sparkdash.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `diagnostics/glm53-full-20261009-night/dash-g6/sparkdash.jsonl` | `59ee0a94b05fdb9347641cdac52bb339fea6956662ada1118adc0b0cd9ac2450` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g5/qeval-screen/qeval-glm53full-cand3win7-w4ehproj-5-q1.json` | `4324be1b7b3975d01053fddaf4e2b8b90619950cd0273ce6824ecd77ad6a0528` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g5/qeval-screen/raw-responses.jsonl` | `c5d80a2bba3f73a45c4acd0b96e337315ee70b08ea9df26c4fd6900d524398b5` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g5/qeval-screen/run.log` | `8ec4cfaa1077484f711bcf07077c1a1a6ef46d0646ba22c58dd5eb72de158780` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g5/qeval-screen-item-diff.json` | `6aee72c4c327f64dec0ab8fd860b31e180d13828694d911ddb710b0bc8432229` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/panel-128000.json` | `26f99a8ce0503a9f12f363cb35a1bc58a21a352dc493cdb3b227771f8ca7dacb` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/panel-16384.json` | `9834b95db7cb1fd083b753f08e2eae865d75ca47ac3539f354507cc3c85fde1e` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/run-128000-1.json` | `f2c3face1e3341ba5e1cc756c391edfd74db659f79c517cde6bfbc00eb15e856` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/run-128000-2.json` | `40654a481b213dd3d6ef7587db32c2813d37d075840313383b9bd76a4529c743` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/run-16384-1.json` | `bcedeafb141c9d2c84b86f073df89f03e6db06f28b63f5e096e644e66aab12b8` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/run-16384-2.json` | `f16aeecba56033b6eb3676c0408f68ad69ebafeb4b1bd03600d64db23d0abecc` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/self-check.json` | `b6b13f0e5a79ed8d33b829a7cbeaa372cdc0421620bb6ebff7bbe254f73a624b` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/needle/summary.json` | `c0aab54216b442fd0d453466b98acbb297855e583bf364c4798245a8130e688a` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/qeval-screen/qeval-glm53full-cand3win7-w4ehproj-6-q1.json` | `76734cb5890b92ed6be3b86d785f6880a479d379971e313e6661add60834f13a` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/qeval-screen/raw-responses.jsonl` | `c8330a660133a71b9701247dde5e0c1ea98025e2bc31b0eed3b6ee981c87b32a` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/qeval-screen/run.log` | `d3f3d419811c6fc096bbf85c1fb1809c3fb4c5f471193f45c9bfc7db723d8b74` |
| `diagnostics/glm53-full-20261009-night/ehattr/driver/w4-ehproj-g6/qeval-screen-item-diff.json` | `8b49b80562b495d68126628a5f0411772781e69878bea6e21fc9488692fda733` |
| `diagnostics/glm53-full-20261009-night/qpanel/B/RESULT.json` | `71d76bc2429c6da549a1701d6ae015ed91801212d133cededd1673f70b3f2d4f` |
| `diagnostics/glm53-full-20261009-night/qpanel/B/boot-receipt.json` | `17e80e786dd883b356761f83f43057e8eec25bf3d52817613f88139d5c001bc1` |
| `diagnostics/glm53-full-20261009-night/qpanel/B/raw-responses.jsonl` | `e271e8c7bb49635b9d51f6a6dd01baabec76357c64da99ddda3871866786a476` |
| `diagnostics/glm53-full-20261009-night/qpanel/DECISION.json` | `1cb3786777cb99fb8fa37e221b912edb29253cac4865044b2b795ac7888acde6` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/COMPLETE.json` | `043e8eac0b4034436a57e51ac31b7994e78fd524d525b7f3e10cf8c3474d4a73` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/dash-vs-det-release.json` | `d2e3fe974e3aa7e67eb27d7a88c23ce45f1c0921e8e21b7eb0f01e0c3b80a1ae` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/done-initgate1.json` | `81611bd5e9a81ac57b25aaa6d92adb450ff3668082037556eb3025633e8549d1` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/done-memstress.json` | `88eece1bf3b0a95f40e1280423644332185d9e92b618d3a8bf0ff420b713ccc6` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/done-soak.json` | `84f2e64a09fe7c4478b192bfbbd4fc624b1de7488b0fe4cbb74df2247f1c8631` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/done-stress.json` | `907fe06d82d2a7c8777eabc3a4f32428f350dec83af5d119871b0420afbce96b` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/done-t0.json` | `f7298e84e16eeb72cd4ab2612b362077f45044a6ea99216135e5927fb6a40a18` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/init-qual-rank0.jsonl` | `f5c845f5e2ca279cf22d130fbe2f78708956a264fa8ec6d733c682aee7499967` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/init-qual-rank1.jsonl` | `920396c5495cc1570d68d44bac840102d0567ddfdd6b8fdcd0f1e37498eff128` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/init-qual-rank2.jsonl` | `a5b5abfcd81c9d6fbdc6fcbad3849a8189d2f1e6affaf7e04066504202093509` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/init-qual-rank3.jsonl` | `5356e4ab419e24bcb71debf9fc4ff751d085e05e67c55db0437b53a48913e416` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/needle/summary.json` | `ad912a1953b963b947ef9ef3c8758e51de2fb5340b899f34580e57af68b83fdb` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/release-run.json` | `32080199ecb5eb92f5414879165473a9723d5deaf035b9394434cdad11db2503` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/release-t0/result.json` | `32af2d4a22a6f134339d4808a35b4ccb4493c4c4421247a82a31975973866ee1` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/release-t0-comparison.json` | `5d8389bef78e10b6466093bf0cd69739d35b19ef5036036ed633b1cba14d7375` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/release-t0-stack2-equality.json` | `3d121106ee3233bd4155b7a62ad86618e6bb5bae57ecbd664411828dccfa2f6e` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/rigmark-receipt.json` | `9cca73d12ffc335d3de6637d09e79e91462541faebcca488c7590c08f4c3a910` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/rigmark-summary.json` | `c6b3977233c5deb1b0d187d0128b64e8b49c63ec4bddcc1f286bea7a66ea3b41` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/soak/RESULT.json` | `8fe67dca9fb140f80c7f4d3c7777fb00c22607e6a59b350b6cc87772d3547412` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/sparkdash.jsonl` | `4a450b5433abec668f3a093e40e14c35f34e5b006c2743e163765c1d81e52dcb` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/c4/apc.json` | `cccb6b0042d2e29aa1cc846a21d1aa4918153a268e8c3f9fd3026f4d8ef909d0` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/c4/c4-trace.json` | `67b688556295d7a9e42b8626400e9459a2142bd6cd99282aedd392ba93abd0ef` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/memory-summary.json` | `d07adcb86a165a7b15392075fdd2703f33ec2ee5a5f5766296f81188f4ad4cdb` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/rank0-memory.jsonl` | `17b2feb107095536f756001ce5860bb43bd916d336c214447c42ba6a8886e36d` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/rank1-memory.jsonl` | `5f5346610e192df36bd6ca1874efd238b401a7fb053d1d8832078da0e6ce2b76` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/rank2-memory.jsonl` | `ac12d01239fe867b3ebf75e9511b7f3341d0ff71620c13af603cfccfbc833f78` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/rank3-memory.jsonl` | `fa0a6c8ae38b8b126d8f8931fd886e3fbd790a8092644f02650789ea8d375014` |
| `diagnostics/glm53-full-20261009-night/relgate/run-1010d/stress/single.json` | `6d3c9d691deb441c3262ec6ff1786954e56c9b8f64ca10816b5a742297c78d57` |
| `diagnostics/glm53-full-20261009-night/stack/driver/stack-g2/A-t0/result.json` | `772877f9180edcff2ba305f31f73dad8b76ecd72eddf013404ce47a39700a9b2` |
| `diagnostics/glm53-full-20261009-night/stack/driver/stack-g2/decisions/glue.json` | `837a377568694d7e1ea0e150d062e76ff1e1039a868129b3b53d836fad333bc0` |
| `diagnostics/glm53-full-20261009-night/stack/driver/stack-g2/done-verify.json` | `458f93675ef8b6e8ae6bb745d3db7ef99f01ec5db2bcf0638729f7b0aefdc0d3` |
| `diagnostics/glm53-full-20261009-night/stack/driver/stack-g2/glue-t0/result.json` | `bface79b2ebab751291281f47182ec7cefe8ea56d169d4a170734669c95d316d` |
| `diagnostics/glm53-full-20261009-night/stack/driver/stack-g2/stack-boot-receipts.json` | `ccca25444276d4febc7b41e5c67d763a3615bd352363bee44f2dbda9f1173aab` |

Reproduce the README tables and provenance offline:

```bash
python3 scripts/release_table.py "$RUN/sparkdash.jsonl" --gate-run "$RUN" \
  --provenance "$RECEIPTS/provenance.md" > "$RECEIPTS/readme-tables.md"
```

The formatter verifies every carried hash, phase identities, T0 report bindings and the fresh boot.
The 250K cell records absence, rather than a measured score. Qeval x3 has no release-boot receipt;
the carried single-run qeval diagnostics are labelled with their original decisions.
