# Full-GLM fresh-clone gate: operator runbook

**Offline packet only; fleet execution is unverified.** The current fleet owner must schedule an exclusive
idle window and run this packet. The gate never stops someone else's serving containers. Launching it
while another window owns `~/fleet_busy` returns a refusal. Keep management SSH/local-console recovery.

The commands below take less than 15 minutes of operator setup with all images/weights already installed.
The automated measurement window is capped at 90 minutes. Builds and weight downloads are preparation,
not hidden work inside that window. The program defaults to a local plan with no SSH or Docker calls.

```bash
python3 scripts/release_gate.py
# CPU checks; no fleet access. Already extracted v11 source required for the entire suite.
GLM_IMAGE_SRC=/path/to/extracted/image-source tests/run_cpu_tests.sh
```

Prepare two config files **outside** the repository, using `.env.example`. Set the same four hosts,
fabric addresses, model/draft paths, HCA and NCCL binary hash. Set `IMAGES=(sha256:... sha256:...
sha256:... sha256:...)` to the complete locally verified ARM64 image IDs. Tags alone are refused.
Use different config files if the independently staged reference image differs. Never include credentials.
Build the pinned `Dockerfile.roce` beforehand on an idle authorised build host, retaining source-pin
and image-ID receipts. Use `--RUN_TESTS` defaults; do not override the pinned base. A full image
rebuild plus 1.5 TB of checkpoint reads is not promised within the measurement window.

Choose the previously qualified full-GLM reference checkout and commit. Both arms must use identical
checkpoint manifests. An unchanged `HEAD` reference is a **cold-boot reproducibility control**, not an
independent full-precision reference or proof of no quantization loss. No Flash baseline is used.
Keep the laptop awake. In another terminal open a management tunnel to the same head configured in
both config files (replace `rank0` if needed):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18095:127.0.0.1:8095 rank0
```

Ensure sparkDash is available at the specified URL and has no active benchmark. Then, from the committed
candidate repository, run:

```bash
nice -n 10 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  python3 scripts/release_gate.py --execute \
  --config /absolute/path/candidate.env --reference-config /absolute/path/reference.env \
  --reference-repo /absolute/path/qualified-full-glm --reference-rev FULL_COMMIT \
  --candidate-rev HEAD --out /absolute/path/new-release-receipts \
  --base http://127.0.0.1:18095 \
  --dash http://127.0.0.1:5555/api/sparks/rank0/llm \
  --window-minutes 90 --aa-max 0.01 --kl-max 0.01 --min-prose-tps 50
```

Freeze those thresholds before the run. They are deliberately conservative screening limits,
not established full-GLM quality tolerances. Do not raise them after observing a failure. If an A/A
panel is unstable, record HOLD and diagnose repeatability before comparing an optimization.
The first 65 minutes allow measurements; the final 25 minutes are reserved for owned stops and a fresh
reference re-admission. Individual HTTP requests, benchmark jobs, qeval subprocesses and admissions are
bounded. A deadline yields partial receipts and HOLD; it never counts omitted cells as passes.
SSH failure can require manual recovery and is outside any wall-clock success guarantee. Safety cleanup
must complete even if the scheduled window expires. No power operation is automatic.

Read `gate.json`, all phase files and `SERVING.md`. Exit 0 means the bounded requested gate passed;
exit 2 means HOLD/PARKED-IMPL. With the current 32k release profile, the complete 4k–128k scope **must**
return HOLD even if every supported request passes. The 32k prefill cell measures 32,767 input tokens
plus one output token, and records both lengths. The 64k/128k cells are explicit unsupported records;
the program does not silently increase the KV pool or context limit.

The reference stop is verified before candidate boot: exact four container names stopped/preserved,
no compute PIDs, no OOM, owned marker gone. Only the gate's exact launcher PID is interrupted. Failed
stops retain the fleet marker; do not retry a boot or remove the marker without investigating. The
candidate stays guarded if qualified. On HOLD, the previously checked reference is freshly booted with
a different empty runtime/cache; if recovery cannot be safely admitted, the fleet is left stopped and
the error is recorded. A surviving foreground watchdog keeps the fleet lease until a coordinated
handoff. Stop using the fresh checkout path in `SERVING.md`; do not delete remote containers or caches.

Record actual fleet results in the administration workspace's SETUP.md (Current state, Next action,
checklist and Verification log, Europe/Warsaw). No fleet result was measured by this offline packet.
