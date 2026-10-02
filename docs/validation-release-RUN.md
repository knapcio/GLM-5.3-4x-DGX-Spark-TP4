# Running the fresh-clone release gate

The gate needs an idle fleet. It never stops someone else's serving containers: while another
deployment holds the launcher's fleet marker on the head, launching it returns a refusal. Keep a
management SSH or local-console recovery path.

With all images and weights already installed, setup takes less than 15 minutes. The automated
measurement window is capped at 90 minutes. Image builds and weight downloads are preparation and are
not part of that window. Without `--execute` the program prints a local plan and makes no SSH or Docker calls.

```bash
python3 scripts/release_gate.py
# CPU checks; no fleet access. The extracted image source is required for the entire suite.
GLM_IMAGE_SRC=/path/to/extracted/image-source tests/run_cpu_tests.sh
```

Prepare two config files **outside** the repository, using `.env.example`. Set the same four hosts,
fabric addresses, model/draft paths, HCA and NCCL binary hash. Set `IMAGES=(sha256:... sha256:...
sha256:... sha256:...)` to the complete locally verified ARM64 image IDs. Tags alone are refused.
Use different config files if the independently staged reference image differs. Never include credentials.
Build the pinned `Dockerfile.roce` beforehand on an idle build host, keeping the source-pin
and image-ID records. Use the `--RUN_TESTS` defaults; do not override the pinned base. A full image
rebuild plus 1.5 TB of checkpoint reads does not fit within the measurement window.

Choose a previously qualified full-GLM reference checkout and commit. Both arms must use identical
checkpoint manifests. An unchanged `HEAD` reference is a **cold-boot reproducibility control**, not an
independent full-precision reference or proof of no quantization loss. No Flash baseline is used.
Keep the workstation awake. In another terminal open a management tunnel to the same head configured in
both config files (replace `Spark_01` if needed):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18095:127.0.0.1:8095 Spark_01
```

Make sure sparkDash is available at the given URL and has no active benchmark. Then, from the committed
candidate repository, run:

```bash
nice -n 10 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  python3 scripts/release_gate.py --execute \
  --config /absolute/path/candidate.env --reference-config /absolute/path/reference.env \
  --reference-repo /absolute/path/qualified-full-glm --reference-rev FULL_COMMIT \
  --candidate-rev HEAD --out /absolute/path/new-release-receipts \
  --base http://127.0.0.1:18095 \
  --dash http://127.0.0.1:5555/api/sparks/spark-01/llm \
  --window-minutes 90 --aa-max 0.01 --kl-max 0.01 --min-prose-tps 50
```

Freeze those thresholds before the run. They are deliberately conservative screening limits,
not established full-GLM quality tolerances. Do not raise them after observing a failure. If an A/A
panel is unstable, record HOLD and diagnose repeatability before comparing an optimization.
The first 65 minutes allow measurements; the final 25 minutes are reserved for the gate's own stops and a
fresh reference re-admission. Individual HTTP requests, benchmark jobs, qeval subprocesses and admissions
are bounded. A deadline yields partial receipts and HOLD; it never counts omitted cells as passes.
An SSH failure can require manual recovery and is outside any wall-clock guarantee. Safety cleanup
completes even if the window expires. No power operation is automatic.

Read `gate.json`, all phase files and `SERVING.md`. Exit 0 means the bounded requested gate passed;
exit 2 means HOLD or an implementation timeout. With the current 32k
release profile, the complete 4k–128k scope **must** return HOLD even if every supported request passes.
The 32k prefill cell measures 32,767 input tokens plus one output token, and records both lengths.
The 64k/128k cells are explicit unsupported records; the program does not silently increase the KV pool
or context limit.

The reference stop is verified before the candidate boots: the exact four container names are stopped and
preserved, there are no compute PIDs, no OOM, and the gate's fleet marker is gone. Only the gate's exact
launcher PID is interrupted. A failed stop keeps the fleet marker; do not retry a boot or remove the
marker without investigating. A qualified candidate stays up under the watchdog. On HOLD, the previously
checked reference is freshly booted with a different empty runtime/cache; if recovery cannot be safely
admitted, the fleet is left stopped and the error is recorded. A surviving foreground watchdog keeps the
fleet marker until it is handed over explicitly. To stop serving, use the fresh checkout path recorded in
`SERVING.md`; do not delete remote containers or caches.
