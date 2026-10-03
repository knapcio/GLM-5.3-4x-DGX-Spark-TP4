# SPDX-License-Identifier: Apache-2.0
"""Run the F1/F2/F3 single-GPU gates in order, one process each; exit 0 only if all PASS.

Inside the pinned image on ONE idle Spark GPU with serving stopped (no multi-GB tests next to a
serving process). Example, from a checkout of this repository:
  docker run --rm --name glm53full-glue-gates-<ts> --gpus all --network none \
    -v <checkout>:/repo:ro -v <out>:/out --entrypoint python3 <image> \
    /repo/tests/gpu/run_gpu_gates.py --out /out
Any FAIL means the glue-lite switches stay off (docs/runtime.md).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATES = (("F1", "gpu_bitexact_f1.py", 300), ("F2", "gpu_gate_f2.py", 900), ("F3", "gpu_gate_f3.py", 600))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    for name, script, limit in GATES:
        if a.only and name not in a.only:
            continue
        begun = time.time()
        receipt = out / f"{name}.json"
        try:
            z = subprocess.run([sys.executable, str(HERE / script), "--out", str(receipt)], capture_output=True,
                               text=True, timeout=limit)
            rc, tail = z.returncode, (z.stdout[-2000:] + z.stderr[-4000:])
        except subprocess.TimeoutExpired:
            rc, tail = 124, "timeout"
        (out / f"{name}.log").write_text(tail)
        verdict = json.loads(receipt.read_text())["verdict"] if receipt.exists() else "NO_RECEIPT"
        rows.append(dict(gate=name, rc=rc, verdict=verdict, seconds=round(time.time() - begun, 1)))
        ok = ok and rc == 0 and verdict == "PASS"
        if not ok:
            break  # any FAIL stops; later gates are not run on a failed device state
    summary = dict(status="PASS_GPU_GATES" if ok and len(rows) == len(a.only or GATES) else "FAIL_GPU_GATES",
                   gates=rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0 if summary["status"] == "PASS_GPU_GATES" else 1


if __name__ == "__main__":
    sys.exit(main())
