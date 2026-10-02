"""In-image checks for the GLM RoCEnante shim (CPU only; run by Dockerfile.roce).

1. Off by default: with GLM_ROCE_ALLREDUCE unset, importing the three vLLM modules
   leaves them untouched and ``b12x`` is not importable (image == base image).
2. On: with GLM_ROCE_ALLREDUCE=1 the ``glm_roce.pth`` hook patches exactly those
   classes in a fresh interpreter, and ``verify_targets`` finds no drift.
3. The vendored runtime imports, reports API_VERSION 1, and the RDMA proxy built
   at image-build time loads (ABI version readable; no RDMA device needed).

If vLLM itself cannot be imported without a GPU in the build container, checks 1-2
are reported as SKIPPED (the 4-rank script repeats them on the fleet).
Usage: CUDA_VISIBLE_DEVICES= python3 /opt/glm-roce/tests/test_glm_roce_image.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

MODULES = (
    "vllm.distributed.parallel_state",
    "vllm.distributed.device_communicators.cuda_communicator",
    "vllm.v1.worker.gpu_worker",
)

PROBE = r"""
import importlib, importlib.util, json, sys
mods = %r
for m in mods:
    importlib.import_module(m)
ps = sys.modules[mods[0]]; cc = sys.modules[mods[1]]; gw = sys.modules[mods[2]]
out = {
    "graph_capture": bool(getattr(ps.GroupCoordinator, "_glm_roce_patched", False)),
    "communicator": bool(getattr(cc.CudaCommunicator, "_glm_roce_patched", False)),
    "worker": bool(getattr(gw.Worker, "_glm_roce_patched", False)),
    "b12x_importable": importlib.util.find_spec("b12x") is not None,
    "hook_loaded": "glm_roce.boot" in sys.modules,
}
if out["communicator"]:
    from glm_roce.install import verify_targets
    out["problems"] = verify_targets()
print("PROBE " + json.dumps(out))
""" % (MODULES,)


def probe(enabled: bool) -> dict:
    env = dict(os.environ)
    env.pop("GLM_ROCE_ALLREDUCE", None)
    if enabled:
        env["GLM_ROCE_ALLREDUCE"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    raise RuntimeError(f"probe failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")


def check_runtime() -> None:
    env = dict(os.environ, GLM_ROCE_ALLREDUCE="1")
    code = (
        "from b12x.comm import roce; from b12x.comm.roce import _proxy;"
        "lib = _proxy.load(); print('RUNTIME', roce.API_VERSION, lib.roce_abi_version(), _proxy._build())"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RUNTIME ")), None)
    assert line is not None, f"b12x runtime check failed:\n{proc.stdout}\n{proc.stderr[-4000:]}"
    _, api, abi, so_path = line.split(" ", 3)
    assert api == "1", f"b12x.comm.roce API_VERSION {api}, the shim needs 1"
    cache = os.environ.get("B12X_ROCE_CACHE_DIR", "")
    assert not cache or so_path.startswith(cache), f"proxy not prebuilt in {cache}: {so_path}"
    print(f"PASS runtime: API_VERSION={api} proxy ABI={abi} at {so_path}")


def main() -> int:
    check_runtime()
    try:
        off = probe(False)
        on = probe(True)
    except RuntimeError as exc:
        text = str(exc)
        if "CUDA" in text or "cuda" in text or "NVML" in text or "driver" in text:
            print(f"SKIPPED vLLM import checks (no GPU in the build container): {text[-600:]}")
            return 0
        raise
    assert not (off["graph_capture"] or off["communicator"] or off["worker"]), f"patched while off: {off}"
    assert not off["b12x_importable"], "b12x importable while GLM_ROCE_ALLREDUCE is off"
    print(f"PASS off: {off}")
    assert on["hook_loaded"] and on["graph_capture"] and on["communicator"] and on["worker"], on
    assert on["b12x_importable"], on
    assert on.get("problems") == [], f"vLLM drift: {on.get('problems')}"
    print(f"PASS on: {on}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
