#!/usr/bin/env bash
# Local Mac/Colima only; fail closed on unavailable Docker or a busy queue.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DAY=${1:?Pass the local diagnostics/glm53-full-20260929/day3 directory}
OUT=${2:?Pass a receipts directory outside the repository}
# Resolve existing symlinks and missing path components before creating OUT or
# touching Docker. A spelling such as ../repo/receipts must also be refused.
OUT=$(python3 - "$ROOT" "$OUT" <<'PY'
from pathlib import Path
import sys
root, out = (Path(p).resolve() for p in sys.argv[1:])
if out == root or root in out.parents:
    sys.exit('OUT must be outside the repository')
print(out)
PY
)
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
[[ $(uname -s) == Darwin ]] || { echo 'Mac only' >&2; exit 2; }
[[ $(docker context show) == colima ]] || { echo 'colima context required' >&2; exit 2; }
running=$(docker ps -q)
[[ -z $running ]] || { echo 'docker ps must be empty' >&2; exit 2; }
docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}'
mkdir -p "$OUT"
exec docker run --pull never --network none --rm --platform linux/arm64 --cpus 2 --memory 8g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/tmp/triton \
  --mount "type=bind,src=$ROOT,dst=/pkg,readonly" \
  --mount "type=bind,src=$DAY,dst=/receipts/day3,readonly" \
  --mount "type=bind,src=$OUT,dst=/results" --entrypoint /usr/bin/nice \
  "$IMAGE" -n 10 python3 -B /pkg/tests/kstop_compat_cpu_suite.py
