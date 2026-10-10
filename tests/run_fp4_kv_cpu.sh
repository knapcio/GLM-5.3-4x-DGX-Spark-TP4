#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Mac/Colima only, pinned installed source, no network or GPU, retained container.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DAY=${1:?Pass the campaign day3 directory}
DUMPS=${2:?Pass fp4probe-20261005-0855}
OUT=${3:?Pass a new receipts directory under Projects, outside the worktree}
OUT=$(python3 - "$ROOT" "$OUT" <<'PY'
from pathlib import Path
import sys
root,out=(Path(p).resolve() for p in sys.argv[1:])
projects=Path('/srv/projects')
if projects not in out.parents or root==out or root in out.parents:
    sys.exit('OUT must be under Projects and outside the worktree')
if out.exists():sys.exit('Choose a new OUT directory')
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
NAME=fp4-kv-cpu-$(date +%s)-$$
printf '%s\n' "$NAME" > "$OUT/container-name.txt"
exec docker run --pull never --network none --name "$NAME" --platform linux/arm64 --cpus 2 --memory 12g \
  --cap-drop ALL --security-opt no-new-privileges \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/results/triton \
  --mount "type=bind,src=$ROOT,dst=/pkg,readonly" \
  --mount "type=bind,src=$DAY,dst=/receipts/day3,readonly" \
  --mount "type=bind,src=$DUMPS,dst=/probe,readonly" \
  --mount "type=bind,src=$OUT,dst=/results" --entrypoint /usr/bin/nice \
  "$IMAGE" -n 10 python3 -B /pkg/tests/fp4_kv_cpu_suite.py
