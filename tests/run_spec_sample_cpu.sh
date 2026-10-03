#!/usr/bin/env bash
# Offline colima only. A failed docker ps cannot be mistaken for an empty queue.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
[[ $(docker context show) == colima ]] || { echo 'colima context required' >&2; exit 2; }
running=$(docker ps -q)
[[ -z $running ]] || { echo 'docker ps must be empty; retry after current jobs finish' >&2; exit 2; }
docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}'
exec docker run --pull never --network none --rm --platform linux/arm64 --cpus 2 --memory 8g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRITON_CACHE_DIR=/tmp/triton \
  --mount "type=bind,src=$ROOT,dst=/pkg,readonly" --entrypoint /usr/bin/nice \
  "$IMAGE" -n 10 python3 -B /pkg/tests/spec_sample_cpu_suite.py
