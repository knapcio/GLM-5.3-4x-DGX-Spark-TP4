#!/usr/bin/env bash
set -euo pipefail
task_repo=/srv/projects/glm53-full-padv2
task_day=/srv/campaign/diagnostics/glm53-full-20260929/day3
task_out=$task_repo/tests/results/padv2/coordinator
task_image=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
cd "$task_repo"
[[ $(uname -s) == Darwin && $(docker context show) == colima ]]
task_running=$(docker ps -q)
[[ -z $task_running ]] || { echo 'docker ps must be empty' >&2; exit 2; }
docker image inspect "$task_image" --format '{{.Id}} {{.Architecture}}'
mkdir -p "$task_out"
docker run --pull never --network none --rm --platform linux/arm64 --cpus 2 --memory 8g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/tmp/triton \
  --workdir /pkg \
  --mount "type=bind,src=$task_repo,dst=/pkg,readonly" \
  --mount "type=bind,src=$task_day,dst=/campaign/day3,readonly" \
  --mount "type=bind,src=$task_out,dst=/results" --entrypoint /usr/bin/nice \
  "$task_image" -n 10 python3 -B /pkg/tests/kstop_compat_cpu_suite.py \
  > "$task_out/suite.txt" 2>&1
cat "$task_out/PASS.json"
