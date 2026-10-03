#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/.." && pwd)
: "${GLM_CAMPAIGN_DAY:?Set GLM_CAMPAIGN_DAY to the external day3 CPU fixture directory}"
task_day=$(cd "$GLM_CAMPAIGN_DAY" && pwd)
task_out=$task_repo/tests/results/padv2/coordinator
task_image=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
cd "$task_repo"
[[ -d "$task_day/mtp-kstop" ]] || { echo "missing mtp-kstop CPU fixtures" >&2; exit 2; }
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
