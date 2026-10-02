#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${GLM_IMAGE_SRC:?Set GLM_IMAGE_SRC to the parent directory of the extracted image vllm/ tree}"
export GLM_IMAGE_SRC
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2
if [[ ${GLM_CPU_NICED:-0} != 1 ]]; then
    export GLM_CPU_NICED=1
    exec nice -n 10 bash "$0" "$@"
fi
python3 "$ROOT/scripts/check_source_pins.py" "$GLM_IMAGE_SRC"
python3 -B "$ROOT/tests/test_recipe.py"
python3 -B "$ROOT/tests/test_launcher.py"
python3 -B "$ROOT/tests/test_release_gate.py"
DEPS=(torch numpy regex transformers pydantic cachetools msgspec pyzmq psutil cloudpickle blake3 openai prometheus_client pyyaml sentencepiece tiktoken einops cbor2 aiohttp openai-harmony pillow pybase64 uvloop py-cpuinfo llguidance xgrammar safetensors)
PY=(uv run --quiet --no-project --python 3.12)
for dep in "${DEPS[@]}"; do PY+=(--with "$dep"); done
"${PY[@]}" python -B "$ROOT/tests/test_swa_pool.py"
"${PY[@]}" python -B "$ROOT/tests/test_dsa_short.py"
"${PY[@]}" python -B "$ROOT/tests/test_dirty_l2.py"
"${PY[@]}" python -B "$ROOT/tests/test_kernel_cpu.py"
"${PY[@]}" python -B "$ROOT/tests/test_glm_fast_load.py"
"${PY[@]}" python -B "$ROOT/roce/tests/test_glm_roce_cpu.py"
