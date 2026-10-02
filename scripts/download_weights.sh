#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS_ROOT=${1:-$HOME/models}
python3 "$ROOT/scripts/weights.py" download target "$MODELS_ROOT/Tech2wild/GLM-5.3-Int4-Int8Mix"
# Optional historical DSpark fallback only:
# python3 "$ROOT/scripts/weights.py" download drafter "$MODELS_ROOT/RedHatAI/GLM-5.3-speculator.dspark"
