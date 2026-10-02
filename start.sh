#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/profiles/current.env"
source "$HERE/.env.example"
[[ ! -f "$HERE/.env" ]] || source "$HERE/.env"
export RECIPE_PROFILE="${RECIPE_PROFILE:-native-mtp-k2}"
case "$RECIPE_PROFILE" in
  native-mtp-k2) export RECIPE_SERVE_ARGS="$HERE/profiles/serve-args.json" ;;
  dspark-k3)
    source "$HERE/profiles/dspark-k3.env"
    export RECIPE_SERVE_ARGS="$HERE/profiles/dspark-k3-args.json"
    ;;
  *) echo 'RECIPE_PROFILE must be native-mtp-k2 or dspark-k3' >&2; exit 1 ;;
esac
export RECIPE_ROOT="$HERE" DRY=${DRY:-0}
export RECIPE_HOSTS="${HOSTS[*]}" RECIPE_IPS="${IPS[*]}"
export RECIPE_IMAGES="${IMAGES[*]:-}"
export FABRIC_IFACE IB_HCA MODEL_DIR DRAFT_DIR NCCL_HOST_DIR OVERLAY_REMOTE IMAGE NCCL_SHA256
exec python3 "$HERE/scripts/cluster.py" "$@"
