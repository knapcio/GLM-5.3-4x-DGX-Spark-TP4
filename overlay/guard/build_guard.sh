#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build and test the ARM64 Linux copy guard in the pinned runtime image.
set -euo pipefail
guard_dir=$(cd "$(dirname "$0")" && pwd)
guard_image=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
docker run --rm --pull never --network none --platform linux/arm64 \
  --cpus 2 --memory 1g --memory-swap 1g \
  --mount "type=bind,src=$guard_dir,dst=/guard" --workdir /guard \
  --entrypoint bash "$guard_image" -c \
  'set -euo pipefail
   bash ./test_guard.sh
   gcc -O2 -Wall -Werror -shared -fPIC dispram_copy_guard.c -o libdispram_copy_guard.so -ldl
   sha256sum dispram_copy_guard.c libdispram_copy_guard.so > SHA256SUMS
   cat SHA256SUMS'
