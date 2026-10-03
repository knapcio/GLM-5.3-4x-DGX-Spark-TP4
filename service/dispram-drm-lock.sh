#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# DRAFT, NOT DEPLOYED. Root helper for dispram-drm-lock.service (install as /usr/local/sbin/dispram-drm-lock).
#   lock    every /dev/dri card/render node to mode 0000 (non-root cannot open DRM, so no dumb buffers)
#   unlock  udev default modes back, ONLY if provably no lender container, no CUDA context and nothing
#           lent; anything unverifiable keeps the lock (the next reboot restores the defaults).
RUN=${DISPRAM_RUN:-/srv/glm-dispram/run}
case "$1" in
lock)
  for n in /dev/dri/card* /dev/dri/renderD*; do [ -e "$n" ] && chmod 0000 "$n"; done; exit 0 ;;
unlock)
  keep() { echo "dispram-drm-lock: $*: DRM stays locked"; exit 0; }
  l=$(docker ps -a -q --filter label=glm.dispram=1 --filter status=running) || keep 'docker unavailable'
  [ -z "$l" ] || keep 'lender container running'
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || keep 'nvidia-smi failed'
  [ -z "$a" ] || keep "CUDA contexts: $a"
  l=$(cat "$RUN/lent.json") || keep 'lent.json unreadable'
  [ "$(printf %s "$l" | tr -d ' \n')" = '[]' ] || keep 'lent.json not empty'
  udevadm trigger --subsystem-match=drm --action=change; echo 'dispram-drm-lock: unlocked' ;;
*) echo "usage: $0 lock|unlock"; exit 1 ;;
esac
