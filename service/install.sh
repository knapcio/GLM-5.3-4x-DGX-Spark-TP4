#!/bin/bash
# DRAFT, NOT RUN. Install on one node, as root, with no GPU process present:
#   sudo bash service/install.sh (after rendering SERVICE_USER in both templates)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
if grep -q SERVICE_USER "$D/dispramd.service" "$D/sudoers-dispram"; then
  echo 'Render SERVICE_USER in both templates before installing' >&2; exit 1
fi
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU process present'; exit 1; }
[ -x /srv/glm-dispram/dispram.sh ] || { echo 'dispram.sh fetch/build first'; exit 1; }
install -m 0755 "$D/dispram-drm-lock.sh" /usr/local/sbin/dispram-drm-lock
install -m 0644 "$D/dispram-drm-lock.service" "$D/dispramd.service" /etc/systemd/system/
install -m 0440 "$D/sudoers-dispram" /etc/sudoers.d/dispram && visudo -cf /etc/sudoers.d/dispram
systemctl daemon-reload
systemctl enable dispram-drm-lock.service dispramd.service
systemctl start dispram-drm-lock.service dispramd.service
sleep 5; systemctl --no-pager status dispramd.service | head -12
