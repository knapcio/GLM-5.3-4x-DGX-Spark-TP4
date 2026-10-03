#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Fetch, build and run kindling's dispramd UNMODIFIED on one Spark (run on the Spark host).
#   dispram.sh fetch|build                       pinned sources, unmodified build, hash checks
#   sudo-pw | dispram.sh drm-lock|drm-unlock     DRM exclusion for the whole lease (device nodes 0000)
#   dispram.sh start|stop                        daemon lifetime (refuses with any GPU process/borrower/lease)
#   dispram.sh status|watch|preborrow|postcheck  watch/preborrow/postcheck exit 3 = NOT safe (fail closed)
#   dispram.sh service                           foreground supervisor for the draft unit (service/)
#
# dispramd and librmlist are third-party (kindling-spark-os, AGPL-3.0, kindlingai). This script only
# fetches them at a pinned commit, compiles rmlist.c against NVIDIA's MIT headers at the driver tag,
# checks hashes and launches the daemon. Nothing of theirs is modified, copied into the recipe or
# imported by the recipe; the vLLM hook talks to the daemon over its socket (INTERFACE.md).
#
# Safety contract (A1 review P1): the daemon runs in the HOST PID namespace (--pid=host) so its
# SO_PEERCRED/proc borrower tracking sees engine processes in sibling containers; it is never
# restarted automatically (--restart no) and never started or stopped while any CUDA context exists
# on the node (nvidia-smi compute apps). DRM is excluded for the whole lease by making /dev/dri nodes
# unopenable (non-root) and checked every watch (modes + users). Any unreadable monitor input, daemon
# exit, DRM user, Xid, SMMU or NVRM error makes watch exit 3: session invalid. Rollback order:
# stop borrowers -> verify no CUDA context -> stop daemon -> drm-unlock -> restore baseline.
set -euo pipefail

KINDLING_URL=https://github.com/kindlingai/kindling-spark-os
KINDLING_COMMIT=5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e            # kindling 0.9.3, 2026-10-01
OGKM_URL=https://github.com/NVIDIA/open-gpu-kernel-modules
OGKM_COMMIT=20e4e6e19cc26ba47b5cbe23130a396be100c427               # tag 580.173.02 (MIT headers)
DRIVER=580.173.02
SHA_DISPRAMD=769cbff80b80af5c667a75b70036fadabcebe36bec375facc57761cc7aaa5cdd
SHA_RMLIST_C=457bf38aa06af41bafe4c049006f89125790bf469b7818d30bc64b82f3f36657
SHA_LIBRMLIST=9f28b4eb143bc4c01b96f06f52bacdaf2c17f33f747a06e58f75263f36b963f0  # gcc 13.3, A1 build
IMAGE=${DISPRAM_IMAGE:-glm53-roce:v11-b58f34ea}                      # serving image: python3 + gcc 13.3
HOME_D=${DISPRAM_HOME:-/srv/glm-dispram}
SRC=$HOME_D/src; BIN=$HOME_D/kindling; RUN=$HOME_D/run; LOG=$HOME_D/log
LABEL=glm.dispram=1

die() { echo "dispram: $*" >&2; exit 1; }
bad() { echo "DISPRAM-INVALID $*"; exit 3; }
ts() { date '+%F %T %Z'; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
running() { docker ps --filter label=$LABEL --format '{{.Names}}'; }
compute_apps() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; }
dri_nodes() { ls ${DISPRAM_DRI_NODES:-/dev/dri/card* /dev/dri/renderD*} 2>/dev/null; }  # override: tests only
dri_users() {
  docker run --rm --name "dispram-dri-scan-$$" --privileged --pid=host --network none --entrypoint bash "$IMAGE" -c '
    for p in /proc/[0-9]*; do for f in $p/fd/*; do t=$(readlink $f 2>/dev/null) || continue
      case $t in /dev/dri/*) echo "${p#/proc/} $(cat $p/comm 2>/dev/null) $t";; esac; done; done; echo SCAN-DONE'
}
query() {  # our own client for INTERFACE.md "info" (no kindling code)
  docker run --rm --name "dispram-query-$$" --network none -v "$RUN:/run/dispram" --entrypoint python3 "$IMAGE" -c '
import json, socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); s.settimeout(5); s.connect("/run/dispram/dispram.sock")
s.send(json.dumps({"key": "kindlingai_1", "op": "info"}).encode()); print(s.recv(65536).decode())'
}
borrowers() {  # running containers, other than the lender, that mount the daemon socket dir; rc!=0 = unknown
  local c ids lender m
  ids=$(docker ps -q) || return 1
  lender=$(docker ps -q --filter label=$LABEL) || return 1
  for c in $ids; do
    echo "$lender" | grep -qx "$c" && continue
    if ! m=$(docker inspect -f '{{.Name}} {{range .Mounts}}{{.Source}};{{end}}' "$c" 2>/dev/null); then
      # fix8: a container that exited and was removed between ps and inspect (our own --rm helpers run
      # concurrently from other dispram.sh calls) is not a borrower; anything else stays "unknown"
      still=$(docker ps -q --no-trunc) || return 1  # (no pipe into grep -q: pipefail + SIGPIPE would hide it)
      case "$still" in *"$c"*) return 1 ;; esac
      continue
    fi
    case "$m" in /dispram-query-*|/dispram-dri-scan-*|/dispram-build-*) continue ;; esac  # our helpers, not borrowers
    echo "$m" | grep -F "$RUN;" | cut -d' ' -f1 | tr -d / || true
  done
}
verified_free() {  # rc 0 only when provably: no CUDA context, no borrower, nothing lent
  local apps b
  apps=$(compute_apps) || return 1; [ -z "$apps" ] || return 1
  b=$(borrowers) || return 1; [ -z "$b" ] || return 1
  ( lease_free ) >/dev/null 2>&1
}
CT=${DISPRAM_CALL_TIMEOUT:-60}
bnd() { timeout --kill-after=5 "$CT" "$@"; }   # every teardown step bounded; a timeout is "unknown"
teardown() {  # session invalid: stop borrowers; release the lender ONLY after a bounded, verified release
  local b n keep=''
  if b=$(bnd "$0" borrowers); then
    for x in $b; do echo "$(ts) stopping borrower $x" | tee -a "$LOG/events.txt"; bnd docker stop -t 30 "$x" >/dev/null || keep=1; done
  else
    keep=1; echo "$(ts) borrower enumeration failed or timed out" | tee -a "$LOG/events.txt"
  fi
  for _ in $(seq 30); do bnd "$0" verified-free >/dev/null 2>&1 && break; sleep 2; done
  if [ -z "$keep" ] && bnd "$0" verified-free >/dev/null 2>&1 && n=$(bnd docker ps --filter label=$LABEL --format '{{.Names}}') && [ -n "$n" ]; then
    bnd docker stop -t 10 "$n" >/dev/null && echo "$(ts) lender stopped after verified release" | tee -a "$LOG/events.txt"
  else
    echo "$(ts) borrowers, CUDA contexts or leases remain (or cannot be verified): lender and DRM lock left up; node recovery (RUN.md section 4)" | tee -a "$LOG/events.txt"
  fi
}
idle_gpu() {
  local apps; apps=$(compute_apps) || die "nvidia-smi failed ($1): refusing"
  [ -z "$apps" ] || die "GPU compute processes present, refusing ($1): $apps"
}
drm_locked() {  # every DRM node mode 0000 (non-root cannot open; DRM dumb buffers need a card node)
  local n m any=0
  for n in $(dri_nodes); do any=1; m=$(stat -c %a "$n") || return 1; [ "$m" = 0 ] || return 1; done
  [ $any = 1 ] || [ ! -d /dev/dri ]
}
check_scan() {  # $1 = scan output; fail closed on an incomplete scan or any user
  echo "$1" | grep -q '^SCAN-DONE$' || bad 'DRM user scan failed'
  local u; u=$(echo "$1" | grep -v '^SCAN-DONE$' || true)
  [ -z "$u" ] || bad "DRM users: $u"
}
# Kernel-line classification (fix4). Input: journalctl -k -o short-unix. Exit 1 + reason on:
#  - any journalctl meta line (no epoch prefix) mentioning permissions/no journal files (unreadable);
#  - any Xid / SMMU / IOMMU-fault / NVRM error-or-fail line, EXCEPT exactly
#    "NVRM: refcntRequestReference_IMPL: Failed to enter state 1 (current state: 0, status: 0x00000056)"
#    whose timestamp lies inside a profiler window declared by a borrower in
#    $RUN/borrower-profiler-windows.log ("start <epoch> <tag>" / "end <epoch> <tag>"; window = [start-1, end+5];
#    an unterminated start counts until now+0). That line is routine on every node at CUPTI sessions (09-28..).
KERNEL_FILTER='
import re, sys, time
ALLOW = re.compile(r"NVRM: refcntRequestReference_IMPL: Failed to enter state 1 \(current state: 0, status: 0x00000056\)$")
# fix6 (PROPOSED, opt-in via DISPRAM_ALLOW_RM_ALLOC_OOM=1, needs the coordinator): an RM allocation that returned
# NV_ERR_NO_MEMORY and was handled by its caller. Routine without dispram: 1136/1268/1245/1608 lines per node in
# the 24 h before 2026-10-02 16:01, in bursts at every engine boot and during serving. Exactly this text; counted.
ALLOC = re.compile(r"NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory \[NV_ERR_NO_MEMORY\] \(0x00000051\) returned from _memdescAllocInternal\(pMemDesc\) @ mem_desc\.c:1359$")
import os
allow_alloc = os.environ.get("DISPRAM_ALLOW_RM_ALLOC_OOM") == "1"
n_alloc = 0
FATAL = re.compile(r"xid|smmu|iommu.*fault|nvrm:.*(error|fail)", re.I)
wins, opened = [], {}
try:
    for row in open(sys.argv[1]):
        f = row.split()
        if len(f) >= 3 and f[0] == "start": opened[f[2]] = float(f[1])
        elif len(f) >= 3 and f[0] == "end" and f[2] in opened: wins.append((opened.pop(f[2]) - 1, float(f[1]) + 5))
except FileNotFoundError:
    pass
wins += [(t - 1, time.time()) for t in opened.values()]
bad = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line: continue
    m = re.match(r"^(\d+\.\d+) ", line)
    if not m:
        if re.search(r"insufficient permissions|no journal files|failed to (open|read)", line, re.I): bad.append("journal unreadable: " + line)
        continue
    if not FATAL.search(line): continue  # fix5: no blanket suppression of any NVRM class
    t = float(m.group(1))
    if ALLOW.search(line) and any(a <= t <= b for a, b in wins): continue
    if allow_alloc and ALLOC.search(line):
        n_alloc += 1
        continue
    bad.append(line)
if bad:
    print("; ".join(bad)[:2000]); sys.exit(1)
if n_alloc:
    print("rm-alloc-oom-lines=%d" % n_alloc)
'
watch_once() {
  local n since k i
  n=$(running) || bad 'docker ps failed'
  [ -n "$n" ] || bad 'daemon not running'
  [ "$(docker inspect -f '{{.State.Running}}' "$n")" = true ] || bad "daemon $n not running"
  [ "${DISPRAM_DRM_LOCKDOWN:-1}" = 0 ] || drm_locked || bad 'DRM nodes not locked (mode != 0000)'
  check_scan "$(dri_users 2>&1)"
  since=$(docker inspect -f '{{.State.StartedAt}}' "$n") || bad 'docker inspect failed'
  k=$(journalctl -k -q -o short-unix --no-pager --since "$(date -d "$since" '+%F %T')" 2>&1) || bad "kernel journal unreadable: $k"
  k=$(printf '%s\n' "$k" | python3 -c "$KERNEL_FILTER" "$RUN/borrower-profiler-windows.log" 2>&1) || bad "kernel: $k"
  i=$(query 2>&1) || bad "daemon info unreachable: $i"
  echo "$i" | grep -q '"size"' || bad "daemon info malformed: $i"
  echo "DISPRAM-OK $n $i"
}
invalid_marker() { echo "$LOG/INVALID-$(cat /proc/sys/kernel/random/boot_id)"; }
lease_free() {  # preborrow/postcheck: nothing lent, no CUDA context, no borrower
  local i apps b
  apps=$(compute_apps) || bad 'nvidia-smi failed'
  [ -z "$apps" ] || bad "GPU compute processes present: $apps"
  b=$(borrowers) || bad 'borrower scan failed'
  [ -z "$b" ] || bad "borrower containers running: $b"
  i=$(query 2>&1) || bad "daemon info unreachable: $i"
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d["free"] == d["size"] == d["largest"] else 1)' "$i" \
    || bad "carveout not fully free: $i"
  [ "$(tr -d ' \n' < "$RUN/lent.json" 2>/dev/null)" = '[]' ] || bad "lent.json not empty: $(cat "$RUN/lent.json" 2>/dev/null)"
}

case "${1:-}" in
fetch)
  mkdir -p "$SRC"
  if [ ! -d "$SRC/kindling/.git" ]; then git init -q "$SRC/kindling"; git -C "$SRC/kindling" remote add origin "$KINDLING_URL"; fi
  git -C "$SRC/kindling" fetch -q --depth 1 origin "$KINDLING_COMMIT"
  git -C "$SRC/kindling" checkout -q --detach FETCH_HEAD
  [ "$(git -C "$SRC/kindling" rev-parse HEAD)" = "$KINDLING_COMMIT" ] || die 'kindling commit mismatch'
  if [ ! -d "$SRC/ogkm/.git" ]; then
    git init -q "$SRC/ogkm"; git -C "$SRC/ogkm" remote add origin "$OGKM_URL"
    git -C "$SRC/ogkm" sparse-checkout set src/common/sdk/nvidia/inc kernel-open/common/inc src/nvidia/arch/nvalloc/unix/include
  fi
  git -C "$SRC/ogkm" fetch -q --depth 1 --filter=blob:none origin "$OGKM_COMMIT"
  git -C "$SRC/ogkm" checkout -q --detach FETCH_HEAD
  [ "$(git -C "$SRC/ogkm" rev-parse HEAD)" = "$OGKM_COMMIT" ] || die 'open-gpu-kernel-modules commit mismatch'
  [ "$(sha "$SRC/kindling/dispram/dispramd.py")" = "$SHA_DISPRAMD" ] || die 'dispramd.py hash mismatch'
  [ "$(sha "$SRC/kindling/dispram/rmlist.c")" = "$SHA_RMLIST_C" ] || die 'rmlist.c hash mismatch'
  echo "fetch OK kindling $KINDLING_COMMIT ogkm $OGKM_COMMIT" ;;
build)
  [ "$(git -C "$SRC/kindling" rev-parse HEAD)" = "$KINDLING_COMMIT" ] || die 'run fetch first'
  mkdir -p "$BIN/out"
  docker run --rm --name "dispram-build-$$" --network none -v "$SRC:/s:ro" -v "$BIN/out:/out" --entrypoint bash "$IMAGE" -c '
    gcc --version | head -1
    gcc -O2 -Wall -Werror -shared -fPIC /s/kindling/dispram/rmlist.c -o /out/librmlist.so \
      -I/s/ogkm/src/common/sdk/nvidia/inc -I/s/ogkm/kernel-open/common/inc -I/s/ogkm/src/nvidia/arch/nvalloc/unix/include'
  got=$(sha "$BIN/out/librmlist.so")
  [ "$got" = "$SHA_LIBRMLIST" ] || die "librmlist.so sha256 $got != $SHA_LIBRMLIST (compiler differs from the A1 build?)"
  cp "$BIN/out/librmlist.so" "$BIN/librmlist.so"
  for f in dispram/dispramd.py dispram/LICENSE dispram/LICENSE-GPL dispram/BUNDLING-EXCEPTION dispram/README.md; do
    cp "$SRC/kindling/$f" "$BIN/"; done
  [ "$(sha "$BIN/dispramd.py")" = "$SHA_DISPRAMD" ] || die 'staged dispramd.py hash mismatch'
  echo "build OK librmlist.so $got" ;;
start)
  [ ! -e "$(invalid_marker)" ] || die "session invalidated in this boot ($(cat "$(invalid_marker)")); reboot or reset-invalid first"
  [ -z "$(running)" ] || die "dispramd already running: $(running)"
  [ "$(sha "$BIN/dispramd.py")" = "$SHA_DISPRAMD" ] && [ "$(sha "$BIN/librmlist.so")" = "$SHA_LIBRMLIST" ] || die 'staged binaries fail hash check'
  grep -q "$DRIVER" /proc/driver/nvidia/version || die "driver is not $DRIVER"
  idle_gpu start
  [ -z "$(borrowers)" ] || die "borrower containers present: $(borrowers)"
  if [ "${DISPRAM_DRM_LOCKDOWN:-1}" = 0 ]; then echo 'WARNING: DRM exclusion by policy + polling only'; else drm_locked || die 'run drm-lock first'; fi
  scan=$(dri_users 2>&1); echo "$scan" | grep -q '^SCAN-DONE$' || die 'DRM scan failed'
  [ "$scan" = SCAN-DONE ] || die "/dev/dri has users, refusing: $scan"
  mkdir -p "$RUN" "$LOG"
  if [ -s "$RUN/lent.json" ] && [ "$(tr -d ' \n' < "$RUN/lent.json")" != '[]' ]; then
    # no CUDA context and no borrower exist on the node (checked above), so every recorded lease is stale
    mv "$RUN/lent.json" "$RUN/lent.json.stale-$(date +%Y%m%d-%H%M%S)"; echo 'stale lent.json set aside'
  fi
  name=dispramd-$(date +%Y%m%d-%H%M%S)
  docker run -d --name "$name" --label $LABEL --restart no --privileged --pid=host --network none \
    -e DISPRAM_DRIVER=$DRIVER -v "$BIN:/opt/dispram:ro" -v "$RUN:/run/dispram" \
    --entrypoint python3 "$IMAGE" /opt/dispram/dispramd.py >/dev/null
  for _ in $(seq 20); do [ -S "$RUN/dispram.sock" ] && break; sleep 0.5; done
  [ -S "$RUN/dispram.sock" ] || { docker logs "$name" 2>&1 | tail; docker stop -t 5 "$name" >/dev/null; die 'socket did not appear'; }
  echo "$(ts) start $name" >> "$LOG/events.txt"; watch_once ;;
drm-lock)  # sudo authentication on stdin; reverted by drm-unlock or reboot
  mkdir -p "$RUN"; nodes=$(dri_nodes || true)
  [ -n "$nodes" ] || { echo 'no DRM nodes'; exit 0; }
  [ -f "$RUN/dri-modes.txt" ] || stat -c '%a %n' $nodes > "$RUN/dri-modes.txt"
  sudo -S -p '' chmod 0000 $nodes
  drm_locked && echo "DRM locked: $nodes" || die 'DRM lock failed' ;;
drm-unlock)  # only after the lender is gone and nothing can hold carveout memory
  r=$(running) || die 'docker ps failed'; [ -z "$r" ] || die 'daemon still running; stop it first'
  idle_gpu drm-unlock; b=$(borrowers) || die 'borrower scan failed'; [ -z "$b" ] || die "borrowers: $b"
  l=$(cat "$RUN/lent.json") || die 'lent.json unreadable: DRM stays locked'
  [ "$(printf %s "$l" | tr -d ' \n')" = '[]' ] || die 'lent.json not empty'
  [ -f "$RUN/dri-modes.txt" ] || die 'no saved modes'
  pw=$(cat); while read -r m n; do printf '%s\n' "$pw" | sudo -S -p '' chmod "$m" "$n"; done < "$RUN/dri-modes.txt"
  mv "$RUN/dri-modes.txt" "$RUN/dri-modes.txt.restored-$(date +%Y%m%d-%H%M%S)"; echo 'DRM modes restored' ;;
status)
  echo "container: $(running || true)"
  for n in $(running); do docker logs "$n" 2>&1 | tail -5; done
  echo "info: $(query 2>&1 || echo unreachable)"
  echo "lent.json: $(cat "$RUN/lent.json" 2>/dev/null || echo absent)"
  echo "borrowers: $(borrowers | tr '\n' ' ')"
  echo "compute apps: $(compute_apps | tr '\n' ';')"
  echo "drm locked: $(drm_locked && echo yes || echo NO)"
  echo "dri users: $(dri_users 2>&1 | tr '\n' ';')" ;;
watch) watch_once ;;
preborrow) watch_once >/dev/null; lease_free; echo 'PREBORROW-OK' ;;
postcheck)  # after a borrower stopped: wait up to 60 s for the CUDA context and the lease to go
  for _ in $(seq 30); do ( lease_free ) >/dev/null 2>&1 && break; sleep 2; done
  watch_once >/dev/null; lease_free; echo 'SHUTDOWN-VERIFIED' ;;
stop)  # no force option: the lender outlives every borrower and context (fix2 review P1-2)
  n=$(running); [ -n "$n" ] || die 'not running'
  verified_free || die 'refusing: a borrower, CUDA context or lease exists or cannot be verified (RUN.md section 4)'
  timeout --kill-after=5 60 docker stop -t 10 "$n" >/dev/null || die 'docker stop failed'
  echo "$(ts) stop $n" >> "$LOG/events.txt"; echo "stopped $n" ;;
service|monitor)  # service starts idle; monitor adopts an already healthy running lender without restarting it
  if [ "$1" = service ]; then "$0" start; else "$0" watch >/dev/null; fi
  while sleep "${DISPRAM_WATCH_S:-30}"; do
    out=$(timeout --kill-after=5 120 "$0" watch 2>&1) && continue
    echo "$(ts) ${out:-watch timed out}" | tee -a "$LOG/events.txt" > "$(invalid_marker)"
    teardown
    exit 3
  done ;;
service-teardown) teardown ;;  # the service's INVALID path, callable for tests and manual recovery
census)  # fix5 pre-window: classify the last 24 h of kernel lines; exit 3 if any would be fatal outside a window
  k=$(journalctl -k -q -o short-unix --no-pager --since -24h 2>&1) || bad "kernel journal unreadable: $k"
  printf '%s\n' "$k" | grep -iE 'xid|smmu|iommu.*fault|nvrm' | sed -E 's/^[0-9.]+ [^ ]+ kernel: //' | sort | uniq -c | sort -rn | head -20
  w=$(printf '%s\n' "$k" | python3 -c "$KERNEL_FILTER" /dev/null 2>&1) || {
    echo "$w" | tr ';' '\n' | grep -vF 'refcntRequestReference_IMPL: Failed to enter state 1 (current state: 0, status: 0x00000056)' | grep -q . \
      && bad "kernel lines that would invalidate a lease: $(echo "$w" | cut -c1-600)"; }
  echo 'CENSUS-OK (only the profiler refcnt line, if any)' ;;
deadman)  # controller-loss guard: $2 = hard epoch, or none for selected serving. Heartbeat mtime.
  # Stale heartbeat (> DEADMAN_STALE s, default 900) or now > hard epoch + 1800: stop every running
  # glm53full-* container that is not the default serving set, then release the lender only after
  # verified_free (teardown); never touches DRM modes (no sudo), never restores serving (needs all 4 nodes).
  HB=$HOME_D/heartbeat; hard=${2:?hard epoch or none}; stale=${DEADMAN_STALE:-900}
  case "$hard" in none) ;; *[!0-9]*|'') die 'hard epoch must be numeric or none' ;; esac
  echo "$(ts) deadman pid $$ hard $hard stale ${stale}s" >> "$LOG/events.txt"
  while sleep "${DEADMAN_POLL:-30}"; do
    [ -e "$HOME_D/deadman.stop" ] && { echo "$(ts) deadman stopped by controller" >> "$LOG/events.txt"; exit 0; }
    now=$(date +%s); age=$(( now - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
    [ "$age" -gt "$stale" ] || { [ "$hard" != none ] && [ "$now" -gt $((hard + 1800)) ]; } || continue
    echo "$(ts) CONTROLLER-LOST heartbeat age ${age}s: stopping borrowers" | tee -a "$LOG/events.txt" > "$HOME_D/CONTROLLER-LOST"
    for c in $(timeout 60 docker ps --format '{{.Names}}' | grep '^glm53full-' | grep -v '^glm53full-rel-default-20261002r-'); do
      timeout --kill-after=5 120 docker stop -t 60 "$c" >/dev/null 2>&1; echo "$(ts) deadman stopped $c" >> "$LOG/events.txt"; done
    [ -n "$(running)" ] && teardown
    exit 4
  done ;;
borrowers) borrowers ;;                     # bounded helpers for teardown (exit != 0 = unknown)
verified-free) verified_free && echo VERIFIED-FREE ;;
invalid-state) [ -e "$(invalid_marker)" ] && { echo "INVALID $(cat "$(invalid_marker)")"; exit 3; } || echo 'no invalidation in this boot' ;;
reset-invalid)  # manual reset after RUN.md section 4 recovery; only with no GPU process and no daemon
  idle_gpu reset-invalid; [ -z "$(running)" ] || die 'daemon running'
  m=$(invalid_marker); [ -e "$m" ] || die 'nothing to reset'
  mv "$m" "$m.reset-$(date +%Y%m%d-%H%M%S)"; echo "$(ts) reset-invalid by $(id -un)" >> "$LOG/events.txt"; echo 'reset' ;;
*) echo "usage: $0 fetch|build|drm-lock|drm-unlock|start|status|watch|preborrow|postcheck|stop|service|monitor|service-teardown|census|deadman|invalid-state|reset-invalid"; exit 1 ;;
esac
