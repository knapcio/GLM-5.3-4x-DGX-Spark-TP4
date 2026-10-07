#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local cache experiment. Root 'flush'; serving user 'begin/ready/end'.

All checkpoint readers MUST call begin before loading and ready only after
full warm admission. The lock excludes sync/drop from checkpoint transitions.
The loader's existing range fadvise remains in charge while loading.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

STATE = Path('/run/glm-kv-headroom')
GiB_KB = 1048576


def meminfo(path=Path('/proc/meminfo')):
    return {k: int(v.split()[0]) for k, v in
            (line.split(':', 1) for line in path.read_text().splitlines())}


def candidate_kb(m):
    # Conservative upper estimate of unmapped clean disk-backed page cache.
    return max(0, m['Cached'] - m['Shmem'] - m['Mapped'] - m['Dirty'] - m['Writeback'])


def eligible(m, state, now):
    if not (state / 'enabled').is_file():
        return 'disabled'
    if (state / 'loading').exists():
        return 'checkpoint-loading'
    ready = state / 'ready'
    if not ready.is_file() or not 0 <= now - ready.stat().st_mtime <= 90:
        return 'no-fresh-serving-lease'
    if m['MemAvailable'] < 8 * GiB_KB:
        return 'below-serving-floor'
    if m['Dirty'] + m['Writeback'] > 64 * 1024:
        return 'writeback-busy'
    if candidate_kb(m) <= 4 * GiB_KB:
        return 'cache-below-4GiB'
    if m['MemFree'] > 4 * GiB_KB:
        return 'enough-free-pages'
    return None


def perform(action, state=STATE, read=meminfo, sync=None, drop=None):
    # No automatic mkdir: absent installation fails closed. Do not flock an
    # unrelated inode after a boot; tmpfiles owns this persistent-in-boot lock.
    with (state / 'lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if action in ('begin', 'end'):
            (state / 'ready').unlink(missing_ok=True)
            if action == 'begin':
                (state / 'loading').touch()
            else:
                (state / 'loading').unlink(missing_ok=True)
            return dict(action=action)
        if action == 'ready':
            (state / 'loading').unlink(missing_ok=True)
            (state / 'ready').touch()
            return dict(action=action)
        before = read()
        reason = eligible(before, state, time.time())
        if reason:
            return dict(action='skip', reason=reason, mem_kB=before,
                        candidate_kB=candidate_kb(before))
        if os.geteuid() != 0 and sync is None:
            raise PermissionError('flush requires root')
        started = time.monotonic()
        if sync:
            sync()
        else:
            subprocess.run(['/usr/bin/sync'], check=True, timeout=15)
        # Recheck readiness/off switch and floors after sync while holding lock.
        current = read()
        reason = eligible(current, state, time.time())
        if reason:
            return dict(action='skip-after-sync', reason=reason, mem_kB=current)
        if drop:
            drop()
        else:
            Path('/proc/sys/vm/drop_caches').write_text('1\n')
        return dict(action='drop_caches=1', elapsed_s=time.monotonic() - started,
                    before_kB=before, after_kB=read(),
                    buddyinfo=Path('/proc/buddyinfo').read_text() if Path('/proc/buddyinfo').exists() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['flush', 'begin', 'ready', 'end'])
    a = p.parse_args()
    try:
        result = perform(a.action)
    except Exception as e:
        print(json.dumps(dict(action='error', error=str(e), epoch=time.time())), flush=True)
        raise SystemExit(1)
    print(json.dumps(dict(result, epoch=time.time())), flush=True)


if __name__ == '__main__':
    main()
