#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-node memory recorder: MemAvailable, swap, memory PSI and buddyinfo as JSON lines. Read-only, stdlib only.

Run one per node (for example as a systemd --user unit). The sampling interval is re-read once per second from
--rate-file (seconds, 0.05..10; 0.1 = 10 Hz for a boot, 1 = 1 Hz afterwards), so the rate changes without a restart.
Stops at --seconds, on SIGTERM, or when --stop-file exists. Never allocates more than a line buffer.

Line keys: t epoch, a MemAvailable kB, f MemFree kB, c Cached kB, sw swap used kB, ps/pf PSI some/full total us,
s10/f10 PSI some/full avg10, b2/b8 free bytes in buddy blocks >= 2 MiB / >= 8 MiB (all zones), hi highest
non-empty order, dt interval.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

PAGE = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096


def meminfo(text):
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        if key in ('MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree'):
            out[key] = int(rest.split()[0])
    return out


def psi(text):
    out = {}
    for line in text.splitlines():
        kind, *fields = line.split()
        values = dict(f.split('=') for f in fields)
        out[kind] = (float(values['avg10']), int(values['total']))
    return out


def buddy(text, page=PAGE):
    b2 = b8 = 0
    high = -1
    for line in text.splitlines():
        counts = [int(x) for x in line.split()[4:]]
        for order, count in enumerate(counts):
            size = page << order
            if count:
                high = max(high, order)
            if size >= 1 << 21:
                b2 += count * size
            if size >= 1 << 23:
                b8 += count * size
    return b2, b8, high


def sample(interval, root=Path('/proc')):
    m = meminfo((root / 'meminfo').read_text())
    p = psi((root / 'pressure/memory').read_text())
    b2, b8, high = buddy((root / 'buddyinfo').read_text())
    return dict(t=round(time.time(), 3), a=m['MemAvailable'], f=m['MemFree'], c=m['Cached'],
                sw=m['SwapTotal'] - m['SwapFree'], ps=p['some'][1], pf=p['full'][1], s10=p['some'][0],
                f10=p['full'][0], b2=b2, b8=b8, hi=high, dt=interval)


def read_rate(path, default):
    try:
        value = float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return default
    return min(10.0, max(0.05, value))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rate-file', type=Path, required=True)
    ap.add_argument('--stop-file', type=Path)
    ap.add_argument('--seconds', type=float, default=6 * 3600)
    a = ap.parse_args()
    ended = []
    signal.signal(signal.SIGTERM, lambda *_: ended.append(1))
    start = time.monotonic()
    interval = read_rate(a.rate_file, 1.0)
    checked = start
    with a.out.open('a', buffering=1) as out:
        out.write(json.dumps(dict(event='start', pid=os.getpid(), t=time.time(), interval=interval, page=PAGE)) + '\n')
        nxt = time.monotonic()
        while not ended and time.monotonic() - start < a.seconds:
            now = time.monotonic()
            if now - checked >= 1.0:
                checked = now
                new = read_rate(a.rate_file, interval)
                if new != interval:
                    interval = new
                    out.write(json.dumps(dict(event='rate', t=time.time(), interval=interval)) + '\n')
                if a.stop_file and a.stop_file.exists():
                    break
            out.write(json.dumps(sample(interval), separators=(',', ':')) + '\n')
            nxt += interval
            delay = nxt - time.monotonic()
            if delay < 0:
                nxt = time.monotonic()
            else:
                time.sleep(delay)
        out.write(json.dumps(dict(event='end', t=time.time())) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
