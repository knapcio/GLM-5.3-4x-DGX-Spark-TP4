#!/usr/bin/env python3
"""Read-only host memory sampler for a future loader boot; run locally on each Spark."""
import argparse
import json
from pathlib import Path
import signal
import time


def sample(proc=Path('/proc')):
    wanted = {'MemTotal', 'MemAvailable', 'Cached', 'Dirty', 'Writeback', 'SwapTotal', 'SwapFree'}
    values = {}
    for line in (proc / 'meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in wanted:
            values[key + '_bytes'] = int(value.split()[0]) * 1024
    if 'MemAvailable_bytes' not in values:
        raise RuntimeError('MemAvailable missing')
    values['used_excluding_available_bytes'] = values['MemTotal_bytes'] - values['MemAvailable_bytes']
    values['wall_epoch'] = time.time()
    values['monotonic'] = time.monotonic()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=range(4), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error('--seconds must be 0..3600')
    stopped = [False]
    def stop(*_): stopped[0] = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.seconds
    with args.out.open('x') as f:
        while not stopped[0] and time.monotonic() < deadline:
            f.write(json.dumps(dict(rank=args.rank, **sample()), sort_keys=True) + '\n')
            f.flush()
            time.sleep(.1)


if __name__ == '__main__':
    main()
