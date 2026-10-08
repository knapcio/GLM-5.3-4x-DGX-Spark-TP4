#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Atomically write a central scheduler sidecar. No network or worker RPC."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def write(path, chunk, sequence, decode_steps=None):
    if type(chunk) is not int or chunk not in (0, 512, 1024, 2048, 4096):
        raise ValueError('chunk must be 0/512/1024/2048/4096')
    if type(sequence) is not int or sequence < 0:
        raise ValueError('sequence must be nonnegative')
    if decode_steps is not None and (type(decode_steps) is not int or decode_steps < 0):
        raise ValueError('decode_steps must be a nonnegative integer')
    if path.exists():
        prior = json.loads(path.read_text())
        if sequence <= prior['sequence']:
            raise ValueError('sequence must increase')
    payload = dict(schema=1, chunk=chunk, sequence=sequence)
    if decode_steps is not None:
        payload.update(schema=2, decode_steps=decode_steps)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as f:
            name = f.name
            f.write(json.dumps(payload, sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name is not None and Path(name).exists(): Path(name).unlink()
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path); p.add_argument('--chunk', required=True, type=int)
    p.add_argument('--sequence', required=True, type=int)
    p.add_argument('--decode-steps', type=int, help='schema 2 pure-decode steps between mixed chunks')
    a = p.parse_args()
    print(json.dumps(write(a.path, a.chunk, a.sequence, a.decode_steps), sort_keys=True))


if __name__ == '__main__': main()
