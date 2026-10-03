#!/usr/bin/env python3
"""Render only: every token of the launch differs solely by GLM_SPEC_SAMPLE."""
import difflib
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def compare():
    vectors = []
    for flag in ('0', '1'):
        result = subprocess.check_output([str(ROOT/'start.sh'), 'serve'], text=True,
                                        env=dict(os.environ, DRY='1', GLM_SPEC_SAMPLE=flag,
                                                 CTN='glm53full-specsample-dry'))
        vectors.append([shlex.split(line) for line in result.splitlines() if line.startswith('docker run ')])
    assert len(vectors[0]) == len(vectors[1]) == 8
    for a, b in zip(*vectors):
        assert a.count('GLM_SPEC_SAMPLE=0') == b.count('GLM_SPEC_SAMPLE=1') == 1
        assert ['GLM_SPEC_SAMPLE=1' if t == 'GLM_SPEC_SAMPLE=0' else t for t in a] == b
    diff = '\n'.join(difflib.unified_diff(
        [shlex.join(v) for v in vectors[0]], [shlex.join(v) for v in vectors[1]],
        fromfile='GLM_SPEC_SAMPLE=0', tofile='GLM_SPEC_SAMPLE=1', lineterm=''))
    return diff


if __name__ == '__main__':
    diff = compare()
    print('DRY SPEC SAMPLE PASS: all 8 commands; only new env flag differs')
    print(diff)
