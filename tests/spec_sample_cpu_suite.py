# SPDX-License-Identifier: Apache-2.0
"""Pinned-image CPU suite entry: no downloads, GPU, or remote operations."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(importlib.util.find_spec('vllm').origin).resolve().parent.parent
env = dict(os.environ, GLM_IMAGE_SRC=str(SOURCE), GLM_SPEC_SAMPLE='0')
checks = [
    ('scripts/check_source_pins.py', str(SOURCE)),
    ('tests/test_recipe.py',),
    ('tests/test_launcher.py',),
    ('tests/test_spec_sample.py',),
    ('tests/spec_sample_compile.py',),
    ('tests/spec_sample_import.py',),
]
for name, *args in checks:
    print('CPU CHECK ' + name, flush=True)
    subprocess.run([sys.executable, '-B', str(ROOT/name), *args], check=True, env=env)
print('CPU SUITE PASS', flush=True)
