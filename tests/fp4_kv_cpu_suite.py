# SPDX-License-Identifier: Apache-2.0
"""Pinned-image checks, real interpreter, and stock release CPU compatibility."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
source=Path(importlib.util.find_spec('vllm').origin).resolve().parent.parent
env=dict(os.environ,GLM_IMAGE_SRC=str(source),GLM_IMAGE_SRC_SECOND=str(source),
         GLM_KV_FORMAT='fp8',FP4_SIM_DIR='/receipts/day3/release-stack',FP4_DUMPS='/probe',FP4_RESULTS='/results')
checks=[('scripts/check_source_pins.py',str(source)),('tests/test_recipe.py',),('tests/test_launcher.py',),
        ('tests/test_persistent_cache.py',),('tests/test_fp4_kv.py',),('tests/test_fp4_kv_integration.py',),('tests/test_fp4_prefill_memory.py',),('tests/test_fp4_mla_prefill.py',),('tests/test_topk_exactness.py',),('tests/test_fp4_kv_admission.py',)]
for file,*args in checks:
    subprocess.run([sys.executable,'-B',str(ROOT/file),*args],env=env,check=True)
subprocess.run([sys.executable,'-B',str(ROOT/'tests/fp4_kv_triton_cpu.py')],
               env=dict(env,TRITON_INTERPRET='1'),check=True)
# Separate processes retain the stock K-stop/short-DSA/padding contracts.
subprocess.run([sys.executable,'-B',str(ROOT/'tests/kstop_compat_cpu_suite.py')],env=env,check=True)
Path('/results/fp4-kv-cpu.json').write_text(json.dumps(dict(status='PASS_PINNED_FP4_KV_CPU',
    source=str(source),checks=checks,interpreter='PASS',kstop_compat='PASS',
    scope='CPU only; no GPU correctness, replay or throughput qualification'),indent=2)+'\n')
