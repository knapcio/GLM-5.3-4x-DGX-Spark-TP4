#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run every Mac CPU test file offline, preserving failures/skips per suite.

Use an already-installed Python with torch/numpy/safetensors and saved source
extracts. No dependency installation, SSH, fleet, image pull or GPU execution.
Local Gloo tests may be blocked by a sandbox which denies socket bind.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--sim-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True)
    env=dict(os.environ,GLM_IMAGE_SRC=str(a.source.resolve()),FP4_SIM_DIR=str(a.sim_dir.resolve()),
             CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    checks=[ROOT/'scripts/check_source_pins.py']+sorted((ROOT/'tests').glob('test_*.py'))+[ROOT/'roce/tests/test_glm_roce_cpu.py']
    rows=[]
    for file in checks:
        args=[sys.executable,'-B',str(file)]
        if file.name=='check_source_pins.py':args.append(str(a.source.resolve()))
        started=time.monotonic();log=a.out/(file.stem+'.log')
        with log.open('x') as f:
            try:rc=subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
            except subprocess.TimeoutExpired:rc=124
        text=log.read_text();count=re.search(r'Ran (\d+) tests?',text);skips=re.search(r'skipped=(\d+)',text)
        row=dict(suite=str(file.relative_to(ROOT)),rc=rc,seconds=round(time.monotonic()-started,3),
                 cases=int(count[1]) if count else None,skips=int(skips[1]) if skips else 0)
        rows.append(row);print(json.dumps(row),flush=True)
        (a.out/'mac-tests.json').write_text(json.dumps(dict(python=sys.executable,source=str(a.source),
           scope='Mac CPU only; full logs record blocked sockets and platform/input skips',suites=rows),indent=2)+'\n')
    return 1 if any(r['rc'] for r in rows) else 0


if __name__=='__main__':raise SystemExit(main())
