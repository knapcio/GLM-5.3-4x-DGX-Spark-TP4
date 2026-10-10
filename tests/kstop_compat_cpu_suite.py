# SPDX-License-Identifier: Apache-2.0
"""Offline pinned-image checks, with separate real-control-flow processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(importlib.util.find_spec('vllm').origin).resolve().parent.parent
ENV=dict(os.environ,GLM_IMAGE_SRC=str(SOURCE),GLM_MTP_KSTOP='0',GLM_PAD_HYGIENE='0',GLM_MTP_KSTOP_CAPTURE_LAYOUT='m12',GLM_INDEXER_SHORTCUT='0',GLM_SPEC_SAMPLE='0')
checks=[('scripts/check_source_pins.py',str(SOURCE)),('tests/test_recipe.py',),('tests/test_launcher.py',),
        ('tests/test_kstop.py',),('tests/test_dsa_short.py',),('tests/test_kstop_compat.py',),('tests/test_pad_hygiene.py',),
        ('tests/test_spec_sample.py',),('tests/test_kstop_spec_distribution.py',),
        ('tests/spec_sample_compile.py',),('tests/spec_sample_import.py',)]
checks.append(('tests/kstop_uniform_rank_cpu.py',))
checks.append(('tests/kstop_specsample_rank_cpu.py','--gloo'))
for name,*args in checks:
    print('CPU CHECK '+name,flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/name),*args],env=ENV,check=True)
for uniform in ('0','1','k2'):
    for sample in ('0','1'):
        for layout in (('m12','reuse') if uniform=='k2' else ('reuse',)):
            for pad in ('0','1'):
                out=Path('/results')/f'controlflow-u{uniform}-s{sample}-{layout}-p{pad}.json'
                subprocess.run([sys.executable,'-B',str(ROOT/'tests/kstop_compat_controlflow.py'),
                    '--campaign','/receipts/day3/mtp-kstop','--uniform',uniform,'--sample',sample,
                    '--capture-layout',layout,'--pad-hygiene',pad,'--out',str(out)],env=ENV,check=True)
            off=json.loads((Path('/results')/f'controlflow-u{uniform}-s{sample}-{layout}-p0.json').read_text())
            on=json.loads((Path('/results')/f'controlflow-u{uniform}-s{sample}-{layout}-p1.json').read_text())
            assert off['token_ids']==on['token_ids'],'T=0 token identity failed'
Path('/results/PASS.json').write_text(json.dumps(dict(status='PASS_CPU_KSTOP_COMPAT',source=str(SOURCE),
    checks=checks,controlflow=dict(shortcut='1',uniform=['0','1','k2'],sample=['0','1'],capture_layout=['m12','reuse'],pad_hygiene=['0','1'],temperature_zero_identity='PASS'),
    scope='CPU only; real models meta-built in the real fix19 control-flow harness; GPU tensor math substituted'),indent=2)+'\n')
print('CPU KSTOP COMPAT SUITE PASS',flush=True)
