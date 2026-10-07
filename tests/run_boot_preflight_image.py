#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mac/Colima historical regression runner; no SSH, network, or GPUs.

--out must be an existing directory on a Colima-visible filesystem.
Real headers are intentionally unavailable in these regression probes; the
header checks must fail explicitly, while construction/hook/API checks run.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import boot_preflight as B


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',default=B.BASE_IMAGE)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();started=time.monotonic()
    a.out=a.out.resolve();a.out.mkdir(exist_ok=True)
    results=[]
    for revision in ('88f27fb','613bf73','current'):
        with tempfile.TemporaryDirectory(prefix='.preflight-regression-',dir=ROOT) as d:
            root=Path(d)
            if revision=='current':root=ROOT
            else:
                archive=subprocess.check_output(['git','archive',revision],cwd=ROOT)
                subprocess.run(['tar','-x','-C',str(root)],input=archive,check=True)
                for name in ('boot_preflight.py','checkpoint_headers.py'):shutil.copy2(ROOT/'scripts'/name,root/'scripts'/name)
            out=a.out/(revision+'.json');log=a.out/(revision+'.log')
            cmd=[sys.executable,str(ROOT/'scripts/boot_preflight.py'),'--root',str(root),
                 '--dry',str(ROOT/'tests/fixtures/boot_preflight_cand2.dry.txt'),
                 '--config',str(ROOT/'tests/fixtures/boot_preflight_glm53_config.json'),
                 '--image',a.image,'--out',str(out)]
            if revision!='current':cmd+=['--rank','0']
            with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=125)
            assert r.returncode==1,'missing headers must refuse admission'
            result=json.loads(out.read_text())
            for row in result['ranks']:
                checks={x['check']:x for x in row['report']['checks']}
                for name in ('settings','target+mtp-meta-construction','adaptive-chunk-startup','process-weights-hooks'):
                    assert checks[name]['status']=='PASS',(revision,row['rank'],checks[name])
                for name in ('checkpoint-headers','parameter-contracts'):assert checks[name]['status']=='FAIL'
                c=checks['kstop-pad-hygiene'];d=checks['mtp-apply-api']
                if revision=='88f27fb':assert c['status']=='FAIL' and 'kstop dead rows require modular Marlin topk hook' in c['error']
                else:assert c['status']=='PASS',c
                if revision=='current':assert d['status']=='PASS',d
                else:assert d['status']=='FAIL' and "has no attribute 'silu_and_mul'" in d['error'],d
            assert result['seconds']<120,result['seconds']
            results.append(dict(revision=revision,seconds=result['seconds'],ranks=len(result['ranks']),status='PASS'))
            print(revision,results[-1],flush=True)
    (a.out/'regressions.json').write_text(json.dumps(dict(ok=True,results=results,seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':main()
