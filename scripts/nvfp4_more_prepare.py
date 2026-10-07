#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mac/local-only packages and all-rank DRY commands; no SSH or fleet execution."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import stress_step as S

ARMS={'B':'attn','S':'attn,shared','SD':'attn,shared,dense','M':'attn,shared,mtp','I':'attn,shared,mtp,indexer'}


def prepare(a):
    a.out=a.out.resolve()
    if a.out==S.ROOT or S.ROOT in a.out.parents:
        raise ValueError('packages must be outside the source checkout')
    a.out.mkdir() # fresh no-clobber window
    records={}
    for arm,selected in ARMS.items():
        if a.with_dense and arm in ('M','I'):selected+=',dense'
        folder=a.out/arm
        S.prepare(argparse.Namespace(reference_clone=a.reference_clone,out=folder,
            boot=f'glm53full-kvhr-nvfp4more-{a.tag}-{arm.lower()}',extra_gib=0,maxlen=98176,
            c4_mode='pool-quarter',cache_policy=False,cycle_probe=S.ROOT/'bench/cycle.py',
            gate_metrics=S.ROOT/'bench/gate_metrics.py',loader=None,sidecar=a.attn_sidecar,param_hash=False))
        env=folder/'clone/.env'
        text=env.read_text()+'\nunset GLM_NVFP4_WSIM GLM_LOADER GLM_PARAM_HASH VLLM_SERVER_DEV_MODE\n'
        if arm=='B':text+='unset GLM_NVFP4_GROUPS GLM_NVFP4_MORE_DIR\n'
        else:text+='export GLM_NVFP4_GROUPS='+shlex.quote(selected)+' GLM_NVFP4_MORE_DIR='+shlex.quote(str(a.more_sidecar))+'\n'
        env.write_text(text)
        result=subprocess.run(['bash',str(folder/'launcher.sh'),'serve'],env=dict(os.environ,DRY='1'),
                              capture_output=True,text=True,timeout=120,check=True)
        (folder/'dry.txt').write_text(result.stdout)
        # Refresh hashes after selector injection. The same source guard/cap/pools are preserved.
        m=json.loads((folder/'step.json').read_text())
        m.update(groups=selected,dry_sha256=S.sha(folder/'dry.txt'),package_sha256=S.package_hash(folder/'clone'))
        m['benchmark_sha256']={str(p.relative_to(folder/'clone')):S.sha(p)
            for p in sorted((folder/'clone/bench').glob('*.py'))}
        (folder/'step.json').write_text(json.dumps(m,indent=2)+'\n')
        records[arm]=m
    result=dict(scope='offline packages only; coordinator owns stop/hold/restore; no unattended arm advancement',
        arms=records,baseline='B-current attn only',predecessors={'S':'B','SD':'S','M':'SD' if a.with_dense else 'S','I':'M'},
        with_dense=a.with_dense,attn_sidecar=str(a.attn_sidecar),more_sidecar=str(a.more_sidecar),
        restore='Use B package through existing serving coordinator handoff; preserve recorded production watch/guard identity')
    (a.out/'window.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Prepared',a.out)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-clone',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--attn-sidecar',type=Path,default=Path('/srv/glm/models/GLM-5.3-attn-nvfp4-87cf357'))
    p.add_argument('--more-sidecar',type=Path,default=Path('/srv/glm/models/GLM-5.3-more-nvfp4-20261006'))
    p.add_argument('--with-dense',action='store_true');a=p.parse_args();prepare(a)
