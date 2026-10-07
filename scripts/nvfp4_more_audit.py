#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline reference audit of real group sidecars; CPU only, original shards read-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'overlay/overlay'),str(Path(__file__).resolve().parents[1]/'tests/fixtures')]
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import glm_nvfp4_format as fmt
import glm_nvfp4_groups as selection
import glm_nvfp4_more as loader
import nvfp4_wsim_reference as reference
from convert_more_nvfp4 import convert,source_file


def audit(src,dst,partial=False,rows=64):
    src,dst=Path(src),Path(dst)
    m=json.loads((dst/'manifest.json').read_text())
    if not partial:loader.read_manifest(dst,m['groups'])
    idx=json.loads((src/'model.safetensors.index.json').read_text())['weight_map']
    expected=selection.inventory(idx,m['groups'])
    if not partial and set(m['tensors'])!=set(expected):raise ValueError('inventory mismatch')
    source=dict(index_sha256=fmt.sha256(src/'model.safetensors.index.json'),config_sha256=fmt.sha256(src/'config.json'),
                shards={s:fmt.sha256(source_file(src,s)) for s in m['source']['shards']})
    if source!=m['source']:raise ValueError('original source hash mismatch')
    results=[]
    for name,e in m['tensors'].items():
        path=dst/e['file']
        if fmt.sha256(path)!=e['sha256']:raise ValueError('sidecar hash mismatch')
        data=load_file(str(path));fmt.validate_tensors(data,e['shape']);n,k=e['shape']
        bf16=e['group']=='indexer';leaf='weight' if bf16 else 'weight_packed'
        if not bf16:
            with safe_open(str(source_file(src,idx[name+'.weight_scale'])),framework='pt') as sf:
                scales=sf.get_tensor(name+'.weight_scale')
        with safe_open(str(source_file(src,idx[name+'.'+leaf])),framework='pt') as sf:
            p=sf.get_slice(name+'.'+leaf)
            def decode(a,b):
                if bf16:return p[a:b].float().numpy()
                sc=scales[a:b]
                if e['group']=='mtp':sc=sc.expand(-1,k//128).contiguous()
                return fmt.decode_int8(p[a:b],sc)
            amax=np.float32(0)
            for a in range(0,n,rows):amax=max(amax,np.max(np.abs(decode(a,min(n,a+rows)))))
            ts=np.float32(amax/np.float32(2688))
            if ts.view(np.uint32)!=np.float32(data['weight_global_scale'].item()).view(np.uint32):raise ValueError('global mismatch')
            mismatch=0;signal=0.;error=0.
            for a in range(0,n,rows):
                b=min(n,a+rows);w=decode(a,b)
                q=fmt.dequant(data['weight_packed'][a:b].numpy(),data['weight_scale'][a:b].view(torch.uint8).numpy(),ts)
                oracle=reference.qdq(w,ts)
                mismatch+=int(np.count_nonzero(q.view(np.uint32)!=oracle.view(np.uint32)))
                signal+=float(np.sum(w.astype(np.float64)**2));error+=float(np.sum((w.astype(np.float64)-q)**2))
            if mismatch:raise ValueError('reference mismatch: '+name)
        results.append(dict(name=name,group=e['group'],shape=[n,k],elements=n*k,bit_mismatches=mismatch,
            weight_qdq_nrmse=(error/signal)**.5 if signal else 0,sha256=e['sha256']))
        print(name,'bit-exact',flush=True)
    after={s:fmt.sha256(source_file(src,s)) for s in source['shards']}
    if after!=source['shards']:raise ValueError('source changed during audit')
    return dict(scope='CPU reference audit; partial cannot serve' if partial else 'CPU full group reference audit; GPU/model qualification pending',
        passed=True,source_unchanged=True,reference_sha256=fmt.sha256(reference.__file__),source=source,
        elements=sum(r['elements'] for r in results),counts={g:sum(r['group']==g for r in results) for g in m['groups']},tensors=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('sidecar',type=Path)
    p.add_argument('--sample-report',type=Path,help='convert only known local samples; unservable partial output')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    if a.out.exists():raise ValueError('fresh audit receipt required')
    if a.sample_report:
        report=json.loads(a.sample_report.read_text())
        names=[r['name'].rsplit('.',1)[0] for r in report['tensors'] if r['group']!='attn']
        convert(a.checkpoint,a.sidecar,groups=('shared','indexer','dense','mtp'),strict=False,sample_modules=names)
    result=audit(a.checkpoint,a.sidecar,partial=bool(a.sample_report))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
