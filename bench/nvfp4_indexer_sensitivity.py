#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline indexer risk screen: real weights, synthetic activations, isolated top-k.

This omits trained activation distributions, RoPE, FP8 index-cache rounding and
runtime scoring. It demonstrates sensitivity, not long-context qualification.
Real selected-index overlap must be collected from identical runtime activations.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/overlay'))
import numpy as np
import torch
from safetensors import safe_open
import glm_nvfp4_format as fmt


def screen(src,layer=1,tokens=4096,queries=8,topk=2048):
    src=Path(src);idx=json.loads((src/'model.safetensors.index.json').read_text())['weight_map']
    torch.set_num_threads(1);rng=np.random.default_rng(5306);weights={};qdq={};stats={}
    for leaf in ('wq_b','wk','weights_proj'):
        name=f'model.layers.{layer}.self_attn.indexer.{leaf}.weight'
        with safe_open(str(src/idx[name]),framework='pt') as sf:
            original=sf.get_tensor(name)
        if original.dtype!=torch.bfloat16:raise ValueError('expected BF16 indexer source')
        w=original.float().numpy();ts=np.float32(np.max(np.abs(w))/np.float32(2688));v,s=fmt.encode(w,ts);q=fmt.dequant(v,s,ts)
        weights[leaf]=torch.from_numpy(w);qdq[leaf]=torch.from_numpy(q)
        stats[leaf]=dict(shape=list(w.shape),nrmse=float(np.linalg.norm(w-q)/np.linalg.norm(w)),
            source_tensor_sha256=__import__('hashlib').sha256(original.view(torch.uint8).numpy().tobytes()).hexdigest())
    hidden=torch.from_numpy(rng.standard_normal((tokens,weights['wk'].shape[1]),dtype=np.float32))
    latent=torch.from_numpy(rng.standard_normal((queries,weights['wq_b'].shape[1]),dtype=np.float32))
    nheads=weights['weights_proj'].shape[0];dim=weights['wk'].shape[0]
    def scores(ws):
        q=(latent@ws['wq_b'].T).view(queries,nheads,dim)
        k=torch.nn.functional.layer_norm(hidden@ws['wk'].T,(dim,))
        h=(hidden[:queries]@ws['weights_proj'].T)*(nheads**-.5)*(dim**-.5)
        return (torch.relu(torch.einsum('qhd,td->qht',q,k))*h[:,:,None]).sum(1)
    before,after=scores(weights),scores(qdq)
    rows=[]
    for i in range(queries):
        ordered=torch.sort(before[i],descending=True).values
        selected=set(torch.topk(before[i],topk).indices.tolist());candidate=set(torch.topk(after[i],topk).indices.tolist())
        gap=float(ordered[topk-1]-ordered[topk]);delta=float((after[i]-before[i]).abs().max())
        rows.append(dict(query=i,topk_overlap=len(selected&candidate)/topk,swapped=len(selected-candidate),
            boundary_gap=gap,max_score_perturbation=delta,sufficient_stability_margin=gap>2*delta))
    return dict(scope=__doc__,seed=5306,tokens=tokens,queries=queries,topk=topk,weights=stats,rows=rows,
        mean_topk_overlap=sum(r['topk_overlap'] for r in rows)/queries,
        recommendation='Indexer opt-in only. Do not ship from weight RMS or simulator quality; require REAL-path long-context retrieval and runtime selected-index overlap.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh sensitivity receipt required')
    a.out.write_text(json.dumps(screen(a.checkpoint),indent=2)+'\n')
