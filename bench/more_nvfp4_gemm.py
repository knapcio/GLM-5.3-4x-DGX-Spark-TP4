#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Later sm121 REAL W4A16 checker, full group inventory, sequential rank slices.

Eager FP32 reference first, then graph timings. No model boot or network.
This linear screen does not replace native MTP routed-GEMM/model acceptance gates.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/overlay'))
import glm_nvfp4_format as fmt
import glm_nvfp4_attn as attn
import glm_nvfp4_more as more


def rank_slice(name,tensors,rank):
    # Whole logical globals are never recomputed after TP slicing.
    if '.indexer.' in name or name.endswith(('q_a_proj','kv_a_proj_with_mqa')):
        return tensors
    result=dict(tensors);n,k=tensors['weight_packed'].shape[0],tensors['weight_packed'].shape[1]*2
    if name.endswith(('down_proj','o_proj')):
        result['weight_packed']=tensors['weight_packed'][:,rank*k//8:(rank+1)*k//8].contiguous()
        result['weight_scale']=tensors['weight_scale'][:,rank*k//64:(rank+1)*k//64].contiguous()
    else:
        result['weight_packed']=tensors['weight_packed'][rank*n//4:(rank+1)*n//4].contiguous()
        result['weight_scale']=tensors['weight_scale'][rank*n//4:(rank+1)*n//4].contiguous()
    return result


if __name__=='__main__':
    import torch
    from safetensors.torch import load_file
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sidecar',type=Path,required=True)
    p.add_argument('--group',choices=('shared','dense','indexer','mtp'),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rank',type=int,choices=range(4),default=0);p.add_argument('--repeats',type=int,default=40);a=p.parse_args()
    if a.out.exists() or a.repeats<1:raise ValueError('fresh receipt and positive repeats required')
    if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('sm121 GB10 required')
    attn.check_sources();more.register()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=more.read_manifest(a.sidecar,[a.group]);rows=[]
    for name,entry in manifest['tensors'].items():
        if entry['group']!=a.group:continue
        t=rank_slice(name,load_file(str(a.sidecar/entry['file'])),a.rank)
        n,k=t['weight_packed'].shape[0],t['weight_packed'].shape[1]*2
        reference=torch.from_numpy(fmt.dequant(t['weight_packed'].numpy(),t['weight_scale'].view(torch.uint8).numpy(),t['weight_global_scale'].item())).cuda()
        mod=torch.nn.Module();mod.params_dtype=torch.bfloat16;mod.logical_widths=[n]
        mod.input_size_per_partition=k;mod.output_size_per_partition=n
        for leaf,value in t.items():mod.register_parameter(leaf,torch.nn.Parameter(value.cuda(),requires_grad=False))
        attn.prepare_layer(mod)
        for m in tuple(range(1,17))+(32,64,128,256,512,1024,2048,4096):
            torch.manual_seed(5306+m);x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.05
            got=attn.apply_layer(mod,x);expected=x.float()@reference.T
            error=(got.float()-expected).norm()/expected.norm().clamp_min(1e-20)
            if not torch.isfinite(got).all() or error.item()>.02:raise ValueError(f'{name} M{m} real FP32 reference error {error.item()}')
            for _ in range(3):attn.apply_layer(mod,x)
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):attn.apply_layer(mod,x)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.repeats):graph.replay()
            end.record();end.synchronize()
            row=dict(name=name,rank=a.rank,m=m,n=n,k=k,nrmse=error.item(),nvfp4_us=start.elapsed_time(end)*1000/a.repeats)
            rows.append(row);print(json.dumps(row),flush=True)
        del mod,reference,t,x,got,expected,graph
        torch.cuda.empty_cache()
    a.out.write_text(json.dumps(dict(scope=__doc__,passed=True,rows=rows,source=manifest['source'],group=a.group,rank=a.rank),indent=2)+'\n')
