#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Later real routed-MTP three-GEMM check with independent gate/up globals.

Two real experts (0,255), TP4 slice, BF16 activations, FP32 eager oracle with
BF16 stage rounding; graph replay only after numerical checks. CPU cannot
qualify this kernel. Model draft acceptance remains a separate mandatory gate.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'overlay/overlay'),str(Path(__file__).resolve().parent)]
from more_nvfp4_gemm import rank_slice
import glm_nvfp4_format as fmt
import glm_nvfp4_more as more
import glm_nvfp4_attn as attn
import glm_nvfp4_mtp as mtp


def run(sidecar,rank,repeats):
    import torch
    from safetensors.torch import load_file
    if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('sm121 GB10 required')
    attn.check_sources();more.register();torch.backends.cuda.matmul.allow_tf32=False
    manifest=more.read_manifest(sidecar,['mtp']);parts={};refs={}
    for site in ('gate_proj','up_proj','down_proj'):
        tensors=[];reference=[]
        for expert in (0,255):
            name=f'model.layers.78.mlp.experts.{expert}.{site}';entry=manifest['tensors'][name]
            t=rank_slice(name,load_file(str(sidecar/entry['file'])),rank);tensors.append(t)
            reference.append(torch.from_numpy(fmt.dequant(t['weight_packed'].numpy(),t['weight_scale'].view(torch.uint8).numpy(),t['weight_global_scale'].item())).cuda())
        parts[site]={leaf:torch.stack([t[leaf] for t in tensors]).cuda() for leaf in tensors[0]}
        refs[site]=reference
    mod=torch.nn.Module();mod.params_dtype=torch.bfloat16;mod.num_experts=2;mod.global_num_experts=2
    mod.hidden_size=refs['gate_proj'][0].shape[1];mod.intermediate_size_per_partition=refs['gate_proj'][0].shape[0]
    mod.expert_map=None;mod.apply_router_weight_on_input=False
    for leaf in ('weight_packed','weight_scale','weight_global_scale'):
        v=torch.cat([parts[p][leaf] for p in ('gate_proj','up_proj')],dim=1)
        mod.register_parameter('w13_'+leaf,torch.nn.Parameter(v,requires_grad=False))
        v=parts['down_proj'][leaf]
        if leaf=='weight_global_scale':v=v.reshape(2)
        mod.register_parameter('w2_'+leaf,torch.nn.Parameter(v,requires_grad=False))
    for leaf in ('w13_input_global_scale','w2_input_global_scale'):
        mod.register_parameter(leaf,torch.nn.Parameter(torch.ones(2,device='cuda'),requires_grad=False))
    mtp.prepare_experts(mod);rows=[]
    for m in (1,3,16,512):
        torch.manual_seed(5306+m);x=torch.randn((m,mod.hidden_size),device='cuda',dtype=torch.bfloat16)*.05
        ids=torch.tensor([0,1],device='cuda',dtype=torch.int32).expand(m,-1).contiguous()
        weights=torch.tensor([.4,.6],device='cuda',dtype=torch.float32).expand(m,-1).contiguous()
        def fn():return mtp.apply_experts(mod,x,weights,ids)
        expected=[]
        for i in range(2):
            gate=(x.float()@refs['gate_proj'][i].T).to(torch.bfloat16)
            up=(x.float()@refs['up_proj'][i].T).to(torch.bfloat16)
            activated=(torch.nn.functional.silu(gate.float())*up.float()).to(torch.bfloat16)
            expected.append(((activated.float()@refs['down_proj'][i].T)*weights[:,i,None]).to(torch.bfloat16))
        reference=torch.stack(expected,dim=1).sum(1);got=fn()
        error=(got.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-20)
        if not torch.isfinite(got).all() or error.item()>.02:raise ValueError(f'real MTP independent-global error M{m}: {error.item()}')
        for _ in range(3):fn()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):graph.replay()
        end.record();end.synchronize()
        rows.append(dict(m=m,rank=rank,experts=[0,255],topk=2,nrmse=error.item(),nvfp4_us=begin.elapsed_time(end)*1000/repeats))
    return dict(scope=__doc__,passed=True,rows=rows,source=manifest['source'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--rank',type=int,choices=range(4),default=0)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=40);a=p.parse_args()
    if a.out.exists() or a.repeats<1:raise ValueError('fresh receipt, positive repeats required')
    a.out.write_text(json.dumps(run(a.sidecar,a.rank,a.repeats),indent=2)+'\n')
