#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Later GB10 kernel check, no model boot/network. Run inside the pinned image."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/overlay'))
import glm_nvfp4_format as fmt
import glm_nvfp4_attn as attn


def main():
    import torch
    from safetensors.torch import load_file
    from safetensors import safe_open
    import vllm._custom_ops as ops
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--sidecar',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--layer',type=int,default=1)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--repeats',type=int,default=40)
    args=parser.parse_args()
    if torch.cuda.get_device_capability() != (12,1):raise RuntimeError('this receipt must be from GB10 sm121')
    if not 1<=args.layer<=77 or not 0<=args.rank<4 or args.repeats<1:raise ValueError('invalid arguments')
    attn.check_sources()
    manifest=fmt.read_manifest(args.sidecar)
    src=Path(args.checkpoint);index=src/'model.safetensors.index.json';idx=json.loads(index.read_text())['weight_map']
    if fmt.sha256(index)!=manifest['source']['index_sha256'] or fmt.sha256(src/'config.json')!=manifest['source']['config_sha256']:
        raise ValueError('checkpoint/sidecar mismatch')
    def read(name):
        with safe_open(src/idx[name],framework='pt',device='cpu') as sf:return sf.get_tensor(name)
    def matrices(proj):
        name=f'model.layers.{args.layer}.self_attn.{proj}'
        t=load_file(str(Path(args.sidecar)/manifest['tensors'][name]['file']))
        p=read(name+'.weight_packed');s=read(name+'.weight_scale');n,k=manifest['tensors'][name]['shape']
        if proj in ('q_b_proj','kv_b_proj'):
            sl=slice(n//4*args.rank,n//4*(args.rank+1));p=p[sl];s=s[sl]
            t['weight_packed']=t['weight_packed'][sl];t['weight_scale']=t['weight_scale'][sl]
        elif proj=='o_proj':
            p=p[:,args.rank*k//16:(args.rank+1)*k//16];s=s[:,args.rank*k//512:(args.rank+1)*k//512]
            t['weight_packed']=t['weight_packed'][:,args.rank*k//8:(args.rank+1)*k//8]
            t['weight_scale']=t['weight_scale'][:,args.rank*k//64:(args.rank+1)*k//64]
        return p.contiguous(),s.contiguous(),t
    ms=list(range(1,17))+[32,64,128,256,512,1024,2048,4096]
    receipt=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),algorithm=fmt.ALGORITHM,
        source=manifest['source'],rank=args.rank,rows=[])
    def timing(fn,x):
        for _ in range(3):fn(x)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn(x)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):graph.replay()
        end.record();end.synchronize()
        return start.elapsed_time(end)*1000/args.repeats
    for site,projs in [('qkv_a',['q_a_proj','kv_a_proj_with_mqa']),('q_b',['q_b_proj']),('o_proj',['o_proj']),('kv_b',['kv_b_proj'])]:
        matrices_cpu=[matrices(p) for p in projs];widths=[p.shape[0] for p,s,t in matrices_cpu]
        p=torch.cat([p for p,s,t in matrices_cpu]).cuda();s=torch.cat([s for p,s,t in matrices_cpu]).cuda()
        n,k=p.shape[0],p.shape[1]*4;empty=torch.empty(0,dtype=torch.int32,device='cuda')
        w8=ops.gptq_marlin_repack(p.T.contiguous(),empty,k,n,8,False)
        sc8=mu.marlin_permute_scales(s.T.contiguous(),k,n,128)
        workspace=mu.marlin_make_workspace_new(torch.device('cuda',torch.cuda.current_device()))
        mod=torch.nn.Module();mod.logical_widths=widths;mod.params_dtype=torch.bfloat16
        mod.output_size_per_partition=n;mod.input_size_per_partition=k
        mod.weight_packed=torch.nn.Parameter(torch.cat([t['weight_packed'] for p,s,t in matrices_cpu]).cuda(),requires_grad=False)
        mod.weight_scale=torch.nn.Parameter(torch.cat([t['weight_scale'] for p,s,t in matrices_cpu]).cuda(),requires_grad=False)
        mod.weight_global_scale=torch.nn.Parameter(torch.cat([t['weight_global_scale'] for p,s,t in matrices_cpu]).cuda(),requires_grad=False)
        attn.prepare_layer(mod)
        refs=[torch.from_numpy(fmt.dequant(t['weight_packed'].numpy(),t['weight_scale'].view(torch.uint8).numpy(),t['weight_global_scale'].item())) for p,s,t in matrices_cpu]
        reference=torch.cat(refs).to(device='cuda',dtype=torch.bfloat16)
        fn8=lambda x:mu.apply_gptq_marlin_linear(x,w8,sc8,empty,empty,empty,workspace,
                    scalar_types.uint8b128,n,k,True,input_dtype=None,use_fp32_reduce=True)
        fn4=lambda x:attn.apply_layer(mod,x)
        for m in ms:
            torch.manual_seed(314+m);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.05
            got=fn4(x);expected=x@reference.T
            error=(got.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-20)
            if not torch.isfinite(got).all() or error.item()>.02:raise RuntimeError(f'{site} M{m} Marlin QDQ reference mismatch {error.item()}')
            # Alternate ordering to limit systematic warm cache/order bias; this is a kernel check,
            # not the full-model dirty-L2 cycle estimate.
            if m%2:t4=timing(fn4,x);t8=timing(fn8,x)
            else:t8=timing(fn8,x);t4=timing(fn4,x)
            row=dict(site=site,m=m,n=n,k=k,int8_us=t8,nvfp4_us=t4,nrmse=error.item())
            receipt['rows'].append(row);print(json.dumps(row),flush=True)
        del mod,reference,p,s,w8,sc8,matrices_cpu,refs
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
