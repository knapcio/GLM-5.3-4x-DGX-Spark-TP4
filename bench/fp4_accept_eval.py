#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline fixed-query acceptance diagnostics; no serving or network imports.

Token ages follow first writes within each single-request capture epoch. Slot
overwrites retain positions (speculative rejected tails). Missing rows are
counted and excluded from attention diagnostics, never filled with zero.
Output error is a proxy, never a measured change in MTP acceptance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/fixtures'))
from fp4_probe_quant_dc5ad9e import fp4_qdq, fp8_codes


def levels(v):
    c = torch.zeros_like(v)
    for mid, upper, up in ((.25,.5,False),(.75,1.,True),(1.25,1.5,False),
                           (1.75,2.,True),(2.5,3.,False),(3.5,4.,True),(5.,6.,False)):
        c = torch.where(v >= mid if up else v > mid, upper, c)
    return c


def variant(x, mode):
    width = 8 if mode == 'block8' else 16
    z = x.float().reshape(-1, 512 // width, width)
    a = z.abs().amax((-2,-1), keepdim=True)
    row = torch.exp2(torch.ceil(torch.log2(torch.where(a > 0, a/2688., 1.))))
    if mode == 'free_row':
        row = torch.where(a > 0, a/2688., 1.)
    s = z.abs().amax(-1, keepdim=True) / 6.
    if mode == 'bf16_scale':
        s = s.bfloat16().float()
    elif mode != 'fp32_scale':
        s = (s/row).clamp(0,448).to(torch.float8_e4m3fn).float() * row
    c = torch.copysign(levels(z.abs()/torch.where(s>0,s,1.)),z)
    if mode == 'mse_scale':
        # One least-squares update; retain the old scale if discrete recoding
        # increases SSE. Existing FP8 block bytes and FP32 row remain unchanged.
        fit = (z*c).sum(-1,keepdim=True)/torch.where(c.square().sum(-1,keepdim=True)>0,
                                                       c.square().sum(-1,keepdim=True),1.)
        fit = (fit/row).clamp(0,448).to(torch.float8_e4m3fn).float()*row
        cc = torch.copysign(levels(z.abs()/torch.where(fit>0,fit,1.)),z)
        better = (z-cc*fit).square().sum(-1,keepdim=True) < (z-c*s).square().sum(-1,keepdim=True)
        c,s = torch.where(better,cc,c),torch.where(better,fit,s)
    return (c*s).reshape_as(x).bfloat16()


class Error:
    def __init__(self): self.ee,self.xx,self.n = 0.,0.,0
    def add(self,x,y):
        self.ee += float((y.float()-x.float()).square().sum())
        self.xx += float(x.float().square().sum()); self.n += x.numel()
    def result(self):
        return dict(n=self.n,sse=self.ee,reference_energy=self.xx,
                    nrmse=(self.ee/self.xx)**.5 if self.xx else None)


def evaluate(captures):
    torch.set_num_threads(2)
    groups = defaultdict(list); inventory=[]
    for path in sorted(captures.glob('*.pt')):
        e = torch.load(path, weights_only=True, map_location='cpu')
        groups[e['case'],e['epoch'],e['layer_id']].append((e['seq'],path,e['stage']))
        inventory.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    modes=('fp4x','free_row','bf16_scale','fp32_scale','block8','mse_scale')
    reports=[]
    for (case,epoch,layer),files in sorted(groups.items()):
        caches={m:torch.zeros(66496,512,dtype=torch.bfloat16) for m in ('bf16','fp8',*modes)}
        ropes=torch.zeros(66496,64,dtype=torch.bfloat16)
        positions=torch.full((66496,),-1,dtype=torch.int64); cursor=0
        lastprefill=max((s for s,p,stage in files if stage=='prefill'), default=-1)
        stats=defaultdict(Error); ages=defaultdict(lambda:dict(mass=0.,selected=0,heads=0,pv_sse=0.,qk_sse=0.,
            fp4_fp8_sse=0.,fp8_energy=0.,fp4_bf16_sse=0.,fp8_bf16_sse=0.,bf16_energy=0.))
        missing=0; queries=0; kinds=defaultdict(int)
        for seq,path,stage in sorted(files):
            e=torch.load(path,weights_only=True,map_location='cpu')
            x=e['latent']; slots=e['write_slots'].long(); live=slots>=0
            scale=float(e['k_scale']); rs=e.get('reader_k_scale',scale)
            ys={'bf16':x,'fp8':(fp8_codes(x,scale)*rs).bfloat16(),
                'fp4x':fp4_qdq(x),**{m:variant(x,m) for m in modes if m!='fp4x'}}
            for i in live.nonzero().flatten().tolist():
                slot=int(slots[i])
                if positions[slot]<0: positions[slot]=cursor; cursor+=1
                for m,y in ys.items(): caches[m][slot]=y[i]
                ropes[slot]=fp8_codes(e['rope'][i],scale).bfloat16()
            for m in modes:
                stats['reconstruction_bf16',m,stage].add(x[live],ys[m][live])
                stats['reconstruction_fp8',m,stage].add(ys['fp8'][live],ys[m][live])
            if stage=='prefill' and seq!=lastprefill: continue
            # Last captured query, real backend inputs; all eight decode cycles.
            j=len(e['attend_slots'])-1; attend=e['attend_slots'][j].long()
            qslot=int(slots[int(e['query_rows'][j])])
            if not len(attend) or qslot<0 or (positions[attend]<0).any(): missing+=1; continue
            age=positions[qslot]-positions[attend]
            if (age<0).any(): raise ValueError('noncausal/ambiguous position mapping')
            q=e['q_nope'][j].float(); qr=e['q_pe'][j].float(); s=e['softmax_scale']
            rope_logits=qr @ ropes[attend].float().T
            def attention(y):
                w=torch.softmax((q @ y.float().T+rope_logits)*s,-1)
                return w,w @ y.float()
            x8=caches['fp8'][attend]; x4=caches['fp4x'][attend]
            w8,o8=attention(x8); w4,o4=attention(x4)
            wuv=e['w_uv'].float()
            def proj(o): return torch.bmm(o[:,None,:],wuv).squeeze(1)
            p8=proj(o8)
            for m in modes:
                _,o=attention(caches[m][attend])
                stats['attention_latent',m,stage].add(o8,o)
                stats['attention_projected',m,stage].add(p8,proj(o))
            for window in (2048,4096,8192):
                recent=age<window
                _,o=attention(torch.where(recent[:,None],x8,x4))
                stats['attention_latent',f'recent{window}',stage].add(o8,o)
                stats['attention_projected',f'recent{window}',stage].add(p8,proj(o))
            for label,lo,hi in (('0-2k',0,2048),('2k-4k',2048,4096),('4k-8k',4096,8192),('8k+',8192,10**9)):
                mask=(age>=lo)&(age<hi); a=ages[stage,label]
                a['mass']+=float(w8[:,mask].sum());a['selected']+=int(mask.sum());a['heads']+=q.shape[0]
                original=caches['bf16'][attend][mask].float()
                a['fp4_fp8_sse']+=float((x4[mask].float()-x8[mask].float()).square().sum())
                a['fp8_energy']+=float(x8[mask].float().square().sum())
                a['fp4_bf16_sse']+=float((x4[mask].float()-original).square().sum())
                a['fp8_bf16_sse']+=float((x8[mask].float()-original).square().sum())
                a['bf16_energy']+=float(original.square().sum())
                # Exact PV and QK decomposition of o4-o8. Bin SSEs do not add
                # because vectors may reinforce or cancel across age ranges.
                a['pv_sse']+=float((w4[:,mask] @ (x4[mask].float()-x8[mask].float())).square().sum())
                a['qk_sse']+=float(((w4-w8)[:,mask] @ x8[mask].float()).square().sum())
            queries+=1;kinds[e['query_kind']]+=1
        reports.append(dict(case=case,epoch=epoch,layer=layer,queries=queries,missing=missing,
            positions=cursor,query_kinds=dict(kinds),
            errors=[dict(metric=k[0],mode=k[1],stage=k[2],**v.result()) for k,v in stats.items()],
            ages=[dict(stage=k[0],bin=k[1],**v,mass_fraction=v['mass']/v['heads']) for k,v in ages.items()]))
        print(case,layer,'queries',queries,'missing',missing,flush=True)
    return dict(schema=1,files=inventory,reports=reports,
        limitations=['Rank-zero four layers; fixed FP8 queries and DSA selections.',
                     'Ages from first writes in single-request epochs; no eviction.',
                     'Last prefill query and last query in each of eight decode cycles.',
                     'FP32 softmax/matmul and W_UV projection; not logits or MTP acceptance.',
                     'BF16 dumps do not contain downstream hidden states or draft logits.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('captures',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=evaluate(a.captures);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('Missing-prefix queries:',sum(g['missing'] for g in r['reports']))
