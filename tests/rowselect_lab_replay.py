# SPDX-License-Identifier: Apache-2.0
"""64 E9 anchors through saved real-weight CPU MTP, full-M vs selected tail.

Use the existing local lab interpreter and weights. The reference executes
each causal prefix row separately; it does not emulate GPU M-dependent tiling.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab',type=Path,required=True);p.add_argument('--roots',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.lab))
    from model import MTP,Context
    from data import decode_mla,decode_index
    from weights import Shards
    torch.set_num_threads(4);started=time.monotonic()
    roots=torch.load(a.roots,mmap=True,weights_only=True)
    assert roots['meta']['summary']['status']=='PASS' and roots['meta']['e9']
    requests=sorted(roots['requests'],key=lambda q:(q['panel'],q['request']))
    panels=sorted({q['panel'] for q in requests})
    groups={panel:[q for q in requests if q['panel']==panel] for panel in panels}
    selected=[]
    for i in range(64):
        panel=panels[i%len(panels)];q=groups[panel][i//len(panels)]
        assert q['split']=='heldout'
        ns=[n for n in q['natural'] if n['t'] in q['hidden'] and n['t']+1 in q['stream']]
        n=ns[len(ns)//2];t=n['t']
        ctx=Context(decode_mla(q['timeline'][:t].clone(),dict(cache_dtype='fp8_e4m3',scale=1.,rope_scale=1.)),
                    decode_index(q['index_rows'][:t].clone()) if t+1>2048 else None,cache_mode='fp8_e4m3')
        selected.append((dict(request=q['request'],panel=panel,position=t),q['hidden'][t].clone(),q['stream'][t+1],ctx))
    del roots
    shards=Shards(a.weights,torch.bfloat16);model=MTP(shards,native_bf16=True,native_tp=4)
    # Split the exact lab method immediately before o_proj; native prefix
    # includes cache append and causal attention for every row in either arm.
    raw=textwrap.dedent(inspect.getsource(MTP.step))
    boundary="    z = self.activation(z + self.linear(val, 'self_attn.o_proj'))\n"
    assert raw.count(boundary)==1
    prefix=raw.split(boundary)[0]+'    return z, val, context, selected\n'
    prefix=prefix.replace('def step(', 'def prefix(',1)
    g=dict(inspect.unwrap(MTP.step).__globals__);exec(compile(prefix,'<rowselect-lab-prefix>','exec'),g)
    def finish(z,val):
        z=model.activation(z+model.linear(val,'self_attn.o_proj'))
        pre=model.activation(z+model.moe(model.norm(z,'post_attention_layernorm')))
        return model.norm(pre,'shared_head.norm')
    # The native proposal head sees only selected rows even in the full-M arm.
    original_logits=model.logits;model.logits=lambda pre:model.norm(pre,'shared_head.norm')
    records=[]
    for start in range(0,64,16):
        batch=selected[start:start+16];full=[];prefixes=[]
        for meta,h,tok,ctx in batch:
            out,feedback,cache,idx=model.step(tok,h,meta['position'],ctx.clone(),0)
            z,val,cache2,idx2=g['prefix'](model,tok,h,meta['position'],ctx.clone(),0)
            assert torch.equal(cache.kv,cache2.kv) and torch.equal(idx,idx2)
            assert cache.keys is None or torch.equal(cache.keys,cache2.keys)
            full.append(out);prefixes.append((z,val))
        # Rotate needed rows across M16, covering every anchor exactly once.
        # Baseline tail above executed all M16; selected tail executes just
        # four rows per gather. No discarded row's KV update is removed.
        for group in range(4):
            ids=list(range(group,16,4))
            reduced=torch.stack([finish(*prefixes[i]) for i in ids])
            expected=torch.stack([full[i] for i in ids])
            assert torch.equal(reduced,expected)
            actual_logits=model.linear(reduced,'shared_head.head')
            expected_logits=model.linear(expected,'shared_head.head')
            assert torch.equal(actual_logits,expected_logits)
            for j,i in enumerate(ids):
                meta=batch[i][0];records.append(meta|dict(argmax=int(actual_logits[j].argmax()),
                     logits_bit_exact=True,feedback_bit_exact=True,kv_bit_exact=True))
        print(json.dumps(dict(anchors=len(records),seconds=time.monotonic()-started)),flush=True)
    model.logits=original_logits
    binding=dict(roots_sha256=hashlib.file_digest(a.roots.open('rb'), 'sha256').hexdigest(),
        weights_fingerprint=shards.fingerprint(),model_sha256=hashlib.sha256((a.lab/'model.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report=dict(anchors=64,all_exact=True,full_tail_rows=64,selected_tail_rows=64,
        arm_geometry='four M16 groups, four M4 selections per group cover all 64 anchors',
        per_proposal_geometry='M16 attention/KV; M16 full tail versus M4 selected tail',
        seconds=time.monotonic()-started,binding=binding,
        scope='real saved INT8 MTP weights; CPU BF16 boundaries/TP4 emulation; per-row reference, no GPU tiling or current NVFP4-body claim')
    (a.out/'positions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    import signal
    signal.alarm(1200)
    with torch.inference_mode():main()
