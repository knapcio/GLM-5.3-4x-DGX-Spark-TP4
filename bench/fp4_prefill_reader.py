#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Owned GPU microbenchmark; no model, fleet lifecycle, or network actions.

Separates pack/store, packed attention, union preparation and plain attention.
Multiply serial per-layer savings by 79 for the model's target+MTP upper count.
Run after the strict GPU fixture, on each rank; do not overlap fleet requests.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'overlay/bringup'))
from glm_fp4_prefill import reserve, bank_for, MLA_ROWS, bank_bytes, LOGITS_BYTES
from glm_fp4_kv_kernel import pack_store
from glm_fp4_mla_kernel import sparse_mla as packed
from glm_full_mla_kernel import sparse_mla as plain_reader
from glm_fp4_mla_prefill import prefill_mla, _mark_selected, _expand_selected


def elapsed(fn, repeat):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / repeat


def run(prefix, rows, repeat, shared):
    device=torch.empty(0,device='cuda').device
    bank=reserve(device,64)
    gen=torch.Generator(device=device).manual_seed(5111)
    cache=torch.empty(MLA_ROWS,368,device=device,dtype=torch.uint8)
    x=torch.randn(prefix,512,device=device,generator=gen).bfloat16()
    rope=torch.randn(prefix,64,device=device,generator=gen).bfloat16()
    write=torch.arange(prefix,device=device,dtype=torch.int64)
    scale=torch.ones(1,device=device)
    pack_store(x,rope,cache,write,scale)
    # Release NoPE head-major layout, with a contiguous inner latent dimension.
    q=torch.randn(16,rows,512,device=device,generator=gen).bfloat16().transpose(0,1)
    qr=torch.randn(rows,16,64,device=device,generator=gen).bfloat16()
    width=min(prefix,2048)
    idx=torch.randint(prefix,(1 if shared else rows,width),device=device,generator=gen,dtype=torch.int32)
    if shared: idx=idx.expand(rows,-1).contiguous()
    selected=bank['selected']
    plain=bank['workspace'][:MLA_ROWS*1152].view(torch.bfloat16).view(MLA_ROWS,576)
    def expand():
        selected.zero_()
        _mark_selected[(rows,(width+255)//256)](idx,selected,width,width,MLA_ROWS,num_warps=4)
        _expand_selected[((MLA_ROWS+15)//16,)](cache,selected,plain,MLA_ROWS,num_warps=4,num_stages=1,enable_fp_fusion=False)
    expected=packed(q,qr,cache,idx,.0625)
    actual=prefill_mla(q,qr,cache,idx,.0625)
    assert torch.allclose(expected.float(),actual.float(),atol=.03,rtol=.02)
    writer_rows=min(rows,prefix)
    pack_ms=elapsed(lambda:pack_store(x[:writer_rows],rope[:writer_rows],cache,write[:writer_rows],scale),repeat)
    packed_ms=elapsed(lambda:packed(q,qr,cache,idx,.0625),repeat)
    union_ms=elapsed(lambda:prefill_mla(q,qr,cache,idx,.0625),repeat)
    prepare_ms=elapsed(expand,repeat)
    plain_ms=elapsed(lambda:plain_reader(q,qr,plain,idx,.0625),repeat)
    return dict(prefix=prefix,queries=rows,shared_selections=shared,union_rows=int(idx.unique().numel()),
                pack_store_ms=pack_ms,packed_attention_ms=packed_ms,
                union_end_to_end_ms=union_ms,union_prepare_ms=prepare_ms,plain_attention_ms=plain_ms,
                modeled_79_layer_saved_s_per_chunk=79*(packed_ms-union_ms)/1000,
                scratch_bytes=bank_bytes(bank)+LOGITS_BYTES,
                device=torch.cuda.get_device_name(),torch=torch.__version__)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=int,nargs='+',default=[4096,16384,65536,100288])
    p.add_argument('--queries',type=int,default=2048);p.add_argument('--repeat',type=int,default=10)
    a=p.parse_args()
    if not 0<a.queries<=4096 or a.repeat<1 or any(not 0<n<=MLA_ROWS for n in a.prefix):p.error('release bounds exceeded')
    for n in a.prefix:
        for shared in (False,True):print(json.dumps(run(n,a.queries,a.repeat,shared)),flush=True)
