# SPDX-License-Identifier: Apache-2.0
"""Pinned interpreter coverage for the fixed prefill tile ABI; CPU only."""
import torch
from glm_fp4_prefill_kernel import _gather_index_tile, _tile_bounds, _finish_indices, _index_prefill_bounds, _prepare_index_queries


def verify_tiles():
    bs=64
    cache=torch.zeros(4,bs,132,dtype=torch.uint8)
    codes=torch.arange(4*bs*128).remainder(256).byte().reshape(4,bs,128)
    scales=torch.arange(4*bs).float().reshape(4,bs)+1
    cache.view(4,-1)[:,:bs*128].copy_(codes.reshape(4,-1))
    cache.view(4,-1)[:,bs*128:].copy_(scales.view(torch.uint8).reshape(4,-1))
    bt=torch.tensor([[2,0],[1,3]],dtype=torch.int32)
    cu=torch.tensor([0,65,140],dtype=torch.int32)
    keys=torch.empty(4096,128,dtype=torch.uint8);sf=torch.empty(4096)
    _gather_index_tile[(256,)](cache,bt,cu,keys,sf,60,bs,bs*132,2,2,num_warps=4)
    for row in range(4096):
        pos=60+row
        if pos>=140:
            assert sf[row]==0 and torch.all(keys[row]==0)
        else:
            req=0 if pos<65 else 1;local=pos-int(cu[req]);block=int(bt[req,local//bs])
            assert torch.equal(keys[row],codes[block,local%bs])
            assert sf[row]==scales[block,local%bs]
    ks=torch.tensor([0,65],dtype=torch.int32);ke=torch.tensor([65,140],dtype=torch.int32)
    starts=torch.empty(64,dtype=torch.int32);ends=torch.empty_like(starts)
    _tile_bounds[(1,)](ks,ke,starts,ends,0,2,60,num_warps=4)
    assert starts[:3].tolist()==[0,5,0] and ends[:3].tolist()==[5,80,0]
    values=torch.full((64,2048),float('-inf'));ids=torch.full((64,2048),-1,dtype=torch.int64)
    values[0,0]=1;ids[0,0]=3;values[1,0]=2;ids[1,0]=70
    out=torch.empty(2,2048,dtype=torch.int32)
    _finish_indices[(2,8)](values,ids,ks,out,0,2,2048,num_warps=4)
    assert out[:,0].tolist()==[3,5] and torch.all(out[:,1:]==-1)
    qsl=torch.tensor([0,3,8],dtype=torch.int32);seq=torch.tensor([50000,70001],dtype=torch.int32)
    cu=torch.tensor([0,50000,120001],dtype=torch.int32)
    ks=torch.empty(5,dtype=torch.int32);ke=torch.empty_like(ks)
    _index_prefill_bounds[(2,1)](qsl,seq,cu,ks,ke,2,7,num_warps=4)
    assert ks.tolist()==[0,50000,50000,50000,50000]
    assert ke.tolist()==[50000,119997,119998,119999,120000]
    # Single-request query sub-chunks (stock splitter above 512 MiB of logits): the second
    # sub-chunk starts at qs_start>0 and must still get every row (fp4x-cal 10-05 80K hang).
    qsl=torch.tensor([0,2048],dtype=torch.int32);seq=torch.tensor([67584],dtype=torch.int32)
    cu=torch.tensor([0,67584],dtype=torch.int32)
    for qs,qe in ((0,1985),(1985,2048),(300,2048),(1,129)):
        ks=torch.full((qe-qs,),-7,dtype=torch.int32);ke=torch.full_like(ks,-7)
        _index_prefill_bounds[(1,(qe-qs+127)//128)](qsl,seq,cu,ks,ke,qs,qe,num_warps=4)
        assert ks.tolist()==[0]*(qe-qs)
        assert ke.tolist()==[67584-2048+c+1 for c in range(qs,qe)]
    q=torch.arange(7*128).remainder(256).byte().reshape(7,1,128)
    w=torch.arange(7).float().reshape(7,1)
    ks=torch.tensor([123,124,125,126,127],dtype=torch.int32);ke=ks+4096
    bq=torch.empty(4,1,128,dtype=torch.uint8);bw=torch.empty(4,1)
    starts=torch.empty(4,dtype=torch.int32);ends=torch.empty_like(starts)
    _prepare_index_queries[(4,)](q,w,ks,ke,bq,bw,starts,ends,4,2,3,1,128,num_warps=4)
    assert torch.equal(bq[:3],q[4:7]) and torch.all(bq[3]==0)
    assert bw[:,0].tolist()==[4.,5.,6.,0.]
    assert starts.tolist()==[125,126,127,0] and ends.tolist()==[4221,4222,4223,0]
    verify_union()
    print('PASS_PINNED_PREFILL_TILE_INTERPRETER')


def verify_union():
    from glm_fp4_kv_kernel import _pack_store, _read_rows
    from glm_fp4_mla_prefill import _mark_selected, _expand_selected
    n=131
    gen=torch.Generator().manual_seed(5112)
    x=torch.randn(n,512,generator=gen).bfloat16()
    rope=torch.randn(n,64,generator=gen).bfloat16()
    cache=torch.empty(n,368,dtype=torch.uint8)
    write=torch.arange(n,dtype=torch.int64)
    _pack_store[(n,)](x,rope,cache,write,torch.ones(1),512,64,368,n,n,
                     num_warps=4,num_stages=1,enable_fp_fusion=False)
    idx=torch.tensor([[0,63,64,130,-1,131,63],[64,65,130,-64,9999,65,-1]],dtype=torch.int32)
    selected=torch.empty(n,dtype=torch.int32)
    plain=torch.full((n,576),float('nan'),dtype=torch.bfloat16)
    expected=torch.empty(n,576,dtype=torch.bfloat16)
    _read_rows[((n+15)//16,)](cache,write,expected,n,n,576,num_warps=4,num_stages=1,enable_fp_fusion=False)
    for _ in range(2):
        selected.zero_()
        _mark_selected[(2,1)](idx,selected,7,7,n,num_warps=4)
        _expand_selected[((n+15)//16,)](cache,selected,plain,n,num_warps=4,num_stages=1,enable_fp_fusion=False)
        want=sorted(set(s for s in idx.flatten().tolist() if 0<=s<n))
        assert selected.nonzero().flatten().tolist()==want
        assert torch.equal(plain[want].view(torch.uint16),expected[want].view(torch.uint16))
        idx.fill_(-1);idx[:,0]=129
    print('PASS_PINNED_PREFILL_UNION_INTERPRETER')
