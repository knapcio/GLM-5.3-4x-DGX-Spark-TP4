# SPDX-License-Identifier: Apache-2.0
"""GPU correctness/replay fixture for an owned qualification window.

Run after static compilation and performance pre-review. This script is
never launched by the Mac CPU suite. No network or model/fleet lifecycle.
"""
import argparse
import os
from pathlib import Path
import sys
import types
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'tests/fixtures')]
from fp4_probe_quant_dc5ad9e import fp4_qdq,fp8_codes
from topk_exactness import assert_topk_exact
from glm_fp4_kv_kernel import pack_store,_read_rows,gather
from glm_fp4_mla_kernel import sparse_mla as unsplit
from glm_fp4_mla_split_kernel import sparse_mla as split



def verify_indexer_tiles(device):
    from glm_fp4_prefill import reserve, tiled_prefill, geometry, LOGITS_BYTES, scratch_bytes, bank_bytes
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm import _custom_ops as ops
    gen=torch.Generator(device=device).manual_seed(5106)
    device=torch.empty(0,device=device).device
    bank=reserve(device,64);addresses={k:v.data_ptr() for k,v in bank.items()}
    assert bank_bytes(bank)+LOGITS_BYTES==scratch_bytes()
    # All bucket transitions, Q tails, request/page seams and query subchunks.
    for tokens,nq in ((12289,67),(32769,1025),(65537,513),(98226,67)):
        bs=64;split_at=tokens//2+1
        n0=(split_at+bs-1)//bs;n1=(tokens-split_at+bs-1)//bs;blocks=n0+n1
        codes=torch.randn(blocks*bs,128,device=device,generator=gen).to(torch.float8_e4m3fn)
        cache=torch.empty(blocks,bs,132,device=device,dtype=torch.uint8)
        cache.view(blocks,-1)[:,:bs*128].copy_(codes.view(torch.uint8).reshape(blocks,-1))
        cache.view(blocks,-1)[:,bs*128:].copy_(torch.ones(blocks,bs,device=device).view(torch.uint8).reshape(blocks,-1))
        bt=torch.zeros(2,max(n0,n1),device=device,dtype=torch.int32)
        bt[0,:n0]=torch.arange(n0,device=device);bt[1,:n1]=torch.arange(n0,blocks,device=device)
        cu=torch.tensor([0,split_at,tokens],device=device,dtype=torch.int32)
        q=torch.randn(nq+2,64,128,device=device,generator=gen).to(torch.float8_e4m3fn)
        w=torch.rand(nq+2,64,device=device,generator=gen)
        ks_cpu=torch.tensor([0]*(nq//2)+[split_at]*(nq-nq//2),dtype=torch.int32)
        ke_cpu=torch.tensor([9]+[split_at]*(nq//2-1)+[split_at+5]+[tokens]*(nq-nq//2-1),dtype=torch.int32)
        ks,ke=ks_cpu.to(device),ke_cpu.to(device)
        logical=torch.cat([codes[:split_at].view(torch.uint8),codes[n0*bs:n0*bs+tokens-split_at].view(torch.uint8)]).view(torch.float8_e4m3fn)
        chunk=types.SimpleNamespace(token_start=2,token_end=nq+2,local_total_seq_lens=tokens,
            block_table=bt,local_cu_seq_lens=cu,cu_seqlen_ks=ks,cu_seqlen_ke=ke)
        for case in ('random','zero_ties'):
            if case=='zero_ties':w.zero_()
            dense=fp8_fp4_mqa_logits((q[2:],None),(logical,torch.ones(tokens,device=device)),w[2:],ks,ke,clean_logits=False)
            expected=torch.full((nq,2048),-1,device=device,dtype=torch.int32)
            ops.top_k_per_row_prefill(dense,ks,ke,expected,nq,dense.stride(0),dense.stride(1),2048)
            result=torch.full((nq+2,2048),77,device=device,dtype=torch.int32)
            for _ in range(2):
                cursor=[0]
                def checked_logits(*args,**kwargs):
                    actual=fp8_fp4_mqa_logits(*args,**kwargs)
                    assert actual.numel()*actual.element_size()==LOGITS_BYTES
                    rows=min(geometry(tokens)[0],nq-cursor[0])
                    for row in range(rows):
                        absolute=cursor[0]+row;start,end=int(ks_cpu[absolute]),int(ke_cpu[absolute])
                        assert torch.equal(actual[row,start:end].view(torch.int32),dense[absolute,start:end].view(torch.int32))
                    cursor[0]+=rows
                    return actual
                tiled_prefill(cache,q,None,w,[chunk],result,checked_logits,ops.top_k_per_row_prefill)
                torch.cuda.synchronize()
                assert cursor[0]==nq and torch.all(result[:2]==77)
                # Stock cutoff ties use atomic arrival order. Require exact
                # strict-score IDs and score bits; allow only equal-bit
                # cutoff tie swaps, retaining bounds/count/sentinel checks.
                for row in range(nq):
                    try:
                        assert_topk_exact(dense[row],result[row+2],expected[row],
                                          int(ks_cpu[row]),int(ke_cpu[row]))
                    except AssertionError as exc:
                        raise AssertionError((tokens,case,row,str(exc))) from exc
                assert addresses=={k:v.data_ptr() for k,v in reserve(device,64).items()}
            del dense
    print('PASS_GPU_FP4X_FIXED_BUDGET_STOCK_TOPK_BITS')


def verify(device):
    from glm_fp4_prefill import reserve, bank_for
    from glm_fp4_mla_prefill import prefill_mla
    reserve(torch.empty(0,device=device).device,64)
    gen=torch.Generator(device=device).manual_seed(5105)
    x=torch.randn(128,512,device=device,generator=gen).bfloat16()
    rope=torch.randn(128,64,device=device,generator=gen).bfloat16()
    scale=torch.tensor([.75],device=device)
    cache=torch.zeros(4,64,368,dtype=torch.uint8,device=device)
    slots=torch.arange(128,device=device,dtype=torch.int64)+62
    pack_store(x,rope,cache,slots,scale)
    out=torch.empty(128,576,dtype=torch.bfloat16,device=device)
    _read_rows[(8,)](cache,slots,out,256,128,576,num_warps=4,num_stages=1,enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert torch.equal((out[:,:512].float()+0.0).view(torch.int32),(fp4_qdq(x).float()+0.0).view(torch.int32))  # +-0 canonicalized
    assert torch.equal(out[:,512:].view(torch.uint16),fp8_codes(rope,scale).bfloat16().view(torch.uint16))
    flat=cache.reshape(-1,368)
    assert torch.all(flat[62:190,292:304]==0)
    assert torch.all(flat[:62]==0) and torch.all(flat[190:]==0)
    # Cached-prefix gather crosses page boundaries with nonzero sequence start.
    bt=torch.tensor([[0,1,2,3]],device=device,dtype=torch.int32)
    cu=torch.tensor([0,128],device=device,dtype=torch.int32)
    ts=torch.zeros(128,device=device,dtype=torch.int32)
    starts=torch.tensor([62],device=device,dtype=torch.int32)
    buf=torch.empty_like(out)
    gather(cache,buf,bt,cu,ts,128,seq_starts=starts)
    assert torch.equal(buf.view(torch.uint16),out.view(torch.uint16))
    for n in (1,3,6,12,16):
        q=torch.randn(n,16,512,device=device,generator=gen).bfloat16()
        qr=torch.randn(n,16,64,device=device,generator=gen).bfloat16()
        idx=torch.full((n,2048),-1,device=device,dtype=torch.int32)
        idx[:,:128]=slots.to(torch.int32)
        if n > 1: idx[-1]=-1
        expected=unsplit(q,qr,cache,idx,.0625)
        union=prefill_mla(q,qr,cache,idx,.0625)
        # BF16 expansion adds no quantization: compare actual reconstructed
        # rows bit-for-bit, then qualify the unchanged plain kernel output.
        plain=bank_for(q.device)['workspace'][:cache.numel()//368*1152].view(torch.bfloat16).view(-1,576)
        if n==1:
            assert torch.equal(plain[62:190].view(torch.uint16),out.view(torch.uint16))
        assert torch.allclose(expected.float(),union.float(),atol=.03,rtol=.02)
        if n>1: assert torch.all(union[-1]==0)
        warm=split(q,qr,cache,idx,.0625)  # allocates scratch outside capture
        assert torch.allclose(expected.float(),warm.float(),atol=.03,rtol=.02)
        if n > 1: assert torch.all(warm[-1]==0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_store(x,rope,cache,slots,scale)
            replayed=split(q,qr,cache,idx,.0625)
        # Overwrite data, replay writer + reader, compare with current eager.
        for _ in range(3):
            x.copy_(torch.randn(x.shape,device=device,generator=gen).bfloat16())
            graph.replay()
            eager=split(q,qr,cache,idx,.0625)
            assert torch.equal(replayed.view(torch.uint16),eager.view(torch.uint16))
    # Invalid speculative slots must not change the null block or other rows.
    pack_store(x[:3],rope[:3],cache,torch.tensor([-1,-64,9999],device=device),scale)
    assert torch.all(cache[0,:62]==0)
    torch.cuda.synchronize()
    print('PASS_GPU_FP4_KV_READ_WRITE_GATHER_REPLAY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');a=p.parse_args()
    if os.environ.get('GLM_MLA_SPLIT_K','32')!='32':raise RuntimeError('release qualification requires split32')
    verify(a.device)
    verify_indexer_tiles(a.device)
