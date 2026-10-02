# SPDX-License-Identifier: Apache-2.0
"""Capture-safe sparse MLA for BF16 queries and plain FP8/BF16 KV.

Uses split NoPE/RoPE dot products, direct sparse gathers and online softmax.
No host plan, dense gathered KV, or runtime autotuning.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_partial(Q, R, KV, IDX, A, MD,
         QS0: tl.constexpr, QS1: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
         KS: tl.constexpr, IS: tl.constexpr, OS0: tl.constexpr,
         H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
         SCALE: tl.constexpr, CKV_SCALE: tl.constexpr, FP8: tl.constexpr,
         SPLITS: tl.constexpr, SEG: tl.constexpr,
         BN: tl.constexpr = 16):
    t = tl.program_id(0).to(tl.int64)
    hb = 0
    part = tl.program_id(1)
    h = hb * 16 + tl.arange(0, 16)
    d = tl.arange(0, 512)
    q = tl.load(Q + t * QS0 + h[:, None] * QS1 + d[None, :],
                mask=h[:, None] < H, other=0)
    if P:
        rd = tl.arange(0, 64)
        qr = tl.load(R + t * RS0 + h[:, None] * RS1 + rd[None, :],
                     mask=h[:, None] < H, other=0)
    m = tl.full((16,), float('-inf'), tl.float32)
    den = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 512), tl.float32)
    for start in range(part * SEG, (part + 1) * SEG, BN):
        c = start + tl.arange(0, BN)
        slot = tl.load(IDX + t * IS + c, mask=c < W, other=-1).to(tl.int64)
        valid = (slot >= 0) & (slot < N)
        k = tl.load(KV + slot[:, None] * KS + d[None, :],
                    mask=valid[:, None], other=0.0).to(tl.bfloat16)
        if FP8:
            k = (k.to(tl.float32) * CKV_SCALE).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k))
        if P:
            kr = tl.load(KV + slot[:, None] * KS + 512 + rd[None, :],
                         mask=valid[:, None], other=0.0).to(tl.bfloat16)
            # Match stock wrapper: kpe_scale=1, independent of ckv scale.
            s += tl.dot(qr, tl.trans(kr))
        s = tl.where(valid[None, :], s * SCALE, float('-inf'))
        mn = tl.maximum(m, tl.max(s, 1))
        safe = tl.maximum(mn, -3.4028234663852886e38)
        alpha = tl.exp2(m - safe)
        prob = tl.exp2(s - safe[:, None])
        den = den * alpha + tl.sum(prob, 1)
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), k)
        m = mn
    tl.store(A + ((t * SPLITS + part) * 16 + h[:, None]) * 512 + d[None, :], acc,
             mask=h[:, None] < H)
    tl.store(MD + ((t * SPLITS + part) * 2) * 16 + h, m, mask=h < H)
    tl.store(MD + ((t * SPLITS + part) * 2 + 1) * 16 + h, den, mask=h < H)


@triton.jit
def _mla_reduce(A, MD, O, OS: tl.constexpr, SPLITS: tl.constexpr, BD: tl.constexpr=64):
    t=tl.program_id(0)
    hd=tl.program_id(1)
    h=hd//8
    d=(hd%8)*BD+tl.arange(0,BD)
    p=tl.arange(0,SPLITS)
    m=tl.load(MD+((t*SPLITS+p)*2)*16+h)
    den=tl.load(MD+((t*SPLITS+p)*2+1)*16+h)
    mx=tl.maximum(tl.max(m,0),-3.4028234663852886e38)
    scale=tl.exp2(m-mx)
    total=tl.sum(den*scale,0)
    acc=tl.load(A+((t*SPLITS+p[:,None])*16+h)*512+d[None,:])
    out=tl.sum(acc*scale[:,None],0)/tl.where(total>0,total,1.0)
    tl.store(O+t*OS+h*512+d,out)


def sparse_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    import os
    t,h,d=q.shape;p=qr.shape[-1]
    if h!=16 or d!=512 or p not in (0,64):raise ValueError('split MLA requires qualified H16/D512/R0|64')
    if q.dtype!=torch.bfloat16 or qr.dtype!=torch.bfloat16:raise ValueError('BF16 queries required')
    if cache.dtype not in (torch.bfloat16,torch.float8_e4m3fn):raise ValueError('plain BF16/FP8 cache required')
    flat=cache.reshape(-1,cache.shape[-1]);w=slots.shape[1]
    splits=int(os.environ.get('GLM_MLA_SPLIT_K','32'))
    if splits not in (8,16,32):raise ValueError('qualified split choices:8/16/32')
    seg=triton.cdiv(w,splits*16)*16
    out=torch.empty_like(q,memory_format=torch.contiguous_format)
    acc=torch.empty((t,splits,16,512),dtype=torch.float32,device=q.device)
    md=torch.empty((t,splits,2,16),dtype=torch.float32,device=q.device)
    if t:
        _mla_partial[(t,splits)](q,qr,flat,slots,acc,md,q.stride(0),q.stride(1),qr.stride(0),qr.stride(1),flat.stride(0),slots.stride(0),out.stride(0),h,w,p,flat.shape[0],float(scale)*1.4426950408889634,float(ckv_scale),cache.dtype==torch.float8_e4m3fn,splits,seg,num_warps=4,num_stages=1)
        _mla_reduce[(t,128)](acc,md,out,out.stride(0),splits,num_warps=4)
    return out
