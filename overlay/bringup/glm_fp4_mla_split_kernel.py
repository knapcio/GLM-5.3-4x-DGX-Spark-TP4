# SPDX-License-Identifier: Apache-2.0
"""Capture-safe sparse MLA for BF16 queries and packed FP4x KV.

Uses split NoPE/RoPE dot products, direct sparse gathers and online softmax.
No host plan, dense gathered KV, or runtime autotuning.
"""
import torch
import triton
import triton.language as tl
from glm_fp4_kv_kernel import _load_latent, _load_rope


@triton.jit
def _mla_partial(Q, R, KV, IDX, A, MD,
         QS0: tl.constexpr, QS1: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
         KS: tl.constexpr, IS: tl.constexpr, OS0: tl.constexpr,
         H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
         SCALE: tl.constexpr, CKV_SCALE: tl.constexpr, FP8: tl.constexpr,
         SPLITS: tl.constexpr, SEG: tl.constexpr,
         BN: tl.constexpr = 16,
         SHADOW=None, MAP=None, TAG=None, POS=None, LOW=None, FLAG=None, CAP: tl.constexpr=0):
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
        k = _load_latent(KV, slot, d, KS, N, SHADOW, MAP, TAG, POS, LOW, FLAG, CAP)
        s = tl.dot(q, tl.trans(k))
        if P:
            kr = _load_rope(KV, slot, rd, KS, N)
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



_SCRATCH = {}


def check(q, qr, cache, slots):
    t, h, d = q.shape
    p = qr.shape[-1]
    if h != 16 or d != 512 or p != 64 or qr.shape[:2] != q.shape[:2]:
        raise ValueError('FP4x MLA requires H16/D512/R64')
    if q.dtype != torch.bfloat16 or qr.dtype != torch.bfloat16:
        raise ValueError('FP4x MLA requires BF16 queries')
    if cache.dtype != torch.uint8 or cache.shape[-1] != 368 or not cache.is_contiguous():
        raise ValueError('FP4x MLA requires contiguous uint8 D368 storage')
    if slots.dtype != torch.int32 or slots.shape[0] != t or slots.stride(-1) != 1:
        raise ValueError('FP4x MLA requires int32 unit-stride slots')
    if q.stride(-1) != 1 or qr.stride(-1) != 1:
        raise ValueError('FP4x MLA requires unit inner query strides')
    return t, h, p


def scratch(q, splits):
    # One stream per worker, sequential target/MTP MLA. Fixed addresses per row
    # shape; eager warmup must establish every capture shape before capture.
    import os
    capacity = int(os.environ.get("GLM_MLA_SPLIT_MAX_ROWS", "36"))
    if not 1 <= capacity <= 36 or q.shape[0] > capacity:
        raise ValueError("FP4x split scratch capacity is 1..36 rows")
    key = (q.device, capacity, splits)
    if key not in _SCRATCH:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('FP4x split scratch not warmed before graph capture')
        t = capacity
        _SCRATCH[key] = (torch.empty((t,splits,16,512), dtype=torch.float32, device=q.device),
                         torch.empty((t,splits,2,16), dtype=torch.float32, device=q.device))
    return _SCRATCH[key]


def sparse_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    import os
    t, h, p = check(q, qr, cache, slots)
    splits = int(os.environ.get('GLM_MLA_SPLIT_K', '32'))
    if splits not in (8, 16, 32):
        raise ValueError('qualified split choices:8/16/32')
    discard = os.environ.get('GLM_DIRTY_L2', '0') == 'discard'
    if discard and splits != 32:
        raise ValueError('dirty-L2 requires split32')
    flat = cache.reshape(-1, 368)
    w = slots.shape[1]
    seg = triton.cdiv(w, splits * 16) * 16
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if t:
        acc, md = scratch(q, splits)
        from glm_recent_kv import reader_args
        shadow,mapping,tags,positions,lower,flag,cap=reader_args(cache)
        _mla_partial[(t,splits)](q,qr,flat,slots,acc,md,q.stride(0),q.stride(1),
            qr.stride(0),qr.stride(1),368,slots.stride(0),out.stride(0),h,w,p,flat.shape[0],
            float(scale)*1.4426950408889634,1.0,False,splits,seg,
            SHADOW=shadow,MAP=mapping,TAG=tags,POS=positions,LOW=lower,FLAG=flag,CAP=cap,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
        if discard:
            from glm_dirty_l2_kernel import _mla_reduce_discard
            if acc.data_ptr() % 128:
                raise RuntimeError('FP4x split scratch must be 128-byte aligned')
            _mla_reduce_discard[(t,128)](acc,md,out,out.stride(0),splits,NT=128,num_warps=4,num_stages=1)
        else:
            _mla_reduce[(t,128)](acc,md,out,out.stride(0),splits,num_warps=4,num_stages=1)
    return out
