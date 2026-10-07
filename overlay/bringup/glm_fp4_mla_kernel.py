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
def _mla(Q, R, KV, IDX, O,
         QS0: tl.constexpr, QS1: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
         KS: tl.constexpr, IS: tl.constexpr, OS0: tl.constexpr,
         H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
         SCALE: tl.constexpr, CKV_SCALE: tl.constexpr, FP8: tl.constexpr,
         BN: tl.constexpr = 16,
         SHADOW=None, MAP=None, TAG=None, POS=None, LOW=None, FLAG=None, CAP: tl.constexpr=0):
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
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
    for start in range(0, W, BN):
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
    out = acc / tl.where(den > 0, den, 1)[:, None]
    tl.store(O + t * OS0 + h[:, None] * 512 + d[None, :], out,
             mask=h[:, None] < H)



def sparse_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
    from glm_fp4_mla_split_kernel import check
    t, h, p = check(q, qr, cache, slots)
    flat = cache.reshape(-1, 368)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if t:
        from glm_recent_kv import reader_args
        shadow,mapping,tags,positions,lower,flag,cap=reader_args(cache)
        _mla[(t, 1)](q, qr, flat, slots, out, q.stride(0), q.stride(1),
            qr.stride(0), qr.stride(1), 368, slots.stride(0), out.stride(0),
            h, slots.shape[1], p, flat.shape[0], float(scale) * 1.4426950408889634,
            1.0, False, SHADOW=shadow,MAP=mapping,TAG=tags,POS=positions,LOW=lower,FLAG=flag,CAP=cap,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out
