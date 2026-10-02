# SPDX-License-Identifier: Apache-2.0
"""Dead MLA split-K partials: drop them from L2 instead of writing them back.

The released split MLA (glm_full_mla_split_kernel.py) writes t x 32 x 16 x 512
fp32 partials (1 MiB per row), reduces them and never reads them again. The dirty
lines stay in L2 and are written back to DRAM while the next bandwidth-bound
kernels (absorbed UV bmm, INT8 o_proj) stream their weights. The reduce below is
the released `_mla_reduce` body followed by a CTA barrier and `discard.global.L2`
on exactly the lines that program consumed, so the write-back never happens.
Arithmetic, reduction order and stores are the released ones; the discard only
touches dead scratch, so the output is bit-identical. The partial kernel is the
released `_mla_partial`, passed in by glm_dirty_l2. Qualified at 32 splits only.
"""
import torch
import triton
import triton.language as tl

LINE_BYTES = 128


@triton.jit
def _l2_discard(ptr, pred):
    # `discard.global.L2` of the 128-byte line at ptr (128-B aligned) where pred != 0.
    # The int32 result is a dummy required by inline_asm_elementwise; is_pure=False
    # keeps the side effect.
    return tl.inline_asm_elementwise(
        '{\n\t.reg .pred dp;\n\tsetp.ne.s32 dp, $2, 0;\n\t@dp discard.global.L2 [$1], 128;\n\tmov.u32 $0, 0;\n\t}',
        '=r,l,r', [ptr, pred], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _discard_lines(A, t, h, hd, SPLITS: tl.constexpr, BD: tl.constexpr, NT: tl.constexpr):
    # The program owns SPLITS x (BD*4/128) lines: (t, p, h, d-chunk) for every split p.
    # NT = threads per CTA; element e < NLINES discards line e, the rest are predicated
    # off, so with one element per thread every owned line is discarded exactly once.
    LINES: tl.constexpr = BD * 4 // 128
    NLINES: tl.constexpr = SPLITS * LINES
    tl.static_assert(NLINES <= NT)
    e = tl.arange(0, NT)
    q = e % NLINES
    off = ((t * SPLITS + q // LINES) * 16 + h) * 512 + (hd % 8) * BD + (q % LINES) * 32
    _l2_discard(A + off, (e < NLINES).to(tl.int32))


@triton.jit
def _mla_reduce_discard(A, MD, O, OS: tl.constexpr, SPLITS: tl.constexpr, BD: tl.constexpr = 64,
                        NT: tl.constexpr = 128):
    # Body identical to the released _mla_reduce, then barrier + discard.
    t = tl.program_id(0)
    hd = tl.program_id(1)
    h = hd // 8
    d = (hd % 8) * BD + tl.arange(0, BD)
    p = tl.arange(0, SPLITS)
    m = tl.load(MD + ((t * SPLITS + p) * 2) * 16 + h)
    den = tl.load(MD + ((t * SPLITS + p) * 2 + 1) * 16 + h)
    mx = tl.maximum(tl.max(m, 0), -3.4028234663852886e38)
    scale = tl.exp2(m - mx)
    total = tl.sum(den * scale, 0)
    acc = tl.load(A + ((t * SPLITS + p[:, None]) * 16 + h) * 512 + d[None, :])
    out = tl.sum(acc * scale[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(O + t * OS + h * 512 + d, out)
    # Every thread's loads of A are complete (values consumed) before any thread
    # of this CTA invalidates the lines. This program is the only reader of its
    # (t, p, h, d-chunk) slices, so no other CTA can observe the discard.
    tl.debug_barrier()
    _discard_lines(A, t, h, hd, SPLITS, BD, NT)


def _check(q, qr, cache):
    t, h, d = q.shape
    p = qr.shape[-1]
    if h != 16 or d != 512 or p not in (0, 64):
        raise ValueError('split MLA requires qualified H16/D512/R0|64')
    if q.dtype != torch.bfloat16 or qr.dtype != torch.bfloat16:
        raise ValueError('BF16 queries required')
    if cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError('plain BF16/FP8 cache required')
    return t, h, p


def sparse_mla_discard(q, qr, cache, slots, scale, ckv_scale=1.0, splits=32, partial=None):
    """Released split32 MLA with discard-on-reduce. `partial` is the released
    `_mla_partial`, injected by glm_dirty_l2 so the partial is the served kernel."""
    t, h, p = _check(q, qr, cache)
    if splits != 32:
        raise ValueError('glm-dirty-l2: qualified at 32 splits only')
    flat = cache.reshape(-1, cache.shape[-1])
    w = slots.shape[1]
    seg = triton.cdiv(w, splits * 16) * 16
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    acc = torch.empty((t, splits, 16, 512), dtype=torch.float32, device=q.device)
    md = torch.empty((t, splits, 2, 16), dtype=torch.float32, device=q.device)
    if acc.data_ptr() % LINE_BYTES:
        raise RuntimeError('glm-dirty-l2: partial buffer is not 128-byte aligned; discard needs whole lines')
    if t:
        partial[(t, splits)](q, qr, flat, slots, acc, md, q.stride(0), q.stride(1), qr.stride(0), qr.stride(1),
                             flat.stride(0), slots.stride(0), out.stride(0), h, w, p, flat.shape[0],
                             float(scale) * 1.4426950408889634, float(ckv_scale),
                             cache.dtype == torch.float8_e4m3fn, splits, seg, num_warps=4, num_stages=1)
        _mla_reduce_discard[(t, 128)](acc, md, out, out.stride(0), splits, NT=128, num_warps=4)
    return out
