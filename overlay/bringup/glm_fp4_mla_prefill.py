# SPDX-License-Identifier: Apache-2.0
"""Expand the union of prefill selections once; use the release plain reader.

Physical slot IDs are retained. DSA gather and attention expansion alias one
constructor arena on the worker stream. No graph/decode dispatch uses it.
"""
import torch
import triton
import triton.language as tl
from glm_fp4_kv_kernel import _load_latent, _load_rope


@triton.jit
def _mark_selected(IDX, SELECTED, IS: tl.constexpr, W: tl.constexpr,
                   N: tl.constexpr, BC: tl.constexpr = 256):
    q = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    slot = tl.load(IDX + q * IS + c, mask=c < W, other=-1)
    valid = (slot >= 0) & (slot < N)
    # Multiple query rows can select a slot: atomic writes avoid a data race.
    tl.atomic_or(SELECTED + slot, 1, mask=valid, sem='relaxed')


@triton.jit
def _expand_selected(KV, SELECTED, OUT, N: tl.constexpr,
                     BN: tl.constexpr = 16,
                     SHADOW=None, MAP=None, TAG=None, POS=None, LOW=None, FLAG=None, CAP: tl.constexpr=0):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    selected = tl.load(SELECTED + row, mask=row < N, other=0) != 0
    slot = tl.where(selected, row, -1).to(tl.int64)
    d = tl.arange(0, 512)
    rd = tl.arange(0, 64)
    k = _load_latent(KV, slot, d, 368, N, SHADOW, MAP, TAG, POS, LOW, FLAG, CAP)
    r = _load_rope(KV, slot, rd, 368, N)
    tl.store(OUT + row[:, None] * 576 + d[None, :], k,
             mask=selected[:, None])
    tl.store(OUT + row[:, None] * 576 + 512 + rd[None, :], r,
             mask=selected[:, None])


def prefill_mla(q, qr, cache, slots, scale):
    from glm_fp4_mla_split_kernel import check
    from glm_fp4_prefill import bank_for, MLA_ROWS
    from glm_full_mla_kernel import sparse_mla
    t, h, p = check(q, qr, cache, slots)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('FP4x union prefill must be eager')
    n = cache.numel() // 368
    if n > MLA_ROWS:
        raise RuntimeError('FP4x prefill exceeds release physical cache capacity')
    bank = bank_for(q.device)
    selected = bank['selected'][:n]
    selected.zero_()
    plain = bank['workspace'][:n * 576 * 2].view(torch.bfloat16).view(n, 576)
    if t:
        from glm_recent_kv import reader_args
        shadow,mapping,tags,positions,lower,flag,cap=reader_args(cache)
        _mark_selected[(t, triton.cdiv(slots.shape[1], 256))](
            slots, selected, slots.stride(0), slots.shape[1], n, num_warps=4)
        _expand_selected[(triton.cdiv(n, 16),)](
            cache, selected, plain, n, SHADOW=shadow,MAP=mapping,TAG=tags,POS=positions,LOW=lower,FLAG=flag,CAP=cap,
            num_warps=4, num_stages=1,
            enable_fp_fusion=False)
    # Unselected rows may contain old DSA bytes. The plain reader masks -1
    # and >=N exactly as the packed reader; every valid selection was marked.
    return sparse_mla(q, qr, plain, slots, scale, 1.0)
