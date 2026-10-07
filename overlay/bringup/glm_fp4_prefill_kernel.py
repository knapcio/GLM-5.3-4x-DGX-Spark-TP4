# SPDX-License-Identifier: Apache-2.0
"""Bounded readers for the unchanged paged DSA E4M3 + page-separated scales."""
import triton
import triton.language as tl


@triton.jit
def _gather_index_tile(KV, BT, CU, K, S, OFFSET,
                       BS: tl.constexpr, PS: tl.constexpr, BTS: tl.constexpr,
                       REQS: tl.constexpr, BN: tl.constexpr = 16):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    pos = OFFSET + row
    req = tl.full((BN,), 0, tl.int32)
    for r in range(1, REQS):
        req = tl.where(pos >= tl.load(CU + r), r, req)
    start = tl.load(CU + req)
    stop = tl.load(CU + req + 1)
    valid = pos < stop
    local = pos - start
    block = tl.load(BT + req * BTS + local // BS, mask=valid, other=0).to(tl.int64)
    slot = local % BS
    d = tl.arange(0, 128)
    base = KV.to(tl.pointer_type(tl.uint8)) + block[:, None] * PS
    codes = tl.load(base + slot[:, None] * 128 + d[None, :], mask=valid[:, None], other=0)
    tl.store(K.to(tl.pointer_type(tl.uint8)) + row[:, None] * 128 + d[None, :], codes)
    scale_ptr = (KV.to(tl.pointer_type(tl.uint8)) + block * PS + BS * 128 + slot * 4).to(tl.pointer_type(tl.float32))
    scales = tl.load(scale_ptr, mask=valid, other=0.)
    tl.store(S + row, scales)


@triton.jit
def _tile_bounds(KS, KE, S, E, QOFF, ROWS, OFFSET,
                 QT: tl.constexpr = 64, KT: tl.constexpr = 4096):
    row = tl.arange(0, QT)
    start = tl.load(KS + QOFF + row, mask=row < ROWS, other=0)
    end = tl.load(KE + QOFF + row, mask=row < ROWS, other=0)
    tl.store(S + row, tl.minimum(tl.maximum(start - OFFSET, 0), KT))
    tl.store(E + row, tl.minimum(tl.maximum(end - OFFSET, 0), KT))


@triton.jit
def _finish_indices(V, IDS, KS, OUT, QOFF, ROWS,
                    OS: tl.constexpr, TOPK: tl.constexpr = 2048,
                    BN: tl.constexpr = 256):
    row = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    score = tl.load(V + row * TOPK + col)
    idx = tl.load(IDS + row * TOPK + col)
    start = tl.load(KS + QOFF + row)
    # Match the stock prefill selector's request-relative token IDs/sentinel.
    idx = tl.where(score > float('-inf'), idx - start, -1)
    tl.store(OUT + row * OS + col, idx.to(tl.int32))


def gather_index_tile(cache, bt, cu, keys, scales, offset):
    _gather_index_tile[(triton.cdiv(keys.shape[0], 16),)](cache, bt, cu, keys, scales, offset,
        cache.shape[1], cache.stride(0), bt.stride(0), bt.shape[0], num_warps=4)


def tile_bounds(ks, ke, starts, ends, qoff, rows, offset):
    _tile_bounds[(1,)](ks, ke, starts, ends, qoff, rows, offset, num_warps=4)


def finish_indices(values, ids, starts, out, qoff, rows):
    _finish_indices[(rows, 8)](values, ids, starts, out, qoff, rows,
                              out.stride(0), num_warps=4)


@triton.jit
def _index_prefill_bounds(QSL, SEQ, CU, KS, KE, QSTART, QSTOP,
                          BN: tl.constexpr = 128):
    req = tl.program_id(0)
    query_start = tl.load(QSL + req)
    query_end = tl.load(QSL + req + 1)
    # The grid covers qstop-qstart rows. A query sub-chunk (stock splitter, P*Q*4 > the
    # logits budget) starts at QSTART > query_start: begin there, as the stock kernel's
    # full-query loop does, or rows >= query_start+cdiv(...)*BN never get bounds.
    col = tl.maximum(QSTART - query_start, 0) + tl.program_id(1) * BN + tl.arange(0, BN)
    seq_len = tl.load(SEQ + req)
    row_start = tl.load(CU + req)
    absolute = query_start + col
    valid = (absolute < query_end) & (absolute >= QSTART) & (absolute < QSTOP)
    out_pos = absolute - QSTART
    tl.store(KS + out_pos, row_start, mask=valid)
    tl.store(KE + out_pos, row_start + seq_len - (query_end-query_start) + col + 1, mask=valid)


def index_prefill_bounds(qsl, seq, cu, local_cu, token_to_seq, ks, ke,
                         qstart, qstop, rank, world, interleave, *, num_reqs, COMPRESS_RATIO):
    if world != 1 or COMPRESS_RATIO != 1:
        raise RuntimeError('FP4x prefill bounds require DCP1/uncompressed DSA')
    _index_prefill_bounds[(num_reqs, triton.cdiv(qstop-qstart,128))](
        qsl, seq, cu, ks, ke, qstart, qstop, num_warps=4)


@triton.jit
def _prepare_index_queries(Q, W, KS, KE, BQ, BW, S, E,
                           BEGIN, QOFF, ROWS,
                           H: tl.constexpr, DH: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, DH)
    h = tl.arange(0, DH // 128)
    # Copy bytes: FP8 is storage here, never an integer numeric conversion.
    codes = tl.load(Q.to(tl.pointer_type(tl.uint8)) + (BEGIN + row) * H * 128 + d,
                    mask=(row < ROWS) & (d < H * 128), other=0)
    tl.store(BQ.to(tl.pointer_type(tl.uint8)) + row * H * 128 + d,
             codes, mask=d < H * 128)
    weight = tl.load(W + (BEGIN + row) * H + h,
                     mask=(row < ROWS) & (h < H), other=0.)
    tl.store(BW + row * H + h, weight, mask=h < H)
    start = tl.load(KS + QOFF + row, mask=row < ROWS, other=0)
    end = tl.load(KE + QOFF + row, mask=row < ROWS, other=0)
    tl.store(S + row, start)
    tl.store(E + row, end)


def prepare_index_queries(q, weights, ks, ke, bq, bw, starts, ends, begin, qoff, rows):
    _prepare_index_queries[(bq.shape[0],)](q, weights, ks, ke, bq, bw, starts, ends,
        begin, qoff, rows, q.shape[1], triton.next_power_of_2(q.shape[1] * 128), num_warps=4)
