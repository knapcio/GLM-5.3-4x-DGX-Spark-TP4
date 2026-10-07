# SPDX-License-Identifier: Apache-2.0
"""FP4x v1: fused row writer and shared sparse/prefill readers.

ABI: E2M1 pairs [0:256], E4M3 scale/16 [256:288], FP32 pow2
row scale [288:292], zero [292:304], E4M3 RoPE [304:368].
Quantization matches glm_fp4_probe_quant at dc5ad9e. No tensor k_scale
on latent reconstruction; RoPE retains the released x/k_scale write.
"""
import torch
import triton
import triton.language as tl

ROW_BYTES = 368


@triton.jit
def _e4m3_rne(x):
    # Encode clamped FP32 directly: the pinned interpreter's FP8 cast rounds
    # ties up and ORs a mantissa carry into the exponent (124.666664 -> 64).
    # Integer RNE includes the carry and agrees with the probe's PyTorch cast.
    bits = x.to(tl.uint32, bitcast=True)
    # Signed arithmetic, clamped below the exponent of 2^127: no wrap even in
    # the unselected normal branch (Triton overflow sanitizers stay quiet).
    mag = tl.minimum(bits & 0x7fffffff, 0x7f000000).to(tl.int32)
    normal = ((mag + 0x7ffff + ((mag >> 20) & 1)) >> 20) - 960
    # E4M3 subnormals are multiples of 2^-9; round in that fixed-point grid.
    v = tl.abs(x) * 512.0
    lo = tl.minimum(v, 8.0).to(tl.uint32)
    frac = v - lo.to(tl.float32)
    sub = lo + ((frac > .5) | ((frac == .5) & ((lo & 1) != 0))).to(tl.uint32)
    code = tl.where(tl.abs(x) < .015625, sub.to(tl.int32), normal)
    code = tl.where(x != x, 127, code)
    return (code | ((bits >> 24) & 128).to(tl.int32)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _parts(z):
    # z is [32,16], FP32. Each CTA owns a full row and both nibbles.
    a = tl.max(tl.max(tl.abs(z), 1), 0)
    row = tl.exp2(tl.ceil(tl.log2(tl.where(a > 0, tl.div_rn(a, 2688.0), 1.0))))
    block = tl.minimum(tl.maximum(tl.div_rn(tl.max(tl.abs(z), 1), 6.0 * row), 0.0), 448.0)
    block8 = _e4m3_rne(block)
    block = block8.to(tl.float32)
    scale = block[:, None] * row
    v = tl.div_rn(tl.abs(z), tl.where(scale > 0, scale, 1.0))
    code = tl.full((32, 16), 0, tl.int32)
    code = tl.where(v > .25, 1, code)
    code = tl.where(v >= .75, 2, code)
    code = tl.where(v > 1.25, 3, code)
    code = tl.where(v >= 1.75, 4, code)
    code = tl.where(v > 2.5, 5, code)
    code = tl.where(v >= 3.5, 6, code)
    code = tl.where(v > 5.0, 7, code)
    sign = (z.to(tl.int32, bitcast=True).to(tl.uint32) >> 31) << 3
    return code | sign.to(tl.int32), block8, row


@triton.jit
def _levels(code):
    mag = code & 7
    val = tl.where(mag <= 4, mag.to(tl.float32) * .5,
                   tl.where(mag == 5, 3., tl.where(mag == 6, 4., 6.)))
    return tl.where((code & 8) != 0, -val, val)


@triton.jit
def _load_latent(KV, slot, d, KS: tl.constexpr, N: tl.constexpr,
                 SHADOW=None, MAP=None, TAG=None, POS=None, LOW=None, FLAG=None,
                 CAP: tl.constexpr=0):
    valid = (slot >= 0) & (slot < N)
    hit = valid & False
    if CAP:
        enabled=tl.load(FLAG)>0
        idx=tl.load(MAP+slot,mask=valid & enabled,other=-1).to(tl.int64)
        ok=valid & enabled & (idx>=0) & (idx<4*CAP)
        tag=tl.load(TAG+idx,mask=ok,other=-1).to(tl.int64)
        pos=tl.load(POS+idx,mask=ok,other=-1)
        lower=tl.load(LOW+idx//CAP,mask=ok,other=2147483647)
        hit=ok & (tag==slot) & (pos>=lower)
    packed = valid & ~hit
    base = KV.to(tl.pointer_type(tl.uint8)) + slot[:, None] * KS
    raw = tl.load(base + d[None, :] // 2, mask=packed[:, None], other=0).to(tl.int32)
    code = (raw >> ((d[None, :] & 1) * 4)) & 15
    # Mask in byte space: Triton's compiler cannot cast integer other=0 to FP8.
    block = tl.load(base + 256 + d[None, :] // 16, mask=packed[:, None], other=0)
    block = block.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    row = tl.load((KV.to(tl.pointer_type(tl.uint8)) + slot * KS + 288).to(tl.pointer_type(tl.float32)),
                  mask=packed, other=1.)
    # Preserve the probe's evaluation order: (code * block) * row, BF16 once.
    value = ((_levels(code) * block) * row[:, None]).to(tl.bfloat16)
    if CAP:
        raw8=tl.load(SHADOW+idx[:,None]*512+d[None,:],mask=hit[:,None],other=0)
        high=raw8.to(tl.float8e4nv,bitcast=True).to(tl.bfloat16)
        value=tl.where(hit[:,None],high,value)
    return value


@triton.jit
def _load_rope(KV, slot, rd, KS: tl.constexpr, N: tl.constexpr):
    valid = (slot >= 0) & (slot < N)
    base = KV.to(tl.pointer_type(tl.uint8)) + slot[:, None] * KS
    rope = tl.load(base + 304 + rd[None, :], mask=valid[:, None], other=0)
    return rope.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)


@triton.jit
def _pack_store(X, R, KV, SLOTS, SCALE,
                XS: tl.constexpr, RS: tl.constexpr, KS: tl.constexpr,
                T: tl.constexpr, N: tl.constexpr,
                SHADOW=None, MAP=None, TAG=None, POS=None, TOKEN_RING=None, TOKEN_POS=None,
                CAP: tl.constexpr=0):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(SLOTS + t).to(tl.int64)
    if (slot >= 0) & (slot < N):
        d = tl.arange(0, 512)
        z = tl.load(X + t * XS + d).to(tl.float32).reshape(32, 16)
        # Bad activations must not silently become a valid cache row.
        tl.device_assert(tl.sum((tl.abs(z) == float('inf')).to(tl.int32), 0) == 0, 'nonfinite FP4 latent')
        tl.device_assert(tl.sum((z != z).to(tl.int32), 0) == 0, 'NaN FP4 latent')
        code, block, row = _parts(z)
        pairs = code.reshape(256, 2)
        packed = (tl.sum(pairs * tl.full((1, 2), 1, tl.int32) * (1 << (tl.arange(0, 2)[None, :] * 4)), 1)).to(tl.uint8)
        base = KV.to(tl.pointer_type(tl.uint8)) + slot * KS
        tl.store(base + tl.arange(0, 256), packed)
        fp8_base = base.to(tl.pointer_type(tl.float8e4nv))
        tl.store(fp8_base + 256 + tl.arange(0, 32), block)
        tl.store((base + 288).to(tl.pointer_type(tl.float32)), row)
        pad = tl.arange(0, 16)
        tl.store(base + 292 + pad, 0, mask=pad < 12)
        rd = tl.arange(0, 64)
        rope = tl.load(R + t * RS + rd).to(tl.float32)
        ks = tl.load(SCALE)
        rope = _e4m3_rne(tl.minimum(tl.maximum(tl.div_rn(rope, ks), -448.), 448.))
        tl.store(fp8_base + 304 + rd, rope)
        if CAP:
            idx=tl.load(TOKEN_RING+t).to(tl.int64)
            pos=tl.load(TOKEN_POS+t)
            tl.device_assert((idx>=0)&(idx<4*CAP),'recent KV staging missing for target row')
            high=_e4m3_rne(tl.minimum(tl.maximum(tl.div_rn(z.reshape(512),ks),-448.),448.))
            # Released FP8 sparse reader applies the host latent scale.
            # Release uses 1.0; refuse a nonunit value in the Python wrapper.
            tl.store(SHADOW.to(tl.pointer_type(tl.float8e4nv))+idx*512+d,high)
            tl.store(TAG+idx,slot.to(tl.int32))
            tl.store(POS+idx,pos)
            tl.store(MAP+slot,idx.to(tl.int32))


def pack_store(x, rope, cache, slots, k_scale, recent=None, reader_scale=1.0):
    if cache.numel() == 0:
        return
    rope = rope.reshape(rope.shape[0], 64)
    slots = slots.flatten()
    if x.dtype != torch.bfloat16 or rope.dtype != torch.bfloat16 or x.shape[-1] != 512:
        raise ValueError('FP4x writer requires BF16 D512/R64')
    if cache.dtype != torch.uint8 or cache.shape[-1] != ROW_BYTES or not cache.is_contiguous():
        raise ValueError('FP4x requires contiguous uint8 368-byte rows')
    if x.stride(-1) != 1 or rope.stride(-1) != 1 or slots.stride(0) != 1:
        raise ValueError('FP4x writer requires unit inner strides')
    if slots.numel() > x.shape[0] or slots.numel() > rope.shape[0]:
        raise ValueError('FP4x slot/input extent mismatch')
    if k_scale.dtype != torch.float32 or k_scale.numel() != 1:
        raise ValueError('FP4x rope needs scalar FP32 k_scale')
    if slots.numel():
        extra = {}
        if recent is not None:
            if float(reader_scale)!=1.0:
                raise ValueError('recent KV requires released unit FP8 reader scale')
            from glm_recent_kv import bind
            bind(cache,recent)
            extra=dict(SHADOW=recent['shadow'],MAP=recent['mapping'],TAG=recent['tags'],
                       POS=recent['positions'],TOKEN_RING=recent['token_ring'],TOKEN_POS=recent['token_pos'],
                       CAP=recent['capacity'])
        _pack_store[(slots.numel(),)](x, rope, cache, slots, k_scale,
            x.stride(0), rope.stride(0), ROW_BYTES, slots.numel(), cache.numel() // ROW_BYTES,
            num_warps=4, num_stages=1, enable_fp_fusion=False, **extra)


@triton.jit
def _qdq(X, O, XS: tl.constexpr, T: tl.constexpr, FLAG=None, RECENT: tl.constexpr=False):
    t = tl.program_id(0)
    d = tl.arange(0, 512)
    z = tl.load(X + t * XS + d).to(tl.float32).reshape(32, 16)
    code, block, row = _parts(z)
    y = (_levels(code) * block.to(tl.float32)[:, None]) * row
    # Match the probe's preservation of nonfinite fresh-prefill inputs.
    y = tl.where((z == z) & (tl.abs(z) != float('inf')), y, z)
    if RECENT:
        # Dense fresh prefill in the FP8 control consumes original BF16 rows.
        y=tl.where(tl.load(FLAG)>0,z,y)
    tl.store(O + t * 512 + d, y.reshape(512).to(tl.bfloat16))


def qdq_prefill(x, out=None, recent=False):
    if out is None:  # standalone numeric/qualification fixture
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError('FP4x QDQ scratch shape/dtype/stride mismatch')
    if x.shape[0]:
        from glm_recent_kv import fresh_flag
        flag=fresh_flag(x.device,recent)
        _qdq[(x.shape[0],)](x, out, x.stride(0), x.shape[0],
            FLAG=flag if flag is not None else x,RECENT=flag is not None,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@triton.jit
def _read_rows(KV, SLOTS, O, N: tl.constexpr, T: tl.constexpr, OS: tl.constexpr,
               BN: tl.constexpr = 16):
    t = tl.program_id(0) * BN + tl.arange(0, BN)
    slot = tl.load(SLOTS + t, mask=t < T, other=-1).to(tl.int64)
    d = tl.arange(0, 512)
    y = _load_latent(KV, slot, d, 368, N)
    tl.store(O + t[:, None] * OS + d[None, :], y, mask=t[:, None] < T)
    rd = tl.arange(0, 64)
    rope = _load_rope(KV, slot, rd, 368, N)
    tl.store(O + t[:, None] * OS + 512 + rd[None, :], rope, mask=t[:, None] < T)


@triton.jit
def _gather(KV, O, BT, CU, TS, START,
            BS: tl.constexpr, BTS: tl.constexpr, OS: tl.constexpr,
            N: tl.constexpr, T: tl.constexpr, HAS_START: tl.constexpr, BN: tl.constexpr = 16,
            SHADOW=None, MAP=None, TAG=None, POS=None, LOW=None, FLAG=None, CAP: tl.constexpr=0):
    t = tl.program_id(0) * BN + tl.arange(0, BN)
    req = tl.load(TS + t, mask=t < T, other=0).to(tl.int64)
    begin = tl.load(CU + req)
    pos = t - begin
    if HAS_START:
        pos += tl.load(START + req)
    block = tl.load(BT + req * BTS + pos // BS, mask=t < T, other=0).to(tl.int64)
    slot = tl.where(t < T, block * BS + pos % BS, -1)
    d = tl.arange(0, 512)
    y = _load_latent(KV, slot, d, 368, N, SHADOW, MAP, TAG, POS, LOW, FLAG, CAP)
    tl.store(O + t[:, None] * OS + d[None, :], y, mask=t[:, None] < T)
    rd = tl.arange(0, 64)
    rope = _load_rope(KV, slot, rd, 368, N)
    tl.store(O + t[:, None] * OS + 512 + rd[None, :], rope, mask=t[:, None] < T)


def gather(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens,
           kv_cache_dtype=None, scale=None, seq_starts=None):
    if src_cache.dtype != torch.uint8 or src_cache.shape[-1] != 368 or not src_cache.is_contiguous():
        raise ValueError('packed prefill requires FP4x rows')
    if dst.dtype != torch.bfloat16 or dst.shape[-1] != 576 or dst.stride(-1) != 1:
        raise ValueError('FP4x context gather requires bounded BF16 D576 workspace')
    if num_tokens > dst.shape[0] or num_tokens > token_to_seq.numel():
        raise ValueError('FP4x context gather extent mismatch')
    if num_tokens:
        from glm_recent_kv import reader_args
        shadow,mapping,tags,positions,lower,flag,cap=reader_args(src_cache)
        _gather[(triton.cdiv(num_tokens, 16),)](src_cache, dst, block_table, cu_seq_lens,
            token_to_seq, seq_starts if seq_starts is not None else cu_seq_lens, src_cache.shape[1], block_table.stride(0), dst.stride(0),
            src_cache.numel() // 368, num_tokens, seq_starts is not None,
            SHADOW=shadow,MAP=mapping,TAG=tags,POS=positions,LOW=lower,FLAG=flag,CAP=cap,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
