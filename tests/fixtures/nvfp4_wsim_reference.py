# SPDX-License-Identifier: Apache-2.0
"""Offline CPU NVFP4 weight QDQ, before TP slicing and Marlin/MLA preparation.

No persistent tensors or kernels are added. FP32 tensor scale is amax/(6*448),
E4M3FN block scales and E2M1 codes use deterministic ties-to-even. Checkpoint
INT8 is biased unsigned bytes in int32 words, not signed byte storage.
Indexer weights are BF16 in this checkpoint: QDQ and cast back to BF16.
"""
import functools
import os
import re
import sys

GROUPS = frozenset(('attn', 'shared', 'indexer', 'dense', 'mtp'))
PROJECTIONS = frozenset(('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa',
                        'kv_b_proj', 'o_proj', 'fused_qkv_a_proj_with_mqa'))


def groups(env=None):
    value = (os.environ if env is None else env).get('GLM_NVFP4_WSIM', '').strip()
    if value in ('', '0'):
        return frozenset()
    parts = value.split(',')
    if any(p not in GROUPS for p in parts) or len(set(parts)) != len(parts):
        raise ValueError('GLM_NVFP4_WSIM must be comma-separated unique attn,shared,indexer,dense,mtp')
    return frozenset(parts)


def classify(name):
    m = re.fullmatch(r'model\.layers\.(\d+)\.(?:mtp_block\.)?(.+)\.(weight_packed|weight_scale|weight)', name)
    if not m:
        return None
    layer, site, leaf = int(m[1]), m[2], m[3]
    if layer == 78:
        # Channelwise INT8 only; eh_proj, norms, indexer and head stay original.
        if leaf != 'weight' and (site.startswith('mlp.experts.') or
                                site.startswith('mlp.shared_experts.') or
                                site.split('.')[-1] in PROJECTIONS):
            return 'mtp'
        return None
    if not 0 <= layer < 78:
        return None
    if site.startswith('self_attn.indexer.') and site.split('.')[-1] in ('wq_b', 'wq', 'wk', 'weights_proj'):
        return 'indexer' if leaf == 'weight' else None
    if layer == 0 or leaf == 'weight':
        return None
    if site.startswith('self_attn.') and site.split('.')[-1] in PROJECTIONS:
        return 'attn'
    if site.startswith('mlp.shared_experts.'):
        return 'shared'
    if layer in (1, 2) and site in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'mlp.gate_up_proj'):
        return 'dense'
    return None


def _tables():
    import numpy as np
    fp4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    fp8 = np.array([i * 2**-9 if i < 8 else
                   (1 + (i % 8)/8) * 2**(i//8 - 7) for i in range(127)], dtype=np.float32)
    return fp4, fp8


def _rne(values, table):
    """Positive float formats, monotonically ordered binary codes. Saturating RNE."""
    import numpy as np
    boundaries = (table[:-1] + table[1:]) * np.float32(.5)
    low = np.searchsorted(boundaries, values, side='left')
    # At midpoint choose the even binary code; includes zero/subnormal ties.
    tie = (low < len(boundaries)) & (values == boundaries[np.minimum(low, len(boundaries)-1)])
    return table[low + (tie & ((low & 1) != 0))]


def qdq(weight, tensor_scale=None):
    """FP32 [N,K], blocks contiguous along K. Scale can be supplied for row chunks."""
    import numpy as np
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError('NVFP4 requires finite matrices with K divisible by 16')
    if tensor_scale is None:
        tensor_scale = np.float32(np.max(np.abs(w)) / np.float32(6*448))
    if tensor_scale == 0:
        return np.zeros_like(w)
    fp4, fp8 = _tables()
    blocks = w.reshape(w.shape[0], -1, 16)
    scale8 = _rne(np.max(np.abs(blocks), axis=-1) / np.float32(6) / tensor_scale, fp8)
    scales = scale8 * tensor_scale
    normal = np.divide(np.abs(blocks), scales[..., None], out=np.zeros_like(blocks), where=scales[..., None] != 0)
    return (np.copysign(_rne(normal, fp4), blocks) * scales[..., None]).reshape(w.shape)


def unpack(packed):
    import numpy as np
    p = np.asarray(packed, dtype=np.int32).astype(np.uint32)
    return (((p[..., None] >> np.array([0, 8, 16, 24], dtype=np.uint32)) & 255)
            .reshape(p.shape[0], -1).astype(np.float32) - np.float32(128))


def pack(codes):
    import numpy as np
    biased = (np.asarray(codes, dtype=np.int32) + 128).astype(np.uint32).reshape(codes.shape[0], -1, 4)
    return np.bitwise_or.reduce(biased << np.array([0, 8, 16, 24], dtype=np.uint32), axis=-1).view(np.int32)


def reencode(weight, scale_dtype, group_size):
    import numpy as np
    import torch
    w = np.asarray(weight, dtype=np.float32)
    width = w.shape[1] if group_size == -1 else group_size
    blocks = w.reshape(w.shape[0], -1, width)
    scale = np.max(np.abs(blocks), axis=-1) / np.float32(127)
    # Use the stored scale for encoding, including its BF16 rounding.
    stored = torch.from_numpy(scale.copy()).to(scale_dtype)
    rounded = stored.float().numpy()
    if not np.isfinite(rounded).all() or np.any((scale > 0) & (rounded == 0)):
        raise ValueError('INT8 scale overflow/underflow')
    scaled = np.divide(blocks, rounded[..., None], out=np.zeros_like(blocks), where=rounded[..., None] != 0)
    codes = np.rint(scaled).clip(-127, 127).reshape(w.shape)
    return torch.from_numpy(pack(codes)), stored


def simulate(packed, scales, group_size=128, rows=64):
    """Two CPU row passes: full logical tensor amax, then QDQ and re-encode.

    Scratch is bounded by rows*K; output retains the original packed/scale shape.
    Never QDQ a TP shard independently (that would change the tensor scale).
    """
    import numpy as np
    import torch
    if packed.device.type != 'cpu' or packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError('expected CPU checkpoint int32 packed matrix')
    n, k = packed.shape[0], packed.shape[1]*4
    width = k if group_size == -1 else group_size
    if k % width or k % 16 or tuple(scales.shape) != (n, k//width) or rows < 1:
        raise ValueError('INT8 packed/scale contract mismatch')
    sc = scales.float().numpy()
    if not np.isfinite(sc).all() or np.any(sc < 0):
        raise ValueError('invalid weight scales')
    def decode(a, b):
        return (unpack(packed[a:b].numpy()).reshape(b-a, -1, width) * sc[a:b, :, None]).reshape(b-a, k)
    amax = np.float32(0)
    for a in range(0, n, rows):
        amax = max(amax, np.max(np.abs(decode(a, min(n, a+rows)))))
    ts = np.float32(amax / np.float32(6*448))
    out = torch.empty_like(packed)
    out_sc = torch.empty_like(scales)
    for a in range(0, n, rows):
        b = min(n, a+rows)
        q, s = reencode(qdq(decode(a, b), ts), scales.dtype, group_size)
        out[a:b].copy_(q); out_sc[a:b].copy_(s)
    return out, out_sc


def simulate_bf16(weight, rows=64):
    import numpy as np
    import torch
    if weight.device.type != 'cpu' or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError('indexer requires CPU BF16 matrix')
    amax = max(float(weight[a:a+rows].float().abs().max()) for a in range(0, len(weight), rows))
    ts = np.float32(amax / (6*448))
    out = torch.empty_like(weight)
    for a in range(0, len(weight), rows):
        out[a:a+rows].copy_(torch.from_numpy(qdq(weight[a:a+rows].float().numpy(), ts)).to(weight.dtype))
    return out


def transform(weights, files, selected):
    """Preserve iterator order; companion scale lookup also works across shards.

    Scale-first streams are handled with a second bounded transformation, never a
    retained packed-weight cache. The fast loader's reusable slabs remain read-only.
    """
    if not selected:
        yield from weights
        return
    from safetensors import safe_open
    from glm_fast_load import parse_header
    paths = {}
    for f in files:
        _, h = parse_header(f)
        for name in h:
            if classify(name) in selected:
                if name in paths:
                    raise RuntimeError('duplicate checkpoint tensor: ' + name)
                paths[name] = f
    def read(name):
        if name not in paths:
            raise RuntimeError('missing QDQ companion: ' + name)
        with safe_open(paths[name], framework='pt', device='cpu') as sf:
            return sf.get_tensor(name)
    pending, seen, counts = {}, set(), {}
    for name, value in weights:
        group = classify(name)
        if group not in selected:
            yield name, value
            continue
        if name in seen:
            raise RuntimeError('duplicate QDQ application: ' + name)
        seen.add(name)
        if name.endswith('.weight'):
            value = simulate_bf16(value)
            counts[group] = counts.get(group, 0) + 1
        elif name.endswith('.weight_packed'):
            sn = name.rsplit('.', 1)[0] + '.weight_scale'
            value, scale = simulate(value, read(sn), -1 if group == 'mtp' else 128)
            if sn not in seen:
                pending[sn] = scale
            counts[group] = counts.get(group, 0) + 1
        else:
            if name in pending:
                value = pending.pop(name)
            else:
                pn = name.rsplit('.', 1)[0] + '.weight_packed'
                _, value = simulate(read(pn), value, -1 if group == 'mtp' else 128)
        yield name, value
    if pending:
        raise RuntimeError('QDQ scales were not consumed: ' + ','.join(pending))
    sys.stderr.write('glm-nvfp4-wsim: transformed ' + str(counts) + '; CPU pre-TP QDQ, original layouts\n')


def wrap_iterator(original):
    @functools.wraps(original)
    def iterator(hf_weights_files, *args, **kwargs):
        selected = groups()
        files = list(hf_weights_files)
        yield from transform(original(files, *args, **kwargs), files, selected)
    iterator._glm_nvfp4_wsim = True
    return iterator
