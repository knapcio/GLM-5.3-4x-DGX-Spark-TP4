# SPDX-License-Identifier: Apache-2.0
"""Offline NVFP4 block16 format, using the qualified simulator's FP32 QDQ order."""
import hashlib
import json
from pathlib import Path

PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj')
MODULES = tuple(f'model.layers.{i}.self_attn.{p}' for i in range(1, 78) for p in PROJECTIONS)
ALGORITHM = 'glm-wsim-fp32-absmax-e2m1-e4m3-rne-block16-v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tables():
    import numpy as np
    return (np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32),
            np.array([i * 2**-9 if i < 8 else (1 + (i % 8)/8) * 2**(i//8 - 7)
                      for i in range(127)], dtype=np.float32))


def rne_codes(values, table):
    import numpy as np
    boundaries = (table[:-1] + table[1:]) * np.float32(.5)
    low = np.searchsorted(boundaries, values, side='left')
    tie = (low < len(boundaries)) & (values == boundaries[np.minimum(low, len(boundaries)-1)])
    return (low + (tie & ((low & 1) != 0))).astype(np.uint8)


def encode(w, tensor_scale):
    import numpy as np
    w = np.asarray(w, dtype=np.float32)
    ts = np.float32(tensor_scale)
    if w.ndim != 2 or w.shape[1] % 16 or not np.isfinite(w).all() or not np.isfinite(ts) or ts < 0:
        raise ValueError('finite [N,K] block16 matrix and nonnegative FP32 global scale required')
    if ts == 0:
        if np.any(w):
            raise ValueError('zero/underflow global scale with nonzero weights')
        return np.zeros((len(w), w.shape[1]//2), np.uint8), np.zeros((len(w), w.shape[1]//16), np.uint8)
    fp4, fp8 = tables()
    blocks = w.reshape(len(w), -1, 16)
    s = rne_codes(np.max(np.abs(blocks), axis=-1) / np.float32(6) / ts, fp8)
    scales = fp8[s] * ts
    normalized = np.divide(np.abs(blocks), scales[..., None], out=np.zeros_like(blocks), where=scales[..., None] != 0)
    codes = rne_codes(normalized, fp4) | (np.signbit(blocks).astype(np.uint8) << 3)
    codes = codes.reshape(w.shape)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), s


def dequant(values, scales, tensor_scale):
    import numpy as np
    fp4, fp8 = tables()
    v = np.asarray(values, dtype=np.uint8)
    s = np.asarray(scales, dtype=np.uint8)
    if v.ndim != 2 or s.shape != (len(v), v.shape[1]//8) or np.any(s > 126):
        raise ValueError('invalid NVFP4 shape/scale codes')
    codes = np.stack((v & 15, v >> 4), axis=-1).reshape(len(v), -1)
    signed = np.copysign(fp4[codes & 7], np.where(codes & 8, np.float32(-1), np.float32(1)))
    # Same multiplication association as qdq(): fp4 * (scale8 * tensor_scale).
    return (signed.reshape(len(v), -1, 16) * (fp8[s] * np.float32(tensor_scale))[..., None]).reshape(len(v), -1)


def decode_int8(packed, scales):
    import numpy as np
    p = packed.numpy().astype(np.uint32)
    codes = (((p[..., None] >> np.array([0, 8, 16, 24], np.uint32)) & 255)
             .reshape(len(p), -1).astype(np.float32) - np.float32(128))
    sc = scales.float().numpy()
    if sc.shape != (len(p), codes.shape[1]//128) or not np.isfinite(sc).all() or np.any(sc < 0):
        raise ValueError('expected finite nonnegative group128 scales')
    return (codes.reshape(len(p), -1, 128) * sc[..., None]).reshape(codes.shape)


def validate_tensors(tensors, shape):
    import torch
    n, k = shape
    v, s, g = (tensors[x] for x in ('weight_packed', 'weight_scale', 'weight_global_scale'))
    if k % 128 or v.dtype != torch.uint8 or tuple(v.shape) != (n, k//2):
        raise ValueError('invalid packed weight')
    if s.dtype != torch.float8_e4m3fn or tuple(s.shape) != (n, k//16):
        raise ValueError('invalid block scale')
    if g.dtype != torch.float32 or tuple(g.shape) != (1,) or not torch.isfinite(g).all() or (g < 0).any():
        raise ValueError('invalid global scale')
    sf = s.float()
    if not torch.isfinite(sf).all() or (sf < 0).any():
        raise ValueError('invalid E4M3 scales')
    # Pinned Marlin zeroes scale*128 < 2. Normalized full-tensor scales have max448,
    # so its automatic scale_factor is 1. Reject rather than silently lose subnormals.
    if ((sf > 0) & (sf < 1/64)).any():
        raise ValueError('scale below exact Marlin range (1/64)')
    if g.item() == 0 and (v.any() or sf.any()):
        raise ValueError('nonzero weight with zero global scale')


def read_manifest(root, modules=MODULES, verify=True):
    root = Path(root).resolve()
    m = json.loads((root/'manifest.json').read_text())
    if m.get('schema') != 1 or m.get('algorithm') != ALGORITHM or not m.get('complete') or set(m['tensors']) != set(modules):
        raise ValueError('incomplete or incompatible NVFP4 manifest')
    for entry in m['tensors'].values():
        p = root/entry['file']
        if p.parent.resolve() != root or p.is_symlink():
            raise ValueError('NVFP4 output must be a regular direct child')
        if verify and sha256(p) != entry['sha256']:
            raise ValueError('NVFP4 output hash mismatch: ' + str(p))
    return m
