# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the actual Triton AST bodies, with byte-addressed pointers.

This is a numeric/indexing model, not Triton compiler or hardware evidence.
Pinned-image tests separately use the real Triton interpreter and compiler.
"""
import ast
from pathlib import Path
import numpy as np
import torch


class DType:
    def __init__(self, name, size):
        self.name, self.size = name, size


DTYPES = {n: DType(n, s) for n, s in (
    ('int32', 4), ('int64', 8), ('uint32', 4), ('uint8', 1),
    ('float32', 4), ('bfloat16', 2), ('float8e4nv', 1))}


def decode(raw, dtype):
    raw = np.ascontiguousarray(raw)
    if dtype.name == 'float8e4nv':
        return torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float().numpy()
    if dtype.name == 'bfloat16':
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return raw.view(np.dtype(dtype.name))


def encode(value, dtype):
    value = np.ascontiguousarray(value)
    if dtype.name in ('float8e4nv', 'bfloat16'):
        target = torch.float8_e4m3fn if dtype.name == 'float8e4nv' else torch.bfloat16
        return torch.from_numpy(value.astype(np.float32)).to(target).view(torch.uint8).numpy()
    return value.astype(dtype.name).view(np.uint8)


def pinned_fp8_cast(value):
    """Model the pinned Triton 3.7.1 FP32->E4M3 cast's finite normal path.

    runtime/interpreter.py::_convert_float adds the cutoff bit (half-up),
    then ORs significand and exponent instead of propagating a carry. This
    mode covers the random-row regression; it is not a full interpreter.
    """
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    exponent = ((bits >> 23) & 255).astype(np.int32)
    mantissa = bits & 0x7fffff
    out_exp = np.clip(exponent - 120, 0, 15).astype(np.uint8)
    out_sig = ((mantissa >> 20) + ((mantissa & 0x80000) != 0)).astype(np.uint8)
    shift = np.where(out_exp == 0, 121 - exponent, 0)
    # Matches the interpreter's post-rounding subnormal shift as well.
    out_sig = np.where((out_exp == 0) & (exponent != 0),
                       (out_sig >> shift) | (1 << (3 - shift)), out_sig).astype(np.uint8)
    return (((bits >> 24) & 128) | (out_exp << 3) | out_sig).astype(np.uint8)


class Tensor(np.ndarray):
    fp8_interpreter = False

    def to(self, dtype, bitcast=False):
        if bitcast:
            if dtype.name == 'float8e4nv':
                return tensor(decode(self.view(np.uint8), dtype).reshape(self.shape))
            return tensor(self.view(dtype.name))
        if dtype.name in ('float8e4nv', 'bfloat16'):
            if dtype.name == 'float8e4nv' and self.fp8_interpreter:
                return tensor(decode(pinned_fp8_cast(self), dtype).reshape(self.shape))
            return tensor(decode(encode(self, dtype), dtype).reshape(self.shape))
        return tensor(self.astype(dtype.name))


def tensor(x):
    return np.asarray(x).view(Tensor)


class Ptr:
    __array_priority__ = 10000

    def __init__(self, raw, dtype, offset=0):
        self.raw, self.dtype, self.offset = raw, dtype, offset

    @classmethod
    def of(cls, x):
        names = {torch.uint8:'uint8', torch.int32:'int32', torch.int64:'int64',
                 torch.float32:'float32', torch.bfloat16:'bfloat16'}
        return cls(x.view(torch.uint8).numpy().reshape(-1), DTYPES[names[x.dtype]])

    def __add__(self, off):
        return Ptr(self.raw, self.dtype, self.offset + np.asarray(off) * self.dtype.size)

    def to(self, dtype):
        return Ptr(self.raw, dtype, self.offset)


class TL:
    def __init__(self):
        self.pid = (0, 0)
        for name, typ in DTYPES.items():
            setattr(self, name, typ)

    def pointer_type(self, dtype): return dtype
    def program_id(self, i): return tensor(np.int32(self.pid[i]))
    def arange(self, a, b): return tensor(np.arange(a, b, dtype=np.int32))
    def where(self, c, a, b): return tensor(np.where(c, a, b))
    def abs(self, a): return tensor(np.abs(a))
    def max(self, a, axis): return tensor(np.max(a, axis=axis))
    def sum(self, a, axis): return tensor(np.sum(a, axis=axis, dtype=a.dtype))
    def full(self, shape, v, dtype): return tensor(np.full(shape, v, dtype=dtype.name))
    def zeros(self, shape, dtype): return self.full(shape, 0, dtype)
    def log2(self, a): return tensor(np.log2(a).astype(np.float32))
    def ceil(self, a): return tensor(np.ceil(a))
    def exp2(self, a): return tensor(np.exp2(a).astype(np.float32))
    def minimum(self, a, b): return tensor(np.minimum(a, b))
    def maximum(self, a, b): return tensor(np.maximum(a, b))
    def div_rn(self, a, b): return tensor(np.asarray(a, dtype=np.float32) / np.asarray(b, dtype=np.float32))
    def trans(self, a): return tensor(a.T)
    def dot(self, a, b): return tensor(a.astype(np.float32) @ b.astype(np.float32))
    def device_assert(self, p, msg):
        if not np.all(p): raise ValueError(msg)

    def atomic_or(self, p, value, mask=True, sem=None):
        self.store(p, self.load(p, mask=mask, other=0).astype(np.int32) | value, mask)

    def load(self, p, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(p.offset, mask)
        offsets = np.where(mask, offsets, 0).astype(np.int64)
        idx = offsets[..., None] + np.arange(p.dtype.size)
        raw = p.raw[idx].copy()
        value = decode(raw.reshape(-1), p.dtype).reshape(offsets.shape)
        return tensor(np.where(mask, value, other))

    def store(self, p, value, mask=True):
        offsets, value, mask = np.broadcast_arrays(p.offset, value, mask)
        off = offsets[mask].astype(np.int64)
        if not off.size: return
        raw = encode(value[mask], p.dtype).reshape(-1, p.dtype.size)
        idx = off[:, None] + np.arange(p.dtype.size)
        p.raw[idx] = raw


def kernels(root):
    tl = TL()
    ns = {'tl': tl, 'float': float, 'range': range}
    for file in ('glm_fp4_kv_kernel.py', 'glm_fp4_mla_kernel.py', 'glm_fp4_mla_split_kernel.py', 'glm_fp4_prefill_kernel.py', 'glm_fp4_mla_prefill.py', 'glm_recent_kv_kernel.py'):
        tree = ast.parse((Path(root)/file).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list]
        for n in nodes:
            n.decorator_list = []
            n.returns = None
            for a in n.args.args:
                a.annotation = None
        mod = ast.Module(body=nodes, type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(Path(root)/file), 'exec'), ns)
    return tl, ns
