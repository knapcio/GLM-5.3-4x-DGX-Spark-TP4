# SPDX-License-Identifier: Apache-2.0
"""Stream-owned align scratch and a graph-capturable C ABI launcher."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading

_LOCK = threading.Lock()
_LIB = None
_BUFFERS = {}
_ARENAS = {}
BLOCKS = (8, 16, 32, 48, 64)


def capacity(n, experts, block, pad=False):
    length = n + experts * (block - 1)
    if pad:
        length = (length + block - 1) // block * block
    return min(n * block, length) if n < experts else length


def prepare_library():
    """Boot/warmup only. Cache identity covers source and exact compiler version."""
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        source = Path(__file__).with_name('kernel.cu')
        nvcc = os.environ.get('GLM_MOE_DET_NVCC', '/usr/local/cuda/bin/nvcc')
        version = subprocess.check_output([nvcc, '--version'], timeout=10)
        key = hashlib.sha256(source.read_bytes() + version + b'-O3-sm_121-v1').hexdigest()
        cache = Path(os.environ.get('GLM_MOE_DET_CACHE', str(Path.home() / '.cache/glm-moe-det')))
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / (key + '.so')
        if not binary.exists():
            # Unique temporary output, atomic publication; concurrent ranks may
            # compile the same bytes but never dlopen a partially written file.
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                output = Path(tmp) / 'kernel.so'
                subprocess.run([nvcc, '-O3', '-std=c++17', '-arch=sm_121',
                                '-shared', '-Xcompiler=-fPIC', '-Xptxas=-v,-warn-spills',
                                str(source), '-o', str(output)], check=True, timeout=240)
                os.replace(output, binary)
        lib = ctypes.CDLL(str(binary))
        lib.glm_det_align.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.glm_det_align.restype = ctypes.c_int
        lib.glm_det_last_error.argtypes = []
        lib.glm_det_last_error.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def check_cuda_error():
    """Benchmark-only: surface unchecked launch errors at a named boundary.

    Device synchronization alone need not expose the host thread's last launch
    error. Call outside capture, before and after synchronizing pending work.
    """
    status = prepare_library().glm_det_last_error()
    if status:
        raise RuntimeError(f'moe-det: CUDA boundary error ({status})')


def prepare_stream(device=None, stream=None):
    """Allocate the bounded arena once during load/warmup, covering every shape."""
    import torch
    stream = torch.cuda.current_stream(device) if stream is None else stream
    device = stream.device if device is None else device
    key = (device.index, stream.cuda_stream)
    arena = _ARENAS.get(key)
    if arena is not None:
        return arena
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('moe-det: prepare this stream before capture')
    with _LOCK:
        arena = _ARENAS.get(key)
        if arena is None:
            with torch.cuda.stream(stream):
                def empty(size):
                    return torch.empty(size, dtype=torch.int32, device=device)
                arena = (empty(65536 + 256 * 63), empty((65536 + 256 * 7) // 8),
                         empty(1), empty(64 * 256), empty(65536))
                _ARENAS[key] = arena
    return arena


def prepare_capture_stream():
    """Prepare the stream selected by torch.cuda.graph without stream=.

    Construct, but do not enter, the same context as the pinned FULL manager.
    PyTorch initializes/reuses its internal side stream in the constructor.
    All arena allocation therefore happens before capture_begin, at model load.
    """
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('moe-det: prepare capture stream outside capture')
    context = torch.cuda.graph(torch.cuda.CUDAGraph())
    stream = getattr(context, 'capture_stream', None)
    if stream is None:
        raise RuntimeError('moe-det: unsupported PyTorch graph stream selection')
    return prepare_stream(stream=stream)


def deterministic_align(topk_ids, block_size, num_experts, expert_map=None,
                        pad_sorted_ids=False, ignore_invalid_experts=False):
    """Returned buffers are scratch: consume before the next same-stream align.

    Prepare/warm each stream first; its fixed arena covers all supported shapes.
    Cold stream capture fails before allocation/JIT. Shape views allocate no
    device memory. Python keys depend only on metadata, never values.
    """
    import torch
    n = topk_ids.numel()
    if (not topk_ids.is_cuda or not topk_ids.is_contiguous() or topk_ids.ndim != 2
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or not 1 <= num_experts <= 256 or not 0 <= n <= 65536
            or block_size not in BLOCKS):
        raise ValueError('moe-det: unsupported routing metadata')
    if expert_map is not None and (expert_map.device != topk_ids.device
            or expert_map.dtype != torch.int32 or not expert_map.is_contiguous()
            or expert_map.shape != (num_experts,)):
        raise ValueError('moe-det: unsupported expert map metadata')
    with torch.cuda.device(topk_ids.device):
        stream = torch.cuda.current_stream()
        capturing = torch.cuda.is_current_stream_capturing()
        if _LIB is None and capturing:
            raise RuntimeError('moe-det: library must be prepared before capture')
        lib = _LIB if _LIB is not None else prepare_library()
        length = capacity(n, num_experts, block_size, pad_sorted_ids)
        key = (topk_ids.device.index, stream.cuda_stream, n, num_experts, block_size, bool(pad_sorted_ids))
        # Arena construction is load/warmup work; a warmed stream never allocates
        # device memory again, including previously unseen eager prefill shapes.
        arena = prepare_stream(topk_ids.device, stream)
        buffers = _BUFFERS.get(key)
        if buffers is None:
            with _LOCK:
                buffers = _BUFFERS.get(key)
                if buffers is None:
                    ids = arena[0].narrow(0, 0, length)
                    experts = arena[1].narrow(0, 0, (length + block_size - 1) // block_size)
                    counts = arena[3].narrow(0, 0, ((n + 1023) // 1024) * num_experts if n > 128 else 0)
                    ranks = arena[4].narrow(0, 0, n if n > 128 else 0)
                    result = (ids, experts, arena[2])
                    buffers = (*result, counts, ranks, result)
                    _BUFFERS[key] = buffers
        ids, experts, post, counts, ranks, result = buffers
        status = lib.glm_det_align(topk_ids.data_ptr(), expert_map.data_ptr() if expert_map is not None else None,
                                  ids.data_ptr(), experts.data_ptr(), post.data_ptr(), counts.data_ptr(),
                                  ranks.data_ptr(), n, num_experts, block_size, length,
                                  topk_ids.element_size(), int(ignore_invalid_experts), stream.cuda_stream)
        if status:
            raise RuntimeError(f'moe-det: CUDA launch failed ({status})')
        return result
