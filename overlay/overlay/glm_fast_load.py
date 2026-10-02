# SPDX-License-Identifier: Apache-2.0
"""Env-gated safetensors slab loader for GB10.

GLM_FAST_LOAD=1 wraps the pinned vLLM lazy iterator. Preadv fills reusable
anonymous/pinned slabs in stock key order, preserving names, shapes, dtypes
and bytes. Slab size, read-ahead, thread count and consumed-page eviction
are environment settings; the current full-model profile selects 64 MiB,
256 MiB, four threads and targeted page eviction. GLM_FAST_LOAD_VERIFY=2
compares two tensors per shard against the stock safetensors mmap.

Credit: Willian-Zhang, vLLM#58726; local DeepSeek/Flash loader port.
"""
import collections
import concurrent.futures
import json
import logging
import os
import struct
import sys
import threading
import time

logger = logging.getLogger("glm_fast_load")

_DTYPES = {
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu",
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "U8": "uint8", "I16": "int16", "I32": "int32", "I64": "int64", "BOOL": "bool",
    "U16": "uint16", "U32": "uint32", "U64": "uint64",
}
_ALIGN = 4096
_CHUNK = 64 << 20


def enabled() -> bool:
    return os.environ.get("GLM_FAST_LOAD", "0").strip().lower() in ("1", "on", "true")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _log(msg):
    # vLLM's logger is configured per process; stderr always reaches `docker logs`.
    sys.stderr.write(f"glm-fast-load: {msg}\n")
    sys.stderr.flush()


def _mem_available_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def parse_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


def _torch_dtype(code):
    import torch

    name = _DTYPES.get(code)
    return getattr(torch, name, None) if name else None


class _Stats:
    def __init__(self):
        self.t0 = time.time()
        self.bytes = 0
        self.eager = 0
        self.passthrough = 0
        self.slabs = 0
        self.slab_bytes = 0
        self.live_peak = 0
        self.mem_start = _mem_available_kb()
        self.mem_low = self.mem_start
        self._last_sample = 0.0
        self.refs = []          # (StorageWeakRef, first tensor name, nbytes)
        self.verified = 0

    def sample(self, force=False):
        now = time.time()
        if force or now - self._last_sample >= 0.5:
            self._last_sample = now
            m = _mem_available_kb()
            if m >= 0 and (self.mem_low < 0 or m < self.mem_low):
                self.mem_low = m

    def live(self):
        return [r for r in self.refs if r[0] is not None and not r[0].expired()]

    def prune(self):
        self.refs = self.live()


def _read_into(fd, mv, off):
    got, n = 0, len(mv)
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at offset {off + got}")
        got += r


def _read_job(fd, flat, file_off, drop):
    """Fill ``flat`` (a uint8 view of a slab) from ``fd`` at ``file_off``; then drop those cache pages."""
    mv = memoryview(flat.numpy())
    try:
        n = len(mv)
        for c in range(0, n, _CHUNK):
            _read_into(fd, mv[c:c + _CHUNK], file_off + c)
    finally:
        mv.release()
    if drop and hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, file_off, flat.numel(), os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _alloc(nbytes, pinned):
    import torch

    if pinned:
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    import mmap

    return torch.frombuffer(mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
                            dtype=torch.uint8)


def _plan_file(names, header, slab_cap):
    """Split one shard into units in yield order: ('slab', [(name, off, info)...], size) or ('mmap', name)."""
    units, cur, used = [], [], 0
    for name in names:
        info = header.get(name)
        n = (info["data_offsets"][1] - info["data_offsets"][0]) if info else -1
        if info is None or n <= 0 or n > slab_cap or _torch_dtype(info["dtype"]) is None:
            if cur:
                units.append(("slab", cur, used)); cur, used = [], 0
            units.append(("mmap", name))
            continue
        off = (used + _ALIGN - 1) // _ALIGN * _ALIGN
        if off + n > slab_cap and cur:
            units.append(("slab", cur, used)); cur, used, off = [], 0, 0
        cur.append((name, off, info))
        used = off + n
    if cur:
        units.append(("slab", cur, used))
    return units


def fast_safetensors_iterator(files, keys_of, open_stock, skip=None, progress=None, stats=None):
    """Yield (name, tensor) for ``files`` in the stock order, with eager slab-backed tensors.

    ``keys_of(path)``: the stock name order of one shard; ``open_stock(path)``: a context manager with
    ``get_tensor`` (safetensors ``safe_open``) for pass-through tensors and verification;
    ``skip(name)``: the stock EP filter; ``progress``: wraps the file list (tqdm) like the stock loop.
    """
    import torch

    threads = max(1, _env_int("GLM_FAST_LOAD_THREADS", 16))
    slab_cap = max(1, _env_int("GLM_FAST_LOAD_SLAB_MB", 256)) << 20
    ahead = max(slab_cap, _env_int("GLM_FAST_LOAD_AHEAD_MB", 1024) << 20)
    pinned = os.environ.get("GLM_FAST_LOAD_PINNED", "1") == "1" and torch.cuda.is_available()
    drop = os.environ.get("GLM_FAST_LOAD_DROP_CACHE", "1") == "1"
    verify_n = _env_int("GLM_FAST_LOAD_VERIFY", 0)
    st = stats if stats is not None else _Stats()
    try:
        from torch.multiprocessing.reductions import StorageWeakRef
    except Exception:  # pragma: no cover
        StorageWeakRef = None

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads, thread_name_prefix="glm-fastload")
    # Lazily expanded stream of units across all shards: (file_state, unit)
    FileState = collections.namedtuple("FileState", "path fd base header last_idx")
    stream = collections.deque()      # planned, not yet submitted
    ready = collections.deque()       # submitted: (fs, unit, slab, futures)
    inflight = [0]
    file_iter = iter(progress(files) if progress else files)
    pending_files = collections.deque()
    open_fds = []

    def plan_next_file():
        try:
            path = next(file_iter)
        except StopIteration:
            return False
        base, header = parse_header(path)
        names = [n for n in keys_of(path) if not (skip and skip(n))]
        units = _plan_file(names, header, slab_cap)
        fd = os.open(path, os.O_RDONLY)
        open_fds.append(fd)
        fs = FileState(path, fd, base, header, len(units) - 1)
        if not units:
            os.close(fd); open_fds.remove(fd)
            return True
        for i, u in enumerate(units):
            stream.append((fs, u, i))
        return True

    def submit_one():
        while not stream:
            if not plan_next_file():
                return False
        fs, unit, i = stream.popleft()
        slab, futs = None, []
        if unit[0] == "slab":
            size = unit[2]
            # Every slab asks for the full cap (one caching-allocator size bin, reused across slabs).
            slab = _alloc(slab_cap if pinned else size, pinned)
            st.slabs += 1
            st.slab_bytes += slab.numel()
            if StorageWeakRef is not None:
                st.refs.append((StorageWeakRef(slab.untyped_storage()), unit[1][0][0], slab.numel()))
            for name, off, info in unit[1]:
                a, b = info["data_offsets"]
                futs.append(pool.submit(_read_job, fs.fd, slab[off:off + (b - a)], fs.base + a, drop))
            inflight[0] += size
        ready.append((fs, unit, i, slab, futs))
        return True

    max_units = max(64, _env_int("GLM_FAST_LOAD_MAX_UNITS", 4096))

    def fill():
        while inflight[0] < ahead and len(ready) < max_units:
            if not submit_one():
                break

    stock_handles = {}

    def stock(path):
        h = stock_handles.get(path)
        if h is None:
            cm = open_stock(path)
            h = (cm, cm.__enter__())
            stock_handles[path] = h
        return h[1]

    def close_file(fs):
        h = stock_handles.pop(fs.path, None)
        if h is not None:
            h[0].__exit__(None, None, None)
        try:
            os.close(fs.fd)
            open_fds.remove(fs.fd)
        except (OSError, ValueError):
            pass

    per_file_verified = collections.Counter()
    try:
        fill()
        while ready:
            fs, unit, i, slab, futs = ready.popleft()
            if unit[0] == "slab":
                for f in futs:
                    f.result()
                inflight[0] -= unit[2]
                fill()
                for name, off, info in unit[1]:
                    a, b = info["data_offsets"]
                    t = slab[off:off + (b - a)].view(_torch_dtype(info["dtype"])).reshape(info["shape"])
                    if per_file_verified[fs.path] < verify_n:
                        ref = stock(fs.path).get_tensor(name)
                        if ref.dtype != t.dtype or tuple(ref.shape) != tuple(t.shape) or not torch.equal(
                                ref.contiguous().reshape(-1).view(torch.uint8),
                                t.contiguous().reshape(-1).view(torch.uint8)):
                            raise RuntimeError(f"glm-fast-load: VERIFY mismatch for {name} in {fs.path}")
                        per_file_verified[fs.path] += 1
                        st.verified += 1
                    st.bytes += b - a
                    st.eager += 1
                    yield name, t
                    del t
                del slab
            else:
                fill()
                st.passthrough += 1
                yield unit[1], stock(fs.path).get_tensor(unit[1])
            st.sample()
            if unit[0] == "slab":
                st.prune()
                st.live_peak = max(st.live_peak, len(st.refs))
            if i == fs.last_idx:
                close_file(fs)
            if not ready:
                fill()
    finally:
        # Stop the readers before their descriptors are closed.
        pool.shutdown(wait=True, cancel_futures=True)
        for fs_ in list(stock_handles):
            h = stock_handles.pop(fs_)
            try:
                h[0].__exit__(None, None, None)
            except Exception:
                pass
        for fd in open_fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _empty_host_cache():
    try:
        import torch

        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass


def release(stats, label, wait_s=None):
    """After a load: return cached pinned blocks and malloc arenas, log a summary, check for escaped slabs.

    The consumer still holds the last yielded tensor when the generator finishes, so one slab is
    normally alive at this point; a watcher thread returns it to the driver once that view dies and
    warns only if a slab is still referenced ``GLM_FAST_LOAD_ESCAPE_S`` (120) seconds later.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    _empty_host_cache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    stats.sample(force=True)
    dt = max(time.time() - stats.t0, 1e-6)
    gib = stats.bytes / (1 << 30)
    _log(f"{label}: {stats.eager} eager + {stats.passthrough} mmap tensors, {gib:.1f} GiB in {dt:.1f} s "
         f"({gib / dt:.2f} GiB/s), {stats.slabs} slabs, peak {stats.live_peak} live, "
         f"verified {stats.verified}, MemAvailable start {stats.mem_start / 1048576:.1f} GiB "
         f"low {stats.mem_low / 1048576:.1f} GiB end {_mem_available_kb() / 1048576:.1f} GiB")
    if not stats.live():
        return
    limit = wait_s if wait_s is not None else _env_int("GLM_FAST_LOAD_ESCAPE_S", 120)

    def watch():
        t0 = time.time()
        while time.time() - t0 < limit:
            if not stats.live():
                _empty_host_cache()
                _log(f"{label}: last slab released after {time.time() - t0:.1f} s")
                return
            time.sleep(0.5)
        live = stats.live()
        if live:
            names = ", ".join(r[1] for r in live[:5])
            _log(f"WARNING {label}: {len(live)} slab(s) still referenced {limit} s after the load "
                 f"({sum(r[2] for r in live) >> 20} MiB host memory), first tensors: {names}")

    threading.Thread(target=watch, name="glm-fastload-release", daemon=True).start()


# --- vLLM wiring ---------------------------------------------------------------------------------

_load_count = [0]


def _make_wrapper(orig, wu):
    def safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy=None,
                                     local_expert_ids=None, **kwargs):
        if safetensors_load_strategy not in (None, "lazy"):
            yield from orig(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy,
                            local_expert_ids=local_expert_ids, **kwargs)
            return
        from safetensors.torch import safe_open

        files = sorted(hf_weights_files, key=wu._natural_sort_key)

        def keys_of(path):
            with safe_open(path, framework="pt") as f:
                return list(f.keys())

        def open_stock(path):
            return safe_open(path, framework="pt")

        def progress(fs):
            return wu.tqdm(fs, desc="Loading safetensors checkpoint shards (glm-fast-load)",
                           disable=not wu.enable_tqdm(use_tqdm_on_load), bar_format=wu._BAR_FORMAT)

        _load_count[0] += 1
        label = f"load {_load_count[0]} ({len(files)} files, {os.path.dirname(files[0]) if files else '-'})"
        stats = _Stats()
        try:
            yield from fast_safetensors_iterator(
                files, keys_of, open_stock,
                skip=lambda n: wu.should_skip_weight(n, local_expert_ids),
                progress=progress, stats=stats)
        finally:
            release(stats, label)

    safetensors_weights_iterator._glm_fast_load = True
    return safetensors_weights_iterator


def _check_source(module):
    import hashlib
    from pathlib import Path
    pins = json.loads((Path(__file__).resolve().parents[1] / 'source_pins.json').read_text())
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != pins[module.__name__]:
        raise RuntimeError('glm-fast-load: source drift ' + module.__name__)


def _patch_weight_utils(module):
    _check_source(module)
    orig = getattr(module, "safetensors_weights_iterator", None)
    if orig is None or getattr(orig, "_glm_fast_load", False):
        return
    for needed in ("_natural_sort_key", "tqdm", "enable_tqdm", "_BAR_FORMAT", "should_skip_weight"):
        if not hasattr(module, needed):
            raise RuntimeError(f"glm-fast-load: vllm weight_utils has no {needed}; engine drifted, refusing")
    module.safetensors_weights_iterator = _make_wrapper(orig, module)
    _log("armed (weight_utils.safetensors_weights_iterator wrapped)")


def _patch_default_loader(module):
    _check_source(module)
    import vllm.model_executor.model_loader.weight_utils as wu

    _patch_weight_utils(wu)
    if hasattr(module, "safetensors_weights_iterator"):
        module.safetensors_weights_iterator = wu.safetensors_weights_iterator


_TARGETS = {
    "vllm.model_executor.model_loader.weight_utils": _patch_weight_utils,
    "vllm.model_executor.model_loader.default_loader": _patch_default_loader,
}


def register():
    import importlib.abc
    import importlib.util

    class _Hook(importlib.abc.MetaPathFinder):
        _glm_fast_load = True

        def find_spec(self, name, path, target=None):
            fn = _TARGETS.get(name)
            if fn is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def patched(module):
                exec_module(module)
                fn(module)

            spec.loader.exec_module = patched
            return spec

    if not any(getattr(h, "_glm_fast_load", False) for h in sys.meta_path):
        sys.meta_path.insert(0, _Hook())
    # Already imported (e.g. registered late): patch in place.
    for name, fn in _TARGETS.items():
        mod = sys.modules.get(name)
        if mod is not None:
            fn(mod)
