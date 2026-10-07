# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allan Clark
# Copyright 2026 knapcio
"""Modified port of ajclark's coalesced.py and streaming.py at f0b64af.

Original: https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark
Changes: preserve stock key order and MTP/EP selection, pack disjoint aligned
ranges, cap staging/owned allocations, account for unified destination commit,
portable buffered reads for Mac tests, and reuse this recipe's pinned wiring.
See LICENSES/ajclark-NOTICE.txt and docs/coalesced-loader.md.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import queue
import struct
import sys
import threading
import time

import glm_fast_load as fast
import glm_mtp_select

ALIGN = 4096
MIB = 1 << 20
FLOOR = 3 << 30  # emergency only; ordinary pressure is baseline-relative
MAX_BATCH = 128 * MIB
MAX_OWNED = 4096 * MIB
MAX_HEADER_BYTES = 64 * MIB
MAX_TENSORS = 200_000


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def options(env=None):
    env = os.environ if env is None else env
    result = {}
    for key, default, maximum in (("BATCH_MB", 128, 128), ("OWNED_MB", 4096, 4096),
                                  ("THREADS", 16, 32)):
        value = int(env.get("GLM_COALESCED_" + key, default))
        if not 1 <= value <= maximum:
            raise ValueError(f"GLM_COALESCED_{key} must be 1..{maximum}")
        result[key.lower()] = value
    direct = env.get("GLM_COALESCED_DIRECT", "1")
    if direct not in ("0", "1"):
        raise ValueError("GLM_COALESCED_DIRECT must be 0 or 1")
    if result['owned_mb'] < 3 * result['batch_mb']:
        raise ValueError("owned budget must fit consumer, queued and producer batches")
    result['direct'] = direct == '1'
    for key, default, maximum in (("EMERGENCY_MB", 3072, 9216), ("MARGIN_MB", 1024, 4096)):
        value = int(env.get("GLM_COALESCED_" + key, default))
        if not 1 <= value <= maximum:
            raise ValueError(f"GLM_COALESCED_{key} must be 1..{maximum}")
        result[key.lower()] = value
    return result


def log(metrics):
    sys.stderr.write("glm-coalesced-load: " + json.dumps(metrics, sort_keys=True) + "\n")
    sys.stderr.flush()


def guard(extra=0, floor=FLOOR):
    available = fast._mem_available_kb()
    if available < 0:
        if sys.platform.startswith('linux'):
            raise MemoryError("coalesced loader cannot read MemAvailable")
        return None  # Mac CPU tests; fleet enforcement is mandatory.
    available *= 1024
    if available - extra < floor:
        raise MemoryError("coalesced loader would cross the emergency memory floor")
    return available


class MemoryGuard:
    """Only loader storage consumes the relative budget; native writes earn credit."""
    def __init__(self, opts, metrics=None, scope=None):
        self.floor = opts['emergency_mb'] * MIB
        self.margin = opts['margin_mb'] * MIB
        self.metrics, self.scope = metrics, scope
        self.staging, self.owned, self.owned_peak = 0, 0, 0
        self.baseline = guard(floor=self.floor)
        self.commit_start = scope.destination.committed if scope else 0
        if metrics is not None:
            metrics.update(mem_available_start_bytes=self.baseline,
                           emergency_floor_bytes=self.floor, margin_bytes=self.margin)
            sample_memory(metrics, self.baseline)

    def check(self, extra=0):
        if self.scope:
            self.scope.team.check()
        available = guard(extra, self.floor)
        transient = self.staging + self.owned
        # Owning outputs can expire while their allocator blocks remain cached.
        # Keep the charged high-water mark until load teardown; it is bounded by
        # the same owned limit and must not be misclassified as external pressure.
        accounted = self.staging + max(self.owned, self.owned_peak)
        committed = self.scope.destination.committed - self.commit_start if self.scope else 0
        if self.baseline is not None and available is not None:
            if transient + extra > self.baseline - self.margin:
                raise MemoryError('coalesced loader transient footprint exceeds relative budget')
            # Credit tracked live transient bytes to isolate unexplained pressure.
            # Destination credit comes from native writes, never checkpoint bytes.
            if available + committed + accounted < self.baseline - self.margin:
                raise MemoryError('coalesced loader detected external memory pressure')
        if self.metrics is not None:
            self.metrics.update(destination_committed_bytes=committed,
                                transient_live_bytes=transient,
                                transient_accounted_bytes=accounted,
                                relative_budget_bytes=(self.baseline - self.margin
                                                       if self.baseline is not None else None))
        sample_memory(self.metrics, available)
        return available


def stamp(path):
    s = os.stat(path)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    dest: int


@dataclass(frozen=True)
class Batch:
    file: str
    segments: tuple
    tensors: tuple  # (name, output byte offset, info), in stock key order
    bytes: int


def pack(path, rows):
    """Merge overlapping/adjacent physical pages; typed views keep stock order."""
    ranges = []
    for _, start, end, _ in sorted(rows, key=lambda r: r[1]):
        a, b = start // ALIGN * ALIGN, align(end)
        if ranges and a <= ranges[-1][1]:
            ranges[-1][1] = max(b, ranges[-1][1])
        else:
            ranges.append([a, b])
    segments, size = [], 0
    for a, b in ranges:
        segments.append(Segment(a, b, size))
        size += b - a
    tensors = []
    for name, a, b, info in rows:
        seg = next(s for s in segments if s.start <= a and b <= s.end)
        tensors.append((name, seg.dest + a - seg.start, info))
    return Batch(path, tuple(segments), tuple(tensors), size)


def inspect(files, keys_of, skip, batch_bytes, owned_limit, memory=None):
    """Validate all headers before placement; apply selection before planning I/O."""
    import torch
    plans, stamps, seen, header_bytes = [], {}, set(), 0
    for path in files:
        path = os.fspath(path)
        if path in stamps:
            raise ValueError('duplicate checkpoint file')
        before = stamps[path] = stamp(path)
        with open(path, 'rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('truncated safetensors header')
            n = struct.unpack('<Q', raw)[0]
            if n > min(100_000_000, before[2] - 8):
                raise ValueError('invalid safetensors header length')
            header_bytes += n
            if header_bytes > MAX_HEADER_BYTES:
                raise MemoryError('checkpoint headers exceed 64 MiB metadata budget')
            (memory.check if memory else guard)(8 * n)  # decoded header reservation
            header = json.loads(f.read(n))
        base, rows, packed_size = 8 + n, [], 0
        # keys_of uses safe_open: the official parser validates all ranges too.
        for name in keys_of(path):
            if skip and skip(name):
                continue
            if name in seen:
                raise ValueError('duplicate checkpoint tensor: ' + name)
            seen.add(name)
            if len(seen) > MAX_TENSORS:
                raise MemoryError('checkpoint exceeds 200000 tensor metadata budget')
            info = header[name]
            dtype = fast._torch_dtype(info['dtype'])
            if dtype is None:
                raise ValueError('unsupported coalesced dtype: ' + info['dtype'])
            a, b = info['data_offsets']
            itemsize = torch.empty((), dtype=dtype).element_size()
            size = itemsize
            for dim in info['shape']:
                size *= dim
            if size != b - a or not 0 <= a <= b <= before[2] - base:
                raise ValueError('invalid checkpoint tensor range: ' + name)
            # Outputs have page-aligned starts; absolute tensor alignment is required.
            if (base + a) % itemsize:
                raise ValueError('unaligned tensor encoding: ' + name)
            if size == 0:
                if rows:
                    plans.append(pack(path, rows)); rows, packed_size = [], 0
                plans.append(Batch(path, (), ((name, 0, info),), 0))
                continue
            padded = align(base + b) - (base + a) // ALIGN * ALIGN
            if padded > owned_limit:
                raise MemoryError('largest tensor exceeds coalesced owned budget: ' + name)
            # Conservative page charge avoids quadratic repacking for each tensor.
            if rows and packed_size + padded > batch_bytes:
                plans.append(pack(path, rows)); rows, packed_size = [], 0
            rows.append((name, base + a, base + b, info))
            packed_size += padded
        if rows:
            plans.append(pack(path, rows))
        if stamp(path) != before:
            raise ValueError('checkpoint changed during header inspection')
    return plans, stamps


def read_aligned(fd, view, offset, required, direct):
    """Upstream short-read handling; portable pread only for buffered Mac tests."""
    done, calls, buffered_fd = 0, 0, None
    try:
        while done < required:
            active = fd if buffered_fd is None else buffered_fd
            if hasattr(os, 'preadv'):
                n = os.preadv(active, [view[done:]], offset + done)
            else:
                # Bound Python bytes even when a test batch contains a large tensor.
                data = os.pread(active, min(len(view) - done, 4 * MIB), offset + done)
                n = len(data)
                view[done:done + n] = data
            calls += 1
            if not n:
                raise EOFError(f'short checkpoint read at {offset + done}')
            done += n
            if done < required and direct and buffered_fd is None:
                buffered_fd = os.open(f'/proc/self/fd/{fd}', os.O_RDONLY)
        return done, calls
    finally:
        if buffered_fd is not None:
            os.close(buffered_fd)


class OwnedStorageBudget:
    """Upstream storage weak references charge aliases, including retained views."""
    def __init__(self, limit, metrics=None, memory=None):
        self.limit, self.live, self.peak, self.refs = limit, 0, 0, []
        self.metrics = metrics
        self.memory = memory

    def reserve(self, size):
        remaining = []
        for ref, charge in self.refs:
            if ref.expired():
                self.live -= charge
            else:
                remaining.append((ref, charge))
        self.refs = remaining
        if self.live + size > self.limit:
            raise MemoryError('native loader retained tensors beyond coalesced owned budget')
        if self.memory:
            self.memory.owned = self.live
            self.memory.owned_peak = self.peak
            self.memory.check(size)
        else:
            sample_memory(self.metrics, guard(size))

    def track(self, tensor):
        from torch.multiprocessing.reductions import StorageWeakRef
        size = tensor.untyped_storage().nbytes()
        self.refs.append((StorageWeakRef(tensor.untyped_storage()), size))
        self.live += size
        self.peak = max(self.peak, self.live)
        if self.memory:
            self.memory.owned = self.live


class RangeReader:
    """Modified upstream two-tile reader: packed ranges, one upload per usual batch."""
    def __init__(self, tile_bytes, threads, cuda, direct, metrics, memory=None):
        import torch
        self.cuda, self.direct, self.metrics = cuda, direct, metrics
        self.tile_bytes, self.threads = tile_bytes, threads
        self.stream = torch.cuda.current_stream() if cuda else None
        self.slots, self.next_slot, self.fd, self.path = [], 0, None, None
        allocation = (1 << (tile_bytes - 1).bit_length()) if cuda else tile_bytes + ALIGN
        sample_memory(metrics, (memory.check if memory else guard)(2 * allocation))
        if memory:
            memory.staging = 2 * allocation
        for _ in range(2):
            # Keep pinned requests in the exact power-of-two allocator bin.
            # Adding 4096 can round a 128 MiB host request up to 256 MiB.
            owner = torch.empty(allocation, dtype=torch.uint8, pin_memory=cuda)
            start = (-owner.data_ptr()) % ALIGN
            if cuda and start:
                raise ValueError('pinned allocator is not page-aligned; refusing extra staging allocation')
            tensor = owner[start:start + tile_bytes]
            self.slots.append([tensor, memoryview(tensor.numpy()), None])
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='glm-coalesced-read')
        metrics['staging_bytes'] = 2 * allocation

    def fill(self, batch, output):
        import torch
        if self.path != batch.file:
            if self.fd is not None:
                os.close(self.fd)
            self.fd = None
            self.fd = os.open(batch.file, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
            self.path = batch.file
            source = os.fstat(self.fd)
            self.file_size = source.st_size
            if hasattr(self, 'stamps') and (source.st_dev, source.st_ino, source.st_size,
                                           source.st_mtime_ns, source.st_ctime_ns) != self.stamps[batch.file]:
                raise ValueError('checkpoint descriptor changed before read')
        final_event = None
        for position in range(0, batch.bytes, self.tile_bytes):
            slot = self.slots[self.next_slot % 2]; self.next_slot += 1
            if slot[2] is not None:
                tick = time.monotonic(); slot[2].synchronize()
                self.metrics['staging_wait_seconds'] += time.monotonic() - tick
            length = min(self.tile_bytes, batch.bytes - position)
            quantum = max(ALIGN, align((length + self.threads - 1) // self.threads))
            futures, tick, error = [], time.monotonic(), None

            def collect(future):
                nonlocal error
                try:
                    received, calls = future.result()
                    self.metrics['read_bytes'] += received
                    self.metrics['read_calls'] += calls
                except BaseException as exc:
                    error = error or exc

            for seg in batch.segments:
                left, right = max(position, seg.dest), min(position + length, seg.dest + seg.end - seg.start)
                for dest in range(left, right, quantum):
                    n = min(quantum, right - dest)
                    offset = seg.start + dest - seg.dest
                    # Required bytes omit final EOF alignment padding.
                    required = min(n, self.file_size - offset)
                    if required <= 0:
                        raise EOFError('checkpoint truncated during read')
                    futures.append(self.pool.submit(read_aligned, self.fd,
                        slot[1][dest - position:dest - position + n], offset, required, self.direct))
                    if len(futures) >= 2 * self.threads:
                        collect(futures.pop(0))
            for future in futures:  # drain every job even after failure
                collect(future)
            self.metrics['reader_wait_seconds'] += time.monotonic() - tick
            if error is not None:
                raise error
            if not self.direct and hasattr(os, 'posix_fadvise'):
                # Target only consumed extents, never global drop_caches.
                for seg in batch.segments:
                    os.posix_fadvise(self.fd, seg.start, seg.end - seg.start, os.POSIX_FADV_DONTNEED)
            tick = time.monotonic()
            output[position:position + length].copy_(slot[0][:length], non_blocking=self.cuda)
            if self.cuda:
                final_event = torch.cuda.Event(); final_event.record(self.stream)
                slot[2] = final_event
            self.metrics['upload_enqueue_seconds'] += time.monotonic() - tick
            self.metrics['uploads'] += 1
        return final_event

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        try:
            if self.cuda:
                tick = time.monotonic(); self.stream.synchronize()
                self.metrics['upload_wait_seconds'] += time.monotonic() - tick
        finally:
            if self.fd is not None:
                os.close(self.fd); self.fd = None
            for slot in self.slots:
                slot[1].release()
            self.slots.clear()
            fast._empty_host_cache()


def sample_memory(metrics, available):
    if metrics is not None and available is not None:
        metrics['mem_available_low_bytes'] = min(metrics.get('mem_available_low_bytes', available), available)
        metrics['mem_available_peak_bytes'] = max(metrics.get('mem_available_peak_bytes', available), available)


def check_direct(stamps):
    import torch
    if not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv'):
        raise ValueError('direct I/O requires Linux; GLM_COALESCED_DIRECT=0 for Mac CPU tests')
    owner = torch.empty(ALIGN * 2, dtype=torch.uint8)
    start = (-owner.data_ptr()) % ALIGN
    view = memoryview(owner[start:start + ALIGN].numpy())
    checked = set()
    try:
        for path, before in stamps.items():
            if before[0] in checked:
                continue
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                if os.preadv(fd, [view], 0) <= 0:
                    raise EOFError('empty checkpoint direct-I/O probe')
            finally:
                os.close(fd)
            checked.add(before[0])
    finally:
        view.release()


def prefetch(iterator, cuda):
    """Modified upstream one-batch queue; consumer stream records owning outputs."""
    import torch
    ready, slot, stop = queue.Queue(maxsize=1), threading.Semaphore(1), threading.Event()
    device = torch.cuda.current_device() if cuda else None
    stream = torch.cuda.Stream(device=device) if cuda else None
    if cuda:
        stream.wait_stream(torch.cuda.current_stream(device))

    def put(value):
        while not stop.is_set():
            try:
                ready.put(value, timeout=.1); return
            except queue.Full:
                pass

    def pump():
        terminal = ('done', None)
        try:
            if cuda:
                torch.cuda.set_device(device); torch.cuda.set_stream(stream)
            while not stop.is_set():
                if not slot.acquire(timeout=.1):
                    continue
                if stop.is_set():
                    break
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                put(('batch', value)); del value
        except BaseException as exc:
            terminal = ('error', exc)
        finally:
            try:
                iterator.close()
            except BaseException as exc:
                terminal = ('error', exc)
        put(terminal)

    thread = threading.Thread(target=pump, name='glm-coalesced-pump', daemon=True)
    thread.start()
    try:
        while True:
            kind, value = ready.get(); slot.release()
            if kind == 'error':
                raise value
            if kind == 'done':
                break
            if cuda:
                value[2].record_stream(torch.cuda.current_stream(device))
            yield value
            del value
    finally:
        stop.set(); thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError('coalesced producer failed to stop; abort worker')


def coalesced_safetensors_iterator(files, keys_of, open_stock, skip=None, progress=None, stats=None):
    import torch
    o = options()
    cuda = torch.cuda.is_available()
    batch_bytes, owned_limit = o['batch_mb'] * MIB, o['owned_mb'] * MIB
    metrics = dict(backend='coalesced', direct=o['direct'], cuda=cuda, batch_bytes=batch_bytes,
                   owned_limit_bytes=owned_limit, threads=o['threads'], batches=0, tensors=0,
                   read_bytes=0, read_calls=0, uploads=0, reader_wait_seconds=0.,
                   staging_wait_seconds=0., upload_enqueue_seconds=0., upload_wait_seconds=0.,
                   placement_wait_seconds=0., consumer_seconds=0., complete=False,
                   start_wall_epoch=time.time())
    import glm_loader_guard
    scope = glm_loader_guard.active()
    memory = MemoryGuard(o, metrics, scope)
    files = list(progress(files) if progress else files)
    plans, stamps = inspect(files, keys_of, skip, batch_bytes, owned_limit, memory)
    if o['direct']:
        check_direct(stamps)  # unsupported FS fails before native placement
    context = glm_mtp_select.active()
    metrics.update(load_kind=context.kind if context else 'other', files=len(files),
                   planned_tensors=sum(len(b.tensors) for b in plans))
    started, budget = time.monotonic(), OwnedStorageBudget(owned_limit, metrics, memory)

    def produce():
        reader = None
        try:
            reader = RangeReader(batch_bytes, o['threads'], cuda, o['direct'], metrics, memory)
            reader.stamps = stamps
            for batch in plans:
                if stamp(batch.file) != stamps[batch.file]:
                    raise ValueError('checkpoint changed before batch read')
                budget.reserve(batch.bytes)
                output = torch.empty(batch.bytes, dtype=torch.uint8, device='cuda' if cuda else 'cpu')
                budget.track(output)
                event = reader.fill(batch, output)
                metrics['batches'] += 1
                yield batch, event, output
                del output
        finally:
            if reader is not None:
                reader.close()

    iterator = prefetch(produce(), cuda)
    try:
        for batch, event, output in iterator:
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            for name, offset, info in batch.tensors:
                size = info['data_offsets'][1] - info['data_offsets'][0]
                tensor = output[offset:offset + size].view(fast._torch_dtype(info['dtype'])).reshape(info['shape'])
                if stats is not None:
                    stats.eager += 1; stats.bytes += size; stats.sample()
                metrics['tensors'] += 1
                tick = time.monotonic()
                yield name, tensor
                metrics['consumer_seconds'] += time.monotonic() - tick
                memory.check()
                del tensor
            if cuda:
                # Retired allocations on the upload stream cannot accumulate behind
                # native placement. The producer still reads/uploads the next batch.
                tick = time.monotonic(); torch.cuda.current_stream().synchronize()
                metrics['placement_wait_seconds'] += time.monotonic() - tick
            del output
        for path, before in stamps.items():
            if stamp(path) != before:
                raise ValueError('checkpoint changed during coalesced loading')
        metrics['complete'] = True
    finally:
        try:
            iterator.close()
        finally:
            metrics.update(seconds=time.monotonic() - started, peak_owned_bytes=budget.peak,
                           end_wall_epoch=time.time())
            if stats is not None:
                stats.coalesced = metrics
            log(metrics)
