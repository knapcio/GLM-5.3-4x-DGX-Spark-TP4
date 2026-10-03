# SPDX-License-Identifier: Apache-2.0
"""Display-carveout lease + one-range CUDA VMM pool (clean-room).

Written from the public CUDA driver API (virtual memory management: cuMemAddressReserve,
cuMemCreate, cuMemImportFromShareableHandle, cuMemMap, cuMemSetAccess) and INTERFACE.md (the
observed dispramd socket interface). No kindling code is used or imported; dispramd runs
unmodified in its own container and only lends an fd.

Pool layout: one virtual range [head | tail]. The head is ordinary device memory (cuMemCreate,
same host cost as cudaMalloc); the tail is the leased carveout slice (no MemAvailable debit).
Callers carve independent tensors (own storage each) out of the range, so kernels see plain
contiguous tensors and nothing in vLLM learns about the seam.

Rules this module enforces:
* every rank takes the same decision from collective results only (never a local fallback);
* the carveout is written and zeroed by SM kernels only (no cudaMemcpy/cudaMemset on it);
* mappings are never torn down while the process lives (no free path after success).
"""
import ctypes
import json
import math
import os
import socket
import struct
import sys

SOCK = '/run/dispram/dispram.sock'
KEY = 'kindlingai_1'
SLICE = 2 << 20
MIB = 1 << 20
ALIGN = 512  # torch caching-allocator alignment; every carved tensor starts on it


class CarveoutError(RuntimeError):
    pass


def log(event, **kw):
    sys.stderr.write('glm-dispram-kv: ' + json.dumps(dict(event=event, **kw), sort_keys=True) + '\n')
    sys.stderr.flush()


def round_down(x, g):
    return x // g * g


def round_up(x, g):
    return -(-x // g) * g


# ---------------------------------------------------------------- dispramd client (INTERFACE.md)
class DispramdClient:
    """One SOCK_SEQPACKET connection; a lease lives as long as this socket."""

    def __init__(self, path=SOCK, timeout=10.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.settimeout(timeout)
        self.sock.connect(path)

    def _call(self, body):
        self.sock.send(json.dumps(dict(key=KEY, **body)).encode())
        data, anc, _flags, _ = self.sock.recvmsg(65536, socket.CMSG_SPACE(4 * struct.calcsize('i')))
        fds = []
        for level, kind, payload in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                n = len(payload) // struct.calcsize('i')
                fds += list(struct.unpack(f'{n}i', payload[:n * struct.calcsize('i')]))
        if not data:
            for fd in fds:
                os.close(fd)
            raise CarveoutError('dispramd closed the connection')
        return json.loads(data), fds

    def info(self):
        rep, fds = self._call(dict(op='info'))
        for fd in fds:
            os.close(fd)
        for k in ('base', 'size', 'free', 'largest'):
            if not isinstance(rep.get(k), int):
                raise CarveoutError(f'info reply without integer {k}: {rep}')
        return rep

    def alloc(self, size):
        rep, fds = self._call(dict(op='alloc', size=int(size)))
        if rep.get('ok') is not True or len(fds) != 1 or rep.get('size') != size:
            for fd in fds:
                os.close(fd)
            raise CarveoutError(f'alloc {size} refused: reply {rep}, {len(fds)} fd')
        return fds[0]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------- CUDA driver VMM via ctypes
class _Loc(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [('compressionType', ctypes.c_ubyte), ('gpuDirectRDMACapable', ctypes.c_ubyte),
                ('usage', ctypes.c_ushort), ('reserved', ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):  # CUmemAllocationProp
    _fields_ = [('type', ctypes.c_int), ('requestedHandleTypes', ctypes.c_int), ('location', _Loc),
                ('win32HandleMetaData', ctypes.c_void_p), ('allocFlags', _AllocFlags)]


class _Access(ctypes.Structure):  # CUmemAccessDesc
    _fields_ = [('location', _Loc), ('flags', ctypes.c_int)]


CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
U64 = ctypes.c_ulonglong


class _CAISpan:
    """__cuda_array_interface__ v3 view of [ptr, ptr+n); keeps the pool alive."""

    def __init__(self, ptr, n, keepalive):
        self._keepalive = keepalive
        self.__cuda_array_interface__ = dict(shape=(n,), typestr='|i1', data=(ptr, False),
                                             version=3, strides=None, stream=None)


class CudaVMM:
    """Thin checked wrapper over the driver calls the pool needs."""

    def __init__(self, lib=None):
        self.cu = lib or ctypes.CDLL('libcuda.so.1')

    def _ck(self, name, *args):
        rc = getattr(self.cu, name)(*args)
        if rc != 0:
            s = ctypes.c_char_p()
            self.cu.cuGetErrorName(rc, ctypes.byref(s))
            raise CarveoutError(f'{name} -> {rc} {s.value.decode() if s.value else ""}')

    def _prop(self, dev):
        p = _Prop()
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED
        p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE
        p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = dev
        return p

    def granularity(self, dev):
        g = ctypes.c_size_t()
        p = self._prop(dev)
        self._ck('cuMemGetAllocationGranularity', ctypes.byref(g), ctypes.byref(p),
                 CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        return g.value

    def reserve(self, size, align):
        va = U64()
        self._ck('cuMemAddressReserve', ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(align),
                 U64(0), U64(0))
        return va.value

    def free_va(self, va, size):
        self._ck('cuMemAddressFree', U64(va), ctypes.c_size_t(size))

    def create(self, size, dev):
        h = U64()
        p = self._prop(dev)
        self._ck('cuMemCreate', ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(p), U64(0))
        return h.value

    def import_fd(self, fd):
        h = U64()
        self._ck('cuMemImportFromShareableHandle', ctypes.byref(h), ctypes.c_void_p(fd),
                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
        return h.value

    def map(self, va, size, handle):
        self._ck('cuMemMap', U64(va), ctypes.c_size_t(size), ctypes.c_size_t(0), U64(handle), U64(0))

    def unmap(self, va, size):
        self._ck('cuMemUnmap', U64(va), ctypes.c_size_t(size))

    def release(self, handle):
        self._ck('cuMemRelease', U64(handle))

    def set_access(self, va, size, dev):
        d = _Access()
        d.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        d.location.id = dev
        d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._ck('cuMemSetAccess', U64(va), ctypes.c_size_t(size), ctypes.byref(d), ctypes.c_size_t(1))

    def tensor(self, va, n, dev, keepalive):
        import torch
        return torch.as_tensor(_CAISpan(va, n, keepalive), device=torch.device('cuda', dev))

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def release_cached(self):
        """fix7: hand torch's cached-but-free blocks back to the driver before the head is created.

        The stock path allocates KV with torch.zeros, which reuses the ~2 GiB that profile_run left cached in
        the caching allocator. The hook allocates the head with cuMemCreate instead, so without this the
        cached blocks stay reserved next to a new 2 GiB head (TP4 B-build 2026-10-02: rank-0 pre-capture
        MemAvailable 8.79 GiB vs 10.3-10.8 on the stock path; torch reserved-but-free 2.15 GB)."""
        import torch
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        return before, torch.cuda.memory_reserved()


# ---------------------------------------------------------------- lease and pool
class Pool:
    """[head | tail] in one VA range. Never unmapped while the process lives."""

    def __init__(self, lease, va, total, head):
        self.lease, self.va, self.total, self.head = lease, va, total, head
        self.spans = []

    def tensor(self, offset, n):
        if offset < 0 or offset + n > self.total or offset % ALIGN:
            raise CarveoutError(f'span [{offset}, {offset + n}) outside pool or misaligned')
        self.spans.append((self.va + offset, n))
        return self.lease.drv.tensor(self.va + offset, n, self.lease.dev, self)

    def overlaps(self, ptr, n):
        return ptr < self.va + self.total and ptr + n > self.va


QUARANTINE = []  # leases whose teardown is uncertain: socket and handle held until process exit


class Lease:
    """A leased slice. State is tracked step by step so teardown only undoes what really happened."""

    def __init__(self, drv, client, dev, size, env=None, rank=0):
        self.drv, self.client, self.dev, self.size = drv, client, dev, size
        self.env, self.rank = (os.environ if env is None else env), rank
        self.handle = self.probe_va = None
        self.mapped = False
        self.pool = None
        self.quarantined = False

    def _step(self, where):
        _inject(self.env, self.rank, where)

    def take(self, fd):
        try:
            self.handle = self.drv.import_fd(fd)
        finally:
            os.close(fd)  # the imported handle holds the memory; the socket holds the lease

    def map_probe(self, gran):
        self.probe_va = self.drv.reserve(self.size, gran)
        self._step('map')
        self.drv.map(self.probe_va, self.size, self.handle)
        self.mapped = True
        self._step('access')  # partial state: mapped, no access yet
        self.drv.set_access(self.probe_va, self.size, self.dev)

    def verify(self, chunk=16 * MIB):
        """Address-dependent pattern and its complement over every 8-byte word; SM kernels only."""
        import torch
        t = self.drv.tensor(self.probe_va, self.size, self.dev, self).view(torch.int64)
        nw = chunk // 8
        ar = torch.arange(nw, dtype=torch.int64, device=t.device)
        ref = torch.empty_like(ar)
        k = 0x9E3779B97F4A7C15 - (1 << 64)
        bad = 0
        for key in (0x5A5A5A5A5A5A5A5A, ~0x5A5A5A5A5A5A5A5A):
            for i in range(0, t.numel(), nw):
                c = t[i:i + nw]
                torch.add(ar[:c.numel()], i, out=ref[:c.numel()]).mul_(k).bitwise_xor_(key)
                torch.bitwise_xor(ref[:c.numel()], 0, out=c)  # SM write into the carveout
            for i in range(0, t.numel(), nw):
                c = t[i:i + nw]
                torch.add(ar[:c.numel()], i, out=ref[:c.numel()]).mul_(k).bitwise_xor_(key)
                bad += int(not torch.equal(c, ref[:c.numel()]))
        del t, ar, ref
        self.drv.synchronize()
        if bad:
            raise CarveoutError(f'carveout pattern verify failed in {bad} chunk(s)')

    def build_pool(self, total, head, gran):
        """Remap the verified slice behind a fresh head allocation in one range.

        Any failure here quarantines the lease (the slice may be mapped in either range) and raises;
        the worker then exits and only process exit returns the slice.
        """
        if self.pool is not None:
            raise CarveoutError('KV pool already built; re-initialisation is not supported')
        if total != head + self.size or head % gran or total % gran or head < 0:
            raise CarveoutError(f'bad pool geometry total={total} head={head} tail={self.size} gran={gran}')
        try:
            self.drv.synchronize()
            self.drv.unmap(self.probe_va, self.size)
            self.mapped = False
            self.drv.free_va(self.probe_va, self.size)
            self.probe_va = None
            va = self.drv.reserve(total, gran)
            if head:
                h = self.drv.create(head, self.dev)
                self.drv.map(va, head, h)
            self.drv.map(va + head, self.size, self.handle)
            self.drv.set_access(va, total, self.dev)
        except Exception as e:
            self._quarantine('build_pool', e)
            raise CarveoutError(f'pool build failed, lease quarantined until process exit: {e!r}') from e
        self.pool = Pool(self, va, total, head)
        return self.pool

    def _quarantine(self, where, err):
        self.quarantined = True
        QUARANTINE.append(self)  # keeps the socket object (and so the lease) alive until exit
        log('quarantine', where=where, error=repr(err), rank=self.rank,
            note='slice may still be mapped; socket held open; process must exit before any re-lend')

    def abandon(self):
        """Pre-agreement failure. Returns True only after a verified clean teardown.

        Any error (CUDA calls can report earlier asynchronous faults) quarantines the lease instead:
        the socket stays open and the handle is not released, so dispramd cannot re-lend bytes this
        context may still reach. Process exit is then the only release.
        """
        try:
            self._step('sync')
            self.drv.synchronize()
            if self.mapped:
                self._step('unmap')
                self.drv.unmap(self.probe_va, self.size)
                self.mapped = False
            if self.probe_va is not None:
                self.drv.free_va(self.probe_va, self.size)
                self.probe_va = None
            if self.handle is not None:
                self._step('release')
                self.drv.release(self.handle)
                self.handle = None
            self.drv.synchronize()
        except Exception as e:
            self._quarantine('abandon', e)
            return False
        self.client.close()
        return True


def parse_mode(env):
    m = env.get('GLM_DISPRAM_KV', '0')
    m = {'1': 'require'}.get(m, m)
    if m not in ('0', 'require', 'auto'):
        raise CarveoutError('GLM_DISPRAM_KV must be 0, 1/require or auto')
    return m


def _inject(env, rank, where):
    # test only, outside the bootfast env contract: "<stage>@<rank>[,<stage>@<rank>...]"
    # stages: probe alloc map access verify (claim) | sync unmap release (teardown)
    if f'{where}@{rank}' in env.get('DISPRAM_A2_INJECT', '').split(','):
        raise CarveoutError(f'injected failure at {where} on rank {rank}')


def negotiate(coll, dev, env=os.environ, drv=None, connect=DispramdClient):
    """Collective claim. coll: .rank, .min(int) -> int over all ranks (CPU group).

    Returns a verified Lease (same size on every rank) or None (auto mode, all ranks).
    Raises CarveoutError on every rank in require mode when any rank fails.
    Every rank executes the same collectives (MIN size, MIN success, MIN clean teardown) regardless of
    local failures; an uncertain teardown on any rank aborts every rank, also in auto mode.
    """
    mode = parse_mode(env)
    cap = int(env.get('GLM_DISPRAM_KV_MAX_MIB', '0')) * MIB
    floor = int(env.get('GLM_DISPRAM_KV_MIN_MIB', '1024')) * MIB
    path = env.get('GLM_DISPRAM_SOCK', SOCK)
    drv = drv or CudaVMM()
    client = lease = None
    usable, why, gran = 0, '', SLICE
    try:
        _inject(env, coll.rank, 'probe')
        gran = math.lcm(SLICE, drv.granularity(dev))
        client = connect(path)
        info = client.info()
        if info['free'] != info['size']:
            raise CarveoutError(f'another lease is outstanding (free {info["free"]} of {info["size"]})')
        usable = round_down(min(info['largest'], cap) if cap else info['largest'], gran)
    except Exception as e:
        usable, why = 0, repr(e)
    agreed = coll.min(usable)
    ok = 0
    if agreed >= floor:
        try:
            _inject(env, coll.rank, 'alloc')
            if agreed % gran:
                raise CarveoutError(f'agreed {agreed} B is not a multiple of this rank\'s granularity {gran}')
            lease = Lease(drv, client, dev, agreed, env, coll.rank)
            lease.take(client.alloc(agreed))
            lease.map_probe(gran)
            lease.verify()
            _inject(env, coll.rank, 'verify')
            ok = 1
        except Exception as e:
            why = why or repr(e)
    all_ok = coll.min(ok) if agreed >= floor else 0
    facts = dict(rank=coll.rank, mode=mode, local_usable=usable, agreed=agreed, floor=floor, gran=gran,
                 ok=ok, all_ok=all_ok, why=why)
    if all_ok:
        lease.gran = gran
        log('lease', **facts)
        return lease
    clean = 1
    if lease is not None:
        clean = int(lease.abandon())
    elif client is not None:
        client.close()
    # third collective: an uncertain teardown anywhere forbids every fallback, also in auto mode
    all_clean = coll.min(clean) if agreed >= floor else clean
    facts.update(clean=clean, all_clean=all_clean)
    log('no-lease', **facts)
    if not all_clean:
        raise CarveoutError(f'carveout teardown uncertain on some rank; aborting (no fallback): {facts}')
    if mode == 'require':
        raise CarveoutError(f'carveout KV required but not available on every rank: {facts}')
    return None


def guard_arm(pool):
    """Tell the optional LD_PRELOAD copy guard (guard/dispram_copy_guard.c) which range to watch."""
    try:
        f = ctypes.CDLL(None).dispram_guard_set
    except (AttributeError, OSError):
        return False
    f.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    f(pool.va, pool.va + pool.total)
    log('copy-guard-range', lo=hex(pool.va), hi=hex(pool.va + pool.total))
    return True


def plan_regions(sizes, lease_size, gran, base_bytes):
    """Offsets of each region (512-aligned) and the [head | tail] split.

    The head may exceed the pinned host budget by less than one granule only: the extra KV
    bytes must come from the carveout, never silently from ordinary memory.
    """
    offs, cur = [], 0
    for s in sizes:
        cur = round_up(cur, ALIGN)
        offs.append(cur)
        cur += s
    total = max(round_up(cur, gran), lease_size)
    head = total - lease_size
    if head > base_bytes + gran:
        raise CarveoutError(f'head {head} B exceeds the pinned budget {base_bytes} B: '
                            'KV would spill into ordinary memory')
    return offs, total, head
