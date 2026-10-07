# SPDX-License-Identifier: Apache-2.0
"""Destination commit accounting and a bounded all-rank load failure channel."""
from datetime import timedelta
import functools
import itertools
import os
import signal
import sys
import threading
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_local = threading.local()
_sequence = 0


def log(message):
    sys.stderr.write('glm-loader-guard: ' + message + '\n')
    sys.stderr.flush()


def active():
    return getattr(_local, 'scope', None)


class DestinationWrites(TorchDispatchMode):
    """Credit unique destination bytes, after native TP/EP slicing.

    Rewrites/aliases never earn a second credit. Unknown storage and excessive
    strided fragmentation earn zero credit rather than guessing a TP ratio.
    """
    def __init__(self, model=None):
        super().__init__()
        self.ranges = {}
        self.committed = 0
        self.lock = threading.Lock()
        self.storage_refs = []
        if model is not None:
            from torch.multiprocessing.reductions import StorageWeakRef
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.device.type != 'meta':
                    storage = tensor.untyped_storage()
                    if storage._cdata not in self.ranges:
                        self.ranges[storage._cdata] = []
                        self.storage_refs.append(StorageWeakRef(storage))

    def credit(self, tensor):
        key = tensor.untyped_storage()._cdata
        if key not in self.ranges or not tensor.numel():
            return
        lo = tensor.storage_offset() * tensor.element_size()
        block = tensor.element_size()
        outer = []
        for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
            if size == 1:
                continue
            stride *= tensor.element_size()
            if stride == block and not outer:
                block *= size
            else:
                outer.append((stride, size))
        fragments = 1
        for _, size in outer:
            fragments *= size
        if fragments > 200_000:
            return  # conservative credit for unusual native layouts
        writes = [(lo + sum(index * stride for index, (stride, _) in zip(indices, outer)), block)
                  for indices in itertools.product(*(range(size) for _, size in outer))]
        writes = [(a, a + size) for a, size in writes]
        with self.lock:
            ranges = self.ranges[key]
            old = sum(b - a for a, b in ranges)
            merged = []
            for a, b in sorted(ranges + writes):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            self.ranges[key] = merged
            self.committed += sum(b - a for a, b in merged) - old

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            # Register the native write before enqueue, so a concurrent producer
            # cannot mistake an in-progress destination commit for external loss.
            # A failed write aborts the load; its credit is never reused.
            self.credit(args[0])
        return func(*args, **(kwargs or {}))


class LoadTeam:
    """One decision before placement; independent abort polling thereafter.

    No per-tensor collective: EP ranks may have different inventories. Store
    RPCs have a 2 s timeout. A peer stuck in native/CUDA/NCCL code receives
    SIGTERM after 30 s, then SIGKILL after 5 s, so a failed rank cannot strand it.
    Healthy loads rendezvous at the end, keeping the monitor alive for peers.
    """
    def __init__(self, store=None, rank=0, world=1, timeout=120, grace=30, terminate=None):
        self.store, self.rank, self.world = store, rank, world
        self.timeout, self.grace = timeout, grace
        self.terminate = terminate or (lambda sig: os.kill(os.getpid(), sig))
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.thread = None
        self.error = None
        self.decisions = 0

    @classmethod
    def distributed(cls, sequence):
        import torch.distributed as dist
        if not dist.is_initialized():
            if int(os.environ.get('WORLD_SIZE', '1')) > 1:
                raise RuntimeError('coalesced loader needs initialized distributed failure channel')
            return cls()
        world = dist.get_world_size()
        if world == 1:
            return cls()
        # An independent client gives this channel a short RPC timeout without
        # changing vLLM's default store/group timeout or introducing NCCL calls.
        base = dist.distributed_c10d._get_default_store()
        while isinstance(base, dist.PrefixStore):
            base = base.underlying_store
        if not isinstance(base, dist.TCPStore):
            raise RuntimeError('coalesced multi-rank load requires a TCPStore failure channel')
        client = dist.TCPStore(base.host, base.port, None, False,
                               timeout=timedelta(seconds=2), wait_for_workers=False)
        store = dist.PrefixStore(f'glm-loader-{sequence}/', client)
        return cls(store, dist.get_rank(), world)

    def start(self):
        if self.store is None:
            return
        def watch():
            failed_at = None
            term_sent = False
            while not self.stop.wait(.1):
                try:
                    if self.store.check(['abort']):
                        self.failed.set()
                except Exception as exc:
                    self.error = exc
                    self.failed.set()
                if self.failed.is_set():
                    failed_at = failed_at or time.monotonic()
                    elapsed = time.monotonic() - failed_at
                    if elapsed >= self.grace + 5:
                        self.terminate(signal.SIGKILL)
                        return
                    if elapsed >= self.grace and not term_sent:
                        self.terminate(signal.SIGTERM)
                        term_sent = True
        self.thread = threading.Thread(target=watch, name='glm-loader-abort', daemon=True)
        self.thread.start()

    def check(self):
        if self.failed.is_set():
            raise RuntimeError('coalesced load aborted by a peer or failure-channel error') from self.error

    def abort(self):
        self.failed.set()
        log(f'abort rank={self.rank} world={self.world}')
        if self.store is not None:
            try:
                self.store.set('abort', '1')
            except Exception as exc:
                self.error = exc

    def wait(self, keys):
        deadline = time.monotonic() + self.timeout
        while not self.store.check(keys):
            self.check()
            if time.monotonic() >= deadline:
                self.abort()
                raise TimeoutError('coalesced load rank rendezvous timed out')
            time.sleep(.01)
        self.check()

    def ready(self, ok):
        if self.store is None:
            return ok
        index = self.decisions
        self.decisions += 1
        keys = [f'ready-{index}-{r}' for r in range(self.world)]
        self.store.set(keys[self.rank], '1' if ok else '0')
        self.wait(keys)
        return all(self.store.get(key) == b'1' for key in keys)

    def finish(self):
        if self.store is not None:
            keys = [f'done-{r}' for r in range(self.world)]
            self.store.set(keys[self.rank], '1')
            self.wait(keys)
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)


class LoadScope:
    def __init__(self, model=None, team=None):
        self.destination = DestinationWrites(model)
        self.team = team or LoadTeam()
        self.placed = False
        self.fallback = False


def wrap_load_weights(orig):
    @functools.wraps(orig)
    def load_weights(self, model, model_config):
        import glm_fast_load as fast
        if fast.loader_mode() != 'coalesced':
            return orig(self, model, model_config)
        global _sequence
        if active() is not None:
            raise RuntimeError('nested coalesced model load')
        _sequence += 1
        scope = LoadScope(model, LoadTeam.distributed(_sequence))
        _local.scope = scope
        scope.team.start()
        log(f'initialized load={_sequence} rank={scope.team.rank} world={scope.team.world} '
            'rpc_timeout=2s rendezvous_timeout=120s abort_grace=30s kill_grace=5s')
        try:
            with scope.destination:
                result = orig(self, model, model_config)
            scope.team.finish()
            log(f'complete load={_sequence} rank={scope.team.rank} fallback={scope.fallback}')
            return result
        except BaseException:
            scope.team.abort()
            # Keep the daemon watchdog armed through stack unwinding. Peers
            # and a worker hung in native teardown must terminate within 35 s.
            raise
        finally:
            _local.scope = None
    load_weights._glm_loader_guard = True
    return load_weights
