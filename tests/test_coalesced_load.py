"""Mac CPU byte/lifetime/failure tests; CUDA qualification belongs to the one-boot plan."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/overlay'))
import torch
from safetensors.torch import safe_open, save_file
import glm_coalesced_load as C
import glm_fast_load as F
import glm_mtp_select as M
import glm_loader_guard as G

torch.set_num_threads(2)


def keys(path):
    with safe_open(path, framework='pt') as f:
        return list(f.keys())


def stock(files, skip=None):
    for path in files:
        with safe_open(path, framework='pt') as f:
            for name in f.keys():
                if not (skip and skip(name)):
                    yield name, f.get_tensor(name)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def manifest(iterator):
    return [(n, str(t.dtype), tuple(t.shape), digest(t)) for n, t in iterator]


def failing(error):
    raise error
    yield  # failure during iteration, before any handoff


def fixtures(root, count=3142):
    gen = torch.Generator().manual_seed(482)
    dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int8,
              torch.bool, torch.float8_e4m3fn, torch.float64]
    shards = [{}, {}, {}]
    for i in range(count):
        dtype = dtypes[i % len(dtypes)]
        # Mixed dtype order differs from physical order; page gaps and scalars.
        shape = () if i % 47 == 0 else (0, 4) if i % 53 == 0 else (13, 17)
        value = torch.randint(0, 100, shape, generator=gen, dtype=torch.int32).to(dtype)
        if i == 17:
            value = torch.randint(0, 255, (2 * C.MIB + 123,), generator=gen, dtype=torch.uint8)
        if dtype == torch.float32 and value.numel():
            value.reshape(-1)[0] = float('nan')  # compare payload bytes, not floating equal()
        name = f'model.layers.{78 if i % 19 == 0 else i % 78:02d}.w{i:05d}'
        shards[i % 3][name] = value
    paths, weight_map = [], {}
    for i, shard in enumerate(shards):
        path = root / f'model-{i + 1:05d}.safetensors'
        save_file(shard, str(path))
        paths.append(str(path))
        weight_map.update({n: path.name for n in shard})
    (root / M.INDEX).write_text(json.dumps({'weight_map': weight_map}))
    return paths


class Coalesced(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, GLM_LOADER='coalesced', GLM_COALESCED_BATCH_MB='1',
                              GLM_COALESCED_OWNED_MB='8', GLM_COALESCED_THREADS='4',
                              GLM_COALESCED_DIRECT='0')
        self.env.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = fixtures(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup(); self.env.stop()

    def iterator(self, **kw):
        return C.coalesced_safetensors_iterator(self.paths, keys,
                                                lambda p: safe_open(p, framework='pt'), **kw)

    def test_3142_tensors_per_rank_hash_equal(self):
        ref = manifest(stock(self.paths))
        self.assertEqual(len(ref), 3142)
        for rank in range(4):
            with patch.dict(os.environ, RANK=str(rank)):
                st = F._Stats()
                got = manifest(self.iterator(stats=st))
            self.assertEqual(ref, got)
            self.assertTrue(st.coalesced['complete'])
            self.assertLessEqual(st.coalesced['peak_owned_bytes'], 8 * C.MIB)
            self.assertEqual(st.coalesced['staging_bytes'], 2 * (C.MIB + C.ALIGN))
            print(f'hash-EQUAL synthetic rank{rank}: {len(got)} tensors')

    def test_ep_and_mtp_selection_before_read(self):
        skip = lambda n: n.endswith('3')
        plans, _ = C.inspect(self.paths, keys, skip, C.MIB, 8 * C.MIB)
        self.assertTrue(all(not skip(n) for b in plans for n, _, _ in b.tensors))
        self.assertEqual(manifest(stock(self.paths, skip)), manifest(self.iterator(skip=skip)))
        pred = lambda n: n.startswith('model.layers.78.')
        for kind, mode in [('target', '1'), ('draft', '1'), ('draft', 'audit')]:
            ctx = M.LoadContext(kind, mode, pred)
            iterator = M.iterate(ctx, self.paths, None, lambda fs, filt:
                C.coalesced_safetensors_iterator(fs, keys, lambda p: safe_open(p, framework='pt'), skip=filt))
            effective = lambda n: pred(n) if kind == 'target' else not pred(n) if mode == '1' else False
            self.assertEqual(manifest(stock(self.paths, effective)), manifest(iterator))
            self.assertTrue(ctx.complete)

    def test_alias_budget_counts_backing_storage(self):
        budget = C.OwnedStorageBudget(1024)
        output = torch.empty(1024, dtype=torch.uint8)
        budget.reserve(1024); budget.track(output)
        alias = output[:1]; del output; gc.collect()
        with self.assertRaisesRegex(MemoryError, 'retained'):
            budget.reserve(1)
        del alias; gc.collect(); budget.reserve(1024)
        self.assertEqual(budget.live, 0)

    def test_memory_floor_and_budget_limits(self):
        with patch.object(F, '_mem_available_kb', return_value=(C.FLOOR + C.MIB) // 1024):
            C.guard(C.MIB)
            with self.assertRaises(MemoryError): C.guard(C.MIB + 1)
        with patch.object(F, '_mem_available_kb', return_value=-1), patch.object(C.sys, 'platform', 'linux'):
            with self.assertRaises(MemoryError): C.guard()
        for key, val in [('BATCH_MB', '129'), ('OWNED_MB', '4097'), ('THREADS', '33'),
                         ('DIRECT', 'yes'), ('OWNED_MB', '2')]:
            with self.subTest(key=key), patch.dict(os.environ, {'GLM_COALESCED_' + key: val}):
                with self.assertRaises(ValueError): C.options()
        with patch.object(F, '_mem_available_kb', return_value=2 * 1048576):
            with self.assertRaises(MemoryError): next(self.iterator())

    def test_observed_unified_commit_95_gib_passes(self):
        # First 8 GiB reproduces 18 -> 10 GiB. Later commits reclaim other
        # resident pages: MemAvailable cannot literally become 18 - 95 GiB.
        # Destination writes total 95 GiB; loader storage never exceeds 640 MiB.
        for rank in range(4):
            scope = G.LoadScope()
            with patch.object(F, '_mem_available_kb', return_value=18 * 1048576):
                memory = C.MemoryGuard(C.options(), {}, scope)
            memory.staging, memory.owned = 256 * C.MIB, 384 * C.MIB
            for commit in range(96):
                scope.destination.committed = commit << 30
                available = max(6 << 30, (18 - commit) << 30) - memory.staging - memory.owned
                with patch.object(F, '_mem_available_kb', return_value=available // 1024):
                    memory.check(128 * C.MIB)
            self.assertEqual(memory.metrics['destination_committed_bytes'], 95 << 30)
            self.assertLess(memory.metrics['mem_available_low_bytes'], 10 << 30)
            self.assertEqual(memory.metrics['transient_live_bytes'], 640 * C.MIB)

    def test_external_pressure_and_emergency_abort(self):
        scope = G.LoadScope()
        with patch.object(F, '_mem_available_kb', return_value=18 * 1048576):
            memory = C.MemoryGuard(C.options(), scope=scope)
        memory.staging, memory.owned = 256 * C.MIB, 384 * C.MIB
        scope.destination.committed = 8 << 30
        # A separate process consumes 2 GiB beyond destination + tracked loader.
        available = (18 - 8 - 2) * 1048576 - 640 * 1024
        with patch.object(F, '_mem_available_kb', return_value=available):
            with self.assertRaisesRegex(MemoryError, 'external'): memory.check()
        scope.destination.committed = 95 << 30
        with patch.object(F, '_mem_available_kb', return_value=2 * 1048576):
            with self.assertRaisesRegex(MemoryError, 'emergency'): memory.check()
        with patch.dict(os.environ, GLM_COALESCED_EMERGENCY_MB='4096'), \
             patch.object(F, '_mem_available_kb', return_value=3 * 1048576):
            with self.assertRaisesRegex(MemoryError, 'emergency'): C.MemoryGuard(C.options())
        for key in ('EMERGENCY_MB', 'MARGIN_MB'):
            with patch.dict(os.environ, {'GLM_COALESCED_' + key: '0'}):
                with self.assertRaises(ValueError): C.options()

    def test_relative_transient_budget_is_bounded(self):
        with patch.object(F, '_mem_available_kb', return_value=18 * 1048576):
            memory = C.MemoryGuard(C.options())
            memory.owned = 17 << 30
            with self.assertRaisesRegex(MemoryError, 'transient'): memory.check(1)

    def test_native_destination_credit_uses_slices_and_unique_bytes(self):
        model = torch.nn.Module()
        model.register_buffer('weight', torch.empty(4, 16, dtype=torch.uint8))
        ledger = G.DestinationWrites(model)
        with ledger:
            # 64-byte checkpoint, native TP rank places only its 16-byte slice.
            model.weight[0].copy_(torch.ones(16, dtype=torch.uint8))
            model.weight[0, :8].copy_(torch.zeros(8, dtype=torch.uint8))
            torch.empty(64, dtype=torch.uint8).copy_(torch.zeros(64, dtype=torch.uint8))
        self.assertEqual(ledger.committed, 16)
        with ledger:
            model.weight[1:].copy_(torch.ones(3, 16, dtype=torch.uint8))
        self.assertEqual(ledger.committed, 64)

    def test_memory_pressure_mid_load_drains_producer(self):
        before = {t.ident for t in threading.enumerate()}
        scope = G.LoadScope()
        with patch.object(G, 'active', return_value=scope), \
             patch.object(F, '_mem_available_kb', return_value=18 * 1048576) as available:
            iterator = self.iterator()
            next(iterator)
            available.return_value = 14 * 1048576
            with self.assertRaisesRegex(MemoryError, 'external'): next(iterator)
        self.assertFalse([t for t in threading.enumerate()
                          if t.ident not in before and t.name.startswith('glm-coalesced')])

    def test_strided_destination_credit_deduplicates_aliases(self):
        model = torch.nn.Module()
        model.register_buffer('weight', torch.zeros(4, 8, dtype=torch.uint8))
        ledger = G.DestinationWrites(model)
        with ledger:
            model.weight[:, ::2].copy_(torch.ones(4, 4, dtype=torch.uint8))
            model.weight[:, 0].copy_(torch.ones(4, dtype=torch.uint8))
        self.assertEqual(ledger.committed, 16)
        with ledger:
            model.weight.t().copy_(torch.ones(8, 4, dtype=torch.uint8))
        self.assertEqual(ledger.committed, 32)

    def test_owning_cache_high_water_is_not_external_pressure(self):
        with patch.object(F, '_mem_available_kb', return_value=18 * 1048576):
            memory = C.MemoryGuard(C.options())
        memory.staging, memory.owned, memory.owned_peak = 256 * C.MIB, 128 * C.MIB, 2 << 30
        available = 18 * 1048576 - (256 + 2048) * 1024
        with patch.object(F, '_mem_available_kb', return_value=available): memory.check()

    def test_automatic_fast_fallback_only_before_handoff(self):
        from tqdm import tqdm
        wu = types.SimpleNamespace(_natural_sort_key=lambda p: p, tqdm=tqdm,
            enable_tqdm=lambda _: False, _BAR_FORMAT='{desc}', should_skip_weight=lambda n, ids: n.endswith('3'))
        def orig(*args, **kw):
            raise AssertionError('must use normal fast iterator')
        wrapper = F._make_wrapper(orig, wu)
        ref = manifest(stock(self.paths, lambda n: n.endswith('3')))
        for failure in (MemoryError('pressure'), OSError('read'), ValueError('direct I/O')):
            with patch.object(C, 'coalesced_safetensors_iterator', side_effect=lambda *a, **k: failing(failure)):
                self.assertEqual(manifest(wrapper(self.paths, False, 'lazy')), ref)
        def partial(*args, **kw):
            yield 'first', torch.zeros(1)
            raise MemoryError('after placement')
        with patch.object(C, 'coalesced_safetensors_iterator', partial), \
             patch.object(F, 'fast_safetensors_iterator', side_effect=AssertionError('unsafe fallback')):
            iterator = wrapper(self.paths, False, 'lazy')
            self.assertEqual(next(iterator)[0], 'first')
            with self.assertRaisesRegex(MemoryError, 'after placement'): next(iterator)

    def test_short_reads_and_eof(self):
        path = Path(self.tmp.name) / 'bytes'
        path.write_bytes(b'abcdefgh')
        fd = os.open(path, os.O_RDONLY)
        try:
            def short(fd_, views, off):
                data = os.pread(fd_, min(3, len(views[0])), off)
                views[0][:len(data)] = data
                return len(data)
            with patch.object(C.os, 'preadv', short, create=True):
                buf = bytearray(12)
                self.assertEqual(C.read_aligned(fd, memoryview(buf), 0, 8, False), (8, 3))
                self.assertEqual(buf[:8], b'abcdefgh')
                with self.assertRaises(EOFError): C.read_aligned(fd, memoryview(buf), 0, 12, False)
                with patch.object(C.os, 'open', return_value=os.dup(fd)) as reopen:
                    C.read_aligned(fd, memoryview(buf), 0, 8, True)
                    reopen.assert_called_once_with(f'/proc/self/fd/{fd}', os.O_RDONLY)
        finally:
            os.close(fd)

    def test_direct_range_alignment_and_eof_padding(self):
        def read(fd, views, off):
            view = views[0]
            self.assertEqual(off % C.ALIGN, 0)
            self.assertEqual(len(view) % C.ALIGN, 0)
            self.assertEqual(torch.frombuffer(view, dtype=torch.uint8).data_ptr() % C.ALIGN, 0)
            data = os.pread(fd, len(view), off)
            view[:len(data)] = data
            return len(data)
        with patch.dict(os.environ, GLM_COALESCED_DIRECT='1'), patch.object(C.os, 'O_DIRECT', 0, create=True), \
             patch.object(C.os, 'preadv', read, create=True):
            self.assertEqual(manifest(stock(self.paths)), manifest(self.iterator()))

    def test_pending_tile_event_is_fenced_before_reuse(self):
        metrics = dict(staging_wait_seconds=0., read_bytes=0, read_calls=0, reader_wait_seconds=0.,
                       upload_enqueue_seconds=0., uploads=0)
        reader = C.RangeReader(C.MIB, 4, False, False, metrics)
        plans, _ = C.inspect(self.paths, keys, None, C.MIB, 8 * C.MIB)
        class Event:
            def __init__(self): self.waited = False
            def synchronize(self): self.waited = True
        event = Event(); reader.slots[0][2] = event
        try:
            batch = next(b for b in plans if b.bytes)
            reader.fill(batch, torch.empty(batch.bytes, dtype=torch.uint8))
            self.assertTrue(event.waited)
        finally: reader.close()

    def test_pinned_allocator_bin_is_charged_in_full(self):
        # Exercise allocation policy on CPU without requiring a CUDA runtime.
        real_empty, requests = torch.empty, []
        def aligned_empty(n, **kw):
            self.assertTrue(kw.pop('pin_memory'))
            requests.append(n)
            owner = real_empty(n + C.ALIGN, **kw)
            start = (-owner.data_ptr()) % C.ALIGN
            return owner[start:start + n]
        metrics = dict(upload_wait_seconds=0.)
        stream = types.SimpleNamespace(synchronize=lambda: None)
        with patch.object(torch, 'empty', aligned_empty), patch.object(torch.cuda, 'current_stream', return_value=stream):
            reader = C.RangeReader(3 * C.MIB, 4, True, False, metrics)
        try:
            self.assertEqual(requests, [4 * C.MIB, 4 * C.MIB])
            self.assertEqual(metrics['staging_bytes'], 8 * C.MIB)
        finally: reader.close()

    def test_consumer_retention_fails_instead_of_deadlocking(self):
        path = Path(self.tmp.name) / 'retained.safetensors'
        save_file({f'w{i}': torch.zeros(512 * 1024, dtype=torch.uint8) for i in range(12)}, str(path))
        self.paths = [str(path)]
        with patch.dict(os.environ, GLM_COALESCED_OWNED_MB='3'):
            held = []
            with self.assertRaisesRegex(MemoryError, 'retained'):
                for _, value in self.iterator(): held.append(value)

    def test_read_error_and_early_close_drain_threads(self):
        before = {t.ident for t in threading.enumerate()}
        iterator = self.iterator()
        for _ in range(10): next(iterator)
        iterator.close()
        self.assertFalse([t for t in threading.enumerate() if t.ident not in before and t.name.startswith('glm-coalesced')])
        with patch.object(C, 'read_aligned', side_effect=OSError('injected read failure')):
            with self.assertRaisesRegex(OSError, 'injected'): next(self.iterator())
        self.assertFalse([t for t in threading.enumerate() if t.ident not in before and t.name.startswith('glm-coalesced')])

    def test_source_changed_or_truncated_aborts(self):
        iterator = self.iterator(); next(iterator)
        with open(self.paths[-1], 'ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'): manifest(iterator)
        path = Path(self.tmp.name) / 'bad.safetensors'; path.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            C.inspect([str(path)], keys, None, C.MIB, 8 * C.MIB)

    def test_wrapper_strategy_selection_and_stock_order(self):
        calls = []
        def orig(files, use_tqdm, strategy=None, **kw):
            calls.append(strategy); yield from stock(sorted(files))
        from tqdm import tqdm
        wu = types.SimpleNamespace(_natural_sort_key=lambda p: p, tqdm=tqdm,
            enable_tqdm=lambda _: False, _BAR_FORMAT='{desc}', should_skip_weight=lambda n, ids: False)
        wrapper = F._make_wrapper(orig, wu)
        ref = manifest(stock(self.paths))
        self.assertEqual(ref, manifest(wrapper(list(reversed(self.paths)), False, 'lazy', local_expert_ids=None)))
        self.assertEqual(calls, [])
        self.assertEqual(ref, manifest(wrapper(self.paths, False, 'eager')))
        self.assertEqual(calls, ['eager'])
        with patch.dict(os.environ, GLM_LOADER=''):
            self.assertEqual(F.loader_mode({'GLM_FAST_LOAD': '1'}), 'fast')
            self.assertEqual(F.loader_mode({}), 'off')
        self.assertEqual(F.loader_mode({'GLM_FAST_LOAD': '0', 'GLM_LOADER': 'coalesced'}), 'coalesced')
        with self.assertRaises(ValueError): F.loader_mode({'GLM_LOADER': 'oops'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
