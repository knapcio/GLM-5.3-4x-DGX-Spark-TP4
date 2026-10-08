# SPDX-License-Identifier: Apache-2.0
"""Execute whole pinned native schedule, with inert KV and no TP/CUDA mocks in policy."""
import ast
import copy
import itertools
import json
import os
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_adaptive_chunk as T
from test_adaptive_chunk import A, W, request, scheduler, Pause, Status
import glm_decode_fair as F


class FairSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.PinnedSchedule.setUpClass()
        cls.source = T.PinnedSchedule.source
        cls.namespace = T.PinnedSchedule.namespace
        cls.adaptive = staticmethod(T.PinnedSchedule.adaptive)

    def fair(self, s, chunk=1024):
        ns = dict(self.namespace, _glm_budget_factory=lambda s: F.FairStepBudget(s, A.THRESHOLD, chunk))
        exec('from __future__ import annotations\n' + A.transform(self.source, True), ns)
        return ns['schedule'](s)

    def test_both_running_orders_all_chunks_and_native_draft_rows(self):
        for chunk in F.CHUNKS:
            for reverse in (False, True):
                rows = [request('d', 100, 100, decode=True), request('p', 100000, 4094)]
                s = scheduler(running=rows[::-1] if reverse else rows)
                out = self.fair(s, chunk)
                self.assertEqual(out.num_scheduled_tokens['d'], 3)
                self.assertLessEqual(out.num_scheduled_tokens['p'], chunk)
                self.assertEqual(out.scheduled_spec_decode_tokens, {'d': [1, 2]})
                self.assertLessEqual(out.total_num_scheduled_tokens + 2 * len(out.num_scheduled_tokens), 4096)

    def test_native_mtp_zero_draft_slots_and_variable_kstop_widths(self):
        path = Path(os.environ['GLM_IMAGE_SRC'])/'vllm/config/speculative.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpeculativeConfig')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'max_num_new_slots_for_drafting')
        fn.decorator_list = []
        ns = {}
        exec(ast.unparse(fn), ns)
        spec = NS(num_speculative_tokens=3, parallel_drafting=False,
                  use_dflash=lambda: False, uses_draft_model=lambda: False)
        slots = ns['max_num_new_slots_for_drafting'](spec)
        self.assertEqual(slots, 0)
        for width in (2, 3, 4):
            for reverse in (False, True):
                d = request('d', 100, 100, decode=True)
                d.num_tokens_with_spec = 100 + width
                d.spec_token_ids = list(range(width-1))
                rows = [request('p', 100000, 4096), d]
                s = scheduler(running=rows[::-1] if reverse else rows, draft_slots=slots)
                out = self.fair(s, 4096)
                self.assertEqual(out.num_scheduled_tokens, {'p': 4096-width, 'd': width})
                self.assertEqual(out.scheduled_spec_decode_tokens['d'], list(range(width-1)))

    def test_aggregate_budget_many_prefills_with_decoder_last(self):
        for order in itertools.permutations(('a', 'b', 'c', 'd')):
            rows = {x: request(x, 100000, 4094) for x in ('a', 'b', 'c')}
            rows['d'] = request('d', 20000, 20000, decode=True)
            s = scheduler(running=[rows[x] for x in order], waiting=[request('queued', 1000)])
            out = self.fair(s)
            self.assertEqual(out.num_scheduled_tokens['d'], 3)
            self.assertEqual(sum(n for rid, n in out.num_scheduled_tokens.items() if rid != 'd'), 1024)
            self.assertEqual(len(s.waiting), 1)
            self.assertEqual([r.request_id for r in s.running], list(order))

    def test_budget_consumed_waiting_prefill_breaks_without_zero_allocation(self):
        s = scheduler(running=[request('d', 100, 100, decode=True)],
                      waiting=[request('a', 100000), request('queued', 100000)])
        out = self.fair(s)
        self.assertEqual(out.num_scheduled_tokens, {'d': 3, 'a': 1024})
        self.assertEqual([r.request_id for r in s.waiting], ['queued'])
        self.assertTrue(all(n > 0 for _, n, _ in s.allocations))
        self.assertTrue(s.allocations[-1][2]['full_sequence_must_fit'])

    def test_multi_step_decode_progress_and_restore_after_decoder_finishes(self):
        s = scheduler(running=[request('p', 100000, 4094), request('d', 100, 100, decode=True)])
        for _ in range(12):
            out = self.fair(s)
            self.assertEqual(out.num_scheduled_tokens, {'p': 1024, 'd': 3})
            d = s.running[1]
            d.num_tokens = d.num_computed_tokens + 1
            d.num_tokens_with_spec = d.num_tokens + 2
            d.spec_token_ids = [1, 2]
        s.running.pop()
        self.assertEqual(self.fair(s).num_scheduled_tokens, {'p': 4094})

    def test_zero_switch_c1_boot_pause_decode_only_exact_adaptive_equivalence(self):
        for chunk in (0, *F.CHUNKS):
            cases = [scheduler(waiting=[request('p', 100000)]),
                     scheduler(running=[request('d', 100, 100, decode=True)]),
                     scheduler(running=[request('a', 100, 100, decode=True), request('b', 30000, 30000, decode=True)]),
                     scheduler(running=[request('d', 100, 100, decode=True)], waiting=[request('p', 100000)], cap=512)]
            if chunk == 0:
                cases.append(scheduler(running=[request('p', 100000, 4094), request('d', 100, 100, decode=True)]))
            for s in cases:
                other = copy.deepcopy(s)
                # Fixture methods close over their scheduler; recreate independently.
                def remake(x):
                    return scheduler(running=copy.deepcopy(x.running), waiting=copy.deepcopy(x.waiting),
                                     cap=x.max_num_scheduled_tokens)
                left, right = remake(s), remake(other)
                self.assertEqual(vars(self.adaptive(left)), vars(self.fair(right, chunk)))
                self.assertEqual(left.allocations, right.allocations)
        s = scheduler(running=[request('p', 100000, 4094), request('d', 100, 100, decode=True)])
        s._pause_state = Pause.PAUSED_ALL
        self.assertEqual(self.fair(s).total_num_scheduled_tokens, 0)

    def test_async_eligibility_finished_decoder_does_not_cap_prefill(self):
        for mode in ('ineligible', 'complete', 'no_work'):
            d = request('d', 100, 100, decode=True)
            if mode == 'ineligible': d.next_decode_eligible_step = 3
            if mode == 'complete':
                d.num_computed_tokens = 613
                d.num_output_placeholders = 1
            if mode == 'no_work': d.num_tokens_with_spec = 100
            s = scheduler(running=[request('p', 100000, 4094), d])
            self.assertEqual(self.fair(s).num_scheduled_tokens['p'], 4094)

    def test_apc_long_delta_and_threshold_compose(self):
        for native_threshold in (0, 512, 2048):
            s = scheduler(running=[request('d', 100, 100, decode=True)],
                          waiting=[request('p', 64448, hits=61376)])
            s.scheduler_config.long_prefill_token_threshold = native_threshold
            out = self.fair(s)
            self.assertEqual(out.num_scheduled_tokens, {'d': 3, 'p': min(native_threshold or 1024, 1024)})
            self.assertEqual(s.running[-1].num_computed_tokens, 61376 + out.num_scheduled_tokens['p'])

    def test_large_reservation_preserves_decodes_instead_of_splitting_for_prefill(self):
        d = request('d', 100, 100, decode=True)
        d.num_tokens_with_spec = 3600
        d.spec_token_ids = []
        s = scheduler(running=[request('p', 100000, 4094), d])
        out = self.fair(s)
        self.assertEqual(out.num_scheduled_tokens['d'], 3500)
        self.assertEqual(out.num_scheduled_tokens['p'], 592)

    def test_rank_invariance_complete_outputs_four_state_replicas(self):
        for chunk in F.CHUNKS:
            outs = []
            for rank in range(4):
                s = scheduler(running=[request('p', 100000, 4094), request('d', 100, 100, decode=True)])
                # Forbidden rank-local values must not affect the schedule.
                s.rank_local_mem_available = rank * 100
                outs.append(vars(self.fair(s, chunk)))
            self.assertTrue(all(out == outs[0] for out in outs))
        root = Path(os.environ['GLM_IMAGE_SRC']) / 'vllm/v1'
        self.assertIn('args=(scheduler_output,)', (root / 'executor/multiproc_executor.py').read_text())
        self.assertIn('scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())',
                      (root / 'engine/core.py').read_text())

    def test_native_kv_refusal_and_config_no_mutation(self):
        s = scheduler(running=[request('d', 100, 100, decode=True)], waiting=[request('p', 100000)])
        allocate = s.kv_cache_manager.allocate_slots
        s.kv_cache_manager.allocate_slots = lambda r, *a, **kw: None if r.request_id == 'p' else allocate(r, *a, **kw)
        before = copy.deepcopy(vars(s.scheduler_config))
        out = self.fair(s)
        self.assertEqual(out.num_scheduled_tokens, {'d': 3})
        self.assertEqual(len(s.waiting), 1)
        self.assertEqual(vars(s.scheduler_config), before)
        self.assertEqual(s._w2_prefill_capacity, 4096)
        with self.assertRaisesRegex(RuntimeError, 'drift'): A.transform(self.source + '\n', True)

    def test_priority_preemption_refunds_prefill_and_keeps_native_victim_rule(self):
        a, b, d = request('a', 100000, 4094), request('b', 100000, 4094), request('d', 100, 100, decode=True)
        # Force the native priority allocator to preempt an already scheduled
        # prefill, refund it, and retry the decoder. No policy-side refunds.
        a.priority, b.priority, d.priority = 10, 0, 0
        for r in (a, b, d): r.arrival_time = 0
        s = scheduler(running=[a, b, d], waiting=[request('queued', 1000)])
        s.policy = 'priority'
        def preempt(r, *_args, **_kwargs):
            r.num_computed_tokens = 0
            r.status = Status.PREEMPTED
            s.waiting.prepend_request(r)
        s._preempt_request = preempt
        s.requires_kv_delivery = False
        allocate = s.kv_cache_manager.allocate_slots
        attempts = []
        def alloc(r, *args, **kwargs):
            if r is d:
                attempts.append(r.request_id)
                if len(attempts) == 1: return None
            return allocate(r, *args, **kwargs)
        s.kv_cache_manager.allocate_slots = alloc
        ns = dict(self.namespace, SchedulingPolicy=NS(PRIORITY='priority'),
                  _glm_budget_factory=lambda s: F.FairStepBudget(s, A.THRESHOLD, 1024))
        exec('from __future__ import annotations\n' + A.transform(self.source, True), ns)
        out = ns['schedule'](s)
        self.assertEqual(out.num_scheduled_tokens, {'d': 3})
        self.assertEqual(attempts, ['d', 'd'])
        self.assertEqual(s.waiting[0].request_id, 'a')

    def test_real_installer_runtime_on_off_on_with_running_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixed, fair = Path(tmp)/'prefill.json', Path(tmp)/'fair.json'
            fixed.write_text('{"schema":1,"chunk":2048,"sequence":1}')
            module = NS(__file__=str(T.PinnedSchedule.path),
                        Scheduler=type('Scheduler', (), {'schedule': T.PinnedSchedule.native}), **self.namespace)
            env = dict(GLM_W2_PREFILL_CONTROL=str(fixed), GLM_KV_FORMAT='fp4x',
                       GLM_PREFILL_CHUNK_ADAPTIVE='1', GLM_DECODE_FAIR='1', GLM_DECODE_FAIR_CONTROL=str(fair))
            with patch.dict(os.environ, env):
                W.install(module)
                s = scheduler()
                W.apply(s, json.loads(fixed.read_text()))
                for seq, chunk in enumerate((0, 1024, 2048, 0, 512, 1024)):
                    s.running = [request('p', 100000, 4094), request('d', 100, 100, decode=True)]
                    (Path(tmp)/'new.json').write_text(json.dumps(dict(schema=1, chunk=chunk, sequence=seq)))
                    (Path(tmp)/'new.json').replace(fair)
                    out = module.Scheduler.schedule(s)
                    self.assertEqual(out.num_scheduled_tokens['p'], chunk or 4094)
                    self.assertEqual(out.num_scheduled_tokens.get('d'), 3 if chunk else None)
                fair.write_text('{"schema":1,"chunk":0,"sequence":1}')
                with self.assertRaisesRegex(RuntimeError, 'stale'): module.Scheduler.schedule(s)


class Controls(unittest.TestCase):
    def test_operator_writer_atomic_sequence_and_mode(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('fair_writer', root/'scripts/decode_fair_control.py')
        writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'control.json'
            self.assertEqual(writer.write(path, 0, 0), dict(schema=1, chunk=0, sequence=0))
            self.assertEqual(F.control_chunk(NS(), str(path), 1024), 0)
            writer.write(path, 1024, 1)
            self.assertEqual(F.control_chunk(NS(), str(path), 512), 1024)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError): writer.write(path, 512, 1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_default_off_and_invalid_boot_values(self):
        self.assertEqual(F.settings({}), (False, 4096, ''))
        valid = dict(GLM_KV_FORMAT='fp4x', GLM_W2_PREFILL_CONTROL='/cache/prefill.json', GLM_DECODE_FAIR='1')
        self.assertEqual(F.settings(valid), (True, 4096, ''))
        for changes in (dict(GLM_DECODE_FAIR='yes'), dict(GLM_DECODE_FAIR_CHUNK='999'),
                        dict(GLM_DECODE_FAIR_CONTROL='relative'), dict(GLM_PREFILL_CHUNK_ADAPTIVE='0'),
                        dict(GLM_W2_PREFILL_CONTROL='')):
            with self.assertRaises(ValueError): F.settings(dict(valid, **changes))

    def test_fail_closed_controls_equal_sequence_replay_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fair.json'
            s = NS()
            with self.assertRaises(FileNotFoundError): F.control_chunk(s, str(path), 1024)
            for value in ([], {}, dict(schema=True, chunk=1024, sequence=1),
                          dict(schema=1, chunk=True, sequence=1), dict(schema=1, chunk=1024, sequence=True),
                          dict(schema=1, chunk=12, sequence=1), dict(schema=1, chunk=1024, sequence=-1)):
                path.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError): F.control_chunk(s, str(path), 1024)
            path.write_text('{')
            with self.assertRaises(json.JSONDecodeError): F.control_chunk(s, str(path), 1024)
            path.write_text('{"schema":1,"chunk":1024,"sequence":1}')
            self.assertEqual(F.control_chunk(s, str(path), 512), 1024)
            self.assertEqual(F.control_chunk(s, str(path), 512), 1024)
            path.write_text('{"schema":1,"chunk":512,"sequence":1}')
            with self.assertRaisesRegex(RuntimeError, 'stale'): F.control_chunk(s, str(path), 512)


if __name__ == '__main__': unittest.main()
