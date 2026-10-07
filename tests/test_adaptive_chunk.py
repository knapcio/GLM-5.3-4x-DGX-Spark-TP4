# SPDX-License-Identifier: Apache-2.0
"""Execute the pinned native schedule on CPU with inert KV/worker boundaries."""
from contextlib import nullcontext
from enum import Enum
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_adaptive_chunk as A
import glm_prefill_switch as W


class Request(NS):
    __hash__ = object.__hash__


class Status(Enum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    WAITING_FOR_REMOTE_KVS = 4


class Pause(Enum):
    UNPAUSED = 1
    PAUSED_ALL = 2


class Queue(list):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.pop(0)

    def prepend_requests(self, requests):
        self[:0] = requests

    def prepend_request(self, request):
        self.insert(0, request)


def request(name, prompt, computed=0, decode=False, hits=0):
    return Request(request_id=name, num_prompt_tokens=prompt,
        num_computed_tokens=computed, num_tokens=prompt + int(decode),
        num_tokens_with_spec=prompt + (3 if decode else 0),
        num_output_placeholders=0, next_decode_eligible_step=0,
        max_tokens=512, is_prefill_chunk=not decode, has_encoder_inputs=False,
        spec_token_ids=[1, 2] if decode else [], lora_request=None,
        num_stale_output_tokens=0, drop_stale_output=False, prefill_stats=None,
        num_preemptions=0, status=Status.RUNNING if computed else Status.WAITING,
        hits=hits, _all_token_ids=[])


def scheduler(running=(), waiting=(), cap=2048, draft_slots=2):
    blocks = NS(get_block_ids=lambda: ([1],))
    allocations = []

    def allocate(req, tokens, **kwargs):
        recorded = dict(kwargs)
        if 'new_computed_blocks' in recorded:
            recorded['new_computed_blocks'] = recorded['new_computed_blocks'].get_block_ids()
        allocations.append((req.request_id, tokens, recorded))
        return blocks

    s = NS(current_step=0, max_num_scheduled_tokens=cap,
        scheduler_config=NS(max_num_batched_tokens=cap, long_prefill_token_threshold=0,
                            enable_chunked_prefill=True),
        _w2_prefill_capacity=4096,
        vllm_config=NS(speculative_config=NS(max_num_new_slots_for_drafting=draft_slots)),
        _pause_state=Pause.UNPAUSED, max_num_encoder_input_tokens=0,
        prefill_capacity_bound=False, running=list(running), waiting=Queue(waiting),
        skipped_waiting=Queue(), policy='fcfs', connector=None, ec_connector=None,
        lora_config=None, max_model_len=200000, num_sampled_tokens_per_step=1,
        need_mamba_block_aligned_split=False, num_lookahead_tokens=3,
        num_waiting_for_streaming_input=0, max_num_running_reqs=4,
        num_spec_tokens=2, dynamic_sd_lookup=None, is_encoder_decoder=False,
        scheduler_reserve_full_isl=True, log_stats=False,
        _inflight_prefills=set(), kv_cache_config=NS(kv_cache_groups=[1]),
        use_v2_model_runner=True, reset_preempted_req_ids=set(), finished_req_ids=set(),
        defer_block_free=False, sched_step_seq=0, observability_config=None,
        encoder_cache_manager=NS(get_freed_mm_hashes=lambda: [], get_manager_metadata=lambda: None),
        _get_new_block_ids_to_zero=lambda: None, allocations=allocations)
    s.kv_cache_manager = NS(new_step_starts=lambda: None, allocate_slots=allocate,
        get_blocks=lambda _: blocks, record_prefix_cache_stats=lambda *_: None,
        get_num_common_prefix_blocks=lambda _: [0], take_kv_cache_block_copies=lambda: ([], []),
        empty_kv_cache_blocks=blocks)
    s._select_waiting_queue_for_scheduling = lambda: s.waiting
    s._is_blocked_waiting_status = lambda _: False
    s._get_local_prefix_cache_hit = lambda r: (blocks, r.hits, 0, False)
    s._make_cached_request_data = lambda *args: [r.request_id for r in args[0]]

    def update(output):
        for r in s.running:
            r.num_computed_tokens += output.num_scheduled_tokens.get(r.request_id, 0)
    s._update_after_schedule = update
    return s


class PinnedSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ['GLM_IMAGE_SRC'])
        cls.path = root / 'vllm/v1/core/sched/scheduler.py'
        cls.source = cls.path.read_text()
        cls.namespace = dict(time=time, PauseState=Pause, RequestStatus=Status,
            create_request_queue=lambda _: Queue(), record_function_or_nullcontext=lambda _: nullcontext(),
            NewRequestData=NS(from_request=lambda r, *_: (r.request_id, r.num_computed_tokens)),
            SchedulerOutput=lambda **kwargs: NS(**kwargs),
            _glm_budget_factory=lambda s: A.StepBudget(s, A.THRESHOLD))
        ns = dict(cls.namespace)
        exec('from __future__ import annotations\n' + A.schedule_source(cls.source), ns)
        cls.native = staticmethod(ns['schedule'])
        ns = dict(cls.namespace)
        exec('from __future__ import annotations\n' + A.transform(cls.source), ns)
        cls.adaptive = staticmethod(ns['schedule'])

    def test_cold_4k_and_20k_split_each_step(self):
        for tokens, splits in ((4096, [2046, 2046, 4]),
                               (20480, [4094, 4094, 4094, 4094, 4094, 10])):
            s = scheduler(waiting=[request('r', tokens)])
            actual = [self.adaptive(s).num_scheduled_tokens['r'] for _ in splits]
            self.assertEqual(actual, splits)
            self.assertEqual(s.running[0].num_computed_tokens, tokens)

    def test_apc_hit_61k_with_3k_delta(self):
        s = scheduler(waiting=[request('r', 61376 + 3072, hits=61376)])
        self.assertEqual(self.adaptive(s).num_scheduled_tokens, {'r': 3072})
        self.assertEqual(s.running[0].num_computed_tokens, 64448)
        for hits, expected in ((1024, {'a': 2046}),
                               (61376, {'a': 3072, 'b': 1020})):
            s = scheduler(waiting=[request('a', hits + 3072, hits=hits),
                                   request('b', 20480)])
            self.assertEqual(self.adaptive(s).num_scheduled_tokens, expected)

    def test_two_concurrent_requests_share_one_budget(self):
        for prompts, expected in (((20480, 20480), {'a': 4094}),
                                  ((1024, 20480), {'a': 1024, 'b': 3068}),
                                  ((1024, 1024), {'a': 1024, 'b': 1020})):
            s = scheduler(waiting=[request('a', prompts[0]), request('b', prompts[1])])
            out = self.adaptive(s)
            self.assertEqual(out.num_scheduled_tokens, expected)
            self.assertLessEqual(out.total_num_scheduled_tokens + 2 * len(expected),
                                 4096 if max(prompts) >= A.THRESHOLD else 2048)

    def test_exact_threshold_uses_total_context_for_running_and_waiting(self):
        for total, budget in ((16383, 2046), (16384, 4094), (16385, 4094)):
            for remaining in (total, 3072, 1):
                for waiting in (False, True):
                    r = request('r', total, computed=0 if waiting else total-remaining,
                                hits=total-remaining if waiting else 0)
                    s = scheduler(waiting=[r]) if waiting else scheduler(running=[r])
                    self.assertEqual(self.adaptive(s).num_scheduled_tokens,
                                     {'r': min(remaining, budget)})

    def test_small_decode_and_mixed_are_native_equivalent(self):
        cases = [([], [4096]), ([], [1024, 4096]), ([100], []),
                 ([100, 30000], []), ([100], [4096]), ([100, 30000], [8192]),
                 ([], [16383])]
        for decodes, prefills in cases:
            def make():
                return scheduler(running=[request(f'd{i}', p, computed=p, decode=True)
                                          for i, p in enumerate(decodes)],
                    waiting=[request(f'p{i}', p) for i, p in enumerate(prefills)])
            native, adaptive = make(), make()
            for _ in range(3):
                left, right = self.native(native), self.adaptive(adaptive)
                self.assertEqual(vars(left), vars(right), (decodes, prefills))
                self.assertEqual(native.allocations, adaptive.allocations)

    def test_mixed_long_keeps_decode_reservations_and_graph_shape(self):
        s = scheduler(running=[request('d', 30000, computed=30000, decode=True)],
                      waiting=[request('p', 20480)])
        out = self.adaptive(s)
        self.assertEqual(out.num_scheduled_tokens, {'d': 3, 'p': 4089})
        self.assertEqual(out.scheduled_spec_decode_tokens, {'d': [1, 2]})

    def test_custom_threshold_and_fixed4096_control(self):
        ns = dict(self.namespace, _glm_budget_factory=lambda s: A.StepBudget(s, 32768))
        exec('from __future__ import annotations\n' + A.transform(self.source), ns)
        for tokens, expected in ((20480, 2046), (32768, 4094)):
            self.assertEqual(ns['schedule'](scheduler(waiting=[request('p', tokens)])).num_scheduled_tokens,
                             {'p': expected})
        # Adaptive normalizes a serving control of 4096 to the small budget
        # until eligible; env=0 keeps the native fixed-cap scheduler instead.
        for tokens, expected in ((4096, 2046), (20480, 4094)):
            self.assertEqual(self.adaptive(scheduler(waiting=[request('p', tokens)], cap=4096)).num_scheduled_tokens,
                             {'p': expected})
        self.assertEqual(self.native(scheduler(waiting=[request('p', 4096)], cap=4096)).num_scheduled_tokens,
                         {'p': 4094})

    def test_exhaustive_small_budget_equivalence(self):
        for slots in (0, 2, 3):
            for first in (1, 64, 1024, 2046, 2048, 4096, 16383):
                for second in (1, 1024, 4096):
                    def make():
                        return scheduler(waiting=[request('a', first), request('b', second)], draft_slots=slots)
                    left, right = make(), make()
                    self.assertEqual(vars(self.native(left)), vars(self.adaptive(right)))
                    self.assertEqual(left.allocations, right.allocations)

    def test_waiting_kv_admission_can_still_refuse_long_request(self):
        s = scheduler(waiting=[request('p', 102400)])
        s.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None
        self.assertEqual(self.adaptive(s).num_scheduled_tokens, {})
        self.assertEqual(len(s.waiting), 1)
        self.assertEqual(s.running, [])

    def test_boot_and_pause_do_not_expand(self):
        s = scheduler(waiting=[request('p', 102400)], cap=512)
        self.assertEqual(vars(self.native(scheduler(waiting=[request('p', 102400)], cap=512))),
                         vars(self.adaptive(s)))
        s = scheduler(waiting=[request('p', 102400)])
        s._pause_state = Pause.PAUSED_ALL
        self.assertEqual(self.adaptive(s).total_num_scheduled_tokens, 0)

    def test_capacity_config_and_admission_unchanged(self):
        for tokens in (4096, 20480, 102400):
            s = scheduler(waiting=[request('p', tokens)])
            before = (s._w2_prefill_capacity, vars(s.scheduler_config).copy(), s.num_lookahead_tokens,
                      s.max_model_len, s.max_num_running_reqs, s.scheduler_reserve_full_isl)
            self.adaptive(s)
            self.assertEqual(before, (s._w2_prefill_capacity, vars(s.scheduler_config), s.num_lookahead_tokens,
                             s.max_model_len, s.max_num_running_reqs, s.scheduler_reserve_full_isl))
            self.assertTrue(s.allocations[0][2]['full_sequence_must_fit'])
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            s._w2_prefill_capacity = 2048
            self.adaptive(s)

    def test_pin_drift_and_switch_composition(self):
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(), W.PIN)
        with self.assertRaisesRegex(RuntimeError, 'drift'):
            A.transform(self.source + '\n')
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / 'control.json'
            control.write_text('{"schema": 1, "chunk": 2048, "sequence": 1}')
            module = NS(__file__=str(self.path), Scheduler=type('Scheduler', (), {'schedule': self.native}),
                        **self.namespace)
            with patch.dict(os.environ, GLM_W2_PREFILL_CONTROL=str(control), GLM_KV_FORMAT='fp4x',
                            GLM_PREFILL_CHUNK_ADAPTIVE='1'):
                W.install(module)
                s = scheduler(waiting=[request('p', 20480)])
                self.assertEqual(module.Scheduler.schedule(s).total_num_scheduled_tokens, 4094)
            module = NS(__file__=str(self.path), Scheduler=type('Scheduler', (), {'schedule': self.native}),
                        **self.namespace)
            with patch.dict(os.environ, GLM_W2_PREFILL_CONTROL=str(control), GLM_KV_FORMAT='fp4x',
                            GLM_PREFILL_CHUNK_ADAPTIVE='0'):
                W.install(module)
                self.assertEqual(module.Scheduler.schedule(scheduler(waiting=[request('p', 20480)])).total_num_scheduled_tokens,
                                 2046)

    def test_broadcast_path_is_centralized(self):
        root = Path(os.environ['GLM_IMAGE_SRC']) / 'vllm/v1'
        core = (root / 'engine/core.py').read_text()
        executor = (root / 'executor/multiproc_executor.py').read_text()
        self.assertIn('scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())', core)
        self.assertIn('self.model_executor.execute_model(scheduler_output, non_block=True)', core)
        self.assertIn('args=(scheduler_output,)', executor)
        self.assertIn('self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))', executor)


class Settings(unittest.TestCase):
    def test_defaults_and_threshold(self):
        self.assertEqual(A.settings({}), (False, 16384))
        self.assertEqual(A.settings({'GLM_KV_FORMAT': 'fp4x'}), (True, 16384))
        self.assertEqual(A.settings({'GLM_KV_FORMAT': 'fp4x', 'GLM_PREFILL_CHUNK_ADAPTIVE': '0'}), (False, 16384))
        self.assertEqual(A.settings({'GLM_PREFILL_CHUNK_ADAPTIVE': '1', 'GLM_PREFILL_CHUNK_THRESHOLD': '32768'}), (True, 32768))
        for env in ({'GLM_PREFILL_CHUNK_ADAPTIVE': 'yes'}, {'GLM_PREFILL_CHUNK_THRESHOLD': '0'}):
            with self.assertRaises(ValueError):
                A.settings(env)


if __name__ == '__main__':
    unittest.main()
