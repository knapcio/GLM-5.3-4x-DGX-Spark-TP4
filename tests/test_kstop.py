# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the staged native MTP K-stop overlay (overlay/kstop). No GPU.

The runtime runs in its fleet control mode (local) on a one-rank Gloo group: the decision guard is a real
all-reduce and its flag is read back through the planner copy, as on the GPU. Source-pin and transform
checks need GLM_IMAGE_SRC (the extracted image sources); the pinned image passes its own.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
KSTOP = ROOT / 'overlay/kstop'
sys.path.insert(0, str(KSTOP))
import deadrow_ops  # noqa: E402,F401  (registers the CPU path of the remap op)
import glm_mtp_kstop as hook  # noqa: E402
import kstop_runtime as rt  # noqa: E402
from kstop_policy import lengths, sources, stop  # noqa: E402

SRC = Path(os.environ['GLM_IMAGE_SRC']) if os.environ.get('GLM_IMAGE_SRC') else None
CONTROL = KSTOP / 'control.json'


def setUpModule():
    if not torch.distributed.is_initialized():
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0' if sys.platform == 'darwin' else 'lo')
        torch.distributed.init_process_group('gloo', store=torch.distributed.HashStore(), rank=0, world_size=1)


class Group:
    rank_in_group = 0
    ranks = [0]
    device_communicator = None

    @property
    def cpu_group(self):
        return torch.distributed.group.WORLD


def runtime(n=4):
    r = rt.Runtime(NS(max_num_reqs=n, device='cpu'))
    r.mode, r.epoch, r.tau, r.synthetic = 'k3-stop', 0, .74, False
    return r


def state(r, m=64):
    rt.STATE = dict(runtime=r, rows=torch.arange(m), src=torch.arange(m), dead=torch.zeros(m, dtype=torch.bool),
                    identity=True)


class Base(unittest.TestCase):
    def setUp(self):
        for target, value in (('kstop_runtime.tp_group', lambda: Group()),):
            p = patch(target, value); p.start(); self.addCleanup(p.stop)
        e = patch.dict(os.environ, GLM_MTP_KSTOP_CONTROL=str(CONTROL)); e.start(); self.addCleanup(e.stop)
        os.environ.pop('GLM_MTP_KSTOP_CONTROL_MODE', None)
        os.environ.pop(rt.BROADCAST_TEST_FLAG, None)

    def tearDown(self):
        rt.STATE = None


class Policy(unittest.TestCase):
    def test_uniform_cumulative_rule(self):
        self.assertEqual(lengths([[.73, 1], [.74, 1], [.8, .8], [1, .74]]), [1, 3, 2, 3])
        self.assertEqual(lengths([[.9, .85]]), [3])
        for p in (float('nan'), float('inf'), -1, 1.1):
            with self.assertRaises(ValueError):
                lengths([[p, .9]])
        for tau in (.59, .86, float('nan')):
            with self.assertRaises(ValueError):
                lengths([[.9, .9]], tau)

    def test_incremental_decisions_equal_the_rule(self):
        rng = random.Random(20261002)
        for _ in range(2000):
            p = [rng.random(), rng.random()]
            lens, cum = stop([3], [1.], [p[0]], 1, .74)
            if lens[0] > 1:
                lens, cum = stop(lens, cum, [p[1]], 2, .74)
            self.assertEqual(lens, lengths([p]))

    def test_dead_rows_reuse_the_request_root_and_keep_the_bonus_row(self):
        src, dead = sources([4, 4, 3], [1, 3, 2])
        self.assertEqual(src, [0, 1, 0, 0, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(dead, [False, False, True, True] + [False] * 7)
        with self.assertRaises(ValueError):
            sources([3], [3])                                   # no live bonus row


@unittest.skipUnless(SRC, 'needs GLM_IMAGE_SRC')
class SourcePins(unittest.TestCase):
    def test_pins_match_and_every_transform_compiles(self):
        self.assertEqual(len(hook.PINS), 8)
        for name, pin in hook.PINS.items():
            raw = (SRC / (name.replace('.', '/') + '.py')).read_text()
            self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), pin, name)
            cooked = hook.transform(name, raw)
            if name != hook.MTP:                                # the MTP speculator is patched after import
                self.assertNotEqual(cooked, raw, name)
            compile(cooked, name, 'exec')
            with self.assertRaisesRegex(RuntimeError, 'source drift'):
                hook.transform(name, raw + '\n')

    def test_k3_graph_widths_come_from_the_transformed_manager(self):
        raw = (SRC / (hook.CG.replace('.', '/') + '.py')).read_text()
        self.assertIn('decode_query_lens = ([2,3,4] if self.decode_query_len > 1 else [1])',
                      hook.transform(hook.CG, raw))


class Register(unittest.TestCase):
    ENV = dict(GLM_MTP_KSTOP='1', GLM_MTP_FIX='1', GLM_MTP_KSTOP_CONTROL=str(CONTROL))

    def test_default_off_installs_nothing(self):
        before = list(sys.meta_path)
        self.assertFalse(hook.register({}))
        self.assertFalse(hook.register({'GLM_MTP_KSTOP': '0'}))
        self.assertEqual(before, sys.meta_path)
        with self.assertRaises(ValueError):
            hook.register({'GLM_MTP_KSTOP': 'yes'})

    def test_refusals(self):
        with self.assertRaisesRegex(RuntimeError, 'loader fix/control'):
            hook.register(dict(self.ENV, GLM_MTP_FIX='0'))
        with self.assertRaisesRegex(RuntimeError, 'loader fix/control'):
            hook.register({k: v for k, v in self.ENV.items() if k != 'GLM_MTP_KSTOP_CONTROL'})
        for flag in ('GLM_MTP_PHASE_K', 'GLM_DSA_SWA_POOL', 'GLM_DSA_DRAFT_FOLD', 'GLM_KVLENS_EXACT'):
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                hook.register(dict(self.ENV, **{flag: '1'}))

    def test_registers_first_and_refuses_late_registration(self):
        before = list(sys.meta_path)
        try:
            self.assertTrue(hook.register(self.ENV))
            self.assertIsInstance(sys.meta_path[0], hook.Hook)
        finally:
            sys.meta_path[:] = before
        name = next(iter(hook.PINS))
        with patch.dict(sys.modules, {name: NS()}), self.assertRaisesRegex(RuntimeError, 'too late'):
            hook.register(self.ENV)
        self.assertEqual(before, sys.meta_path)

    def test_capture_layout_values(self):
        for bad in ('', '6', 'm6', 'yes'):
            with self.assertRaisesRegex(ValueError, 'CAPTURE_LAYOUT must be m12 or reuse'):
                hook.register(dict(self.ENV, GLM_MTP_KSTOP_CAPTURE_LAYOUT=bad))

    def test_capture_layout_is_bound_in_the_cold_agreement(self):
        for layout in ('m12','reuse'):
            payloads=[]
            with patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout,
                            GLM_MTP_KSTOP_CONTROL=str(CONTROL)), \
                    patch.object(rt.Runtime,'agree',lambda self,stage,payload,valid=True:payloads.append((stage,payload))):
                r=rt.Runtime(NS(max_num_reqs=4,device='cpu'));r.load_control()
            self.assertEqual(payloads,[('control',['k3-stop',0,.74,'uniform-batch','k2']+
                                        (['reuse-m6'] if layout=='reuse' else []))])

    def test_shipped_control(self):
        control = json.loads(CONTROL.read_text())
        self.assertEqual(rt.check_control(control), dict(schema=1, mode='k3-stop', epoch=0, tau=.74))
        for bad in (dict(control, tau=.9), dict(control, mode='k4'), dict(control, extra=1), dict(control, epoch=-1)):
            with self.assertRaises(ValueError):
                rt.check_control(bad)


class Runtime(Base):
    def test_broadcast_control_is_refused_without_the_offline_flag(self):
        with self.assertRaisesRegex(ValueError, 'prohibited on the fleet'):
            rt.control_mode({'GLM_MTP_KSTOP_CONTROL_MODE': 'broadcast'})
        with self.assertRaisesRegex(ValueError, 'prohibited on the fleet'):
            rt.Runtime(NS(max_num_reqs=1, device='cpu'), control='broadcast')
        self.assertEqual(rt.control_mode({}), 'local')

    def test_control_file_read_once_per_process(self):
        r = rt.Runtime(NS(max_num_reqs=4, device='cpu'))
        b = NS(req_ids=['a'], num_reqs=1)
        with patch.object(rt.Runtime, 'load_control', wraps=r.load_control) as load:
            self.assertEqual(r.begin(b), 3)
            r.begin(b)
            self.assertEqual(load.call_count, 1)
        self.assertEqual((r.mode, r.epoch, r.tau), ('k3-stop', 0, .74))

    def test_guard_rides_the_planner_copy_without_an_extra_sync(self):
        r = runtime()
        r.confidence[:] = torch.tensor([.9, .8, .7, .6])
        r.lengths, r.cumulative, r.ids = [3] * 4, [1.] * 4, list('abcd')
        r.arm(1, 4)
        self.assertIsNotNone(r.guard_pending)
        calls = []
        original = torch.Tensor.cpu

        def counted(t, *a, **kw):
            calls.append(t.numel())
            return original(t, *a, **kw)
        with patch.object(torch.Tensor, 'cpu', counted):
            out = r.packed_host(torch.tensor([2**30, 17, 3, 1]))
        self.assertEqual(calls, [4 + 4 + 1])                    # one copy: values, probabilities, guard flag
        self.assertEqual(out.tolist(), [2**30, 17, 3, 1])
        self.assertEqual((r.guards, r.fallback_syncs, r.guard_pending), (1, 0, None))
        self.assertTrue(r.after_metadata(1, 4))
        self.assertEqual(r.lengths, [3, 3, 1, 1])               # .9/.8 continue, .7/.6 stop after pass 1

    def test_guard_mismatch_refuses_at_the_common_check(self):
        r = runtime()
        r.lengths, r.cumulative, r.ids = [3], [1.], ['a']
        real = torch.distributed.all_reduce

        def disagree(t, *a, **kw):
            real(t, *a, **kw)
            t[0] += 1                                           # a peer with a different digest
        with patch.object(torch.distributed, 'all_reduce', disagree):
            r.arm(1, 1)
        with self.assertRaisesRegex(RuntimeError, 'collective-safe kstop refusal at decision-1'):
            r.packed_host(torch.tensor([5]))

    def test_c1_selects_the_physical_width_without_changing_the_schedule(self):
        r = runtime()
        runner = NS(speculator=NS(_kstop=r))
        r.proposals = {'r': 1}
        o = NS(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
               scheduled_spec_decode_tokens={'r': [-1] * 3}, num_scheduled_tokens={'r': 4}, total_num_scheduled_tokens=4)
        view = rt.select_inputs(runner, o)
        self.assertEqual(view.num_scheduled_tokens, {'r': 2})
        self.assertEqual(len(view.scheduled_spec_decode_tokens['r']), 1)
        self.assertEqual((o.total_num_scheduled_tokens, len(o.scheduled_spec_decode_tokens['r'])), (4, 3))
        self.assertFalse(runner._kstop_needs_remap)

    def test_c4_dead_rows_reuse_root_experts_and_keep_bonus_rows(self):
        r = runtime()
        state(r)
        r.proposals = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
        b = NS(req_ids=list(r.proposals), num_reqs=4, query_start_loc_np=[0, 4, 8, 12, 16],
               num_tokens=16, num_tokens_after_padding=16, is_prefilling_np=[False] * 4)
        rt.prepare(NS(device='cpu'), b, False)
        self.assertEqual(r.fallback_syncs, 1)                   # no planner copy in this unit: counted sync check
        w = torch.arange(16, dtype=torch.float32).view(-1, 1)
        ids = torch.arange(16).view(-1, 1)
        live = w.clone()
        rt.remap(w, ids)
        self.assertEqual(ids[:, 0].tolist(), [0, 1, 0, 0, 4, 5, 6, 4, 8, 9, 10, 11, 12, 13, 12, 12])
        self.assertEqual(w[[1, 6, 11, 13]].tolist(), live[[1, 6, 11, 13]].tolist())
        self.assertEqual(rt.mask_drafts(torch.arange(16), torch.arange(16)).tolist(),
                         [0, 1, -1, -1, 4, 5, 6, -1, 8, 9, 10, 11, 12, 13, -1, -1])

    def test_identity_batches_launch_no_remap(self):
        r = runtime(1)
        state(r)
        r.proposals = {'r': 1}
        b = NS(req_ids=['r'], num_reqs=1, query_start_loc_np=[0, 2], num_tokens=2, num_tokens_after_padding=2, is_prefilling_np=[False])
        with patch.object(torch.ops.glm_deadrow, 'remap_', side_effect=AssertionError('identity launch')):
            rt.prepare(NS(device='cpu'), b, False)
            rt.remap(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64))

    def test_synthetic_cycles_do_not_communicate(self):
        r = rt.Runtime(NS(max_num_reqs=4, device='cpu', _kstop_synthetic=True))
        with patch.object(rt, 'tp_group', side_effect=AssertionError('synthetic collective')):
            self.assertEqual(r.begin(NS(req_ids=['dummy'], num_reqs=1)), 3)
            r.arm(1, 1)
            self.assertIsNone(r.pending)
            self.assertTrue(r.after_metadata(1, 1))
            self.assertEqual(r.finish(torch.ones(1, 3)).tolist(), [[1, 1, 1]])

    def test_skipped_draft_slots_hold_valid_ids(self):
        r = runtime()
        r.ids, r.lengths = ['a', 'b'], [1, 3]
        out = r.finish(torch.full((2, 3), 7))
        self.assertEqual(out.tolist(), [[7, 0, 0], [7, 7, 7]])
        self.assertEqual(r.proposals, {'a': 1, 'b': 3})


class UniformBatch(Base):
    """GLM_MTP_KSTOP_UNIFORM_BATCH: batches of more than one request draft all three tokens."""

    def make(self, flag):
        with patch.dict(os.environ, {rt.UNIFORM_FLAG: flag}):
            return rt.Runtime(NS(max_num_reqs=4, device='cpu'))

    def cycle(self, r, ids, probs):
        """begin -> arm/planner/decision per pass (as the AR speculator loop) -> finish."""
        b = NS(req_ids=list(ids), num_reqs=len(ids))
        r.begin(b)
        n = len(ids)
        for step in (1, 2):
            if not r.can_advance(step, n):
                break
            r.confidence[:n] = torch.tensor([p[step - 1] for p in probs])
            r.arm(step, n)
            r.packed_host(torch.arange(2))                       # the draft planner's existing D2H
            if not r.after_metadata(step, n):
                break
        r.finish(torch.ones(n, 3, dtype=torch.int64))
        return dict(r.proposals)

    def test_flag_values(self):
        self.assertFalse(rt.uniform_batch({}))
        self.assertFalse(rt.uniform_batch({rt.UNIFORM_FLAG: '0'}))
        self.assertTrue(rt.uniform_batch({rt.UNIFORM_FLAG: '1'}))
        self.assertTrue(rt.uniform_batch({rt.UNIFORM_FLAG: 'k2'}))
        for bad in ('yes', '2', 'on'):
            with self.assertRaises(ValueError):
                rt.uniform_batch({rt.UNIFORM_FLAG: bad})
            with self.assertRaisesRegex(ValueError, 'UNIFORM_BATCH must be 0, 1 or k2'):
                hook.register(dict(Register.ENV, GLM_MTP_KSTOP_UNIFORM_BATCH=bad))

    def test_k2_count_policy_and_guarded_verify_rectangles(self):
        for n in (2, 3, 4):
            with self.subTest(n=n):
                r = self.make('k2')
                ids = list('abcd'[:n])
                # All confidence values are unusable: count-only batches never read them.
                self.assertEqual(self.cycle(r, ids, [(float('nan'), float('nan'))]*n),
                                 dict.fromkeys(ids, 2))
                self.assertFalse(r.can_advance(2, n))
                self.assertEqual((r.guards, r.fallback_syncs), (0, 0))
                runner = NS(speculator=NS(_kstop=r))
                o = NS(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
                       scheduled_spec_decode_tokens={rid: [-1]*3 for rid in ids},
                       num_scheduled_tokens=dict.fromkeys(ids, 4), total_num_scheduled_tokens=n*4)
                view = rt.select_inputs(runner, o)
                self.assertEqual(view.num_scheduled_tokens, dict.fromkeys(ids, 3))
                self.assertEqual(view.total_num_scheduled_tokens, n*3)
                self.assertTrue(all(len(t)==2 for t in view.scheduled_spec_decode_tokens.values()))
                self.assertFalse(runner._kstop_needs_remap)
                self.assertEqual(o.total_num_scheduled_tokens, n*4)  # scheduler accounting stays intact
                state(r)
                b = NS(req_ids=ids, num_reqs=n, query_start_loc_np=list(range(0, n*3+1, 3)),
                       num_tokens=n*3, num_tokens_after_padding=n*3, is_prefilling_np=[False]*n)
                rt.guard_inputs(NS(device='cpu'), b, False)
                r.packed_host(torch.arange(n))  # the actual target prepare guard
                rt.prepare(NS(device='cpu'), b, False)
                self.assertTrue(rt.STATE['identity'])
                self.assertIsNone(r.guard_pending)
                self.assertIsNone(r.bad)
                self.assertEqual(r.fallback_syncs, 0)

    def test_k2_keeps_c1_stop_and_binds_policy_in_cold_agreement(self):
        r = self.make('k2')
        payloads = []
        real = rt.Runtime.agree
        def spy(self_, stage, payload, valid=True):
            payloads.append((stage, list(payload)))
            return real(self_, stage, payload, valid)
        with patch.object(rt.Runtime, 'agree', spy):
            self.assertEqual(self.cycle(r, 'a', [(.5, .9)]), dict(a=1))
        self.assertEqual(payloads, [('control', ['k3-stop', 0, .74, 'uniform-batch', 'k2'])])
        self.assertEqual(self.cycle(r, 'b', [(.9, .7)])['b'], 2)
        self.assertEqual(self.cycle(r, 'c', [(.9, .9)])['c'], 3)
        self.assertEqual(self.cycle(r, 'cd', [(.1, .1)]*2)['c'], 2)
        self.assertEqual(self.cycle(r, 'c', [(.1, .1)])['c'], 1)

    def test_default_off_keeps_the_packet_behaviour(self):
        off, on = self.make('0'), self.make('1')
        payloads = []
        real = rt.Runtime.agree

        def spy(self_, stage, payload, valid=True):
            payloads.append((stage, list(payload)))
            return real(self_, stage, payload, valid)
        probs = [(.9, .9), (.5, .9), (.8, .8), (.95, .5)]
        with patch.object(rt.Runtime, 'agree', spy):
            got = self.cycle(off, 'abcd', probs)
        self.assertEqual(payloads, [('control', ['k3-stop', 0, .74])])        # no flag in the agreement
        self.assertEqual(got, dict(zip('abcd', [3, 1, 2, 2])))                 # per-request stop at c4
        self.assertEqual(off.guards, 2)                                         # one decision guard per pass
        self.assertNotIn('uniform_batch', off.audit())
        self.assertFalse(off.uniform_now)
        payloads.clear()
        with patch.object(rt.Runtime, 'agree', spy):
            self.cycle(on, 'abcd', probs)
        self.assertEqual(payloads, [('control', ['k3-stop', 0, .74, 'uniform-batch'])])

    def test_multi_request_batches_draft_three_without_a_decision(self):
        r = self.make('1')
        got = self.cycle(r, 'abcd', [(.1, .1), (.5, .9), (.8, .8), (.95, .5)])
        self.assertEqual(got, dict(a=3, b=3, c=3, d=3))
        self.assertEqual((r.guards, r.fallback_syncs, r.uniform_cycles), (0, 0, 1))
        self.assertEqual(r.audit()['uniform_cycles'], 1)
        # The next verify rectangle is uniform q4 and keeps its graph (no remap, no eager fallback).
        runner = NS(speculator=NS(_kstop=r))
        o = NS(finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[],
               scheduled_spec_decode_tokens={k: [-1] * 3 for k in 'abcd'}, num_scheduled_tokens={k: 4 for k in 'abcd'},
               total_num_scheduled_tokens=16)
        view = rt.select_inputs(runner, o)
        self.assertFalse(runner._kstop_needs_remap)
        self.assertEqual(view.num_scheduled_tokens, {k: 4 for k in 'abcd'})
        state(r)
        b = NS(req_ids=list('abcd'), num_reqs=4, query_start_loc_np=[0, 4, 8, 12, 16], num_tokens=16, num_tokens_after_padding=16,
               is_prefilling_np=[False] * 4)
        rt.prepare(NS(device='cpu'), b, False)
        self.assertTrue(rt.STATE['identity'])

    def test_one_request_keeps_the_confidence_stop(self):
        r = self.make('1')
        self.assertEqual(self.cycle(r, 'a', [(.5, .9)]), dict(a=1))
        self.assertEqual(self.cycle(r, 'b', [(.9, .7)]), dict(a=1, b=2))
        self.assertEqual(r.guards, 3)                                            # guards as without the flag
        self.assertEqual(r.uniform_cycles, 0)
        # c1 -> c2 -> c1 transitions: the flag follows the current draft batch only.
        self.cycle(r, 'cd', [(.1, .1), (.1, .1)])
        self.assertEqual((r.proposals['c'], r.proposals['d']), (3, 3))
        self.assertEqual(self.cycle(r, 'c', [(.1, .1)])['c'], 1)

    def test_uniform_choice_is_folded_into_the_guarded_trail(self):
        a, b = self.make('1'), self.make('1')
        self.cycle(a, 'ab', [(.9, .9), (.9, .9)])
        self.cycle(b, 'ab', [(.1, .1), (.2, .2)])
        self.assertEqual(a.trail, b.trail)                                       # probabilities are not read
        c = self.make('0')
        self.cycle(c, 'ab', [(.9, .9), (.9, .9)])
        self.assertNotEqual(a.trail, c.trail)

    def test_synthetic_and_other_modes_are_untouched(self):
        r = self.make('1')
        r.mode, r.epoch, r.tau = 'k2', 0, .74
        self.assertEqual(r.begin(NS(req_ids=list('ab'), num_reqs=2)), 2)
        self.assertFalse(r.uniform_now)
        s = rt.Runtime(NS(max_num_reqs=4, device='cpu', _kstop_synthetic=True))
        s.uniform = True
        self.assertEqual(s.begin(NS(req_ids=['x', 'y'], num_reqs=2)), 3)
        self.assertFalse(s.uniform_now)


class DeadRowOp(unittest.TestCase):
    def test_cpu_remap_reads_both_snapshots_before_writing(self):
        w = torch.tensor([[1.], [2.], [3.]])
        i = torch.tensor([[10], [20], [30]])
        torch.ops.glm_deadrow.remap_(w, i, torch.tensor([0, 0, 1]))
        self.assertEqual((w[:, 0].tolist(), i[:, 0].tolist()), ([1., 1., 2.], [10, 10, 20]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
