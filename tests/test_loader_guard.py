"""Mac-only all-rank failure/fallback simulations; no fleet or network calls."""
import os
from pathlib import Path
import signal
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'overlay/overlay'))
import torch
import glm_loader_guard as G


class Store:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()
    def check(self, keys):
        with self.lock:
            return all(key in self.values for key in keys)
    def set(self, key, value):
        with self.lock:
            self.values[key] = value.encode()
    def get(self, key):
        with self.lock:
            return self.values[key]


class Teams(unittest.TestCase):
    def teams(self, world=4, **kw):
        store = Store()
        teams = [G.LoadTeam(store, r, world, timeout=.5, terminate=lambda sig: None, **kw)
                 for r in range(world)]
        for team in teams:
            team.start()
        self.addCleanup(lambda: [team.stop.set() for team in teams])
        return teams

    def parallel(self, teams, fn):
        results, errors = {}, {}
        def run(rank):
            try:
                results[rank] = fn(teams[rank], rank)
            except Exception as exc:
                errors[rank] = exc
        threads = [threading.Thread(target=run, args=(r,)) for r in range(len(teams))]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads), 'rank deadlock')
        return results, errors

    def test_one_rank_refusal_makes_all_four_fallback(self):
        teams = self.teams()
        results, errors = self.parallel(teams, lambda team, rank: team.ready(rank != 2))
        self.assertEqual(errors, {})
        self.assertEqual(results, {r: False for r in range(4)})
        results, errors = self.parallel(teams, lambda team, rank: team.ready(True))
        self.assertEqual(errors, {})
        self.assertEqual(results, {r: True for r in range(4)})
        _, errors = self.parallel(teams, lambda team, rank: team.finish())
        self.assertEqual(errors, {})
        self.assertTrue(all(not team.thread.is_alive() for team in teams))

    def test_partial_rank_abort_reaches_completed_and_loading_peers(self):
        teams = self.teams()
        teams[1].abort()
        time.sleep(.15)
        _, errors = self.parallel(teams, lambda team, rank: team.finish())
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(isinstance(exc, RuntimeError) for exc in errors.values()))

    def test_missing_rank_times_out_instead_of_deadlocking(self):
        teams = self.teams()
        with self.assertRaises(TimeoutError): teams[0].ready(True)
        time.sleep(.15)
        for team in teams:
            with self.assertRaises(RuntimeError): team.check()

    def test_stuck_native_peer_has_bounded_termination(self):
        signals = []
        team = G.LoadTeam(Store(), grace=.01, terminate=signals.append)
        team.start()
        self.addCleanup(team.stop.set)
        team.abort()
        time.sleep(.25)
        self.assertEqual(signals, [signal.SIGTERM])
        # No 5 s sleep in the suite; drive the watchdog's monotonic deadline.
        now = time.monotonic()
        with patch.object(G.time, 'monotonic', return_value=now + 10):
            time.sleep(.15)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])

    def test_store_failure_is_fail_closed(self):
        team = G.LoadTeam(Store(), terminate=lambda sig: None)
        team.start()
        self.addCleanup(team.stop.set)
        with patch.object(team.store, 'check', side_effect=OSError('channel lost')):
            time.sleep(.15)
            with self.assertRaises(RuntimeError): team.check()

    def test_wrapper_accounts_native_writes_and_clears_scope(self):
        model = torch.nn.Linear(4, 4, bias=False)
        team = G.LoadTeam()
        def load(self_, model_, config):
            with torch.no_grad():
                model_.weight[:2].copy_(torch.ones(2, 4))
            self.assertEqual(G.active().destination.committed, 32)
            return 'loaded'
        with patch.dict(os.environ, GLM_LOADER='coalesced'), \
             patch.object(G.LoadTeam, 'distributed', return_value=team):
            self.assertEqual(G.wrap_load_weights(load)(None, model, None), 'loaded')
        self.assertIsNone(G.active())

    def test_wrapper_failure_signals_peers_and_clears_scope(self):
        team = G.LoadTeam()
        def load(*args):
            raise MemoryError('partial native failure')
        with patch.dict(os.environ, GLM_LOADER='coalesced'), \
             patch.object(G.LoadTeam, 'distributed', return_value=team):
            with self.assertRaises(MemoryError): G.wrap_load_weights(load)(None, torch.nn.Module(), None)
        self.assertTrue(team.failed.is_set())
        self.assertIsNone(G.active())

    def test_multi_rank_without_failure_channel_refuses(self):
        with patch.dict(os.environ, WORLD_SIZE='4'), \
             patch.object(torch.distributed, 'is_initialized', return_value=False):
            with self.assertRaises(RuntimeError): G.LoadTeam.distributed(1)

    def test_actual_wrapper_all_ranks_fallback_before_first_handoff(self):
        import glm_fast_load as F
        import glm_coalesced_load as C
        teams = self.teams()
        wu = types.SimpleNamespace(_natural_sort_key=lambda p: p, tqdm=lambda fs, **kw: fs,
             enable_tqdm=lambda _: False, _BAR_FORMAT='{desc}', should_skip_weight=lambda n, ids: False)
        def candidate(*args, **kw):
            if G.active().team.rank == 2:
                raise MemoryError('rank 2 pre-placement pressure')
            yield 'candidate', torch.zeros(1)
        def fast(*args, **kw):
            yield 'normal', torch.zeros(1)
        wrapper = F._make_wrapper(lambda *a, **kw: iter(()), wu)
        def run(team, rank):
            G._local.scope = G.LoadScope(team=team)
            try:
                names = [name for name, _ in wrapper([], False, 'lazy')]
                self.assertTrue(G.active().fallback)
                team.finish()
                return names
            finally:
                G._local.scope = None
        with patch.dict(os.environ, GLM_LOADER='coalesced'), \
             patch.object(C, 'coalesced_safetensors_iterator', candidate), \
             patch.object(F, 'fast_safetensors_iterator', fast), patch.object(F, 'release'):
            results, errors = self.parallel(teams, run)
        self.assertEqual(errors, {})
        self.assertEqual(results, {r: ['normal'] for r in range(4)})

    def test_fallback_is_sticky_across_model_sources(self):
        import glm_fast_load as F
        import glm_coalesced_load as C
        scope = G.LoadScope()
        calls = []
        wu = types.SimpleNamespace(_natural_sort_key=lambda p: p, tqdm=lambda fs, **kw: fs,
             enable_tqdm=lambda _: False, _BAR_FORMAT='{desc}', should_skip_weight=lambda n, ids: False)
        def candidate(*args, **kw):
            calls.append('coalesced')
            raise MemoryError('before placement')
            yield
        def fast(*args, **kw):
            calls.append('fast')
            yield 'normal', torch.zeros(1)
        wrapper = F._make_wrapper(lambda *a, **kw: iter(()), wu)
        with patch.dict(os.environ, GLM_LOADER='coalesced'), patch.object(G, 'active', return_value=scope), \
             patch.object(C, 'coalesced_safetensors_iterator', candidate), \
             patch.object(F, 'fast_safetensors_iterator', fast), patch.object(F, 'release'):
            for _ in range(2):
                self.assertEqual([name for name, _ in wrapper([], False, 'lazy')], ['normal'])
        self.assertEqual(calls, ['coalesced', 'fast', 'fast'])

    def test_prior_destination_write_prohibits_fallback(self):
        import glm_fast_load as F
        import glm_coalesced_load as C
        scope = G.LoadScope()
        scope.destination.committed = 1
        wu = types.SimpleNamespace(_natural_sort_key=lambda p: p, tqdm=lambda fs, **kw: fs,
             enable_tqdm=lambda _: False, _BAR_FORMAT='{desc}', should_skip_weight=lambda n, ids: False)
        def candidate(*args, **kw):
            raise MemoryError('early in this source, late in the model load')
            yield
        wrapper = F._make_wrapper(lambda *a, **kw: iter(()), wu)
        with patch.dict(os.environ, GLM_LOADER='coalesced'), patch.object(G, 'active', return_value=scope), \
             patch.object(C, 'coalesced_safetensors_iterator', candidate), \
             patch.object(F, 'fast_safetensors_iterator', side_effect=AssertionError('unsafe fallback')), \
             patch.object(F, 'release'):
            with self.assertRaisesRegex(RuntimeError, 'after destination placement'):
                list(wrapper([], False, 'lazy'))

    def test_tcpstore_loopback_four_rank_failure_channel(self):
        # Mac loopback only: exercise the real PrefixStore/TCPStore API used
        # on the fleet, including a separate client's 2 s timeout contract.
        dist = torch.distributed
        server = dist.TCPStore('127.0.0.1', 0, None, True, wait_for_workers=False)
        base = dist.PrefixStore('test-default/', server)
        teams = []
        with patch.object(dist, 'is_initialized', return_value=True), \
             patch.object(dist, 'get_world_size', return_value=4), \
             patch.object(dist.distributed_c10d, '_get_default_store', return_value=base):
            for rank in range(4):
                with patch.object(dist, 'get_rank', return_value=rank):
                    teams.append(G.LoadTeam.distributed(9001))
        for team in teams:
            team.terminate = lambda sig: None
            team.timeout = 1
            team.start()
        self.addCleanup(lambda: [team.stop.set() for team in teams])
        results, errors = self.parallel(teams, lambda team, rank: team.ready(rank != 3))
        self.assertEqual(errors, {})
        self.assertEqual(results, {r: False for r in range(4)})
        teams[2].abort()
        time.sleep(.15)
        _, errors = self.parallel(teams, lambda team, rank: team.finish())
        self.assertEqual(len(errors), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
