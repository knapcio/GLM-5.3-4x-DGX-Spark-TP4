# SPDX-License-Identifier: Apache-2.0
"""CPU/Torch laws + negative controls; use GLM_IMAGE_SRC on a host without vLLM."""
import ast
import copy
import json
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/bringup'))
import glm_spec_sample as S
from spec_sample_distribution import (cases, processor, assert_exact, score,
    batch_cases, batch_layout, ConditionalCounts, assert_cache_layout, assert_stock_bias)

SOURCE = Path(os.environ.get('GLM_IMAGE_SRC') or Path(importlib.util.find_spec('vllm').origin).parent.parent)
DRAWS = 1_000_000


def simulate(p, q, shared):
    """Same rejection/residual law; independent stream is domain-separated.

    This models independence, not Triton's PRNG: the GPU script tests Philox.
    """
    draft_rng = torch.Generator().manual_seed(20261002)
    accept_rng = torch.Generator().manual_seed(91239)
    residual_rng = torch.Generator().manual_seed(20261002 ^ S.RESIDUAL_SALT)
    counts = torch.zeros(len(p), dtype=torch.int64)
    accepted = 0
    for start in range(0, DRAWS, 100_000):
        n = min(100_000, DRAWS - start)
        noise = -torch.log(-torch.log(torch.rand(n, len(p), generator=draft_rng, dtype=torch.float64)))
        x = (q.log() + noise).argmax(-1)
        accept = torch.rand(n, generator=accept_rng, dtype=torch.float64) < (p[x] / q[x]).clamp(max=1)
        r = (p - q).clamp(min=0)
        noise2 = noise if shared else -torch.log(-torch.log(torch.rand(n, len(p), generator=residual_rng, dtype=torch.float64)))
        y = (r.log() + noise2).argmax(-1)
        counts += torch.bincount(torch.where(accept, x, y), minlength=len(p))
        accepted += int(accept.sum())
    return counts, accepted


def simulate_batch(profiles, batch, offset, rngs, shared=False, dtype=torch.float32, draft_sampler=None):
    """Torch equivalent of K2 branch and cache semantics, not Triton's PRNG."""
    kinds, idx, temps, _, target, proposal, placeholders, cache = batch_layout(profiles, batch, offset)
    proposal, cache = proposal.to(dtype), cache.to(dtype)
    draft_rng, accept_rng, residual_rng = rngs
    def noise(shape, rng):
        return -torch.log(-torch.log(torch.rand(shape, generator=rng, dtype=torch.float64)))
    g = noise(target.shape, draft_rng)
    g2 = noise(target.shape, residual_rng)
    k=proposal.shape[1]
    u = torch.rand(batch, k, generator=accept_rng, dtype=torch.float64)
    temperature = temps[idx.long()].double()
    greedy = temperature == 0
    drafts = torch.full((batch, k), -1, dtype=torch.int64)
    for s in range(k):
        live = (placeholders < 0) | (s < placeholders)
        if draft_sampler is None:cache[idx[live].long(), s] = proposal[live, s]
        logits = proposal[:, s].double() / temperature.clamp(min=1e-10)[:, None]
        x = torch.where(greedy, proposal[:, s].argmax(-1), (logits + g[:, s]).argmax(-1))
        if draft_sampler is not None:
            x=draft_sampler(proposal[:,s],torch.where(live,idx,-1),temps,cache,s,g[:,s])
        drafts[live, s] = x[live]
    assert_cache_layout(cache, proposal, idx, placeholders)
    sampled = torch.full((batch, k+1), -1, dtype=torch.int64)
    lengths = torch.ones(batch, dtype=torch.int32)
    reach = torch.ones(batch, dtype=torch.bool)
    for s in range(k+1):
        p = target[:, s].double().softmax(-1)
        if s == k:
            y = torch.where(greedy, target[:, s].argmax(-1), (p.log() + g[:, s]).argmax(-1))
        else:
            x = drafts[:, s].clamp(min=0)
            valid = drafts[:, s] >= 0
            q = (cache[idx.long(), s].double() / temperature.clamp(min=1e-10)[:, None]).softmax(-1)
            accept = valid & torch.where(greedy, x == target[:, s].argmax(-1),
                u[:, s] < (p.gather(1, x[:, None]) / q.gather(1, x[:, None])).flatten().clamp(max=1))
            r = (p - q).clamp(min=0)
            # Poisoned placeholders go straight to target; no residual/cache law.
            residual = torch.where(valid[:, None], r, p)
            resample_noise = torch.where((valid & ~greedy)[:, None], g[:, s] if shared else g2[:, s], g[:, s])
            y = torch.where(greedy, target[:, s].argmax(-1), (residual.log() + resample_noise).argmax(-1))
            y = torch.where(accept, drafts[:, s], y)
        sampled[reach, s] = y[reach]
        if s < k:
            reach &= accept
            lengths += reach.int()
    return kinds, sampled, lengths, drafts


class Exactness(unittest.TestCase):
    def test_independent_passes_stock_fails(self):
        for name, _, p, q in cases(processor(SOURCE)):
            fixed, accept = simulate(p, q, False)
            result = assert_exact(fixed, p)
            # Symbolic law: accepted mass min(p,q) + rejection_mass * r = p.
            r = (p - q).clamp(min=0)
            law = torch.minimum(p, q) + r
            self.assertTrue(torch.allclose(law, p, atol=1e-15, rtol=0))
            self.assertLess(abs(accept / DRAWS - float(torch.minimum(p, q).sum())), .002)
            stock, stock_accept = simulate(p, q, True)
            self.assertEqual(accept, stock_accept)
            negative = score(stock, p)
            if name == 'plain':
                assert_stock_bias(stock)
            if name not in ('equal', 'disjoint'):
                with self.assertRaises(AssertionError):
                    assert_exact(stock, p)
            print(json.dumps({'case': name, 'fixed': result, 'stock': negative}), flush=True)

    def test_heterogeneous_k2_conditional_laws_and_poison(self):
        profiles = batch_cases(processor(SOURCE))
        for dtype in (torch.float32, torch.bfloat16):
            for c in profiles:
                c.q = (c.proposal.to(dtype).float().double() / (c.temperature or 1.)).softmax(-1)
            tally = ConditionalCounts(profiles)
            rngs = [torch.Generator().manual_seed(seed) for seed in (20261002, 91239, 20261002 ^ S.RESIDUAL_SALT)]
            draws = DRAWS * len(profiles)
            for offset in range(0, draws, 20_000):
                tally.add(*simulate_batch(profiles, min(20_000, draws - offset), offset, rngs, dtype=dtype))
            results = tally.check()
            print(json.dumps({'suite': 'CPU_heterogeneous_K2', 'dtype': str(dtype),
                              'checks': len(results), 'min_conditional_draws': min(r['draws'] for r in results)}), flush=True)

    def test_conditional_oracle_rejects_remap_and_poison_mistakes(self):
        profiles = batch_cases(processor(SOURCE))
        _, idx, _, _, _, proposal, placeholders, cache = batch_layout(profiles, 100, 0)
        for s in range(2):
            live = (placeholders < 0) | (s < placeholders)
            cache[idx[live].long(), s] = proposal[live, s]
        assert_cache_layout(cache, proposal, idx, placeholders)
        with self.assertRaisesRegex(AssertionError, 'cache remap/column'):
            assert_cache_layout(cache.flip(0), proposal, idx, placeholders)
        poisoned = cache.clone()
        poisoned[0, 2] = 0
        with self.assertRaisesRegex(AssertionError, 'poisoned cache slot overwritten'):
            assert_cache_layout(poisoned, proposal, idx, placeholders)
        # Swapping heterogeneous output laws must fail the distribution oracle.
        for position, condition in ((0, 'reach'), (1, 'prefix0=0'), (2, 'prefix=0,1')):
            tally = ConditionalCounts(profiles)
            tally.counts[(0, position, condition)] = (profiles[1].p[position] * DRAWS).round().long()
            with self.assertRaises(AssertionError):
                tally.check()

    def test_mixed_greedy_bonus_placeholder_stock_identity(self):
        profiles = batch_cases(processor(SOURCE))
        def run(shared):
            rngs = [torch.Generator().manual_seed(seed) for seed in (7, 17, 27)]
            return simulate_batch(profiles, 20_000, 0, rngs, shared=shared)
        kinds, a, alen, _ = run(True)
        _, b, blen, _ = run(False)
        self.assertTrue(torch.equal(alen, blen))
        greedy = torch.tensor([c.temperature == 0 for c in profiles])[kinds]
        valid = torch.arange(3)[None] < alen[:, None]
        self.assertTrue(torch.equal(a[valid & greedy[:, None]], b[valid & greedy[:, None]]))
        placeholder = torch.tensor([c.placeholder for c in profiles])[kinds]
        terminal = alen.long() - 1
        unchanged = (alen == 3) | ((placeholder >= 0) & (terminal == placeholder))
        rows = torch.arange(len(kinds))[unchanged]
        self.assertTrue(torch.equal(a[rows, terminal[rows]], b[rows, terminal[rows]]))

    def test_pins_and_only_noise_delta(self):
        for name in S.PINS:
            mod = types.SimpleNamespace(__name__=name, __file__=str(SOURCE / (name.replace('.', '/') + '.py')))
            S.check_source(mod)
        def fn(source, name):
            return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        stock_resample = fn((SOURCE / (S.REJECTION.replace('.', '/') + '.py')).read_text(), '_resample_kernel')
        kernel = (ROOT / 'overlay/bringup/glm_spec_sample_kernel.py').read_text()
        fixed_resample = fn(kernel, '_resample_kernel')
        salt_nodes = []
        for n in ast.walk(fixed_resample):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'residual_gumbel_block_argmax':
                n.func.id = 'gumbel_block_argmax'
                salt_nodes = [k.value for k in n.keywords if k.arg == 'SEED_SALT']
                n.keywords = [k for k in n.keywords if k.arg != 'SEED_SALT']
        self.assertEqual(len(salt_nodes), 1)
        self.assertEqual(ast.unparse(salt_nodes[0]), 'tl.where(HAS_DRAFT_LOGITS & (temp != 0.0) & (not is_bonus) & is_valid_rejected_draft, RESIDUAL_SALT, 0)')
        self.assertEqual(ast.dump(stock_resample), ast.dump(fixed_resample))
        stock_helper = fn((SOURCE / 'vllm/v1/worker/gpu/sample/gumbel.py').read_text(), 'gumbel_block_argmax')
        fixed_helper = fn(kernel, 'residual_gumbel_block_argmax')
        fixed_helper.name = 'gumbel_block_argmax'
        fixed_helper.args.args = [a for a in fixed_helper.args.args if a.arg != 'SEED_SALT']
        for n in ast.walk(fixed_helper):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'randint':
                self.assertEqual(ast.unparse(n.args[0]), 'seed ^ SEED_SALT')
                n.args[0] = ast.Name(id='seed', ctx=ast.Load())
        self.assertEqual(ast.dump(stock_helper), ast.dump(fixed_helper))

    def test_default_inert_and_config_gate(self):
        before = list(sys.meta_path)
        self.assertFalse(S.register({}))
        self.assertEqual(before, sys.meta_path)
        for value in ('', '2', 'yes'):
            with self.assertRaises(ValueError):
                S.enabled({'GLM_SPEC_SAMPLE': value})
        class Config:
            def __post_init__(self):
                self.seen = self.draft_sample_method
        module = types.SimpleNamespace(__name__=S.CONFIG, SpeculativeConfig=Config)
        with patch.object(S, 'check_source'):
            S.install(module)
            S.install(module)
        c = Config()
        c.method, c.rejection_sample_method, c.use_local_argmax_reduction = 'mtp', 'standard', False
        c.draft_sample_method = 'greedy'
        c.__post_init__()
        self.assertEqual(c.seen, 'probabilistic')
        for key, value in (('method', 'dspark'), ('rejection_sample_method', 'block'), ('use_local_argmax_reduction', True)):
            bad = copy.copy(c); setattr(bad, key, value)
            with self.assertRaises(ValueError):
                bad.__post_init__()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
