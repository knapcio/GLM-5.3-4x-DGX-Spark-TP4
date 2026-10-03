# SPDX-License-Identifier: Apache-2.0
"""Shared toy laws and chi-square oracle for CPU and real GPU kernels."""
import ast
import hashlib
from pathlib import Path
import torch

TOPK_PIN = 'bc8c1d91ad624368d2bd87a53c0361e550d28b060bcd460462d2f86565c26426'


def processor(source_root):
    path = Path(source_root) / 'vllm/v1/sample/ops/topk_topp_sampler.py'
    source = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TOPK_PIN, 'target processor drift'
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'apply_top_k_top_p_pytorch')
    # Execute the pinned Torch body without importing vLLM/CUDA on CPU.
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[node.name]


def cases(process):
    for name, raw_p, q, top_k, top_p in (
        ('plain', [.5, .3, .2], [.7, .2, .1], None, None),
        ('top_p', [.45, .27, .18, .10], [.65, .16, .09, .10], None, .85),
        ('top_k', [.45, .27, .18, .10], [.65, .16, .09, .10], 3, None),
        ('top_k_top_p', [.45, .27, .18, .06, .04], [.65, .16, .09, .06, .04], 4, .9),
        ('equal', [.5, .3, .2], [.5, .3, .2], None, None),
        ('disjoint', [0., .6, .4], [1., 0., 0.], None, None),
    ):
        logits = torch.tensor(raw_p, dtype=torch.float64).log()[None]
        logits = process(logits, None if top_k is None else torch.tensor([top_k]),
                         None if top_p is None else torch.tensor([top_p], dtype=torch.float64))
        p = logits.softmax(-1)[0]
        yield name, logits[0], p, torch.tensor(q, dtype=torch.float64)


def score(counts, p):
    counts = counts.cpu().double()
    p = p.cpu().double()
    assert counts[p == 0].sum() == 0, 'tokens outside processed target support'
    expected = counts.sum() * p[p > 0]
    chi2 = ((counts[p > 0] - expected).square() / expected).sum()
    df = int((p > 0).sum()) - 1
    tail = torch.special.gammaincc(torch.tensor(df / 2.), chi2 / 2.).item() if df else 1.
    return {'chi2': chi2.item(), 'df': df, 'pvalue': tail,
            'draws': int(counts.sum()), 'counts': counts.tolist(), 'target': p.tolist()}


def assert_exact(counts, p):
    result = score(counts, p)
    assert result['pvalue'] >= 1e-6, result
    return result


# Each profile shares a vocabulary but has different laws/processing. The three
# rows are K2 target positions (including bonus); q has two proposal positions.
def batch_cases(process, k=2):
    from types import SimpleNamespace
    profiles = (
        ('plain', 1., 4, 1., -1, [.5, .3, .15, .05], [.7, .2, .08, .02]),
        ('temperature', .65, 4, 1., -1, [.2, .5, .2, .1], [.1, .7, .15, .05]),
        ('top_p', 1.3, 4, .85, -1, [.45, .27, .18, .1], [.65, .16, .09, .1]),
        ('top_k', .8, 3, 1., -1, [.1, .2, .5, .2], [.2, .1, .6, .1]),
        ('top_k_top_p', 1.7, 3, .9, -1, [.45, .27, .18, .1], [.65, .16, .09, .1]),
        ('placeholder0', .9, 4, 1., 0, [.2, .1, .4, .3], [.7, .1, .1, .1]),
        ('placeholder1', 1.2, 3, .9, 1, [.3, .4, .2, .1], [.5, .2, .2, .1]),
        ('greedy_accept', 0., 4, 1., -1, [.5, .3, .15, .05], [.7, .2, .08, .02]),
        ('greedy_reject', 0., 3, .85, -1, [.2, .1, .5, .2], [.7, .2, .08, .02]),
        ('greedy_tie', 0., 4, 1., -1, [.4, .4, .1, .1], [.5, .3, .1, .1]),
    )
    result = []
    for name, temp, top_k, top_p, placeholder, raw_p, raw_q in profiles:
        # Rotate by position so a wrong cache column/target row changes the law.
        raw = torch.tensor(raw_p, dtype=torch.float64).log()
        target = torch.stack([raw.roll(s) for s in range(k+1)])
        target = target / (temp or 1.)
        target = process(target, torch.full((k+1,), top_k), torch.full((k+1,), top_p, dtype=torch.float64))
        proposal = torch.stack([torch.tensor(raw_q, dtype=torch.float64).log().roll(s)
                                for s in range(k)])
        result.append(SimpleNamespace(name=name, temperature=temp, placeholder=placeholder,
                                      target=target, p=target.softmax(-1), proposal=proposal,
                                      q=(proposal / (temp or 1.)).softmax(-1)))
    return result


def batch_layout(profiles, batch, offset, device='cpu', remap=True):
    """Request order differs from sparse persistent state/cache order."""
    kinds = torch.arange(offset, offset + batch, device=device) % len(profiles)
    idx = torch.arange(batch, dtype=torch.int32, device=device)
    idx = 2 * (idx.flip(0) if remap else idx) + 1
    states = 2 * batch + 3
    temps = torch.full((states,), float('nan'), device=device)
    temps[idx.long()] = torch.tensor([c.temperature for c in profiles], device=device)[kinds]
    seeds = torch.zeros(states, dtype=torch.int64, device=device)
    seeds[idx.long()] = torch.arange(offset, offset + batch, device=device) * 104729 + 9719
    target = torch.stack([c.target for c in profiles]).to(device=device, dtype=torch.float32)[kinds]
    proposal = torch.stack([c.proposal for c in profiles]).to(device=device, dtype=torch.float32)[kinds]
    placeholders = torch.tensor([c.placeholder for c in profiles], device=device)[kinds]
    # No consumer may read gap slots, placeholder columns or the bonus column.
    cache = torch.full((states, target.shape[1], target.shape[-1]), float('nan'), device=device)
    return kinds, idx, temps, seeds, target, proposal, placeholders, cache


class ConditionalCounts:
    """Count only valid outputs, conditioned on reach and accepted prefix.

    Later p rows must be checked conditional on reaching the row; counting all
    padded bytes or conditioning on *acceptance of that row* is incorrect.
    """
    def __init__(self, profiles):
        self.profiles = profiles
        self.counts = {}

    def add(self, kinds, sampled, lengths, drafts):
        kinds, sampled, lengths, drafts = (v.cpu() for v in (kinds, sampled, lengths, drafts))
        vocab = self.profiles[0].p.shape[-1]
        k=len(self.profiles[0].q)
        assert bool(((lengths >= 1) & (lengths <= k+1)).all()), 'invalid speculative output length'
        for ci, c in enumerate(self.profiles):
            rows = kinds == ci
            if c.temperature == 0:
                expected = c.target.argmax(-1)
                prefix = torch.ones_like(rows)
                expected_len = torch.ones_like(lengths)
                for s in range(k):
                    prefix &= drafts[:, s] == expected[s]
                    expected_len += prefix.int()
                assert torch.equal(lengths[rows], expected_len[rows]), c.name
                for s in range(k+1):
                    valid = rows & (lengths > s)
                    assert bool((sampled[valid, s] == expected[s]).all()), (c.name, s)
                continue
            for s in range(k+1):
                reach = rows & (lengths > s)
                selections = [('reach', reach)]
                if s:
                    # Every accepted prefix, rather than just aggregate later rows.
                    for token in range(vocab):
                        selections.append((f'prefix0={token}', reach & (sampled[:, 0] == token)))
                if s >= 2:
                    for x in range(vocab):
                        for y in range(vocab):
                            selections.append((f'prefix={x},{y}', reach & (sampled[:, 0] == x)
                                               & (sampled[:, 1] == y)))
                if s < k and (c.placeholder < 0 or s < c.placeholder):
                    selections.append(('residual', rows & (lengths == s + 1)))
                for label, mask in selections:
                    key = (ci, s, label)
                    counts = torch.bincount(sampled[mask, s], minlength=vocab)
                    self.counts[key] = self.counts.get(key, torch.zeros(vocab, dtype=torch.int64)) + counts

    def check(self):
        results = []
        for (ci, s, label), counts in self.counts.items():
            c = self.profiles[ci]
            # Impossible prefixes/rows are expected to have no outputs.
            reachable = c.placeholder < 0 or s <= c.placeholder
            if label.startswith('prefix'):
                tokens = label.split('=')[1].split(',')
                reachable &= all(float(torch.minimum(c.p[j], c.q[j])[int(t)]) > 0
                                 for j, t in enumerate(tokens))
            if not reachable:
                assert counts.sum() == 0, (c.name, s, label)
                continue
            assert counts.sum() >= 100, (c.name, s, label, 'insufficient conditional draws')
            p = c.p[s]
            if label == 'residual':
                p = (p - c.q[s]).clamp(min=0)
                p = p / p.sum()
            result = assert_exact(counts, p)
            results.append({'case': c.name, 'position': s, 'condition': label, **result})
        assert results, 'no probabilistic coverage'
        return results


def assert_stock_bias(counts):
    """The required shared-noise negative control is recognizable, not just red."""
    counts = counts.cpu().double()
    observed = counts / counts.sum()
    expected = torch.tensor([.500, .291, .209], dtype=torch.float64)
    assert bool((abs(observed - expected) < .003).all()), observed.tolist()
    assert score(counts, torch.tensor([.5, .3, .2]))['pvalue'] < 1e-6


def assert_cache_layout(cache, proposal, idx, placeholders):
    k=proposal.shape[1]
    valid = torch.zeros(cache.shape[:2], dtype=torch.bool, device=cache.device)
    for s in range(k):
        live = (placeholders < 0) | (s < placeholders)
        valid[idx[live].long(), s] = True
        assert torch.equal(cache[idx[live].long(), s], proposal[live, s].to(cache.dtype)), 'cache remap/column'
    assert bool(torch.isnan(cache[~valid]).all()), 'poisoned cache slot overwritten'
