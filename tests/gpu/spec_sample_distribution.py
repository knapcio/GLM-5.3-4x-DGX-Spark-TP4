#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU real kernels: K1 negative control + heterogeneous conditional K2.

Run later, inside the pinned image with this checkout mounted at /pkg:
  python3 /pkg/tests/gpu/spec_sample_distribution.py --draws 1000000
No model weights, network, server, TP, or overlay startup needed.
--draws is trials per profile, including each profile of the mixed K2 batch.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'overlay/bringup')]
from spec_sample_distribution import (cases, processor, assert_exact, score,
    batch_cases, batch_layout, ConditionalCounts, assert_cache_layout, assert_stock_bias)
import glm_spec_sample as S


def inputs(batch, p_logits, q, offset, dtype=torch.float32):
    device = 'cuda'
    vocab = len(q)
    idx = torch.arange(batch, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * 2
    pos = torch.tensor([19, 20], dtype=torch.int64, device=device).repeat(batch)
    # Every draw has its own seed; request remapping is exercised below.
    seeds = (torch.arange(offset, offset + batch, dtype=torch.int64, device=device) * 104729 + 9719)
    target = p_logits.float().to(device).repeat(2 * batch, 1)
    cache = torch.empty(batch, 1, vocab, dtype=dtype, device=device)
    proposal = q.log().to(device=device, dtype=dtype).repeat(batch, 1)
    return idx, cu, pos, seeds, target, cache, proposal


def verify(U, p_logits, q, batch, offset, temperature=1., probabilistic=True,
           dtype=torch.float32, placeholders=False, remap=False, fp64=False):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    idx, cu, pos, seeds, target, cache, proposal = inputs(batch, p_logits, q, offset, dtype)
    if remap:
        idx = idx.flip(0).contiguous()
    temps = (torch.full((batch,), temperature, dtype=torch.float32, device='cuda')
             if isinstance(temperature, (int, float)) else torch.tensor(temperature, dtype=torch.float32, device='cuda'))
    if probabilistic:
        x = gumbel_sample(proposal, idx, temps, seeds, pos[::2].contiguous(),
                          apply_temperature=True, logits_cache=cache,
                          logits_cache_col=torch.tensor(0, device='cuda', dtype=torch.int32),
                          use_fp64=fp64)
    else:
        x = proposal.argmax(-1)
    if placeholders:
        x[:] = -1
    drafts = torch.stack((torch.zeros_like(x), x), -1).flatten()
    sampled, lengths = U.rejection_sample(
        target, cache if probabilistic else None, drafts, cu, pos,
        idx, idx.repeat_interleave(2), torch.tensor([0, 1], dtype=torch.int32, device='cuda').repeat(batch),
        temps, seeds, 1, use_fp64=fp64,
    )
    return sampled, lengths


def verify_batch(U, profiles, batch, offset, stock, fixed, dtype, fp64):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    kinds, idx, temps, seeds, target, proposal, placeholders, cache = batch_layout(
        profiles, batch, offset, device='cuda')
    proposal = proposal.to(dtype)
    cache = cache.to(dtype)
    # Absolute positions must match between draft noise and residual/bonus noise.
    pos = torch.tensor([37, 38, 39], device='cuda', dtype=torch.int64).repeat(batch)
    drafts = torch.full((batch, 3), -1, device='cuda', dtype=torch.int64)
    drafts[:, 0] = 0  # already emitted token, outside verification
    for s in range(2):
        live = (placeholders < 0) | (s < placeholders)
        mapping = torch.where(live, idx, -1).contiguous()
        x = gumbel_sample(proposal[:, s].contiguous(), mapping, temps, seeds,
                          pos.reshape(batch, 3)[:, s].contiguous(),
                          apply_temperature=True, logits_cache=cache,
                          logits_cache_col=torch.tensor(s, device='cuda', dtype=torch.int32),
                          use_fp64=fp64)
        drafts[live, s + 1] = x[live]
    assert_cache_layout(cache, proposal, idx, placeholders)
    args = (target.flatten(0, 1), cache, drafts.flatten(),
            torch.arange(batch + 1, device='cuda', dtype=torch.int32) * 3,
            pos, idx, idx.repeat_interleave(3),
            torch.tensor([0, 1, 2], device='cuda', dtype=torch.int32).repeat(batch),
            temps, seeds, 2)
    try:
        U._resample_kernel = stock
        a, alen = U.rejection_sample(*args, use_fp64=fp64)
        U._resample_kernel = fixed
        b, blen = U.rejection_sample(*args, use_fp64=fp64)
        assert torch.equal(alen, blen), 'residual salt changed acceptance lengths'
        # Greedy valid bytes, and terminal target-only placeholder/bonus samples,
        # use precisely the stock seed and must retain token/length identity.
        greedy = temps[idx.long()] == 0
        valid = torch.arange(3, device='cuda')[None] < alen[:, None]
        assert torch.equal(a[valid & greedy[:, None]], b[valid & greedy[:, None]]), 'mixed T=0'
        terminal = blen.long() - 1
        unchanged = (blen == 3) | ((placeholders >= 0) & (terminal == placeholders))
        rows = torch.arange(batch, device='cuda')[unchanged]
        assert torch.equal(a[rows, terminal[rows]], b[rows, terminal[rows]]), 'bonus/placeholder seed changed'
        return kinds, b, blen, drafts[:, 1:]
    finally:
        U._resample_kernel = stock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draws', type=int, default=1_000_000)
    ap.add_argument('--batch', type=int, default=4096)
    args = ap.parse_args()
    assert args.draws >= 1_000_000 and args.batch > 0
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expose exactly one GPU'
    U = importlib.import_module(S.REJECTION)
    for name in S.PINS:
        S.check_source(importlib.import_module(name))
    import vllm
    source_root = Path(vllm.__file__).resolve().parent.parent
    stock = U._resample_kernel
    from glm_spec_sample_kernel import _resample_kernel as fixed
    for name, logits, p, q in cases(processor(source_root)):
        results = {}
        for mode, kernel in (('stock', stock), ('fixed', fixed)):
            U._resample_kernel = kernel
            counts = torch.zeros(len(q), device='cuda', dtype=torch.int64)
            for offset in range(0, args.draws, args.batch):
                b = min(args.batch, args.draws - offset)
                sampled, _ = verify(U, logits, q, b, offset)
                counts += torch.bincount(sampled[:, 0], minlength=len(q))
            results[mode] = score(counts, p)
            if mode == 'stock' and name == 'plain':
                assert_stock_bias(counts)
            if mode == 'fixed' or name in ('equal', 'disjoint'):
                assert_exact(counts, p)
            else:
                assert results[mode]['pvalue'] < 1e-6, 'stock negative control did not fail'
        print(json.dumps({'case': name, **results}), flush=True)
    # K2 heterogeneous requests in the SAME batch, including mixed T=0/T>0,
    # per-position top-k/top-p target laws, sparse remapping and NaN cache poison.
    profiles = batch_cases(processor(source_root))
    for dtype, fp64 in ((torch.float32, False), (torch.bfloat16, False), (torch.float32, True)):
        # bf16 proposal cache changes q; construct the oracle from those exact
        # pre-temperature bytes, as used by draft sampling and verification.
        for c in profiles:
            c.q = (c.proposal.to(dtype).float().double() / (c.temperature or 1.)).softmax(-1)
        tally = ConditionalCounts(profiles)
        draws = args.draws * len(profiles)
        for offset in range(0, draws, args.batch):
            tally.add(*verify_batch(U, profiles, min(args.batch, draws - offset), offset,
                                    stock, fixed, dtype, fp64))
        for result in tally.check():
            print(json.dumps({'suite': 'heterogeneous_K2', 'dtype': str(dtype), 'fp64': fp64,
                              **result}), flush=True)
    # Native bf16 proposal cache, fp64 noise, placeholder and remapped request checks.
    for dtype, fp64, placeholder in ((torch.bfloat16, False, False), (torch.float32, True, False), (torch.bfloat16, False, True)):
        logits = torch.tensor([.5, .3, .2], dtype=torch.float64).log()
        q = torch.tensor([.7, .2, .1], dtype=torch.float64)
        counts = torch.zeros(3, device='cuda', dtype=torch.int64)
        U._resample_kernel = fixed
        for offset in range(0, args.draws, args.batch):
            sampled, _ = verify(U, logits, q, min(args.batch, args.draws-offset), offset,
                                dtype=dtype, placeholders=placeholder, remap=True, fp64=fp64)
            counts += torch.bincount(sampled[:, 0], minlength=3)
        print(json.dumps({'dtype': str(dtype), 'fp64': fp64, 'placeholder': placeholder,
                          **assert_exact(counts, logits.softmax(-1))}), flush=True)
    # Stock greedy vs fixed probabilistic T=0: accepted, rejected, ties, bonus.
    for q in (torch.tensor([.7, .2, .1]), torch.tensor([.1, .2, .7]), torch.tensor([.4, .4, .2])):
        for target in ([.5, .3, .2], [.4, .4, .2]):
            logits = torch.tensor(target).log()
            U._resample_kernel = stock
            a, alen = verify(U, logits, q, 64, 0, temperature=0., probabilistic=False, dtype=torch.bfloat16)
            U._resample_kernel = fixed
            b, blen = verify(U, logits, q, 64, 0, temperature=0., probabilistic=True, dtype=torch.bfloat16)
            mask = torch.arange(a.shape[1], device='cuda')[None, :] < alen[:, None]
            assert torch.equal(alen, blen) and torch.equal(a[mask], b[mask]), 'T=0 token/length bits changed'
    # Mixed batches: temperature-zero rows retain the same valid token bytes.
    logits = torch.tensor([.5, .3, .2]).log()
    q = torch.tensor([.1, .2, .7])
    temps = [0., 1.] * 32
    U._resample_kernel = stock
    a, alen = verify(U, logits, q, 64, 0, temperature=temps, probabilistic=False)
    U._resample_kernel = fixed
    b, blen = verify(U, logits, q, 64, 0, temperature=temps, probabilistic=True)
    rows = torch.tensor(temps, device='cuda') == 0
    mask = rows[:, None] & (torch.arange(a.shape[1], device='cuda')[None, :] < alen[:, None])
    assert torch.equal(alen[rows], blen[rows]) and torch.equal(a[mask], b[mask]), 'mixed T=0 changed'
    U._resample_kernel = stock
    torch.cuda.synchronize()
    print(json.dumps({'PASS': True, 'T0_bit_identity': True, 'draws_per_case': args.draws,
                      'GPU': torch.cuda.get_device_name(0)}), flush=True)


if __name__ == '__main__':
    main()
