#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline format-aware sim-v4 extension. No fleet/container actions.

Use the campaign release-stack/sim_admission.py as the calibrated memory
anchor. Correct its FP8-only head rounding and replace capacity geometry.
Long-prefill loss is calibrated separately from steady boot admission.
No credit for the new tiled path is taken before its fleet memory rerun.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from fp4_kv_layout import layout, ORDINARY_BYTES
GiB = 1 << 30
CALIBRATION = Path(__file__).resolve().parents[1]/'tests/fixtures/fp4_prefill_memory_20261005.json'


def long_prefill_bytes(mode, prefix):
    c = json.loads(CALIBRATION.read_text())
    a = c['fp8']
    rates = [(warm-low)*GiB/a['prefix_tokens']
             for warm,low in zip(a['post_warm_GiB'],a['prefill_min_GiB'])]
    if mode == 'fp4x':
        b = c['fp4x']
        rates[0] = max(rates[0], (b['post_warm_rank0_GiB']-b['prefill_rank0_GiB'])*GiB/b['prefix_tokens'])
    return [rate*prefix for rate in rates]


def safe_length(sim_dir, mode, live_only=False):
    # All charged terms are monotone; search the 64-token release grid.
    low, high = 0, int(layout(mode)['max_model_len'])//64
    while low < high:
        middle = (low+high+1)//2
        r = forecast(sim_dir, mode, middle*64)
        if r['passes_live_8_0'] and (live_only or r['passes_8_5']):
            low = middle
        else:
            high = middle-1
    return low*64


def forecast(sim_dir, mode, maxlen, ordinary_bytes=ORDINARY_BYTES):
    path = Path(sim_dir)/'sim_admission.py'
    spec = importlib.util.spec_from_file_location('release_sim',path)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    mem = S.load_memory()
    base = S.project(mem,ORDINARY_BYTES,True,4,66112,64,3,[1,4,12,16])
    l = layout(mode, ordinary_bytes)
    if maxlen > int(l['max_model_len']):
        raise ValueError('context does not fit this format with six-block margin')
    # Stock sim assumes every available byte is FP8 ordinary KV. Here dispram
    # carries the tail; only the exact ordinary head affects MemAvailable.
    rounding_debit = (ordinary_bytes-base['kv_allocated_bytes'])/GiB
    extension = max(0,maxlen-66112)
    scratch = (36*32*16*512*4 + 36*32*2*16*4) if mode=='fp4x' else 0
    # Fixed four-slot constructor banks + constant-byte DeepGEMM output.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/bringup'))
    from glm_fp4_prefill import scratch_bytes, LOGITS_BYTES
    prefill_bank = scratch_bytes(pool_tokens=l['tokens'], max_model_len=maxlen) - LOGITS_BYTES if mode=='fp4x' else 0
    # Separate value/scales simultaneous workspace: 40 rows/context token,
    # 132 bytes each. Larger decode logits at the entire split dispatch cap.
    index_gather = extension*40*132 if mode=='fp8' else 0
    logits = extension*36*4
    # Overestimate independent block tables for all 101 tensors/4 slots.
    tables = 101*4*((maxlen+63)//64-(66112+63)//64)*4 if extension else 0
    # Extra prefill/dequant compiler temporaries, page alignment, and warm
    # allocator fragmentation allowance beyond the sim's existing 32 MiB.
    warm = 64<<20 if mode=='fp4x' else 0
    costs = dict(apc=0.06*GiB, fp4_split_scratch=scratch, tiled_prefill_bank=prefill_bank,
                 prefill_logits_tile=LOGITS_BYTES if mode=='fp4x' else 0,
                 index_gather_growth=index_gather, decode_logits_growth=logits,
                 block_table_growth=tables, extra_warm_allowance=warm)
    debit = sum(costs.values())/GiB+rounding_debit
    transient = long_prefill_bytes(mode, maxlen)
    # Replace the anchor's flat 0.09 GiB estimate; don't charge it twice.
    flat = S.LONG_CONTEXT_GIB
    steady_point = [v-debit+flat for v in base['per_rank_point_GiB']]
    steady_lower = [v-debit+flat for v in base['per_rank_lower_GiB']]
    point = [v-t/GiB for v,t in zip(steady_point, transient)]
    lower = [v-t/GiB for v,t in zip(steady_lower, transient)]
    live = min(lower)>=8.0
    need=1+(maxlen+3+63)//64
    # Geometry reasons belong to the old ordinary-only pool. Replace them,
    # preserving any unrelated simulator reason and the same >=8.5 rule.
    reasons=[r for r in base['reasons'] if not r.startswith(('KV pool ', 'vLLM startup check:'))]
    if need > l['blocks']:reasons.append('format-aware KV geometry does not fit')
    if min(lower)<8.5:reasons.append('sim lower including long-prefill transient below 8.5 GiB')
    if not live:reasons.append('projected prefill lower below 8.0 GiB live floor')
    return dict(format=mode,max_model_len=maxlen,layout=l,required_blocks=need,
        sim_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        memory_source_sha256=hashlib.sha256(S._memory_path().read_bytes()).hexdigest(),
        graph_layout=base['graphs'],bank_rule=base['bank_rule'],per_lever_bytes=costs,
        exact_head_rounding_debit_GiB=rounding_debit,rank0_point_GiB=point[0],
        rank0_lower_GiB=lower[0],per_rank_lower_GiB=lower,passes_8_5=not reasons,
        passes_live_8_0=live,per_rank_steady_lower_GiB=steady_lower,
        long_prefill_transient_bytes=transient,
        calibration_sha256=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
        prefill_policy='observed loss retained; zero speculative savings credit for tiled fix',
        reasons=reasons,scope='SIM ONLY: byte bounds plus calibrated v4 anchor; recheck on final package and all warmed graphs',
        unmeasured_terms=[term for term in base['unmeasured_terms'] if 'context above 32768' not in term]+[
            'FP4 split scratch plus fixed four-slot prefill bank/256 MiB logits; 64 MiB warm allowance',
            'Fixed-path residual not measured; retain boot B loss including allocator/API/sampler uncertainty',
            '50K boot B prefix is an owner estimate, not a logged exact token counter',
            'GPU register/shared usage, graph replay peak and fixed-path new-max prefill peak not measured'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    rows=[forecast(a.sim_dir,'fp8',66112),forecast(a.sim_dir,'fp4x',66112),forecast(a.sim_dir,'fp4x',100288)]
    safe = {mode:safe_length(a.sim_dir, mode) for mode in ('fp8','fp4x')}
    live_safe = {mode:safe_length(a.sim_dir, mode, live_only=True) for mode in ('fp8','fp4x')}
    rows += [forecast(a.sim_dir, 'fp4x', 80000), forecast(a.sim_dir, 'fp4x', safe['fp4x']), forecast(a.sim_dir, 'fp4x', safe['fp4x']+64)]
    a.out.write_text(json.dumps(dict(rule='sim lower >=8.5 GiB and projected live lower >=8.0 GiB on every rank',
        largest_safe_max_model_len=safe, largest_live_only_max_model_len=live_safe, fp4x_at_least_80K=safe['fp4x']>=80000, table=rows),indent=2)+'\n')
    for r in rows:
        print(r['format'],r['max_model_len'],r['layout']['blocks'],r['rank0_lower_GiB'],r['passes_8_5'])
