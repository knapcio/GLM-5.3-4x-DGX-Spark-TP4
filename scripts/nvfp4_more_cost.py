#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TP4 memory ledger and x2-discounted cycle forecast, never GPU evidence."""
import argparse
import json


def w8(n,k,channel=False):return n*k+2*n*(1 if channel else k//128)
def fp4(n,k):return n*k*9//16+4

def model(cycle_ms=75):
    if cycle_ms<=0:raise ValueError('positive B-current cycle prior required')
    shared=[(512,6144),(512,6144),(6144,512)]
    dense=[(3072,6144),(3072,6144),(6144,3072)]
    indexer=[(4096,2048),(128,6144),(32,6144)]
    mtp=[(2048,6144),(576,6144),(4096,2048),(6144,4096),*shared]
    routed=sum(w8(n,k,True) for n,k in shared)
    linears=sum(w8(n,k,True) for n,k in mtp)
    records={}
    for name,shapes,count,trace in [('shared',shared,75,3.959),('dense',dense,2,.535),('indexer',indexer,21,None)]:
        old=sum((2*n*k if name=='indexer' else w8(n,k)) for n,k in shapes)*count
        new=sum(fp4(n,k) for n,k in shapes)*count
        # Indexer's 32-output weights_proj is independently padded to N64 by Marlin.
        padding=21*32*6144*9//16 if name=='indexer' else 0
        workspace=(63 if name=='indexer' else count)*48*4
        traffic_gain=(old-new)/255e6/2
        saved=.65 if name=='indexer' else trace*(1-new/old)/2
        records[name]=dict(resident_source_bytes=old,resident_nvfp4_bytes=new,padding_bytes=padding,
            extra_workspace_bytes=workspace,freed_bytes=old-new-padding-workspace,
            read_source_bytes_per_cycle=old,read_nvfp4_bytes_per_cycle=new+padding,
            bandwidth_saved_ms_x2=traffic_gain,trace_saved_ms_x2=saved,
            added_split_calls_per_cycle=count,
            split_launch_allowance_ms=count*(.5/77),
            predicted_saved_ms=max(0,saved-count*(.5/77)))
    resident8=linears+w8(7168,512,True)+256*routed
    resident4=sum(fp4(n,k) for n,k in mtp)+fp4(7168,512)+256*sum(fp4(n,k) for n,k in shared)
    read8=2*linears+routed*(17.6+8)
    ratio=sum(fp4(n,k) for n,k in shared)/routed
    records['mtp']=dict(resident_source_bytes=resident8,resident_nvfp4_bytes=resident4,
        freed_bytes=resident8-resident4-48*4*4,extra_workspace_allowance_bytes=48*4*4,
        read_source_bytes_per_cycle=read8,read_nvfp4_bytes_per_cycle=read8*ratio,
        bandwidth_saved_ms_x2=(read8-read8*ratio)/255e6/2,
        trace_saved_ms_x2_range=[.13,.41],predicted_saved_ms_range=[.13-2*(.5/77),.41-2*(.5/77)],
        split_launch_allowance_ms=2*(.5/77),
        assumptions='K2 prior: two MTP passes, distinct expert unions17.6+8; K-stop and acceptance change this; UK/UV BF16 unchanged; three routed Marlin GEMMs instead of two')
    for name,r in records.items():
        r['freed_GB_per_rank']=r['freed_bytes']/1e9
        r['conservative_realizable_GB_per_rank_x2']=r['freed_bytes']/2e9
        savings=r.get('predicted_saved_ms_range',[r.get('predicted_saved_ms',0)])
        r['predicted_cycle_ms_gain_pct']=[100*s/cycle_ms for s in savings]
        r['predicted_cycle_rate_gain_pct']=[100*(cycle_ms/(cycle_ms-s)-1) for s in savings]
        r['predicted_exceeds_1pct_gate']=all(s/cycle_ms>.01 for s in savings)
    return dict(scope='Analytic forecast only; memory ledger is exact tensor bytes minus stated allowances, no admission/cap change',
        baseline_cycle_prior_ms=cycle_ms,baseline_note='75ms illustrative current B prior, replace with paired same-day cycle.py median',
        bandwidth_GBs=255,discount=2,groups=records,
        attention_incremental_gain=0,
        conversion=dict(elements=dict(shared=2831155200,dense=452984832,indexer=196804608,mtp=9866444800),
            total_elements=13347389440,nvfp4_payload_bytes_before_headers=7507906560,source_shards=81,
            quantization_rate_elements_per_s=[10000000,40000000],estimated_total_minutes=[10,35],reserve_minutes=45),
        caveat='Trace priors from nvfp4w PROBE.md; split allowance borrowed from attention, not measured for these groups. Memory savings are not discounted physical tensor bytes; x2 realizable column is a conservative credit only. DRAM/L2/reloads/activation/KV/TP/graph/transient costs unmeasured.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cycle-ms',type=float,default=75);a=p.parse_args()
    print(json.dumps(model(a.cycle_ms),indent=2))
