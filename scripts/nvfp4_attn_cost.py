#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-labelled TP4 byte and bandwidth model; these are not GPU measurements."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fp4_kv_layout as kv

SHAPES={'q_a':(2048,6144),'kv_a':(576,6144),'q_b':(4096,2048),'o_proj':(6144,4096),'kv_b':(7168,512)}
MS=tuple(range(1,17))+(32,64,128,256,512,1024,2048,4096)

def ledger():
    rows=[]
    for name,(n,k) in SHAPES.items():
        w8=n*k+n*(k//128)*2;w4=n*k//2+n*(k//16)+4
        rows.append(dict(name=name,n=n,k=k,int8_bytes=w8,nvfp4_bytes=w4,
            decode_call=name!='kv_b',int8_stream_us=w8/255000,nvfp4_stream_us=w4/255000,
            roofline=[dict(m=m,int8_us=max(w8/255000,2*m*n*k/100000000),
                nvfp4_us=max(w4/255000,2*m*n*k/100000000)) for m in MS]))
    b8=sum(r['int8_bytes'] for r in rows)*77;b4=sum(r['nvfp4_bytes'] for r in rows)*77
    # One extra SM-count int workspace for each separately executed A projection.
    workspace_delta=77*48*4
    raw_freed=b8-b4;freed=raw_freed-workspace_delta
    base=kv.layout('fp4x');head=kv.ORDINARY_BYTES+freed
    blocks=(head+kv.CARVE_BYTES)//base['block_bytes']
    while kv.aligned_bytes('fp4x',blocks)>head+kv.CARVE_BYTES:blocks-=1
    decode8=sum(r['int8_bytes'] for r in rows if r['decode_call'])*77
    decode4=sum(r['nvfp4_bytes'] for r in rows if r['decode_call'])*77
    saved_bandwidth=(decode8-decode4)/255e6/2-.5
    saved_trace=18.973*(1-.5625/1.015625)/2-.5
    return dict(rows=rows,bandwidth_GBs=255,bf16_effective_TFLOPs_assumption=100,
      roofline_note='optimistic lower bound, same BF16 compute rate for both formats; no scale/dequant, reload, launch or KV costs',
      int8_resident_bytes=b8,nvfp4_resident_bytes=b4,raw_freed_bytes=raw_freed,
      extra_workspace_bytes=workspace_delta,freed_bytes=freed,freed_GB=freed/1e9,freed_GiB=freed/2**30,
      decode_int8_bytes=decode8,decode_nvfp4_bytes=decode4,
      retained_absorbed_ukuv_bytes=78*7340032,
      cycle_prior_ms=83,extra_split_and_concat_allowance_ms=.5,
      forecast_saved_ms=[saved_bandwidth,saved_trace],
      forecast_cycle_reduction_pct=[100*saved_bandwidth/83,100*saved_trace/83],
      forecast_cycle_rate_gain_pct=[100*(83/(83-saved_bandwidth)-1),100*(83/(83-saved_trace)-1)],
      kv=dict(base_blocks=base['blocks'],base_ordinary_bytes=kv.ORDINARY_BYTES,
        unchanged_dispram_bytes=kv.CARVE_BYTES,new_ordinary_bytes=head,
        block_bytes=base['block_bytes'],new_blocks=blocks,additional_blocks=blocks-base['blocks'],
        physical_tokens=blocks*64,six_margin_max_model_len=(blocks-6)*64,
        allocated_aligned_bytes=kv.aligned_bytes('fp4x',blocks),
        qualification='capacity ledger only; keep served cap98176 and original pool for performance A/B; increased metadata, graphs, memory admission and context quality need a separate later gate'))
if __name__=='__main__':print(json.dumps(ledger(),indent=2))
