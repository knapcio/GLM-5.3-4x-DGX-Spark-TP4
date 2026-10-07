#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix 5 offline ledger; aggregate data constrain, but cannot profile, stages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/bringup'))
from glm_fp4_prefill import MLA_ROWS, scratch_bytes, WORKSPACE_BYTES

LAYERS=79
CHUNK=2048
TOPK=2048


def fp8(n):
    # Same exact-token owner anchors as Fix 4; 98K remains extrapolation.
    n16,n64=16336,65508
    b=(82/n64-20/n16)/(n64-n16);a=20/n16-b*n16
    return a*n+b*n*n


def records(path):
    return [r for line in path.read_text().splitlines() if line.startswith('{')
            if isinstance((r:=json.loads(line)).get('prompt_tokens'),int)
            and isinstance(r.get('ttft_s'),(int,float))]


def linear_fit(rows):
    xs=[r['prompt_tokens'] for r in rows];ys=[r['ttft_s'] for r in rows]
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    offset=ym-slope*xm
    return dict(slope_ms_per_token=slope*1000,intercept_s=offset,
                residuals_s=[y-offset-slope*x for x,y in zip(xs,ys)])


def model(old,new,reader_ms=.55,bf16_ms=.08,bandwidth_gbs=100,launch_us=5):
    preds=[]
    for r in records(new):
        n=r['prompt_tokens'];active=max(n-CHUNK,0)
        # Conservatively assume every prefix row belongs to the union.
        prefixes=[min(end,n) for end in range(2*CHUNK,n+CHUNK,CHUNK)]
        expand_bytes=LAYERS*sum(prefixes)*(368+1152)
        # Mark traffic includes ID loads and worst-case uncached 4B RMW pairs.
        mark_bytes=LAYERS*active*TOPK*12
        prepare_bytes=expand_bytes+mark_bytes+LAYERS*len(prefixes)*MLA_ROWS*4
        prepare_s=prepare_bytes/(bandwidth_gbs*1e9)+LAYERS*len(prefixes)*3*launch_us*1e-6
        # Reader penalty is a stated stage hypothesis, NOT an aggregate fit.
        # Preserve all observed residual, including startup/DSA/writer effects,
        # except the attributed .55 ms per active sparse token. BF16's extra
        # DRAM debit is .08 ms/token = 16% miss fraction at 200 GB/s.
        saving=active*reader_ms/1000
        debit=active*bf16_ms/1000
        preds.append(dict(prompt_tokens=n,observed_fix4_ttft_s=r['ttft_s'],fp8_anchor_or_extrapolation_s=fp8(n),
                          attributed_reader_s=saving,preserved_residual_s=r['ttft_s']-fp8(n)-saving,
                          bf16_traffic_debit_s=debit,union_prepare_s=prepare_s,
                          predicted_fix5_ttft_s=r['ttft_s']-saving+debit+prepare_s,
                          traffic_only_pessimistic_ttft_s=r['ttft_s']-saving+active*.466/1000+prepare_s,
                          union_expansion_logical_bytes=expand_bytes,mark_logical_bytes=mark_bytes))
    row=dict(selected_rows_per_query=TOPK,layers=LAYERS,
             packed_logical_read_bytes_per_token=LAYERS*TOPK*368,
             fp8_logical_read_bytes_per_token=LAYERS*TOPK*576,
             bf16_logical_read_bytes_per_token=LAYERS*TOPK*1152,
             latent_values_reconstructed_per_token=LAYERS*TOPK*512,
             ordered_scale_multiplies_per_token=LAYERS*TOPK*512*2,
             qk_pv_flops_per_token=LAYERS*TOPK*16*2*(576+512),
             writer_output_bytes_per_token=LAYERS*368,
             writer_latent_values_per_token=LAYERS*512)
    return dict(scope='OFFLINE SOURCE AND AGGREGATE MODEL; no stage timing measured',
                inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old,new)},
                fix3=records(old),fix4=records(new),ledger=row,
                measured_linear_fits=dict(fix3=linear_fit(records(old)),fix4=linear_fit(records(new))),
                scratch_bytes=scratch_bytes(),workspace_bytes=WORKSPACE_BYTES,
                additional_scratch_vs_fix4_bytes=scratch_bytes()-359153664,
                physical_rows=MLA_ROWS,assumed_reader_penalty_ms_per_active_token=reader_ms,
                assumed_bf16_debit_ms_per_active_token=bf16_ms,
                assumed_union_bandwidth_GBs=bandwidth_gbs,assumed_prepare_launch_us=launch_us,
                predictions=preds,
                refutation=['At Q2048, warm per-layer packed minus union latency below 10 ms on the slowest rank at 16K/64K/98K refutes the central savings (expected about 12 ms).',
                            'Measured preparation above the modeled per-chunk bytes/bandwidth + 15 us/layer refutes the union debit.',
                            'Matched cold TTFT above 30/101/157 s at the exact 16K/64K/98K ladder prompts refutes the central forecast (>10% margin).',
                            'If reader savings summed over layers explain <50% of the excess, reader dominance is refuted; profile writer, indexer, dense first chunk and host gaps.',
                            'BF16 miss fraction near 100% instead of 16% makes the unchanged plain reader bandwidth-bound; pessimistic forecast is supplied.'],
                alternatives={'a':'Chosen: mark union and expand into BF16 once, alias DSA arena; exact values, unchanged release plain reader.',
                              'b':'H16 reuse already exists; vector unpack still repeats reconstruction across queries. A query-sharing reader requires a new attention layout/compiler qualification.',
                              'c':'Original fresh BF16 differs from FP4 QDQ; bypass would change attention. Exact fresh QDQ can avoid only current-chunk selections, declining as chunk/prefix.'},
                attribution='Repeated packed reconstruction is the dominant source/model hypothesis. Aggregate TTFT cannot establish exclusive causality. MTP is one of 79 layers, not an extra 78-layer multiplier; no added per-chunk host sync exists in the audited paths.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-ladder',type=Path,required=True);p.add_argument('--new-ladder',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=model(a.old_ladder,a.new_ladder);a.out.write_text(json.dumps(d,indent=2)+'\n')
    print('scratch MiB',d['scratch_bytes']/2**20)
    for r in d['predictions']:print(r['prompt_tokens'],round(r['predicted_fix5_ttft_s'],2))
