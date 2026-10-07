#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline launch/byte ledger and explicitly conditional cold-TTFT forecast.

No GPU timings are inferred from the launch count alone. Fit the excess over
owner FP8 anchors to per-chunk and per-score-call costs. The unchanged-reader
share is not identifiable from aggregate TTFT; expose it as a sensitivity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'overlay/bringup'))
from glm_fp4_prefill import geometry, scratch_bytes, LOGITS_BYTES

CHUNK = 2048
INDEX_LAYERS = 22  # conservative target+MTP upper count
STOCK_BUDGET = 512 << 20


def chunk_counts(prefix, queries=CHUNK):
    max_q = STOCK_BUDGET // (4 * prefix)
    qt, kt = geometry(prefix)
    splits=[min(max_q,queries-off) for off in range(0,queries,max_q)]
    old_q=sum(math.ceil(n/64) for n in splits)
    old_calls=old_q*math.ceil(prefix/4096)
    new_calls=sum(math.ceil(n/qt) for n in splits)
    # Old: 8 fills/copies per Q tile, finish=1; per K tile gather,
    # bounds, score, logits copy, IDs add, topk>=1, IDs gather, 2 copies.
    # Excludes clean_logits=True initialization, extra torch.topk internal
    # kernels and the common output-buffer clear/quant/writer launches.
    return dict(prefix=prefix,queries=queries,metadata_splits=splits,
                geometry=[qt,kt],old_q_tiles=old_q,old_score_calls=old_calls,
                old_candidate_merges=old_calls,old_launches_min=9*(old_calls+old_q),
                stock_score_calls=len(splits),stock_launches=1+2*len(splits),
                new_score_calls=new_calls,new_candidate_merges=0,
                new_launches=len(splits)+3*new_calls,
                # Full score writes upper bound (causal scheduler skips).
                old_score_matrix_bytes=old_calls*64*4096*4,
                stock_score_matrix_bytes=queries*prefix*4,
                new_logits_allocation_bytes=LOGITS_BYTES,
                stock_selector_min_traffic_bytes=queries*prefix*4+queries*2048*4,
                new_selector_min_traffic_bytes=queries*prefix*4+queries*2048*4,
                old_gather_cache_read_bytes=old_calls*4096*132,
                stock_gather_cache_read_bytes=prefix*132,
                new_gather_cache_read_bytes=len(splits)*prefix*132,
                new_gather_bank_write_bytes=len(splits)*kt*132,
                # One logical pass; PyTorch topk can scan more than once.
                old_merge_min_traffic_bytes=old_calls*merge_tile_bytes())


def merge_tile_bytes():
    q,k,t=64,4096,2048
    return (2*q*k*4 + q*k*8                 # logits copy + new IDs write
            + q*(k+t)*4 + q*t*(4+8)       # candidate read + topk outputs
            + q*t*8*3                     # gather selection/IDs/read/write
            + 2*q*t*(4+8))                # retained values/IDs copies


def prompt_counts(tokens):
    # First <=2048 fresh tokens use the retained dense-MHA index bypass.
    rows=[chunk_counts(off+min(CHUNK,tokens-off),min(CHUNK,tokens-off))
          for off in range(CHUNK,tokens,CHUNK)]
    keys=('old_score_calls','old_q_tiles','old_launches_min','new_score_calls','new_launches')
    result={k:sum(r[k] for r in rows)*INDEX_LAYERS for k in keys}
    result['serving_chunks']=math.ceil(tokens/CHUNK)
    return result


def forecast(ladder, reader_share=.05):
    records=[json.loads(line) for line in Path(ladder).read_text().splitlines() if line.startswith('{')]
    # Quadratic interpolation through owner anchors, seconds per exact prompt
    # token. 98K is extrapolated; no FP8 98K measurement exists in this input.
    n16,n64=16336,65508
    b=(82/n64-20/n16)/(n64-n16);a=20/n16-b*n16
    def fp8(n):return a*n+b*n*n
    counts=[prompt_counts(r['prompt_tokens']) for r in records]
    x=[(c['serving_chunks'],c['old_score_calls']) for c in counts]
    y=[r['ttft_s']-fp8(r['prompt_tokens']) for r in records]
    xx=sum(v[0]**2 for v in x);zz=sum(v[1]**2 for v in x);xz=sum(v[0]*v[1] for v in x)
    xy=sum(v[0]*w for v,w in zip(x,y));zy=sum(v[1]*w for v,w in zip(x,y));det=xx*zz-xz*xz
    alpha=(xy*zz-zy*xz)/det;beta=(zy*xx-xy*xz)/det
    predictions=[]
    for r,c in zip(records,counts):
        def predict(share):
            # Per-chunk term includes old Q64 preparation/selection overhead
            # and possibly packed MLA/writer cost. Reduce only its launch
            # component by the change in query slice count; retain reader share.
            ratio=c['new_score_calls']/c['old_q_tiles']
            return fp8(r['prompt_tokens'])+alpha*c['serving_chunks']*(share+(1-share)*ratio)+beta*c['new_score_calls']
        predictions.append(dict(prompt_tokens=r['prompt_tokens'],observed_fix3_ttft_s=r['ttft_s'],
            fp8_anchor_or_extrapolation_s=fp8(r['prompt_tokens']),predicted_fix4_ttft_s=predict(reader_share),
            reader_share_sensitivity_s={str(f):predict(f) for f in (0,.05,.2,1)},
            fit_residual_s=r['ttft_s']-fp8(r['prompt_tokens'])-alpha*c['serving_chunks']-beta*c['old_score_calls'],
            prompt_counts=c,last_chunk=chunk_counts(r['prompt_tokens'])))
    return dict(scope='OFFLINE COST MODEL; predicted, not measured speed or GPU exactness',
        ladder_sha256=hashlib.sha256(Path(ladder).read_bytes()).hexdigest(),index_layer_upper_count=INDEX_LAYERS,
        serving_chunk=CHUNK,fp8_model=dict(linear_s_per_token=a,quadratic_s_per_token2=b,owner_anchors_s=[20,82]),
        fitted_excess=dict(s_per_serving_chunk=alpha,s_per_old_score_merge_call=beta),
        assumed_unchanged_reader_share=reader_share,
        caveats=['Aggregate TTFT cannot identify the packed-reader share; central 5% is an explicit hypothesis',
                 '98K FP8 anchor is extrapolated, not measured; compiler cold starts excluded from forecast',
                 'Stock selector is identical; score-bit and cutoff-tie GPU fixture must still pass',
                 'GPU launches are a lower bound for Fix3, not profiler measurements'],
        scratch_bytes=scratch_bytes(),constructor_bank_bytes=scratch_bytes()-LOGITS_BYTES,
        returned_logits_bytes=LOGITS_BYTES,old_merge_min_bytes_per_q64_k4096_tile=merge_tile_bytes(),
        additional_scratch_vs_fix3_bytes=scratch_bytes()-(12648960+(1<<20)),predictions=predictions)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ladder',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reader-share',type=float,default=.05)
    args=p.parse_args()
    if not 0<=args.reader_share<=1:p.error('reader-share must be in [0,1]')
    result=forecast(args.ladder,args.reader_share);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scratch_bytes','constructor_bank_bytes','returned_logits_bytes','fitted_excess')}))
    for row in result['predictions']:
        print(row['prompt_tokens'],round(row['predicted_fix4_ttft_s'],2),row['reader_share_sensitivity_s'])
