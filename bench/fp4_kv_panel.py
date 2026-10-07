#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare identical public prompts and run two-boot context/needle panels.

Uses the pinned campaign retrieval generator/scorer. No fleet lifecycle,
SSH, locks or deployment; execution requires an explicit HTTP endpoint.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

PROBE_SHA = '24e943ae1fa565a0db64792af530372d868643923087e36744fade8927af8b66'
COMMON = (2048,16384,32768,63488)
NEW_MAX_PROMPT = 99200


def sha(data):return hashlib.sha256(data).hexdigest()


def probe(file):
    if sha(file.read_bytes())!=PROBE_SHA:raise ValueError('campaign retrieval source drift')
    spec=importlib.util.spec_from_file_location('long_probe',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def prepare(args,L):
    args.out.mkdir(parents=True,exist_ok=False)
    tok=L.LocalTokens(args.tokenizer_dir)
    panels=[]
    for seed in range(20260930,20260930+args.seeds):
        for target in (16384,63488,NEW_MAX_PROMPT):
            r=L.generate(target,seed,tok)
            assert L.oracle(r['request']['messages'][0]['content'])==r['expected']
            panels.append(r)
    (args.out/'needles.json').write_text(json.dumps(panels)+'\n')
    contexts=[]
    filler=tok.encode('Public archive: region north; status open; count twelve.\n').ids
    tasks={'prose':'Write a detailed explanation of how a town can design a useful public library. Use clear prose.',
           'code':'Write a Python LRU cache with a capacity limit, get and put methods, and a short usage example.'}
    marker='__KV_CONTEXT_5105__'
    for kind,task in tasks.items():
        rendered=tok.render(marker+'\n'+task)
        head,tail=rendered.split(marker)
        left,right=tok.encode(head).ids,tok.encode(tail).ids
        for target in (*COMMON,NEW_MAX_PROMPT):
            n=target-len(left)-len(right)
            ids=left+(filler*((n+len(filler)-1)//len(filler)))[:n]+right
            assert len(ids)==target
            contexts.append(dict(kind=kind,input_tokens=target,prompt=ids,
                                 prompt_sha256=sha(json.dumps(ids,separators=(',',':')).encode())))
    (args.out/'contexts.json').write_text(json.dumps(contexts)+'\n')
    (args.out/'manifest.json').write_text(json.dumps(dict(probe_sha256=PROBE_SHA,tokenizer=tok.pins,
        seeds=args.seeds,common_contexts=COMMON,new_max_prompt=NEW_MAX_PROMPT,
        needles_sha256=sha((args.out/'needles.json').read_bytes()),
        contexts_sha256=sha((args.out/'contexts.json').read_bytes())),indent=2)+'\n')
    print('PREPARED',len(panels),'needles;',len(contexts),'context fixtures')


def decode(args,fixtures):
    # Fixed 512 generated tokens, greedy, streaming raw event timestamps.
    # Capture cycle-ms/committed-token metrics separately from the worker
    # profiler; stream timing alone cannot isolate sub-millisecond changes.
    out=args.out/'decode.jsonl'
    with out.open('x') as f:
        for round_id in range(args.rounds+1):
            cells=list(fixtures)
            if round_id%2:cells.reverse()
            for cell in cells:
                if args.arm=='fp8' and cell['input_tokens']==NEW_MAX_PROMPT:continue
                body=dict(model=args.model,prompt=cell['prompt'],temperature=0,seed=0,
                          max_tokens=512,min_tokens=512,stream=True,stream_options={'include_usage':True})
                url=args.endpoint.rstrip('/')+'/v1/completions'
                req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
                start=time.monotonic();events=[];done=False
                with urllib.request.urlopen(req,timeout=900) as response:
                    for raw in response:
                        line=raw.decode().strip()
                        if line=='data: [DONE]':
                            done=True
                        elif line.startswith('data: '):
                            events.append(dict(seconds=time.monotonic()-start,chunk=json.loads(line[6:])))
                finishes=[c.get('finish_reason') for e in events for c in e['chunk'].get('choices',[]) if c.get('finish_reason')]
                assert done and finishes,'incomplete SSE protocol'
                texts=[e for e in events if any(c.get('text') for c in e['chunk'].get('choices',[]))]
                usage=next((e['chunk']['usage'] for e in reversed(events) if e['chunk'].get('usage')),None)
                assert texts and usage and usage['prompt_tokens']==cell['input_tokens']
                assert usage['completion_tokens']==512,'fixed-width decode truncated'
                elapsed=texts[-1]['seconds']-texts[0]['seconds']
                assert elapsed>0,'decode timing unavailable'
                row=dict(arm=args.arm,round=round_id,warmup=round_id==0,kind=cell['kind'],
                         input_tokens=cell['input_tokens'],prompt_sha256=cell['prompt_sha256'],
                         ttft_s=texts[0]['seconds'],decode_s=elapsed,
                         stream_decode_tps=511/elapsed,usage=usage,events=events,stream_done=done,finish_reasons=finishes)
                f.write(json.dumps(row)+'\n');f.flush()
                print(args.arm,round_id,cell['kind'],cell['input_tokens'],row['stream_decode_tps'],flush=True)


def needles(args,L,fixtures):
    runs=[]
    for receipt in fixtures:
        target=receipt['target_input_tokens']
        if args.arm=='fp8' and target==NEW_MAX_PROMPT:continue
        for repeat in (1,2):
            receipt['request']['model']=args.model
            run=L.execute(args.endpoint,dict(receipt,repeat=repeat))
            if run.get('server_minus_local_prompt_tokens')!=0:
                run['error']=run.get('error') or 'server/local prompt token count mismatch or missing usage'
            run['seed']=receipt['seed'];runs.append(run)
            path=args.out/f'needle-{receipt["seed"]}-{target}-{repeat}.json'
            with path.open('x') as f:json.dump(run,f)
    # Apply the same 16K versus long rule independently at 63K and new max.
    comparisons=[]
    for seed in sorted({r['seed'] for r in runs}):
        for target in ((63488,) if args.arm=='fp8' else (63488,NEW_MAX_PROMPT)):
            group=[r for r in runs if r['seed']==seed and r['target_input_tokens'] in (16384,target)]
            L.TARGETS=(16384,target)
            verdict=L.verdict(group)
            comparisons.append(dict(seed=seed,target=target,verdict=verdict))
    (args.out/'needles-verdict.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    if any(r['verdict']['status']!='PASS' for r in comparisons):raise RuntimeError('needle panel failed')
    print('NEEDLES PASS',len(comparisons),'seed/length comparisons')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('prepare','decode','needles'))
    ap.add_argument('--long-probe',type=Path,required=True)
    ap.add_argument('--tokenizer-dir',type=Path)
    ap.add_argument('--fixtures',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--endpoint')
    ap.add_argument('--arm',choices=('fp8','fp4x'))
    ap.add_argument('--seeds',type=int,default=20)
    ap.add_argument('--rounds',type=int,default=7)
    ap.add_argument('--model',default='GLM-5.3')
    a=ap.parse_args()
    if a.seeds<1 or a.rounds<1:ap.error('positive seeds/rounds required')
    L=probe(a.long_probe)
    if a.mode=='prepare':
        if a.endpoint or not a.tokenizer_dir:ap.error('prepare requires local tokenizer and no endpoint')
        prepare(a,L)
    else:
        if not a.endpoint or not a.arm or not a.fixtures:ap.error('execution requires endpoint, arm, fixtures')
        manifest=json.loads((a.fixtures/'manifest.json').read_text())
        file=a.fixtures/('contexts.json' if a.mode=='decode' else 'needles.json')
        key='contexts_sha256' if a.mode=='decode' else 'needles_sha256'
        if sha(file.read_bytes())!=manifest[key]:raise ValueError('fixture hash mismatch')
        a.out.mkdir(parents=True,exist_ok=False)
        cells=json.loads(file.read_text())
        if a.mode=='decode':decode(a,cells)
        else:needles(a,L,cells)
