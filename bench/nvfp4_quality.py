#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-window probe helpers. Network only for explicit freeze/collect/needles/dash commands.

Frozen next-token drift is |log p_candidate(t|prompt)-log p_normal(t|prompt)|
for the normal boot's chosen token, scored by teacher forcing on identical IDs.
It is neither greedy-text equality nor truncated-support KL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

PROMPTS = [
 ('prose-en','Explain why a cache can make a repeated request faster, in two sentences.'),
 ('prose-pl','Wyjaśnij krótko, dlaczego zima jest chłodniejsza od lata.'),
 ('code-python','Write a Python function that returns the first non-repeated character in a string.\n```python\ndef first_unique(s):\n'),
 ('code-sql','Write SQL to find customers with no orders. Tables: customers(id), orders(id, customer_id).\nSELECT'),
 ('math','Compute 17 * 19. The result is'),
 ('reason','All blue tiles are square. Some square tiles are heavy. Does it follow that any blue tile is heavy? Explain.'),
 ('json','Return only JSON with keys name and count. Name is probe and count is 3.\n'),
 ('long-prose','Public record: consistency matters more than changing terminology.\n'*256+'Summarize the record in one sentence.'),
 ('long-code','def increment(x):\n    return x + 1\n\n'*256+'# Write a unit test for increment\n'),
 ('logic','A box contains 3 red balls and 2 blue balls. After removing one red ball, the chance the next ball is blue is'),
 ('needle-short','The access code is KZQHWNPT. Ignore a decoy code BCDRSTUV. Repeat the access code only:'),
 ('unicode','Uzupełnij po polsku: Zażółć gęślą'),
]


def request(base,path,body=None):
    req=urllib.request.Request(base.rstrip('/')+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1200) as r:return json.load(r)


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze(base,out):
    panel=[]
    for name,text in PROMPTS:
        ids=request(base,'/tokenize',dict(model='GLM-5.3',prompt=text))['tokens']
        r=request(base,'/v1/completions',dict(model='GLM-5.3',prompt=ids,max_tokens=1,temperature=0,logprobs=1,return_token_ids=True,seed=53))
        token_ids=r['choices'][0].get('token_ids')
        if not token_ids or len(token_ids)!=1:
            raise ValueError('generated token IDs unavailable; freeze requires return_token_ids')
        panel.append(dict(id=name,text=text,tokens=ids+[token_ids[0]],normal_next_token=token_ids[0]))
    save(out,panel)


def collect(base,panel_path,out):
    panel=json.loads(Path(panel_path).read_text());rows=[]
    for item in panel:
        ids=item['tokens']
        r=request(base,'/v1/completions',dict(model='GLM-5.3',prompt=ids,max_tokens=1,temperature=0,prompt_logprobs=0,return_token_ids=True,cache_salt='nvfp4-'+item['id']))
        ch=r['choices'][0];lp=ch.get('prompt_logprobs') or r.get('prompt_logprobs')
        if ch.get('prompt_token_ids')!=ids or not lp or len(lp)!=len(ids):raise ValueError('teacher-forced ID/logprob mismatch')
        obj=lp[-1].get(str(ids[-1]),lp[-1].get(ids[-1]))
        value=obj['logprob'] if isinstance(obj,dict) else obj
        if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('missing actual-token logprob')
        rows.append(dict(id=item['id'],tokens=ids,logprob=value));save(out,dict(panel_sha256=hashlib.sha256(Path(panel_path).read_bytes()).hexdigest(),rows=rows))


def drift(a,b):
    if a['panel_sha256']!=b['panel_sha256'] or len(a['rows'])!=len(PROMPTS) or len(b['rows'])!=len(PROMPTS):raise ValueError('incomplete or different panels')
    rows=[]
    for p,q in zip(a['rows'],b['rows']):
        if p['id']!=q['id'] or p['tokens']!=q['tokens']:raise ValueError('frozen ID mismatch')
        d=abs(q['logprob']-p['logprob'])
        if not math.isfinite(d):raise ValueError('nonfinite drift')
        rows.append(dict(id=p['id'],absolute_nats=d))
    return dict(rows=rows,max_nats=max(r['absolute_nats'] for r in rows),passed=all(r['absolute_nats']<=.3 for r in rows))


def qgate(a,b):
    def counts(folder):
        runs=[json.loads(p.read_text()) for p in sorted(Path(folder).glob('qeval-*.json'))]
        if len(runs)!=3:raise ValueError('exactly three qeval runs required')
        ids=None;out=[];primary=[]
        for run in runs:
            rs=run['results'];grid=[(r['id'],r['category']) for r in rs]
            if len(rs)!=75 or len(set(x[0] for x in grid))!=75 or (ids is not None and ids!=grid) or any(r.get('why','').startswith('request failed:') for r in rs):raise ValueError('incomplete/failed qeval')
            ids=grid;out.append(sum(bool(r['pass']) for r in rs));primary.append(sum(bool(r['pass']) for r in rs if r['category'] in ('code','reason','math')))
        return ids,out,primary
    ia,ca,pa=counts(a);ib,cb,pb=counts(b)
    if ia!=ib:raise ValueError('qeval grid mismatch')
    drop=statistics.mean(ca)-statistics.mean(cb)
    return dict(normal=ca,candidate=cb,mean_drop_tasks=drop,primary_normal=pa,primary_candidate=pb,passed=drop<=.7)


def needles(base,out,lengths=(16384,63488,96000)):
    rows=[]
    # 96,000 fits the served 98,176 cap with output/lookahead room; 98,304 does not.
    filler=request(base,'/tokenize',dict(model='GLM-5.3',prompt='Public unrelated note: the decoy access code is ABCDEFGHJKLM.\n'*4096))['tokens']
    if not filler or not all(type(t) is int for t in filler):raise ValueError('invalid filler token IDs')
    for length in lengths:
        padding=(filler*((length+len(filler)-1)//len(filler)))[:length]
        for seed in range(20):
            rng=random.Random(5300+seed);code=''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ',k=12))
            marker=request(base,'/tokenize',dict(model='GLM-5.3',prompt='\nThe unique authoritative access code is '+code+'.\n'))['tokens']
            question=request(base,'/tokenize',dict(model='GLM-5.3',prompt='\nReturn only the unique authoritative access code, ignoring all decoy codes.\nAnswer:'))['tokens']
            remaining=length-len(marker)-len(question)
            frac=(.02,.25,.5,.75,.98)[seed%5];pos=int(remaining*frac)
            ids=padding[:pos]+marker+padding[pos:remaining]+question
            if len(ids)!=length:raise ValueError('needle tokenizer length mismatch')
            r=request(base,'/v1/completions',dict(model='GLM-5.3',prompt=ids,max_tokens=64,temperature=0,seed=seed,cache_salt='nvfp4-needle-'+str(length)+'-'+str(seed)))
            ch=r['choices'][0];text=ch['text'].strip().rstrip('.')
            row=dict(length=length,seed=seed,depth=frac,expected=code,content=text,finish_reason=ch['finish_reason'],passed=text==code and ch['finish_reason']=='stop',tokens_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
            rows.append(row);save(out,rows)
            if not row['passed']:raise RuntimeError('needle failed: '+str(row))


def ngate(a,b,lengths=(16384,63488,96000)):
    grid=[(length,seed) for length in lengths for seed in range(20)]
    for rows in (a,b):
        if [(r['length'],r['seed']) for r in rows]!=grid:raise ValueError('incomplete needle grid')
        if any(not r['passed'] or r['finish_reason']!='stop' or r['content']!=r['expected'] for r in rows):
            return dict(passed=False,reason='at least one needle failed')
    if any(p['tokens_sha256']!=q['tokens_sha256'] or p['expected']!=q['expected'] for p,q in zip(a,b)):
        raise ValueError('needle fixture/tokenizer mismatch')
    return dict(passed=True,cases=len(grid),lengths=list(lengths))


def dash(api,out):
    rows=[]
    for phase in ('warmup','scored1','scored2','scored3','scored4','scored5'):
        for kind in ('prose','code'):
            config=dict(port=8095,modelId='GLM-5.3',concurrencies=[1],maxTokens=256,promptType=kind)
            if request(api,'/bench').get('active'):raise RuntimeError('sparkDash busy')
            posted=request(api,'/bench',config);started=time.monotonic()
            while True:
                state=request(api,'/bench')
                if not state.get('active'):break
                if time.monotonic()-started>300:raise TimeoutError('sparkDash timeout')
                time.sleep(2)
            job=state['last']
            if job.get('benchId')!=posted.get('benchId') or job.get('status')!='completed':raise RuntimeError('stale/failed sparkDash job')
            cell=job['results'][0]
            if cell.get('streamsOk')!=1 or cell.get('streamsFailed',0) or cell.get('error'):raise RuntimeError('sparkDash stream failed')
            rows.append(dict(phase=phase,kind=kind,job=job));save(out,rows)
            time.sleep(3.1)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    for cmd in ('freeze','collect','needles','dash'):
        p=sub.add_parser(cmd);p.add_argument('--base',required=True);p.add_argument('--out',required=True)
        if cmd=='collect':p.add_argument('--panel',required=True)
    for cmd in ('drift','qgate','ngate'):
        p=sub.add_parser(cmd);p.add_argument('normal');p.add_argument('candidate');p.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.cmd=='freeze':freeze(a.base,a.out)
    elif a.cmd=='collect':collect(a.base,a.panel,a.out)
    elif a.cmd=='needles':needles(a.base,a.out)
    elif a.cmd=='dash':dash(a.base,a.out)
    else:
        if a.cmd=='qgate':result=qgate(a.normal,a.candidate)
        else:
            fn=drift if a.cmd=='drift' else ngate
            result=fn(json.loads(Path(a.normal).read_text()),json.loads(Path(a.candidate).read_text()))
        save(a.out,result)
        if not result['passed']:sys.exit(1)

if __name__=='__main__':main()
