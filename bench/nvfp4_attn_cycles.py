#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Later exclusive-c1 cycle gate: request-bound metrics, 20 matched runs per kind."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import urllib.request
from gate_metrics import GateMetrics,paired

PROMPTS={
 'prose':'Write a detailed practical guide to maintaining a large vegetable garden through all four seasons. Continue with concrete examples until the token limit.',
 'code':'Write a complete Python module implementing a persistent task queue with SQLite, retries, leases, transactions and a command line interface. Continue implementing until the token limit.'}


def run(base,metrics,out,arm):
    gate=GateMetrics(metrics,out,metadata=dict(arm=arm,k_mode='served-kstop',target_M='adaptive'))
    for kind in ('prose','code'):
        for i in range(-1,20):
            pair_key=f'{kind}-{i}'
            prompt=f'Trial {pair_key}:\n'+PROMPTS[kind]
            body=dict(model='GLM-5.3',messages=[dict(role='user',content=prompt)],
                temperature=0,seed=5300+i,max_tokens=256,stream=True,stream_options=dict(include_usage=True),
                cache_salt='nvfp4-attn-'+pair_key,chat_template_kwargs=dict(reasoning_effort='low'))
            digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
            req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with gate.request(f'{arm}-{kind}-{i+1}',exclusive=True,
                 metadata=dict(pair_key=pair_key,prompt_sha256=digest,seed=5300+i,kind=kind,warmup=i<0)) as probe:
                done=False;usage=None
                with urllib.request.urlopen(req,timeout=180) as response:
                    for raw in response:
                        line=raw.decode().strip()
                        if not line.startswith('data:'):continue
                        data=line[5:].strip()
                        if data=='[DONE]':done=True;break
                        chunk=json.loads(data)
                        if chunk.get('error'):raise RuntimeError(str(chunk['error']))
                        if chunk.get('usage'):usage=chunk['usage']
                if not done or not usage or usage.get('completion_tokens',0)<128:
                    raise RuntimeError('incomplete or too-short streaming cycle probe')
                # No token() call: gate_metrics uses the server request decode time sum.
            print(json.dumps(probe.record),flush=True)


def compare(a,b,out):
    def rows(folder):
        records=[json.loads(l) for l in (Path(folder)/'requests.jsonl').read_text().splitlines()]
        records=[r for r in records if not r['metadata']['warmup']]
        keys=[r['metadata']['pair_key'] for r in records]
        expected={f'{k}-{i}' for k in PROMPTS for i in range(20)}
        if len(keys)!=40 or set(keys)!=expected:raise ValueError('need exactly 20 valid runs for each prose/code kind')
        return {r['metadata']['pair_key']:r for r in records}
    aa,bb=rows(a),rows(b);result={}
    for kind in PROMPTS:
        pairs=[paired(aa[f'{kind}-{i}'],bb[f'{kind}-{i}']) for i in range(20)]
        cycle=statistics.median(p['cycle_ratio'] for p in pairs)
        committed=sum(p['B_committed_per_cycle'] for p in pairs)/sum(p['A_committed_per_cycle'] for p in pairs)
        result[kind]=dict(pairs=pairs,cycle_ratio_median=cycle,committed_ratio_pooled=committed,
                         passed=cycle<=.97 and committed>=.99)
    result['passed']=all(result[k]['passed'] for k in PROMPTS)
    result['metric_kind']='request decode wall / draft opportunities; GPU-event cycle measurement remains separate'
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit('cycle/committed gate FAILED')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    r=s.add_parser('run');r.add_argument('--base',required=True);r.add_argument('--metrics',required=True)
    r.add_argument('--out',required=True);r.add_argument('--arm',required=True,choices=('A','B'))
    r=s.add_parser('compare');r.add_argument('normal');r.add_argument('candidate');r.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='run':run(a.base,a.metrics,a.out,a.arm)
    else:compare(a.normal,a.candidate,a.out)
