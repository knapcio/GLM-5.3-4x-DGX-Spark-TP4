#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict incremental REAL-path cycle/qeval gates; English prose and code only."""
import argparse
import json
from pathlib import Path
import statistics
from gate_metrics import paired
from nvfp4_quality import needles,ngate,dash


def cycle_gate(a,b):
    def read(path):
        rows=[json.loads(l) for l in Path(path).read_text().splitlines()]
        selected={}
        for row in rows:
            if row.get('status')!='OK':continue
            key=row.get('metadata',{}).get('pair_key')
            if key in selected:raise ValueError('duplicate valid cycle pair')
            selected[key]=row
        expected={f'{kind}-{i}' for kind in ('prose','code') for i in range(10)}
        if set(selected)!=expected:raise ValueError('need ten paired valid runs per English/code kind')
        return selected
    aa,bb=read(a),read(b);results={}
    for kind in ('prose','code'):
        ps=[paired(aa[f'{kind}-{i}'],bb[f'{kind}-{i}']) for i in range(10)]
        ratio=statistics.median(p['cycle_ratio'] for p in ps)
        commit=sum(p['B_committed_per_cycle'] for p in ps)/sum(p['A_committed_per_cycle'] for p in ps)
        results[kind]=dict(pairs=ps,cycle_ratio_median=ratio,cycle_gain_pct=100*(1-ratio),
            committed_ratio_pooled=commit,passed=ratio<.99 and commit>.99)
    return dict(passed=all(r['passed'] for r in results.values()),kinds=results,
        gate='strict cycle gain >1%, committed/cycle drop <1%; preceding accepted arm, plus compare with B-current',
        metric='request decode wall/draft opportunities; not a GPU-event cycle timer')


def qeval_gate(folder):
    files=sorted(Path(folder).glob('qeval-*.json'))
    if len(files)!=3:raise ValueError('exactly three qeval runs required')
    ids=None;scores=[];primary=[]
    for file in files:
        run=json.loads(file.read_text());rs=run['results'];grid=[(r['id'],r['category']) for r in rs]
        if len(grid)!=75 or len(set(k[0] for k in grid))!=75 or (ids is not None and ids!=grid):
            raise ValueError('incomplete/mismatched qeval grid')
        if any(type(r['pass']) is not bool or r.get('why','').startswith('request failed:') for r in rs):
            raise ValueError('invalid/transport-failed qeval row')
        if run.get('concurrency')!=1:raise ValueError('qeval must be c1')
        ids=grid;scores.append(sum(r['pass'] for r in rs))
        primary.append(sum(r['pass'] for r in rs if r['category'] in ('code','reason','math')))
    mean=statistics.mean(scores)
    return dict(passed=mean>=70.553,scores=scores,mean_passed_tasks=mean,primary=primary,threshold=70.553,
                units='number passed out of75, mean across three runs; not percent')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('cycle');q.add_argument('baseline');q.add_argument('candidate');q.add_argument('--out',required=True)
    q=s.add_parser('qeval');q.add_argument('folder');q.add_argument('--out',required=True)
    q=s.add_parser('needles');q.add_argument('--base',required=True);q.add_argument('--out',required=True);q.add_argument('--max-input',type=int,default=98048)
    q=s.add_parser('ngate');q.add_argument('baseline');q.add_argument('candidate');q.add_argument('--out',required=True);q.add_argument('--max-input',type=int,default=98048)
    q=s.add_parser('dash');q.add_argument('--base',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.cmd=='needles':
        needles(a.base,a.out,lengths=(16384,63488,a.max_input));raise SystemExit(0)
    if a.cmd=='dash':dash(a.base,a.out);raise SystemExit(0)
    if a.cmd=='cycle':result=cycle_gate(a.baseline,a.candidate)
    elif a.cmd=='qeval':result=qeval_gate(a.folder)
    else:result=ngate(json.loads(Path(a.baseline).read_text()),json.loads(Path(a.candidate).read_text()),lengths=(16384,63488,a.max_input))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit('REAL-path qualification gate FAILED')
