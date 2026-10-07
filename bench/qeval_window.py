#!/usr/bin/env python3
"""WINDOW ONLY qeval x3 using the Flash fixed checker, model alias only adapted.
Preserves full raw response (including reasoning) beside original scoring.
A hard local deadline terminates the explicit local child, never a fleet process.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import os
import sys
import signal
import time
import urllib.request


def worker(args):
    bench=args.bench.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    src=(bench/'qeval.py').read_text()
    anchor='"model": "GLM-5.3"'
    if src.count(anchor)!=1:raise ValueError('qeval model anchor drift')
    src=src.replace(anchor,'"model": '+json.dumps(args.model))
    sys.path.insert(0,str(bench))
    ns={'__name__':'window_qeval','__file__':str(bench/'qeval.py')};exec(compile(src,str(bench/'qeval.py'),'exec'),ns)
    meta=json.loads(args.meta.read_text())
    checker_sha=hashlib.sha256((bench/'qeval_tasks.py').read_bytes()).hexdigest()
    source_sha=hashlib.sha256((bench/'qeval.py').read_bytes()).hexdigest()
    original=urllib.request.urlopen
    class Response:
        def __init__(self,response,body):self.response=response;self.body=body
        def read(self,*args,**kwargs):
            raw=self.response.read(*args,**kwargs)
            with (out/'raw-responses.jsonl').open('a') as f:
                f.write(json.dumps(dict(request=self.body,response=json.loads(raw)))+'\n')
            return raw
        def __getattr__(self,k):return getattr(self.response,k)
    def open_record(request,*a,**kw):
        return Response(original(request,*a,**kw),json.loads(request.data))
    urllib.request.urlopen=open_record
    os.chdir(out)
    saved_stdout,saved_stderr=sys.stdout,sys.stderr
    with (out/'run.log').open('w') as log:
        sys.stdout=sys.stderr=log
        for repeat in range(1,args.repeats+1):
            label=f'{args.label}-q{repeat}'
            ns['run'](label,args.url,args.timeout,1,[],0)
            p=out/f'qeval-{label}.json';d=json.loads(p.read_text())
            d.update(meta=meta,checker_sha256=checker_sha,qeval_source_sha256=source_sha,model=args.model)
            p.write_text(json.dumps(d,indent=2)+'\n')
    sys.stdout,sys.stderr=saved_stdout,saved_stderr
    urllib.request.urlopen=original


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fleet-window',action='store_true');ap.add_argument('--url',required=True);ap.add_argument('--model',default='GLM-5.3');ap.add_argument('--label',required=True)
    ap.add_argument('--meta',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bench',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--budget-s',type=float,default=850);ap.add_argument('--timeout',type=float,default=45)
    ap.add_argument('--repeats',type=int,default=3,choices=(1,2,3))  # default keeps the x3 gate
    a=ap.parse_args()
    if not a.fleet_window:ap.error('--fleet-window required; inference is forbidden for an offline job')
    m=json.loads(a.meta.read_text())
    if not m.get('fleet_owner'):ap.error('ownership receipt required')
    p=mp.Process(target=worker,args=(a,));p.start();deadline=time.monotonic()+a.budget_s
    def interrupted(signum,frame):
        if p.is_alive():
            p.terminate();p.join(timeout=5)
            if p.is_alive():p.kill();p.join()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    while p.is_alive() and time.monotonic()<deadline:p.join(timeout=min(1,max(.01,deadline-time.monotonic())))
    if p.is_alive():
        p.terminate();p.join();a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'deadline.json').write_text(json.dumps(dict(status='INCOMPLETE',reason='qeval x3 exceeded budget; no qualification'))+'\n');return 2
    return p.exitcode

if __name__=='__main__':raise SystemExit(main())
