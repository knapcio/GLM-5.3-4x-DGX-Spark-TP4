#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One fresh clone / one boot / one KV-head step. Default is offline planning.

prepare is local-only (DRY launcher); run --execute is allowed only on rank0
under its local systemd user manager. No automatic deployment or power recovery.
Literal c4 at per-request max is refused unless its resident KV fits. The
explicit pool-quarter screen is useful evidence, but cannot qualify literal c4.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from fp4_kv_layout import layout, ORDINARY_BYTES
from fp4_kv_admission import forecast

ROOT = Path(__file__).resolve().parents[1]
GiB = 1 << 30
BASE_MAX = 98176
OUTPUT = 1024
# Exactly the routine RM allocation line handled by its caller (dispram.sh fix6, DISPRAM_ALLOW_RM_ALLOC_OOM=1,
# set by every recipe here): bursts at every engine boot. Counted, never fatal. Everything else stays fatal.
KERNEL_ROUTINE = r'^NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory \[NV_ERR_NO_MEMORY\] \(0x00000051\) returned from _memdescAllocInternal\(pMemDesc\) @ mem_desc\.c:1359$'
KERNEL_FATAL = r'oom-kill|out of memory|killed process|hung.task|blocked for more than|NV_ERR_NO_MEMORY|soft lockup|rcu.*stall|SMMU.*fault|\bXid\b|page allocation failure'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_hash(root):
    h = hashlib.sha256()
    for directory in ('scripts', 'overlay', 'profiles', 'roce', 'manifests', 'tests'):
        for p in sorted((root / directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    for name in ('start.sh', '.env', '.env.example'):
        h.update(name.encode() + b'\0' + (root / name).read_bytes())
    return h.hexdigest()


def geometry(extra_gib, c4='literal', maxlen=None):
    if type(extra_gib) not in (int, float) or not math.isfinite(extra_gib) or not 0 <= extra_gib <= 8 or extra_gib * 2 != int(extra_gib * 2):
        raise ValueError('extra GiB must be a half-GiB increment in 0..8; capacity is not admission')
    head = ORDINARY_BYTES + int(extra_gib * GiB)
    l = layout('fp4x', head)
    # Retain the serving boot's 39-block slack, not just the theoretical six.
    length = BASE_MAX + (l['blocks'] - layout('fp4x')['blocks']) * 64
    length = length if maxlen is None else maxlen
    if length % 64 or not OUTPUT + 64 <= length <= int(l['max_model_len']):
        raise ValueError('maxlen must be on the 64-token grid and fit the head')
    c4_length = length if c4 == 'literal' else min(length, ((l['blocks'] - 1)//4 - 1)*64)
    need = 1 + 4 * math.ceil((c4_length + 3)/64)
    return dict(extra_GiB=extra_gib, ordinary_bytes=head, max_model_len=length,
                layout=l, c4_mode=c4, c4_total_tokens_per_request=c4_length,
                c4_prompt_tokens=c4_length-OUTPUT, decode_tokens=OUTPUT,
                c4_required_blocks=need, c4_resident_possible=need <= l['blocks'],
                qualifies_requested_worst_case=c4 == 'literal' and need <= l['blocks'])


def plan(a):
    rows = []
    for n in range(4):
        g = geometry(n)
        if a.sim_dir:
            g['conservative_sim'] = forecast(a.sim_dir, 'fp4x', g['max_model_len'], g['ordinary_bytes'])
        rows.append(g)
    result = dict(scope='OFFLINE ONLY; capacity is not admission', steps=rows,
                  rule='all rank sim lower >=8.5 GiB; page-cache credit=0; live rank0>=8.0, any rank>=6.0',
                  literal_c4='4 independent max contexts require about 4x aggregate KV; queue/preemption cannot qualify')
    if a.out:
        a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def prepare(a):
    if not re.fullmatch(r'glm53full-kvhr-[a-zA-Z0-9-]+', a.boot):
        raise ValueError('unique boot name must start glm53full-kvhr-')
    g = geometry(a.extra_gib, a.c4_mode, a.maxlen)
    dest = a.out.resolve()
    if dest.exists():
        raise ValueError('fresh output directory required; no resume/reboot in same step')
    ref = a.reference_clone.resolve()
    for path in (ref/'.env', ref/'overlay/guard/libdispram_copy_guard.so'):
        if not path.is_file():
            raise ValueError('qualified reference clone missing ' + str(path))
    dest.mkdir(parents=True)
    clone = dest/'clone'
    shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns('.git', '.env', 'state', 'logs', 'cache', '__pycache__', '*.pyc'))
    (clone/'overlay/guard').mkdir(exist_ok=True)
    shutil.copy2(ref/'overlay/guard/libdispram_copy_guard.so', clone/'overlay/guard')
    # Existing qualified .env can reference FP4X_*; bind them before sourcing.
    pre = f'export FP4X_FORMAT=fp4x FP4X_MAXLEN={g["max_model_len"]} FP4X_BOOT={shlex.quote(a.boot)}\n'
    tail = dict(CTN=a.boot, GLM_KV_FORMAT='fp4x', RECIPE_DISPRAM='require',
                RECIPE_MAX_MODEL_LEN=str(g['max_model_len']), RECIPE_KV_HEAD_BYTES=str(g['ordinary_bytes']),
                RECIPE_PAGE_CACHE_POLICY=str(int(a.cache_policy)), DISPRAM_ALLOW_RM_ALLOC_OOM='1',
                OVERLAY_REMOTE='/srv/glm/'+a.boot)
    if getattr(a, 'loader', None):
        tail['GLM_LOADER'] = a.loader
    if getattr(a, 'sidecar', None):
        tail.update(GLM_ATTN_WEIGHTS='nvfp4', GLM_ATTN_NVFP4_DIR=str(a.sidecar))
    if getattr(a, 'param_hash', False):
        tail.update(GLM_PARAM_HASH='1', VLLM_SERVER_DEV_MODE='1')
    (clone/'.env').write_text(pre + (ref/'.env').read_text()+'\n'+
                            '\n'.join('export '+k+'='+shlex.quote(v) for k,v in tail.items())+'\n')
    wrapper = dest/'launcher.sh'
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nR=$(cd "$(dirname "$0")" && pwd)\n'
                       'export PATH=$HOME/glm-control/bin:/usr/local/bin:/usr/bin:/bin\n'
                       'export PYTHONUNBUFFERED=1\ncd "$R/clone"\nexec ./start.sh "$@"\n')
    wrapper.chmod(0o755)
    dry = subprocess.run(['bash', str(wrapper), 'serve'], env=dict(os.environ, DRY='1'),
                         capture_output=True, text=True, timeout=120, check=True).stdout
    (dest/'dry.txt').write_text(dry)
    ranks=[]
    for line in dry.splitlines():
        if not line.startswith('docker run -d '):
            continue
        parts=shlex.split(line)
        def arg(k): return parts[parts.index(k)+1]
        ranks.append(dict(rank=int(arg('--node-rank')), head=int(arg('--kv-cache-memory-bytes')),
                          maxlen=int(arg('--max-model-len'))))
    if sorted(r['rank'] for r in ranks) != [0,1,2,3] or any(
            (r['head'],r['maxlen']) != (g['ordinary_bytes'],g['max_model_len']) for r in ranks):
        raise ValueError('DRY ranks do not match requested geometry')
    # Freeze measured helpers, not their changing campaign paths. GateMetrics
    # credits the think10 probe authors; preserve source and hashes verbatim.
    if a.cycle_probe:
        shutil.copy2(a.cycle_probe, dest/'cycle.py')
        if not a.gate_metrics:
            raise ValueError('--cycle-probe needs --gate-metrics')
        shutil.copy2(a.gate_metrics, dest/'gate_metrics.py')
    manifest = dict(geometry=g, boot=a.boot, dry_sha256=sha(dest/'dry.txt'),
                    package_sha256=package_hash(clone), reference_env_sha256=sha(ref/'.env'),
                    guard_sha256=sha(clone/'overlay/guard/libdispram_copy_guard.so'),
                    helper_sha256={p.name:sha(p) for p in (dest/'cycle.py',dest/'gate_metrics.py') if p.exists()},
                    prepared_epoch=time.time(), cache_policy=a.cache_policy)
    (dest/'step.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def admit(a):
    """Write a source-bound conservative sim receipt; refused rows stay refused."""
    m=json.loads((a.step/'step.json').read_text());g=m['geometry']
    result=forecast(a.sim_dir,'fp4x',g['max_model_len'],g['ordinary_bytes'])
    result.update(package_sha256=m['package_sha256'],dry_sha256=m['dry_sha256'],
                  ordinary_bytes=g['ordinary_bytes'],blocks=g['layout']['blocks'],
                  stress_mode=g['c4_mode'],page_cache_credit_GiB=0)
    if not g['c4_resident_possible']:
        result['reasons'].append('four resident max requests exceed aggregate KV capacity')
        result['passes_8_5']=False
    result['headroom_sim_sha256']=sha(ROOT/'scripts/fp4_kv_admission.py')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('ADMITTED' if result['passes_8_5'] else 'REFUSED',result['per_rank_lower_GiB'])


# Persistent sampler, transported as one short Python program per rank. Sampling
# reads counters only. No tensor allocations, GPU commands, or pinned side tests.
NODE_CODE = r'''
import json,os,pathlib,subprocess,threading,time
p=pathlib.Path
probe=subprocess.run(['journalctl','-k','-b','-n','1','--no-pager'],capture_output=True,text=True,timeout=5)
if probe.returncode or not probe.stdout.strip() or 'No entries' in probe.stdout or 'permission' in probe.stderr.lower():
 print(json.dumps({'error':'kernel journal unreadable','stderr':probe.stderr}),flush=True);raise SystemExit(1)
j=subprocess.Popen(['journalctl','-k','-b','-n','0','-f','-o','json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
events=[];lock=threading.Lock()
def readjournal():
 for line in j.stdout:
  try:
   with lock:events.append(json.loads(line).get('MESSAGE',line))
  except Exception:
   with lock:events.append('journal decode failure')
threading.Thread(target=readjournal,daemon=True).start()
pss_snapshot=[]
def readpss():
 global pss_snapshot
 while True:
  q=subprocess.run(['docker','top',CTN,'-eo','pid,args'],capture_output=True,text=True,timeout=3)
  processes=[]
  if q.returncode==0:
   for line in q.stdout.splitlines()[1:]:
    fields=line.strip().split(None,1)
    if not fields or not fields[0].isdigit():continue
    pid=fields[0]
    try:
     sm={k:int(v.split()[0]) for k,v in (s.split(':',1) for s in p('/proc/'+pid+'/smaps_rollup').read_text().splitlines() if ':' in s)}
     processes.append({'pid':pid,'command':fields[1] if len(fields)>1 else '', 'pss_kB':sm.get('Pss'),'rss_kB':sm.get('Rss'),'locked_kB':sm.get('Locked')})
    except (OSError,ValueError):processes.append({'pid':pid,'pss':'unreadable'})
  pss_snapshot=processes
  time.sleep(10)
threading.Thread(target=readpss,daemon=True).start()
try:
 tick=0
 while True:
  start=time.monotonic()
  m={k:int(v.split()[0]) for k,v in (s.split(':',1) for s in p('/proc/meminfo').read_text().splitlines())}
  vm={k:int(v) for k,v in (s.split() for s in p('/proc/vmstat').read_text().splitlines())}
  with lock:kernel=events[:];events.clear()
  if j.poll() is not None:raise RuntimeError('kernel journal follower exited')
  row={'epoch':time.time(),'mem_kB':m,'buddyinfo':p('/proc/buddyinfo').read_text(),
       'page_size':os.sysconf('SC_PAGE_SIZE'),'vmstat':{k:v for k,v in vm.items() if k.startswith(('pswp','compact','allocstall','pgmajfault'))},'kernel':kernel}
  buddy=[]
  for line in row['buddyinfo'].splitlines():
   fields=line.split();counts=list(map(int,fields[4:]));page=row['page_size']
   buddy.append({'node':fields[1].strip(','),'zone':fields[3], 'order_counts':counts,
                 'free_bytes_in_blocks_at_least_2MiB':sum(c*(page<<order) for order,c in enumerate(counts) if (page<<order)>=2**21),
                 'free_bytes_in_blocks_at_least_8MiB':sum(c*(page<<order) for order,c in enumerate(counts) if (page<<order)>=2**23)})
  row['buddy_zones']=buddy
  row["processes"]=pss_snapshot
  print(json.dumps(row),flush=True);tick+=1
  time.sleep(max(0,1-(time.monotonic()-start)))
finally:
 j.terminate()
'''


class Monitor:
    def __init__(self, out, boot):
        self.out,self.boot=out,boot
        self.latest={};self.base={};self.minimum={};self.problem=None
        self.closed=threading.Event();self.procs=[];self.lock=threading.Lock();self.admitted=False
        self.on_trip=None;self.routine={};self.min_floor_gib=6.0;self.rank0_floor_gib=8.0
    def start(self):
        for rank in range(4):
            cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',os.environ['RECIPE_HOSTS'].split()[rank],
                 'python3 -u -c '+shlex.quote('CTN='+repr(f'{self.boot}-r{rank}')+'\n'+NODE_CODE)]
            err=(self.out/f'rank{rank}-sampler.stderr').open('w')
            p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,text=True)
            err.close();self.procs.append(p)
            threading.Thread(target=self.reader,args=(rank,p),daemon=True).start()
        threading.Thread(target=self.watch,daemon=True).start()
    def reader(self, rank, process):
        try:
            with (self.out/f'rank{rank}-memory.jsonl').open('a',buffering=1) as f:
                for line in process.stdout:
                    f.write(line)
                    row=json.loads(line)
                    if row.get('error'):raise RuntimeError(row['error'])
                    m=row['mem_kB'];avail=m['MemAvailable']/1048576;swap=m['SwapTotal']-m['SwapFree']
                    with self.lock:
                        self.latest[rank]=(time.monotonic(),row)
                        self.base.setdefault(rank,swap)
                        self.minimum[rank]=min(self.minimum.get(rank,avail),avail)
                        if avail<self.min_floor_gib: self.problem=f'rank {rank} MemAvailable <{self.min_floor_gib} GiB'
                        elif self.admitted and rank==0 and avail<self.rank0_floor_gib: self.problem=f'rank0 below serving {self.rank0_floor_gib:g} GiB floor'
                        elif swap-self.base[rank]>65536: self.problem=f'rank {rank} swap growth >64 MiB'
                        routine=[x for x in row['kernel'] if re.search(KERNEL_ROUTINE,str(x).strip())]
                        self.routine[rank]=self.routine.get(rank,0)+len(routine)
                        if any(re.search(KERNEL_FATAL,str(x),re.I) for x in row['kernel'] if x not in routine): self.problem=f'rank {rank} kernel OOM/stall/fault'
        except Exception as e:
            self.problem=f'rank {rank} sampler: {e}'
        finally:
            if not self.closed.is_set():self.problem=f'rank {rank} sample stream ended'
    def watch(self):
        started=time.monotonic()
        while not self.closed.wait(1):
            with self.lock:
                for rank in range(4):
                    if rank not in self.latest:
                        if time.monotonic()-started>15:self.problem=f'rank {rank} no initial sample'
                    elif time.monotonic()-self.latest[rank][0]>3:self.problem=f'rank {rank} sample stale >3s'
            if self.problem and self.on_trip:
                callback,self.on_trip=self.on_trip,None
                callback()  # interrupts blocking HTTP / phase waits on the main thread
    def check(self):
        if self.problem:raise RuntimeError(self.problem)
    def ready(self):
        return len(self.latest)==4
    def close(self):
        self.closed.set()
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        (self.out/'memory-summary.json').write_text(json.dumps(dict(min_GiB=self.minimum,swap_base_KiB=self.base,problem=self.problem,routine_rm_alloc_lines=self.routine),indent=2)+'\n')


def http(url, data=None, timeout=30):
    request=urllib.request.Request(url,data=json.dumps(data).encode() if data is not None else None,
                                  headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as r:return json.load(r)


def metric(text,name):
    values=[float(line.split()[-1]) for line in text.splitlines()
            if line.startswith(name+'{') or line.startswith(name+' ')]
    if not values:
        raise RuntimeError('required metric missing: '+name)
    return sum(values)


def metrics(base):
    with urllib.request.urlopen(base+'/metrics',timeout=5) as r:return r.read().decode()


def wait_drained(base,mon):
    for _ in range(30):
        mon.check();t=metrics(base)
        if metric(t,'vllm:num_requests_running')==metric(t,'vllm:num_requests_waiting')==0:return t
        time.sleep(1)
    raise RuntimeError('API not drained / foreign traffic')


def stream(base,model,tokens,out,index,barrier=None,activity=None,output_tokens=OUTPUT):
    if barrier:barrier.wait(timeout=15)
    body=dict(model=model,prompt=tokens,max_tokens=output_tokens,temperature=0,seed=0,
              ignore_eos=True,stream=True,stream_options={'include_usage':True})
    req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    first=last=None;usage=None
    with (out/f'stream-{index}.jsonl').open('w') as f, urllib.request.urlopen(req,timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b'data: ') or raw.strip()==b'data: [DONE]':continue
            ev=json.loads(raw[6:]);f.write(json.dumps(ev)+'\n')
            if ev.get('error'):raise RuntimeError(str(ev['error']))
            if any(c.get('text') for c in ev.get('choices',[])):
                now=time.monotonic();first=first or now;last=now
                if activity is not None:activity[index]=(first,last)
            usage=ev.get('usage') or usage
    if not usage or usage.get('completion_tokens')!=output_tokens or usage.get('prompt_tokens')!=len(tokens):
        raise RuntimeError('stream usage does not match exact prompt / 1024 decode')
    return dict(index=index,usage=usage,first=first,last=last,decode_wall_s=last-first if first and last else None)


def prompt_tokens(base,model,n,index,tag):
    # Tokenize distinct public-domain synthetic text once, then use token IDs.
    # Different first tokens prevent accidental c4 prefix sharing. No huge
    # string generation and no host pinned allocations.
    line=f'{tag} stream {index}. Public test: explain arithmetic, geometry and weather in clear prose. '
    t=http(base+'/tokenize',dict(model=model,prompt=line))['tokens']
    prefix=http(base+'/tokenize',dict(model=model,prompt=f'{tag} independent-{index}: '))['tokens']
    if len(prefix)>=n:raise RuntimeError('prompt target too small')
    return prefix+(t*math.ceil((n-len(prefix))/len(t)))[:n-len(prefix)]


def c4(a,m,mon,out,model):
    g=m['geometry'];base=a.endpoint
    before=wait_drained(base,mon)
    preempt_before=metric(before,'vllm:num_preemptions_total')
    prompts=[prompt_tokens(base,model,g['c4_prompt_tokens'],i,m['boot']) for i in range(4)]
    trace=[];rows=[]
    # First cold-fill all four prompts, then replay cached prefixes for the
    # full decode phase. Completion of four HTTP submits alone is not a gate.
    for phase,outputs in (('cold-prefill',1),('resident-decode',OUTPUT)):
        activity={};barrier=threading.Barrier(4);observed_c4=False
        stage=out/phase;stage.mkdir()
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            futures=[pool.submit(stream,base,model,t,stage,i,barrier,activity,outputs) for i,t in enumerate(prompts)]
            while not all(f.done() for f in futures):
                mon.check();t=metrics(base)
                running=metric(t,'vllm:num_requests_running')
                trace.append(dict(phase=phase,epoch=time.time(),running=running,
                                  waiting=metric(t,'vllm:num_requests_waiting'),
                                  preemptions=metric(t,'vllm:num_preemptions_total')))
                if len(activity)==4 and running==4 and not any(f.done() for f in futures):observed_c4=True
                time.sleep(1)
            rows.extend(dict(phase=phase,**f.result()) for f in futures)
        finally:
            # A monitor trip must reach docker stop immediately; waiting for
            # blocked request threads before cleanup would defeat the guard.
            pool.shutdown(wait=False,cancel_futures=True)
        boundary=wait_drained(base,mon)
        if phase=='cold-prefill':
            decode_before=boundary
        else:
            if not observed_c4:raise RuntimeError('four resident decoding streams not observed')
            hits=(metric(boundary,'vllm:prefix_cache_hits_total')-
                  metric(decode_before,'vllm:prefix_cache_hits_total'))
            if hits < 4*max(0,len(prompts[0])-128):
                raise RuntimeError('all four prefills not retained in APC before decode')
    after=wait_drained(base,mon)
    (out/'c4-trace.json').write_text(json.dumps(dict(trace=trace,streams=rows,observed_four_decoding=observed_c4),indent=2)+'\n')
    if not observed_c4:raise RuntimeError('four resident decoding streams not observed')
    if metric(after,'vllm:num_preemptions_total')!=preempt_before:raise RuntimeError('preemption: literal resident stress invalid')
    # APC repeat after full c4 drain, same token IDs. Verify query/hit deltas.
    tokens=prompts[0];hits_before=metric(after,'vllm:prefix_cache_hits_total')
    queries_before=metric(after,'vllm:prefix_cache_queries_total')
    repeated=stream(base,model,tokens,out,'apc-repeat')
    for _ in range(10):
        mon.check();after=wait_drained(base,mon)
        hits=metric(after,'vllm:prefix_cache_hits_total')-hits_before
        if hits>0:break
        time.sleep(1)
    receipt=dict(stream=repeated,hit_delta=hits,query_delta=metric(after,'vllm:prefix_cache_queries_total')-queries_before)
    (out/'apc.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if hits<=0:raise RuntimeError('APC repeat did not record any hits')


def run_subprocess(cmd,out,mon,timeout):
    with out.open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        started=time.monotonic()
        try:
            while p.poll() is None:
                mon.check()
                if time.monotonic()-started>timeout:raise RuntimeError('phase timeout')
                time.sleep(1)
            if p.returncode:raise RuntimeError('phase failed: '+str(out))
        finally:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def cycle(a,m,mon,out):
    for name,digest in m['helper_sha256'].items():
        if sha(a.step/name)!=digest:raise ValueError('cycle helper drift')
    if not {'cycle.py','gate_metrics.py'} <= m['helper_sha256'].keys():
        raise ValueError('prepare with cycle.py and gate_metrics.py; speed cannot be unmeasured')
    # Local frozen gate_metrics import has priority over the campaign path in cycle.py.
    cmd=[sys.executable,'-c',
         'import sys,runpy;sys.path.insert(0,sys.argv[1]);import gate_metrics;'
         'sys.argv=[sys.argv[1]+"/cycle.py"]+sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")',
         str(a.step),'--endpoint',a.endpoint,'--out',str(out),'--boot',m['boot']]
    run_subprocess(cmd,out/'cycle.log',mon,3600)
    result=json.loads((out/'cycle-summary.json').read_text())
    baseline=json.loads(a.baseline_cycle.read_text())
    ratios={}
    for kind in ('prose','code'):
        r,b=result['summary'][kind],baseline['summary'][kind]
        if r['n']!=10 or b['n']!=10:raise RuntimeError('need 10 attributable cycles per kind in baseline and candidate')
        ratios[kind]=r['cycle_ms_median']/b['cycle_ms_median']
    (out/'cycle-comparison.json').write_text(json.dumps(dict(ratios=ratios,baseline_sha256=sha(a.baseline_cycle)),indent=2)+'\n')
    if max(ratios.values())>1.02:raise RuntimeError('decode cycle-ms regression >2%')


def dash(a,mon,out):
    url=a.dash_endpoint+'/bench';rows=[]
    for i in range(4):
        mon.check()
        if http(url).get('active'):raise RuntimeError('sparkDash already active')
        config=dict(port=8095,modelId='GLM-5.3',concurrencies=[1],maxTokens=256,promptType='prose')
        post=http(url,config);start=time.monotonic()
        while True:
            time.sleep(2);mon.check();status=http(url)
            if not status.get('active'):break
            if time.monotonic()-start>300:raise RuntimeError('sparkDash timeout')
        result=status['last']
        if result.get('benchId') not in (None,post.get('benchId')):raise RuntimeError('stale sparkDash job')
        cells=result.get('results',[])
        if not cells or any(x.get('streamsOk')!=1 or x.get('streamsFailed',0) or x.get('error') for x in cells):
            raise RuntimeError('sparkDash c1 failed')
        rows.append(dict(warmup=i==0,post=post,result=result))
    (out/'sparkDash-c1.json').write_text(json.dumps(rows,indent=2)+'\n')


def receipts(a,m,out):
    records={}
    hosts=os.environ.get('RECIPE_HOSTS','').split()
    if len(hosts)!=4:raise ValueError('RECIPE_HOSTS must contain four rank-ordered hosts')
    for i in range(4):
        p=subprocess.run(['ssh',hosts[i],'docker logs '+shlex.quote(m['boot']+f'-r{i}')+' 2>&1'],
                         capture_output=True,text=True,timeout=30,check=True)
        (out/f'rank{i}-boot.log').write_text(p.stdout)
        events=[]
        for line in p.stdout.splitlines():
            if 'glm-dispram-kv: {' in line:events.append(json.loads(line.split('glm-dispram-kv: ',1)[1]))
        pools=[x for x in events if x.get('event')=='pool'];leases=[x for x in events if x.get('event')=='lease']
        want=m['geometry']['layout']
        if not pools or any(x['num_blocks']!=want['blocks'] or x['head']!=want['kv_bytes'] for x in pools):
            raise RuntimeError('rank block/head receipt mismatch')
        if not leases or any(x.get('all_ok')!=1 or x.get('agreed')!=2145386496 for x in leases):
            raise RuntimeError('rank lease mismatch')
        if f'Using max model len {m["geometry"]["max_model_len"]}' not in p.stdout:raise RuntimeError('maxlen mismatch')
        records[i]=dict(pools=pools,leases=leases)
    (out/'receipts.json').write_text(json.dumps(records,indent=2)+'\n')


def execute(a):
    # Refuse before any process/network activity on a Mac, even with --execute.
    if not a.execute or platform.system()!='Linux' or platform.node().split('.')[0]!=os.environ.get('RECIPE_COORDINATOR_HOSTNAME','rank0'):
        raise ValueError('run requires --execute on rank0; use plan/prepare on the Mac')
    if not os.environ.get('INVOCATION_ID'):
        raise ValueError('run in a rank0 systemd --user transient unit; no Mac-owned controller')
    a.step=a.step.resolve();m=json.loads((a.step/'step.json').read_text());g=m['geometry']
    if package_hash(a.step/'clone')!=m['package_sha256'] or sha(a.step/'dry.txt')!=m['dry_sha256']:
        raise ValueError('prepared clone / launch vector drift')
    admission=json.loads(a.admission.read_text())
    if (admission.get('package_sha256'),admission.get('dry_sha256'))!=(m['package_sha256'],m['dry_sha256']):
        raise ValueError('sim receipt must bind exact prepared package and launch vector')
    expected=(g['ordinary_bytes'],g['max_model_len'],g['layout']['blocks'],g['c4_mode'])
    actual=tuple(admission.get(k) for k in ('ordinary_bytes','max_model_len','blocks','stress_mode'))
    lows=admission.get('per_rank_lower_GiB',[])
    if actual!=expected or len(lows)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) and x>=8.5 for x in lows):
        raise ValueError('exact-layout sim lower must be >=8.5 GiB for all ranks and requested stress mode')
    if admission.get('page_cache_credit_GiB')!=0 or not admission.get('sim_source_sha256'):
        raise ValueError('zero flusher credit / sim source binding required')
    if not g['c4_resident_possible']:raise ValueError('literal c4 at max cannot fit the shared KV pool; no boot')
    if not a.baseline_cycle:raise ValueError('baseline cycle-summary.json required')
    out=a.step/'run';out.mkdir()  # one boot per step; a failure requires a fresh step
    os.environ['PATH']=os.path.expanduser('~/glm-control/bin')+':/usr/local/bin:/usr/bin:/bin'
    os.environ.pop('LD_PRELOAD',None);os.environ.pop('PYTHONPATH',None)
    if Path.home().joinpath('fleet_busy').exists():raise RuntimeError('fleet lock already held')
    p=subprocess.run(['systemctl','--user','is-active','glm-serving-watch.service'],capture_output=True,text=True)
    if p.stdout.strip()=='active':raise RuntimeError('serving watch still active; perform documented operator handoff first')
    # Existing launcher's preflight verifies GPU idle and borrowers/lenders and
    # acquires its own no-clobber fleet lock before launching any rank.
    mon=Monitor(out,m['boot']);launcher=None;keep=threading.Event();outcome='FAILED';error=None
    def hold():
        while not keep.wait(20):
            state=a.step/'clone/state/deployment.json'
            if launcher and launcher.poll() is None and state.exists() and json.loads(state.read_text()).get('ctn')==m['boot']:
                Path.home().joinpath('glm-control/hold').write_text(str(int(time.time())+120)+'\n')
    def interrupted(signum, frame):raise RuntimeError(mon.problem or f'driver signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    mon.on_trip=lambda: os.kill(os.getpid(),signal.SIGTERM)
    try:
        mon.start();threading.Thread(target=hold,daemon=True).start()
        deadline=time.monotonic()+20
        while not mon.ready():
            mon.check()
            if time.monotonic()>deadline:raise RuntimeError('samplers not ready')
            time.sleep(.2)
        with (out/'serve.log').open('w') as f:
            launcher=subprocess.Popen(['bash',str(a.step/'launcher.sh'),'serve'],stdout=f,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while 'Admission PASS' not in (out/'serve.log').read_text(errors='replace'):
            mon.check()
            if launcher.poll() is not None:raise RuntimeError('launcher exited before admission')
            if time.monotonic()-start>1800:raise RuntimeError('preflight+boot timeout')
            time.sleep(1)
        mon.admitted=True
        receipts(a,m,out)
        model=http(a.endpoint+'/v1/models')['data'][0]['id']
        cycle_before=out/'cycle-before';cycle_before.mkdir()
        cycle(a,m,mon,cycle_before)
        single_tokens=prompt_tokens(a.endpoint,model,g['max_model_len']-OUTPUT,0,m['boot']+'-single-max')
        stream(a.endpoint,model,single_tokens,out,'single-max')
        mon.check()
        c4(a,m,mon,out,model)
        dash(a,mon,out)
        cycle_after=out/'cycle-after';cycle_after.mkdir()
        cycle(a,m,mon,cycle_after)
        mon.check();outcome='PASS' if g['qualifies_requested_worst_case'] else 'SCREEN_ONLY'
    except BaseException as exc:
        error=str(exc)
    finally:
        mon.on_trip=None
        keep.set()
        # Terminate explicit launcher PID, then use its own stop/postcheck path.
        # No rm, kill by pattern, controller on Mac, autoreboot, or power action.
        if launcher and launcher.poll() is None:
            launcher.terminate()
            try:launcher.wait(timeout=5)
            except subprocess.TimeoutExpired:launcher.kill();launcher.wait()
        try:
            if (a.step/'clone/state/deployment.json').exists():
                subprocess.run(['bash',str(a.step/'launcher.sh'),'stop'],stdout=(out/'stop.log').open('w'),
                               stderr=subprocess.STDOUT,timeout=360,check=True)
        except BaseException as exc:
            error=(error or '')+'; stop failed: '+str(exc);outcome='RECOVERY_REQUIRED'
        mon.close()
        (out/'RESULT.json').write_text(json.dumps(dict(outcome=outcome,error=error,geometry=g,
             admission_sha256=sha(a.admission),epoch=time.time(),restore='manual best qualified full GLM after postcheck; never automatic'),indent=2)+'\n')
    if outcome not in ('PASS','SCREEN_ONLY'):raise RuntimeError(error or outcome)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--sim-dir',type=Path);q.add_argument('--out',type=Path)
    q=sub.add_parser('prepare')
    q.add_argument('--reference-clone',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--boot',required=True);q.add_argument('--extra-gib',type=float,required=True)
    q.add_argument('--loader',choices=['fast','coalesced'])
    q.add_argument('--sidecar',type=Path);q.add_argument('--param-hash',action='store_true')
    q.add_argument('--maxlen',type=int);q.add_argument('--cache-policy',action='store_true')
    q.add_argument('--c4-mode',choices=['literal','pool-quarter'],default='literal')
    q.add_argument('--cycle-probe',type=Path);q.add_argument('--gate-metrics',type=Path)
    q=sub.add_parser('admit');q.add_argument('--step',type=Path,required=True)
    q.add_argument('--sim-dir',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--step',type=Path,required=True);q.add_argument('--admission',type=Path,required=True)
    q.add_argument('--baseline-cycle',type=Path);q.add_argument('--execute',action='store_true')
    q.add_argument('--endpoint',default='http://127.0.0.1:8095')
    q.add_argument('--dash-endpoint',default='http://127.0.0.1:5555/api/sparks/rank0/llm')
    a=p.parse_args()
    {'plan':plan,'prepare':prepare,'admit':admit,'run':execute}[a.command](a)


if __name__=='__main__':
    main()
