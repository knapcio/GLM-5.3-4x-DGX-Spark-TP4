#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline prepare/plan; future rank0 coordinator, one explicit step per run.

Step 0 is the normal-loader, unchanged-capacity NVFP4 hash/cycle control.
Steps 1..4 need independently recalibrated, package-bound sim admission.
Every window restores the recorded B configuration after a verified stop.
"""
import argparse
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
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import stress_step as S

EXTRAS = (1.0, 1.5, 2.0, 2.5)
SIDECAR = '/srv/glm/models/GLM-5.3-attn-nvfp4-87cf357'
B_CTN = 'glm53full-nvfp4a-20261006-b'
B_CLONE = '/srv/glm/glm-control/runs/nvfp4a-20261006/clone-a'
GiB = 1 << 30


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def ladder(receipts, sim_dir=None):
    b = json.loads((receipts/'b/done-mem.json').read_text())
    a = json.loads((receipts/'a/done-mem.json').read_text())
    minima = json.loads((receipts/'b/memwatch-minima.json').read_text())
    stressed = [min(row['min_GiB'][r] for phase,row in minima.items() if phase != 'boot') for r in range(4)]
    # Quiet median gain is evidence of steady saving, not a transient calibration.
    gain = [x-y for x,y in zip(b['quiet_median_GiB'],a['quiet_median_GiB'])]
    sys.path.insert(0,str(S.ROOT/'overlay/bringup'))
    from glm_fp4_prefill import scratch_bytes
    base=S.geometry(0,'pool-quarter')
    base_scratch=scratch_bytes(pool_tokens=base['layout']['tokens'],max_model_len=98176)
    rows=[]
    for i,n in enumerate(EXTRAS,1):
        g=S.geometry(n,'pool-quarter')
        workspace_growth=scratch_bytes(pool_tokens=g['layout']['tokens'],max_model_len=g['max_model_len'])-base_scratch
        row=dict(step=i,**g,workspace_growth_bytes=workspace_growth,quiet_projection_GiB=[x-n for x in b['quiet_min_GiB']],
                 observed_stress_projection_GiB=[x-n for x in stressed],
                 projection_scope='subtract ordinary head only; workspace, uncertainty and longer stress still uncharged')
        if sim_dir:
            old=S.forecast(sim_dir,'fp4x',g['max_model_len'],g['ordinary_bytes'])
            row['legacy_sim']=old
            row['quiet_gain_sensitivity_lower_GiB']=[x+y for x,y in zip(old['per_rank_lower_GiB'],gain)]
            row['admission']='REFUSED: legacy sim retained; quiet gain sensitivity is not recalibration'
        rows.append(row)
    return dict(scope='OFFLINE capacity proposal; no fleet qualification',steps=rows,
                quiet_gain_GiB=gain,observed_stress_min_GiB=stressed,
                source_sha256={str(p.relative_to(receipts)):S.sha(p) for p in
                               (receipts/'a/done-mem.json',receipts/'b/done-mem.json',receipts/'b/memwatch-minima.json')},
                bytes_per_token=31976,block_bytes=2046464,carveout_bytes=2145386496,
                sim_rule='all four lower bounds >=8.5 GiB; zero cache credit; exact package/shape/stress',
                live_rule='rank0 >=8.0 GiB continuously after admission; any rank <6 emergency stop')


def prepare(a):
    a.out=a.out.resolve();a.out.mkdir() # no-clobber run window
    save(a.out/'ladder.json',ladder(a.receipts,a.sim_dir))
    for i,n in enumerate((0,)+EXTRAS):
        S.prepare(argparse.Namespace(reference_clone=a.reference_clone,out=a.out/f'step{i}',
                  boot=f'glm53full-kvhr-integ1006-{a.tag}-s{i}',extra_gib=n,maxlen=None,
                  c4_mode='pool-quarter',cache_policy=False,cycle_probe=a.cycle_probe,gate_metrics=a.gate_metrics,
                  loader='fast' if i==0 else 'coalesced',sidecar=Path(SIDECAR),param_hash=True))
        if i:
            S.admit(argparse.Namespace(step=a.out/f'step{i}',sim_dir=a.sim_dir,out=a.out/f'step{i}/admission.json'))
    # One fresh original-B restore package for each window, preserving old containers.
    for i in range(5):
        restore=a.out/f'restore{i}';restore.mkdir();cl=restore/'clone'
        shutil.copytree(a.reference_clone,cl,ignore=shutil.ignore_patterns('.git','.env','state','logs','cache','__pycache__','*.pyc'))
        ctn=f'glm53full-integ1006-{a.tag}-restore-b-{i}'
        env=(a.reference_clone/'.env').read_text()
        (cl/'.env').write_text(f'export FP4X_FORMAT=fp4x FP4X_MAXLEN=98176 FP4X_BOOT={ctn}\n'+env+'\n'+
             f'export CTN={ctn} GLM_ATTN_WEIGHTS=nvfp4 GLM_ATTN_NVFP4_DIR={SIDECAR}\n'
             'export DISPRAM_ALLOW_RM_ALLOC_OOM=1 GLM_FAST_LOAD=1\nunset GLM_LOADER GLM_PARAM_HASH VLLM_SERVER_DEV_MODE\n')
        wrapper=restore/'launcher.sh'
        wrapper.write_text('#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")/clone"\n'
                           'export PATH=$HOME/glm-control/bin:/usr/local/bin:/usr/bin:/bin\nexec ./start.sh "$@"\n')
        wrapper.chmod(0o755)
        dry=subprocess.run(['bash',str(wrapper),'serve'],env=dict(os.environ,DRY='1'),capture_output=True,text=True,timeout=120,check=True).stdout
        (restore/'dry.txt').write_text(dry)
        shutil.copy2(a.receipts/'rank0/memwatch.py',restore/'memwatch.py')
        save(restore/'restore.json',dict(boot=ctn,package_sha256=S.package_hash(cl),dry_sha256=S.sha(restore/'dry.txt'),
             memwatch_sha256=S.sha(restore/'memwatch.py')))
    save(a.out/'window.json',dict(tag=a.tag,baseline_ctn=a.serving_ctn,baseline_clone=a.serving_clone,
         sidecar=SIDECAR,steps=list(EXTRAS),last_passed_index=-1,baseline_guard_unit=a.serving_guard_unit))
    print('Prepared',a.out,'; execution requires fresh sim admission; step0 captures compatible normal hashes.')


def remote(rank, command, timeout=80):
    hosts=os.environ.get('RECIPE_HOSTS','').split()
    if len(hosts)!=4:raise ValueError('RECIPE_HOSTS must contain four rank-ordered hosts')
    r=subprocess.run(['ssh','-n','-o','BatchMode=yes','-o','ConnectTimeout=5',hosts[rank],command],
                     capture_output=True,text=True,timeout=timeout,check=True)
    return r.stdout.strip()


def acquire_window_lock(path, deployment):
    import fcntl
    token='integ1006:'+uuid.uuid4().hex
    if path.exists():
        with path.open('r+') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if f.read()!=deployment['token']:raise RuntimeError('fleet lock owned by another coordinator')
            if path.stat().st_ino!=os.fstat(f.fileno()).st_ino:raise RuntimeError('lock path changed')
            f.seek(0);f.write(token);f.truncate();f.flush()
    else:
        with path.open('x') as f:f.write(token)
    return token, True


def release_window_lock(path, token):
    if not path.exists() or path.read_text()!=token:raise RuntimeError('window lock changed; recovery required')
    path.unlink()


class LoaderSamples:
    """100 ms /proc reads on each host, independent of one-second safety telemetry."""
    def __init__(self,out,mon):
        self.out,self.mon=out,mon;self.procs=[];self.latest={};self.closed=False
    def start(self):
        helper=(S.ROOT/'scripts/loader_memwatch.py').read_text().split("if __name__ == '__main__':")[0]
        code=helper+"\nwhile True:\n print(json.dumps(sample()),flush=True)\n time.sleep(.1)\n"
        hosts=os.environ.get('RECIPE_HOSTS','').split()
        if len(hosts)!=4:raise ValueError('RECIPE_HOSTS must contain four rank-ordered hosts')
        for r in range(4):
            with (self.out/f'rank{r}-loader-memory.stderr').open('w') as err:
                p=subprocess.Popen(['ssh','-n','-o','BatchMode=yes','-o','ConnectTimeout=5',hosts[r],
                                    'python3 -u -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=err,text=True)
            self.procs.append(p);threading.Thread(target=self.reader,args=(r,p),daemon=True).start()
        threading.Thread(target=self.watch,daemon=True).start()
    def reader(self,r,p):
        try:
            with (self.out/f'rank{r}-loader-memory.jsonl').open('w',buffering=1) as f:
                for line in p.stdout:
                    json.loads(line);f.write(line);self.latest[r]=time.monotonic()
        except Exception as exc:self.mon.problem='100ms loader sampler: '+str(exc)
        finally:
            if not self.closed:self.mon.problem=f'rank {r} loader sampler ended'
    def watch(self):
        started=time.monotonic()
        while not self.closed:
            if time.monotonic()-started>15:
                if len(self.latest)!=4 or any(time.monotonic()-t>3 for t in self.latest.values()):
                    self.mon.problem='100ms loader telemetry stale/missing >3s';return
            time.sleep(1)
    def ready(self):return len(self.latest)==4
    def close(self):
        self.closed=True
        for p in self.procs:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def rendered_images(dry):
    images={}
    for line in dry.read_text().splitlines():
        if line.startswith('docker run -d '):
            parts=shlex.split(line);images[int(parts[parts.index('--node-rank')+1])]=parts[parts.index('serve')-1]
    if len(images)!=4:raise RuntimeError('image vector missing')
    return images


def transport(step,m,out,mon):
    """Gate the actual pinned CUDA image on every idle rank; no image builds/pulls."""
    images=rendered_images(step/'dry.txt')
    # Images appear immediately before the vllm serve argument in the rendered vector.
    if len(images)!=4:raise RuntimeError('transport image vector missing')
    hosts=os.environ.get('RECIPE_HOSTS','').split()
    if len(hosts)!=4:raise ValueError('RECIPE_HOSTS must contain four rank-ordered hosts')
    for r in range(4):
        mon.check();stage='/srv/glm/glm-control/integ-transport/'+m['boot']
        remote(r,'mkdir -p /srv/glm/glm-control/integ-transport; mkdir '+shlex.quote(stage)+'; mkdir '+shlex.quote(stage+'/tmp'))
        subprocess.run(['rsync','-a','--exclude=state','--exclude=logs','--exclude=cache',str(step/'clone')+'/',hosts[r]+':'+stage+'/src/'],check=True,timeout=120)
        command=shlex.join(['docker','run','--pull','never','--network','none','--name',m['boot']+f'-transport-r{r}',
             '--gpus','all','--ipc','host','--ulimit','memlock=-1:-1','-e','TMPDIR=/transport-tmp',
             '--mount','type=bind,src='+stage+'/src,dst=/pkg,readonly',
             '--mount','type=bind,src='+stage+'/tmp,dst=/transport-tmp',
             '--entrypoint','python3',images[r],'-B','/pkg/tests/gpu/coalesced_transport.py'])
        (out/f'transport-r{r}.log').write_text(remote(r,command,180)+'\n')
        # Gate container is exited and preserved; next recipe preflight verifies GPU idle.

def check_admission(step, m):
    g=m['geometry'];p=step/'admission.json';ad=json.loads(p.read_text())
    expected=(m['package_sha256'],m['dry_sha256'],g['ordinary_bytes'],g['max_model_len'],g['layout']['blocks'],g['c4_mode'])
    actual=tuple(ad.get(k) for k in ('package_sha256','dry_sha256','ordinary_bytes','max_model_len','blocks','stress_mode'))
    lows=ad.get('per_rank_lower_GiB',[])
    if expected!=actual or len(lows)!=4 or not all(type(x) in (float,int) and math.isfinite(x) and x>=8.5 for x in lows):
        raise ValueError('REFUSED: package-bound sim lower must be >=8.5 GiB on every rank')
    if ad.get('page_cache_credit_GiB')!=0 or not ad.get('sim_source_sha256') or not g['c4_resident_possible']:
        raise ValueError('REFUSED: exact sim source, zero cache credit and resident shared-pool c4 required')
    if not ad.get('calibration_sources') or not ad.get('calibration_model'):
        raise ValueError('REFUSED: expanded NVFP4 stress needs explicit recalibration source files/model')
    if ad['sim_source_sha256'] not in ad['calibration_sources'].values():
        raise ValueError('sim implementation source must be retained in calibration sources')
    for rel,digest in ad['calibration_sources'].items():
        if S.sha(step/rel)!=digest:raise ValueError('calibration source drift: '+rel)
    if not ad.get('passes_8_5') or ad.get('reasons'):
        raise ValueError('REFUSED: sim reasons remain')
    return ad


def verify_step(step, m):
    if S.package_hash(step/'clone')!=m['package_sha256'] or S.sha(step/'dry.txt')!=m['dry_sha256']:
        raise ValueError('prepared source/vector drift')
    dry=subprocess.run(['bash',str(step/'launcher.sh'),'serve'],env=dict(os.environ,DRY='1'),
                       capture_output=True,text=True,timeout=120,check=True).stdout
    if dry!=(step/'dry.txt').read_text():raise ValueError('Spark DRY differs from Mac DRY')
    for name,digest in m['helper_sha256'].items():
        if S.sha(step/name)!=digest:raise ValueError('helper drift')


def smoke(endpoint, mon, out):
    model=S.http(endpoint+'/v1/models')['data'][0]['id']
    result=S.http(endpoint+'/v1/chat/completions',dict(model=model,messages=[dict(role='user',content='Reply with exactly OK')],
                  temperature=0,max_tokens=32,chat_template_kwargs=dict(enable_thinking=False)))
    if result['choices'][0]['message']['content'].strip()!='OK':raise RuntimeError('exact OK failed')
    tokens=S.prompt_tokens(endpoint,model,4096,0,str(out))
    S.wait_drained(endpoint,mon)
    S.stream(endpoint,model,tokens,out,'apc-cold',output_tokens=16)
    before=S.wait_drained(endpoint,mon)
    S.stream(endpoint,model,tokens,out,'apc-warm',output_tokens=16)
    after=S.wait_drained(endpoint,mon)
    hits=S.metric(after,'vllm:prefix_cache_hits_total')-S.metric(before,'vllm:prefix_cache_hits_total')
    if hits<3968:raise RuntimeError('restore/control APC repeat not retained')
    save(out/'smoke.json',dict(exact_OK=result,apc_hit_delta=hits));return model


def hashes(a,m,mon,out):
    # Guarded read-only worker RPCs. No unknown RPC reaches an uninstrumented rank.
    for r in range(4):
        env=remote(r,'docker inspect -f '+shlex.quote('{{json .Config.Env}}')+' '+m['boot']+f'-r{r}')
        if not {'GLM_PARAM_HASH=1','VLLM_SERVER_DEV_MODE=1'}<=set(json.loads(env)):
            raise RuntimeError('hash env absent')
        log=(out/f'rank{r}-boot.log').read_text()
        if 'glm-phash: worker methods attached' not in log:raise RuntimeError('hash attach receipt absent')
    S.wait_drained(a.endpoint,mon)
    def rpc(method,args):
        return S.http(a.endpoint+'/collective_rpc',dict(method=method,args=[str(x) for x in args]),timeout=200)['results']
    statuses=rpc('glm_phash_status',[])
    if sorted(x.get('tp_rank') for x in statuses)!=list(range(4)) or any(x.get('error') for x in statuses):
        raise RuntimeError('hash status not four healthy ranks')
    dest=out/'manifests';dest.mkdir();deadline=time.monotonic()+1200
    while True:
        mon.check()
        rows=rpc('glm_phash_run',[m['boot'],json.dumps(dict(chunk_mb=32,segment_mb=256,threads=6,max_seconds=120,mem_floor_gib=8.5,attrs=True))])
        if len(rows)!=4 or any(x.get('error') for x in rows):raise RuntimeError('hash worker failed: '+str(rows))
        if all(x.get('done') for x in rows):break
        if time.monotonic()>deadline:raise RuntimeError('hash time budget exceeded')
    H=module(a.step/'clone/overlay/overlay/glm_param_hash.py','integ_hash_compare');report={}
    for row in rows:
        r=row['tp_rank'];mounts=json.loads(remote(r,'docker inspect -f '+shlex.quote('{{json .Mounts}}')+' '+m['boot']+f'-r{r}'))
        runtime=[x['Source'] for x in mounts if x['Destination']=='/cache']
        path=row['path']
        if len(runtime)!=1 or not path.startswith('/cache/'):raise RuntimeError('invalid hash path/cache mount')
        data=remote(r,'cat '+shlex.quote(runtime[0]+path[len('/cache'):]),300);(dest/f'rank{r}.json').write_text(data+'\n')
        candidate=json.loads(data)
        if candidate['header']['tp_rank']!=r or candidate['digest_params_buffers']!=row['digest_params_buffers']:
            raise RuntimeError('manifest/RPC identity differs')
        if a.index:
            baseline=json.loads((a.window/'step0/run/manifests'/f'rank{r}.json').read_text())
            for key in ('tp_rank','host','models','chunk_mb','segment_mb'):
                if baseline['header'][key]!=candidate['header'][key]:raise RuntimeError('incompatible weight baseline: '+key)
            compare=H.compare(baseline,candidate);report[r]=compare
            if compare['verdict']!='EQUAL' or not compare['digest_equal']:raise RuntimeError('weights DIFFERENT rank '+str(r))
    save(out/'weight-comparison.json',dict(verdict='EQUAL' if a.index else 'CONTROL_CAPTURED',ranks=report))


def boot_ledger(out, started, health_epoch, admission_epoch, coalesced=False):
    ranks={}
    for r in range(4):
        text=(out/f'rank{r}-boot.log').read_text();loads=[]
        for line in text.splitlines():
            if 'glm-coalesced-load: {' in line:loads.append(json.loads(line.split('glm-coalesced-load: ',1)[1]))
        if coalesced and not loads:raise RuntimeError('coalesced receipts absent')
        if loads and (not {'target','draft'}<=set(x['load_kind'] for x in loads) or
                      any(not all(x.get(k) for k in ('complete','cuda','direct')) for x in loads)):
            raise RuntimeError('coalesced target/draft GPU/direct receipts incomplete')
        weight_times=[float(x) for x in re.findall(r'Loading weights took ([0-9.]+) seconds',text)]
        model_time=re.findall(r'Model loading took [0-9.]+ GiB and ([0-9.]+) seconds',text)
        graph_time=re.findall(r'Graph capturing finished in ([0-9.]+) secs',text)
        model=float(model_time[-1]) if model_time else None
        phase=dict(main_weights_s=weight_times[0] if weight_times else None,mtp_s=weight_times[1] if len(weight_times)>1 else None,
                   model_total_s=model,other_construction_repack_s=model-sum(weight_times) if model is not None and len(weight_times)==2 else None,
                   graphs_s=float(graph_time[-1]) if graph_time else None)
        fast=[]
        path=out/f'rank{r}-loader-memory.jsonl'
        if path.exists():fast=[json.loads(line) for line in path.read_text().splitlines()]
        intervals=[]
        for load in loads:
            samples=[x for x in fast if load['start_wall_epoch']<=x['wall_epoch']<=load['end_wall_epoch']]
            if not samples:raise RuntimeError('no 100ms samples in loader interval')
            intervals.append(dict(load_kind=load['load_kind'],samples=len(samples),
                min_available_bytes=min(x['MemAvailable_bytes'] for x in samples),max_available_bytes=max(x['MemAvailable_bytes'] for x in samples),
                max_used_bytes=max(x['used_excluding_available_bytes'] for x in samples),max_dirty_bytes=max(x['Dirty_bytes'] for x in samples),
                max_writeback_bytes=max(x['Writeback_bytes'] for x in samples),
                swap_growth_bytes=max(x['SwapTotal_bytes']-x['SwapFree_bytes'] for x in samples)-(fast[0]['SwapTotal_bytes']-fast[0]['SwapFree_bytes'])))
        if 'glm-nvfp4-attn: loaded 385 logical attention matrices' not in text or not re.search(r'mtp-only .*check PASS',text):
            raise RuntimeError('NVFP4/native MTP load receipt missing rank '+str(r))
        ranks[r]=dict(loader=loads,phases=phase,loader_intervals=intervals,phase_lines=[line for line in text.splitlines() if any(k in line for k in
                      ('Loading weights took','Model loading took','Graph capturing','Capturing','glm-nvfp4-attn: loaded','glm-window-memory'))])
    save(out/'boot-breakdown.json',dict(launch_epoch=started,health_epoch=health_epoch,admission_epoch=admission_epoch,
         health_seconds=health_epoch-started,admission_seconds=admission_epoch-health_epoch,ranks=ranks,
         exclusive_phases='main weights; MTP; other construction/repack; graphs; profile/KV/warm; launch/API residual',
         unresolved='raw timestamp lines retained; inseparable phases grouped/unknown; overlapping loader counters never summed',
         reduction='slowest-rank critical path, never sum parallel rank durations; hash time excluded'))


def halt_launcher(process, step, out):
    if process and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    state=step/'clone/state/deployment.json'
    if state.exists():
        with (out/'stop.log').open('w') as f:
            subprocess.run(['bash',str(step/'launcher.sh'),'stop'],stdout=f,stderr=subprocess.STDOUT,timeout=360,check=True)
    if Path.home().joinpath('fleet_busy').exists():raise RuntimeError('lock retained; recovery required')


def wait_admission(process, log, mon, endpoint):
    start=time.monotonic();health_epoch=None
    while 'Admission PASS' not in log.read_text(errors='replace'):
        mon.check()
        if process.poll() is not None:raise RuntimeError('launcher exited before admission')
        if time.monotonic()-start>1800:raise RuntimeError('boot timeout')
        if health_epoch is None:
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=1) as response:
                    if response.status==200:health_epoch=time.time()
            except Exception:pass
        time.sleep(1)
    if health_epoch is None:raise RuntimeError('health timestamp missing')
    mon.admitted=True;mon.check()
    if any(mon.latest[r][1]['mem_kB']['MemAvailable']<8*1048576 for r in range(4)):
        raise RuntimeError('admission latest sample below 8 GiB')
    return health_epoch,time.time()


def handoff_restore(step, m, process, mon, out):
    if process.poll() is not None:raise RuntimeError('restore launcher exited')
    process.terminate();process.wait(timeout=15) # SIGTERM detaches; containers stay up
    state=step/'clone/state/deployment.json';d=json.loads(state.read_text());swap={}
    for r in range(4):
        if remote(r,'docker inspect -f '+shlex.quote('{{.State.Running}}')+' '+m['boot']+f'-r{r}')!='true':
            raise RuntimeError('restore rank down')
        swap[str(r)]=int(remote(r,"awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo"))
    d['swap_base']=swap;save(state,d)
    di=Path.home()/'.config/systemd/user/glm-serving-watch.service.d/fp4x-serving.conf'
    di.write_text('[Service]\nWorkingDirectory='+str(step/'clone')+'\nEnvironment=FP4X_FORMAT=fp4x FP4X_MAXLEN=98176 FP4X_BOOT='+m['boot']+
                  ' CTN='+m['boot']+' GLM_ATTN_WEIGHTS=nvfp4 GLM_ATTN_NVFP4_DIR='+SIDECAR+'\nExecStart=\nExecStart='+str(step/'clone/start.sh')+' watch\n')
    logs=step/'clone/logs'/m['boot'];logs.mkdir(parents=True,exist_ok=True)
    link=Path.home()/'glm-control/clone-w4final/logs'/m['boot'];link.symlink_to(logs)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start','glm-serving-watch.service'],check=True)
    guard_unit='glm-integ1006-floor-'+m['boot']
    guard_out=out/'floor';guard_out.mkdir()
    subprocess.run(['systemd-run','--user','--unit='+guard_unit,
        '--setenv=PATH=/srv/glm/glm-control/bin:/usr/local/bin:/usr/bin:/bin',
        '/usr/bin/python3',str(step/'memwatch.py'),'--out',str(guard_out),'--serve-log',str(out/'serve.log'),
        '--unit','glm-serving-watch.service','--launcher',str(step/'launcher.sh'),'--floor','8.05'],check=True)
    deadline=time.monotonic()+100
    heartbeat=Path.home()/'glm-control/log/heartbeat.log';size=heartbeat.stat().st_size
    while time.monotonic()<deadline:
        mon.check()
        active=subprocess.run(['systemctl','--user','is-active','glm-serving-watch.service'],capture_output=True,text=True).stdout.strip()
        fresh=list(logs.glob('watchdog.jsonl'))
        guard_active=subprocess.run(['systemctl','--user','is-active',guard_unit],capture_output=True,text=True).stdout.strip()
        if active=='active' and guard_active=='active' and fresh and time.time()-fresh[0].stat().st_mtime<30:
            hold=Path.home()/'glm-control/hold'
            if hold.exists():hold.rename(out/'hold-retired') # let watch own heartbeat
            if 'touched(watch)' in heartbeat.read_text()[size:]:break
        time.sleep(1)
    else:raise RuntimeError('restored watch/heartbeat handoff unverified')
    # Match the existing serving handoff: release only the verified new owner
    # after watch, floor guard and heartbeat are active.
    release_window_lock(Path.home()/'fleet_busy',d['token'])
    save(out/'RESTORED-B.json',dict(boot=m['boot'],config='NVFP4 B / FP4x / adaptive / 98176 / ordinary1GiB',swap_base=swap,watch=True,heartbeat='touched(watch)',floor_guard_unit=guard_unit,guard_floor_GiB=8.05))


def boot_preflight(step, out, headers=None, image=None, enabled=True):
    """Advisory before fleet inspection; absent cache and failures never block."""
    from boot_preflight import advisory
    return advisory(step/'clone', step/'dry.txt', out, headers, image, enabled)


def run(a):
    if not a.execute or platform.system()!='Linux' or platform.node().split('.')[0]!=os.environ.get('RECIPE_COORDINATOR_HOSTNAME','rank0') or not os.environ.get('INVOCATION_ID'):
        raise ValueError('run requires --execute on rank0 in systemd --user; Mac execution refused')
    a.window=a.window.resolve();a.step=a.window/f'step{a.index}'
    m=json.loads((a.step/'step.json').read_text());verify_step(a.step,m)
    boot_preflight(a.step,a.step/'boot-preflight.json',getattr(a,'preflight_headers',None),
                   getattr(a,'preflight_image',None),getattr(a,'boot_preflight',True))
    if a.index:
        check_admission(a.step,m)
        for r in range(4):json.loads((a.window/'step0/run/manifests'/f'rank{r}.json').read_text())
        a.baseline_cycle=a.window/'step0/run/cycle-before/cycle-summary.json'
        json.loads(a.baseline_cycle.read_text())
    elif (m['geometry']['ordinary_bytes'],m['geometry']['max_model_len'])!=(GiB,98176):
        raise ValueError('normal control must retain qualified B capacity')
    restore=a.window/f'restore{a.index}';rm=json.loads((restore/'restore.json').read_text())
    if S.package_hash(restore/'clone')!=rm['package_sha256']:raise ValueError('original B restore source drift')
    if S.sha(restore/'memwatch.py')!=rm['memwatch_sha256']:raise ValueError('original serving floor-guard source drift')
    restored_dry=subprocess.run(['bash',str(restore/'launcher.sh'),'serve'],env=dict(os.environ,DRY='1'),capture_output=True,text=True,timeout=120,check=True).stdout
    if restored_dry!=(restore/'dry.txt').read_text():raise ValueError('restore DRY drift')
    out=a.step/'run';out.mkdir();home=Path.home();busy=home/'fleet_busy';hold=home/'glm-control/hold'
    cfg=json.loads((a.window/'window.json').read_text());bcl=Path(cfg['baseline_clone'])
    if cfg['last_passed_index']!=a.index-1:raise ValueError('run steps in order; previous step must pass and restore B')
    if a.index:
        for name,digest in cfg['control_hashes'].items():
            if S.sha(a.window/'step0/run/manifests'/name)!=digest:raise ValueError('normal control manifest drift')
        if S.sha(a.baseline_cycle)!=cfg['control_cycle_sha256']:raise ValueError('control cycle drift')
    d=json.loads((bcl/'state/deployment.json').read_text())
    if d['ctn']!=cfg['baseline_ctn']:raise RuntimeError('current B identity differs; regenerate window')
    if busy.exists() and busy.read_text()!=d['token']:raise RuntimeError('fleet owned by another coordinator')
    # Snapshot concrete current state before detaching its watch.
    save(out/'serving-before.json',d)
    shutil.copy2(home/'.config/systemd/user/glm-serving-watch.service.d/fp4x-serving.conf',out/'watch-before.conf')
    images=rendered_images(a.step/'dry.txt')
    restore_images=rendered_images(restore/'dry.txt')
    for r in range(4):
        raw=remote(r,'docker inspect '+d['ctn']+f'-r{r}')
        (out/f'serving-before-r{r}.json').write_text(raw+'\n')
        serving=json.loads(raw)[0]
        for image in (images[r],restore_images[r]):
            pinned=remote(r,'docker image inspect -f '+shlex.quote('{{.Id}}')+' '+shlex.quote(image))
            if serving['Image']!=pinned:raise RuntimeError('candidate/restore image differs from current B rank '+str(r))
        if not serving['State']['Running']:raise RuntimeError('current B rank not running')
    owner_token,operator_lock=acquire_window_lock(busy,d)
    save(out/'window-owner.json',dict(token=owner_token,operator_lock=operator_lock))
    hold.write_text(str(int(time.time())+120)+'\n');stop_hold=threading.Event()
    def refresh():
        while not stop_hold.wait(20):hold.write_text(str(int(time.time())+120)+'\n')
    threading.Thread(target=refresh,daemon=True).start()
    mon=S.Monitor(out,m['boot']);fast=LoaderSamples(out,mon);process=None;detached=False;stopped=False;restore_process=None;restore_mon=None
    outcome='FAILED';error=None
    def trip(*_):raise RuntimeError((restore_mon.problem if restore_mon else None) or mon.problem or 'coordinator signal')
    signal.signal(signal.SIGTERM,trip);signal.signal(signal.SIGINT,trip)
    try:
        previous_guard=cfg.get('baseline_guard_unit')
        if previous_guard:
            status=subprocess.run(['systemctl','--user','is-active',previous_guard],capture_output=True,text=True)
            (out/'floor-guard-before.txt').write_text(status.stdout+status.stderr)
            if status.stdout.strip()=='active':subprocess.run(['systemctl','--user','stop',previous_guard],check=True)
        subprocess.run(['systemctl','--user','stop','glm-serving-watch.service'],check=True);detached=True
        with (out/'stop-b.log').open('w') as f:
            subprocess.run(['bash','./start.sh','stop'],cwd=bcl,
                 env=dict(os.environ,FP4X_FORMAT='fp4x',FP4X_MAXLEN='98176',FP4X_BOOT=d['ctn'],CTN=d['ctn'],
                          GLM_ATTN_WEIGHTS='nvfp4',GLM_ATTN_NVFP4_DIR=SIDECAR,DISPRAM_ALLOW_RM_ALLOC_OOM='1'),
                 stdout=f,stderr=subprocess.STDOUT,timeout=360,check=True)
        if operator_lock:release_window_lock(busy,owner_token)
        if busy.exists():raise RuntimeError('B stop retained lock')
        stopped=True;mon.on_trip=lambda:os.kill(os.getpid(),signal.SIGTERM);mon.start();fast.start()
        deadline=time.monotonic()+20
        while not (mon.ready() and fast.ready()):
            mon.check()
            if time.monotonic()>deadline:raise RuntimeError('samplers not ready')
            time.sleep(.2)
        if a.index:transport(a.step,m,out,mon)
        started=time.time()
        with (out/'serve.log').open('w') as f:process=subprocess.Popen(['bash',str(a.step/'launcher.sh'),'serve'],stdout=f,stderr=subprocess.STDOUT)
        health,admission=wait_admission(process,out/'serve.log',mon,a.endpoint)
        S.receipts(a,m,out);boot_ledger(out,started,health,admission, bool(a.index))
        model=smoke(a.endpoint,mon,out);hashes(a,m,mon,out)
        if a.index:
            normal=json.loads((a.window/'step0/run/boot-breakdown.json').read_text())
            candidate=json.loads((out/'boot-breakdown.json').read_text())
            save(out/'boot-vs-normal.json',dict(control=normal,candidate=candidate,
                 health_delta_s=candidate['health_seconds']-normal['health_seconds'],
                 caveat='KV capacity differs; this is an integration boot comparison, not a loader-only causal benchmark'))
        panel=out/'cycle-before';panel.mkdir()
        if not a.index:
            a.baseline_cycle=panel/'cycle-summary.json' # capture same matched panel, ratio 1
        S.cycle(a,m,mon,panel)
        if a.index:
            S.stream(a.endpoint,model,S.prompt_tokens(a.endpoint,model,m['geometry']['max_model_len']-1024,0,m['boot']+'-single'),out,'single-max')
            S.c4(a,m,mon,out,model);S.dash(a,mon,out)
            panel=out/'cycle-after';panel.mkdir();S.cycle(a,m,mon,panel)
        mon.check();outcome='SHARED_POOL_PASS' if a.index else 'CONTROL_CAPTURED'
    except BaseException as exc:error=str(exc)
    finally:
        mon.on_trip=None
        try:
            if stopped:
                for r in range(4):
                    name=m['boot']+f'-transport-r{r}'
                    if a.index:
                        remote(r, 'if docker inspect '+shlex.quote(name)+' >/dev/null 2>&1; then docker stop -t 20 '+shlex.quote(name)+'; fi')
                halt_launcher(process,a.step,out)
            elif detached:raise RuntimeError('initial B stop incomplete')
            if stopped:
                fast.close();mon.close()
                ro=restore/'run';ro.mkdir();restore_mon=S.Monitor(ro,rm['boot']);restore_mon.start()
                deadline=time.monotonic()+20
                while not restore_mon.ready():
                    restore_mon.check()
                    if time.monotonic()>deadline:raise RuntimeError('restore telemetry not ready')
                    time.sleep(.2)
                restore_mon.on_trip=lambda:os.kill(os.getpid(),signal.SIGTERM)
                with (ro/'serve.log').open('w') as f:restore_process=subprocess.Popen(['bash',str(restore/'launcher.sh'),'serve'],stdout=f,stderr=subprocess.STDOUT)
                wait_admission(restore_process,ro/'serve.log',restore_mon,a.endpoint)
                smoke(a.endpoint,restore_mon,ro)
                stop_hold.set();handoff_restore(restore,rm,restore_process,restore_mon,ro)
                cfg.update(baseline_ctn=rm['boot'],baseline_clone=str(restore/'clone'),baseline_guard_unit='glm-integ1006-floor-'+rm['boot'])
                if error is None:
                    cfg['last_passed_index']=a.index
                    if a.index==0:
                        cfg['control_hashes']={f'rank{r}.json':S.sha(out/'manifests'/f'rank{r}.json') for r in range(4)}
                        cfg['control_cycle_sha256']=S.sha(out/'cycle-before/cycle-summary.json')
                save(a.window/'window.json',cfg)
        except BaseException as exc:
            error=(error or '')+'; restore/recovery: '+str(exc);outcome='RECOVERY_REQUIRED'
            # Bound cleanup after a restore boot/handoff error; never release an unknown lock.
            if restore_process:
                try:
                    subprocess.run(['systemctl','--user','stop','glm-integ1006-floor-'+rm['boot']],timeout=30)
                    subprocess.run(['systemctl','--user','stop','glm-serving-watch.service'],timeout=30,check=True)
                    halt_launcher(restore_process,restore,restore/'run')
                except BaseException:pass
        finally:
            stop_hold.set();fast.close();mon.close()
            if restore_mon:restore_mon.on_trip=None;restore_mon.close()
            save(out/'RESULT.json',dict(outcome=outcome,error=error,geometry=m['geometry'],
                 restored_B=(restore/'run/RESTORED-B.json').exists(),epoch=time.time()))
    if error or outcome=='RECOVERY_REQUIRED':raise RuntimeError(error or outcome)
    print(outcome,'; B restored; no automatic next step')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--receipts',type=Path,required=True);q.add_argument('--sim-dir',type=Path);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--receipts',type=Path,required=True);q.add_argument('--sim-dir',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True);q.add_argument('--tag',required=True);q.add_argument('--reference-clone',type=Path,required=True)
    q.add_argument('--serving-ctn',default=B_CTN);q.add_argument('--serving-clone',default=B_CLONE)
    q.add_argument('--serving-guard-unit',default='nvfp4a-memwatch-b')
    q.add_argument('--cycle-probe',type=Path,required=True);q.add_argument('--gate-metrics',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--window',type=Path,required=True);q.add_argument('--index',type=int,choices=range(5),required=True)
    q.add_argument('--boot-preflight',action=argparse.BooleanOptionalAction,default=True)
    q.add_argument('--preflight-headers',type=Path);q.add_argument('--preflight-image')
    q.add_argument('--execute',action='store_true');q.add_argument('--endpoint',default='http://127.0.0.1:8095');q.add_argument('--dash-endpoint',default='http://127.0.0.1:5555/api/sparks/rank0/llm')
    a=p.parse_args()
    if a.command=='plan':save(a.out,ladder(a.receipts,a.sim_dir))
    elif a.command=='prepare':prepare(a)
    else:
        os.environ['PATH']=os.path.expanduser('~/glm-control/bin')+':/usr/local/bin:/usr/bin:/bin'
        os.environ.pop('LD_PRELOAD',None);os.environ.pop('PYTHONPATH',None);run(a)


if __name__=='__main__':main()
