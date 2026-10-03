#!/usr/bin/env python3
"""Fresh-clone full-GLM release gate. Execution requires --execute in an idle fleet window."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bench'))
from release_probe import Probes, compare_teacher, quality_gate, save, http

CAPS = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
            VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='2', BLIS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def clone(repo, rev, destination):
    # --no-local avoids hardlinks and excludes untracked/local runtime state.
    commit = git(repo, 'rev-parse', rev+'^{commit}')
    subprocess.run(['git', 'clone', '--quiet', '--no-local', '--no-checkout', str(repo), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', commit], check=True)
    if git(destination, 'status', '--porcelain'):
        raise RuntimeError('fresh checkout is dirty')
    return commit


def read_config(repo, config):
    # Same bash semantics as start.sh; print only non-secret allowlisted fields.
    code = 'source "$1/profiles/current.env"; source "$1/.env.example"; source "$2"; '
    code += 'export RECIPE_HOSTS="${HOSTS[*]}" RECIPE_IPS="${IPS[*]}" RECIPE_IMAGES="${IMAGES[*]:-}"; '
    code += 'export IMAGE MODEL_DIR DRAFT_DIR NCCL_HOST_DIR OVERLAY_REMOTE FABRIC_IFACE IB_HCA NCCL_SHA256; '
    code += "exec python3 -c 'import os,json; print(json.dumps({k:os.environ.get(k,\"\") for k in "
    code += repr(['RECIPE_HOSTS', 'RECIPE_IPS', 'RECIPE_IMAGES', 'IMAGE', 'MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE', 'FABRIC_IFACE', 'IB_HCA', 'NCCL_SHA256']).replace("'", '"') + "}))'"
    data = json.loads(subprocess.check_output(['bash', '-c', code, 'gate-config', str(repo), str(config)], text=True))
    images = data['RECIPE_IMAGES'].split() or [data['IMAGE']]*4
    if len(images) != 4 or any(not re.fullmatch(r'sha256:[0-9a-f]{64}', image) for image in images):
        raise ValueError('Pin four full Docker image IDs in IMAGES; mutable tags are refused')
    if len(data['RECIPE_HOSTS'].split()) != 4:
        raise ValueError('four hosts required')
    return data


def plan():
    return dict(max_window_s=5400, workload_deadline_s=3900, recovery_reserve_s=1500,
        preparation='Pre-stage pinned ARM64 images, weights and NCCL; builds/downloads are outside the fleet window.',
        stages=['fresh reference clone + image/weights/NCCL verification + cold boot + watchdog',
                'reference correctness + c1/c4 teacher A1/A2 + original qeval x3',
                'reference stop/idle/lock checks',
                'fresh candidate clone + verification + cold boot + correctness',
                'sparkDash prose/code/structured/json client c1-c16 + c1/c4 supplements',
                'cold input-token prefill 4k/8k/16k/32k/64k/128k',
                'prefix first/repeat + T=0/T=0.8 scans + teacher B1/B2 c1/c4 + qeval x3',
                'compare same-window reference; preserve qualified candidate or restore reference'],
        known_gaps=['Default profile max_model_len=32768; 64k/128k unsupported, full requested gate HOLD.',
                    'Client c>4 queues on four active slots; no actual batch16 graph claim.',
                    'Prefix caching (on in the native profile): cold prefill and teacher requests carry a nonce or a fresh '
                    'cache_salt; the first/repeat scans check answers, not the hit rate.',
                    'No historical Flash KL threshold imported; full-GLM A/A must satisfy frozen limits.'])


class Gate:
    def __init__(self, args):
        self.args = args; self.out = args.out.resolve(); self.out.mkdir(parents=True, exist_ok=False)
        self.end = time.monotonic()+args.window_minutes*60
        self.work_end = self.end-1500
        self.report = dict(status='RUNNING', date=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),
                           plan=plan(), stages=[], checks={}, limits=vars(args).copy())
        self.report['limits'] = {k: str(v) if isinstance(v, Path) else v for k,v in self.report['limits'].items()}
        self.active = None; self.proc = None; self.handle = None; self.reference_ready = False
        self.receipt()

    def receipt(self):
        self.report['remaining_window_s'] = max(0, self.end-time.monotonic())
        save(self.out/'gate.json', self.report)

    def check(self, recovery=False):
        if time.monotonic() >= (self.end if recovery else self.work_end):
            raise TimeoutError('window deadline; incomplete measurements are HOLD')
        if self.proc and self.proc.poll() is not None:
            raise RuntimeError('launcher/watchdog exited; see launch.log')

    def stage(self, name):
        self.check(); self.report['stages'].append(dict(name=name, utc=time.time())); self.receipt()
        print(name, flush=True)

    def command(self, command, cwd, output, timeout=900):
        self.check()
        with output.open('x') as log:
            subprocess.run(command, cwd=cwd, env={**os.environ, **CAPS}, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(timeout, max(1, self.work_end-time.monotonic())))
        self.check()

    def prepare(self, name, repo, rev, config):
        self.stage('clone-'+name)
        dest = self.out/name/'checkout'; dest.parent.mkdir()
        commit = clone(repo, rev, dest); conf = read_config(dest, config)
        runtime = conf['OVERLAY_REMOTE'].rstrip('/') + '-gate-' + name + '-' + uuid.uuid4().hex[:12]
        # Store only the launch allowlist; never copy arbitrary credentials from .env.
        lines = ['HOSTS=('+shlex.join(conf['RECIPE_HOSTS'].split())+')',
                 'IPS=('+shlex.join(conf['RECIPE_IPS'].split())+')',
                 'IMAGES=('+shlex.join(conf['RECIPE_IMAGES'].split() or [conf['IMAGE']]*4)+')']
        lines += [key+'='+shlex.quote(conf[key]) for key in ('MODEL_DIR','DRAFT_DIR','NCCL_HOST_DIR','FABRIC_IFACE','IB_HCA','NCCL_SHA256')]
        lines += ['OVERLAY_REMOTE='+shlex.quote(runtime)]
        (dest/'.env').write_text('\n'.join(lines)+'\n')
        checker_sha = hashlib.sha256((dest/'bench/qeval_tasks.py').read_bytes()).hexdigest()
        if checker_sha != '54719522d26996198c870264dfe5a93e2dd23f33436626c2477f1ac71206ffd2':
            raise ValueError('original 75-task qeval checker/prompt/budget hash mismatch')
        args = json.loads((dest/'profiles/serve-args.json').read_text())
        flags = [flag for flag in ('--enable-prefix-caching', '--no-enable-prefix-caching') if flag in args]
        if len(flags) != 1 or args.count(flags[0]) != 1:
            raise ValueError('the release profile must carry exactly one prefix-caching flag')
        prefix_caching = flags[0] == '--enable-prefix-caching'
        limit = int(args[args.index('--max-model-len')+1])
        if args[args.index('--port')+1] != '8095' or args[args.index('--max-num-seqs')+1] != '4':
            raise ValueError('gate expects port8095 and four active request slots')
        info = dict(qeval_checker_sha256=checker_sha, name=name, commit=commit, config=conf, runtime=runtime, max_model_len=limit,
                    prefix_caching=prefix_caching,
                    manifests={kind: hashlib.sha256((dest/f'manifests/{kind}.json').read_bytes()).hexdigest() for kind in ('target','drafter')})
        self.report[name] = info; self.receipt()
        return dest

    def boot(self, repo, name, recovery=False):
        self.check(recovery)
        if recovery and self.end-time.monotonic() < 930:
            raise TimeoutError('insufficient reserve for safe reference re-admission; fleet remains stopped')
        self.active = repo
        self.handle = (repo.parent/('recovery-launch.log' if recovery else 'launch.log')).open('x')
        env = {**os.environ, **CAPS, 'CTN': 'glm53full-gate-'+name+'-'+uuid.uuid4().hex[:10]}
        # No inherited dry flag, fleet lease or image override may bypass the clone's .env.
        env.pop('DRY', None); env.pop('RECIPE_FLEET_TOKEN', None)
        self.proc = subprocess.Popen(['./start.sh','serve'], cwd=repo, env=env, stdout=self.handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        begin = time.monotonic(); launched = None
        while True:
            self.check(recovery)
            log = Path(self.handle.name).read_text()
            if 'Watchdog active in foreground' in log and launched is None: launched = time.monotonic()
            if 'Admission PASS:' in log:
                self.report['stages'].append(dict(name=name+'-admitted', wall_s=time.monotonic()-begin,
                    boot_s=None if launched is None else time.monotonic()-launched,
                    log=str(Path(self.handle.name).relative_to(self.out))))
                self.receipt(); return
            if launched is not None and time.monotonic()-launched > 910:
                raise TimeoutError('boot admission >900s')
            time.sleep(2)

    def stop(self):
        if not self.active: return
        repo = self.active
        # Interrupt only this exact watchdog PID. Its handler stops its recorded ranks.
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try: self.proc.wait(timeout=270)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(timeout=10)
        path = repo/'state/deployment.json'
        if path.exists() and self.handle and 'Four containers stopped and preserved; owned lock released' in Path(self.handle.name).read_text():
            deployment = json.loads(path.read_text())
            path.rename(path.with_name('stopped-'+deployment['ctn']+'.json'))
        elif path.exists():
            with (repo.parent/'stop.log').open('a') as log:
                subprocess.run(['./start.sh','stop'], cwd=repo, env={**os.environ, **CAPS},
                               stdout=log, stderr=subprocess.STDOUT, timeout=270, check=True)
        # Verify no compute processes and exact token absent. Never stop unrelated jobs.
        stopped = list((repo/'state').glob('stopped-*.json'))
        if not stopped: raise RuntimeError('missing stop receipt')
        deployment = json.loads(max(stopped, key=lambda p:p.stat().st_mtime).read_text())
        checks = []
        for rank, host in enumerate(deployment['hosts']):
            name = deployment['ctn']+f'-r{rank}'
            code = "import json,pathlib,subprocess; d=json.loads(subprocess.check_output(['docker','inspect',"+repr(name)+"]))[0]; "
            code += "assert not d['State']['Running'] and not d['State']['OOMKilled']; "
            code += "assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(); "
            if rank == 0:
                code += 'p=pathlib.Path.home()/"fleet_busy"; assert not p.exists(); '
            code += "print('STOP_IDLE_PASS')"
            result = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                                             'python3 -c '+shlex.quote(code)], text=True, timeout=25)
            checks.append(dict(rank=rank, check=result.strip()))
        save(repo.parent/'stop-checks.json', checks)
        self.report['checks'][repo.parent.name+'-stop'] = True; self.receipt()
        self.active = None; self.proc = None
        if self.handle: self.handle.close(); self.handle = None

    def runtime(self, repo):
        deployment = json.loads((repo/'state/deployment.json').read_text())
        records = []
        for rank, host in enumerate(deployment['hosts']):
            self.check()
            name = deployment['ctn']+f'-r{rank}'
            command = 'docker inspect '+shlex.quote(name)
            data = json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command], text=True, timeout=25))[0]
            logs = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                'docker logs '+shlex.quote(name)+' 2>&1'], text=True, timeout=25)
            (repo.parent/f'rank{rank}.log').write_text(logs)
            env = dict(v.split('=',1) for v in data['Config']['Env'] if '=' in v)
            expected = dict(GLM_FULL_MLA='triton', GLM_MLA_SPLIT_K='32', GLM_DSA_SWA_POOL='1', GLM_ROCE_ALLREDUCE='1')
            passed = data['State']['Running'] and not data['State']['OOMKilled'] and data['RestartCount'] == 0 and data['Image'] == deployment['images'][rank]
            passed = passed and all(env.get(k) == v for k,v in expected.items())
            markers = ['glm-full-mla: ARMED', 'glm-dsa-swa-pool: registered',
                       'glm-window-memory: capture guard source pin PASS', 'capture-finished',
                       'glm-dsa-swa-pool: pool blocks=', f'GLM_ROCE_READY rank={rank} world=4', 'GLM_ROCE_ROUTE all_reduce']
            passed = passed and all(marker in logs for marker in markers)
            # Config/env are selected explicitly; Docker metadata may contain credentials.
            kv = re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', logs)
            kv_tokens = [int(value.replace(',', '')) for value in kv]
            # Engine capacity is logged by the head only; workers report pool geometry.
            if rank == 0:
                passed = passed and bool(kv_tokens) and min(kv_tokens) >= 32768
            records.append(dict(kv_tokens=kv_tokens, rank=rank, name=name, image=data['Image'], state=data['State'],
                                restarts=data['RestartCount'], cmd=data['Config']['Cmd'],
                                env={k:env.get(k) for k in expected}, markers={m:m in logs for m in markers}, passed=bool(passed)))
        save(repo.parent/'runtime.json', records)
        server = http(self.args.base.rstrip('/')+'/server_info?config_format=json')
        # Persist configuration only, excluding system/environment inventories.
        vconfig = server.get('vllm_config')
        args = json.loads((repo/'profiles/serve-args.json').read_text())
        expected_config = [('parallel_config','tensor_parallel_size',4),
                           ('model_config','max_model_len',int(args[args.index('--max-model-len')+1])),
                           ('scheduler_config','max_num_seqs',4),
                           ('speculative_config','num_speculative_tokens',3),
                           ('cache_config','kv_cache_memory_bytes',2147483648)]
        if not isinstance(vconfig,dict) or any((vconfig.get(group) or {}).get(key) != value for group,key,value in expected_config):
            raise RuntimeError('endpoint does not report the launched TP4/K3/KV/context configuration')
        save(repo.parent/'server-config.json', {group:{key:(vconfig.get(group) or {}).get(key)} for group,key,_ in expected_config})
        models = http(self.args.base.rstrip('/')+'/v1/models')
        save(repo.parent/'models.json', models)
        if not all(x['passed'] for x in records) or 'GLM-5.3' not in [x['id'] for x in models.get('data', [])]:
            raise RuntimeError('runtime pin/hook/RoCE/model checks HOLD')
        self.report['checks'][repo.parent.name+'-runtime'] = True
        self.receipt()

    def qeval(self, repo, name):
        results = []
        for repeat in range(1, 4):
            self.stage(f'{name}-qeval-{repeat}')
            label = name+f'-q{repeat}'; folder = repo.parent
            self.command([sys.executable, str(repo/'bench/qeval.py'), 'run', label, '--url',
                          self.args.base.rstrip('/')+'/v1/chat/completions', '--timeout','180','--concurrency','1'],
                         folder, folder/(label+'.log'), timeout=1200)
            results.append(json.loads((folder/('qeval-'+label+'.json')).read_text()))
        return results

    def run(self):
        ref = candidate = None
        try:
            ref = self.prepare('reference', self.args.reference_repo or ROOT, self.args.reference_rev, self.args.reference_config or self.args.config)
            candidate = self.prepare('candidate', ROOT, self.args.candidate_rev, self.args.config)
            if self.report['reference']['config']['RECIPE_HOSTS'] != self.report['candidate']['config']['RECIPE_HOSTS']:
                raise ValueError('both arms must use the same fleet')
            if self.report['reference']['manifests'] != self.report['candidate']['manifests']:
                raise ValueError('same-target reference requires identical target/draft manifests')
            self.stage('reference-boot'); self.boot(ref, 'reference')
            rp = Probes(self.args.base, ref.parent/'probes', self.check)
            if not rp.correctness(): raise RuntimeError('reference correctness HOLD')
            self.runtime(ref)
            panel = json.loads((ROOT/'bench/release/panel.json').read_text())
            # Public long-context teacher panel fixed by reference tokenizer, reused byte-for-byte.
            for count in (8192, 24576):
                ids = rp.request('/tokenize', dict(model='GLM-5.3', prompt='Public calibration note about rivers, arithmetic and sorting. '*count))['tokens'][:count]
                panel.append(dict(id='long-'+str(count), text=ids))
            save(self.out/'panel.json', panel)
            a1 = rp.teacher(panel, 'teacher-A1'); a2 = rp.teacher(panel, 'teacher-A2')
            a4 = rp.teacher(panel, 'teacher-A4-1', 4); a4r = rp.teacher(panel, 'teacher-A4-2', 4)
            aa = compare_teacher(panel, a1, a2); aa4 = compare_teacher(panel, a4, a4r)
            save(self.out/'reference-aa.json', dict(c1=aa, c4=aa4))
            rq = self.qeval(ref,'reference'); self.reference_ready = True
            self.stop()
            self.stage('candidate-boot'); self.boot(candidate,'candidate')
            cp = Probes(self.args.base, candidate.parent/'probes', self.check)
            checks = self.report['checks']; checks['correctness'] = cp.correctness(); self.runtime(candidate); self.receipt()
            self.stage('candidate-sparkdash'); checks['prose_c1_median_tps'] = cp.sparkdash(self.args.dash)
            self.stage('candidate-cold-prefill'); checks['prefill_full_scope'] = cp.prefill(self.report['candidate']['max_model_len'])
            self.stage('candidate-prefix-and-sampling'); checks['prefix_sampling'] = cp.prefix_scan()
            self.stage('candidate-teacher'); b1 = cp.teacher(panel,'teacher-B1'); b2 = cp.teacher(panel,'teacher-B2')
            b4 = cp.teacher(panel,'teacher-B4-1',4); b4r = cp.teacher(panel,'teacher-B4-2',4)
            bb = compare_teacher(panel,b1,b2); ab = compare_teacher(panel,a1,b1)
            bb4 = compare_teacher(panel,b4,b4r); ab4 = compare_teacher(panel,a4,b4)
            # Freeze limits before run. Neither a noisy A/A nor infinite values grants tolerance.
            checks['kl'] = dict(reference_aa=aa, candidate_aa=bb, reference_candidate=ab, reference_aa_c4=aa4, candidate_aa_c4=bb4, reference_candidate_c4=ab4,
                aa_max=self.args.aa_max, drift_max=self.args.kl_max,
                passed=max(item['mean'] for panel_result in (aa,bb,aa4,bb4) for item in panel_result['per_item']) <= self.args.aa_max and max(item['mean'] for panel_result in (ab,ab4) for item in panel_result['per_item']) <= self.args.kl_max)
            save(self.out/'kl.json',checks['kl'])
            cq = self.qeval(candidate,'candidate'); checks['qeval'] = quality_gate(rq,cq)
            passed = checks['correctness'] and checks['prefix_sampling'] and checks['prefill_full_scope'] and checks['kl']['passed'] and checks['qeval']['passed'] and checks['prose_c1_median_tps'] >= self.args.min_prose_tps
            self.report.update(status='PASS' if passed else 'HOLD', complete=True,
                               candidate_retained=passed, decision='bounded release screen; no intelligence-equivalence claim')
            self.receipt()
            if passed:
                self.report['serving'] = self.handoff(candidate)
                return 0
        except BaseException as exc:
            self.report.update(status='PARKED-IMPL' if not self.reference_ready else 'HOLD', complete=False,
                               error=f'{type(exc).__name__}: {exc}')
            self.receipt()
        finally:
            if not self.report.get('candidate_retained'):
                try:
                    self.stop()
                    if ref and self.reference_ready:
                        # Fresh cache and preserved prior runtime: restore the checked reference.
                        with (ref/'.env').open('a') as f:
                            f.write('\nOVERLAY_REMOTE='+shlex.quote(self.report['reference']['runtime']+'-restore')+'\n')
                        self.boot(ref,'reference-restore',recovery=True)
                        self.report['serving'] = self.handoff(ref)
                except BaseException as exc:
                    self.report['recovery_error'] = f'{type(exc).__name__}: {exc}'
                    # Do not leave a partially admitted boot unguarded.
                    try: self.stop()
                    except BaseException as stop_exc: self.report['stop_error'] = str(stop_exc)
            self.receipt()
        return 2

    def handoff(self, repo):
        deployment = json.loads((repo/'state/deployment.json').read_text())
        path = repo.parent/'SERVING.md'
        path.write_text('Serving on head loopback :8095. Watchdog PID '+str(self.proc.pid)+'.\n\n'
                        'Keep this workstation running. Foreground watchdog remains attached to the recorded deployment.\n'
                        'Stop with `cd '+str(repo)+' && ./start.sh stop`. Containers are preserved.\n'
                        'Fleet marker remains owned while serving; arrange an explicit handoff with the next coordinator.\n')
        return dict(checkout=str(repo), watchdog_pid=self.proc.pid, deployment=deployment, instructions=str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--reference-config', type=Path)
    ap.add_argument('--reference-repo', type=Path)
    ap.add_argument('--reference-rev', default='HEAD')
    ap.add_argument('--candidate-rev', default='HEAD')
    ap.add_argument('--base', default='http://localhost:18095')
    ap.add_argument('--dash', default='http://localhost:5555/api/sparks/configured-node/llm')
    ap.add_argument('--window-minutes', type=int, default=90)
    ap.add_argument('--aa-max', type=float, default=0.01)
    ap.add_argument('--kl-max', type=float, default=0.01)
    ap.add_argument('--min-prose-tps', type=float, default=50)
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(plan(), indent=2)); return 0
    if not args.config or not args.out:
        ap.error('--execute requires --config and a new --out directory')
    if not 30 <= args.window_minutes <= 90 or any(not (0 <= x < float('inf')) for x in (args.aa_max,args.kl_max,args.min_prose_tps)):
        ap.error('window must be 30..90min; limits finite and nonnegative')
    args.config = args.config.resolve()
    if args.reference_config: args.reference_config = args.reference_config.resolve()
    # Signal exceptions enter the same stop/restore path as ordinary failures.
    def interrupted(signum, frame): raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    return Gate(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
