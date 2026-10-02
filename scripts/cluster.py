#!/usr/bin/env python3
"""Workstation launcher for four TP ranks; dry rendering has no fleet access."""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(os.environ['RECIPE_ROOT'])
ENV = os.environ
HOSTS = ENV['RECIPE_HOSTS'].split()
IPS = ENV['RECIPE_IPS'].split()
IMAGES = ENV.get('RECIPE_IMAGES', '').split() or [ENV['IMAGE']] * 4
STATE = ROOT / 'state/deployment.json'
SERVE_ARGS = Path(ENV.get('RECIPE_SERVE_ARGS', ROOT / 'profiles/serve-args.json'))
PROFILE_KEYS = re.findall(r'^export (\w+)=', (ROOT / 'profiles/current.env').read_text(), re.M)
GiB = 1 << 30


def validate():
    if len(HOSTS) != 4 or len(IPS) != 4 or len(IMAGES) != 4:
        raise ValueError('Exactly four hosts, fabric addresses and images required')
    if len(set(HOSTS)) != 4 or len(set(IPS)) != 4:
        raise ValueError('Hostnames and fabric addresses must be unique')
    for host in HOSTS:
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host):
            raise ValueError('Invalid SSH host')
    for key in ('MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE'):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', ENV[key]):
            raise ValueError(key + ' requires an absolute path without shell syntax')
    if len({ENV[k] for k in ('MODEL_DIR', 'DRAFT_DIR', 'NCCL_HOST_DIR', 'OVERLAY_REMOTE')}) != 4:
        raise ValueError('Model, draft, NCCL and runtime paths must be distinct')
    nccl_hashes()


def nccl_hashes():
    # One hash for identical binaries, or four comma-separated hashes in HOSTS order.
    values = ENV['NCCL_SHA256'].split(',')
    values = values * 4 if len(values) == 1 else values
    if len(values) != 4 or any(not re.fullmatch(r'[0-9a-f]{64}', v) for v in values):
        raise ValueError('NCCL_SHA256 must be one SHA256 or four comma-separated SHA256 values in host order')
    return values


def remote(rank, command, timeout=45):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                    HOSTS[rank], command], text=True, timeout=timeout).strip()


def rank_env(rank):
    env = {k: ENV[k] for k in PROFILE_KEYS}
    env.update(VLLM_HOST_IP=IPS[rank], NCCL_SOCKET_IFNAME='=' + ENV['FABRIC_IFACE'],
               GLOO_SOCKET_IFNAME=ENV['FABRIC_IFACE'], NCCL_IB_HCA='=' + ENV['IB_HCA'],
               B12X_ROCE_HCA=ENV['IB_HCA'])
    return env


def rank_args(rank):
    args = json.loads(SERVE_ARGS.read_text())
    args[args.index('--node-rank') + 1] = str(rank)
    args[args.index('--master-addr') + 1] = IPS[0]
    if rank:
        # Preserve the saved worker vector's order.
        args.insert(args.index('--no-enable-prefix-caching'), '--headless')
    return args



def prefill_control(chunk, sequence):
    path = ENV.get('GLM_W2_PREFILL_CONTROL')
    if not path:
        return
    if path != '/cache/d2w2-prefill-control.json' or chunk not in (512, 2048, 4096):
        raise ValueError('Unsupported prefill control path/cap')
    control = dict(schema=1, chunk=chunk, sequence=sequence)
    code = 'import json,os,pathlib; p=pathlib.Path(PATH); q=p.with_name(p.name+".new"); q.write_text(PAYLOAD); os.replace(q,p); print(p.read_text())'
    code = code.replace('PATH', repr(ENV['OVERLAY_REMOTE'] + '/cache/d2w2-prefill-control.json')).replace('PAYLOAD', repr(json.dumps(control)))
    for rank in range(4):
        if json.loads(remote(rank, 'python3 -c ' + shlex.quote(code))) != control:
            raise RuntimeError('Prefill readback disagreement')


def select_serving_prefill():
    if not ENV.get('GLM_W2_PREFILL_CONTROL'):
        return
    cap = int(ENV.get('GLM_W2_PREFILL_CHUNK', '512'))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        metrics = remote(0, 'curl -fsS -m5 http://127.0.0.1:8095/metrics')
        counts = []
        for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            vals = [float(line.split()[-1]) for line in metrics.splitlines() if line.startswith(key+'{') or line.startswith(key+' ')]
            if not vals:
                raise RuntimeError('Missing drain metric')
            counts.append(sum(vals))
        if counts == [0, 0]:
            prefill_control(cap, time.time_ns())
            print('Serving prefill cap %d; constructor capacity4096' % cap, flush=True)
            return
        time.sleep(.25)
    raise RuntimeError('Prefill cap selection did not drain in30s')


def uses_external_draft():
    args = json.loads(SERVE_ARGS.read_text())
    return json.loads(args[args.index('--speculative-config')+1]).get('model') == '/draft'


def docker_command(rank, ctn):
    cmd = ['docker', 'run', '-d', '--restart', 'no', '--name', f'{ctn}-r{rank}',
           '--gpus', 'all', '--network', 'host', '--ipc', 'host', '--device', '/dev/infiniband',
           '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
           '--ulimit', 'nofile=1048576:1048576', '--memory', '112g', '--memory-swap', '112g',
           '--entrypoint', 'vllm']
    mounts = [(ENV['MODEL_DIR'], '/model', True),
                              (ENV['OVERLAY_REMOTE'], '/overlay', True),
                              (ENV['OVERLAY_REMOTE'] + '/cache', '/cache', False),
                              (ENV['NCCL_HOST_DIR'], '/opt/nccl', True)]
    if uses_external_draft():
        mounts.append((ENV['DRAFT_DIR'], '/draft', True))
    for source, dest, ro in mounts:
        cmd += ['-v', source + ':' + dest + (':ro' if ro else '')]
    for key, value in sorted(rank_env(rank).items()):
        cmd += ['-e', key + '=' + value]
    return cmd + [IMAGES[rank]] + rank_args(rank)


# Cold FlashInfer JIT modules that the serving process would otherwise compile with nvcc right after
# graph capture (sampling, about 2-2.7 GiB of host memory for about 37 s) and during the first FULL graph
# (batch MLA). They are built into the fresh per-deployment cache from the same image and environment
# before the model containers start; the serving build then finds them up to date and runs no nvcc.
JIT_PREP_TIMEOUT_S = 600
JIT_PREP_MLA = ('batch_mla_attention_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32'
                '_head_dim_ckv_512_head_dim_kpe_64_profiler_False')
# One line, so DRY output stays one command per line.
JIT_PREP_CODE = '; '.join([
    'import torch',
    'from flashinfer.jit.attention import gen_batch_mla_module',
    'from flashinfer.jit.sampling import gen_sampling_module',
    "specs = [gen_sampling_module(), gen_batch_mla_module('fa2', torch.bfloat16, torch.float8_e4m3fn, "
    "torch.bfloat16, torch.int32, 512, 64, False)]",
    f"assert [s.name for s in specs] == ['sampling', {JIT_PREP_MLA!r}], [s.name for s in specs]",
    '[s.build(need_lock=True) for s in specs]',
    'assert all(s.is_compiled for s in specs), [s.name for s in specs if not s.is_compiled]',
    "print(chr(10).join('JIT PREP %s %s' % (s.name, s.get_library_path()) for s in specs), flush=True)"])


def jit_prep_command(rank, ctn):
    # Same image, environment and /cache mount as the rank's model container. PYTHONPATH is
    # emptied so the overlay hooks do not run; no GPU, no network. The in-container timeout
    # ends the container (and its compilers) itself; --rm leaves nothing behind.
    env = rank_env(rank)
    env['PYTHONPATH'] = ''
    cmd = ['docker', 'run', '--rm', '--name', f'{ctn}-jitprep-r{rank}', '--network', 'none',
           '--memory', '16g', '--memory-swap', '16g',
           '-v', ENV['OVERLAY_REMOTE'] + '/cache:/cache', '-v', ENV['NCCL_HOST_DIR'] + ':/opt/nccl:ro']
    for key, value in sorted(env.items()):
        cmd += ['-e', key + '=' + value]
    return cmd + ['--entrypoint', 'timeout', IMAGES[rank], '-k', '10', str(JIT_PREP_TIMEOUT_S),
                  'python3', '-c', JIT_PREP_CODE]


def jit_prep(ctn):
    def prep_rank(rank):
        out = remote(rank, shlex.join(jit_prep_command(rank, ctn)), timeout=JIT_PREP_TIMEOUT_S + 60)
        built = [line for line in out.splitlines() if line.startswith('JIT PREP ')]
        if len(built) != 2:
            raise RuntimeError('JIT prep incomplete: ' + out[-400:])
        return built
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Fail closed: every rank must finish before compaction and launch.
        futures = [pool.submit(prep_rank, rank) for rank in range(4)]
        errors = []
        for rank, future in enumerate(futures):
            try:
                for line in future.result():
                    print(HOSTS[rank], line, flush=True)
            except Exception as exc:
                errors.append((rank, str(exc)[-400:]))
    if errors:
        raise RuntimeError('JIT prep failed: ' + repr(errors))
    print('JIT PREP PASS', flush=True)


ARG_MAX_STRLEN = 131072   # Linux MAX_ARG_STRLEN: one argv string, here the remote shell's -c program


def verify_command(path, manifest, mode='cached'):
    # Full CPU checksum verification streams each file, never initializes CUDA. The manifest
    # is embedded once and bound to a name, so the single remote argument stays small.
    if mode not in ('cached', 'full'):
        raise ValueError('VERIFY_WEIGHTS must be cached or full')
    verify = (ROOT / 'scripts/weights.py').read_text().split("if __name__ == '__main__':")[0]
    receipt = 'None' if mode == 'full' else 'receipt_path(ROOT_DIR, MANIFEST)'
    call = f'\nMANIFEST = {manifest!r}\nROOT_DIR = Path({path!r})\nverify(ROOT_DIR, MANIFEST, {receipt})\n'
    command = 'nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -c ' + shlex.quote(verify + call)
    if len(command.encode()) >= ARG_MAX_STRLEN - 4096:
        raise RuntimeError('weight verification program too large for one remote argument')
    return command


def preflight(ctn):
    # Two independent single-thread checksum streams; preserve every rank receipt.
    def check_rank(rank):
        code = '''import json, pathlib, subprocess
p=pathlib.Path
image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
assert image['Architecture']=='arm64', 'linux/arm64 image required'
assert not (HEAD and p.home().joinpath('fleet_busy').exists()), 'fleet is owned'
assert not subprocess.check_output(['docker','ps','-a','--filter','name=^'+NAME+'$','--format','{{.Names}}']).strip(), 'name already used'
running=json.loads(subprocess.check_output(['docker','ps','-q']).decode() and subprocess.check_output(['docker','inspect']+subprocess.check_output(['docker','ps','-q']).decode().split()) or '[]')
for c in running:
 req=c['HostConfig'].get('DeviceRequests') or []
 assert not any('gpu' in caps for d in req for caps in d.get('Capabilities',[])), 'GPU container is running: '+c['Name']
 assert not any(m['Source']==OVERLAY for m in c.get('Mounts',[])), 'runtime path in use'
assert not p(OVERLAY).exists(), 'runtime path already exists; use a fresh path'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(), 'GPU process is running'
assert p('/dev/infiniband').is_dir(), 'RDMA device missing'
for d in ([MODEL,DRAFT] if EXTERNAL_DRAFT else [MODEL]):
 assert p(d).joinpath('config.json').is_file(), 'weights missing'
 assert not list(p(d).rglob('*.part')), 'partial downloads present'
assert len(list(p(MODEL).glob('*.safetensors')))==282, 'expected 282 target shards'
mem=dict((l.split(':')[0],int(l.split()[1])) for l in p('/proc/meminfo').read_text().splitlines())
assert mem['MemAvailable']>=110*1048576, 'preboot headroom below 110 GiB'
assert p('/usr/local/sbin/spark-compact-mem.sh').is_file(), 'compaction helper missing'
assert p(NCCL).joinpath('libnccl.so.2.30.7').is_file(), 'NCCL missing'
print(json.dumps(dict(architecture=image['Architecture'],image=image['Id'],MemAvailable_kB=mem['MemAvailable'])))
'''
        setup = '\n'.join(k + '=' + repr(v) for k, v in dict(IMAGE=IMAGES[rank], NAME=f'{ctn}-r{rank}',
                      OVERLAY=ENV['OVERLAY_REMOTE'], MODEL=ENV['MODEL_DIR'], DRAFT=ENV['DRAFT_DIR'],
                      NCCL=ENV['NCCL_HOST_DIR'], EXTERNAL_DRAFT=uses_external_draft(), HEAD=rank == 0).items())
        print(HOSTS[rank], remote(rank, 'python3 -c ' + shlex.quote(setup + '\n' + code)))
        checkpoints = [('target', ENV['MODEL_DIR'])]
        if uses_external_draft():
            checkpoints.append(('drafter', ENV['DRAFT_DIR']))
        for kind, path in checkpoints:
            manifest = json.loads((ROOT / f'manifests/{kind}.json').read_text())
            result = remote(rank, verify_command(path, manifest, ENV.get('VERIFY_WEIGHTS', 'cached')), timeout=1200)
            print(HOSTS[rank], result, flush=True)
        sha = remote(rank, 'sha256sum ' + shlex.quote(ENV['NCCL_HOST_DIR'] + '/libnccl.so.2.30.7')).split()[0]
        if sha != nccl_hashes()[rank]:
            raise RuntimeError('NCCL binary hash mismatch on ' + HOSTS[rank])
        addresses = remote(rank, 'ip -4 -o addr show dev ' + shlex.quote(ENV['FABRIC_IFACE']))
        if IPS[rank] + '/' not in addresses:
            raise RuntimeError('Fabric IP/interface mismatch on ' + HOSTS[rank])
        remote(rank, 'sudo -n -l /usr/local/sbin/spark-compact-mem.sh >/dev/null')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Await every task, including failures, before admitting any container.
        results = [pool.submit(check_rank, rank) for rank in range(4)]
        errors = []
        for rank, future in enumerate(results):
            try:
                future.result()
            except Exception as exc:
                errors.append((rank, str(exc)))
        if errors:
            raise RuntimeError('Preflight failed: ' + repr(errors))
    print('PREFLIGHT PASS', flush=True)


def lock(token):
    remote(0, 'set -C; printf %s ' + shlex.quote(token) + ' > "$HOME/fleet_busy"')


def unlock(token):
    # Only unlink the exact lock owned by this deployment.
    remote(0, 'python3 -c ' + shlex.quote('from pathlib import Path; p=Path.home()/"fleet_busy"; '
                    f'\np.unlink() if p.exists() and p.read_text()=={token!r} else None'))


def stop(deployment):
    def stop_rank(rank):
        remote(rank, 'docker stop -t 60 ' + shlex.quote(deployment['ctn'] + f'-r{rank}'), timeout=80)
        status = remote(rank, 'docker inspect -f ' + shlex.quote('{{.State.Running}}') + ' ' + shlex.quote(deployment['ctn'] + f'-r{rank}'))
        if status != 'false':
            raise RuntimeError('container still running')
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stop_rank, rank) for rank in range(4)]
        for rank, future in enumerate(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append((rank, str(exc)))
    if failed:
        raise RuntimeError('Stop incomplete; lock retained: ' + repr(failed))
    unlock(deployment['token'])
    print('Four containers stopped and preserved; owned lock released', flush=True)


SAMPLE_CODE = '''import json,pathlib,subprocess
p=pathlib.Path
m={l.split(':')[0]:int(l.split()[1]) for l in p('/proc/meminfo').read_text().splitlines()}
v={l.split()[0]:int(l.split()[1]) for l in p('/proc/vmstat').read_text().splitlines()}
d=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
logs=subprocess.check_output(['docker','logs','--tail','150',NAME],stderr=subprocess.STDOUT).decode(errors='replace')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits']).decode()
print(json.dumps(dict(mem=m,vm=v,state=d['State'],restarts=d['RestartCount'],logs=logs,gpu=gpu)))
'''


def health():
    return remote(0, "curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8095/health || true") == '200'


def fatal_line(line):
    # Log-level ERROR is matched case-sensitively: status JSON such as RoCE's
    # "error_hca": 0 counters in the readiness line is not an error.
    if 'import_utils' in line and 'WARNING' in line:
        return False
    return bool(re.search(r'CUDA error|illegal memory access|device-side assert|Traceback|EngineDead|\bnan\b|\binf\b', line, re.I)
                or re.search(r'Worker.*(\bERROR\b|\bfailed\b|\bFailed\b)', line))


def inspect_samples(samples, swap_base, growing):
    available = []
    for rank, sample in enumerate(samples):
        mem = sample['mem']; available.append(mem['MemAvailable'])
        if sample.get('restarts', 0):
            raise RuntimeError(f'rank {rank} container restarted')
        if not sample['state']['Running'] or sample['state']['OOMKilled']:
            raise RuntimeError(f'rank {rank} exited or was OOM killed')
        if mem['MemAvailable'] < 6 * 1048576:
            raise RuntimeError(f'rank {rank} MemAvailable below 6 GiB')
        used = mem['SwapTotal'] - mem['SwapFree']
        swap_base.setdefault(rank, used)
        growing[rank] = growing.get(rank, 0) + 1 if used - swap_base[rank] >= 64 * 1024 else 0
        if growing[rank] >= 3:
            raise RuntimeError(f'rank {rank} sustained swap growth >=64 MiB')
        errors = [line for line in sample['logs'].splitlines() if fatal_line(line)]
        if errors:
            raise RuntimeError(f'rank {rank} runtime error: {errors[0]}')
    return available


def capture_headroom(samples, avail):
    # The in-process hook refuses capture below 10 GiB before it starts. The capture
    # progress lines stay in the log tail until health 200, so this 8 GiB floor covers
    # capture, the post-capture warm-up and API start. The cold FlashInfer JIT builds
    # that dipped below it are made before launch by jit_prep().
    capturing = any('Capturing CUDA graph' in s['logs'] or 'Capturing cudagraph' in s['logs'] for s in samples)
    if capturing and min(avail) < 8 * 1048576:
        raise RuntimeError('capture headroom below 8 GiB')


def progress_fingerprint(samples, metrics=None):
    if metrics is not None:
        return '\n'.join(l for l in metrics.splitlines() if l.startswith('vllm:generation_tokens_total') or l.startswith('vllm:prompt_tokens_total'))
    # Health/access logs cannot keep a stalled cold boot alive.
    return '\n'.join(line for sample in samples for line in sample['logs'].splitlines()
                     if not re.search(r'\"(GET|POST) |/health|/metrics|Avg prompt throughput|Avg generation throughput', line))


def monitor(deployment, boot=False):
    started = time.monotonic()
    steady = None
    swap_base = {int(k): v for k, v in deployment.get('swap_base', {}).items()}
    growing = {}
    progress = {}
    ready = False
    directory = ROOT / 'logs' / deployment['ctn']
    directory.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            now = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(remote, rank, 'python3 -c ' + shlex.quote(
                    'NAME=' + repr(deployment['ctn'] + f'-r{rank}') + '\n' + SAMPLE_CODE)) for rank in range(4)]
                samples = [json.loads(f.result()) for f in futures]
            with (directory / 'watchdog.jsonl').open('a') as out:
                out.write(json.dumps(dict(elapsed_s=now-started, samples=samples)) + '\n')
            avail = inspect_samples(samples, swap_base, growing)
            if boot and not ready:
                capture_headroom(samples, avail)
            if boot and now-started > 900:
                raise RuntimeError('boot exceeded 900 seconds')
            if boot and not ready:
                ready = health()
                if ready:
                    print('health 200 after %.1f s' % (now-started), flush=True)
            if ready and boot:
                steady = (steady or now) if min(avail) >= 8 * 1048576 else None
                if steady is not None and now-steady >= 60:
                    print('Admission PASS: >=8 GiB on all ranks for 60 s', flush=True)
                    select_serving_prefill()
                    boot = False
            if not boot:
                metrics = remote(0, 'curl -fsS -m 5 http://127.0.0.1:8095/metrics')
                fingerprint = progress_fingerprint(samples, metrics)
            else:
                fingerprint = progress_fingerprint(samples)
            if progress.get('value') != fingerprint:
                progress = dict(value=fingerprint, time=now)
            busy = any(float(v.strip()) > 20 for sample in samples for v in sample['gpu'].splitlines())
            if busy and now-progress.get('time', now) > 180:
                raise RuntimeError('GPU busy without progress for 180 seconds')
            time.sleep(10)
    except BaseException:
        # SSH loss is fatal; no automatic restart or power operation.
        stop(deployment)
        raise


def serve(ctn):
    if STATE.exists():
        raise RuntimeError('Local deployment state exists; use status/stop before another serve')
    preflight(ctn)
    token = ctn + ':' + uuid.uuid4().hex
    lock(token)
    deployment = dict(ctn=ctn, token=token, hosts=HOSTS, ips=IPS, images=IMAGES,
                      runtime=ENV['OVERLAY_REMOTE'], profile=ENV.get('RECIPE_PROFILE', 'native-mtp-k2'), created=time.time())
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(deployment, indent=2) + '\n')
    launched = []
    try:
        for rank in range(4):
            remote(rank, 'mkdir ' + shlex.quote(ENV['OVERLAY_REMOTE']))
            subprocess.run(['rsync', '-a', '--exclude=__pycache__', str(ROOT / 'overlay') + '/',
                            HOSTS[rank] + ':' + ENV['OVERLAY_REMOTE'] + '/'], check=True)
            remote(rank, 'mkdir ' + shlex.quote(ENV['OVERLAY_REMOTE'] + '/cache'))
        # Before compaction, so the compilers' memory is gone before the preboot floor check.
        jit_prep(ctn)
        for rank in range(4):
            remote(rank, 'sudo -n /usr/local/sbin/spark-compact-mem.sh', timeout=180)
            # Compaction must succeed and must preserve the 110 GiB preboot floor.
            available = int(remote(rank, "awk '/MemAvailable/{print $2}' /proc/meminfo"))
            if available < 110 * 1048576:
                raise RuntimeError('Preboot memory floor after compaction')
        prefill_control(512, 0)
        deployment['swap_base'] = {rank: int(remote(rank, "awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo")) for rank in range(4)}
        STATE.write_text(json.dumps(deployment, indent=2) + '\n')
        for rank in (3, 2, 1, 0):
            print(HOSTS[rank], remote(rank, shlex.join(docker_command(rank, ctn))), flush=True)
            launched.append(rank)
    except BaseException:
        stopped = True
        for rank in launched:
            try:
                remote(rank, 'docker stop -t 60 ' + shlex.quote(f'{ctn}-r{rank}'), timeout=80)
            except Exception:
                stopped = False
        if stopped:
            unlock(token)
        raise
    print('Watchdog active in foreground; use another terminal for status and benchmarks. Ctrl-C stops the deployment.', flush=True)
    monitor(deployment, boot=True)


def main():
    validate()
    command = sys.argv[1] if len(sys.argv) > 1 else 'status'
    ctn = ENV.get('CTN', 'glm53full-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    if not re.fullmatch(r'glm53full-[A-Za-z0-9_-]+', ctn):
        raise ValueError('CTN must begin glm53full- and contain only letters, digits, _ or -')
    if ENV.get('DRY') == '1':
        if command != 'serve':
            raise ValueError('DRY=1 supports serve only')
        for rank in range(4):
            print(f'# jit-prep rank={rank} host={HOSTS[rank]}')
            print(shlex.join(jit_prep_command(rank, ctn)))
        for rank in (3, 2, 1, 0):
            print(f'# rank={rank} host={HOSTS[rank]}')
            print(shlex.join(docker_command(rank, ctn)))
        return
    if command == 'preflight':
        preflight(ctn)
        return
    if command == 'serve':
        serve(ctn)
        return
    deployment = json.loads(STATE.read_text())
    if deployment['hosts'] != HOSTS or deployment['runtime'] != ENV['OVERLAY_REMOTE']:
        raise RuntimeError('Saved deployment differs from .env; restore its host/runtime settings')
    if command == 'stop':
        stop(deployment)
        STATE.rename(STATE.with_name('stopped-' + deployment['ctn'] + '.json'))
    elif command == 'watch':
        monitor(deployment)
    elif command == 'status':
        for rank in range(4):
            print(HOSTS[rank], remote(rank, 'docker inspect -f ' + shlex.quote('{{.State.Running}} {{.State.OOMKilled}}') + ' ' + shlex.quote(deployment['ctn'] + f'-r{rank}')))
        print('health', health())
    elif command == 'logs':
        rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if rank not in range(4):
            raise ValueError('rank must be 0..3')
        print(remote(rank, 'docker logs --tail 100 ' + shlex.quote(deployment['ctn'] + f'-r{rank}') + ' 2>&1'))
    else:
        raise ValueError('usage: start.sh preflight|serve|status|logs [rank]|watch|stop')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
