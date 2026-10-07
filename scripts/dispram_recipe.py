# SPDX-License-Identifier: Apache-2.0
"""Launcher side of the optional display-carveout KV profile (RECIPE_DISPRAM).

RECIPE_DISPRAM=0 (default) | auto (also 1) | require. When enabled, `serve` makes sure dispramd is up and
healthy on all four nodes BEFORE any engine starts:
  per node, first and unconditionally: `dispram.sh invalid-state` (a node invalidated in this boot by an
  Xid, SMMU fault, DRM user or daemon crash is UNSAFE whether or not its daemon still runs); then
  `dispram.sh watch`: any INVALID other than "daemon not running" is UNSAFE; a daemon that is simply not
  running is started once (installed unit via `sudo -n systemctl start dispramd.service`, else
  `dispram.sh start`, all refusals apply); a refused start is UNAVAILABLE; after a start, an unhealthy
  daemon is UNSAFE; finally `dispram.sh preborrow` (a failure means a lease, borrower or context exists:
  UNSAFE).
  any UNSAFE      -> abort before any engine starts (both modes): the node needs RUN.md section 4.
  all READY       -> engines get GLM_DISPRAM_KV=<mode> and the socket mount.
  any UNAVAILABLE -> require: abort; auto: borrowing explicitly disabled on EVERY rank (no socket mount,
                     GLM_DISPRAM_KV=0); note the bootfast contract then differs from the dispram layout.
`stop` leaves the daemon running and runs `dispram.sh postcheck` on every node (lease returned, no CUDA
context); a failing postcheck keeps the fleet lock (no re-lend before recovery).
dispramd/librmlist (kindling-spark-os, AGPL-3.0, kindlingai) are fetched and built unmodified on the
nodes by `dispram.sh fetch && dispram.sh build` (`start.sh dispram-setup`); nothing of them is in this repo.
"""
import concurrent.futures
import shlex
import subprocess
import time

HOME = '/srv/glm/dispram'


def mode(env):
    m = env.get('RECIPE_DISPRAM', '0')
    m = {'1': 'auto'}.get(m, m)
    if m not in ('0', 'auto', 'require'):
        raise ValueError('RECIPE_DISPRAM must be 0, 1/auto or require')
    return m


def home(env):
    return env.get('DISPRAM_HOME', HOME)


ACTIVE = None  # set by ensure(): True = borrow, False = explicitly disabled; None = not decided (DRY render)


def container_args(env, active=None):
    """Extra docker run arguments for every engine rank (identical on all ranks)."""
    m = mode(env)
    active = ACTIVE if active is None else active
    if m == '0':
        return [], {}
    if active is False:
        return [], {'GLM_DISPRAM_KV': '0'}
    return ['-v', home(env) + '/run:/run/dispram'], {'GLM_DISPRAM_KV': m}


def _run(host, command, timeout=120):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                       text=True, capture_output=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


READY, UNAVAILABLE, UNSAFE = 'ready', 'unavailable', 'unsafe'


def ensure_node(host, env, run=_run, sleep=time.sleep):
    ds = shlex.quote(home(env) + '/dispram.sh')
    if env.get('DISPRAM_ALLOW_RM_ALLOC_OOM') == '1':  # fix6, see dispram.sh KERNEL_FILTER
        ds = 'DISPRAM_ALLOW_RM_ALLOC_OOM=1 ' + ds
    rc, inv = run(host, ds + ' invalid-state')
    if rc != 0:
        return UNSAFE, 'invalidated in this boot: ' + inv[-300:]
    rc, out = run(host, ds + ' watch')
    if rc != 0:
        if 'daemon not running' not in out:
            return UNSAFE, 'unhealthy: ' + out[-300:]
        rc, _ = run(host, 'systemctl cat dispramd.service >/dev/null 2>&1')
        start = 'sudo -n systemctl start dispramd.service' if rc == 0 else ds + ' start'
        rc, out = run(host, start)
        if rc != 0:
            return UNAVAILABLE, 'start refused: ' + out[-300:]
        for _ in range(15):
            rc, out = run(host, ds + ' watch')
            if rc == 0:
                break
            sleep(2)
        if rc != 0:
            return UNSAFE, 'not healthy after start: ' + out[-300:]
    rc, out = run(host, ds + ' preborrow')
    return (READY, out[-300:]) if rc == 0 else (UNSAFE, 'preborrow: ' + out[-300:])


def ensure(hosts, env, run=_run, sleep=time.sleep):
    """Decide once for all ranks. Returns True (borrow) or False (explicitly disabled); raises when unsafe
    or when require cannot be met. Sets ACTIVE for container_args()."""
    global ACTIVE
    m = mode(env)
    if m == '0':
        ACTIVE = False
        return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        res = dict(zip(hosts, pool.map(lambda h: ensure_node(h, env, run, sleep), hosts)))
    unsafe = {h: why for h, (st, why) in res.items() if st == UNSAFE}
    if unsafe:
        raise RuntimeError('dispram UNSAFE on ' + repr(unsafe) + '; not booting (RUN.md section 4)')
    missing = {h: why for h, (st, why) in res.items() if st != READY}
    if not missing:
        ACTIVE = True
        print('DISPRAM READY on all nodes', flush=True)
        return True
    if m == 'require':
        raise RuntimeError('RECIPE_DISPRAM=require but dispramd is unavailable: ' + repr(missing))
    ACTIVE = False
    print('DISPRAM unavailable on ' + repr(missing) + '; borrowing disabled on every rank '
          '(no socket mount, GLM_DISPRAM_KV=0), plain 2 GiB pool', flush=True)
    return False


def postcheck(hosts, env, run=_run):
    if mode(env) == '0':
        return
    bad = {}
    for h in hosts:
        pre = 'DISPRAM_ALLOW_RM_ALLOC_OOM=1 ' if env.get('DISPRAM_ALLOW_RM_ALLOC_OOM') == '1' else ''
        rc, out = run(h, pre + shlex.quote(home(env) + '/dispram.sh') + ' postcheck')
        if rc != 0:
            bad[h] = out[-300:]
    if bad:
        raise RuntimeError('dispram postcheck failed (lease or CUDA context not released); lock retained, '
                           'see dispram recovery: ' + repr(bad))
    print('dispram postcheck: carveout released on all nodes; daemons left running', flush=True)


def setup(hosts, env, root, run=_run):
    """One-time per node: copy dispram.sh, fetch pinned kindling + NVIDIA headers, build unmodified."""
    for h in hosts:
        subprocess.run(['ssh', h, 'mkdir -p ' + shlex.quote(home(env))], check=True)
        subprocess.run(['rsync', '-a', str(root / 'scripts/dispram.sh'), h + ':' + home(env) + '/dispram.sh'], check=True)
        ds = shlex.quote(home(env) + '/dispram.sh')
        rc, out = run(h, ds + ' fetch && ' + ds + ' build', timeout=900)
        print(h, out[-400:], flush=True)
        if rc != 0:
            raise RuntimeError('dispram setup failed on ' + h)
