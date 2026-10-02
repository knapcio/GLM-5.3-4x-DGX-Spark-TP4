#!/usr/bin/env python3
"""Compare dry vectors to saved launch plus qualified profile deltas."""
import difflib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIAG = {'GLM_W4_TAIL', 'GLM_W6_GRAPH_RECEIPTS'}


def clean_args(args):
    out = []
    i = 0
    while i < len(args):
        item = args[i]
        if item.startswith('--profiler-config.'):
            i += 1
            continue
        if item == '--speculative-config':
            if 'kv_cache_dtype' in json.loads(args[i+1]):
                out += args[i:i+2]
            i += 2
            continue
        out.append(item)
        i += 1
    return out


def compare():
    run_env = dict(os.environ, DRY='1', CTN='glm53full-dry-verification')
    rendered = subprocess.check_output([str(ROOT/'start.sh'), 'serve'], env=run_env, text=True)
    actual = {}
    for line in rendered.splitlines():
        if not line.startswith('docker run ') or line.startswith('docker run --rm '):
            continue   # the --rm lines are the JIT prep containers, checked by tests/test_launcher.py
        cmd = shlex.split(line)
        env = dict(cmd[i+1].split('=', 1) for i, token in enumerate(cmd) if token == '-e')
        # The fixed image is followed by vLLM's serve command.
        args = cmd[cmd.index('serve'):]
        rank = int(args[args.index('--node-rank')+1])
        actual[rank] = dict(env=env, args=args)
    if set(actual) != set(range(4)):
        raise AssertionError('Four distinct rendered ranks required')
    saved = json.loads((ROOT/'docs/results/best-launch.json').read_text())
    diffs = []
    for rank in saved['ranks']:
        expected = dict(env={k:v for k,v in rank['env'].items() if k not in DIAG},
                        args=clean_args(rank['args']))
        selected = json.loads((ROOT/'docs/results/w4-profile.json').read_text())
        for key, value in selected['changed_arguments'].items():
            expected['args'][expected['args'].index(key)+1] = value
        expected['env'].update(selected['added_env'])
        mtp = json.loads((ROOT/'docs/results/e2b-profile.json').read_text())
        expected['env'].update(mtp['changed_env'])
        expected['env'].update(json.loads((ROOT/'docs/results/short-dsa-profile.json').read_text())['added_env'])
        expected['env'].update(json.loads((ROOT/'docs/results/dirty-l2-profile.json').read_text())['added_env'])
        expected['args'] = list(mtp['args'])
        expected['args'][expected['args'].index('--node-rank')+1] = str(rank['rank'])
        if rank['rank']:
            expected['args'].insert(expected['args'].index('--no-enable-prefix-caching'), '--headless')
        a = json.dumps(expected, indent=2, sort_keys=True).splitlines(True)
        b = json.dumps(actual[rank['rank']], indent=2, sort_keys=True).splitlines(True)
        diff = ''.join(difflib.unified_diff(a, b, fromfile='best-clean-r%d' % rank['rank'], tofile='dry-r%d' % rank['rank']))
        if diff:
            diffs.append(diff)
    return rendered, diffs


if __name__ == '__main__':
    rendered, diffs = compare()
    report = '''# Dry launch versus the selected control

The Docker commands below were rendered locally by `DRY=1 ./start.sh serve`: four short
`docker run --rm` JIT prep commands (same image and environment with PYTHONPATH empty, no GPU
or network; they build FlashInfer's sampling and batch MLA modules into the fresh per-deployment
cache before the model containers start), then the four model commands compared here.
No SSH, container, GPU, cache seed or fleet mutation occurs in dry mode.
`scripts/compare_dry.py` parses each command with shlex and compares every explicit
container environment key and ordered vLLM argument against the saved per-rank
Boot B BEST_FULL_GLM.json vectors (source SHA256 in results/best-launch.json).

Allowed removals: GLM_W4_TAIL, GLM_W6_GRAPH_RECEIPTS, and six profiler-config
arguments. PROFILE=1 belongs to the diagnostic wrapper, not the container env.
The earlier duplicate speculative-config is removed. The W4 profile changes constructor
capacity512->4096 and adds drained serving cap2048 (results/w4-profile.json).
The native MTP delta in results/e2b-profile.json selects method=mtp, K2, TP4 draft,
compressed-tensors, FP8 draft KV, block64, async scheduling and graphs1/3/6/12;
GLM_MTP_FIX=1 and the DSpark SWA/low-memory hooks are disabled. The /draft mount
is omitted. results/short-dsa-profile.json adds GLM_INDEXER_SHORTCUT=1 and
results/dirty-l2-profile.json adds GLM_DIRTY_L2=discard. Greedy drafting and standard rejection remain explicit. The DSpark
alternative preserves the historical K3 vector separately.

Docker's inherited CUDA/base-image environment is not a launcher override and is
outside this comparison. Image IDs differ across the four saved builds; each is
recorded in results/best-launch.json. The recipe rebuild uses the same pinned
base and RoCE source and requires its own image-source and fleet validation.
Host source paths and unique names differ; /model, /draft, /overlay and /cache
container paths retain their meanings. The three startup entry files contain
only used hooks. Inactive diagnostic branches were removed from the MLA adapter;
the split32 and larger-row unsplit kernel bodies preserve deployed arithmetic.
The pre-capture 10 GiB floor remains; per-load phase logging is omitted.

## Environment / argument diff

'''
    report += '```diff\n' + (''.join(diffs) if diffs else '# Empty: all 4 ranks match after the stated removals.\n') + '```\n\n## Rendered commands\n\n```bash\n' + rendered + '```\n'
    (ROOT/'docs/dry-vs-best.md').write_text(report)
    print('DRY VECTOR PASS: all 4 ranks' if not diffs else 'DRY VECTOR FAIL')
    if diffs:
        print(''.join(diffs))
        raise SystemExit(1)
