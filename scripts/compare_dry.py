#!/usr/bin/env python3
"""Compare dry vectors to saved launch plus qualified profile deltas."""
import difflib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

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
    # Historical comparison uses a synthetic explicit configuration, never site defaults.
    saved = json.loads((ROOT/'docs/results/best-launch.json').read_text())
    config = "HOSTS=(rank0 rank1 rank2 rank3)\nIPS=("+' '.join(r['env']['VLLM_HOST_IP'] for r in saved['ranks'])+")\n"
    config += "FABRIC_IFACE=enp1s0f0np0\nIB_HCA=rocep1s0f0,roceP2p1s0f0\n"
    config += "MODEL_DIR=/srv/model\nDRAFT_DIR=/srv/draft\nNCCL_HOST_DIR=/srv/nccl\nOVERLAY_REMOTE=/srv/runtime\n"
    config += "RECIPE_DISPRAM=0\nGLM_KV_FORMAT=fp8\nRECIPE_MAX_MODEL_LEN=32768\nRECIPE_KV_HEAD_BYTES=1073741824\n"
    config += "GLM_DRAFT_HEAD=0\nGLM_DRAFT_HEAD_INIT=0\nGLM_DECODE_FAIR=0\nGLM_DECODE_FAIR_DECODE_STEPS=0\nGLM_DECODE_FAIR_CONTROL=\nGLM_MOE_DET_ALIGN=0\n"
    config += "GLM_ATTN_WEIGHTS=int8\nGLM_NVFP4_GROUPS=attn\nGLM_LOADER=\nGLM_FP4_RECENT_WINDOW=0\n"
    config += "NCCL_SHA256="+"0"*64+"\n"
    config += "GLM_PAD_HYGIENE="+run_env.get('GLM_PAD_HYGIENE', '0')+"\nGLM_INDEXER_SHORTCUT=0\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env') as fixture:
        fixture.write(config); fixture.flush()
        rendered = subprocess.check_output([str(ROOT/'start.sh'), 'serve'],
                      env=dict(run_env, RECIPE_CONFIG=fixture.name), text=True)
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
        expected['env'].update(json.loads((ROOT/'docs/results/boot-fast-profile.json').read_text())['added_env'])
        expected['env'].update(json.loads((ROOT/'docs/results/glue-lite-profile.json').read_text())['added_env'])
        expected['env'].update(json.loads((ROOT/'docs/results/kstop-profile.json').read_text())['added_env'])
        expected['env']['GLM_SPEC_SAMPLE'] = run_env.get('GLM_SPEC_SAMPLE', '0')
        stack = json.loads((ROOT/'docs/results/stack-1002b-profile.json').read_text())
        expected['env'].update(stack['changed_env'])
        expected['args'] = list(mtp['args'])
        for old, new in stack['flipped_flags'].items():
            expected['args'][expected['args'].index(old)] = new
        shape = stack['kstop_launch_shape']
        i = expected['args'].index('--speculative-config')+1
        expected['args'][i] = json.dumps({**json.loads(expected['args'][i]), **shape['speculative_config']})
        i = expected['args'].index('--compilation-config')+1
        expected['args'][i] = json.dumps({**json.loads(expected['args'][i]), **shape['compilation_config']})
        # K2 reuse default: exact c2 M6 from the existing capture bank.
        expected['env']['GLM_MTP_KSTOP_UNIFORM_BATCH'] = 'k2'
        expected['env']['GLM_PAD_HYGIENE'] = run_env.get('GLM_PAD_HYGIENE', '0')
        expected['env']['GLM_MTP_KSTOP_CAPTURE_LAYOUT'] = run_env.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT', 'reuse')
        # Release 2026-10-06: the in-process pre-capture floor became a profile key (earlier fixed at 10 GiB).
        expected['env']['GLM_PRECAPTURE_FLOOR_GIB'] = run_env.get('GLM_PRECAPTURE_FLOOR_GIB', '7.5')
        expected['args'][i] = json.dumps({**json.loads(expected['args'][i]),
                                       'cudagraph_capture_sizes': [1, 4, 12, 16],
                                       'max_cudagraph_capture_size': 16})
        expected['args'][expected['args'].index('--kv-cache-memory-bytes')+1] = shape['kv_cache_memory_bytes']
        expected['args'][expected['args'].index('--max-model-len')+1] = shape['max_model_len']
        expected['args'][expected['args'].index('--node-rank')+1] = str(rank['rank'])
        if rank['rank']:
            expected['args'].insert(expected['args'].index('--enable-prefix-caching'), '--headless')
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
The earlier duplicate speculative-config is removed. D2-W4 changes constructor
capacity512->4096 and adds drained serving cap2048 (results/w4-profile.json).
The native MTP delta in results/e2b-profile.json selects method=mtp, K2, TP4 draft,
compressed-tensors, FP8 draft KV, block64, async scheduling and graphs1/3/6/12;
GLM_MTP_FIX=1 and the DSpark SWA/low-memory hooks are disabled. The /draft mount
is omitted. results/short-dsa-profile.json adds GLM_INDEXER_SHORTCUT=1 and
results/dirty-l2-profile.json adds GLM_DIRTY_L2=discard. results/boot-fast-profile.json adds GLM_MTP_ONLY_LOAD=1 and
GLM_TARGET_SKIP_MTP=1 (native MTP shard selection);
results/glue-lite-profile.json adds the five glue-lite keys with every switch off and
results/kstop-profile.json adds GLM_MTP_KSTOP, its control path and GLM_MTP_KSTOP_UNIFORM_BATCH.
results/stack-1002b-profile.json then turns prefix caching on (one flag token), turns K-stop and its
uniform-batch policy on with the launcher's K3 / [1, 4, 16] / 544-block layout, and sets
GLM_INDEXER_SHORTCUT=0. Greedy drafting and standard rejection remain explicit. The DSpark
alternative preserves the historical K3 vector separately.

Docker's inherited CUDA/base-image environment is not a launcher override and is
outside this comparison. Image IDs differ across the four saved builds; each is
recorded in results/best-launch.json. The recipe rebuild uses the same pinned
base and RoCE source and requires its own image-source and fleet validation.
Host source paths and unique names differ; /model, /draft, /overlay and /cache
container paths retain their meanings. The three startup entry files contain
only used hooks. Inactive diagnostic branches were removed from the MLA adapter;
the split32 and larger-row unsplit kernel bodies preserve deployed arithmetic.
The pre-capture floor is the profile key GLM_PRECAPTURE_FLOOR_GIB (7.5 GiB; earlier fixed 10 GiB); per-load phase logging is omitted.

## Environment / argument diff

'''
    report += '```diff\n' + (''.join(diffs) if diffs else '# Empty: all 4 ranks match after the stated removals.\n') + '```\n\n## Rendered commands\n\n```bash\n' + rendered + '```\n'
    (ROOT/'docs/dry-vs-best.md').write_text(report)
    print('DRY VECTOR PASS: all 4 ranks' if not diffs else 'DRY VECTOR FAIL')
    if diffs:
        print(''.join(diffs))
        raise SystemExit(1)
