#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release guard / exact stress on an already owned boot. No boot or network by default."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
import stress_step as S

ROOT = Path(__file__).resolve().parents[1]


def geometry(head_bytes, max_model_len):
    """Shared-pool c4 (four distinct prompts resident in one quarter of the pool each) + one max-length request."""
    from fp4_kv_layout import layout
    l = layout('fp4x', head_bytes)
    if max_model_len % 64 or not S.OUTPUT + 64 <= max_model_len <= int(l['max_model_len']):
        raise ValueError('max_model_len must be on the 64-token grid and fit the head')
    c4_length = min(max_model_len, ((l['blocks'] - 1)//4 - 1)*64)
    return dict(ordinary_bytes=head_bytes, max_model_len=max_model_len, layout=l, c4_mode='pool-quarter',
                c4_total_tokens_per_request=c4_length, c4_prompt_tokens=c4_length-S.OUTPUT, decode_tokens=S.OUTPUT,
                c4_required_blocks=1 + 4*math.ceil((c4_length + 3)/64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('guard', 'stress'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--boot', required=True)
    p.add_argument('--base', default='http://127.0.0.1:8095')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=4500)
    p.add_argument('--handoff-file', type=Path)
    p.add_argument('--max-model-len', type=int, default=int(os.environ.get('RECIPE_MAX_MODEL_LEN') or 262144))
    p.add_argument('--head-bytes', type=int, default=int(os.environ.get('RECIPE_KV_HEAD_BYTES') or 6318718976))
    # Release values (profiles/current.env); the 165312 gate used guard 8.05 and stress 8.4.
    p.add_argument('--live-floor', type=float, default=float(os.environ.get('RECIPE_LIVE_FLOOR_GIB') or 4.5))
    p.add_argument('--stress-floor', type=float, default=float(os.environ.get('RECIPE_STRESS_FLOOR_GIB') or 4.5))
    a = p.parse_args()
    g = geometry(a.head_bytes, a.max_model_len)
    single_prompt = a.max_model_len - S.OUTPUT
    if not a.execute:
        print(json.dumps(dict(mode=a.mode, geometry=g, single_prompt=single_prompt, output_tokens=S.OUTPUT,
                              floor_GiB=a.live_floor, stress_floor_GiB=a.stress_floor, seconds=a.seconds)))
        return 0
    if not os.environ.get('INVOCATION_ID') or len(os.environ.get('RECIPE_HOSTS', '').split()) != 4:
        p.error('execute requires coordinator systemd user unit and RECIPE_HOSTS in rank order')
    if not 1 <= a.seconds <= 4500:
        p.error('seconds must be 1..4500')
    deployment = json.loads((ROOT/'state/deployment.json').read_text())
    if deployment['ctn'] != a.boot:
        p.error('boot differs from the owned deployment')
    a.out.mkdir(parents=True, exist_ok=False)
    mon = S.Monitor(a.out, a.boot)
    mon.min_floor_gib = mon.rank0_floor_gib = a.live_floor
    mon.admitted = True
    ended = [False]
    signal.signal(signal.SIGTERM, lambda *_: ended.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: ended.__setitem__(0, True))
    def trip():
        subprocess.run(['bash', str(ROOT/'start.sh'), 'stop'], timeout=360, check=True)
    mon.on_trip = trip
    status = 'FAILED'
    started = time.monotonic()
    try:
        mon.start()
        for _ in range(15):
            mon.check()
            if mon.ready(): break
            time.sleep(1)
        if not mon.ready(): raise RuntimeError('all-rank telemetry unavailable')
        if a.mode == 'guard':
            while not ended[0] and time.monotonic()-started < a.seconds:
                mon.check()
                if a.handoff_file and a.handoff_file.is_file():
                    handoff = json.loads(a.handoff_file.read_text())
                    if (handoff.get('ctn'), handoff.get('token')) != (a.boot, deployment['token']) or not (
                        handoff.get('watch_active') is True and handoff.get('heartbeat_verified') is True):
                        raise RuntimeError('invalid serving handoff receipt')
                    break
                time.sleep(1)
            else:
                raise RuntimeError('release guard ended without verified serving handoff')
        else:
            model = S.http(a.base+'/v1/models')['data'][0]['id']
            out = a.out/'c4'; out.mkdir()
            S.c4(SimpleNamespace(endpoint=a.base), dict(geometry=g, boot=a.boot), mon, out, model)
            S.wait_drained(a.base, mon)
            tokens = S.prompt_tokens(a.base, model, single_prompt, 9, a.boot+'-single')
            single = S.stream(a.base, model, tokens, a.out, 'single-max')
            mon.check()
            (a.out/'single.json').write_text(json.dumps(single, indent=2)+'\n')
            if min(mon.minimum.values()) < a.stress_floor:
                raise RuntimeError('stress minimum below release %g GiB criterion' % a.stress_floor)
        status = 'PASS'
    except BaseException:
        trip()
        raise
    finally:
        mon.close()
        (a.out/'RESULT.json').write_text(json.dumps(dict(status=status, geometry=g,
             single_prompt=single_prompt, output_tokens=S.OUTPUT, seconds=time.monotonic()-started,
             live_floor_GiB=a.live_floor, stress_floor_GiB=a.stress_floor, min_GiB=mon.minimum), indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
