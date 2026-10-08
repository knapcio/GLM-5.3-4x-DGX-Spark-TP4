#!/usr/bin/env python3
"""Release soak on an owned boot (2026-10-06 gate; run on the head node as a systemd --user unit, cwd = the clone).

Blocks of ~400 s. Each block starts with an exclusive c1 probe (fixed prose + code prompt, T=0, thinking off,
512 tokens, GateMetrics cycle_ms) and then mixed load at a block-specific concurrency target (1/2/4/3/4/2):
thinking-on prose chats (T=0.7), code chats, long token prompts 16K..200K (+256 out), APC repeats of earlier long
prompts, and one client-aborted stream per block. In-flight prompt+max_tokens is kept <= 250K tokens (one pool).
Memory: stress_step.Monitor (1 s, all ranks, kernel journal; trip <4.5 GiB any rank, swap growth, kernel faults).
Pass: no request errors (aborts excluded), preemptions <= 5, monitor clean, rank-0 quiet MemAvailable end-start
within 0.3 GiB, probe cycle_ms median (last 10 min / first 10 min) <= 1.02 for prose and code (one-sided).
"""
import argparse, concurrent.futures, hashlib, json, os, random, statistics, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'bench'))
os.environ.pop('LD_PRELOAD', None)
import stress_step as S
from gate_metrics import GateMetrics

PROSE = ['Explain in plain prose, for a curious adult, why the sky is blue during the day and reddish at sunset.',
         'Describe how a city water supply system works, from reservoir to tap, in a few clear paragraphs.',
         'Write a short essay on why people keep diaries and what they gain from writing regularly.',
         'Explain how vaccines train the immune system, using plain language and no lists.',
         'Describe the life cycle of a star like the Sun, from its birth in a nebula to its final stages.',
         'Write a reflective piece about the experience of learning to ride a bicycle as a child.',
         'Napisz krótki esej o tym, dlaczego ludzie lubią chodzić po górach, prostym językiem.',
         'Explain why the seasons change on Earth, and correct the common misconception about distance to the Sun.']
CODE = ['Write a Python class implementing an LRU cache with get and put in O(1), with type hints.',
        'Write a JavaScript function that debounces another function, and explain its parameters in comments.',
        'Write a C function that reverses a singly linked list in place, including the node struct.',
        'Write a Rust function that returns the n-th prime number using a sieve, with a unit test.',
        'Write a SQL schema for a small library system (books, members, loans) and three example queries.']
PROBE = {'prose': 'Explain in plain prose, for a curious adult, why the sky is blue during the day and reddish at sunset.',
         'code': 'Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests.'}
TARGETS = [1, 2, 4, 3, 4, 2]
BLOCK_S = 400
BUDGET = int(os.environ.get('SOAK_BUDGET') or 250000)


def post_stream(base, path, body, on_chunk=None, abort_after=None, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.monotonic(); first = last = None; n = 0; usage = None; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
                continue
            ev = json.loads(raw[6:])
            if ev.get('error'):
                raise RuntimeError(str(ev['error'])[:300])
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = ch.get('text') or d.get('content') or d.get('reasoning_content') or d.get('reasoning')
                if piece:
                    now = time.monotonic(); first = first or now; last = now; n += 1
                finish = ch.get('finish_reason') or finish
            usage = ev.get('usage') or usage
            if abort_after and n >= abort_after:
                break   # closing the connection aborts the request server-side
    return dict(ttft_s=(first - t0) if first else None, decode_s=(last - first) if first and last else None,
                chunks=n, usage=usage, finish=finish, wall_s=time.monotonic() - t0)


class Soak:
    def __init__(self, a):
        self.a = a; self.base = a.base; self.out = a.out
        self.rng = random.Random(a.seed)
        self.lock = threading.Lock(); self.inflight = {}; self.rid = 0
        self.records = (self.out / 'requests.jsonl').open('a', buffering=1)
        self.longs = []   # previously sent long prompts (token lists) for APC repeats
        self.unit = None

    def tokens(self, n, tag):
        return S.prompt_tokens(self.base, 'GLM-5.3', n, self.rid, tag)

    def record(self, row):
        with self.lock:
            self.records.write(json.dumps(row) + '\n')

    def run_one(self, kind, rid, cost, spec):
        row = dict(rid=rid, kind=kind, start=time.time(), **{k: v for k, v in spec.items() if k != 'tokens'})
        try:
            if kind in ('prose_think', 'code', 'abort'):
                think = kind == 'prose_think'
                body = dict(model='GLM-5.3', messages=[{'role': 'user', 'content': spec['prompt']}],
                            max_tokens=spec['max_tokens'], temperature=0.7 if think else 0, stream=True,
                            stream_options={'include_usage': True},
                            chat_template_kwargs={'enable_thinking': think, 'reasoning_effort': 'low'} if think else {'enable_thinking': False})
                r = post_stream(self.base, '/v1/chat/completions', body, abort_after=spec.get('abort_after'))
            else:
                body = dict(model='GLM-5.3', prompt=spec['tokens'], max_tokens=spec['max_tokens'], temperature=0, seed=0,
                            stream=True, stream_options={'include_usage': True})
                r = post_stream(self.base, '/v1/completions', body)
            row.update(r)
            u = r.get('usage') or {}
            if kind != 'abort' and u.get('completion_tokens') and r.get('decode_s'):
                row['tps'] = (u['completion_tokens'] - 1) / r['decode_s'] if r['decode_s'] > 0 else None
            if kind != 'abort' and not u:
                row['error'] = 'no usage'
        except Exception as e:
            row['error'] = repr(e)[:300]
        row['end'] = time.time()
        self.record(row)
        with self.lock:
            self.inflight.pop(rid, None)
        return row

    def submit(self, pool, kind, spec, cost):
        with self.lock:
            self.rid += 1; rid = self.rid
            self.inflight[rid] = (kind, cost)
        return pool.submit(self.run_one, kind, rid, cost, spec)

    def budget_used(self):
        with self.lock:
            return sum(c for _, c in self.inflight.values()), len(self.inflight), sum(1 for k, _ in self.inflight.values() if k in ('long', 'apc'))

    def drain(self, mon, limit=900):
        t0 = time.monotonic()
        while True:
            mon.check()
            used, n, _ = self.budget_used()
            if n == 0:
                try:
                    return S.wait_drained(self.base, mon)
                except RuntimeError:
                    pass
            if time.monotonic() - t0 > limit:
                raise RuntimeError('drain timeout with %d in flight' % n)
            time.sleep(1)

    def probe(self, gate, block, mon):
        self.drain(mon)
        rows = {}
        for kind, prompt in PROBE.items():
            body = dict(model='GLM-5.3', messages=[{'role': 'user', 'content': prompt}], max_tokens=512, temperature=0,
                        stream=True, stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False})
            for attempt in range(2):
                try:
                    with gate.request(f'probe-{block}-{kind}-{attempt}', exclusive=True, metadata=dict(kind=kind, block=block)) as pr:
                        r = post_stream(self.base, '/v1/chat/completions', body)
                        if r['decode_s']:
                            pr.set_decode_wall(r['decode_s'], delivered_tokens=(r['usage'] or {}).get('completion_tokens'))
                    rec = pr.record
                    rows[kind] = dict(cycle_ms=rec['cycle_ms'], committed=rec['committed_per_cycle'], ttft_s=r['ttft_s'],
                                      tps=((r['usage'] or {}).get('completion_tokens', 1) - 1) / r['decode_s'] if r['decode_s'] else None,
                                      t=time.time())
                    break
                except Exception as e:
                    rows.setdefault('errors', []).append(repr(e)[:200]); time.sleep(3)
        return rows

    def load(self, pool, block, mon, end_at):
        target = TARGETS[block % len(TARGETS)]
        t0 = time.monotonic(); futures = []; aborted = False
        very_long = block % 2 == 0
        while time.monotonic() - t0 < 330 and time.monotonic() < end_at:
            mon.check()
            bt = time.monotonic() - t0
            used, n, nlong = self.budget_used()
            if n >= target:
                time.sleep(0.5); continue
            r = self.rng.random()
            kind = None
            if very_long and bt < 45 and nlong == 0:
                size = self.rng.choice([163840, 200000]); kind = 'long'; very_long = False
            elif bt < 200 and nlong == 0 and r < 0.30:
                size = self.rng.choice([16384, 32768, 65536, 98304]); kind = 'long'
            elif self.longs and bt < 300 and nlong < 2 and r < 0.45:
                kind = 'apc'
            elif not aborted and bt > 60 and r < 0.55:
                kind = 'abort'; aborted = True
            if kind == 'long':
                if used + size + 256 > BUDGET:
                    kind = None
                else:
                    toks = self.tokens(size, f'soak{block}-{self.rid}')
                    self.longs.append(toks); self.longs = self.longs[-4:]
                    futures.append(self.submit(pool, 'long', dict(tokens=toks, prompt_tokens=size, max_tokens=256), size + 256))
                    continue
            if kind == 'apc':
                base = self.rng.choice(self.longs)
                suffix = self.tokens(64, f'apc{self.rid}')
                toks = base + suffix
                if used + len(toks) + 256 <= BUDGET:
                    futures.append(self.submit(pool, 'apc', dict(tokens=toks, prompt_tokens=len(toks), max_tokens=256), len(toks) + 256))
                    continue
            if kind == 'abort':
                futures.append(self.submit(pool, 'abort', dict(prompt=self.rng.choice(PROSE), max_tokens=1500, abort_after=40), 1700))
                continue
            if self.rng.random() < 0.6:
                futures.append(self.submit(pool, 'prose_think', dict(prompt=self.rng.choice(PROSE), max_tokens=1500), 1700))
            else:
                futures.append(self.submit(pool, 'code', dict(prompt=self.rng.choice(CODE), max_tokens=600), 800))
        return futures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--boot', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--minutes', type=float, default=40)
    ap.add_argument('--base', default='http://127.0.0.1:8095')
    ap.add_argument('--seed', type=int, default=1006)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    mon = S.Monitor(a.out, a.boot)
    mon.min_floor_gib = mon.rank0_floor_gib = float(os.environ.get('RECIPE_LIVE_FLOOR_GIB') or 4.5)
    mon.admitted = True
    soak = Soak(a)
    gate = GateMetrics(a.base + '/metrics', a.out / 'probes', labels={'model_name': 'GLM-5.3'}, metadata={'boot': a.boot})
    result = dict(boot=a.boot, start=time.time(), minutes=a.minutes, probes=[], reasons=[])
    reasons = result['reasons']
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    try:
        mon.start()
        for _ in range(30):
            if mon.ready(): break
            time.sleep(1)
        if not mon.ready(): raise RuntimeError('all-rank telemetry unavailable')
        before = soak.drain(mon)
        time.sleep(20)
        result['quiet_start'] = {r: mon.latest[r][1]['mem_kB']['MemAvailable'] / 1048576 for r in range(4)}
        result['preempt_start'] = S.metric(S.metrics(a.base), 'vllm:num_preemptions_total')
        t_start = time.monotonic(); end_at = t_start + a.minutes * 60
        block = 0
        while time.monotonic() < end_at - 60:
            p = soak.probe(gate, block, mon); p['block'] = block; p['elapsed_s'] = time.monotonic() - t_start
            result['probes'].append(p); print('probe', json.dumps(p), flush=True)
            futures = soak.load(pool, block, mon, end_at)
            print('block', block, 'submitted', len(futures), 'elapsed', round(time.monotonic() - t_start), flush=True)
            block += 1
        p = soak.probe(gate, block, mon); p['block'] = block; p['elapsed_s'] = time.monotonic() - t_start
        result['probes'].append(p); print('probe', json.dumps(p), flush=True)
        result['soak_s'] = time.monotonic() - t_start
        soak.drain(mon)
        time.sleep(25)
        mon.check()
        vals = {r: [] for r in range(4)}
        for _ in range(20):
            for r in range(4): vals[r].append(mon.latest[r][1]['mem_kB']['MemAvailable'] / 1048576)
            time.sleep(1)
        result['quiet_end'] = {r: statistics.median(v) for r, v in vals.items()}
        result['preempt_end'] = S.metric(S.metrics(a.base), 'vllm:num_preemptions_total')
    except Exception as e:
        reasons.append('exception: ' + repr(e)[:400])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        mon.close()
    rows = [json.loads(l) for l in (a.out / 'requests.jsonl').read_text().splitlines()] if (a.out / 'requests.jsonl').exists() else []
    errors = [r for r in rows if r.get('error') and r['kind'] != 'abort']
    abort_errors = [r for r in rows if r['kind'] == 'abort' and r.get('error')]
    result.update(requests=len(rows), by_kind={k: sum(1 for r in rows if r['kind'] == k) for k in sorted({r['kind'] for r in rows})},
                  errors=len(errors), error_samples=[r['error'] for r in errors[:5]], abort_errors=len(abort_errors),
                  min_GiB=mon.minimum, monitor_problem=mon.problem, routine_rm_alloc=mon.routine)
    if 'preempt_end' in result:
        result['preemptions'] = result['preempt_end'] - result['preempt_start']
    if 'quiet_end' in result:
        result['mem_drift_GiB'] = {r: round(result['quiet_end'][r] - result['quiet_start'][r], 3) for r in range(4)}
        result['mem_drift_r0_GiB'] = result['mem_drift_GiB'][0]
    def med(kind, lo, hi):
        v = [p[kind]['cycle_ms'] for p in result['probes'] if kind in p and lo <= p['elapsed_s'] <= hi]
        return statistics.median(v) if v else None
    T = result.get('soak_s', 0)
    result['step_ratio'] = {k: (round(med(k, T - 660, T + 60) / med(k, -1, 600), 4) if med(k, -1, 600) and med(k, T - 660, T + 60) else None) for k in PROBE}
    lat = {}
    for k in sorted({r['kind'] for r in rows}):
        g = [r for r in rows if r['kind'] == k and not r.get('error')]
        if g:
            lat[k] = dict(n=len(g), ttft_med_s=statistics.median([r['ttft_s'] for r in g if r.get('ttft_s')] or [0]),
                          ttft_max_s=max([r['ttft_s'] for r in g if r.get('ttft_s')] or [0]),
                          tps_med=statistics.median([r['tps'] for r in g if r.get('tps')] or [0]))
    result['latency'] = lat
    if errors: reasons.append(f'{len(errors)} request errors')
    if result.get('preemptions', 99) > 5: reasons.append(f'preemptions {result.get("preemptions")}')
    if mon.problem: reasons.append('monitor: ' + mon.problem)
    if 'mem_drift_r0_GiB' not in result or abs(result['mem_drift_r0_GiB']) > 0.3: reasons.append(f'rank-0 drift {result.get("mem_drift_r0_GiB")}')
    # One-sided: the check catches slowdowns (last 10 min slower than the first 10 min by > 2 %); a faster end
    # (ratio < 1) is single-probe noise (T=0 outputs differ run to run). Coordinator 2026-10-06 22:15.
    for k, v in result['step_ratio'].items():
        if v is None or v > 1.02: reasons.append(f'{k} step ratio {v}')
    if result.get('soak_s', 0) < a.minutes * 60 - 120: reasons.append('soak shorter than requested')
    result['verdict'] = 'PASS' if not reasons else 'FAIL'
    result['end'] = time.time()
    (a.out / 'RESULT.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print('RESULT', json.dumps(dict(verdict=result['verdict'], reasons=reasons)), flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
