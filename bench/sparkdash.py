"""sparkDash decode and prefill sweep behind the README results tables. Run it on the head node, next to sparkDash
and the endpoint on port 8095:

    SPARKDASH_API=http://127.0.0.1:5555/api/sparks/rank0/llm python3 bench/sparkdash.py full 32768 > sparkdash.jsonl

Arguments: mode (`full` or `short`) and the served max_model_len (default 32768).

mode full  (README tables): decode DecodeBench, 256 tokens, thinking off, for prose, code, structured, json at
           c1 x5, c2 x3, c4 x3, c8 x2 (one c1 warm-up per prompt type and one prose c4 warm-up, discarded), then
           the cold prefill bench at 4k/8k/16k/32k (one 4k warm-up, then three rounds). About 19 min.
mode short: prose c1 x5, code c1 x5, prose c4 x3. About 4 min.

The served context is 32,768 tokens by default, so a 32,768-token prompt cannot be served; the 32k cell then
asks for max_model_len - 512 tokens and records the size it used. Every job is printed as one JSON line; the
last line is {"summary": [...three short cells...], "sweep": {...}} (medians over scored runs whose streams all
completed). The server has four slots: at c8 four streams queue behind the first four, and sparkDash sends the
same prompt to every stream at c > 1.
"""
import json, os, statistics, sys, time, urllib.request

MODE = sys.argv[1] if len(sys.argv) > 1 else 'short'
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 32768
API = os.environ.get('SPARKDASH_API', 'http://127.0.0.1:5555/api/sparks/rank0/llm')
PORT = 8095
POLL = float(os.environ.get('SPARKDASH_POLL_S', '2'))  # offline tests set 0
KINDS = ('prose', 'code', 'structured', 'json')
CONC = {1: 5, 2: 3, 4: 3, 8: 2}
PREFILL = (4096, 8192, 16384, 32768)
rows = []; T0 = time.monotonic()


def http(url, data=None, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode() or 'null')


def emit(row):
    rows.append(row); print(json.dumps(row), flush=True)


def bench(kind, c, phase):
    url = API + '/bench'
    if http(url).get('active'): raise RuntimeError('sparkDash already active')
    config = {'port': PORT, 'modelId': 'GLM-5.3', 'concurrencies': [c], 'maxTokens': 256, 'promptType': kind}
    t = time.monotonic(); post = http(url, config)
    while True:
        time.sleep(POLL); s = http(url)
        if not s.get('active'): break
        if time.monotonic() - t > 300: raise RuntimeError('bench deadline')
    emit({'phase': phase, 'kind': kind, 'concurrency': c, 'config': config, 'post': post, 'result': s['last'],
          'wall_s': time.monotonic() - t})


def prefill(sizes, phase):
    url = API + '/prefill-bench'
    st = http(f'{url}?port={PORT}')
    if st and st.get('active'): raise RuntimeError('sparkDash prefill bench already active')
    t = time.monotonic(); row = {'phase': phase, 'kind': 'prefill', 'sizes': sizes}
    try:
        post = http(url, {'port': PORT, 'contextSizes': sizes}, timeout=120); row['post'] = post
        while True:
            time.sleep(POLL * 1.5); s = http(f'{url}?port={PORT}')
            if s and not s.get('active'): break
            if time.monotonic() - t > 1800: raise RuntimeError('prefill deadline')
        last = s.get('last') or {}
        row['result'] = last
        if last.get('benchId') not in (None, post.get('benchId')): row['error'] = 'stale prefill job'
        elif last.get('status') not in (None, 'completed'): row['error'] = f"prefill job {last.get('status')}: {last.get('error')}"
    except Exception as e:  # one failed size must not lose the decode sweep or the other sizes
        row['error'] = repr(e)
    row['wall_s'] = time.monotonic() - t; emit(row)


def ok(x, c): return x.get('streamsOk') == c and x.get('streamsFailed', 0) == 0 and not x.get('error')


def decode_cell(kind, c):
    a = [r['result']['results'][0] for r in rows if r.get('phase') == 'scored' and r['kind'] == kind and r['concurrency'] == c]
    good = [x for x in a if ok(x, c)]
    if not good: return {'kind': kind, 'concurrency': c, 'n': 0, 'status': 'unmeasured' if not a else 'unmeasured-stream-failure'}
    return {'kind': kind, 'concurrency': c, 'n': len(good), 'runs_failed': len(a) - len(good),
            'median_per_stream_decode_tps': statistics.median(x['meanDecodeTps'] for x in good),
            'median_aggregate_decode_tps': statistics.median(x['aggregateDecodeTps'] for x in good),
            'aggregate_runs': [x['aggregateDecodeTps'] for x in good]}


def prefill_cells(s32):
    out = []
    for label, size in zip(('4k', '8k', '16k', '32k'), PREFILL[:3] + (s32,)):
        res = [x for r in rows if r.get('phase') == 'prefill-scored' and not r.get('error')
               for x in (r.get('result') or {}).get('results', []) if int(x.get('targetTokens') or x.get('contextSize') or 0) == size]
        tps = [float(x['prefillTps']) for x in res if x.get('prefillTps')]
        cell = {'cell': label, 'target_tokens': size, 'n': len(tps)}
        if tps:
            cell.update(median_prefill_tps=statistics.median(tps), runs=tps,
                        median_ttft_ms=statistics.median(float(x.get('ttftMs') or 0) for x in res if x.get('prefillTps')))
        out.append(cell)
    return out


if MODE == 'full':
    for kind in KINDS:
        bench(kind, 1, 'warmup-discarded')
        for c, n in ((CONC | {3: 3}) if kind == 'prose' else CONC).items():
            if kind == 'prose' and c == 4: bench(kind, 4, 'warmup-discarded')
            for _ in range(n): bench(kind, c, 'scored')
    s32 = 32768 if MAX_LEN >= 32768 + 512 else MAX_LEN - 512
    prefill([4096], 'prefill-warmup-discarded')
    for _ in range(3):
        prefill(list(PREFILL[:3]), 'prefill-scored')
        prefill([s32], 'prefill-scored')
else:
    for kind in ('prose', 'code'):
        bench(kind, 1, 'warmup-discarded')
        for _ in range(5): bench(kind, 1, 'scored')
    bench('prose', 4, 'warmup-discarded')
    for _ in range(3): bench('prose', 4, 'scored')

summary = [decode_cell(k, c) for k, c in (('prose', 1), ('code', 1), ('prose', 4))]
sweep = None
if MODE == 'full':
    sweep = {'decode': [decode_cell(k, c) for k in KINDS for c in (tuple(CONC)+( (3,) if k == 'prose' else () ))], 'prefill': prefill_cells(s32),
             'max_model_len': MAX_LEN, 'prefill_32k_tokens': s32}
print(json.dumps({'summary': summary, 'sweep': sweep, 'mode': MODE, 'duration_s': time.monotonic() - T0}), flush=True)
