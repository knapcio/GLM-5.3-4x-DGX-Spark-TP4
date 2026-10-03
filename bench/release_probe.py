#!/usr/bin/env python3
"""Public release probes; stdlib only. Credit: knapcio Flash/DS gates and sparkDash."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import urllib.error
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import uuid

MODEL = 'GLM-5.3'
KINDS = ('prose', 'code', 'structured', 'json')
SIZES = (4096, 8192, 16384, 32768, 65536, 131072)


def http(url, body=None, timeout=180):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def cells():
    # All integer client concurrencies, not a claim of 16 active model slots.
    return [(kind, c) for kind in KINDS for c in range(1, 17)]


def validate_job(job, posted, config):
    errors = []
    for key in ('benchId', 'startedAt', 'sparkId'):
        if job.get(key) != posted.get(key):
            errors.append('job identity changed: ' + key)
    if job.get('sparkId') != os.environ.get('SPARKDASH_SPARK_ID', 'configured-node') or job.get('status') != 'completed' or job.get('error'):
        errors.append('job failed or wrong Spark')
    if any((job.get('config') or {}).get(k) != v for k, v in config.items()):
        errors.append('config mismatch')
    if not positive(job.get('startedAt')) or not positive(job.get('completedAt')) or job['completedAt'] < job['startedAt']:
        errors.append('invalid timestamps')
    if (job.get('progress') or {}).get('completedLevels') != 1 or (job.get('progress') or {}).get('totalLevels') != 1:
        errors.append('incomplete progress')
    waves = job.get('results') or []
    if len(waves) != 1:
        return errors + ['missing wave']
    wave = waves[0]; c = config['concurrencies'][0]
    for key, expected in dict(concurrency=c, streamsOk=c, streamsFailed=0,
                              totalCompletionTokens=c*256, totalDecodeTokens=c*255).items():
        if type(wave.get(key)) is not int or wave[key] != expected:
            errors.append('bad wave ' + key)
    if wave.get('error') or not all(positive(wave.get(k)) for k in ('meanDecodeTps', 'aggregateDecodeTps')):
        errors.append('wave error/timing')
    streams = wave.get('streams') or []
    if [s.get('index') for s in streams] != list(range(c)):
        errors.append('stream identity/count')
    for s in streams:
        if s.get('error') or s.get('completionTokens') != 256 or s.get('decodeTokens') != 255 or s.get('reasoningChunks') != 0 or not positive(s.get('decodeTps')):
            errors.append('stream token/reasoning/timing error')
    return errors


class Probes:
    def __init__(self, base, out, check, timeout=180):
        self.base, self.out, self.check, self.timeout = base.rstrip('/'), Path(out), check, timeout
        self.out.mkdir(parents=True, exist_ok=True)

    def request(self, path, body=None):
        self.check()
        return http(self.base + path, body, self.timeout)

    def chat(self, prompt, **extra):
        body = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                    max_tokens=128, temperature=0, chat_template_kwargs={'enable_thinking': False})
        body.update(extra)
        return self.request('/v1/chat/completions', body)

    def correctness(self):
        rows = []
        for prompt, expected in [('Reply only with the capital of Poland.', 'Warsaw'),
                                 ('Compute 17 * 19. Reply only with the integer.', '323'),
                                 ('Reply only with the numbers 1 through 10 separated by spaces.', '1 2 3 4 5 6 7 8 9 10'),
                                 ('Odpowiedz tylko słowem: gotowe', 'gotowe')]:
            r = self.chat(prompt); ch = r['choices'][0]; content = ch['message'].get('content') or ''
            rows.append(dict(prompt=prompt, expected=expected, response=r, passed=content.strip().rstrip('.') == expected))
            save(self.out/'correctness.json', rows)
        # Explicit reasoning semantics, tool round trip and client cancellation.
        for enabled in (False, True):
            r = self.chat('Compute 12 * 13. Finish with the integer answer.', max_tokens=1024,
                          chat_template_kwargs={'enable_thinking': enabled, 'reasoning_effort': 'high' if enabled else 'low'})
            msg = r['choices'][0]['message']; final = msg.get('content') or ''
            reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
            rows.append(dict(kind='thinking', enabled=enabled, response=r,
                             passed='156' in final and (enabled or not reasoning)))
        tool = dict(type='function', function=dict(name='echo', description='Return the supplied value',
                    parameters=dict(type='object', properties={'value': {'type': 'string'}}, required=['value'])))
        body = dict(model=MODEL, messages=[dict(role='user', content='Call echo with value release-ready.')],
                    tools=[tool], tool_choice={'type': 'function', 'function': {'name': 'echo'}},
                    temperature=0, max_tokens=256, chat_template_kwargs={'enable_thinking': False})
        r = self.request('/v1/chat/completions', body); msg = r['choices'][0]['message']
        calls = msg.get('tool_calls') or []; ok = len(calls) == 1 and calls[0]['function']['name'] == 'echo'
        if ok:
            ok = json.loads(calls[0]['function']['arguments']) == {'value': 'release-ready'}
            body['messages'] += [msg, dict(role='tool', tool_call_id=calls[0]['id'], content='release-ready')]
            body['tool_choice'] = 'none'; roundtrip = self.request('/v1/chat/completions', body)
            ok = ok and 'release-ready' in (roundtrip['choices'][0]['message'].get('content') or '')
        else:
            roundtrip = None
        rows.append(dict(kind='tool-roundtrip', response=r, roundtrip=roundtrip, passed=ok))
        save(self.out/'correctness.json', rows)
        self.check()
        request = urllib.request.Request(self.base+'/v1/chat/completions', headers={'Content-Type': 'application/json'},
            data=json.dumps(dict(model=MODEL, messages=[dict(role='user', content='Write a long travel diary.')],
                stream=True, max_tokens=4096, temperature=0, chat_template_kwargs={'enable_thinking': False})).encode())
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    raise RuntimeError('cancel stream ended before a token')
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    chunk = json.loads(line[6:]); choices = chunk.get('choices') or []
                    if choices and choices[0].get('delta', {}).get('content'):
                        break
        # Recovery requires completion, not health-only progress.
        r = self.chat('Reply only with CANCEL_OK.')
        rows.append(dict(kind='cancel-recovery', response=r,
                         passed=(r['choices'][0]['message'].get('content') or '').strip() == 'CANCEL_OK'))
        save(self.out/'correctness.json', rows)
        return all(x['passed'] for x in rows)

    def sparkdash(self, dash):
        url = dash.rstrip('/') + '/bench'; seen = set(); rows = []
        state = http(url)
        if state.get('active'):
            raise RuntimeError('sparkDash already active')
        seen.update(x.get('benchId') for x in [state.get('last'), *(state.get('history') or [])] if isinstance(x, dict))
        schedule = [('warmup', 'prose', 1), ('warmup', 'code', 1)]
        schedule += [('matrix', k, c) for k, c in cells()]
        schedule += [('supplement', 'prose', 1)]*2 + [('supplement', 'prose', 4)]*2
        for phase, kind, c in schedule:
            self.check()
            config = dict(port=8095, modelId=MODEL, concurrencies=[c], maxTokens=256, promptType=kind)
            posted = http(url, config)
            row = dict(phase=phase, config=config, posted=posted); rows.append(row)
            save(self.out/'sparkdash.json', rows)
            if not posted.get('benchId') or posted['benchId'] in seen or posted.get('status') != 'running' or any((posted.get('config') or {}).get(k) != v for k, v in config.items()):
                raise RuntimeError('sparkDash did not return a unique fresh running job')
            seen.add(posted['benchId']); started = time.monotonic()
            while True:
                self.check()
                job = http(url+'/'+urllib.parse.quote(posted['benchId'], safe=''))
                row['job'] = job; save(self.out/'sparkdash.json', rows)
                if job.get('benchId') != posted['benchId']:
                    raise RuntimeError('sparkDash job identity mismatch')
                if job.get('status') != 'running':
                    break
                if time.monotonic()-started > 300:
                    raise TimeoutError('sparkDash cell >300 seconds')
                time.sleep(2)
            row['errors'] = validate_job(job, posted, config); save(self.out/'sparkdash.json', rows)
            if row['errors']:
                raise RuntimeError('; '.join(row['errors']))
            time.sleep(3.1)
        scored = [r['job']['results'][0]['meanDecodeTps'] for r in rows if r['phase'] != 'warmup' and r['config']['promptType'] == 'prose' and r['config']['concurrencies'] == [1]]
        return statistics.median(scored)

    def prefill(self, limit):
        rows = []
        for n in SIZES:
            # Leave space for generation. 32k cell uses 32767 input tokens.
            target = n-1 if n == limit else n
            row = dict(requested_tokens=n, target_input_tokens=target); rows.append(row)
            if n > limit:
                row.update(status='UNSUPPORTED_PROFILE', max_model_len=limit)
                save(self.out/'prefill.json', rows); continue
            text = 'release-' + uuid.uuid4().hex + '\n' + 'Public numbered note. '*n
            ids = self.request('/tokenize', dict(model=MODEL, prompt=text))['tokens'][:target]
            if len(ids) != target or not all(type(x) is int for x in ids):
                raise RuntimeError('tokenizer did not supply exact prefill length')
            body = dict(model=MODEL, prompt=ids, max_tokens=1, temperature=0, stream=True,
                        stream_options={'include_usage': True}, cache_salt=uuid.uuid4().hex)
            self.check(); start = time.monotonic(); first = None; usage = None; done = False
            req = urllib.request.Request(self.base+'/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as response:
                for raw in response:
                    if not raw.startswith(b'data: '): continue
                    if raw.strip() == b'data: [DONE]': done = True; break
                    event = json.loads(raw[6:]); choices = event.get('choices') or []
                    if choices and choices[0].get('text') and first is None: first = time.monotonic()-start
                    if event.get('usage'): usage = event['usage']
            row.update(usage=usage, ttft_s=first, done=done)
            if not done or not first or not usage or usage.get('prompt_tokens') != target or usage.get('completion_tokens') != 1:
                save(self.out/'prefill.json', rows); raise RuntimeError('incomplete cold prefill')
            cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', usage.get('cached_tokens'))
            # The nonce prevents prefix-cache reuse (prefix caching is on in the native profile). Missing usage is labelled.
            if cached not in (None, 0): raise RuntimeError('cold prefill used cached tokens')
            row.update(status='MEASURED', cached_tokens_reported=cached is not None, prefill_tps=target/first)
            save(self.out/'prefill.json', rows)
        return all(r['status'] == 'MEASURED' for r in rows)

    def prefix_scan(self):
        rng = random.Random(730); codes = [''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=8)) for _ in range(512)]
        prefix = '\n'.join(f'Record {i}: code={code}.' for i, code in enumerate(codes))
        rows = []
        for temp in (0, 0.8):
            for repeat in range(3):
                salt = uuid.uuid4().hex
                for phase in ('first', 'repeat'):
                    def collect(target):
                        prompt = prefix + f'\nReturn JSON only: {{"record":{target},"code":"the exact code from that record"}}.'
                        r = self.chat(prompt, temperature=temp, top_p=0.95, seed=target+17, cache_salt=salt,
                                      max_tokens=128, logprobs=True, top_logprobs=5)
                        ch = r['choices'][0]; text = ch['message'].get('content') or ''
                        try: parsed = json.loads(text); ok = parsed == {'record': target, 'code': codes[target]}
                        except ValueError: ok = False
                        logs = (ch.get('logprobs') or {}).get('content')
                        finite = bool(logs) and all(type(x.get('logprob')) in (int, float) and math.isfinite(x['logprob']) for x in logs)
                        return dict(temp=temp, repeat=repeat, phase=phase, target=target, expected=codes[target],
                                    cache_scope='one cache_salt per repeat: first is cold, repeat can hit when prefix caching is on', response=r, passed=ok and finite)
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        for row in pool.map(collect, (0, 31, 255, 511)):
                            rows.append(row); save(self.out/'prefix-sampling.json', rows)
        return all(r['passed'] for r in rows)

    def teacher(self, panel, label, concurrency=1):
        rows = []
        def collect(item):
            body = dict(model=MODEL, prompt=item['text'], max_tokens=1, temperature=0,
                        prompt_logprobs=20, return_token_ids=True, cache_salt=uuid.uuid4().hex)
            r = self.request('/v1/completions', body); ch = r['choices'][0]
            lp = ch.get('prompt_logprobs') or r.get('prompt_logprobs')
            ids = ch.get('prompt_token_ids')
            if not isinstance(lp, list) or not isinstance(ids, list) or len(lp) != len(ids) or len(lp) < 2 or lp[0] is not None:
                raise ValueError('teacher-forced prompt logprobs/IDs unavailable; no greedy fallback')
            converted = [None] + [{str(k): v['logprob'] if isinstance(v, dict) else v for k, v in pos.items()} for pos in lp[1:]]
            return dict(id=item['id'], tokens=ids, prompt_lp=converted)
        # Four HTTP readers for fleet c4, not a CPU compute test. CPU mocks run at c1.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(collect, panel):
                rows.append(result); save(self.out/(label+'.json'), rows)
        return rows


def folded_kl(p, q):
    """Conservative top20 folded-tail estimate; absent q support uses q floor.

    Credit: knapcio Flash kld_probe.kl_top. This is not full-vocabulary KL.
    """
    for d in (p, q):
        if not isinstance(d, dict) or len(d) not in (20, 21) or not all(k.isdecimal() and str(int(k)) == k and type(v) in (int, float) and math.isfinite(v) and v <= 0 for k, v in d.items()):
            raise ValueError('invalid/incomplete top20 support')
        if sum(math.exp(v) for v in d.values()) > 1.00001:
            raise ValueError('invalid probability mass')
    floor = min(q.values()); pm = qm = value = 0.
    for k, lp in p.items():
        lq = q.get(k, floor); mass = math.exp(lp)
        value += mass*(lp-lq); pm += mass; qm += math.exp(lq)
    pt, qt = max(1e-12, 1-pm), max(1e-12, 1-min(qm, 1-1e-12))
    value += pt*math.log(pt/qt)
    if not math.isfinite(value) or value < -1e-5:
        raise ValueError('unstable folded-tail estimate')
    return max(0., value)


def compare_teacher(panel, a, b):
    ids = [p['id'] for p in panel]
    if not ids or len(ids) != len(set(ids)) or [x['id'] for x in a] != ids or [x['id'] for x in b] != ids:
        raise ValueError('incomplete/reordered panel')
    values = []; lengths = []; items = []
    for p, q in zip(a, b):
        if p['tokens'] != q['tokens'] or len(p['prompt_lp']) != len(q['prompt_lp']) or len(p['prompt_lp']) != len(p['tokens']) or len(p['tokens']) < 2 or p['prompt_lp'][0] is not None or q['prompt_lp'][0] is not None:
            raise ValueError('teacher token/length mismatch')
        lengths.append(len(p['tokens']))
        item_values = [folded_kl(u, v) for u, v in zip(p['prompt_lp'][1:], q['prompt_lp'][1:])]
        values.extend(item_values)
        items.append(dict(id=p['id'], positions=len(item_values), mean=statistics.mean(item_values)))
    return dict(items=len(ids), per_item=items, positions=len(values), lengths=lengths, mean=statistics.mean(values),
                p99=sorted(values)[min(len(values)-1, int(.99*len(values)))], estimator='top20 folded-tail estimate')


def quality_gate(reference, candidate):
    if len(reference) != 3 or len(candidate) != 3:
        raise ValueError('qeval requires three complete repeats per arm')
    def count(data):
        results = data['results']
        if len(results) != 75 or len({r['id'] for r in results}) != 75 or any(r.get('why', '').startswith('request failed:') for r in results):
            raise ValueError('incomplete qeval or request errors')
        primary = sum(r['pass'] for r in results if r['category'] in ('code', 'reason', 'math'))
        return primary
    for a, b in zip(reference, candidate):
        if [(r['id'], r['category']) for r in a['results']] != [(r['id'], r['category']) for r in b['results']]:
            raise ValueError('qeval task grid mismatch')
    ref = [count(x) for x in reference]; arm = [count(x) for x in candidate]
    # Small suite cannot establish <1% intelligence loss. Zero median net breaks is a screen.
    return dict(reference_primary=ref, candidate_primary=arm,
                passed=statistics.median(arm) >= statistics.median(ref), resolution='55 primary tasks; no <1% equivalence claim')
