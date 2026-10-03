#!/usr/bin/env python3
"""Prose decode at added context 0/16K/30K/60K, c1 T=0, thinking off.

Portable CLI adaptation of the W4 collector. Three interleaved repeats, distinct
nonce/seeded filler, 512 output tokens. Decode=(completion_tokens-1)/(last-first
content arrival); TTFT=first content arrival-send. Requires tokenizers and the
checkpoint tokenizer.json. Raw SSE timings and usage accompany every result.
"""
import argparse, json, random, statistics, sys, time, urllib.request, uuid
from pathlib import Path

BASE = None
TOK = None
TARGETS = (0, 16384, 30720, 61440)
REPEATS = 3
QUESTION = ('Write a clear explanation, for a curious adult reader, of how ocean tides work: the roles of the Moon '
            'and the Sun, why most coasts see two high tides a day, and what spring and neap tides are. '
            'Use plain prose paragraphs, no lists.')
PLACES = ['harbour', 'orchard', 'library', 'workshop', 'meadow', 'station', 'bakery', 'garden', 'museum', 'bridge',
          'market', 'quarry', 'school', 'mill', 'lighthouse', 'farm', 'observatory', 'pier', 'archive', 'chapel']
THINGS = ['crates of apples', 'spools of thread', 'boxes of tiles', 'bundles of reeds', 'jars of honey',
          'sacks of flour', 'rolls of canvas', 'barrels of cider', 'stacks of slate', 'baskets of plums']
VERBS = ['delivered', 'counted', 'stored', 'repaired', 'inspected', 'painted', 'weighed', 'sorted', 'labelled', 'moved']


def filler(seed, n_tokens, tok):
    if n_tokens <= 0:
        return ''
    rnd = random.Random(seed)
    parts, used, i = [f'[{uuid.uuid4().hex}] Background notes (not needed for the question below).'], 0, 0
    while True:
        s = (f'Note {i}: on day {rnd.randint(1, 365)} the {rnd.choice(PLACES)} team {rnd.choice(VERBS)} '
             f'{rnd.randint(2, 99)} {rnd.choice(THINGS)} near the {rnd.choice(PLACES)}.')
        parts.append(s); i += 1
        if i % 200 == 0:
            used = len(tok.encode(' '.join(parts)).ids)
            if used >= n_tokens:
                break
    # trim to the target
    while len(parts) > 2 and len(tok.encode(' '.join(parts)).ids) > n_tokens:
        parts = parts[:-20] if len(tok.encode(' '.join(parts)).ids) > n_tokens + 400 else parts[:-1]
    return ' '.join(parts)


def stream(messages):
    body = dict(model='GLM-5.3', messages=messages, max_tokens=512, temperature=0, stream=True,
                stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic(); first = last = None; usage = None; text = []; finish = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
                continue
            ev = json.loads(raw[6:])
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = d.get('content') or d.get('reasoning_content') or d.get('reasoning')
                if piece:
                    now = time.monotonic(); first = first or now; last = now; text.append(piece)
                finish = ch.get('finish_reason') or finish
            usage = ev.get('usage') or usage
    n = (usage or {}).get('completion_tokens', 0)
    return dict(ttft_s=(first - start) if first else None, decode_s=(last - first) if first and last else None,
                completion_tokens=n, prompt_tokens=(usage or {}).get('prompt_tokens'),
                cached_tokens=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
                finish_reason=finish, decode_tps=(n - 1) / (last - first) if first and last and last > first and n > 1 else None,
                text=''.join(text))


def main():
    global BASE, TOK
    parser = argparse.ArgumentParser(description='Prose decode versus context, identical W4 protocol')
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--tokenizer-json', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    BASE, TOK = args.endpoint.rstrip('/'), args.tokenizer_json
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists')
    rows = []
    with out.open('x') as f:
        for rep in range(1, REPEATS + 1):
            for tgt in TARGETS:
                ctx = filler(1000 * rep + tgt // 1024, tgt - 120 if tgt else 0, tok)
                content = (ctx + '\n\n' + QUESTION) if ctx else QUESTION
                r = stream([{'role': 'user', 'content': content}])
                r.update(rep=rep, target_ctx=tgt, filler_local_tokens=len(tok.encode(ctx).ids) if ctx else 0,
                         utc=time.strftime('%FT%TZ', time.gmtime()))
                rows.append(r); f.write(json.dumps(r) + '\n'); f.flush()
                print(rep, tgt, 'prompt', r['prompt_tokens'], 'cached', r['cached_tokens'], 'ttft %.2f' % (r['ttft_s'] or -1),
                      'tok', r['completion_tokens'], 'tps %.2f' % (r['decode_tps'] or -1), r['finish_reason'], flush=True)
    summ = {}
    for tgt in TARGETS:
        g = [r for r in rows if r['target_ctx'] == tgt and r['decode_tps']]
        summ[str(tgt)] = dict(n=len(g), prompt_tokens=[r['prompt_tokens'] for r in g],
                              decode_tps=[round(r['decode_tps'], 2) for r in g],
                              decode_tps_median=statistics.median(r['decode_tps'] for r in g) if g else None,
                              ttft_s=[round(r['ttft_s'], 3) for r in g],
                              ttft_median_s=statistics.median(r['ttft_s'] for r in g) if g else None,
                              completion_tokens=[r['completion_tokens'] for r in g],
                              cached_tokens=[r['cached_tokens'] for r in g])
    (out.parent / (out.stem + '-summary.json')).write_text(json.dumps(summ, indent=1) + '\n')
    print(json.dumps({k: (v['decode_tps_median'], v['ttft_median_s']) for k, v in summ.items()}))


if __name__ == '__main__':
    main()
