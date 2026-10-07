#!/usr/bin/env python3
"""fp4x-cal cycle-time probe (coordinator rule, 2026-10-06): per-request engine /metrics deltas around each
exclusive c1 streaming request (think10/round5-gpu/gate_metrics.py GateMetrics/RequestProbe, unchanged).

One discarded warm-up, then prose x10 and code x10 (ten distinct public prompts each, identical across boots),
T=0, thinking off, max_tokens 512. Decode wall = client first-to-last content chunk. cycle_ms = wall / num_drafts,
committed/cycle = (accepted + drafts) / drafts. A row whose metrics cannot be attributed (foreign traffic, publish
lag) is retried once under a new request id; both rows stay in requests.jsonl.
"""
import argparse, hashlib, json, statistics, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_metrics import GateMetrics

PROSE = [
    'Explain in plain prose, for a curious adult, why the sky is blue during the day and reddish at sunset.',
    'Describe how a city water supply system works, from reservoir to tap, in a few clear paragraphs.',
    'Write a short essay on why people keep diaries and what they gain from writing regularly.',
    'Explain how vaccines train the immune system, using plain language and no lists.',
    'Describe the life cycle of a star like the Sun, from its birth in a nebula to its final stages.',
    'Write a reflective piece about the experience of learning to ride a bicycle as a child.',
    'Explain why the seasons change on Earth, and correct the common misconception about distance to the Sun.',
    'Describe how bread rises, covering yeast, gluten and what happens in the oven, in flowing prose.',
    'Write a brief history of the printing press and its effects on European society.',
    'Explain how a refrigerator keeps food cold, describing the refrigeration cycle in plain words.',
]
CODE = [
    'Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests.',
    'Write a Python class implementing an LRU cache with get and put in O(1), with type hints.',
    'Write a JavaScript function that debounces another function, and explain its parameters in comments.',
    'Write a Python script that counts word frequencies in a text file and prints the top 20 words.',
    'Write a C function that reverses a singly linked list in place, including the node struct.',
    'Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat.',
    'Write a Rust function that returns the n-th prime number using a sieve, with a unit test.',
    'Write a SQL schema for a small library system (books, members, loans) and three example queries.',
    'Write a Python implementation of binary search over a sorted list that returns the insertion point.',
    'Write a Go HTTP handler that returns the current server time as JSON, with error handling.',
]


def stream(base, prompt, probe):
    body = dict(model='GLM-5.3', messages=[{'role': 'user', 'content': prompt}], max_tokens=512, temperature=0,
                stream=True, stream_options={'include_usage': True}, chat_template_kwargs={'enable_thinking': False})
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    first = last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            if not raw.startswith(b'data: ') or raw.strip() == b'data: [DONE]':
                continue
            ev = json.loads(raw[6:])
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = d.get('content') or d.get('reasoning_content') or d.get('reasoning')
                if piece:
                    now = time.monotonic(); first = first or now; last = now; text.append(piece)
            usage = ev.get('usage') or usage
    n = (usage or {}).get('completion_tokens')
    if probe is not None and first and last and last > first:
        probe.set_decode_wall(last - first, delivered_tokens=n)
    return dict(completion_tokens=n, wall=(last - first) if first and last else None, text=''.join(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', default='http://127.0.0.1:18095')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--boot', required=True)
    a = ap.parse_args()
    gate = GateMetrics(a.endpoint + '/metrics', a.out, labels={'model_name': 'GLM-5.3'}, metadata={'boot': a.boot})
    stream(a.endpoint, 'Say hello in one short sentence.', None)   # discarded warm-up (no probe)
    rows = []
    for kind, prompts in (('prose', PROSE), ('code', CODE)):
        for i, p in enumerate(prompts):
            meta = dict(pair_key=f'{kind}-{i}', prompt_sha256=hashlib.sha256(p.encode()).hexdigest(), seed=0,
                        k_mode='boot-default', target_M='boot-default', kind=kind)
            for attempt in range(2):
                rid = f'{kind}-{i}-a{attempt}'
                try:
                    with gate.request(rid, exclusive=True, metadata=meta) as probe:
                        out = stream(a.endpoint, p, probe)
                    rec = probe.record
                except Exception as e:
                    print(rid, 'INVALID', repr(e)[:200], flush=True)
                    time.sleep(5)
                    continue
                rows.append(dict(kind=kind, i=i, request_id=rid, cycle_ms=rec['cycle_ms'], committed=rec['committed_per_cycle'],
                                 tokens=out['completion_tokens'], wall=rec['decode_wall_s'], tps=(out['completion_tokens'] - 1) / out['wall'] if out['wall'] else None))
                print(json.dumps(rows[-1]), flush=True)
                break
    summ = {}
    for kind in ('prose', 'code'):
        g = [r for r in rows if r['kind'] == kind]
        summ[kind] = dict(n=len(g), cycle_ms_median=statistics.median(r['cycle_ms'] for r in g),
                          committed_median=statistics.median(r['committed'] for r in g),
                          tps_median=statistics.median(r['tps'] for r in g if r['tps']))
    (a.out / 'cycle-summary.json').write_text(json.dumps(dict(summary=summ, rows=rows), indent=1) + '\n')
    print(json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
