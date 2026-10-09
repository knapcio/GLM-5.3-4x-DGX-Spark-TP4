#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sequential T=0 probe; standard library only. Cold mode resets prefix cache."""
import argparse
import hashlib
import http.client
import json
import math
import re
import secrets
import socket
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

# Fixed English panel. Roughly 40-105 tokens each; exact model token counts are
# recorded from usage.prompt_tokens rather than claimed from a surrogate tokenizer.
PANEL = [
    ('prose', 'Explain why a city library is useful even when many books are available online. Use plain language for a reader who has never visited one. Mention learning, a quiet place, and community activities. Answer in two concise sentences without a heading.'),
    ('prose', 'Write a short description of a small harbor just after sunrise. Include one sound, one color, and one movement of the water. Keep the tone calm and concrete rather than dramatic. Use two sentences and avoid naming a real location.'),
    ('prose', 'Explain the difference between weather and climate to a curious twelve year old. Give a simple everyday example for each term. Avoid technical vocabulary and do not discuss policy. Limit the answer to two or three short sentences in ordinary English.'),
    ('prose', 'Rewrite this announcement to sound friendly and clear: The building will be closed on Monday because routine maintenance is scheduled. Visitors should return on Tuesday. Keep both facts, remove unnecessary formality, and produce only the revised announcement in two short sentences.'),
    ('prose', 'Describe a useful habit for remembering the names of people you meet at a small gathering. Explain the habit with a concrete example and a respectful tone. Do not recommend writing personal information in a public place. Keep your answer to two sentences.'),
    ('prose', 'Write a tiny story about a gardener who finds an unexpected blue flower beside an old gate. The story should have a beginning and a gentle ending. Use ordinary words, no dialogue, and no magic. Keep the story to three short sentences.'),
    ('prose', 'Explain why letting bread dough rest can change its texture. Address someone baking for the first time and connect your explanation to a practical action in the kitchen. Avoid detailed chemistry or quantities. Answer in two clear sentences without a title or list.'),
    ('prose', 'Compare taking a walk alone with walking alongside a friend. Mention one benefit of each option and finish with a neutral observation about choosing between them. Use a warm but matter of fact tone and keep the entire answer to three short sentences.'),
    ('prose', 'Summarize this situation for a neighborhood notice: A volunteer group planted six trees in the park on Saturday. Rain began afterward, so the group postponed painting the benches until next weekend. Preserve the sequence and all important facts in two concise sentences.'),
    ('prose', 'Give a brief explanation of how a reusable shopping bag can be convenient during an ordinary grocery trip. Mention carrying items and storing the empty bag at home. Avoid claims about environmental statistics or costs. Write two concise sentences in plain English.'),
    ('code', 'Write a Python function named clamp with arguments value, low, and high. Return low when value is below low and high when it exceeds high; otherwise return value. Assume low is no greater than high. Return only code, with no imports or explanation.'),
    ('code', 'Write a Python function named count_even that accepts a list of integers and returns how many entries are even. Negative integers and zero should be handled normally. Do not modify the input list and do not import anything. Return only the function definition.'),
    ('code', 'Write a JavaScript function named lastOrNull that accepts an array. It should return the final element if the array is nonempty and null otherwise. Do not change the array or use external libraries. Return only code and use a conventional function declaration.'),
    ('code', 'Write a Python function named unique_ordered that accepts a list of strings and returns distinct strings in their first appearance order. The input list must remain unchanged. Use a set to track seen values. Return only code without type annotations, imports, or commentary.'),
    ('code', 'Write a SQL query against a table named orders with columns customer_id and amount. Return each customer_id and the sum of amount as total_amount. Include only customers whose sum exceeds 100. Do not sort the result. Return only the SQL query with a semicolon.'),
    ('code', 'Write a Python function named is_palindrome that takes a string and returns whether it reads the same forward and backward. This comparison is case sensitive and includes spaces and punctuation. The empty string should return True. Return only code with no explanatory prose.'),
    ('code', 'Write a JavaScript function named sumPositive that takes an array of numbers and returns the sum of entries strictly greater than zero. Zero and negative numbers should contribute nothing. Do not change the input array. Return only code and avoid any external dependencies.'),
    ('code', 'Write a Python function named safe_get that takes a dictionary, a key, and a default value. Return the stored value when the key exists, even if that value is None; otherwise return the default. Return only a short function definition with no imports.'),
    ('code', 'Write a Python function named pairs that accepts a list and returns a list of adjacent two element tuples. For example, [1, 2, 3] should produce [(1, 2), (2, 3)]. Empty and single element inputs should produce an empty list. Return only code.'),
    ('code', 'Write a Bash snippet that loops over the literal names alpha, beta, and gamma and prints each name on its own line. Use a for loop, quote the variable expansion, and use printf rather than echo. Return only the snippet without a shebang or commentary.'),
    ('reasoning', 'A box contains twelve red balls and eight blue balls. One ball is selected uniformly at random without looking. What is the probability that it is blue? State the answer as a reduced fraction and show one short calculation. Do not discuss repeated draws or replacement.'),
    ('reasoning', 'A train travels at a constant speed of sixty kilometers per hour for forty five minutes. How far does it travel during that time? Convert minutes to hours explicitly, then give the distance in kilometers. Use at most two short sentences and no extra assumptions.'),
    ('reasoning', 'Find the next integer in the sequence 3, 7, 11, 15, 19 under the rule that the same number is added at each step. State the common difference and then the next integer. Keep the explanation to one short sentence without proposing alternative sequence rules.'),
    ('reasoning', 'Three friends share a bill of forty eight dollars equally, then each adds a tip of two dollars. How much does each friend pay in total? Show the equal share and the added tip in one short calculation. Assume there are no taxes or other charges.'),
    ('reasoning', 'A rectangle has length nine centimeters and width four centimeters. Compute its perimeter and area. Label both results and include the correct units for each. Show only the two simple calculations and do not introduce a diagram or discuss other geometric shapes.'),
    ('reasoning', 'Solve the equation 3x + 5 = 26 for x. Give one intermediate step that subtracts the constant term, then state the result. Work with ordinary real numbers and keep the response to no more than two short lines without a general algebra lesson.'),
    ('reasoning', 'All robins are birds, and all birds have feathers. Based only on these two statements, does it follow that every robin has feathers? Answer yes or no and explain the logical chain in a single short sentence. Do not add biological exceptions or outside facts.'),
    ('reasoning', 'A shop reduces the price of a twenty dollar notebook by fifteen percent. What is the discount in dollars and what is the final price? Show both amounts with a simple calculation. Assume the reduction is applied once and there is no additional sales tax.'),
    ('reasoning', 'A jar initially has five coins. Four coins are added, then three coins are removed. How many coins remain? Show the arithmetic in the order the events occur and give the final count. Keep your response to one line without adding any unstated events.'),
    ('reasoning', 'Two fair six sided dice are rolled. How many ordered outcomes have a total of seven, and what is their probability among all ordered outcomes? State the count and a reduced fraction. Include a brief explanation of the total number of possible ordered outcomes.'),
]
REPEATS = 3
NONCE_BYTES = 16


class ProbeError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


class HTTPError(ProbeError):
    def __init__(self, code, body):
        super().__init__('HTTP_ERROR', 'HTTP %s: %s' % (code, body[:2000]))
        self.code = code
        self.body = body


class Transport:
    """Deadline-bound HTTP without proxies, redirects, retries, or SDK defaults."""
    def __init__(self, deadline, timeout, abort):
        self.deadline = deadline
        self.timeout = timeout
        self.abort = abort

    def call(self, url, payload=None, timeout=None, expected_status=None):
        if self.abort.is_set():
            raise ProbeError('SAFETY_ABORT', 'Metrics guard stopped the panel')
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeError('TOTAL_TIMEOUT', 'Total wall-clock cap reached')
        configured_timeout = self.timeout if timeout is None else timeout
        total_limited = remaining <= configured_timeout
        budget = min(configured_timeout, remaining)
        until = time.monotonic() + budget
        parts = urlsplit(url)
        cls = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
        conn = cls(parts.hostname, parts.port, timeout=budget)
        done = threading.Event()
        result = {}

        def worker():
            try:
                if self.abort.is_set():
                    raise ProbeError('SAFETY_ABORT', 'Metrics guard stopped the panel')
                body = None if payload is None else request_bytes(payload)
                path = parts.path or '/'
                if parts.query:
                    path += '?' + parts.query
                conn.request('GET' if payload is None else 'POST', path, body,
                             {'Content-Type': 'application/json'} if body else {})
                result['socket'] = conn.sock
                response = conn.getresponse()
                data = response.read().decode('utf-8')
                if response.status >= 400 or (expected_status is not None and response.status != expected_status):
                    raise HTTPError(response.status, data)
                result['data'] = data
            except (socket.timeout, TimeoutError) as exc:
                result['error'] = ProbeError('TOTAL_TIMEOUT' if total_limited else 'REQUEST_TIMEOUT',
                                             str(exc) or 'HTTP timeout')
            except ProbeError as exc:
                result['error'] = exc
            except Exception as exc:
                result['error'] = ProbeError('HTTP_ERROR', str(exc))
            finally:
                conn.close()
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                # Check abort/deadline even if an immediate response has arrived.
                if self.abort.is_set():
                    raise ProbeError('SAFETY_ABORT', 'Metrics guard stopped the panel')
                now = time.monotonic()
                if now >= self.deadline:
                    raise ProbeError('TOTAL_TIMEOUT', 'Total wall-clock cap reached')
                if now >= until:
                    raise ProbeError('REQUEST_TIMEOUT', 'HTTP request exceeded %.3fs' % budget)
                if done.wait(min(0.02, until - now)):
                    break
            if 'error' in result:
                raise result['error']
            return result['data']
        finally:
            if not done.is_set():
                # Interrupt an in-flight read on safety abort or absolute deadline.
                sock = result.get('socket') or conn.sock
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                conn.close()

    def json(self, url, payload=None):
        data = self.call(url, payload)
        try:
            # Reject nonstandard NaN/Infinity instead of losing the whole report
            # later when writing strict JSON.
            def reject_constant(value):
                raise ValueError('nonfinite JSON number: ' + value)
            return json.loads(data, parse_constant=reject_constant)
        except ValueError as exc:
            raise ProbeError('INVALID_RESPONSE', 'Invalid JSON: %s' % exc) from exc


def request_bytes(payload):
    return json.dumps(payload, allow_nan=False).encode('utf-8')


def server_root(base_url):
    base = base_url.rstrip('/')
    return base[:-3] if base.endswith('/v1') else base


def reset_prefix_cache(transport, url):
    try:
        transport.call(url, {}, expected_status=200)
    except ProbeError as exc:
        raise ProbeError('RESET_UNAVAILABLE', '%s: %s' % (url, exc)) from exc
    return {'url': url, 'http_status': 200}


def prefix_cache_hits(body, name=None):
    """Aggregate one counter family, retaining series to detect counter resets."""
    names = ('vllm:prefix_cache_hits_total', 'vllm:gpu_prefix_cache_hits_total',
             'vllm:gpu_prefix_cache_hits', 'gpu_prefix_cache_hits_total', 'gpu_prefix_cache_hits')
    counters = {key: {} for key in names}
    pattern = re.compile(r'^([^\s{]+)(\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?\s*$')
    for line in body.splitlines():
        match = pattern.match(line)
        if not match or match.group(1) not in counters:
            continue
        metric, labels, raw = match.groups()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ProbeError('METRICS_ERROR', 'Invalid prefix-cache hit counter') from exc
        if not math.isfinite(value) or value < 0:
            raise ProbeError('METRICS_ERROR', 'Nonfinite or negative prefix-cache hit counter')
        key = labels or ''
        if key in counters[metric]:
            raise ProbeError('METRICS_ERROR', 'Duplicate prefix-cache hit series')
        counters[metric][key] = value
    selected = name or next((key for key in names if counters[key]), None)
    if selected not in counters or not counters[selected]:
        raise ProbeError('METRICS_ERROR', 'Required prefix-cache hit counter unavailable: %s' % (name or names,))
    return {'counter': selected, 'value': sum(counters[selected].values()), 'series': counters[selected]}


def hit_delta(before, after):
    if before['counter'] != after['counter'] or before['series'].keys() != after['series'].keys():
        raise ProbeError('METRICS_ERROR', 'Prefix-cache hit counter series changed during request')
    if any(after['series'][key] < value for key, value in before['series'].items()):
        raise ProbeError('METRICS_ERROR', 'Prefix-cache hit counter decreased during request')
    return after['value'] - before['value']


def metrics_counts(body):
    values = {'running': [], 'waiting': []}
    pattern = re.compile(r'^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?\s*$')
    for line in body.splitlines():
        match = pattern.match(line)
        if match:
            try:
                value = float(match.group(2))
            except ValueError as exc:
                raise ProbeError('METRICS_ERROR', 'Invalid gauge value') from exc
            if not math.isfinite(value) or value < 0:
                raise ProbeError('METRICS_ERROR', 'Nonfinite or negative gauge')
            values[match.group(1)].append(value)
    if not all(values.values()):
        raise ProbeError('METRICS_ERROR', 'Both vllm running and waiting gauges are required')
    return {key: sum(entries) for key, entries in values.items()}


class Guard:
    def __init__(self, transport, url, interval):
        self.transport = transport
        self.url = url
        self.interval = interval
        self.stop = threading.Event()
        self.samples = []
        self.error = None
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.thread = None

    def check(self, idle=False):
        try:
            counts = metrics_counts(self.transport.call(self.url, timeout=min(2.0, self.transport.timeout)))
            sample = dict(counts, elapsed_s=round(time.monotonic() - self.started, 6), idle_check=idle)
            with self.lock:
                self.samples.append(sample)
            if (idle and (counts['running'] != 0 or counts['waiting'] != 0)) or counts['running'] > 1:
                raise ProbeError('FOREIGN_TRAFFIC', 'vLLM gauges: running=%s waiting=%s' %
                                 (counts['running'], counts['waiting']))
        except ProbeError as exc:
            with self.lock:
                if self.error is None:
                    self.error = exc
            self.transport.abort.set()
            raise

    def start(self):
        def watch():
            while not self.stop.is_set():
                try:
                    self.check()
                except ProbeError:
                    return
                self.stop.wait(self.interval)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=max(0, min(2.1, self.transport.deadline - time.monotonic())))


def output_record(response):
    try:
        choice = response['choices'][0]
        message = choice.get('message', {})
        logprobs = choice.get('logprobs') or {}
        entries = logprobs.get('content') or []
        if not isinstance(entries, list):
            raise ValueError('logprobs.content must be a list')
        text = message.get('content') or ''
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if not isinstance(text, str) or not isinstance(reasoning, str):
            raise ValueError('Expected string content and reasoning text')
        token_ids = choice.get('token_ids')
        if token_ids is None:
            token_ids = logprobs.get('token_ids')
        if token_ids is None and entries and all(isinstance(e.get('token_id'), int) for e in entries):
            token_ids = [e['token_id'] for e in entries]
        if token_ids is not None:
            if not isinstance(token_ids, list) or not all(isinstance(t, int) for t in token_ids):
                raise ValueError('invalid token_ids')
            basis, sequence = 'token_ids', token_ids
        elif entries:
            basis = 'logprob_tokens_and_bytes'
            sequence = []
            for entry in entries:
                if not isinstance(entry.get('token'), str):
                    raise ValueError('missing logprob token')
                sequence.append({'token': entry['token'], 'bytes': entry.get('bytes')})
        else:
            # No tokenizer guessing for a text-only API. Channel text is also
            # compared independently so reasoning/content boundary changes count.
            basis = 'text_characters'
            sequence = list(reasoning + text)
        if not (token_ids or entries or text or reasoning):
            raise ValueError('No generated tokens or text in response')
        return {'basis': basis, 'sequence': sequence, 'token_ids': token_ids,
                'tokens': entries, 'text': text, 'reasoning_text': reasoning,
                'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage'),
                'raw_response': response}
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ProbeError('INVALID_RESPONSE', str(exc)) from exc


def divergence(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def margin_at(record, index):
    if record['basis'] == 'token_ids' and len(record['sequence']) != len(record['tokens']):
        return {'available': False, 'reason': 'Token ids and logprob entries cannot be aligned'}
    if index >= len(record['tokens']):
        return {'available': False, 'reason': 'No logprob entry at divergence (possibly sequence ended)'}
    entry = record['tokens'][index]
    candidates = entry.get('top_logprobs') or []
    # A selected token outside top-1 is not necessarily rank 2. Only returned
    # top_logprobs can establish the actual top-1/top-2 margin.
    distinct = {}
    for candidate in candidates:
        lp = candidate.get('logprob')
        token = candidate.get('token')
        if token is None or not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp <= -9999:
            continue  # vLLM may use -9999 as an unavailable-logprob sentinel.
        key = json.dumps([token, candidate.get('bytes')], ensure_ascii=False)
        if key not in distinct or lp > distinct[key]['logprob']:
            distinct[key] = {'token': token, 'bytes': candidate.get('bytes'), 'logprob': lp}
    top = sorted(distinct.values(), key=lambda item: item['logprob'], reverse=True)
    if len(top) < 2:
        return {'available': False, 'reason': 'Fewer than two distinct scored candidates; top_logprobs=1 often supplies only top-1'}
    return {'available': True, 'margin': top[0]['logprob'] - top[1]['logprob'],
            'top_1': top[0], 'top_2': top[1], 'source': 'returned top_logprobs'}


def analyze_row(row):
    repeats = row['repeats']
    pairs = []
    for i in range(len(repeats)):
        for j in range(i + 1, len(repeats)):
            a, b = repeats[i].get('output'), repeats[j].get('output')
            if a is None or b is None:
                continue
            if a['basis'] != b['basis']:
                raise ProbeError('INVALID_RESPONSE', 'Token representation changed between repeats')
            index = divergence(a['sequence'], b['sequence'])
            token_basis = a['basis'] != 'text_characters'
            # Also detect differences in channel text when exposed logprobs omit it.
            text_differs = (a['text'], a['reasoning_text']) != (b['text'], b['reasoning_text'])
            pairs.append({'repeats': [i + 1, j + 1], 'first_divergence_index': index,
                          'index_unit': 'token' if token_basis else 'text_character',
                          'first_generated_token_differs': (index == 0 if token_basis else None),
                          'text_differs': text_differs,
                          'near_tie': {'left': margin_at(a, index), 'right': margin_at(b, index)}
                          if index is not None and token_basis else None})
    complete = len(repeats) == REPEATS and all(r.get('output') for r in repeats)
    indexes = [p['first_divergence_index'] for p in pairs if p['first_divergence_index'] is not None]
    any_diff = any(p['first_divergence_index'] is not None or p['text_differs'] for p in pairs)
    observable = bool(pairs) and all(p['first_generated_token_differs'] is not None for p in pairs)
    apc_hit = row.get('apc_mode') == 'cold' and any(
        (attempt.get('prefix_cache_hits', {}).get('delta') or 0) > 0
        for repeat in repeats for attempt in repeat.get('attempts', []))
    bodies = [repeat.get('request_body') for repeat in repeats]
    inputs_identical = (all(body is not None for body in bodies) and len(set(bodies)) == 1
                        if complete and 'apc_mode' in row else None)
    row.update({'complete': complete, 'pairs': pairs,
                'inputs_identical': inputs_identical,
                'status': 'APC_HIT' if apc_hit else 'INPUT_MISMATCH' if inputs_identical is False else
                          'COMPLETE' if complete else 'INCOMPLETE',
                'gate_eligible': complete and not apc_hit and inputs_identical is not False,
                'identical': complete and not any_diff,
                'any_divergence': any_diff,
                'first_divergence_index': min(indexes) if indexes else None,
                'first_generated_token_differs': any(p['first_generated_token_differs'] is True for p in pairs) if observable else None,
                'first_token_observable': observable})


def summarize(rows, status, mode, arm):
    complete = [r for r in rows if r.get('complete')]
    eligible = [r for r in complete if r.get('gate_eligible')]
    indexes = [r['first_divergence_index'] for r in eligible if r['first_divergence_index'] is not None]
    units = {p['index_unit'] for r in eligible for p in r['pairs']}
    identical = sum(r['identical'] for r in eligible)
    passed = status == 'COMPLETE' and mode == 'cold' and len(eligible) == 30 and identical == 30
    return {'rows_planned': 30, 'rows_complete': len(complete), 'rows_identical': identical,
            'rows_eligible': len(eligible), 'rows_apc_hit': sum(r.get('status') == 'APC_HIT' for r in rows),
            'rows_with_any_divergence': sum(r['any_divergence'] for r in eligible),
            'first_token_flips': sum(r['first_generated_token_differs'] is True for r in eligible),
            'first_token_unobservable_rows': sum(not r['first_token_observable'] for r in eligible),
            'median_first_divergence': statistics.median(indexes) if indexes and len(units) == 1 else None,
            'divergence_index_unit': next(iter(units)) if len(units) == 1 else 'mixed_or_unavailable',
            'det_align_on_cold_pass': passed if arm == 'ON' and mode == 'cold' else None}


def write_artifacts(report, out_dir):
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report['summary'] = summarize(report['rows'], report['status'], report['apc_mode'], report['det_align'])
    s = report['summary']
    lines = ['# T=0 reproducibility result', '', 'Status: **%s**; det-align label: **%s**; APC mode: **%s**.' %
             (report['status'], report['det_align'], report['apc_mode']),
             'Model: `%s`; requests: %s; elapsed: %.3f s.' %
             (report.get('model', 'unknown'), report['requests_attempted'], report['elapsed_s']), '',
             '- Complete rows: %s/30' % s['rows_complete'],
             '- Eligible rows: %s/30; APC_HIT rows excluded: %s' % (s['rows_eligible'], s['rows_apc_hit']),
             '- Eligible rows identical across all three repeats: %s' % s['rows_identical'],
             '- Eligible rows with any divergence: %s' % s['rows_with_any_divergence'],
             '- First generated token flips: %s (unobservable rows: %s)' %
             (s['first_token_flips'], s['first_token_unobservable_rows']),
             '- Median first divergence (zero-based, %s): %s' %
             (s['divergence_index_unit'], s['median_first_divergence']), '',
             'Pass criterion: det-align ON must complete with 30/30 eligible identical rows in --apc-cold; every hit delta must be zero.',
             'OFF expectation: approximately 20+ diverging rows; this is an expectation, not an observed result.',
             'ON cold verdict: %s.' % ('PASS' if s['det_align_on_cold_pass'] is True else
                                        'FAIL' if s['det_align_on_cold_pass'] is False else 'NOT APPLICABLE'), '',
             'Repeats use byte-identical requests. Cold mode records one fixed nonce per row and resets prefix cache before every generation attempt (HTTP 200 required).',
             'Prefix-cache hit counter: `%s`; before/after values and deltas are recorded per attempt. Cold rows with hits are APC_HIT and excluded from the gate.' % (report.get('prefix_cache_hit_counter') or 'unavailable'),
             'Metrics are polled, so traffic entirely between samples cannot be excluded.',
             'Top-2 margins are unavailable when only one distinct scored token is returned.',
             'Text-only indices count characters in reasoning+content; first-token flips are unknown.', '',
             'Full prompts, nonces, repeat outputs, pairwise divergence evidence, capabilities and metrics: `result.json`.']
    if report.get('error'):
        lines.extend(['', 'Abort/error: `%s`: %s' % (report['status'], report['error'])])
    for name, content in [('result.json', json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'),
                          ('RESULT.md', '\n'.join(lines) + '\n')]:
        temp = directory / ('.' + name + '.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(directory / name)


def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password or parts.fragment or parts.query:
        raise argparse.ArgumentTypeError('Use an http(s) URL without credentials, query, or fragment')
    try:
        parts.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return url.rstrip('/')


def positive_capped(value, cap):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= cap:
        raise argparse.ArgumentTypeError('must be positive and <= %s' % cap)
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', type=validate_url, default='http://127.0.0.1:8095/v1')
    p.add_argument('--metrics-url', type=validate_url, help='Default: server root + /metrics')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--apc-cold', dest='apc_mode', action='store_const', const='cold')
    mode.add_argument('--apc-warm', dest='apc_mode', action='store_const', const='warm')
    p.set_defaults(apc_mode='cold')
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--request-timeout', type=lambda v: positive_capped(v, 300), default=300.0)
    p.add_argument('--total-cap', type=lambda v: positive_capped(v, 1800), default=1800.0)
    p.add_argument('--metrics-interval', type=lambda v: positive_capped(v, 1), default=0.25)
    p.add_argument('--out-dir', default='t0-result')
    p.add_argument('--det-align', choices=['OFF', 'ON', 'unknown'], default='unknown', help='Metadata only; never changes the server')
    p.add_argument('--boot-label', default='unknown', help='Operator-supplied boot identity, not independently verified')
    p.add_argument('--dry-run', action='store_true', help='Print full plan; no HTTP or file writes')
    return p


def run(args):
    started = time.monotonic()
    root = server_root(args.base_url)
    metrics_url = args.metrics_url or root + '/metrics'
    reset_url = root + '/reset_prefix_cache'
    report = {'schema_version': 2, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'base_url': args.base_url, 'metrics_url': metrics_url, 'model': None,
              'reset_prefix_cache_url': reset_url if args.apc_mode == 'cold' else None,
              'prefix_cache_hit_counter': None,
              'apc_mode': args.apc_mode, 'det_align': args.det_align, 'boot_label': args.boot_label,
              'configuration': {'temperature': 0, 'top_p': 1, 'max_tokens': 64, 'seed': args.seed,
                                'stream': False, 'logprobs': True, 'top_logprobs': 1,
                                'repeats': REPEATS, 'concurrency': 1,
                                'request_timeout_s': args.request_timeout, 'total_cap_s': args.total_cap,
                                'metrics_interval_s': args.metrics_interval},
              'capabilities': {'top_logprobs': 'unconfirmed', 'logprobs': 'unconfirmed'},
              'panel_sha256': hashlib.sha256(json.dumps(PANEL).encode()).hexdigest(),
              'status': 'RUNNING', 'requests_attempted': 0, 'rows': [], 'metrics_samples': []}
    abort = threading.Event()
    transport = Transport(started + args.total_cap, args.request_timeout, abort)
    guard = Guard(transport, metrics_url, args.metrics_interval)
    use_top = True
    try:
        guard.check(idle=True)
        catalog = transport.json(args.base_url + '/models')
        report['model_catalog'] = catalog
        models = catalog.get('data', [])
        if len(models) != 1 or not isinstance(models[0].get('id'), str) or not models[0]['id']:
            raise ProbeError('MODEL_ERROR', 'Expected exactly one model id from /v1/models')
        report['model'] = models[0]['id']
        guard.check(idle=True)
        guard.start()
        nonces = set()
        for number, (category, prompt) in enumerate(PANEL, 1):
            nonce = None
            if args.apc_mode == 'cold':
                nonce = secrets.token_hex(NONCE_BYTES)
                while nonce in nonces:
                    nonce = secrets.token_hex(NONCE_BYTES)
                nonces.add(nonce)
            prefix = ('nonce:' + nonce + '\n') if nonce else ''
            row = {'row': number, 'category': category, 'prompt': prompt, 'nonce': nonce,
                   'apc_mode': args.apc_mode, 'repeats': []}
            report['rows'].append(row)
            for repeat in range(1, REPEATS + 1):
                guard.check(idle=True)  # Endpoint must be fully idle before each reset+generation.
                request = {'model': report['model'], 'messages': [{'role': 'user', 'content': prefix + prompt}],
                           'temperature': 0, 'top_p': 1, 'max_tokens': 64, 'seed': args.seed,
                           'stream': False, 'logprobs': True}
                if use_top:
                    request['top_logprobs'] = 1
                record = {'repeat': repeat, 'nonce': nonce, 'nonce_line': prefix or None,
                          'request': request, 'attempts': []}
                row['repeats'].append(record)
                began = time.monotonic()
                try:
                    while True:
                        body = request_bytes(request)
                        previous = [r['request_body'] for r in row['repeats'] if 'request_body' in r]
                        if previous and body.decode('utf-8') != previous[0]:
                            raise ProbeError('INPUT_MISMATCH', 'Request bytes changed within row %s' % number)
                        attempt = {'request': dict(request), 'request_body': body.decode('utf-8'),
                                   'request_sha256': hashlib.sha256(body).hexdigest()}
                        record['attempts'].append(attempt)
                        guard.check(idle=True)
                        if args.apc_mode == 'cold':
                            attempt['reset'] = reset_prefix_cache(transport, reset_url)
                            guard.check(idle=True)
                        before = prefix_cache_hits(transport.call(metrics_url), report['prefix_cache_hit_counter'])
                        report['prefix_cache_hit_counter'] = before['counter']
                        hits = {'counter': before['counter'], 'before': before, 'after': None, 'delta': None}
                        attempt['prefix_cache_hits'] = hits
                        report['requests_attempted'] += 1
                        try:
                            try:
                                response = transport.json(args.base_url + '/chat/completions', request)
                            finally:
                                # Preserve any available hit evidence on rejected/failed attempts too.
                                if not abort.is_set() and time.monotonic() < transport.deadline:
                                    after = prefix_cache_hits(transport.call(metrics_url), before['counter'])
                                    hits['after'] = after
                                    hits['delta'] = hit_delta(before, after)
                            record['request_body'] = attempt['request_body']
                            record['request_sha256'] = attempt['request_sha256']
                            record['prefix_cache_hits'] = hits
                            break
                        except HTTPError as exc:
                            attempt['error'] = str(exc)
                            # One narrowly scoped capability fallback, no generation retries.
                            if use_top and exc.code in (400, 422) and re.search(r'top_logprobs', exc.body, re.I) and re.search(r'unsupported|not supported|unknown|unrecognized|extra|not permitted', exc.body, re.I):
                                use_top = False
                                report['capabilities']['top_logprobs'] = 'unsupported'
                                request.pop('top_logprobs')
                                guard.check()
                                continue
                            raise
                    record['output'] = output_record(response)
                    entries = record['output']['tokens']
                    if entries:
                        report['capabilities']['logprobs'] = 'returned'
                    if use_top and any(e.get('top_logprobs') for e in entries):
                        report['capabilities']['top_logprobs'] = 'returned'
                    guard.check()
                except ProbeError as exc:
                    record['error'] = {'status': exc.status, 'detail': str(exc)}
                    raise
                finally:
                    record['elapsed_s'] = round(time.monotonic() - began, 6)
            analyze_row(row)
            print('row %02d/30 %s: %s' % (number, category, row['status'] if row['status'] == 'APC_HIT' else
                                        'identical' if row['identical'] else 'DIVERGING'), flush=True)
        guard.close()
        if guard.error:
            raise guard.error
        guard.check()
        report['status'] = 'COMPLETE'
    except KeyboardInterrupt:
        report['status'] = 'INTERRUPTED'
        report['error'] = 'Operator interruption'
        abort.set()
    except ProbeError as exc:
        cause = guard.error or exc
        report['status'], report['error'] = cause.status, str(cause)
    except Exception as exc:
        report['status'], report['error'] = 'ERROR', '%s: %s' % (type(exc).__name__, exc)
    finally:
        guard.close()
        report['elapsed_s'] = round(time.monotonic() - started, 6)
        with guard.lock:
            report['metrics_samples'] = list(guard.samples)
        for row in report['rows']:
            if 'complete' not in row:
                try:
                    analyze_row(row)
                except ProbeError:
                    row.update(complete=False, pairs=[], identical=False, any_divergence=False,
                               first_divergence_index=None, first_generated_token_differs=None,
                               first_token_observable=False)
        write_artifacts(report, args.out_dir)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps({'base_url': args.base_url, 'model': 'discover exactly one id from /v1/models',
                          'metrics_url': args.metrics_url or server_root(args.base_url) + '/metrics',
                          'reset_prefix_cache_url': server_root(args.base_url) + '/reset_prefix_cache' if args.apc_mode == 'cold' else None,
                          'cold_protocol': 'one recorded nonce per row; byte-identical repeats; reset HTTP 200 before every attempt; exclude APC_HIT rows',
                          'requests': 90, 'concurrency': 1, 'repeats': REPEATS,
                          'parameters': {'temperature': 0, 'top_p': 1, 'max_tokens': 64, 'seed': args.seed,
                                         'stream': False, 'logprobs': True, 'top_logprobs': 1},
                          'apc_mode': args.apc_mode, 'nonce_line': 'nonce:<32 random recorded hex characters, fixed per row>\\n' if args.apc_mode == 'cold' else None,
                          'request_timeout_s': args.request_timeout, 'total_cap_s': args.total_cap,
                          'safety': 'idle running=waiting=0; poll throughout; abort running>1 or metrics failure',
                          'out_dir': args.out_dir, 'det_align': args.det_align, 'boot_label': args.boot_label,
                          'panel': [{'row': i, 'category': c, 'prompt': p} for i, (c, p) in enumerate(PANEL, 1)]}, indent=2))
        return 0
    report = run(args)
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'out_dir': args.out_dir}, indent=2))
    if report['status'] != 'COMPLETE':
        return 2
    if args.det_align == 'ON' and args.apc_mode == 'cold' and not report['summary']['det_align_on_cold_pass']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
