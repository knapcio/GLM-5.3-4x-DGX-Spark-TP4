# SPDX-License-Identifier: Apache-2.0
"""CPU-only fake HTTP server tests; socket pairs by default, no network.

Set T0_TEST_TCP=1 to use ephemeral loopback TCP listeners instead.
Both paths exercise http.client against BaseHTTPRequestHandler wire responses.
"""
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import t0_probe as probe  # noqa: E402


def completion(words, ids=False, top_two=False):
    entries = []
    for word in words:
        item = {'token': word, 'bytes': list(word.encode()), 'logprob': -0.6932,
                'top_logprobs': [{'token': word, 'bytes': list(word.encode()), 'logprob': -0.6932}]}
        if top_two:
            item['top_logprobs'].append({'token': 'other', 'bytes': list(b'other'), 'logprob': -0.6933})
        entries.append(item)
    choice = {'message': {'role': 'assistant', 'content': ''.join(words)},
              'finish_reason': 'stop', 'logprobs': {'content': entries}}
    if ids:
        choice['token_ids'] = [ord(w[0]) for w in words]
    return {'choices': [choice], 'usage': {'prompt_tokens': 65, 'completion_tokens': len(words)}}


class State:
    def __init__(self, mode):
        self.mode = mode
        self.requests = []
        self.request_bodies = []
        self.paths = []
        self.resets = 0
        self.hits = 100
        self.running = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.foreign_seen = False


@contextlib.contextmanager
def fake_server(mode='identical'):
    state = State(mode)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def send(self, data, code=200, content_type='application/json'):
            body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            try:
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            with state.lock:
                state.paths.append(self.path)
                active = state.active
                if self.path == '/metrics':
                    hits = state.hits
                    running = state.running
                    waiting = 0
                    if mode == 'idle_foreign':
                        waiting = 1
                    if mode == 'foreign' and state.started.is_set():
                        running = 2
                        state.foreign_seen = True
                    if mode == 'missing_metrics':
                        self.send('vllm:num_requests_running 0\n', content_type='text/plain')
                        return
            if self.path == '/metrics':
                counter = 'vllm:gpu_prefix_cache_hits' if mode == 'legacy_hits' else 'vllm:prefix_cache_hits_total'
                hit_line = '%s{model_name="fake"} %s\n' % (counter, hits)
                if mode == 'missing_hits':
                    hit_line = ''
                if mode == 'invalid_hits':
                    hit_line = '%s NaN\n' % counter
                self.send('vllm:num_requests_running{model_name="fake"} %s\n'
                          'vllm:num_requests_waiting{model_name="fake"} %s\n' % (running, waiting) + hit_line,
                          content_type='text/plain')
            elif self.path == '/v1/models':
                self.send({'data': [{'id': 'glm-5.3-fake'}]})
            else:
                self.send({'error': 'unknown path'}, 404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            data = json.loads(body)
            if self.path == '/reset_prefix_cache':
                with state.lock:
                    state.paths.append(self.path)
                    state.resets += 1
                    resets = state.resets
                    if mode == 'reset_counters':
                        state.hits = 0
                if mode.startswith('reset_status_'):
                    self.send('', int(mode.rsplit('_', 1)[1]))
                elif mode == 'late_reset_failure' and resets == 2:
                    self.send('reset failed', 500)
                else:
                    self.send('')  # The real dev endpoint may return an empty HTTP 200.
                return
            with state.lock:
                state.paths.append(self.path)
                state.requests.append(data)
                state.request_bodies.append(body)
                n = len(state.requests) - 1
                state.active += 1
                state.max_active = max(state.max_active, state.active)
                state.running = 1
                state.started.set()
            try:
                if mode in ('timeout', 'foreign', 'total_timeout'):
                    state.release.wait(1.5)
                if ((mode in ('unsupported_top', 'unsupported_top_hit') or mode == 'late_unsupported_top' and n == 1)
                        and 'top_logprobs' in data):
                    if mode == 'unsupported_top_hit':
                        with state.lock:
                            state.hits += 1
                    self.send({'error': 'top_logprobs is not supported'}, 400)
                    return
                if mode == 'bad_request':
                    self.send({'error': 'bad top_logprobs value'}, 400)
                    return
                words = ['a', 'b', 'c']
                if mode in ('diverging', 'hit_diverging') and n % 3 == 1:
                    words[1] = 'x'
                if mode == 'text_diverging' and n % 3 == 1:
                    words[1] = 'x'
                if mode == 'first_flip' and n % 3 == 1:
                    words[0] = 'z'
                if mode == 'length' and n % 3 == 1:
                    words.pop()
                response = completion(words, ids=mode == 'ids', top_two=mode == 'diverging')
                if mode in ('text_only', 'text_diverging'):
                    response['choices'][0].pop('logprobs')
                if mode == 'empty':
                    response['choices'][0]['message']['content'] = ''
                    response['choices'][0]['logprobs']['content'] = []
                with state.lock:
                    if (mode in ('hit_first', 'hit_diverging') and n == 0) or mode == 'warm_hits':
                        state.hits += 4
                    if mode == 'decreasing_hits':
                        state.hits -= 1
                self.send(response)
            finally:
                with state.lock:
                    state.active -= 1
                    state.running = 0

    if os.environ.get('T0_TEST_TCP') == '1':
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            yield state, 'http://127.0.0.1:%s/v1' % server.server_port
        finally:
            state.release.set()
            server.shutdown()
            server.server_close()
            thread.join(1)
    else:
        peers = []
        threads = []
        server = SimpleNamespace(server_name='fake', server_port=8095)

        def connect(connection):
            client, peer = socket.socketpair()
            client.settimeout(connection.timeout)
            connection.sock = client
            peers.append(peer)

            def serve():
                try:
                    Handler(peer, ('127.0.0.1', 0), server)
                except (OSError, ValueError):
                    pass  # Cancellation deliberately closes the client stream.
                finally:
                    peer.close()

            thread = threading.Thread(target=serve, daemon=True)
            threads.append(thread)
            thread.start()

        with patch.object(probe.http.client.HTTPConnection, 'connect', connect):
            try:
                yield state, 'http://127.0.0.1:8095/v1'
            finally:
                state.release.set()
                for thread in threads:
                    thread.join(.1)
                for peer in peers:
                    peer.close()


class ProbeTests(unittest.TestCase):
    def execute(self, mode='identical', extra=()):
        with fake_server(mode) as (state, url), tempfile.TemporaryDirectory() as out:
            args = probe.parser().parse_args(['--base-url', url, '--out-dir', out,
                                             '--request-timeout', '2', '--total-cap', '15',
                                             '--metrics-interval', '.01'] + list(extra))
            with contextlib.redirect_stdout(io.StringIO()):
                report = probe.run(args)
            stored = json.loads((Path(out) / 'result.json').read_text())
            self.assertEqual(stored, report)
            markdown = (Path(out) / 'RESULT.md').read_text()
            return report, state, markdown

    def test_identical_full_cold_panel_and_parameters(self):
        report, state, md = self.execute(extra=['--det-align', 'ON'])
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['summary']['rows_identical'], 30)
        self.assertTrue(report['summary']['det_align_on_cold_pass'])
        self.assertEqual(len(state.requests), 90)
        self.assertEqual(state.max_active, 1)
        self.assertIn('PASS', md)
        nonces = []
        for index, row in enumerate(report['rows']):
            self.assertEqual(len(row['repeats']), 3)
            self.assertEqual(len(row['pairs']), 3)
            bodies = state.request_bodies[index * 3:index * 3 + 3]
            self.assertEqual(bodies, [bodies[0]] * 3)
            self.assertEqual([r['request_body'].encode() for r in row['repeats']], bodies)
            self.assertEqual(len({r['request_sha256'] for r in row['repeats']}), 1)
            self.assertEqual({r['nonce'] for r in row['repeats']}, {row['nonce']})
            self.assertTrue(row['gate_eligible'])
            self.assertTrue(row['inputs_identical'])
            for repeat in row['repeats']:
                request = repeat['request']
                self.assertEqual(request['model'], 'glm-5.3-fake')
                for key, value in {'temperature': 0, 'top_p': 1, 'max_tokens': 64,
                                   'seed': 1729, 'stream': False, 'logprobs': True, 'top_logprobs': 1}.items():
                    self.assertEqual(request[key], value)
                nonce = repeat['nonce']
                self.assertEqual(len(nonce), 32)
                nonces.append(nonce)
                self.assertEqual(request['messages'][0]['content'], 'nonce:' + nonce + '\n' + row['prompt'])
                self.assertEqual(repeat['output']['tokens'][0]['bytes'], [97])
                self.assertEqual(repeat['prefix_cache_hits']['counter'], 'vllm:prefix_cache_hits_total')
                self.assertEqual(repeat['prefix_cache_hits']['delta'], 0)
                self.assertEqual(repeat['attempts'][0]['reset']['http_status'], 200)
        self.assertEqual(len(set(nonces)), 30)
        self.assertEqual(state.resets, 90)
        operations = [path for path in state.paths if path != '/metrics' and path != '/v1/models']
        self.assertEqual(operations, ['/reset_prefix_cache', '/v1/chat/completions'] * 90)
        self.assertEqual(report['prefix_cache_hit_counter'], 'vllm:prefix_cache_hits_total')
        self.assertTrue(report['metrics_samples'][0]['idle_check'])

    def test_warm_identical_prompts(self):
        report, state, _ = self.execute(extra=['--apc-warm'])
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(state.resets, 0)
        for index, row in enumerate(report['rows']):
            contents = [r['request']['messages'][0]['content'] for r in row['repeats']]
            self.assertEqual(contents, [row['prompt']] * 3)
            self.assertTrue(all(r['nonce'] is None for r in row['repeats']))
            bodies = state.request_bodies[index * 3:index * 3 + 3]
            self.assertEqual(bodies, [bodies[0]] * 3)

    def test_reset_requires_exact_http_200_and_aborts_before_generation(self):
        for code in (202, 204, 302, 404, 500):
            with self.subTest(code=code):
                report, state, md = self.execute('reset_status_%s' % code, ['--det-align', 'ON'])
                self.assertEqual(report['status'], 'RESET_UNAVAILABLE')
                self.assertIn('HTTP %s' % code, report['error'])
                self.assertEqual(state.requests, [])
                self.assertEqual(state.resets, 1)
                self.assertEqual(report['requests_attempted'], 0)
                self.assertFalse(report['summary']['det_align_on_cold_pass'])
                self.assertIn('RESET_UNAVAILABLE', md)

    def test_late_reset_failure_preserves_partial_results(self):
        report, state, _ = self.execute('late_reset_failure')
        self.assertEqual(report['status'], 'RESET_UNAVAILABLE')
        self.assertEqual(len(state.requests), 1)
        self.assertEqual(state.resets, 2)
        self.assertIn('output', report['rows'][0]['repeats'][0])
        self.assertEqual(report['rows'][0]['repeats'][1]['error']['status'], 'RESET_UNAVAILABLE')

    def test_reset_transport_failure_is_reset_unavailable(self):
        transport = probe.Transport(time.monotonic() + 1, 1, threading.Event())
        with patch.object(transport, 'call', side_effect=probe.ProbeError('HTTP_ERROR', 'connection refused')):
            with self.assertRaises(probe.ProbeError) as raised:
                probe.reset_prefix_cache(transport, 'http://fake/reset_prefix_cache')
        self.assertEqual(raised.exception.status, 'RESET_UNAVAILABLE')

    def test_cold_hit_row_excluded_even_with_identical_outputs(self):
        report, _, md = self.execute('hit_first', ['--det-align', 'ON'])
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['summary']['rows_complete'], 30)
        self.assertEqual(report['summary']['rows_eligible'], 29)
        self.assertEqual(report['summary']['rows_identical'], 29)
        self.assertEqual(report['summary']['rows_apc_hit'], 1)
        self.assertFalse(report['summary']['det_align_on_cold_pass'])
        row = report['rows'][0]
        self.assertEqual(row['status'], 'APC_HIT')
        self.assertTrue(row['identical'])
        self.assertFalse(row['gate_eligible'])
        self.assertEqual([r['prefix_cache_hits']['delta'] for r in row['repeats']], [4, 0, 0])
        self.assertEqual(row['repeats'][0]['prefix_cache_hits']['before']['value'], 100)
        self.assertEqual(row['repeats'][0]['prefix_cache_hits']['after']['value'], 104)
        self.assertIn('APC_HIT rows excluded: 1', md)
        self.assertNotIn('nonces change input', md)

    def test_hit_row_excluded_from_divergence_statistics(self):
        report, _, _ = self.execute('hit_diverging')
        self.assertEqual(report['summary']['rows_with_any_divergence'], 29)
        self.assertTrue(report['rows'][0]['any_divergence'])
        self.assertEqual(report['rows'][0]['status'], 'APC_HIT')

    def test_warm_hits_recorded_without_reset_or_exclusion(self):
        report, state, _ = self.execute('warm_hits', ['--apc-warm', '--det-align', 'ON'])
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(state.resets, 0)
        self.assertEqual(report['summary']['rows_apc_hit'], 0)
        self.assertEqual(report['summary']['rows_identical'], 30)
        self.assertIsNone(report['summary']['det_align_on_cold_pass'])
        self.assertTrue(all(r['prefix_cache_hits']['delta'] == 4 for row in report['rows'] for r in row['repeats']))

    def test_legacy_hit_counter_recorded(self):
        report, _, _ = self.execute('legacy_hits', ['--det-align', 'ON'])
        self.assertTrue(report['summary']['det_align_on_cold_pass'])
        self.assertEqual(report['prefix_cache_hit_counter'], 'vllm:gpu_prefix_cache_hits')
        self.assertIn('vllm:gpu_prefix_cache_hits', report['rows'][0]['repeats'][0]['prefix_cache_hits']['counter'])

    def test_missing_or_invalid_hit_counter_fails_closed(self):
        for mode in ('missing_hits', 'invalid_hits'):
            with self.subTest(mode=mode):
                report, state, _ = self.execute(mode, ['--det-align', 'ON'])
                self.assertEqual(report['status'], 'METRICS_ERROR')
                self.assertEqual(state.requests, [])
                self.assertFalse(report['summary']['det_align_on_cold_pass'])

    def test_hit_counter_decrease_cannot_pass_as_cold(self):
        report, state, _ = self.execute('decreasing_hits', ['--det-align', 'ON'])
        self.assertEqual(report['status'], 'METRICS_ERROR')
        self.assertEqual(len(state.requests), 1)
        self.assertFalse(report['summary']['det_align_on_cold_pass'])

    def test_counter_reset_before_snapshot_is_allowed(self):
        report, _, _ = self.execute('reset_counters', ['--det-align', 'ON'])
        self.assertTrue(report['summary']['det_align_on_cold_pass'])
        self.assertEqual(report['rows'][0]['repeats'][0]['prefix_cache_hits']['before']['value'], 0)

    def test_divergence_and_near_tie(self):
        report, _, _ = self.execute('diverging', ['--det-align', 'ON'])
        s = report['summary']
        self.assertEqual(s['rows_with_any_divergence'], 30)
        self.assertEqual(s['rows_identical'], 0)
        self.assertEqual(s['median_first_divergence'], 1)
        self.assertEqual(s['first_token_flips'], 0)
        self.assertFalse(s['det_align_on_cold_pass'])
        for row in report['rows']:
            evidence = row['pairs'][0]['near_tie']['left']
            self.assertTrue(evidence['available'])
            self.assertAlmostEqual(evidence['margin'], .0001)

    def test_first_token_flip(self):
        report, _, _ = self.execute('first_flip')
        self.assertEqual(report['summary']['first_token_flips'], 30)
        self.assertEqual(report['summary']['median_first_divergence'], 0)
        self.assertFalse(report['rows'][0]['pairs'][0]['near_tie']['left']['available'])

    def test_foreign_traffic_aborts_during_pending_request(self):
        began = time.monotonic()
        report, state, md = self.execute('foreign')
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(report['status'], 'FOREIGN_TRAFFIC')
        self.assertEqual(len(state.requests), 1)
        self.assertTrue(state.foreign_seen)
        self.assertIn('FOREIGN_TRAFFIC', md)
        self.assertEqual(report['summary']['rows_complete'], 0)
        self.assertTrue(any(s['running'] > 1 for s in report['metrics_samples']))

    def test_busy_initial_waiting_no_generation(self):
        report, state, _ = self.execute('idle_foreign')
        self.assertEqual(report['status'], 'FOREIGN_TRAFFIC')
        self.assertEqual(state.requests, [])
        self.assertNotIn('/v1/models', state.paths)

    def test_request_timeout_preserves_partial_and_aborts(self):
        began = time.monotonic()
        report, state, md = self.execute('timeout', ['--request-timeout', '.12'])
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(report['status'], 'REQUEST_TIMEOUT')
        self.assertEqual(len(state.requests), 1)
        self.assertEqual(report['rows'][0]['repeats'][0]['error']['status'], 'REQUEST_TIMEOUT')
        self.assertIn('REQUEST_TIMEOUT', md)
        self.assertEqual(report['summary']['rows_identical'], 0)

    def test_total_timeout_caps_pending_request(self):
        began = time.monotonic()
        report, state, _ = self.execute('total_timeout', ['--total-cap', '.15'])
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(report['status'], 'TOTAL_TIMEOUT')
        self.assertEqual(len(state.requests), 1)

    def test_missing_metrics_fails_closed(self):
        report, state, _ = self.execute('missing_metrics')
        self.assertEqual(report['status'], 'METRICS_ERROR')
        self.assertEqual(state.requests, [])

    def test_ids_preferred_and_logprob_tokens_retained(self):
        report, _, _ = self.execute('ids')
        output = report['rows'][0]['repeats'][0]['output']
        self.assertEqual(output['basis'], 'token_ids')
        self.assertEqual(output['sequence'], [97, 98, 99])
        self.assertEqual(output['tokens'][0]['token'], 'a')

    def test_text_fallback_does_not_guess_first_token(self):
        report, _, _ = self.execute('text_only')
        self.assertEqual(report['summary']['rows_identical'], 30)
        self.assertEqual(report['summary']['first_token_unobservable_rows'], 30)
        self.assertIsNone(report['rows'][0]['pairs'][0]['first_generated_token_differs'])
        self.assertIsNone(report['rows'][0]['first_generated_token_differs'])

    def test_text_fallback_divergence_uses_character_offset(self):
        report, _, _ = self.execute('text_diverging')
        self.assertEqual(report['summary']['rows_with_any_divergence'], 30)
        self.assertEqual(report['summary']['median_first_divergence'], 1)
        self.assertEqual(report['summary']['divergence_index_unit'], 'text_character')
        self.assertEqual(report['summary']['first_token_unobservable_rows'], 30)

    def test_top_one_and_selected_token_do_not_establish_top_two(self):
        response = completion(['a'])
        entry = response['choices'][0]['logprobs']['content'][0]
        entry['top_logprobs'] = [{'token': 'b', 'bytes': [98], 'logprob': -.1}]
        entry['logprob'] = -2  # Selected token could be rank 3 or lower.
        self.assertFalse(probe.margin_at(probe.output_record(response), 0)['available'])

    def test_recorded_bytes_detect_token_difference(self):
        a = probe.output_record(completion(['same']))
        b = probe.output_record(completion(['same']))
        b['sequence'][0]['bytes'] = [1, 2]
        row = {'repeats': [{'output': a}, {'output': b}, {'output': a}]}
        probe.analyze_row(row)
        self.assertEqual(row['first_divergence_index'], 0)
        self.assertTrue(row['first_generated_token_differs'])

    def test_preexisting_safety_abort_never_starts_http(self):
        abort = threading.Event()
        abort.set()
        transport = probe.Transport(time.monotonic() + 1, 1, abort)
        with patch.object(probe.http.client, 'HTTPConnection') as connection:
            with self.assertRaises(probe.ProbeError) as raised:
                transport.call('http://127.0.0.1:8095/v1/chat/completions', {})
            self.assertEqual(raised.exception.status, 'SAFETY_ABORT')
            connection.assert_not_called()

    def test_cli_exit_codes_and_warm_not_cold_gate(self):
        # Full fake HTTP execution, including JSON/Markdown and CLI exit logic.
        for mode, extra, expected in [('identical', ['--apc-cold'], 0),
                                      ('diverging', ['--apc-cold'], 1),
                                      ('hit_first', ['--apc-cold'], 1),
                                      ('diverging', ['--apc-warm'], 0),
                                      ('reset_status_404', ['--apc-cold'], 2),
                                      ('foreign', ['--apc-cold'], 2)]:
            with self.subTest(mode=mode, extra=extra), fake_server(mode) as (_, url), tempfile.TemporaryDirectory() as out:
                with contextlib.redirect_stdout(io.StringIO()):
                    actual = probe.main(['--base-url', url, '--out-dir', out, '--det-align', 'ON',
                                         '--metrics-interval', '.01'] + extra)
                self.assertEqual(actual, expected)
                report = json.loads((Path(out) / 'result.json').read_text())
                if '--apc-warm' in extra:
                    self.assertIsNone(report['summary']['det_align_on_cold_pass'])

    def test_length_difference(self):
        report, _, _ = self.execute('length')
        self.assertEqual(report['summary']['rows_with_any_divergence'], 30)
        self.assertEqual(report['summary']['median_first_divergence'], 2)

    def test_unsupported_top_logprobs_only_retries_once(self):
        report, state, _ = self.execute('unsupported_top')
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(len(state.requests), 91)
        self.assertEqual(report['capabilities']['top_logprobs'], 'unsupported')
        self.assertEqual(sum('top_logprobs' in r for r in state.requests), 1)
        first = report['rows'][0]['repeats'][0]
        self.assertEqual(len(first['attempts']), 2)
        self.assertEqual(state.requests[0]['messages'], state.requests[1]['messages'])
        self.assertEqual(state.resets, 91)
        operations = [path for path in state.paths if path not in ('/metrics', '/v1/models')]
        self.assertEqual(operations, ['/reset_prefix_cache', '/v1/chat/completions'] * 91)
        self.assertEqual(state.request_bodies[1:4], [state.request_bodies[1]] * 3)
        self.assertTrue(all(row['inputs_identical'] for row in report['rows']))

    def test_hits_on_rejected_attempt_still_exclude_row(self):
        report, state, _ = self.execute('unsupported_top_hit', ['--det-align', 'ON'])
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(state.resets, 91)
        row = report['rows'][0]
        self.assertEqual(row['status'], 'APC_HIT')
        self.assertEqual(row['repeats'][0]['attempts'][0]['prefix_cache_hits']['delta'], 1)
        self.assertFalse(report['summary']['det_align_on_cold_pass'])

    def test_late_capability_change_cannot_change_input_across_repeats(self):
        report, state, _ = self.execute('late_unsupported_top', ['--det-align', 'ON'])
        self.assertEqual(report['status'], 'INPUT_MISMATCH')
        self.assertEqual(len(state.requests), 2)
        self.assertEqual(state.request_bodies, [state.request_bodies[0]] * 2)
        self.assertFalse(report['summary']['det_align_on_cold_pass'])

    def test_other_http_error_not_retried(self):
        report, state, _ = self.execute('bad_request')
        self.assertEqual(report['status'], 'HTTP_ERROR')
        self.assertEqual(len(state.requests), 1)

    def test_empty_response_cannot_pass(self):
        report, _, _ = self.execute('empty', ['--det-align', 'ON'])
        self.assertEqual(report['status'], 'INVALID_RESPONSE')
        self.assertFalse(report['summary']['det_align_on_cold_pass'])

    def test_dry_run_no_http_or_files(self):
        with tempfile.TemporaryDirectory() as parent, patch.object(probe.Transport, 'call', side_effect=AssertionError('HTTP forbidden')):
            out = Path(parent) / 'absent'
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(probe.main(['--dry-run', '--out-dir', str(out)]), 0)
            plan = json.loads(captured.getvalue())
            self.assertEqual(len(plan['panel']), 30)
            self.assertEqual(plan['requests'], 90)
            self.assertEqual(plan['apc_mode'], 'cold')
            self.assertEqual(plan['reset_prefix_cache_url'], 'http://127.0.0.1:8095/reset_prefix_cache')
            self.assertIn('fixed per row', plan['nonce_line'])
            self.assertFalse(out.exists())

    def test_panel_mix_and_reasonable_size(self):
        self.assertEqual(len(probe.PANEL), 30)
        for category in ('prose', 'code', 'reasoning'):
            self.assertEqual(sum(c == category for c, _ in probe.PANEL), 10)
        for _, text in probe.PANEL:
            self.assertGreaterEqual(len(text.split()), 35)
            self.assertLess(len(text.split()), 150)

    def test_metrics_labels_and_invalid_values(self):
        self.assertEqual(probe.metrics_counts('vllm:num_requests_running{engine="0"} 1\n'
                                              'vllm:num_requests_running{engine="1"} 0\n'
                                              'vllm:num_requests_waiting 0 1234\n'), {'running': 1, 'waiting': 0})
        for value in ('NaN', '+Inf', '-1'):
            with self.assertRaises(probe.ProbeError):
                probe.metrics_counts('vllm:num_requests_running %s\nvllm:num_requests_waiting 0' % value)

    def test_hit_counter_aliases_labels_and_preference(self):
        for name in ('vllm:prefix_cache_hits_total', 'vllm:gpu_prefix_cache_hits_total',
                     'vllm:gpu_prefix_cache_hits', 'gpu_prefix_cache_hits_total', 'gpu_prefix_cache_hits'):
            with self.subTest(name=name):
                snapshot = probe.prefix_cache_hits('%s{engine="0"} 10\n%s{engine="1"} 20 123\n' % (name, name))
                self.assertEqual(snapshot['counter'], name)
                self.assertEqual(snapshot['value'], 30)
                self.assertEqual(len(snapshot['series']), 2)
        snapshot = probe.prefix_cache_hits('vllm:prefix_cache_hits_total 10\nvllm:gpu_prefix_cache_hits 999\n')
        self.assertEqual(snapshot['value'], 10)
        self.assertEqual(snapshot['counter'], 'vllm:prefix_cache_hits_total')
        for raw in ('NaN', '+Inf', '-1', 'bad'):
            with self.subTest(raw=raw), self.assertRaises(probe.ProbeError):
                probe.prefix_cache_hits('vllm:prefix_cache_hits_total %s\n' % raw)

    def test_counter_family_or_series_change_fails_closed(self):
        before = probe.prefix_cache_hits('vllm:prefix_cache_hits_total{engine="0"} 10\n')
        after = probe.prefix_cache_hits('vllm:prefix_cache_hits_total{engine="1"} 10\n')
        with self.assertRaises(probe.ProbeError):
            probe.hit_delta(before, after)
        with self.assertRaises(probe.ProbeError):
            probe.prefix_cache_hits('vllm:gpu_prefix_cache_hits 10\n', before['counter'])
        # A decrease in one series must not be masked by an increase in another.
        before = probe.prefix_cache_hits('vllm:prefix_cache_hits_total{engine="0"} 10\nvllm:prefix_cache_hits_total{engine="1"} 10\n')
        after = probe.prefix_cache_hits('vllm:prefix_cache_hits_total{engine="0"} 9\nvllm:prefix_cache_hits_total{engine="1"} 11\n')
        with self.assertRaises(probe.ProbeError):
            probe.hit_delta(before, after)

    def test_server_root_removes_only_trailing_v1(self):
        self.assertEqual(probe.server_root('http://localhost:8095/v1'), 'http://localhost:8095')
        self.assertEqual(probe.server_root('https://example.invalid/proxy/v1/'), 'https://example.invalid/proxy')
        self.assertEqual(probe.server_root('http://localhost:8095'), 'http://localhost:8095')

    def test_cli_enforces_caps_and_exclusive_modes(self):
        for argv in (['--request-timeout', '301'], ['--total-cap', '1801'],
                     ['--total-cap', '0'], ['--apc-cold', '--apc-warm']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parser().parse_args(argv)


if __name__ == '__main__':
    unittest.main()
