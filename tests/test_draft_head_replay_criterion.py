# SPDX-License-Identifier: Apache-2.0
"""Actual shared qualifier: write coverage, exact tokens and bounded floats."""
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import torch
from test_draft_head_init import fixture, boot, dh, qual

@dataclass(frozen=True)
class PrefillDesc:
    cg_mode: str = 'FULL'
    num_tokens: int = 16
    num_reqs: int = 4
    uniform_token_count: int = 4
    short_context: bool = False
    num_active_loras: int = 0


def descriptor_fixture(fault='none', descriptor=PrefillDesc()):
    r = fixture(); sp = r.speculator
    sp.max_num_reqs = 4
    sp.hidden_states = torch.zeros(16, 256, dtype=torch.bfloat16)
    sp.draft_tokens = torch.zeros(4, 3, dtype=torch.long)
    sp._kstop.confidence = torch.zeros(4)
    sp.temperature = torch.zeros(4)
    sp.idx_mapping = torch.zeros(4, dtype=torch.long)
    sp.last_token_indices = torch.zeros(4, dtype=torch.long)
    count = [0]
    zero = []
    omitted = set()
    executions = {}
    last_inputs = {}
    zero_visits = {}
    def forward(mode, omit=None, manager=0, descriptor=descriptor):
        value = float(sp.hidden_states[4, 0])
        first_hidden = float(sp.hidden_states[0, 0])
        pattern = int(first_hidden < 0) if abs(first_hidden) >= 8 else None
        key = (manager, descriptor)
        inputs = (value, pattern)
        changed = last_inputs.get(key) != inputs
        last_inputs[key] = inputs
        if changed and value == 0:
            zero_visits[key, pattern] = zero_visits.get((key, pattern), 0) + 1
        execution_key = (key, inputs)
        executions[execution_key] = executions.get(execution_key, 0) + int(r._draft_head.on)
        one_omit = (fault == 'one_omit' and manager == 0 and r._draft_head.on
                    and mode is None and value == .125 and key not in omitted)
        if one_omit:
            omitted.add(key)
            omit = 'element_not_written'
        sparse_feedback = sp.hidden_states[2, 7].clone()
        step = int(sp.current_draft_step)
        omit_token = (fault == 'decode_tokens_unwritten' and manager == 1
                      and r._draft_head.on and mode is None and key not in omitted)
        if omit_token: omitted.add(key)
        logits = r._draft_head.bank(sp.hidden_states[:4])
        if omit_token:
            sp.draft_tokens[:2, step].copy_(logits[:2].argmax(-1))
            sp.draft_tokens[3:, step].copy_(logits[3:].argmax(-1))
        else:
            sp.draft_tokens[:, step].copy_(logits.argmax(-1))
        # Omit the actual feedback store, rather than writing a stale output.
        for row in range(4):
            # Native decode leaves hidden input storage untouched.
            if manager == 1 and fault == 'decode_tokens_unwritten': continue
            if omit == 'row_not_written' and row == 2:
                continue
            if omit in ('element_not_written', 'sparse_partial_write') and row == 2:
                sp.hidden_states[row, :7].fill_(2 * value + .25)
                sp.hidden_states[row, 8:].fill_(2 * value + .25)
            else:
                sp.hidden_states[row].fill_(2 * value + .25)
        if fault in ('sparse_partial_write', 'one_omit') and omit not in ('sparse_partial_write', 'element_not_written'):
            sp.hidden_states[2, 7] = torch.nextafter(sparse_feedback, torch.full_like(sparse_feedback, float('inf')))
        if fault == 'signed_zero_nondeterministic': sp.hidden_states[:4, 7].zero_()
        sp._kstop.confidence.fill_(.9 + value / 100)
        if sp.draft_logits is not None:
            sp.draft_logits[:, int(sp.current_draft_step)].fill_(value + 1)
        # Repeated two-state execution order: E=Y, G1=X, G2=Y on 4x4;
        # generic pairs alternate Y/X at the constant +.125 probe too.
        if (r._draft_head.on and fault == 'two_state_flip' and value == .125
                and pattern in (0, None) and executions[execution_key] % 2):
            feedback = sp.hidden_states[:4, 7]
            for _ in range(8):
                feedback.copy_(torch.nextafter(feedback, torch.full_like(feedback, float('inf'))))
            sp._kstop.confidence.add_(3e-7)
        if r._draft_head.on and fault == 'eager_tokens' and mode is not None and value == .125:
            sp.draft_tokens[:, step].add_(1)
        if r._draft_head.on and value == 0 and zero_visits.get((key, pattern), 0) >= 3:
            if fault in ('roundtrip_drift', 'roundtrip_error'):
                feedback = sp.hidden_states[:4, 7]
                for _ in range(8 if fault == 'roundtrip_drift' else 17):
                    feedback.copy_(torch.nextafter(feedback, torch.full_like(feedback, float('inf'))))
            if fault == 'roundtrip_tokens': sp.draft_tokens[:, 0].add_(1)
    def capture():
        for i, manager in enumerate(dh.managers(r)):
            manager._capture_descs = {'FULL': [descriptor]}
            manager._k4_drafthead_factory = lambda desc, warmup, i=i: (
                lambda mode: forward(mode, manager=i, descriptor=desc))
            def replay(i=i):
                value = float(sp.hidden_states[4, 0])
                if fault == 'noop' and i == 0 and r._draft_head.on: return
                forward(None, omit=fault if i == 0 and r._draft_head.on else None, manager=i)
                if i != 0 or not r._draft_head.on: return
                count[0] += 1
                if fault == 'feedback_error': sp.hidden_states[:4, 7].add_(.125)
                if fault == 'tokens' or (fault == 'graph2_tokens' and value == .125 and count[0] % 2 == 0):
                    sp.draft_tokens[:, 0].add_(1)
                if fault == 'hidden_nonfinite': sp.hidden_states[2, 7].fill_(float('nan'))
                if fault == 'nonfinite': sp._kstop.confidence.fill_(float('nan'))
                if fault == 'logits_nonfinite': sp.draft_logits[:, 0].fill_(float('nan'))
                if fault == 'logits_nondeterministic': sp.draft_logits[:, 0, 0].add_(count[0] % 2)
                if fault == 'inactive_logits': sp.draft_logits[:, 1].fill_(3.)
                steps = {'float_drift': 1, 'float_two_ulp': 2, 'float_three_ulp': 3,
                         'float_sixteen_ulp': 16, 'float_seventeen_ulp': 17,
                         'graph1_drift': 8 if count[0] % 2 else 0,
                         'graph2_excess': 8 if count[0] % 2 else 17}.get(fault, 0)
                if steps and value != 0:
                    feedback = sp.hidden_states[:4, 7]
                    if fault == 'float_seventeen_ulp': feedback = feedback[:1]
                    for _ in range(steps):
                        feedback.copy_(torch.nextafter(feedback, torch.full_like(feedback, float('inf'))))
                    sp._kstop.confidence.add_(3e-7)
                if fault in ('diff_fraction', 'diff_fraction_boundary') and value != 0:
                    num_diff = 17 if fault == 'diff_fraction' else 16
                    feedback = sp.hidden_states[:4].flatten()[:num_diff]
                    feedback.copy_(torch.nextafter(feedback, torch.full_like(feedback, float('inf'))))
                if fault == 'confidence_error': sp._kstop.confidence.add_(2e-5)
                if fault == 'logits_drift': sp.draft_logits[:, 0, 0].add_(1 / 16)
                if fault == 'logits_error': sp.draft_logits[:, 0, 0].add_(1 / 8)
                if fault == 'signed_zero_nondeterministic': sp.hidden_states[:4, 7].fill_(-0.0 if count[0] % 2 else 0.0)
                if fault == 'nondeterministic':
                    sp.hidden_states[:4, 7].add_(count[0] % 2)
                if fault == 'stateful' and value == 0 and count[0] > 2:
                    sp.hidden_states[:4, 7].add_(1)
                if not zero:
                    zero.extend([sp.hidden_states.clone(), sp._kstop.confidence.clone(), sp.draft_tokens.clone()])
                if fault == 'stale_row': sp.hidden_states[2].copy_(zero[0][2])
                if fault in ('stale_hidden', 'stale_all'):
                    sp.hidden_states.copy_(zero[0])
                    if fault == 'stale_all':
                        sp._kstop.confidence.copy_(zero[1]); sp.draft_tokens.copy_(zero[2])
            manager.graphs = {descriptor: NS(replay=replay)}
    sp.capture = capture; capture()
    return r


class ReplayCriterionTests(unittest.TestCase):
    @staticmethod
    def failed_terms(r, check):
        return [t for vote in r._draft_head_init_failure['votes'] for t in vote['terms']
                if t['check'] == check and not all(t['conditions'].values())]

    @staticmethod
    def comparisons(r, manager=None):
        return [t for t in r._draft_head_init_terms if t['check'] == 'replay_compare'
                and (manager is None or t['measured']['manager'] == manager)]

    def test_two_state_flip_passes_all_descriptors_with_pair_receipts_and_restore(self):
        for desc in (PrefillDesc(), replace(PrefillDesc(), short_context=True),
                     replace(PrefillDesc(), num_tokens=12, uniform_token_count=3)):
            with self.subTest(desc=desc):
                r = descriptor_fixture('two_state_flip', desc)
                before = qual.tensor_state(r.speculator); boot(r)
                self.assertTrue(r._draft_head_ready and r._draft_head.on)
                terms = self.comparisons(r)
                self.assertTrue(all(all(t['conditions'].values()) for t in terms))
                self.assertFalse(any(t['check'] == 'warm_compare' for t in r._draft_head_init_terms))
                for manager in (0, 1):
                    term = next(t for t in terms if t['measured']['manager'] == manager
                                and t['measured']['value'] == .125
                                and t['measured']['sentinel_pattern'] in (0, None))
                    pair = term['measured']['pairs']['eager_vs_graph1']
                    hidden = pair['outputs'][1]
                    self.assertEqual((hidden['mismatches'], hidden['max_bf16_steps']), (4, 8))
                    self.assertEqual(hidden['first_differing_flat_indices'], [7, 263, 519, 775])
                    self.assertGreater(pair['outputs'][2]['max_abs'], 0)
                    self.assertLess(pair['outputs'][2]['max_abs'], 1e-5)
                    self.assertTrue(pair['tokens_exact'] and pair['drift_bounded'])
                    self.assertFalse(term['measured']['bit_exact'])
                    self.assertNotIn('bit_exact', term['conditions'])
                    if term['measured'].get('criterion'):
                        self.assertFalse(term['measured']['deterministic'])
                        self.assertFalse(term['measured']['feedback_eager_exact'])
                        self.assertFalse(term['measured']['eager_bit_exact'])
                        for key in ('deterministic', 'feedback_eager_exact', 'eager_bit_exact'):
                            self.assertNotIn(key, term['conditions'])
                        self.assertTrue(term['measured']['pairs']['eager_vs_graph2']['outputs'][1]['bit_exact'])
                        self.assertEqual(term['measured']['pairs']['graph1_vs_graph2']['outputs'][1]['max_bf16_steps'], 8)
                    else:
                        self.assertTrue(term['conditions']['tokens_exact'] and term['conditions']['drift_bounded'])
                rows = r._draft_head_qualification['cases']
                self.assertEqual([row['executions'] for row in rows],
                                 [30, 6] if desc == PrefillDesc() else [6, 6])
                for manager, row in enumerate(rows):
                    pairs = [p for t in self.comparisons(r, manager) for p in t['measured']['pairs'].values()]
                    self.assertNotIn('warm_executions', row)
                    self.assertEqual(row['float_exact_pairs'], sum(all(o['bit_exact'] for o in p['outputs'][1:]) for p in pairs))
                    self.assertEqual(row['float_bounded_pairs'], len(pairs))
                    self.assertLess(row['float_exact_pairs'], row['float_bounded_pairs'])
                for t, saved in before: self.assertTrue(torch.equal(t, saved))
                self.assertTrue(r._draft_head_qualification['target_fixed_hidden']['bit_exact'])

    def test_tp_summary_totals_allow_rank_local_drift_differences(self):
        r = descriptor_fixture('two_state_flip')
        # Other ranks can have a different count of exact pairs while every
        # pair remains bounded. Each descriptor returns the same TP totals.
        calls = []
        def gather(counts):
            calls.append(dict(counts))
            return [counts, dict(float_exact_pairs=0, float_bounded_pairs=counts['float_bounded_pairs'])]
        with patch.object(torch.distributed, 'is_initialized', return_value=True), \
             patch.object(torch.distributed, 'get_rank', return_value=0), \
             patch.object(dh, 'gather_receipts', side_effect=gather):
            boot(r)
        self.assertTrue(r._draft_head_ready)
        receipts = [t['measured'] for t in r._draft_head_init_terms if t['check'] == 'replay_descriptor']
        self.assertEqual(len(calls), 2)
        for row, local, receipt in zip(r._draft_head_qualification['cases'], calls, receipts):
            self.assertEqual(row['float_exact_pairs'], local['float_exact_pairs'])
            self.assertEqual(row['float_bounded_pairs'], 2 * local['float_bounded_pairs'])
            self.assertEqual(receipt['float_exact_pairs'], local['float_exact_pairs'])
            self.assertEqual(receipt['float_bounded_pairs'], local['float_bounded_pairs'])

    def test_every_prefill_descriptor_checks_row_element_and_one_shot_writes(self):
        for desc in (PrefillDesc(), replace(PrefillDesc(), short_context=True),
                     replace(PrefillDesc(), num_tokens=12, uniform_token_count=3)):
            for fault, elements in (('row_not_written', 256), ('element_not_written', 1),
                                    ('sparse_partial_write', 1), ('one_omit', 1)):
                with self.subTest(desc=desc, fault=fault):
                    r = descriptor_fixture(fault, desc); before = qual.tensor_state(r.speculator)
                    boot(r)
                    self.assertFalse(r._draft_head_ready or r._draft_head.on)
                    term = self.failed_terms(r, 'replay_execution')[0]
                    self.assertFalse(term['conditions']['hidden_feedback_written'])
                    self.assertTrue(term['conditions']['draft_tokens_written'])
                    self.assertEqual(term['measured']['hidden_unwritten_rows'], [2])
                    self.assertEqual(term['measured']['hidden_unwritten_elements'], elements)
                    self.assertEqual(term['measured']['phase'], 'steady')
                    if fault == 'one_omit':
                        self.assertEqual((term['measured']['value'], term['measured']['replay']), (.125, True))
                    for t, saved in before: self.assertTrue(torch.equal(t, saved))

    def test_decode_token_omission_refuses_without_hidden_write_coverage(self):
        r = descriptor_fixture('decode_tokens_unwritten'); before = qual.tensor_state(r.speculator)
        boot(r)
        self.assertFalse(r._draft_head_ready or r._draft_head.on)
        term = self.failed_terms(r, 'replay_execution')[0]
        self.assertFalse(term['conditions']['draft_tokens_written'])
        self.assertTrue(term['conditions']['hidden_feedback_written'])
        self.assertEqual((term['measured']['manager'], term['measured']['phase'],
                          term['measured']['replay']), (1, 'steady', True))
        self.assertEqual(term['measured']['hidden_unwritten_elements'], 0)
        for t, saved in before: self.assertTrue(torch.equal(t, saved))

    def test_vote_order_is_input_independent(self):
        from test_draft_head_init import completed_vote
        sequences = []
        for fault in ('none', 'two_state_flip', 'float_drift'):
            votes = []
            def vote(payload, valid):
                votes.append(payload); completed_vote(payload, valid)
            r = descriptor_fixture(fault); boot(r, vote)
            self.assertTrue(r._draft_head_ready)
            # The final report intentionally exposes data-dependent float
            # exactness counts; execution/input/compare vote payloads must not.
            votes = deepcopy(votes)
            final = next(p['initial_qualified'] for p in votes if 'initial_qualified' in p)
            for row in final['cases']:
                row.pop('float_exact_pairs')
                row.pop('float_bounded_pairs')
            sequences.append(votes)
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(sequences[0], sequences[2])
        self.assertFalse(any('warm_compare' in p for p in sequences[0]))

    def test_stale_nondeterministic_stateful_nonfinite_proposal_errors_refuse(self):
        for fault in ('noop', 'stale_hidden', 'stale_all', 'stale_row', 'nondeterministic',
                      'stateful', 'nonfinite', 'tokens', 'feedback_error'):
            with self.subTest(fault=fault):
                r = descriptor_fixture(fault); boot(r)
                self.assertFalse(r._draft_head.on or r._draft_head_ready)
                self.assertFalse(dh.status(r)['dh_gate_passed'])

    def test_exact_tokens_on_every_proposal_pair(self):
        for fault in ('tokens', 'eager_tokens', 'graph2_tokens'):
            with self.subTest(fault=fault):
                r = descriptor_fixture(fault); boot(r)
                self.assertFalse(r._draft_head_ready)
                term = self.failed_terms(r, 'replay_compare')[0]
                self.assertFalse(term['conditions']['proposals_exact'])
                pairs = term['measured']['pairs']
                expected = {'tokens': ('eager_vs_graph1', 'eager_vs_graph2'),
                            'eager_tokens': ('eager_vs_graph1', 'eager_vs_graph2'),
                            'graph2_tokens': ('graph1_vs_graph2', 'eager_vs_graph2')}[fault]
                self.assertTrue(all(not pairs[p]['tokens_exact'] for p in expected))

    def test_generic_descriptors_gate_tokens_and_bounds(self):
        for desc in (replace(PrefillDesc(), short_context=True),
                     replace(PrefillDesc(), num_tokens=12, uniform_token_count=3)):
            for fault, accepted in (('graph1_drift', True), ('float_drift', True),
                                    ('tokens', False), ('eager_tokens', False),
                                    ('float_seventeen_ulp', False), ('diff_fraction', False),
                                    ('confidence_error', False), ('hidden_nonfinite', False)):
                with self.subTest(desc=desc, fault=fault):
                    r = descriptor_fixture(fault, desc); boot(r)
                    self.assertEqual(r._draft_head_ready, accepted)
                    if not accepted:
                        self.assertTrue(self.failed_terms(r, 'replay_compare'))
        # K4 capture remains outside the 4x4 proposal criterion and refuses
        # excessive drift through the generic bounds.
        r = descriptor_fixture('float_seventeen_ulp'); r.speculator._kstop.k4_capture = True
        boot(r); self.assertFalse(r._draft_head_ready)
        term = self.failed_terms(r, 'replay_compare')[0]
        self.assertNotIn('criterion', term['measured'])
        self.assertFalse(term['conditions']['drift_bounded'])

    def test_logits_finite_bounds_and_inactive_columns_are_enforced(self):
        for fault, accepted in (('float_drift', True), ('two_state_flip', True),
                                ('logits_drift', True), ('logits_error', False),
                                ('logits_nonfinite', False), ('logits_nondeterministic', False),
                                ('inactive_logits', False)):
            with self.subTest(fault=fault):
                r = descriptor_fixture(fault)
                r.speculator.draft_logits = torch.full((4, 3, 64), float('nan'))
                boot(r)
                self.assertEqual(r._draft_head_ready, accepted)
                self.assertTrue(torch.isnan(r.speculator.draft_logits).all())

    def test_runtime_step_fraction_confidence_bounds_and_hidden_nonfinite(self):
        for fault, accepted in (('float_drift', True), ('float_two_ulp', True), ('float_three_ulp', True),
                                ('float_sixteen_ulp', True), ('float_seventeen_ulp', False),
                                ('diff_fraction_boundary', True), ('diff_fraction', False),
                                ('confidence_error', False), ('hidden_nonfinite', False)):
            with self.subTest(fault=fault):
                r = descriptor_fixture(fault); boot(r)
                self.assertEqual(dh.status(r)['dh_gate_passed'], accepted)

    def test_bf16_bound_counts_steps_at_sign_zero_and_exponent_boundaries(self):
        eager = torch.tensor([0, 0x3f7f, -16513, -32768], dtype=torch.int16).view(torch.bfloat16)
        graph = torch.tensor([2, 0x3f81, -16511, 0], dtype=torch.int16).view(torch.bfloat16)
        self.assertEqual(qual.bf16_feedback_distance(eager, graph)['max_bf16_ulps'], 2)
        self.assertFalse(qual.replay_equal(eager[-1:], graph[-1:], 1))
        graph[0] = torch.tensor([17], dtype=torch.int16).view(torch.bfloat16)[0]
        self.assertEqual(qual.bf16_feedback_distance(eager, graph)['elements_over_bound'], 1)
        with self.assertRaisesRegex(RuntimeError, 'BF16'):
            qual.bf16_feedback_distance(eager.float(), graph.float())

    def test_distinct_finite_sentinel_patterns_and_excessive_feedback_error(self):
        output = torch.zeros(4, 256, dtype=torch.bfloat16)
        a, b = [qual.feedback_sentinel(output, p, .125) for p in (0, 1)]
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        self.assertTrue((a != b).all())
        self.assertTrue((a[0] != a[1]).any())
        r = descriptor_fixture('feedback_error'); boot(r)
        term = self.failed_terms(r, 'replay_compare')[0]
        self.assertTrue(term['conditions']['proposals_exact'])
        self.assertFalse(term['conditions']['feedback_eager_bounded'])
        hidden = term['measured']['pairs']['eager_vs_graph1']['outputs'][1]
        # Adding .125 to .25 is 64 ordered BF16 steps, already excessive.
        self.assertEqual((hidden['mismatches'], hidden['max_bf16_steps']), (4, 64))

    def test_graph1_only_drift_passes_with_pair_receipts(self):
        r = descriptor_fixture('graph1_drift'); boot(r)
        self.assertTrue(r._draft_head_ready)
        term = next(t for t in self.comparisons(r, 0) if t['measured']['value'] == .125)
        measured = term['measured']
        self.assertFalse(measured['deterministic'])
        self.assertTrue(term['conditions']['repeat_bounded'])
        self.assertTrue(term['conditions']['feedback_eager_bounded'])
        hidden = measured['pairs']['graph1_vs_graph2']['outputs'][1]
        self.assertEqual((hidden['mismatches'], hidden['max_bf16_steps']), (4, 8))
        self.assertEqual(hidden['first_differing_flat_indices'], [7, 263, 519, 775])

    def test_graph2_excess_remains_refused(self):
        r = descriptor_fixture('graph2_excess'); boot(r)
        self.assertFalse(r._draft_head.on or r._draft_head_ready)
        term = self.failed_terms(r, 'replay_compare')[0]
        self.assertFalse(term['conditions']['feedback_eager_bounded'])
        self.assertEqual(term['measured']['pairs']['eager_vs_graph2']['outputs'][1]['max_bf16_steps'], 17)

    def test_signed_zero_repeat_difference_is_diagnostic(self):
        r = descriptor_fixture('signed_zero_nondeterministic'); boot(r)
        self.assertTrue(r._draft_head_ready)
        terms = self.comparisons(r, 0)
        self.assertTrue(all(t['measured']['deterministic'] for t in terms))
        self.assertTrue(all(t['measured']['feedback_eager_exact'] for t in terms))
        self.assertTrue(all(t['measured']['graph_repeat_mismatches'][1] == 4 for t in terms))
        self.assertTrue(all(t['conditions']['repeat_bounded'] for t in terms))
        self.assertTrue(all(not t['measured']['pairs']['graph1_vs_graph2']['outputs'][1]['bit_exact'] for t in terms))

    def test_zero_roundtrip_requires_exact_tokens_and_bounded_floats(self):
        for fault, accepted in (('roundtrip_drift', True), ('roundtrip_error', False), ('roundtrip_tokens', False)):
            with self.subTest(fault=fault):
                r = descriptor_fixture(fault); boot(r)
                self.assertEqual(r._draft_head_ready, accepted)
                terms = self.comparisons(r, 0) if accepted else self.failed_terms(r, 'replay_compare')
                term = next(t for t in terms if t['measured']['value'] == 0.
                            and t['measured']['probe'] == 4)
                self.assertEqual(term['conditions']['zero_input_roundtrip'], accepted)
                self.assertTrue(term['measured']['feedback_eager_exact'])
                self.assertTrue(term['measured']['deterministic'])
                pair = term['measured']['pairs']['zero_roundtrip']
                if fault == 'roundtrip_tokens':
                    self.assertFalse(pair['tokens_exact'])
                else:
                    self.assertEqual(pair['outputs'][1]['max_bf16_steps'], 8 if accepted else 17)

    def test_output_scratch_is_charged_before_clone(self):
        for logits in (False, True):
            with self.subTest(logits=logits):
                r = descriptor_fixture(); sp = r.speculator
                if logits: sp.draft_logits = torch.zeros(4, 3, 64)
                snapshots = qual.tensor_state(sp)
                snapshot_bytes = sum(t.numel()*t.element_size() for t, _ in snapshots)
                output_bytes = sum(t.numel()*t.element_size() for t in
                    (sp.hidden_states[:4], sp.draft_tokens[:4], sp._kstop.confidence[:4]))
                if logits: output_bytes += sp.draft_logits[:4].numel()*sp.draft_logits.element_size()
                bound = snapshot_bytes + 10 * output_bytes + 16 * sp.hidden_states[:4].numel() * sp.hidden_states.element_size()
                with patch.object(torch.Tensor, 'clone', side_effect=AssertionError('cloned before budget')):
                    with self.assertRaisesRegex(RuntimeError, 'scratch'):
                        qual.tensor_state(sp, scratch_limit=bound - 1)
                self.assertTrue(qual.tensor_state(sp, scratch_limit=bound))

if __name__ == '__main__': unittest.main()
