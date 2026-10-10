# SPDX-License-Identifier: Apache-2.0
"""Shared native draft replay qualification for INIT and combined toggles.

Every descriptor checks output-write coverage and exact draft tokens, with
bounded finite float comparisons and pair receipts. The GPU-observed two-state
flip at +0.125, pattern 0 (cand3win6) can recur between executions. Hidden,
confidence and logits limits are policy bounds, not a kernel error proof.
The unchanged target model, whose LM head remains BF16, verifies drafts through
standard speculative verification; this does not establish bitwise-identical
output across runs.
"""
import torch
import glm_draft_head as dh
import kstop_runtime as rt

SCRATCH_LIMIT = 256 << 20
LIVE_FLOOR = 4.5 * (1 << 30)
HIDDEN_MAX_BF16_STEPS = 16
HIDDEN_MAX_DIFF_FRACTION = 1 / 64
CONFIDENCE_MAX_ABS = 1e-5
LOGITS_MAX_ABS = 1 / 16


def feedback_sentinel(output, pattern, value):
    """Finite, BF16-exact row/element patterns; opposite signs across patterns.

    Native prefill aliases input/output storage. These are deliberately also
    model inputs, identical for eager and repeated graph executions. Never
    compare different patterns as if they were the same input.
    """
    columns = torch.arange(output.shape[1], device=output.device)
    rows = torch.arange(output.shape[0], device=output.device)[:, None]
    return ((16 + (columns + 17 * rows) % 128 / 8 + value)
            * (1 if pattern == 0 else -1)).to(output.dtype)


def replay_equal(a, b, output_index):
    # Diagnostic exactness includes the sign of feedback zero.
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) if output_index == 1 else torch.equal(a, b)


def bf16_feedback_distance(eager, graph):
    """Count adjacent BF16 values, including exponent boundaries and signs.

    Collapse +/- zero to the same ordered value. Finiteness is gated separately.
    Reject other dtypes rather than silently rounding their errors to BF16.
    """
    if eager.dtype != torch.bfloat16 or graph.dtype != torch.bfloat16:
        raise RuntimeError('feedback bound requires native BF16 outputs')
    def ordered(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)
    distance = (ordered(eager) - ordered(graph)).abs()
    different = distance != 0
    return dict(max_bf16_ulps=int(distance.max()),
                elements_over_bound=int((distance > HIDDEN_MAX_BF16_STEPS).sum()),
                bound_bf16_ulps=HIDDEN_MAX_BF16_STEPS,
                mismatches=int(different.sum()),
                first_differing_flat_indices=different.flatten().nonzero().flatten()[:16].tolist(),
                max_diff_fraction=HIDDEN_MAX_DIFF_FRACTION)


def bounded_pair(a, b):
    """Exact tokens and bounded finite floats, with per-output diagnostics."""
    labels = ('draft_tokens', 'hidden_states', 'confidence', 'draft_logits')
    tokens_exact = bool(torch.equal(a[0], b[0]))
    drift_bounded = True
    outputs = []
    for i, (left, right) in enumerate(zip(a, b)):
        output = dict(name=labels[i], shape=list(left.shape),
            bit_exact=bool(replay_equal(left, right, i)), mismatches=int((left != right).sum()),
            first_differing_flat_indices=(left != right).flatten().nonzero().flatten()[:16].tolist(),
            eager_nonfinite=int((~torch.isfinite(left)).sum()),
            graph_nonfinite=int((~torch.isfinite(right)).sum()),
            max_abs=float((left.float() - right.float()).abs().nan_to_num().max()))
        if i == 1:
            distance = bf16_feedback_distance(left, right)
            output.update(distance, max_bf16_steps=distance['max_bf16_ulps'])
            bounded = (distance['elements_over_bound'] == 0 and
                       distance['mismatches'] <= left.numel() * HIDDEN_MAX_DIFF_FRACTION)
        elif i > 1:
            bound = CONFIDENCE_MAX_ABS if i == 2 else LOGITS_MAX_ABS
            output['bound_max_abs'] = bound
            bounded = output['max_abs'] <= bound
        if i > 0:
            bounded = bounded and output['eager_nonfinite'] == output['graph_nonfinite'] == 0
            output['bounded'] = bounded
            drift_bounded = drift_bounded and bounded
        outputs.append(output)
    return dict(tokens_exact=tokens_exact, drift_bounded=drift_bounded, outputs=outputs)


def graph_key(runner):
    managers = [runner.cudagraph_manager, *dh.managers(runner)]
    return [[repr(k) for k in m.graphs] for m in managers]


def tensor_state(sp, scratch_limit=SCRATCH_LIMIT):
    """Persistent work tensors only; never duplicate model/KV allocations."""
    objects = (sp, sp._kstop, sp.input_buffers, sp.target_input_buffers, sp.block_tables)
    unique = {}
    for obj in objects:
        for value in vars(obj).values():
            values = value if isinstance(value, (tuple, list)) else (value,)
            for t in values:
                if isinstance(t, torch.Tensor) and t.numel():
                    unique.setdefault((t.data_ptr(), t.shape, t.stride()), t)
    if rt.STATE is not None and rt.STATE['runtime'] is sp._kstop:
        for t in rt.STATE.values():
            if isinstance(t, torch.Tensor) and t.numel():
                unique.setdefault((t.data_ptr(), t.shape, t.stride()), t)
    # Charge every clone (including views of a shared storage) before copying.
    # Two retained zero-pattern references plus three current executions,
    # a sentinel, plus diagnostics' temporary float/bool output tensors.
    output_bound = 10 * sum(t.numel()*t.element_size() for t in
        (sp.hidden_states[:sp.max_num_reqs], sp.draft_tokens[:sp.max_num_reqs],
         sp._kstop.confidence[:sp.max_num_reqs]))
    if sp.draft_logits is not None:
        output_bound += 10*sp.draft_logits[:sp.max_num_reqs].numel()*sp.draft_logits.element_size()
    output_bound += 16 * sp.hidden_states[:sp.max_num_reqs].numel() * sp.hidden_states.element_size()
    if sum(t.numel()*t.element_size() for t in unique.values()) + output_bound > scratch_limit:
        raise RuntimeError('combined replay scratch exceeds 256 MiB')
    return [(t, t.clone()) for t in unique.values()]


def proposal_replay_criterion(sp, index, desc):
    """Only the observed regular 4x4 draft prefill bank gets this criterion.

    This descriptor adds bounded repeated graphs, eager-vs-graph2 and zero-input
    roundtrips. Every descriptor checks output-write coverage, exact tokens and
    bounded finite floats. Target verification remains exact.
    """
    return (index == 0 and sp.num_speculative_steps == 3
            and not getattr(sp._kstop, 'k4_capture', False)
            and desc.num_tokens == 16 and desc.num_reqs == 4
            and getattr(desc, 'uniform_token_count', None) == 4
            and getattr(desc, 'short_context', None) is False
            and getattr(desc, 'num_active_loras', 0) == 0)


@torch.inference_mode()
def qualify(runner, sync=torch.cuda.synchronize, vote=dh.agree, mem=dh.memory,
            scratch_limit=SCRATCH_LIMIT):
    """Recheck every native draft FULL graph at three changed hidden inputs.

    Capture factories use dummy slot -1 and null block 0, preserving APC/KV.
    Scratch and native work buffers are restored before every execution.
    The observed regular 4x4 prefill descriptor additionally runs A/B/A/C/A
    with two finite sentinel input patterns and two graph replays per pattern.
    Every descriptor gates exact tokens and bounded finite floats on every pair;
    byte exactness remains diagnostic. Pair receipts expose the GPU-observed
    two-state flip at +0.125, pattern 0 (cand3win6). These are policy bounds,
    not a kernel error proof. Changed inputs and sign-flipped patterns detect
    stale feedback. Every prefill feedback element and every active draft token
    must replace its sentinel; decode hidden storage is not an output.
    Synchronized probes do not prove production stream ordering.
    Target weights/graphs are never recaptured. Full request lifecycle checks
    and populated-KV accuracy remain separate fleet acceptance requirements.
    """
    if getattr(runner.speculator, "_glm_rowselect", None) is not None:
        from glm_rowselect_qual import qualify as qualify_rowselect
        return qualify_rowselect(runner, sync=sync, vote=vote, mem=mem,
                                 scratch_limit=scratch_limit)
    from vllm.config.compilation import CUDAGraphMode
    sp = runner.speculator
    snapshots = []
    error = None
    try:
        snapshots = tensor_state(sp, scratch_limit)
        for manager in dh.managers(runner):
            if getattr(manager, '_k4_drafthead_factory', None) is None or not manager.graphs:
                raise RuntimeError('missing native captured draft factory')
            if any(d.cg_mode != CUDAGraphMode.FULL for d in manager.graphs):
                raise RuntimeError('combined replay requires FULL draft graphs')
    except Exception as exc:
        error = exc
    available = mem()
    valid = dh.qualification_term('replay_prepare', dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR), dict(graphs=graph_key(runner), mem_available=available, floor_bytes=int(LIVE_FLOOR), scratch_bytes=sum(t.numel()*t.element_size() for t, _ in snapshots)), error)
    try:
        vote({'replay_prepare': graph_key(runner)}, valid)
    except BaseException:
        snapshots.clear()
        raise
    if error is not None:
        raise RuntimeError('combined replay preparation failed') from error
    scratch_bytes = sum(t.numel()*t.element_size() for t, _ in snapshots)
    rows = []
    try:
        for index, manager in enumerate(dh.managers(runner)):
            factory = getattr(manager, '_k4_drafthead_factory', None)
            if factory is None or not manager.graphs:
                raise RuntimeError('missing native captured draft factory')
            for desc, graph in manager.graphs.items():
                if desc.cg_mode != CUDAGraphMode.FULL:
                    raise RuntimeError('combined replay requires FULL draft graphs')
                n = desc.num_reqs or min(desc.num_tokens, sp.max_num_reqs)
                proposal_criterion = proposal_replay_criterion(sp, index, desc)
                zero_by_pattern = {}
                float_exact_pairs = float_bounded_pairs = 0
                zero_outputs = None
                probes = (0., .125, 0., -.125, 0.) if proposal_criterion else (0., .125, -.125)
                trials = [(probe, value, pattern) for probe, value in enumerate(probes)
                          for pattern in ((0, 1) if proposal_criterion else (None,))]
                def execute(phase, probe, value, pattern, execution, replay):
                    begin = sp.on_prefill_begin if index == 0 else sp.on_multi_step_decode_begin
                    end = sp.on_prefill_end if index == 0 else sp.on_multi_step_decode_end
                    stage = dict(manager=index, descriptor=repr(desc), value=value,
                                 input_index=(0., .125, -.125).index(value),
                                 probe=probe, sentinel_pattern=pattern,
                                 execution=execution, replay=replay, phase=phase)
                    error = None
                    try:
                        for tensor, saved in snapshots:
                            tensor.copy_(saved)
                        begin(n)
                        forward = factory(desc, warmup=False)
                        # Fixed, finite, rank-identical inputs; no RNG drift.
                        sp.hidden_states.fill_(value)
                        sp.temperature.zero_()
                        sp.idx_mapping.copy_(torch.arange(sp.idx_mapping.numel(),
                            device=sp.idx_mapping.device, dtype=sp.idx_mapping.dtype).reshape_as(sp.idx_mapping))
                        sp.last_token_indices.zero_()
                        step = 0 if index == 0 else sp.num_speculative_steps - 1
                        sp.current_draft_step.fill_(step)
                        sp.draft_tokens[:n, step].fill_(-1)
                        sp._kstop.confidence[:n].fill_(float('nan'))
                        if sp.draft_logits is not None:
                            sp.draft_logits[:n, step].fill_(float('nan'))
                        if index == 0:
                            # Hidden input/output storage aliases: the same
                            # finite pattern is supplied to eager and graphs.
                            # Tokens/confidence/logits are output-only.
                            hidden_sentinel = feedback_sentinel(sp.hidden_states[:n],
                                pattern if pattern is not None else 0, value)
                            sp.hidden_states[:n].copy_(hidden_sentinel)
                        sync()
                    except Exception as exc:
                        error = exc
                    available = mem()
                    valid = dh.qualification_term('replay_inputs', dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR), dict(**stage, mem_available=available, current_draft_step=0 if index == 0 else sp.num_speculative_steps - 1), error)
                    vote({'replay_inputs': stage}, valid)
                    if error is not None:
                        raise RuntimeError('combined replay inputs failed') from error
                    inactive_columns = []
                    inactive_unchanged = True
                    hidden_written = True
                    tokens_written = False
                    hidden_unwritten_rows = []
                    hidden_unwritten_elements = 0
                    try:
                        if replay:
                            graph.replay()
                        else:
                            forward(CUDAGraphMode.NONE)
                        sync()
                        step = 0 if index == 0 else sp.num_speculative_steps - 1
                        outputs = [sp.draft_tokens[:n].clone(), sp.hidden_states[:n].clone(),
                                   sp._kstop.confidence[:n].clone()]
                        tokens_written = bool((outputs[0][:, step] != -1).all())
                        if sp.draft_logits is not None:
                            outputs.append(sp.draft_logits[:n, step].clone())
                            cache_saved = next(saved for tensor, saved in snapshots
                                if tensor is sp.draft_logits)
                            for column in range(sp.draft_logits.shape[1]):
                                if column == step:
                                    continue
                                # Unwritten NaN/-inf sentinels are not fresh logits,
                                # but neither pass may corrupt those columns.
                                unchanged = torch.isclose(sp.draft_logits[:n,column],
                                    cache_saved[:n,column], rtol=0, atol=0, equal_nan=True)
                                mismatches = int((~unchanged).sum())
                                inactive_columns.append(dict(column=column, mismatches=mismatches))
                                inactive_unchanged = inactive_unchanged and mismatches == 0
                        if index == 0:
                            unwritten = outputs[1] == hidden_sentinel
                            hidden_unwritten_elements = int(unwritten.sum())
                            hidden_unwritten_rows = unwritten.any(dim=1).nonzero().flatten().tolist()
                            hidden_written = hidden_unwritten_elements == 0
                        end(n)
                    except Exception as exc:
                        error = exc
                    available = mem()
                    valid = dh.qualification_term('replay_execution', dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR, inactive_columns_unchanged=inactive_unchanged, hidden_feedback_written=hidden_written, draft_tokens_written=tokens_written), dict(**stage, mem_available=available, inactive_columns=inactive_columns, hidden_unwritten_elements=hidden_unwritten_elements, hidden_unwritten_rows=hidden_unwritten_rows), error)
                    vote({'replay_execution': stage}, valid)
                    if error is not None:
                        raise RuntimeError('combined replay execution failed') from error
                    return outputs, stage

                for probe, value, pattern in trials:
                    zero_outputs = zero_by_pattern.get(pattern)
                    results = []
                    for execution, replay in enumerate((False, True, True) if proposal_criterion else (False, True)):
                        outputs, stage = execute('steady', probe, value, pattern, execution, replay)
                        results.append(outputs)
                    outputs = None
                    same = finite = False
                    error = None
                    measured = dict(stage)
                    try:
                        same = all(torch.equal(a, b) for a, b in zip(results[0], results[1]))
                        finite = all(torch.isfinite(t).all() for result in results for t in result[1:])
                        eager_graph1 = bounded_pair(results[0], results[1])
                        measured.update(bit_exact=same, outputs=eager_graph1['outputs'],
                            pairs={'eager_vs_graph1': eager_graph1})
                        if proposal_criterion:
                            # Exact tokens and bounded floats gate each pair.
                            graph_repeat = bounded_pair(results[1], results[2])
                            eager_graph2 = bounded_pair(results[0], results[2])
                            measured['pairs'].update(graph1_vs_graph2=graph_repeat,
                                eager_vs_graph2=eager_graph2)
                            deterministic = all(torch.equal(a, b) for a, b in zip(results[1], results[2]))
                            feedback_eager_exact = same and all(torch.equal(a, b)
                                for a, b in zip(results[0], results[2]))
                            repeat_bounded = graph_repeat['drift_bounded']
                            feedback_eager_bounded = eager_graph1['drift_bounded'] and eager_graph2['drift_bounded']
                            if zero_outputs is None:
                                zero_outputs = results[1]
                                zero_by_pattern[pattern] = zero_outputs
                            roundtrip = True
                            if value == 0.:
                                zero_roundtrip = bounded_pair(zero_outputs, results[1])
                                measured['pairs']['zero_roundtrip'] = zero_roundtrip
                                roundtrip = zero_roundtrip['tokens_exact'] and zero_roundtrip['drift_bounded']
                            sensitive = value == 0. or bool((zero_outputs[1] != results[1][1]).any(dim=1).all())
                            proposals_exact = all(pair['tokens_exact'] for pair in measured['pairs'].values())
                            measured.update(criterion='draft_4x4_prefill_proposal_replay',
                                eager_bit_exact=all(o['bit_exact'] for o in measured['outputs']),
                                graph_repeat_mismatches=[int(((a.view(torch.int16) != b.view(torch.int16)) if i == 1 else (a != b)).sum())
                                    for i, (a, b) in enumerate(zip(results[1], results[2]))],
                                proposals_exact=proposals_exact, deterministic=deterministic,
                                feedback_eager_exact=feedback_eager_exact,
                                repeat_bounded=repeat_bounded,
                                feedback_eager_bounded=feedback_eager_bounded,
                                feedback_bound=bf16_feedback_distance(results[0][1], results[1][1]),
                                changed_input_sensitive=sensitive, zero_input_roundtrip=roundtrip)
                        else:
                            measured.update(tokens_exact=eager_graph1['tokens_exact'],
                                drift_bounded=eager_graph1['drift_bounded'])
                        float_exact_pairs += sum(all(o['bit_exact'] for o in pair['outputs'][1:])
                            for pair in measured['pairs'].values())
                        float_bounded_pairs += sum(pair['drift_bounded']
                            for pair in measured['pairs'].values())
                    except Exception as exc:
                        error = exc
                    conditions = dict(no_exception=error is None, finite=finite)
                    if proposal_criterion:
                        conditions.update({key: measured.get(key, False) for key in
                            ('proposals_exact', 'repeat_bounded', 'feedback_eager_bounded', 'changed_input_sensitive', 'zero_input_roundtrip')})
                    else:
                        conditions.update({key: measured.get(key, False) for key in
                            ('tokens_exact', 'drift_bounded')})
                    valid = dh.qualification_term('replay_compare', conditions, measured, error)
                    vote({'replay_compare': stage}, valid)
                    if error is not None:
                        raise RuntimeError('native draft output comparison failed') from error
                    if not valid:
                        raise RuntimeError('native draft replay criterion failed or nonfinite')
                    del results
                zero_outputs = None
                zero_by_pattern.clear()
                row = dict(manager=index, descriptor=repr(desc), changed_inputs=3,
                    criterion='draft_4x4_prefill_proposal_replay' if proposal_criterion else 'tokens_exact_bounded_floats',
                    executions=len(trials) * (3 if proposal_criterion else 2))
                local_counts = dict(float_exact_pairs=float_exact_pairs,
                                    float_bounded_pairs=float_bounded_pairs)
                dh.qualification_term('replay_descriptor', dict(no_exception=True),
                    dict(**row, **local_counts))
                # The caller hashes the returned report in its final TP vote.
                # Preserve rank-local counts in receipts; report TP-wide totals
                # so accepted drift can differ by rank without a payload mismatch.
                counts = (dh.gather_receipts(local_counts)
                          if torch.distributed.is_initialized() else [local_counts])
                rows.append(dict(**row, **{key: sum(count[key] for count in counts)
                                          for key in local_counts}))
    finally:
        for tensor, saved in snapshots:
            tensor.copy_(saved)
        sync()
        snapshots.clear()
        tensor = saved = outputs = results = zero_outputs = None
        zero_by_pattern = None
    return dict(cases=rows, scratch_bytes=scratch_bytes,
                scope='native draft FULL replay; write coverage on every descriptor/execution; exact tokens and bounded finite floats on every pair: hidden <=16 BF16 steps and <=1/64 differing elements, confidence <=1e-5, logits <=1/16; audited 4x4 repeats/roundtrips; pair receipts expose GPU-observed two-state flip at +0.125 p0 (cand3win6); policy bounds, not a kernel error proof; all active draft tokens and prefill hidden elements must replace sentinels; decode hidden is not an output; dummy KV, APC retained; production stream ordering unverified')
