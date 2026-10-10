# SPDX-License-Identifier: Apache-2.0
"""Boot-only rowselect full-M/selected qualification; all-output byte-exact pairs and write receipts."""
import contextlib
import torch
import glm_draft_head as dh
import glm_draft_head_qual as dq

SCRATCH_LIMIT = 256 << 20
LIVE_FLOOR = 4.5 * (1 << 30)


def graph_key(runner):
    managers = [runner.cudagraph_manager, *dh.managers(runner)]
    return [[repr(k) for k in m.graphs] for m in managers]


def tensor_state(sp, scratch_limit=SCRATCH_LIMIT):
    # Share fix6's conservative output/sentinel/diagnostic workspace charge.
    return dq.tensor_state(sp, scratch_limit)


def bounded_pair(left, right, native_logits, rowselect, *, near_zero_relief=False):
    """Retain fix6 diagnostics, but require every output byte exact.

    The legacy near_zero_relief argument cannot enable relief on this branch.
    KV writes precede the selected MoE tail and stay byte exact too.
    """
    count = 4 if native_logits else 3
    pair = dq.bounded_pair(left[:count], right[:count])
    cursor = count
    if rowselect:
        logits = dq.bounded_pair([*left[:3], left[cursor]],
                                 [*right[:3], right[cursor]])['outputs'][-1]
        logits['name'] = 'rowselect_logits'
        pair['outputs'].append(logits)
        pair['drift_bounded'] = pair['drift_bounded'] and logits['bounded']
        cursor += 1
    exact = []
    kv_count = len(left) - cursor - 4
    names = [f'kv_{i}' for i in range(kv_count)] + [
        'last_token_indices', 'idx_mapping', 'slot_mappings', 'positions']
    for name, a, b in zip(names, left[cursor:], right[cursor:]):
        exact.append(dict(name=name,
                          shape=list(a.shape), bit_exact=bool(a.dtype == b.dtype and a.shape == b.shape and
                              torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))),
                          mismatches=int((a != b).sum()),
                          first_differing_flat_indices=(a != b).flatten().nonzero().flatten()[:16].tolist(),
                          nonfinite_left=int((~torch.isfinite(a)).sum()),
                          nonfinite_right=int((~torch.isfinite(b)).sum())))
    # This branch admits no draft float drift, including signed zero.
    all_exact = all(a.shape == b.shape and a.dtype == b.dtype and
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        for a, b in zip(left, right)) and len(left) == len(right)
    pair['bit_exact'] = all_exact
    pair['drift_bounded'] = pair['drift_bounded'] and all_exact
    for output, a, b in zip(pair['outputs'], left[:cursor], right[:cursor]):
        output['bit_exact'] = bool(a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    pair['exact_outputs'] = exact
    pair['kv_selection_exact'] = all(o['bit_exact'] and
        o['nonfinite_left'] == o['nonfinite_right'] == 0 for o in exact)
    return pair


def selection_state(sp, n, m):
    return (sp.last_token_indices[:n], sp.idx_mapping[:n],
            sp.block_tables.slot_mappings[:, :m], sp.input_buffers.positions[:m])


@torch.inference_mode()
def qualify(runner, sync=torch.cuda.synchronize, vote=dh.agree, mem=dh.memory,
            scratch_limit=SCRATCH_LIMIT):
    """Recheck every native draft FULL graph at three changed hidden inputs.

    Capture factories use dummy slot -1; rowselect tests write reserved null
    block 0 on all input rows. Saved null blocks restore before admission.
    Scratch and native work buffers are restored before every arm and on exit.
    Target weights/graphs are never recaptured. Full request lifecycle checks
    and populated-KV accuracy remain separate fleet acceptance requirements.
    """
    from vllm.config.compilation import CUDAGraphMode
    sp = runner.speculator
    snapshots = []
    rowselect = getattr(sp, "_glm_rowselect", None) is not None
    kv = []
    error = None
    try:
        extra = 0
        if rowselect:
            import glm_mtp_rowselect as rs
            if sp._kstop.capture_layout != 'reuse':
                raise RuntimeError('rowselect requires qualified reuse runtime capture layout')
            kv = rs.kv_scratch(sp)
            logits_bytes = sp._glm_rowselect_logits.numel()*sp._glm_rowselect_logits.element_size()
            # Saved KV, two retained zero baselines, four current arms and diagnostics.
            # Full-M input sentinels exceed the ordinary active feedback rows.
            extra = 10*sum(t.numel()*t.element_size() for t in kv) + 10*logits_bytes
            descriptors = sp.prefill_cudagraph_manager.graphs
            max_tokens = max(d.num_tokens for manager in dh.managers(runner) for d in manager.graphs)
            extra += 10*sum(t.numel()*t.element_size() for t in
                selection_state(sp, sp.max_num_reqs, max_tokens))
            # Full-M feedback_sentinel uses temporary int64/FP32 patterns;
            # reserve their peak independently of fix6's active-row budget.
            extra += 16*sp.hidden_states[:max_tokens].numel()*sp.hidden_states.element_size()
            if not {2, 3, 4, 6, 12, 16}.issubset({d.num_tokens for d in descriptors}):
                raise RuntimeError('rowselect missing required full-M capture descriptors')
            for m in (2, 3, 4, 12):
                if {bool(d.short_context) for d in descriptors if d.num_tokens == m} != {False, True}:
                    raise RuntimeError('rowselect missing short/full attention descriptors')
        snapshots = tensor_state(sp, scratch_limit - extra)
        snapshots += [(t, t.clone()) for t in kv]
        for manager in dh.managers(runner):
            if getattr(manager, '_k4_drafthead_factory', None) is None or not manager.graphs:
                raise RuntimeError('missing native captured draft factory')
            if any(d.cg_mode != CUDAGraphMode.FULL for d in manager.graphs):
                raise RuntimeError('combined replay requires FULL draft graphs')
    except Exception as exc:
        error = exc
    available = mem()
    valid = dh.qualification_term('replay_prepare',
        dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR),
        dict(graphs=graph_key(runner), mem_available=available, floor_bytes=int(LIVE_FLOOR),
             scratch_bytes=sum(t.numel()*t.element_size() for t, _ in snapshots)), error)
    try:
        vote({'replay_prepare': graph_key(runner)}, valid)
    except BaseException:
        snapshots.clear()
        raise
    if error is not None:
        raise RuntimeError('combined replay preparation failed') from error
    rows = []
    selected_cases = []
    zero_by_pattern = {}
    try:
        for index, manager in enumerate(dh.managers(runner)):
            factory = getattr(manager, '_k4_drafthead_factory', None)
            if factory is None or not manager.graphs:
                raise RuntimeError('missing native captured draft factory')
            for desc, graph in manager.graphs.items():
                if desc.cg_mode != CUDAGraphMode.FULL:
                    raise RuntimeError('combined replay requires FULL draft graphs')
                n = desc.num_reqs or min(desc.num_tokens, sp.max_num_reqs)
                width = desc.num_tokens // n
                positions = range(width) if rowselect and index == 0 else (0,)
                float_exact_pairs = float_bounded_pairs = 0
                for position in positions:
                    # Retain only graph1's first A for each sentinel pattern at
                    # this position. Never compare different positions/patterns.
                    zero_by_pattern = {}
                    for probe, value, pattern in ((i, v, pattern)
                            for i, v in enumerate((0., .125, 0., -.125, 0.))
                            for pattern in (0, 1)):
                        results = []
                        selected = rowselect and index == 0
                        arms = ('full', 'eager', 'graph1', 'graph2') if selected else ('eager', 'graph1', 'graph2')
                        for arm in arms:
                            replay = arm.startswith('graph')
                            begin = sp.on_prefill_begin if index == 0 else sp.on_multi_step_decode_begin
                            end = sp.on_prefill_end if index == 0 else sp.on_multi_step_decode_end
                            stage = dict(manager=index, descriptor=repr(desc), value=value, replay=replay, arm=arm, position=position, probe=probe, sentinel_pattern=pattern)
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
                                if rowselect and index == 0:
                                    sp.last_token_indices[:n].copy_(torch.arange(n, device=sp.last_token_indices.device)*width + position)
                                    sp.block_tables.slot_mappings[:, :desc.num_tokens].copy_(
                                        torch.arange(desc.num_tokens, device=sp.hidden_states.device))
                                    sp.input_buffers.positions[:desc.num_tokens].copy_(
                                        torch.arange(desc.num_tokens, device=sp.hidden_states.device))
                                step = 0 if index == 0 else sp.num_speculative_steps - 1
                                sp.current_draft_step.fill_(step)
                                sp.draft_tokens[:n, step].fill_(-1)
                                sp._kstop.confidence[:n].fill_(float('nan'))
                                if sp.draft_logits is not None:
                                    sp.draft_logits[:n, step].fill_(float('nan'))
                                if selected:
                                    sp._glm_rowselect_logits[:n].fill_(float('nan'))
                                if index == 0:
                                    # Native feedback aliases input/output. Every arm
                                    # receives the same finite row/element sentinels.
                                    hidden_sentinel = dq.feedback_sentinel(
                                        sp.hidden_states[:desc.num_tokens], pattern, value)
                                    sp.hidden_states[:desc.num_tokens].copy_(hidden_sentinel)
                                expected_selection = [t.clone() for t in selection_state(sp, n, desc.num_tokens)]
                                if index == 0:
                                    # Native _prefill compacts selected positions
                                    # into the first n rows after sampling.
                                    expected_selection[-1][:n].copy_(
                                        sp.input_buffers.positions[sp.last_token_indices[:n]])
                                sync()
                            except Exception as exc:
                                error = exc
                            available = mem()
                            valid = dh.qualification_term('replay_inputs',
                                dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR),
                                dict(**stage, mem_available=available), error)
                            vote({'replay_inputs': stage}, valid)
                            if error is not None:
                                raise RuntimeError('combined replay inputs failed') from error
                            tokens_written = hidden_written = selection_correct = False
                            inactive_unchanged = True
                            hidden_unwritten_elements = 0
                            hidden_unwritten_rows = []
                            inactive_columns = []
                            try:
                                if replay:
                                    graph.replay()
                                else:
                                    context = rs.full_reference(sp) if arm == 'full' else contextlib.nullcontext()
                                    with context:
                                        forward(CUDAGraphMode.NONE)
                                sync()
                                outputs = [sp.draft_tokens[:n].clone(), sp.hidden_states[:n].clone(),
                                           sp._kstop.confidence[:n].clone()]
                                if sp.draft_logits is not None:
                                    outputs.append(sp.draft_logits[:n, step].clone())
                                    cache_saved = next(saved for tensor, saved in snapshots if tensor is sp.draft_logits)
                                    for column in range(sp.draft_logits.shape[1]):
                                        if column == step:
                                            continue
                                        mismatches = int((~torch.isclose(sp.draft_logits[:n, column],
                                            cache_saved[:n, column], rtol=0, atol=0, equal_nan=True)).sum())
                                        inactive_columns.append(dict(column=column, mismatches=mismatches))
                                        inactive_unchanged = inactive_unchanged and mismatches == 0
                                if rowselect and index == 0:
                                    outputs.append(sp._glm_rowselect_logits[:n].clone())
                                    outputs.extend(t.clone() for t in kv)
                                tokens_written = bool((outputs[0][:, step] != -1).all())
                                hidden_written = True
                                if index == 0:
                                    unwritten = outputs[1] == hidden_sentinel[:n]
                                    hidden_unwritten_elements = int(unwritten.sum())
                                    hidden_unwritten_rows = unwritten.any(dim=1).nonzero().flatten().tolist()
                                    hidden_written = hidden_unwritten_elements == 0
                                metadata = [t.clone() for t in selection_state(sp, n, desc.num_tokens)]
                                selection_correct = all(torch.equal(a, b) for a, b in
                                    zip(expected_selection[:4 if index == 0 else 3], metadata))
                                outputs.extend(metadata)
                                results.append(outputs)
                                end(n)
                            except Exception as exc:
                                error = exc
                            available = mem()
                            valid = dh.qualification_term('replay_execution',
                                dict(no_exception=error is None, memory_floor=available >= LIVE_FLOOR,
                                     draft_tokens_written=tokens_written, hidden_feedback_written=hidden_written,
                                     selection_correct=selection_correct, inactive_columns_unchanged=inactive_unchanged),
                                dict(**stage, mem_available=available, hidden_unwritten_elements=hidden_unwritten_elements,
                                     hidden_unwritten_rows=hidden_unwritten_rows, inactive_columns=inactive_columns), error)
                            vote({'replay_execution': stage}, valid)
                            if not valid:
                                raise RuntimeError('native draft write coverage/selection failed')
                            if error is not None:
                                raise RuntimeError('combined replay execution failed') from error
                        pairs = {}
                        sensitivity = {}
                        roundtrip = False
                        error = None
                        try:
                            for i, left in enumerate(results):
                                for j in range(i+1, len(results)):
                                    pairs[f'{arms[i]}_vs_{arms[j]}'] = bounded_pair(
                                        left, results[j], sp.draft_logits is not None, selected,
                                        near_zero_relief=False)
                            if probe == 0:
                                zero_by_pattern[pattern] = results[arms.index('graph1')]
                            zero_outputs = zero_by_pattern[pattern]
                            # Prefill feedback is an output; decode hidden is an
                            # input and cannot establish changed-input sensitivity.
                            if index == 0 and value != 0.:
                                sensitivity = {arm: (zero_outputs[1] != result[1])
                                    .any(dim=1).tolist() for arm, result in zip(arms, results)}
                            if value == 0. and probe != 0:
                                for arm, result in zip(arms, results):
                                    pairs[f'zero_roundtrip_{arm}'] = bounded_pair(
                                        zero_outputs, result, sp.draft_logits is not None, selected)
                            roundtrip = all(p['tokens_exact'] and p['drift_bounded']
                                and p['kv_selection_exact'] for name, p in pairs.items()
                                if name.startswith('zero_roundtrip_'))
                            float_exact_pairs += sum(all(o['bit_exact'] for o in pair['outputs'][1:])
                                for pair in pairs.values())
                            float_bounded_pairs += sum(pair['drift_bounded'] for pair in pairs.values())
                        except Exception as exc:
                            error = exc
                        valid = dh.qualification_term('replay_compare',
                            dict(no_exception=error is None,
                                 changed_input_sensitive=all(all(rows) for rows in sensitivity.values()),
                                 zero_input_roundtrip=roundtrip,
                                 finite=bool(pairs) and all(o['eager_nonfinite'] == o['graph_nonfinite'] == 0
                                     for p in pairs.values() for o in p['outputs']),
                                 tokens_exact=bool(pairs) and all(p['tokens_exact'] for p in pairs.values()),
                                 drift_bounded=bool(pairs) and all(p['drift_bounded'] for p in pairs.values()),
                                 kv_selection_exact=bool(pairs) and all(p['kv_selection_exact'] for p in pairs.values()),
                                 bit_exact=bool(pairs) and all(p['bit_exact'] for p in pairs.values())),
                            dict(**stage, criterion='all_outputs_byte_exact_det_align',
                                 changed_rows_by_arm=sensitivity, pairs=pairs), error)
                        vote({'replay_compare': stage}, valid)
                        if not valid:
                            raise RuntimeError('native draft bounded replay criterion failed') from error
                        if rowselect and index == 0:
                            selected_cases.append(dict(descriptor=repr(desc), position=position, value=value, probe=probe, sentinel_pattern=pattern,
                                full_m_selected=True, eager_replay=True, kv_blocks=len(kv)))
                        del results
                        outputs = left = result = None
                    zero_by_pattern.clear()
                    zero_outputs = None
                row = dict(manager=index, descriptor=repr(desc), changed_inputs=3,
                    sentinel_patterns=2, probes=5, input_sequence=[0., .125, 0., -.125, 0.],
                    prefill_sensitivity_checked=index == 0, zero_input_roundtrip=True, criterion='all_outputs_byte_exact_det_align')
                local_counts = dict(float_exact_pairs=float_exact_pairs, float_bounded_pairs=float_bounded_pairs)
                dh.qualification_term('replay_descriptor', dict(no_exception=True), dict(**row, **local_counts))
                # Report TP totals: valid rank-local drift must not change the
                # payload hashed in the final READY vote.
                counts = dh.gather_receipts(local_counts) if torch.distributed.is_initialized() else [local_counts]
                rows.append(dict(**row, **{key: sum(c[key] for c in counts) for key in local_counts}))
    finally:
        for tensor, saved in snapshots:
            tensor.copy_(saved)
        sync()
        scratch_bytes = sum(t.numel()*t.element_size() for t, _ in snapshots)
        snapshots.clear()
        zero_by_pattern.clear()
        tensor = saved = outputs = results = zero_outputs = expected_selection = metadata = hidden_sentinel = None
    report = dict(cases=rows, scratch_bytes=scratch_bytes,
                scope='native draft FULL full-M/eager/repeated-graph and A/B/A/C/A roundtrip pairs: ALL output bytes exact, finite, write-covered, selection-correct; no near-zero relief; det-align required; dummy KV restored; GPU qualification pending')
    if rowselect:
        report['rowselect'] = dict(full_m_selected=True, changed_positions=True, bit_exact=True, det_align=True,
            logits_confidence_feedback_kv=True, cases=selected_cases,
            kv_scope='reserved null block 0 written at every input row; restored before admission')
    return report
