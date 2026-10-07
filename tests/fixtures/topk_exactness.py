# SPDX-License-Identifier: Apache-2.0
"""Exact stock top-k parity, allowing only bitwise-equal cutoff tie swaps."""
import torch


def assert_topk_exact(logits, actual, expected, start, end):
    """Check one FP32 score row and its request-relative selected IDs.

    Stock assigns cutoff ties by atomic arrival order. IDs above the cutoff
    remain mandatory; only equal score bits at the cutoff may change IDs.
    """
    assert logits.dtype == torch.float32 and logits.ndim == 1
    assert actual.ndim == expected.ndim == 1 and actual.shape == expected.shape
    assert 0 <= start <= end <= logits.numel()
    scores = logits[start:end]
    assert not torch.isnan(scores).any(), 'NaN valid score'
    count = min(actual.numel(), end-start)
    selections = []
    for ids in (actual, expected):
        assert torch.all((ids == -1) | ((ids >= 0) & (ids < end-start))), 'invalid ID'
        valid = ids[ids != -1].long().sort().values
        assert valid.numel() == count, 'selection/sentinel count'
        assert valid.unique().numel() == count, 'duplicate ID'
        selections.append(valid)
    if not count:
        return
    a, b = selections
    sa, sb = scores[a], scores[b]
    cutoff = sb.min()
    required = torch.nonzero(scores > cutoff).flatten()
    for ids, selected in ((a, sa), (b, sb)):
        assert torch.all(selected >= cutoff), 'selected score below cutoff'
        assert torch.equal(ids[selected > cutoff], required), 'strict-score IDs'
    assert torch.equal(sa.view(torch.int32).sort().values,
                       sb.view(torch.int32).sort().values), 'selected score bits'
    # Numeric equality alone is insufficient (notably +0 versus -0). All
    # symmetric-difference IDs must have the same cutoff score bits.
    changed = torch.cat((a[~torch.isin(a, b)], b[~torch.isin(b, a)]))
    if changed.numel():
        changed_scores = scores[changed]
        assert torch.all(changed_scores == cutoff), 'ID swap outside cutoff'
        bits = changed_scores.view(torch.int32)
        assert torch.all(bits == bits[0]), 'cutoff swap score bits'
