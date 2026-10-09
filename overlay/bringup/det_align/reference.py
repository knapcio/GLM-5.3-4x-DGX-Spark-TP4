# SPDX-License-Identifier: Apache-2.0
"""Independent list oracle for tests/bench only; never imported by step code."""

def stock_canonical(flat, e, b, mapping=None, ignore=False, pad=False):
    """Independent Python stock layout followed by sorting each expert segment."""
    n = len(flat)
    groups = [[] for _ in range(e)]
    for i, raw in enumerate(flat):
        raw = ((raw + 2**31) % 2**32) - 2**31  # pinned CUDA signed-int conversion
        if 0 <= raw < e:
            expert = mapping[raw] if mapping is not None and ignore else raw
            if expert != -1:
                groups[expert].append(i)
    capacity = n + e * (b - 1)
    if pad: capacity = (capacity + b - 1) // b * b
    if n < e: capacity = min(n * b, capacity)
    ids, experts = [], []
    for expert, tokens in enumerate(groups):
        rounded = (len(tokens) + b - 1) // b * b
        ids.extend(sorted(tokens) + [n] * (rounded - len(tokens)))
        experts.extend([expert] * (rounded // b))
    post = len(ids)
    ids.extend([n] * (capacity - post))
    experts.extend([-1] * ((capacity + b - 1) // b - len(experts)))
    if mapping is not None and not ignore:
        experts = [mapping[x] for x in experts]
    return ids, experts, [post]

