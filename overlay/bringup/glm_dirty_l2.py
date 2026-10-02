# SPDX-License-Identifier: Apache-2.0
"""GLM_DIRTY_L2=discard: drop consumed split32 MLA partials from L2. Inert unless set.

The released split-K sparse MLA writes 1 MiB of fp32 partials per row, reduces them once
and never reads them again. Left dirty in L2, they are written back to DRAM while the
following UV bmm and INT8 o_proj stream their weights. This wraps the source-pinned
`glm_full_mla_split_kernel.sparse_mla`: the released `_mla_partial` runs unchanged, and the
reduce is the released body followed by a CTA barrier and one predicated
`discard.global.L2` per consumed 128-byte line. Output bits are unchanged; no buffer,
graph or memory layout changes. Qualified at 32 splits only.
"""
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

SPLIT_MODULE = 'glm_full_mla_split_kernel'
SPLIT_PIN = '16eebe816e7c77b387e091aa71c4090b7af9c22cb47d42b9985745c3bfb9f54c'
SPLITS = 32
_first = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_dispatch(released):
    from glm_dirty_l2_kernel import sparse_mla_discard
    stock = released.sparse_mla
    partial = released._mla_partial

    def sparse_mla(q, qr, cache, slots, scale, ckv_scale=1.0):
        if not _first:
            _first.append(q.shape[0])
            sys.stderr.write(f'glm-dirty-l2: first discard dispatch rows={q.shape[0]}\n')
        return sparse_mla_discard(q, qr, cache, slots, scale, ckv_scale, splits=SPLITS, partial=partial)

    sparse_mla._glm_dirty_l2 = 'discard'
    sparse_mla._stock = stock
    return sparse_mla


def install_split(module):
    if sha(module.__file__) != SPLIT_PIN:
        raise RuntimeError('glm-dirty-l2: split kernel source drift ' + module.__file__)
    if getattr(module.sparse_mla, '_glm_dirty_l2', None):
        return
    module.sparse_mla = make_dispatch(module)
    sys.stderr.write('glm-dirty-l2: armed (discard, split 32)\n')


class Hooks(importlib.abc.MetaPathFinder):
    """After-import callbacks; composes with the other startup finders."""

    def __init__(self):
        self.callbacks = {}
        self.busy = set()

    def after_import(self, name, fn):
        if name in sys.modules:
            fn(sys.modules[name])
            return
        self.callbacks.setdefault(name, []).append(fn)

    def find_spec(self, name, path=None, target=None):
        if name not in self.callbacks or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None:
            raise ImportError(name)
        original = spec.loader.exec_module
        callbacks = self.callbacks.pop(name)

        def exec_module(module):
            original(module)
            for fn in callbacks:
                fn(module)
        spec.loader.exec_module = exec_module
        return spec


def register(env=None):
    e = os.environ if env is None else env
    mode = e.get('GLM_DIRTY_L2', '0') or '0'
    if mode == '0':
        return False
    if mode != 'discard':
        raise ValueError('GLM_DIRTY_L2 must be 0 or discard')
    if e.get('GLM_FULL_MLA') != 'triton' or e.get('GLM_MLA_SPLIT_K') != str(SPLITS):
        raise ValueError('GLM_DIRTY_L2=discard needs GLM_FULL_MLA=triton and GLM_MLA_SPLIT_K=32')
    hooks = Hooks()
    sys.meta_path.insert(0, hooks)
    hooks.after_import(SPLIT_MODULE, install_split)
    return True
