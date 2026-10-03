# SPDX-License-Identifier: Apache-2.0
"""Native MTP shard selection for the fast checkpoint loader.

The full GLM-5.3 checkpoint keeps its single MTP layer (``model.layers.78.*``) in a handful of
the 282 shards. The pinned vLLM draft loader (``DeepSeekMTP.load_weights``) and target loader
(``DeepseekV2Model.load_weights``) both start every weight with the same classifier,
``get_spec_layer_idx_from_weight_name``, and ``continue`` before any side effect:

* the draft drops every name that is *not* an MTP-layer name. ``model.embed_tokens`` and
  ``lm_head`` are dropped too; the proposer shares the target's embedding and head after load.
* the target drops every name that *is* an MTP-layer name.

``GLM_MTP_ONLY_LOAD=1`` feeds the draft only the MTP-layer tensors, read from only the shards
that hold them. ``GLM_TARGET_SKIP_MTP=1`` stops the target load from reading those tensors.
Both consumers then see the same names, order and bytes as with a full scan, minus names they
discard unread. ``GLM_MTP_ONLY_LOAD=audit`` keeps the full scan and runs the same checks.

Fail-closed checks (draft): shard headers, not the index, decide what is read; the index must
agree with the headers; every expected tensor must be yielded exactly once; after the load,
every draft parameter that the checkpoint can fill must be in the loaded set.
"""
import collections
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

DRAFT_CLASSES = {('vllm.model_executor.models.deepseek_mtp', 'DeepSeekMTP')}
TARGET_CLASSES = {('vllm.model_executor.models.deepseek_v2', 'GlmMoeDsaForCausalLM')}
UTILS = 'vllm.model_executor.models.utils'
INDEX = 'model.safetensors.index.json'
# Draft parameters replaced by the target's modules right after the draft load
# (vllm/v1/worker/gpu/spec_decode/eagle/utils.py: load_eagle_model).
SHARED_FROM_TARGET = re.compile(r'(^|\.)embed_tokens\.|(^|\.)shared_head\.head\.|^lm_head\.')
FUSED_EXPERT_PREFIX = re.compile(r'^w(13|2|1|3)_')

_local = threading.local()


def _log(msg):
    sys.stderr.write(f"glm-fast-load: {msg}\n")
    sys.stderr.flush()


def flags(env=None):
    env = os.environ if env is None else env
    draft = env.get('GLM_MTP_ONLY_LOAD', '0')
    target = env.get('GLM_TARGET_SKIP_MTP', '0')
    if draft not in ('0', '1', 'audit'):
        raise ValueError('GLM_MTP_ONLY_LOAD must be 0, 1 or audit')
    if target not in ('0', '1'):
        raise ValueError('GLM_TARGET_SKIP_MTP must be 0 or 1')
    return draft, target


def active():
    return getattr(_local, 'ctx', None)


def _pins():
    return json.loads((Path(__file__).resolve().parents[1] / 'source_pins.json').read_text())


def check_pinned(module):
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != _pins().get(module.__name__):
        raise RuntimeError('glm-mtp-select: source drift ' + module.__name__)


def spec_predicate(config):
    """True for names the pinned classifier assigns to an MTP layer (the stock function itself)."""
    import importlib

    utils = importlib.import_module(UTILS)
    check_pinned(utils)
    fn = utils.get_spec_layer_idx_from_weight_name
    if not getattr(config, 'num_nextn_predict_layers', 0):
        raise RuntimeError('glm-mtp-select: config has no MTP layers')
    return lambda name: fn(config, name) is not None


def classify(model):
    key = (type(model).__module__, type(model).__name__)
    if key in DRAFT_CLASSES:
        return 'draft'
    if key in TARGET_CLASSES:
        return 'target'
    return None


class LoadContext:
    """One DefaultModelLoader.load_weights call of the draft or the target."""

    def __init__(self, kind, mode, predicate):
        self.kind, self.mode, self.pred = kind, mode, predicate
        self.expected = None          # Counter of MTP names from the shard headers (draft)
        self.yielded = collections.Counter()
        self.skipped = 0
        self.complete = False
        self.selected_files = None

    # --- draft planning -------------------------------------------------------------------
    def plan_draft(self, files, stock_skip=None):
        from glm_fast_load import parse_header

        expected, per_file, nbytes, total = collections.Counter(), {}, 0, 0
        for path in files:
            _, header = parse_header(path)
            names = []
            for name, info in header.items():
                n = info['data_offsets'][1] - info['data_offsets'][0]
                total += n
                if self.pred(name) and not (stock_skip and stock_skip(name)):
                    names.append(name)
                    nbytes += n
            if names:
                per_file[path] = names
                expected.update(names)
        if not expected:
            raise RuntimeError('glm-mtp-select: no MTP tensors in the checkpoint shards')
        if any(c != 1 for c in expected.values()):
            raise RuntimeError('glm-mtp-select: duplicate MTP tensor names across shards')
        self._check_index(files, per_file)
        self.expected = expected
        self.selected_files = [f for f in files if f in per_file]
        _log(f"mtp-only {self.mode}: {len(self.selected_files)}/{len(files)} shards hold "
             f"{len(expected)} MTP tensors, {nbytes / (1 << 30):.2f} of {total / (1 << 30):.1f} GiB; "
             f"index agrees: {', '.join(os.path.basename(f) for f in self.selected_files)}")
        return self.selected_files

    def _check_index(self, files, per_file):
        folders = {os.path.dirname(f) for f in files}
        if len(folders) != 1:
            raise RuntimeError('glm-mtp-select: checkpoint shards span several folders')
        path = os.path.join(folders.pop(), INDEX)
        if not os.path.isfile(path):
            raise RuntimeError('glm-mtp-select: ' + INDEX + ' missing; cannot cross-check the shards')
        with open(path) as f:
            weight_map = json.load(f)['weight_map']
        from_index = {n: s for n, s in weight_map.items() if self.pred(n)}
        from_headers = {n: os.path.basename(p) for p, names in per_file.items() for n in names}
        if from_index != from_headers:
            only_i = sorted(set(from_index) - set(from_headers))[:3]
            only_h = sorted(set(from_headers) - set(from_index))[:3]
            moved = sorted(n for n in set(from_index) & set(from_headers) if from_index[n] != from_headers[n])[:3]
            raise RuntimeError(f'glm-mtp-select: index and shard headers disagree on MTP tensors: '
                               f'index-only {only_i}, header-only {only_h}, other shard {moved}')

    # --- iteration ------------------------------------------------------------------------
    def keep(self, name):
        """Names this load must yield (stock skip already applied by the caller)."""
        if self.kind == 'draft':
            return self.mode == 'audit' or self.pred(name)
        if self.pred(name):
            self.skipped += 1
            return False
        return True

    def record(self, name):
        if self.kind == 'draft' and self.pred(name):
            self.yielded[name] += 1

    def finish_iteration(self):
        if self.kind == 'draft' and self.yielded != self.expected:
            missing = sorted((self.expected - self.yielded).elements())[:3]
            extra = sorted((self.yielded - self.expected).elements())[:3]
            raise RuntimeError(f'glm-mtp-select: MTP tensors yielded differ from the shard headers: '
                               f'missing {missing}, unexpected/duplicate {extra}')
        self.complete = True
        if self.kind == 'target':
            _log(f'target skip-mtp: {self.skipped} MTP-layer tensors not read by the target load')

    # --- post-load ------------------------------------------------------------------------
    def check_loaded(self, model, loaded):
        if self.kind != 'draft':
            return
        if not self.complete:
            raise RuntimeError('glm-mtp-select: draft weight stream was not consumed to the end')
        if loaded is None:
            raise RuntimeError('glm-mtp-select: draft load_weights returned no loaded set')
        missing = unloaded_required(model, loaded, self.expected)
        if missing:
            raise RuntimeError(f'glm-mtp-select: {len(missing)} MTP parameter(s) left unloaded: {missing[:5]}')
        _log(f'mtp-only {self.mode} check PASS: {len(self.expected)} tensors -> {len(loaded)} parameters loaded')


def leaf(name):
    return FUSED_EXPERT_PREFIX.sub('', name.rsplit('.', 1)[-1])


def unloaded_required(model, loaded, checkpoint_names):
    """Draft parameters that a checkpoint tensor kind can fill but that the load did not fill.

    A parameter is required when its leaf (``weight``, ``weight_packed``, ``weight_scale``, ...;
    fused-expert ``w13_``/``w2_`` prefixes removed) is a leaf of some checkpoint MTP tensor.
    Kernel-side parameters with no checkpoint counterpart (``g_idx``, sort indices, KV scales)
    and the embedding/head shared from the target are not required.
    """
    leaves = {leaf(n) for n in checkpoint_names}
    loaded = set(loaded)
    return sorted(n for n, _ in model.named_parameters()
                  if not SHARED_FROM_TARGET.search(n) and leaf(n) in leaves and n not in loaded)


# --- vLLM wiring ----------------------------------------------------------------------------

def wrap_load_weights(orig):
    """Wrap DefaultModelLoader.load_weights: set the per-load context, capture the loaded set."""
    def load_weights(self, model, model_config):
        kind = classify(model)
        draft_mode, target_mode = flags()
        mode = draft_mode if kind == 'draft' else target_mode if kind == 'target' else '0'
        if mode == '0':
            return orig(self, model, model_config)
        if active() is not None:
            raise RuntimeError('glm-mtp-select: nested model load')
        check_pinned(sys.modules[type(model).__module__])
        ctx = LoadContext(kind, mode, spec_predicate(model.config))
        captured = {}
        bound = model.load_weights

        def capture(weights):
            captured['loaded'] = bound(weights)
            return captured['loaded']

        model.load_weights = capture
        _local.ctx = ctx
        try:
            result = orig(self, model, model_config)
        finally:
            _local.ctx = None
            del model.load_weights
        ctx.check_loaded(model, captured.get('loaded'))
        return result

    load_weights._glm_mtp_select = True
    return load_weights


def iterate(ctx, files, stock_skip, run):
    """Drive one weights-iterator call under ``ctx``.

    ``run(files, skip)`` returns the underlying (name, tensor) iterator for those files with a
    name-level skip that is honoured before any data is read.
    """
    if ctx.kind == 'draft' and ctx.mode == '1':
        files = ctx.plan_draft(files, stock_skip)
    elif ctx.kind == 'draft':
        ctx.plan_draft(files, stock_skip)

    def skip(name):
        return bool(stock_skip and stock_skip(name)) or not ctx.keep(name)

    for name, tensor in run(files, skip):
        ctx.record(name)
        yield name, tensor
    ctx.finish_iteration()
