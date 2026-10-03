"""Native MTP shard selection: the pinned vLLM loaders see the same tensors and bytes.

The draft runs through the image's own DefaultModelLoader.load_weights, weight_utils iterator
and DeepSeekMTP.load_weights (on a stub module whose parameters record every weight_loader
call with a SHA-256 of the bytes). The reference is the unpatched stock vLLM iterator scanning
every shard. Run inside tests/run_cpu_tests.sh (GLM_IMAGE_SRC set). With GLM_MTP_LAB=<dir of
real shards + model.safetensors.index.json + config.json> the same comparison runs on the real
full-GLM MTP shards.
"""
import ast
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get('GLM_IMAGE_SRC', '/image-source'))
SHIM = ROOT / 'tests/.cpu-shim/vllm-0.1.dev20051+cpu.dist-info'
SHIM.mkdir(parents=True, exist_ok=True)
(SHIM / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: 0.1.dev20051+cpu\n')
sys.path[:0] = [str(ROOT / 'overlay/overlay'), str(SHIM.parent), str(SRC)]
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ.update(GLM_MTP_ONLY_LOAD='1', GLM_TARGET_SKIP_MTP='1', GLM_FAST_LOAD_THREADS='2',
                  GLM_FAST_LOAD_VERIFY='2')

import torch  # noqa: E402
torch.set_num_threads(2)
from safetensors.torch import save_file  # noqa: E402

import glm_fast_load as gfl  # noqa: E402
import glm_mtp_select as sel  # noqa: E402
import vllm.model_executor.model_loader.weight_utils as wu  # noqa: E402
import vllm.model_executor.model_loader.default_loader as dl  # noqa: E402
import vllm.model_executor.models.deepseek_mtp as mtp  # noqa: E402
import vllm.model_executor.models.deepseek_v2 as dsv2  # noqa: E402
from vllm.config import LoadConfig  # noqa: E402

STOCK_ITER = wu.safetensors_weights_iterator
gfl._patch_default_loader(dl)          # flags are non-zero here, so load_weights is wrapped too
assert getattr(dl.DefaultModelLoader.load_weights, '_glm_mtp_select', False)

STACKED = [('gate_up_proj', 'gate_proj'), ('gate_up_proj', 'up_proj'),
           ('fused_qkv_a_proj', 'q_a_proj'), ('fused_qkv_a_proj', 'kv_a_proj_with_mqa'),
           ('wk_weights_proj', 'wk'), ('wk_weights_proj', 'weights_proj')]


def digest(t):
    t = t.contiguous()
    h = hashlib.sha256()
    h.update(t.reshape(-1).view(torch.uint8).numpy().data if t.numel() else b'')
    return h.hexdigest()


class Param:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def weight_loader(self, param, weight, *args, **kwargs):
        kwargs.pop('return_success', None)
        self.calls.append((self.name, repr(args), repr(sorted(kwargs.items())), str(weight.dtype),
                           tuple(weight.shape), digest(weight)))
        return True


def config_ns(path):
    cfg = json.loads(Path(path).read_text())
    return NS(**{k: cfg[k] for k in ('num_hidden_layers', 'num_nextn_predict_layers', 'n_routed_experts',
                                     'n_shared_experts')})


def draft_param_names(checkpoint_names, config):
    """The draft's parameter names the pinned load_weights addresses, plus kernel/shared extras."""
    layer = config.num_hidden_layers
    out = set()
    for n in checkpoint_names:
        r = mtp.DeepSeekMTP._rewrite_spec_layer_name(None, layer, n)
        m = re.search(r'mlp\.experts\.\d+\.(gate|up|down)_proj\.(.+)$', r)
        if m:
            out.add(r[:m.start()] + 'mlp.experts.routed_experts.' + ('w2_' if m.group(1) == 'down' else 'w13_') + m.group(2))
            continue
        for fused, part in STACKED:
            if part in r:
                r = r.replace(part, fused)
                break
        out.add(r)
    p = f'model.layers.{layer}.'
    extras = ['model.embed_tokens.weight', p + 'shared_head.head.weight',
              p + 'mtp_block.mlp.experts.routed_experts.w13_weight_g_idx', p + 'mtp_block.mlp.experts.routed_experts.w2_weight_g_idx',
              p + 'mtp_block.self_attn.mla_attn.k_scale']
    return sorted(out | set(extras))


def stub(module, name, load_weights, config, params, calls):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = NS(mtp_start_layer_idx=config.num_hidden_layers,
                        num_mtp_layers=config.num_nextn_predict_layers)

    body = dict(__init__=__init__, load_weights=load_weights,
                named_parameters=lambda self, *a, **k: iter([(n, Param(n, calls)) for n in params]))
    if name == 'DeepSeekMTP':
        body['_rewrite_spec_layer_name'] = mtp.DeepSeekMTP._rewrite_spec_layer_name
    cls = type(name, (torch.nn.Module,), body)
    cls.__module__ = module
    return cls()


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@contextlib.contextmanager
def stock_iterator():
    patched = wu.safetensors_weights_iterator
    wu.safetensors_weights_iterator = dl.safetensors_weights_iterator = STOCK_ITER
    try:
        yield
    finally:
        wu.safetensors_weights_iterator = dl.safetensors_weights_iterator = patched


@contextlib.contextmanager
def files_seen():
    seen = []
    inner = gfl.fast_safetensors_iterator

    def recording(files, *a, **kw):
        files = list(files)
        seen.append([os.path.basename(f) for f in files])
        return inner(files, *a, **kw)

    gfl.fast_safetensors_iterator = recording
    try:
        yield seen
    finally:
        gfl.fast_safetensors_iterator = inner


def load(folder, model):
    loader = dl.DefaultModelLoader(LoadConfig())
    loader._init_ep_weight_filter = lambda model_config: None     # no expert parallelism here
    loader.load_weights(model, NS(model=str(folder), revision=None, quantization='compressed-tensors'))


def run_draft(folder, mode, params=None, stock=False):
    config = config_ns(Path(folder) / 'config.json')
    if params is None:
        index = json.loads((Path(folder) / sel.INDEX).read_text())['weight_map']
        params = draft_param_names([n for n in index if n.startswith(f'model.layers.{config.num_hidden_layers}.')],
                                   config)
    calls, out = [], {}

    def load_weights(self, weights):
        out['loaded'] = mtp.DeepSeekMTP.load_weights(self, weights)
        return out['loaded']

    model = stub('vllm.model_executor.models.deepseek_mtp', 'DeepSeekMTP', load_weights, config, params, calls)
    log = io.StringIO()
    t0 = time.time()
    with env(GLM_MTP_ONLY_LOAD=mode), files_seen() as seen, contextlib.redirect_stderr(log), \
            (stock_iterator() if stock else contextlib.nullcontext()):
        load(folder, model)
    return dict(calls=calls, loaded=out['loaded'], files=seen, log=log.getvalue(), seconds=time.time() - t0)


def run_target(folder, mode):
    config = config_ns(Path(folder) / 'config.json')
    received = []

    def load_weights(self, weights):
        for name, t in weights:
            received.append((name, digest(t)))
        return set()

    model = stub('vllm.model_executor.models.deepseek_v2', 'GlmMoeDsaForCausalLM', load_weights, config, [], [])
    with env(GLM_TARGET_SKIP_MTP=mode), files_seen() as seen, contextlib.redirect_stderr(io.StringIO()):
        load(folder, model)
    return received, seen


# --- synthetic checkpoint ----------------------------------------------------------------------

def layer_tensors(i, mtp_layer, g):
    p = f'model.layers.{i}.'
    t = {p + 'input_layernorm.weight': torch.randn(16, generator=g).bfloat16()}
    for lin in ('self_attn.q_a_proj', 'self_attn.kv_a_proj_with_mqa', 'mlp.shared_experts.gate_proj',
                'mlp.shared_experts.up_proj', 'mlp.shared_experts.down_proj'):
        t[p + lin + '.weight_packed'] = torch.randint(-2**31, 2**31 - 1, (8, 4), generator=g, dtype=torch.int32)
        t[p + lin + '.weight_scale'] = torch.randn(8, 1, generator=g).bfloat16()
        t[p + lin + '.weight_shape'] = torch.tensor([8, 32], dtype=torch.int64)
    t[p + 'self_attn.indexer.wk.weight'] = torch.randn(4, 16, generator=g).bfloat16()
    t[p + 'self_attn.indexer.weights_proj.weight'] = torch.randn(2, 16, generator=g).bfloat16()
    t[p + 'mlp.gate.weight'] = torch.randn(4, 16, generator=g)
    t[p + 'mlp.gate.e_score_correction_bias'] = torch.randn(4, generator=g)
    for e in range(4):
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            q = f'{p}mlp.experts.{e}.{proj}.'
            t[q + 'weight_packed'] = torch.randint(-2**31, 2**31 - 1, (8, 2), generator=g, dtype=torch.int32)
            t[q + 'weight_scale'] = torch.randn(8, 1, generator=g).bfloat16()
            t[q + 'weight_shape'] = torch.tensor([8, 16], dtype=torch.int64)
    if mtp_layer:
        for n in ('enorm', 'hnorm', 'shared_head.norm'):
            t[p + n + '.weight'] = torch.randn(16, generator=g).bfloat16()
        t[p + 'eh_proj.weight'] = torch.randn(16, 32, generator=g).bfloat16()
    return t


def synthetic(folder, move_in_index=None, drop_from_index=None):
    g = torch.Generator().manual_seed(0)
    l0, l1, l2 = layer_tensors(0, False, g), layer_tensors(1, False, g), layer_tensors(2, True, g)
    attn2 = {k: v for k, v in l2.items() if '.self_attn.' in k}
    shards = {
        'model-00001-of-00004.safetensors': {'model.embed_tokens.weight': torch.randn(32, 16, generator=g).bfloat16(),
                                             'lm_head.weight': torch.randn(32, 16, generator=g).bfloat16(), **l0},
        'model-00002-of-00004.safetensors': {**l1, **attn2},
        'model-00003-of-00004.safetensors': {k: v for k, v in l2.items() if k not in attn2},
        'model-00004-of-00004.safetensors': {'model.norm.weight': torch.randn(16, generator=g).bfloat16()},
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, os.path.join(folder, name))
        weight_map.update({k: name for k in tensors})
    if move_in_index:
        weight_map[move_in_index] = 'model-00004-of-00004.safetensors'
    if drop_from_index:
        del weight_map[drop_from_index]
    (Path(folder) / sel.INDEX).write_text(json.dumps(dict(metadata={}, weight_map=weight_map)))
    (Path(folder) / 'config.json').write_text(json.dumps(dict(
        num_hidden_layers=2, num_nextn_predict_layers=1, n_routed_experts=4, n_shared_experts=1)))
    return shards


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.shards = synthetic(cls.dir)
        cls.ref = run_draft(cls.dir, '0', stock=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_draft_same_calls_and_loaded_set(self):
        for mode in ('0', '1', 'audit'):
            with self.subTest(mode=mode):
                got = run_draft(self.dir, mode)
                self.assertEqual(got['calls'], self.ref['calls'])
                self.assertEqual(got['loaded'], self.ref['loaded'])
        self.assertGreater(len(self.ref['calls']), 50)
        self.assertTrue(all(c[0].startswith('model.layers.2.') for c in self.ref['calls']))

    def test_draft_reads_only_mtp_shards(self):
        got = run_draft(self.dir, '1')
        self.assertEqual(got['files'], [['model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors']])
        self.assertIn('mtp-only 1: 2/4 shards', got['log'])
        self.assertIn('check PASS', got['log'])
        self.assertEqual(run_draft(self.dir, 'audit')['files'], [sorted(self.shards)])

    def test_target_skips_mtp_layer_only(self):
        full, seen_full = run_target(self.dir, '0')
        skip, seen_skip = run_target(self.dir, '1')
        self.assertEqual(skip, [x for x in full if not x[0].startswith('model.layers.2.')])
        self.assertLess(len(skip), len(full))
        self.assertEqual(seen_full, seen_skip)       # same files; shard 3 contributes only its header

    def test_unloaded_mtp_parameter_raises(self):
        config = config_ns(Path(self.dir) / 'config.json')
        index = json.loads((Path(self.dir) / sel.INDEX).read_text())['weight_map']
        params = draft_param_names([n for n in index if n.startswith('model.layers.2.')], config)
        for mode in ('1', 'audit'):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'left unloaded'):
                run_draft(self.dir, mode, params=params + ['model.layers.2.mtp_block.self_attn.o_proj.weight_packed'])

    def test_index_disagreement_raises(self):
        for kw in (dict(move_in_index='model.layers.2.enorm.weight'),
                   dict(drop_from_index='model.layers.2.mlp.experts.3.down_proj.weight_scale')):
            with self.subTest(**kw), tempfile.TemporaryDirectory() as d:
                synthetic(d, **kw)
                with self.assertRaisesRegex(RuntimeError, 'index and shard headers disagree'):
                    run_draft(d, '1')

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic(d)
            os.unlink(os.path.join(d, sel.INDEX))
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                run_draft(d, '1', params=draft_param_names(['model.layers.2.enorm.weight'], config_ns(Path(d) / 'config.json')))

    def test_stream_mismatch_raises(self):
        ctx = sel.LoadContext('draft', '1', lambda n: n.startswith('model.layers.2.'))
        ctx.expected = collections.Counter(['model.layers.2.a', 'model.layers.2.b'])
        ctx.record('model.layers.2.a')
        with self.assertRaisesRegex(RuntimeError, 'yielded differ'):
            ctx.finish_iteration()
        ctx.record('model.layers.2.b'); ctx.record('model.layers.2.b')
        with self.assertRaisesRegex(RuntimeError, 'yielded differ'):
            ctx.finish_iteration()
        with self.assertRaisesRegex(RuntimeError, 'not consumed'):
            ctx.check_loaded(NS(named_parameters=lambda: iter([])), set())

    def test_inert_and_bad_flags(self):
        self.assertEqual(sel.flags({}), ('0', '0'))
        for bad in (dict(GLM_MTP_ONLY_LOAD='yes'), dict(GLM_TARGET_SKIP_MTP='audit')):
            with self.assertRaises(ValueError):
                sel.flags(bad)
        self.assertIsNone(sel.classify(torch.nn.Linear(1, 1)))

    def test_source_drift_raises(self):
        with tempfile.NamedTemporaryFile(suffix='.py') as f:
            f.write(Path(mtp.__file__).read_bytes() + b'\n'); f.flush()
            with self.assertRaisesRegex(RuntimeError, 'source drift'):
                sel.check_pinned(NS(__name__=mtp.__name__, __file__=f.name))
        sel.check_pinned(mtp)
        sel.check_pinned(dsv2)


class SourceProof(unittest.TestCase):
    """Both pinned consumers classify every name with the same function and discard first."""

    def loop_head(self, path, cls, fn='load_weights'):
        tree = ast.parse(Path(path).read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        func = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        loop = next(n for n in ast.walk(func) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'weights')
        return [ast.unparse(s) for s in loop.body[:3]]

    def test_draft_consumer_drops_non_mtp_names_first(self):
        head = self.loop_head(mtp.__file__, 'DeepSeekMTP')
        self.assertEqual(head, ["if 'rotary_emb.inv_freq' in name:\n    continue",
                                'spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)',
                                'if spec_layer is None:\n    continue'])

    def test_target_consumer_drops_mtp_names_first(self):
        head = self.loop_head(dsv2.__file__, 'DeepseekV2Model')
        self.assertEqual(head, ["if 'rotary_emb.inv_freq' in name:\n    continue",
                                'spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)',
                                'if spec_layer is not None:\n    continue'])
        tree = ast.parse(Path(dsv2.__file__).read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertEqual([ast.unparse(s) for s in classes['GlmMoeDsaForCausalLM'].body], ['pass'])
        causal = next(n for n in classes['DeepseekV2ForCausalLM'].body
                      if isinstance(n, ast.FunctionDef) and n.name == 'load_weights')
        self.assertEqual([ast.unparse(s) for s in causal.body],
                         ['loader = AutoWeightsLoader(self)', 'return loader.load_weights(weights)'])
        # AutoWeightsLoader hands every "model." name to DeepseekV2Model.load_weights unchanged
        # apart from the stripped prefix; the classifier accepts both spellings.
        from vllm.model_executor.models.utils import get_spec_layer_idx_from_weight_name as f
        cfg = NS(num_hidden_layers=78, num_nextn_predict_layers=1)
        self.assertEqual((f(cfg, 'model.layers.78.x'), f(cfg, 'layers.78.x'), f(cfg, 'model.layers.7.x'),
                          f(cfg, 'model.layers.780.x'), f(cfg, 'lm_head.weight')), (78, 78, None, None, None))

    def test_proposer_replaces_embedding_and_head(self):
        text = (SRC / 'vllm/v1/worker/gpu/spec_decode/eagle/utils.py').read_text()
        self.assertIn('draft_inner.embed_tokens = target_embed', text)
        self.assertIn('sh.head = target_lm_head', text)
        self.assertNotIn('has_own_embed_tokens', Path(mtp.__file__).read_text())
        self.assertNotIn('has_own_lm_head', Path(mtp.__file__).read_text())


@unittest.skipUnless(os.environ.get('GLM_MTP_LAB'), 'set GLM_MTP_LAB to the real-shard folder')
class RealShards(unittest.TestCase):
    def test_real_mtp_shards_byte_identical(self):
        folder = Path(os.environ['GLM_MTP_LAB'])
        index = json.loads((folder / sel.INDEX).read_text())['weight_map']
        mtp_shards = sorted({s for n, s in index.items() if n.startswith('model.layers.78.')})
        self.assertEqual(mtp_shards, [f'model-{i:05d}-of-00282.safetensors' for i in range(270, 275)])
        self.assertEqual(len(set(index.values())), 282)
        # The lab folder holds 6 of the 282 shards: let the stock folder filter keep the present
        # ones instead of refusing the missing ones. Everything downstream is unchanged.
        stock_filter = dl.filter_duplicate_safetensors_files

        def present_only(files, folder_, index_file):
            names = set(json.loads(Path(folder_, index_file).read_text())['weight_map'].values())
            return [f for f in files if os.path.basename(f) in names]

        dl.filter_duplicate_safetensors_files = present_only
        try:
            ref = run_draft(folder, '0', stock=True)
            got = run_draft(folder, '1')
            audit = run_draft(folder, 'audit')
        finally:
            dl.filter_duplicate_safetensors_files = stock_filter
        self.assertEqual(got['files'], [mtp_shards])
        self.assertEqual(len(got['calls']), len(ref['calls']))
        self.assertEqual(got['calls'], ref['calls'])
        self.assertEqual(got['loaded'], ref['loaded'])
        self.assertEqual((audit['calls'], audit['loaded']), (ref['calls'], ref['loaded']))
        self.assertIn('mtp-only audit check PASS', audit['log'])
        nbytes = sum(os.path.getsize(folder / s) for s in mtp_shards)
        everything = sum(os.path.getsize(p) for p in folder.glob('*.safetensors'))
        summary = dict(tensors_loaded=len(got['calls']), params_loaded=len(got['loaded']),
                       call_sha256=hashlib.sha256(repr(got['calls']).encode()).hexdigest(),
                       shards_read=mtp_shards, shard_bytes_read=nbytes, lab_bytes=everything,
                       stock_full_scan_s=round(ref['seconds'], 1), mtp_only_s=round(got['seconds'], 1),
                       audit_s=round(audit['seconds'], 1),
                       log=[l for l in got['log'].splitlines() if 'mtp-only' in l])
        print('REAL MTP SHARDS PASS ' + json.dumps(summary))


if __name__ == '__main__':
    unittest.main(verbosity=2)
