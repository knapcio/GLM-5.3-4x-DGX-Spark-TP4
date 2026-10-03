#!/usr/bin/env python3
"""CPU tests for overlay/overlay/glm_param_hash.py (opt-in per-rank weight manifest).

Runs on the Mac (uv torch) and inside the pinned image; set GLM_PHASH_REQUIRE_VLLM=1 in the
image so the real vllm.v1.worker.gpu_worker attach test must run instead of being skipped.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/overlay'))
import glm_param_hash as H  # noqa: E402

FP8 = getattr(torch, 'float8_e4m3fn', None)


class Head(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head


class Target(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.embed_tokens = nn.Embedding(37, 24)
        self.embed_tokens.weight.data = torch.randn(37, 24, generator=g).to(torch.bfloat16)
        self.layers = nn.ModuleList([nn.Linear(24, 40, bias=True) for _ in range(2)])
        for lin in self.layers:
            lin.weight.data = torch.randn(40, 24, generator=g).to(torch.bfloat16)
            lin.bias.data = torch.randn(40, generator=g)
        self.layers[0].weight_packed = nn.Parameter(
            torch.randint(-128, 127, (40, 12), generator=g, dtype=torch.int8), requires_grad=False)
        if FP8 is not None:
            self.layers[1].weight_fp8 = nn.Parameter(
                torch.randn(16, 24, generator=g).to(FP8), requires_grad=False)
        base = torch.randn(24, 9, generator=g)
        wide = torch.randn(8, 32, generator=g)
        self._keep = dict(wide=wide)                                           # not a tensor attr
        self.w_t = nn.Parameter(base.t(), requires_grad=False)                 # non-contiguous
        self.w_pad = nn.Parameter(wide[:, :20], requires_grad=False)           # view of padded storage
        self.scale = nn.Parameter(torch.tensor(0.125), requires_grad=False)    # 0-dim
        self.empty = nn.Parameter(torch.empty(0, 4), requires_grad=False)
        self.register_buffer('cos_sin_cache', torch.randn(64, 8, generator=g), persistent=False)
        self.register_buffer('flags', torch.tensor([True, False, True]))
        self.big = nn.Parameter(torch.randn(3000, generator=g), requires_grad=False)  # 12000 B, many segments
        self.lm_head = Head(nn.Linear(24, 37, bias=False)).head
        self.lm_head.weight = self.embed_tokens.weight                         # tied
        self.workspace = torch.zeros(128, dtype=torch.uint8)                   # scratch attr
        self.topk_indices_buffer = torch.zeros(4, 8, dtype=torch.int32)        # shared with draft
        self.kv_cache = torch.zeros(2, 64)                                     # bound KV, skipped


class Draft(nn.Module):
    def __init__(self, g, target):
        super().__init__()
        self.mtp = nn.Linear(48, 24, bias=False)
        self.mtp.weight.data = torch.randn(24, 48, generator=g).to(torch.bfloat16)
        self.embed_tokens = target.embed_tokens
        self.shared_head = Head(target.lm_head)
        self.topk_indices_buffer = target.topk_indices_buffer


def models(seed=0):
    g = torch.Generator().manual_seed(seed)
    t = Target(g)
    d = Draft(g, t)
    return [('target', t), ('draft', d)]


def manifest(ms, **opts):
    o = dict(chunk_mb=1024 / 2 ** 20, segment_mb=4096 / 2 ** 20)   # 1 KiB chunks, 4 KiB segments
    o.update(opts)
    return H.build_manifest(ms, opts=o)


def ref_sha(t):
    t = t.detach().contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def flip(t, index):
    with torch.no_grad():
        flat = t.data.reshape(-1).view(torch.uint8) if t.is_contiguous() else None
        if flat is None:          # flip through the logical C-order position
            c = t.data.contiguous().reshape(-1).view(torch.uint8)
            c[index] ^= 1
            t.data.copy_(c.view(t.dtype).reshape(t.shape))
        else:
            flat[index] ^= 1


def get(ms, name):
    prefix, rest = name.split('.', 1)
    m = dict(ms)[prefix]
    for part in rest.split('.')[:-1]:
        m = getattr(m, part)
    return getattr(m, rest.split('.')[-1])


class HashTests(unittest.TestCase):
    def test_reference_bytes_all_chunkings(self):
        ms = models()
        rows, todo = H.plan(ms)
        self.assertTrue(any(not t.is_contiguous() for _, t in todo))
        for chunk in (64, 1024, 1 << 20):
            for name, t in todo:
                sha, crcs = H.hash_tensor(t, chunk, max(chunk, 4096))
                self.assertEqual(sha, ref_sha(t), (name, chunk))
                n = t.numel() * t.element_size()
                self.assertEqual(len(crcs), max(1, -(-n // max(chunk, 4096))), name)

    def test_identical_weights_hash_equal(self):
        a, b = manifest(models(0)), manifest(models(0))
        self.assertEqual(a['digest_params_buffers'], b['digest_params_buffers'])
        self.assertEqual(a['digest_attrs'], b['digest_attrs'])
        r = H.compare(a, b)
        self.assertEqual(r['verdict'], 'EQUAL')
        self.assertTrue(r['digest_equal'])
        self.assertEqual(sum(r['counts'].values()), 0)
        c = manifest(models(1))
        self.assertEqual(H.compare(a, c)['verdict'], 'DIFFERENT')

    def test_manifest_contents(self):
        m = manifest(models())
        e = {d['name']: d for d in m['entries']}
        self.assertEqual(e['draft.embed_tokens.weight']['alias_of'], 'target.embed_tokens.weight')
        self.assertEqual(e['draft.shared_head.head.weight']['alias_of'], 'target.embed_tokens.weight')
        self.assertEqual(e['target.lm_head.weight']['alias_of'], 'target.embed_tokens.weight')
        self.assertEqual(e['target.kv_cache']['skipped'], 'kv_cache')
        self.assertEqual(e['target.workspace']['kind'], 'attr')
        self.assertEqual(e['draft.topk_indices_buffer']['alias_of'], 'target.topk_indices_buffer')
        self.assertEqual(e['target.cos_sin_cache']['kind'], 'buffer')
        self.assertEqual(e['target.w_pad']['nbytes'], 8 * 20 * 4)
        self.assertFalse(e['target.w_t']['contiguous'])
        self.assertEqual(e['target.empty']['sha256'], hashlib.sha256(b'').hexdigest())
        self.assertGreater(len(e['target.big']['seg_crc32']), 2)
        json.dumps(m)

    def test_single_flipped_byte_is_detected_by_name(self):
        base = manifest(models())
        cases = [('target.layers.0.weight', 0, ['target.layers.0.weight']),
                 ('target.layers.0.weight_packed', 479, ['target.layers.0.weight_packed']),
                 ('target.w_t', 100, ['target.w_t']),
                 ('target.w_pad', 8 * 20 * 4 - 1, ['target.w_pad']),
                 ('target.scale', 3, ['target.scale']),
                 ('target.cos_sin_cache', 777, ['target.cos_sin_cache']),
                 ('target.flags', 2, ['target.flags']),
                 ('draft.mtp.weight', 1000, ['draft.mtp.weight']),
                 ('draft.embed_tokens.weight', 5, ['draft.embed_tokens.weight', 'draft.shared_head.head.weight',
                                                   'target.embed_tokens.weight', 'target.lm_head.weight'])]
        if FP8 is not None:
            cases.append(('target.layers.1.weight_fp8', 16 * 24 - 1, ['target.layers.1.weight_fp8']))
        for name, index, expect in cases:
            ms = models()
            flip(get(ms, name), index)
            r = H.compare(base, manifest(ms))
            self.assertEqual(r['verdict'], 'DIFFERENT', name)
            self.assertEqual(sorted(x['name'] for x in r['hash_diff']), sorted(expect), name)
            self.assertEqual(r['counts']['attr_hash_diff'], 0, name)

    def test_segment_localisation(self):
        base = manifest(models())
        ms = models()
        flip(ms[0][1].big, 9000)          # byte 9000 of 12000 -> segment 2 of 4 KiB segments
        diff = H.compare(base, manifest(ms))['hash_diff']
        self.assertEqual(diff[0]['diff_segments'], [2])

    def test_padding_outside_view_is_ignored(self):
        base = manifest(models())
        ms = models()
        with torch.no_grad():
            ms[0][1]._keep['wide'][:, 20:] += 1.0
        self.assertEqual(H.compare(base, manifest(ms))['verdict'], 'EQUAL')

    def test_alias_break_detected_even_with_equal_bytes(self):
        base = manifest(models())
        ms = models()
        d = ms[1][1]
        d.embed_tokens = nn.Embedding(37, 24)
        d.embed_tokens.weight = nn.Parameter(ms[0][1].embed_tokens.weight.detach().clone())
        r = H.compare(base, manifest(ms))
        self.assertEqual(r['verdict'], 'DIFFERENT')
        self.assertEqual([x['name'] for x in r['alias_diff']], ['draft.embed_tokens.weight'])
        self.assertEqual(r['hash_diff'], [])

    def test_attr_change_reported_but_not_decisive(self):
        base = manifest(models())
        ms = models()
        ms[0][1].workspace[7] = 9
        ms[0][1].kv_cache += 1
        r = H.compare(base, manifest(ms))
        self.assertEqual(r['verdict'], 'EQUAL')
        self.assertEqual([x['name'] for x in r['attr_hash_diff']], ['target.workspace'])

    def test_missing_and_shape_changes(self):
        base = manifest(models())
        ms = models()
        del ms[1][1].mtp
        ms[0][1].layers[1].bias = nn.Parameter(torch.zeros(41))
        r = H.compare(base, manifest(ms))
        self.assertEqual([x['name'] for x in r['missing_in_b']], ['draft.mtp.weight'])
        self.assertEqual([x['name'] for x in r['meta_diff']], ['target.layers.1.bias'])


class FakeRunner:
    def __init__(self, ms):
        self.ms = dict(ms)
        self.kv_caches = [self.ms['target'].kv_cache]

    def get_model(self):
        return self.ms['target']

    def get_draft_model(self):
        return self.ms['draft']


class RpcTests(unittest.TestCase):
    def setUp(self):
        H.STATE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.opts = dict(out_dir=self.tmp.name, chunk_mb=1024 / 2 ** 20, segment_mb=4096 / 2 ** 20, threads=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_resume_and_file(self):
        w = types.SimpleNamespace(model_runner=FakeRunner(models()))
        first = H.rpc_run(w, 'A-audit0', json.dumps(dict(self.opts, max_seconds=0)))
        self.assertFalse(first['done'], first)
        self.assertNotIn('error', first)
        self.assertEqual(first['hashed'], 3)          # one tensor per thread even with max_seconds=0
        done = H.rpc_run(w, 'A-audit0', json.dumps(dict(self.opts, max_seconds=60)))
        self.assertTrue(done['done'], done)
        self.assertEqual(done['calls'], 2)
        path = Path(done['path'])
        self.assertEqual(path, Path(self.tmp.name) / 'A-audit0' / 'rank-1.json')
        on_disk = json.loads(path.read_text())
        ref = manifest(models())
        self.assertEqual(on_disk['digest_params_buffers'], ref['digest_params_buffers'])
        self.assertEqual(H.compare(on_disk, ref)['verdict'], 'EQUAL')
        self.assertEqual(on_disk['header']['models'], dict(target='Target', draft='Draft'))
        again = H.rpc_run(w, 'A-audit0', '{}')
        self.assertEqual(again['calls'], 2)
        status = H.rpc_status(w)
        self.assertEqual(status['labels'], {'A-audit0': dict(done=True, todo=0)})

    def test_probe(self):
        w = types.SimpleNamespace(model_runner=FakeRunner(models()))
        H.rpc_run(w, 'P', json.dumps(self.opts))
        out = H.rpc_probe(w, 'P', json.dumps(['target.scale', 'nope']))
        p = out['probes']['target.scale']
        self.assertEqual(p['head_hex'], torch.tensor([0.125]).view(torch.uint8).numpy().tobytes().hex())
        self.assertEqual(out['probes']['nope'], dict(error='not found'))

    def test_errors_are_returned_not_raised(self):
        w = types.SimpleNamespace(model_runner=FakeRunner(models()))
        self.assertIn('error', H.rpc_run(w, 'bad/label', '{}'))
        self.assertIn('error', H.rpc_run(w, 'X', json.dumps(dict(bogus=1))))
        self.assertIn('error', H.rpc_run(w, 'X', 'not json'))
        self.assertIn('error', H.rpc_run(types.SimpleNamespace(), 'Y', json.dumps(self.opts)))
        self.assertIn('error', H.rpc_probe(w, 'never-run', '[]'))
        runner = FakeRunner(models())
        runner.get_draft_model = lambda: None
        nodraft = types.SimpleNamespace(model_runner=runner, vllm_config=types.SimpleNamespace(speculative_config=object()))
        self.assertEqual(H.rpc_run(nodraft, 'Z', json.dumps(self.opts))['error'],
                         'speculative config present but no draft model found')

    @unittest.skipUnless(Path('/proc/meminfo').exists(), 'needs /proc/meminfo')
    def test_memory_floor_stops(self):
        w = types.SimpleNamespace(model_runner=FakeRunner(models()))
        out = H.rpc_run(w, 'M', json.dumps(dict(self.opts, mem_floor_gib=1e6)))
        self.assertIn('below floor', out['error'])
        self.assertFalse(out['done'])


class RegisterTests(unittest.TestCase):
    def test_flag_values(self):
        self.assertFalse(H.register({'GLM_PARAM_HASH': '0'}))
        self.assertFalse(H.register({}))
        with self.assertRaises(ValueError):
            H.register({'GLM_PARAM_HASH': 'yes'})

    def test_hook_attaches_to_fake_worker_module(self):
        meta = list(sys.meta_path)
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'phash_fake_worker.py').write_text('class Worker:\n    pass\n')
            sys.path.insert(0, d)
            saved = H.WORKER
            try:
                H.WORKER = 'phash_fake_worker'
                self.assertTrue(H.register({'GLM_PARAM_HASH': '1'}))
                mod = importlib.import_module('phash_fake_worker')
                self.assertIs(mod.Worker.glm_phash_run, H.rpc_run)
                self.assertIs(mod.Worker.glm_phash_status, H.rpc_status)
                self.assertIs(mod.Worker.glm_phash_probe, H.rpc_probe)
            finally:
                H.WORKER = saved
                sys.path.remove(d)
                sys.modules.pop('phash_fake_worker', None)
                sys.meta_path[:] = meta

    def test_real_worker_attach(self):
        import importlib.util
        have = importlib.util.find_spec('vllm') is not None
        if not have:
            if os.environ.get('GLM_PHASH_REQUIRE_VLLM') == '1':
                self.fail('vllm required in the image test')
            self.skipTest('vllm not installed')
        meta = list(sys.meta_path)
        try:
            self.assertTrue(H.register({'GLM_PARAM_HASH': '1'}))
            mod = importlib.import_module(H.WORKER)
            self.assertIs(mod.Worker.glm_phash_run, H.rpc_run)
        finally:
            sys.meta_path[:] = meta

    def test_startup_wiring_is_opt_in(self):
        boot = (ROOT / 'overlay/bringup/sitecustomize.py').read_text()
        self.assertLess(boot.index('glm_param_hash.register()'), boot.index("tail = os.environ.get"))
        self.assertNotIn('GLM_PARAM_HASH', (ROOT / 'profiles/current.env').read_text())
        self.assertNotIn('VLLM_SERVER_DEV_MODE', (ROOT / 'profiles/current.env').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
