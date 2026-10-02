# SPDX-License-Identifier: Apache-2.0
"""Dirty-L2 fix: wiring, pins and (with Triton installed) kernel checks on CPU.

Without Triton (macOS) only the wiring and pin tests run. In the pinned serving image:
* TRITON_INTERPRET=1: the discard reduce against the released split32 kernel on the same
  inputs, bit for bit. `discard.global.L2` is replaced by a model that overwrites each
  discarded line with NaN; the interpreter runs programs one after another, so a program
  that discarded a line another program still reads would turn that output into NaN, and
  a missed line would stay finite. This checks line ownership, ordering and coverage.
* TRITON_INTERPRET=0: sm_121 PTX from `triton.compile` (no driver): every discard is
  predicated, after the CTA barrier, and no global load follows the barrier.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRINGUP = ROOT / 'overlay/bringup'
sys.path.insert(0, str(BRINGUP))
import glm_dirty_l2 as D

PROFILE = json.loads((ROOT / 'docs/results/dirty-l2-profile.json').read_text())
INTERP = os.environ.get('TRITON_INTERPRET') == '1'
try:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _poison(ptr, pred):
        # Model of discard.global.L2: the 128-byte line (32 fp32) becomes garbage (NaN).
        lane = tl.arange(0, 32)
        tl.store(ptr[:, None] + lane[None, :], float('nan'), mask=(pred != 0)[:, None])
        return lane
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def needs_interp(fn):
    return unittest.skipUnless(HAVE_TRITON and INTERP, 'needs Triton with TRITON_INTERPRET=1')(fn)


def needs_compile(fn):
    return unittest.skipUnless(HAVE_TRITON and not INTERP, 'needs Triton without the interpreter')(fn)


def inputs(t, w, valid, n=4096, seed=0, fp8=False):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(t, 16, 512, generator=g) * 0.5).bfloat16()
    qr = (torch.randn(t, 16, 64, generator=g) * 0.5).bfloat16()
    cache = torch.randn(n, 576, generator=g) * 0.5
    cache = cache.to(torch.float8_e4m3fn) if fp8 else cache.bfloat16()
    slots = torch.full((t, w), -1, dtype=torch.int32)
    for i in range(t):
        k = valid[i] if isinstance(valid, (list, tuple)) else valid
        slots[i, :k] = torch.randperm(n, generator=g)[:k].int()
    return q, qr, cache, slots


class Wiring(unittest.TestCase):
    def setUp(self):
        self.meta = list(sys.meta_path)

    def tearDown(self):
        sys.meta_path[:] = self.meta

    def test_off_by_default_installs_nothing(self):
        self.assertFalse(D.register({}))
        self.assertFalse(D.register({'GLM_DIRTY_L2': '0'}))
        self.assertFalse(D.register({'GLM_DIRTY_L2': ''}))
        self.assertEqual(sys.meta_path, self.meta)

    def test_only_static_discard_at_split32(self):
        ok = {'GLM_DIRTY_L2': 'discard', 'GLM_FULL_MLA': 'triton', 'GLM_MLA_SPLIT_K': '32'}
        for bad in ('ab', '1', 'yes'):
            with self.assertRaises(ValueError):
                D.register(dict(ok, GLM_DIRTY_L2=bad))
        for key, value in (('GLM_FULL_MLA', '0'), ('GLM_MLA_SPLIT_K', '16'), ('GLM_MLA_SPLIT_K', '8'),
                           ('GLM_MLA_SPLIT_K', '0')):
            with self.assertRaises(ValueError):
                D.register(dict(ok, **{key: value}))
        self.assertEqual(sys.meta_path, self.meta)
        self.assertTrue(D.register(ok))
        self.assertIsInstance(sys.meta_path[0], D.Hooks)
        source = (BRINGUP / 'glm_dirty_l2.py').read_text() + (BRINGUP / 'glm_dirty_l2_kernel.py').read_text()
        for absent in ('VLLM_SERVER_DEV_MODE', 'collective_rpc', '_ab(', 'CTRL', 'SPLIT_MAP'):
            self.assertNotIn(absent, source)

    def test_profiles_and_startup_order(self):
        self.assertIn("export GLM_DIRTY_L2='discard'", (ROOT / 'profiles/current.env').read_text())
        self.assertIn("export GLM_DIRTY_L2='0'", (ROOT / 'profiles/dspark-k3.env').read_text())
        self.assertEqual(PROFILE['added_env'], {'GLM_DIRTY_L2': 'discard'})
        boot = (BRINGUP / 'sitecustomize.py').read_text()
        self.assertLess(boot.index('glm_dsa_short.register()'), boot.index('glm_dirty_l2.register()'))
        self.assertLess(boot.index('glm_dirty_l2.register()'), boot.index('except BaseException'))

    def test_split_kernel_pin_and_kernel_ast(self):
        split = BRINGUP / 'glm_full_mla_split_kernel.py'
        self.assertEqual(D.sha(split), D.SPLIT_PIN)
        self.assertEqual(PROFILE['split_kernel_sha256'], D.SPLIT_PIN)
        text = (BRINGUP / 'glm_dirty_l2_kernel.py').read_text()
        tree = ast.parse(text)
        got = {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest()
               for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PROFILE['kernel_source_sha256']}
        self.assertEqual(got, PROFILE['kernel_source_sha256'])
        # The discard reduce repeats the released _mla_reduce arithmetic statement for statement.
        released = next(n for n in ast.parse(split.read_text()).body if getattr(n, 'name', '') == '_mla_reduce')
        discard = next(n for n in tree.body if getattr(n, 'name', '') == '_mla_reduce_discard')
        body = lambda fn: [ast.dump(s) for s in fn.body
                           if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        self.assertEqual(body(discard)[:len(body(released))], body(released))
        tail = [ast.unparse(s) for s in discard.body[-2:]]
        self.assertEqual(tail, ['tl.debug_barrier()', '_discard_lines(A, t, h, hd, SPLITS, BD, NT)'])
        self.assertEqual(len(body(discard)), len(body(released)) + 2)

    def test_source_pin_refuses_drift(self):
        fake = types.ModuleType(D.SPLIT_MODULE)
        fake.__file__ = __file__
        fake.sparse_mla = lambda *a, **k: None
        with self.assertRaises(RuntimeError):
            D.install_split(fake)

    def test_dispatch_passes_released_partial_at_split32(self):
        calls = []
        kernel = types.ModuleType('glm_dirty_l2_kernel')
        kernel.sparse_mla_discard = lambda *a, **k: calls.append((a, k)) or 'out'
        saved = sys.modules.get('glm_dirty_l2_kernel')
        sys.modules['glm_dirty_l2_kernel'] = kernel
        released = types.ModuleType(D.SPLIT_MODULE)
        released.__file__ = str(BRINGUP / 'glm_full_mla_split_kernel.py')
        stock = released.sparse_mla = lambda *a, **k: 'stock'
        released._mla_partial = object()
        try:
            D.install_split(released)
            D.install_split(released)                    # idempotent
            self.assertIs(released.sparse_mla._stock, stock)
            q = types.SimpleNamespace(shape=(3, 16, 512))
            self.assertEqual(released.sparse_mla(q, 'qr', 'cache', 'slots', 0.1, 0.5), 'out')
            (args, kwargs), = calls
            self.assertEqual(args, (q, 'qr', 'cache', 'slots', 0.1, 0.5))
            self.assertEqual(kwargs, dict(splits=32, partial=released._mla_partial))
        finally:
            if saved is None:
                sys.modules.pop('glm_dirty_l2_kernel', None)
            else:
                sys.modules['glm_dirty_l2_kernel'] = saved

    def test_hook_runs_after_import(self):
        seen = []
        hooks = D.Hooks()
        sys.meta_path.insert(0, hooks)
        sys.modules.pop('json.tool', None)
        hooks.after_import('json.tool', lambda m: seen.append(m.__name__))
        import json.tool  # noqa: F401
        self.assertEqual(seen, ['json.tool'])

    def test_startup_registers_and_aborts_on_bad_mode(self):
        armed = ('import sys\nimport sitecustomize\nimport glm_dirty_l2 as D\n'
                 'assert any(isinstance(h, D.Hooks) for h in sys.meta_path)\n')
        base = dict(PATH=os.environ.get('PATH', ''), PYTHONPATH=str(BRINGUP), GLM_FULL_MLA='triton',
                    GLM_MLA_SPLIT_K='32')
        run = lambda code, **env: subprocess.run([sys.executable, '-S', '-c', code], capture_output=True,
                                                 text=True, env=dict(base, **env))
        ok = run(armed, GLM_DIRTY_L2='discard')
        self.assertEqual(ok.returncode, 0, ok.stderr)
        for env in (dict(GLM_DIRTY_L2='ab'), dict(GLM_DIRTY_L2='discard', GLM_MLA_SPLIT_K='16'),
                    dict(GLM_DIRTY_L2='discard', GLM_FULL_MLA='0')):
            with self.subTest(**env):
                bad = run(armed, **env)
                self.assertEqual(bad.returncode, 78, bad.stderr)
                self.assertIn('GLM_DIRTY_L2', bad.stderr)
        off = run('import sitecustomize, sys\nassert "glm_dirty_l2" not in sys.modules', GLM_DIRTY_L2='0')
        self.assertEqual(off.returncode, 0, off.stderr)


class InterpreterNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (HAVE_TRITON and INTERP):
            return
        import glm_full_mla_split_kernel as R
        import glm_dirty_l2_kernel as K
        cls.R, cls.K = R, K
        cls.real_discard = K._l2_discard
        K._l2_discard = _poison

    @classmethod
    def tearDownClass(cls):
        if HAVE_TRITON and INTERP:
            cls.K._l2_discard = cls.real_discard

    def released(self, q, qr, cache, slots):
        os.environ['GLM_MLA_SPLIT_K'] = '32'
        return self.R.sparse_mla(q, qr, cache, slots, 0.0625, 0.75 if cache.dtype != q.dtype else 1.0)

    def discard(self, q, qr, cache, slots, call=None):
        captured = {}
        real_empty = torch.empty

        def spy(*a, **k):
            x = real_empty(*a, **k)
            if x.dtype == torch.float32 and x.dim() == 4 and x.shape[-1] == 512:
                captured['acc'] = x
            return x
        torch.empty = spy
        try:
            scale = 0.75 if cache.dtype != q.dtype else 1.0
            out = (call or (lambda *a: self.K.sparse_mla_discard(*a, splits=32, partial=self.R._mla_partial)))(
                q, qr, cache, slots, 0.0625, scale)
        finally:
            torch.empty = real_empty
        return out, captured.get('acc')

    def check(self, out, ref, acc):
        self.assertTrue(torch.equal(out.view(torch.int16), ref.view(torch.int16)), 'output bits changed')
        self.assertTrue(torch.isfinite(out.float()).all())
        if acc is not None and acc.numel():
            self.assertTrue(bool(torch.isnan(acc).all()), 'a consumed partial line was not discarded')

    @needs_interp
    def test_bit_identical_and_full_coverage(self):
        for t, valid in ((1, 2048), (3, [2048, 700, 31]), (2, 0)):
            with self.subTest(t=t, valid=valid):
                q, qr, cache, slots = inputs(t, 2048, valid, seed=32 + t)
                ref = self.released(q, qr, cache, slots)
                out, acc = self.discard(q, qr, cache, slots)
                self.check(out, ref, acc)

    @needs_interp
    def test_fp8_cache(self):
        try:
            q, qr, cache, slots = inputs(2, 2048, [1500, 64], seed=5, fp8=True)
            ref = self.released(q, qr, cache, slots)
        except Exception as exc:  # interpreter without fp8 support
            self.skipTest('interpreter fp8 unsupported: ' + repr(exc)[:120])
        out, acc = self.discard(q, qr, cache, slots)
        self.check(out, ref, acc)

    @needs_interp
    def test_installed_dispatch_matches_released(self):
        stock = self.R.sparse_mla
        try:
            D.install_split(self.R)
            self.assertEqual(self.R.sparse_mla._glm_dirty_l2, 'discard')
            q, qr, cache, slots = inputs(3, 2048, [2048, 1200, 9], seed=77)
            os.environ['GLM_MLA_SPLIT_K'] = '32'
            ref = stock(q, qr, cache, slots, 0.0625, 1.0)
            out, acc = self.discard(q, qr, cache, slots, call=self.R.sparse_mla)
            self.check(out, ref, acc)
        finally:
            self.R.sparse_mla = stock


class StaticPTX(unittest.TestCase):
    @needs_compile
    def test_reduce_discards_after_barrier(self):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        import glm_dirty_l2_kernel as K
        fn = K._mla_reduce_discard
        signature = dict(A='*fp32', MD='*fp32', O='*bf16')
        attrs = {(fn.arg_names.index(n),): [['tt.divisibility', 16]] for n in signature}
        src = ASTSource(fn, signature=signature, constexprs=dict(OS=8192, SPLITS=32, BD=64, NT=128), attrs=attrs)
        ptx = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4, num_stages=1)).asm['ptx']
        lines = [l.strip() for l in ptx[ptx.index('.entry'):].splitlines()]
        disc = [i for i, l in enumerate(lines) if 'discard.global.L2' in l]
        bar = [i for i, l in enumerate(lines) if re.match(r'(bar\.sync|barrier\.sync)', l)]
        loads = [i for i, l in enumerate(lines) if re.match(r'(@%p\d+ )?ld\.global', l) or ' ld.global' in l]
        self.assertTrue(disc, 'no discard emitted')
        self.assertTrue(all(', 128;' in lines[i] and lines[i].startswith('@') for i in disc),
                        'each discard must be predicated (one owner thread per line)')
        last_barrier = max(i for i in bar if i < disc[0])
        self.assertLess(max(loads), last_barrier, 'a global load is issued after the barrier')


if __name__ == '__main__':
    unittest.main(verbosity=2)
