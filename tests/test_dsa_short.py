"""Short-context DSA shortcut: source pins, transforms, eligibility and graph dispatch on CPU."""
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get('GLM_IMAGE_SRC', '/image-source'))
SHIM = Path(tempfile.mkdtemp(prefix='glm-dsa-cpu-')) / 'vllm-0.1.dev20051+cpu.dist-info'
SHIM.mkdir(parents=True, exist_ok=True)
(SHIM / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: 0.1.dev20051+cpu\n')
sys.path[:0] = [str(ROOT / 'overlay/bringup'), str(SHIM.parent), str(SRC)]
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
import glm_dsa_short as D

# SHA256 of each pinned module after the transform. The same values come from the
# qualified in-boot lever source with only its module name changed.
TRANSFORMED = {
    D.MODEL: '2b68633a831658ceb7953f449b58f3e7768658379ff05a9746cd307ae5f2b7a2',
    D.SPARSE: '2f2e729d5b1263ce1d3e784a6f5239d1b30dd3f97629379d3776b29d0fd26420',
    D.AUTO: '2d76f034740d0ad7056155bfeabd1d66afb75145a058dbfadea2d028323d04e1',
    D.CG: '9890b8b4e2f42ebf701629cc80863d64f2e144d3d2fa199c8001b5bd0734c93c',
    D.RUN: '101ef063621b02d1275148b2721bb9402bdaad590488a6817b5151a2ad698aea',
}


def source(name):
    return (SRC / (name.replace('.', '/') + '.py')).read_text()


def schedule(n=1, ctx=100, spec=2, new=False):
    ids = [f'r{i}' for i in range(n)]
    return NS(num_scheduled_tokens={r: 3 for r in ids}, scheduled_new_reqs=[1] if new else [],
              scheduled_cached_reqs=NS(resumed_req_ids=set()), total_num_scheduled_tokens=3 * n,
              scheduled_spec_decode_tokens={r: [0] * spec for r in ids}), \
        NS(req_ids=ids, num_tokens=3 * n, has_prefill=False, num_scheduled_tokens=[3] * n), [ctx] * n


class Inert(unittest.TestCase):
    def test_off_by_default_and_bad_value(self):
        prior = list(sys.meta_path)
        self.assertFalse(D.register({}))
        self.assertFalse(D.register({'GLM_INDEXER_SHORTCUT': '0'}))
        self.assertEqual(sys.meta_path, prior)
        with self.assertRaises(ValueError):
            D.register({'GLM_INDEXER_SHORTCUT': 'yes'})

    def test_profile_flags(self):
        current = (ROOT / 'profiles/current.env').read_text()
        # Shortcut remains opt-in; K-stop compatibility does not change defaults.
        self.assertIn("export GLM_INDEXER_SHORTCUT='1'", current)
        self.assertIn("export GLM_MTP_KSTOP='1'", current)
        self.assertIn("export GLM_INDEXER_SHORTCUT='0'", (ROOT / 'profiles/dspark-k3.env').read_text())
        boot = (ROOT / 'overlay/bringup/sitecustomize.py').read_text()
        self.assertLess(boot.index('glm_prefill_switch.register()'), boot.index('glm_dsa_short.register()'))


class Eligibility(unittest.TestCase):
    def test_target_verify(self):
        for n in (1, 4):
            s, b, c = schedule(n, 100)
            self.assertTrue(D.eligible(s, b, c, 3))
        s, b, c = schedule(1, 2045)
        self.assertTrue(D.eligible(s, b, c, 3))      # rows up to 2048 tokens
        s, b, c = schedule(1, 2046)
        self.assertFalse(D.eligible(s, b, c, 3))     # one row would exceed 2048
        for n in (2, 3):
            s, b, c = schedule(n, 100)
            self.assertFalse(D.eligible(s, b, c, 3))  # no graph width for 2/3 requests
        s, b, c = schedule(1, 100, new=True)
        self.assertFalse(D.eligible(s, b, c, 3))
        s, b, c = schedule(1, 100, spec=1)
        self.assertFalse(D.eligible(s, b, c, 3))
        s, b, c = schedule(1, 100)
        self.assertFalse(D.eligible(s, b, c, 2))
        self.assertFalse(D.eligible(s, b, c, 3, supported_model=False))

    def test_draft(self):
        self.assertTrue(D.draft_eligible([100], 1, 3, 3, 0))
        self.assertTrue(D.draft_eligible([100] * 4, 4, 4, 1, 1))
        self.assertTrue(D.draft_eligible([2047], 1, 1, 1, 1))
        self.assertFalse(D.draft_eligible([2048], 1, 1, 1, 1))
        self.assertFalse(D.draft_eligible([100], 1, 3, 3, 0, has_prefill=True))
        self.assertFalse(D.draft_eligible([100], 1, 3, 3, 0, dummy=True))
        self.assertFalse(D.draft_eligible([100] * 2, 2, 2, 1, 1))


class Sources(unittest.TestCase):
    def test_pins_and_transforms(self):
        for name, expected in TRANSFORMED.items():
            text = source(name)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), D.PINS[name])
            out = D.transform(name, text)
            compile(out, name, 'exec')
            self.assertEqual(hashlib.sha256(out.encode()).hexdigest(), expected)
            with self.assertRaises(RuntimeError):
                D.transform(name, text + '\n')

    def test_refuses_late_registration(self):
        before = list(sys.meta_path)
        saved = sys.modules.get(D.CG)
        sys.modules[D.CG] = NS()
        try:
            with self.assertRaises(RuntimeError):
                D.register({'GLM_INDEXER_SHORTCUT': '1'})
        finally:
            if saved is None:
                del sys.modules[D.CG]
            else:
                sys.modules[D.CG] = saved
        self.assertEqual(sys.meta_path, before)


class Dispatch(unittest.TestCase):
    """Real pinned cudagraph_utils through the finder; wrappers driven by a stock stand-in."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest('torch not installed; run through tests/run_cpu_tests.sh')
        assert D.register({'GLM_INDEXER_SHORTCUT': '1'})
        import importlib
        cls.cg = importlib.import_module(D.CG)
        D.ENABLED = False

    def desc(self, tokens, reqs, uniform, short=False):
        cg = self.cg
        return cg.BatchExecutionDescriptor(cg.CUDAGraphMode.FULL, tokens, reqs, uniform, short_context=short)

    def stand_in(self, stock_descs, result):
        cg = self.cg

        class Manager:
            def __init__(self):
                self._capture_descs = {cg.CUDAGraphMode.FULL: list(stock_descs)}
                self._graphs_captured = False
                self.graphs = {}

            def capture(self, factory, progress_bar_desc=''):
                for d in self._capture_descs[cg.CUDAGraphMode.FULL]:
                    self.graphs[d] = factory(d, False)(cg.CUDAGraphMode.NONE)
                self._graphs_captured = True

            def dispatch(self, num_reqs, num_tokens, uniform, loras, max_query_len=None):
                return result[0]

        class ModelManager(Manager):
            pass
        mod = NS(CudaGraphManager=Manager, ModelCudaGraphManager=ModelManager, CUDAGraphMode=cg.CUDAGraphMode)
        D.install_cg(mod)
        D.capture_summary(mod)
        return ModelManager

    def test_real_module_transformed_and_wrapped(self):
        cg = self.cg
        self.assertIn('short_context', {f.name for f in dataclasses.fields(cg.BatchExecutionDescriptor)})
        self.assertNotEqual(self.desc(3, 1, 3), self.desc(3, 1, 3, True))
        self.assertTrue(hasattr(cg.CudaGraphManager.dispatch, '__wrapped__'))
        self.assertTrue(hasattr(cg.CudaGraphManager.capture, '__wrapped__'))

    def test_capture_doubles_eligible_widths_under_short_context(self):
        stock = [self.desc(12, 4, 3), self.desc(6, 2, 3), self.desc(3, 1, 3), self.desc(1, 1, 1)]
        manager = self.stand_in(stock, [None])()
        got = sorted((d.num_tokens, d.num_reqs, d.short_context) for d in manager._capture_descs[self.cg.CUDAGraphMode.FULL])
        self.assertEqual(got, [(1, 1, False), (1, 1, True), (3, 1, False), (3, 1, True),
                               (6, 2, False), (12, 4, False), (12, 4, True)])
        manager.capture(lambda d, warmup: (lambda mode: D.SHORT.get()))
        self.assertTrue(all(flag == d.short_context for d, flag in manager.graphs.items()))

    def test_dispatch_promotes_only_when_eligible(self):
        result = [None]
        stock = [self.desc(3, 1, 3), self.desc(12, 4, 3), self.desc(6, 2, 3)]
        manager = self.stand_in(stock, result)()
        manager.capture(lambda d, warmup: (lambda mode: None))
        D.ENABLED = True
        try:
            for reqs, tokens in ((1, 3), (4, 12)):
                result[0] = self.desc(tokens, reqs, 3)
                with D.context(True):
                    self.assertEqual(manager.dispatch(reqs, tokens, 3, 0), self.desc(tokens, reqs, 3, True))
                with D.context(False):
                    self.assertEqual(manager.dispatch(reqs, tokens, 3, 0), result[0])
            result[0] = self.desc(6, 2, 3)
            with D.context(True):
                self.assertEqual(manager.dispatch(2, 6, 3, 0), result[0])   # no short width for two requests
            D.ENABLED = False
            result[0] = self.desc(3, 1, 3)
            with D.context(True):
                self.assertEqual(manager.dispatch(1, 3, 3, 0), result[0])   # switch off: stock graph
            D.ENABLED = True
            del manager.graphs[self.desc(3, 1, 3, True)]
            with D.context(True), self.assertRaises(RuntimeError):
                manager.dispatch(1, 3, 3, 0)
            self.assertGreater(D.COUNTS['short_dispatch'], 0)
        finally:
            D.ENABLED = False


if __name__ == '__main__':
    unittest.main(verbosity=2)
