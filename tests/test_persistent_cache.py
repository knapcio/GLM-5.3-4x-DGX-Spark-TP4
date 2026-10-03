"""Persistent compile cache helper (runs in a container on the fleet; plain files here)."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pcache', ROOT / 'overlay/tools/glm_persistent_cache.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

PARTS = dict(schema=1, image='sha256:' + 'a' * 64, overlay='b' * 64, arch={'TORCH_CUDA_ARCH_LIST': '12.1a'},
             cache_env={'TRITON_CACHE_DIR': '/cache/triton'}, flashinfer='0.6.9')
EXCLUDE = ['d2w2-prefill-control.json', 'd2w2-prefill-control.json.new']


class PersistentCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root, self.cache = base / 'persist', base / 'cache'
        self.root.mkdir()
        self.cache.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def fill(self, cache):
        (cache / 'fi/cached_ops/sampling').mkdir(parents=True)
        lib = cache / 'fi/cached_ops/sampling/sampling.so'
        lib.write_bytes(b'\x7fELF' + b'x' * 100)
        os.utime(lib, (1_700_000_000, 1_700_000_000))
        (cache / 'triton/abc').mkdir(parents=True)
        (cache / 'triton/abc/kernel.cubin').write_bytes(b'cubin')
        (cache / 'd2w2-prefill-control.json').write_text('{"chunk": 2048}')

    def run_seed(self, parts=PARTS, cache=None):
        return P.seed(parts, str(self.root), str(cache or self.cache))

    def run_save(self, parts=PARTS, key=None, max_bytes=1 << 30):
        return P.save(parts, key or P.key_of(parts), EXCLUDE, max_bytes, str(self.root), str(self.cache))

    def test_miss_save_then_hit_on_a_fresh_cache(self):
        self.assertEqual(self.run_seed(), 'MISS')
        self.fill(self.cache)
        self.assertEqual(self.run_save(), 'SAVED')
        gen = self.root / P.key_of(PARTS)
        ready = json.loads((gen / 'READY').read_text())
        self.assertEqual(ready['key'], P.key_of(PARTS))
        manifest = json.loads((gen / 'manifest.json').read_text())
        self.assertEqual(sorted(manifest['files']), ['fi/cached_ops/sampling/sampling.so', 'triton/abc/kernel.cubin'])
        self.assertEqual(json.loads((gen / 'key.json').read_text()), PARTS)
        self.assertFalse((gen / 'cache/d2w2-prefill-control.json').exists())   # deployment state stays out
        self.assertEqual([p.name for p in self.root.iterdir()], [gen.name])     # no temporary left
        fresh = Path(self.tmp.name) / 'fresh-cache'
        fresh.mkdir()
        self.assertEqual(self.run_seed(cache=fresh), 'HIT')
        lib = fresh / 'fi/cached_ops/sampling/sampling.so'
        self.assertEqual(lib.read_bytes(), (self.cache / 'fi/cached_ops/sampling/sampling.so').read_bytes())
        self.assertEqual(int(lib.stat().st_mtime), 1_700_000_000)            # ninja sees outputs as up to date
        self.assertTrue((fresh / 'triton/abc/kernel.cubin').is_file())
        self.assertFalse((fresh / 'd2w2-prefill-control.json').exists())

    def test_existing_generation_is_never_overwritten(self):
        self.fill(self.cache)
        self.assertEqual(self.run_save(), 'SAVED')
        (self.cache / 'triton/abc/kernel.cubin').write_bytes(b'changed')
        self.assertEqual(self.run_save(), 'KEEP')
        gen = self.root / P.key_of(PARTS)
        self.assertEqual((gen / 'cache/triton/abc/kernel.cubin').read_bytes(), b'cubin')

    def test_other_key_is_a_miss_and_builds_fresh(self):
        self.fill(self.cache)
        self.run_save()
        for change in (dict(image='sha256:' + 'c' * 64), dict(flashinfer='0.7.0'), dict(overlay='d' * 64),
                       dict(arch={'TORCH_CUDA_ARCH_LIST': '12.0'})):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                self.assertEqual(self.run_seed(dict(PARTS, **change), Path(d)), 'MISS')
                self.assertEqual(os.listdir(d), [])

    def test_other_key_metadata_is_a_mismatch(self):
        self.fill(self.cache)
        self.run_save()
        gen = self.root / P.key_of(PARTS)
        for text in (json.dumps(dict(PARTS, image='sha256:' + 'e' * 64)), '{not json'):
            original = (gen / 'key.json').read_text()
            (gen / 'key.json').write_text(text)
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(self.run_seed(cache=Path(d)), 'MISMATCH')
                self.assertEqual(os.listdir(d), [])
            (gen / 'key.json').write_text(original)
        (gen / 'READY').unlink()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_seed(cache=Path(d)), 'MISS')

    def corrupt_cases(self):
        def flip(gen):
            (gen / 'cache/triton/abc/kernel.cubin').write_bytes(b'cubiN')          # same size, other bytes
        def extra(gen):
            (gen / 'cache/triton/abc/injected.so').write_bytes(b'x')
        def missing(gen):
            (gen / 'cache/fi/cached_ops/sampling/sampling.so').unlink()
        def relink(gen):
            os.symlink('/etc/passwd', gen / 'cache/triton/abc/link')
        def manifest(gen):
            m = json.loads((gen / 'manifest.json').read_text())
            m['files']['triton/abc/kernel.cubin']['sha256'] = '0' * 64
            (gen / 'manifest.json').write_text(json.dumps(m))
        def ready_key(gen):
            r = json.loads((gen / 'READY').read_text()); r['key'] = 'f' * 32
            (gen / 'READY').write_text(json.dumps(r))
        def ready_legacy(gen):
            (gen / 'READY').write_text(P.key_of(PARTS))                           # no manifest binding
        def no_manifest(gen):
            (gen / 'manifest.json').unlink()
        def fifo(gen):
            os.mkfifo(gen / 'cache/triton/abc/pipe')
        return dict(flip=flip, extra=extra, missing=missing, relink=relink, manifest=manifest,
                    ready_key=ready_key, ready_legacy=ready_legacy, no_manifest=no_manifest, fifo=fifo)

    def test_corrupt_payload_fails_closed_before_copy(self):
        self.fill(self.cache)
        self.run_save()
        gen = self.root / P.key_of(PARTS)
        pristine = Path(self.tmp.name) / 'pristine'
        import shutil
        shutil.copytree(gen, pristine, symlinks=True)
        for name, damage in self.corrupt_cases().items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as d:
                shutil.rmtree(gen); shutil.copytree(pristine, gen, symlinks=True)
                damage(gen)
                with self.assertRaises(P.Corrupt):
                    self.run_seed(cache=Path(d))
                self.assertEqual(os.listdir(d), [])                                 # nothing copied
        shutil.rmtree(gen); shutil.copytree(pristine, gen, symlinks=True)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_seed(cache=Path(d)), 'HIT')

    def test_copy_is_verified_and_removed_on_mismatch(self):
        self.fill(self.cache)
        self.run_save()
        real_run = P.subprocess.run
        def tampering_copy(cmd, **kw):
            result = real_run(cmd, **kw)
            Path(cmd[-1], 'triton/abc/kernel.cubin').write_bytes(b'XXXXX')       # changed during the copy
            return result
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(P.subprocess, 'run', tampering_copy):
            with self.assertRaises(P.Corrupt):
                self.run_seed(cache=Path(d))
            self.assertEqual(os.listdir(d), [])

    def test_main_exit_codes(self):
        import subprocess, sys
        self.fill(self.cache)
        self.run_save()
        gen = self.root / P.key_of(PARTS)
        (gen / 'cache/triton/abc/kernel.cubin').write_bytes(b'cubiN')
        stub = Path(self.tmp.name) / 'stub'
        (stub / 'flashinfer').mkdir(parents=True)
        (stub / 'flashinfer/__init__.py').write_text("__version__ = '0.6.9'\n")
        parts = {k: v for k, v in PARTS.items() if k != 'flashinfer'}
        fresh = Path(self.tmp.name) / 'fresh'; fresh.mkdir()
        env = dict(os.environ, PYTHONPATH=str(stub), PCACHE_ROOT=str(self.root), PCACHE_CACHE=str(fresh),
                   PCACHE_PARTS=json.dumps(parts))
        out = subprocess.run([sys.executable, str(ROOT / 'overlay/tools/glm_persistent_cache.py'), 'seed'],
                             env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, P.CORRUPT_EXIT, out.stderr)
        self.assertIn('PERSISTENT CACHE CORRUPT ' + P.key_of(PARTS), out.stdout)
        self.assertEqual(os.listdir(fresh), [])

    def test_hit_requires_an_empty_deployment_cache(self):
        self.fill(self.cache)
        self.run_save()
        with self.assertRaisesRegex(RuntimeError, 'not empty'):
            self.run_seed()

    def test_key_drift_and_size_limit(self):
        self.fill(self.cache)
        with self.assertRaisesRegex(RuntimeError, 'key changed'):
            self.run_save(key='0' * 32)
        self.assertEqual(self.run_save(max_bytes=10), 'SKIP')
        self.assertEqual(os.listdir(self.root), [])

    def test_output_is_one_parseable_line(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.run_seed()
        line, = out.getvalue().splitlines()
        self.assertEqual(line.split()[:3], ['PERSISTENT', 'CACHE', 'MISS'])
        self.assertRegex(line.split()[3], r'^[0-9a-f]{32}$')


if __name__ == '__main__':
    unittest.main(verbosity=2)
