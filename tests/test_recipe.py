"""Offline configuration, drift, manifest, kernel preservation and dry-mode checks."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'overlay/bringup'), str(ROOT/'overlay/swa-pool')]
import compare_dry
import weights
import glm_full_mla
import glm_dsa_swa_pool


class Recipe(unittest.TestCase):
    def test_four_dry_vectors(self):
        rendered, diff = compare_dry.compare()
        self.assertEqual(diff, [])
        self.assertEqual(rendered.count('docker run '), 8)
        self.assertEqual(rendered.count('docker run --rm '), 4)              # JIT prep, one per rank
        self.assertEqual(rendered.count('docker run -d '), 4)
        self.assertLess(rendered.rindex('docker run --rm '), rendered.index('docker run -d '))
        self.assertEqual(rendered.count('--speculative-config'), 4)
        self.assertNotIn('--profiler', rendered)

    def test_dry_has_no_fleet_or_state_access(self):
        with tempfile.TemporaryDirectory() as d:
            shim = Path(d)/'ssh'
            shim.write_text('#!/bin/sh\nexit 99\n');shim.chmod(0o755)
            before = (ROOT/'state').exists()
            subprocess.run([str(ROOT/'start.sh'), 'serve'], check=True, stdout=subprocess.DEVNULL,
                           env=dict(os.environ, DRY='1', PATH=d+':'+os.environ['PATH']))
            self.assertEqual((ROOT/'state').exists(), before)

    def test_mla_inert_and_bad_mode(self):
        prior = list(sys.meta_path)
        self.assertFalse(glm_full_mla.register({}))
        self.assertEqual(sys.meta_path, prior)
        with self.assertRaises(RuntimeError):
            glm_full_mla.register({'GLM_FULL_MLA':'bad'})

    def test_mla_pin_and_drift(self):
        source = Path(os.environ['GLM_IMAGE_SRC'])/'vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py'
        glm_full_mla.check_source(source)
        with tempfile.NamedTemporaryFile() as f:
            f.write(source.read_bytes()+b'\n');f.flush()
            with self.assertRaises(RuntimeError):
                glm_full_mla.check_source(f.name)

    def test_kernel_body_pins(self):
        expected = json.loads((ROOT/'docs/results/source-provenance.json').read_text())
        for path in ['bringup/glm_full_mla_kernel.py','bringup/glm_full_mla_split_kernel.py']:
            tree = ast.parse((ROOT/'overlay'/path).read_text())
            actual = hashlib.sha256(ast.dump(ast.Module(body=tree.body[1:], type_ignores=[]), include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, expected['kernel_ast_sha256'][path])

    def test_split_dispatch_contract(self):
        source = (ROOT/'overlay/bringup/glm_full_mla.py').read_text()
        self.assertIn("t <= int(os.environ.get('GLM_MLA_SPLIT_MAX_ROWS','36'))", source)
        self.assertIn('return split_mla(qn,qr,cache,slots,self.scale,ks), None', source)
        self.assertNotIn('glm_w4_mla_splitk', source)
        self.assertNotIn('glm_w6_mla_oracle', source)

    def test_model_revisions_and_complete_manifests(self):
        for kind, rev, tensors in [('target','206507bbb047d8223964a0414cd83230c59428f9',282),
                                   ('drafter','b374b95663447ea0e935151be4f3d6666e36e6d7',1)]:
            manifest = json.loads((ROOT/f'manifests/{kind}.json').read_text())
            weights.validate_manifest(manifest)
            self.assertEqual(manifest['sha'], rev)
            self.assertEqual(sum(f['f'].endswith('.safetensors') for f in manifest['files']), tensors)
            text = ''.join(f"{f['sha256']}  {f['f']}\n" for f in manifest['files'])
            self.assertEqual(text,(ROOT/f'manifests/{kind}.sha256').read_text())

    def test_verify_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.json').write_bytes(b'original')
            manifest=dict(sha='0'*40,repo='example/model',files=[dict(f='config.json',size=8,sha256=hashlib.sha256(b'original').hexdigest())])
            weights.verify(root,manifest)
            (root/'config.json').write_bytes(b'corrupt!')
            with self.assertRaises(ValueError): weights.verify(root,manifest)

    def test_verify_receipt_skips_only_unchanged_files(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.json').write_bytes(b'original');receipt=root/'r/receipt.json'
            manifest=dict(sha='0'*40,repo='example/model',files=[dict(f='config.json',size=8,sha256=hashlib.sha256(b'original').hexdigest())])
            def run():
                out=io.StringIO()
                with contextlib.redirect_stdout(out): weights.verify(root,manifest,receipt)
                return out.getvalue()
            self.assertIn('SHA256 PASS',run());self.assertIn('SHA256 CACHED',run())
            (root/'config.json').write_bytes(b'corrupt!')        # same size, new mtime/ctime
            with self.assertRaises(ValueError): run()
            (root/'config.json').write_bytes(b'original')
            self.assertIn('SHA256 PASS',run())
            other=dict(manifest,sha='1'*40)                      # different manifest: no reuse
            self.assertNotEqual(weights.receipt_path(root,manifest),weights.receipt_path(root,other))
            receipt.write_text('not json');self.assertIn('SHA256 PASS',run())

    def test_verify_receipt_expires_after_seven_days(self):
        import contextlib, io, json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config.json').write_bytes(b'original');receipt=root/'r/receipt.json'
            manifest=dict(sha='0'*40,repo='example/model',files=[dict(f='config.json',size=8,sha256=hashlib.sha256(b'original').hexdigest())])
            def run(now):
                out=io.StringIO()
                with contextlib.redirect_stdout(out): weights.verify(root,manifest,receipt,now=now)
                return out.getvalue()
            t0=1_800_000_000.0
            self.assertIn('SHA256 PASS',run(t0))
            self.assertEqual(json.loads(receipt.read_text())['written'],t0)
            self.assertIn('SHA256 CACHED',run(t0+weights.RECEIPT_MAX_AGE_S-1))
            self.assertIn('SHA256 PASS',run(t0+weights.RECEIPT_MAX_AGE_S))      # expired: full pass, new receipt
            self.assertIn('SHA256 CACHED',run(t0+weights.RECEIPT_MAX_AGE_S+1))
            self.assertIn('SHA256 PASS',run(t0))                                 # receipt from the future: full pass
            self.assertEqual(weights.RECEIPT_MAX_AGE_S,7*24*3600)
            receipt.write_text(json.dumps({'config.json':weights.identity(root/'config.json')}))  # old format
            self.assertIn('SHA256 PASS',run(t0))

    def test_verify_rejects_path_escape(self):
        manifest=dict(sha='0'*40,files=[dict(f='../outside',size=0,sha256='0'*64)])
        with self.assertRaises(ValueError): weights.validate_manifest(manifest)

    def test_qeval_count_and_final_content(self):
        spec=importlib.util.spec_from_file_location('qeval_tasks',ROOT/'bench/qeval_tasks.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertEqual(len(mod.TASKS),75)
        source=(ROOT/'bench/qeval.py').read_text()
        self.assertIn('content = msg.get("content") or ""',source)

    def test_startup_order_and_inert_gates(self):
        for enabled in (False, True):
            code = self.startup_code(enabled)
            subprocess.run([sys.executable, '-S', '-c', code], check=True)

    def test_startup_registration_error_aborts(self):
        code = self.startup_code(True, fail=True)
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 78)
        self.assertIn('registration failure', result.stderr)

    def startup_code(self, enabled, fail=False):
        modules = ['glm_dsa_swa_pool','glm_fast_load','glm_full_mla','glm_window_memory','glm_draft_lowmem']
        env = dict(GLM_DSA_SWA_POOL='1',GLM_FAST_LOAD='1',GLM_FULL_MLA='triton',GLM_DRAFT_LOWMEM='1',
                   GLM_FLASH_SITECUSTOMIZE=str(ROOT/'overlay/swa-pool/sitecustomize.py')) if enabled else {}
        expected = modules if enabled else []
        return '\n'.join([
            'import sys,os,types', 'seen=[]',
            'os.environ.clear()', f'os.environ.update({env!r})',
            f'for name in {modules!r}:',
            ' m=types.ModuleType(name)',
            " m.register=lambda name=name: seen.append(name)",
            " sys.modules[name]=m",
            "if "+repr(fail)+": sys.modules['glm_full_mla'].register=lambda: (_ for _ in ()).throw(RuntimeError('registration failure'))",
            f'path={str(ROOT/"overlay/bringup/sitecustomize.py")!r}',
            "exec(compile(open(path).read(),path,'exec'),{'__file__':path})",
            f'assert seen=={expected!r},seen'])

    def test_native_mtp_source_and_mapping_drift(self):
        import glm_mtp_fix as fix
        import types
        source = Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/models/deepseek_mtp.py'
        fix.check_source(source)
        cls = type('MTP', (), {'__init__': lambda self, **kw: None})
        module = types.SimpleNamespace(__file__=str(source), DeepSeekMTP=cls)
        fix.install(module)
        initialized = cls.__init__
        fix.install(module)
        self.assertIs(initialized, cls.__init__)
        self.assertEqual(cls.packed_modules_mapping['fused_qkv_a_proj'], ['q_a_proj', 'kv_a_proj_with_mqa'])
        cls.packed_modules_mapping['fused_qkv_a_proj'] = ['wrong']
        with self.assertRaises(RuntimeError): fix.install(module)

    def test_native_mtp_starts_loader_without_swa_pool(self):
        modules = ['glm_fast_load', 'glm_mtp_fix', 'glm_full_mla', 'glm_window_memory', 'glm_prefill_switch']
        env = dict(GLM_DSA_SWA_POOL='0', GLM_DRAFT_LOWMEM='0', GLM_FAST_LOAD='1',
                   GLM_MTP_FIX='1', GLM_FULL_MLA='triton', GLM_W2_PREFILL_CONTROL='/cache/control.json')
        code = '\n'.join(['import sys,os,types', 'seen=[]', 'os.environ.clear()',
            f'os.environ.update({env!r})', f'for name in {modules!r}:',
            ' m=types.ModuleType(name)', ' m.register=lambda name=name: seen.append(name)',
            ' sys.modules[name]=m', f'path={str(ROOT/"overlay/bringup/sitecustomize.py")!r}',
            "exec(compile(open(path).read(),path,'exec'),{'__file__':path})", f'assert seen=={modules!r},seen'])
        subprocess.run([sys.executable, '-S', '-c', code], check=True)

    def test_all_python_parses(self):
        for path in ROOT.rglob('*.py'):
            if '.cpu-shim' not in path.parts:
                ast.parse(path.read_text(),filename=str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
