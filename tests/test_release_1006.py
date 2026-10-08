# SPDX-License-Identifier: Apache-2.0
"""Release default-vector and fail-safe boundaries; no fleet or accelerator."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import release_1006_probe as P
import memrec as M
import boot_preflight as B


class Release(unittest.TestCase):
    def test_fresh_default_four_rank_vector_and_explicit_recent_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('scripts', 'profiles', 'overlay'):
                shutil.copytree(ROOT/name, root/name, ignore=shutil.ignore_patterns('__pycache__', '*.so'))
            for name in ('start.sh', '.env.example'):
                shutil.copy2(ROOT/name, root/name)
            (root/'overlay/guard/libdispram_copy_guard.so').touch()  # existence-only DRY fixture
            (root/'.env').write_text('HOSTS=(rank0 rank1 rank2 rank3)\nIPS=(192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4)\n'
                'FABRIC_IFACE=fabric0\nIB_HCA=hca0,hca1\nMODEL_DIR=/srv/model\nNCCL_HOST_DIR=/srv/nccl\n'
                'OVERLAY_REMOTE=/srv/runtime\nGLM_ATTN_NVFP4_DIR=/srv/attn\nGLM_NVFP4_MORE_DIR=/srv/more\n'
                'NCCL_SHA256='+64*'0'+'\n')
            def render():
                text = subprocess.check_output(['bash', str(root/'start.sh'), 'serve'], text=True,
                      env=dict(PATH=os.environ['PATH'], HOME=str(root), DRY='1'))
                return [shlex.split(line) for line in text.splitlines() if line.startswith('docker run -d')]
            vectors = render()
            self.assertEqual(len(vectors), 4)
            text = subprocess.check_output(['bash', str(root/'start.sh'), 'serve'], text=True,
                  env=dict(PATH=os.environ['PATH'], HOME=str(root), DRY='1'))
            self.assertIn('# Memory floors (GiB MemAvailable per rank): live 4.5, capture headroom 6, admission 6.5 for 60 s, '
                          'in-process pre-capture 7.5', text.splitlines())
            for cmd in vectors:
                env = dict(cmd[i+1].split('=', 1) for i,v in enumerate(cmd) if v == '-e')
                self.assertEqual(cmd[cmd.index('--max-model-len')+1], '262144')
                self.assertEqual(cmd[cmd.index('--kv-cache-memory-bytes')+1], '6318718976')
                self.assertEqual(env['GLM_FP4_POOL_BLOCKS'], '4135')
                self.assertEqual(env['GLM_FP4_MAX_MODEL_LEN'], '262144')
                self.assertEqual(env['GLM_PRECAPTURE_FLOOR_GIB'], '7.5')
                self.assertEqual(env['GLM_NVFP4_GROUPS'], 'attn,shared,dense,mtp')
                self.assertEqual(env['GLM_ATTN_NVFP4_DIR'], '/attn-nvfp4')
                self.assertEqual(env['GLM_LOADER'], 'fast')
                self.assertNotIn('GLM_FP4_RECENT_WINDOW', env)
                self.assertNotIn('VLLM_SERVER_DEV_MODE', env)
                self.assertNotIn('GLM_DRAFT_HEAD', env)
                self.assertEqual(env['GLM_DECODE_FAIR'], '1')
                self.assertEqual(env['GLM_DECODE_FAIR_CHUNK'], '4096')
                self.assertEqual(env['GLM_DECODE_FAIR_DECODE_STEPS'], '40')
                self.assertEqual(env['GLM_DECODE_FAIR_CONTROL'], '')
            with (root/'.env').open('a') as f:
                f.write('GLM_FP4_RECENT_WINDOW=2048\nGLM_FP4_RECENT_AB=1\nGLM_FP4_RECENT_INIT=0\nVLLM_SERVER_DEV_MODE=1\n')
            for cmd in render():
                env = dict(cmd[i+1].split('=', 1) for i,v in enumerate(cmd) if v == '-e')
                self.assertEqual(env['GLM_FP4_RECENT_WINDOW'], '2048')
                self.assertEqual(env['GLM_FP4_RECENT_INIT'], '0')
            # Plain .env assignments must reach the Python launcher too.
            with (root/'.env').open('a') as f:
                f.write('GLM_DECODE_FAIR=1\nGLM_DECODE_FAIR_CHUNK=1024\nGLM_DECODE_FAIR_CONTROL=/cache/decode-fair.json\n')
            for cmd in render():
                env = dict(cmd[i+1].split('=', 1) for i,v in enumerate(cmd) if v == '-e')
                self.assertEqual(env['GLM_DECODE_FAIR'], '1')
                self.assertEqual(env['GLM_DECODE_FAIR_CHUNK'], '1024')
                self.assertEqual(env['GLM_DECODE_FAIR_CONTROL'], '/cache/decode-fair.json')

    def test_probe_geometry_262144_shared_pool_and_single_max(self):
        g = P.geometry(6318718976, 262144)
        self.assertEqual(g['layout']['blocks'], 4135)
        self.assertEqual(g['c4_prompt_tokens'], 65024)
        self.assertLessEqual(g['c4_required_blocks'], 4135)
        with self.assertRaises(ValueError): P.geometry(6318718976, 264320)   # beyond packed capacity
        g = P.geometry(5876219904, 248320)
        self.assertEqual(g['layout']['blocks'], 3919)

    def test_memrec_parsers_and_rate_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'pressure').mkdir()
            (root/'meminfo').write_text('MemTotal: 100 kB\nMemFree: 10 kB\nMemAvailable: 20 kB\nCached: 5 kB\n'
                                        'SwapTotal: 8 kB\nSwapFree: 6 kB\n')
            (root/'pressure/memory').write_text('some avg10=1.50 avg60=0.00 avg300=0.00 total=7\n'
                                                'full avg10=0.25 avg60=0.00 avg300=0.00 total=3\n')
            # 4 KiB pages: order 9 = 2 MiB, order 11 = 8 MiB.
            (root/'buddyinfo').write_text('Node 0, zone   Normal ' + ' '.join(['1']*9 + ['2', '0', '1']) + '\n')
            row = M.sample(0.1, root)
            self.assertEqual((row['a'], row['f'], row['c'], row['sw'], row['ps'], row['pf']), (20, 10, 5, 2, 7, 3))
            self.assertEqual((row['s10'], row['f10'], row['hi']), (1.5, 0.25, 11))
            if M.PAGE == 4096:
                self.assertEqual(row['b2'], 2*(2<<20) + (8<<20))
                self.assertEqual(row['b8'], 8<<20)
            (root/'rate').write_text('0.001'); self.assertEqual(M.read_rate(root/'rate', 1.0), 0.05)
            (root/'rate').write_text('bad'); self.assertEqual(M.read_rate(root/'rate', 1.0), 1.0)

    def test_preflight_failure_or_bad_receipt_never_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);headers=root/'headers.json';headers.write_text('{}')
            with patch.object(B.subprocess, 'run', side_effect=TimeoutError('local deadline')):
                result=B.advisory(root,root/'dry',root/'out',headers)
            self.assertEqual(result['status'], 'WARNING')
            with patch.object(B.subprocess, 'run', return_value=type('Run', (), {'returncode':0})()):
                (root/'out').write_text('{broken')
                self.assertEqual(B.advisory(root,root/'dry',root/'out',headers)['status'], 'WARNING')

    def test_guard_deadline_stops_owned_boot_and_handoff_keeps_it(self):
        class Monitor:
            def __init__(self, *args): self.minimum={i:9. for i in range(4)}
            def start(self): pass
            def check(self): pass
            def ready(self): return True
            def close(self): pass
        for handoff in (False, True):
            with self.subTest(handoff=handoff), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'state').mkdir()
                (root/'state/deployment.json').write_text('{"ctn":"owned","token":"owned-token"}')
                h=root/'handoff.json'
                if handoff: h.write_text('{"ctn":"owned","token":"owned-token","watch_active":true,"heartbeat_verified":true}')
                args=['probe','guard','--execute','--boot','owned','--out',str(root/'out'),
                      '--seconds','1','--handoff-file',str(h)]
                with patch.object(P,'ROOT',root), patch.object(P.S,'Monitor',Monitor), \
                     patch.object(P.signal,'signal'), patch.object(P.subprocess,'run') as stop, \
                     patch.dict(os.environ,INVOCATION_ID='unit',RECIPE_HOSTS='r0 r1 r2 r3'), \
                     patch.object(sys,'argv',args), patch.object(P.time,'monotonic',side_effect=[0,0 if handoff else 2,3]):
                    if handoff:
                        self.assertEqual(P.main(),0);stop.assert_not_called()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'without verified serving handoff'): P.main()
                        self.assertEqual(stop.call_count,1)
                self.assertEqual(json.loads((root/'out/RESULT.json').read_text())['status'], 'PASS' if handoff else 'FAILED')

    def test_probe_default_is_offline_and_execute_requires_owner(self):
        with patch.object(P.S, 'http', side_effect=AssertionError('network')):
            with patch.object(sys, 'argv', ['probe', 'stress', '--boot', 'fixture', '--out', '/unused']), patch('builtins.print'):
                self.assertEqual(P.main(), 0)
        with patch.dict(os.environ, {'INVOCATION_ID':'','RECIPE_HOSTS':''}), \
             patch.object(sys, 'argv', ['probe', 'stress', '--execute', '--boot', 'fixture', '--out', '/unused']):
            with self.assertRaises(SystemExit): P.main()


if __name__ == '__main__':
    unittest.main()
