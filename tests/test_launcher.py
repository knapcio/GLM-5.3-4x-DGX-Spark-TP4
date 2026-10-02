"""Safety decisions tested with recorded-shaped samples and a fake SSH boundary."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import shlex
import json
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(RECIPE_ROOT=str(ROOT), RECIPE_HOSTS='s1 s2 s3 s4',
                  RECIPE_IPS='10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4', IMAGE='image:local',
                  MODEL_DIR='/models/target',DRAFT_DIR='/models/draft',NCCL_HOST_DIR='/nccl',OVERLAY_REMOTE='/runtime',
                  FABRIC_IFACE='eth1',IB_HCA='hca0,hca1')
spec=importlib.util.spec_from_file_location('cluster',ROOT/'scripts/cluster.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def samples():
    return [dict(mem=dict(MemAvailable=8*1048576,SwapTotal=2*1048576,SwapFree=2*1048576),
                 state=dict(Running=True,OOMKilled=False),logs='healthy') for _ in range(4)]


class Launcher(unittest.TestCase):
    def test_preflight_lock_check_is_head_only(self):
        # lock()/unlock() own Spark_01's marker; workers may hold unrelated files of that name.
        scripts = []
        def fake(rank, command, timeout=45):
            scripts.append((rank, command))
            raise RuntimeError('stop after first remote call')
        with patch.object(C, 'remote', fake), self.assertRaises(RuntimeError):
            C.preflight('glm53full-test')
        for rank, command in scripts:
            self.assertIn('HEAD=' + repr(rank == 0), shlex.split(command)[-1])
            self.assertIn("assert not (HEAD and p.home().joinpath('fleet_busy').exists())", shlex.split(command)[-1])

    def test_verify_command_binds_manifest_once_and_fits_one_argument(self):
        manifest=json.loads((ROOT/'manifests/target.json').read_text())
        for mode in ('cached','full'):
            command=C.verify_command('/models/target',manifest,mode)
            program=shlex.split(command)[-1]
            self.assertEqual(program.count(repr(manifest)),1)
            self.assertLess(len(command.encode()),C.ARG_MAX_STRLEN//2)   # 64 KiB of the 128 KiB single-argument limit
            self.assertEqual('receipt_path(ROOT_DIR, MANIFEST)' in program, mode=='cached')
        with self.assertRaisesRegex(ValueError, 'VERIFY_WEIGHTS'):
            C.verify_command('/m', manifest, 'fast')

    def test_verify_command_runs(self):
        import hashlib, subprocess, sys
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_bytes(b'original')
            manifest=dict(sha='0'*40,repo='example/model',files=[dict(f='config.json',size=8,sha256=hashlib.sha256(b'original').hexdigest())])
            for mode, home in (('full', d), ('cached', d)):
                program=shlex.split(C.verify_command(d,manifest,mode))[-1]
                out=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,env=dict(os.environ,HOME=home))
                self.assertEqual(out.returncode,0,out.stderr)
                self.assertIn('SHA256 PASS',out.stdout)
            program=shlex.split(C.verify_command(d,manifest,'cached'))[-1]
            again=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,env=dict(os.environ,HOME=d))
            self.assertIn('SHA256 CACHED',again.stdout)
            full=subprocess.run([sys.executable,'-c',shlex.split(C.verify_command(d,manifest,'full'))[-1]],capture_output=True,text=True,env=dict(os.environ,HOME=d))
            self.assertIn('SHA256 PASS',full.stdout)                  # VERIFY_WEIGHTS=full never uses the receipt

    def test_jit_prep_matches_model_container(self):
        # Same image, env (PYTHONPATH emptied) and /cache as the model container; no GPU, no network.
        import ast
        def env_of(cmd): return dict(cmd[i+1].split('=', 1) for i, t in enumerate(cmd) if t == '-e')
        for rank in range(4):
            with patch.dict(C.ENV, {key: 'v-' + key for key in C.PROFILE_KEYS}):
                prep, model = C.jit_prep_command(rank, 'glm53full-t'), C.docker_command(rank, 'glm53full-t')
            self.assertEqual(prep[:3], ['docker', 'run', '--rm'])
            self.assertEqual(prep[prep.index('--name')+1], f'glm53full-t-jitprep-r{rank}')
            self.assertEqual(prep[prep.index('--network')+1], 'none')
            self.assertNotIn('--gpus', prep)
            self.assertNotIn('-d', prep)
            self.assertIn('/runtime/cache:/cache', prep)
            self.assertIn('/nccl:/opt/nccl:ro', prep)
            self.assertNotIn('/runtime:/overlay:ro', prep)
            expected = env_of(model); expected['PYTHONPATH'] = ''
            self.assertEqual(env_of(prep), expected)
            self.assertEqual(prep[prep.index('--entrypoint')+2], C.IMAGES[rank])
            self.assertIn(C.IMAGES[rank], model)
            self.assertEqual(prep[prep.index('--entrypoint')+1:prep.index('--entrypoint')+2], ['timeout'])
            tail = prep[prep.index('--entrypoint')+3:]
            self.assertEqual(tail[:5], ['-k', '10', str(C.JIT_PREP_TIMEOUT_S), 'python3', '-c'])
            ast.parse(tail[5])
            self.assertIn(repr(C.JIT_PREP_MLA), tail[5])
            self.assertIn('gen_sampling_module()', tail[5])
            self.assertLess(len(C.shlex.join(prep).encode()), C.ARG_MAX_STRLEN // 4)
            self.assertNotIn('\n', C.shlex.join(prep))                         # one DRY line per command

    def _serve(self, prep_output):
        calls = []
        def fake(rank, command, timeout=45):
            calls.append((rank, command, timeout))
            if command.startswith('docker run --rm '):
                return prep_output(rank)
            if command.startswith('awk'):
                return str(118 * 1048576) if 'MemAvailable' in command else '0'
            return ''
        with tempfile.TemporaryDirectory() as d, patch.object(C, 'STATE', Path(d)/'state.json'), \
             patch.dict(C.ENV, {key: '0' for key in C.PROFILE_KEYS}), \
             patch.object(C, 'remote', fake), patch.object(C, 'preflight'), patch.object(C, 'lock'), \
             patch.object(C, 'prefill_control'), \
             patch.object(C, 'unlock') as unlock, patch.object(C, 'monitor') as monitor, \
             patch.object(C.subprocess, 'run'):
            try:
                C.serve('glm53full-t')
                error = None
            except RuntimeError as exc:
                error = exc
        return calls, error, unlock, monitor

    def test_serve_runs_jit_prep_before_compaction(self):
        ok = lambda rank: 'JIT PREP sampling /cache/x.so\nJIT PREP %s /cache/y.so' % C.JIT_PREP_MLA
        calls, error, unlock, monitor = self._serve(ok)
        self.assertIsNone(error)
        kinds = [('cache' if c.endswith('/cache') and c.startswith('mkdir') else 'prep' if c.startswith('docker run --rm ')
                  else 'compact' if 'spark-compact-mem' in c else 'launch' if c.startswith('docker run -d ') else None, r, t)
                 for r, c, t in calls]
        order = [k for k, _, _ in kinds if k]
        self.assertEqual(order, ['cache']*4 + ['prep']*4 + ['compact']*4 + ['launch']*4)
        self.assertEqual(sorted(r for k, r, _ in kinds if k == 'prep'), [0, 1, 2, 3])
        self.assertTrue(all(t == C.JIT_PREP_TIMEOUT_S + 60 for k, _, t in kinds if k == 'prep'))
        monitor.assert_called_once()
        unlock.assert_not_called()

    def test_jit_prep_failure_is_fail_closed(self):
        def bad(rank):
            if rank == 2:
                raise C.subprocess.CalledProcessError(1, 'ssh', output='Ninja build failed.')
            return 'JIT PREP sampling /cache/x.so\nJIT PREP %s /cache/y.so' % C.JIT_PREP_MLA
        for prep in (bad, lambda rank: 'JIT PREP sampling /cache/x.so'):
            calls, error, unlock, monitor = self._serve(prep)
            self.assertRegex(str(error), 'JIT prep')
            self.assertFalse(any('spark-compact-mem' in c or c.startswith('docker run -d ') for _, c, _ in calls))
            self.assertEqual(sum(c.startswith('docker run --rm ') for _, c, _ in calls), 4)   # every rank awaited
            unlock.assert_called_once()
            monitor.assert_not_called()

    def test_nccl_hash_one_or_four_in_host_order(self):
        a, b = 'a'*64, 'b'*64
        with patch.dict(C.ENV, NCCL_SHA256=a):
            self.assertEqual(C.nccl_hashes(), [a]*4)
        with patch.dict(C.ENV, NCCL_SHA256=','.join([a, b, a, a])):
            self.assertEqual(C.nccl_hashes()[1], b)
        for bad in (a+','+b, 'x'*64, a+',', ','.join([a]*5)):
            with patch.dict(C.ENV, NCCL_SHA256=bad), self.assertRaises(ValueError):
                C.nccl_hashes()

    def test_native_mtp_does_not_mount_external_draft(self):
        # The selected native model is loaded from /model, without RedHat assets.
        with patch.dict(C.ENV, {key: '0' for key in C.PROFILE_KEYS}):
            command = C.docker_command(0, 'native-test')
        self.assertFalse(C.uses_external_draft())
        self.assertNotIn('/models/draft:/draft:ro', command)
        self.assertIn('/models/target:/model:ro', command)

    def test_prefill_control_requires_all_rank_readback(self):
        calls=[]
        def fake(rank, command, timeout=45):
            calls.append(rank)
            return json.dumps(dict(schema=1, chunk=2048 if rank != 2 else 512, sequence=7))
        with patch.dict(C.ENV, GLM_W2_PREFILL_CONTROL='/cache/d2w2-prefill-control.json'), patch.object(C,'remote',fake):
            with self.assertRaisesRegex(RuntimeError,'readback disagreement'):
                C.prefill_control(2048,7)
        self.assertEqual(calls,[0,1,2])

    def test_prefill_busy_endpoint_never_mutates_control(self):
        metrics='vllm:num_requests_running{model="GLM"} 1\nvllm:num_requests_waiting{model="GLM"} 0'
        with patch.dict(C.ENV, GLM_W2_PREFILL_CONTROL='/cache/d2w2-prefill-control.json', GLM_W2_PREFILL_CHUNK='2048'), \
             patch.object(C,'remote',return_value=metrics), patch.object(C,'prefill_control') as mutate, \
             patch.object(C.time,'monotonic',side_effect=[0,1,31]), patch.object(C.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'did not drain'):
                C.select_serving_prefill()
            mutate.assert_not_called()

    def test_prefill_cap_selected_only_after_drain(self):
        metrics='vllm:num_requests_running{model="GLM"} 0\nvllm:num_requests_waiting{model="GLM"} 0'
        with patch.dict(C.ENV, GLM_W2_PREFILL_CONTROL='/cache/d2w2-prefill-control.json', GLM_W2_PREFILL_CHUNK='2048'), \
             patch.object(C,'remote',return_value=metrics), patch.object(C,'prefill_control') as mutate:
            C.select_serving_prefill()
            self.assertEqual(mutate.call_args.args[0],2048)
            self.assertGreater(mutate.call_args.args[1],0)

    def test_floor_stopped_rank_and_numeric_errors(self):
        for change in ('memory','stopped','oom','cuda','numeric'):
            s=samples()
            if change=='memory':s[2]['mem']['MemAvailable']=6*1048576-1
            if change=='stopped':s[2]['state']['Running']=False
            if change=='oom':s[2]['state']['OOMKilled']=True
            if change=='cuda':s[2]['logs']='CUDA error: illegal memory access'
            if change=='numeric':s[2]['logs']='worker output nan detected'
            with self.subTest(change=change),self.assertRaises(RuntimeError):C.inspect_samples(s,{}, {})

    def test_capture_floor(self):
        s = samples()
        for x in s: x['logs'] = 'Capturing CUDA graphs (FULL):  40%'
        C.capture_headroom(s, [int(9.5 * 1048576)] * 4)      # measured MTP K2 capture transient
        with self.assertRaisesRegex(RuntimeError, 'below 8 GiB'):
            C.capture_headroom(s, [int(7.9 * 1048576)] + [12 * 1048576] * 3)
        quiet = samples()
        C.capture_headroom(quiet, [7 * 1048576] * 4)           # not capturing: the 6 GiB floor applies

    def test_roce_ready_counters_are_not_errors(self):
        ready=('(Worker pid=124) INFO 10-01 19:31:16 [adapter.py:202] GLM_ROCE_READY rank=0 world=4 '
               '{"error_hca": 0, "error_peer": 0, "error_seq": 0, "writes_completed": 0}')
        self.assertFalse(C.fatal_line(ready))
        for bad in ('(Worker pid=7) ERROR 10-01 boom', '(Worker_TP1 pid=9) worker failed to start',
                    'RuntimeError: CUDA error: an illegal memory access', 'Traceback (most recent call last):',
                    'logits contain nan'):
            self.assertTrue(C.fatal_line(bad), bad)

    def test_swap_requires_three_samples_over_baseline(self):
        base={};growing={};s=samples();C.inspect_samples(s,base,growing)
        s[1]['mem']['SwapFree']-=64*1024
        C.inspect_samples(s,base,growing);C.inspect_samples(s,base,growing)
        with self.assertRaises(RuntimeError):C.inspect_samples(s,base,growing)

    def test_preexisting_swap_and_optional_warning(self):
        s=samples()
        s[0]['mem']['SwapFree']-=200*1024
        s[0]['logs']='WARNING import_utils optional DeepEP import'
        b={};g={}
        for _ in range(5):self.assertEqual(len(C.inspect_samples(s,b,g)),4)

    def test_stop_targets_only_owned_names(self):
        calls=[]
        def fake(rank,cmd,timeout=45):
            calls.append((rank,cmd))
            return 'false' if 'docker inspect' in cmd else ''
        with patch.object(C,'remote',fake):C.stop(dict(ctn='glm53full-owned',token='owned:123'))
        stopped=sorted((rank,c) for rank,c in calls if c.startswith('docker stop'))
        self.assertEqual(len(stopped),4)
        for rank,cmd in stopped:self.assertEqual(cmd,f'docker stop -t 60 glm53full-owned-r{rank}')
        self.assertTrue(any("p.read_text()=='owned:123'" in shlex.split(cmd)[2] for _,cmd in calls if cmd.startswith('python3 -c')))
        self.assertFalse(any('docker rm' in cmd for _,cmd in calls))

    def test_ssh_failure_retains_lock(self):
        calls=[]
        def fake(rank,cmd,timeout=45):
            calls.append(cmd)
            if rank==2:raise RuntimeError('SSH lost')
            return 'false'
        with patch.object(C,'remote',fake),self.assertRaises(RuntimeError):C.stop(dict(ctn='glm53full-owned',token='owned'))
        self.assertFalse(any('p.unlink' in cmd for cmd in calls))

    def test_boot_deadline_includes_admission(self):
        s=samples()
        for item in s:item['gpu']='0'
        def fake(rank,cmd,timeout=45):
            return '200' if cmd.startswith('curl') else json.dumps(s[rank])
        with tempfile.TemporaryDirectory() as directory, patch.object(C,'ROOT',Path(directory)), \
             patch.object(C,'remote',fake), patch.object(C.time,'monotonic',side_effect=[0,1,901]), \
             patch.object(C.time,'sleep'), patch.object(C,'stop') as stopped:
            with self.assertRaisesRegex(RuntimeError,'boot exceeded'):
                C.monitor(dict(ctn='glm53full-test',token='owned'),boot=True)
            stopped.assert_called_once()

    def test_progress_ignores_http_polls_and_static_metrics(self):
        a=samples();b=copy.deepcopy(a)
        a[0]['logs']='loaded shard 1\nINFO: "GET /health HTTP/1.1" 200'
        b[0]['logs']='loaded shard 1\nINFO: "GET /health HTTP/1.1" 503\nAvg generation throughput: 0'
        self.assertEqual(C.progress_fingerprint(a),C.progress_fingerprint(b))
        metrics='vllm:generation_tokens_total{model="GLM"} 16\nhttp_requests_total 10'
        self.assertEqual(C.progress_fingerprint(a,metrics),C.progress_fingerprint(b,metrics.replace('total 10','total 20')))
        self.assertNotEqual(C.progress_fingerprint(a,metrics),C.progress_fingerprint(a,metrics.replace('} 16','} 17')))

    def test_restart_and_preboot_swap_baseline_trip(self):
        s=samples();s[0]['restarts']=1
        with self.assertRaisesRegex(RuntimeError,'restarted'):C.inspect_samples(s,{}, {})
        s=samples();s[0]['mem']['SwapFree']-=64*1024
        base={0:0};growing={}
        C.inspect_samples(s,base,growing);C.inspect_samples(s,base,growing)
        with self.assertRaisesRegex(RuntimeError,'swap growth'):C.inspect_samples(s,base,growing)

    def test_lock_does_not_release_another_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            lock=Path(directory)/'fleet_busy';lock.write_text('another-owner')
            def fake(rank,command,timeout=45):
                with patch.object(Path,'home',return_value=Path(directory)):
                    exec(shlex.split(command)[2],{})
                return ''
            with patch.object(C,'remote',fake):
                C.unlock('owned')
                self.assertEqual(lock.read_text(),'another-owner')
                C.unlock('another-owner')
                self.assertFalse(lock.exists())


if __name__=='__main__':unittest.main(verbosity=2)
