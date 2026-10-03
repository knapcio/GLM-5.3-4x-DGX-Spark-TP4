"""Safety decisions tested with recorded-shaped samples and a fake SSH boundary."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import shlex
import json
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(RECIPE_ROOT=str(ROOT), RECIPE_HOSTS='s1 s2 s3 s4',
                  RECIPE_IPS='test-address-1 test-address-2 test-address-3 test-address-4', IMAGE='image:local',
                  MODEL_DIR='/models/target',DRAFT_DIR='/models/draft',NCCL_HOST_DIR='/nccl',OVERLAY_REMOTE='/runtime',
                  FABRIC_IFACE='eth1',IB_HCA='hca0,hca1')
spec=importlib.util.spec_from_file_location('cluster',ROOT/'scripts/cluster.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def samples():
    return [dict(mem=dict(MemAvailable=8*1048576,SwapTotal=2*1048576,SwapFree=2*1048576),
                 state=dict(Running=True,OOMKilled=False),logs='healthy') for _ in range(4)]


class Launcher(unittest.TestCase):
    def test_preflight_lock_check_is_head_only(self):
        # lock()/unlock() own NODE_1's marker; workers may hold unrelated files of that name.
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
            with patch.dict(C.ENV, {key: 'v-' + key for key in C.PROFILE_KEYS} | {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '0', 'GLM_MTP_KSTOP_CAPTURE_LAYOUT': 'm12', 'GLM_PAD_HYGIENE': '0'}):
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

    def _steady_stall_clock(self, busy_at, exit_at):
        clock = [0.0]
        s = samples()
        def fake(rank, cmd, timeout=45):
            if cmd.startswith('curl'):
                return 'vllm:generation_tokens_total 0\n'
            row = copy.deepcopy(s[rank]);row['gpu'] = '96' if clock[0] >= busy_at else '0'
            if clock[0] >= exit_at:row['state']['Running'] = False
            return json.dumps(row)
        def sleep(seconds):clock[0] += seconds
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'ROOT', Path(directory)), \
             patch.object(C, 'remote', fake), patch.object(C.time, 'monotonic', lambda: clock[0]), \
             patch.object(C.time, 'sleep', sleep), patch.object(C, 'stop') as stopped:
            with self.assertRaises(RuntimeError) as caught:
                C.monitor(dict(ctn='glm53full-test', token='owned'), boot=False)
            stopped.assert_called_once()
        return str(caught.exception), clock[0]

    def test_new_request_after_long_idle_has_fresh_stall_allowance(self):
        error, elapsed = self._steady_stall_clock(busy_at=300, exit_at=350)
        self.assertIn('exited', error)
        self.assertEqual(elapsed, 350)

    def test_continuous_busy_without_token_progress_still_aborts(self):
        error, elapsed = self._steady_stall_clock(busy_at=0, exit_at=1000)
        self.assertIn('GPU busy without progress', error)
        self.assertGreater(elapsed, 180)
        self.assertLessEqual(elapsed, 180 + C.SAMPLE_STEADY_S)

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


    def _clocked_monitor(self, deployment, ready_at=12.5, end_at=150, persist=None):
        """Monitor on a fake clock; ``persist(rank, cmd)`` answers the save helper's remote calls."""
        clock = [0.0]; events = []
        s = samples()
        for item in s: item['gpu'] = '0'
        def fake(rank, cmd, timeout=45):
            if 'pcache' in cmd or cmd.startswith('docker inspect -f {{.Image}}'):
                events.append(('persist', clock[0], rank, cmd))
                return persist(rank, cmd)
            if cmd.startswith('curl -s -m 3'):
                events.append(('health', clock[0]))
                return '200' if clock[0] >= ready_at else '000'
            if cmd.startswith('curl -fsS'):
                return 'vllm:generation_tokens_total 0'
            if rank == 0:
                events.append(('sample', clock[0]))
            if clock[0] >= end_at:
                s[rank]['state']['Running'] = False
            return json.dumps(s[rank])
        def sleep(seconds):
            self.assertGreater(seconds, 0)
            clock[0] += seconds
        out = []
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'ROOT', Path(directory)), \
             patch.object(C, 'remote', fake), patch.object(C.time, 'monotonic', lambda: clock[0]), \
             patch.object(C.time, 'sleep', sleep), patch.object(C, 'stop') as stopped, \
             patch.object(C, 'select_serving_prefill'), \
             patch.dict(C.ENV, {key: '0' for key in C.PROFILE_KEYS}), patch.dict(C.ENV, PERSISTENT_CACHE_DIR='/persist/glm'), \
             patch('builtins.print', lambda *a, **k: out.append(' '.join(map(str, a)))):
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                C.monitor(deployment, boot=True)
            stopped.assert_called_once()
        return events, out

    def test_health_every_second_and_admission_samples_every_five(self):
        events, out = self._clocked_monitor(dict(ctn='glm53full-test', token='owned'))
        health = [t for k, t, *_ in events if k == 'health']
        self.assertEqual(health, [float(t) for t in range(14)])              # 1 s polls until 200 at t=13
        self.assertIn('health 200 after 13.0 s', out)
        boot_samples = [t for k, t, *_ in events if k == 'sample' and t <= 73]
        self.assertEqual(boot_samples[:3], [0.0, 5.0, 10.0])
        self.assertTrue(all(b - a <= C.SAMPLE_BOOT_S for a, b in zip(boot_samples, boot_samples[1:])))
        admission = [line for line in out if line.startswith('Admission PASS')]
        self.assertEqual(admission, ['Admission PASS: >=8 GiB on all ranks for 60 s (73.0 s after start)'])
        steady = [t for k, t, *_ in events if k == 'sample' and t > 73]
        self.assertEqual(steady[:2], [83.0, 93.0])
        self.assertFalse(any(k == 'persist' for k, *_ in events))            # flag off: nothing saved

    def _blocking_save(self):
        release = threading.Event(); started = []
        def persist(rank, cmd):
            if cmd.startswith('docker inspect'):
                return 'sha256:img'
            if cmd.startswith('docker stop -t 10 '):
                release.set()                                                  # the helper container ends
                return ''
            started.append(rank)
            release.wait(30)
            raise C.subprocess.CalledProcessError(143, 'ssh', output='terminated')
        return persist, release, started

    def deployment_with_cache(self):
        return dict(ctn='glm53full-test', token='owned',
                    persistent_cache={str(r): dict(image='sha256:img', key='k'*32, seed='MISS') for r in range(4)})

    def test_save_runs_beside_monitoring_and_is_cancelled_on_abort(self):
        persist, release, started = self._blocking_save()
        events, out = self._clocked_monitor(self.deployment_with_cache(), end_at=150, persist=persist)
        samples_after = [t for k, t, *_ in events if k == 'sample' and t > 73]
        self.assertEqual(samples_after[:3], [83.0, 93.0, 103.0])                # watchdog kept sampling
        stops = sorted(e[2] for e in events if e[0] == 'persist' and e[3].startswith('docker stop -t 10 '))
        self.assertEqual(stops, [0, 1, 2, 3])                                   # helpers stopped on abort
        for e in events:
            if e[0] == 'persist' and e[3].startswith('docker stop'):
                self.assertIn(C.persist_save_name('glm53full-test', e[2]), e[3])
        self.assertTrue(release.is_set())

    def test_save_deadline_stops_helpers_and_serving_continues(self):
        persist, release, started = self._blocking_save()
        end = 73 + C.PERSIST_SAVE_DEADLINE_S + 60
        events, out = self._clocked_monitor(self.deployment_with_cache(), end_at=end, persist=persist)
        deadline = [l for l in out if 'deadline' in l]
        self.assertEqual(len(deadline), 1)
        stops = [e for e in events if e[0] == 'persist' and e[3].startswith('docker stop -t 10 ')]
        self.assertEqual(sorted(e[2] for e in stops), [0, 1, 2, 3])            # once, not again at abort
        self.assertTrue(all(73 + C.PERSIST_SAVE_DEADLINE_S < e[1] < end for e in stops))
        self.assertTrue(any(k == 'sample' and t > stops[0][1] for k, t, *_ in events))

    def test_save_results_and_failures_are_reported_not_raised(self):
        def ok(rank, cmd, timeout=45):
            if cmd.startswith('docker inspect'):
                return 'sha256:img' if rank != 3 else 'sha256:other'
            if rank == 2:
                raise RuntimeError('ssh lost during save')
            self.assertLessEqual(timeout, C.PERSIST_SAVE_DEADLINE_S)
            return 'PERSISTENT CACHE SAVED ' + 'k'*32 + ' 123'
        printed = []
        with patch.object(C, 'remote', ok), patch('builtins.print', lambda *a, **k: printed.append(' '.join(map(str, a)))), \
             patch.dict(C.ENV, {key: '0' for key in C.PROFILE_KEYS}), patch.dict(C.ENV, PERSISTENT_CACHE_DIR='/persist/glm'):
            saver = C.PersistentCacheSave(self.deployment_with_cache(), 0.0)
            for t in saver.threads: t.join(5)
            saver.poll(1.0); saver.poll(2.0)
            saver.cancel()                                                        # finished: no remote stop
        self.assertEqual(len(printed), 4)
        self.assertIn('SAVED ' + 'k'*32, printed[0])
        self.assertIn('SAVE FAILED (serving continues): ssh lost', printed[2])
        self.assertIn('SKIP image changed since seed', printed[3])

    def test_persistent_cache_flag_and_path_validation(self):
        with patch.dict(C.ENV, RECIPE_PERSISTENT_CACHE='yes', NCCL_SHA256='a'*64), self.assertRaises(ValueError):
            C.validate()
        for path in ('relative/x', '/runtime/cache', '/models/target/c', '/', '/a/../b', '/x;rm'):
            with self.subTest(path=path), patch.dict(C.ENV, RECIPE_PERSISTENT_CACHE='1', PERSISTENT_CACHE_DIR=path, NCCL_SHA256='a'*64), \
                 self.assertRaises(ValueError):
                C.validate()
        with patch.dict(C.ENV, RECIPE_PERSISTENT_CACHE='1', PERSISTENT_CACHE_DIR='/persist/glm', NCCL_SHA256='a'*64):
            C.validate()
        with patch.dict(C.ENV, RECIPE_PERSISTENT_CACHE='0', PERSISTENT_CACHE_DIR='relative', NCCL_SHA256='a'*64):
            C.validate()                                                       # inert when off

    def test_persist_commands_are_isolated_one_line_and_keyed(self):
        image = 'sha256:' + '1'*64
        with patch.dict(C.ENV, {key: 'v-' + key for key in C.PROFILE_KEYS} | {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '0', 'GLM_MTP_KSTOP_CAPTURE_LAYOUT': 'm12', 'GLM_PAD_HYGIENE': '0'}), \
                patch.dict(C.ENV, PERSISTENT_CACHE_DIR='/persist/glm'):
            seed, save = C.persist_command(2, 'glm53full-t', image, 'seed'), C.persist_command(2, 'glm53full-t', image, 'save', 'k'*32)
            parts = C.persist_parts(image)
        for cmd, phase in ((seed, 'seed'), (save, 'save')):
            self.assertEqual(cmd[:3], ['docker', 'run', '--rm'])
            self.assertEqual(cmd[cmd.index('--network')+1], 'none')
            self.assertNotIn('--gpus', cmd)
            self.assertIn('/runtime:/overlay:ro', cmd)
            self.assertIn('/persist/glm:/persist', cmd)
            self.assertIn('/runtime/cache:/cache' + (':ro' if phase == 'save' else ''), cmd)
            self.assertEqual(cmd[cmd.index('--entrypoint')+2], image)          # the keyed image, by ID
            self.assertEqual(cmd[-3:], ['python3', '/overlay/tools/glm_persistent_cache.py', phase])
            self.assertIn('PYTHONPATH=', cmd)
            self.assertNotIn('\n', C.shlex.join(cmd))
            self.assertIn('PCACHE_PARTS=' + json.dumps(parts, sort_keys=True), cmd)
        self.assertIn('PCACHE_KEY=' + 'k'*32, save)
        self.assertEqual(parts['image'], image)
        self.assertEqual(parts['arch'], {'FLASHINFER_CUDA_ARCH_LIST': 'v-FLASHINFER_CUDA_ARCH_LIST', 'TORCH_CUDA_ARCH_LIST': 'v-TORCH_CUDA_ARCH_LIST'})
        with self.assertRaises(ValueError):
            C.persist_command(0, 'glm53full-t', image, 'purge')

    def test_overlay_hash_tracks_the_rsynced_tree(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'overlay/overlay').mkdir(parents=True)
            (Path(d)/'overlay/overlay/a.py').write_text('a')
            with patch.object(C, 'ROOT', Path(d)):
                first = C.overlay_hash()
                (Path(d)/'overlay/overlay/__pycache__').mkdir()
                (Path(d)/'overlay/overlay/__pycache__/a.pyc').write_bytes(b'x')
                self.assertEqual(C.overlay_hash(), first)                     # rsync excludes __pycache__
                (Path(d)/'overlay/overlay/a.py').write_text('b')
                self.assertNotEqual(C.overlay_hash(), first)

    def test_persist_result_parsing(self):
        self.assertEqual(C.persist_result('noise\nPERSISTENT CACHE HIT ' + 'a'*32, 'seed'), ('HIT', 'a'*32))
        self.assertEqual(C.persist_result('PERSISTENT CACHE SAVED ' + 'b'*32 + ' 123', 'save'), ('SAVED', 'b'*32))
        for bad, phase in (('', 'seed'), ('PERSISTENT CACHE SAVED ' + 'a'*32, 'seed'), ('PERSISTENT CACHE HIT short', 'seed'),
                           ('PERSISTENT CACHE MISS ' + 'a'*32 + '\nPERSISTENT CACHE MISS ' + 'a'*32, 'seed')):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                C.persist_result(bad, phase)

    def test_serve_seeds_before_jit_prep_and_fails_closed(self):
        ok = lambda rank: 'JIT PREP sampling /cache/x.so\nJIT PREP %s /cache/y.so' % C.JIT_PREP_MLA
        def run(seed_out):
            calls = []
            def fake(rank, command, timeout=45):
                calls.append(command)
                if command.startswith('docker image inspect'):
                    return 'sha256:' + '2'*64
                if 'glm_persistent_cache.py seed' in command:
                    return seed_out(rank)
                if command.startswith('docker run --rm '):
                    return ok(rank)
                if command.startswith('awk'):
                    return str(118 * 1048576) if 'MemAvailable' in command else '0'
                return ''
            with tempfile.TemporaryDirectory() as d, patch.object(C, 'STATE', Path(d)/'state.json'), \
                 patch.dict(C.ENV, {key: '0' for key in C.PROFILE_KEYS}), \
                 patch.dict(C.ENV, RECIPE_PERSISTENT_CACHE='1', PERSISTENT_CACHE_DIR='/persist/glm'), \
                 patch.object(C, 'remote', fake), patch.object(C, 'preflight'), patch.object(C, 'lock'), \
                 patch.object(C, 'prefill_control'), patch.object(C, 'unlock'), \
                 patch.object(C, 'monitor') as monitor, patch.object(C.subprocess, 'run'):
                try:
                    C.serve('glm53full-t')
                    error, state = None, monitor.call_args.args[0]
                except RuntimeError as exc:
                    error, state = exc, None
            kinds = ['seed' if 'glm_persistent_cache.py seed' in c else 'prep' if c.startswith('docker run --rm ')
                     else 'compact' if 'spark-compact-mem' in c else 'launch' if c.startswith('docker run -d ') else None
                     for c in calls]
            return [k for k in kinds if k], error, state
        order, error, state = run(lambda rank: 'PERSISTENT CACHE %s %s' % ('HIT' if rank else 'MISS', 'c'*32))
        self.assertIsNone(error)
        self.assertEqual(order, ['seed']*4 + ['prep']*4 + ['compact']*4 + ['launch']*4)
        self.assertEqual(state['persistent_cache']['0'], dict(image='sha256:' + '2'*64, key='c'*32, seed='MISS'))
        self.assertEqual(state['persistent_cache']['3']['seed'], 'HIT')
        order, error, state = run(lambda rank: 'Traceback' if rank == 1 else 'PERSISTENT CACHE MISS ' + 'c'*32)
        self.assertRegex(str(error), 'Persistent cache seed failed')
        self.assertEqual(order, ['seed']*4)                                   # no prep, compaction or launch
SAVED = json.loads((ROOT/'profiles/serve-args.json').read_text())


def compilation(args):
    return json.loads(args[args.index('--compilation-config')+1])


class LaunchShape(unittest.TestCase):
    def test_defaults_reproduce_the_saved_vector(self):
        with patch.dict(C.ENV, {k: v for k, v in C.LAUNCH_DEFAULTS.items()}):
            self.assertEqual(C.launch_shape(SAVED), SAVED)
        with patch.dict(C.ENV, {}, clear=False):
            for k in C.LAUNCH_DEFAULTS:
                C.ENV.pop(k, None)
            self.assertEqual(C.launch_shape(SAVED), SAVED)

    def test_switches_are_not_container_variables(self):
        self.assertFalse(any(k.startswith('RECIPE_') for k in C.PROFILE_KEYS))
        profile = (ROOT/'profiles/current.env').read_text()
        for key, value in C.LAUNCH_DEFAULTS.items():
            self.assertIn(f"export {key}='{value}'", profile)       # released default is written down
            self.assertIn(f"export {key}='{value}'", (ROOT/'profiles/dspark-k3.env').read_text())

    def test_c4_pass2_graph_adds_width_four_only(self):
        with patch.dict(C.ENV, {'RECIPE_C4_PASS2_GRAPH': '1'}):
            args = C.launch_shape(SAVED)
        self.assertEqual(compilation(args)['cudagraph_capture_sizes'], [1, 3, 4, 6, 12])
        self.assertEqual(compilation(args)['max_cudagraph_capture_size'], 12)
        self.assertEqual(compilation(args)['cudagraph_mode'], 'FULL_DECODE_ONLY')
        changed = [i for i, (a, b) in enumerate(zip(SAVED, args)) if a != b]
        self.assertEqual((len(args), changed), (len(SAVED), [SAVED.index('--compilation-config')+1]))
        self.assertEqual(args[changed[0]], '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", '
                         '"cudagraph_capture_sizes": [1, 3, 4, 6, 12], "max_cudagraph_capture_size": 12}')

    def test_bad_values_and_other_profiles_refused(self):
        for bad in ('2', 'yes', '', 'on'):
            with patch.dict(C.ENV, {'RECIPE_C4_PASS2_GRAPH': bad}), self.assertRaises(ValueError):
                C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_C4_PASS2_GRAPH': '1', 'RECIPE_PROFILE': 'dspark-k3'}):
            with self.assertRaisesRegex(ValueError, 'native-mtp-k2 profile only'):
                C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_C4_PASS2_GRAPH': '0', 'RECIPE_PROFILE': 'dspark-k3'}):
            C.launch_switches()

    def test_context_uses_the_whole_two_gib_pool(self):
        self.assertEqual(C.KV_PIN_BYTES['0'] // C.KV_BLOCK_BYTES, 693)          # logged pool: 44,352 tokens
        self.assertEqual((693 - 1) * C.KV_BLOCK_TOKENS, 44288)                    # minus vLLM's null block
        self.assertEqual(C.KV_PIN_BYTES['1'], 821 * C.KV_BLOCK_BYTES)             # L1 pin, 2.369 GiB
        self.assertAlmostEqual(C.KV_PIN_BYTES['1'] / 2**30, 2.369, places=3)
        with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '44288'}):
            args = C.launch_shape(SAVED)
        changed = [i for i, (a, b) in enumerate(zip(SAVED, args)) if a != b]
        self.assertEqual(changed, [SAVED.index('--max-model-len') + 1])
        self.assertEqual(args[changed[0]], '44288')
        self.assertEqual(args[args.index('--kv-cache-memory-bytes') + 1], '2147483648')

    def test_kv_pin_l1_changes_only_the_pool(self):
        for length in ('32768', '44288'):
            with patch.dict(C.ENV, {'RECIPE_KV_PIN_L1': '1', 'RECIPE_MAX_MODEL_LEN': length}):
                args = C.launch_shape(SAVED)
            self.assertEqual(args[args.index('--kv-cache-memory-bytes') + 1], '2543549952')
            self.assertEqual(args[args.index('--max-model-len') + 1], length)
            self.assertEqual(sum(a != b for a, b in zip(SAVED, args)), 1 + (length != '32768'))

    def test_context_values_are_closed_and_must_fit(self):
        for bad in ('44352', '65536', '32k', ''):
            with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': bad}), self.assertRaises(ValueError):
                C.launch_switches()
        with patch.object(C, 'MAX_MODEL_LENS', C.MAX_MODEL_LENS + ('44352',)):
            with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '44352'}):
                with self.assertRaisesRegex(ValueError, 'does not fit'):
                    C.launch_switches()
            with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '44352', 'RECIPE_KV_PIN_L1': '1'}):
                C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '44288', 'RECIPE_PROFILE': 'dspark-k3'}):
            with self.assertRaisesRegex(ValueError, 'native-mtp-k2 profile only'):
                C.launch_switches()

    def test_nccl_no_ll128_adds_only_the_protocol_variable(self):
        env = {key: 'v-' + key for key in C.PROFILE_KEYS} | {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '0', 'GLM_MTP_KSTOP_CAPTURE_LAYOUT': 'm12', 'GLM_PAD_HYGIENE': '0'}
        with patch.dict(C.ENV, env):
            base = C.rank_env(0)
            with patch.dict(C.ENV, {'RECIPE_NCCL_NO_LL128': '1'}):
                switched, cmd = C.rank_env(0), C.docker_command(0, 'glm53full-t')
                prep = C.jit_prep_command(0, 'glm53full-t')
        self.assertNotIn('NCCL_PROTO', base)
        self.assertEqual({k: v for k, v in switched.items() if k not in base}, {'NCCL_PROTO': '^LL128'})
        self.assertEqual({k: v for k, v in switched.items() if k in base}, base)
        self.assertIn('NCCL_PROTO=^LL128', cmd)
        self.assertIn('NCCL_PROTO=^LL128', prep)                               # same environment in JIT prep
        self.assertEqual(C.launch_shape(SAVED), SAVED)                          # no argument change
        with patch.dict(C.ENV, dict(env, RECIPE_NCCL_NO_LL128='true')), self.assertRaises(ValueError):
            C.rank_env(0)

    KSTOP = {'GLM_MTP_KSTOP': '1', 'GLM_INDEXER_SHORTCUT': '0'}

    def test_kstop_selects_k3_graph_widths_1_4_16_and_a_544_block_pool(self):
        spec_at = SAVED.index('--speculative-config') + 1
        for c4, uniform in (('0', '0'), ('1', '0'), ('0', '1')):
            with patch.dict(C.ENV, dict(self.KSTOP, RECIPE_C4_PASS2_GRAPH=c4, GLM_MTP_KSTOP_UNIFORM_BATCH=uniform)):
                args = C.launch_shape(SAVED)
            self.assertEqual(json.loads(args[spec_at]), dict(json.loads(SAVED[spec_at]), num_speculative_tokens=3))
            self.assertEqual(compilation(args), dict(compilation(SAVED), cudagraph_capture_sizes=[1, 4, 16],
                                                     max_cudagraph_capture_size=16))
            self.assertEqual(compilation(args)['cudagraph_mode'], 'FULL_DECODE_ONLY')
            changed = [i for i, (a, b) in enumerate(zip(SAVED, args)) if a != b]
            self.assertEqual(changed, sorted([spec_at, SAVED.index('--compilation-config') + 1,
                                              SAVED.index('--kv-cache-memory-bytes') + 1]))
        self.assertEqual(args[args.index('--kv-cache-memory-bytes') + 1], str(544 * C.KV_BLOCK_BYTES))
        self.assertEqual(544 * C.KV_BLOCK_BYTES, 1685372928)
        for key, value in (('--max-num-seqs', '4'), ('--block-size', '64'), ('--max-model-len', '32768')):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--async-scheduling', args)
        with patch.dict(C.ENV, self.KSTOP):
            self.assertIn('544 KV blocks, max_model_len 32768, NCCL LL128 on', C.kstop_note())
        with patch.dict(C.ENV, {'GLM_MTP_KSTOP': '0'}):
            self.assertIsNone(C.kstop_note())

    def test_kstop_full_pool_needs_no_ll128_and_is_unmeasured(self):
        full = dict(self.KSTOP, RECIPE_MAX_MODEL_LEN='44224')
        with patch.dict(C.ENV, full), self.assertRaisesRegex(ValueError, 'requires RECIPE_NCCL_NO_LL128=1'):
            C.launch_switches()
        with patch.dict(C.ENV, dict(full, RECIPE_NCCL_NO_LL128='1')):
            args = C.launch_shape(SAVED)
            note = C.kstop_note()
        self.assertEqual(args[args.index('--kv-cache-memory-bytes') + 1], '2147483648')
        self.assertEqual(int(args[args.index('--kv-cache-memory-bytes') + 1]) // C.KV_BLOCK_BYTES, 693)
        self.assertEqual(args[args.index('--max-model-len') + 1], '44224')
        self.assertEqual(compilation(args)['cudagraph_capture_sizes'], [1, 4, 16])
        self.assertIn('693 KV blocks, max_model_len 44224, NCCL LL128 off', note)
        self.assertIn('UNMEASURED', note)
        # One 44,224-token request + K3 lookahead + vLLM's null block fills the 693-block pool exactly.
        self.assertEqual(-(-(44224 + 3) // 64) + 1, 693)

    def test_kstop_refuses_every_other_pool_and_context(self):
        for extra, message in ((dict(RECIPE_MAX_MODEL_LEN='44288'), 'RECIPE_MAX_MODEL_LEN 32768'),
                               (dict(RECIPE_MAX_MODEL_LEN='44288', RECIPE_NCCL_NO_LL128='1'), 'RECIPE_MAX_MODEL_LEN 32768'),
                               (dict(RECIPE_KV_PIN_L1='1'), 'RECIPE_KV_PIN_L1 must be 0'),
                               (dict(RECIPE_KV_PIN_L1='1', RECIPE_MAX_MODEL_LEN='44224', RECIPE_NCCL_NO_LL128='1'),
                                'RECIPE_KV_PIN_L1 must be 0')):
            with self.subTest(extra=extra), patch.dict(C.ENV, dict(self.KSTOP, **extra)):
                with self.assertRaisesRegex(ValueError, message):
                    C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '44224'}):
            with self.assertRaisesRegex(ValueError, 'requires GLM_MTP_KSTOP=1'):
                C.launch_switches()

    def test_no_switch_combination_launches_the_refused_kstop_layout(self):
        # The 1..16 / 693-block layout (predicted rank-0 7.82 GiB) and any K-stop layout with LL128 on at 693
        # blocks must be impossible; every launchable K-stop vector is one of the two admitted layouts.
        import itertools
        seen = set()
        for kstop, c4, ctx, l1, noll, uni in itertools.product('01', '01', C.MAX_MODEL_LENS, '01', '01', '01'):
            env = dict(GLM_MTP_KSTOP=kstop, GLM_INDEXER_SHORTCUT='0', RECIPE_C4_PASS2_GRAPH=c4, RECIPE_MAX_MODEL_LEN=ctx,
                       RECIPE_KV_PIN_L1=l1, RECIPE_NCCL_NO_LL128=noll, GLM_MTP_KSTOP_UNIFORM_BATCH=uni)
            with patch.dict(C.ENV, env):
                try:
                    args = C.launch_shape(SAVED)
                except ValueError:
                    continue
            sizes = compilation(args)['cudagraph_capture_sizes']
            blocks = int(args[args.index('--kv-cache-memory-bytes') + 1]) // C.KV_BLOCK_BYTES
            k = json.loads(args[args.index('--speculative-config') + 1])['num_speculative_tokens']
            self.assertNotEqual(sizes, list(range(1, 17)))
            if kstop == '1':
                self.assertEqual((k, sizes), (3, [1, 4, 16]))
                self.assertIn((blocks, ctx, noll), {(544, '32768', '0'), (544, '32768', '1'), (693, '44224', '1')})
                seen.add((blocks, ctx))
            else:
                self.assertEqual(k, 2)
        self.assertEqual(seen, {(544, '32768'), (693, '44224')})

    def test_uniform_batch_switch_is_closed_and_needs_kstop(self):
        for bad in ('yes', '2', 'on'):
            with patch.dict(C.ENV, dict(self.KSTOP, GLM_MTP_KSTOP_UNIFORM_BATCH=bad)), self.assertRaises(ValueError):
                C.launch_switches()
        with patch.dict(C.ENV, {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '1'}):
            with self.assertRaisesRegex(ValueError, 'requires GLM_MTP_KSTOP=1'):
                C.launch_switches()
        with patch.dict(C.ENV, dict(self.KSTOP, GLM_MTP_KSTOP_UNIFORM_BATCH='1')):
            self.assertIn('uniform batch K on', C.kstop_note())
            self.assertEqual(C.launch_shape(SAVED), C.launch_shape(SAVED))

    def test_uniform_k2_shapes_and_request_sizes_on_every_rank(self):
        env = {key: 'v-' + key for key in C.PROFILE_KEYS}
        env.update(self.KSTOP, GLM_PAD_HYGIENE='0', GLM_MTP_KSTOP_UNIFORM_BATCH='k2', GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse')
        with patch.dict(C.ENV, env):
            for rank in range(4):
                args = C.rank_args(rank)
                self.assertEqual(compilation(args)['cudagraph_capture_sizes'], [1, 4, 12, 16])
                self.assertEqual(compilation(args)['max_cudagraph_capture_size'], 16)
                self.assertEqual(json.loads(args[args.index('--speculative-config')+1])['num_speculative_tokens'], 3)
                self.assertEqual(C.rank_env(rank)['GLM_MTP_KSTOP_UNIFORM_BATCH'], 'k2')
                self.assertEqual(C.rank_env(rank)['GLM_MTP_KSTOP_CAPTURE_LAYOUT'], 'reuse')
            self.assertIn('uniform batch K k2, capture layout reuse', C.kstop_note())
        with patch.dict(C.ENV, {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': 'k2'}):
            with self.assertRaisesRegex(ValueError, 'requires GLM_MTP_KSTOP=1'):
                C.launch_switches()

    def test_pad_hygiene_validation_and_rank_propagation(self):
        for bad in ('', 'yes', '2'):
            with patch.dict(C.ENV, dict(self.KSTOP, GLM_PAD_HYGIENE=bad)), self.assertRaisesRegex(ValueError, 'GLM_PAD_HYGIENE'):
                C.launch_switches()
        with patch.dict(C.ENV, GLM_MTP_KSTOP='0', GLM_PAD_HYGIENE='1'), self.assertRaisesRegex(ValueError, 'requires GLM_MTP_KSTOP'):
            C.launch_switches()
        env = {key: 'v-' + key for key in C.PROFILE_KEYS}
        env.update(self.KSTOP, GLM_PAD_HYGIENE='1', GLM_MTP_KSTOP_UNIFORM_BATCH='k2', GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse')
        with patch.dict(C.ENV, env):
            for rank in range(4):
                self.assertEqual(C.rank_env(rank)['GLM_PAD_HYGIENE'], '1')

    def test_capture_layout_refuses_unknown_values(self):
        for bad in ('', '6', 'm6', 'yes'):
            with patch.dict(C.ENV, dict(self.KSTOP, GLM_MTP_KSTOP_CAPTURE_LAYOUT=bad)):
                with self.assertRaisesRegex(ValueError, 'CAPTURE_LAYOUT'):
                    C.launch_switches()
        for layout in ('m12', 'reuse'):
            with patch.dict(C.ENV, dict(self.KSTOP, GLM_MTP_KSTOP_UNIFORM_BATCH='k2', GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout)):
                self.assertEqual(compilation(C.rank_args(0))['cudagraph_capture_sizes'], [1,4,12,16])

    def test_kstop_refusals(self):
        with patch.dict(C.ENV, dict(self.KSTOP, GLM_INDEXER_SHORTCUT='1')):
            C.launch_switches()
        for bad in ('yes', '2', ''):
            with patch.dict(C.ENV, {'GLM_MTP_KSTOP': bad}), self.assertRaises(ValueError):
                C.launch_switches()
        with patch.dict(C.ENV, {'GLM_MTP_KSTOP': '1', 'GLM_INDEXER_SHORTCUT': '0', 'RECIPE_PROFILE': 'dspark-k3'}):
            with self.assertRaisesRegex(ValueError, 'native-mtp-k2 profile only'):
                C.launch_switches()

    def test_kstop_switch_and_control_reach_the_container(self):
        self.assertIn('GLM_MTP_KSTOP', C.PROFILE_KEYS)
        self.assertIn('GLM_MTP_KSTOP_CONTROL', C.PROFILE_KEYS)
        self.assertIn('GLM_MTP_KSTOP_UNIFORM_BATCH', C.PROFILE_KEYS)
        profile = (ROOT/'profiles/current.env').read_text()
        self.assertIn("export GLM_MTP_KSTOP='1'", profile)
        self.assertIn("export GLM_MTP_KSTOP_CONTROL='/overlay/kstop/control.json'", profile)
        self.assertIn("export GLM_MTP_KSTOP='0'", (ROOT/'profiles/dspark-k3.env').read_text())
        self.assertIn("export GLM_MTP_KSTOP_UNIFORM_BATCH='k2'", profile)
        self.assertIn("export GLM_MTP_KSTOP_UNIFORM_BATCH='0'", (ROOT/'profiles/dspark-k3.env').read_text())
        control = json.loads((ROOT/'overlay/kstop/control.json').read_text())
        self.assertEqual(control, dict(schema=1, mode='k3-stop', epoch=0, tau=0.74))

    def test_rank_vectors_carry_the_switch(self):
        with patch.dict(C.ENV, {'RECIPE_C4_PASS2_GRAPH': '1'}):
            for rank in range(4):
                self.assertEqual(compilation(C.rank_args(rank))['cudagraph_capture_sizes'], [1, 3, 4, 6, 12])


class StackDefaults(unittest.TestCase):
    """release/stack-1002b: prefix caching on, opt-in MLA plan skip keys, spec-sample/K-stop refusal, dispram hook."""
    KSTOP = {'GLM_MTP_KSTOP': '1', 'GLM_INDEXER_SHORTCUT': '0'}

    def setUp(self):
        env = {key: 'v-' + key for key in C.PROFILE_KEYS} | {'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '0', 'GLM_MTP_KSTOP_CAPTURE_LAYOUT': 'm12', 'GLM_PAD_HYGIENE': '0',
                                                              'GLM_SPEC_SAMPLE': '0', 'GLM_INDEXER_SHORTCUT': '0'}
        p = patch.dict(C.ENV, env); p.start(); self.addCleanup(p.stop)

    def test_prefix_caching_is_on_in_the_native_vector_only(self):
        self.assertEqual(SAVED.count('--enable-prefix-caching'), 1)
        self.assertNotIn('--no-enable-prefix-caching', SAVED)
        dspark = json.loads((ROOT/'profiles/dspark-k3-args.json').read_text())
        self.assertEqual(C.prefix_flag(dspark), '--no-enable-prefix-caching')
        for rank in (1, 2, 3):
            args = C.rank_args(rank)
            self.assertEqual(args[args.index('--enable-prefix-caching') - 1], '--headless')
        self.assertNotIn('--headless', C.rank_args(0))
        for bad in (SAVED + ['--no-enable-prefix-caching'], [a for a in SAVED if a != '--enable-prefix-caching']):
            with self.assertRaisesRegex(ValueError, 'exactly one prefix-caching flag'):
                C.prefix_flag(bad)

    def test_optional_skip_mla_keys_reach_the_container_only_when_set(self):
        base = C.rank_env(0)
        for key in C.OPTIONAL_KEYS:
            self.assertNotIn(key, base)
        with patch.dict(C.ENV, {'GLM_SKIP_MLA_PLAN': '0', 'GLM_SKIP_MLA_PLAN_AB_INIT': '0', 'VLLM_SERVER_DEV_MODE': '0'}):
            self.assertEqual(C.rank_env(0), base)
        with patch.dict(C.ENV, {'GLM_FULL_MLA': 'triton', 'GLM_SKIP_MLA_PLAN': '1'}):
            self.assertEqual(C.rank_env(0)['GLM_SKIP_MLA_PLAN'], '1')
        ab = {'GLM_FULL_MLA': 'triton', 'GLM_SKIP_MLA_PLAN': 'ab', 'VLLM_SERVER_DEV_MODE': '1'}
        with patch.dict(C.ENV, ab):
            env = C.rank_env(0)
            self.assertEqual((env['GLM_SKIP_MLA_PLAN'], env['VLLM_SERVER_DEV_MODE']), ('ab', '1'))
            self.assertIn('-e', C.docker_command(3, 'glm53full-t'))
            self.assertIn('GLM_SKIP_MLA_PLAN=ab', C.docker_command(3, 'glm53full-t'))
        for bad, message in (({'GLM_SKIP_MLA_PLAN': 'yes', 'GLM_FULL_MLA': 'triton'}, 'must be 0, 1 or ab'),
                             ({'GLM_SKIP_MLA_PLAN': '1', 'GLM_FULL_MLA': '0'}, 'needs GLM_FULL_MLA=triton'),
                             ({'GLM_SKIP_MLA_PLAN': 'ab', 'GLM_FULL_MLA': 'triton'}, 'only with GLM_SKIP_MLA_PLAN=ab'),
                             ({'VLLM_SERVER_DEV_MODE': '1'}, 'only with GLM_SKIP_MLA_PLAN=ab'),
                             (dict(ab, VLLM_SERVER_DEV_MODE='true'), 'must be 0 or 1'),
                             ({'GLM_SKIP_MLA_PLAN_AB_INIT': '1', 'GLM_SKIP_MLA_PLAN': '1', 'GLM_FULL_MLA': 'triton'},
                              'applies to GLM_SKIP_MLA_PLAN=ab only')):
            with self.subTest(bad=bad), patch.dict(C.ENV, bad), self.assertRaisesRegex(ValueError, message):
                C.launch_switches()

    def test_spec_sample_and_shortcut_with_kstop(self):
        with patch.dict(C.ENV, dict(self.KSTOP, GLM_SPEC_SAMPLE='1')):
            C.launch_switches()
        with patch.dict(C.ENV, dict(self.KSTOP, GLM_SPEC_SAMPLE='1', GLM_INDEXER_SHORTCUT='1')):
            C.launch_switches()
        with patch.dict(C.ENV, dict(self.KSTOP, GLM_SPEC_SAMPLE='0')):
            C.launch_switches()
        with patch.dict(C.ENV, {'GLM_MTP_KSTOP': '0', 'GLM_SPEC_SAMPLE': '1'}):
            C.launch_switches()

    def test_dispram_hook_is_off_and_refused_without_its_integration(self):
        self.assertIsNone(C.dispram())
        cmd = C.docker_command(0, 'glm53full-t')
        self.assertFalse(any('dispram' in token.lower() for token in cmd))
        for value in ('1', 'auto', 'require'):
            with self.subTest(value=value), patch.dict(C.ENV, {'RECIPE_DISPRAM': value}):
                with self.assertRaisesRegex(ValueError, 'needs the dispram integration'):
                    C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_DISPRAM': 'yes'}), self.assertRaisesRegex(ValueError, '0, 1/auto or require'):
            C.launch_switches()

    def test_dispram_hook_points_call_the_integration_when_present(self):
        import sys, types
        calls = []
        fake = types.ModuleType('dispram_recipe')
        fake.container_args = lambda env: (calls.append('args') or ['-v', '/d/run:/run/dispram'], {'GLM_DISPRAM_KV': 'auto'})
        fake.postcheck = lambda hosts, env: calls.append('postcheck')
        on = {'RECIPE_DISPRAM': 'require', 'GLM_MTP_KSTOP': '1', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '1',
              'RECIPE_MAX_MODEL_LEN': '66112'}
        fake.container_args = lambda env: (calls.append('args') or ['-v', '/d/run:/run/dispram'], {'GLM_DISPRAM_KV': 'require'})
        with patch.dict(sys.modules, {'dispram_recipe': fake}), patch.object(C, 'dispram_guard_present', lambda: True):
            with patch.dict(C.ENV, on):
                self.assertIs(C.dispram(), fake)
                cmd, prep, args = C.docker_command(0, 'glm53full-t'), C.jit_prep_command(0, 'glm53full-t'), C.rank_args(0)
            for bad, msg in (({'RECIPE_DISPRAM': '1'}, 'requires RECIPE_DISPRAM=require'),
                             ({'RECIPE_MAX_MODEL_LEN': '32768'}, 'RECIPE_MAX_MODEL_LEN=66112 only'),
                             ({'GLM_MTP_KSTOP': '0', 'GLM_MTP_KSTOP_UNIFORM_BATCH': '0', 'RECIPE_MAX_MODEL_LEN': '32768'}, 'K-stop layout only')):
                with self.subTest(bad=bad), patch.dict(C.ENV, dict(on, **bad)), self.assertRaisesRegex(ValueError, msg):
                    C.launch_switches()
        with patch.dict(C.ENV, {'RECIPE_MAX_MODEL_LEN': '66112'}), self.assertRaisesRegex(ValueError, 'requires GLM_MTP_KSTOP=1'):
            C.launch_switches()
        with patch.dict(sys.modules, {'dispram_recipe': fake}), patch.object(C, 'dispram_guard_present', lambda: False):
            with patch.dict(C.ENV, on), self.assertRaisesRegex(ValueError, 'copy guard'):
                C.launch_switches()
        self.assertIn('/d/run:/run/dispram', cmd)
        self.assertIn('GLM_DISPRAM_KV=require', cmd)
        self.assertIn('DISPRAM_GUARD_REQUIRED=1', cmd)
        self.assertTrue(any(t.startswith('LD_PRELOAD=') and t.endswith(':' + C.DISPRAM_GUARD) for t in cmd))
        self.assertFalse(any('dispram' in t.lower() for t in prep))                 # guard on engines only
        self.assertEqual((args[args.index('--max-model-len') + 1], args[args.index('--kv-cache-memory-bytes') + 1]),
                         ('66112', '1073741824'))
        self.assertEqual(C.KSTOP_DISPRAM_LAYOUT['blocks'], 1039)
        self.assertGreaterEqual(C.KSTOP_DISPRAM_LAYOUT['blocks'] - 1, -(-(66112 + 3) // 64))
        self.assertEqual(C.DISPRAM_HEAD_KV_BYTES, 1073741824)
        self.assertIn('RECIPE_DISPRAM=${RECIPE_DISPRAM:-require}', (ROOT/'.env.example').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
