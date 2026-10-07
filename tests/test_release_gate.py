"""Offline oracles for release evidence, fresh clones and fatal cleanup. No fleet calls."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bench'))
import release_probe as P
spec = importlib.util.spec_from_file_location('release_gate', ROOT/'scripts/release_gate.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)


def fixture(c=1):
    config = dict(port=8095, modelId='GLM-5.3', concurrencies=[c], maxTokens=256, promptType='prose')
    posted = dict(benchId='fresh-123', startedAt=100, sparkId='rank0')
    job = {**posted, 'completedAt': 200, 'status':'completed', 'error':None, 'config':config,
           'progress':dict(completedLevels=1,totalLevels=1),
           'results':[dict(concurrency=c, streamsOk=c, streamsFailed=0, totalCompletionTokens=c*256,
                totalDecodeTokens=c*255, meanDecodeTps=33., aggregateDecodeTps=33.*c,
                streams=[dict(index=i, completionTokens=256, decodeTokens=255, reasoningChunks=0, decodeTps=33.) for i in range(c)])]}
    return job, posted, config


def teacher():
    dist = {str(i): math.log(.04) for i in range(20)}
    return [dict(id='a', tokens=[1,2,3], prompt_lp=[None,dist,dist])]


def quality():
    return dict(results=[dict(id=str(i), category='code' if i < 55 else 'json', **{'pass':True}, why='') for i in range(75)])


class Evidence(unittest.TestCase):
    def test_full_sweep_all_integer_concurrencies(self):
        self.assertEqual(len(P.cells()),64)
        for kind in P.KINDS: self.assertEqual([c for k,c in P.cells() if k==kind],list(range(1,17)))

    def test_stream_identity_tokens_reasoning_and_job_pins(self):
        job, posted, config = fixture(16)
        self.assertEqual(P.validate_job(job,posted,config),[])
        changes = [('benchId','stale'),('sparkId','rank1'),('status','failed'),('completedAt',float('nan'))]
        for key,value in changes:
            d=copy.deepcopy(job);d[key]=value
            self.assertTrue(P.validate_job(d,posted,config))
        for key,value in [('reasoningChunks',1),('decodeTokens',254),('completionTokens',255),('index',17),('decodeTps',float('inf'))]:
            d=copy.deepcopy(job);d['results'][0]['streams'][0][key]=value
            self.assertTrue(P.validate_job(d,posted,config))
        d=copy.deepcopy(job);d['results'][0]['streams'].pop()
        self.assertTrue(P.validate_job(d,posted,config))

    def test_kl_equal_and_changed_distributions(self):
        a=teacher(); panel=[dict(id='a')]
        self.assertLess(P.compare_teacher(panel,a,a)['mean'],1e-10)
        b=copy.deepcopy(a);b[0]['prompt_lp'][1]['0']=math.log(.005)
        self.assertGreater(P.compare_teacher(panel,a,b)['mean'],0)

    def test_kl_refuses_missing_support_infinite_or_invalid_mass(self):
        a=teacher(); panel=[dict(id='a')]
        for value in (float('nan'),float('inf'),0.1):
            b=copy.deepcopy(a);b[0]['prompt_lp'][1]['0']=value
            with self.assertRaises(ValueError):P.compare_teacher(panel,a,b)
        b=copy.deepcopy(a);b[0]['prompt_lp'][1].pop('0')
        with self.assertRaises(ValueError):P.compare_teacher(panel,a,b)
        b=copy.deepcopy(a);b[0]['prompt_lp'][1]={str(i):math.log(.1) for i in range(20)}
        with self.assertRaises(ValueError):P.compare_teacher(panel,a,b)

    def test_kl_refuses_reordered_or_truncated_tokens(self):
        a=teacher(); panel=[dict(id='a')]
        for field,value in [('id','b'),('tokens',[1,2]),('tokens',[1,2,4]),('prompt_lp',[None])]:
            b=copy.deepcopy(a);b[0][field]=value
            with self.assertRaises(ValueError):P.compare_teacher(panel,a,b)
        with self.assertRaises(ValueError):P.compare_teacher(panel,a,[])

    def test_qeval_original_grid_three_repeats_and_request_failure(self):
        a=[quality() for _ in range(3)];b=copy.deepcopy(a)
        self.assertTrue(P.quality_gate(a,b)['passed'])
        for d in b:d['results'][0]['pass']=False
        self.assertFalse(P.quality_gate(a,b)['passed'])
        b=copy.deepcopy(a);b[1]['results'].pop()
        with self.assertRaises(ValueError):P.quality_gate(a,b)
        b=copy.deepcopy(a);b[1]['results'][0]['why']='request failed: timeout'
        with self.assertRaises(ValueError):P.quality_gate(a,b)
        with self.assertRaises(ValueError):P.quality_gate(a,a[:2])

    def test_teacher_no_greedy_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=P.Probes('http://fake',tmp,lambda:None)
            with patch.object(p,'request',return_value={'choices':[{'text':'greedy only'}]}):
                with self.assertRaises(ValueError):p.teacher([dict(id='a',text='public')],'A')

    def test_teacher_preserves_actual_ids_and_all_positions(self):
        a=teacher()[0]
        with tempfile.TemporaryDirectory() as tmp:
            p=P.Probes('http://fake',tmp,lambda:None)
            with patch.object(p,'request',return_value={'choices':[dict(prompt_token_ids=a['tokens'],prompt_logprobs=a['prompt_lp'])]}):
                rows=p.teacher([dict(id='a',text='public')],'A')
            self.assertEqual(rows,[a]);self.assertEqual(json.loads((Path(tmp)/'A.json').read_text()),rows)

    def test_default_plan_has_no_ssh_or_subprocess(self):
        with patch.object(G.subprocess,'Popen',side_effect=AssertionError('no fleet')):
            with patch.object(sys,'argv',['release_gate.py']), patch('builtins.print'):self.assertEqual(G.main(),0)

    def test_config_refuses_tags_and_keeps_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'config';cfg.write_text('IMAGE=mutable:tag\n')
            with self.assertRaises(ValueError):G.read_config(ROOT,cfg)
            cfg.write_text('HOSTS=(rank0 rank1 rank2 rank3)\nIPS=(192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4)\nFABRIC_IFACE=enp1s0f0np0\nIMAGES=('+' '.join(['sha256:'+'1'*64]*4)+')\nTOKEN=never-save-this\n')
            result=G.read_config(ROOT,cfg)
            self.assertEqual(len(result['RECIPE_IMAGES'].split()),4)
            self.assertNotIn('TOKEN',result)
            self.assertEqual(result['FABRIC_IFACE'],'enp1s0f0np0')

    @unittest.skipUnless(shutil.which("git"), "git absent in runtime image; clone boundary tested on workstation")
    def test_clone_excludes_untracked_state_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'src';src.mkdir()
            subprocess.run(['git','init','-q',str(src)],check=True)
            (src/'tracked').write_text('pinned')
            subprocess.run(['git','-C',str(src),'add','tracked'],check=True)
            subprocess.run(['git','-C',str(src),'-c','user.name=knapcio','-c','user.email=knapcio@gmail.com','commit','-qm','fixture'],check=True)
            (src/'.env').write_text('not cloned');(src/'state').mkdir()
            commit=G.clone(src,'HEAD',Path(tmp)/'dest')
            self.assertEqual(len(commit),40);self.assertFalse((Path(tmp)/'dest/.env').exists())
            self.assertFalse((Path(tmp)/'dest/state').exists())

    def test_deadline_and_dead_watchdog_abort(self):
        args=type('Args',(),dict(out=None,window_minutes=90))()
        with tempfile.TemporaryDirectory() as tmp:
            args.out=Path(tmp)/'run';g=G.Gate(args);g.work_end=0
            with self.assertRaises(TimeoutError):g.check()
            g.work_end=float('inf');g.proc=type('Dead',(),{'poll':lambda self:1})()
            with self.assertRaises(RuntimeError):g.check()

    def test_runtime_accepts_worker_pool_geometry_without_head_kv_line(self):
        args=type('Args',(),dict(out=None,window_minutes=90,base='http://fake'))()
        image='sha256:'+'1'*64
        config=dict(parallel_config={'tensor_parallel_size':4}, model_config={'max_model_len':32768},
                    scheduler_config={'max_num_seqs':4},speculative_config={'num_speculative_tokens':3},
                    cache_config={'kv_cache_memory_bytes':2147483648}, hf_token='do-not-save')
        with tempfile.TemporaryDirectory() as tmp:
            args.out=Path(tmp)/'run';g=G.Gate(args)
            repo=Path(tmp)/'candidate/checkout';(repo/'state').mkdir(parents=True);(repo/'profiles').mkdir()
            (repo/'profiles/serve-args.json').write_text((ROOT/'profiles/serve-args.json').read_text())
            (repo/'state/deployment.json').write_text(json.dumps(dict(ctn='glm53full-owned',hosts=['s1','s2','s3','s4'],images=[image]*4)))
            def ssh(command,**kw):
                rank=int(command[-1].split('-r')[-1].split()[0])
                if 'docker inspect' in command[-1]:
                    return json.dumps([dict(Image=image,State=dict(Running=True,OOMKilled=False),RestartCount=0,
                        Config=dict(Cmd=['serve'],Env=['GLM_FULL_MLA=triton','GLM_MLA_SPLIT_K=32','GLM_DSA_SWA_POOL=1','GLM_ROCE_ALLREDUCE=1','SECRET=no-save']))])
                return ('glm-full-mla: ARMED\nglm-dsa-swa-pool: registered\n'
                        'glm-window-memory: capture guard source pin PASS\ncapture-finished\n'
                        'glm-dsa-swa-pool: pool blocks=(654,1545)\n'
                        f'GLM_ROCE_READY rank={rank} world=4\nGLM_ROCE_ROUTE all_reduce\n'
                        + ('GPU KV cache size: 41,728 tokens' if rank==0 else ''))
            with patch.object(G.subprocess,'check_output',side_effect=ssh),patch.object(G,'http',side_effect=[{'vllm_config':config},{'data':[{'id':'GLM-5.3'}]}]):
                g.runtime(repo)
            self.assertTrue(g.report['checks']['candidate-runtime'])
            self.assertNotIn('hf_token',(repo.parent/'server-config.json').read_text())
            self.assertNotIn('SECRET',(repo.parent/'runtime.json').read_text())

    def test_sparkdash_refuses_stale_job_before_any_poll(self):
        job, posted, config = fixture();posted.update(status='running',config=config)
        with tempfile.TemporaryDirectory() as tmp:
            p=P.Probes('http://fake',tmp,lambda:None)
            stale={**posted,'benchId':'old'}
            with patch.object(P,'http',side_effect=[{'active':False,'last':{'benchId':'old'}},stale]) as h:
                with self.assertRaisesRegex(RuntimeError,'unique fresh'):p.sparkdash('http://dash')
                self.assertEqual(h.call_count,2)
            self.assertTrue((Path(tmp)/'sparkdash.json').exists())

    def test_prefill_never_sends_unsupported_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=P.Probes('http://fake',tmp,lambda:None)
            with patch.object(p,'request',side_effect=AssertionError('unsupported must never send')):
                self.assertFalse(p.prefill(2048))
            rows=json.loads((Path(tmp)/'prefill.json').read_text())
            self.assertEqual([r['requested_tokens'] for r in rows],list(P.SIZES))
            self.assertTrue(all(r['status']=='UNSUPPORTED_PROFILE' for r in rows))

    def test_cpu_tests_force_two_loader_readers(self):
        source=(ROOT/'tests/test_glm_fast_load.py').read_text()
        self.assertIn('os.environ["GLM_FAST_LOAD_THREADS"] = "2"',source)

    def test_complete_32k_window_holds_and_restores_reference(self):
        args=type('Args',(),dict(out=None,window_minutes=90,reference_repo=None,reference_rev='HEAD',
                  reference_config=None,config=None,candidate_rev='HEAD',base='http://fake',dash='http://dash',
                  aa_max=.01,kl_max=.01,min_prose_tps=50))()
        with tempfile.TemporaryDirectory() as tmp:
            args.out=Path(tmp)/'run';g=G.Gate(args)
            def prepare(name,*unused):
                repo=g.out/name/'checkout';repo.mkdir(parents=True);(repo/'.env').write_text('')
                g.report[name]=dict(config={'RECIPE_HOSTS':'s1 s2 s3 s4'},manifests={'target':'pinned','drafter':'pinned'},
                                    runtime='/runtime-'+name,max_model_len=32768)
                return repo
            class FakeProbes:
                def __init__(self,*a):pass
                def correctness(self):return True
                def request(self,*a):return {'tokens':list(range(24576))}
                def teacher(self,panel,*a):
                    t=teacher()[0]
                    return [dict(t,id=x['id']) for x in panel]
                def sparkdash(self,*a):return 51.
                def prefill(self,*a):return False
                def prefix_scan(self):return True
            with patch.object(g,'prepare',side_effect=prepare),patch.object(g,'boot') as boot, \
                 patch.object(g,'runtime'),patch.object(g,'stop') as stop,patch.object(g,'handoff',return_value={'reference':'guarded'}), \
                 patch.object(g,'qeval',return_value=[quality() for _ in range(3)]),patch.object(G,'Probes',FakeProbes),patch('builtins.print'):
                self.assertEqual(g.run(),2)
            self.assertEqual(g.report['status'],'HOLD');self.assertTrue(g.report['complete'])
            self.assertFalse(g.report['candidate_retained']);self.assertEqual(boot.call_count,3)
            self.assertEqual(boot.call_args.kwargs,{'recovery':True});self.assertEqual(stop.call_count,2)
            self.assertEqual(g.report['serving'],{'reference':'guarded'})
            self.assertFalse(g.report['checks']['prefill_full_scope'])

    def test_prepare_failure_does_not_restore_unqualified_reference(self):
        args=type('Args',(),dict(out=None,window_minutes=90,reference_repo=None,reference_rev='HEAD',
                  reference_config=None,config=None,candidate_rev='HEAD'))()
        with tempfile.TemporaryDirectory() as tmp:
            args.out=Path(tmp)/'run';g=G.Gate(args)
            with patch.object(g,'prepare',side_effect=RuntimeError('fleet not available')),patch.object(g,'boot') as boot,patch.object(g,'stop') as stop:
                self.assertEqual(g.run(),2)
                boot.assert_not_called();stop.assert_called_once()
            self.assertEqual(g.report['status'],'PARKED-IMPL')


if __name__=='__main__':unittest.main()
