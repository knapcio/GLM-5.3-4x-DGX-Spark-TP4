# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic safetensors + real native loader and INIT fallback checks."""
import gc
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
import weakref
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'tests')]
import glm_draft_ehproj as eh
import glm_draft_head as dh
from test_draft_head_init import fixture, completed_vote, COMPILATION


class RollbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from boot_preflight import fake_hardware, Report
        sys.path.insert(0,os.environ['GLM_IMAGE_SRC'])
        fake_hardware(Report(),0)
        from vllm.config.load import LoadConfig
        cls.load_config=LoadConfig(load_format='safetensors')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'model-00001-of-00001.safetensors'
        self.name='model.layers.78.eh_proj.weight'
        self.weight=(torch.arange(256*512).reshape(256,512)%43-21).bfloat16()/64
        self.weight[0,0]=-0.0
        save_file({self.name:self.weight,'unrelated.weight':torch.ones(7)},str(self.path))

    def runner(self,fault='mismatch'):
        r=fixture(fault)
        n,k=self.weight.shape
        native=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16,device='meta')
        native.weight=torch.nn.Parameter(torch.empty((n,k),dtype=torch.bfloat16),False)
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        default_weight_loader(native.weight,load_file(str(self.path))[self.name])
        r.speculator.draft_model_config=NS(model=self.temp.name,revision=None)
        r.load_config=self.load_config
        source=eh.checkpoint_source(r,native,'78')
        bank=torch.nn.Module()
        bank.weight=torch.nn.Parameter(torch.zeros(3,dtype=torch.int32),False)
        bank.weight_scale=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16),False)
        bank.workspace=torch.zeros(48,dtype=torch.int32)
        # Deliberately lossy ON projection; OFF must use the checkpoint bytes.
        bank.forward=lambda x:torch.zeros((*x.shape[:-1],256),dtype=x.dtype)
        r.speculator.model.model.layers['78'].eh_proj=bank
        r._draft_ehproj=bank;r._draft_ehproj_source=source;r._draft_ehproj_layer='78'
        r._draft_ehproj_ready=False
        r._draft_ehproj_receipt={'total':dh.storage_bytes(bank)}
        r.model_memory_usage=1000
        self.native=native
        self.old_parameters=[weakref.ref(p) for p in bank.parameters()]
        self.workspace=weakref.ref(bank.workspace)
        return r

    def boot(self,r,vote=completed_vote,mem=lambda:8<<30,outer_installer=None):
        capture=r.speculator.capture
        events=[]
        def observed_capture():
            events.append((r._draft_head.on,type(r._draft_ehproj)))
            result=capture()
            if not r._draft_head.on and tuple(r._draft_ehproj.weight.shape)==(256,512):
                # OFF graphs capture the actual restored projection and native
                # head, rather than the ON mismatch injector in the fixture.
                def native_replay():
                    sp=r.speculator
                    projected=r._draft_ehproj(torch.cat((sp.hidden_states,sp.hidden_states),-1))
                    logits=r.model.lm_head.quant_method.apply(r.model.lm_head,projected)
                    sp.hidden_states.copy_(projected)
                    sp.draft_tokens.fill_(int(logits.argmax(-1)[0]))
                    sp._kstop.confidence.fill_(.9)
                for manager in dh.managers(r):
                    for graph in manager.graphs.values():graph.replay=native_replay
            return result
        r.speculator.capture=observed_capture
        cls=type('GPUModelRunner',(),dict(load_model=lambda s:None,
            capture_model=lambda s:'captured' if not s.speculator.capture() else None))
        dh.install_runner(NS(GPUModelRunner=cls))
        eh.install_runner(NS(GPUModelRunner=cls))
        if outer_installer is not None:outer_installer(NS(GPUModelRunner=cls))
        with patch.dict(sys.modules,{'vllm.config.compilation':COMPILATION}), \
             patch.object(dh,'agree',vote),patch.object(dh,'memory',mem), \
             patch.object(torch.cuda,'synchronize',lambda:None), \
             patch.object(torch.cuda,'empty_cache',lambda:None), \
             patch.object(dh,'renew_draft_pool',lambda r:None):
            result=cls.capture_model(r)
        self.events=events
        return result

    def assert_native(self,r):
        bank=r._draft_ehproj
        self.assertIs(bank,r.speculator.model.model.layers['78'].eh_proj)
        self.assertIs(type(bank),torch.nn.Linear)
        self.assertEqual(bank.weight.detach().view(torch.uint8).numpy().tobytes(),
                         self.native.weight.detach().view(torch.uint8).numpy().tobytes())
        receipt=r._draft_ehproj_receipt['ehproj_rollback']
        self.assertTrue(receipt['qualified'])
        self.assertEqual(receipt['sha256'],receipt['expected_sha256'])
        self.assertEqual(receipt['bytes'],self.weight.numel()*2)
        self.assertFalse(r._draft_head.on);self.assertFalse(r._draft_head_ready)
        self.assertFalse(r._draft_ehproj_ready);self.assertIsNone(r.speculator._kstop.bad)
        self.assertFalse(hasattr(bank,'workspace'));self.assertFalse(hasattr(bank,'weight_scale'))
        gc.collect()
        self.assertTrue(all(p() is None for p in self.old_parameters))
        self.assertIsNone(self.workspace())
        before=r.speculator.hidden_states.clone()
        expected=self.native(torch.cat((before,before),-1))
        next(iter(dh.managers(r)[0].graphs.values())).replay()
        self.assertEqual(r.speculator.hidden_states.view(torch.uint8).numpy().tobytes(),
                         expected.view(torch.uint8).numpy().tobytes())
        expected_logits=r.model.lm_head.quant_method.apply(r.model.lm_head,expected)
        self.assertTrue(bool((r.speculator.draft_tokens==expected_logits.argmax(-1)[0]).all()))
        # Same BF16 projection and native OFF head/logits path as ehproj unset.
        for m in (1,4,12,16):
            x=torch.arange(m*512).reshape(m,512).remainder(31).bfloat16()/16
            actual=bank(x);expected=self.native(x)
            self.assertEqual(actual.view(torch.uint8).numpy().tobytes(),expected.view(torch.uint8).numpy().tobytes())
            a=r._draft_head.quant_method.apply(r._draft_head,actual)
            b=r.model.lm_head.quant_method.apply(r.model.lm_head,expected)
            self.assertEqual(a.view(torch.uint8).numpy().tobytes(),b.view(torch.uint8).numpy().tobytes())

    def test_refusal_restores_native_before_capture_and_qualifies_same_off_checks(self):
        r=self.runner();identity=id(r._draft_ehproj);terms=[]
        original=dh.qualification_term
        def term(*a,**kw):terms.append(a[0]);return original(*a,**kw)
        with patch.object(dh,'qualification_term',term):self.assertEqual(self.boot(r),'captured')
        self.assertEqual(id(r._draft_ehproj),identity)
        self.assertEqual(self.events[-1],(False,torch.nn.Linear))
        self.assertLess(terms.index('ehproj_rollback_loaded'),terms.index('initial_fallback_off'))
        self.assert_native(r)

    def test_rowselect_refusal_disables_before_bf16_recapture(self):
        r=self.runner();r.speculator._glm_rowselect={'disabled':False,'ready':True}
        def disable(sp,refusal):sp._glm_rowselect.update(disabled=True,ready=False)
        with patch.dict(sys.modules,{'glm_mtp_rowselect':NS(disable=disable)}):self.boot(r)
        self.assertTrue(r.speculator._glm_rowselect['disabled']);self.assert_native(r)

    def test_second_rollback_attempt_is_hard_refusal(self):
        r=self.runner();self.boot(r)
        with patch.object(dh,'agree',completed_vote),patch.object(dh,'memory',lambda:8<<30), \
             patch.object(torch.cuda,'synchronize',lambda:None), \
             patch.object(torch.cuda,'empty_cache',lambda:None):
            with self.assertRaises(dh.VoteRefused):
                dh.initial_fallback(r,dh.VoteRefused({'second-attempt':True}))
        self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])
        self.assertIsNotNone(r.speculator._kstop.bad)

    def test_outer_rowselect_refusal_after_fp8_ready_restores_native_and_clears_ready(self):
        import glm_mtp_rowselect as rs
        import glm_draft_head_qual as qual
        r=self.runner(None);r.speculator._glm_rowselect={'disabled':False,'ready':False}
        def disable(sp,refusal):sp._glm_rowselect.update(disabled=True,ready=False)
        ready_at_refusal=[]
        def vote(payload,valid):
            if 'rowselect_descriptors' in payload:ready_at_refusal.append(r._draft_ehproj_ready)
            completed_vote(payload,valid)
        with patch.object(rs,'disable',disable),patch.object(rs,'capture_logits',lambda sp:None), \
             patch.object(rs,'account',lambda sp:[{'descriptor':'synthetic'}]), \
             patch.object(qual,'qualify',return_value={'cases':[{'native_fixture':True}]}):
            self.boot(r,vote=vote,outer_installer=rs.install_runner)
        self.assertEqual(ready_at_refusal,[True])
        self.assertTrue(r.speculator._glm_rowselect['disabled'])
        self.assert_native(r)

    def test_memory_floor_refuses_before_open_or_allocation(self):
        r=self.runner()
        with patch('safetensors.safe_open',side_effect=AssertionError('payload opened')), \
             patch.object(torch,'empty',side_effect=AssertionError('allocated')):
            with self.assertRaises(dh.VoteRefused):self.boot(r,mem=lambda:eh.FLOOR_BYTES)
        self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])
        self.assertIsNot(type(r._draft_ehproj),torch.nn.Linear)
        self.assertIsNotNone(r.speculator._kstop.bad)

    def test_vote_disagreement_before_commit_stays_hard_refusal(self):
        for stage in ('ehproj_rollback_admission','ehproj_rollback_loaded',
                      'ehproj_rollback_restored','initial_fallback_off',
                      'ehproj_rollback','ehproj_native_fallback'):
            with self.subTest(stage=stage):
                r=self.runner()
                def vote(payload,valid):
                    if stage in payload:raise dh.VoteRefused({'payload_mismatch':True})
                    completed_vote(payload,valid)
                with self.assertRaises(dh.VoteRefused):self.boot(r,vote)
                self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])
                self.assertFalse(r._draft_ehproj_ready);self.assertIsNotNone(r.speculator._kstop.bad)
                if stage in ('ehproj_rollback_admission','ehproj_rollback_loaded'):
                    self.assertIsNot(type(r._draft_ehproj),torch.nn.Linear)

    def test_post_allocation_memory_or_native_capture_failure_remains_poisoned(self):
        for fault in ('post-load-memory','capture'):
            with self.subTest(fault=fault):
                r=self.runner()
                def mem():
                    receipt=r._draft_ehproj_receipt.get('ehproj_rollback',{})
                    return eh.FLOOR_BYTES-1 if fault=='post-load-memory' and receipt.get('sha256') else 8<<30
                capture=r.speculator.capture
                def fail_capture():
                    if not r._draft_head.on:raise RuntimeError('native OFF capture failed')
                    return capture()
                r.speculator.capture=fail_capture
                with self.assertRaises(dh.VoteRefused):self.boot(r,mem=mem)
                self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])
                self.assertIsNotNone(r.speculator._kstop.bad)

    def test_changed_missing_wrong_dtype_checkpoint_fail_closed(self):
        for fault in ('changed','missing','dtype'):
            with self.subTest(fault=fault):
                save_file({self.name:self.weight},str(self.path));r=self.runner()
                if fault=='missing':self.path.unlink()
                else:save_file({self.name:self.weight+1 if fault=='changed' else self.weight.float()},str(self.path))
                with self.assertRaises(dh.VoteRefused):self.boot(r)
                self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])
                self.assertIsNot(type(r._draft_ehproj),torch.nn.Linear)
                self.assertIsNotNone(r.speculator._kstop.bad)

    def test_no_refusal_keeps_only_fp8_and_scalar_source(self):
        r=self.runner(None)
        self.boot(r)
        self.assertTrue(r._draft_ehproj_ready)
        self.assertNotIn('ehproj_rollback',r._draft_ehproj_receipt)
        self.assertTrue(all(not isinstance(v,(torch.Tensor,torch.nn.Module)) for v in r._draft_ehproj_source.values()))

    def test_full_replicated_geometry_hash_and_reload(self):
        self.weight=torch.full((6144,12288),.02,dtype=torch.bfloat16)
        save_file({self.name:self.weight,'unrelated.weight':torch.ones(7)},str(self.path))
        r=self.runner()
        self.boot(r)
        self.assertEqual(r._draft_ehproj_receipt['ehproj_rollback']['bytes'],150994944)
        self.assertEqual(eh.tensor_sha(r._draft_ehproj.weight),eh.tensor_sha(self.native.weight))
        self.assertTrue(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])

    def test_production_prepare_retains_no_native_bf16_module_or_parameter(self):
        from test_draft_ehproj import fixture as projection_fixture, bank
        self.weight=torch.full((6144,12288),.02,dtype=torch.bfloat16)
        save_file({self.name:self.weight},str(self.path))
        r,original=projection_fixture()
        original.weight=torch.nn.Parameter(self.weight,False)
        original_ref=weakref.ref(original);weight_ref=weakref.ref(original.weight)
        r.load_config=self.load_config
        r.speculator.draft_model_config=NS(model=self.temp.name,revision=None)
        eh.prepare(r,factory=bank,vote=completed_vote)
        self.weight=None
        del original
        gc.collect()
        self.assertIsNone(original_ref());self.assertIsNone(weight_ref())
        self.assertEqual(r._draft_ehproj_source['bytes'],150994944)
        self.assertTrue(all(not isinstance(v,(torch.Tensor,torch.nn.Module)) for v in r._draft_ehproj_source.values()))

    def test_real_agree_four_thread_peers_refuse_before_commit_without_sockets(self):
        # Execute dh.agree's actual MIN/MAX digest and validity protocol. Only
        # transport is substituted; rank-local memory/reload faults are real.
        class Team:
            def __init__(self):
                self.local=threading.local();self.barrier=threading.Barrier(4,timeout=20)
                self.rows=[None]*4
            def reduce(self,tensor,op,group):
                self.rows[self.local.rank]=tensor.clone();self.barrier.wait()
                rows=torch.stack(self.rows)
                tensor.copy_(rows.amin(0) if op==torch.distributed.ReduceOp.MIN else rows.amax(0))
                self.barrier.wait()
            def gather(self,rows,value,group):
                self.rows[self.local.rank]=value;self.barrier.wait()
                rows[:]=self.rows;self.barrier.wait()
        from vllm.model_executor.model_loader import weight_utils
        original_loader=weight_utils.default_weight_loader
        real_vote=dh.agree
        for fault in (None,'sha-disagreement','memory','reload','compiled'):
            with self.subTest(fault=fault):
                team=Team();runners=[self.runner() for _ in range(4)]
                for r in runners:
                    r._draft_head.on=False
                    for m in dh.managers(r):m.graphs.clear()
                if fault=='sha-disagreement':
                    runners[2]._draft_ehproj_source['expected_sha256']='different'
                if fault=='compiled':runners[2].speculator.model.compiled=True
                def loader(param,weight):
                    if fault=='reload' and team.local.rank==2:raise OSError('rank-local reload failure')
                    original_loader(param,weight)
                def peer(rank):
                    team.local.rank=rank
                    try:eh.rollback(runners[rank])
                    except dh.VoteRefused:return False
                    return True
                with patch.object(dh,'agree',lambda payload,valid:real_vote(payload,valid,group=team)), \
                     patch.object(dh,'memory',lambda:eh.FLOOR_BYTES if fault=='memory' and team.local.rank==2 else 8<<30), \
                     patch.object(torch.distributed,'all_reduce',team.reduce), \
                     patch.object(torch.distributed,'all_gather_object',team.gather), \
                     patch.object(torch.distributed,'get_world_size',lambda group:4), \
                     patch.object(torch.distributed,'get_rank',lambda group:team.local.rank), \
                     patch.object(weight_utils,'default_weight_loader',loader):
                    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(peer,range(4)))
                self.assertEqual(results,[fault is None]*4)
                for r in runners:
                    self.assertEqual(type(r._draft_ehproj) is torch.nn.Linear,fault is None)
                    self.assertFalse(r._draft_ehproj_receipt['ehproj_rollback']['qualified'])

    def test_default_and_explicit_off_do_not_install_any_projection_hooks(self):
        for env in ({},{'GLM_DRAFT_EHPROJ':'0','GLM_DRAFT_EHPROJ_INIT':'0'}):
            with patch.object(eh,'install_runner',side_effect=AssertionError('hook installed')), \
                 patch.object(eh,'checkpoint_source',side_effect=AssertionError('checkpoint inspected')):
                self.assertFalse(eh.register(env))
            native=self.native if hasattr(self,'native') else torch.nn.Linear(512,256,bias=False,dtype=torch.bfloat16)
            x=torch.ones(1,512,dtype=torch.bfloat16)
            before=native(x).view(torch.uint8).detach().numpy().tobytes()
            eh.register(env)
            self.assertEqual(native(x).view(torch.uint8).detach().numpy().tobytes(),before)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
