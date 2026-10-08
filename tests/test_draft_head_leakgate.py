# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch
import torch
from test_draft_head import runner, prepare_cpu
import glm_draft_head as dh

class LeakgateTests(unittest.TestCase):
    def test_logging_precedes_vote_and_refusal_preserves_evidence(self):
        r=runner();prepare_cpu(r);bank=r._draft_head.bank
        r._draft_head=NS(on=False,bank=bank,original=NS(weight=NS(device=NS(type='cuda'))))
        events=[];output=io.StringIO();calls=[]
        def toggle(r,on,epoch,drained):
            calls.append(on);r._draft_head.on=bool(on);r._draft_head_epoch=epoch
        def vote(payload,valid):
            if 'leakcheck_step' in payload:
                events.append(payload['leakcheck_step'])
                row=json.loads(output.getvalue().splitlines()[-1].split(' ',1)[1])
                self.assertEqual(row['index'],payload['leakcheck_step'])
                self.assertIn('bank_hash',row);self.assertIn('draft_pools',row)
                if not valid:raise dh.VoteRefused({'votes':[{'rank':0,'valid':False}]})
        def sample(r):return dict(allocated=100,reserved=200+(17<<20 if len(calls)==5 else 0),mem_available=8<<30)
        with patch.object(dh,'switch',toggle),contextlib.redirect_stdout(output):
            report=dh.leakcheck(NS(model_runner=r),True,vote=vote,sync=lambda:None,
                collect=lambda:None,empty=lambda:None,sample=sample,gather=lambda history:[history])
        self.assertEqual(events,[0,1,2]);self.assertTrue(report['refused'])
        self.assertEqual(report['failed_conditions'][0]['condition'],'reserved_drift')
        self.assertFalse(report['resume_allowed']);self.assertEqual(len(report['samples']),3)

    def test_runtime_setter_and_microcheck_refusals_return(self):
        cls=type('Worker',(),{});dh.install_worker(NS(Worker=cls));w=cls();w.model_runner=runner();prepare_cpu(w.model_runner)
        def refused(*a,**kw):raise dh.VoteRefused({'votes':[{'rank':2,'condition':'memory_floor','measured':4<<30}]})
        with patch.object(dh,'switch',refused),patch.object(dh,'microcheck',refused):
            for call,args in ((w.draft_head_set,('1','3','true')),(w.draft_head_microcheck,('true',))):
                receipt=call(*args);self.assertTrue(receipt['refused']);self.assertFalse(receipt['ready'])
                self.assertEqual(receipt['refusal']['votes'][0]['rank'],2)

    def test_named_k4_control_refusal_and_other_errors(self):
        r=runner();prepare_cpu(r);r.speculator._kstop.epoch=4;r.speculator._kstop.k4_on=False
        def callback(w):raise RuntimeError('collective-safe kstop refusal at k4-switch (planner)')
        with patch.object(dh,'gather_receipts',lambda row:[row]),patch.object(dh,'memory',lambda:4<<30):
            result=dh.runtime_rpc(callback)(NS(model_runner=r))
            self.assertTrue(result['refused'])
            self.assertEqual(result['refusal']['votes'][0]['condition'],'k4_switch_vote')
        def device_fault(w):raise RuntimeError('device lost')
        with self.assertRaisesRegex(RuntimeError,'device lost'):
            dh.runtime_rpc(device_fault)(NS(model_runner=r))

    def test_stable_stream_native_context_and_target_isolation(self):
        contexts=[];created=[]
        @contextlib.contextmanager
        def native(device,graph_capture_context=None):
            contexts.append(graph_capture_context);yield graph_capture_context
        mod=NS(graph_capture=native)
        def capture(self,factory):
            with mod.graph_capture(device='cuda'):pass
            return 42
        base=type('CudaGraphManager',(),{'capture':capture})
        draft=type('SpeculatorCudaGraphManager',(base,),{})
        mod.CudaGraphManager=base;dh.install_graph(mod)
        def stream(**kw):
            s=NS(cuda_stream=len(created));created.append(s);return s
        fake=ModuleType('vllm.distributed.parallel_state');fake.GraphCaptureContext=lambda s:NS(stream=s)
        m1,m2,target=draft(),draft(),base()
        with patch.object(torch.cuda,'Stream',stream),patch.dict(sys.modules,{'vllm.distributed.parallel_state':fake}):
            for _ in range(24):
                self.assertEqual(m1.capture(object()),42);m2.capture(object())
                m1.graphs={};m2.graphs={};m1.pool=None;m2.pool=None
            target.capture(object())
        self.assertEqual(len(created),2)
        self.assertIs(contexts[0],contexts[2]);self.assertIsNot(contexts[0],contexts[1]);self.assertIsNone(contexts[-1])
        self.assertIsNone(dh._capture_manager.get())

if __name__=='__main__':unittest.main()
