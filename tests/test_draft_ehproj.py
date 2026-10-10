# SPDX-License-Identifier: Apache-2.0
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]
import glm_draft_ehproj as eh
from glm_draft_ehproj_config import options,byte_cost

ENV=dict(GLM_DRAFT_EHPROJ='fp8',GLM_DRAFT_EHPROJ_INIT='1',VLLM_USE_V2_MODEL_RUNNER='1',
         GLM_MTP_KSTOP='1',GLM_DRAFT_HEAD='nvfp4',GLM_DRAFT_HEAD_INIT='1')

class DeepSeekMTP(torch.nn.Module):
    def __init__(self,original):
        super().__init__();self.model=torch.nn.Module();layer=torch.nn.Module();layer.eh_proj=original
        self.model.layers=torch.nn.ModuleDict({'78':layer})


def fixture(shared=False):
    original=torch.nn.Linear(12288,6144,bias=False,device='meta',dtype=torch.bfloat16)
    target=torch.nn.Module();target.lm_head=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
    if shared:target.alias=original.weight
    return NS(model=target,speculator=NS(model=DeepSeekMTP(original),_kstop=NS(bad=None)),model_memory_usage=1<<30),original


def bank(original):
    b=torch.nn.Module();b.workspace=torch.zeros(48,dtype=torch.int32);b.resident_bytes=byte_cost()['total'];return b


def prepare(runner,**kw):
    # Metadata-only preparation fixture: rollback I/O is covered separately.
    return eh.prepare(runner,source_reader=lambda *a:dict(path='synthetic',tensor='eh_proj',
        shape=[6144,12288],bytes=150994944,expected_sha256='fixture',
        replicated=True,loader='default_weight_loader'),**kw)


def vote(payload,valid):
    if not valid:raise RuntimeError('all-rank refusal')

class Tests(unittest.TestCase):
    def test_options(self):
        self.assertEqual(options({}),'0');self.assertEqual(options(ENV),'fp8')
        for e in ({'GLM_DRAFT_EHPROJ':'nvfp4'},ENV|{'GLM_DRAFT_EHPROJ_INIT':'0'},
                  ENV|{'GLM_DRAFT_HEAD_INIT':'0'},{'GLM_DRAFT_EHPROJ_INIT':'1'}):
            with self.assertRaises(ValueError):options(e)
    def test_bytes(self):
        self.assertEqual(byte_cost()['total'],75509952)
        self.assertEqual(byte_cost()['delta'],-75484992)
        self.assertEqual(byte_cost(retain=True)['delta'],75509952)
    def test_unshared_release_and_shared_debit(self):
        for shared in (False,True):
            r,orig=fixture(shared);head=r.model.lm_head;before=r.model_memory_usage
            prepare(r,factory=bank,vote=vote)
            self.assertIs(r.model.lm_head,head)
            self.assertIs(r.speculator.model.model.layers['78'].eh_proj,r._draft_ehproj)
            self.assertEqual(r.model_memory_usage-before,byte_cost(retain=shared)['delta'])
            self.assertNotIn(orig,list(r.speculator.model.modules()))
            if shared:self.assertIs(r.model.alias,orig.weight)
            with self.assertRaises(RuntimeError):prepare(r,factory=bank,vote=vote)
    def test_peer_failure_before_attach(self):
        r,orig=fixture()
        def refuse(*a):raise RuntimeError('peer failed')
        with self.assertRaises(RuntimeError):prepare(r,factory=bank,vote=refuse)
        self.assertIs(r.speculator.model.model.layers['78'].eh_proj,orig)
        self.assertEqual(r.model_memory_usage,1<<30)
    def test_quantizer_scale_storage_and_zero(self):
        w=torch.tensor([[0,0,0,0],[1.,.1,-.03,-1.]],dtype=torch.bfloat16)
        q,s=eh.encode_fp8(w);self.assertEqual(s.dtype,torch.bfloat16)
        self.assertEqual(q.dtype,torch.float8_e4m3fn);self.assertTrue(torch.isfinite(q.float()).all())
        self.assertTrue(torch.equal(q[0].float(),torch.zeros(4)))
        expected=((w.float()/s.float()).clamp(-448,448).to(torch.float8_e4m3fn).float()*s.float()).bfloat16()
        self.assertTrue(torch.equal(expected,(q.float()*s.float()).bfloat16()))
    def test_capture_ready_and_refusal(self):
        for fault in (False,True):
            r,_=fixture();prepare(r,factory=bank,vote=vote)
            def cap(self):
                self._draft_head_ready=True
                self._draft_head_qualification=None if fault else dict(cases=[1],target_fixed_hidden=dict(bit_exact=True))
            cls=type('Runner',(),dict(load_model=lambda s:None,capture_model=cap))
            eh.install_runner(NS(GPUModelRunner=cls))
            r.model.lm_head.quant_method=object()
            with patch.object(eh.dh,'agree',vote):
                if fault:
                    with self.assertRaises(RuntimeError):cls.capture_model(r)
                    self.assertFalse(r._draft_ehproj_ready)
                else:cls.capture_model(r);self.assertTrue(r._draft_ehproj_ready)
    def test_missing_combined_qualification_dispatches_native_fallback(self):
        r,_=fixture();prepare(r,factory=bank,vote=vote)
        def cap(self):
            self._draft_head_ready=False
            self._draft_head_qualification=None
            self._draft_head_init_failure={'votes':[{'valid':False}]}
        cls=type('Runner',(),dict(load_model=lambda s:None,capture_model=cap))
        eh.install_runner(NS(GPUModelRunner=cls))
        r.model.lm_head.quant_method=object()
        rows=[]
        term=eh.dh.qualification_term
        def receipt(*a,**kw):
            rows.append(a)
            return term(*a,**kw)
        def refuse(payload,valid):
            if not valid:raise eh.dh.VoteRefused({'votes':[{'valid':False}]})
        with patch.object(eh.dh,'agree',refuse),patch.object(eh.dh,'qualification_term',receipt), \
             patch.object(eh.dh,'initial_fallback',side_effect=eh.dh.VoteRefused({'failed':True})) as fallback:
            with self.assertRaises(eh.dh.VoteRefused):cls.capture_model(r)
        fallback.assert_called_once()
        self.assertFalse(r._draft_ehproj_ready)
        self.assertFalse(r._draft_head_ready)
        self.assertFalse(rows[-1][1]['combined_init_qualified'])

    def test_no_runtime_control(self):
        self.assertFalse(hasattr(eh,'set'));self.assertFalse(hasattr(eh,'set_projection'))

if __name__=='__main__':unittest.main()
