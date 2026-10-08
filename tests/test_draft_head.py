# SPDX-License-Identifier: Apache-2.0
import ast
import hashlib
import os
from pathlib import Path
import sys
import types
import gc
import weakref
import unittest
from unittest.mock import patch

import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay'),str(ROOT/'overlay/kstop')]
import glm_draft_head as d
import glm_nvfp4_format as fmt


class UnquantizedEmbeddingMethod:
    def apply(self,head,x,bias=None):
        return torch.nn.functional.linear(x,head.weight,bias)


class CPUHead(torch.nn.Module):
    def __init__(self,w):
        super().__init__();self.weight=torch.nn.Parameter(w,False)
        self.quant_method=UnquantizedEmbeddingMethod();self.tp_size=4


class CPUBank(torch.nn.Module):
    def __init__(self,head,kind):
        super().__init__();w=head.weight
        if kind=='int8':
            q,s=d.quantize_int8(w)
            z=(q.float().reshape(len(q),-1,128)*s.float().unsqueeze(-1)).reshape_as(w)
        else:
            ts=float(w.abs().max())/(6*448)
            v,s=fmt.encode(w.float().numpy(),ts)
            z=torch.from_numpy(fmt.dequant(v,s,ts))
        self.register_buffer('weight',z.bfloat16());self.resident_bytes=d.storage_bytes(self)
        self.workspace=torch.zeros(48,dtype=torch.int32)
    def forward(self,x,bias=None):
        return torch.nn.functional.linear(x,self.weight,bias)


def runner():
    torch.manual_seed(43)
    head=CPUHead(torch.randn(64,256).bfloat16())
    layer=types.SimpleNamespace(shared_head=types.SimpleNamespace(head=head))
    inner=types.SimpleNamespace(layers={'78':layer},logits_processor=types.SimpleNamespace(
        head_dtype=None,logits_as_input=False,use_all_gather=True))
    draft=type('DeepSeekMTP',(),{})();draft.model=inner
    class Policy:
        pending=ready=guard_pending=None
        def __init__(self):self.used={'old'};self.proposals={'old':[1]};self.trail=bytes(32)
        def forget(self,ids):
            for rid in ids:self.proposals.pop(rid,None);self.used.discard(rid)
    target=types.SimpleNamespace(lm_head=head)
    target_graph={'target':object()}
    sp=types.SimpleNamespace(model=draft,_kstop=Policy(),prefill_cudagraph_manager=types.SimpleNamespace(graphs={'M4':object()}),
        decode_cudagraph_manager=types.SimpleNamespace(graphs={'M1':object()}))
    r=types.SimpleNamespace(model=target,speculator=sp,model_memory_usage=0,execute_model_state=None,
        cudagraph_manager=types.SimpleNamespace(graphs=target_graph))
    r._test_target_graph=target_graph
    def capture():
        facade=r._draft_head
        # Freeze selected operation, as FULL CUDA capture does. Replay cannot
        # observe later changes to a Python bool.
        op=(lambda x:facade.bank(x)) if facade.on else (lambda x:head.quant_method.apply(head,x))
        sp.prefill_cudagraph_manager.graphs['M4']=op
        sp.decode_cudagraph_manager.graphs['M1']=op
    sp.capture=capture
    return r


def local_vote(payload,valid):
    if not valid:raise RuntimeError('refused')


def prepare_cpu(r,kind='nvfp4'):
    d.FORMAT=kind;d.prepare(r,CPUBank,local_vote)
    r.speculator.capture();r._draft_head_ready=True


def change(r,on,epoch,**kw):
    return d.switch(r,on,epoch,True,vote=local_vote,sync=lambda:None,
        mem=lambda:8*(1<<30),collect=lambda:None,empty=lambda:None,**kw)


class DraftTests(unittest.TestCase):
    def test_boot_on_before_first_capture(self):
        r=runner()
        with patch.object(d, 'INITIAL_ON', True):
            prepare_cpu(r)
        self.assertTrue(r._draft_head.on)
        self.assertEqual(r._draft_head_epoch, -1)
        x=torch.ones(1,256).bfloat16()
        self.assertTrue(torch.equal(r._draft_head.bank(x),
            r.speculator.decode_cudagraph_manager.graphs['M1'](x)))
        for invalid in ({'GLM_DRAFT_HEAD_INIT':'1'},
                        {'GLM_DRAFT_HEAD':'nvfp4','GLM_DRAFT_HEAD_INIT':'yes'}):
            with self.assertRaises(ValueError):d.initial_on(invalid)
        self.assertFalse(d.initial_on({}))

    def test_initial_capture_owns_separate_pool_without_runtime_toggle(self):
        r=runner();prepare_cpu(r)
        global_pool=object();new_pool=object()
        r.cudagraph_manager.pool=global_pool
        for m in d.managers(r):
            m.pool=global_pool;m._capture_descs={'FULL':[1]}
        cls=type('GPUModelRunner',(),{'load_model':lambda self:None,
            'capture_model':lambda self:self.speculator.capture()})
        d.install_runner(types.SimpleNamespace(GPUModelRunner=cls))
        fake=types.SimpleNamespace(CUDAGraphMode=types.SimpleNamespace(FULL='FULL'))
        renew=d.renew_draft_pool
        with patch.dict(sys.modules,{'vllm.config.compilation':fake}),patch.object(d,'agree',local_vote),             patch.object(torch.cuda,'synchronize',lambda:None),patch.object(torch.cuda,'empty_cache',lambda:None),             patch.object(d,'renew_draft_pool',lambda r:renew(r,lambda:new_pool)):
            cls.capture_model(r)
        self.assertIs(r.cudagraph_manager.pool,global_pool)
        self.assertTrue(all(m.pool is new_pool for m in d.managers(r)))
        self.assertTrue(r._draft_head_ready)
        self.assertEqual(r._draft_head_epoch,-1)

    def test_twenty_toggles_allocator_pool_and_capture_references(self):
        # Model the native allocator: cached graph blocks remain charged while
        # any graph in that pool is alive. Target pins the old global pool.
        class Allocator:
            def __init__(self):self.pools=[]
            def pool(self):
                p={'owners':0,'allocated':0,'reserved':0};self.pools.append(p);return p
            def empty(self):
                for p in self.pools:
                    if not p['owners']:p['allocated']=p['reserved']=0
            def sample(self):
                return tuple(sum(p[k] for p in self.pools) for k in ('allocated','reserved'))
        class Graph:
            def __init__(self,pool):
                self.pool=pool;pool['owners']+=1
                pool['allocated']+=32<<20;pool['reserved']+=40<<20
            def reset(self):
                if self.pool is not None:self.pool['owners']-=1;self.pool=None
            def __del__(self):self.reset()
        for fixed in (False,True):
            r=runner();prepare_cpu(r);allocator=Allocator()
            target_pool=allocator.pool();target=Graph(target_pool)
            retired=[]
            def capture():
                pool=d.managers(r)[0].pool
                if pool is None:
                    pool=allocator.pool();d.renew_draft_pool(r,lambda:pool)
                for m,key in zip(d.managers(r),('M4','M1')):
                    graph=Graph(pool);m.graphs[key]=graph;retired.append(weakref.ref(graph))
                    # Capture-only buffers in native factory closure.
                    m._k4_drafthead_factory=lambda g=graph: g
            d.renew_draft_pool(r,allocator.pool if fixed else lambda:target_pool)
            capture();baseline=allocator.sample();bank=r._draft_head.bank
            for epoch in range(20):
                r.speculator.capture=capture
                def broken_release(r):
                    for m in d.managers(r):m.graphs.clear()
                with patch.object(d,'release_draft_graphs',d.release_draft_graphs if fixed else broken_release):
                    d.switch(r,1-epoch%2,epoch,True,vote=local_vote,sync=lambda:None,
                        mem=lambda:8<<30,collect=gc.collect,empty=allocator.empty)
                if fixed:
                    self.assertEqual(allocator.sample(),baseline)
                    self.assertTrue(all(ref() is None for ref in retired[:-2]))
                    self.assertIs(r._draft_head.bank,bank)
                    self.assertIs(target.pool,target_pool)
            if not fixed:
                self.assertGreater(allocator.sample()[0]-baseline[0], 1<<30)

    def test_gpu_leakcheck_controlflow_and_tolerances_without_gpu(self):
        for leak in (False,True):
            r=runner();prepare_cpu(r)
            r._draft_head=types.SimpleNamespace(on=False,bank=r._draft_head.bank,
                original=types.SimpleNamespace(weight=types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))))
            calls=[]
            def toggle(r,on,epoch,drained):
                calls.append(on);r._draft_head.on=bool(on);r._draft_head_epoch=epoch
            def sample(r):
                drift=max(0,len(calls)-4)*(64<<20) if leak else 0
                return dict(allocated=(100<<20)+drift,reserved=(120<<20)+drift,mem_available=8<<30)
            with patch.object(d,'switch',toggle):
                if leak:
                    def vote(payload, valid):
                        if not valid:raise d.VoteRefused({'votes':[{'rank':0,'valid':False}]})
                    report=d.leakcheck(types.SimpleNamespace(model_runner=r),True,
                        vote=vote,sync=lambda:None,collect=lambda:None,empty=lambda:None,
                        sample=sample,gather=lambda row:[row])
                    self.assertTrue(report['refused'])
                    self.assertEqual(len(report['samples']),3)
                    self.assertFalse(r._draft_head_ready)
                else:
                    report=d.leakcheck(types.SimpleNamespace(model_runner=r),True,vote=local_vote,
                        sync=lambda:None,collect=lambda:None,empty=lambda:None,sample=sample)
                    self.assertEqual(report['transitions_checked'],20)
                    self.assertEqual(len(report['samples']),22)
                    self.assertFalse(r._draft_head.on)
                    self.assertEqual(len(calls),24)

    def test_graph_reset_failure_refuses_before_capture(self):
        r=runner();prepare_cpu(r)
        class BadGraph:
            def reset(self):raise RuntimeError('reset')
        r.speculator.decode_cudagraph_manager.graphs['M1']=BadGraph()
        with patch.object(r.speculator,'capture') as capture:
            with self.assertRaises(RuntimeError):change(r,1,0)
            capture.assert_not_called()
        self.assertFalse(r._draft_head_ready)

    def test_default_cold_and_options(self):
        hooks=list(sys.meta_path)
        self.assertFalse(d.register({}));self.assertEqual(hooks,sys.meta_path)
        for value in ('1','fp8','NVFP4',''):
            with self.assertRaises(ValueError):d.options({'GLM_DRAFT_HEAD':value})
        for mode in ('nvfp4','int8'):
            with self.assertRaises(ValueError):d.options({'GLM_DRAFT_HEAD':mode})
            self.assertEqual(d.options({'GLM_DRAFT_HEAD':mode,'VLLM_USE_V2_MODEL_RUNNER':'1','GLM_MTP_KSTOP':'1'}),mode)

    def test_exact_byte_accounting(self):
        self.assertEqual(d.byte_cost('nvfp4')['total'],133816516)
        self.assertEqual(d.byte_cost('int8')['total'],241613008)
        self.assertEqual(d.byte_cost('nvfp4')['bf16_retained'],475791360)

    def test_original_native_target_and_draft_processor(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/layers/logits_processor.py'
        cls=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LogitsProcessor')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_head')
        code=compile('from __future__ import annotations\n'+ast.unparse(method),str(src),'exec')
        ns={'torch':torch};exec(code,ns);native=ns['_apply_head']
        lp=types.SimpleNamespace(head_dtype=None)
        for kind in ('nvfp4','int8'):
            r=runner();target=r.model.lm_head;prepare_cpu(r,kind)
            # Exercise M1/M4 plus wider verify rows and changed-input replay.
            for m in (1,4,12,16,64):
                x=torch.randn(m,256).bfloat16()
                before=native(lp,target,x,None).clone()
                off=native(lp,r._draft_head,x,None)
                self.assertTrue(torch.equal(before,off))
                change(r,1,m*2)
                self.assertIs(r.model.lm_head,target)
                self.assertTrue(torch.equal(before,native(lp,target,x,None)))
                draft=native(lp,r._draft_head,x,None)
                self.assertFalse(torch.equal(draft,before))
                self.assertTrue(torch.equal(draft,r.speculator.decode_cudagraph_manager.graphs['M1'](x)))
                change(r,0,m*2+1)
                self.assertTrue(torch.equal(before,r.speculator.decode_cudagraph_manager.graphs['M1'](x)))

    def test_graph_recapture_and_target_identity(self):
        r=runner();prepare_cpu(r);target=r.cudagraph_manager.graphs.copy()
        old=r.speculator.decode_cudagraph_manager.graphs['M1']
        for epoch,on in enumerate((1,0,1,0,1,0)):
            change(r,on,epoch)
            self.assertEqual(r.cudagraph_manager.graphs,target)
            self.assertEqual(r._draft_head.on,bool(on))
        self.assertIsNot(old,r.speculator.decode_cudagraph_manager.graphs['M1'])
        self.assertFalse(r.speculator._kstop.used);self.assertFalse(r.speculator._kstop.proposals)

    def test_stale_busy_drain_and_memory_refuse_before_mutation(self):
        for fault in ('epoch','busy','pending','guard','drain','memory','ready'):
            r=runner();prepare_cpu(r);before=r.speculator.decode_cudagraph_manager.graphs.copy()
            args=dict(on=1,epoch=1,scheduler_drained=True,vote=local_vote,sync=lambda:None,
                mem=lambda:8*(1<<30),collect=lambda:None,empty=lambda:None)
            if fault=='epoch':args['epoch']=-1
            if fault=='busy':r.execute_model_state=object()
            if fault=='pending':r.speculator._kstop.pending=object()
            if fault=='guard':r.speculator._kstop.guard_pending=object()
            if fault=='drain':args['scheduler_drained']=False
            if fault=='memory':args['mem']=lambda:5*(1<<30)
            if fault=='ready':r._draft_head_ready=False
            with self.assertRaises(RuntimeError):d.switch(r,**args)
            self.assertFalse(r._draft_head.on);self.assertEqual(before,r.speculator.decode_cudagraph_manager.graphs)

    def test_capture_failure_never_publishes_ready(self):
        r=runner();prepare_cpu(r)
        r.speculator.capture=lambda:(_ for _ in ()).throw(ValueError('capture'))
        with self.assertRaises(RuntimeError):change(r,1,0)
        self.assertFalse(r._draft_head_ready)

    def test_wrong_alias_and_quant_method_refused(self):
        for fault in ('alias','method','dtype','logits'):
            r=runner()
            if fault=='alias':r.speculator.model.model.layers['78'].shared_head.head=CPUHead(torch.zeros(64,256).bfloat16())
            if fault=='method':r.model.lm_head.quant_method=object()
            if fault=='dtype':r.model.lm_head.weight.data=r.model.lm_head.weight.float()
            if fault=='logits':r.speculator.model.model.logits_processor.head_dtype=torch.float32
            with self.assertRaises(RuntimeError):d.prepare(r,CPUBank,local_vote)
            self.assertFalse(hasattr(r,'_draft_head'))

    def test_integer_packing_and_zeros(self):
        w=torch.zeros(2,128).bfloat16();q,s=d.quantize_int8(w)
        self.assertTrue(torch.equal(s,torch.ones_like(s)))
        packed=d.int8_pack(q)
        unpack=((packed.to(torch.int64).unsqueeze(-1)>>torch.tensor([0,8,16,24]))&255).flatten(-2)-128
        self.assertTrue(torch.equal(unpack,q))
        for bad in (torch.ones(2,127),torch.full((2,128),float('nan'))):
            with self.assertRaises(ValueError):d.quantize_int8(bad)

    def test_source_closure(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])
        for name,want in d.PINS.items():
            self.assertEqual(hashlib.sha256((src/(name.replace('.','/')+'.py')).read_bytes()).hexdigest(),want)

    def test_serialized_worker_arguments(self):
        cls=type('Worker',(),{})
        mod=types.SimpleNamespace(Worker=cls);d.install_worker(mod)
        obj=cls();obj.model_runner=runner();prepare_cpu(obj.model_runner)
        fake=types.ModuleType('vllm.distributed');fake.get_tp_group=lambda:types.SimpleNamespace(rank_in_group=0)
        with patch.object(d,'switch') as changed,patch.dict(sys.modules,{'vllm.distributed':fake}):
            obj.draft_head_set('1','4','true')
            changed.assert_called_once_with(obj.model_runner,1,4,True)

    def test_enabled_registration_source_check_without_cuda(self):
        prior=list(sys.meta_path)
        env=dict(GLM_DRAFT_HEAD='nvfp4',VLLM_USE_V2_MODEL_RUNNER='1',GLM_MTP_KSTOP='1')
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm'
        with patch('importlib.util.find_spec',return_value=types.SimpleNamespace(submodule_search_locations=[str(src)])):
            self.assertTrue(d.register(env))
        self.assertEqual(set(sys.meta_path[0].callbacks),{d.RUNNER,d.WORKER,
            'vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils'})
        sys.meta_path[:]=prior

    def test_microcheck_controlflow_with_cpu_graph_substitute(self):
        r=runner();prepare_cpu(r)
        class Graph:
            current=None
            def __enter__(self):Graph.current=self;return self
            def __exit__(self,*a):Graph.current=None
            def replay(self):self.out.copy_(self.fn())
        forward=CPUBank.forward
        def recorded(bank,x,bias=None):
            result=forward(bank,x,bias)
            if Graph.current is not None:
                Graph.current.out=result
                Graph.current.fn=lambda:forward(bank,x,bias)
            return result
        fake=types.ModuleType('vllm.distributed');fake.get_tp_group=lambda:types.SimpleNamespace(rank_in_group=0)
        with patch.object(d,'agree',local_vote),patch.object(d,'memory',lambda:8*(1<<30)),\
             patch.object(torch.cuda,'synchronize',lambda:None),\
             patch.object(torch.cuda,'CUDAGraph',Graph),patch.object(torch.cuda,'graph',lambda g:g),\
             patch.object(CPUBank,'forward',recorded),patch.dict(sys.modules,{'vllm.distributed':fake}):
            receipt=d.microcheck(r,True)
        self.assertEqual([v['m'] for v in receipt['cases']],[1,4,12,16])
        self.assertTrue(all(v['target_bit_exact'] and v['changed_input_replays']==3 for v in receipt['cases']))
        self.assertFalse(r._draft_head.on)


if __name__=='__main__':unittest.main()
