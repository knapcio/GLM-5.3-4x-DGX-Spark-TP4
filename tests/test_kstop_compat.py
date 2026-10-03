# SPDX-License-Identifier: Apache-2.0
"""Composition, changing K, shared-index bounds and sampler/cache CPU contracts."""
import ast
import dataclasses
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import tempfile
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop')]
import glm_dsa_short as D
import glm_mtp_kstop as K
SRC=Path(os.environ.get('GLM_IMAGE_SRC','/image-source'))


class Composition(unittest.TestCase):
    def setUp(self):
        p=patch.multiple(D,ENABLED=True,KSTOP=True);p.start();self.addCleanup(p.stop)

    def test_exact_overlap_and_both_transforms(self):
        self.assertEqual(set(D.PINS)&set(K.PINS),D.SHARED)
        for name in D.SHARED:
            raw=(SRC/(name.replace('.','/')+'.py')).read_text()
            self.assertEqual(D.PINS[name],K.PINS[name])
            cooked=K.combined_transform(name,raw)
            compile(cooked,name,'exec')
            self.assertIn('short_context: bool' if name==K.CG else 'glm_dsa_short',cooked)
            self.assertIn('_kstop' if name!=K.CG else '([2,3,4]',cooked)
            with self.assertRaisesRegex(RuntimeError,'source drift'):
                K.combined_transform(name,raw+'\n')
            self.assertIsNone(D.Finder().find_spec(name))

    def test_unmodified_sampling_and_sharing_sources_are_pinned(self):
        pins=json.loads((ROOT/'overlay/kstop/compat_source_pins.json').read_text())
        self.assertEqual(len(pins),7)
        for name,want in pins.items():
            with self.subTest(module=name):
                self.assertEqual(hashlib.sha256((SRC/(name.replace('.','/')+'.py')).read_bytes()).hexdigest(),want,name)
        import glm_spec_sample as S
        for name in set(pins)&set(S.PINS):
            self.assertEqual(pins[name],S.PINS[name],name)

    def test_one_code_object_through_a_wrapper_loader(self):
        # Same wrapper-loader shape as the full-MLA startup chain: it exposes
        # exec_module only and calls the original SourceFileLoader internally.
        name='kstop_compat_chain_fixture';raw="value=['raw']\n"
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self,n,path=None,target=None):
                if n!=name:return None
                plain=importlib.machinery.PathFinder.find_spec(n)
                inner=plain.loader
                class Loader:
                    def create_module(self,spec):return None
                    def exec_module(self,mod):inner.exec_module(mod);mod.value.append('wrapper')
                plain.loader=Loader();return plain
        original=K.transform
        def kstop(n,text):return original(n,text)+"value.append('kstop')\n"
        before=list(sys.meta_path);code_method=importlib.machinery.SourceFileLoader.get_code
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,name+'.py').write_text(raw);sys.path.insert(0,tmp)
            try:
                with patch.dict(K.PINS,{name:hashlib.sha256(raw.encode()).hexdigest()}), \
                     patch.dict(D.PINS,{name:hashlib.sha256(raw.encode()).hexdigest()}), \
                     patch.object(D,'SHARED',{name}),patch.object(K,'transform',kstop), \
                     patch.dict(D.TRANSFORMS,{name:lambda text:text+"value.append('short')\n"}), \
                     patch.dict(D.INSTALL,{name:lambda mod:mod.value.append('short-install')}):
                    sys.meta_path[:0]=[K.Hook(),D.Finder(),Finder()]
                    mod=importlib.import_module(name)
                    self.assertEqual(mod.value,['raw','kstop','short','wrapper','short-install'])
                    self.assertIs(importlib.machinery.SourceFileLoader.get_code,code_method)
            finally:
                sys.meta_path[:]=before;sys.path.remove(tmp);sys.modules.pop(name,None)

    def test_variable_verify_widths_and_boundaries(self):
        from test_dsa_short import schedule
        for n in (1,4):
            for q in (2,3,4,2,4,3):
                s,b,c=schedule(n,2048-q)
                s.num_scheduled_tokens={r:q for r in b.req_ids}
                s.total_num_scheduled_tokens=b.num_tokens=q*n
                s.scheduled_spec_decode_tokens={r:[0]*(q-1) for r in b.req_ids}
                b.num_scheduled_tokens=[q]*n
                self.assertTrue(D.eligible(s,b,c,q))
                self.assertFalse(D.eligible(s,b,[2049-q]*n,q))
                b.num_tokens+=1
                self.assertFalse(D.eligible(s,b,c,q))
        s,b,c=schedule(4)
        b.num_scheduled_tokens=[2,3,4,3]
        self.assertFalse(D.eligible(s,b,c,None))

    def test_whole_shared_index_cycle_is_short_or_stock(self):
        for n in (1,4):
            for q in (2,3,4):
                self.assertTrue(D.draft_eligible([2046]*n,n,n*q,q,0))
                self.assertFalse(D.draft_eligible([2047]*n,n,n*q,q,0))
            for step in (1,2):
                self.assertTrue(D.draft_eligible([2046]*n,n,n,1,step))
                self.assertFalse(D.draft_eligible([2047]*n,n,n,1,step))
        self.assertEqual(D.graph_widths(),(1,2,3,4))


    def test_capture_and_dispatch_all_kstop_widths(self):
        from enum import Enum
        class Mode(Enum):NONE=0;FULL=1
        @dataclasses.dataclass(frozen=True)
        class Desc:
            cg_mode: Mode
            num_tokens: int
            num_reqs: int
            uniform_token_count: int
            num_active_loras: int=0
            short_context: bool=False
        current=[None]
        class Manager:
            def __init__(self):
                self._capture_descs={Mode.FULL:[Desc(Mode.FULL,n*q,n,q) for n in (1,4) for q in (1,2,3,4)]}
                self._graphs_captured=False;self.graphs={}
            def capture(self,factory):
                for d in self._capture_descs[Mode.FULL]:self.graphs[d]=factory(d,False)(Mode.NONE)
                self._graphs_captured=True
            def dispatch(self,*a):return current[0]
        class ModelManager(Manager):pass
        D.install_cg(NS(CudaGraphManager=Manager,ModelCudaGraphManager=ModelManager,CUDAGraphMode=Mode))
        m=ModelManager();m.capture(lambda d,w:lambda mode:D.active())
        self.assertEqual(len(m.graphs),16)
        self.assertTrue(all(d.short_context==value for d,value in m.graphs.items()))
        for n in (1,4):
            for q in (2,4,3,1,4,2):
                current[0]=Desc(Mode.FULL,n*q,n,q)
                with D.context(True):self.assertTrue(m.dispatch(n,n*q,q,0).short_context)
                with D.context(False):self.assertFalse(m.dispatch(n,n*q,q,0).short_context)
        m=ModelManager()
        m._capture_descs={Mode.FULL:[Desc(Mode.FULL,16,4,4)]}
        # Exercise the installer with sparse native candidates, as [1,4,16].
        class SparseManager(Manager):
            def __init__(self):
                self._capture_descs={Mode.FULL:[Desc(Mode.FULL,16,4,4),Desc(Mode.FULL,3,1,3)]}
                self._graphs_captured=False;self.graphs={}
                self.max_num_reqs=4;self.decode_query_len=4
        class SparseModel(SparseManager):pass
        D.install_cg(NS(CudaGraphManager=SparseManager,ModelCudaGraphManager=SparseModel,CUDAGraphMode=Mode))
        m=SparseModel();m.capture(lambda d,w:lambda mode:D.active())
        current[0]=Desc(Mode.NONE,8,4,2)
        with D.context(False):self.assertEqual(m.dispatch(4,8,2,0),current[0])
        for q in (2,3,4):
            with D.context(True):
                got=m.dispatch(4,4*q,q,0)
                self.assertEqual((got.cg_mode,got.num_tokens,got.uniform_token_count,got.short_context),(Mode.FULL,4*q,q,True))
        current[0]=Desc(Mode.FULL,8,4,2)
        del m.graphs[dataclasses.replace(current[0],short_context=True)]
        with D.context(True),self.assertRaisesRegex(RuntimeError,'missing'):m.dispatch(4,8,2,0)


    def test_uniform_batches_capture_only_reachable_short_shapes(self):
        # K-stop verifies q2..q4 and, with uniform batches, every four-request rectangle is q4: the exact short
        # captures are c1 q2/q3/q4 and c4 q4 on verify/first-draft managers and q1 c1/c4 on the later-draft manager.
        from enum import Enum
        class Mode(Enum):NONE=0;FULL=1
        @dataclasses.dataclass(frozen=True)
        class Desc:
            cg_mode: Mode
            num_tokens: int
            num_reqs: int
            uniform_token_count: int
            num_active_loras: int=0
            short_context: bool=False
        stock={4:[(2,1,2),(3,1,3),(4,1,4),(4,2,2),(6,2,3),(16,4,4)],1:[(1,1,1),(4,4,1)]}   # [1,4,16], 4 slots
        current=[None]
        def manager(dql):
            class Manager:
                def __init__(self):
                    self._capture_descs={Mode.FULL:[Desc(Mode.FULL,t,r,q) for t,r,q in stock[dql]]}
                    self._graphs_captured=False;self.graphs={};self.max_num_reqs=4;self.decode_query_len=dql
                def capture(self,factory):
                    for d in self._capture_descs[Mode.FULL]:self.graphs[d]=factory(d,False)(Mode.NONE)
                    self._graphs_captured=True
                def dispatch(self,*a):return current[0]
            class Model(Manager):pass
            return Manager,Model
        for uniform,c4 in ((True,[(16,4,4)]),(False,[(8,4,2),(12,4,3),(16,4,4)])):
            with patch.object(D,'UNIFORM',uniform):
                for dql,want in ((4,[(2,1,2),(3,1,3),(4,1,4)]+c4),(1,[(1,1,1),(4,4,1)])):
                    Manager,Model=manager(dql)
                    D.install_cg(NS(CudaGraphManager=Manager,ModelCudaGraphManager=Model,CUDAGraphMode=Mode))
                    m=Model()
                    short=sorted((d.num_tokens,d.num_reqs,d.uniform_token_count)
                                 for d in m._capture_descs[Mode.FULL] if d.short_context)
                    self.assertEqual(short,sorted(want),(uniform,dql))
                    self.assertEqual(sorted((d.num_tokens,d.num_reqs,d.uniform_token_count)
                                            for d in m._capture_descs[Mode.FULL] if not d.short_context),sorted(stock[dql]))
        # Uniform: a (never scheduled) c4 q2 rectangle under the short context replays the stock graph.
        with patch.object(D,'UNIFORM',True):
            Manager,Model=manager(4)
            D.install_cg(NS(CudaGraphManager=Manager,ModelCudaGraphManager=Model,CUDAGraphMode=Mode))
            m=Model();m.capture(lambda d,w:lambda mode:D.active())
            current[0]=Desc(Mode.FULL,16,4,4)
            with D.context(True):self.assertFalse(m.dispatch(4,8,2,0).short_context)
            with D.context(True):self.assertTrue(m.dispatch(4,16,4,0).short_context)

    def test_register_reads_the_uniform_flag_with_kstop_only(self):
        for env,want in (({'GLM_MTP_KSTOP':'1','GLM_MTP_KSTOP_UNIFORM_BATCH':'1'},True),
                         ({'GLM_MTP_KSTOP':'1','GLM_MTP_KSTOP_UNIFORM_BATCH':'k2'},True),
                         ({'GLM_MTP_KSTOP':'1','GLM_MTP_KSTOP_UNIFORM_BATCH':'0'},False),
                         ({'GLM_MTP_KSTOP':'0','GLM_MTP_KSTOP_UNIFORM_BATCH':'1'},False)):
            with patch.object(D,'ENABLED',False),patch.object(D,'UNIFORM',None),patch.object(sys,'meta_path',list(sys.meta_path)), \
                    patch.object(D,'TRANSFORMS',{}):
                D.register(dict(env,GLM_INDEXER_SHORTCUT='1'))
                self.assertIs(D.UNIFORM,want)
                self.assertEqual(D.UNIFORM_Q, 3 if env['GLM_MTP_KSTOP_UNIFORM_BATCH']=='k2' else 4)

    def test_k2_pinned_descriptors_match_capture_plan_and_reuse_m6(self):
        # Execute the source-pinned builder/dispatcher, then compare every
        # selected descriptor with the simulated capture bank on the Mac.
        from collections import defaultdict
        from itertools import product
        from enum import Enum
        class Mode(Enum):
            NONE=0; FULL=1; FULL_DECODE_ONLY=2
            def decode_mode(self):return Mode.FULL
            def mixed_mode(self):return Mode.NONE
            def separate_routine(self):return True
            def __bool__(self):return self!=Mode.NONE
        raw=(SRC/'vllm/v1/worker/gpu/cudagraph_utils.py').read_text()
        tree=ast.parse(K.combined_transform(K.CG,raw))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CudaGraphManager')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)
                 and n.name in ('_init_candidates','_resolve_effective_loras','dispatch')]
        cls.body=methods
        nodes=[n for n in tree.body if getattr(n,'name',None) in ('BatchExecutionDescriptor','_is_compatible')]+[cls]
        import kstop_runtime as R
        ns=dict(os=os,_kstop=R,dataclass=dataclasses.dataclass,CUDAGraphMode=Mode,defaultdict=defaultdict,product=product,
                round_up=lambda n,q: -(-n//q)*q)
        exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),ns)
        Native=ns['CudaGraphManager'];Desc=ns['BatchExecutionDescriptor']
        class Manager(Native):
            def __init__(self, q, sizes):
                self.compilation_config=NS(cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=max(sizes))
                self.vllm_config=NS(speculative_config=NS(uses_dynamic_speculative_decoding=lambda:False))
                self.cudagraph_mode=Mode.FULL_DECODE_ONLY;self.decode_query_len=q;self.max_num_reqs=4
                self.lora_capture_cases=[0];self.varlen_decode=False
                self._candidates={};self._capture_descs={};self._lora_dispatch_map={};self._max_lora_case=0
                self.graphs={};self._graphs_captured=False;self._init_candidates()
            def capture(self,factory):
                self.graphs={d:None for ds in self._capture_descs.values() for d in ds}
                self._graphs_captured=True
        class Model(Manager):pass
        D.install_cg(NS(CudaGraphManager=Manager,ModelCudaGraphManager=Model,CUDAGraphMode=Mode))
        layouts=[([1,4,12,16],28,170),([1,4,6,12,16],32,198),
                 ([1,4,6,9,12,16],34,216),([1,4,8,12,16],34,220),([1,2,4,6,12,16],33,200)]
        for sizes,count,rows in layouts:
            for layout in ('m12','reuse'):
                with self.subTest(sizes=sizes,layout=layout), \
                        patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout), \
                        patch.object(D,'UNIFORM',True),patch.object(D,'UNIFORM_Q',3):
                    target=Model(4,sizes);target.capture(None)
                    first=Model(4,sizes);first.capture(None)
                    later=Model(1,sizes);later.capture(None)
                    managers=(target,first,later)
                    self.assertEqual((sum(len(m.graphs) for m in managers),
                                      sum(d.num_tokens for m in managers for d in m.graphs)),(count,rows))
                    self.assertIn(Desc(Mode.FULL,6,2,3),target.graphs)
                    short=sorted((d.num_tokens,d.num_reqs,d.uniform_token_count) for d in target.graphs if d.short_context)
                    self.assertEqual(short,[(2,1,2),(3,1,3),(4,1,4),(12,4,3)])
                    for manager in (target,first):
                        for n in (1,2,3,4):
                            for q in (2,3,4):
                                # Includes K3 synthetic warmup under K2. The
                                # fallback must refuse uncaptured short widths.
                                ok=D.draft_eligible([100]*n,n,n*q,q,0)
                                self.assertEqual(ok,n==1 or (n==4 and q==3))
                                for shortcut in (False,True):
                                    with D.context(shortcut and ok):desc=manager.dispatch(n,n*q,q,0)
                                    if desc.cg_mode==Mode.FULL:self.assertIn(desc,manager.graphs)
                                    if n==1:
                                        self.assertEqual(desc,Desc(Mode.FULL,q,1,q,short_context=shortcut))
                                    if n==4 and q==4:
                                        self.assertEqual((desc.cg_mode,desc.num_tokens,desc.short_context),(Mode.FULL,16,False))
                                    if n>1 and q==3:
                                        padded=(12 if layout=='m12' else (6 if n==2 else 9 if n==3 and (8 in sizes or 9 in sizes) else 12))
                                        self.assertEqual((desc.cg_mode,desc.num_tokens,desc.num_reqs),
                                                         (Mode.FULL,padded,padded//3))
                                        self.assertEqual(desc.short_context,shortcut and n==4)
                        # Removing an actual selected descriptor must fail here,
                        # before a coordinator can ever reach graph replay.
                        for n,short_context in ((1,True),(2,False),(4,True),(4,False)):
                            with D.context(short_context):desc=manager.dispatch(n,n*3,3,0)
                            graph=manager.graphs.pop(desc)
                            try:
                                with D.context(short_context),self.assertRaisesRegex(RuntimeError,'descriptor missing'):
                                    manager.dispatch(n,n*3,3,0)
                            finally:manager.graphs[desc]=graph
                    for n in (1,2,3,4):
                        for shortcut in (False,True):
                            ok=D.draft_eligible([100]*n,n,n,1,1)
                            with D.context(shortcut and ok):desc=later.dispatch(n,n,1,0)
                            self.assertEqual(desc.cg_mode,Mode.FULL)
                            self.assertIn(desc,later.graphs)
        # Policies 0/1 ignore the new layout switch, preserving stock K3.
        for uniform in ('0','1'):
            with patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH=uniform), \
                    patch.object(D,'UNIFORM',uniform=='1'),patch.object(D,'UNIFORM_Q',4):
                for layout in ('m12','reuse'):
                    with patch.dict(os.environ,GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout):
                        manager=Model(4,[1,4,16]);manager.capture(None)
                        for n in (1,2,3,4):
                            with D.context(False):desc=manager.dispatch(n,n*4,4,0)
                            self.assertEqual((desc.cg_mode,desc.num_tokens),(Mode.FULL,4 if n==1 else 16))
                            self.assertIn(desc,manager.graphs)


    def test_uniform_batches_keep_uncaptured_c4_widths_on_the_stock_path(self):
        # Fleet 2026-10-03: with k2 uniform batches only c4 q3 has a short graph; a warm-up c4 q4 rectangle
        # selected the short context and found no captured descriptor. Eligibility now refuses such shapes.
        for uniform_q, ok4 in ((3, (3,)), (4, (4,))):
            with patch.object(D, 'UNIFORM', True), patch.object(D, 'UNIFORM_Q', uniform_q):
                for q in (2, 3, 4):
                    self.assertEqual(D.captured_shape(4, q), q in ok4, (uniform_q, q))
                    self.assertTrue(D.captured_shape(1, q))
                    self.assertEqual(D.draft_eligible([100] * 4, 4, 4 * q, q, 0), q in ok4)
                self.assertTrue(D.draft_eligible([100] * 4, 4, 4, 1, 1))
        with patch.object(D, 'UNIFORM', False):
            self.assertTrue(all(D.captured_shape(4, q) for q in (2, 3, 4)))


class Sampling(unittest.TestCase):
    def test_controlflow_cache_dtypes_all_uniform_sample_combinations(self):
        import torch
        import kstop_runtime as rt
        from kstop_compat_controlflow import cpu_gumbel
        # Execute the pinned stock dtype contract without importing vLLM.
        stock=ast.parse((SRC/'vllm/v1/worker/gpu/spec_decode/speculator.py').read_text())
        dtype_method=next(n for n in ast.walk(stock) if isinstance(n,ast.FunctionDef)
                          and n.name=='draft_logits_spec')
        namespace={}
        exec('from __future__ import annotations\n'+ast.unparse(dtype_method),namespace)
        class Speculator:
            def __init__(self,*a):pass
            def propose(self,*a):pass
            def _configure_fused_multi_step_decode(self):pass
        rt.install_mtp(Speculator)
        idx=torch.tensor([5,-1,1,7],dtype=torch.int32)
        positions=torch.tensor([100,999,200,300])
        temperatures=torch.zeros(8)  # real control-flow harness is T=0
        seeds=torch.arange(8)
        raw=torch.tensor([[3.14159,1.,0.],[-2.,8.,0.],[0.,4.12345,1.],[2.,2.,0.]])
        for uniform in ('0','1'):
            for sample in ('0','1'):
                for head_dtype in (torch.bfloat16,torch.float32):
                    dtype,fill=namespace['draft_logits_spec'](None,NS(model_config=NS(head_dtype=head_dtype)))
                    self.assertEqual((dtype,fill),(head_dtype,0.0))
                    for logits_dtype in (torch.bfloat16,torch.float32):
                        for step in range(3):
                            for col in (torch.tensor(step),torch.tensor([step])):
                                with self.subTest(uniform=uniform,sample=sample,cache=dtype,
                                                  logits=logits_dtype,step=step,col_shape=col.shape), \
                                     patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH=uniform,GLM_SPEC_SAMPLE=sample):
                                    logits=raw.to(logits_dtype);original=logits.clone()
                                    owner=NS(model=NS(compute_logits=lambda h:logits),use_fp64_gumbel=False,
                                             max_num_reqs=8,device='cpu',speculative_config=NS(
                                                 draft_sample_method='probabilistic' if sample=='1' else 'greedy'))
                                    owner._kstop=rt.Runtime(owner)
                                    self.assertEqual(owner._kstop.uniform,uniform=='1')
                                    cache=torch.full((8,3,3),float('nan'),dtype=dtype) if sample=='1' else None
                                    calls=[]
                                    def native(*args,**kwargs):
                                        calls.append((args,kwargs))
                                        return cpu_gumbel(*args,**kwargs)
                                    with patch.dict(sys.modules,{'vllm.v1.worker.gpu.spec_decode.speculator':NS(gumbel_sample=native)}):
                                        result=Speculator.sample_draft(owner,None,positions,idx,temperatures,seeds,col,cache)
                                    self.assertTrue(torch.equal(result,logits.argmax(-1)))
                                    self.assertTrue(torch.equal(logits,original))
                                    z=logits.float()
                                    self.assertTrue(torch.equal(owner._kstop.confidence[:4],
                                        (z.amax(-1)-z.logsumexp(-1)).exp()))
                                    self.assertEqual(len(calls),int(sample))
                                    if cache is not None:
                                        expected=torch.full_like(cache,float('nan'))
                                        expected[idx[idx>=0].long(),step]=logits[idx>=0].to(dtype)
                                        torch.testing.assert_close(cache,expected,rtol=0,atol=0,equal_nan=True)
                                        self.assertEqual(cache.dtype,head_dtype)
                                        self.assertIs(calls[0][0][0],logits)
                                        self.assertTrue(torch.equal(calls[0][0][4],positions+1))
                                        self.assertIs(calls[0][1]['logits_cache'],cache)

    def test_draft_gumbel_call_matches_stock_image(self):
        def call(source, function):
            tree=ast.parse(source)
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==function)
            return next(n for n in ast.walk(fn) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Name) and n.func.id=='gumbel_sample')
        stock=(SRC/'vllm/v1/worker/gpu/spec_decode/speculator.py').read_text()
        runtime=(ROOT/'overlay/kstop/kstop_runtime.py').read_text()
        # Binds logits, mapping, temperature, seeds, position+1, raw cache
        # column and fp64 selection to the stock call without importing vLLM.
        self.assertEqual(ast.dump(call(stock,'sample_draft')),ast.dump(call(runtime,'sample')))

    def test_raw_max_q_and_one_native_cache_write(self):
        import torch
        import kstop_runtime as rt
        class Speculator:
            def __init__(self,*a):pass
            def propose(self,*a):pass
            def _configure_fused_multi_step_decode(self):pass
        rt.install_mtp(Speculator)
        logits=torch.tensor([[3.,1.,0.],[0.,4.,1.],[2.,2.,0.]])
        owner=NS(model=NS(compute_logits=lambda h:logits),use_fp64_gumbel=False,
                 _kstop=NS(confidence=torch.zeros(3)))
        cache=torch.full((8,3,3),float('nan'));idx=torch.tensor([5,1,7])
        temp=torch.tensor([1.,.5,1.,1.,1.,0.,1.,2.]);seeds=torch.arange(8)
        module=NS()
        calls=[]
        def native(z,m,t,s,p,**kw):
            calls.append((z,m,t,s,p,kw))
            kw['logits_cache'][m,kw['logits_cache_col']]=z
            return torch.tensor([0,2,1])  # includes a non-max sampled token
        module.gumbel_sample=native
        with patch.dict(sys.modules,{'vllm.v1.worker.gpu.spec_decode.speculator':module}):
            result=Speculator.sample_draft(owner,None,torch.tensor([100,200,300]),idx,temp,seeds,1,cache)
        self.assertEqual(result.tolist(),[0,2,1])
        self.assertTrue(torch.equal(owner._kstop.confidence,(logits.amax(-1)-logits.logsumexp(-1)).exp()))
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][4].tolist(),[101,201,301])
        self.assertTrue(torch.equal(cache[idx,1],logits))
        self.assertTrue(torch.isnan(cache[:,0]).all() and torch.isnan(cache[:,2]).all())
        greedy=Speculator.sample_draft(owner,None,None,None,None,None,None,None)
        self.assertTrue(torch.equal(greedy,logits.argmax(-1)))

    def test_guard_binds_seeds_temperature_and_shortcut_bounds(self):
        import numpy as np
        import torch
        import kstop_runtime as rt
        ss=NS(**{k:NS(np=np.array(v)) for k,v in dict(temperature=[0.,1.],top_p=[1.,.9],min_p=[0.,0.],
             top_k=[3,2],seeds=[11,99]).items()})
        b=NS(req_ids=['b'],num_reqs=1,query_start_loc_np=np.array([0,4]),is_prefilling_np=[False],
             idx_mapping_np=np.array([1]),seq_lens_cpu_upper_bound=torch.tensor([100]),has_prefill=False,num_tokens_after_padding=4)
        r=rt.Runtime(NS(max_num_reqs=1,device='cpu'))
        r.sample_mode='probabilistic';r.shortcut=True;r.proposals={'b':3}
        with patch.object(rt,'STATE',dict(runner=NS(sampler=NS(sampling_states=ss)))):
            _,valid,base=rt._inputs(r,b);self.assertTrue(valid)
            for key,value in (('seeds',100),('temperature',.7)):
                old=getattr(ss,key).np[1];getattr(ss,key).np[1]=value
                self.assertNotEqual(base,rt._inputs(r,b)[2],key)
                getattr(ss,key).np[1]=old
            b.seq_lens_cpu_upper_bound+=1
            self.assertNotEqual(base,rt._inputs(r,b)[2])
        with patch.object(rt,'STATE',{}):self.assertFalse(rt._inputs(r,b)[1])


if __name__=='__main__':unittest.main(verbosity=2)
