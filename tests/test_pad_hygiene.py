# SPDX-License-Identifier: Apache-2.0
"""Pinned Marlin top-k hook exactness, with CPU expert math; no CUDA claim."""
import ast
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from enum import Enum
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/kstop'),str(ROOT/'overlay/bringup')]
import deadrow_ops
import glm_mtp_kstop as K
import kstop_runtime as R
SRC=Path(os.environ['GLM_IMAGE_SRC'])


def capture_plan(q,sizes,varlen=False):
    """Execute the pinned/composed builder, dispatcher AND capture loop."""
    import glm_dsa_short as D
    class Mode(Enum):
        NONE=0;FULL=1;PIECEWISE=2;FULL_DECODE_ONLY=3
        def decode_mode(self):return Mode.FULL
        def mixed_mode(self):return Mode.NONE
        def separate_routine(self):return True
        def __bool__(self):return self!=Mode.NONE
    raw=(SRC/(K.CG.replace('.','/')+'.py')).read_text()
    with patch.multiple(D,ENABLED=True,KSTOP=True):
        tree=ast.parse(K.combined_transform(K.CG,raw))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CudaGraphManager')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
              ('_init_candidates','_resolve_effective_loras','dispatch','capture')]
    for fn in cls.body:fn.decorator_list=[]
    nodes=[n for n in tree.body if getattr(n,'name',None) in
           ('BatchExecutionDescriptor','_is_compatible')]+[cls]
    ns=dict(os=os,_kstop=R,dataclass=dataclass,CUDAGraphMode=Mode,
            defaultdict=defaultdict,product=product,round_up=lambda n,q:-(-n//q)*q,
            graph_capture=lambda **kw:nullcontext(),is_global_first_rank=lambda:False,
            logger=NS(debug=lambda *a:None),torch=torch,
            compilation_counter=NS(num_cudagraph_captured=0),
            get_offloader=lambda:NS(sync_prev_onload=lambda:None,join_after_forward=lambda:None),
            set_graph_pool_id=lambda *a:None,current_platform=NS(graph_pool_handle=lambda:None))
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),ns)
    class Manager(ns['CudaGraphManager']):
        def __init__(self):
            self.compilation_config=NS(cudagraph_capture_sizes=sizes,max_cudagraph_capture_size=max(sizes))
            self.vllm_config=NS(speculative_config=NS(uses_dynamic_speculative_decoding=lambda:False))
            self.cudagraph_mode=Mode.FULL_DECODE_ONLY;self.decode_query_len=q;self.max_num_reqs=4
            self.lora_capture_cases=[0];self.varlen_decode=varlen
            self._candidates={};self._capture_descs={};self._lora_dispatch_map={};self._max_lora_case=0
            self.graphs={};self._graphs_captured=False;self.device='cpu';self.pool=None
            self.use_breakable_cg=False;self._init_candidates()
    class Model(Manager):pass
    if not varlen:
        D.install_cg(NS(CudaGraphManager=Manager,ModelCudaGraphManager=Model,CUDAGraphMode=Mode))
    return Model(),Mode,ns['_is_compatible']


def method(name,which):
    raw=(SRC/(name.replace('.','/')+'.py')).read_text()
    tree=ast.parse(K.transform(name,raw))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==which)
    fn.decorator_list=[];fn.returns=None
    for a in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs):a.annotation=None
    scope=dict(torch=torch,_kstop=R,SharedExpertsOrder=NS(NO_OVERLAP=0,MULTI_STREAM_OVERLAPPED=1))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),name,'exec'),scope)
    return scope[which]


def state(flag,m=32):
    with patch.dict(os.environ,GLM_PAD_HYGIENE=str(flag)):
        r=R.Runtime(NS(max_num_reqs=4,device='cpu'))
    R.STATE=dict(runtime=r,rows=torch.arange(m),src=torch.arange(m),
                 dead=torch.zeros(m,dtype=torch.bool),identity=True)
    return r


class Padding(unittest.TestCase):
    def tearDown(self):R.STATE=None

    def test_per_descriptor_table_and_c1_graphs_contain_no_remap_call(self):
        import glm_dsa_short as D
        launcher=ast.parse((ROOT/'scripts/cluster.py').read_text())
        plan=next(n for n in launcher.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='KSTOP_K2_CAPTURE_SIZES' for t in n.targets))
        sizes=ast.literal_eval(plan.value)
        apply=method(K.MOE,'_apply_quant_method')
        active=[]
        class Graph:
            def __init__(self):self.remaps=0
        class Recording:
            def __init__(self,g,*a):self.g=g
            def __enter__(self):active.append(self.g)
            def __exit__(self,*a):active.pop()
        op=torch.ops.glm_deadrow.remap_
        def record(*args):
            if active:active[-1].remaps+=1
            return op(*args)
        def factory(d,warmup):
            weights=torch.ones(d.num_tokens,8);ids=torch.zeros(d.num_tokens,8,dtype=torch.long)
            layer=NS(_shared_experts=None,_maybe_apply_shared_experts=lambda *a:None,
                _quant_method=NS(topk_indices_dtype=torch.int64),
                router=NS(select_experts=lambda **kw:(weights,ids)),
                routed_experts=NS(quant_method=NS(is_monolithic=False),forward_modular=lambda **kw:kw['x']))
            return lambda mode:apply(layer,torch.ones(d.num_tokens,32),torch.ones(d.num_tokens,256),None)
        for layout in ('reuse','m12'):
            with patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout), \
                    patch.multiple(D,ENABLED=True,KSTOP=True,UNIFORM=True,UNIFORM_Q=3):
                for role,q in (('target',4),('first_mtp',4),('later_mtp',1)):
                    state(1)
                    manager,mode,compatible=capture_plan(q,sizes)
                    with patch.object(torch.cuda,'CUDAGraph',Graph),patch.object(torch.cuda,'graph',Recording), \
                            patch.object(torch.ops.glm_deadrow,'remap_',side_effect=record):
                        manager.capture(factory)
                    table={(d.num_tokens,d.num_reqs,d.uniform_token_count,d.short_context):g.remaps
                           for d,g in manager.graphs.items()}
                    stock={(2,1,2,False):0,(3,1,3,False):0,(4,1,4,False):0,
                           (4,2,2,False):0,(6,2,3,False):0,
                           (12,4,3,False):1,(12,3,4,False):1,(16,4,4,False):0}
                    if q==1:
                        expected={(1,1,1,False):0,(4,4,1,False):1,
                                  (1,1,1,True):0,(4,4,1,True):0}
                    else:
                        expected=dict(stock)
                        expected.update({(2,1,2,True):0,(3,1,3,True):0,
                                         (4,1,4,True):0,(12,4,3,True):0})
                    self.assertEqual(table,expected,(role,layout))
                    self.assertIsNone(R.STATE.get('capture_pad'))
                    # Independent dispatch oracle: enumerate physically valid
                    # uniform shapes through the actual composed dispatcher.
                    selected={d:False for d in manager.graphs}
                    for n in range(1,5):
                        for width in range(1,q+1):
                            for short in (False,True):
                                eligible=D.draft_eligible([100]*n,n,n*width,width,0 if q==4 else 1)
                                with D.context(short and eligible):d=manager.dispatch(n,n*width,width,0)
                                if d in selected:selected[d]|=n*width<d.num_tokens
                    self.assertEqual({d:bool(g.remaps) for d,g in manager.graphs.items()},selected)
                    # c1 graph hook executes during capture but contains zero
                    # remap calls for every q and both DSA contexts.
                    for width in ((1,) if q==1 else (2,3,4)):
                        for short in (False,True):
                            with D.context(short):d=manager.dispatch(1,width,width,0)
                            self.assertEqual(manager.graphs[d].remaps,0)
                    # c2 reuse exact; c3 padded; c4 short exact M12.
                    if q==4 and layout=='reuse':
                        with D.context(False):
                            self.assertEqual(manager.graphs[manager.dispatch(2,6,3,0)].remaps,0)
                            self.assertEqual(manager.graphs[manager.dispatch(3,9,3,0)].remaps,1)
                        with D.context(True):
                            self.assertEqual(manager.graphs[manager.dispatch(4,12,3,0)].remaps,0)

    def test_alternate_plans_and_eager_exact_widths(self):
        import glm_dsa_short as D
        for sizes in ([1,4,16],[1,4,6,12,16],[1,4,6,9,12,16],[1,4,8,12,16],[1,2,4,6,12,16]):
            for policy in ('0','1','k2'):
                for layout in ('reuse','m12'):
                    with patch.dict(os.environ,GLM_MTP_KSTOP_UNIFORM_BATCH=policy,GLM_MTP_KSTOP_CAPTURE_LAYOUT=layout), \
                            patch.multiple(D,ENABLED=True,KSTOP=True,UNIFORM=policy!='0',UNIFORM_Q=3 if policy=='k2' else 4):
                        for q in (1,4):
                            m,mode,compatible=capture_plan(q,sizes)
                            m.graphs={d:None for ds in m._capture_descs.values() for d in ds};m._graphs_captured=True
                            padded=set()
                            for n in range(1,5):
                                for width in range(1,q+1):
                                    for short in (False,True):
                                        eligible=D.draft_eligible([100]*n,n,n*width,width,0 if q==4 else 1)
                                        with D.context(short and eligible):d=m.dispatch(n,n*width,width,0)
                                        if d.cg_mode==mode.FULL and d.num_tokens>n*width:padded.add(d)
                            self.assertEqual({d for d in m.graphs if R.capture_can_pad(m,d,compatible)},padded)
        state(1)
        for live,padded in ((2,2),(3,3),(4,4),(6,6),(12,12),(16,16),(6,12),(3,4)):
            R.prepare_tail(live,padded,reset=True)
            with patch.object(torch.ops.glm_deadrow,'remap_') as op:
                R.remap(torch.ones(padded,8),torch.zeros(padded,8,dtype=torch.long))
                self.assertEqual(op.call_count,int(live<padded))
        # Existing heterogeneous verify remap is required even without tails.
        R.STATE['identity']=False
        with patch.object(torch.ops.glm_deadrow,'remap_') as op:
            R.remap(torch.ones(4,8),torch.zeros(4,8,dtype=torch.long));op.assert_called_once()

    def test_varlen_plan_and_capture_scope_restoration(self):
        m,mode,compatible=capture_plan(4,[1,4,12,16],varlen=True)
        m._graphs_captured=True
        table={d:R.capture_can_pad(m,d,compatible) for ds in m._capture_descs.values() for d in ds}
        self.assertEqual({d.num_tokens:v for d,v in table.items()},{1:False,4:True,12:True,16:True})
        oracle={d:False for d in table}
        for n in range(1,5):
            for tokens in range(n,n*4+1):
                d=m.dispatch(n,tokens,None,0,4)
                if d in oracle:oracle[d]|=tokens<d.num_tokens
        self.assertEqual(table,oracle)
        state(1)
        d=next(d for d in table if d.num_tokens==12)
        def fail(mode):
            self.assertTrue(R.STATE['capture_pad'])
            raise ValueError('capture fixture')
        run=R.capture_forward(m,d,compatible,fail)
        with self.assertRaisesRegex(ValueError,'capture fixture'):run(mode.NONE)
        self.assertIsNone(R.STATE.get('capture_pad'))

    def test_flag_off_never_reads_padding_fields(self):
        class OldBatch(NS):
            def __getattribute__(self,name):
                if name in ('num_tokens','num_tokens_after_padding'):
                    raise AssertionError('padding-only field read with flag off: '+name)
                return super().__getattribute__(name)
        for dummy,warmup in ((False,False),(True,False),(False,True)):
            with self.subTest(dummy=dummy,warmup=warmup):
                r=state(0);r.owner._kstop_warmup=warmup
                r.control='broadcast';r.proposals={'a':1,'b':3}
                b=OldBatch(num_reqs=2,req_ids=['a','b'],query_start_loc_np=[0,4,8],
                           is_prefilling_np=[False,False])
                R.STATE['src'].zero_();R.STATE['dead'].fill_(True)
                with patch.object(R,'prepare_tail') as tail,patch.object(r,'agree') as agree:
                    R.prepare(NS(device='cpu'),b,dummy)
                    tail.assert_not_called()
                if dummy or warmup:
                    agree.assert_not_called()
                    self.assertTrue(R.STATE['identity'])
                    self.assertEqual(R.STATE['src'].tolist(),list(range(32)))
                    self.assertFalse(R.STATE['dead'].any())
                else:
                    agree.assert_called_once_with('prepare',
                        dict(request_ids=['a','b'],widths=[4,4],prefilling=[False,False],
                             proposals=[('a',1),('b',3)]),True)
                    self.assertEqual(R.STATE['src'][:12].tolist(),[0,1,0,0,4,5,6,7,8,9,10,11])
                    self.assertEqual(R.STATE['dead'][:12].tolist(),[False,False,True,True]+[False]*8)
                    self.assertEqual(r.proposals,{})

    def test_flag_on_with_pinned_input_batch_fields(self):
        # Execute the pinned dataclass fields; omit only GPU/Triton methods and
        # annotations so the actual InputBatch constructor can run on the Mac.
        raw=(SRC/'vllm/v1/worker/gpu/input_batch.py').read_text()
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                         '3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410')
        cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='InputBatch')
        cls.body=[n for n in cls.body if isinstance(n,ast.AnnAssign)]
        for n in cls.body:n.annotation=ast.Name(id='object',ctx=ast.Load())
        scope=dict(dataclass=dataclass,__name__=__name__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'InputBatch','exec'),scope)
        batch_type=scope['InputBatch']
        values={f.name:None for f in fields(batch_type)}
        values.update(num_reqs=2,num_reqs_after_padding=4,req_ids=['a','b'],
                      num_tokens=6,num_tokens_after_padding=12,
                      query_start_loc_np=[0,3,6],is_prefilling_np=[False,False])
        for stub in (batch_type,NS):
            with self.subTest(stub=stub.__name__):
                b=stub(**values)
                r=state(1);r.control='broadcast';r.proposals={'a':2,'b':2}
                with patch.object(r,'agree') as agree:R.prepare(NS(device='cpu'),b,False)
                agree.assert_called_once_with('prepare',
                    dict(request_ids=['a','b'],widths=[3,3],prefilling=[False,False],
                         proposals=[('a',2),('b',2)],**{'pad-hygiene':[6,12]}),True)
                self.assertEqual(R.STATE['src'][:12].tolist(),list(range(6))+[0]*6)
                self.assertEqual(R.STATE['dead'][:12].tolist(),[False]*6+[True]*6)
                self.assertEqual(r.proposals,{})

    def test_pinned_topk_hook_exactness_changing_replay_tail(self):
        apply=method(K.MOE,'_apply_quant_method')
        # Substitute only router/expert tensor kernels. Execute the actual
        # source-pinned transformed modular/Marlin hook immediately before math.
        generator=torch.Generator().manual_seed(20261003)
        x=torch.randn(16,32,generator=generator)
        z=torch.randn(16,256,generator=generator)
        def run(flag,live,padded,reset):
            r=state(flag)
            ptr=R.STATE['src'].data_ptr()
            weights,ids=z[:padded].topk(8)
            weights=weights.softmax(-1)
            snapshots=(weights.clone(),ids.clone())
            routed={}
            def experts(**kw):
                w,i=kw['topk_weights'],kw['topk_ids']
                routed.update(w=w.clone(),ids=i.clone())
                # Fixed top-k reduction order, affine CPU expert substitutes.
                parts=kw['x'][:,None,:]*(i[:,:,None].float()+1)
                return (parts*w[:,:,None]).sum(1)
            layer=NS(_shared_experts=None,_maybe_apply_shared_experts=lambda *a:None,
                _quant_method=NS(topk_indices_dtype=torch.int64),
                router=NS(select_experts=lambda **kw:(weights,ids)),
                routed_experts=NS(quant_method=NS(is_monolithic=False),forward_modular=experts))
            # Use target preparation for target, reset=True for both draft roles.
            if reset:R.prepare_tail(live,padded,reset=True)
            else:R.prepare(NS(),NS(num_tokens=live,num_tokens_after_padding=padded),True)
            result=apply(layer,x[:padded],z[:padded],None)[1]
            self.assertEqual(ptr,R.STATE['src'].data_ptr())
            self.assertTrue(torch.equal(snapshots[0][:live],routed['w'][:live]))
            self.assertTrue(torch.equal(snapshots[1][:live],routed['ids'][:live]))
            if flag:
                self.assertTrue(torch.equal(routed['ids'][live:],ids[:1].expand(padded-live,-1)))
                self.assertTrue(torch.equal(routed['w'][live:],weights[:1].expand(padded-live,-1)))
                self.assertTrue(set(routed['ids'][live:].flatten().tolist())<=set(ids[0].tolist()))
                self.assertTrue(R.STATE['dead'][live:padded].all())
                # Sampler live indices are unaffected; tails cannot be sampled.
                tokens=torch.arange(padded)
                self.assertTrue(torch.equal(R.mask_drafts(tokens,torch.arange(padded))[:live],tokens[:live]))
                self.assertTrue((R.mask_drafts(tokens,torch.arange(padded))[live:]==-1).all())
            else:self.assertTrue(torch.equal(snapshots[1],routed['ids']))
            return result[:live]
        for live,padded in ((6,12),(9,12),(2,2),(3,3),(4,4),(2,4),(3,4),(12,12),(16,16)):
            for role in ('target','first_mtp','later_mtp'):
                with self.subTest(live=live,padded=padded,role=role):
                    self.assertTrue(torch.equal(run(0,live,padded,role!='target'),run(1,live,padded,role!='target')))

    def test_captured_remap_reads_stable_map_after_identity_capture(self):
        state(1)
        ptr=R.STATE['src'].data_ptr()
        def captured():
            ids=torch.arange(96).reshape(12,8);weights=ids.float()
            R.remap(weights,ids)
            return ids
        m,mode,compatible=capture_plan(4,[1,4,12,16])
        d=next(d for ds in m._capture_descs.values() for d in ds
               if d.num_tokens==12 and d.uniform_token_count==3 and not d.short_context)
        replay=R.capture_forward(m,d,compatible,lambda mode:captured())
        for live in (12,6,9,12,3,6):
            R.prepare_tail(live,12,reset=True)
            ids=replay(mode.NONE)
            self.assertTrue(torch.equal(ids[:live],torch.arange(96).reshape(12,8)[:live]))
            self.assertTrue(torch.equal(ids[live:],ids[:1].expand(12-live,-1)))
            self.assertEqual(ptr,R.STATE['src'].data_ptr())
        R.STATE['runtime'].pad_hygiene=False
        with patch.object(torch.ops.glm_deadrow,'remap_') as op:
            R.remap(torch.ones(12,8),torch.zeros(12,8,dtype=torch.long));op.assert_not_called()

    def test_heterogeneous_map_and_draft_reset(self):
        r=state(1);r.proposals=dict(a=1,b=3);r.control='broadcast';r.agree=lambda *a:None
        b=NS(num_tokens=8,num_tokens_after_padding=12,num_reqs=2,req_ids=['a','b'],
             query_start_loc_np=[0,4,8],is_prefilling_np=[False,False])
        R.prepare(NS(device='cpu'),b,False)
        self.assertEqual(R.STATE['src'][:12].tolist(),[0,1,0,0,4,5,6,7,0,0,0,0])
        R.prepare_tail(2,4,reset=True)
        self.assertEqual(R.STATE['src'][:12].tolist(),[0,1,0,0,4,5,6,7,8,9,10,11])
        self.assertEqual(R.STATE['dead'][:12].tolist(),[False,False,True,True]+[False]*8)

    def test_boot_validation_cold_digest_and_prepare_counts(self):
        for bad in ('','yes','2'):
            with self.assertRaisesRegex(ValueError,'GLM_PAD_HYGIENE'):
                K.register(dict(GLM_MTP_KSTOP='0',GLM_PAD_HYGIENE=bad))
        with self.assertRaisesRegex(ValueError,'requires GLM_MTP_KSTOP'):
            K.register(dict(GLM_PAD_HYGIENE='1'))
        payloads=[]
        for flag in (0,1):
            r=state(flag);r.agree=lambda stage,payload,valid:payloads.append(payload)
            with patch.dict(os.environ,GLM_MTP_KSTOP_CONTROL=str(ROOT/'overlay/kstop/control.json')):r.load_control()
        self.assertEqual(payloads[1],payloads[0]+['pad-hygiene'])
        r=state(1);r.proposals={'a':2}
        b=NS(num_reqs=1,req_ids=['a'],query_start_loc_np=[0,3],is_prefilling_np=[False],num_tokens=3,num_tokens_after_padding=4)
        _,ok,payload=R._inputs(r,b)
        self.assertTrue(ok)
        self.assertEqual(payload,dict(request_ids=['a'],widths=[3],prefilling=[False],
            proposals=[('a',2)],**{'pad-hygiene':[3,4]}))

    def test_named_padding_counts_remain_bound_in_guard_digests(self):
        b=NS(num_reqs=1,req_ids=['a'],query_start_loc_np=[0,3],
             is_prefilling_np=[False],num_tokens=3,num_tokens_after_padding=4)
        r=state(1);r.proposals={'a':2}
        _,valid,payload=R._inputs(r,b)
        self.assertTrue(valid)
        r.fold('prepare',payload,valid)
        r.guard_confidence=torch.zeros(R.GUARD_SLOTS,dtype=torch.int64)
        baseline=r.trail
        fields=r.guard_field_digests()
        self.assertEqual(fields['0:prepare.pad-hygiene'],
            hashlib.sha256(json.dumps([3,4],sort_keys=True,allow_nan=True).encode()).hexdigest())
        for key in ('request_ids','widths','prefilling','proposals'):
            self.assertIn('0:prepare.'+key,fields)
        peer=[]
        def gather(rows,row,group):
            if not peer:peer.append(row.clone())
            rows[0].copy_(peer[0])
        group=NS(ranks=[0],cpu_group=None)
        with patch.object(R,'tp_group',return_value=group), \
                patch.object(torch.distributed,'all_gather',side_effect=gather):
            r.agree('prepare',payload,valid)
        for live,padded in ((2,4),(3,5)):
            with self.subTest(live=live,padded=padded):
                b.num_tokens=live;b.num_tokens_after_padding=padded
                r=state(1);r.proposals={'a':2}
                _,valid,payload=R._inputs(r,b)
                self.assertTrue(valid)
                r.fold('prepare',payload,valid)
                r.guard_confidence=torch.zeros(R.GUARD_SLOTS,dtype=torch.int64)
                self.assertNotEqual(r.trail,baseline)
                self.assertNotEqual(r.guard_field_digests()['0:prepare.pad-hygiene'],
                                    fields['0:prepare.pad-hygiene'])
                with patch.object(R,'tp_group',return_value=group), \
                        patch.object(torch.distributed,'all_gather',side_effect=gather), \
                        self.assertRaisesRegex(RuntimeError,'collective-safe kstop refusal at prepare'):
                    r.agree('prepare',payload,valid)

    def test_padding_preserves_sampling_and_shortcut_guard_fields(self):
        r=state(1);r.proposals={'a':2};r.sample_mode='probabilistic';r.shortcut=True
        ss=NS(temperature=NS(np=[1.]),seeds=NS(np=[17]))
        R.STATE['runner']=NS(sampler=NS(sampling_states=ss))
        b=NS(num_reqs=1,req_ids=['a'],query_start_loc_np=[0,3],
             is_prefilling_np=[False],num_tokens=3,num_tokens_after_padding=4,
             idx_mapping_np=[0],seq_lens_cpu_upper_bound=torch.tensor([11]),has_prefill=False)
        _,valid,payload=R._inputs(r,b)
        self.assertTrue(valid)
        self.assertEqual(payload,{
            'request_ids':['a'],'widths':[3],'prefilling':[False],'proposals':[('a',2)],
            'sampling.temperature':[('a',float(1).hex())],
            'sampling.seeds':[('a',17)],'sampling.lengths':[('a',11)],'pad-hygiene':[3,4],
            'short-dsa.lengths':[11],'short-dsa.prefill':False,'short-dsa.padding':4})
        ss.temperature.np[0]=float('nan')
        _,valid,payload=R._inputs(r,b)
        self.assertFalse(valid)
        self.assertTrue(payload['sampling.missing'])
        self.assertEqual(payload['pad-hygiene'],[3,4])

    def test_native_later_mtp_loop_updates_map_before_every_replay(self):
        native=method(K.AR,'_multi_step_decode')
        calls=[]
        r=state(1)
        # The metadata kernel/graph math are substitutes; the native loop and
        # transformed hook execute unchanged, including step ordering.
        sp=NS(num_speculative_steps=3,advance_draft_positions=True,
              _kstop=NS(pad_hygiene=True,can_advance=lambda *a:True,arm=lambda *a:None,after_metadata=lambda *a:True),
              input_buffers=NS(positions=torch.arange(32),query_start_loc=torch.arange(5)),
              idx_mapping=torch.arange(4),current_draft_step=torch.tensor(0),
              decode_cudagraph_manager=NS(run_fullgraph=lambda desc:calls.append(R.STATE['src'][:4].clone())))
        native.__globals__['CUDAGraphMode']=NS(FULL=1)
        for live in (4,2,3,1,4):
            calls.clear()
            native(sp,live,True,NS(num_tokens=4,cg_mode=1),None,torch.ones(live))
            self.assertEqual(len(calls),2)
            expected=torch.arange(4);expected[live:]=0
            for src in calls:self.assertTrue(torch.equal(src,expected))
        sp._kstop.pad_hygiene=False
        class OldDescriptor:
            cg_mode=1
            @property
            def num_tokens(self):raise AssertionError('draft padding count read with flag off')
        with patch.object(R,'prepare_tail') as tail:
            native(sp,2,True,OldDescriptor(),None,torch.ones(2))
            tail.assert_not_called()

    def test_native_padding_safety_contracts(self):
        raw=(SRC/(K.RUNNER.replace('.','/')+'.py')).read_text()
        self.assertIn('num_tokens_padded=input_batch.num_tokens_after_padding',raw)
        self.assertIn('query_start_loc_np[num_reqs + 1 :] = num_tokens',raw)
        bt=(SRC/'vllm/v1/worker/gpu/block_table.py').read_text()
        self.assertIn('actual_num_tokens = tl.load(query_start_loc + batch_idx)',bt)
        self.assertIn('tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)',bt)
        self.assertIn('PAD_ID=PAD_SLOT_ID',bt)
        self.assertIn('PAD_SLOT_ID = -1',(SRC/'vllm/v1/attention/backends/utils.py').read_text())
        ar=(SRC/(K.AR.replace('.','/')+'.py')).read_text()
        self.assertIn('tl.store(draft_seq_lens_ptr + block, 0, mask=mask)',ar)
        self.assertIn('tl.store(seq_lens_ptr + block, 0, mask=mask)',ar)
        self.assertIn('tl.store(last_token_indices_ptr + block, 0, mask=mask)',ar)
        indexer=(SRC/'vllm/model_executor/layers/sparse_attn_indexer.py').read_text()
        self.assertIn('topk_indices_buffer[: hidden_states.shape[0]] = -1',indexer)
        self.assertIn('seq_lens = decode_metadata.seq_lens[:batch_size]',indexer)

    def test_pinned_target_and_both_draft_replay_hooks(self):
        for name in K.PINS:
            raw=(SRC/(name.replace('.','/')+'.py')).read_text();compile(K.combined_transform(name,raw),name,'exec')
        ar=K.transform(K.AR,(SRC/(K.AR.replace('.','/')+'.py')).read_text())
        self.assertLess(ar.index('_kstop.prepare_tail(num_tokens,'),ar.index('self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)'))
        self.assertLess(ar.index('_kstop.prepare_tail(num_reqs,'),ar.index('self.decode_cudagraph_manager.run_fullgraph(batch_desc)'))
        guards=[n for n in ast.walk(ast.parse(ar)) if isinstance(n,ast.If)
                and ast.unparse(n.test)=='self._kstop.pad_hygiene']
        calls=[n for guard in guards for n in ast.walk(guard)
               if isinstance(n,ast.Call) and ast.unparse(n.func)=='_kstop.prepare_tail']
        self.assertEqual(len(calls),2)  # both draft argument reads are flag-guarded
        # Fused decode remains explicitly disabled by install_mtp.
        self.assertIn('self.use_fused_multi_step_decode=False',Path(R.__file__).read_text())


if __name__=='__main__':unittest.main(verbosity=2)
