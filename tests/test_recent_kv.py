# SPDX-License-Identifier: Apache-2.0
"""Actual writer/reader/stager AST execution on Mac, plus pinned integration.

No CUDA compiler/replay claim. Exercise aliasing, wrap, request reorder, APC
misses, rejected-tail writes, padding and the device scalar used in graphs.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'tests'),str(ROOT/'tests/fixtures')]
import glm_recent_kv as G
from fp4_kv_cpu_model import kernels,Ptr,tensor
from fp4_probe_quant_dc5ad9e import fp4_qdq,fp8_codes


class Recent(unittest.TestCase):
    def setUp(self):
        self.tl,self.k=kernels(ROOT/'overlay/bringup')
        self.cap=12; self.n=96
        self.cache=torch.full((self.n,368),165,dtype=torch.uint8)
        self.shadow=torch.full((4*self.cap,512),165,dtype=torch.uint8)
        self.mapping=torch.full((self.n,),-1,dtype=torch.int32)
        self.tags=torch.full((4*self.cap,),-1,dtype=torch.int32)
        self.pos=torch.full((4*self.cap,),-1,dtype=torch.int32)
        self.lower=torch.zeros(4,dtype=torch.int32)
        self.flag=torch.tensor([4],dtype=torch.int32)
        self.tokenring=torch.full((128,),-1,dtype=torch.int32)
        self.tokenpos=torch.full((128,),-1,dtype=torch.int32)
        self.extra=tuple(Ptr.of(x) for x in (self.shadow,self.mapping,self.tags,self.pos,self.lower,self.flag))+(self.cap,)

    def stage(self,owners,starts,positions):
        owners=torch.tensor(owners,dtype=torch.int64)
        starts=torch.tensor(starts,dtype=torch.int32)
        positions=torch.tensor(positions,dtype=torch.int64)
        self.tl.pid=(0,0)
        self.k['_stage_rows'](*(Ptr.of(x) for x in (starts,owners,positions,self.tokenring,self.tokenpos,self.lower,self.flag)),
                              len(positions),len(owners),self.cap)

    def write(self,x,slots,scale=1.):
        slots=torch.tensor(slots,dtype=torch.int64);rope=torch.zeros(len(slots),64,dtype=torch.bfloat16)
        scale=torch.tensor([scale],dtype=torch.float32)
        for i in range(len(slots)):
            self.tl.pid=(i,0)
            self.k['_pack_store'](*(Ptr.of(t) for t in (x,rope,self.cache,slots,scale)),512,64,368,len(slots),self.n,
                *(Ptr.of(t) for t in (self.shadow,self.mapping,self.tags,self.pos,self.tokenring,self.tokenpos)),self.cap)

    def read(self,slots):
        return torch.from_numpy(self.k['_load_latent'](Ptr.of(self.cache),tensor(slots),tensor(range(512)),368,self.n,*self.extra).copy()).bfloat16()

    def test_stable_request_reorder_and_padding(self):
        self.stage([3,1],[0,2,3],[21,22,70,0])
        self.assertEqual(self.tokenring[:4].tolist(),[3*self.cap+21%self.cap,3*self.cap+22%self.cap,self.cap+70%self.cap,-1])
        self.assertEqual(self.lower.tolist(),[0,67,0,18])
        self.stage([1,3],[0,1,2],[71,23])
        self.assertEqual(self.tokenring[:2].tolist(),[self.cap+71%self.cap,3*self.cap+23%self.cap])

    def test_padded_request_descriptor_does_not_overread_idx_mapping(self):
        # Real V2 idx_mapping has actual request count; graph descriptors may
        # have more padded requests. Execute the actual Python launch wrapper.
        tree=ast.parse((ROOT/'overlay/bringup/glm_recent_kv_kernel.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage_rows')
        test=self
        class Launch:
            def __getitem__(self,grid):
                def run(*args,**kwargs):
                    test.assertEqual(args[-2],1)
                    test.tl.pid=(0,0)
                    test.k['_stage_rows'](*(Ptr.of(x) for x in args[:7]),*args[7:])
                return run
        ns=dict(triton=types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),_stage_rows=Launch())
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-stage-wrapper>','exec'),ns)
        batch=types.SimpleNamespace(num_tokens_after_padding=4,num_reqs=1,num_reqs_after_padding=4,
            query_start_loc=torch.tensor([0,2,2,2,2],dtype=torch.int32),
            idx_mapping=torch.tensor([2],dtype=torch.int64),positions=torch.tensor([0,1,0,0],dtype=torch.int64))
        bank=dict(token_ring=self.tokenring,token_pos=self.tokenpos,lower=self.lower,flag=self.flag)
        ns['stage_rows'](batch,bank,self.cap)
        self.assertEqual(self.tokenring[:4].tolist(),[2*self.cap,2*self.cap+1,-1,-1])

    def test_recent_fp8_old_fp4_and_device_switch(self):
        torch.manual_seed(43);x=torch.randn(8,512).bfloat16()
        self.stage([0],[0,8],list(range(8)))
        self.write(x,list(range(2,10)),scale=.75)
        # Entire current query batch is high precision, including guard rows.
        self.assertTrue(torch.equal(self.read(list(range(2,10))),fp8_codes(x,.75).bfloat16()))
        self.flag[0]=0
        self.assertTrue(torch.equal(self.read(list(range(2,10))),fp4_qdq(x)))
        self.flag[0]=4
        self.stage([0],[0,1],[10])
        self.assertTrue(torch.equal(self.read([2,3]),fp4_qdq(x[:2])))
        self.assertTrue(torch.equal(self.read([9]),fp8_codes(x[7:],.75).bfloat16()))

    def test_wrap_invalidates_old_mapping_and_rejected_tail_overwrites(self):
        torch.manual_seed(44);x=torch.randn(2,512).bfloat16()
        self.stage([0],[0,1],[0]);self.write(x[:1],[5])
        self.stage([0],[0,1],[self.cap]);self.write(x[1:],[7])
        self.lower.zero_() # isolate tag validation from age validation
        self.assertTrue(torch.equal(self.read([5]),fp4_qdq(x[:1])))
        self.assertTrue(torch.equal(self.read([7]),fp8_codes(x[1:],1).bfloat16()))
        # Same physical slot and logical position, different accepted token.
        self.write(x[:1],[7])
        self.assertTrue(torch.equal(self.read([7]),fp8_codes(x[:1],1).bfloat16()))

    def test_null_and_unpopulated_apc_rows_fall_back(self):
        x=torch.ones(2,512,dtype=torch.bfloat16)
        self.stage([2],[0,2],[5,6]);self.write(x,[-1,9])
        self.assertTrue(torch.all(self.cache[0]==165))
        self.assertTrue(torch.all(self.read([-1,self.n])==0))
        self.mapping[9]=-1
        self.assertTrue(torch.equal(self.read([9]),fp4_qdq(x[1:])))

    def test_fresh_prefill_switch_is_device_backed(self):
        torch.manual_seed(12);x=torch.randn(2,512).bfloat16();out=torch.empty_like(x)
        for flag in (0,2048,0):
            self.flag[0]=flag
            for i in range(2):
                self.tl.pid=(i,0)
                self.k['_qdq'](Ptr.of(x),Ptr.of(out),512,2,Ptr.of(self.flag),True)
            self.assertTrue(torch.equal(out,x if flag else fp4_qdq(x)))

    def test_all_read_paths_consume_same_tagged_loader(self):
        torch.manual_seed(46);x=torch.randn(4,512).bfloat16()
        self.stage([0],[0,4],list(range(4)));self.write(x,[2,3,4,5])
        selected=torch.zeros(self.n,dtype=torch.int32);selected[2:6]=1
        out=torch.full((self.n,576),17,dtype=torch.bfloat16)
        for p in range((self.n+15)//16):
            self.tl.pid=(p,0)
            self.k['_expand_selected'](Ptr.of(self.cache),Ptr.of(selected),Ptr.of(out),self.n,16,*self.extra)
        self.assertTrue(torch.equal(out[2:6,:512],fp8_codes(x,1).bfloat16()))
        self.assertTrue(torch.all(out[:2]==17))
        # Cached-prefix gather uses physical slots with the same sidecar.
        bt=torch.tensor([[0]],dtype=torch.int32);cu=torch.tensor([0,4],dtype=torch.int32)
        ts=torch.zeros(4,dtype=torch.int32);start=torch.tensor([2],dtype=torch.int32)
        gathered=torch.empty(4,576,dtype=torch.bfloat16);self.tl.pid=(0,0)
        self.k['_gather'](*(Ptr.of(t) for t in (self.cache,gathered,bt,cu,ts,start)),64,1,576,self.n,4,True,16,*self.extra)
        self.assertTrue(torch.equal(gathered[:,:512],fp8_codes(x,1).bfloat16()))
        # Sparse unsplit vs split both use high rows, original rotary and same
        # online reduction algebra. Include null slots in the selection.
        q=torch.randn(1,16,512).bfloat16();r=torch.zeros(1,16,64,dtype=torch.bfloat16)
        idx=torch.tensor([[2,3,4,5,-1]],dtype=torch.int32)
        o=torch.empty_like(q);self.tl.pid=(0,0)
        base=(Ptr.of(q),Ptr.of(r),Ptr.of(self.cache),Ptr.of(idx))
        self.k['_mla'](*base,Ptr.of(o),q.stride(0),q.stride(1),r.stride(0),r.stride(1),368,5,o.stride(0),16,5,64,self.n,.01,1.,False,16,*self.extra)
        acc=torch.empty(1,32,16,512);md=torch.empty(1,32,2,16)
        for p in range(32):
            self.tl.pid=(0,p)
            self.k['_mla_partial'](*base,Ptr.of(acc),Ptr.of(md),q.stride(0),q.stride(1),r.stride(0),r.stride(1),368,5,o.stride(0),16,5,64,self.n,.01,1.,False,32,16,16,*self.extra)
        # Compare partial 0 to unsplit; remaining splits are entirely padded.
        partial=(acc[0,0]/md[0,0,1,:,None]).bfloat16()
        self.assertTrue(torch.equal(o[0],partial))

    def test_fleet_rows_reconstruct_exact_released_fp8(self):
        root=Path(os.environ['FP4_DUMPS'])/'normale/captures'
        seen=set();rows=0
        for file in sorted(root.glob('*.pt')):
            e=torch.load(file,weights_only=True,map_location='cpu')
            key=(e['case'],e['layer_id'],e['stage'])
            if key in seen or e['layer_id']==78:continue
            seen.add(key);x=e['latent'][e['write_slots']>=0]
            ids=sorted(set((0,len(x)//2,len(x)-1)));x=x[ids].contiguous()
            self.stage([0],[0,len(ids)],list(range(len(ids))))
            self.write(x,list(range(2,2+len(ids))),float(e['k_scale']))
            self.assertTrue(torch.equal(self.read(list(range(2,2+len(ids)))),fp8_codes(x,float(e['k_scale'])).bfloat16()))
            rows+=len(ids)
        self.assertEqual(len(seen),18)
        print('RECENT_FLEET_BITEXACT',len(seen),'groups;',rows,'BF16 rows')


class Wiring(unittest.TestCase):
    def test_options_and_capacity_cost(self):
        self.assertEqual(G.options({}),(0,0,False))
        good=dict(GLM_KV_FORMAT='fp4x',VLLM_USE_V2_MODEL_RUNNER='1',GLM_FP4_RECENT_WINDOW='2048')
        self.assertEqual(G.options(good),(2048,2048,False))
        for change in ({'GLM_KV_FORMAT':'fp8'},{'GLM_FP4_RECENT_WINDOW':'17'},
                       {'GLM_FP4_RECENT_AB':'1'},{'GLM_FP4_RECENT_INIT':'0'}):
            with self.assertRaises(ValueError):G.options(dict(good,**change))
        self.assertEqual(G.memory_bytes(0),0)
        self.assertEqual(G.memory_bytes(2048),1034085652)

    def test_switch_uses_existing_tensor_and_refuses_unreserved_size(self):
        flag=torch.tensor([2048],dtype=torch.int32);address=flag.data_ptr()
        with patch.object(G,'_AB',True),patch.object(G,'_WINDOW',2048),patch.object(G,'_INITIAL',2048), \
             patch.dict(G._DEVICES,{'cpu':{'flag':flag}},clear=True):
            G.set_window(0);self.assertEqual(int(flag[0]),0)
            G.set_window(2048);self.assertEqual(int(flag[0]),2048)
            self.assertEqual(flag.data_ptr(),address)
            with self.assertRaises(ValueError):G.set_window(4096)
        with patch.object(G,'_AB',False):
            with self.assertRaises(RuntimeError):G.set_window(0)

    def test_runner_hook_stages_both_real_and_dummy_batches(self):
        raw=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/v1/worker/gpu/model_runner.py'
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),G.RUNNER_PIN)
        events=[]
        class Runner:
            def prepare_attn(self,batch):events.append('real');return ('tables','slots')
            def prepare_dummy_attn(self,batch):events.append('dummy');return ('dummy','slots')
        mod=types.SimpleNamespace(__file__=str(raw),GPUModelRunner=Runner)
        with patch.object(G,'stage',lambda batch:events.append(batch)):
            G.install_runner(mod)
            self.assertEqual(Runner().prepare_attn('batch'),('tables','slots'))
            Runner().prepare_dummy_attn('warm')
        self.assertEqual(events,['real','batch','dummy','warm'])

    def test_constructor_reserves_only_target_banks_and_binds_cache(self):
        impl=types.SimpleNamespace(topk_indices_buffer=torch.empty(1))
        layer=types.SimpleNamespace(layer_name='model.layers.0.self_attn.attn',impl=impl)
        mtp=types.SimpleNamespace(layer_name='model.layers.78.self_attn.attn',
                                 impl=types.SimpleNamespace())
        with patch.object(G,'_WINDOW',2048),patch.object(G,'_INITIAL',2048), \
             patch.dict(G._DEVICES,{},clear=True),patch.dict(G._CACHES,{},clear=True), \
             patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            G.reserve(layer);G.reserve(mtp)
            self.assertTrue(layer._glm_recent_target);self.assertFalse(mtp._glm_recent_target)
            self.assertIsNone(mtp.impl._glm_recent_bank)
            bank=layer.impl._glm_recent_bank
            self.assertEqual(bank['shadow'].shape,(4*(2048+4096+36),512))
            cache=torch.zeros((64,368),dtype=torch.uint8)
            G.bind(cache,bank)
            self.assertIs(G.reader_args(cache)[0],bank['shadow'])
            self.assertEqual(G.reader_args(cache)[-1],6180)


if __name__=='__main__':unittest.main()
