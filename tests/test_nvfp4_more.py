# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from test_nvfp4_attn import ROOT, fixture, layer, f, a, ref, mtp
import glm_nvfp4_groups as groups
import glm_nvfp4_more as more
import glm_nvfp4_mtp as expert
import convert_more_nvfp4 as converter
sys.path.insert(0,str(ROOT/'bench'))
from more_nvfp4_gemm import rank_slice
from nvfp4_more_gate import cycle_gate,qeval_gate

NAMES=('model.layers.77.mlp.shared_experts.gate_proj',
       'model.layers.1.self_attn.indexer.wk','model.layers.1.self_attn.indexer.weights_proj',
       'model.layers.1.mlp.down_proj','model.layers.78.mlp.experts.0.gate_proj',
       'model.layers.78.mlp.experts.0.up_proj','model.layers.78.mlp.experts.0.down_proj')


def mixed_fixture(root):
    tensors=fixture(root,NAMES)
    rng=np.random.default_rng(49)
    for name in NAMES:
        if groups.classify(name)=='indexer':
            for leaf in ('weight_packed','weight_scale','weight_shape'):
                del tensors[name+'.'+leaf]
            tensors[name+'.weight']=torch.from_numpy(rng.normal(0,.1,(64,256)).astype(np.float32)).to(torch.bfloat16)
        elif groups.classify(name)=='mtp':
            tensors[name+'.weight_packed'],tensors[name+'.weight_scale']=ref.reencode(
                rng.normal(0,.1,(64,256)).astype(np.float32),torch.bfloat16,-1)
    scales={n:t for n,t in tensors.items() if n.endswith('.weight_scale')}
    save_file(scales,str(root/'scales.safetensors'))
    save_file({n:t for n,t in tensors.items() if n not in scales},str(root/'weights.safetensors'))
    idx={n:'scales.safetensors' if n in scales else 'weights.safetensors' for n in tensors}
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':idx}))
    return tensors


class MoreTests(unittest.TestCase):
    def test_inventory_and_switches(self):
        self.assertEqual(groups.groups({}),('attn',))
        self.assertEqual(groups.groups({'GLM_NVFP4_GROUPS':'mtp,dense,attn'}),('attn','dense','mtp'))
        for value in ('','0','all','attn,attn','shared, indexer'):
            with self.assertRaises(ValueError):groups.groups({'GLM_NVFP4_GROUPS':value})
        self.assertIsNone(groups.classify('model.layers.78.eh_proj'))
        self.assertIsNone(groups.classify('model.layers.0.mlp.down_proj'))
        self.assertEqual(groups.classify('model.layers.78.mtp_block.mlp.experts'),'mtp')
        source=Path('receipts/glm53-full-20260929/day3/mtp-accept-lab/weights/e8-tech2wild')
        if source.exists():
            inv=groups.inventory(json.loads((source/'model.safetensors.index.json').read_text())['weight_map'],groups.ORDER)
            self.assertEqual({g:sum(e['group']==g for e in inv.values()) for g in groups.ORDER},groups.COUNTS)

    def test_cross_shard_exact_resumption_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src';t=mixed_fixture(src);selected=('shared','indexer','dense','mtp')
            before={p.name:f.sha256(p) for p in src.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                m=converter.convert(src,Path(d)/'one',rows=3,groups=selected,strict=False)
                converter.convert(src,Path(d)/'one',rows=16,groups=selected,strict=False)
                converter.convert(src,Path(d)/'two',rows=64,groups=selected,strict=False)
            self.assertEqual((Path(d)/'one/SHA256SUMS').read_bytes(),(Path(d)/'two/SHA256SUMS').read_bytes())
            for name,e in m['tensors'].items():
                data=load_file(str(Path(d)/'one'/e['file']))
                if e['group']=='indexer':w=t[name+'.weight'].float().numpy()
                elif e['group']=='mtp':
                    w=ref.unpack(t[name+'.weight_packed'].numpy())*t[name+'.weight_scale'].float().numpy()
                else:w=f.decode_int8(t[name+'.weight_packed'],t[name+'.weight_scale'])
                q=f.dequant(data['weight_packed'].numpy(),data['weight_scale'].view(torch.uint8).numpy(),data['weight_global_scale'].item())
                self.assertTrue(np.array_equal(q.view(np.uint32),ref.qdq(w).view(np.uint32)),name)
            self.assertEqual(before,{p.name:f.sha256(p) for p in src.iterdir()})
            files=[str(src/'weights.safetensors'),str(src/'scales.safetensors')]
            for kind,wanted in (('target',{'shared','indexer','dense'}),('draft',{'mtp'})):
                with patch.object(mtp,'active',return_value=types.SimpleNamespace(kind=kind)):
                    rows=list(more.transform(iter(t.items()),files,Path(d)/'one',selected,manifest=m))
                replacements={n.rsplit('.',1)[0] for n,v in rows if n.endswith('weight_global_scale')}
                self.assertEqual(replacements,{n for n,e in m['tensors'].items() if e['group'] in wanted})
                if kind=='target':
                    for n,v in rows:
                        if groups.classify(n.rsplit('.',1)[0])=='mtp':self.assertIs(v,t[n])
            (Path(d)/'one'/next(iter(m['tensors'].values()))['file']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):converter.convert(src,Path(d)/'one',groups=selected,strict=False)

    def test_group_bytes_across_processes(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src';mixed_fixture(src);sums=[]
            code="import sys;sys.path[:0]=sys.argv[1:3];import convert_more_nvfp4 as c;c.convert(sys.argv[3],sys.argv[4],groups=('shared','indexer','dense','mtp'),strict=False)"
            for i in range(3):
                out=Path(d)/str(i)
                subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts'),str(ROOT/'overlay/overlay'),str(src),str(out)],check=True,capture_output=True)
                sums.append((out/'SHA256SUMS').read_bytes())
            self.assertEqual(len(set(sums)),1)

    def test_real_scheme_group_selection(self):
        class Config:
            def get_scheme(self,layer,layer_name=None):return 'old'
            def get_quant_method(self,layer,prefix):return 'original'
        module=types.SimpleNamespace(CompressedTensorsConfig=Config)
        a.patch_config(module)
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='shared,dense,mtp,indexer'),patch.object(a,'new_scheme',return_value='real'):
            for n in NAMES:self.assertEqual(Config().get_scheme(None,n),'real')
            self.assertEqual(Config().get_scheme(None,f.MODULES[0]),'old')
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='attn'),patch.object(a,'new_scheme',return_value='real'):
            self.assertEqual(Config().get_scheme(None,NAMES[0]),'old')
            self.assertEqual(Config().get_scheme(None,f.MODULES[0]),'real')

    def test_indexer_fp8_helper_does_not_swallow_nvfp4_scales(self):
        original=lambda *a,**kw:'stock-fp8'
        module=types.SimpleNamespace(_try_load_fp8_indexer_wk=original)
        more.patch_indexer_helper(module);wrapped=module._try_load_fp8_indexer_wk
        more.patch_indexer_helper(module);self.assertIs(wrapped,module._try_load_fp8_indexer_wk)
        prefix='model.layers.1.self_attn.indexer.wk.'
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='attn,indexer'):
            for leaf in ('weight_packed','weight_scale','weight_global_scale'):
                self.assertFalse(wrapped(prefix+leaf,None))
            self.assertEqual(wrapped(prefix+'weight',None),'stock-fp8')
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='attn'):
            self.assertEqual(wrapped(prefix+'weight_scale',None),'stock-fp8')

    def test_indexer_constructor_replaces_explicit_none_only_when_selected(self):
        class Merged:
            def __init__(self,input_size,output_sizes,quant_config=None,prefix='',disable_tp=False):
                self.quant_config=quant_config
        module=types.SimpleNamespace(MergedColumnParallelLinear=Merged)
        more.patch_linear(module)
        prefix='model.layers.1.self_attn.indexer.wk_weights_proj'
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='attn,indexer'):
            mod=Merged(256,[128,32],quant_config=None,prefix=prefix,disable_tp=True)
            self.assertIsNotNone(mod.quant_config)
            self.assertTrue(hasattr(mod.quant_config,'get_quant_method'))
            with self.assertRaisesRegex(ValueError,'replicated'):Merged(256,[128,32],prefix=prefix)
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_NVFP4_GROUPS='attn'):
            self.assertIsNone(Merged(256,[128,32],prefix=prefix).quant_config)

    def test_tp_slices_preserve_dequant_and_full_globals(self):
        mod=layer((128,),512)
        t={leaf:getattr(mod,leaf).detach() for leaf in ('weight_packed','weight_scale','weight_global_scale')}
        full=f.dequant(t['weight_packed'].numpy(),t['weight_scale'].view(torch.uint8).numpy(),t['weight_global_scale'].item())
        for name,axis in (('model.layers.1.mlp.gate_proj',0),('model.layers.78.mlp.experts.0.down_proj',1),('model.layers.0.self_attn.indexer.wk',None)):
            shards=[]
            for r in range(4):
                s=rank_slice(name,t,r);self.assertIs(s['weight_global_scale'],t['weight_global_scale'])
                q=f.dequant(s['weight_packed'].numpy(),s['weight_scale'].view(torch.uint8).numpy(),s['weight_global_scale'].item())
                if axis is None:self.assertTrue(np.array_equal(q.view(np.uint32),full.view(np.uint32)))
                else:shards.append(q)
            if axis is not None:self.assertTrue(np.array_equal(np.concatenate(shards,axis=axis).view(np.uint32),full.view(np.uint32)))

    def test_strict_incremental_cycle_boundary_and_qeval_count_floor(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);aa=[];bb=[]
            for kind in ('prose','code'):
                for i in range(10):
                    metadata=dict(pair_key=f'{kind}-{i}',prompt_sha256='fixed',seed=0,k_mode='same',target_M='same')
                    aa.append(dict(status='OK',request_id=str(i),metadata=metadata,time_source='wall',labels={},cycle_ms=100.,committed_per_cycle=2.,cycle_ms_kind='decode/drafts'))
                    bb.append(dict(aa[-1],cycle_ms=98.,committed_per_cycle=1.999))
            def write():
                for name,rows in (('a',aa),('b',bb)):(d/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write();self.assertTrue(cycle_gate(d/'a',d/'b')['passed'])
            for row in bb:row['cycle_ms']=99.
            write();self.assertFalse(cycle_gate(d/'a',d/'b')['passed'])
            for row in bb:row.update(cycle_ms=98.,committed_per_cycle=1.98)
            write();self.assertFalse(cycle_gate(d/'a',d/'b')['passed'])
            for i in range(3):
                rows=[dict(id=str(j),category='code',**{'pass':j<71}) for j in range(75)]
                (d/f'qeval-{i}.json').write_text(json.dumps(dict(concurrency=1,results=rows)))
            self.assertTrue(qeval_gate(d)['passed'])
            for i in range(3):
                rows=[dict(id=str(j),category='code',**{'pass':j<70}) for j in range(75)]
                (d/f'qeval-{i}.json').write_text(json.dumps(dict(concurrency=1,results=rows)))
            self.assertFalse(qeval_gate(d)['passed'])

    def test_mtp_split_globals_and_three_gemms(self):
        mod=torch.nn.Module();n=128;k=256;e=2
        mod.num_experts=e;mod.hidden_size=k;mod.intermediate_size_per_partition=n
        mod.global_num_experts=e;mod.expert_map=None;mod.apply_router_weight_on_input=False
        pieces=[layer((n,n),k) for _ in range(e)];downs=[layer((k,),n) for _ in range(e)]
        for name,values in {'w13_weight_packed':[p.weight_packed for p in pieces],
            'w13_weight_scale':[p.weight_scale for p in pieces],
            'w13_weight_global_scale':[p.weight_global_scale for p in pieces],
            'w2_weight_packed':[p.weight_packed for p in downs],
            'w2_weight_scale':[p.weight_scale for p in downs],
            'w2_weight_global_scale':[p.weight_global_scale.reshape(()) for p in downs]}.items():
            mod.register_parameter(name,torch.nn.Parameter(torch.stack(values),requires_grad=False))
        for name in ('w13_input_global_scale','w2_input_global_scale'):mod.register_parameter(name,torch.nn.Parameter(torch.ones(e),requires_grad=False))
        expected=mod.w13_weight_global_scale.detach().clone();calls=[]
        def prepare(l,*args,**kwargs):
            self.assertFalse(kwargs['is_act_and_mul']);calls.append(args)
            l.workspace=torch.zeros(1,dtype=torch.int32)
            return args
        expert.prepare_experts(mod,prepare)
        self.assertEqual(len(calls),2)
        self.assertTrue(torch.equal(mod._glm_gate_global,expected[:,0]));self.assertTrue(torch.equal(mod._glm_up_global,expected[:,1]))
        class Ops:
            def __init__(self):self.calls=[]
            def moe_wna16_marlin_gemm(self,*args,**kw):
                self.calls.append((args,kw));args[1].fill_(1);return args[1]
            def silu_and_mul(self,out,inp):out.copy_(torch.nn.functional.silu(inp[:,:n])*inp[:,n:])
        ops=Ops();weights=torch.ones((3,2),dtype=torch.float32);ids=torch.zeros((3,2),dtype=torch.int32)
        out=expert.apply_experts(mod,torch.ones((3,k),dtype=torch.bfloat16),weights,ids,ops=ops,
            align=lambda *a,**kw:(torch.zeros(1,dtype=torch.int32),)*3,scalar='FP4')
        self.assertEqual(out.shape,(3,k));self.assertEqual(len(ops.calls),3)
        self.assertEqual([kw['top_k'] for args,kw in ops.calls],[2,2,1])
        self.assertTrue(all(kw['use_fp32_reduce'] and not kw['use_atomic_add'] for args,kw in ops.calls))
        self.assertTrue(all(args[5] is None for args,kw in ops.calls))

    def test_dry_default_identical_to_integration_and_groups_all_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'ssh';fake.write_text('#!/bin/sh\nexit 99\n');fake.chmod(0o755)
            env=dict(os.environ,RECIPE_CONFIG=str(ROOT/'tests/fixtures/legacy-dry.env'),DRY='1',CTN='glm53full-nvfp4-more-dry',GLM_ATTN_WEIGHTS='nvfp4',GLM_ATTN_NVFP4_DIR='/tmp/attn',PATH=d+':'+os.environ['PATH'])
            env['GLM_NVFP4_GROUPS']='attn'
            def run(root,ev):return subprocess.run([str(root/'start.sh'),'serve'],env=ev,text=True,capture_output=True,check=True).stdout
            default=run(ROOT,env)
            self.assertEqual(default.count('GLM_ATTN_WEIGHTS=nvfp4'), 8)
            self.assertNotIn('GLM_NVFP4_GROUPS=', default)
            env.update(GLM_NVFP4_GROUPS='attn,shared,dense,mtp,indexer',GLM_NVFP4_MORE_DIR='/tmp/more')
            output=run(ROOT,env)
            docker=[shlex.split(l) for l in output.splitlines() if l.startswith('docker run -d')]
            self.assertEqual(len(docker),4)
            for cmd in docker:
                self.assertIn('GLM_NVFP4_GROUPS=attn,shared,indexer,dense,mtp',cmd)
                self.assertIn('/tmp/more:/more-nvfp4:ro',cmd)

if __name__=='__main__':unittest.main()
