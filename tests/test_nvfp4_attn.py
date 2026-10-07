# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
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
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/overlay'),str(ROOT/'scripts'),str(ROOT/'tests/fixtures')]
import glm_nvfp4_format as f
import glm_nvfp4_attn as a
import glm_fast_load as fast
import glm_mtp_select as mtp
import convert_attn_nvfp4 as c
import nvfp4_wsim_reference as ref


def fixture(root,names):
    root.mkdir()
    (root/'config.json').write_text('{}')
    tensors={};rng=np.random.default_rng(77)
    for name in names:
        p,s=ref.reencode(rng.normal(0,.04,(64,256)).astype(np.float32),torch.bfloat16,128)
        tensors.update({name+'.weight_packed':p,name+'.weight_scale':s,
                        name+'.weight_shape':torch.tensor([64,256])})
    save_file(tensors,str(root/'weights.safetensors'))
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{n:'weights.safetensors' for n in tensors}}))
    return tensors


def layer(widths=(64,),k=256):
    mod=torch.nn.Module();mod.params_dtype=torch.bfloat16
    mod.logical_widths=list(widths);mod.output_size_per_partition=sum(widths);mod.input_size_per_partition=k
    rng=np.random.default_rng(4)
    values=[];scales=[];globals=[]
    for i,n in enumerate(widths):
        w=rng.normal(0,.01*(i+1),(n,k)).astype(np.float32);ts=np.float32(np.max(np.abs(w))/np.float32(2688))
        v,s=f.encode(w,ts);values.append(torch.from_numpy(v));scales.append(torch.from_numpy(s).view(torch.float8_e4m3fn));globals.append(ts)
    mod.weight_packed=torch.nn.Parameter(torch.cat(values),requires_grad=False)
    mod.weight_scale=torch.nn.Parameter(torch.cat(scales),requires_grad=False)
    mod.weight_global_scale=torch.nn.Parameter(torch.tensor(np.array(globals)),requires_grad=False)
    return mod


class Numerics(unittest.TestCase):
    def test_bit_exact_fp32_qdq_random_and_chunking(self):
        rng=np.random.default_rng(41)
        for n,k in ((1,128),(19,256),(64,512)):
            p,s=ref.reencode(rng.normal(0,.04,(n,k)).astype(np.float32),torch.bfloat16,128)
            w=f.decode_int8(p,s);ts=np.float32(np.max(np.abs(w))/np.float32(2688))
            v,sc=f.encode(w,ts)
            actual=f.dequant(v,sc,ts)
            self.assertTrue(np.array_equal(actual.view(np.uint32),ref.qdq(w).view(np.uint32)))
            chunks=[f.encode(w[i:i+3],ts) for i in range(0,n,3)]
            self.assertTrue(np.array_equal(v,np.concatenate([x[0] for x in chunks])))
            self.assertTrue(np.array_equal(sc,np.concatenate([x[1] for x in chunks])))

    def test_every_format_tie_and_signed_zero(self):
        for table in f.tables():
            mids=(table[:-1]+table[1:])*np.float32(.5)
            expected=np.array([i if i%2==0 else i+1 for i in range(len(mids))],np.uint8)
            self.assertTrue(np.array_equal(f.rne_codes(mids,table),expected))
        w=np.zeros((2,128),np.float32);w[0,1]=-0.;w[1,0]=6
        v,s=f.encode(w,np.float32(1/448))
        self.assertTrue(np.array_equal(f.dequant(v,s,1/448).view(np.uint32),ref.qdq(w, np.float32(1/448)).view(np.uint32)))

    def test_zero_nonfinite_and_invalid_scales(self):
        v,s=f.encode(np.zeros((64,128),np.float32),0)
        self.assertFalse(v.any());self.assertFalse(s.any())
        for w,ts in ((np.ones((1,128),np.float32),0),(np.ones((1,17),np.float32),1),(np.full((1,128),np.nan,np.float32),1)):
            with self.assertRaises(ValueError):f.encode(w,ts)
        mod=layer();mod.weight_scale.data[0,0]=torch.tensor(2**-9).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError,'exact Marlin range'):a.prepare_layer(mod,lambda x:None)

    def test_simulator_int8_reencoding_is_a_distinct_grid(self):
        rng=np.random.default_rng(8);p,s=ref.reencode(rng.normal(0,.03,(64,256)).astype(np.float32),torch.bfloat16,128)
        direct=ref.qdq(f.decode_int8(p,s));sp,ss=ref.simulate(p,s)
        served=f.decode_int8(sp,ss)
        self.assertGreater(np.max(np.abs(direct-served)),0)
        self.assertLess(np.linalg.norm(direct-served)/np.linalg.norm(direct),.02)

    def test_explicit_inventory_is_exact_checkpoint_group(self):
        lines=(ROOT/'docs/nvfp4-attn/tensors.txt').read_text().splitlines()
        self.assertEqual(lines,[n+'.weight_packed' for n in f.MODULES]);self.assertEqual(len(lines),385)
        for name in f.MODULES:self.assertEqual(ref.classify(name+'.weight_packed'),'attn')
        self.assertFalse(a.selected('model.layers.0.self_attn.q_b_proj'))
        self.assertFalse(a.selected('model.layers.78.self_attn.q_a_proj'))
        self.assertFalse(a.selected('model.layers.4.self_attn.indexer.wq_b'))
        self.assertTrue(a.selected('model.layers.4.self_attn.fused_qkv_a_proj'))


class Converter(unittest.TestCase):
    def test_cross_shard_resumption_and_source_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';out=Path(d)/'new';names=f.MODULES[:2];tensors=fixture(src,names)
            # Put scales in a different shard, and ensure global absmax precedes row chunking.
            scale_keys=[n for n in tensors if n.endswith('weight_scale')]
            save_file({n:tensors[n] for n in scale_keys},str(src/'scales.safetensors'))
            save_file({n:t for n,t in tensors.items() if n not in scale_keys},str(src/'weights.safetensors'))
            idx=json.loads((src/'model.safetensors.index.json').read_text())
            for n in scale_keys:idx['weight_map'][n]='scales.safetensors'
            (src/'model.safetensors.index.json').write_text(json.dumps(idx))
            before={p.name:f.sha256(p) for p in src.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):m=c.convert(src,out,rows=3,modules=names)
            mtimes={p.name:p.stat().st_mtime_ns for p in out.glob('*.safetensors')}
            c.convert(src,out,rows=16,modules=names)
            self.assertEqual(mtimes,{p.name:p.stat().st_mtime_ns for p in out.glob('*.safetensors')})
            self.assertEqual(before,{p.name:f.sha256(p) for p in src.iterdir()})
            for name in names:
                data=load_file(str(out/m['tensors'][name]['file']))
                w=f.decode_int8(tensors[name+'.weight_packed'],tensors[name+'.weight_scale'])
                actual=f.dequant(data['weight_packed'].numpy(),data['weight_scale'].view(torch.uint8).numpy(),data['weight_global_scale'].item())
                self.assertTrue(np.array_equal(actual.view(np.uint32),ref.qdq(w).view(np.uint32)))
            f.read_manifest(out,names)
            self.assertIn('manifest.json',(out/'SHA256SUMS').read_text())
            (out/m['tensors'][names[0]]['file']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash'):c.convert(src,out,modules=names)

    def test_output_bytes_are_reproducible_across_processes(self):
        # Every node converts its own checkpoint copy; identical sources must give identical files and SHA256SUMS.
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';names=f.MODULES[:5];fixture(src,names)
            code=('import sys;sys.path.insert(0,sys.argv[1]);import convert_attn_nvfp4 as c,glm_nvfp4_format as f;'
                  'c.convert(sys.argv[2],sys.argv[3],modules=f.MODULES[:5])')
            sums=[]
            for i in range(4):
                out=Path(d)/f'new{i}'
                subprocess.run([sys.executable,'-c',code,str(Path(c.__file__).resolve().parent),str(src),str(out)],
                               check=True,capture_output=True)
                sums.append((out/'SHA256SUMS').read_text())
            self.assertEqual(len(set(sums)),1)
            meta=safe_open(str(out/'attn-000.safetensors'),framework='pt').metadata()
            self.assertEqual(list(meta),['glm_nvfp4'])
            self.assertEqual(json.loads(meta['glm_nvfp4']),dict(algorithm=f.ALGORITHM,module=names[0]))

    def test_incomplete_resume_and_source_drift(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';out=Path(d)/'new';names=f.MODULES[:1];fixture(src,names)
            with contextlib.redirect_stdout(io.StringIO()):m=c.convert(src,out,modules=names)
            m['complete']=False;c.atomic_json(out/'manifest.json',m)
            with self.assertRaises(ValueError):f.read_manifest(out,names)
            c.convert(src,out,modules=names);f.read_manifest(out,names)
            data=load_file(str(src/'weights.safetensors'));data[names[0]+'.weight_packed'][0,0]+=1
            save_file(data,str(src/'weights.safetensors'))
            with self.assertRaisesRegex(ValueError,'source/algorithm'):c.convert(src,out,modules=names)

    def test_directory_safety_and_missing_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';fixture(src,f.MODULES[:1])
            for dst in (src,src/'child',src.parent):
                with self.assertRaises(ValueError):c.convert(src,dst,modules=f.MODULES[:1])
            with self.assertRaisesRegex(ValueError,'complete 385'):c.convert(src,Path(d)/'new')


class Integration(unittest.TestCase):
    def test_fused_scales_not_collapsed_and_bf16_activations(self):
        mod=layer((64,128));original=mod.weight_global_scale.detach().clone();calls=[]
        def prepare(p):p.workspace=torch.zeros(1,dtype=torch.int32);calls.append(p)
        a.prepare_layer(mod,prepare)
        self.assertEqual(len(calls),2);self.assertFalse(hasattr(mod,'weight_packed'))
        self.assertTrue(torch.equal(torch.stack([p.weight_global_scale for p in calls]),original))
        def apply(**kw):
            self.assertIsNone(kw['input_dtype']);self.assertTrue(kw['use_fp32_reduce'])
            return torch.ones((*kw['input'].shape[:-1],kw['size_n']),dtype=torch.bfloat16)
        self.assertEqual(tuple(a.apply_layer(mod,torch.zeros(3,256,dtype=torch.bfloat16),apply=apply).shape),(3,192))
        with self.assertRaises(ValueError):a.apply_layer(mod,torch.zeros(3,256),apply=apply)

    def test_single_preparation_no_persistent_duplicate(self):
        mod=layer();a.prepare_layer(mod,lambda p:setattr(p,'workspace',torch.zeros(1)))
        self.assertFalse(hasattr(mod,'weight_packed'));self.assertFalse(hasattr(mod,'_glm_nvfp4_parts'))
        self.assertEqual(mod.weight_global_scale.ndim,0)
        self.assertEqual(set(dict(mod.named_parameters())),{'weight','weight_scale','weight_global_scale'})

    def test_iterator_order_companions_counts_draft_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';out=Path(d)/'new';names=f.MODULES[:2];tensors=fixture(src,names)
            with contextlib.redirect_stdout(io.StringIO()):m=c.convert(src,out,modules=names)
            files=[str(src/'weights.safetensors')]
            weights=list(reversed(list(tensors.items())))+[('model.layers.78.self_attn.q_a_proj.weight_scale',torch.ones(1))]
            with patch.object(mtp,'active',return_value=None):got=dict(a.transform(iter(weights),files,out,names))
            self.assertEqual(len(got),7);self.assertNotIn(names[0]+'.weight_shape',got)
            self.assertIs(got[weights[-1][0]],weights[-1][1])
            with patch.object(mtp,'active',return_value=types.SimpleNamespace(kind='draft')):
                self.assertEqual(list(a.transform(iter(weights),files,'/does/not/exist',names)),weights)
            with self.assertRaisesRegex(ValueError,'incomplete'):list(a.transform(iter(weights[:-3]),files,out,names))
            with self.assertRaisesRegex(ValueError,'duplicate'):list(a.transform(iter(weights+weights[:1]),files,out,names))
            (out/m['tensors'][names[0]]['file']).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'hash'):list(a.transform(iter(weights),files,out,names))

    def test_scheme_patch_default_and_selection_idempotent(self):
        class Config:
            def get_scheme(self,layer,layer_name=None):return 'old'
        module=types.SimpleNamespace(CompressedTensorsConfig=Config)
        a.patch_config(module);first=Config.get_scheme;a.patch_config(module)
        self.assertIs(first,Config.get_scheme)
        with patch.object(a,'new_scheme',return_value='new'),patch.dict(os.environ,{'GLM_ATTN_WEIGHTS':'nvfp4'}):
            self.assertEqual(Config().get_scheme(None,'model.layers.1.self_attn.q_b_proj'),'new')
            self.assertEqual(Config().get_scheme(None,'model.layers.78.self_attn.q_b_proj'),'old')
        with patch.dict(os.environ,{'GLM_ATTN_WEIGHTS':'int8'}):self.assertEqual(Config().get_scheme(None,'model.layers.1.self_attn.q_b_proj'),'old')

    def test_default_inert_invalid_mode_and_source_pins(self):
        prior=list(sys.meta_path);self.assertFalse(a.register({}));self.assertEqual(prior,sys.meta_path)
        with self.assertRaises(ValueError):a.mode({'GLM_ATTN_WEIGHTS':'auto'})
        with self.assertRaises(ValueError):a.mode({'GLM_ATTN_WEIGHTS':'nvfp4','GLM_NVFP4_WSIM':'attn'})
        src=Path(os.environ['GLM_IMAGE_SRC'])
        for name,h in json.loads((ROOT/'overlay/overlay/nvfp4_source_pins.json').read_text()).items():
            p=src/(name.replace('.','/')+'.py')
            if not p.exists():p=src/name.replace('.','/')/'__init__.py'
            self.assertEqual(f.sha256(p),h)

    def test_fast_and_stock_iterator_hook(self):
        for enabled in ('0','1'):
            original=lambda *args,**kw:iter(())
            module=types.SimpleNamespace(safetensors_weights_iterator=original,_natural_sort_key=lambda x:x,
              tqdm=lambda x,**kw:x,enable_tqdm=lambda x:False,_BAR_FORMAT='',should_skip_weight=lambda *x:False)
            with patch.dict(os.environ,{'GLM_ATTN_WEIGHTS':'nvfp4','GLM_FAST_LOAD':enabled}),patch.object(fast,'_check_source'):
                fast._patch_weight_utils(module);first=module.safetensors_weights_iterator
                fast._patch_weight_utils(module);self.assertIs(first,module.safetensors_weights_iterator)
                self.assertTrue(first._glm_nvfp4_attn)

    def test_dry_default_and_all_ranks_mount(self):
        for mode in ('int8','nvfp4'):
            with tempfile.TemporaryDirectory() as tmp:
                shim=Path(tmp)/'ssh';shim.write_text('#!/bin/sh\nexit 99\n');shim.chmod(0o755)
                env=dict(os.environ,RECIPE_CONFIG=str(ROOT/'tests/fixtures/legacy-dry.env'),GLM_NVFP4_GROUPS='attn',DRY='1',GLM_ATTN_WEIGHTS=mode,GLM_ATTN_NVFP4_DIR='/srv/glm/models/attn-nvfp4',PATH=tmp+':'+os.environ['PATH'])
                r=subprocess.run([str(ROOT/'start.sh'),'serve'],env=env,text=True,capture_output=True,check=True)
            commands=[shlex.split(l) for l in r.stdout.splitlines() if l.startswith('docker run -d')]
            self.assertEqual(len(commands),4)
            for cmd in commands:
                flags=[x for x in cmd if x.startswith('GLM_ATTN_')]
                self.assertEqual(flags,[] if mode=='int8' else ['GLM_ATTN_NVFP4_DIR=/attn-nvfp4','GLM_ATTN_WEIGHTS=nvfp4'])
                self.assertEqual('/tmp/attn:/attn-nvfp4:ro' in cmd,mode=='nvfp4')

    def test_real_pinned_parameter_tp_slicing(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/parameter.py'
        distributed=types.ModuleType('vllm.distributed');distributed.get_tensor_model_parallel_rank=lambda:2;distributed.get_tensor_model_parallel_world_size=lambda:4
        logger=types.ModuleType('vllm.logger');logger.init_logger=lambda name:types.SimpleNamespace()
        platform=types.ModuleType('vllm.platforms');platform.current_platform=types.SimpleNamespace(use_sync_weight_loader=lambda:False)
        with patch.dict(sys.modules,{'vllm.distributed':distributed,'vllm.logger':logger,'vllm.platforms':platform}):
            spec=importlib.util.spec_from_file_location('_nvfp4_pinned_parameter',src);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
            for cls in (p.ModelWeightParameter,p.GroupQuantScaleParameter):
                for dim in (0,1):
                    full=torch.arange(64*128).reshape(64,128).to(torch.uint8)
                    shape=list(full.shape);shape[dim]//=4
                    param=cls(data=torch.empty(shape,dtype=torch.uint8),input_dim=1,output_dim=0,weight_loader=lambda *x:None)
                    (param.load_column_parallel_weight if dim==0 else param.load_row_parallel_weight)(full)
                    self.assertTrue(torch.equal(param,full.chunk(4,dim)[2]))
            scalar=p.PerTensorScaleParameter(data=torch.empty(2),weight_loader=lambda *x:None)
            scalar.load_merged_column_weight(torch.tensor([.001]),shard_id=0)
            scalar.load_merged_column_weight(torch.tensor([.004]),shard_id=1)
            self.assertTrue(torch.equal(scalar,torch.tensor([.001,.004])))



class PinnedContracts(unittest.TestCase):
    def test_actual_pinned_scheme_creation_and_parameter_load(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])
        distributed=types.ModuleType('vllm.distributed');distributed.get_tensor_model_parallel_rank=lambda:0;distributed.get_tensor_model_parallel_world_size=lambda:1
        logger=types.ModuleType('vllm.logger');logger.init_logger=lambda name:types.SimpleNamespace()
        platform=types.ModuleType('vllm.platforms');platform.current_platform=types.SimpleNamespace(use_sync_weight_loader=lambda:False)
        kernel_module=types.ModuleType('vllm.model_executor.kernels.linear')
        class MarlinNvFp4LinearKernel:pass
        kernel_module.init_nvfp4_linear_kernel=lambda use_a16=False:MarlinNvFp4LinearKernel() if use_a16 else None
        fusion=types.ModuleType('vllm.model_executor.layers.fusion.quant_activation');fusion.expose_input_quant_key=lambda *x:None
        base=types.ModuleType('vllm.model_executor.layers.quantization.compressed_tensors.schemes');base.CompressedTensorsScheme=object
        name='vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4'
        modules={'vllm.distributed':distributed,'vllm.logger':logger,'vllm.platforms':platform,
          'vllm.model_executor.kernels.linear':kernel_module,'vllm.model_executor.layers.fusion.quant_activation':fusion,
          'vllm.model_executor.layers.quantization.compressed_tensors.schemes':base}
        with patch.dict(sys.modules,modules):
            spec=importlib.util.spec_from_file_location('vllm.model_executor.parameter',src/'vllm/model_executor/parameter.py')
            parameters=importlib.util.module_from_spec(spec);spec.loader.exec_module(parameters)
            with patch.dict(sys.modules,{'vllm.model_executor.parameter':parameters}):
                spec=importlib.util.spec_from_file_location(name,src/(name.replace('.','/')+'.py'))
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                with patch.dict(sys.modules,{name:module}):
                    scheme=a.new_scheme();self.assertTrue(scheme.use_a16)
                    mod=torch.nn.Module();mod.params_dtype=torch.bfloat16
                    scheme.create_weights(mod,[64,128],256,torch.bfloat16,lambda *x:None)
                    self.assertFalse(hasattr(mod,'input_global_scale'))
                    self.assertEqual(tuple(mod.weight_packed.shape),(192,128))
                    self.assertEqual(tuple(mod.weight_scale.shape),(192,16))
                    self.assertEqual(mod.weight_global_scale.dtype,torch.float32)
                    for offset,width,scale in ((0,64,.01),(64,128,.02)):
                        mod.weight_packed.load_merged_column_weight(torch.zeros(width,128,dtype=torch.uint8),shard_offset=offset,shard_size=width)
                        mod.weight_scale.load_merged_column_weight(torch.full((width,16),448,dtype=torch.float8_e4m3fn),shard_offset=offset,shard_size=width)
                        mod.weight_global_scale.load_merged_column_weight(torch.tensor([scale]),shard_id=0 if offset==0 else 1)
                    with patch('glm_nvfp4_attn.prepare_layer') as prepare:
                        scheme.process_weights_after_loading(mod);prepare.assert_called_once_with(mod)

    def test_pinned_fp4_dispatch_every_requested_m_and_shape(self):
        import ast
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py'
        tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_fp4_marlin_linear')
        calls=[];current=[None]
        def gemm(**kwargs):
            calls.append(kwargs)
            return torch.empty(kwargs['size_m'],kwargs['size_n'],dtype=torch.bfloat16,device='meta')
        namespace={'torch':torch,'USE_FP32_REDUCE_DEFAULT':True,
            'marlin_repacked_nk':lambda *x,**kw:current[0],
            'marlin_pad_dim':lambda x,*args:x,'should_use_atomic_add_reduce':lambda **kw:False,
            'marlin_unpad_output':lambda x,*args:x,'ops':types.SimpleNamespace(marlin_gemm=gemm),
            'scalar_types':types.SimpleNamespace(float4_e2m1f='fp4')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
        from nvfp4_attn_cost import SHAPES,MS
        for n,k in SHAPES.values():
            self.assertTrue(n%64==0 and k%128==0);current[0]=(n,k)
            for m in MS:
                x=torch.empty(m,k,dtype=torch.bfloat16,device='meta')
                y=namespace['apply_fp4_marlin_linear'](x,torch.empty(1),torch.empty(1),torch.empty(1),torch.empty(1),n,k,input_dtype=None)
                self.assertEqual(tuple(y.shape),(m,n));self.assertIs(calls[-1]['a']._base,x)
                self.assertIsNone(calls[-1]['a_scales']);self.assertTrue(calls[-1]['use_fp32_reduce'])
        self.assertEqual(len(calls),5*len(MS))

    def test_cost_model_and_memory_ledger(self):
        import nvfp4_attn_cost as cost
        m=cost.ledger()
        self.assertEqual(m['freed_bytes'],1861269564)
        self.assertEqual(m['kv']['new_blocks'],2482)
        self.assertGreater(m['forecast_cycle_reduction_pct'][0],3)
        self.assertEqual(m['retained_absorbed_ukuv_bytes'],572522496)


class CycleGate(unittest.TestCase):
    def make_rows(self,folder,cycle=1,committed=1):
        rows=[]
        for kind in ('prose','code'):
            for i in range(20):
                key=f'{kind}-{i}'
                rows.append(dict(status='OK',request_id='request-'+key,cycle_ms=83*cycle,
                    committed_per_cycle=2.5*committed,time_source='server_request_decode_time_sum',
                    cycle_ms_kind='wall proxy',labels={},metadata=dict(warmup=False,kind=kind,
                    pair_key=key,prompt_sha256='frozen-'+key,seed=i,k_mode='served-kstop',target_M='adaptive')))
        folder.mkdir();(folder/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return rows
    def test_paired_cycle_and_committed_gate(self):
        sys.path.insert(0,str(ROOT/'bench'));import nvfp4_attn_cycles as cycles
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_rows(root/'A');self.make_rows(root/'B',.96,.995)
            cycles.compare(root/'A',root/'B',root/'gate.json')
            self.assertTrue(json.loads((root/'gate.json').read_text())['passed'])
            rows=[json.loads(l) for l in (root/'B/requests.jsonl').read_text().splitlines()]
            for row in rows:row['committed_per_cycle']=2.5*.98
            (root/'B/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(SystemExit):cycles.compare(root/'A',root/'B',root/'gate.json')
            (root/'B/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaises(ValueError):cycles.compare(root/'A',root/'B',root/'gate.json')

    def test_replication_status_uses_pinned_linear_method(self):
        import ast
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/layers/linear.py'
        tree=ast.parse(src.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LinearBase')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_param_tp_status')
        namespace={'BasevLLMParameter':torch.nn.Parameter}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
        mod=layer((64,128));mod.tp_rank=0;mod.tp_size=1
        for param in mod.parameters():param.tp_rank=2;param.tp_size=4
        namespace['update_param_tp_status'](mod)
        for param in mod.parameters():self.assertEqual((param.tp_rank,param.tp_size),(0,1))


class ConverterRecovery(unittest.TestCase):
    def test_resume_after_mid_conversion_failure(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';out=Path(d)/'new';names=f.MODULES[:2];fixture(src,names)
            encode=f.encode;calls=[0]
            def fail_second(*args):
                calls[0]+=1
                if calls[0]==2:raise RuntimeError('interrupted')
                return encode(*args)
            with patch.object(f,'encode',side_effect=fail_second),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):c.convert(src,out,rows=64,modules=names)
            m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['complete'])
            self.assertEqual(set(m['tensors']),{names[0]});first=out/m['tensors'][names[0]]['file'];mtime=first.stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):c.convert(src,out,modules=names)
            self.assertEqual(first.stat().st_mtime_ns,mtime);f.read_manifest(out,names)

    def test_output_symlinks_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source';out=Path(d)/'new';fixture(src,f.MODULES[:1]);out.mkdir()
            (out/'manifest.json.tmp').symlink_to(src/'config.json');before=f.sha256(src/'config.json')
            with self.assertRaisesRegex(ValueError,'symlink'):c.convert(src,out,modules=f.MODULES[:1])
            self.assertEqual(before,f.sha256(src/'config.json'))


class StartupSafety(unittest.TestCase):
    def test_overlay_only_invalid_or_missing_sidecar_fails_closed(self):
        path=ROOT/'overlay/overlay/sitecustomize.py'
        code=f"import sys;sys.path.insert(0,{str(path.parent)!r});exec(compile(open({str(path)!r}).read(),{str(path)!r},'exec'))"
        for mode in ('bad','nvfp4'):
            env=dict(os.environ,GLM_ATTN_WEIGHTS=mode);env.pop('GLM_ATTN_NVFP4_DIR',None)
            r=subprocess.run([sys.executable,'-S','-c',code],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,78);self.assertIn('ValueError',r.stderr)

if __name__=='__main__':unittest.main()
