#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU serving-image load preflight. No SSH, payload reads, GPU, or model boot.

Host: --dry dry.txt --image <local pinned v11> --headers headers.json --out report.json
Image worker: --worker --vector vector.json --headers headers.json --out report.json
Missing checkpoint metadata is a FAIL, never an admission. --config is useful
for independent construction/regression diagnostics but cannot certify headers.
"""
import argparse
from collections import Counter
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = 'ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6'


def vectors(text):
    result=[]
    for line in text.splitlines():
        if not line.startswith('docker run '):continue
        parts=shlex.split(line)
        if 'serve' not in parts:continue
        i=parts.index('serve');env={};mounts={}
        for j,t in enumerate(parts[:i-1]):
            if t in ('-e','--env'):
                key,sep,value=parts[j+1].partition('=')
                if not sep:raise ValueError('DRY must contain explicit env values')
                env[key]=value
            if t in ('-v','--volume'):
                source,dest,*_=parts[j+1].split(':');mounts[dest]=source
        args=parts[i+1:];rank=int(args[args.index('--node-rank')+1]) if '--node-rank' in args else 0
        result.append(dict(rank=rank,image=parts[i-1],env=env,args=args,mounts=mounts))
    if not result or len({v['rank'] for v in result})!=len(result):raise ValueError('missing/duplicate DRY rank vector')
    return sorted(result,key=lambda v:v['rank'])


class Report:
    def __init__(self):self.rows=[];self.started=time.monotonic();self.fakes=[]
    def check(self,name,fn):
        start=time.monotonic()
        try:
            detail=fn();row=dict(check=name,status='PASS',detail=detail)
        except (Exception,SystemExit) as exc:
            row=dict(check=name,status='FAIL',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        row['seconds']=round(time.monotonic()-start,3);self.rows.append(row)
        print(f"{row['status']} {name}: {row.get('error',row.get('detail'))}",flush=True)
        return row['status']=='PASS'
    def result(self):
        return dict(schema=1,ok=all(r['status']=='PASS' for r in self.rows),checks=self.rows,
                    seconds=round(time.monotonic()-self.started,3),fakes=self.fakes,
                    scope='CPU structure, loader contracts and API reachability only; no CUDA/numeric/quality admission')


def option(args,name,default=None):
    if name not in args:return default
    return args[args.index(name)+1]


def overlay_paths(root):
    sys.path[:0]=[str(root/'overlay'/p) for p in ('overlay','bringup','kstop')]


def settings(vector):
    overlay_paths(ROOT)
    import glm_nvfp4_attn as attn,glm_nvfp4_groups as groups
    import glm_adaptive_chunk as adaptive,glm_fp4_kv as kv,glm_mtp_kstop as kstop
    mode=attn.mode();selected=groups.groups();kvfmt=kv.kv_format(os.environ)
    # Execute actual startup parsers, rather than duplicating their accepted values.
    if os.environ.get('GLM_MTP_FIX')=='1':
        import glm_mtp_fix
        glm_mtp_fix.register()
    import glm_full_mla,glm_recent_kv
    glm_full_mla.register()
    kv.register()
    glm_recent_kv.register()
    recent=glm_recent_kv.options(os.environ)
    kstop.register(os.environ)
    adaptive_on,threshold=adaptive.settings(os.environ)
    if adaptive_on and int(option(vector['args'],'--max-num-batched-tokens',4096))!=4096:
        raise ValueError('adaptive prefill requires fixed 4096 constructor capacity')
    if mode=='nvfp4':
        attn.register()
    return dict(weights=mode,groups=selected,kv=kvfmt,adaptive=adaptive_on,threshold=threshold,recent=recent)


def fake_hardware(report,rank):
    """FAKE: SM121 capability + TP/PP groups, no drivers/collectives are used.

    Retain real quant-method/backend selection. Hardware availability probes
    are shape-independent and cannot be tested on this CPU.
    """
    import torch
    from types import SimpleNamespace as NS
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    current_platform.device_type='cuda'
    type(current_platform).get_device_capability=classmethod(lambda cls,*a,**k:DeviceCapability(12,1))
    current_platform.get_device_name=lambda *a,**k:'FAKE SM121 CPU preflight'
    current_platform.get_device_total_memory=lambda *a,**k:128<<30
    import vllm.distributed.parallel_state as ps
    ps._TP=NS(world_size=4,rank_in_group=rank,device_group=NS(size=lambda:4,rank=lambda:rank))
    single=NS(world_size=1,rank_in_group=0,is_first_rank=True,is_last_rank=True,device_group=NS(size=lambda:1,rank=lambda:0))
    ps._PP=single;ps._PCP=single;ps._DCP=single;ps._DP=single;ps._EP=single
    report.fakes.extend(['SM121 capability and TP4/EP1 group metadata (no collectives)',
                         'MLA backend GPU implementation; real MLA, projections and indexer constructors run'])
    return current_platform


_FAKE_LIBS=[]

def fake_silu():
    import torch
    lib=torch.library.Library('_C','FRAGMENT')
    try:lib.define('silu_and_mul(Tensor(a!) result, Tensor input) -> ()')
    except RuntimeError:pass
    def silu(out,x):
        out.copy_(torch.nn.functional.silu(x[...,:x.shape[-1]//2])*x[...,x.shape[-1]//2:])
    lib.impl('silu_and_mul',silu,'CPU')
    _FAKE_LIBS.append(lib)


def build(vector,config_dir,report):
    import torch
    platform=fake_hardware(report,vector['rank'])
    from vllm.engine.arg_utils import EngineArgs
    args=vector['args']
    # Consume real launch arguments with the pinned parser; only execution-only
    # network/tokenizer/compile options are changed for a CPU construction probe.
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    p=make_arg_parser(FlexibleArgumentParser())
    ns=p.parse_args([str(config_dir),*args[1:]])
    ns.model=str(config_dir);ns.tokenizer=str(config_dir);ns.skip_tokenizer_init=True
    ns.reasoning_parser=None;ns.tool_call_parser=None
    ns.enforce_eager=True;ns.nnodes=1;ns.node_rank=0;ns.distributed_executor_backend='mp'
    engine_args=EngineArgs.from_cli_args(ns)
    cfg=engine_args.create_engine_config()
    cfg.device_config.device=torch.device('meta')
    from vllm.config import set_current_vllm_config
    from vllm.platforms.interface import PlatformEnum
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    import vllm.model_executor.layers.attention.mla_attention as mla
    class FakeMLAImpl:
        is_sparse=True;supports_dense_mha_prefill=False;dcp_world_size=1
        supports_quant_query_input=False;q_pad_num_heads=None
        def __init__(self,*a,**kw):self.topk_indices_buffer=kw['topk_indices_buffer']
    class FakeMLABackend:
        @staticmethod
        def get_name():return option(args,'--attention-backend','FLASHINFER_MLA_SPARSE_SM90')
        @staticmethod
        def is_mla():return True
        @staticmethod
        def get_impl_cls():return FakeMLAImpl
    mla.get_attn_backend=lambda *a,**kw:FakeMLABackend
    fake_silu()
    from types import SimpleNamespace as NS
    import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared
    shared.aux_stream=lambda:NS()
    torch.cuda.Event=lambda *a,**kw:NS()
    report.fakes.append('shared-expert auxiliary CUDA stream/events (construction only)')
    type(platform)._enum=PlatformEnum.CUDA;platform.device_type='meta'
    # No CUDA context: the marlin availability probe also consults torch.cuda.
    torch.cuda.is_available=lambda:True
    torch.cuda.get_device_capability=lambda *a:(12,1)
    torch.cuda.current_device=lambda:0
    torch.cuda.is_current_stream_capturing=lambda:False
    report.fakes.append('CUDA stream capture query = false during load')
    from vllm.model_executor.model_loader.utils import configure_quant_config
    configure_quant_config(cfg.quant_config,GlmMoeDsaForCausalLM)
    torch.set_default_dtype(cfg.model_config.dtype)
    with set_current_vllm_config(cfg),torch.device('meta'):
        target=GlmMoeDsaForCausalLM(vllm_config=cfg)
        draft_cfg=copy.copy(cfg)
        draft_cfg.compilation_config=copy.copy(cfg.compilation_config)
        draft_cfg.compilation_config.static_forward_context={}
        draft_cfg.model_config=cfg.speculative_config.draft_model_config
        draft_cfg.quant_config=copy.deepcopy(cfg.quant_config)
        configure_quant_config(draft_cfg.quant_config,DeepSeekMTP)
        # The actual MTP model shares the real checkpoint's config and methods.
        with set_current_vllm_config(draft_cfg):draft=DeepSeekMTP(vllm_config=draft_cfg)
    return target,draft,cfg


DTYPES = {'U8':'uint8','I8':'int8','I16':'int16','I32':'int32','I64':'int64',
          'BF16':'bfloat16','F16':'float16','F32':'float32','F64':'float64',
          'F8_E4M3':'float8_e4m3fn','F8_E5M2':'float8_e5m2','BOOL':'bool'}


def check_adaptive(models):
    import glm_adaptive_chunk as adaptive
    from types import SimpleNamespace as NS
    enabled,threshold=adaptive.settings(os.environ)
    if not enabled:return 'disabled'
    import importlib.util
    spec=importlib.util.find_spec('vllm.v1.core.sched.scheduler')
    text=adaptive.transform(Path(spec.origin).read_text())
    compile('from __future__ import annotations\n'+text,spec.origin,'exec')
    scheduler=NS(scheduler_config=models['cfg'].scheduler_config,max_num_scheduled_tokens=2048)
    scheduler.scheduler_config.max_num_batched_tokens=4096
    budget=adaptive.StepBudget(scheduler,threshold)
    return dict(threshold=threshold,constructor_capacity=4096,initial_budget=budget.limit,source_pin=adaptive.PIN)


def effective_headers(data):
    from checkpoint_headers import unpack,sha
    import glm_nvfp4_groups as groups,glm_nvfp4_format as fmt,glm_nvfp4_attn as attn
    tensors=dict(data['tensors']);enabled=groups.groups()
    if attn.mode()!='nvfp4':return tensors
    sidecars=[]
    for cache in data['cache']['sidecars']:
        docs,hs=unpack(cache);m=json.loads(docs['manifest.json'])
        if not m.get('complete') or m.get('algorithm')!=fmt.ALGORITHM:raise ValueError('incompatible sidecar manifest')
        for key,file in [('index','model.safetensors.index.json'),('config','config.json')]:
            if m['source'][key+'_sha256']!=sha(data['docs'][file]):raise ValueError('sidecar source '+key+' mismatch')
        sidecars.append((m,hs))
    for group in enabled:
        matches=[(m,hs) for m,hs in sidecars if (m.get('schema')==1 if group=='attn' else group in m.get('groups',[]))]
        if len(matches)!=1:raise ValueError('missing/duplicate sidecar for '+group)
        m,hs=matches[0];entries={n:e for n,e in m['tensors'].items() if (group=='attn' or e['group']==group)}
        if len(entries)!=groups.COUNTS[group]:raise ValueError('incomplete sidecar inventory: '+group)
        for prefix,e in entries.items():
            if groups.classify(prefix)!=group:raise ValueError('sidecar name/group mismatch: '+prefix)
            n,k=e['shape']
            expected={'weight_packed':('U8',[n,k//2]),'weight_scale':('F8_E4M3',[n,k//16]),'weight_global_scale':('F32',[1])}
            if k%128:raise ValueError('sidecar input size must divide 128')
            for leaf,(dtype,shape) in expected.items():
                # Each converter output is one logical matrix with unprefixed leaves.
                candidates=[t for key,t in hs.items() if key==e['file']+'/'+leaf and t['file']==e['file']]
                if not candidates:raise ValueError('missing sidecar header: '+prefix+'.'+leaf)
                h=candidates[0]
                if h['dtype']!=dtype or h['shape']!=shape:raise ValueError('sidecar shape/dtype mismatch: '+prefix+'.'+leaf)
            leaves=e.get('leaves',['weight_packed','weight_scale','weight_shape'])
            for leaf in leaves:
                if prefix+'.'+leaf not in tensors:raise ValueError('missing source companion: '+prefix+'.'+leaf)
                del tensors[prefix+'.'+leaf]
            for leaf in expected:tensors[prefix+'.'+leaf]=next(t for key,t in hs.items() if key==e['file']+'/'+leaf and t['file']==e['file'])
    return tensors


def check_parameters(models,data,vector,report):
    if not data:raise ValueError('real headers unavailable; model construction alone cannot certify names/shapes/dtypes')
    import torch
    headers=effective_headers(data);seen={};loaded={}
    # Actual model.load_weights performs expert/stacked/indexer/MTP remapping and
    # TP slicing. copy_ becomes a structural checker on meta tensors, including
    # dtype (torch.copy_ would otherwise silently cast a wrong checkpoint dtype).
    original=torch.Tensor.copy_
    def structural_copy(dst,src,*args,**kw):
        if dst.shape!=src.shape:raise ValueError(f'loader shape mismatch {seen.get("name")}: {tuple(src.shape)} -> {tuple(dst.shape)}')
        if dst.dtype!=src.dtype:raise ValueError(f'loader dtype mismatch {seen.get("name")}: {src.dtype} -> {dst.dtype}')
        return dst
    report.fakes.append('weight_shape payload inferred from packed companion dimensions; tensor copy is shape/dtype-only')
    def weights(kind):
        for name,h in headers.items():
            is_draft=bool(re.search(r'\.layers\.78\.',name))
            if kind=='target' and is_draft:continue
            if kind=='draft' and not is_draft and name!='model.embed_tokens.weight':continue
            seen['name']=name
            if h['dtype'] not in DTYPES:raise ValueError('unsupported checkpoint dtype '+h['dtype'])
            if name.endswith('.weight_shape'):
                if h['shape']!=[2]:raise ValueError('weight_shape header must be a pair: '+name)
                packed=headers[name.removesuffix('weight_shape')+'weight_packed']
                # CT int32 packs 4 int8 / 8 int4 along input dimension.
                routed=bool(re.search(r'\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.mlp\.experts\.',name))
                group=8 if routed else 4
                # Weight_shape is an int64 pair in this checkpoint. Its values
                # cannot be read from headers; loaders only retain these buffers.
                t=torch.tensor([packed['shape'][0],packed['shape'][1]*group],dtype=getattr(torch,DTYPES[h['dtype']]))
            else:t=torch.empty(h['shape'],dtype=getattr(torch,DTYPES[h['dtype']]),device='meta')
            yield name,t
    try:
        torch.Tensor.copy_=structural_copy
        for kind in ('target','draft'):
            model=models[kind];loaded[kind]=model.load_weights(weights(kind))
            missing=set(dict(model.named_parameters()))-loaded[kind]
            # input globals are intentionally fabricated unity for W4A16 MTP;
            # rotary/cache scales absent from the checkpoint are not weights.
            missing={n for n in missing if not n.endswith(('w13_input_global_scale','w2_input_global_scale','kv_scale','k_scale','v_scale','q_scale','prob_scale'))}
            if missing:raise ValueError(kind+' unloaded parameters: '+', '.join(sorted(missing)[:20]))
    finally:torch.Tensor.copy_=original
    return {k:len(v) for k,v in loaded.items()}


@contextlib.contextmanager
def fake_values(report):
    import torch
    # FAKE tensor values only: headers cannot establish finiteness, positivity,
    # scale range or equality. All Python name/dtype/shape/backend assertions run.
    old_item=torch.Tensor.item;old_any=torch.Tensor.any;old_all=torch.Tensor.all;old_close=torch.allclose
    torch.Tensor.item=lambda t,*a,**kw:1 if t.is_meta else old_item(t,*a,**kw)
    torch.Tensor.any=lambda t,*a,**kw:torch.tensor(False,device='cpu') if t.is_meta else old_any(t,*a,**kw)
    torch.Tensor.all=lambda t,*a,**kw:torch.tensor(True,device='cpu') if t.is_meta else old_all(t,*a,**kw)
    torch.allclose=lambda a,b,*args,**kw:True if a.is_meta or b.is_meta else old_close(a,b,*args,**kw)
    report.fakes.append('meta value reductions: unity globals, finite nonnegative scales; no tensor-value validation')
    try:yield
    finally:
        torch.Tensor.item=old_item;torch.Tensor.any=old_any;torch.Tensor.all=old_all;torch.allclose=old_close


def process_hooks(models,report):
    import torch
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    from vllm.config import set_current_vllm_config
    old_units=mu.num_compute_units;mu.num_compute_units=lambda *a:40
    old_workspace=mu.marlin_make_workspace_new
    mu.marlin_make_workspace_new=lambda device,*a,**kw:torch.empty(1024,device='meta',dtype=torch.int32)
    patched={}
    def patch(name,fn):patched[name]=getattr(ops,name,None);setattr(ops,name,fn)
    patch('gptq_marlin_repack',lambda b_q_weight,perm,size_k,size_n,num_bits,**kw:torch.empty((size_k//16,size_n*16//(32//num_bits)),device='meta',dtype=torch.int32))
    patch('awq_marlin_repack',lambda b_q_weight,size_k,size_n,num_bits,**kw:torch.empty((size_k//16,size_n*16//(32//num_bits)),device='meta',dtype=torch.int32))
    patch('gptq_marlin_moe_repack',lambda w,perm,size_k,size_n,num_bits,**kw:torch.empty((w.shape[0],size_k//16,size_n*16//(32//num_bits)),device='meta',dtype=torch.int32))
    # Imported aliases of workspace must be patched, too.
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mf
    old_fp4_workspace=mf.marlin_make_workspace_new;mf.marlin_make_workspace_new=mu.marlin_make_workspace_new
    import vllm.model_executor.layers.attention.mla_attention as mla
    old_dequant=mla.get_and_maybe_dequant_weights
    mla.get_and_maybe_dequant_weights=lambda layer,out_dtype:torch.empty(
        (layer.output_size_per_partition,layer.input_size_per_partition),device='meta',dtype=out_dtype)
    report.fakes.append('CUDA KV-b projection dequantization: meta shape only; real MLA post-load assertions run')
    counts=Counter();errors=[]
    report.fakes.append('CUDA Marlin repack and workspace allocation (meta shape outputs)')
    try:
        with fake_values(report),set_current_vllm_config(models['cfg']),torch.device('meta'):
            for kind in ('target','draft'):
                for name,layer in list(models[kind].named_modules()):
                    method=getattr(layer,'quant_method',None)
                    if method is None or not hasattr(method,'process_weights_after_loading'):continue
                    try:method.process_weights_after_loading(layer);counts[type(method).__name__]+=1
                    except Exception as exc:
                        errors.append(kind+'.'+name+': '+type(exc).__name__+': '+str(exc))
                        if len(errors)==1:print(traceback.format_exc(),flush=True)
            for kind in ('target','draft'):
                for name,layer in models[kind].named_modules():
                    if isinstance(layer,mla.MLAAttention):
                        try:layer.process_weights_after_loading(models['cfg'].model_config.dtype);counts['MLAAttention']+=1
                        except Exception as exc:
                            errors.append(kind+'.'+name+': '+type(exc).__name__+': '+str(exc))
                            if len(errors)==1:print(traceback.format_exc(),flush=True)
    finally:
        mla.get_and_maybe_dequant_weights=old_dequant
        mu.num_compute_units=old_units
        mu.marlin_make_workspace_new=old_workspace;mf.marlin_make_workspace_new=old_fp4_workspace
        for name,fn in patched.items():
            if fn is None:delattr(ops,name)
            else:setattr(ops,name,fn)
    if errors:raise RuntimeError('; '.join(errors[:8])+f' ({len(errors)} failed hooks)')
    return dict(counts)


def check_kstop(models,report):
    if os.environ.get('GLM_MTP_KSTOP','0')!='1':return 'disabled'
    import torch,kstop_runtime as k
    from types import SimpleNamespace as NS
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    # Run real initialize, including its exact boot-c method rejection. Binding
    # and local CUDA control are checked separately; no collective is attempted.
    old_binding=k.model_binding;old_local=k.check_local;old_state=k.STATE
    moes=[m for m in models['target'].modules() if isinstance(m,DeepseekV2MoE)]
    def binding(runner):
        cfg=runner.model.config
        bad={key:getattr(cfg,key,None) for key,value in k.FULL_GLM.items() if getattr(cfg,key,None)!=value}
        if bad:raise ValueError('full GLM binding differs: '+str(bad))
        want=sum(i>=cfg.first_k_dense_replace and i%(getattr(cfg,'moe_layer_freq',1) or 1)==0 for i in range(cfg.num_hidden_layers))
        if len(moes)!=want or any(m.n_routed_experts!=256 for m in moes):raise ValueError('target MoE count/expert count differs')
        import importlib.util
        vroot=Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
        pins=json.loads(Path(k.__file__).with_name('compat_source_pins.json').read_text())
        for name,digest in pins.items():
            path=vroot/(name.removeprefix('vllm.').replace('.','/')+'.py')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise RuntimeError('kstop compatibility source drift: '+name)
        if type(models['target']).__name__!=k.TARGET_ARCH or type(models['draft']).__name__!=k.DRAFT_ARCH:
            raise ValueError('target/draft class binding differs')
        return moes
    owner=NS(max_num_reqs=4,device='cpu',speculative_config=models['cfg'].speculative_config)
    runtime=k.Runtime(owner)
    control=Path(os.environ['GLM_MTP_KSTOP_CONTROL'])
    k.check_control(json.loads(control.read_text()))
    runner=NS(model=models['target'],speculator=NS(model=models['draft'],_kstop=runtime),
              max_num_tokens=16,device='cpu')
    report.fakes.append('kstop speculator shell and local CUDA control/row-remap; real initialize guard, target/draft binding and compatibility pins checked')
    k.model_binding=binding;k.check_local=lambda *a:None;k.STATE=None
    # deadrow_ops registers a CUDA custom op; registration has no GPU work.
    try:k.initialize(runner)
    finally:k.model_binding=old_binding;k.check_local=old_local;k.STATE=old_state
    return dict(target_moes=len(moes),pad_hygiene=runner.speculator._kstop.pad_hygiene)


def check_apply(models,report):
    import torch
    from vllm import _custom_ops as ops
    import glm_nvfp4_mtp
    methods=[(m,getattr(m,'_quant_method',None)) for m in models['draft'].modules()]
    selected=[(layer,q) for layer,q in methods if type(q).__name__=='NativeMTPNVFP4']
    if not selected:return 'NVFP4 MTP not selected'
    # Leave _custom_ops API intact (boot d must fail). Replace only GPU bodies.
    originals={n:getattr(ops,n,None) for n in ('moe_wna16_marlin_gemm',)}
    ops.moe_wna16_marlin_gemm=lambda inp,out,*a,**kw:out.zero_()
    # The pinned CUDA SiLU op is represented by an explicitly registered CPU fake.
    # Do NOT invent a _custom_ops.silu_and_mul wrapper: that hid boot d.
    report.fakes.append('moe_wna16_marlin_gemm and _C.silu_and_mul CPU body; real apply_experts API lookups run')
    from types import SimpleNamespace as NS
    # Tiny activations but real method implementation. Repacked weights are not
    # read by fake GEMMs; retain real layer attributes and dtype contracts.
    probe=NS(intermediate_size_per_partition=128,num_experts=256,global_num_experts=256,
             expert_map=None,apply_router_weight_on_input=False,workspace=torch.empty(0))
    for which in ('gate','up','down'):
        for suffix in ('','_scale','_global'):setattr(probe,'_glm_'+which+suffix,torch.empty(0))
    try:
        out=glm_nvfp4_mtp.apply_experts(probe,torch.ones((1,128),dtype=torch.bfloat16),
                torch.ones((1,8),dtype=torch.float32),torch.zeros((1,8),dtype=torch.int32),
                align=lambda *a,**kw:(torch.zeros(8,dtype=torch.int32),torch.zeros(1,dtype=torch.int32),torch.tensor([8],dtype=torch.int32)),scalar=object())
        if out.shape!=(1,128):raise ValueError('unexpected MTP output shape')
    finally:
        for n,fn in originals.items():
            if fn is None:delattr(ops,n)
            else:setattr(ops,n,fn)
    return 'real apply_experts reached pinned SiLU API'


def worker(a):
    report=Report();vector=json.loads(a.vector.read_text())
    # Preserve all rendered feature env. No sitecustomize auto-exit at interpreter startup.
    os.environ.update(vector['env'])
    for key,value in list(os.environ.items()):
        if key.startswith('GLM_') and value.startswith('/overlay/'):
            os.environ[key]=str(ROOT/'overlay'/value.removeprefix('/overlay/'))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
    data={};models={}
    report.check('settings',lambda:settings(vector))
    def headers():
        if not a.headers:raise ValueError('real header cache missing; export with scripts/checkpoint_headers.py')
        from checkpoint_headers import unpack,sha
        cache=json.loads(a.headers.read_text())
        if cache.get('schema')!=1:raise ValueError('unsupported header cache schema')
        expected_model=vector['mounts'].get('/model')
        if expected_model and cache['model']['source']!=expected_model:raise ValueError('cache checkpoint path differs from DRY mount')
        if vector['env'].get('GLM_ATTN_WEIGHTS')=='nvfp4':
            import glm_nvfp4_groups
            selected=glm_nvfp4_groups.groups(vector['env'])
            dirs=['GLM_ATTN_NVFP4_DIR'] if 'attn' in selected else []
            if set(selected)-{'attn'}:dirs.append('GLM_NVFP4_MORE_DIR')
            for key in dirs:
                mount=vector['mounts'].get(vector['env'].get(key))
                if not mount or sum(s['source']==mount for s in cache['sidecars'])!=1:
                    raise ValueError('cache sidecar path differs from DRY mount: '+key)
        docs,tensors=unpack(cache['model'])
        pinned=json.loads((ROOT/'manifests/target.json').read_text())
        files={f['f']:f for f in pinned['files']}
        for name in ('config.json','model.safetensors.index.json'):
            if sha(docs[name])!=files[name]['sha256']:raise ValueError('checkpoint metadata differs from pinned target: '+name)
        for name,receipt in cache['model']['files'].items():
            if name not in files or receipt['size']!=files[name]['size']:raise ValueError('checkpoint shard size differs from pinned target: '+name)
        data.update(cache=cache,docs=docs,tensors=tensors)
        for sidecar in cache['sidecars']:unpack(sidecar)
        return dict(checkpoint_tensors=len(tensors),sidecars=len(cache['sidecars']))
    report.check('checkpoint-headers',headers)
    with tempfile.TemporaryDirectory() as td:
        cfgdir=Path(td)
        if data:
            (cfgdir/'config.json').write_bytes(data['docs']['config.json'])
        elif a.config:
            (cfgdir/'config.json').write_bytes(a.config.read_bytes())
        def construct():
            if not (cfgdir/'config.json').exists():raise ValueError('real config missing')
            target,draft,cfg=build(vector,cfgdir,report)
            models.update(target=target,draft=draft,cfg=cfg)
            return {kind:dict(parameters=len(list(model.named_parameters())),methods=dict(Counter(
                type(m.quant_method).__name__
                for m in model.modules() if hasattr(m,'quant_method'))))
                for kind,model in [('target',target),('draft',draft)]}
        report.check('target+mtp-meta-construction',construct)
        if models:
            report.check('adaptive-chunk-startup',lambda:check_adaptive(models))
            report.check('parameter-contracts',lambda:check_parameters(models,data,vector,report))
            report.check('process-weights-hooks',lambda:process_hooks(models,report))
            report.check('kstop-pad-hygiene',lambda:check_kstop(models,report))
            report.check('mtp-apply-api',lambda:check_apply(models,report))
    result=report.result()
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted((ROOT/'overlay').rglob('*.py'))}
    result['rank']=vector['rank']
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['ok'] else 1


def host(a):
    vs=vectors(a.dry.read_text())
    if a.rank is None and [v['rank'] for v in vs]!=list(range(4)):
        raise ValueError('full preflight requires DRY ranks 0..3; --rank is diagnostic only')
    if a.rank is not None:vs=[v for v in vs if v['rank']==a.rank]
    if not vs:raise ValueError('requested rank absent')
    for path in (a.headers,a.config):
        if path and not path.is_file():raise ValueError('missing metadata file: '+str(path))
    started=time.monotonic();reports=[]
    # Same vector/features for each rank; rank itself affects loader slices.
    with tempfile.TemporaryDirectory(prefix='.boot-preflight-',dir=a.root) as td:
        work=Path(td)
        for v in vs:
            remain=a.budget-(time.monotonic()-started)
            if remain<=0:
                reports.append(dict(rank=v['rank'],report=dict(ok=False,error='120s total preflight budget exceeded')))
                break
            (work/'vector.json').write_text(json.dumps(v))
            container_name='boot-preflight-'+work.name.strip('.')+'-r'+str(v['rank'])
            command=['docker','run','--rm','--name',container_name,'--pull','never','--network','none','--cpus','4','--memory','8g',
                     '-v',str(a.root.resolve())+':/pkg:ro','-v',str(work)+':/work',
                     '--entrypoint','python3',a.image or v['image'],'-B','/pkg/scripts/boot_preflight.py',
                     '--worker','--vector','/work/vector.json','--out','/work/report.json']
            # File mounts must precede image, never inherit DRY GPU/network/LD_PRELOAD.
            for name,path in [('headers',a.headers),('config',a.config)]:
                if path:
                    pos=command.index('--entrypoint');command[pos:pos]=['-v',str(path.resolve())+':/'+name+'.json:ro']
                    command+=['--'+name,'/'+name+'.json']
            try:
                r=subprocess.run(command,capture_output=True,text=True,timeout=remain)
                print('\n'.join(line for line in r.stdout.splitlines() if line.startswith(('PASS ','FAIL '))),flush=True)
                if (work/'report.json').exists():row=json.loads((work/'report.json').read_text());(work/'report.json').unlink()
                else:row=dict(ok=False,error='image worker failed',stdout=r.stdout,stderr=r.stderr)
                if r.returncode:row['ok']=False
            except subprocess.TimeoutExpired:
                subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=5)
                row=dict(ok=False,error='120s total preflight budget exceeded')
            reports.append(dict(rank=v['rank'],image=a.image or v['image'],rendered_image=v['image'],report=row))
    result=dict(ok=all(x['report']['ok'] for x in reports),ranks=reports,seconds=round(time.monotonic()-started,3),
                dry_sha256=hashlib.sha256(a.dry.read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result,indent=2)+'\n');return 0 if result['ok'] else 1


def advisory(root, dry, out, headers=None, image=None, enabled=True):
    """Best-effort cached-header diagnostic; never a boot admission decision."""
    result = dict(ok=None, status='SKIPPED', reason='disabled')
    if enabled and (not headers or not Path(headers).is_file()):
        result['reason'] = 'header cache absent'
    elif enabled:
        command = [sys.executable, str(Path(root)/'scripts/boot_preflight.py'),
                   '--root', str(root), '--dry', str(dry), '--headers', str(headers),
                   '--out', str(out), '--budget', '120']
        if image: command += ['--image', str(image)]
        try:
            run = subprocess.run(command, timeout=125, check=False)
            result = json.loads(Path(out).read_text())
            if run.returncode or result.get('ok') is not True:
                result.update(status='WARNING', reason='cached-header preflight failed')
            else:
                result['status'] = 'PASS'
        except Exception as exc:
            result = dict(ok=False, status='WARNING', reason=type(exc).__name__+': '+str(exc))
    if result['status'] != 'PASS':
        print('WARNING: CPU boot preflight '+result['reason']+'; continuing with launcher safety checks', file=sys.stderr)
    try:
        Path(out).write_text(json.dumps(result, indent=2)+'\n')
    except Exception as exc:
        print('WARNING: could not save CPU preflight receipt: '+str(exc), file=sys.stderr)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--vector',type=Path);p.add_argument('--dry',type=Path);p.add_argument('--headers',type=Path)
    p.add_argument('--rank',type=int,choices=range(4));p.add_argument('--config',type=Path);p.add_argument('--image');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--budget',type=float,default=120);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.worker and not a.vector:p.error('--worker requires --vector')
    if not a.worker and not a.dry:p.error('--dry required')
    try:return worker(a) if a.worker else host(a)
    except Exception as exc:
        a.out.write_text(json.dumps(dict(ok=False,error=type(exc).__name__+': '+str(exc)))+'\n')
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
