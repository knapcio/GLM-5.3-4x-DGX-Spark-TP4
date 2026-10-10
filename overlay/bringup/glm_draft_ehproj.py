# SPDX-License-Identifier: Apache-2.0
"""Boot-only replicated FP8 W8A16 MTP eh_proj; target unchanged."""
import functools
import hashlib
import importlib.util
import json
import sys
import os
from pathlib import Path
import torch
import glm_draft_head as dh
from glm_draft_ehproj_config import options, byte_cost

FORMAT='0'
FP8_MODULE='vllm.model_executor.layers.quantization.utils.marlin_utils_fp8'
FLOOR_BYTES=int(4.5*(1 << 30))


def tensor_sha(weight):
    """Hash native BF16 bytes with at most 64 rows of host scratch."""
    digest=hashlib.sha256()
    for first in range(0,len(weight),64):
        chunk=weight[first:first+64].detach().cpu().contiguous().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
        del chunk
    return digest.hexdigest()


def checkpoint_source(runner, original, layer_name):
    """Resolve the native loader's local files; never download or iterate weights.

    The pinned MTP eh_proj is a replicated nn.Linear, not a TP-sharded linear.
    Its checkpoint name is unchanged by DeepSeekMTP._rewrite_spec_layer_name.
    """
    from safetensors import safe_open
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    draft=runner.speculator.model
    config=runner.speculator.draft_model_config
    if not Path(config.model).is_dir() or getattr(draft,'secondary_weights',()):
        raise ValueError('rollback requires one local native checkpoint source')
    if getattr(original.weight,'weight_loader',default_weight_loader) is not default_weight_loader:
        raise ValueError('replicated native eh_proj loader required')
    loader=DefaultModelLoader(runner.load_config)
    _,files,safe=loader._prepare_weights(config.model,None,config.revision,
        getattr(draft,'fall_back_to_pt_during_load',True),
        getattr(draft,'allow_patterns_overrides',None))
    if not safe:raise ValueError('rollback requires safetensors')
    name=f'model.layers.{layer_name}.eh_proj.weight'
    matches=[]
    for path in files:
        with safe_open(path,framework='pt',device='cpu') as sf:
            if name not in sf.keys():continue
            view=sf.get_slice(name)
            if view.get_shape()!=list(original.weight.shape) or view.get_dtype()!='BF16':
                raise ValueError('checkpoint eh_proj geometry/dtype mismatch')
            matches.append(str(Path(path).resolve()))
    if len(matches)!=1:raise ValueError('exactly one checkpoint eh_proj required')
    return dict(path=matches[0],tensor=name,shape=list(original.weight.shape),
                bytes=original.weight.numel()*2,expected_sha256=tensor_sha(original.weight),
                replicated=True,loader='default_weight_loader')


def rollback(runner):
    """Single boot attempt, only after draft graphs are released and arms OFF."""
    source=runner._draft_ehproj_source
    receipt=dict(source=dict(file=source['path'],tensor=source['tensor'],
                            replicated=True,loader=source['loader']),
                 bytes=source['bytes'],sha256=None,
                 expected_sha256=source['expected_sha256'],qualified=False)
    runner._draft_ehproj_receipt['ehproj_rollback']=receipt
    runner._draft_ehproj_ready=False
    print('ehproj_rollback '+json.dumps(receipt,sort_keys=True),flush=True)
    error=None;native=None;available=None;reserve=None
    try:
        if getattr(runner,'_draft_ehproj_rollback_attempted',False):
            raise RuntimeError('eh_proj rollback is single-use per boot')
        runner._draft_ehproj_rollback_attempted=True
        bank=runner._draft_ehproj
        draft=runner.speculator.model
        layer=draft.model.layers[runner._draft_ehproj_layer]
        if getattr(draft,'compiled',False) or getattr(draft,'aot_compiled_fn',None) is not None:
            raise RuntimeError('eh_proj rollback refuses cached compiled draft callables')
        if (layer.eh_proj is not bank or runner._draft_head.on or
            any(m.graphs for m in dh.managers(runner)) or
            (getattr(runner.speculator,'_glm_rowselect',None) is not None and
             not runner.speculator._glm_rowselect.get('disabled',False))):
            raise RuntimeError('rollback requires drained native OFF arms')
        device=next(bank.parameters()).device
        available=dh.memory()
        # Reserve both the mmap's touched pages and the device BF16 allocation,
        # plus two 64-row hash buffers. FP8 is still resident at this boundary.
        reserve=2*source['bytes']+2*min(64,source['shape'][0])*source['shape'][1]*2
        if available-reserve<FLOOR_BYTES:
            raise RuntimeError('eh_proj rollback memory floor refused')
    except Exception as exc:error=exc
    valid=dh.qualification_term('ehproj_rollback_admission',dict(no_exception=error is None),
        dict(receipt=receipt,mem_available=available,
             reserve_bytes=reserve,floor_bytes=FLOOR_BYTES),error)
    dh.agree({'ehproj_rollback_admission':receipt},valid)
    if error is not None:raise RuntimeError('eh_proj rollback admission refused') from error
    try:
        from safetensors import safe_open
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        with safe_open(source['path'],framework='pt',device='cpu') as sf:
            view=sf.get_slice(source['tensor'])
            if view.get_shape()!=source['shape'] or view.get_dtype()!='BF16':
                raise ValueError('reloaded eh_proj geometry/dtype mismatch')
            # Only this tensor is faulted into memory. No other model weights.
            weight=view[:,:]
            receipt['sha256']=tensor_sha(weight)
            if receipt['sha256']!=source['expected_sha256']:
                raise ValueError('reloaded eh_proj SHA differs from native load')
            n,k=source['shape']
            native=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
            native.weight=torch.nn.Parameter(torch.empty((n,k),device=device,dtype=torch.bfloat16),False)
            default_weight_loader(native.weight,weight)
            if tensor_sha(native.weight)!=source['expected_sha256']:
                raise ValueError('restored device eh_proj SHA mismatch')
            weight=None
        if dh.memory()<FLOOR_BYTES:raise RuntimeError('eh_proj rollback post-load memory floor refused')
    except Exception as exc:
        error=exc
        error.__traceback__=None
    valid=dh.qualification_term('ehproj_rollback_loaded',dict(no_exception=error is None),receipt,error)
    print('ehproj_rollback '+json.dumps(receipt,sort_keys=True),flush=True)
    dh.agree({'ehproj_rollback_loaded':receipt},valid)
    if error is not None:raise RuntimeError('eh_proj rollback reload refused') from error
    # Preserve the module identity used by native forward factories, but remove
    # every packed parameter, scale and workspace before fallback graph capture.
    try:
        bank.__class__=torch.nn.Linear
        bank.__dict__.clear()
        bank.__dict__.update(native.__dict__)
        runner.model_memory_usage+=source['bytes']-runner._draft_ehproj_receipt['total']
    except Exception as exc:error=exc
    valid=dh.qualification_term('ehproj_rollback_restored',dict(no_exception=error is None),receipt,error)
    dh.agree({'ehproj_rollback_restored':receipt},valid)
    if error is not None:raise RuntimeError('eh_proj rollback replacement refused') from error
    print('ehproj_rollback '+json.dumps(receipt,sort_keys=True),flush=True)


def rollback_qualified(runner):
    receipt=runner._draft_ehproj_receipt['ehproj_rollback']
    dh.agree({'ehproj_rollback':dict(receipt,qualified=True)},
             type(runner._draft_ehproj) is torch.nn.Linear and
             runner._draft_ehproj.weight.dtype is torch.bfloat16)
    receipt['qualified']=True
    print('ehproj_rollback '+json.dumps(receipt,sort_keys=True),flush=True)


def encode_fp8(w):
    """Per-output-channel BF16 scales; same QDQ as held-out falsifier."""
    if w.ndim!=2 or not torch.isfinite(w).all():
        raise ValueError('finite projection matrix required')
    z=w.float()
    s=(z.abs().amax(-1,keepdim=True).clamp_min(1e-10)/448).bfloat16()
    q=(z/s.float()).clamp(-448,448).to(torch.float8_e4m3fn)
    return q,s


class FP8Bank(torch.nn.Module):
    def __init__(self, original, prepare=None, sms=None):
        super().__init__()
        w=original.weight.detach()
        if w.dtype!=torch.bfloat16 or tuple(w.shape)!=(6144,12288) or original.bias is not None:
            raise ValueError('bias-free BF16 replicated 6144x12288 eh_proj required')
        if prepare is None:
            if not w.is_cuda:raise ValueError('CUDA projection required')
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
            prepare=prepare_fp8_layer_for_marlin
        self.input_size_per_partition=12288
        self.output_size_per_partition=6144
        self.orig_dtype=torch.bfloat16
        self.weight_block_size=None
        self.logical_widths=[6144]
        self.weight=torch.nn.Parameter(torch.empty(w.shape,device=w.device,dtype=torch.float8_e4m3fn),False)
        self.weight_scale=torch.nn.Parameter(torch.empty((6144,1),device=w.device,dtype=torch.bfloat16),False)
        for first in range(0,6144,64):
            q,s=encode_fp8(w[first:first+64])
            self.weight.data[first:first+len(q)].copy_(q)
            self.weight_scale.data[first:first+len(s)].copy_(s)
        prepare(self,size_k_first=False,input_dtype=None)
        self.resident_bytes=dh.storage_bytes(self)
        sms=torch.cuda.get_device_properties(w.device).multi_processor_count if sms is None else sms
        if self.resident_bytes!=byte_cost(sms=sms)['total']:
            raise RuntimeError('FP8 projection persistent byte mismatch')

    def forward(self,x):
        if x.dtype!=torch.bfloat16 or x.shape[-1]!=12288:
            raise ValueError('BF16 concatenated projection input required')
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
        return apply_fp8_marlin_linear(input=x,weight=self.weight,weight_scale=self.weight_scale,
            workspace=self.workspace,size_n=6144,size_k=12288,bias=None,
            input_dtype=None,use_fp32_reduce=True)


def storage_ptrs(module):
    return {t.untyped_storage().data_ptr() for t in
        list(module.parameters())+list(module.buffers()) if t.numel()}


def prepare(runner,factory=FP8Bank,vote=dh.agree,source_reader=checkpoint_source):
    error=None;descriptor={}
    try:
        if getattr(runner,'_draft_ehproj',None) is not None:
            raise RuntimeError('projection preparation is single-use per boot')
        draft=runner.speculator.model
        layers=list(draft.model.layers.values())
        if type(draft).__name__!='DeepSeekMTP' or draft is runner.model or len(layers)!=1:
            raise ValueError('one native draft MTP layer required')
        layer_name=next(iter(draft.model.layers))
        original=layers[0].eh_proj
        if type(original) is not torch.nn.Linear or tuple(original.weight.shape)!=(6144,12288):
            raise ValueError('unmodified replicated native eh_proj required')
        reload_source=source_reader(runner,original,layer_name)
        source=original.weight.untyped_storage()
        # Scan both parameters and buffers: the target may keep a storage alias.
        retain=source.data_ptr() in storage_ptrs(runner.model)
        if sum(m is original for _,m in draft.named_modules(remove_duplicate=False))!=1:
            raise ValueError('unexpected draft projection owner')
        target=runner.model.lm_head
        bank=factory(original)
        cost=byte_cost(sms=bank.workspace.numel(),retain=retain)
        if bank.resident_bytes!=cost['total'] or source.nbytes()!=cost['bf16_source']:
            raise RuntimeError('projection/source byte accounting mismatch')
        descriptor=dict(format=FORMAT,shape=[6144,12288],replicated=True,
                        rollback_source=reload_source,**cost)
    except Exception as exc:error=exc
    vote({'ehproj_prepare':descriptor},error is None)
    if error is not None:raise RuntimeError('eh_proj preparation refused') from error
    # Native nn.Linear returns only a Tensor. The bank preserves that interface.
    layers[0].eh_proj=bank
    if runner.model.lm_head is not target:
        raise RuntimeError('target ownership changed')
    runner._draft_ehproj=bank
    runner._draft_ehproj_source=reload_source
    runner._draft_ehproj_layer=layer_name
    runner._draft_ehproj_receipt=descriptor
    runner._draft_ehproj_ready=False
    runner.model_memory_usage+=cost['delta']
    sys.stderr.write('GLM_DRAFT_EHPROJ prepared '+json.dumps(descriptor,sort_keys=True)+'\n')
    # No reference to original is retained here. Shared target aliases stay alive.


def install_runner(mod):
    cls=mod.GPUModelRunner
    load,capture=cls.load_model,cls.capture_model
    @functools.wraps(load)
    def loaded(self,*a,**kw):
        result=load(self,*a,**kw)
        prepare(self)
        return result
    @functools.wraps(capture)
    def captured(self,*a,**kw):
        self._draft_ehproj_ready=False
        target=self.model.lm_head
        method=target.quant_method
        # eh_proj was attached after load and before the first graph capture.
        # The inner cand3 INIT path runs glm_draft_head_qual on this exact bank.
        result=capture(self,*a,**kw)
        rollback_receipt=self._draft_ehproj_receipt.get('ehproj_rollback')
        if rollback_receipt is not None:
            # Inner DH/rowselect refusal has already restored and qualified OFF.
            try:
                dh.agree({'ehproj_native_fallback':rollback_receipt},
                    rollback_receipt.get('qualified') is True and
                    not self._draft_head.on and not self._draft_head_ready and
                    type(self._draft_ehproj) is torch.nn.Linear)
            except BaseException:
                rollback_receipt['qualified']=False
                self.speculator._kstop.bad='eh_proj native fallback agreement refused'
                raise
            return result
        error=None
        try:
            report=self._draft_head_qualification
            if (not self._draft_head_ready or not report or
                report.get('target_fixed_hidden',{}).get('bit_exact') is not True or
                not report.get('cases') or self.model.lm_head is not target or
                target.quant_method is not method or
                self.speculator.model.model.layers[self._draft_ehproj_layer].eh_proj is not self._draft_ehproj):
                raise RuntimeError('missing combined INIT native replay qualification')
        except Exception as exc:error=exc
        valid=dh.qualification_term('ehproj_captured',
            dict(combined_init_qualified=error is None),
            dict(receipt=self._draft_ehproj_receipt), error)
        try:
            dh.agree({'ehproj_captured':self._draft_ehproj_receipt},valid)
        except dh.VoteRefused as exc:
            dh.initial_fallback(self,exc)
            return result
        if error is not None:
            self._draft_head_ready=False
            self.speculator._kstop.bad='eh_proj qualification failed; whole candidate stop required'
            raise RuntimeError('eh_proj capture refused') from error
        self._draft_ehproj_ready=True
        self._draft_ehproj_receipt['qualification']=report
        sys.stderr.write('GLM_DRAFT_EHPROJ READY '+json.dumps(self._draft_ehproj_receipt,sort_keys=True)+'\n')
        return result
    cls.load_model,cls.capture_model=loaded,captured


def register(env=None):
    global FORMAT
    env=os.environ if env is None else env
    FORMAT=options(env)
    if FORMAT=='0':return False
    # The head import hooks must execute first: its load/INIT qualification is inner.
    from glm_skip_mla_plan import Hooks
    root=Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    pins=dh.PINS|{FP8_MODULE:'72c324a9de072e60c1f126ad3d5e7f077f6a8752590435e088634f798683d12a',
        'vllm.model_executor.model_loader.default_loader':'9c9d54b1b650bf5affc924ebb8b8c73711e187ae7486cadd9f737bb1279cc7b8',
        'vllm.model_executor.model_loader.weight_utils':'b42ff2bd9301c4034592598f86a03df7d60a9fc7e13576054deb44bce9ceccf7'}
    for name,expected in pins.items():
        path=root/(name.removeprefix('vllm.').replace('.','/')+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('eh_proj source drift '+name)
    if dh.RUNNER in sys.modules:raise RuntimeError('eh_proj must register before runner import')
    hooks=Hooks();sys.meta_path.insert(0,hooks)
    hooks.after_import(dh.RUNNER,install_runner)
    return True
