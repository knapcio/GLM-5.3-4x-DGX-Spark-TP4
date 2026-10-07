# SPDX-License-Identifier: Apache-2.0
"""Default-off, source-pinned attention-only NVFP4 W4A16 loader and Marlin scheme."""
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import glm_nvfp4_format as fmt

CONFIG_MODULE = 'vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors'


def mode(env=None):
    env = os.environ if env is None else env
    value = env.get('GLM_ATTN_WEIGHTS', 'int8')
    if value not in ('int8','nvfp4'):
        raise ValueError('GLM_ATTN_WEIGHTS must be int8 or nvfp4')
    if value == 'nvfp4' and env.get('GLM_NVFP4_WSIM','') not in ('','0'):
        raise ValueError('real NVFP4 and weight simulator cannot be combined')
    return value


def selected(prefix):
    if not isinstance(prefix,str):return False
    m = re.fullmatch(r'(?:.*\.)?layers\.(\d+)\.self_attn\.(.+)',prefix)
    return bool(m and 1 <= int(m[1]) <= 77 and m[2] in (*fmt.PROJECTIONS,'fused_qkv_a_proj'))


def check_sources():
    pins=json.loads((Path(__file__).with_name('nvfp4_source_pins.json')).read_text())
    spec=importlib.util.find_spec('vllm')
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError('glm-nvfp4-attn: vLLM package is unavailable')
    root=Path(next(iter(spec.submodule_search_locations)))
    for name, expected in pins.items():
        relative=name.removeprefix('vllm.').replace('.','/')
        source=root/(relative+'.py')
        if not source.is_file():source=root/relative/'__init__.py'
        if not source.is_file() or fmt.sha256(source) != expected:
            raise RuntimeError('glm-nvfp4-attn: pinned source drift ' + name)


def prepare_layer(layer, prepare=None):
    import torch
    if prepare is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
        prepare=prepare_fp4_layer_for_marlin
    n,k=layer.output_size_per_partition,layer.input_size_per_partition
    widths=layer.logical_widths
    if sum(widths) != n or len(widths) != layer.weight_global_scale.numel():
        raise ValueError('NVFP4 logical shard/global scale mismatch')
    if len(widths) == 1:
        fmt.validate_tensors(dict(weight_packed=layer.weight_packed,weight_scale=layer.weight_scale,
                                  weight_global_scale=layer.weight_global_scale),[n,k])
        layer.weight=torch.nn.Parameter(layer.weight_packed.data,requires_grad=False)
        del layer.weight_packed
        layer.weight_global_scale=torch.nn.Parameter(layer.weight_global_scale.reshape(()),requires_grad=False)
        prepare(layer)
    else:
        # Stock CT takes max(global_scales), which changes independently QDQ'd matrices.
        # Each logical projection gets its exact FP32 multiplier and a Marlin GEMM.
        parts=[];offset=0
        for i,width in enumerate(widths):
            part=torch.nn.Module()
            part.input_size_per_partition=k;part.output_size_per_partition=width
            part.params_dtype=layer.params_dtype
            part.weight=torch.nn.Parameter(layer.weight_packed[offset:offset+width].contiguous(),requires_grad=False)
            part.weight_scale=torch.nn.Parameter(layer.weight_scale[offset:offset+width].contiguous(),requires_grad=False)
            part.weight_global_scale=torch.nn.Parameter(layer.weight_global_scale[i].reshape(()),requires_grad=False)
            fmt.validate_tensors(dict(weight_packed=part.weight,weight_scale=part.weight_scale,
                                      weight_global_scale=part.weight_global_scale.reshape(1)),[width,k])
            prepare(part);parts.append(part);offset+=width
        del layer.weight_packed,layer.weight_scale,layer.weight_global_scale
        layer._glm_nvfp4_parts=torch.nn.ModuleList(parts)


def apply_layer(layer,x,bias=None,apply=None):
    import torch
    if x.dtype != torch.bfloat16:
        raise ValueError('qualified attention NVFP4 path requires BF16 activations')
    if apply is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
        apply=apply_fp4_marlin_linear
    def run(part,b):
        return apply(input=x,weight=part.weight,weight_scale=part.weight_scale,
                     weight_global_scale=part.weight_global_scale,workspace=part.workspace,
                     size_n=part.output_size_per_partition,size_k=part.input_size_per_partition,
                     bias=b,input_dtype=None,use_fp32_reduce=True)
    if not hasattr(layer,'_glm_nvfp4_parts'):
        return run(layer,bias)
    outputs=[];offset=0
    for part in layer._glm_nvfp4_parts:
        n=part.output_size_per_partition
        outputs.append(run(part,None if bias is None else bias[offset:offset+n]));offset+=n
    return torch.cat(outputs,dim=-1)


def new_scheme():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4
    class AttentionNVFP4(CompressedTensorsW4A4Fp4):
        def __init__(self):
            super().__init__(use_a16=True)
            if type(self.kernel).__name__ != 'MarlinNvFp4LinearKernel':
                raise RuntimeError('attention NVFP4 requires pinned Marlin W4A16')
        def process_weights_after_loading(self,layer):
            # Sidecar stores a multiplier, not the CT stock reciprocal convention.
            prepare_layer(layer)
        def apply_weights(self,layer,x,bias=None):
            return apply_layer(layer,x,bias)
    return AttentionNVFP4()


def patch_config(module):
    cls=module.CompressedTensorsConfig
    original=cls.get_scheme
    if getattr(original,'_glm_nvfp4_attn',False):return
    @functools.wraps(original)
    def get_scheme(self,layer,layer_name=None):
        import glm_nvfp4_groups as selection
        enabled = selection.groups()
        if mode() == 'nvfp4' and (('attn' in enabled and selected(layer_name)) or
                (selection.classify(layer_name or '') in enabled and not selected(layer_name))):
            return new_scheme()
        return original(self,layer,layer_name)
    get_scheme._glm_nvfp4_attn=True
    cls.get_scheme=get_scheme
    import glm_nvfp4_groups as selection
    if 'mtp' in selection.groups():
        import glm_nvfp4_more
        glm_nvfp4_more.patch_moe_config(module)


def transform(weights,files,root,modules=fmt.MODULES):
    from safetensors.torch import load_file
    import glm_mtp_select
    ctx=glm_mtp_select.active()
    if ctx is not None and ctx.kind == 'draft':
        yield from weights
        return
    m=fmt.read_manifest(root,modules)
    if not files:
        raise ValueError('NVFP4 target loader has no checkpoint files')
    checkpoint=Path(files[0]).resolve().parent
    if fmt.sha256(checkpoint/'model.safetensors.index.json') != m['source']['index_sha256'] or fmt.sha256(checkpoint/'config.json') != m['source']['config_sha256']:
        raise ValueError('NVFP4 sidecar does not match the original checkpoint')
    seen=set();companions=set()
    for name,value in weights:
        prefix,_,leaf=name.rpartition('.')
        if prefix not in m['tensors']:
            yield name,value
            continue
        if leaf not in ('weight_packed','weight_scale','weight_shape') or name in companions:
            raise ValueError('duplicate/unexpected attention checkpoint companion: ' + name)
        companions.add(name)
        if leaf != 'weight_packed':continue
        if prefix in seen:raise ValueError('duplicate attention weight: ' + prefix)
        seen.add(prefix)
        entry=m['tensors'][prefix]
        tensors=load_file(str(Path(root)/entry['file']),device='cpu')
        fmt.validate_tensors(tensors,entry['shape'])
        for out_leaf,tensor in tensors.items():yield prefix+'.'+out_leaf,tensor
    if seen != set(modules) or len(companions) != len(modules)*3:
        raise ValueError(f'incomplete attention replacement: {len(seen)}/{len(modules)}')
    import glm_nvfp4_groups as selection
    draft_note = 'MTP real group selected' if 'mtp' in selection.groups() else 'MTP INT8'
    sys.stderr.write(f'glm-nvfp4-attn: loaded {len(seen)} logical attention matrices; Marlin W4A16; {draft_note}\n')


def wrap_iterator(original):
    if getattr(original,'_glm_nvfp4_attn',False):return original
    @functools.wraps(original)
    def iterator(files,*args,**kwargs):
        files=list(files)
        import glm_nvfp4_groups as selection
        enabled = selection.groups()
        weights = original(files,*args,**kwargs)
        if 'attn' in enabled:
            weights = transform(weights,files,os.environ['GLM_ATTN_NVFP4_DIR'])
        if enabled != ('attn',):
            import glm_nvfp4_more
            weights = glm_nvfp4_more.transform(weights,files,os.environ['GLM_NVFP4_MORE_DIR'],enabled)
        yield from weights
    iterator._glm_nvfp4_attn=True
    iterator._glm_fast_load=True
    return iterator


def register(env=None):
    env=os.environ if env is None else env
    if mode(env) == 'int8':return False
    if not env.get('GLM_ATTN_NVFP4_DIR'):raise ValueError('NVFP4 needs GLM_ATTN_NVFP4_DIR')
    import glm_nvfp4_groups as selection
    if selection.groups(env) != ('attn',):
        if not env.get('GLM_NVFP4_MORE_DIR'):
            raise ValueError('extra groups require GLM_NVFP4_MORE_DIR')
        if 'mtp' in selection.groups(env) and (env.get('GLM_MTP_ONLY_LOAD') != '1' or env.get('GLM_TARGET_SKIP_MTP') != '1'):
            raise ValueError('MTP NVFP4 requires native selected target/draft loading')
        import glm_nvfp4_more
        glm_nvfp4_more.register()
    check_sources()
    class Hook(importlib.abc.MetaPathFinder):
        _glm_nvfp4_attn=True
        def find_spec(self,name,path,target=None):
            if name != CONFIG_MODULE:return None
            sys.meta_path.remove(self)
            try:spec=importlib.util.find_spec(name)
            finally:sys.meta_path.insert(0,self)
            execute=spec.loader.exec_module
            def patched(module):execute(module);patch_config(module)
            spec.loader.exec_module=patched
            return spec
    if not any(getattr(h,'_glm_nvfp4_attn',False) for h in sys.meta_path):sys.meta_path.insert(0,Hook())
    if CONFIG_MODULE in sys.modules:patch_config(sys.modules[CONFIG_MODULE])
    import glm_fast_load
    glm_fast_load.register()
    return True
