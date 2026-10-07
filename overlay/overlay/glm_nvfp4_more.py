# SPDX-License-Identifier: Apache-2.0
"""Real W4A16 group sidecars; no INT8 re-encoding or BF16 resident bank."""
import functools
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import glm_nvfp4_format as fmt
import glm_nvfp4_groups as selection

LINEAR = 'vllm.model_executor.layers.linear'
MODEL = 'vllm.model_executor.models.deepseek_v2'


def read_manifest(root, enabled, verify=True):
    root = Path(root).resolve()
    m = json.loads((root/'manifest.json').read_text())
    wanted = set(enabled)-{'attn'}
    if (m.get('schema') != 2 or m.get('algorithm') != fmt.ALGORITHM or not m.get('complete') or
        not wanted.issubset(m.get('groups', []))):
        raise ValueError('incomplete/incompatible group sidecar')
    for name,e in m['tensors'].items():
        if selection.classify(name) != e['group'] or e['group'] not in m['groups']:
            raise ValueError('invalid sidecar group inventory')
        p = root/e['file']
        if p.parent.resolve() != root or p.is_symlink() or (verify and fmt.sha256(p) != e['sha256']):
            raise ValueError('group sidecar hash/path mismatch')
    counts = {g:sum(e['group'] == g for e in m['tensors'].values()) for g in m['groups']}
    if counts != {g:selection.COUNTS[g] for g in m['groups']}:
        raise ValueError('incomplete full group inventory')
    return m


def transform(weights, files, root, enabled, manifest=None):
    from safetensors.torch import load_file
    import glm_mtp_select
    ctx = glm_mtp_select.active()
    draft = ctx is not None and ctx.kind == 'draft'
    wanted = {'mtp'} & set(enabled) if draft else set(enabled)-{'attn','mtp'}
    if not wanted:
        yield from weights
        return
    # MTP context is mandatory: otherwise a full scan could be mistaken for a target load.
    if 'mtp' in enabled and ctx is None:
        raise ValueError('MTP NVFP4 requires native MTP load context')
    m = read_manifest(root, enabled) if manifest is None else manifest
    checkpoint = Path(files[0]).resolve().parent
    for leaf in ('index','config'):
        filename = 'model.safetensors.index.json' if leaf == 'index' else 'config.json'
        if fmt.sha256(checkpoint/filename) != m['source'][leaf+'_sha256']:
            raise ValueError('group sidecar source mismatch')
    chosen = {n:e for n,e in m['tensors'].items() if e['group'] in wanted}
    seen = set()
    companions = set()
    for name,value in weights:
        prefix,_,leaf = name.rpartition('.')
        if prefix not in chosen:
            yield name,value
            continue
        entry = chosen[prefix]
        if leaf not in entry['leaves'] or name in companions:
            raise ValueError('duplicate/unexpected group companion: '+name)
        companions.add(name)
        if leaf != entry['leaves'][0]:
            continue
        seen.add(prefix)
        tensors = load_file(str(Path(root)/entry['file']),device='cpu')
        fmt.validate_tensors(tensors,entry['shape'])
        for out_leaf,tensor in tensors.items():
            yield prefix+'.'+out_leaf,tensor
    if seen != set(chosen) or len(companions) != sum(len(e['leaves']) for e in chosen.values()):
        raise ValueError('incomplete real group replacement')
    counts = {g:sum(e['group'] == g for e in chosen.values()) for g in sorted(wanted)}
    sys.stderr.write('glm-nvfp4-more: loaded '+json.dumps(counts,sort_keys=True)+'; '+('draft' if draft else 'target')+'; real Marlin W4A16\n')


def patch_moe_config(module):
    cls = module.CompressedTensorsConfig
    original = cls.get_quant_method
    if getattr(original,'_glm_nvfp4_more',False):
        return
    @functools.wraps(original)
    def get_quant_method(self,layer,prefix):
        from glm_nvfp4_attn import mode
        if mode() == 'nvfp4' and 'mtp' in selection.groups() and selection.classify(prefix) == 'mtp' and prefix.endswith('.mlp.experts'):
            from glm_nvfp4_mtp import new_method
            return new_method(layer.moe_config)
        return original(self,layer,prefix)
    get_quant_method._glm_nvfp4_more = True
    cls.get_quant_method = get_quant_method


def patch_linear(module):
    # The pinned fused WK/weights projection deliberately passes quant_config=None.
    # Supply a CT linear method only at that exact constructor, before parameter allocation.
    cls = module.MergedColumnParallelLinear
    original = cls.__init__
    if getattr(original,'_glm_nvfp4_more',False):
        return
    signature = inspect.signature(original)
    class IndexerConfig:
        def get_quant_method(self,layer,prefix):
            from glm_nvfp4_attn import new_scheme
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
            layer.scheme = new_scheme()
            return CompressedTensorsLinearMethod(self)
    @functools.wraps(original)
    def init(*args,**kwargs):
        bound = signature.bind(*args,**kwargs)
        prefix = bound.arguments.get('prefix','')
        from glm_nvfp4_attn import mode
        if mode() == 'nvfp4' and 'indexer' in selection.groups() and selection.classify(prefix) == 'indexer' and prefix.endswith('.wk_weights_proj'):
            if not bound.arguments.get('disable_tp',False):
                raise ValueError('indexer fused projection must remain replicated')
            bound.arguments['quant_config'] = IndexerConfig()
        return original(*bound.args,**bound.kwargs)
    init._glm_nvfp4_more = True
    cls.__init__ = init


def patch_indexer_helper(module):
    original = module._try_load_fp8_indexer_wk
    if getattr(original,'_glm_nvfp4_more',False):
        return
    @functools.wraps(original)
    def load(name,tensor,*args,**kwargs):
        prefix,_,leaf = name.rpartition('.')
        from glm_nvfp4_attn import mode
        # The stock FP8 helper treats any WK weight_scale as an FP8 companion
        # and would swallow our block16 E4M3 scale before the fused CT loader.
        if (mode() == 'nvfp4' and 'indexer' in selection.groups() and
            selection.classify(prefix) == 'indexer' and leaf in
                ('weight_packed','weight_scale','weight_global_scale')):
            return False
        return original(name,tensor,*args,**kwargs)
    load._glm_nvfp4_more = True
    module._try_load_fp8_indexer_wk = load


def register():
    spec = importlib.util.find_spec('vllm')
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError('NVFP4 extra groups require pinned vLLM sources')
    source_root = Path(next(iter(spec.submodule_search_locations)))
    pins = json.loads(Path(__file__).with_name('nvfp4_more_source_pins.json').read_text())
    for relative, expected in pins.items():
        source = source_root/relative
        if not source.is_file() or fmt.sha256(source) != expected:
            raise RuntimeError('NVFP4 extra group source drift: '+relative)
    class Hook(importlib.abc.MetaPathFinder):
        _glm_nvfp4_more = True
        def find_spec(self,name,path,target=None):
            if name not in (LINEAR,MODEL):
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0,self)
            execute = spec.loader.exec_module
            def patched(module):
                execute(module)
                (patch_linear if name == LINEAR else patch_indexer_helper)(module)
            spec.loader.exec_module = patched
            return spec
    if not any(getattr(h,'_glm_nvfp4_more',False) for h in sys.meta_path):
        sys.meta_path.insert(0,Hook())
    if LINEAR in sys.modules:
        patch_linear(sys.modules[LINEAR])
    if MODEL in sys.modules:
        patch_indexer_helper(sys.modules[MODEL])
