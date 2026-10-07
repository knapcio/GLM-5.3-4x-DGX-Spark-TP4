# SPDX-License-Identifier: Apache-2.0
"""Boot import composition, physical spec/allocator contracts, and A/B launcher."""
import ast
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'scripts')]
import glm_fp4_kv as G
import glm_full_mla as F
import glm_mtp_kstop as K


class Integration(unittest.TestCase):
    def test_actual_hook_order_composes_with_kstop_source_loader(self):
        name='fp4_chain_fixture'
        raw='''
trace=['source']
class FlashInferMLASparseSM90Impl:
    def forward_mqa(self,*args):pass
class FlashInferMLASparseSM90Backend:pass
class _SM90State:
    def __init__(self):
        self.wrapper=type('Wrapper',(),{'run':lambda *a:None})()
_SM90_STATE=None
'''
        digest=hashlib.sha256(raw.encode()).hexdigest()
        before=list(sys.meta_path)
        modules={name:sys.modules.get(name) for name in ('glm_full_mla_kernel','glm_fp4_mla_kernel')}
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            Path(tmp,name+'.py').write_text(raw)
            sys.path.insert(0,tmp)
            for n in modules:
                sys.modules[n]=types.SimpleNamespace(sparse_mla=lambda *a:None)
            try:
                with patch.object(F,'TARGET',name),patch.object(F,'PIN',digest), \
                     patch.dict(G.PINS,{name:digest}),patch.object(G,'SPARSE',name),patch.object(G,'_FORMAT',None), \
                     patch.dict(K.PINS,{name:digest}):
                    F.register({'GLM_FULL_MLA':'triton'})
                    G.register(dict(GLM_KV_FORMAT='fp4x',GLM_FULL_MLA='triton',VLLM_USE_V2_MODEL_RUNNER='1', GLM_MLA_SPLIT_K='32'))
                    sys.meta_path.insert(0,K.Hook())
                    m=importlib.import_module(name)
                    self.assertTrue(m._glm_fp4x_installed)
                    self.assertTrue(m.FlashInferMLASparseSM90Impl.forward_mqa._glm_full_mla)
                    self.assertEqual(m.FlashInferMLASparseSM90Backend.get_kv_cache_shape(1573,64,1,576), (1573,64,368))
                    with self.assertRaisesRegex(RuntimeError,'forbidden'):
                        m._SM90State().wrapper.run()
            finally:
                sys.meta_path[:]=before;sys.path.remove(tmp);sys.modules.pop(name,None)
                for n,mod in modules.items():
                    if mod is None:sys.modules.pop(n,None)
                    else:sys.modules[n]=mod

    def test_spec_semantics_merge_and_mixed_indexer_page_bytes(self):
        import torch
        # Execute the actual pinned spec classes, replacing only their import
        # dependencies. This avoids importing GPU-dependent vLLM on the Mac.
        src=os.environ.get('GLM_IMAGE_SRC')
        if not src:self.skipTest('GLM_IMAGE_SRC needed')
        raw=(Path(src)/'vllm/v1/kv_cache_interface.py').read_text()
        from enum import Enum
        class Quant(Enum):
            NONE=0;INT4_PER_TOKEN_HEAD=1
            @property
            def is_nvfp4(self):return False
            @property
            def is_per_token_head(self):return False
        ns=dict(torch=torch,dataclass=dataclasses.dataclass,replace=dataclasses.replace,
                fields=dataclasses.fields,KVQuantMode=Quant,
                get_dtype_size=lambda dtype:torch.empty((),dtype=dtype).element_size(),
                _apply_alignment_padding=lambda spec:None)
        want={'KVCacheSpec','AttentionSpec','FullAttentionSpec','MLAAttentionSpec','UniformTypeKVCacheSpecs'}
        nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name in want]
        code=compile('from __future__ import annotations\n'+ '\n'.join(ast.unparse(n) for n in nodes),'<pinned-cache-specs>','exec')
        module=types.ModuleType('fp4_spec_fixture');sys.modules[module.__name__]=module
        module.__dict__.update(ns)
        try:
            exec(code,module.__dict__)
            G.install_spec(module)
            C=module.MLAAttentionSpec
            packed=C(block_size=64,num_kv_heads=1,head_size=576,dtype=torch.uint8,cache_dtype_str=G.ABI)
            stock=C(block_size=64,num_kv_heads=1,head_size=576,dtype=torch.float8_e4m3fn,cache_dtype_str='fp8_e4m3')
            idx=C(block_size=64,num_kv_heads=1,head_size=132,dtype=torch.uint8)
            self.assertEqual(packed.head_size,576)
            self.assertEqual(packed.page_size_bytes,64*368)
            self.assertEqual(stock.page_size_bytes,64*576)
            self.assertEqual(idx.page_size_bytes,64*132)
            self.assertNotEqual(stock,packed)
            self.assertEqual(C.merge([packed,packed]),packed)
            with self.assertRaises(AssertionError):C.merge([packed,stock])
            group=module.UniformTypeKVCacheSpecs(block_size=64,kv_cache_specs={
                **{'mla'+str(i):packed for i in range(79)},**{'idx'+str(i):idx for i in range(22)}})
            self.assertEqual(group.page_size_bytes,2046464)
        finally:sys.modules.pop(module.__name__,None)

    def test_new_max_and_ab_use_one_profile_and_original_fp8_geometry(self):
        env=dict(RECIPE_ROOT=str(ROOT),RECIPE_HOSTS='s1 s2 s3 s4',RECIPE_IPS='192.0.2.1 192.0.2.2 192.0.2.3 192.0.2.4',
            IMAGE='image:local',GLM_FULL_MLA='triton',VLLM_USE_V2_MODEL_RUNNER='1', GLM_MLA_SPLIT_K='32',GLM_MTP_KSTOP='1',
            RECIPE_DISPRAM='require',GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse')
        with patch.dict(os.environ,env):
            sp=importlib.util.spec_from_file_location('fp4_cluster',ROOT/'scripts/cluster.py')
            C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
        # The integration accessor and guard file are not fleet reads.
        with patch.dict(C.ENV,env),patch.object(C,'dispram',return_value=object()):
            for mode,maxlen,blocks in (('fp8','66112',1039),('fp4x','66112',1573),('fp4x','100288',1573),('fp4x','28352',1573)):
                with patch.dict(C.ENV,dict(GLM_KV_FORMAT=mode,RECIPE_MAX_MODEL_LEN=maxlen)):
                    sw=C.launch_switches();lay=C.kstop_layout(sw)
                    self.assertEqual(lay['blocks'],blocks)
                    self.assertEqual(lay['kv_bytes'],1<<30)
            with patch.dict(C.ENV,dict(GLM_KV_FORMAT='fp8',RECIPE_MAX_MODEL_LEN='100288')):
                with self.assertRaisesRegex(ValueError,'requires'):C.launch_switches()
            with patch.dict(C.ENV,dict(GLM_KV_FORMAT='fp4x',RECIPE_PROFILE='dspark-k3')):
                with self.assertRaisesRegex(ValueError,'one native'):C.launch_switches()

    def test_dispram_region_plan_seam_and_alignment(self):
        import glm_carveout as C
        from fp4_kv_layout import layout,sizes
        for mode in ('fp8','fp4x'):
            l=layout(mode);ss=sizes(mode,l['blocks'])
            offs,total,head=C.plan_regions(ss,2145386496,2<<20,1<<30)
            self.assertTrue(all(o%512==0 for o in offs))
            self.assertLessEqual(head,(1<<30)+(2<<20))
            self.assertGreater(total,head)
            self.assertTrue(any(o<head<o+s for o,s in zip(offs,ss)))
            self.assertTrue(all(offs[i]+ss[i]<=offs[i+1] for i in range(len(ss)-1)))


if __name__=='__main__':unittest.main()
