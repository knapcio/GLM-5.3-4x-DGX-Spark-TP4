# SPDX-License-Identifier: Apache-2.0
"""FP4x ABI, actual kernel-body CPU model, import and release compatibility."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'overlay/bringup'))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))
import glm_fp4_kv as G
from fp4_kv_layout import layout


def reference():
    p = ROOT/'tests/fixtures/fp4_probe_quant_dc5ad9e.py'
    m = importlib.util.spec_from_file_location('fp4_reference', p)
    module = importlib.util.module_from_spec(m)
    m.loader.exec_module(module)
    return module


class Wiring(unittest.TestCase):
    def test_fp8_is_inert_and_boot_fixed(self):
        before = list(sys.meta_path)
        with patch.object(G, '_FORMAT', None):
            self.assertFalse(G.register({'GLM_KV_FORMAT':'fp8'}))
            self.assertEqual(before, sys.meta_path)
            with self.assertRaisesRegex(RuntimeError, 'fixed'):
                G.register(dict(GLM_KV_FORMAT='fp4x', GLM_FULL_MLA='triton', VLLM_USE_V2_MODEL_RUNNER='1', GLM_MLA_SPLIT_K='32'))

    def test_explicit_fp8_dry_matches_unset_release(self):
        import compare_dry
        with patch.dict(os.environ, GLM_KV_FORMAT='fp8'):
            explicit, diff = compare_dry.compare()
            self.assertEqual(diff, [])
            self.assertNotIn('GLM_KV_FORMAT=', explicit)
        with patch.dict(os.environ):
            os.environ.pop('GLM_KV_FORMAT', None)
            default, diff = compare_dry.compare()
            self.assertEqual(diff, [])
        self.assertEqual(explicit, default)

    def test_bad_formats_and_incompatible_flags(self):
        for mode in ('', 'fp4', 'FP4x'):
            with self.assertRaises(ValueError): G.kv_format({'GLM_KV_FORMAT':mode})
        with self.assertRaises(ValueError): G.kv_format({'GLM_KV_FORMAT':'fp4x'})
        with self.assertRaises(ValueError):
            G.kv_format(dict(GLM_KV_FORMAT='fp4x', GLM_FULL_MLA='triton', VLLM_USE_V2_MODEL_RUNNER='1', GLM_MLA_SPLIT_K='32', GLM_FP4_PROBE_SIM='fp4x'))

    def test_original_fp8_kernels_unchanged(self):
        # Includes the pinned split32 body and dirty-L2 reduce.
        expected = json.loads((ROOT/'tests/fixtures/fp4_fp8_originals.json').read_text())
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest, name)

    def test_fp8_byte_loads_and_pointer_casts(self):
        # Mac has no Triton compiler; guard the real-compiler failure in source.
        # Integer masked-load defaults cannot be converted directly to FP8.
        readers = {'_load_latent', '_load_rope'}
        seen = set()
        for path in (ROOT/'overlay/bringup').glob('glm_fp4*kernel.py'):
            tree = ast.parse(path.read_text())
            for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
                for node in ast.walk(fn):
                    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'to' and node.args):
                        continue
                    target = ast.unparse(node.args[0])
                    if target == 'tl.pointer_type(tl.float8e4nv)':
                        # Reinterpret a known pointer before adding FP8 offsets.
                        self.assertIsInstance(node.func.value, ast.Name, (path.name, fn.name))
                        self.assertNotIn(fn.name, readers)
                    if fn.name in readers and target == 'tl.float8e4nv':
                        self.assertIn('bitcast=True', ast.unparse(node), fn.name)
                        seen.add(fn.name)
                if fn.name in readers:
                    self.assertIn('KV.to(tl.pointer_type(tl.uint8))', ast.unparse(fn))
        self.assertEqual(seen, readers)

    def test_reference_is_exact_probe_file(self):
        self.assertEqual(hashlib.sha256((ROOT/'tests/fixtures/fp4_probe_quant_dc5ad9e.py').read_bytes()).hexdigest(),
                         'af0d499cf0b1dca1f165f38b3b194c992245dc84db65eeed676e91ecc4cb6e65')

    def test_sizing_alignment_null_and_k3(self):
        for mode, blocks, tokens, maxlen in (('fp8',1039,66496,66112),('fp4x',1573,100672,100288)):
            l = layout(mode)
            self.assertEqual((l['blocks'],l['tokens'],int(l['max_model_len'])),(blocks,tokens,maxlen))
            self.assertLessEqual(1+(maxlen+3+63)//64, blocks)
            self.assertLessEqual(l['aligned_bytes'], (1<<30)+2145386496)
        self.assertEqual(layout('fp4x')['block_bytes'],2046464)

    def test_pinned_sources_and_reinterpretation_transform(self):
        src = os.environ.get('GLM_IMAGE_SRC')
        if not src:
            self.skipTest('set GLM_IMAGE_SRC to the installed image or saved source parent')
        for name, digest in G.PINS.items():
            p = Path(src)/(name.replace('.', '/')+'.py')
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest, name)
        raw = (Path(src)/(G.COMMON.replace('.', '/')+'.py')).read_text()
        transformed = G.transform_forward(raw)
        ast.parse(transformed)
        self.assertIn('and not self._glm_fp4x', transformed)
        self.assertNotIn('fp4_qdq', transformed)


class Numeric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from fp4_kv_cpu_model import kernels, Ptr, Tensor, tensor
        cls.torch, cls.Ptr, cls.Tensor, cls.tensor = torch, Ptr, Tensor, staticmethod(tensor)
        cls.tl, cls.k = kernels(ROOT/'overlay/bringup')
        cls.ref = reference()

    def pack_read(self, x, rope=None, scale=.75):
        t = self.torch
        n = x.shape[0]
        if rope is None: rope = t.randn(n,64).bfloat16()
        cache = t.full((n+4,368),165,dtype=t.uint8)
        slots = t.arange(n,dtype=t.int64)+2
        ks = t.tensor([scale],dtype=t.float32)
        for row in range(n):
            self.tl.pid=(row,0)
            self.k['_pack_store'](self.Ptr.of(x),self.Ptr.of(rope),self.Ptr.of(cache),self.Ptr.of(slots),
                self.Ptr.of(ks),x.stride(0),rope.stride(0),368,n,n+4)
        out=t.empty(n,576,dtype=t.bfloat16)
        for row in range((n+15)//16):
            self.tl.pid=(row,0)
            self.k['_read_rows'](self.Ptr.of(cache),self.Ptr.of(slots),self.Ptr.of(out),n+4,n,576)
        self.assertTrue(t.equal(out[:,:512].view(t.uint16),self.ref.fp4_qdq(x).view(t.uint16)))
        self.assertTrue(t.equal(out[:,512:].view(t.uint16),self.ref.fp8_codes(rope,ks).bfloat16().view(t.uint16)))
        # Compare the encoder itself, including signed zero and both scales.
        code,block,row_scale=self.ref.fp4_parts(x)
        levels=t.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
        nibble=(code.abs()[...,None]==levels).to(t.int32).argmax(-1).to(t.uint8)
        nibble=(nibble | (t.signbit(code).to(t.uint8)*8)).reshape(n,256,2)
        packed=nibble[:,:,0] | (nibble[:,:,1]<<4)
        self.assertTrue(t.equal(cache[2:2+n,:256],packed))
        self.assertTrue(t.equal(cache[2:2+n,256:288],block.to(t.float8_e4m3fn).view(t.uint8).reshape(n,32)))
        self.assertTrue(t.equal(cache[2:2+n,288:292],row_scale.reshape(n,1).view(t.uint8)))
        self.assertTrue(t.equal(cache[2:2+n,304:],self.ref.fp8_codes(rope,ks).to(t.float8_e4m3fn).view(t.uint8)))
        self.assertTrue(t.all(cache[2:2+n,292:304]==0))
        self.assertTrue(t.all(cache[:2]==165) and t.all(cache[n+2:]==165))
        return cache,slots,out

    def test_pinned_interpreter_random_scale075_bitexact(self):
        t=self.torch
        # Exact line-49 input from the pinned-image failure, including RoPE.
        g=t.Generator().manual_seed(5105)
        x=t.randn(96,512,generator=g).bfloat16();x[0]=0;x[1]=-0.
        for i,e in enumerate((-110,-60,-10,10,60,110)):x[i+2]*=2.**e
        rope=t.randn(96,64,generator=g).bfloat16()
        with patch.object(self.Tensor,'fp8_interpreter',True):
            # Prove this model exposes both defects hidden by PyTorch casts.
            bad=self.tensor([200.,124.666664]).to(self.tl.float8e4nv)
            self.assertEqual(bad.tolist(),[208.,64.])
            self.pack_read(x,rope,.75)
            out=t.empty_like(x)
            for row in range(len(x)):
                self.tl.pid=(row,0)
                self.k['_qdq'](self.Ptr.of(x),self.Ptr.of(out),512,len(x))
            self.assertTrue(t.equal(out.view(t.uint16),self.ref.fp4_qdq(x).view(t.uint16)))

    def test_e4m3_encoder_boundaries_subnormals_signs_bitexact(self):
        import numpy as np
        t=self.torch
        levels=t.arange(127,dtype=t.uint8).view(t.float8_e4m3fn).float().numpy()
        mid=(levels[:-1]+levels[1:])*.5
        values=np.concatenate([levels,mid,np.nextafter(mid,-np.inf),np.nextafter(mid,np.inf)])
        values=np.concatenate([values,-values])
        # Include every finite BF16 RoPE input divided by the failing scale.
        bf=t.arange(65536,dtype=t.int32).to(t.uint16).view(t.bfloat16).float()
        rope=(bf[t.isfinite(bf)]/.75).clamp(-448,448).numpy()
        values=np.concatenate([values,rope]).astype(np.float32)
        expected=t.from_numpy(values).to(t.float8_e4m3fn).view(t.uint8)
        for mode in (False,True):
            with patch.object(self.Tensor,'fp8_interpreter',mode):
                out=self.k['_e4m3_rne'](self.tensor(values))
                actual=t.from_numpy(np.asarray(out).copy()).to(t.float8_e4m3fn).view(t.uint8)
                self.assertTrue(t.equal(actual,expected))

    def test_random_zero_sign_ties_and_dynamic_range_bitexact(self):
        t=self.torch
        g=t.Generator().manual_seed(5105)
        x=t.randn(96,512,generator=g).bfloat16()
        x[0]=0; x[1]=-0.
        for i,e in enumerate((-110,-60,-10,10,60,110)):
            x[i+2]=x[i+2]*2.**e
        self.pack_read(x)
        # At scale=1, scale code448 and row=1/448 make exact E2M1 ties
        # independently test the actual threshold body via _parts.
        vals=t.tensor([.25,.75,1.25,1.75,2.5,3.5,5.,6.]).repeat(64).reshape(1,512).bfloat16()
        self.pack_read(vals)

    def test_negative_oob_slots_and_nonfinite_fail_closed(self):
        t=self.torch
        x=t.ones(3,512,dtype=t.bfloat16);r=t.ones(3,64,dtype=t.bfloat16)
        cache=t.full((4,368),165,dtype=t.uint8);slots=t.tensor([-1,4,-200],dtype=t.int64)
        ks=t.ones(1)
        for row in range(3):
            self.tl.pid=(row,0)
            self.k['_pack_store'](self.Ptr.of(x),self.Ptr.of(r),self.Ptr.of(cache),self.Ptr.of(slots),self.Ptr.of(ks),512,64,368,3,4)
        self.assertTrue(t.all(cache==165))
        out=t.empty(3,576,dtype=t.bfloat16)
        self.tl.pid=(0,0)
        self.k['_read_rows'](self.Ptr.of(cache),self.Ptr.of(slots),self.Ptr.of(out),4,3,576)
        self.assertTrue(t.all(out==0))
        x[0,0]=float('nan');slots[0]=1
        self.tl.pid=(0,0)
        with self.assertRaisesRegex(ValueError,'NaN'):
            self.k['_pack_store'](self.Ptr.of(x),self.Ptr.of(r),self.Ptr.of(cache),self.Ptr.of(slots),self.Ptr.of(ks),512,64,368,3,4)

    def test_page_crossing_gather_and_prefix_resume(self):
        t=self.torch
        cache,slots,decoded=self.pack_read(t.randn(130,512).bfloat16())
        # Permuted physical pages, partial context start, two requests.
        bt=t.tensor([[1,0,2],[0,2,1]],dtype=t.int32)
        cu=t.tensor([0,20,42],dtype=t.int32)
        ts=t.cat([t.zeros(20,dtype=t.int32),t.ones(22,dtype=t.int32)])
        starts=t.tensor([55,60],dtype=t.int32)
        out=t.empty(42,576,dtype=t.bfloat16)
        for pid in range(3):
            self.tl.pid=(pid,0)
            self.k['_gather'](self.Ptr.of(cache),self.Ptr.of(out),self.Ptr.of(bt),self.Ptr.of(cu),
                self.Ptr.of(ts),self.Ptr.of(starts),64,3,576,134,42,True)
        for i in range(42):
            req=int(ts[i]);pos=i-int(cu[req])+int(starts[req])
            slot=int(bt[req,pos//64])*64+pos%64
            if 2<=slot<132:
                self.assertTrue(t.equal(out[i].view(t.uint16),decoded[slot-2].view(t.uint16)))

    def test_sparse_unsplit_split32_and_empty_rows(self):
        t=self.torch
        cache,_,decoded=self.pack_read(t.randn(96,512).bfloat16())
        q=t.randn(2,16,512).bfloat16();qr=t.randn(2,16,64).bfloat16()
        idx=t.full((2,64),-1,dtype=t.int32)
        idx[0,:35]=t.arange(2,37)
        out=t.empty_like(q)
        common=(q.stride(0),q.stride(1),qr.stride(0),qr.stride(1),368,64,out.stride(0),16,64,64,100,.0625*1.4426950408889634,1.,False)
        for row in range(2):
            self.tl.pid=(row,0)
            self.k['_mla'](self.Ptr.of(q),self.Ptr.of(qr),self.Ptr.of(cache),self.Ptr.of(idx),self.Ptr.of(out),*common)
        self.assertTrue(t.all(out[1]==0))
        keys=decoded[:35,:512].float();rope=decoded[:35,512:].float()
        scores=(q[0].float()@keys.T+qr[0].float()@rope.T)*.0625
        expected=t.softmax(scores,dim=1).bfloat16().float()@keys
        self.assertTrue(t.allclose(out[0].float(),expected,atol=.025,rtol=.02))
        for splits in (8,16,32):
            acc=t.empty(2,splits,16,512);md=t.empty(2,splits,2,16)
            merged=t.empty_like(q)
            for row in range(2):
                for part in range(splits):
                    self.tl.pid=(row,part)
                    self.k['_mla_partial'](self.Ptr.of(q),self.Ptr.of(qr),self.Ptr.of(cache),self.Ptr.of(idx),
                        self.Ptr.of(acc),self.Ptr.of(md),*common,splits,16)
                for hd in range(128):
                    self.tl.pid=(row,hd)
                    self.k['_mla_reduce'](self.Ptr.of(acc),self.Ptr.of(md),self.Ptr.of(merged),merged.stride(0),splits)
            self.assertTrue(t.all(merged[1]==0))
            self.assertTrue(t.allclose(out.float(),merged.float(),atol=.025,rtol=.02))

    def test_fresh_prefill_qdq_bitexact(self):
        t=self.torch
        x=t.randn(17,512).bfloat16();x[0,0]=float('nan');x[1,0]=float('inf')
        out=t.empty_like(x)
        for row in range(17):
            self.tl.pid=(row,0)
            with __import__('numpy').errstate(invalid='ignore',divide='ignore'):
                self.k['_qdq'](self.Ptr.of(x),self.Ptr.of(out),512,17)
        expect=self.ref.fp4_qdq(x)
        finite=t.isfinite(expect)
        self.assertTrue(t.equal(out[finite].view(t.uint16),expect[finite].view(t.uint16)))
        self.assertTrue(t.equal(t.isnan(out),t.isnan(expect)))
        self.assertTrue(t.equal(t.isinf(out),t.isinf(expect)))

    def test_fleet_bf16_dumps_bitexact(self):
        root=os.environ.get('FP4_DUMPS')
        if not root:
            self.skipTest('set FP4_DUMPS to the probe campaign')
        files=sorted(Path(root).glob('normale/captures/*.pt'))
        self.assertTrue(files,'fleet BF16 captures missing')
        seen=set();rows=0;manifest=[]
        for p in files:
            e=self.torch.load(p,map_location='cpu',weights_only=True)
            key=(e['case'],e['layer_id'],e['stage'])
            if key in seen:continue
            seen.add(key)
            x=e['latent']; live=e['write_slots']>=0
            x=x[live]
            indices=sorted(set([0,len(x)//2,len(x)-1]))
            sample=x[indices].contiguous()
            for mode in (False,True):
                with patch.object(self.Tensor,'fp8_interpreter',mode):
                    self.pack_read(sample,e['rope'][live][indices].contiguous(),float(e['k_scale']))
            rows+=len(sample)
            manifest.append(dict(file=str(p.relative_to(Path(root))),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                 rows=indices,case=key[0],layer=key[1],stage=key[2]))
        self.assertEqual({key[1] for key in seen},{0,39,77,78})
        self.assertEqual({key[0] for key in seen},{'prose','code','long16k','long60k'})
        if os.environ.get('FP4_RESULTS'):
            Path(os.environ['FP4_RESULTS'],'kernel-ast-bitexact.json').write_text(json.dumps(dict(
                status='PASS_KERNEL_AST_BITEXACT',torch=self.torch.__version__,groups=len(seen),rows=rows,dumps=manifest,
                scope='Actual kernel AST CPU model, not Triton interpreter/compiler or GPU evidence'),indent=2)+'\n')
        print('FLEET_BITEXACT',len(seen),'case/layer/stage groups;',rows,'BF16 rows')


if __name__=='__main__':unittest.main()
