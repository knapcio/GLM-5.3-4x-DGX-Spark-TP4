# SPDX-License-Identifier: Apache-2.0
"""Actual union/expansion ASTs and release dispatch, without a GPU."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay/bringup'), str(ROOT/'tests')]
from fp4_kv_cpu_model import kernels, Ptr
import glm_fp4_prefill as P
import glm_full_mla as F


class UnionPrefill(unittest.TestCase):
    def test_union_expand_exact_values_poisoned_unused_rows_seams_and_reuse(self):
        tl, k = kernels(ROOT/'overlay/bringup')
        gen = torch.Generator().manual_seed(5110)
        n = 131  # page seam plus partial BN tail
        x = torch.randn(n, 512, generator=gen).bfloat16()
        r = torch.randn(n, 64, generator=gen).bfloat16()
        cache = torch.zeros(n, 368, dtype=torch.uint8)
        write = torch.arange(n, dtype=torch.int64)
        scale = torch.tensor([.75])
        for row in range(n):
            tl.pid = (row, 0)
            k['_pack_store'](*map(Ptr.of, (x, r, cache, write, scale)), 512, 64, 368, n, n)
        selected = torch.zeros(n, dtype=torch.int32)
        plain = torch.full((n, 576), float('nan'), dtype=torch.bfloat16)
        slots = torch.tensor([[0, 63, 64, 130, -1, 131, 63],
                              [64, 65, 130, -64, 9999, 65, -1]], dtype=torch.int32)
        for iteration in range(2):
            selected.zero_()
            for row in range(2):
                tl.pid = (row, 0)
                k['_mark_selected'](*map(Ptr.of, (slots, selected)), 7, 7, n)
            want = sorted(set(s for s in slots.flatten().tolist() if 0 <= s < n))
            self.assertEqual(selected.nonzero().flatten().tolist(), want)
            for tile in range((n+15)//16):
                tl.pid = (tile, 0)
                k['_expand_selected'](*map(Ptr.of, (cache, selected, plain)), n)
            for row in want:
                latent = k['_load_latent'](Ptr.of(cache), torch.tensor([row]).numpy(),
                                         tl.arange(0, 512), 368, n)
                rope = k['_load_rope'](Ptr.of(cache), torch.tensor([row]).numpy(),
                                     tl.arange(0, 64), 368, n)
                expected = torch.cat((torch.from_numpy(latent.copy()).flatten(),
                                      torch.from_numpy(rope.copy()).flatten())).bfloat16()
                self.assertTrue(torch.equal(plain[row].view(torch.uint16), expected.view(torch.uint16)))
            # Compare attention against the actual packed reader, using the
            # same online-softmax BN16 ordering and no tolerance for CPU bits.
            raw = ast.parse((ROOT/'overlay/bringup/glm_full_mla_kernel.py').read_text())
            node = copy.deepcopy(next(n for n in raw.body if isinstance(n, ast.FunctionDef) and n.name == '_mla'))
            node.decorator_list = []
            for a in node.args.args: a.annotation = None
            ns = dict(tl=tl)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<plain-mla>', 'exec'), ns)
            q = torch.randn(2, 16, 512, generator=gen).bfloat16()
            qr = torch.randn(2, 16, 64, generator=gen).bfloat16()
            packed_out = torch.empty_like(q); plain_out = torch.empty_like(q)
            for row in range(2):
                tl.pid = (row, 0)
                args = (q.stride(0), q.stride(1), qr.stride(0), qr.stride(1))
                common = (7, 8192, 16, 7, 64, n, .0625, 1., False)
                k['_mla'](*map(Ptr.of, (q, qr, cache, slots, packed_out)), *args, 368, *common)
                ns['_mla'](*map(Ptr.of, (q, qr, plain, slots, plain_out)), *args, 576, *common)
            self.assertTrue(torch.equal(packed_out.view(torch.uint16), plain_out.view(torch.uint16)))
            slots.fill_(-1); slots[:, 0] = 129

    def test_workspace_alias_and_fixed_total(self):
        bank = P.reserve(torch.device('cpu'), 1)
        self.assertEqual(P.bank_bytes(bank) + P.LOGITS_BYTES, P.scratch_bytes(1))
        self.assertLessEqual(P.scratch_bytes(64), 400 << 20)
        self.assertEqual(bank['keys'].untyped_storage().data_ptr(), bank['workspace'].data_ptr())
        self.assertEqual(bank['scales'].untyped_storage().data_ptr(), bank['workspace'].data_ptr())
        plain = bank['workspace'][:P.MLA_ROWS*1152].view(torch.bfloat16).view(P.MLA_ROWS,576)
        ptrs = {name: v.data_ptr() for name,v in bank.items()}
        plain[-1].fill_(2)
        self.assertEqual(ptrs, {name:v.data_ptr() for name,v in P.reserve(torch.device('cpu'),1).items()})

    def test_prefill_decode_and_mixed_dispatch(self):
        name = 'fp4_union_dispatch_fixture'
        raw = 'class FlashInferMLASparseSM90Impl:\n    def forward_mqa(self,*args):pass\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.py'; path.write_text(raw)
            mod = types.SimpleNamespace(__file__=str(path), FlashInferMLASparseSM90Impl=type('Impl',(),{'forward_mqa':lambda *a:None}))
            trace=[]
            def reader(kind):
                def run(q,qr,cache,slots,scale,*a):
                    trace.append((kind,len(q),slots[:,0].tolist()))
                    return torch.full_like(q, {'packed':1,'split':2,'union':3}[kind])
                return run
            modules = dict(glm_full_mla_kernel=types.SimpleNamespace(sparse_mla=reader('plain')),
                           glm_fp4_mla_kernel=types.SimpleNamespace(sparse_mla=reader('packed')),
                           glm_fp4_mla_split_kernel=types.SimpleNamespace(sparse_mla=reader('split')),
                           glm_fp4_mla_prefill=types.SimpleNamespace(prefill_mla=reader('union')))
            mod.triton_convert_req_index_to_global_index = lambda *a, **kw: (a[2],None)
            with patch.dict(sys.modules,modules), patch('glm_fp4_kv.active',return_value=True), \
                 patch.object(F,'PIN',hashlib.sha256(raw.encode()).hexdigest()), \
                 patch.dict(os.environ,GLM_MLA_SPLIT_K='32',GLM_MLA_SPLIT_MAX_ROWS='36'):
                F.install(mod)
                impl=mod.FlashInferMLASparseSM90Impl()
                impl.num_heads=16; impl.kv_lora_rank=512; impl.qk_rope_head_dim=64
                impl.kv_cache_dtype='fp8_e4m3'; impl.scale=.0625
                impl.topk_indices_buffer=torch.arange(40,dtype=torch.int32)[:,None]
                q=torch.zeros(40,16,512,dtype=torch.bfloat16);qr=torch.zeros(40,16,64,dtype=torch.bfloat16)
                for t,nd,np,want in ((3,3,0,[('split',3)]),(40,40,0,[('packed',40)]),
                                     (3,3,1,[('split',3)]),
                                     (3,0,1,[('union',3)]),(40,3,1,[('packed',3),('union',37)]),
                                     (12,3,1,[('split',3),('union',9)])):
                    trace.clear()
                    md=types.SimpleNamespace(num_prefills=np,num_decode_tokens=nd,
                                             req_id_per_token=torch.zeros(t),block_table=None,block_size=64)
                    impl.forward_mqa((q[:t],qr[:t]),torch.zeros(1,64,368,dtype=torch.uint8),md,None)
                    self.assertEqual([(kind,n) for kind,n,_ in trace],want)
                    if nd and np and t>nd: self.assertEqual(trace[-1][2],list(range(nd,t)))


if __name__ == '__main__': unittest.main()
