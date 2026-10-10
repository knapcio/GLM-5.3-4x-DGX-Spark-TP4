# SPDX-License-Identifier: Apache-2.0
"""Execute pinned native MLA, decoder, MTP layer and _prefill bodies on CPU.

Small deterministic modules substitute CUDA kernels, weights and metadata;
these are plumbing/exactness tests, not native Marlin or CUDA qualification.
"""
import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay/bringup'), str(ROOT/'overlay/overlay')]
import glm_mtp_rowselect as R
from glm_mtp_rowselect_config import options, descriptor_rows
NS = types.SimpleNamespace
ENV = dict(GLM_MOE_DET_ALIGN='1', GLM_MTP_ROWSELECT='1', GLM_MTP_KSTOP='1', GLM_MTP_FIX='1',
           VLLM_USE_V2_MODEL_RUNNER='1', GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse', GLM_DRAFT_HEAD='nvfp4', GLM_DRAFT_HEAD_INIT='1')


def source(name):
    root = os.environ.get('GLM_IMAGE_SRC')
    if root:
        path = Path(root)/(name.replace('.', '/')+'.py')
    else:
        path = Path('/tmp/glm53-rowselect-native')/(name.removeprefix('vllm.').replace('.', '_')+'.py')
    raw = path.read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == R.PINS[name], name
    return raw


def native_method(name, cls, method, globals_):
    raw = source(name)
    tree = ast.parse(raw)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method)
    # Native function's exact source segment preserves its body/annotations.
    text = '\n'.join(raw.splitlines()[f.lineno-1:f.end_lineno])
    import textwrap
    g = dict(globals_)
    import linecache
    code = 'from __future__ import annotations\n'+textwrap.dedent(text)
    filename = '<rowselect-'+cls+'-'+method+'>'
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    exec(compile(code, filename, 'exec'), g)
    return g[method]


class Linear(torch.nn.Module):
    def __init__(self, k, n, tuple_result=True):
        super().__init__()
        self.weight = torch.nn.Parameter(((torch.arange(n*k).reshape(n,k)%7-3)/8).bfloat16(), False)
        self.tuple_result = tuple_result
        self.rows = []
    def forward(self, x):
        self.rows.append(len(x))
        y = (x.float()[:,None,:]*self.weight.float()[None,:,:]).sum(-1).bfloat16()
        return (y, None) if self.tuple_result else y


class Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        y = x if residual is None else (x+residual).bfloat16()
        out = (y.float()*torch.rsqrt(y.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
        return out if residual is None else (out, y)


class Attention(torch.nn.Module):
    def forward(self, q, kv, kpe, output_shape, **kwargs):
        # Explicit full-row KV write occurs before causal attention.
        self.kv = torch.cat((kv,kpe.squeeze(1)), -1).clone()
        values = self.kv[:,:2].repeat(1,2).float()
        scores = torch.einsum('mhd,nhd->mn',q.float(),q.float())
        scores.masked_fill_(torch.ones_like(scores,dtype=torch.bool).triu(1), -float('inf'))
        return (scores.softmax(-1)@values).bfloat16()


class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.rows=[]
        self.experts=torch.nn.ModuleList([Linear(4,4,False),Linear(4,4,False)])
        self.shared=Linear(4,4,False)
    def forward(self,x):
        self.rows.append(len(x))
        gate=x.float().sigmoid().mean(-1,keepdim=True)
        return (gate*self.experts[0](x)+(1-gate)*self.experts[1](x)+self.shared(x)).bfloat16()


def fixture():
    g = dict(torch=torch, DeepseekAttention=type('DeepseekAttention',(),{}),
             DeepseekV2MLP=type('DeepseekV2MLP',(),{}), tensor_model_parallel_all_gather=lambda x,d:x)
    mla = torch.nn.Module()
    mla.forward = types.MethodType(native_method('vllm.model_executor.layers.mla',
        'MultiHeadLatentAttentionWrapper','forward',g), mla)
    mla.q_lora_rank=2; mla.kv_lora_rank=2; mla.qk_rope_head_dim=2
    mla.qk_nope_head_dim=0; mla.qk_head_dim=2; mla.num_heads=2; mla.v_head_dim=2
    mla.fused_qkv_a_proj=Linear(4,6); mla.q_a_layernorm=Norm(); mla.kv_a_layernorm=Norm()
    mla.q_b_proj=Linear(2,4); mla.fuse_qkv_rmsnorm=False; mla.dcp_q_replicate=False
    mla.rotary_emb=None; mla.indexer=None; mla.is_sparse=True; mla.g_proj=None
    mla.mla_attn=Attention(); mla.o_proj=Linear(4,4)
    block = torch.nn.Module()
    block.forward=types.MethodType(native_method('vllm.model_executor.models.deepseek_v2',
        'DeepseekV2DecoderLayer','forward',g),block)
    block.use_sequence_parallel_moe=False; block.use_mha=False; block.layer_idx=78
    block.self_attn=mla; block.input_layernorm=Norm(); block.post_attention_layernorm=Norm()
    block.mlp=MoE(); block.routed_scaling_factor=1.
    layer=torch.nn.Module();layer.mtp_block=block;layer.enorm=Norm();layer.hnorm=Norm()
    layer.eh_proj=Linear(8,4,False);layer.shared_head=Norm()
    layer.forward=types.MethodType(native_method('vllm.model_executor.models.deepseek_mtp',
        'DeepSeekMultiTokenPredictorLayer','forward',g),layer)
    return layer,block,mla


class Tests(unittest.TestCase):
    def test_boot_option_default_and_refusal(self):
        self.assertFalse(options({}));self.assertTrue(options(ENV))
        for key in ENV:
            with self.subTest(key=key),self.assertRaises(ValueError):options(ENV|{key:'ab'})
        self.assertFalse(R.register({}))
        self.assertFalse(hasattr(R,'set_switch'))

    def test_descriptor_accounting(self):
        for m,c in ((2,1),(3,1),(4,1),(6,2),(12,4),(16,4),(4096,4)):
            self.assertEqual(descriptor_rows(m,c),dict(input_rows=m,kv_write_rows=m,attention_rows=m,
                             tail_rows=c,discarded_tail_rows=m-c))
        d=NS(num_tokens=16,num_reqs=4,cg_mode='FULL')
        class D:
            num_tokens=16;num_reqs=4;cg_mode='FULL'
        manager=NS(graphs={D():object()})
        rows=R.account(NS(prefill_cudagraph_manager=manager,decode_cudagraph_manager=manager,max_num_reqs=4))
        self.assertEqual([r['tail_rows'] for r in rows],[4,16])

    def test_native_bodies_64_anchors_selected_logits_feedback_kv_exact(self):
        head=Linear(4,11,False)
        for anchor in range(64):
            m,c=((2,1),(3,1),(4,1),(6,2),(12,4),(16,4),(9,3),(17,4))[anchor%8]
            positions=torch.arange(m)
            hidden=((torch.arange(m*4).reshape(m,4)+anchor)%19/16-0.5).bfloat16()
            embeds=((torch.arange(m*4).reshape(m,4)*3+anchor)%23/16-0.5).bfloat16()
            indices=torch.tensor([(anchor+j*3)%m for j in range(c)],dtype=torch.int64)
            if anchor % 11 == 0 and c > 1:
                indices.zero_()
            full,block,mla=fixture()
            # Distinct target instance, with byte-identical class source/weights.
            target,tb,tm=fixture();target_before=copy.deepcopy(target.state_dict())
            logits,feedback=full(torch.zeros(m,dtype=torch.int64),positions,hidden,embeds)
            kv=mla.mla_attn.kv.clone()
            R.attach(NS(),dict(block=block,mla=mla,block_fn=R.clone_method(block.forward.__func__,'block'),
                mla_fn=R.clone_method(mla.forward.__func__,'mla'),prefill_fn=lambda *a,**kw:None))
            block._glm_rowselect_indices=mla._glm_rowselect_indices=indices
            selected,recycled=full(torch.zeros(m,dtype=torch.int64),positions,hidden,embeds)
            expected=head(Norm()(logits.index_select(0,indices)))
            actual=head(Norm()(selected))
            self.assertTrue(torch.equal(expected,actual),anchor)
            self.assertTrue(torch.equal(feedback.index_select(0,indices),recycled),anchor)
            self.assertTrue(torch.equal(kv,mla.mla_attn.kv),anchor)
            self.assertTrue(torch.equal(expected.argmax(-1),actual.argmax(-1)))
            self.assertTrue(torch.equal(expected.float().softmax(-1).amax(-1),actual.float().softmax(-1).amax(-1)))
            self.assertEqual(mla.o_proj.rows[-2:],[m,c]);self.assertEqual(block.mlp.rows[-2:],[m,c])
            self.assertTrue(all(torch.equal(v,target.state_dict()[k]) for k,v in target_before.items()))
            # Later pass, without selection, still computes every row.
            block._glm_rowselect_indices=mla._glm_rowselect_indices=None
            again,_=full(torch.zeros(m,dtype=torch.int64),positions,hidden,embeds)
            self.assertTrue(torch.equal(logits,again))

    def test_native_prefill_dynamic_indices_sampling_and_cleanup(self):
        fn=native_method('vllm.v1.worker.gpu.spec_decode.autoregressive.speculator',
                         'AutoRegressiveSpeculator','_prefill',dict(torch=torch,CUDAGraphMode=NS(NONE=0)))
        for alias in (False,True):
            sp=NS(last_token_indices=torch.tensor([3,1]),idx_mapping=torch.arange(2),
                  input_buffers=NS(positions=torch.arange(4)),hidden_states=torch.zeros(4,3),
                  draft_tokens=torch.zeros(2,1,dtype=torch.int64),temperature=None,seeds=None,
                  current_draft_step=None,draft_logits=None)
            block,mla=NS(),NS()
            native=R.clone_method(fn,'prefill')
            def run(*a,**kw):
                x=torch.arange(12).reshape(4,3).float().index_select(0,block._glm_rowselect_indices)
                return x,x if alias else x+10
            sp._run_model=run;sp.sample_draft=lambda h,*a:h.argmax(-1)
            R.attach(sp,dict(block=block,mla=mla,block_fn=lambda:None,mla_fn=lambda:None,prefill_fn=native))
            for order in ([3,1],[0,2]):
                sp.last_token_indices.copy_(torch.tensor(order));sp.input_buffers.positions.copy_(torch.arange(4))
                sp._prefill(2,4,None,None,None)
                expected=torch.arange(12).reshape(4,3).float()[order]+(0 if alias else 10)
                self.assertTrue(torch.equal(expected,sp.hidden_states[:2]))
                self.assertIsNone(block._glm_rowselect_indices);self.assertIsNone(mla._glm_rowselect_indices)
            sp._run_model=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('failure'))
            with self.assertRaises(RuntimeError):sp._prefill(2,4,None,None,None)
            self.assertIsNone(block._glm_rowselect_indices);self.assertIsNone(mla._glm_rowselect_indices)

    def test_anchor_drift(self):
        for kind in ('mla','block','prefill'):
            with self.assertRaises(RuntimeError):R.rewrite(kind,'def broken(): pass')

    def test_capture_ready_requires_qualification_and_all_rank_vote(self):
        import glm_draft_head as dh
        class D:
            num_tokens=4;num_reqs=1;cg_mode='FULL'
        for qualified,peer_good in ((True,True),(False,True),(True,False)):
            cls=type('Runner',(),dict(load_model=lambda s:None,capture_model=lambda s:setattr(s,'_draft_head_ready',s.simulated_qualified)))
            R.install_runner(NS(GPUModelRunner=cls));r=cls();r.simulated_qualified=qualified
            manager=NS(graphs={D():object()})
            r.speculator=NS(prefill_cudagraph_manager=manager,decode_cudagraph_manager=manager,
                           max_num_reqs=4,_glm_rowselect=dict(ready=False))
            r._draft_head_ready=qualified;r._draft_head_qualification={'rowselect': dict(full_m_selected=True, changed_positions=True, bit_exact=True, det_align=True, logits_confidence_feedback_kv=True, cases=[{}])} if qualified else None
            def vote(payload,valid):
                if not valid or not peer_good:raise RuntimeError('collective refusal')
            with patch.object(dh,'agree',vote), patch.object(R,'capture_logits',lambda sp:None):
                if qualified and peer_good:cls.capture_model(r);self.assertTrue(r.speculator._glm_rowselect['ready'])
                else:
                    with self.assertRaises(RuntimeError):cls.capture_model(r)
                    self.assertFalse(r.speculator._glm_rowselect['ready'])
                    self.assertFalse(r._draft_head_ready)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
