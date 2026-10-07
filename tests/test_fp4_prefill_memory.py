# SPDX-License-Identifier: Apache-2.0
"""Execute tiled prefill orchestration and real reader ASTs on the Mac."""
import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay/bringup'), str(ROOT/'tests'), str(ROOT/'scripts')]
import torch
import glm_fp4_prefill as P
from fp4_kv_cpu_model import kernels, Ptr


class PrefillMemory(unittest.TestCase):
    def setUp(self):
        self.tl, self.k = kernels(ROOT/'overlay/bringup')

    def test_actual_paged_tile_gather_seam_multiple_requests_tail(self):
        bs = 64
        cache = torch.zeros(5, bs, 132, dtype=torch.uint8)
        codes = torch.arange(5*bs*128).remainder(256).byte().reshape(5, bs, 128)
        scales = torch.arange(5*bs).float().reshape(5, bs) + 1
        cache.view(5,-1)[:,:bs*128].copy_(codes.reshape(5,-1))
        cache.view(5,-1)[:,bs*128:].copy_(scales.view(torch.uint8).reshape(5,-1))
        bt = torch.tensor([[2,0,4],[1,3,0]], dtype=torch.int32)
        cu = torch.tensor([0,129,219], dtype=torch.int32)
        keys = torch.empty(4096,128,dtype=torch.uint8); sf = torch.empty(4096)
        for offset in (0,64,128,210,4096):
            for pid in range(256):
                self.tl.pid=(pid,0)
                self.k['_gather_index_tile'](*map(Ptr.of,(cache,bt,cu,keys,sf)),offset,bs,bs*132,3,2)
            for row in range(4096):
                pos=offset+row
                if pos>=219:
                    self.assertEqual(float(sf[row]),0.)
                    self.assertTrue(torch.all(keys[row]==0))
                else:
                    req=0 if pos<129 else 1; local=pos-int(cu[req]); block=int(bt[req,local//bs])
                    self.assertTrue(torch.equal(keys[row],codes[block,local%bs]))
                    self.assertEqual(float(sf[row]),float(scales[block,local%bs]))

    def test_bounds_and_finish_request_relative_ids_and_sentinels(self):
        ks=torch.tensor([100,50,200],dtype=torch.int32);ke=torch.tensor([160,60,205],dtype=torch.int32)
        starts=torch.empty(64,dtype=torch.int32);ends=torch.empty_like(starts)
        self.k['_tile_bounds'](*map(Ptr.of,(ks,ke,starts,ends)),0,3,120)
        self.assertEqual(starts[:4].tolist(),[0,0,80,0]);self.assertEqual(ends[:4].tolist(),[40,0,85,0])
        values=torch.full((64,2048),float('-inf'));ids=torch.full((64,2048),-1,dtype=torch.int64)
        values[0,:2]=torch.tensor([3.,2.]);ids[0,:2]=torch.tensor([110,130])
        out=torch.full((3,2056),77,dtype=torch.int32)
        for row in range(3):
            for col in range(8):
                self.tl.pid=(row,col)
                self.k['_finish_indices'](*map(Ptr.of,(values,ids,ks,out)),0,3,2056)
        self.assertEqual(out[0,:3].tolist(),[10,30,-1]);self.assertTrue(torch.all(out[:,2048:]==77))

    def test_budget_geometry_and_fixed_bank_addresses(self):
        bank=P.reserve(torch.device('cpu'),1)
        self.assertEqual(P.bank_bytes(bank) + P.LOGITS_BYTES, P.scratch_bytes(1))
        self.assertLessEqual(P.scratch_bytes(64),400<<20)
        self.assertEqual(P.geometry(16336),(2048,32768))
        self.assertEqual(P.geometry(65508),(1024,65536))
        self.assertEqual(P.geometry(98226),(512,131072))
        for prefix in (1,32768,32769,65536,65537,131072,131073,262144,262145,524288):
            qt,kt=P.geometry(prefix)
            self.assertEqual(qt*kt*4,P.LOGITS_BYTES)
            self.assertGreaterEqual(kt,prefix)
        for prefix in (0,524289):
            with self.assertRaises(RuntimeError):P.geometry(prefix)
        self.assertEqual({k:v.data_ptr() for k,v in bank.items()},
                         {k:v.data_ptr() for k,v in P.reserve(torch.device('cpu'),1).items()})

    def test_actual_query_prepare_subchunk_and_padding(self):
        q=torch.arange(7*128).remainder(256).byte().reshape(7,1,128)
        w=torch.arange(7).float().reshape(7,1)
        ks=torch.tensor([123,124,125,126,127],dtype=torch.int32)
        ke=ks+4096
        bq=torch.full((4,1,128),99,dtype=torch.uint8);bw=torch.empty(4,1)
        starts=torch.empty(4,dtype=torch.int32);ends=torch.empty_like(starts)
        for row in range(4):
            self.tl.pid=(row,0)
            self.k['_prepare_index_queries'](*map(Ptr.of,(q,w,ks,ke,bq,bw,starts,ends)),4,2,3,1,128)
        self.assertTrue(torch.equal(bq[:3],q[4:7]));self.assertTrue(torch.all(bq[3]==0))
        self.assertEqual(bw[:,0].tolist(),[4.,5.,6.,0.])
        self.assertEqual(starts.tolist(),[125,126,127,0]);self.assertEqual(ends.tolist(),[4221,4222,4223,0])

    def test_stock_selection_unique_ties_and_multi_request_subchunk(self):
        # Exercise the real orchestration with an independent deterministic
        # selector stand-in; the GPU fixture calls the actual stock selector.
        bank=P.reserve(torch.device('cpu'),1)
        ptrs={k:v.data_ptr() for k,v in bank.items()}
        starts=torch.tensor([0,0,25000,25000,49000],dtype=torch.int32)
        ends=torch.tensor([9,21000,25005,49000,49100],dtype=torch.int32)
        chunk=types.SimpleNamespace(token_start=2,token_end=7,local_total_seq_lens=49100,
            block_table=None,local_cu_seq_lens=None,cu_seqlen_ks=starts,cu_seqlen_ke=ends)
        calls=[]
        def gather(cache,bt,cu,keys,scales,offset):
            calls.append('gather');self.assertEqual(keys.shape[0],65536);self.assertEqual(offset,0)
        def prepare(q,w,ks,ke,bq,bw,s,e,begin,qoff,rows):
            calls.append('prepare')
            self.assertEqual((begin,qoff,rows),(2,0,5))
            s.zero_();e.zero_();s[:rows].copy_(ks);e[:rows].copy_(ke)
        def select(logits,ks,ke,out,rows,s0,s1,topk):
            calls.append('select');self.assertEqual(rows,5)
            out.fill_(-1)
            for row in range(rows):
                start,end=int(ks[row]),int(ke[row])
                order=torch.argsort(logits[row,start:end],descending=True,stable=True)[:topk]
                out[row,:order.numel()].copy_(order.int())
        mode=['unique']
        def score(pos):
            return ((pos*7919%49117).float() if mode[0]=='unique' else (pos%7).float())
        def logits(q,kv,w,s,e,clean_logits):
            calls.append('score');self.assertFalse(clean_logits)
            self.assertEqual(q[0].shape,(1024,1,128))
            val=torch.empty(1024,65536)
            # Poison outside valid intervals: stock selection must ignore it.
            val[:5].fill_(1e20)
            for row in range(5):
                start,end=int(s[row]),int(e[row])
                val[row,start:end]=score(torch.arange(start,end))
            return val
        fake=types.SimpleNamespace(gather_index_tile=gather,prepare_index_queries=prepare)
        out=torch.full((7,2048),77,dtype=torch.int32)
        q=torch.zeros(7,1,128).to(torch.float8_e4m3fn)
        for case in ('unique','ties'):
            mode[0]=case;calls.clear()
            with patch.dict(sys.modules,glm_fp4_prefill_kernel=fake):
                P.tiled_prefill(torch.empty(1,64,132,dtype=torch.uint8),q,None,torch.ones(7,1),[chunk],out,logits,select)
            self.assertEqual(calls,['gather','prepare','score','select'])
            for row in range(5):
                pos=torch.arange(int(starts[row]),int(ends[row]))
                want=torch.argsort(score(pos),descending=True,stable=True)[:2048].int()
                self.assertTrue(torch.equal(out[row+2,:want.numel()],want))
                self.assertTrue(torch.all(out[row+2,want.numel():]==-1))
            self.assertTrue(torch.all(out[:2]==77))
        self.assertEqual(ptrs,{k:v.data_ptr() for k,v in P.reserve(torch.device('cpu'),1).items()})

    def test_q512_crosses_stock_metadata_subchunks_without_host_reads(self):
        P.reserve(torch.device('cpu'),1)
        q=torch.zeros(2050,1,128).to(torch.float8_e4m3fn)
        weights=torch.ones(2050,1);out=torch.full((2050,2048),77,dtype=torch.int32)
        chunks=[]
        for begin,end in ((2,1368),(1368,2050)):
            chunks.append(types.SimpleNamespace(token_start=begin,token_end=end,local_total_seq_lens=98226,
                block_table=None,local_cu_seq_lens=None,
                cu_seqlen_ks=torch.zeros(end-begin,dtype=torch.int32),
                cu_seqlen_ke=torch.arange(begin,end,dtype=torch.int32)+90000))
        trace=[]
        def gather(*args):trace.append('gather')
        def prepare(q,w,ks,ke,bq,bw,s,e,begin,qoff,rows):
            trace.append((begin,qoff,rows));s.zero_();e.zero_();e[:rows].copy_(ke[qoff:qoff+rows])
        def logits(*args,**kwargs):
            return types.SimpleNamespace(numel=lambda:P.LOGITS_BYTES//4,element_size=lambda:4,
                                         stride=lambda d:131072 if d==0 else 1)
        def select(logits,ks,ke,out,rows,*args):out.copy_(ke[:rows,None].expand(-1,2048))
        fake=types.SimpleNamespace(gather_index_tile=gather,prepare_index_queries=prepare)
        with patch.dict(sys.modules,glm_fp4_prefill_kernel=fake):
            P.tiled_prefill(torch.empty(1,64,132,dtype=torch.uint8),q,None,weights,chunks,out,logits,select)
        self.assertEqual(trace,['gather',(2,0,512),(514,512,512),(1026,1024,342),
                                'gather',(1368,0,512),(1880,512,170)])
        self.assertTrue(torch.all(out[:2]==77))
        self.assertTrue(torch.equal(out[2:,0],torch.arange(2,2050,dtype=torch.int32)+90000))

    def test_actual_metadata_clone_allocates_only_query_fields(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/v1/attention/backends/mla/indexer.py'
        class Metadata:
            def __init__(self, **kwargs):self.__dict__.update(kwargs)
        def build(qsl,seq,cu,local_cu,token_map,ks,ke,qstart,qstop,rank,world,interleave,**kwargs):
            self.assertEqual(token_map.numel(),0)
            for req in range(kwargs['num_reqs']):
                for col in range((qstop-qstart+127)//128):
                    self.tl.pid=(req,col)
                    self.k['_index_prefill_bounds'](*map(Ptr.of,(qsl,seq,cu,ks,ke)),qstart,qstop)
        ns=dict(torch=torch,DeepseekV32IndexerPrefillChunkMetadata=Metadata,_glm_fp4x_build_bounds=build)
        exec(compile(P.transform_metadata(src.read_text()),'<pinned-metadata>','exec'),ns)
        qsl=torch.tensor([0,3,8],dtype=torch.int32);seq=torch.tensor([50000,70001],dtype=torch.int32)
        bt=torch.zeros(2,1100,dtype=torch.int32)
        md=ns['build_prefill_chunk_metadata'](0,2,qsl,qsl,seq,seq,seq,bt,1,query_slice=slice(2,7))
        self.assertEqual(md.cu_seqlen_ks.tolist(),[0,50000,50000,50000,50000])
        self.assertEqual(md.cu_seqlen_ke.tolist(),[50000,119997,119998,119999,120000])
        self.assertEqual(md.token_to_seq.numel(),0)
        self.assertEqual((md.token_start,md.token_end),(2,7))

    def test_sparse_metadata_skips_dead_long_prefix_map_keeps_dense_prefix(self):
        calls=[]
        class Builder:
            def _build_chunked_context_fields(self,*args):calls.append(args);return 'dense'
        P.install_sparse_metadata(types.SimpleNamespace(SparseMLACommonMetadataBuilder=Builder))
        b=Builder();b.topk_mask_workspace=None;b.model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(index_topk=2048))
        common=types.SimpleNamespace(seq_lens_cpu_upper_bound=torch.tensor([17,50000]))
        self.assertIsNone(b._build_chunked_context_fields(common,0,2,torch.tensor([1,2048])))
        common.seq_lens_cpu_upper_bound=torch.tensor([17,2048])
        self.assertEqual(b._build_chunked_context_fields(common,0,2,torch.tensor([1,100])),'dense')
        self.assertEqual(len(calls),1)

    def test_installed_hook_routes_eager_prefill_preserves_insert_and_decode(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/layers/sparse_attn_indexer.py'
        calls=[]
        class Indexer:
            def __init__(self):
                self.k_cache=types.SimpleNamespace(prefix='index',kv_cache=torch.empty(1,64,132,dtype=torch.uint8))
                self.topk_indices_buffer=torch.empty(4,2048,dtype=torch.int32)
                self.quant_block_size=128;self.scale_fmt='ue8m0';self.topk_tokens=2048;self.head_dim=128
                self.max_model_len=100288;self.max_total_seq_len=40*100288;self.skip_k_cache_insert=False
                self.use_pcp=False;self.dense_mha_metadata_layer_name='';self.use_fp4_cache=False
                self.dcp_rank=0;self.dcp_world_size=1;self.cp_kv_cache_interleave_size=1
            def forward_cuda(self,*args):calls.append('stock');return 'decode'
        md=types.SimpleNamespace(slot_mapping=torch.arange(4),num_prefills=1,num_decodes=0,num_decode_tokens=0,
                                 prefill=types.SimpleNamespace(chunks=[]))
        context=types.SimpleNamespace(attn_metadata={'index':md},cudagraph_runtime_mode=None)
        def gather(k,slots,*args):return k,slots
        def insert(*args):calls.append('insert')
        def tiled(*args):calls.append('tiled');args[-3].fill_(7)
        mod=types.SimpleNamespace(__file__=str(src),torch=torch,LayerNameType=str,SparseAttnIndexer=Indexer,
            get_current_vllm_config=lambda:types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(index_n_heads=1))),
            get_forward_context=lambda:context,current_platform=types.SimpleNamespace(fp8_dtype=lambda:torch.float8_e4m3fn),
            _resolve_layer_name=lambda x:x,DeepseekV32IndexerMetadata=types.SimpleNamespace,
            CUDAGraphMode=types.SimpleNamespace(FULL='full'),maybe_gather_indexer_k=gather,
            ops=types.SimpleNamespace(indexer_k_quant_and_cache=insert,top_k_per_row_prefill=lambda *a:None),fp8_fp4_mqa_logits=lambda *a:None)
        P.install_indexer(mod)
        index=Indexer();self.assertEqual(index.max_total_seq_len,P.K_CAPACITY)
        q=torch.zeros(4,1,128).to(torch.float8_e4m3fn)
        with patch.object(torch.cuda,'is_current_stream_capturing',return_value=False),patch.object(P,'tiled_prefill',tiled):
            result=index.forward_cuda(torch.zeros(4,2),q,torch.zeros(4,128),torch.ones(4,1))
        self.assertEqual(calls,['insert','tiled']);self.assertTrue(torch.all(result==7))
        md.num_prefills=0;md.num_decodes=1
        self.assertEqual(index.forward_cuda(None,q,None,None),'decode')
        self.assertEqual(calls[-1],'stock')

    def test_source_transform_preserves_writer_decode_and_has_no_prefix_workspace(self):
        src=Path(os.environ['GLM_IMAGE_SRC'])/'vllm/model_executor/layers/sparse_attn_indexer.py'
        raw=src.read_text(); new=P.transform_indexer(raw);ast.parse(new)
        self.assertIn('ops.indexer_k_quant_and_cache(',new)
        prefill=new[new.index('    if has_prefill:'):new.index('    if has_decode:')]
        self.assertNotIn('get_simultaneous',prefill);self.assertNotIn('cp_gather',prefill)
        self.assertIn('tiled_prefill(',prefill)
        self.assertIn('ops.top_k_per_row_prefill)',prefill)
        profile=new[new.index('    if not isinstance(attn_metadata, dict):'):new.index('    attn_metadata_narrowed =')]
        self.assertNotIn('get_simultaneous(',profile)
        self.assertIn('256 * 1024 * 1024',profile)
        self.assertEqual(new[new.index('    if has_decode:'):],raw[raw.index('    if has_decode:'):raw.index('\ndef sparse_attn_indexer_fake')].rstrip()+'\n')


if __name__=='__main__':unittest.main()
