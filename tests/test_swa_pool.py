"""CPU integration with the pinned image's real cache managers and BlockPools."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get('GLM_IMAGE_SRC', '/image-source'))
SHIM = ROOT / 'tests/.cpu-shim/vllm-0.1.dev20051+cpu.dist-info'
SHIM.mkdir(parents=True, exist_ok=True)
(SHIM / 'METADATA').write_text('Metadata-Version: 2.1\nName: vllm\nVersion: 0.1.dev20051+cpu\n')
sys.path[:0] = [str(ROOT / 'overlay/swa-pool'), str(SHIM.parent), str(SRC)]
os.environ['GLM_DSA_SWA_POOL'] = '1'
os.environ['GLM_DSA_DRAFT_FOLD'] = '0'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
import glm_dsa_swa_pool as G
G.register()
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec, FullAttentionSpec, get_kv_quant_mode
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams
from vllm.config import VllmConfig, set_current_vllm_config

FIX = ROOT / 'tests/fixtures'
GiB = 1 << 30


def config(length=32768, seqs=4, flight=4096):
    return NS(model_config=NS(max_model_len=length, original_max_model_len=length),
              scheduler_config=NS(max_num_seqs=seqs, max_num_batched_tokens=4096,
                                  disable_hybrid_kv_cache_manager=False),
              cache_config=NS(num_gpu_blocks_override=None, block_size=64,
                              prefix_match_unit=None, enable_prefix_caching=True),
              parallel_config=NS(pipeline_parallel_size=1, decode_context_parallel_size=1,
                                 prefill_context_parallel_size=1),
              speculative_config=NS(num_speculative_tokens=8), max_in_flight_tokens=flight,
              use_v2_model_runner=True, kv_transfer_config=None)


def specs(arm='rh'):
    t = {}
    for i in range(78):
        t[f'model.layers.{i}.self_attn.attn'] = MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=torch.uint8,
            cache_dtype_str='fp8_e4m3', kv_quant_mode=get_kv_quant_mode('fp8_e4m3'))
    for i in range(21):
        t[f'model.layers.{i}.self_attn.indexer.k_cache'] = MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=132, dtype=torch.uint8)
    name = {'rh': 'redhat_dspark_config.json', 'alaya': 'alaya_dspark_config.json',
            'dflash2': 'incoai_dflash2_config.json'}[arm]
    raw = json.loads((FIX / name).read_text())
    raw = raw.get('transformer_layer_config', raw)
    for i, lt in enumerate(raw['layer_types']):
        kw = dict(block_size=16 if lt == 'sliding_attention' else 64,
                  num_kv_heads=raw['num_key_value_heads']//4, head_size=raw['head_dim'],
                  dtype=torch.bfloat16, indexes_kv_by_block_stride=True)
        if lt == 'sliding_attention':
            s = SlidingWindowSpec(**kw, sliding_window=raw['sliding_window'])
        else:
            s = FullAttentionSpec(**kw)
        t[f'model.layers.{78+i}.self_attn.attn'] = s
    return t


def setup(arm='rh', length=32768, seqs=4, flight=4096, caching=True):
    vc = config(length, seqs, flight)
    source = specs(arm)
    cfg = U.get_kv_cache_configs(vc, [source]*4, [4*GiB]*4)[0]
    cfg = U.generate_scheduler_kv_cache_config([cfg])
    bs, hs = U.resolve_kv_cache_block_sizes(cfg, vc)
    manager = KVCacheManager(cfg, length, bs, hs, max_in_flight_tokens=flight,
                             enable_caching=caching, use_eagle=True)
    return vc, cfg, manager


def request(rid='r', length=6000):
    return Request(rid, list(range(length)), SamplingParams(max_tokens=30000), None,
                   block_hasher=U.get_request_block_hasher(16, U.sha256_cbor))


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = set_current_vllm_config(VllmConfig()); cls.ctx.__enter__()
        KVCacheSpecRegistry._ensure_registered(VllmConfig())
        U.init_none_hash(U.sha256_cbor)

    def test_geometry_and_capacity(self):
        rows = []
        for arm in ('rh', 'alaya', 'dflash2'):
            for flight in (4096, 8192):
                vc = config(flight=flight); s = specs(arm)
                groups = U.get_kv_cache_groups(vc, s)
                original = {n: z for n, z in s.items() if type(z) is MLAAttentionSpec}
                for n, z in G.inner(groups[0]).items():
                    self.assertIs(z, original[n])
                cfgs = U.get_kv_cache_configs(vc, [s]*4, [4*GiB, 4*GiB-1, 4*GiB, 4*GiB])
                self.assertTrue(all(c.glm_pool_blocks == cfgs[0].glm_pool_blocks for c in cfgs))
                cfg = cfgs[0]
                self.assertFalse(cfg.needs_kv_cache_zeroing)
                self.assertLessEqual(sum(t.size for t in cfg.kv_cache_tensors), 4*GiB-1)
                self.assertEqual([g.kv_cache_spec.block_size for g in groups], [64,16] + ([64] if arm=='alaya' else []))
                self.assertEqual([g.is_eagle_group for g in groups], [False, True]+([True] if arm=='alaya' else []))
                for group, count in zip(groups, cfg.glm_pool_blocks):
                    for n, z in G.inner(group).items():
                        tensor = next(t for t in cfg.kv_cache_tensors if t.shared_by == [n])
                        self.assertEqual(tensor.size, count*z.page_size_bytes)
                rows.append(dict(arm=arm, flight=flight, blocks=cfg.glm_pool_blocks,
                                 physical_target_tokens=cfg.num_blocks*64, usable_target_tokens=(cfg.num_blocks-1)*64,
                                 reserved_bytes=sum(t.size for t in cfg.kv_cache_tensors if t.shared_by[0] not in original)))
        (ROOT / 'tests/memory-results.json').write_text(json.dumps(rows,indent=2)+'\n')

    def test_source_pins_and_refusal(self):
        for name in G.PINS:
            path = SRC / (name.replace('.','/')+'.py')
            source = path.read_text()
            transformed = G.transform_source(name, source)
            ast.parse(transformed)
            with self.assertRaises(RuntimeError):
                G.transform_source(name, source+'\n')
        with self.assertRaises(ValueError):
            G.make_groups(U, config(), {**specs(), 'odd': object()})
        for attr, value in [('use_v2_model_runner', False), ('kv_transfer_config', object())]:
            vc = config(); setattr(vc, attr, value)
            with self.assertRaises(ValueError): U.get_kv_cache_groups(vc,specs())
        vc=config(); vc.model_config.original_max_model_len=-1
        with self.assertRaises(ValueError): U.get_kv_cache_groups(vc,specs())

    def test_minimal_profile_config(self):
        vc=config(); gs=U.get_kv_cache_groups(vc,specs('alaya'))
        vc.cache_config.num_gpu_blocks_override=36
        c=U.get_kv_cache_config_from_groups(vc,gs,0)
        self.assertEqual(c.glm_pool_blocks,(36,36,36))
        with self.assertRaises(ValueError): U.get_kv_cache_config_from_groups(vc,gs,4*GiB)

    def test_alloc_cancel_deferred_preemption(self):
        for arm in ('rh', 'alaya'):
            _, cfg, km = setup(arm)
            req = request(); initial = [p.get_num_free_blocks() for p in km.block_pool.pools]
            self.assertIsNotNone(km.allocate_slots(req, 4096, num_lookahead_tokens=9))
            req.num_computed_tokens=4096
            self.assertIsNotNone(km.allocate_slots(req, 1904, num_lookahead_tokens=9))
            req.num_computed_tokens=6000
            km.remove_skipped_blocks(req.request_id,6000)
            for gid,m in enumerate(km.coordinator.single_type_managers):
                real=[b for b in m.req_to_blocks['r'] if not b.is_null]
                self.assertLess(len(real), cfg.glm_pool_blocks[gid])
                if gid==1: self.assertLessEqual(len(real),130)
            pending=km.pop_blocks_for_free(req)
            self.assertTrue(all(not group for group in km.get_blocks('r').blocks))
            self.assertTrue(any(p.get_num_free_blocks()!=n for p,n in zip(km.block_pool.pools,initial)))
            self.assertFalse(km.reset_prefix_cache())
            km.block_pool.free_blocks(reversed(pending))
            self.assertEqual([p.get_num_free_blocks() for p in km.block_pool.pools],initial)
            req.num_computed_tokens=0; req.status=RequestStatus.PREEMPTED
            self.assertIsNotNone(km.allocate_slots(req, 4096, num_lookahead_tokens=9))
            km.free(req)
            self.assertEqual([p.get_num_free_blocks() for p in km.block_pool.pools],initial)
            self.assertTrue(km.reset_prefix_cache())

    def test_pool_specific_exhaustion_atomic(self):
        _, _, km=setup()
        p=km.block_pool.pools[1]
        held=p.get_new_blocks(p.get_num_free_blocks())
        before=km.block_pool.pools[0].get_num_free_blocks()
        self.assertIsNone(km.allocate_slots(request(),32,num_lookahead_tokens=9))
        self.assertEqual(km.block_pool.pools[0].get_num_free_blocks(),before)
        p.free_blocks(held)

    def test_prefix_hit_reconciles_and_missing_tail_replays(self):
        for arm in ('rh','alaya'):
            _,cfg,km=setup(arm)
            req=request(length=8192)
            for end in (4096,8192):
                self.assertIsNotNone(km.allocate_slots(req,end-req.num_computed_tokens,num_lookahead_tokens=9))
                req.num_computed_tokens=end
            km.free(req)
            other=request('other',8192)
            blocks,n,_=km.get_computed_blocks(other)
            self.assertGreater(n,2048)
            self.assertEqual(n%64,0)
            self.assertEqual(len(blocks.blocks[0])*64,n)
            if arm=='alaya': self.assertEqual(len(blocks.blocks[2])*64,n)
            self.assertIsNotNone(km.allocate_slots(other,8192-n,num_new_computed_tokens=n,
                                                 new_computed_blocks=blocks,num_lookahead_tokens=9))
            km.free(other)
            # Keep the target cache, invalidate the entire drafter pool.
            km.block_pool.pools[1].evict_blocks(set(range(1,cfg.glm_pool_blocks[1])))
            missing=request('missing',8192)
            _,n,_=km.get_computed_blocks(missing)
            self.assertEqual(n,0)  # no uninitialized drafter tail behind a target-only hit

    def test_rejection_page_boundary_and_committed_eviction(self):
        for arm in ('rh','alaya'):
            _,_,km=setup(arm)
            req=request(length=7000)
            for end in (2047,2063,4095,4111):
                self.assertIsNotNone(km.allocate_slots(req,end-req.num_computed_tokens,num_lookahead_tokens=9))
                req.num_computed_tokens=end
            before=copy.copy(km.coordinator.single_type_managers[1].req_to_blocks['r'])
            # Optimistic batch spans 16/64 page boundaries; seven rows reject.
            req.num_in_flight_tokens=16
            self.assertIsNotNone(km.allocate_slots(req,16,num_lookahead_tokens=9))
            req.num_computed_tokens+=16
            req.num_computed_tokens-=7
            req.num_in_flight_tokens=0
            self.assertIsNotNone(km.allocate_slots(req,8,num_lookahead_tokens=9))
            m=km.coordinator.single_type_managers[1]
            first=max(0,req.num_computed_tokens-2047)//16
            self.assertTrue(all(not b.is_null for b in m.req_to_blocks['r'][first:]))
            for i in range(max(0,(4111-16-2047)//16),first):
                if i<len(before) and not before[i].is_null:
                    # Old physical pages cannot disappear before the committed window advances.
                    self.assertTrue(before[i].block_id>=1)
            km.free(req)

    def test_long_decode_plateau_and_multirequest_refs(self):
        _,cfg,km=setup('rh',length=65536)
        reqs=[request(str(i),60000) for i in range(4)]
        for end in range(4096,24001,4096):
            for req in reqs:
                self.assertIsNotNone(km.allocate_slots(req,end-req.num_computed_tokens,num_lookahead_tokens=9))
                req.num_computed_tokens=end
        m=km.coordinator.single_type_managers[1]
        live=sum(not b.is_null for r in reqs for b in m.req_to_blocks[r.request_id])
        self.assertLessEqual(live,4*386)
        for req in reqs: km.free(req)
        self.assertEqual(km.block_pool.pools[1].get_num_free_blocks(),cfg.glm_pool_blocks[1]-1)

    def test_no_cache_and_switch(self):
        _,_,km=setup(caching=False)
        req=request(); self.assertIsNotNone(km.allocate_slots(req,64,num_lookahead_tokens=9))
        km.free(req)
        os.environ['GLM_DSA_DRAFT_FOLD']='1'
        try:
            with self.assertRaises(ValueError): G.enabled()
        finally: os.environ['GLM_DSA_DRAFT_FOLD']='0'

    def test_prefix_shared_refs_and_payload(self):
        # Model physical writes with token/position markers. Real allocator,
        # hashes and refcounts; no simulated replacement allocator.
        for arm in ('rh', 'alaya'):
            _, cfg, km = setup(arm)
            payload = [dict() for _ in cfg.kv_cache_groups]
            def write(req, start, end):
                for gid, manager in enumerate(km.coordinator.single_type_managers):
                    table=manager.req_to_blocks[req.request_id]
                    for pos in range(start,end):
                        block=table[pos//manager.block_size]
                        if not block.is_null:
                            payload[gid][(block.block_id,pos%manager.block_size)]=(pos,req.all_token_ids[pos])
            def check(req, boundary):
                for gid, manager in enumerate(km.coordinator.single_type_managers):
                    table=manager.req_to_blocks[req.request_id]
                    start=max(0,boundary-2047) if gid==1 else 0
                    for pos in range(start,boundary):
                        block=table[pos//manager.block_size]
                        self.assertFalse(block.is_null)
                        self.assertEqual(payload[gid][(block.block_id,pos%manager.block_size)],
                                         (pos,req.all_token_ids[pos]))
            base=request('base',8192)
            for end in (4096,8192):
                start=base.num_computed_tokens
                self.assertIsNotNone(km.allocate_slots(base,end-start,num_lookahead_tokens=9))
                write(base,start,end); base.num_computed_tokens=end
            km.free(base)
            peers=[]
            for rid in ('a','b'):
                req=request(rid,8192)
                blocks,boundary,_=km.get_computed_blocks(req)
                self.assertGreater(boundary,2048)
                self.assertIsNotNone(km.allocate_slots(req,8192-boundary,
                    num_new_computed_tokens=boundary,new_computed_blocks=blocks,num_lookahead_tokens=9))
                check(req,boundary)
                write(req,boundary,8192); req.num_computed_tokens=8192
                check(req,8192); peers.append(req)
            shared=km.coordinator.single_type_managers[0].req_to_blocks['b'][0]
            self.assertEqual(shared.ref_cnt,2)
            km.free(peers[0]); self.assertEqual(shared.ref_cnt,1)
            check(peers[1],8192)
            pending=km.pop_blocks_for_free(peers[1])
            self.assertEqual(shared.ref_cnt,1)  # cancelled in-flight request still pins pages
            km.block_pool.free_blocks(reversed(pending)); self.assertEqual(shared.ref_cnt,0)

    def test_full_attention_miss_reconciles_all_tables(self):
        _,cfg,km=setup('alaya')
        req=request(length=8192)
        for end in (4096,8192):
            self.assertIsNotNone(km.allocate_slots(req,end-req.num_computed_tokens,num_lookahead_tokens=9))
            req.num_computed_tokens=end
        full=km.coordinator.single_type_managers[2]
        victim=full.req_to_blocks['r'][32].block_id
        km.free(req)
        full.block_pool.evict_blocks({victim})
        blocks,n,_=km.get_computed_blocks(request('hit',8192))
        self.assertLessEqual(n,2048)
        for gid in (0,2): self.assertEqual(len(blocks.blocks[gid])*64,n)

    def test_async_peak_and_admission_gate(self):
        _,cfg,km=setup('rh',flight=8192)
        req=request(length=20000)
        self.assertIsNotNone(km.allocate_slots(req,4096,num_lookahead_tokens=9,full_sequence_must_fit=True))
        req.num_computed_tokens=4096; req.num_in_flight_tokens=4096
        self.assertIsNotNone(km.allocate_slots(req,4096,num_lookahead_tokens=9))
        req.num_computed_tokens=8192; req.num_in_flight_tokens=8192
        self.assertIsNotNone(km.allocate_slots(req,8,num_lookahead_tokens=9))
        m=km.coordinator.single_type_managers[1]
        self.assertTrue(all(not b.is_null for b in m.req_to_blocks['r'][:512]))
        req.num_computed_tokens-=7; req.num_in_flight_tokens=0
        self.assertIsNotNone(km.allocate_slots(req,8,num_lookahead_tokens=9))
        first=max(0,req.num_computed_tokens-2047)//16
        self.assertTrue(all(b.is_null for b in m.req_to_blocks['r'][:first]))
        self.assertTrue(all(not b.is_null for b in m.req_to_blocks['r'][first:]))
        km.free(req)

    def test_worker_rejects_group_blind_zero_and_copy(self):
        source=(SRC/(G.R_NAME.replace('.','/')+'.py')).read_text()
        tree=ast.parse(G.transform_source(G.R_NAME,source))
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.If)
                and '_glm_is_pool' in ast.unparse(n.test)]
        self.assertEqual(len(guards),2)
        fn=ast.FunctionDef(name='guard', args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg='self'),ast.arg(arg='scheduler_output')],kwonlyargs=[],
            kw_defaults=[],defaults=[]), body=guards, decorator_list=[])
        module=ast.Module(body=[fn],type_ignores=[]); ast.fix_missing_locations(module)
        namespace={'_glm_is_pool':G.is_pool}
        exec(compile(module,'<worker-zero-copy-guards>','exec'),namespace)
        pool_cfg=NS(glm_pool_blocks=(2,2)); ordinary=NS()
        for zero,copies in (([1],[]),([],[object()])):
            out=NS(new_block_ids_to_zero=zero,kv_cache_block_copies=copies)
            with self.assertRaises(RuntimeError): namespace['guard'](NS(kv_cache_config=pool_cfg),out)
            namespace['guard'](NS(kv_cache_config=ordinary),out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
