# SPDX-License-Identifier: Apache-2.0
"""CPU algorithm proof + pinned layout oracle + real target/MTP hook tests.

This simulates warp peer and tile-prefix ownership; it does not execute CUDA.
The fleet microbench independently checks actual GPU outputs against the oracle.
"""
import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay/bringup'), str(ROOT/'overlay/overlay')]
import glm_moe_det as det
from det_align import runtime


from det_align.reference import stock_canonical


def simulate(flat, e, b, mapping=None, ignore=False, pad=False, seed=0):
    """Model thread/warp and CTA scheduling, rank storage, in-place tile scan.

    Poison outputs and require exactly one write to every slot; randomized CTA
    and token execution exposes overlap/races rather than relying on write order.
    """
    n = len(flat)
    converted = [((x + 2**31) % 2**32) - 2**31 for x in flat]
    routes = [mapping[x] if mapping is not None and ignore and 0 <= x < e
              else x if 0 <= x < e else -1 for x in converted]
    tile_size = 256 if n <= 128 else 1024
    tiles = max(1, (n + tile_size - 1) // tile_size)
    counts, ranks = [[0] * e for _ in range(tiles)], [None] * n
    rng = random.Random(seed)
    tile_order = list(range(tiles)); rng.shuffle(tile_order)
    for tile in tile_order:
        prior = [0] * e
        for part in range(0, tile_size, 256):
            hist = [[0] * e for _ in range(8)]
            for warp in range(8):
                begin = tile * tile_size + part + warp * 32
                for lane in range(32):
                    i = begin + lane
                    if i < n and routes[i] >= 0:
                        expert = routes[i]
                        ranks[i] = hist[warp][expert]
                        hist[warp][expert] += 1
            for i in range(tile * tile_size + part, min(n, tile * tile_size + part + 256)):
                if routes[i] >= 0:
                    ranks[i] += prior[routes[i]] + sum(hist[w][routes[i]] for w in range((i % 256) // 32))
            prior = [prior[x] + sum(hist[w][x] for w in range(8)) for x in range(e)]
        counts[tile] = prior
    totals = [0] * e
    for tile in range(tiles):
        for expert in range(e):
            value = counts[tile][expert]
            counts[tile][expert] = totals[expert]
            totals[expert] += value
    starts, ends, total = [], [], 0
    for count in totals:
        starts.append(total)
        total += (count + b - 1) // b * b
        ends.append(total)
    for tile in range(tiles):
        for expert in range(e): counts[tile][expert] += starts[expert]
    length = runtime.capacity(n, e, b, pad)
    ids, experts = [None] * length, [None] * ((length + b - 1) // b)
    def write(buf, i, value):
        if not 0 <= i < len(buf): raise AssertionError('OOB write')
        if buf[i] is not None: raise AssertionError('multiple writers')
        buf[i] = value
    for expert in range(e):
        label = mapping[expert] if mapping is not None and not ignore else expert
        for j in range(starts[expert] // b, ends[expert] // b): write(experts, j, label)
        for j in range(starts[expert] + totals[expert], ends[expert]): write(ids, j, n)
    for j in range(total, length): write(ids, j, n)
    tail = mapping[-1] if mapping is not None and not ignore else -1
    for j in range(total // b, len(experts)): write(experts, j, tail)
    indices = list(range(n)); rng.shuffle(indices)
    for i in indices:
        if routes[i] >= 0: write(ids, counts[i // tile_size][routes[i]] + ranks[i], i)
    if None in ids or None in experts: raise AssertionError('unwritten output')
    return ids, experts, [total]


class LayoutTests(unittest.TestCase):
    cases = 0
    def prove(self, flat, e, b, mapping=None, ignore=False, pad=False, seed=0):
        self.assertEqual(simulate(flat, e, b, mapping, ignore, pad, seed),
                         stock_canonical(flat, e, b, mapping, ignore, pad))
        LayoutTests.cases += 1

    def test_random_decode_and_prefill(self):
        for seed in range(12):
            rng = random.Random(seed)
            for n in (0, 8, 32, 64, 128, 129, 256, 257, 512, 4096, 32768, 65536):
                flat = [rng.randrange(-2, 259) for _ in range(n)]
                for b in runtime.BLOCKS:
                    self.prove(flat, 256, b, pad=bool(seed % 2), seed=seed)

    def test_empty_skew_invalid_and_padding_boundaries(self):
        for e in (1, 4, 32, 64, 256):
            for b in runtime.BLOCKS:
                for n in (0, 1, b-1, b, b+1, 127, 128, 129, 255, 256, 257, 1024):
                    for value in (-2, -1, 0, e-1, e, 2**40):
                        self.prove([value]*n, e, b, pad=bool(n % 2))

    def test_ep_both_modes_duplicate_local_maps_and_remote_tail(self):
        for seed in range(8):
            rng = random.Random(seed)
            e = 256
            mapping = [rng.choice([-1, -1, rng.randrange(e)]) for _ in range(e)]
            mapping[:4], mapping[-1] = [-1]*4, 7
            for n in (8, 128, 129, 257, 1023, 1024, 1025, 4096, 65536):
                flat = [rng.randrange(-1, e+1) for _ in range(n)]
                for b in runtime.BLOCKS:
                    for ignore in (False, True):
                        self.prove(flat, e, b, mapping, ignore, bool(seed % 2), seed)
        for ignore in (False, True):
            self.prove(list(range(256))*4, 256, 48, [-1]*256, ignore, True)

    def test_full_prefill_skew_and_cta_round_boundaries(self):
        for n in (511,512,513,1023,1024,1025,65536):
            for b in runtime.BLOCKS:
                for value in (-1,0,255):
                    self.prove([value]*n,256,b,pad=True)
                    self.prove([value]*n,256,b,[-1]*256,True,False)

    def test_int64_stock_narrowing_before_expert_bounds(self):
        flat = [-2**63,2**63-1,2**32,2**32+255,2**32-1,2**40,-2**32+1,-1,256]
        mapping = list(reversed(range(256))); mapping[0] = -1
        for b in runtime.BLOCKS:
            for ignore in (False,True):
                self.prove(flat,256,b,mapping,ignore,True)

    def test_actual_pinned_python_wrapper_layout(self):
        import torch
        source = Path(os.environ['GLM_IMAGE_SRC'])
        if (source/'vllm').is_dir(): source /= 'vllm'
        p = source/'model_executor/layers/fused_moe/moe_align_block_size.py'
        text = p.read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), det.PINS[det.ALIGN])
        tree = ast.parse(text)
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'moe_align_block_size')
        def op(topk, e, b, ids, experts, post, mapping):
            out = stock_canonical(topk.reshape(-1).tolist(), e, b,
                                  mapping.tolist() if mapping is not None else None, mapping is not None)
            ids.fill_(topk.numel()); experts.fill_(-1)
            ids[:len(out[0])].copy_(torch.tensor(out[0], dtype=torch.int32))
            experts[:len(out[1])].copy_(torch.tensor(out[1], dtype=torch.int32))
            post.copy_(torch.tensor(out[2], dtype=torch.int32))
        scope = dict(torch=torch, ops=NS(moe_align_block_size=op),
                     triton=NS(cdiv=lambda a,b:(a+b-1)//b), round_up=lambda a,b:(a+b-1)//b*b)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), 'exec'), scope)
        for n in (0, 8, 128, 129, 512, 65536):
            topk = (torch.arange(n, dtype=torch.int64) % 259 - 1).reshape(-1, 1)
            mapping = torch.arange(256, dtype=torch.int32).flip(0); mapping[:4] = -1
            for b in runtime.BLOCKS:
                for map_arg in (None, mapping):
                    for ignore in (False, True):
                        for pad in (False, True):
                            output = scope['moe_align_block_size'](topk, b, 256, map_arg, pad, ignore)
                            self.assertEqual(tuple(t.tolist() for t in output), stock_canonical(
                                topk.reshape(-1).tolist(), 256, b,
                                map_arg.tolist() if map_arg is not None else None, ignore, pad))

    def test_cuda_source_has_no_atomic_placement_or_synchronizing_host_api(self):
        source = (ROOT/'overlay/bringup/det_align/kernel.cu').read_text()
        self.assertNotIn('atomic', source)
        self.assertNotIn('cudaDeviceSynchronize', source)
        self.assertNotIn('cudaMalloc', source)
        self.assertIn('__match_any_sync', source)
        self.assertIn('ExclusiveSum', source)

    @classmethod
    def tearDownClass(cls):
        print(f'DET_LAYOUT_CASES={cls.cases}', flush=True)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.state = det.STATS.copy()
        det.STATS.update(installs=0, sites=[], mtp_checked=False, logged=False)
    def tearDown(self):
        det.STATS.clear(); det.STATS.update(self.state)

    def test_off_no_import_compile_or_hook_and_switch_conflict(self):
        before = list(sys.meta_path)
        with patch.object(runtime, 'prepare_library', side_effect=AssertionError('compile')):
            self.assertFalse(det.register({}))
            self.assertFalse(det.register({'GLM_MOE_DET_ALIGN':'0'}))
        self.assertEqual(before, sys.meta_path)
        with self.assertRaises(ValueError): det.register({'GLM_MOE_DET_ALIGN':'yes'})
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            det.register({'GLM_MOE_DET_ALIGN':'1','GLM_MOE_CANON_ALIGN':'1'})

    def modules(self):
        def align(topk_ids, block_size, num_experts, expert_map=None,
                  pad_sorted_ids=False, ignore_invalid_experts=False): pass
        a = NS(__name__=det.ALIGN, moe_align_block_size=align)
        m = NS(__name__=det.MARLIN, moe_align_block_size=align, fused_marlin_moe=lambda: None,
               BatchedMarlinExperts=type('B',(),{}), MarlinExperts=type('M',(),{}))
        import glm_nvfp4_mtp
        return a, m, glm_nvfp4_mtp

    def test_real_target_and_mtp_default_reach_kernel_wrapper(self):
        import torch
        a,m,mtp = self.modules()
        class Reached(Exception): pass
        def probe(*args):
            self.assertTrue(args[-1]); raise Reached
        with patch.object(runtime, 'deterministic_align', side_effect=probe):
            for module in (a,m,mtp): det.install(module, strict=False); det.install(module, strict=False)
            self.assertIs(a.moe_align_block_size._glm_moe_det_original, m.moe_align_block_size._glm_moe_det_original)
            with patch.dict(sys.modules, {det.ALIGN:a,det.MARLIN:m,det.MTP:mtp}), contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(det.verify_installs())
                with self.assertRaises(Reached): m.moe_align_block_size(None,8,256,None,False,True)
                layer = NS(num_experts=256,global_num_experts=256,intermediate_size_per_partition=4,expert_map=None)
                with self.assertRaises(Reached): mtp.apply_experts(layer, torch.zeros((1,4),dtype=torch.bfloat16),
                    torch.ones((1,8)),torch.zeros((1,8),dtype=torch.int64),ops=NS(),scalar=object())
            self.assertEqual(det.STATS['installs'],2)
            with self.assertRaises(ValueError): m.BatchedMarlinExperts()
            with self.assertRaises(ValueError): m.MarlinExperts().set_lora_context(None)

    def test_missing_sites_signatures_and_source_drift_fail(self):
        with self.assertRaises(RuntimeError): det.verify_installs()
        with self.assertRaises(RuntimeError): det.install(NS(__name__=det.ALIGN), strict=False)
        with self.assertRaises(RuntimeError): det.install(NS(__name__=det.ALIGN,__file__=__file__))
        with self.assertRaises(ValueError): det.deterministic_wrapper(lambda: None)
        with self.assertRaises(RuntimeError): det.install(NS(__name__=det.MTP,apply_experts=lambda:None),strict=False)

    def test_register_boot_compile_and_import_hooks_once(self):
        mods = self.modules(); before = list(sys.meta_path)
        try:
            with patch.object(det,'_REGISTERED',False), patch.object(runtime,'prepare_library') as compile_kernel, \
                 patch.dict(sys.modules, {m.__name__:m for m in mods}):
                self.assertTrue(det.register({'GLM_MOE_DET_ALIGN':'1','GLM_MOE_DET_STRICT':'0'}))
                count = len(sys.meta_path)
                self.assertTrue(det.register({'GLM_MOE_DET_ALIGN':'1'}))
                self.assertEqual(count,len(sys.meta_path)); compile_kernel.assert_called_once()
        finally: sys.meta_path[:] = before

    def test_lora_and_admission_missing_sites_fail_before_capture(self):
        cls = type('Runner',(),{'load_model':lambda self:'loaded'})
        with patch.object(det,'_check_source'): det.install(NS(__name__=det.RUNNER,GPUModelRunner=cls))
        runner = cls(); runner.vllm_config = NS(lora_config=object())
        with self.assertRaises(ValueError): runner.load_model()
        runner.vllm_config.lora_config = None
        with self.assertRaises(RuntimeError): runner.load_model()



class RuntimeTests(unittest.TestCase):
    def test_benchmark_error_poll_surfaces_runtime_status(self):
        for status in (0,1,9):
            with patch.object(runtime,'_LIB',NS(glm_det_last_error=lambda:status)):
                if status:
                    with self.assertRaisesRegex(RuntimeError,f'boundary error \\({status}\\)'):
                        runtime.check_cuda_error()
                else:
                    runtime.check_cuda_error()

    def fake_torch(self):
        # Metadata-only CUDA stand-in. Data reads are deliberately unavailable.
        class Tensor:
            pointer = 0
            def __init__(self,size,dtype='i32',device=None):
                Tensor.pointer += 1
                self.shape = (size,) if isinstance(size,int) else size
                self.ndim = len(self.shape); self.dtype = dtype
                self.device = device or NS(index=0); self.is_cuda = True
                self.ptr = Tensor.pointer
            def numel(self):
                result = 1
                for x in self.shape: result *= x
                return result
            def is_contiguous(self): return True
            def data_ptr(self): return self.ptr
            def narrow(self, dim, start, length):
                view = Tensor(length,self.dtype,self.device); view.ptr = self.ptr
                return view
            def element_size(self): return 8 if self.dtype == 'i64' else 4
        torch = NS(int32='i32',int64='i64',Tensor=Tensor)
        torch.empty = lambda size,**kw: Tensor(size,**kw)
        torch.cuda = NS(device=lambda d:contextlib.nullcontext(),
                        current_stream=lambda *args:NS(cuda_stream=11,device=NS(index=0)),
                        stream=lambda s:contextlib.nullcontext(),is_current_stream_capturing=lambda:False)
        return torch

    def test_model_load_prepares_pinned_full_manager_internal_capture_stream(self):
        # Execute the pinned manager's entire capture method, including its
        # warmup factory and torch.cuda.graph(graph, self.pool) with no stream=.
        source = Path(os.environ['GLM_IMAGE_SRC'])
        if (source/'vllm').is_dir(): source /= 'vllm'
        path = source/'v1/worker/gpu/cudagraph_utils.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f')
        tree = ast.parse(path.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CudaGraphManager')
        capture = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'capture')
        capture.decorator_list = []
        capture.returns = None
        for arg in capture.args.args: arg.annotation = None
        torch = self.fake_torch()
        load_stream = NS(cuda_stream=11,device=NS(index=0))
        warm_stream = NS(cuda_stream=22,device=NS(index=0))
        internal = NS(cuda_stream=33,device=NS(index=0))
        current = [load_stream]
        capturing = [False]
        events = []
        @contextlib.contextmanager
        def stream_ctx(stream):
            previous = current[0]; current[0] = stream
            try: yield
            finally: current[0] = previous
        class GraphContext:
            def __init__(self, graph, pool=None, stream=None):
                self.capture_stream = internal if stream is None else stream
                self.ctx = stream_ctx(self.capture_stream)
            def __enter__(self):
                self.ctx.__enter__(); capturing[0] = True
            def __exit__(self,*args):
                capturing[0] = False; self.ctx.__exit__(*args)
        torch.cuda.current_stream = lambda *a:current[0]
        torch.cuda.is_current_stream_capturing = lambda:capturing[0]
        torch.cuda.stream = stream_ctx
        torch.cuda.graph = GraphContext
        torch.cuda.CUDAGraph = object
        modes = NS(PIECEWISE=0,FULL=1,NONE=2)
        desc = NS(cg_mode=NS(name='FULL'))
        scope = dict(torch=torch,graph_capture=lambda **kw:stream_ctx(warm_stream),CUDAGraphMode=modes,
                     is_global_first_rank=lambda:False,logger=NS(debug=lambda *a:None),
                     get_offloader=lambda:NS(sync_prev_onload=lambda:None,join_after_forward=lambda:None),
                     set_graph_pool_id=lambda p:None,current_platform=NS(graph_pool_handle=lambda:None),
                     compilation_counter=NS(num_cudagraph_captured=0))
        exec(compile(ast.Module(body=[capture],type_ignores=[]),str(path),'exec'),scope)
        runner_cls = type('Runner',(),{'load_model':lambda self:'loaded'})
        det.install(NS(__name__=det.RUNNER,GPUModelRunner=runner_cls),strict=False)
        routes = torch.Tensor((1,8),'i64')
        def launch(*args):
            events.append((args[-1],capturing[0])); return 0
        def factory(desc,warmup):
            def forward(mode):
                if not warmup:
                    with patch.object(torch,'empty',side_effect=AssertionError('capture allocation')), \
                         patch.object(runtime,'prepare_library',side_effect=AssertionError('capture JIT')):
                        runtime.deterministic_align(routes,8,256)
                else: runtime.deterministic_align(routes,8,256)
            return forward
        with patch.dict(sys.modules,torch=torch),patch.object(runtime,'_LIB',NS(glm_det_align=launch)), \
             patch.object(runtime,'_ARENAS',{}),patch.object(runtime,'_BUFFERS',{}), \
             patch.object(det,'verify_installs',return_value=True):
            runner = runner_cls(); runner.vllm_config = NS(lora_config=None)
            self.assertEqual(runner.load_model(),'loaded')
            self.assertEqual(set(runtime._ARENAS),{(0,11),(0,33)})
            manager = NS(device=load_stream.device,_capture_descs={1:[object()]},
                         use_breakable_cg=False,pool=None,graphs={})
            # Descriptor must be hashable, with the exact cg_mode metadata.
            class Descriptor: cg_mode = desc.cg_mode
            manager._capture_descs = {1:[Descriptor()]}
            scope['capture'](manager,factory)
            self.assertEqual(events,[(22,False),(33,True)])
            self.assertTrue(manager._graphs_captured)
            self.assertEqual(len(manager.graphs),1)
            self.assertEqual(scope['compilation_counter'].num_cudagraph_captured,1)

    def test_capture_stream_preparation_fails_closed_on_drift_or_nested_capture(self):
        torch = self.fake_torch()
        torch.cuda.CUDAGraph = object
        torch.cuda.graph = lambda *a:NS()
        with patch.dict(sys.modules,torch=torch):
            with self.assertRaisesRegex(RuntimeError,'unsupported PyTorch'):
                runtime.prepare_capture_stream()
            with patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'outside capture'):
                    runtime.prepare_capture_stream()

    def test_real_pytorch_graph_constructor_selects_the_prepared_stream(self):
        import torch
        fake = self.fake_torch()
        side = NS(cuda_stream=44,device=NS(index=0))
        with patch.object(torch.cuda.graph,'default_capture_stream',None), \
             patch.object(torch.cuda,'CUDAGraph',object), \
             patch.object(torch.cuda,'Stream',return_value=side), \
             patch.object(torch.cuda,'stream',return_value=contextlib.nullcontext()), \
             patch.object(torch.cuda,'is_current_stream_capturing',return_value=False), \
             patch.object(torch,'empty',side_effect=fake.empty) as allocate, \
             patch.object(runtime,'_ARENAS',{}):
            arena = runtime.prepare_capture_stream()
            context = torch.cuda.graph(torch.cuda.CUDAGraph(),None)
            self.assertIs(context.capture_stream,side)
            self.assertIs(arena,runtime._ARENAS[(0,44)])
            self.assertIs(arena,runtime.prepare_capture_stream())
            self.assertEqual(allocate.call_count,5)

    def test_warmed_step_and_capture_allocate_nothing_and_use_stream(self):
        torch = self.fake_torch(); lib = NS(glm_det_align=lambda *args:0)
        with patch.dict(sys.modules,torch=torch), patch.object(runtime,'_LIB',lib),patch.object(runtime,'_BUFFERS',{}),patch.object(runtime,'_ARENAS',{}):
            routes = torch.Tensor((16,8),'i64')
            result = runtime.deterministic_align(routes,8,256)
            with patch.object(torch,'empty',side_effect=AssertionError('step allocation')), \
                 patch.object(torch.cuda,'is_current_stream_capturing',return_value=True), \
                 patch.object(runtime,'prepare_library',return_value=lib):
                self.assertIs(result,runtime.deterministic_align(routes,8,256))
                unseen = runtime.deterministic_align(torch.Tensor((17,8),'i64'),8,256)
                self.assertEqual(result[0].data_ptr(),unseen[0].data_ptr())
                self.assertEqual(unseen[0].shape,(136*8,))
                with patch.object(torch.cuda,'current_stream',return_value=NS(cuda_stream=99,device=NS(index=0))):
                    with self.assertRaisesRegex(RuntimeError,'prepare this stream'):
                        runtime.deterministic_align(routes,8,256)
            with patch.object(torch.cuda,'current_stream',return_value=NS(cuda_stream=12,device=NS(index=0))):
                other = runtime.deterministic_align(routes,8,256)
                self.assertNotEqual(result[0].data_ptr(),other[0].data_ptr())
            self.assertEqual(len(runtime._BUFFERS),3)
            self.assertEqual(len(runtime._ARENAS),2)

    def test_cold_capture_never_compiles_or_allocates(self):
        torch = self.fake_torch()
        with patch.dict(sys.modules,torch=torch),patch.object(runtime,'_LIB',None), \
             patch.object(torch.cuda,'is_current_stream_capturing',return_value=True), \
             patch.object(runtime,'prepare_library',side_effect=AssertionError('capture JIT')), \
             patch.object(torch,'empty',side_effect=AssertionError('capture allocation')):
            with self.assertRaisesRegex(RuntimeError,'prepared before capture'):
                runtime.deterministic_align(torch.Tensor((1,8),'i64'),8,256)

    def test_invalid_metadata_and_launch_errors(self):
        torch = self.fake_torch()
        with patch.dict(sys.modules,torch=torch),patch.object(runtime,'_LIB',NS(glm_det_align=lambda *a:9)), \
             patch.object(runtime,'_BUFFERS',{}),patch.object(runtime,'_ARENAS',{}):
            with self.assertRaisesRegex(RuntimeError,'launch failed'):
                runtime.deterministic_align(torch.Tensor((1,8),'i64'),8,256)
            for shape,dtype,block,e in (((1,8),'fp32',8,256),((1,8),'i64',7,256),
                                      ((1,8),'i64',8,257),((8193,8),'i64',8,256),((8,),'i32',8,256)):
                with self.assertRaises(ValueError): runtime.deterministic_align(torch.Tensor(shape,dtype),block,e)
            with self.assertRaises(ValueError):
                runtime.deterministic_align(torch.Tensor((1,8),'i64'),8,256,torch.Tensor(256,'i64'))


class LaunchVectorTests(unittest.TestCase):
    def test_default_off_rank_environment_and_vectors_identical(self):
        import test_launcher
        C = test_launcher.C
        profile = {key:'v-'+key for key in C.PROFILE_KEYS}
        # Keep launch-policy selectors valid; all unrelated optional switches cold.
        profile.update(GLM_MTP_KSTOP='0',GLM_MTP_KSTOP_UNIFORM_BATCH='0',GLM_PAD_HYGIENE='0',
                       GLM_DRAFT_HEAD='0',GLM_DECODE_FAIR='0',GLM_ATTN_WEIGHTS='int8',
                       GLM_NVFP4_GROUPS='attn',GLM_MOE_CANON_ALIGN='0')
        with patch.dict(C.ENV,profile), patch.object(C,'dispram',return_value=None):
            C.ENV.pop('GLM_MOE_DET_ALIGN',None)
            before = [C.rank_env(r) for r in range(4)]
            vectors = [C.rank_args(r) for r in range(4)]
            commands = [C.docker_command(r,'det-test') for r in range(4)]
            with patch.dict(C.ENV,GLM_MOE_DET_ALIGN='0'):
                self.assertEqual(before,[C.rank_env(r) for r in range(4)])
                self.assertEqual(vectors,[C.rank_args(r) for r in range(4)])
                self.assertEqual(commands,[C.docker_command(r,'det-test') for r in range(4)])
                self.assertTrue(all('GLM_MOE_DET_ALIGN' not in row for row in before))
            with patch.dict(C.ENV,GLM_MOE_DET_ALIGN='1'):
                for r in range(4):
                    self.assertEqual(C.rank_env(r),{**before[r],'GLM_MOE_DET_ALIGN':'1'})
                    self.assertEqual(vectors[r],C.rank_args(r))
            with patch.dict(C.ENV,GLM_MOE_DET_ALIGN='garbage'):
                with self.assertRaises(ValueError): C.rank_env(0)
            with patch.dict(C.ENV,GLM_MOE_DET_ALIGN='1',GLM_MOE_CANON_ALIGN='1'):
                with self.assertRaises(ValueError): C.rank_env(0)


if __name__ == "__main__": unittest.main()
