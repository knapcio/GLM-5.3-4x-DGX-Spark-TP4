"""MLA plan skip (GLM_SKIP_MLA_PLAN): wiring, fail-closed fallback and rank identity on CPU.

Without vLLM (macOS) only the wiring tests run. In the pinned serving image (CPU only, no
GPU) the rest run against the image's installed flashinfer_mla_sparse_sm90.py:
* booby trap: while armed, builds and the overlay forward never touch `_SM90_STATE` or the
  device positions (both replaced by objects that raise on any access); with the skip off
  the same trap fires, so it is sensitive;
* fallback: with the stock forward on the class the build plans, byte for byte the stock
  plan; a stock run on a skipped or missing plan raises; after a stock run the skip is
  latched off;
* rank identity: four worker processes with different rank-local data replay one message
  stream (builds and switch messages) and produce identical decision traces;
* startup: the real sitecustomize chain arms the MLA overlay and then this module.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
BRINGUP = ROOT / 'overlay/bringup'
sys.path.insert(0, str(BRINGUP))
import glm_skip_mla_plan as S

TOPK = 2048
try:
    import torch
    torch.cuda.is_current_stream_capturing = lambda: False     # CPU: no CUDA runtime
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as M
    HAVE_IMAGE = True
except Exception:                                              # macOS / no vLLM
    HAVE_IMAGE = False


def needs_image(fn):
    return unittest.skipUnless(HAVE_IMAGE, 'needs the pinned image (vLLM installed)')(fn)


# --------------------------------------------------------------------------- wiring (anywhere)
class Wiring(unittest.TestCase):
    def setUp(self):
        self.meta = list(sys.meta_path)
        self.mode, self.state = S.MODE, dict(S.STATE)

    def tearDown(self):
        sys.meta_path[:] = self.meta
        S.MODE = self.mode
        S.STATE.update(self.state)

    def test_off_by_default_installs_nothing(self):
        for env in ({}, {'GLM_SKIP_MLA_PLAN': '0'}, {'GLM_SKIP_MLA_PLAN': ''}):
            self.assertFalse(S.register(env))
        self.assertEqual(sys.meta_path, self.meta)

    def test_refusals(self):
        ok = {'GLM_SKIP_MLA_PLAN': 'ab', 'GLM_FULL_MLA': 'triton', 'VLLM_SERVER_DEV_MODE': '1'}
        bad = [dict(ok, GLM_SKIP_MLA_PLAN=v) for v in ('yes', 'on', '2', 'AB')]
        bad += [dict(ok, GLM_FULL_MLA=v) for v in ('0', 'flashinfer', '')]
        bad += [dict(ok, VLLM_SERVER_DEV_MODE='0'), {k: v for k, v in ok.items() if k != 'VLLM_SERVER_DEV_MODE'}]
        bad += [dict(ok, GLM_SKIP_MLA_PLAN_AB_INIT=v) for v in ('2', 'on', '')]
        bad += [{'GLM_SKIP_MLA_PLAN': '1', 'GLM_FULL_MLA': '0'}]
        for env in bad:
            with self.assertRaises(ValueError, msg=env):
                S.register(env)
        self.assertEqual(sys.meta_path, self.meta)

    def test_modes_register_hooks(self):
        seen = []
        original = S.Hooks.after_import
        S.Hooks.after_import = lambda hooks, name, fn: seen.append((name, fn.__name__))
        try:
            self.assertTrue(S.register({'GLM_SKIP_MLA_PLAN': '1', 'GLM_FULL_MLA': 'triton'}))
            self.assertEqual((S.MODE, S.STATE['on']), ('1', True))
            self.assertEqual(seen, [(S.TARGET, 'install')])
            self.assertIsInstance(sys.meta_path[0], S.Hooks)
            sys.meta_path[:] = self.meta
            env = {'GLM_SKIP_MLA_PLAN': 'ab', 'GLM_FULL_MLA': 'triton', 'VLLM_SERVER_DEV_MODE': '1'}
            for init, on in (('0', False), ('1', True)):
                seen.clear()
                self.assertTrue(S.register(dict(env, GLM_SKIP_MLA_PLAN_AB_INIT=init)))
                self.assertEqual((S.MODE, S.STATE['on']), ('ab', on))
                self.assertEqual(seen, [(S.TARGET, 'install'), (S.WORKER, 'install_worker')])
                sys.meta_path[:] = self.meta
        finally:
            S.Hooks.after_import = original

    def test_default_off_in_profiles_and_startup_order(self):
        for profile in ('current.env', 'dspark-k3.env'):
            self.assertNotIn('GLM_SKIP_MLA_PLAN', (ROOT / 'profiles' / profile).read_text())
        boot = (BRINGUP / 'sitecustomize.py').read_text()
        self.assertLess(boot.index('glm_full_mla.register()'), boot.index('glm_skip_mla_plan.register()'))
        self.assertLess(boot.index('glm_skip_mla_plan.register()'), boot.index('except BaseException'))

    def test_pins_match_the_recipe(self):
        pins = json.loads((ROOT / 'overlay/source_pins.json').read_text())
        self.assertEqual(pins[S.TARGET], S.PIN)
        self.assertEqual(pins[S.WORKER], S.WORKER_PIN)
        import glm_full_mla
        self.assertEqual(glm_full_mla.PIN, S.PIN)

    def test_decision_reads_only_rank_invariant_state(self):
        tree = ast.parse((BRINGUP / 'glm_skip_mla_plan.py').read_text())
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        def names(fn):
            return {n.id for n in ast.walk(fns[fn]) if isinstance(n, ast.Name)}
        self.assertEqual(names('skipping'), {'MODE', 'STATE', 'forward_armed', 'mod'})
        self.assertEqual(names('forward_armed'), {'getattr', 'mod'})
        subscripts = {n.slice.value for n in ast.walk(fns['skipping']) if isinstance(n, ast.Subscript)}
        self.assertEqual(subscripts, {'on', 'run_seen'})
        # The switch changes only in set_switch (the collective RPC) and register (env).
        writers = {fn for fn, node in fns.items() for n in ast.walk(node)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                                        and t.value.id == 'STATE' and t.slice.value == 'on' for t in n.targets)}
        self.assertEqual(writers, {'set_switch', 'register'})


# --------------------------------------------------------------------------- image helpers
class Recorder:
    """Stand-in for BatchMLAPagedAttentionWrapper: snapshots every plan argument."""

    def __init__(self):
        self.plans, self.runs = [], 0

    def plan(self, *a, **kw):
        snap = [('T', str(x.dtype), tuple(x.shape), x.numpy().tobytes().hex()) if isinstance(x, torch.Tensor)
                else ('S', repr(x)) for x in a]
        self.plans.append((snap, sorted((k, repr(v)) for k, v in kw.items())))

    def run(self, *a, **kw):
        self.runs += 1
        return 'stock-out'


class Poison:
    def __init__(self, log, tag):
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_tag', tag)

    def __getattribute__(self, k):
        object.__getattribute__(self, '_log').append((object.__getattribute__(self, '_tag'), k))
        raise AssertionError('booby trap read: ' + k)

    def __getitem__(self, k):
        object.__getattribute__(self, '_log').append((object.__getattribute__(self, '_tag'), '[]'))
        raise AssertionError('booby trap read: []')


def make_state():
    st = object.__new__(M._SM90State)
    st.device = torch.device('cpu'); st.num_heads = 16; st.kv_dtype = torch.float8_e4m3fn
    st.max_tokens = 4096; st.topk_width = TOPK; st.kv_lora_rank = 512
    st.qk_rope_head_dim = 64; st.sm_scale = 0.0625
    st.kv_indices = torch.zeros(1, dtype=torch.int32)
    st.wrapper = Recorder()
    S.guard_wrapper(st)
    return st


def cam_for(num_reqs, q, first_pos, offset=0):
    qsl = torch.arange(0, (num_reqs + 1) * q, q, dtype=torch.int32)
    pos = torch.cat([torch.arange(p + offset, p + offset + q) for p in first_pos]).to(torch.int64)
    seq = torch.tensor([p + offset + q for p in first_pos], dtype=torch.int32)
    return types.SimpleNamespace(num_reqs=num_reqs, query_start_loc_cpu=qsl, positions=pos,
                                 seq_lens_cpu_upper_bound=seq, seq_lens=seq)


WIDTHS = (('target_M3', 1, 3, [100]), ('target_M12', 4, 3, [700, 2045, 20900, 31000]),
          ('pass2_M1', 1, 1, [2047]), ('pass2_M4', 4, 1, [5, 2400, 9000, 4000]), ('prefill_2048', 1, 2048, [0]))
_ENV = {}


def setup_image():
    """Install the MLA overlay (recorder kernels) and this module on the real image module, once."""
    if _ENV:
        return _ENV
    rec = []
    for name in ('glm_full_mla_kernel', 'glm_full_mla_split_kernel'):
        m = types.ModuleType(name)
        m.sparse_mla = (lambda tag: (lambda qn, qr, cache, slots, scale, ks:
                                     rec.append(tag) or torch.zeros(qn.shape[0], 16, 512)))(name)
        sys.modules[name] = m
    import glm_full_mla
    stock_forward = M.FlashInferMLASparseSM90Impl.forward_mqa
    glm_full_mla.install(M)
    S.MODE = 'ab'
    S.install(M)
    parent_calls = []
    parent = M.FlashInferMLASparseMetadataBuilder

    def parent_build(self, cpl, cam, fast_build=False):
        parent_calls.append((cpl, id(cam), fast_build))
        return ('parent-metadata', id(cam))
    parent.build = parent_build
    builder = object.__new__(M.FlashInferMLASparseSM90Builder)
    builder._async_scheduling = True; builder._index_topk = TOPK; builder._index_kpool = 1
    _ENV.update(rec=rec, stock_forward=stock_forward, overlay_forward=M.FlashInferMLASparseSM90Impl.forward_mqa,
                parent_calls=parent_calls, builder=builder)
    return _ENV


def reset(on):
    S.STATE.update(on=on, plan='none', run_seen=False, switches=0)
    S.COUNTS.update(planned=0, skipped=0, fallback=0, kstop=0)


def overlay_forward(rows, split):
    fake = types.SimpleNamespace(num_heads=16, kv_lora_rank=512, qk_rope_head_dim=64, dcp_world_size=1,
                                 kv_cache_dtype='fp8_e4m3', topk_indices_buffer=torch.zeros(4096, TOPK, dtype=torch.int32),
                                 scale=0.0625)
    md = types.SimpleNamespace(req_id_per_token=torch.zeros(4096, dtype=torch.int32),
                               block_table=torch.zeros(4, 512, dtype=torch.int32), block_size=64)
    os.environ['GLM_MLA_SPLIT_K'] = split
    os.environ['GLM_MLA_SPLIT_MAX_ROWS'] = '36'
    return M.FlashInferMLASparseSM90Impl.forward_mqa(
        fake, (torch.zeros(rows, 16, 512), torch.zeros(rows, 16, 64)), torch.zeros(16, 64, 576, dtype=torch.uint8), md,
        types.SimpleNamespace(_k_scale_float=1.0))


# --------------------------------------------------------------------------- image tests
@unittest.skipUnless(HAVE_IMAGE, 'needs the pinned image (vLLM installed)')
class Image(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = setup_image()
        M.triton_convert_req_index_to_global_index = lambda rid, bt, topk, **kw: (
            torch.zeros(topk.shape, dtype=torch.int32), torch.zeros(topk.shape[0], dtype=torch.int32))

    def setUp(self):
        self.saved_state = M._SM90_STATE
        M.FlashInferMLASparseSM90Impl.forward_mqa = self.env['overlay_forward']

    def tearDown(self):
        M._SM90_STATE = self.saved_state
        M.FlashInferMLASparseSM90Impl.forward_mqa = self.env['overlay_forward']
        reset(False)

    def test_source_pin_and_idempotent_install(self):
        self.assertEqual(S.sha(M.__file__), S.PIN)
        build = M.FlashInferMLASparseSM90Builder.build
        S.install(M)
        self.assertIs(M.FlashInferMLASparseSM90Builder.build, build)
        self.assertTrue(build._glm_skip_mla_plan)

    def test_install_refuses_drift_and_unarmed_forward(self):
        def fake(path, forward):
            impl = type('Impl', (), {'forward_mqa': forward})
            return types.SimpleNamespace(__file__=str(path), FlashInferMLASparseSM90Impl=impl,
                                         FlashInferMLASparseSM90Builder=type('B', (), {'build': lambda *a: None}))
        with self.assertRaisesRegex(RuntimeError, 'source drift'):
            S.install(fake(__file__, self.env['overlay_forward']))
        with self.assertRaisesRegex(RuntimeError, 'not armed'):
            S.install(fake(M.__file__, self.env['stock_forward']))

    def test_booby_trap_never_read_while_armed(self):
        reset(True)
        touched = []
        M._SM90_STATE = Poison(touched, 'state')
        calls = self.env['parent_calls']
        calls.clear()
        for label, n, q, pos in WIDTHS:
            cam = cam_for(n, q, pos)
            cam.positions = Poison(touched, 'positions')
            cam.seq_lens = Poison(touched, 'seq_lens')
            out = self.env['builder'].build(0, cam, False)
            self.assertEqual(out, ('parent-metadata', id(cam)), label)
        self.assertEqual(len(calls), len(WIDTHS))
        self.assertEqual(S.COUNTS, dict(planned=0, skipped=len(WIDTHS), fallback=0, kstop=0))
        self.assertEqual(S.STATE['plan'], 'skipped')
        rec = self.env['rec']
        for split in ('32', '0'):
            for rows in (1, 3, 4, 12, 36, 37, 4096):
                rec.clear()
                overlay_forward(rows, split)
                self.assertEqual(len(rec), 1)
        self.assertEqual(touched, [])
        # Sensitivity: with the switch off the same trap fires on the stock plan path.
        S.STATE['on'] = False
        cam = cam_for(1, 3, [100])
        cam.positions = Poison(touched, 'positions')
        with self.assertRaises(AssertionError):
            self.env['builder'].build(0, cam, False)
        self.assertTrue(touched)

    def test_fallback_plans_exactly_as_stock(self):
        reset(True)
        M.FlashInferMLASparseSM90Impl.forward_mqa = self.env['stock_forward']    # overlay not armed
        self.assertFalse(S.skipping(M))
        for label, n, q, pos in WIDTHS:
            M._SM90_STATE = state = make_state()
            reference = make_state()
            cam = cam_for(n, q, pos, offset=-1)
            self.env['builder'].build(0, cam, False)
            rows, lens = M.FlashInferMLASparseSM90Builder._kv_lens_host(self.env['builder'], cam)
            reference.plan(rows, lens)
            self.assertEqual(state.wrapper.plans, reference.wrapper.plans, label)
            self.assertEqual(len(state.wrapper.plans), 1)
            self.assertEqual(S.STATE['plan'], 'fresh')
        self.assertEqual(S.COUNTS['fallback'], len(WIDTHS))
        self.assertEqual(S.COUNTS['skipped'], 0)
        # The guarded stock run passes on a fresh plan and latches the skip off.
        self.assertEqual(state.wrapper.run('q'), 'stock-out')
        self.assertTrue(S.STATE['run_seen'])
        M.FlashInferMLASparseSM90Impl.forward_mqa = self.env['overlay_forward']
        self.assertFalse(S.skipping(M))
        with self.assertRaisesRegex(RuntimeError, 'stays off'):
            S.set_switch(1)
        self.env['builder'].build(0, cam_for(1, 3, [100]), False)
        self.assertEqual(S.STATE['plan'], 'fresh')

    def test_kstop_payload_rides_a_stock_plan(self):
        # Native MTP K-stop reads its decision probabilities and guard flag through this planner's host copy:
        # while one is pending an armed build plans as stock; otherwise it skips. Not counted as planned/fallback.
        reset(True)
        M._SM90_STATE = state = make_state()
        runtime = types.SimpleNamespace(pending=None, guard_pending=None)
        fake = types.ModuleType('kstop_runtime')
        fake.STATE = dict(runtime=runtime)
        saved = sys.modules.get('kstop_runtime')
        sys.modules['kstop_runtime'] = fake
        try:
            builder = self.env['builder']
            for field, value in (('pending', (1, 1)), ('guard_pending', ('prepare', None))):
                setattr(runtime, field, value)
                self.assertTrue(S.kstop_payload())
                builder.build(0, cam_for(1, 3, [100]), False)
                self.assertEqual(S.STATE['plan'], 'fresh')
                setattr(runtime, field, None)
            self.assertEqual(len(state.wrapper.plans), 2)
            self.assertFalse(S.kstop_payload())
            builder.build(0, cam_for(1, 1, [101]), False)
            self.assertEqual(S.STATE['plan'], 'skipped')
            self.assertEqual(len(state.wrapper.plans), 2)
            self.assertEqual(S.COUNTS, dict(planned=0, skipped=1, fallback=0, kstop=2))
            # Switch off: stock planning, no K-stop count.
            S.STATE['on'] = False
            runtime.pending = (1, 1)
            builder.build(0, cam_for(1, 3, [102]), False)
            self.assertEqual(S.COUNTS, dict(planned=1, skipped=1, fallback=0, kstop=2))
            # No K-stop runtime initialized: no payload.
            fake.STATE = None
            self.assertFalse(S.kstop_payload())
        finally:
            if saved is None:
                sys.modules.pop('kstop_runtime', None)
            else:
                sys.modules['kstop_runtime'] = saved

    def test_stock_run_on_stale_or_missing_plan_raises(self):
        reset(True)
        M._SM90_STATE = state = make_state()
        with self.assertRaisesRegex(RuntimeError, 'none plan'):
            state.wrapper.run('q')
        self.env['builder'].build(0, cam_for(1, 3, [100]), False)     # armed: skipped
        self.assertEqual(state.wrapper.plans, [])
        with self.assertRaisesRegex(RuntimeError, 'skipped plan'):
            state.wrapper.run('q')
        self.assertEqual(state.wrapper.runs, 0)
        self.assertFalse(S.STATE['run_seen'])
        # Switch off -> the next build plans again and the stock run is legal.
        S.set_switch(0)
        self.env['builder'].build(0, cam_for(1, 3, [100]), False)
        self.assertEqual(len(state.wrapper.plans), 1)
        self.assertEqual(state.wrapper.run('q'), 'stock-out')

    def test_new_state_is_guarded(self):
        import flashinfer.mla as FM
        saved = (FM.BatchMLAPagedAttentionWrapper, M._get_sm90_workspace, torch.cuda.get_device_capability)
        FM.BatchMLAPagedAttentionWrapper = lambda *a, **k: Recorder()
        M._get_sm90_workspace = lambda device: torch.empty(0)
        torch.cuda.get_device_capability = lambda *a: (12, 1)
        try:
            st = M._SM90State(torch.device('cpu'), 16, torch.float8_e4m3fn, 8, TOPK, 512, 64, 0.0625)
        finally:
            FM.BatchMLAPagedAttentionWrapper, M._get_sm90_workspace, torch.cuda.get_device_capability = saved
        self.assertTrue(st.wrapper.run._glm_skip_mla_plan)
        S.STATE['plan'] = 'skipped'
        with self.assertRaises(RuntimeError):
            st.wrapper.run('q')

    def test_switch_rpc_contract(self):
        reset(False)
        with self.assertRaises(ValueError):
            S.set_switch(2)
        with self.assertRaises(ValueError):
            S.set_switch(1.0)
        self.env['builder'].build(0, cam_for(1, 3, [100]), False)
        out = S.set_switch(1)
        self.assertEqual(out['at_counts'], dict(planned=1, skipped=0, fallback=0, kstop=0))
        self.assertEqual((out['on'], out['skipping'], out['switches']), (True, True, 1))
        S.MODE = '1'
        try:
            with self.assertRaisesRegex(RuntimeError, 'only in ab mode'):
                S.set_switch(0)
        finally:
            S.MODE = 'ab'


@unittest.skipUnless(HAVE_IMAGE, 'needs the pinned image (vLLM installed)')
class RankIdentity(unittest.TestCase):
    STREAM = ([('build', 'target_M3'), ('build', 'pass2_M1')] * 3 + [('set', 1)]
              + [('build', 'target_M3'), ('build', 'pass2_M1'), ('build', 'prefill_2048')] * 3 + [('set', 0)]
              + [('build', 'target_M12'), ('build', 'pass2_M4')] * 2 + [('set', 1), ('status',)]
              + [('build', 'target_M12'), ('build', 'pass2_M4')] * 4 + [('set', 1), ('set', 0), ('build', 'target_M3')])

    def test_four_ranks_identical_decisions(self):
        procs = []
        for rank in range(4):
            env = dict(os.environ, GLM_TEST_RANK=str(rank), GLM_TEST_FAKE_MEM_KIB=str(9_000_000 + 377_000 * rank))
            procs.append(subprocess.Popen([sys.executable, __file__, '--rank-worker', json.dumps(self.STREAM)],
                                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        outs = []
        for p in procs:
            out, err = p.communicate(timeout=600)
            self.assertEqual(p.returncode, 0, err[-3000:])
            outs.append(json.loads(out.strip().splitlines()[-1]))
        self.assertEqual({o['rank'] for o in outs}, {0, 1, 2, 3})
        self.assertEqual(len({o['local_data_sha'] for o in outs}), 4)       # rank-local inputs really differ
        traces = [json.dumps(o['trace'], sort_keys=True) for o in outs]
        self.assertEqual(len(set(traces)), 1, traces)
        sets = [r for r in outs[0]['trace'] if r[0] == 'set']
        self.assertEqual([r[1]['on'] for r in sets], [True, False, True, True, False])
        decisions = [r[1] for r in outs[0]['trace'] if r[0] == 'build']
        self.assertEqual(decisions[:6], ['planned'] * 6)
        self.assertEqual(decisions[6:15], ['skipped'] * 9)
        self.assertEqual(decisions[15:19], ['planned'] * 4)
        self.assertEqual(decisions[19:27], ['skipped'] * 8)
        self.assertEqual(decisions[27:], ['planned'])


def rank_worker(stream):
    """One fake TP rank: real module, rank-local positions and memory, the shared message stream."""
    import random
    rank = int(os.environ['GLM_TEST_RANK'])
    random.seed(rank)
    env = setup_image()
    S.MODE = 'ab'
    reset(False)
    S._rank = lambda: rank
    M._SM90_STATE = make_state()
    fake_worker = types.SimpleNamespace(__file__=None, Worker=type('Worker', (), {}))
    import importlib.util
    fake_worker.__file__ = importlib.util.find_spec(S.WORKER).origin
    S.install_worker(fake_worker)
    worker = fake_worker.Worker()
    widths = {w[0]: w for w in WIDTHS}
    trace, local = [], []
    for msg in stream:
        if msg[0] == 'build':
            _, n, q, pos = widths[msg[1]]
            offset = random.randrange(0, 3) + 5 * rank                    # rank-local positions
            local.append((offset, os.environ['GLM_TEST_FAKE_MEM_KIB'], os.getpid()))
            before = dict(S.COUNTS)
            env['builder'].build(0, cam_for(n, q, pos, offset=offset), False)
            trace.append(['build', 'skipped' if S.COUNTS['skipped'] > before['skipped'] else 'planned'])
        elif msg[0] == 'set':
            r = worker.skip_mla_plan_set(str(msg[1]) if len(trace) % 2 else msg[1])   # JSON int or string
            trace.append(['set', dict(on=r['on'], at_counts=r['at_counts'], switches=r['switches'], skipping=r['skipping'])])
        else:
            r = worker.skip_mla_plan_status()
            trace.append(['status', {k: r[k] for k in ('mode', 'on', 'skipping', 'plan', 'run_seen', 'counts', 'forward_armed')}])
    print(json.dumps(dict(rank=rank, trace=trace,
                          local_data_sha=hashlib.sha256(json.dumps(local).encode()).hexdigest())))


@unittest.skipUnless(HAVE_IMAGE, 'needs the pinned image (vLLM installed)')
class Startup(unittest.TestCase):
    # PYTHONPATH=overlay/bringup: the interpreter imports overlay/bringup/sitecustomize.py at
    # startup, exactly as in the serving containers.
    def run_python(self, code, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GLM_', 'VLLM_SERVER'))}
        env.update(PYTHONPATH=str(BRINGUP), **extra)
        return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=600)

    def test_sitecustomize_chain_arms_mla_then_skip(self):
        code = '\n'.join([
            'import sys',
            "assert 'glm_skip_mla_plan' in sys.modules and 'glm_full_mla' in sys.modules",
            'import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as M',
            'import vllm.v1.worker.gpu_worker as W',
            'import glm_skip_mla_plan as S',
            "assert M.FlashInferMLASparseSM90Impl.forward_mqa._glm_full_mla is True",
            "assert M.FlashInferMLASparseSM90Builder.build._glm_skip_mla_plan is True",
            "assert callable(W.Worker.skip_mla_plan_set) and callable(W.Worker.skip_mla_plan_status)",
            "assert S.MODE == 'ab' and S.STATE['on'] is False and S.skipping(M) is False",
            "print('STARTUP OK')"])
        r = self.run_python(code, GLM_FULL_MLA='triton', GLM_SKIP_MLA_PLAN='ab', VLLM_SERVER_DEV_MODE='1',
                            GLM_MLA_SPLIT_K='32')
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn('STARTUP OK', r.stdout)
        armed = [line for line in r.stderr.splitlines() if 'ARMED' in line]
        self.assertEqual(len(armed), 2, r.stderr[-4000:])
        self.assertIn('glm-full-mla: ARMED', armed[0])
        self.assertIn('glm-skip-mla-plan: ARMED mode=ab on=0', armed[1])

    def test_static_mode_needs_no_dev_api(self):
        code = ('import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as M, glm_skip_mla_plan as S\n'
                "assert S.MODE == '1' and S.skipping(M) is True\nprint('STATIC OK')")
        r = self.run_python(code, GLM_FULL_MLA='triton', GLM_SKIP_MLA_PLAN='1')
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn('glm-skip-mla-plan: ARMED mode=1 on=1', r.stderr)

    def test_sitecustomize_refuses_bad_env(self):
        for extra, message in ((dict(GLM_FULL_MLA='0', GLM_SKIP_MLA_PLAN='1'), 'needs GLM_FULL_MLA=triton'),
                               (dict(GLM_FULL_MLA='triton', GLM_SKIP_MLA_PLAN='ab'), 'needs VLLM_SERVER_DEV_MODE=1'),
                               (dict(GLM_FULL_MLA='triton', GLM_SKIP_MLA_PLAN='yes'), 'must be 0, 1 or ab')):
            r = self.run_python('pass', **extra)
            self.assertEqual(r.returncode, 78, extra)
            self.assertIn(message, r.stderr)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--rank-worker':
        rank_worker(json.loads(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
