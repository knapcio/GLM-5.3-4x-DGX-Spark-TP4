# SPDX-License-Identifier: Apache-2.0
"""CPU tests for overlay/bringup/glm_glue_lite.py. No GPU; vllm imported only by the import-pin test.

The pin test hashes GLM_IMAGE_SRC (or, inside the pinned image, its own vllm sources); the import-pin
test needs the pinned image and is skipped elsewhere.
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay" / "bringup"))
import glm_glue_lite as G  # noqa: E402

LEVERS = G.LEVERS
os.environ.update({k: "1" for k in LEVERS})  # call-time switches on for the functional tests
os.environ.pop("GLM_GLUE_LITE_BANKABLE", None)

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SRC = Path(os.environ["GLM_IMAGE_SRC"]) / "vllm" if os.environ.get("GLM_IMAGE_SRC") else VLLM


def reset_stats():
    G.STATS.update(router_armed={}, router_refused=[], idx_armed={}, ws_alloc=0, pins_checked=[],
                   counts={}, captures={}, ws_streams={}, ws_capture_streams={}, runner_hooked=False,
                   load_model_armed=0)
    G._WS.clear()
    G._HOME.clear()
    G._IDX.cache, G._IDX.reuse = None, False
    G._CHECK[0] = False


class Pins(unittest.TestCase):
    @unittest.skipUnless(SRC.exists(), "needs GLM_IMAGE_SRC or the pinned image")
    def test_image_sources_match_pins(self):
        for mod, pin in G.PINS.items():
            rel = mod.split(".", 1)[1].replace(".", "/") + ".py"
            G.check_pin(mod, SRC / rel)
        self.assertEqual(set(G.STATS["pins_checked"]), set(G.PINS))

    def test_every_pin_only_module_is_pinned(self):
        for names in G.PIN_ONLY.values():
            for n in names:
                self.assertIn(n, G.PINS)

    @unittest.skipUnless(VLLM.exists(), "image-only")
    def test_sparse_utils_pin_enforced_at_import(self):
        # Registration must hash sparse_utils (and the other pin-only modules) on import.
        reset_stats()
        env = {"GLM_GLUE_DSA_IDX_CACHE": "1", "GLM_GLUE_STRICT": "1"}
        saved = list(sys.meta_path)
        try:
            G._FINDER = None
            for n in G.PIN_ONLY["idx"]:
                sys.modules.pop(n, None)
            G.register(env=env)
            importlib.import_module("vllm.v1.attention.backends.mla.sparse_utils")
            self.assertIn("vllm.v1.attention.backends.mla.sparse_utils", G.STATS["pins_checked"])
        finally:
            sys.meta_path[:] = saved
            G._FINDER = None


# ---------------------------------------------------------------- F1 fakes
class _RepLinear(torch.nn.Module):
    def __init__(self, k, n, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=dtype) * 0.02, requires_grad=False)
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _Gate(_RepLinear):
    """Mirror of GateLinear Tier 6 on a platform without specialised tiers (SM12x)."""
    def __init__(self, k=6144, n=256):
        super().__init__(k, n)
        self.out_dtype = torch.float32
        for t in G.GATE_TIERS:
            setattr(self, t, False)

    def forward(self, x):
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias


class _Router:
    def __init__(self, scoring="sigmoid", bias=True, groups=1, topk_group=1, topk=8, ne=256, rs=2.5,
                 renorm=True, bias_dtype=torch.float32):
        self.scoring_func, self.num_expert_group, self.topk_group = scoring, groups, topk_group
        self.top_k, self.global_num_experts, self.routed_scaling_factor = topk, ne, rs
        self.renormalize, self.num_fused_shared_experts = renorm, 0
        self.e_score_correction_bias = torch.zeros(ne, dtype=bias_dtype) if bias else None


class _QM:
    is_monolithic = False


class _Runner(torch.nn.Module):
    """Stand-in for MoERunner: holds the gate and the router, applies the gate itself."""
    def __init__(self, gate, router, monolithic=False, fse=False):
        super().__init__()
        self.gate = gate
        self.router = router
        self.shared_expert_gate = None
        self._fse_fuse_gate = fse
        self.routed_experts = types.SimpleNamespace(quant_method=types.SimpleNamespace(is_monolithic=monolithic))


class _SubGate(_Gate):
    pass


def _gate_mod():
    return types.SimpleNamespace(GateLinear=_Gate, ReplicatedLinear=_RepLinear)


def _router_mod():
    return types.SimpleNamespace(GroupedTopKRouter=_Router)


def _cuda_platform():
    return types.SimpleNamespace(is_cuda=lambda: True)


ENVS = types.SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=True)


def _preprocess(logits):
    """What single_group_topk does first for any input dtype: widen to float."""
    return logits.float() if logits.dtype != torch.float32 else logits


def _route(logits, bias, k=8):
    s = torch.sigmoid(_preprocess(logits))
    ids = torch.topk(s + bias, k, dim=-1).indices
    w = s.gather(1, ids)
    return ids, w / w.sum(-1, keepdim=True)


def _fresh_gate_forward():
    if getattr(_Gate.forward, "_glm_glue", False):
        _Gate.forward = _Gate.forward.__wrapped__
    G.install_router_forward(_gate_mod())


class RouterF1(unittest.TestCase):
    def setUp(self):
        reset_stats()
        _fresh_gate_forward()

    def test_armed_logits_are_the_pre_cast_bf16(self):
        torch.manual_seed(0)
        gate = _Gate()
        x = torch.randn(3, 6144, dtype=torch.bfloat16)
        ref, _ = gate(x)  # unarmed: Tier 6 with the cast
        self.assertEqual(ref.dtype, torch.float32)
        gate._glm_bf16_logits = True
        got, b = gate(x)
        self.assertIsNone(b)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got.float(), ref))           # widening is exact
        self.assertTrue(torch.equal(_preprocess(got), ref))      # kernel sees identical floats
        bias = torch.randn(256) * 0.01
        for a, r in zip(_route(got, bias), _route(ref, bias)):
            self.assertTrue(torch.equal(a, r))

    def test_armed_gate_refuses_non_bf16_activations(self):
        gate = _Gate()
        gate._glm_bf16_logits = True
        with self.assertRaises(RuntimeError):
            gate(torch.randn(2, 6144, dtype=torch.float32))

    def _arm(self, runners, env=None, tag="target"):
        model = torch.nn.ModuleList(runners)
        return G.arm_router({tag: model}, _gate_mod(), _router_mod(), ENVS, _cuda_platform(),
                            env=env if env is not None else {"GLM_GLUE_STRICT": "0"})

    def test_contract_accepts_exactly_e256_g1_tg1_top8(self):
        ok = _Runner(_Gate(), _Router())
        armed, refused = self._arm([ok])
        self.assertEqual((armed, refused), ({"target": 1}, []))
        self.assertTrue(ok.gate._glm_bf16_logits)

    def test_contract_refusals(self):
        cases = {
            "softmax": _Runner(_Gate(), _Router(scoring="softmax")),
            "nobias": _Runner(_Gate(), _Router(bias=False)),
            "bf16bias": _Runner(_Gate(), _Router(bias_dtype=torch.bfloat16)),
            "e256_g2": _Runner(_Gate(), _Router(groups=2, topk_group=1)),     # generic fallback, review #1
            "e256_g8_tg4": _Runner(_Gate(), _Router(groups=8, topk_group=4)),
            "top6": _Runner(_Gate(), _Router(topk=6)),
            "e128": _Runner(_Gate(6144, 128), _Router(ne=128)),
            "rs3": _Runner(_Gate(), _Router(rs=3.0)),
            "norenorm": _Runner(_Gate(), _Router(renorm=False)),
            "monolithic": _Runner(_Gate(), _Router(), monolithic=True),
            "fse": _Runner(_Gate(), _Router(), fse=True),
            "subclass_gate": _Runner(_SubGate(), _Router()),
            "hidden4096": _Runner(_Gate(4096, 256), _Router()),
        }
        cublas = _Runner(_Gate(), _Router())
        cublas.gate.allow_cublas_router_gemm = True
        cases["cublas"] = cublas
        spec = _Runner(_Gate(), _Router())
        spec.gate.allow_specialized_router_gemm = True
        cases["specialized"] = spec
        fp16out = _Runner(_Gate(), _Router())
        fp16out.gate.out_dtype = torch.bfloat16
        cases["out_bf16"] = fp16out
        for name, r in cases.items():
            with self.subTest(name):
                reset_stats()
                armed, refused = self._arm([r])
                self.assertEqual(armed, {"target": 0}, name)
                self.assertEqual(len(refused), 1, name)
                self.assertFalse(getattr(r.gate, "_glm_bf16_logits", False), name)

    def test_strict_refuses_any_out_of_contract_router(self):
        with self.assertRaises(RuntimeError):
            self._arm([_Runner(_Gate(), _Router()), _Runner(_Gate(), _Router(groups=2))],
                      env={"GLM_GLUE_STRICT": "1"})

    def test_strict_expected_counts(self):
        target = torch.nn.ModuleList([_Runner(_Gate(), _Router()) for _ in range(3)])
        mtp = torch.nn.ModuleList([_Runner(_Gate(), _Router())])
        env = {"GLM_GLUE_STRICT": "1", "GLM_GLUE_ROUTER_EXPECT": "target:3,mtp:1"}
        armed, _ = G.arm_router({"target": target, "mtp": mtp}, _gate_mod(), _router_mod(), ENVS,
                                _cuda_platform(), env=env)
        self.assertEqual(armed, {"target": 3, "mtp": 1})
        reset_stats()
        with self.assertRaises(RuntimeError):
            G.arm_router({"target": target, "mtp": None}, _gate_mod(), _router_mod(), ENVS, _cuda_platform(),
                         env=env)

    def test_fused_topk_disabled_refuses_and_strict_raises(self):
        envs = types.SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=False)
        model = torch.nn.ModuleList([_Runner(_Gate(), _Router())])
        with self.assertRaises(RuntimeError):
            G.arm_router({"target": model}, _gate_mod(), _router_mod(), envs, _cuda_platform(),
                         env={"GLM_GLUE_STRICT": "1"})


# ---------------------------------------------------------------- F2 fakes
def _marlin_mod():
    calls = []
    mod = types.SimpleNamespace(
        marlin_make_workspace_new=lambda dev, n: torch.zeros(48 * n, dtype=torch.int32, device=dev))

    # Same leading parameters as the pinned fused_marlin_moe; workspace is the 27th.
    def fused_marlin_moe(hidden_states, w1, w2, bias1, bias2, w1_scale, w2_scale, topk_weights, topk_ids,
                         quant_type_id, apply_router_weight_on_input=False, global_num_experts=-1,
                         activation=None, activation_func=None, moe_sum=None, expert_map=None,
                         input_global_scale1=None, input_global_scale2=None, global_scale1=None,
                         global_scale2=None, g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
                         w1_zeros=None, w2_zeros=None, workspace=None, intermediate_cache13=None,
                         intermediate_cache2=None, is_k_full=True, output=None, input_dtype=None,
                         activation_config=None):
        calls.append(workspace)
        return hidden_states
    mod.fused_marlin_moe = fused_marlin_moe
    return mod, calls


def _kw(x):
    return dict(hidden_states=x, w1=None, w2=None, bias1=None, bias2=None, w1_scale=None, w2_scale=None,
                topk_weights=None, topk_ids=None, quant_type_id=0)


class MoeWorkspaceF2(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def test_one_persistent_workspace_preallocated(self):
        mod, calls = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: False, stream_id=lambda d: 7)
        ws = G.ws_prepare(mod, "cpu")
        x = torch.zeros(3, 8)
        for _ in range(3):
            mod.fused_marlin_moe(**_kw(x))
        self.assertTrue(all(c is ws for c in calls))
        self.assertEqual(int(ws.abs().sum()), 0)
        self.assertEqual(G.STATS["ws_alloc"], 1)

    def test_cold_capture_fails_closed(self):
        mod, calls = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: True, stream_id=lambda d: 7)
        with self.assertRaises(RuntimeError):
            mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
        self.assertEqual(calls, [])

    def test_warm_capture_uses_persistent(self):
        mod, calls = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: True, stream_id=lambda d: 7)
        ws = G.ws_prepare(mod, "cpu")
        mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
        self.assertIs(calls[0], ws)

    def test_capture_streams_recorded_per_graph(self):
        mod, calls = _marlin_mod()
        sid = [5]
        G.install_moe_ws(mod, capturing=lambda: True, stream_id=lambda d: sid[0])
        G.ws_prepare(mod, "cpu")
        G._CAPTURE_CTX.append("serve|2|M|d|graph")
        try:
            mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
            mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
            self.assertEqual(G.status()["ws_capture_multi_stream"], 0)
            sid[0] = 6                                  # a second stream inside one graph
            mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
            self.assertEqual(G.status()["ws_capture_multi_stream"], 1)
        finally:
            G._CAPTURE_CTX.pop()

    def test_explicit_keyword_workspace_untouched(self):
        mod, calls = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: False, stream_id=lambda d: 7)
        mine = torch.zeros(4, dtype=torch.int32)
        mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)), workspace=mine)
        self.assertIs(calls[0], mine)

    def test_positional_workspace_preserved_no_duplicate(self):
        mod, calls = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: False, stream_id=lambda d: 7)
        mine = torch.zeros(4, dtype=torch.int32)
        pos = [torch.zeros(3, 8)] + [None] * 9 + [False, -1, None, None, None, None] + [None] * 10 + [mine]
        mod.fused_marlin_moe(*pos)                 # explicit positional workspace kept
        self.assertIs(calls[-1], mine)
        pos[-1] = None
        ws = G.ws_prepare(mod, "cpu")
        mod.fused_marlin_moe(*pos)                 # positional None: injected once, no TypeError
        self.assertIs(calls[-1], ws)

    def test_foreign_stream_falls_back_to_stock(self):
        mod, calls = _marlin_mod()
        sid = [1]
        G.install_moe_ws(mod, capturing=lambda: False, stream_id=lambda d: sid[0])
        ws = G.ws_prepare(mod, "cpu")
        mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
        sid[0] = 2
        mod.fused_marlin_moe(**_kw(torch.zeros(3, 8)))
        self.assertIs(calls[0], ws)
        self.assertIsNone(calls[1])
        self.assertEqual(G.STATS["counts"]["-1|eager"]["ws_foreign_stream_stock"], 1)


# ---------------------------------------------------------------- F3 fakes
def _fake_convert_factory(log):
    def convert(req_id, block_table, token_indices, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048,
                return_valid_counts=False, **kw):
        log.append(token_indices.data_ptr())
        blk = token_indices // BLOCK_SIZE
        out = block_table[req_id.long()].gather(1, blk.clamp(min=0).long()) * BLOCK_SIZE + token_indices % BLOCK_SIZE
        out = torch.where(token_indices < 0, torch.full_like(out, -1), out).int()
        cnt = (out >= 0).sum(1).int()
        return (out, cnt) if return_valid_counts else out
    return convert


class _Mla(torch.nn.Module):
    """Stand-in for MultiHeadLatentAttentionWrapper: optional indexer write, then convert."""
    def __init__(self, fi, buf, md, skip, writes):
        super().__init__()
        self.fi, self.md, self.skip_topk, self.is_sparse, self.writes = fi, md, skip, True, writes
        self.topk_indices_buffer = buf

    def forward(self, positions, hidden_states):
        buf = self.topk_indices_buffer
        if not self.skip_topk:
            self.writes(buf)          # indexer rewrites the shared topk buffer
        t = hidden_states.shape[0]
        return self.fi.triton_convert_req_index_to_global_index(
            self.md["req"][:t], self.md["bt"], buf[:t], BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=buf.shape[1], return_valid_counts=True)


class DsaIndexCacheF3(unittest.TestCase):
    def _setup(self, pattern, t=3, seed=0, arm=True, mtp_pattern=""):
        torch.manual_seed(seed)
        reset_stats()
        log = []
        fi = types.SimpleNamespace(triton_convert_req_index_to_global_index=_fake_convert_factory(log))
        G.install_idx_convert(fi)
        if getattr(_Mla.forward, "_glm_glue", False):
            _Mla.forward = _Mla.forward.__wrapped__
        mla_mod = types.SimpleNamespace(MultiHeadLatentAttentionWrapper=_Mla)
        G.install_idx_mla(mla_mod)
        buf = torch.full((16, 128), -1, dtype=torch.int32)
        md = {"req": torch.zeros(16, dtype=torch.int32), "bt": torch.randint(0, 1000, (4, 64), dtype=torch.int32)}

        def writes(b):
            b[:t] = torch.randint(-1, 2048, (t, b.shape[1]), dtype=torch.int32)
        layers = torch.nn.ModuleList([_Mla(fi, buf, md, s == "S", writes) for s in pattern])
        mtp = torch.nn.ModuleList([_Mla(fi, buf, md, s == "S", writes) for s in mtp_pattern])
        if arm:
            G.arm_idx({"target": layers, "mtp": mtp if mtp_pattern else None}, mla_mod, env={"GLM_GLUE_STRICT": "0"})
        return fi, layers, log, buf, md, mtp

    def _reference(self, pattern, seed=0):
        """Same forward with the cache disabled (stock)."""
        torch.manual_seed(seed)
        conv = _fake_convert_factory([])
        buf = torch.full((16, 128), -1, dtype=torch.int32)
        md = {"req": torch.zeros(16, dtype=torch.int32), "bt": torch.randint(0, 1000, (4, 64), dtype=torch.int32)}
        outs = []
        for s in pattern:
            if s == "C":
                buf[:3] = torch.randint(-1, 2048, (3, buf.shape[1]), dtype=torch.int32)
            outs.append(conv(md["req"][:3], md["bt"], buf[:3], BLOCK_SIZE=64, NUM_TOPK_TOKENS=128,
                             return_valid_counts=True))
        return outs

    def test_skip_layers_reuse_and_match_stock(self):
        pattern = "CSSSCSSSCSSS"
        fi, layers, log, *_ = self._setup(pattern)
        x = torch.zeros(3, 4)
        got = [L.forward(None, x) for L in layers]
        ref = self._reference(pattern)
        for (a, ac), (b, bc) in zip(got, ref):
            self.assertTrue(torch.equal(a, b) and torch.equal(ac, bc))
        self.assertEqual(len(log), 3)               # one real convert per computing layer
        self.assertEqual(G.STATS["counts"]["-1|eager"]["idx_hit"], 9)
        self.assertEqual(G.STATS["idx_armed"]["target"], dict(skip_armed=9, other=3))

    def test_unarmed_skip_layer_never_reuses(self):
        # MTP model layers are never armed, even if skip_topk were set at runtime.
        fi, layers, log, buf, md, mtp = self._setup("CS", mtp_pattern="S")
        x = torch.zeros(3, 4)
        layers[0].forward(None, x)
        mtp[0].forward(None, x)                     # same key, skip_topk=True, but unarmed: converts
        self.assertEqual(len(log), 2)
        self.assertNotIn("idx_hit", G.STATS["counts"]["-1|eager"])
        self.assertEqual(G.STATS["idx_armed"]["mtp"], dict(skip_armed=0, other=1))

    def test_no_arming_no_reuse(self):
        fi, layers, log, *_ = self._setup("CSS", arm=False)
        for L in layers:
            L.forward(None, torch.zeros(3, 4))
        self.assertEqual(len(log), 3)

    def test_arm_strict_expect(self):
        fi, layers, *_ = self._setup("CSS", arm=False)
        mla_mod = types.SimpleNamespace(MultiHeadLatentAttentionWrapper=_Mla)
        with self.assertRaises(RuntimeError):
            G.arm_idx({"target": layers}, mla_mod, env={"GLM_GLUE_STRICT": "1", "GLM_GLUE_IDX_EXPECT": "57"})
        reset_stats()
        out = G.arm_idx({"target": layers}, mla_mod, env={"GLM_GLUE_STRICT": "1", "GLM_GLUE_IDX_EXPECT": "2"})
        self.assertEqual(out["target"]["skip_armed"], 2)

    def test_arm_refuses_split_topk_buffers(self):
        fi, layers, *_ = self._setup("CS", arm=False)
        layers[1].topk_indices_buffer = layers[1].topk_indices_buffer.clone()
        with self.assertRaises(RuntimeError):
            G.arm_idx({"target": layers}, types.SimpleNamespace(MultiHeadLatentAttentionWrapper=_Mla),
                      env={"GLM_GLUE_STRICT": "0"})

    def test_width_change_misses(self):
        fi, layers, log, buf, md, _ = self._setup("CS")
        layers[0].forward(None, torch.zeros(3, 4))
        layers[1].forward(None, torch.zeros(1, 4))  # different t -> recompute
        self.assertEqual(len(log), 2)
        self.assertEqual(G.STATS["counts"]["-1|eager"]["idx_miss_skip"], 1)

    def test_block_table_swap_misses(self):
        fi, layers, log, buf, md, _ = self._setup("CS")
        layers[0].forward(None, torch.zeros(3, 4))
        md["bt"] = md["bt"].clone()                  # new metadata tensor -> new pointer
        layers[1].forward(None, torch.zeros(3, 4))
        self.assertEqual(len(log), 2)

    def test_check_mode_catches_unseen_writer(self):
        fi, layers, log, buf, md, _ = self._setup("CSS")
        G._CHECK[0] = True
        layers[0].forward(None, torch.zeros(3, 4))
        layers[1].forward(None, torch.zeros(3, 4))      # honest hit: fresh == cached
        buf[:3] += 1                                      # a writer the cache cannot see
        with self.assertRaises(RuntimeError):
            layers[2].forward(None, torch.zeros(3, 4))

    def test_next_forward_invalidates(self):
        fi, layers, log, buf, md, _ = self._setup("CS")
        for _ in range(2):
            for L in layers:
                L.forward(None, torch.zeros(3, 4))
        self.assertEqual(len(log), 2)                # layer 0 recomputes every forward
        self.assertEqual(G.STATS["counts"]["-1|eager"]["idx_hit"], 2)


# ---------------------------------------------------------------- import-hook composition
class _TransformFinder(importlib.abc.MetaPathFinder):
    """glm_ab.Finder's contract: needs loader.get_source, sets loader.get_code, chains exec."""
    def __init__(self, name, log):
        self.name, self.log = name, log

    def find_spec(self, name, path=None, target=None):
        if name != self.name:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if not hasattr(spec.loader, "get_source"):
            raise RuntimeError("source loader required: " + name)
        src = spec.loader.get_source(name).replace("VALUE = 1", "VALUE = 2")
        code = compile(src, spec.origin, "exec")
        spec.loader.get_code = lambda fullname: code
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            self.log.append("transform-post")
        spec.loader.exec_module = execute
        return spec


class ImportComposition(unittest.TestCase):
    def _run(self, glue_first):
        tmp = tempfile.mkdtemp()
        name = "glue_compose_mod_" + ("a" if glue_first else "b")
        Path(tmp, name + ".py").write_text("VALUE = 1\n")
        sys.path.insert(0, tmp)
        saved = list(sys.meta_path)
        log = []
        try:
            G._FINDER = None
            tf = _TransformFinder(name, log)
            if glue_first:
                sys.meta_path.insert(0, tf)
                G.after_import(name, lambda m: log.append(("glue", m.VALUE)))   # inserted in front
            else:
                G.after_import(name, lambda m: log.append(("glue", m.VALUE)))
                sys.meta_path.insert(0, tf)                                    # glm_ab moves itself first
            mod = importlib.import_module(name)
            self.assertEqual(mod.VALUE, 2)                      # transformed code executed
            self.assertIn(("glue", 2), log)                     # post-import hook saw it
            self.assertIn("transform-post", log)
        finally:
            sys.meta_path[:] = saved
            sys.path.remove(tmp)
            G._FINDER = None

    def test_glue_finder_in_front(self):
        self._run(True)

    def test_transform_finder_in_front(self):
        self._run(False)


# ---------------------------------------------------------------- receipts / bankable
class _FakeAB(types.SimpleNamespace):
    pass


def _fake_ab(bank_values):
    state = {"cur": 0}
    specs = [dict(zip(LEVERS, v)) for v in bank_values]
    ab = _FakeAB(ACTIVE=True, KNOWN=set(LEVERS), CONFIG_HASH="cfg", _specs=specs,
                 env=lambda n, d=None: specs[state["cur"]].get(n, d),
                 truthy=lambda v: v is not None and str(v).lower() not in ("", "0", "off", "false", "no"),
                 current=lambda: state["cur"], descriptor=lambda d: ("FULL", d.num_tokens),
                 exchange=lambda v: [v, v, v, v])
    return ab, state


class Bankable(unittest.TestCase):
    def setUp(self):
        reset_stats()
        self.ab, self.state = _fake_ab([("0", "0", "0"), ("0", "0", "0"), ("1", "1", "1")])
        sys.modules["glm_ab"] = self.ab

    def tearDown(self):
        sys.modules.pop("glm_ab", None)
        os.environ.pop("GLM_GLUE_LITE_BANKABLE", None)

    def test_control_bank_is_stock(self):
        _fresh_gate_forward()
        gate = _Gate(64, 256)
        gate._glm_bf16_logits = True
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        self.assertEqual(gate(x)[0].dtype, torch.float32)          # control bank: cast kept
        self.state["cur"] = 2
        self.assertEqual(gate(x)[0].dtype, torch.bfloat16)         # glue bank
        self.assertEqual(G.STATS["counts"]["0|eager"]["router_stock"], 1)
        self.assertEqual(G.STATS["counts"]["2|eager"]["router_bf16"], 1)

    def test_ws_follows_bank(self):
        mod, seen = _marlin_mod()
        G.install_moe_ws(mod, capturing=lambda: False, stream_id=lambda d: 1)
        G.ws_prepare(mod, "cpu")
        mod.fused_marlin_moe(**_kw(torch.zeros(1)))
        self.state["cur"] = 2
        mod.fused_marlin_moe(**_kw(torch.zeros(1)))
        self.assertIsNone(seen[0])
        self.assertIsNotNone(seen[1])

    def test_bankable_register_installs_all(self):
        self.assertEqual(G.wanted({"GLM_GLUE_LITE_BANKABLE": "1"}), {"router": True, "ws": True, "idx": True})

    def test_bankable_without_harness_refuses(self):
        sys.modules.pop("glm_ab")
        os.environ["GLM_GLUE_LITE_BANKABLE"] = "1"
        with self.assertRaises(RuntimeError):
            G.lever("GLM_GLUE_MOE_WS")

    def test_capture_receipts_per_bank_and_descriptor(self):
        _fresh_gate_forward()

        class Manager:
            def capture(self, create_forward_fn, progress_bar_desc="x"):
                for desc in (types.SimpleNamespace(num_tokens=3), types.SimpleNamespace(num_tokens=12)):
                    create_forward_fn(desc, warmup=True)(None)
                    create_forward_fn(desc, warmup=False)(None)
        G.install_cg(types.SimpleNamespace(CudaGraphManager=Manager))
        gate = _Gate(64, 256)
        gate._glm_bf16_logits = True
        x = torch.randn(3, 64, dtype=torch.bfloat16)

        def factory(desc, warmup):
            return lambda mode: gate(x)
        real = G._capturing
        G._capturing = lambda: True
        try:
            for b in (0, 2):
                self.state["cur"] = b
                Manager().capture(factory)
        finally:
            G._capturing = real
        caps = G.STATS["captures"]
        self.assertEqual(caps["serve|2|Manager|('FULL', 12)|graph"], {"router_bf16": 1})
        self.assertEqual(caps["serve|0|Manager|('FULL', 3)|graph"], {"router_stock": 1})
        self.assertEqual(len(caps), 8)

    def test_status_and_agreed_view(self):
        s = G.status()
        self.assertEqual(s["config"], "cfg")
        self.assertEqual(s["specs"][2], dict(zip(LEVERS, ("1", "1", "1"))))
        view = G.agreed_view(dict(s, rank=3, mem_available_bytes=1))
        self.assertNotIn("rank", view)
        self.assertNotIn("counts", view)            # eager counts are reported, not agreement-gated


class Register(unittest.TestCase):
    def test_inert_by_default(self):
        self.assertEqual(G.register(env={}), {"router": False, "ws": False, "idx": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
