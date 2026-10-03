# SPDX-License-Identifier: Apache-2.0
"""GLM glue-lite: three bit-exact, Python-only launch removals for full GLM-5.3 (FIX1).

Pinned to vLLM 487ecf187 (image sha256:4def0ef6..., V2 model runner). Inert unless a
switch is set. Every switch only removes a kernel whose result is provably unused or
recomputed identically; none changes arithmetic. FIX1 narrows every guard to exactly
what the exactness argument and the GPU gates cover (tests/gpu/, docs/runtime.md).

GLM_GLUE_ROUTER_BF16=1   (F1) GateLinear Tier 6 returns the BF16 router logits instead of
    `output.to(float32)` (gate_linear.py:226-232). Armed per MoE runner only for the exact
    proven contract: GateLinear/GroupedTopKRouter (exact types), BF16 [256,6144] weight,
    FP32 out_dtype, no bias, every specialised tier off, runner applies this gate itself
    (no fused shared-expert gate), modular (non-monolithic) experts, fused CUDA
    grouped_topk enabled, sigmoid + FP32 [256] correction bias, E=256, num_expert_group=1,
    topk_group=1, top_k=8, renormalize, routed_scaling_factor in {1.0, 2.5}. That contract
    reaches single_group_topk (block kernel M<=1024, warp kernel above), whose
    preprocess_score widens each BF16 logit with __bfloat162float before any arithmetic,
    i.e. the same float the cast produced (grouped_topk_kernels.cu:866-881). Multi-group
    layouts (generic fallback rounds sigmoid/bias in T) are refused.
GLM_GLUE_MOE_WS=1        (F2) fused_marlin_moe gets one persistent, zero-initialised lock
    workspace per device instead of torch.zeros per call (marlin_moe.py:101-102). It is
    preallocated at GPUModelRunner.load_model; a capture that finds no workspace fails
    closed; a caller-supplied workspace (keyword or positional) is never touched; an eager
    call on a stream other than the first one seen falls back to the stock allocation and
    is counted. Reuse relies on barrier_release(reset=last) zeroing every lock after a
    completed call (marlin_template.h:200, :2162): serial, stream-ordered reuse only.
GLM_GLUE_DSA_IDX_CACHE=1 (F3) armed static skip_topk layers of the TARGET model (never the
    MTP model) reuse the (slots, counts) produced by the last computing layer of the same
    forward when every input pointer/shape/stride/flag is identical, so
    triton_convert_req_index_to_global_index and its full_like(-1) are not launched. Every
    MLA forward that is not an armed skip layer invalidates the cache.

GLM_GLUE_STRICT (default 1): refuse source drift (all PINS, including sparse_utils,
deepseek_v2/mtp, moe_runner and marlin_utils), refuse any MoE router outside the F1
contract, and require exact arming counts when GLM_GLUE_ROUTER_EXPECT ("target:75,mtp:1")
/ GLM_GLUE_IDX_EXPECT ("57") are set.
GLM_GLUE_LITE_BANKABLE=1: install all hooks but decide each switch at call/capture time
through an external in-boot A/B harness (glm_ab env(); the three names are in its
KNOWN/DEFAULTS/normalize), so one boot captures control banks and a glue bank. The harness
is not part of this repository; the serving startup refuses this mode.
GLM_GLUE_DSA_IDX_CHECK=1: every eager F3 hit recomputes and raises on a mismatch (toggled at
runtime by the drained glue_lite_control RPC; off for scored cells).
Receipts: Worker.glue_lite_status (rank-agreed when asked) and per-bank/per-descriptor
capture counters (install_cg).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.util
import inspect
import os
import sys
from pathlib import Path

PINS = {
    "vllm.model_executor.layers.fused_moe.router.gate_linear":
        "6fca8c7fba361ac79418dd56d6d3c281ef04f34ebfdd2044cb1d12e1fa9e70d7",
    "vllm.model_executor.layers.fused_moe.router.grouped_topk_router":
        "6a541bfdcf9e9a09fc888ea3f4a2e9689181e889e3a3ac9f49716d8d59cca142",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner":
        "a19ecf2417c80b93c0fe5594e74528f686665847732196f9b29ef967b38110b0",
    "vllm.model_executor.layers.fused_moe.experts.marlin_moe":
        "f00460310b033bcb84644677e16af83773464f026c0dad15bcfe25bd28b39745",
    "vllm.model_executor.layers.quantization.utils.marlin_utils":
        "6f91cbf2b0ae276abd4b57b80b20f260307f9df3e24d40ee2b0abf76f4c7f100",
    "vllm.model_executor.layers.mla":
        "936b06c4671d52fce52ae85bb24b986db17855f32740bf55b226053fc385c4b4",
    # image copy (differs from git HEAD 84ee251b...); same pin as bringup/glm_full_mla.py
    "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90":
        "4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136",
    "vllm.v1.attention.backends.mla.sparse_utils":
        "cd95caf494f1d2a946ac87abfb4951f738827fe7749aeefd6c0fd83d1cee072d",
    # skip_topk pattern (:1084-1110) and MTP non-skip initialisation (:1169-1174)
    "vllm.model_executor.models.deepseek_v2":
        "58d8916458de7c6f73b40bfef9d2f57bdd6fa0fb79be9e269331af6e66149fe2",
    "vllm.model_executor.models.deepseek_mtp":
        "88521bcf3bfec6c773998dbefe30448504640907a4ac98f4e822d0ed38e2436f",
    "vllm.v1.worker.gpu.model_runner":
        "f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415",
}
# Modules that are only pinned (their code is read, never patched here).
PIN_ONLY = {
    "router": ("vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
               "vllm.model_executor.layers.fused_moe.runner.moe_runner"),
    "ws": ("vllm.model_executor.layers.quantization.utils.marlin_utils",),
    "idx": ("vllm.v1.attention.backends.mla.sparse_utils",
            "vllm.model_executor.models.deepseek_v2",
            "vllm.model_executor.models.deepseek_mtp"),
}
LEVERS = ("GLM_GLUE_ROUTER_BF16", "GLM_GLUE_MOE_WS", "GLM_GLUE_DSA_IDX_CACHE")

# F1 contract: exactly what the source argument and tools/gpu_bitexact_f1.py cover.
ROUTER_CONTRACT = dict(num_experts=256, hidden=6144, num_expert_group=1, topk_group=1, top_k=8,
                       scoring_func="sigmoid", renormalize=True, routed_scaling_factors=(1.0, 2.5))
GATE_TIERS = ("allow_specialized_router_gemm", "allow_ll_bf16_gemm", "allow_dsv3_router_gemm",
              "allow_fp32_router_gemm", "allow_bf16x3_router_gemm", "allow_cublas_router_gemm")

STATS = {"router_armed": {}, "router_refused": [], "idx_armed": {}, "ws_alloc": 0,
         "pins_checked": [], "counts": {}, "captures": {}, "ws_streams": {}, "ws_capture_streams": {},
         "runner_hooked": False,
         "load_model_armed": 0}


def _on(name, env=None):
    env = os.environ if env is None else env
    return env.get(name, "0").strip().lower() not in ("0", "", "off", "false", "no")


def wanted(env=None):
    bank = _on("GLM_GLUE_LITE_BANKABLE", env)
    return {"router": bank or _on("GLM_GLUE_ROUTER_BF16", env), "ws": bank or _on("GLM_GLUE_MOE_WS", env),
            "idx": bank or _on("GLM_GLUE_DSA_IDX_CACHE", env)}


def _ab():
    ab = sys.modules.get("glm_ab")
    return ab if ab is not None and getattr(ab, "ACTIVE", False) else None


def lever(name):
    """Call/capture-time switch: the in-boot A/B bank value when glm_ab is active, else the env."""
    ab = _ab()
    if ab is not None and name in getattr(ab, "KNOWN", ()):
        return ab.truthy(ab.env(name))
    if _on("GLM_GLUE_LITE_BANKABLE"):
        raise RuntimeError("glm-glue-lite: bankable mode without an active glm_ab bank harness")
    return _on(name)


def bank():
    ab = _ab()
    return ab.current() if ab is not None else -1


def strict(env=None):
    env = os.environ if env is None else env
    return env.get("GLM_GLUE_STRICT", "1") != "0"


def check_pin(modname, path):
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != PINS[modname]:
        raise RuntimeError(f"glm-glue-lite: source drift {modname} {path}: {got} != {PINS[modname]}")
    if modname not in STATS["pins_checked"]:
        STATS["pins_checked"].append(modname)


# --------------------------------------------------------------------------- receipts
_CAPTURE_CTX = []  # stack of (bank, manager, descriptor, warmup) while a capture factory runs


def _capturing():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def tick(name, n=1):
    """Python-path counters. Graph replays do not run Python: capture counters are the
    per-graph receipts (each tick on a capture = one stock launch removed or kept)."""
    phase = "capture" if _capturing() else "eager"
    key = f"{bank()}|{phase}"
    row = STATS["counts"].setdefault(key, {})
    row[name] = row.get(name, 0) + n
    if phase == "capture" and _CAPTURE_CTX:
        ctx = _CAPTURE_CTX[-1]
        row = STATS["captures"].setdefault(ctx, {})
        row[name] = row.get(name, 0) + n


def install_cg(mod):
    """Wrap CudaGraphManager.capture's forward factory (same pattern as dsa_short_fix) so
    every capture/warmup of every bank records which glue paths it took."""
    cls = mod.CudaGraphManager
    if getattr(cls.capture, "_glm_glue", False):
        return
    capture = cls.capture

    @functools.wraps(capture)
    def captured(self, factory, *a, **kw):
        def create(desc, warmup):
            fn = factory(desc, warmup)

            def run(mode):
                ab = _ab()
                d = ab.descriptor(desc) if ab is not None else (str(desc.cg_mode), desc.num_tokens)
                prof = bool(ab is not None and getattr(ab, "_state", {}).get("profiling"))
                _CAPTURE_CTX.append(f"{'profile' if prof else 'serve'}|{bank()}|{type(self).__name__}|{d}|"
                                    f"{'warmup' if warmup else 'graph'}")
                try:
                    return fn(mode)
                finally:
                    _CAPTURE_CTX.pop()
            return run
        return capture(self, create, *a, **kw)

    captured._glm_glue = True
    cls.capture = captured


def status(env=None):
    import torch
    out = dict(module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               wanted=wanted(env), strict=strict(env), pins_checked=sorted(STATS["pins_checked"]),
               router_armed=dict(STATS["router_armed"]), router_refused=list(STATS["router_refused"]),
               idx_armed=dict(STATS["idx_armed"]), ws_alloc=STATS["ws_alloc"],
               ws_bytes={str(k): int(v.numel() * v.element_size()) for k, v in _WS.items()},
               ws_zero={str(k): int(torch.count_nonzero(v).item()) == 0 for k, v in _WS.items()},
               idx_check=_CHECK[0], bank=bank(), counts=dict(STATS["counts"]),
               runner_hooked=STATS["runner_hooked"], load_model_armed=STATS["load_model_armed"],
               ws_capture_multi_stream=sum(len(v) > 1 for v in STATS["ws_capture_streams"].values()),
               captures=dict(STATS["captures"]))
    ab = _ab()
    if ab is not None:
        out.update(config=ab.CONFIG_HASH, specs=[{k: s.get(k, "0") for k in LEVERS} for s in ab._specs])
    return out


def agreed_view(s):
    """Fields every TP rank must agree on (no pointers, streams, memory or eager counts)."""
    keep = ("module_sha256", "wanted", "strict", "pins_checked", "router_armed", "router_refused",
            "runner_hooked", "load_model_armed", "ws_capture_multi_stream",
            "idx_armed", "ws_bytes", "ws_zero", "idx_check", "bank", "captures", "config", "specs")
    return {k: s.get(k) for k in keep}


def install_worker(mod):
    def glue_lite_status(self, agree=False):
        from vllm.distributed.parallel_state import get_tp_group
        rank = get_tp_group().rank_in_group
        s = status()
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        s.update(rank=rank, mem_available_bytes=int(mem["MemAvailable"].split()[0]) * 1024,
                 ws_ptr={k: int(v.data_ptr()) for k, v in ((str(a), b) for a, b in _WS.items())},
                 ws_streams=dict(STATS["ws_streams"]))
        if agree:
            ab = _ab()
            if ab is None:
                raise RuntimeError("glue_lite_status(agree) requires the glm_ab TP exchange")
            import torch
            torch.cuda.synchronize()
            views = ab.exchange(agreed_view(s))
            s["ranks_agree"] = all(v == views[0] for v in views)
            if not s["ranks_agree"]:
                diff = sorted({k for v in views for k in v if v.get(k) != views[0].get(k)})
                s["disagreeing_fields"] = diff
        return s

    def glue_lite_control(self, check):
        if not isinstance(check, bool):
            raise ValueError("check must be a bool")
        _CHECK[0] = check
        from vllm.distributed.parallel_state import get_tp_group
        return dict(rank=get_tp_group().rank_in_group, idx_check=_CHECK[0])

    mod.Worker.glue_lite_status = glue_lite_status
    mod.Worker.glue_lite_control = glue_lite_control


# --------------------------------------------------------------------------- import hooks
class _AfterImport(importlib.abc.MetaPathFinder):
    """Post-import callbacks that keep the found spec's loader object.

    FIX1: the old finder replaced spec.loader with a wrapper class without get_source, so a
    source-transforming finder in front of it (glm_ab.Finder, which requires get_source and
    sets loader.get_code) refused the module. Chaining the instance's exec_module keeps
    get_source/get_code and composes in either finder order.
    """

    def __init__(self):
        self.hooks = {}

    def add(self, name, fn):
        self.hooks.setdefault(name, []).append(fn)

    def find_spec(self, name, path=None, target=None):
        fns = self.hooks.get(name)
        if not fns or name in sys.modules:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            for fn in fns:
                fn(module)

        loader.exec_module = exec_module
        return spec


_FINDER = None


def after_import(name, fn):
    global _FINDER
    if name in sys.modules:
        fn(sys.modules[name])
        return
    if _FINDER is None:
        _FINDER = _AfterImport()
        sys.meta_path.insert(0, _FINDER)
    _FINDER.add(name, fn)


def _pinned(modname, fn, env=None):
    def run(mod):
        if strict(env):
            check_pin(modname, mod.__file__)
        fn(mod)
    return run


# --------------------------------------------------------------------------- F1 router logits
def router_eligible(owner, gate, router, envs, platform, gate_cls, router_cls):
    """'' when the BF16 logits provably reach single_group_topk's widening load, else why not."""
    import torch
    c = ROUTER_CONTRACT
    if type(gate) is not gate_cls:
        return "gate is not exactly GateLinear"
    if type(router) is not router_cls:
        return "router is not exactly GroupedTopKRouter"
    if getattr(owner, "gate", None) is not gate:
        return "owner does not apply this gate"
    if getattr(owner, "_fse_fuse_gate", False) or getattr(owner, "shared_expert_gate", None) is not None:
        return "fused shared-expert gate (logits come from F.linear on combined weights)"
    qm = getattr(getattr(owner, "routed_experts", None), "quant_method", None)
    if qm is None or getattr(qm, "is_monolithic", True):
        return "monolithic/unknown expert kernel (logits would reach the expert kernel)"
    w = getattr(gate, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != (c["num_experts"], c["hidden"]):
        return "weight is not BF16 [256, 6144]"
    if gate.out_dtype != torch.float32:
        return "out_dtype is not FP32"
    if getattr(gate, "bias", None) is not None:
        return "gate has bias"
    for tier in GATE_TIERS:
        if getattr(gate, tier, False):
            return f"{tier} active (not the Tier-6 cast path)"
    if not getattr(envs, "VLLM_USE_FUSED_MOE_GROUPED_TOPK", False) or not platform.is_cuda():
        return "fused CUDA grouped_topk disabled"
    if router.scoring_func != c["scoring_func"]:
        return "scoring is not sigmoid"
    bias = router.e_score_correction_bias
    if bias is None or bias.dtype != torch.float32 or tuple(bias.shape) != (c["num_experts"],):
        return "correction bias is not FP32 [256]"
    if (router.global_num_experts != c["num_experts"] or router.num_expert_group != c["num_expert_group"]
            or router.topk_group != c["topk_group"] or router.top_k != c["top_k"]):
        return (f"routing E{router.global_num_experts}/G{router.num_expert_group}/"
                f"TG{router.topk_group}/top{router.top_k} outside the proven E256/G1/TG1/top8 contract")
    if router.renormalize is not c["renormalize"]:
        return "renormalize differs from the gated contract"
    if float(router.routed_scaling_factor) not in c["routed_scaling_factors"]:
        return f"routed_scaling_factor {router.routed_scaling_factor} not covered by the GPU gate"
    if getattr(router, "num_fused_shared_experts", 0):
        return "fused shared experts in the router"
    return ""


def install_router_forward(mod_gate):
    cls = mod_gate.GateLinear
    if getattr(cls.forward, "_glm_glue", False):
        return
    original = cls.forward
    base = mod_gate.ReplicatedLinear  # Tier 6 calls super().forward(x) == ReplicatedLinear.forward
    assert cls.__mro__[1] is base, "GateLinear must derive directly from ReplicatedLinear"

    @functools.wraps(original)
    def forward(self, x):
        if getattr(self, "_glm_bf16_logits", False):
            if lever("GLM_GLUE_ROUTER_BF16"):
                if x.dtype != self.weight.dtype:
                    raise RuntimeError("glm-glue-lite: armed router saw non-BF16 activations")
                tick("router_bf16")
                out, out_bias = base.forward(self, x)  # == Tier 6 minus `.to(float32)`
                return out, out_bias
            tick("router_stock")
        return original(self, x)

    forward._glm_glue = True
    cls.forward = forward


def _expect(env, name):
    raw = (os.environ if env is None else env).get(name, "").strip()
    if not raw:
        return None
    if ":" not in raw:
        return int(raw)
    return {k: int(v) for k, v in (x.split(":") for x in raw.split(","))}


def arm_router(models, mod_gate, mod_router, envs, platform, env=None):
    """models: {'target': nn.Module, 'mtp': nn.Module|None}."""
    armed, refused, seen = {}, [], set()
    for tag, model in models.items():
        if model is None:
            continue
        armed.setdefault(tag, 0)
        for name, m in model.named_modules():
            gate, router = getattr(m, "gate", None), getattr(m, "router", None)
            if gate is None or router is None or id(gate) in seen:
                continue
            seen.add(id(gate))
            why = router_eligible(m, gate, router, envs, platform, mod_gate.GateLinear, mod_router.GroupedTopKRouter)
            if why:
                refused.append((tag + ":" + name, why))
                continue
            gate._glm_bf16_logits = True
            armed[tag] += 1
    for tag, n in armed.items():
        STATS["router_armed"][tag] = STATS["router_armed"].get(tag, 0) + n
    STATS["router_refused"] += refused
    sys.stderr.write(f"glm-glue-lite: F1 router bf16 logits armed {armed}, refused {len(refused)}"
                     + (f" (first: {refused[0]})" if refused else "") + "\n")
    if strict(env):
        if refused:
            raise RuntimeError(f"glm-glue-lite: MoE router outside the F1 contract: {refused[:3]}")
        expect = _expect(env, "GLM_GLUE_ROUTER_EXPECT")
        if expect is None and sum(armed.values()) == 0:
            raise RuntimeError("glm-glue-lite: F1 requested but no eligible layer")
        if expect is not None and armed != expect:
            raise RuntimeError(f"glm-glue-lite: F1 armed {armed} != expected {expect}")
    return armed, refused


# --------------------------------------------------------------------------- F2 Marlin workspace
_WS = {}
_HOME = {}


def ws_prepare(mod_marlin, device):
    import torch
    device = torch.device(device)
    if device not in _WS:
        _WS[device] = mod_marlin.marlin_make_workspace_new(device, 4)  # zeros, same size as per-call
        STATS["ws_alloc"] += 1
    return _WS[device]


def install_moe_ws(mod_marlin, capturing=None, stream_id=None):
    if getattr(mod_marlin.fused_marlin_moe, "_glm_glue", False):
        return
    original = mod_marlin.fused_marlin_moe
    sig = inspect.signature(original)
    names = list(sig.parameters)
    if names[0] != "hidden_states" or "workspace" not in names:
        raise RuntimeError("glm-glue-lite: fused_marlin_moe signature drift")
    ws_index = names.index("workspace")
    if capturing is None:
        capturing = _capturing
    if stream_id is None:
        def stream_id(device):
            import torch
            return torch.cuda.current_stream(device).cuda_stream

    @functools.wraps(original)
    def fused_marlin_moe(*args, **kw):
        if len(args) > ws_index:  # positional workspace (possibly None): bind to keep one value
            bound = sig.bind(*args, **kw)
            args, kw = (), dict(bound.arguments)
        given = kw.get("workspace")
        if given is not None or not lever("GLM_GLUE_MOE_WS"):
            if given is None:
                tick("ws_stock")
            return original(*args, **kw)
        device = (kw["hidden_states"] if "hidden_states" in kw else args[0]).device
        ws = _WS.get(device)
        if capturing():
            if ws is None:
                raise RuntimeError("glm-glue-lite: F2 capture with a cold workspace (fail closed)")
            # One graph must use the workspace from one stream only (serialized reuse).
            ctx = _CAPTURE_CTX[-1] if _CAPTURE_CTX else "uncontexted"
            seen = STATS["ws_capture_streams"].setdefault(ctx, [])
            sid = stream_id(device)
            if sid not in seen:
                seen.append(sid)
        else:
            sid = stream_id(device)
            home = _HOME.setdefault(device, sid)
            key = f"{device}|{'home' if sid == home else 'foreign'}"
            STATS["ws_streams"][key] = STATS["ws_streams"].get(key, 0) + 1
            if sid != home:
                tick("ws_foreign_stream_stock")  # serialized reuse is not proven across streams
                return original(*args, **kw)
            if ws is None:
                ws = ws_prepare(mod_marlin, device)
        kw["workspace"] = ws  # workspace is not positional here (len(args) <= ws_index)
        tick("ws_persistent")
        return original(*args, **kw)

    fused_marlin_moe._glm_glue = True
    mod_marlin.fused_marlin_moe = fused_marlin_moe


# --------------------------------------------------------------------------- F3 DSA index cache
class _IdxState:
    cache = None   # (key, result) produced by the last computing (non-skip) MLA layer
    reuse = False  # True while an armed skip_topk layer runs


_IDX = _IdxState()
_CHECK = [_on("GLM_GLUE_DSA_IDX_CHECK")]


def _key(req_id, block_table, token_indices, kw):
    import torch
    if any(torch.is_tensor(v) for v in kw.values()):
        return None  # prefill-workspace variants: never cached
    return (req_id.data_ptr(), tuple(req_id.shape), tuple(req_id.stride()), str(req_id.dtype),
            block_table.data_ptr(), tuple(block_table.shape), tuple(block_table.stride()), str(block_table.dtype),
            token_indices.data_ptr(), tuple(token_indices.shape), tuple(token_indices.stride()),
            str(token_indices.dtype), str(token_indices.device), tuple(sorted((k, repr(v)) for k, v in kw.items())))


def arm_idx(models, mod_mla, env=None):
    """Arm only static skip layers of the target model; MTP layers are never armed."""
    cls = mod_mla.MultiHeadLatentAttentionWrapper
    out = {}
    for tag, model in models.items():
        if model is None:
            continue
        skip = compute = 0
        buffers = set()
        for _, m in model.named_modules():
            if type(m) is not cls or not getattr(m, "is_sparse", False):
                continue
            buffers.add(id(getattr(m, "topk_indices_buffer", None)))
            if getattr(m, "skip_topk", False) and tag == "target":
                m._glm_idx_skip_ok = True
                skip += 1
            else:
                compute += 1
        if tag == "target" and skip and len(buffers) != 1:
            raise RuntimeError("glm-glue-lite: F3 skip and computing layers do not share one topk buffer")
        out[tag] = dict(skip_armed=skip, other=compute)
    STATS["idx_armed"].update(out)
    sys.stderr.write(f"glm-glue-lite: F3 armed {out}\n")
    if strict(env):
        expect = _expect(env, "GLM_GLUE_IDX_EXPECT")
        got = out.get("target", {}).get("skip_armed", 0)
        if (expect is None and got == 0) or (expect is not None and got != expect):
            raise RuntimeError(f"glm-glue-lite: F3 armed {got} skip layers, expected {expect}")
    return out


def install_idx_mla(mod_mla):
    cls = mod_mla.MultiHeadLatentAttentionWrapper
    if getattr(cls.forward, "_glm_glue", False):
        return
    original = cls.forward

    @functools.wraps(original)
    def forward(self, *args, **kw):
        skip = bool(getattr(self, "_glm_idx_skip_ok", False) and getattr(self, "is_sparse", False)
                    and getattr(self, "skip_topk", False))
        if not skip:
            _IDX.cache = None  # any computing/unarmed layer may rewrite buffers or metadata
        tick("mla_skip" if skip else "mla_other")
        prev, _IDX.reuse = _IDX.reuse, skip
        try:
            return original(self, *args, **kw)
        finally:
            _IDX.reuse = prev

    forward._glm_glue = True
    cls.forward = forward


def install_idx_convert(mod_fi):
    original = mod_fi.triton_convert_req_index_to_global_index
    if getattr(original, "_glm_glue", False):
        return

    @functools.wraps(original)
    def convert(req_id, block_table, token_indices, *args, **kw):
        on = not args and lever("GLM_GLUE_DSA_IDX_CACHE")
        key = _key(req_id, block_table, token_indices, kw) if on else None
        c = _IDX.cache
        if key is not None and _IDX.reuse and c is not None and c[0] == key:
            tick("idx_hit")
            if _CHECK[0] and not _capturing():
                fresh = original(req_id, block_table, token_indices, *args, **kw)
                a = fresh if isinstance(fresh, tuple) else (fresh,)
                b = c[1] if isinstance(c[1], tuple) else (c[1],)
                if len(a) != len(b) or not all(x.equal(y) for x, y in zip(a, b)):
                    raise RuntimeError("glm-glue-lite: DSA index cache hit differs from a fresh convert")
                tick("idx_checked")
            return c[1]
        out = original(req_id, block_table, token_indices, *args, **kw)
        tick("idx_miss_skip" if (on and _IDX.reuse) else "idx_convert")
        if key is not None and not _IDX.reuse:
            _IDX.cache = (key, out)
        return out

    convert._glm_glue = True
    mod_fi.triton_convert_req_index_to_global_index = convert


# --------------------------------------------------------------------------- registration
def _install_runner(env, w):
    def install(mod):
        cls = mod.GPUModelRunner
        if getattr(cls.load_model, "_glm_glue", False):
            return
        original = cls.load_model

        @functools.wraps(original)
        def load_model(self, *a, **kw):
            out = original(self, *a, **kw)
            spec = getattr(self, "speculator", None)
            models = {"target": self.model, "mtp": getattr(spec, "model", None)}
            if w["router"]:
                import vllm.envs as envs
                from vllm.platforms import current_platform
                gl = sys.modules["vllm.model_executor.layers.fused_moe.router.gate_linear"]
                gr = sys.modules["vllm.model_executor.layers.fused_moe.router.grouped_topk_router"]
                arm_router(models, gl, gr, envs, current_platform, env)
            if w["ws"]:
                ws_prepare(sys.modules["vllm.model_executor.layers.fused_moe.experts.marlin_moe"], self.device)
            if w["idx"]:
                arm_idx(models, sys.modules["vllm.model_executor.layers.mla"], env)
            STATS["load_model_armed"] += 1
            return out

        load_model._glm_glue = True
        cls.load_model = load_model
        STATS["runner_hooked"] = True
    return install


RUN = "vllm.v1.worker.gpu.model_runner"


def runner_installer(env=None):
    """GPUModelRunner hook for glm_ab.hooks() to compose into its RUN installer.

    FIX1 composed-startup finding: several later-registered bring-up finders resolve
    model_runner with importlib.machinery.PathFinder directly, which shadows any finder
    behind them, so a post-import hook registered mid-chain never ran. glm_ab.Finder is
    moved to the front at the end of the chain and always runs its mapping, so in the
    bank harness the runner hook goes through it.
    """
    return _pinned(RUN, _install_runner(env, wanted(env)), env)


def register(env=None):
    w = wanted(env)
    if not any(w.values()):
        return w
    if w["router"]:
        after_import("vllm.model_executor.layers.fused_moe.router.gate_linear",
                     _pinned("vllm.model_executor.layers.fused_moe.router.gate_linear", install_router_forward, env))
    if w["ws"]:
        after_import("vllm.model_executor.layers.fused_moe.experts.marlin_moe",
                     _pinned("vllm.model_executor.layers.fused_moe.experts.marlin_moe", install_moe_ws, env))
    if w["idx"]:
        after_import("vllm.model_executor.layers.mla", _pinned("vllm.model_executor.layers.mla", install_idx_mla, env))
        after_import("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90",
                     _pinned("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90", install_idx_convert, env))
    for key, names in PIN_ONLY.items():
        if w[key]:
            for name in names:
                after_import(name, _pinned(name, lambda mod: None, env))
    if _ab() is None:
        # Standalone: must be registered after any finder that resolves model_runner via
        # PathFinder (see runner_installer); the composed-startup test is the receipt.
        after_import(RUN, runner_installer(env))
    sys.stderr.write(f"glm-glue-lite: registered {w}\n")
    return w
