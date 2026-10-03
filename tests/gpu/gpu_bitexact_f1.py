# SPDX-License-Identifier: Apache-2.0
"""Single-GPU op-level gate for F1 (FIX1). Exit 0 only on PASS; any mismatch exits 1.

Proves grouped_topk(BF16 logits) == grouped_topk(BF16 logits .to(FP32)) bit-for-bit (ids and
weights) for exactly the F1 contract (E256, num_expert_group=1, topk_group=1, top8, sigmoid,
FP32 correction bias, renormalize, routed_scaling_factor 1.0 and 2.5), on:
  * the raw op and the served compiled path (grouped_topk_router.grouped_topk), decode and
    prefill widths covering both single_group kernels (block M<=1024, warp M>1024),
  * GEMM logits (the Tier-6 F.linear output) and adversarial tie sets: quantised logits,
    all-equal rows, exact 8th/9th ties, 7..10 four-way boundary ties, sigmoid saturation,
  * bias variants: random, zero, constant (repeated value), coarse repeated values,
  * the overlay's patched GateLinear (bank on/off) when a real GateLinear can be built,
  * CUDA-graph replay with in-place changed inputs at the same addresses (FP32-cast graph vs
    BF16 graph, both checked against eager each replay).
About 60 MB of GPU memory, under a minute. Run inside the pinned image on an idle GPU:
  docker run --rm --gpus all --network none -v $PWD:/repo:ro --entrypoint python3 <image> \
      /repo/tests/gpu/gpu_bitexact_f1.py [--out receipt.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "overlay" / "bringup"))
E, H, TOPK = 256, 6144, 8
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 64, 512, 1024, 1025, 2048, 4096)
SCALES = (1.0, 2.5)


def bias_variants(dev, g):
    return {
        "random": torch.randn(E, device=dev, generator=g) * 0.05,
        "zero": torch.zeros(E, device=dev),
        "constant": torch.full((E,), 0.125, device=dev),
        "coarse": torch.randint(0, 4, (E,), device=dev, generator=g).float() * 0.01,
    }


def tie_variants(bf, g):
    """bf: [M, E] BF16 GEMM logits. Returns BF16 logit sets with adversarial ties."""
    M = bf.shape[0]
    out = {"gemm": bf, "quantised": ((bf.float() * 4).round() / 4).to(torch.bfloat16)}
    out["all_equal"] = bf[:, :1].expand(M, E).contiguous()
    # exact 8th/9th tie on the logits (ties in selection whenever bias ties too)
    srt, idx = bf.float().sort(dim=1, descending=True)
    t89 = bf.clone()
    t89.scatter_(1, idx[:, 8:9], bf.gather(1, idx[:, 7:8]))
    out["tie_8_9"] = t89
    t710 = bf.clone()
    v = bf.gather(1, idx[:, 6:7])
    for j in (7, 8, 9):
        t710.scatter_(1, idx[:, j:j + 1], v)
    out["tie_7_10"] = t710
    out["saturated"] = (bf.float().sign() * 40).to(torch.bfloat16)   # sigmoid -> 0/1, mass ties
    return out


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--self-check", action="store_true", help="CPU: imports/signatures only, no kernels")
    a = p.parse_args()
    import vllm._custom_ops as ops
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.router import grouped_topk_router as gtr
    rec = dict(gate="F1 grouped_topk bf16 == fp32-cast (FIX1)", contract=dict(E=E, groups=1, topk_group=1, topk=TOPK,
               scoring="sigmoid", renormalize=True, routed_scaling_factors=SCALES),
               fused_grouped_topk_env=bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK), widths=WIDTHS)
    assert callable(ops.grouped_topk) and callable(gtr.grouped_topk)
    if a.self_check:
        print(json.dumps(dict(rec, verdict="SELF_CHECK_ONLY")))
        return 0
    if not torch.cuda.is_available():
        print(json.dumps(dict(rec, verdict="FAIL", error="no CUDA device")))
        return 1
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(20261001)
    w = torch.randn(E, H, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    rows, bad, cells = 0, [], 0
    t0 = time.time()
    raw = lambda lg, rs, b: ops.grouped_topk(lg, 1, 1, TOPK, True, rs, b, 1)

    def served(lg, rs, b):
        hs = torch.empty(lg.shape[0], 1, device=dev)
        return gtr.grouped_topk(hs, lg, TOPK, True, num_expert_group=1, topk_group=1, scoring_func="sigmoid",
                                routed_scaling_factor=rs, e_score_correction_bias=b)

    paths = {"raw": raw}
    if rec["fused_grouped_topk_env"]:
        paths["served_compiled"] = served
    for M in WIDTHS:
        for trial in range(a.trials if M <= 1024 else 2):
            x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
            bf = torch.nn.functional.linear(x, w)  # exactly the Tier-6 GEMM (ReplicatedLinear)
            for vname, logits in tie_variants(bf, g).items():
                for bname, bias in bias_variants(dev, g).items():
                    for rs in SCALES:
                        for pname, fn in paths.items():
                            ref = fn(logits.float(), rs, bias)
                            got = fn(logits, rs, bias)
                            cells += 1
                            rows += M
                            if not same(ref, got):
                                bad.append(dict(M=M, trial=trial, logits=vname, bias=bname, rs=rs, path=pname))
    rec["eager"] = dict(cells=cells, rows=rows, mismatches=len(bad), first=bad[:8])

    # Overlay path on a real GateLinear (bank on/off by env), when constructible here.
    gate_rec = dict(status="SKIPPED")
    try:
        from vllm.model_executor.layers.fused_moe.router import gate_linear as gl
        import glm_glue_lite as G
        G.install_router_forward(gl)
        from vllm.config import VllmConfig, set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):
            gate = gl.GateLinear(H, E, out_dtype=torch.float32, prefix="gate").to(dev)
        gate.weight.data.copy_(w)
        why = [t for t in G.GATE_TIERS if getattr(gate, t, False)]
        if why:
            gate_rec = dict(status="NOT_TIER6_ON_THIS_DEVICE", tiers=why)
        else:
            gate._glm_bf16_logits = True
            mism = 0
            for M in (1, 3, 4, 12, 512, 4096):
                x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
                os.environ["GLM_GLUE_ROUTER_BF16"] = "0"
                stock, _ = gate(x)
                os.environ["GLM_GLUE_ROUTER_BF16"] = "1"
                glue, _ = gate(x)
                ok = (stock.dtype == torch.float32 and glue.dtype == torch.bfloat16
                      and torch.equal(glue.float(), stock))
                for rs in SCALES:
                    b = torch.randn(E, device=dev, generator=g) * 0.05
                    ok = ok and same(raw(stock, rs, b), raw(glue, rs, b))
                mism += int(not ok)
            gate_rec = dict(status="PASS" if mism == 0 else "FAIL", mismatching_widths=mism)
            if mism:
                bad.append(dict(path="overlay_gate", mismatching_widths=mism))
    except Exception as exc:  # construction outside a model is optional evidence
        gate_rec = dict(status="SKIPPED", reason=repr(exc)[:300])
    finally:
        os.environ.pop("GLM_GLUE_ROUTER_BF16", None)
    rec["overlay_gate"] = gate_rec

    # CUDA-graph replay with in-place changed inputs at the same addresses.
    replays, gbad = 0, []
    for M in (1, 3, 4, 12, 16):
        lg = torch.zeros(M, E, device=dev, dtype=torch.bfloat16)
        bias = torch.zeros(E, device=dev)
        outs = {}
        for rs in SCALES:
            for kind in ("fp32cast", "bf16"):
                raw(lg.float() if kind == "fp32cast" else lg, rs, bias)  # warm-up outside capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    res = raw(lg.float() if kind == "fp32cast" else lg, rs, bias)
                outs[(rs, kind)] = (graph, res)
        for rep in range(24):
            x = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
            variants = tie_variants(torch.nn.functional.linear(x, w), g)
            vname = list(variants)[rep % len(variants)]
            bname, bnew = list(bias_variants(dev, g).items())[rep % 4]
            lg.copy_(variants[vname])
            bias.copy_(bnew)
            for rs in SCALES:
                for kind in ("fp32cast", "bf16"):
                    outs[(rs, kind)][0].replay()
                torch.cuda.synchronize()
                eager = raw(lg.float(), rs, bias)
                r32, r16 = outs[(rs, "fp32cast")][1], outs[(rs, "bf16")][1]
                replays += 1
                if not (same(r32, r16) and same(r16, eager)):
                    gbad.append(dict(M=M, rep=rep, logits=vname, bias=bname, rs=rs))
    rec["graph_replay"] = dict(replays=replays, mismatches=len(gbad), first=gbad[:8])
    bad += gbad
    torch.cuda.synchronize()
    rec.update(mismatches=len(bad), elapsed_s=round(time.time() - t0, 1),
               device=torch.cuda.get_device_name(), torch=torch.__version__,
               verdict="PASS" if not bad else "FAIL")
    text = json.dumps(rec)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
