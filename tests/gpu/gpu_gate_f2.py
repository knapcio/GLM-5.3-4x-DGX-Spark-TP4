# SPDX-License-Identifier: Apache-2.0
"""Single-GPU op-level gate for F2 (persistent Marlin MoE lock workspace). Exit 0 only on PASS.

Calls the pinned image's fused_marlin_moe through the overlay wrapper (lever by env) with the
served per-rank MoE shapes: E=256, K=6144, N=512 (TP4), top8, BF16 activations,
  target: uint4b8 group 128 (Tech2wild Int4 routed experts),
  mtp:    uint8b128 channelwise (layer-78 W8 routed experts).
Random packed weights are valid Marlin inputs (any int32 bit pattern is a legal packed tile).
Checks, per width (target 1/3/4/6/12, MTP 1/3/4/12, prefill 64/512/2048/4096):
  A/A   stock (fresh torch.zeros) vs stock on identical inputs: the operator's own determinism;
  F2    persistent vs fresh: torch.equal on the output;
  locks workspace count_nonzero()==0 after every completed call and replay;
  addr  workspace data_ptr stable across all calls, replays and widths;
  graph decode widths: FULL CUDA graph captured with the persistent workspace (and a stock
        graph), 12 replays with in-place changed hidden/topk ids/weights at the same
        addresses, each equal to a fresh eager stock call on the same inputs.
Fails if the A/A control itself is not bit-exact (then F2 cannot be judged, never waived).
Needs ~4 GB GPU memory (one dtype set at a time), a few minutes. Fleet stopped.
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
E, K, N, TOPK = 256, 6144, 512, 8
TARGET_W, MTP_W, PREFILL_W = (1, 3, 4, 6, 12), (1, 3, 4, 12), (64, 512, 2048, 4096)


def weights(kind, dev, g):
    from vllm.scalar_type import scalar_types
    if kind == "target_int4_g128":
        qt, bits, group, scale = scalar_types.uint4b8, 4, 128, 0.01
    else:
        qt, bits, group, scale = scalar_types.uint8b128, 8, -1, 0.001
    pack = 32 // bits
    rnd = lambda *s: torch.randint(-2**31, 2**31 - 1, s, dtype=torch.int32, device=dev, generator=g)
    w1 = rnd(E, K // 16, 2 * N * 16 // pack)
    w2 = rnd(E, N // 16, K * 16 // pack)
    g1 = K // group if group > 0 else 1
    g2 = N // group if group > 0 else 1
    s1 = (torch.rand(E, g1, 2 * N, device=dev, generator=g) * scale + scale / 10).to(torch.bfloat16)
    s2 = (torch.rand(E, g2, K, device=dev, generator=g) * scale + scale / 10).to(torch.bfloat16)
    return dict(w1=w1, w2=w2, w1_scale=s1, w2_scale=s2, quant_type_id=qt.id)


def routing(M, dev, g):
    ids = torch.rand(M, E, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
    wts = torch.rand(M, TOPK, device=dev, generator=g)
    return ids, (wts / wts.sum(1, keepdim=True)).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    import glm_glue_lite as G
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
    G.install_moe_ws(mm)
    fn = mm.fused_marlin_moe
    rec = dict(gate="F2 persistent Marlin MoE workspace (FIX1)", shapes=dict(E=E, K=K, N=N, topk=TOPK),
               widths=dict(target=TARGET_W, mtp=MTP_W, prefill=PREFILL_W), wrapper=getattr(fn, "_glm_glue", False))
    if a.self_check:
        print(json.dumps(dict(rec, verdict="SELF_CHECK_ONLY")))
        return 0
    if not torch.cuda.is_available():
        print(json.dumps(dict(rec, verdict="FAIL", error="no CUDA device")))
        return 1
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev)
    g.manual_seed(20261001)
    ws = G.ws_prepare(mm, dev)
    ptr = ws.data_ptr()
    t0 = time.time()
    fails, aa_fails, lock_fails, cells, replays = [], [], [], 0, 0

    def call(lever, x, ids, wts, W):
        os.environ["GLM_GLUE_MOE_WS"] = "1" if lever else "0"
        return fn(hidden_states=x, w1=W["w1"], w2=W["w2"], bias1=None, bias2=None, w1_scale=W["w1_scale"],
                  w2_scale=W["w2_scale"], topk_weights=wts, topk_ids=ids, quant_type_id=W["quant_type_id"],
                  global_num_experts=E)

    def locks_clear(tag):
        torch.cuda.synchronize()
        if ws.data_ptr() != ptr or int(torch.count_nonzero(ws).item()) != 0:
            lock_fails.append(dict(tag=tag, ptr_stable=ws.data_ptr() == ptr,
                                   nonzero=int(torch.count_nonzero(ws).item())))

    for kind, widths in (("target_int4_g128", TARGET_W + PREFILL_W), ("mtp_int8_channel", MTP_W + PREFILL_W[:2])):
        W = weights(kind, dev, g)
        for M in widths:
            for r in range(a.repeats if M <= 64 else 2):
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
                ids, wts = routing(M, dev, g)
                s1 = call(False, x, ids, wts, W).clone()
                s2 = call(False, x, ids, wts, W).clone()
                pz = call(True, x, ids, wts, W).clone()
                locks_clear(f"{kind}/M{M}/r{r}")
                cells += 1
                if not torch.equal(s1, s2):
                    aa_fails.append(dict(kind=kind, M=M, r=r))
                if not torch.equal(s1, pz):
                    fails.append(dict(kind=kind, M=M, r=r, max_abs=float((s1.float() - pz.float()).abs().max())))
                if not torch.isfinite(s1.float()).all():
                    fails.append(dict(kind=kind, M=M, r=r, error="non-finite test output"))
        # FULL graphs at decode widths, in-place changed inputs at the same addresses.
        for M in (TARGET_W if kind.startswith("target") else MTP_W):
            x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
            ids = torch.zeros(M, TOPK, device=dev, dtype=torch.int32)
            wts = torch.zeros(M, TOPK, device=dev, dtype=torch.float32)
            x.copy_(torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g))
            i0, w0 = routing(M, dev, g)
            ids.copy_(i0)
            wts.copy_(w0)
            graphs = {}
            for lever in (True, False):
                call(lever, x, ids, wts, W)  # eager warm-up on the side stream, like the runner
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                os.environ["GLM_GLUE_MOE_WS"] = "1" if lever else "0"
                with torch.cuda.graph(graph):
                    out = call(lever, x, ids, wts, W)
                graphs[lever] = (graph, out)
            locks_clear(f"{kind}/M{M}/capture")
            for rep in range(12):
                x.copy_(torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g))
                i1, w1_ = routing(M, dev, g)
                ids.copy_(i1)
                wts.copy_(w1_)
                graphs[True][0].replay()
                locks_clear(f"{kind}/M{M}/replay{rep}")
                graphs[False][0].replay()
                torch.cuda.synchronize()
                fresh = call(False, x, ids, wts, W)
                replays += 1
                if not torch.equal(graphs[False][1], fresh):
                    aa_fails.append(dict(kind=kind, M=M, rep=rep, path="stock-graph vs stock-eager"))
                if not torch.equal(graphs[True][1], fresh):
                    fails.append(dict(kind=kind, M=M, rep=rep, path="persistent-graph vs stock-eager"))
            del graphs
        del W
        torch.cuda.empty_cache()
    os.environ.pop("GLM_GLUE_MOE_WS", None)
    counts = G.STATS["counts"]
    verdict = "PASS" if not (fails or aa_fails or lock_fails) else ("FAIL_AA_CONTROL" if aa_fails and not fails else "FAIL")
    rec.update(cells=cells, replays=replays, f2_mismatches=len(fails), aa_mismatches=len(aa_fails),
               lock_or_address_failures=len(lock_fails), first=dict(f2=fails[:5], aa=aa_fails[:5], locks=lock_fails[:5]),
               workspace=dict(numel=ws.numel(), dtype=str(ws.dtype), ptr_stable=ws.data_ptr() == ptr),
               wrapper_counts=counts, streams=G.STATS["ws_streams"], elapsed_s=round(time.time() - t0, 1),
               device=torch.cuda.get_device_name(), torch=torch.__version__, verdict=verdict)
    text = json.dumps(rec)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
