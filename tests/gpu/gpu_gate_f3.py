# SPDX-License-Identifier: Apache-2.0
"""Single-GPU gate for F3 (DSA index-conversion reuse on skip layers). Exit 0 only on PASS.

Uses the pinned image's real Triton triton_convert_req_index_to_global_index (sparse_utils)
through the overlay's patched convert and MLA-wrapper hooks, with the served GLM skip pattern
(78 target layers, computing layers 0, 1, 5, 9, ..., 77 = 21; 57 armed skip layers) and one
unarmed MTP layer that shares the same metadata buffers.

Each layer = stand-in MLA wrapper: computing layers copy their indexer result into the
shared topk buffer (indexer write), every layer converts and copies (slots, counts) into
its own output buffers (the attention read). Widths 1/3/4/12 (target M3/M12, MTP
M1/M3/M4/M12). Per width a glue graph and a stock graph are captured, then replayed with
IN-PLACE changes at the same addresses: request ids per token, block-table contents, topk
indices with -1 padding / short valid lengths / rows beyond t, and target -> MTP -> target
transitions (MTP graph mutates the same metadata between target replays). After every
replay each layer's slots/counts must equal a fresh eager stock convert of the inputs that
layer saw. Also checks the eager check-mode path and that skip layers launched no convert.
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "overlay" / "bringup"))
LAYERS, TOPK, BLOCK, REQS, MAXBLK = 78, 2048, 64, 4, 512
WIDTHS = (1, 3, 4, 12)


def computing(layer):  # deepseek_v2.py:1093 with offset 2, freq 4
    return max(layer - 2 + 1, 0) % 4 == 0


class Layer(torch.nn.Module):
    """Stand-in MultiHeadLatentAttentionWrapper (the overlay patches this class's forward)."""
    def __init__(self, fi, buf, md, src, skip, out_slots, out_counts):
        super().__init__()
        self.fi, self.md, self.src = fi, md, src
        self.is_sparse, self.skip_topk = True, skip
        self.topk_indices_buffer = buf
        self.out_slots, self.out_counts = out_slots, out_counts

    def forward(self, t):
        buf = self.topk_indices_buffer
        if not self.skip_topk:
            buf[:t].copy_(self.src[:t])          # the indexer rewrites the shared buffer
        slots, counts = self.fi.triton_convert_req_index_to_global_index(
            self.md["req"][:t], self.md["bt"], buf[:t], BLOCK_SIZE=BLOCK, NUM_TOPK_TOKENS=buf.shape[1],
            return_valid_counts=True)
        self.out_slots[:t].copy_(slots)
        self.out_counts[:t].copy_(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--replays", type=int, default=16)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    import glm_glue_lite as G
    from vllm.v1.attention.backends.mla import sparse_utils as su
    real = su.triton_convert_req_index_to_global_index
    rec = dict(gate="F3 DSA index reuse, real Triton convert, captured replay with in-place changes (FIX1)",
               layers=LAYERS, computing=sum(computing(i) for i in range(LAYERS)), widths=WIDTHS)
    if a.self_check:
        print(json.dumps(dict(rec, verdict="SELF_CHECK_ONLY", convert=real.__name__)))
        return 0
    if not torch.cuda.is_available():
        print(json.dumps(dict(rec, verdict="FAIL", error="no CUDA device")))
        return 1
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(20261001)
    fi = types.SimpleNamespace(triton_convert_req_index_to_global_index=real)
    G.install_idx_convert(fi)
    mla_mod = types.SimpleNamespace(MultiHeadLatentAttentionWrapper=Layer)
    G.install_idx_mla(mla_mod)
    T = max(WIDTHS)
    buf = torch.full((T + 4, TOPK), -1, dtype=torch.int32, device=dev)          # shared target buffer
    mtp_buf = torch.full((T + 4, TOPK), -1, dtype=torch.int32, device=dev)
    md = dict(req=torch.zeros(T + 4, dtype=torch.int32, device=dev),
              bt=torch.zeros(REQS, MAXBLK, dtype=torch.int32, device=dev))
    srcs = [torch.full((T + 4, TOPK), -1, dtype=torch.int32, device=dev) for _ in range(LAYERS + 1)]
    outs = [(torch.zeros(T + 4, TOPK, dtype=torch.int32, device=dev), torch.zeros(T + 4, dtype=torch.int32, device=dev))
            for _ in range(LAYERS + 1)]
    target = torch.nn.ModuleList([Layer(fi, buf, md, srcs[i], not computing(i), *outs[i]) for i in range(LAYERS)])
    mtp = torch.nn.ModuleList([Layer(fi, mtp_buf, md, srcs[LAYERS], False, *outs[LAYERS])])
    G.arm_idx({"target": target, "mtp": mtp}, mla_mod, env={"GLM_GLUE_STRICT": "1", "GLM_GLUE_IDX_EXPECT": "57"})

    def mutate(t, mode):
        """In-place changes at the same addresses."""
        md["req"].copy_(torch.randint(0, REQS, md["req"].shape, device=dev, generator=g, dtype=torch.int32))
        md["bt"].copy_(torch.randint(0, 1 << 20, md["bt"].shape, device=dev, generator=g, dtype=torch.int32))
        ctx = int(torch.randint(1, MAXBLK * BLOCK, (1,), generator=g, device=dev).item())
        for s in srcs:
            v = torch.randint(0, ctx, s.shape, device=dev, generator=g, dtype=torch.int32)
            if mode % 3 == 0:     # -1 padding tail
                keep = int(torch.randint(1, TOPK, (1,), device=dev, generator=g).item())
                v[:, keep:] = -1
            elif mode % 3 == 1:   # random holes
                v[torch.rand(v.shape, device=dev, generator=g) < 0.3] = -1
            s.copy_(v)
        buf[t:].fill_(-1)          # rows beyond t keep padding

    def reference(t, layers, buffer):
        """Fresh eager stock converts on the inputs each layer saw."""
        exp, cur = [], buffer[:t].clone()
        for L in layers:
            if not L.skip_topk:
                cur = L.src[:t].clone()
            exp.append(real(md["req"][:t], md["bt"], cur, BLOCK_SIZE=BLOCK, NUM_TOPK_TOKENS=TOPK,
                            return_valid_counts=True))
        return exp

    def compare(t, layers, exp, tag, bad):
        for i, (L, (s, c)) in enumerate(zip(layers, exp)):
            if not (torch.equal(L.out_slots[:t], s) and torch.equal(L.out_counts[:t], c)):
                bad.append(dict(tag=tag, layer=i, t=t))
                return

    def fwd(layers, t):
        for L in layers:
            L(t)

    import os
    bad, replays = [], 0
    t0 = time.time()
    # Eager check mode: every hit recomputes and must match (the in-boot warm-up path).
    G._CHECK[0] = True
    for t in WIDTHS:
        mutate(t, t)
        os.environ["GLM_GLUE_DSA_IDX_CACHE"] = "1"
        fwd(target, t)
        torch.cuda.synchronize()
        compare(t, target, reference(t, target, buf), f"eager-check/t{t}", bad)
    eager_counts = dict(G.STATS["counts"].get("-1|eager", {}))
    G._CHECK[0] = False
    graphs = {}
    for t in WIDTHS:
        for lever in ("1", "0"):
            os.environ["GLM_GLUE_DSA_IDX_CACHE"] = lever
            mutate(t, 0)
            fwd(target, t)          # warm-up (eager), as CudaGraphManager.capture does
            torch.cuda.synchronize()
            before = dict(G.STATS["counts"].get("-1|capture", {}))
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fwd(target, t)
            after = dict(G.STATS["counts"].get("-1|capture", {}))
            delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
            graphs[(t, lever)] = gr
            want = dict(idx_hit=57, idx_convert=21) if lever == "1" else dict(idx_convert=78)
            got = {k: delta.get(k, 0) for k in ("idx_hit", "idx_convert", "idx_miss_skip")}
            if any(got.get(k, 0) != v for k, v in want.items()) or got["idx_miss_skip"]:
                bad.append(dict(tag=f"capture-receipt/t{t}/lever{lever}", got=got, want=want))
        os.environ["GLM_GLUE_DSA_IDX_CACHE"] = "1"
        gm = torch.cuda.CUDAGraph()
        fwd(mtp, t)
        torch.cuda.synchronize()
        with torch.cuda.graph(gm):
            fwd(mtp, t)
        graphs[(t, "mtp")] = gm
    for t in WIDTHS:
        for rep in range(a.replays):
            for lever in ("1", "0"):
                mutate(t, rep)
                graphs[(t, lever)].replay()
                torch.cuda.synchronize()
                compare(t, target, reference(t, target, buf), f"replay/t{t}/lever{lever}/rep{rep}", bad)
                replays += 1
            # target -> MTP -> target transition with in-place metadata changes in between
            mutate(t, rep + 1)
            graphs[(t, "mtp")].replay()
            torch.cuda.synchronize()
            compare(t, mtp, reference(t, mtp, mtp_buf), f"mtp/t{t}/rep{rep}", bad)
            mutate(t, rep + 2)
            graphs[(t, "1")].replay()
            torch.cuda.synchronize()
            compare(t, target, reference(t, target, buf), f"after-mtp/t{t}/rep{rep}", bad)
            replays += 2
    os.environ.pop("GLM_GLUE_DSA_IDX_CACHE", None)
    rec.update(replays=replays, mismatches=len(bad), first=bad[:8], eager_counts=eager_counts,
               capture_counts=G.STATS["counts"].get("-1|capture", {}), armed=G.STATS["idx_armed"],
               elapsed_s=round(time.time() - t0, 1), device=torch.cuda.get_device_name(), torch=torch.__version__,
               verdict="PASS" if not bad else "FAIL")
    text = json.dumps(rec)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
