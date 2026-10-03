# SPDX-License-Identifier: Apache-2.0
"""One launch/layer AFTER final top-k and BEFORE expert alignment/Marlin."""
import torch


@torch.library.custom_op('glm_deadrow::remap_', mutates_args=('weights', 'ids'))
def remap_(weights: torch.Tensor, ids: torch.Tensor, src: torch.Tensor) -> None:
    if not ids.is_cuda:
        # Read BOTH snapshots before writing. Live anchors are never mutated.
        w = weights.index_select(0, src)
        i = ids.index_select(0, src)
        weights.copy_(w)
        ids.copy_(i)
        return
    from vllm.triton_utils import triton, tl
    remap_kernel[(triton.cdiv(ids.shape[0],128),)](
        weights, ids, src, ids.shape[0], weights.stride(0), weights.stride(1),
        ids.stride(0), ids.stride(1), K=ids.shape[1], BLOCK=128)


@remap_.register_fake
def _(weights, ids, src):
    return None


try:
    from vllm.triton_utils import triton, tl
except ImportError:
    pass  # CPU custom-op tests don't require vLLM or Triton.
else:
    @triton.jit
    def remap_kernel(W, I, SRC, M, SW0, SW1, SI0, SI1,
                     K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
        source = tl.load(SRC+row, row<M, 0)
        dead = (row<M) & (source != row)
        for j in tl.static_range(K):
            w = tl.load(W+source*SW0+j*SW1, dead, 0.)
            i = tl.load(I+source*SI0+j*SI1, dead, 0)
            tl.store(W+row*SW0+j*SW1, w, dead)
            tl.store(I+row*SI0+j*SI1, i, dead)
