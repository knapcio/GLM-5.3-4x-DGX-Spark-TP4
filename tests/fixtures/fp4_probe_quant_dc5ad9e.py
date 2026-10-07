# SPDX-License-Identifier: Apache-2.0
"""Original FP4 reference, shared by the probe and offline evaluator.

No custom kernels. No host tensor reads or data-dependent Python branches.
The format is the concrete candidate in the FP4 study, not a recovered Mia ABI.
"""


def fp4_parts(x):
    import torch
    if x.shape[-1] != 512:
        raise ValueError('fp4-probe requires D512')
    z = x.float().reshape(*x.shape[:-1], 32, 16)
    a = z.abs().amax(dim=(-2, -1), keepdim=True)
    # BF16 inputs cannot underflow this FP32 row scale. Zero rows use 1.
    row = torch.exp2(torch.ceil(torch.log2(torch.where(a > 0, a / 2688., torch.ones_like(a)))))
    block = (z.abs().amax(dim=-1, keepdim=True) / (6 * row)).clamp(0, 448)
    block = block.to(torch.float8_e4m3fn).float()
    scale = block * row
    v = z.abs() / torch.where(scale > 0, scale, torch.ones_like(scale))
    # Ties use the even E2M1 code, including ties at .25, 1.25, 2.5, 5.
    code = torch.zeros_like(v)
    for mid, upper, tie_up in ((.25, .5, False), (.75, 1., True),
                              (1.25, 1.5, False), (1.75, 2., True),
                              (2.5, 3., False), (3.5, 4., True), (5., 6., False)):
        code = torch.where(v >= mid if tie_up else v > mid, upper, code)
    code = torch.copysign(code, z)
    return code, block, row


def fp4_qdq(x):
    import torch
    code, block, row = fp4_parts(x)
    y = (code * block * row).reshape_as(x).to(x.dtype)
    # Preserve nonfinite inputs, rather than quietly turning a bad row into zero.
    return torch.where(torch.isfinite(x), y, x)


def fp8_codes(x, k_scale):
    import torch
    s = torch.as_tensor(k_scale, dtype=torch.float32, device=x.device)
    return (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()


def fp8_qdq(x, k_scale):
    import torch
    s = torch.as_tensor(k_scale, dtype=torch.float32, device=x.device)
    return fp8_codes(x, s) * s
