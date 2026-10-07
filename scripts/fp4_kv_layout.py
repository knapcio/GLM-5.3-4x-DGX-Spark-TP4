# SPDX-License-Identifier: Apache-2.0
"""Release cache byte ledger. Semantic dimensions remain D512/R64."""
FORMATS = {'fp8': dict(mla=576, index=132, abi='fp8_e4m3'),
           'fp4x': dict(mla=368, index=132, abi='fp4x_v1_e2m1_s16_e4m3_pow2_r64')}
ORDINARY_BYTES = 1 << 30
CARVE_BYTES = 2145386496
BLOCK_TOKENS = 64


def sizes(mode, blocks):
    f = FORMATS[mode]
    return [blocks * BLOCK_TOKENS * f['mla']] * 79 + [blocks * BLOCK_TOKENS * f['index']] * 22


def aligned_bytes(mode, blocks):
    return sum((s + 511) // 512 * 512 for s in sizes(mode, blocks))


def layout(mode, ordinary_bytes=ORDINARY_BYTES):
    if mode not in FORMATS:
        raise ValueError('GLM_KV_FORMAT must be fp8 or fp4x')
    if type(ordinary_bytes) is not int or ordinary_bytes < ORDINARY_BYTES or ordinary_bytes % (2 << 20):
        raise ValueError('ordinary KV head must be >=1 GiB and aligned to 2 MiB')
    f = FORMATS[mode]
    block_bytes = BLOCK_TOKENS * (79 * f['mla'] + 22 * f['index'])
    blocks = (ordinary_bytes + CARVE_BYTES) // block_bytes
    while aligned_bytes(mode, blocks) > ordinary_bytes + CARVE_BYTES:
        blocks -= 1
    return dict(format=mode, abi=f['abi'], mla_row_bytes=f['mla'], index_row_bytes=f['index'],
                block_bytes=block_bytes, kv_bytes=ordinary_bytes, blocks=blocks,
                tokens=blocks * BLOCK_TOKENS, aligned_bytes=aligned_bytes(mode, blocks),
                max_model_len=str((blocks - 6) * BLOCK_TOKENS), requires_no_ll128=False,
                status='FP4x staged; exact-layout sim >=8.5 GiB and fleet qualification required'
                       if mode == 'fp4x' else 'release FP8 dispram layout')
