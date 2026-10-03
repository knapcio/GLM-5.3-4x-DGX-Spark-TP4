# SPDX-License-Identifier: Apache-2.0
"""Compile actual salted kernels for sm_121 without a GPU/driver."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'overlay/bringup'))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from glm_spec_sample_kernel import _resample_kernel

POINTERS = {
    'resampled_local_argmax_ptr': '*i64', 'resampled_local_max_ptr': '*fp32',
    'target_logits_ptr': '*fp32', 'target_rejected_logsumexp_ptr': '*fp32',
    'draft_logits_ptr': '*bf16', 'draft_rejected_logsumexp_ptr': '*fp32',
    'rejected_step_ptr': '*i32', 'cu_num_logits_ptr': '*i32',
    'expanded_idx_mapping_ptr': '*i32', 'draft_sampled_ptr': '*i64',
    'temp_ptr': '*fp32', 'seed_ptr': '*i64', 'pos_ptr': '*i64',
    'cumulative_log_p_ptr': '*fp32',
}
for has_draft in (False, True):
    for fp64 in (False, True):
        constants = dict(BLOCK_SIZE=1024, HAS_DRAFT_LOGITS=has_draft,
                         USE_FP64=fp64, USE_BLOCK_VERIFICATION=False)
        signature = {n: POINTERS.get(n, 'i32' if n == 'vocab_size' else 'i64')
                     for n in _resample_kernel.arg_names if n not in constants}
        src = ASTSource(_resample_kernel, signature, constexprs=constants)
        ptx = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                             options=dict(num_warps=4, num_stages=1)).asm['ptx']
        assert '.entry' in ptx
        print(f'COMPILE PASS sm_121 draft={has_draft} fp64={fp64}', flush=True)
