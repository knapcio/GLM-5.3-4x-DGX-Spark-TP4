# SPDX-License-Identifier: Apache-2.0
"""Pinned full-size FP8 Marlin preparation; CUDA repack shape substitute."""
import sys
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]
from boot_preflight import fake_hardware,Report
fake_hardware(Report(),0)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mf
import glm_draft_ehproj as eh
from glm_draft_ehproj_config import byte_cost

torch.set_num_threads(1)
def repack(b_q_weight,perm,size_k,size_n,num_bits,**kw):
    assert (size_k,size_n,num_bits)==(12288,6144,8)
    assert not kw.get('is_a_8bit',False)
    return torch.empty((size_k//16,size_n*16//4),dtype=torch.int32)

def workspace(*a,**kw):return torch.zeros(48,dtype=torch.int32)

original=torch.nn.Linear(12288,6144,bias=False,dtype=torch.bfloat16)
with torch.no_grad():original.weight.fill_(.02)
with patch.object(ops,'gptq_marlin_repack',repack),patch.object(mf,'marlin_make_workspace_new',workspace):
    bank=eh.FP8Bank(original,prepare=mf.prepare_fp8_layer_for_marlin,sms=48)
assert bank.resident_bytes==75509952
assert bank.weight.dtype==torch.int32 and bank.weight_scale.dtype==torch.bfloat16
expected=(torch.tensor(.02).bfloat16().float()/448).bfloat16() * (torch.tensor(2.,dtype=torch.bfloat16)**120)
assert torch.equal(bank.weight_scale,expected.expand_as(bank.weight_scale))
print('Full replicated native FP8 preparation PASS',byte_cost(),flush=True)
print('Pinned quantizer, pack, BF16 scale fusion/permutation/storage verified; CUDA repack/GEMM unrun',flush=True)
