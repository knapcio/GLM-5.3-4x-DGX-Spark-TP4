# SPDX-License-Identifier: Apache-2.0
"""Real pinned vLLM head schemes and repack accounting on CPU tensors.

SM121 discovery and the CUDA repack op are explicit shape-only substitutes.
Quantizer values, parameter classes, permutations, scales, and byte debit run.
"""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import argparse
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]
from test_draft_head import CPUHead
import glm_draft_head as d
from boot_preflight import fake_hardware,Report

report=Report();platform=fake_hardware(report,0)
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
from vllm.platforms.interface import PlatformEnum
from vllm.config import VllmConfig,set_current_vllm_config
from vllm import _custom_ops as ops
ops.cutlass_scaled_mm_supports_fp8=lambda *a:False
ops.cutlass_scaled_mm_supports_fp4=lambda *a:False
ops.cutlass_group_gemm_supported=lambda *a:False
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mf
type(platform)._enum=PlatformEnum.CUDA

calls=[]
def repack(b_q_weight,perm,size_k,size_n,num_bits,**kw):
    calls.append(dict(k=size_k,n=size_n,bits=num_bits,activation8=kw.get('is_a_8bit',False)))
    return torch.empty((size_k//16,size_n*16//(32//num_bits)),dtype=torch.int32)
def workspace(device,*a,**kw):return torch.zeros(48,dtype=torch.int32)

with patch.object(torch.Tensor,'is_cuda',property(lambda t:True)),\
     patch.object(torch.cuda,'is_available',lambda:True),\
     patch.object(torch.cuda,'current_device',lambda:0),\
     patch.object(torch.cuda,'is_current_stream_capturing',lambda:False),\
     patch.object(torch.cuda,'get_device_properties',lambda *a:NS(multi_processor_count=48)),\
     patch.object(ops,'gptq_marlin_repack',repack),\
     patch.object(mu,'marlin_make_workspace_new',workspace),\
     patch.object(mf,'marlin_make_workspace_new',workspace),\
     set_current_vllm_config(VllmConfig()):
    torch.manual_seed(77)
    n,k=(38720,6144) if args.full else (64,256)
    head=CPUHead(torch.randn(n,k,dtype=torch.bfloat16))
    for kind in ('nvfp4','int8'):
        bank=d.Bank(head,kind)
        expected=d.byte_cost(kind,n,k)['total']
        assert bank.resident_bytes==expected,(kind,bank.resident_bytes,expected)
        assert calls[-1]['activation8'] is False
        print(kind,bank.resident_bytes,type(bank.scheme.kernel).__name__,calls[-1],flush=True)
print('NATIVE PREPARATION PASS; CUDA repack values and replay unverified',flush=True)
