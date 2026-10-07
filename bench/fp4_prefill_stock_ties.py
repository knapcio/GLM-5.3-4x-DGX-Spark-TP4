#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check raw stock selector self-parity before attributing cutoff-tie drift.

GPU-only diagnostic, no model or network. The qualification fixture remains
strict and continues to compare directly against the raw stock selector.
"""
import argparse
import json
import torch
from vllm import _custom_ops as ops


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeat',type=int,default=10)
    a=p.parse_args()
    if a.repeat<2:p.error('at least two identical stock calls required')
    rows,cols=67,12289
    dense=torch.zeros(rows,cols,device='cuda')
    ks=torch.tensor([0]*33+[6145]*34,device='cuda',dtype=torch.int32)
    ke=torch.tensor([9]+[6145]*32+[6150]+[cols]*33,device='cuda',dtype=torch.int32)
    out=torch.empty(rows,2048,device='cuda',dtype=torch.int32)
    baseline=None;different=[]
    for _ in range(a.repeat):
        ops.top_k_per_row_prefill(dense,ks,ke,out,rows,dense.stride(0),dense.stride(1),2048)
        current=out.sort(dim=1).values
        if baseline is None:baseline=current.clone()
        different.append(int((current!=baseline).any(dim=1).sum()))
    print(json.dumps(dict(scope='DIAGNOSTIC_RAW_STOCK_SELF_PARITY',rows=rows,columns=cols,
                          repeat=a.repeat,differing_candidate_set_rows=different,
                          self_parity=not any(different),device=torch.cuda.get_device_name()),indent=2))
