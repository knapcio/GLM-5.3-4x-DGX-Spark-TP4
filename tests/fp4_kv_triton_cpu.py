# SPDX-License-Identifier: Apache-2.0
"""Actual Triton interpreter read/write equality. Run only in pinned image.

No GPU. This script deliberately fails if interpreter/FP8 support is missing;
there is no skip or CPU-model substitution in the pinned qualification path.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

if os.environ.get('TRITON_INTERPRET')!='1':
    raise RuntimeError('TRITON_INTERPRET=1 required')
if torch.cuda.is_available():
    raise RuntimeError('CPU only')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/bringup'))
from glm_fp4_kv_kernel import _pack_store,_read_rows,_qdq,_gather
p=ROOT/'tests/fixtures/fp4_probe_quant_dc5ad9e.py'
sp=importlib.util.spec_from_file_location('reference',p)
R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)


def same_values(a,b):
    """Bit-equal after mapping -0 to +0: the kernel stores an unsigned zero code
    where the reference keeps the sign of a value that rounds to zero; both
    contribute identically to every QK/PV product."""
    a=a.float()+0.0; b=b.float()+0.0
    return torch.equal(a.view(torch.int32),b.view(torch.int32))


def compare(x,rope,scale):
    n=x.shape[0]
    cache=torch.full((n+4,368),165,dtype=torch.uint8)
    slots=torch.arange(n,dtype=torch.int64)+2
    k=torch.tensor([scale],dtype=torch.float32)
    _pack_store[(n,)](x,rope,cache,slots,k,x.stride(0),rope.stride(0),368,n,n+4,
                     num_warps=4,num_stages=1,enable_fp_fusion=False,debug=True)
    out=torch.empty(n,576,dtype=torch.bfloat16)
    _read_rows[((n+15)//16,)](cache,slots,out,n+4,n,576,num_warps=4,num_stages=1,enable_fp_fusion=False)
    assert same_values(out[:,:512],R.fp4_qdq(x))
    assert torch.equal(out[:,512:].view(torch.uint16),R.fp8_codes(rope,k).bfloat16().view(torch.uint16))
    assert torch.all(cache[2:n+2,292:304]==0)
    assert torch.all(cache[:2]==165) and torch.all(cache[n+2:]==165)
    qdq=torch.empty_like(x)
    _qdq[(n,)](x,qdq,x.stride(0),n,num_warps=4,num_stages=1,enable_fp_fusion=False)
    assert same_values(qdq,R.fp4_qdq(x))
    return cache,out


g=torch.Generator().manual_seed(5105)
x=torch.randn(96,512,generator=g).bfloat16();x[0]=0;x[1]=-0.
for i,e in enumerate((-110,-60,-10,10,60,110)):x[i+2]*=2.**e
compare(x,torch.randn(96,64,generator=g).bfloat16(),.75)
# Invalid slots write no bytes; a live row overwrites the rejected tail.
x=torch.ones(3,512,dtype=torch.bfloat16);rope=torch.ones(3,64,dtype=torch.bfloat16)
cache=torch.full((4,368),165,dtype=torch.uint8);slots=torch.tensor([-1,4,-3],dtype=torch.int64)
_pack_store[(3,)](x,rope,cache,slots,torch.ones(1),512,64,368,3,4,
                  num_warps=4,num_stages=1,enable_fp_fusion=False,debug=True)
assert torch.all(cache==165)
slots[0]=1
_pack_store[(3,)](x,rope,cache,slots,torch.ones(1),512,64,368,3,4,
                  num_warps=4,num_stages=1,enable_fp_fusion=False,debug=True)
assert torch.all(cache[0]==165) and torch.all(cache[2:]==165)

# Real BF16 activations: every case/layer/stage, early/middle/late rows.
files=sorted(Path(os.environ['FP4_DUMPS']).glob('normale/captures/*.pt'))
assert files,'BF16 probe dumps missing'
seen=set();manifest=[]
for file in files:
    e=torch.load(file,map_location='cpu',weights_only=True)
    key=(e['case'],e['layer_id'],e['stage'])
    if key in seen:continue
    seen.add(key)
    x=e['latent'][e['write_slots']>=0];r=e['rope'][e['write_slots']>=0]
    rows=sorted(set([0,len(x)//2,len(x)-1]))
    compare(x[rows].contiguous(),r[rows].contiguous(),float(e['k_scale']))
    manifest.append(dict(file=str(file.relative_to(Path(os.environ['FP4_DUMPS']))),
                         sha256=hashlib.sha256(file.read_bytes()).hexdigest(),rows=rows,case=e['case'],layer=e['layer_id'],stage=e['stage']))
assert {k[1] for k in seen}=={0,39,77,78}
assert {k[0] for k in seen}=={'prose','code','long16k','long60k'}
report=dict(status='PASS_TRITON_INTERPRETER_BITEXACT',torch=torch.__version__,
            triton=__import__('triton').__version__,groups=len(seen),dumps=manifest,
            scope='CPU interpreter; compiled GPU arithmetic and graph replay require qualification')
Path(os.environ['FP4_RESULTS'],'triton-interpreter.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],len(manifest),'fleet groups')

from fp4_prefill_interpreter import verify_tiles
verify_tiles()
