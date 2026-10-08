# SPDX-License-Identifier: Apache-2.0
"""Native capture/prefill/decode bodies with CPU model, metadata and CUDA substitutes."""
import contextlib, json, sys, types
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'tests')]
from boot_preflight import fake_hardware, Report
fake_hardware(Report(),0)
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.spec_decode.autoregressive import cudagraph_utils as scg, speculator as native
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from test_draft_head_init import fixture, completed_vote
import glm_draft_head as dh
import glm_dsa_short as dsa
from dataclasses import dataclass
@dataclass(frozen=True)
class Desc:
    num_reqs:int
    num_tokens:int
    short_context:bool
    cg_mode:object=Mode.FULL
    num_active_loras:int=0
    uniform_token_count:int=1
active=[];captured=[]
class Graph:
    def replay(self):
        with dsa.context(self.short):self.fn()
    def reset(self):pass
@contextlib.contextmanager
def graph(g,pool):
    active.append(g)
    try:yield
    finally:active.pop()
@contextlib.contextmanager
def capture_context(**kw):yield NS()
# Actual production wrapper registration order from sitecustomize: DSA then DH.
cg.graph_capture=capture_context
dsa.install_cg(cg)
baseline_mode=len(sys.argv)>1
if baseline_mode:
    baseline=types.ModuleType('dh_baseline')
    exec(compile(Path(sys.argv[1]).read_text(),'9cd0116/glm_draft_head.py','exec'),baseline.__dict__)
    baseline.install_graph(scg)
else:
    dh.install_graph(scg)
r=fixture();sp=r.speculator
sp.max_num_reqs=4;sp.num_speculative_steps=3;sp.device=torch.device('cpu')
sp.max_model_len=128;sp.advance_draft_positions=True;sp.use_fused_multi_step_decode=False
sp.hidden_states=torch.zeros(16,256,dtype=torch.bfloat16)
sp.draft_tokens=torch.zeros(4,3,dtype=torch.long);sp._kstop.confidence=torch.zeros(4)
sp.last_token_indices=torch.zeros(4,dtype=torch.long);sp.idx_mapping=torch.zeros(4,dtype=torch.int32)
sp.temperature=torch.zeros(4);sp.seeds=torch.zeros(4,dtype=torch.long)
sp.input_buffers=NS(input_ids=torch.zeros(16,dtype=torch.long),positions=torch.zeros(16,dtype=torch.long))
sp.target_input_buffers=NS(input_ids=torch.zeros(16,dtype=torch.long),positions=torch.zeros(16,dtype=torch.long))
sp.block_tables=NS();sp.model_state=NS();sp.attn_groups=[];sp.target_attn_groups=[];sp.kv_cache_config=NS()
sp.share_mtp_topk_indices=False
for name in ('on_prefill_begin','on_prefill_end','on_multi_step_decode_begin','on_multi_step_decode_end'):
    setattr(sp,name,MethodType(getattr(MTPSpeculator,name),sp))
sp._prepare_eplb_forward=lambda n:None
sp._run_model=lambda n,*args,**kw:(sp.hidden_states[:n]+(1 if dsa.SHORT.get() else 2),)*2
def sample(hidden,*args):
    logits=r._draft_head.bank(hidden) if r._draft_head.on else r.model.lm_head.quant_method.apply(r.model.lm_head,hidden)
    sp._kstop.confidence[:len(hidden)].fill_(.9)
    return logits.argmax(-1)
sp.sample_draft=sample
for name in ('_prefill','_generate_draft'):
    actual=getattr(native.AutoRegressiveSpeculator,name)
    def run(*args,_actual=actual,**kw):
        if active:
            active[-1].short=dsa.SHORT.get()
            active[-1].fn=lambda:_actual(sp,*args,**kw)
        return _actual(sp,*args,**kw)
    setattr(sp,name,run)
sp.capture=MethodType(native.AutoRegressiveSpeculator.capture,sp)
for index in range(2):
    manager=object.__new__(scg.SpeculatorCudaGraphManager)
    manager.device=torch.device('cpu');manager.max_num_reqs=4;manager.dp_size=1
    manager.pool=(1,2);manager.use_breakable_cg=False;manager.graphs={};manager._graphs_captured=False
    manager._capture_descs={Mode.FULL:[Desc(n,t,short,uniform_token_count=t//n) for n,t in ((1,1),(4,4 if index else 16)) for short in (False,True)]}
    setattr(sp,('prefill_cudagraph_manager','decode_cudagraph_manager')[index],manager)
def update(tokens,step,hidden,drafts,hs,buffers,n,*args,**kw):
    drafts[:n,int(step)]=tokens;hs[:n].copy_(hidden[:n]);buffers.input_ids[:n].copy_(tokens)
offloader=NS(sync_prev_onload=lambda:None,join_after_forward=lambda:None)
cls=type('GPUModelRunner',(),dict(load_model=lambda self:None,
    capture_model=lambda self:(self.speculator.capture(),'capture-result')[1]))
dh.install_runner(NS(GPUModelRunner=cls))
with patch.object(dh,'agree',completed_vote),patch.object(dh,'memory',lambda:8<<30), \
     patch.object(torch.cuda,'synchronize',lambda:None),patch.object(torch.cuda,'empty_cache',lambda:None), \
     patch.object(dh,'renew_draft_pool',lambda r:None),patch.object(cg,'is_global_first_rank',lambda:False),patch.object(cg,'get_offloader',lambda:offloader), \
     patch.object(cg,'set_graph_pool_id',lambda p:None),patch.object(torch.cuda,'CUDAGraph',Graph), \
     patch.object(torch.cuda,'graph',graph),patch.object(torch.cuda,'Stream',lambda **kw:NS()), \
     patch.object(scg,'prepare_inputs_to_capture',lambda *a,**kw:(None,None)), \
     patch.object(native,'update_draft_inputs',update):
    assert cls.capture_model(r)=='capture-result'
if baseline_mode:
    assert not r._draft_head_ready and not r._draft_head.on
    assert not dh.status(r)['dh_gate_passed']
    failed=[t for v in r._draft_head_init_failure['votes'] for t in v['terms']
            if t['check']=='replay_compare' and not t['conditions']['bit_exact']]
    assert failed and 'short_context=True' in failed[0]['measured']['descriptor']
else:
    assert r._draft_head_ready and r._draft_head.on
    assert len(r._draft_head_qualification['cases'])==8
print(json.dumps(dict(ok=True,baseline_factory=baseline_mode,dh_gate=dh.status(r)['dh_gate_passed'],cases=8,native_bodies=['CudaGraphManager.capture','SpeculatorCudaGraphManager.capture',
    'AutoRegressiveSpeculator.capture','AutoRegressiveSpeculator._prefill','AutoRegressiveSpeculator._generate_draft'],
    wrapper_order='DSA then DH',substitutes=['CPU model/metadata','CUDA graph/stream','Triton update'],
    scope='CPU control order and changed-input replay; GPU numerical qualification unverified')))
