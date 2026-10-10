# SPDX-License-Identifier: Apache-2.0
"""Actual native imports, method preparation and combined boot wrapper order."""
import os
from pathlib import Path
import sys
import types
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'overlay/overlay')]
from boot_preflight import fake_hardware,Report
fake_hardware(Report(),0)
os.environ.update(GLM_MTP_KSTOP='1',
    GLM_MTP_FIX='1',GLM_MTP_KSTOP_CONTROL=str(ROOT/'overlay/kstop/control.json'),
    GLM_MTP_KSTOP_CAPTURE_LAYOUT='reuse',GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CONTROL_MODE='local',VLLM_SERVER_DEV_MODE='1',
    VLLM_USE_V2_MODEL_RUNNER='1',GLM_DRAFT_HEAD='nvfp4',GLM_DRAFT_HEAD_INIT='1',
    GLM_DRAFT_EHPROJ='fp8',GLM_DRAFT_EHPROJ_INIT='1',GLM_MTP_ROWSELECT='1',
    GLM_MOE_DET_ALIGN='1',GLM_GLUE_ROUTER_BF16='1',GLM_GLUE_MOE_WS='1',GLM_GLUE_DSA_IDX_CACHE='1',GLM_SKIP_MLA_PLAN='ab',GLM_INDEXER_SHORTCUT='1')
import glm_mtp_fix,glm_mtp_kstop,glm_draft_head as dh,glm_draft_ehproj as eh,glm_mtp_rowselect as r
import glm_dsa_short
assert glm_mtp_fix.register();assert glm_dsa_short.register();assert glm_mtp_kstop.register();assert dh.register();assert eh.register();assert r.register()
import glm_glue_lite as glue
glue.register()
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
assert GPUModelRunner.load_model.__code__.co_filename==glue.__file__
assert GPUModelRunner.load_model.__wrapped__.__code__.co_filename==r.__file__
assert GPUModelRunner.load_model.__wrapped__.__wrapped__.__code__.co_filename==eh.__file__
assert GPUModelRunner.load_model.__wrapped__.__wrapped__.__wrapped__.__code__.co_filename==dh.__file__
assert GPUModelRunner.capture_model.__code__.co_filename==r.__file__
assert not hasattr(Worker,'mtp_rowselect_set')

def empty(cls):
    obj=cls.__new__(cls);torch.nn.Module.__init__(obj);return obj
block=empty(DeepseekV2DecoderLayer);mla=empty(MultiHeadLatentAttentionWrapper)
block.use_mha=False;block.use_sequence_parallel_moe=False
block.self_attn=torch.nn.Module();block.self_attn.mla_attn=mla;block.mlp=torch.nn.Module()
mla.dcp_q_replicate=False;mla.o_proj=torch.nn.Module()
layer=torch.nn.Module();layer.mtp_block=block
draft=empty(DeepSeekMTP);draft.model=torch.nn.Module();draft.model.layers=torch.nn.ModuleDict({'78':layer})
sp=object.__new__(MTPSpeculator);sp.model=draft;sp.vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(mode=0))
target=torch.nn.Module();target.head=torch.nn.Linear(2,2)
class_methods=(DeepseekV2DecoderLayer.forward,MultiHeadLatentAttentionWrapper.forward,AutoRegressiveSpeculator._prefill)
for fault in ('sp','compile','target_alias','override'):
    if fault=='sp':block.use_sequence_parallel_moe=True
    if fault=='compile':sp.vllm_config.compilation_config.mode=1
    if fault=='target_alias':target.alias=mla.o_proj
    if fault=='override':block.forward=types.MethodType(lambda *a,**kw:None,block)
    try:r.prepare(sp,target)
    except RuntimeError:pass
    else:raise AssertionError('unsupported preparation accepted: '+fault)
    block.use_sequence_parallel_moe=False;sp.vllm_config.compilation_config.mode=0
    if fault=='target_alias':del target.alias
    if fault=='override':del block.forward
prepared=r.prepare(sp,target);r.attach(sp,prepared)
assert class_methods==(DeepseekV2DecoderLayer.forward,MultiHeadLatentAttentionWrapper.forward,AutoRegressiveSpeculator._prefill)
assert sp._prefill.__func__.__code__.co_filename==r.__file__
assert block.forward.__func__.__code__.co_filename==r.__file__
assert mla.forward.__func__.__code__.co_filename==r.__file__
try:r.prepare(sp,target)
except RuntimeError:pass
else:raise AssertionError('second attachment accepted')
print('Combined rowselect/FP8-ehproj/DH boot ordering; actual native instance preparation; unchanged target classes PASS')

# Generate the real source-transformed native descriptor tables on CPU.
# Only process-group identity and graph-pool allocation are substituted.
import json
from vllm.v1.worker.gpu import cudagraph_utils as cg
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
from vllm.config.compilation import CUDAGraphMode
cg.get_pp_group=lambda:types.SimpleNamespace(is_first_rank=True,is_last_rank=True)
cg.current_platform.get_global_graph_pool=lambda:None
config=types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=4),
    compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[1,4,12,16],max_cudagraph_capture_size=16),
    parallel_config=types.SimpleNamespace(data_parallel_size=1,tensor_parallel_size=4),speculative_config=None)
target_manager=cg.ModelCudaGraphManager(config,torch.device('cpu'),CUDAGraphMode.FULL_DECODE_ONLY,4)
first=SpeculatorCudaGraphManager(config,torch.device('cpu'),CUDAGraphMode.FULL_DECODE_ONLY,4)
later=SpeculatorCudaGraphManager(config,torch.device('cpu'),CUDAGraphMode.FULL_DECODE_ONLY,1)
for manager in (target_manager,first,later):
    manager.graphs={d:object() for ds in manager._capture_descs.values() for d in ds}
sp.prefill_cudagraph_manager=first;sp.decode_cudagraph_manager=later;sp.max_num_reqs=4
assert set(target_manager.graphs)==set(first.graphs)
rows=r.account(sp)
assert all(x['tail_rows']<=x['input_rows'] for x in rows)
assert any(x['input_rows']==16 and x['tail_rows']==4 for x in rows if x['stage']=='pass1')
report=dict(scope='actual native candidate descriptor generation, CPU graph handles substituted; no CUDA capture',
    capture_sizes=[1,4,12,16],target_descriptor_count=len(target_manager.graphs),
    pass1_descriptor_count=len(first.graphs),later_descriptor_count=len(later.graphs),
    pass1_full_tail_rows=sum(x['input_rows'] for x in rows if x['stage']=='pass1'),
    pass1_selected_tail_rows=sum(x['tail_rows'] for x in rows if x['stage']=='pass1'),rows=rows)
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
