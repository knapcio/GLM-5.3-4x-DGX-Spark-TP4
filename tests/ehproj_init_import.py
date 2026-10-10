# SPDX-License-Identifier: Apache-2.0
"""Pinned frozen-config import ordering; CUDA library preparation is substituted."""
import os,sys
from unittest.mock import patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'overlay/overlay')]
from boot_preflight import fake_hardware,Report
fake_hardware(Report(),0)
os.environ.update(GLM_MTP_KSTOP='1',
    GLM_MTP_FIX='1',GLM_MTP_KSTOP_CONTROL=str(ROOT/'overlay/kstop/control.json'),
    GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CONTROL_MODE='local',VLLM_SERVER_DEV_MODE='1',
    VLLM_USE_V2_MODEL_RUNNER='1',GLM_DRAFT_HEAD='nvfp4',GLM_DRAFT_HEAD_INIT='1')
import glm_mtp_kstop as k
import glm_draft_head as d
import glm_mtp_fix
glm_mtp_fix.register();assert k.register();assert d.register();assert d.INITIAL_ON
import glm_draft_ehproj as eh
os.environ.update(GLM_DRAFT_EHPROJ='fp8',GLM_DRAFT_EHPROJ_INIT='1',
    GLM_MOE_DET_ALIGN='1',GLM_GLUE_ROUTER_BF16='1',GLM_GLUE_MOE_WS='1',GLM_GLUE_DSA_IDX_CACHE='0')
assert eh.register()
import glm_moe_det as det
with patch('det_align.runtime.prepare_library',return_value=object()):
    assert det.register()
import glm_glue_lite as glue
assert glue.register()
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
import inspect
capture=SpeculatorCudaGraphManager.__bases__[0].capture
assert '_k4_drafthead_factory' in capture.__code__.co_names,capture
assert capture.__code__.co_filename == d.__file__
assert hasattr(Worker,'draft_head_status')
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
assert GPUModelRunner.load_model.__code__.co_filename == glue.__file__
assert GPUModelRunner.load_model.__wrapped__.__code__.co_filename == det.__file__
assert GPUModelRunner.load_model.__wrapped__.__wrapped__.__code__.co_filename == eh.__file__
assert GPUModelRunner.capture_model.__code__.co_filename == eh.__file__
assert GPUModelRunner.capture_model.__wrapped__.__code__.co_filename == d.__file__
assert not hasattr(Worker,'draft_ehproj_set')
print('Frozen det-align/FP8 eh_proj/F1+F2 import order and combined INIT qualification PASS')
