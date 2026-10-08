# SPDX-License-Identifier: Apache-2.0
"""Pinned standalone Worker + native manager import verifies INIT factory hook."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'overlay/overlay')]
from boot_preflight import fake_hardware,Report
fake_hardware(Report(),0)
os.environ.update(GLM_MTP_KSTOP='1',GLM_MTP_K4_CAPTURE='0',GLM_MTP_K4_GATE='0',
    GLM_MTP_FIX='1',GLM_MTP_KSTOP_CONTROL=str(ROOT/'overlay/kstop/control.json'),
    GLM_MTP_KSTOP_UNIFORM_BATCH='k2',GLM_MTP_KSTOP_CONTROL_MODE='local',VLLM_SERVER_DEV_MODE='1',
    VLLM_USE_V2_MODEL_RUNNER='1',GLM_DRAFT_HEAD='nvfp4',GLM_DRAFT_HEAD_INIT='1')
import glm_mtp_kstop as k
import glm_draft_head as d
import glm_mtp_fix
glm_mtp_fix.register();assert k.register();assert d.register();assert d.INITIAL_ON
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
import inspect
capture=SpeculatorCudaGraphManager.__bases__[0].capture
assert '_k4_drafthead_factory' in capture.__code__.co_names,capture
assert capture.__code__.co_filename == d.__file__
from vllm.v1.worker.gpu import cudagraph_utils as native_manager
assert inspect.unwrap(native_manager.graph_capture).__code__.co_filename == d.__file__
assert hasattr(Worker,'draft_head_status')
print('Standalone INIT native Worker/manager callback PASS')
