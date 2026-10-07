# SPDX-License-Identifier: Apache-2.0
"""Target-only FP8 recent latent ring over the unchanged FP4x backing cache.

Reserve before profiling. A V2 runner hook stages stable request-state indices,
logical positions and window bounds before every target forward/replay. Keep
all current query rows plus W previous rows for multi-token prefill/verify.
MTP remains FP4x. Missing/overwritten sidecar rows always use FP4x backing.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import re
import sys

RUNNER = 'vllm.v1.worker.gpu.model_runner'
RUNNER_PIN = 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415'
WORKER = 'vllm.v1.worker.gpu_worker'
WORKER_PIN = 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1'
MAX_QUERY = 4096
MAX_VERIFY = 36
REQUESTS = 4
_WINDOW = 0
_INITIAL = 0
_AB = False
_REGISTERED = None
_DEVICES = {}
_CACHES = {}


def options(env):
    value = env.get('GLM_FP4_RECENT_WINDOW', '0')
    if str(value) not in ('0', '2048', '4096', '8192'):
        raise ValueError('GLM_FP4_RECENT_WINDOW must be 0, 2048, 4096 or 8192')
    window = int(value)
    ab = env.get('GLM_FP4_RECENT_AB', '0')
    initial = env.get('GLM_FP4_RECENT_INIT', str(window))
    if ab not in ('0','1') or str(initial) not in ('0','2048','4096','8192') or int(initial)>window:
        raise ValueError('recent KV AB/init invalid or larger than reserved window')
    if not window and (ab!='0' or 'GLM_FP4_RECENT_INIT' in env):
        raise ValueError('recent KV AB/init needs a reserved window')
    if window and (env.get('GLM_KV_FORMAT')!='fp4x' or env.get('VLLM_USE_V2_MODEL_RUNNER')!='1'):
        raise ValueError('recent KV requires FP4x and the V2 runner')
    if ab=='1' and env.get('VLLM_SERVER_DEV_MODE')!='1':
        raise ValueError('recent KV AB requires private VLLM_SERVER_DEV_MODE=1')
    if ab=='0' and int(initial)!=window:
        raise ValueError('recent KV init is only switchable in AB mode')
    return window,int(initial),ab=='1'


def memory_bytes(window, rows=1573*64, layers=78):
    if not window: return 0
    ring = window + MAX_QUERY + MAX_VERIFY
    # Latent bytes, physical-slot tag and position per ring row, reverse map.
    return layers * (REQUESTS*ring*(512+4+4) + rows*4) + MAX_QUERY*8 + REQUESTS*4 + 4


def device_bank(device):
    import torch
    key=str(device)
    if key not in _DEVICES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('recent KV device bank not reserved before capture')
        _DEVICES[key]=dict(flag=torch.full((1,),_INITIAL,dtype=torch.int32,device=device),
            token_ring=torch.full((MAX_QUERY,),-1,dtype=torch.int32,device=device),
            token_pos=torch.full((MAX_QUERY,),-1,dtype=torch.int32,device=device),
            lower=torch.zeros(REQUESTS,dtype=torch.int32,device=device))
    return _DEVICES[key]


def reserve(layer):
    if not _WINDOW: return
    import torch
    from glm_fp4_prefill import MLA_ROWS
    m=re.search(r'\.layers\.(\d+)\.',layer.layer_name)
    if not m or not 0<=int(m[1])<=78:
        raise RuntimeError('recent KV requires pinned target layers 0..77 / MTP78')
    target=int(m[1])<78
    layer._glm_recent_target=target
    layer.impl._glm_recent_bank=None
    if not target: return
    device=layer.impl.topk_indices_buffer.device
    shared=device_bank(device); ring=_WINDOW+MAX_QUERY+MAX_VERIFY
    layer.impl._glm_recent_bank=dict(**shared,capacity=ring,
        shadow=torch.empty((REQUESTS*ring,512),dtype=torch.uint8,device=device),
        mapping=torch.full((MLA_ROWS,),-1,dtype=torch.int32,device=device),
        tags=torch.full((REQUESTS*ring,),-1,dtype=torch.int32,device=device),
        positions=torch.full((REQUESTS*ring,),-1,dtype=torch.int32,device=device))


def bind(cache, bank):
    if bank is None or not cache.numel(): return
    if cache.numel()//368>bank['mapping'].numel():
        raise RuntimeError('recent KV physical cache exceeds constructor reservation')
    key=(str(cache.device),cache.data_ptr())
    if key in _CACHES and _CACHES[key] is not bank:
        raise RuntimeError('recent KV cache ownership changed')
    _CACHES[key]=bank


def reader_args(cache):
    bank=_CACHES.get((str(cache.device),cache.data_ptr())) if _WINDOW else None
    if bank is None:
        return (cache,)*6+(0,)
    return (bank['shadow'],bank['mapping'],bank['tags'],bank['positions'],
            bank['lower'],bank['flag'],bank['capacity'])


def fresh_flag(device, target):
    return device_bank(device)['flag'] if _WINDOW and target else None


def stage(batch):
    if batch.num_reqs_after_padding>REQUESTS or batch.num_tokens_after_padding>MAX_QUERY:
        raise RuntimeError('recent KV supports at most four requests / 4096 query rows')
    from glm_recent_kv_kernel import stage_rows
    bank=device_bank(batch.positions.device)
    stage_rows(batch,bank,_WINDOW+MAX_QUERY+MAX_VERIFY)


def install_runner(mod):
    if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()!=RUNNER_PIN:
        raise RuntimeError('recent KV V2 runner source drift')
    for name in ('prepare_attn','prepare_dummy_attn'):
        original=getattr(mod.GPUModelRunner,name)
        @functools.wraps(original)
        def prepare(self,batch,_original=original):
            result=_original(self,batch)
            stage(batch)
            return result
        setattr(mod.GPUModelRunner,name,prepare)


def status():
    return dict(reserved_window=_WINDOW,active_window=_INITIAL,ab=_AB,
                ring_capacity=_WINDOW+MAX_QUERY+MAX_VERIFY if _WINDOW else 0,
                target_layers=78,mtp=False,bound_caches=len(_CACHES))


def set_window(window):
    global _INITIAL
    if not _AB: raise RuntimeError('recent KV runtime switch requires AB boot')
    if type(window) is not int or window not in (0,2048,4096,8192) or window>_WINDOW:
        raise ValueError('recent KV switch exceeds reserved window')
    # Worker collective RPC runs between model executions. Captured kernels
    # read this device scalar; no Python-only graph switch or recapture.
    for bank in _DEVICES.values(): bank['flag'].fill_(window)
    _INITIAL=window
    return status()


def install_worker(mod):
    if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()!=WORKER_PIN:
        raise RuntimeError('recent KV worker source drift')
    def recent_kv_set(self,window):
        if isinstance(window,str):
            if window not in ('0','2048','4096','8192'):raise ValueError('invalid recent KV RPC window')
            window=int(window)
        return dict(rank=int(os.environ.get('RANK','-1')),**set_window(window))
    def recent_kv_status(self):
        return dict(rank=int(os.environ.get('RANK','-1')),**status())
    mod.Worker.recent_kv_set=recent_kv_set
    mod.Worker.recent_kv_status=recent_kv_status


def register(env=None):
    global _WINDOW,_INITIAL,_AB,_REGISTERED
    config=options(os.environ if env is None else env)
    if _REGISTERED is not None:
        if config!=_REGISTERED:raise RuntimeError('recent KV reservation is boot-time only; restart required')
        return bool(_WINDOW)
    _REGISTERED=config
    _WINDOW,_INITIAL,_AB=config
    if not _WINDOW: return False
    callbacks={RUNNER:install_runner}
    if _AB: callbacks[WORKER]=install_worker
    class Hooks(importlib.abc.MetaPathFinder):
        busy=set()
        def find_spec(self,name,path=None,target=None):
            if name not in callbacks or name in self.busy:return None
            self.busy.add(name)
            try: spec=importlib.util.find_spec(name)
            finally:self.busy.discard(name)
            if spec is None: raise ImportError(name)
            original=spec.loader.exec_module
            def execute(module): original(module);callbacks[name](module)
            spec.loader.exec_module=execute
            return spec
    sys.meta_path.insert(0,Hooks())
    for name,callback in callbacks.items():
        if name in sys.modules:callback(sys.modules[name])
    sys.stderr.write('recent KV ARMED '+str(status())+'\n')
    return True
