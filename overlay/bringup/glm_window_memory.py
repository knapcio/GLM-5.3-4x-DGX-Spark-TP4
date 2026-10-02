import hashlib, importlib.abc, importlib.machinery, importlib.util, sys, time, json, os
from pathlib import Path
TARGETS={'vllm.v1.worker.gpu_model_runner':'5be64e11d25f9802a1970089e9203ae4b18977b340471a6d0761f0274b54c4b6','vllm.v1.worker.gpu.model_runner':'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415'}
PIN='5be64e11d25f9802a1970089e9203ae4b18977b340471a6d0761f0274b54c4b6'
def install(mod):
 if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()!=TARGETS[mod.__name__]: raise RuntimeError('window-memory source drift')
 orig=mod.GPUModelRunner.capture_model
 def capture(self,*a,**kw):
  mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
  import torch
  d=dict(phase='pre-capture',pid=os.getpid(),mem_kB=mem,torch_allocated_B=torch.cuda.memory_allocated(),torch_reserved_B=torch.cuda.memory_reserved(),wall_epoch=time.time())
  sys.stderr.write('glm-window-memory: '+json.dumps(d)+'\n');sys.stderr.flush()
  if mem['MemAvailable']<10*1048576:raise RuntimeError('window-memory pre-capture below 10 GiB')
  start=time.monotonic();z=orig(self,*a,**kw)
  sys.stderr.write('glm-window-memory: '+json.dumps(dict(phase='capture-finished',wall_s=time.monotonic()-start,torch_allocated_B=torch.cuda.memory_allocated(),torch_reserved_B=torch.cuda.memory_reserved()))+'\n')
  return z
 mod.GPUModelRunner.capture_model=capture
 sys.stderr.write('glm-window-memory: capture guard source pin PASS\n')
def register():
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname not in TARGETS:return
   sys.meta_path.remove(self)
   try:spec=importlib.util.find_spec(fullname)
   finally:sys.meta_path.insert(0,self)
   original_exec=spec.loader.exec_module
   def execute(module):original_exec(module);install(module)
   spec.loader.exec_module=execute
   return spec
 sys.meta_path.insert(0,Finder())
