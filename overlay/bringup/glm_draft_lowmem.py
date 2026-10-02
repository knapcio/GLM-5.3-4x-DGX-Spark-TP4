# SPDX-License-Identifier: Apache-2.0
"""Draft-only loader policy; leaves target fast-loader configuration unchanged."""
import os,sys

def register():
 if os.environ.get('GLM_DRAFT_LOWMEM','0')!='1':return
 import glm_fast_load as m
 orig=m.fast_safetensors_iterator
 def iterator(files,*a,**kw):
  files=list(files)
  if not files or not all(str(f).startswith('/draft/') for f in files):
   yield from orig(files,*a,**kw);return
  keys={'GLM_FAST_LOAD_SLAB_MB':'16','GLM_FAST_LOAD_AHEAD_MB':'32','GLM_FAST_LOAD_THREADS':'1','GLM_FAST_LOAD_PINNED':'0','GLM_FAST_LOAD_DROP_CACHE':'1'}
  old={k:os.environ.get(k) for k in keys}
  os.environ.update(keys)
  sys.stderr.write('glm-draft-lowmem: draft-only slabs16MiB ahead32MiB thread1 pageable staging; target policy unchanged\n');sys.stderr.flush()
  try:yield from orig(files,*a,**kw)
  finally:
   for k,v in old.items():
    if v is None:os.environ.pop(k,None)
    else:os.environ[k]=v
   # Drop only the already consumed checkpoint pages, using the existing targeted API.
   for f in files:
    fd=os.open(f,os.O_RDONLY)
    try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
 m.fast_safetensors_iterator=iterator
