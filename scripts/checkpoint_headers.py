#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export checkpoint/sidecar metadata without reading tensor payloads. No network.

Run later on a node (by the operator), then copy the JSON cache to the Mac:
 python3 scripts/checkpoint_headers.py --model /path/model --sidecar /path/attn \
   --sidecar /path/more --out headers.json
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct

DTYPE_BYTES = {'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'I32':4,'U32':4,
               'I64':8,'U64':8,'F16':2,'BF16':2,'F32':4,'F64':8,
               'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1}
MAX_HEADER = 64 << 20


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key: '+key)
        result[key]=value
    return result


def validate(header, payload_bytes):
    spans=[]
    for name,t in header.items():
        if name=='__metadata__':continue
        if not isinstance(t,dict) or set(t)!={'dtype','shape','data_offsets'}:
            raise ValueError('invalid tensor header: '+name)
        shape=t['shape'];offset=t['data_offsets'];dtype=t['dtype']
        if dtype not in DTYPE_BYTES or not isinstance(shape,list) or any(type(n)!=int or n<0 for n in shape):
            raise ValueError('invalid dtype/shape: '+name)
        if len(offset)!=2 or any(type(n)!=int for n in offset) or not 0<=offset[0]<=offset[1]<=payload_bytes:
            raise ValueError('invalid data offsets: '+name)
        if offset[1]-offset[0]!=math.prod(shape)*DTYPE_BYTES[dtype]:
            raise ValueError('shape/byte count mismatch: '+name)
        spans.append((*offset,name))
    end=0
    for lo,hi,name in sorted(spans):
        if lo!=end:raise ValueError('overlap/gap in data offsets: '+name)
        end=hi
    if end!=payload_bytes:raise ValueError('unclaimed tensor payload')


def read_header(path):
    # Deliberately two bounded reads; never safe_open/load_file/mmap tensor data.
    with path.open('rb') as f:
        size=f.read(8)
        if len(size)!=8:raise ValueError('truncated safetensors length')
        n=struct.unpack('<Q',size)[0]
        if not 2<=n<=MAX_HEADER or n>path.stat().st_size-8:raise ValueError('invalid header length')
        raw=f.read(n)
        if len(raw)!=n:raise ValueError('truncated safetensors header')
    h=json.loads(raw,object_pairs_hook=unique)
    validate(h,path.stat().st_size-8-n)
    return dict(raw_b64=base64.b64encode(raw).decode(),sha256=sha(raw),size=path.stat().st_size,header_bytes=n)


def safe_file(root,name):
    p=root/name
    if Path(name).name!=name or p.is_symlink() or not p.is_file():raise ValueError('unsafe/missing file: '+name)
    return p


def export(root, sidecar=False):
    requested=str(root.absolute());root=root.resolve()
    names=('manifest.json',) if sidecar else ('config.json','model.safetensors.index.json')
    documents={n:base64.b64encode(safe_file(root,n).read_bytes()).decode() for n in names}
    if sidecar:
        manifest=json.loads(base64.b64decode(documents['manifest.json']))
        shards={e['file'] for e in manifest['tensors'].values()}
    else:
        index=json.loads(base64.b64decode(documents['model.safetensors.index.json']))
        shards=set(index['weight_map'].values())
    return dict(source=requested,resolved_source=str(root),documents=documents,files={n:read_header(safe_file(root,n)) for n in sorted(shards)})


def unpack(bundle):
    docs={n:base64.b64decode(v,validate=True) for n,v in bundle['documents'].items()}
    tensors={}
    for filename,receipt in bundle['files'].items():
        raw=base64.b64decode(receipt['raw_b64'],validate=True)
        if sha(raw)!=receipt['sha256'] or len(raw)!=receipt['header_bytes']:raise ValueError('cached header digest/length mismatch')
        h=json.loads(raw,object_pairs_hook=unique)
        validate(h,receipt['size']-8-len(raw))
        for name,t in h.items():
            if name=='__metadata__':continue
            key=filename+'/'+name if 'manifest.json' in docs else name
            if key in tensors:raise ValueError('duplicate checkpoint tensor: '+name)
            tensors[key]=dict(t,file=filename)
    if 'model.safetensors.index.json' in docs:
        index=json.loads(docs['model.safetensors.index.json'],object_pairs_hook=unique)['weight_map']
        if set(index)!=set(tensors):raise ValueError('index/header tensor inventory mismatch')
        for name,file in index.items():
            if tensors[name]['file']!=file:raise ValueError('index/header shard mismatch: '+name)
    return docs,tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--sidecar',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=dict(schema=1,model=export(a.model),sidecars=[export(s,True) for s in a.sidecar])
    # Refuse overwrites and verify the cache through the same reader as preflight.
    for b in [result['model'],*result['sidecars']]:unpack(b)
    with a.out.open('x') as f:json.dump(result,f,separators=(',',':'));f.write('\n')
    print('HEADERS ONLY',a.out)

if __name__=='__main__':main()
