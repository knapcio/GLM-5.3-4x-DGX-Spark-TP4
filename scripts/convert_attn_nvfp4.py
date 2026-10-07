#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline sidecar conversion. No network, CUDA, or writes to the source checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'overlay/overlay'))
import glm_nvfp4_format as fmt


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def source_file(src, filename):
    p = src/filename
    if p.parent.resolve() != src or not p.is_file():
        raise ValueError('missing/unsafe checkpoint shard: ' + str(p))
    return p


def convert(src, dst, rows=64, modules=fmt.MODULES):
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if src == dst or src in dst.parents or dst in src.parents or rows < 1:
        raise ValueError('output must be a separate NEW directory outside the source checkpoint')
    dst.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in dst.iterdir()):
        raise ValueError('output directory contains a symlink')
    with open(dst/'.convert.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mp = dst/'manifest.json'
        if not mp.exists() and any(p.name != '.convert.lock' for p in dst.iterdir()):
            raise ValueError('new output directory is not empty')
        index_path = src/'model.safetensors.index.json'
        idx = json.loads(index_path.read_text())['weight_map']
        required = [name+'.'+leaf for name in modules for leaf in ('weight_packed','weight_scale','weight_shape')]
        if any(name not in idx for name in required):
            raise ValueError('checkpoint does not contain the complete 385-tensor attention group')
        files = sorted({idx[name] for name in required})
        if any(dst in source_file(src,f).resolve().parents for f in files):
            raise ValueError('output overlaps original shard storage')
        # Hash every input read before allowing resume; changing weights under an unchanged index is detected.
        source = dict(index_sha256=fmt.sha256(index_path), config_sha256=fmt.sha256(src/'config.json'),
                      shards={f:fmt.sha256(source_file(src,f)) for f in files})
        manifest = dict(schema=1, algorithm=fmt.ALGORITHM, source=source, complete=False, tensors={}, outputs={})
        if mp.exists():
            manifest = json.loads(mp.read_text())
            if manifest.get('source') != source or manifest.get('algorithm') != fmt.ALGORITHM or manifest.get('schema') != 1 or not set(manifest['tensors']).issubset(modules):
                raise ValueError('resume source/algorithm mismatch')
        manifest['complete'] = False
        atomic_json(mp, manifest)
        def read(name):
            with safe_open(source_file(src,idx[name]), framework='pt', device='cpu') as sf:
                return sf.get_tensor(name)
        begin = time.monotonic()
        for ordinal, name in enumerate(modules):
            entry = manifest['tensors'].get(name)
            filename = f'attn-{ordinal:03d}.safetensors'
            path = dst/filename
            if entry:
                if entry['file'] != filename or path.is_symlink() or fmt.sha256(path) != entry['sha256']:
                    raise ValueError('resume output hash/path mismatch: ' + name)
                fmt.validate_tensors(load_file(str(path)), entry['shape'])
                continue
            shape_tensor = read(name+'.weight_shape')
            if shape_tensor.dtype not in (torch.int32,torch.int64) or tuple(shape_tensor.shape) != (2,):
                raise ValueError('invalid original weight_shape tensor')
            n,k = map(int,shape_tensor.tolist())
            if n < 1 or k < 128 or k % 128:
                raise ValueError('invalid original logical weight shape')
            scales = read(name+'.weight_scale')
            if scales.dtype != torch.bfloat16 or tuple(scales.shape) != (n,k//128) or k % 128:
                raise ValueError('expected original BF16 group128 W8A16 scales')
            with safe_open(source_file(src,idx[name+'.weight_packed']), framework='pt', device='cpu') as sf:
                p = sf.get_slice(name+'.weight_packed')
                if p.get_dtype() != 'I32' or p.get_shape() != [n,k//4]:
                    raise ValueError('packed INT8 shape/dtype mismatch')
                ts = np.float32(0)
                for a in range(0,n,rows):
                    b=min(a+rows,n)
                    ts = max(ts, np.max(np.abs(fmt.decode_int8(p[a:b], scales[a:b]))))
                ts = np.float32(ts / np.float32(6*448))
                values=np.empty((n,k//2),np.uint8); scale_codes=np.empty((n,k//16),np.uint8)
                for a in range(0,n,rows):
                    b=min(a+rows,n)
                    values[a:b],scale_codes[a:b] = fmt.encode(fmt.decode_int8(p[a:b],scales[a:b]),ts)
            tensors=dict(weight_packed=torch.from_numpy(values),
                         weight_scale=torch.from_numpy(scale_codes).view(torch.float8_e4m3fn),
                         weight_global_scale=torch.tensor([ts],dtype=torch.float32))
            fmt.validate_tensors(tensors,[n,k])
            tmp=path.with_suffix('.tmp')
            if path.is_symlink() or tmp.is_symlink():
                raise ValueError('unsafe output symlink')
            # One metadata key: safetensors serializes __metadata__ from a hash map, so with two keys the
            # header order (and the file hash) changed between converter processes on identical tensors.
            save_file(tensors,str(tmp),metadata={'glm_nvfp4':json.dumps(dict(algorithm=fmt.ALGORITHM,module=name),
                                                                    sort_keys=True,separators=(',',':'))})
            with open(tmp,'rb') as f: os.fsync(f.fileno())
            os.replace(tmp,path)
            manifest['tensors'][name]=dict(file=filename,sha256=fmt.sha256(path),shape=[n,k],
                                          bytes=path.stat().st_size,global_scale=float(ts))
            atomic_json(mp,manifest)
            print(f'{ordinal+1}/{len(modules)} {name} {time.monotonic()-begin:.1f}s',flush=True)
        inventory=dst/'tensor-list.txt'
        inventory.write_text(''.join(n+'.weight_packed\n' for n in modules))
        manifest['outputs']['tensor-list.txt']=fmt.sha256(inventory)
        manifest['complete']=True
        atomic_json(mp,manifest)
        # A manifest cannot contain its own hash. External checksum covers it and every data output.
        sums=[(e['sha256'],e['file']) for e in manifest['tensors'].values()]
        sums += [(h,p) for p,h in manifest['outputs'].items()]+[(fmt.sha256(mp),'manifest.json')]
        checksum=dst/'SHA256SUMS'; tmp=dst/'SHA256SUMS.tmp'
        tmp.write_text(''.join(f'{h}  {p}\n' for h,p in sorted(sums,key=lambda x:x[1])))
        os.replace(tmp,checksum)
        print('SHA256SUMS sha256=' + fmt.sha256(checksum),flush=True)
        return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('output');parser.add_argument('--rows',type=int,default=64)
    args=parser.parse_args()
    convert(args.checkpoint,args.output,args.rows)
