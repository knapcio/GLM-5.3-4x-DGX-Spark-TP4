#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read frozen campaign receipts only; never connects to hosts."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def summarize(root):
    sources, snapshots, summaries = {}, [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        name = path.name
        if not (name in ('boot-mem.json', 'receipts.json', 'memwatch-minima.json') or
                name.startswith('hold-') and name.endswith('.json') or
                re.fullmatch(r'rank[0-3]-(boot|full)\.log', name) or
                name.startswith('memwatch') and name.endswith('.log')):
            continue
        raw = path.read_bytes()
        rel = str(path.relative_to(root))
        sources[rel] = hashlib.sha256(raw).hexdigest()
        if name.endswith('.json'):
            obj = json.loads(raw)
            if name.startswith('hold-'):
                summaries.append(dict(source=rel, min_GiB=obj.get('min_GiB'),
                    swap_max_KiB=obj.get('swap_used_kB_max'), samples=obj.get('samples')))
            elif name == 'memwatch-minima.json':
                summaries.append(dict(source=rel, phases=obj))
            elif name == 'boot-mem.json':
                summaries.append(dict(source=rel, pre_capture={k:v.get('pre_capture_GiB') for k,v in obj.get('ranks',{}).items()}))
        if name.endswith('-boot.log'):  # full log duplicates the same startup snapshot
            for line in raw.decode(errors='replace').splitlines():
                if 'glm-window-memory: {' not in line:
                    continue
                obj = json.loads(line.split('glm-window-memory: ', 1)[1])
                if 'mem_kB' in obj:
                    m = obj['mem_kB']
                    obj['file_cache_upper_kB'] = max(0, m['Cached']-m['Shmem'])
                    obj['clean_unmapped_candidate_kB'] = max(0, m['Cached']-m['Shmem']-m['Mapped']-m['Dirty']-m['Writeback'])
                    # This is an accounting residual, not a CUDA driver counter.
                    obj['nonfree_nonfile_nonslab_nonanon_residual_GiB'] = (
                        m['MemTotal']-m['MemFree']-(m['Cached']-m['Shmem'])-
                        m['Shmem']-m['Buffers']-m['AnonPages']-m['Slab'])/1048576
                snapshots.append(dict(source=rel, rank=int(name[4]), **obj))
    return dict(scope='offline frozen receipts; pre-capture snapshots are not under-load PSS measurements',
                sources_sha256=sources, snapshots=snapshots, holds_and_minima=summaries)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipts', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(summarize(a.receipts),indent=2)+'\n')
