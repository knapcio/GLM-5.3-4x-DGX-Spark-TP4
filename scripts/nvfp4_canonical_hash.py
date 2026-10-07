#!/usr/bin/env python3
"""nvfp4a window: canonical sidecar content hash (stdlib only). safetensors writes __metadata__ from a Rust HashMap,
so header key order (and the file sha256) differs between converter processes while tensor bytes are identical.
Canonical per-file hash = sha256(sorted-key header JSON + data bytes). Prints one line per file + an aggregate."""
import hashlib, json, struct, sys
from pathlib import Path
root = Path(sys.argv[1])
man = json.loads((root / 'manifest.json').read_text())
agg = hashlib.sha256()
rows = []
for name in sorted(man['tensors'], key=lambda n: man['tensors'][n]['file']):
    e = man['tensors'][name]
    b = (root / e['file']).read_bytes()
    n = struct.unpack('<Q', b[:8])[0]
    h = json.loads(b[8:8 + n])
    c = hashlib.sha256(json.dumps(h, sort_keys=True, separators=(',', ':')).encode() + b[8 + n:]).hexdigest()
    d = hashlib.sha256(b[8 + n:]).hexdigest()
    agg.update(f'{e["file"]} {c} {e["global_scale"]!r} {e["shape"]}\n'.encode())
    rows.append((e['file'], c, d))
print(json.dumps(dict(files=len(rows), canonical_aggregate=agg.hexdigest(), source=man['source']['index_sha256'],
                      shards=len(man['source']['shards']), complete=man['complete'],
                      bytes=sum(e['bytes'] for e in man['tensors'].values()))))
