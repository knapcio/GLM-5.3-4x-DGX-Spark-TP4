# SPDX-License-Identifier: Apache-2.0
"""Per-rank SHA-256 manifest of the final GPU weights of a running engine (opt-in, GLM_PARAM_HASH=1).

Purpose: decide whether two boots (for example the full-scan loader and the MTP-only loader)
end with the same weight bytes on every rank after load, post-processing and warm-up.

Scope per rank: every named parameter and registered buffer of the target model and of the
draft model (``remove_duplicate=False``, so shared tensors such as the draft's embedding and
``shared_head.head`` appear under every name and record ``alias_of``), plus plain tensor
attributes of their modules (kind ``attr``: post-processing products and runtime scratch such
as ``topk_indices_buffer``; reported separately because scratch legitimately changes).
KV-cache tensors are skipped by storage identity and by name.

Bytes hashed: the logical contents in C order (``t.contiguous().view(torch.uint8)``), never
storage padding outside the view. The copy is chunked: contiguous tensors are hashed through
fixed host buffers; non-contiguous tensors are made contiguous in row blocks of at most one
chunk. No tensor is ever copied whole to the host. Per 256 MiB segment a CRC32 is kept so a
mismatch can be localised.

RPCs attached to ``vllm.v1.worker.gpu_worker.Worker`` (all args are strings, as the
``/collective_rpc`` dev endpoint passes them). None of them raises: on this vLLM build an
exception or an unknown method in a collective_rpc kills the EngineCore, so every error is
returned as ``{"error": ...}``.

  glm_phash_status()                       -> attach receipt, rank, pending labels
  glm_phash_run(label, opts_json)          -> hashes until done or ``max_seconds``; call again
                                              with the same label to continue; writes
                                              <out_dir>/<label>/rank<tp>.json when done
  glm_phash_probe(label, names_json)       -> first/last bytes, zero-byte fraction, segment
                                              CRCs for named tensors (mismatch triage)
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import zlib

SCHEMA = 1
WORKER = 'vllm.v1.worker.gpu_worker'
LABEL_RE = re.compile(r'[A-Za-z0-9_.-]{1,64}')
KV_NAME_RE = re.compile(r'(^|\.)(kv_cache|k_cache|v_cache|kv_caches)$')
ENV_KEYS = ('GLM_MTP_ONLY_LOAD', 'GLM_TARGET_SKIP_MTP', 'GLM_FAST_LOAD', 'GLM_FAST_LOAD_VERIFY',
            'GLM_MTP_FIX', 'GLM_FULL_MLA', 'GLM_DIRTY_L2', 'GLM_INDEXER_SHORTCUT', 'GLM_PARAM_HASH')
DEFAULTS = dict(chunk_mb=32, segment_mb=256, threads=6, max_seconds=120.0, mem_floor_gib=7.0,
                attrs=True, out_dir='/cache/param-hash')
STATE = {}          # label -> run state (plan, results, timings)
ATTACHED = dict(pid=None, time=None)
_LOCK = threading.Lock()


def _torch():
    import torch
    return torch


def _mem_available_kib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _ranks():
    tp = glob = None
    try:
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group().rank_in_group
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            glob = dist.get_rank()
    except Exception:
        pass
    if tp is None:
        tp = int(os.environ.get('GLM_PHASH_RANK', os.environ.get('RANK', '-1')))
    return tp, glob


class Sink:
    """SHA-256 over all bytes plus CRC32 per fixed-size segment of the logical byte stream."""

    def __init__(self, segment_bytes):
        self.sha = hashlib.sha256()
        self.segment = segment_bytes
        self.crcs = []
        self.crc = 0
        self.fill = 0
        self.total = 0

    def update(self, buf):
        mv = memoryview(buf).cast('B')
        self.sha.update(mv)
        self.total += len(mv)
        pos = 0
        while pos < len(mv):
            take = min(self.segment - self.fill, len(mv) - pos)
            self.crc = zlib.crc32(mv[pos:pos + take], self.crc)
            self.fill += take
            pos += take
            if self.fill == self.segment:
                self.crcs.append('%08x' % self.crc)
                self.crc = 0
                self.fill = 0

    def result(self):
        crcs = list(self.crcs)
        if self.fill or not crcs:
            crcs.append('%08x' % self.crc)
        return self.sha.hexdigest(), crcs


def _blocks(t, max_bytes, stats):
    """Yield 1-D contiguous uint8 tensors whose concatenation is t's logical bytes in C order."""
    if t.numel() == 0:
        return
    if t.is_contiguous():
        flat = t.reshape(-1).view(_torch().uint8)
        for i in range(0, flat.numel(), max_bytes):
            yield flat[i:i + max_bytes]
        return
    if t.dim() == 0:
        yield t.reshape(1).contiguous().view(_torch().uint8)
        return
    row_bytes = (t.numel() // t.shape[0]) * t.element_size()
    if row_bytes <= max_bytes:
        step = max(1, max_bytes // max(1, row_bytes))
        for i in range(0, t.shape[0], step):
            blk = t[i:i + step]
            if not blk.is_contiguous():
                blk = blk.contiguous()
                stats['temp_blocks'] += 1
            yield blk.reshape(-1).view(_torch().uint8)
        return
    for i in range(t.shape[0]):
        yield from _blocks(t[i], max_bytes, stats)


def _plain(t):
    torch = _torch()
    t = t.detach()
    if type(t) is not torch.Tensor:
        t = t.as_subclass(torch.Tensor)
    return t


def hash_tensor(t, chunk_bytes, segment_bytes, host=None, stats=None, probe=False):
    """Return (sha256, segment crcs[, probe dict]) of t's logical bytes, copying in chunks."""
    torch = _torch()
    stats = stats if stats is not None else dict(temp_blocks=0)
    t = _plain(t)
    sink = Sink(segment_bytes)
    head = bytearray()
    tail = b''
    zeros = 0
    on_gpu = t.device.type == 'cuda'
    if on_gpu and host is None:
        host = torch.empty(chunk_bytes, dtype=torch.uint8)
    ctx = torch.cuda.device(t.device) if on_gpu else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for blk in _blocks(t, chunk_bytes, stats):
            for i in range(0, blk.numel(), chunk_bytes):
                part = blk[i:i + chunk_bytes]
                n = part.numel()
                if on_gpu:
                    host[:n].copy_(part)
                    arr = host[:n].numpy()
                else:
                    arr = part.contiguous().numpy()
                sink.update(arr)
                if probe:
                    import numpy as np
                    zeros += n - int(np.count_nonzero(arr))
                    if len(head) < 64:
                        head += arr[:64 - len(head)].tobytes()
                    tail = (tail + arr[-64:].tobytes())[-64:]
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    sha, crcs = sink.result()
    if sink.total != t.numel() * t.element_size():
        raise RuntimeError('byte count mismatch %d != %d' % (sink.total, t.numel() * t.element_size()))
    if probe:
        return sha, crcs, dict(head_hex=bytes(head).hex(), tail_hex=tail.hex(),
                               zero_byte_fraction=(zeros / sink.total) if sink.total else None)
    return sha, crcs


def _key(t):
    return (t.device.type, t.device.index, t.data_ptr(), str(t.dtype), tuple(t.shape), tuple(t.stride()))


def _storage_ptr(t):
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        return None


def _models(worker):
    torch = _torch()
    runner = getattr(worker, 'model_runner', worker)
    target = runner.get_model() if hasattr(runner, 'get_model') else getattr(runner, 'model', None)
    if hasattr(runner, 'get_draft_model'):
        draft = runner.get_draft_model()
    else:
        draft = getattr(getattr(runner, 'drafter', None), 'model', None)
    out = []
    for prefix, model in (('target', target), ('draft', draft)):
        for attr in ('runnable', 'model'):
            if model is not None and not isinstance(model, torch.nn.Module) and hasattr(model, attr):
                model = getattr(model, attr)
        if model is not None:
            out.append((prefix, model))
    kv_ptrs = set()
    caches = getattr(runner, 'kv_caches', None)
    if isinstance(caches, dict):
        caches = list(caches.values())
    for c in caches or ():
        if isinstance(c, torch.Tensor):
            kv_ptrs.add(_storage_ptr(c))
    kv_ptrs.discard(None)
    return out, kv_ptrs


def enumerate_tensors(models, kv_ptrs=(), attrs=True):
    """Ordered entries (name, kind, tensor-or-None, skip-reason) over all models."""
    torch = _torch()
    entries = []
    for prefix, model in models:
        for name, p in model.named_parameters(remove_duplicate=False):
            entries.append((prefix + '.' + name, 'param', p, None))
        for name, b in model.named_buffers(remove_duplicate=False):
            if b is not None:
                entries.append((prefix + '.' + name, 'buffer', b, None))
        if not attrs:
            continue
        for mname, mod in model.named_modules(remove_duplicate=False):
            own = set(mod._parameters) | set(mod._buffers)
            for aname, val in sorted(vars(mod).items()):
                if aname in own or not isinstance(val, torch.Tensor):
                    continue
                full = prefix + '.' + (mname + '.' if mname else '') + aname
                skip = None
                if KV_NAME_RE.search(full) or _storage_ptr(val) in kv_ptrs:
                    skip = 'kv_cache'
                entries.append((full, 'attr', val, skip))
    return entries


def _describe(name, kind, t, skip):
    d = dict(name=name, kind=kind)
    if t is None:
        d['skipped'] = skip or 'none'
        return d
    d.update(dtype=str(t.dtype).replace('torch.', ''), shape=list(t.shape), stride=list(t.stride()),
             contiguous=bool(t.is_contiguous()), nbytes=t.numel() * t.element_size(),
             device=str(t.device), ptype=type(t).__name__)
    if skip:
        d['skipped'] = skip
    elif t.device.type == 'meta':
        d['skipped'] = 'meta'
    return d


def plan(models, kv_ptrs=(), attrs=True):
    """Manifest skeleton plus the unique tensors to hash (largest first)."""
    rows, first, todo = [], {}, []
    for name, kind, t, skip in enumerate_tensors(models, kv_ptrs, attrs):
        d = _describe(name, kind, t, skip)
        if 'skipped' not in d:
            k = _key(t)
            if k in first:
                d['alias_of'] = first[k]
            else:
                first[k] = name
                todo.append((name, t))
        rows.append(d)
    todo.sort(key=lambda x: -x[1].numel() * x[1].element_size())
    return rows, todo


def canonical_digest(rows, kinds):
    h = hashlib.sha256()
    for d in rows:
        if d['kind'] in kinds:
            h.update(json.dumps([d['name'], d['kind'], d.get('dtype'), d.get('shape'), d.get('sha256'),
                                 d.get('alias_of'), d.get('skipped')], separators=(',', ':')).encode() + b'\n')
    return h.hexdigest()


def finalize(rows, hashes):
    """Fill sha256/crc into rows (aliases inherit their first name's hash)."""
    for d in rows:
        if 'skipped' not in d:
            sha, crcs = hashes[d.get('alias_of', d['name'])]
            d['sha256'] = sha
            if len(crcs) > 1:
                d['seg_crc32'] = crcs
    return rows


def build_manifest(models, kv_ptrs=(), opts=None, header=None):
    """Synchronous single-call manifest (used by tests and by the RPC once a plan completes)."""
    o = dict(DEFAULTS, **(opts or {}))
    rows, todo = plan(models, kv_ptrs, o['attrs'])
    stats = dict(temp_blocks=0)
    hashes = {n: hash_tensor(t, int(o['chunk_mb'] * 2 ** 20), int(o['segment_mb'] * 2 ** 20), stats=stats)
              for n, t in todo}
    return _assemble(rows, hashes, o, dict(header or {}, temp_blocks=stats['temp_blocks']))


def _assemble(rows, hashes, o, header):
    rows = finalize(rows, hashes)
    totals = {}
    for d in rows:
        t = totals.setdefault(d['kind'], dict(entries=0, hashed_unique=0, bytes_unique=0, aliases=0, skipped=0))
        t['entries'] += 1
        if 'skipped' in d:
            t['skipped'] += 1
        elif 'alias_of' in d:
            t['aliases'] += 1
        else:
            t['hashed_unique'] += 1
            t['bytes_unique'] += d['nbytes']
    return dict(schema=SCHEMA, header=dict(header, chunk_mb=o['chunk_mb'], segment_mb=o['segment_mb']),
                totals=totals,
                digest_params_buffers=canonical_digest(rows, ('param', 'buffer')),
                digest_attrs=canonical_digest(rows, ('attr',)),
                entries=rows)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    os.replace(tmp, path)


def _summary(state, extra=None):
    m = state.get('manifest')
    s = dict(label=state['label'], tp_rank=state['tp_rank'], global_rank=state['global_rank'],
             host=state['host'], done=m is not None, hashed=len(state['hashes']), todo=len(state['todo']),
             bytes_hashed=state['bytes'], elapsed_s=round(state['elapsed'], 2), calls=state['calls'],
             mem_min_gib=None if state['mem_min'] is None else round(state['mem_min'] / 1048576, 2))
    if m is not None:
        s.update(path=state['path'], digest_params_buffers=m['digest_params_buffers'],
                 digest_attrs=m['digest_attrs'], totals=m['totals'])
    s.update(extra or {})
    return s


def _run(worker, label, opts_json):
    torch = _torch()
    if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
        return dict(error='bad label')
    o = dict(DEFAULTS)
    o.update(json.loads(opts_json or '{}'))
    unknown = set(o) - set(DEFAULTS)
    if unknown:
        return dict(error='unknown options %s' % sorted(unknown))
    chunk = int(o['chunk_mb'] * 2 ** 20)
    seg = int(o['segment_mb'] * 2 ** 20)
    if chunk < 64 or chunk % 64 or seg < chunk or not 1 <= int(o['threads']) <= 16:
        return dict(error='bad chunk/segment/threads')
    with _LOCK:
        state = STATE.get(label)
        if state is None:
            models, kv_ptrs = _models(worker)
            if not models:
                return dict(error='no models found on worker')
            cfg = getattr(worker, 'vllm_config', None)
            if getattr(cfg, 'speculative_config', None) is not None and 'draft' not in dict(models):
                return dict(error='speculative config present but no draft model found')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rows, todo = plan(models, kv_ptrs, o['attrs'])
            tp, glob = _ranks()
            out = Path(o['out_dir']) / label / ('rank%d.json' % tp)
            state = STATE[label] = dict(
                label=label, opts=o, rows=rows, todo=todo, hashes={}, bytes=0, elapsed=0.0, calls=0,
                tp_rank=tp, global_rank=glob, host=socket.gethostname(), path=str(out), manifest=None,
                mem_min=_mem_available_kib(), temp=dict(temp_blocks=0), started=time.time(),
                models={p: type(m).__name__ for p, m in models})
        if state['manifest'] is not None:
            return _summary(state)
        state['calls'] += 1
        t0 = time.monotonic()
        deadline = t0 + float(o['max_seconds'])
        floor = float(o['mem_floor_gib']) * 1048576
        err = None
        lock = threading.Lock()
        it = iter(list(state['todo']))
        local = threading.local()

        def work():
            nonlocal err
            first = True        # every thread hashes at least one tensor per call: progress guarantee
            while True:
                with lock:
                    if err is not None or (time.monotonic() > deadline and not first):
                        return
                    first = False
                    mem = _mem_available_kib()
                    if mem is not None:
                        state['mem_min'] = mem if state['mem_min'] is None else min(state['mem_min'], mem)
                        if mem < floor:
                            err = 'MemAvailable %.2f GiB below floor' % (mem / 1048576)
                            return
                    nxt = next(it, None)
                if nxt is None:
                    return
                name, t = nxt
                if t.device.type == 'cuda' and getattr(local, 'host', None) is None:
                    local.host = torch.empty(chunk, dtype=torch.uint8)
                try:
                    res = hash_tensor(t, chunk, seg, host=getattr(local, 'host', None), stats=state['temp'])
                except Exception as exc:  # recorded, never raised
                    with lock:
                        err = '%s: %r' % (name, exc)
                    return
                with lock:
                    state['hashes'][name] = res
                    state['bytes'] += t.numel() * t.element_size()

        threads = [threading.Thread(target=work, name='glm-phash-%d' % i, daemon=True)
                   for i in range(int(o['threads']))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        state['todo'] = [(n, t) for n, t in state['todo'] if n not in state['hashes']]
        state['elapsed'] += time.monotonic() - t0
        if err is None and not state['todo']:
            if state['temp']['temp_blocks'] and torch.cuda.is_available():
                torch.cuda.empty_cache()
            header = dict(label=label, tp_rank=state['tp_rank'], global_rank=state['global_rank'],
                          host=state['host'], pid=os.getpid(), models=state['models'],
                          env={k: os.environ.get(k) for k in ENV_KEYS},
                          torch=torch.__version__, vllm=_vllm_version(), started=state['started'],
                          finished=time.time(), elapsed_s=round(state['elapsed'], 2), calls=state['calls'],
                          temp_blocks=state['temp']['temp_blocks'],
                          mem_min_gib=None if state['mem_min'] is None else round(state['mem_min'] / 1048576, 2))
            m = _assemble(state['rows'], state['hashes'], o, header)
            _write(Path(state['path']), m)
            state['manifest'] = m
            state['todo'] = []
        return _summary(state, dict(error=err) if err else None)


def _vllm_version():
    try:
        import vllm
        return getattr(vllm, '__version__', None)
    except Exception:
        return None


def rpc_run(self, label, opts_json='{}'):
    try:
        return _run(self, label, opts_json)
    except BaseException as exc:  # never propagate into the executor
        return dict(error='unhandled %r' % (exc,), label=str(label)[:64])


def rpc_status(self=None):
    try:
        tp, glob = _ranks()
        return dict(attached=ATTACHED, tp_rank=tp, global_rank=glob, host=socket.gethostname(),
                    labels={k: dict(done=v['manifest'] is not None, todo=len(v['todo'])) for k, v in STATE.items()},
                    mem_available_gib=round((_mem_available_kib() or 0) / 1048576, 2))
    except BaseException as exc:
        return dict(error='unhandled %r' % (exc,))


def rpc_probe(self, label, names_json):
    try:
        state = STATE.get(label)
        if state is None:
            return dict(error='unknown label')
        names = json.loads(names_json)
        if not isinstance(names, list) or len(names) > 64:
            return dict(error='names must be a list of <= 64 names')
        models, kv_ptrs = _models(self)
        tensors = {n: t for n, _, t, s in enumerate_tensors(models, kv_ptrs, True) if t is not None}
        chunk = int(state['opts']['chunk_mb'] * 2 ** 20)
        seg = int(state['opts']['segment_mb'] * 2 ** 20)
        out = {}
        for n in names:
            t = tensors.get(n)
            if t is None:
                out[n] = dict(error='not found')
                continue
            sha, crcs, extra = hash_tensor(t, chunk, seg, probe=True)
            out[n] = dict(extra, sha256=sha, seg_crc32=crcs, dtype=str(t.dtype), shape=list(t.shape))
        return dict(tp_rank=state['tp_rank'], host=state['host'], probes=out)
    except BaseException as exc:
        return dict(error='unhandled %r' % (exc,))


def attach(module):
    cls = getattr(module, 'Worker', None)
    if cls is None:
        raise RuntimeError('glm-phash: %s has no Worker' % WORKER)
    cls.glm_phash_run = rpc_run
    cls.glm_phash_status = rpc_status
    cls.glm_phash_probe = rpc_probe
    ATTACHED.update(pid=os.getpid(), time=time.time())
    sys.stderr.write('glm-phash: worker methods attached (pid %d)\n' % os.getpid())
    sys.stderr.flush()


def register(env=None):
    env = os.environ if env is None else env
    flag = env.get('GLM_PARAM_HASH', '0')
    if flag not in ('0', '1'):
        raise ValueError('GLM_PARAM_HASH must be 0 or 1')
    if flag == '0':
        return False
    import importlib.abc
    import importlib.util

    class _Hook(importlib.abc.MetaPathFinder):
        _glm_param_hash = True

        def find_spec(self, name, path, target=None):
            if name != WORKER:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def patched(module):
                exec_module(module)
                attach(module)

            spec.loader.exec_module = patched
            return spec

    if WORKER in sys.modules:
        attach(sys.modules[WORKER])
    elif not any(getattr(h, '_glm_param_hash', False) for h in sys.meta_path):
        sys.meta_path.insert(0, _Hook())
    return True


# ---------------------------------------------------------------- comparison (Mac side)

def _norm(name):
    return re.sub(r'\.\d+\.', '.N.', name)


def compare(a, b):
    """Compare two manifests of the same rank. Returns a dict with a verdict and lists by class."""
    ea = {(d['name'], d['kind']): d for d in a['entries']}
    eb = {(d['name'], d['kind']): d for d in b['entries']}
    res = dict(missing_in_a=[], missing_in_b=[], meta_diff=[], hash_diff=[], alias_diff=[], skip_diff=[],
               attr_hash_diff=[], attr_other_diff=[])
    for k in sorted(set(ea) | set(eb)):
        da, db = ea.get(k), eb.get(k)
        name, kind = k
        if da is None or db is None:
            (res['missing_in_a'] if da is None else res['missing_in_b']).append(dict(name=name, kind=kind))
            continue
        attr = kind == 'attr'
        if da.get('skipped') != db.get('skipped'):
            res['attr_other_diff' if attr else 'skip_diff'].append(
                dict(name=name, kind=kind, a=da.get('skipped'), b=db.get('skipped')))
            continue
        if (da.get('dtype'), da.get('shape')) != (db.get('dtype'), db.get('shape')):
            res['attr_other_diff' if attr else 'meta_diff'].append(
                dict(name=name, kind=kind, a=[da.get('dtype'), da.get('shape')], b=[db.get('dtype'), db.get('shape')]))
            continue
        if da.get('alias_of') != db.get('alias_of'):
            res['attr_other_diff' if attr else 'alias_diff'].append(
                dict(name=name, kind=kind, a=da.get('alias_of'), b=db.get('alias_of')))
        if da.get('sha256') != db.get('sha256'):
            segs = None
            if da.get('seg_crc32') and db.get('seg_crc32'):
                segs = [i for i, (x, y) in enumerate(zip(da['seg_crc32'], db['seg_crc32'])) if x != y]
            row = dict(name=name, kind=kind, dtype=da.get('dtype'), shape=da.get('shape'),
                       nbytes=da.get('nbytes'), alias_of=da.get('alias_of'), diff_segments=segs)
            res['attr_hash_diff' if attr else 'hash_diff'].append(row)
    decisive = ('missing_in_a', 'missing_in_b', 'meta_diff', 'hash_diff', 'alias_diff', 'skip_diff')
    missing_attr = [r for r in res['missing_in_a'] + res['missing_in_b'] if r['kind'] == 'attr']
    for key in ('missing_in_a', 'missing_in_b'):
        res[key] = [r for r in res[key] if r['kind'] != 'attr']
    res['attr_other_diff'] += [dict(r, missing=True) for r in missing_attr]
    groups = {}
    for key in decisive + ('attr_hash_diff', 'attr_other_diff'):
        for r in res[key]:
            g = groups.setdefault(key, {})
            g[_norm(r['name'])] = g.get(_norm(r['name']), 0) + 1
    res['groups'] = groups
    res['counts'] = {k: len(res[k]) for k in decisive + ('attr_hash_diff', 'attr_other_diff')}
    res['digest_equal'] = a.get('digest_params_buffers') == b.get('digest_params_buffers')
    res['verdict'] = 'EQUAL' if not any(res[k] for k in decisive) else 'DIFFERENT'
    return res
