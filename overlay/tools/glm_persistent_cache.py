# SPDX-License-Identifier: Apache-2.0
"""Persistent compile cache, run inside a short container of the keyed image (as root).

seed: copy the generation for this key into the fresh, empty deployment cache, if one exists.
save: publish the deployment cache as the generation for this key, if none exists yet.

The key is SHA-256 over PCACHE_PARTS (image ID, overlay tree hash, CUDA arch lists and cache
layout, from the launcher) plus the FlashInfer version read here. A generation lives in
<root>/<key>/{cache/,manifest.json,key.json,READY}; it is published by one rename and never
overwritten. READY holds the key and the SHA-256 of manifest.json; the manifest lists every
regular file (size, SHA-256) and symlink (target) of the payload.

Outcomes. No READY, or a key.json that differs from the parts: MISS/MISMATCH, the boot builds
fresh and nothing is reused. A generation whose READY, manifest or payload does not verify
(changed, missing or extra file, other file type): CORRUPT, exit 3, the boot stops; nothing is
copied, or a partial copy is removed. The payload is verified before the copy and the copied
deployment cache again after it. Output: one "PERSISTENT CACHE <STATUS> <key> ..." line.
"""
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import uuid

ROOT = os.environ.get('PCACHE_ROOT', '/persist')
CACHE = os.environ.get('PCACHE_CACHE', '/cache')
CORRUPT_EXIT = 3


class Corrupt(Exception):
    pass


def flashinfer_version():
    import flashinfer

    return flashinfer.__version__


def key_of(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def parts_from_env(fi_version=None):
    parts = json.loads(os.environ['PCACHE_PARTS'])
    parts['flashinfer'] = fi_version if fi_version is not None else flashinfer_version()
    return parts


def report(status, key, *extra):
    print(' '.join(['PERSISTENT CACHE', status, key] + [str(x) for x in extra]), flush=True)
    return status


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_manifest(tree):
    """Every regular file (size, sha256) and symlink (target) below ``tree``; other types refuse."""
    files, links = {}, {}
    for d, dirs, names in os.walk(tree):
        for name in dirs + names:
            path = os.path.join(d, name)
            rel = os.path.relpath(path, tree)
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                links[rel] = os.readlink(path)
            elif stat.S_ISREG(mode):
                files[rel] = dict(size=os.lstat(path).st_size, sha256=file_sha256(path))
            elif not stat.S_ISDIR(mode):
                raise Corrupt('unsupported file type: ' + rel)
    return dict(schema=1, files=files, links=links)


def manifest_bytes(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()


def verify_tree(tree, manifest):
    actual = build_manifest(tree)
    if actual != manifest:
        exp_f, act_f = manifest.get('files', {}), actual['files']
        diff = sorted(set(exp_f) ^ set(act_f)) + sorted(n for n in set(exp_f) & set(act_f) if exp_f[n] != act_f[n])
        diff += sorted(set(manifest.get('links', {}).items()) ^ set(actual['links'].items()))
        raise Corrupt('payload differs from manifest: %r' % (diff[:5],))


def load_generation(gen, key):
    """The verified manifest of a published generation, or Corrupt."""
    try:
        with open(os.path.join(gen, 'READY')) as f:
            ready = json.load(f)
        with open(os.path.join(gen, 'manifest.json'), 'rb') as f:
            raw = f.read()
    except (OSError, ValueError) as exc:
        raise Corrupt('unreadable READY/manifest: %s' % exc)
    if not isinstance(ready, dict) or ready.get('key') != key:
        raise Corrupt('READY does not name this key')
    if ready.get('manifest_sha256') != hashlib.sha256(raw).hexdigest():
        raise Corrupt('manifest.json does not match READY')
    manifest = json.loads(raw)
    verify_tree(os.path.join(gen, 'cache'), manifest)
    return manifest


def clear(tree):
    for name in os.listdir(tree):
        path = os.path.join(tree, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def seed(parts, root=ROOT, cache=CACHE):
    key = key_of(parts)
    gen = os.path.join(root, key)
    if not os.path.isfile(os.path.join(gen, 'READY')):
        return report('MISS', key)
    try:
        with open(os.path.join(gen, 'key.json')) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        stored = None
    if stored != parts:
        return report('MISMATCH', key)
    if os.listdir(cache):
        raise RuntimeError('deployment cache is not empty')
    try:
        manifest = load_generation(gen, key)                       # before anything is copied
        try:
            subprocess.run(['cp', '-a', os.path.join(gen, 'cache') + '/.', cache + '/'], check=True)
            verify_tree(cache, manifest)                           # what the boot will actually use
        except BaseException:
            clear(cache)
            raise
    except Corrupt as exc:
        report('CORRUPT', key, json.dumps(str(exc)), 'remove ' + gen)
        raise
    return report('HIT', key, len(manifest['files']))


def save(parts, expected_key, exclude, max_bytes, root=ROOT, cache=CACHE):
    key = key_of(parts)
    if key != expected_key:
        raise RuntimeError('key changed since seed: %s != %s' % (key, expected_key))
    gen = os.path.join(root, key)
    if os.path.exists(gen):
        return report('KEEP', key)
    size = sum(os.lstat(os.path.join(d, f)).st_size for d, _, files in os.walk(cache) for f in files)
    if size > max_bytes:
        return report('SKIP', key, size)
    tmp = os.path.join(root, '.tmp-%s-%s' % (key, uuid.uuid4().hex))
    try:
        os.makedirs(os.path.join(tmp, 'cache'))
        subprocess.run(['cp', '-a', cache + '/.', os.path.join(tmp, 'cache') + '/'], check=True)
        for name in exclude:
            path = os.path.join(tmp, 'cache', name)
            if os.path.lexists(path):
                os.unlink(path)
        # The manifest describes the private copy, so it is one coherent snapshot even if the
        # serving process writes into /cache meanwhile.
        raw = manifest_bytes(build_manifest(os.path.join(tmp, 'cache')))
        with open(os.path.join(tmp, 'manifest.json'), 'wb') as f:
            f.write(raw)
        with open(os.path.join(tmp, 'key.json'), 'w') as f:
            json.dump(parts, f, sort_keys=True)
        with open(os.path.join(tmp, 'READY'), 'w') as f:
            json.dump(dict(key=key, manifest_sha256=hashlib.sha256(raw).hexdigest()), f)
        subprocess.run(['sync'], check=True)
        try:
            os.rename(tmp, gen)
        except OSError:
            if os.path.isfile(os.path.join(gen, 'READY')):
                return report('KEEP', key)      # another deployment published this key first
            raise
        return report('SAVED', key, size)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    if len(argv) != 2 or argv[1] not in ('seed', 'save'):
        raise SystemExit('usage: glm_persistent_cache.py seed|save')
    # docker stop / timeout send SIGTERM: unwind so a partial copy or temporary is removed.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parts = parts_from_env()
    try:
        if argv[1] == 'seed':
            seed(parts)
        else:
            save(parts, os.environ['PCACHE_KEY'], json.loads(os.environ['PCACHE_EXCLUDE']),
                 int(os.environ['PCACHE_MAX']))
    except Corrupt:
        sys.exit(CORRUPT_EXIT)


if __name__ == '__main__':
    main(sys.argv)
