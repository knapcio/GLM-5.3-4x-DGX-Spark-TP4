"""Local composition check against a supplied perf/nvfp4-attn source checkout.

python tests/test_coalesced_nvfp4.py /path/to/glm53-full-nvfp4a
Uses that branch's real converter/manifest/transform on toy matrices only.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'overlay/overlay'))
from test_coalesced_load import C, F, keys, manifest, safe_open, stock


def main(source):
    spec = importlib.util.spec_from_file_location('nvfp4_composition_fixture', source / 'tests/test_nvfp4_attn.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    os.environ.update(GLM_LOADER='coalesced', GLM_COALESCED_DIRECT='0',
                      GLM_COALESCED_BATCH_MB='1', GLM_COALESCED_OWNED_MB='8')
    with tempfile.TemporaryDirectory() as d:
        source_dir, sidecar = Path(d) / 'checkpoint', Path(d) / 'sidecar'
        names = fixture.f.MODULES[:2]
        fixture.fixture(source_dir, names)
        fixture.c.convert(source_dir, sidecar, modules=names)
        files = [str(p) for p in sorted(source_dir.glob('*.safetensors'))]
        original = fixture.a.transform(stock(files), files, sidecar, names)
        loaded = C.coalesced_safetensors_iterator(files, keys, lambda p: safe_open(p, framework='pt'))
        transformed = fixture.a.transform(loaded, files, sidecar, names)
        reference, candidate = manifest(original), manifest(transformed)
        assert reference == candidate
        # Exercise the merged registration path, including explicit coalesced
        # selection with GLM_FAST_LOAD unset. The sidecar transform stays outermost.
        def orig(paths,*args,**kwargs):return stock(paths)
        wu=types.SimpleNamespace(safetensors_weights_iterator=orig,_natural_sort_key=lambda p:p,
            tqdm=lambda fs,**kw:fs,enable_tqdm=lambda _:False,_BAR_FORMAT='',should_skip_weight=lambda n,ids:False)
        real_transform=fixture.a.transform
        def small_transform(weights,paths,root):return real_transform(weights,paths,root,names)
        with patch.dict(os.environ,GLM_ATTN_WEIGHTS='nvfp4',GLM_ATTN_NVFP4_DIR=str(sidecar)), \
             patch.object(F,'_check_source'),patch.object(fixture.a,'transform',side_effect=small_transform):
            os.environ.pop('GLM_FAST_LOAD',None)
            with patch.object(C,'coalesced_safetensors_iterator',wraps=C.coalesced_safetensors_iterator) as coalesced:
                F._patch_weight_utils(wu)
                first=wu.safetensors_weights_iterator;F._patch_weight_utils(wu)
                assert first is wu.safetensors_weights_iterator
                assert manifest(first(files,False,'lazy'))==reference
                assert coalesced.call_count==1, 'explicit coalesced must not fall back to stock'
            with patch.dict(os.environ,GLM_LOADER='fast',GLM_FAST_LOAD_PINNED='0'):
                assert manifest(first(files,False,'lazy'))==reference
            # Lazy strategies other than the two reader modes retain stock bytes.
            assert manifest(first(files,False,'eager'))==reference
        print(f'NVFP4 sidecar composition and registered fast/coalesced/stock hash-EQUAL: {len(candidate)} tensors; source {source}')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT)
