"""CPU test: glm_fast_load yields the stock names, order, dtypes, shapes and bytes.

uv run --with torch --with safetensors --with numpy python tests/test_glm_fast_load.py
"""
import gc
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay", "overlay"))

os.environ["GLM_FAST_LOAD_SLAB_MB"] = "1"      # small slabs: many slab boundaries + pass-through
os.environ["GLM_FAST_LOAD_AHEAD_MB"] = "3"
os.environ["GLM_FAST_LOAD_THREADS"] = "2"
os.environ["GLM_FAST_LOAD_VERIFY"] = "5"

import torch  # noqa: E402
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from safetensors.torch import safe_open, save_file  # noqa: E402

import glm_fast_load as gfl  # noqa: E402


def make_files(d, nfiles=3):
    rng = random.Random(0)
    paths = []
    dtypes = [torch.bfloat16, torch.float32, torch.uint8, torch.float8_e4m3fn, torch.int32, torch.float16]
    for fi in range(nfiles):
        tensors = {}
        for i in range(120):
            dt = rng.choice(dtypes)
            kind = rng.random()
            if kind < 0.05:
                shape = []                                  # scalar
            elif kind < 0.08:
                shape = [0, 4]                              # empty
            elif kind < 0.12:
                shape = [700, 1024]                         # > 1 MiB slab for >= 2-byte dtypes
            else:
                shape = [rng.randint(1, 300), rng.randint(1, 257)]
            name = f"model.layers.{fi}.mlp.experts.{i}.{rng.choice(['gate_proj', 'up_proj', 'down_proj'])}.w{i}"
            if dt.is_floating_point:
                t = torch.randn(shape).to(dt) if dt != torch.float8_e4m3fn else torch.randn(shape).clamp(-4, 4).to(dt)
            else:
                t = torch.randint(0, 100, shape, dtype=dt)
            tensors[name] = t
        p = os.path.join(d, f"model-{fi:05d}.safetensors")
        save_file(tensors, p)
        paths.append(p)
    return paths


def stock(paths, skip=None):
    for p in paths:
        with safe_open(p, framework="pt") as f:
            for n in f.keys():
                if skip and skip(n):
                    continue
                yield n, f.get_tensor(n)


def as_bytes(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def main():
    with tempfile.TemporaryDirectory() as d:
        paths = make_files(d)
        skip = lambda n: n.endswith("7")  # noqa: E731
        ref = [(n, t.clone()) for n, t in stock(paths, skip)]

        def keys_of(p):
            with safe_open(p, framework="pt") as f:
                return list(f.keys())

        st = gfl._Stats()
        got = []
        for n, t in gfl.fast_safetensors_iterator(paths, keys_of, lambda p: safe_open(p, framework="pt"),
                                                  skip=skip, stats=st):
            got.append((n, t.clone()))
        assert [n for n, _ in got] == [n for n, _ in ref], "order/names differ"
        for (n, a), (_, b) in zip(ref, got):
            assert a.dtype == b.dtype and a.shape == b.shape, (n, a.dtype, b.dtype, a.shape, b.shape)
            assert torch.equal(as_bytes(a), as_bytes(b)), n
        assert st.passthrough > 0 and st.eager > 0 and st.slabs > 3 and st.verified > 0, vars(st)
        del got, t
        gc.collect()
        assert not st.live(), "slabs still referenced after the load"
        gfl.release(st, "test", wait_s=0)

        # Early close of the generator must not leak threads or descriptors.
        it = gfl.fast_safetensors_iterator(paths, keys_of, lambda p: safe_open(p, framework="pt"))
        for _ in range(10):
            next(it)
        it.close()
        print(f"OK: {len(ref)} tensors, {st.eager} eager, {st.passthrough} pass-through, {st.slabs} slabs, "
              f"{st.verified} verified")


if __name__ == "__main__":
    main()


def test_wiring():
    """The vLLM wrapper: stock signature, lazy strategy only, EP filter, other strategies untouched."""
    import types

    from tqdm.auto import tqdm

    calls = []

    def orig(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy=None, local_expert_ids=None, *,
             safetensors_prefetch_num_threads=8, safetensors_prefetch_block_size=1):
        calls.append(safetensors_load_strategy)
        yield from stock(sorted(hf_weights_files))

    wu = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    wu.__file__ = os.path.join(os.environ["GLM_IMAGE_SRC"], "vllm/model_executor/model_loader/weight_utils.py")
    wu.safetensors_weights_iterator = orig
    wu._natural_sort_key = lambda p: p
    wu.tqdm = tqdm
    wu.enable_tqdm = lambda use: False
    wu._BAR_FORMAT = "{desc}"
    wu.should_skip_weight = lambda n, ids: ids is not None and n.endswith("3")
    gfl._patch_weight_utils(wu)
    assert wu.safetensors_weights_iterator is not orig
    gfl._patch_weight_utils(wu)  # idempotent
    with tempfile.TemporaryDirectory() as d:
        paths = make_files(d, 2)
        ref = [(n, t.clone()) for n, t in stock(paths, lambda n: n.endswith("3"))]
        got = [(n, t.clone()) for n, t in wu.safetensors_weights_iterator(
            list(reversed(paths)), True, None, local_expert_ids={1},
            safetensors_prefetch_num_threads=8, safetensors_prefetch_block_size=1)]
        assert [n for n, _ in got] == [n for n, _ in ref]
        assert all(torch.equal(as_bytes(a), as_bytes(b)) for (_, a), (_, b) in zip(ref, got))
        assert calls == []
        list(wu.safetensors_weights_iterator(paths, True, "eager"))
        assert calls == ["eager"]
    print("OK: wiring")


if __name__ == "__main__":
    test_wiring()
    import time as _t
    _t.sleep(1.5)
