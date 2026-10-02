# SPDX-License-Identifier: Apache-2.0
"""GLM_FULL_MLA=triton: lazy, source-pinned sparse MLA replacement.

Inert unless enabled. Refuses drift and unsupported layouts. Kernel errors
propagate; the broken stock FP8/RoPE launch is never used as error recovery.
"""
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

TARGET = 'vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90'
PIN = '4449ea25921dcae1ec7988581ff26a6ff5a9e5d6d4a0ac4a68a0017ea108f136'


def check_source(path):
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != PIN:
        raise RuntimeError(f'glm-full-mla: source drift {path}: {got} != {PIN}')


def install(mod):
    check_source(mod.__file__)
    cls = mod.FlashInferMLASparseSM90Impl
    if getattr(cls.forward_mqa, '_glm_full_mla', False):
        return
    from glm_full_mla_kernel import sparse_mla

    def forward(self, q, cache, md, layer):
        import torch
        if not isinstance(q, tuple) or len(q) != 2 or md is None:
            raise RuntimeError('glm-full-mla: split queries and metadata required')
        if self.num_heads != 16 or self.kv_lora_rank != 512 or self.qk_rope_head_dim not in (0, 64):
            raise RuntimeError('glm-full-mla: unsupported dimensions')
        if getattr(self, 'dcp_world_size', 1) != 1:
            raise RuntimeError('glm-full-mla: DCP unsupported')
        if self.kv_cache_dtype not in ('auto', 'bfloat16', 'fp8', 'fp8_e4m3'):
            raise RuntimeError('glm-full-mla: packed cache unsupported')
        qn, qr = q
        t = qn.shape[0]
        slots, counts = mod.triton_convert_req_index_to_global_index(
            md.req_id_per_token[:t], md.block_table, self.topk_indices_buffer[:t],
            BLOCK_SIZE=md.block_size, NUM_TOPK_TOKENS=self.topk_indices_buffer.shape[1],
            return_valid_counts=True)
        fp8 = self.kv_cache_dtype in ('fp8', 'fp8_e4m3')
        if fp8:
            cache = cache.view(torch.float8_e4m3fn)
        ks = float(layer._k_scale_float or 1.0) if fp8 else 1.0
        if os.environ.get('GLM_MLA_SPLIT_K','0') != '0' and t <= int(os.environ.get('GLM_MLA_SPLIT_MAX_ROWS','36')):
            from glm_full_mla_split_kernel import sparse_mla as split_mla
            return split_mla(qn,qr,cache,slots,self.scale,ks), None
        return sparse_mla(qn, qr, cache, slots, self.scale, ks), None

    forward._glm_full_mla = True
    cls.forward_mqa = forward
    sys.stderr.write('glm-full-mla: ARMED triton H16 D512 R0|64; capture enabled\n')


def register(env=None):
    env = os.environ if env is None else env
    mode = env.get('GLM_FULL_MLA', '0')
    if mode == '0':
        return False
    if mode != 'triton':
        raise RuntimeError('GLM_FULL_MLA must be 0 or triton')
    target_name = TARGET
    installer = install
    if target_name in sys.modules:
        installer(sys.modules[target_name])
        return True

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                raise ImportError(TARGET)
            if mode == 'triton':
                check_source(spec.origin)
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    installer(module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    return True
