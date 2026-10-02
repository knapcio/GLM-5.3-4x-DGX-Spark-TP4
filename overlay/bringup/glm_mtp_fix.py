# SPDX-License-Identifier: Apache-2.0
"""Pinned GLM MTP quant repair. Inert unless GLM_MTP_FIX=1.

Register before vLLM imports (append register() to the existing bootstrap).
No kernels, weights, target config, or norm semantics are changed.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

NAME = 'vllm.model_executor.models.deepseek_mtp'
PIN = '88521bcf3bfec6c773998dbefe30448504640907a4ac98f4e822d0ed38e2436f'
PACKED = {'fused_qkv_a_proj': ['q_a_proj', 'kv_a_proj_with_mqa'],
          'gate_up_proj': ['gate_proj', 'up_proj'],
          'wk_weights_proj': ['wk', 'weights_proj']}


def flag(env, name):
    value = env.get(name, '0')
    if value not in ('0', '1'):
        raise ValueError(name + ' must be 0 or 1')
    return value == '1'


def speculative_config(k=3, env=None):
    env = os.environ if env is None else env
    if type(k) is not int or not 1 <= k <= 5:
        raise ValueError('K must be an integer in 1..5')
    config = dict(method='mtp', num_speculative_tokens=k,
                  draft_tensor_parallel_size=4, kv_cache_dtype='fp8_e4m3',
                  draft_sample_method='greedy', rejection_sample_method='standard',
                  attention_backend='FLASHINFER_MLA_SPARSE_SM90')
    if flag(env, 'GLM_MTP_FIX'):
        config['quantization'] = 'compressed-tensors'
    return config


def check_source(path):
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != PIN:
        raise RuntimeError('MTP source drift: ' + got)


def install(mod):
    check_source(mod.__file__)
    cls = mod.DeepSeekMTP
    current = dict(getattr(cls, 'packed_modules_mapping', {}) or {})
    for key, parts in PACKED.items():
        if key in current and current[key] != parts:
            raise RuntimeError('conflicting MTP packed mapping: ' + key)
        current[key] = list(parts)
    cls.packed_modules_mapping = current
    if getattr(cls.__init__, '_glm_mtp_fix', False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def initialize(self, *, vllm_config, prefix=''):
        sc = vllm_config.speculative_config
        hf = sc.draft_model_config.hf_config
        if (sc.method != 'mtp' or hf.num_hidden_layers != 78 or
                hf.num_nextn_predict_layers != 1 or hf.hidden_size != 6144):
            raise RuntimeError('GLM_MTP_FIX requires full GLM-5.3 native MTP')
        if sc.quantization != 'compressed-tensors':
            raise RuntimeError('explicit speculative quantization is required')
        quant = vllm_config.quant_config
        if quant is None or quant.get_name() != 'compressed-tensors':
            raise RuntimeError('draft quant config missing')
        # configure_quant_config runs before construction; verify that fact.
        for key, parts in PACKED.items():
            if quant.packed_modules_mapping.get(key) != parts:
                raise RuntimeError('draft quant mapping not configured: ' + key)
        sharing = os.environ.get('GLM_MTP_INDEX_SHARE', 'checkpoint')
        if sharing not in ('checkpoint', '0', '1'):
            raise ValueError('GLM_MTP_INDEX_SHARE must be checkpoint, 0 or 1')
        if sharing != 'checkpoint':
            hf.index_share_for_mtp_iteration = sharing == '1'
        original(self, vllm_config=vllm_config, prefix=prefix)
        sys.stderr.write('GLM_MTP_FIX loaded: packed W8A16; index_share=' +
                         str(hf.index_share_for_mtp_iteration) + '\n')

    initialize._glm_mtp_fix = True
    cls.__init__ = initialize


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('missing pinned MTP module')
        check_source(spec.origin)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            install(module)

        spec.loader.exec_module = execute
        return spec


def register(env=None):
    env = os.environ if env is None else env
    if not flag(env, 'GLM_MTP_FIX'):
        return False
    if NAME in sys.modules:
        install(sys.modules[NAME])
    elif not any(isinstance(h, Hook) for h in sys.meta_path):
        sys.meta_path.insert(0, Hook())
    return True
