# SPDX-License-Identifier: Apache-2.0
"""GLM_SPEC_SAMPLE=1: pinned probabilistic MTP + independent residual noise.

Boot-time switch: the stock speculator allocates its logits cache at construction.
Stock gumbel_sample uses argmax on temperature-zero rows, including mixed batches.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

PINS = json.loads(Path(__file__).with_name('spec_sample_pins.json').read_text())
CONFIG = 'vllm.config.speculative'
REJECTION = 'vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils'
RESIDUAL_SALT = 0x53504543


def enabled(env=None):
    env = os.environ if env is None else env
    value = env.get('GLM_SPEC_SAMPLE', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_SPEC_SAMPLE must be 0 or 1')
    return value == '1'


def check_source(module):
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != PINS[module.__name__]:
        raise RuntimeError('GLM_SPEC_SAMPLE source drift: ' + module.__name__)


def install(module):
    check_source(module)
    if module.__name__ == CONFIG:
        cls = module.SpeculativeConfig
        if getattr(cls.__post_init__, '_glm_spec_sample', False):
            return
        original = cls.__post_init__

        @functools.wraps(original)
        def initialize(self):
            if (self.method != 'mtp' or self.rejection_sample_method != 'standard'
                    or self.use_local_argmax_reduction):
                raise ValueError('GLM_SPEC_SAMPLE requires native MTP, standard rejection, full logits')
            self.draft_sample_method = 'probabilistic'
            original(self)

        initialize._glm_spec_sample = True
        cls.__post_init__ = initialize
    elif module.__name__ == REJECTION:
        from glm_spec_sample_kernel import _resample_kernel
        module._resample_kernel = _resample_kernel
        sys.stderr.write('GLM_SPEC_SAMPLE: probabilistic T>0, independent residual noise\n')


class Hook(importlib.abc.MetaPathFinder):
    """Compose with existing overlay finders and refuse source drift."""
    def __init__(self):
        self.busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in PINS or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None:
            raise ImportError(name)
        original = spec.loader.exec_module

        def execute(module):
            original(module)
            install(module)

        spec.loader.exec_module = execute
        return spec


def register(env=None):
    if not enabled(env):
        return False
    if not any(isinstance(h, Hook) for h in sys.meta_path):
        sys.meta_path.insert(0, Hook())
    for name in PINS:
        if name in sys.modules:
            install(sys.modules[name])
    return True
