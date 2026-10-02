# SPDX-License-Identifier: Apache-2.0
"""Drained scheduling-cap switch; allocation and captured graph shapes stay fixed."""
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

MOD = 'vllm.v1.core.sched.scheduler'
PIN = '4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a'


def apply(scheduler, control):
    if set(control) != {'schema', 'chunk', 'sequence'} or control['schema'] != 1:
        raise RuntimeError('prefill control schema')
    chunk, sequence = control['chunk'], control['sequence']
    if type(chunk) is not int or chunk not in (512, 2048, 4096):
        raise RuntimeError('prefill chunk must be 512/2048/4096')
    if type(sequence) is not int or sequence < 0:
        raise RuntimeError('prefill sequence')
    capacity=getattr(scheduler, '_w2_prefill_capacity', scheduler.scheduler_config.max_num_batched_tokens)
    if chunk > capacity:
        raise RuntimeError('prefill chunk exceeds allocated capacity')
    prior = getattr(scheduler, '_w2_prefill_control', None)
    if prior == control:
        return False
    if prior is not None and sequence <= prior['sequence']:
        raise RuntimeError('prefill stale sequence')
    if scheduler.running:
        raise RuntimeError('prefill switch requires drained running requests')
    scheduler._w2_prefill_capacity=capacity
    scheduler.max_num_scheduled_tokens = chunk
    # Match native scheduling's input budget, including K3's two reserved slots.
    # Worker processes keep their original4096 allocation/configuration.
    scheduler.scheduler_config.max_num_batched_tokens = chunk
    scheduler._w2_prefill_control = dict(control)
    return True


def install(module):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != PIN:
        raise RuntimeError('prefill scheduler source drift')
    original = module.Scheduler.schedule

    @functools.wraps(original)
    def schedule(self, *args, **kwargs):
        path = Path(os.environ['GLM_W2_PREFILL_CONTROL'])
        # Read atomically replaced content; missing or malformed controls fail closed.
        control = json.loads(path.read_text())
        if apply(self, control):
            print('GLM_W2_PREFILL ' + json.dumps(control, sort_keys=True), file=sys.stderr, flush=True)
        result = original(self, *args, **kwargs)
        if result.total_num_scheduled_tokens > control['chunk']:
            raise RuntimeError('prefill scheduler cap receipt failed')
        return result

    module.Scheduler.schedule = schedule


def register():
    if not os.environ.get('GLM_W2_PREFILL_CONTROL'):
        return
    if MOD in sys.modules:
        install(sys.modules[MOD])
        return

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != MOD:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                install(module)

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Hook())
