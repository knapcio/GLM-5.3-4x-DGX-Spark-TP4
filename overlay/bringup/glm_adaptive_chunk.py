# SPDX-License-Identifier: Apache-2.0
"""Central scheduler budgets after APC lookup; worker capacity stays 4096."""
import ast
import hashlib
import os
from pathlib import Path

PIN = '4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a'
SMALL, LARGE, THRESHOLD = 2048, 4096, 16384


def settings(env):
    mode = env.get('GLM_PREFILL_CHUNK_ADAPTIVE', '1' if env.get('GLM_KV_FORMAT') == 'fp4x' else '0')
    if mode not in ('0', '1'):
        raise ValueError('GLM_PREFILL_CHUNK_ADAPTIVE must be 0 or 1')
    threshold = int(env.get('GLM_PREFILL_CHUNK_THRESHOLD', str(THRESHOLD)))
    if threshold <= 0:
        raise ValueError('GLM_PREFILL_CHUNK_THRESHOLD must be positive')
    return mode == '1', threshold


class StepBudget:
    def __init__(self, scheduler, threshold):
        capacity = getattr(scheduler, '_w2_prefill_capacity',
                           scheduler.scheduler_config.max_num_batched_tokens)
        if capacity != LARGE:
            raise RuntimeError('adaptive prefill requires fixed 4096 constructor capacity')
        self.limit = min(scheduler.max_num_scheduled_tokens, SMALL)
        self.input_limit = min(scheduler.scheduler_config.max_num_batched_tokens, SMALL)
        self.threshold = threshold
        # The 512-token boot/admission phase never expands.
        self.serving = self.limit == SMALL and self.input_limit == SMALL

    def request(self, request, computed, token_budget, input_budget):
        remaining = request.num_prompt_tokens - computed
        cap = SMALL if remaining > 0 else LARGE
        # Total attention context includes APC hits and earlier prefill chunks.
        context_tokens = computed + remaining
        if self.serving and remaining > 0 and context_tokens >= self.threshold:
            cap = LARGE
            token_budget += LARGE - self.limit
            input_budget += LARGE - self.input_limit
            self.limit = self.input_limit = LARGE
        return token_budget, input_budget, cap


def schedule_source(source):
    """Extract only schedule; retain all native admission/reservation logic."""
    if hashlib.sha256(source.encode()).hexdigest() != PIN:
        raise RuntimeError('adaptive prefill scheduler source drift')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    return ast.unparse(fn)


def transform(source):
    text = schedule_source(source)
    replacements = (
        ('token_budget = self.max_num_scheduled_tokens',
         '_glm_budget = _glm_budget_factory(self)\n    token_budget = _glm_budget.limit'),
        ('input_budget = self.scheduler_config.max_num_batched_tokens',
         'input_budget = _glm_budget.input_limit'),
        ('num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)',
         'token_budget, input_budget, _glm_cap = _glm_budget.request(request, request.num_computed_tokens, token_budget, input_budget)\n'
         '        num_new_tokens = min(num_new_tokens, _glm_cap, token_budget, input_budget - draft_slots)'),
        ('request_token_budget = min(token_budget, input_budget - draft_slots)',
         'token_budget, input_budget, _glm_cap = _glm_budget.request(request, num_computed_tokens, token_budget, input_budget)\n'
         '                request_token_budget = min(_glm_cap, token_budget, input_budget - draft_slots)'),
        ('assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens',
         'assert total_num_scheduled_tokens <= _glm_budget.limit'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError('adaptive prefill transform anchor drift: ' + old)
        text = text.replace(old, new)
    return text


def install(module, env=None):
    enabled, threshold = settings(os.environ if env is None else env)
    if not enabled:
        return False
    source = Path(module.__file__).read_text()
    namespace = dict(module.__dict__, _glm_budget_factory=lambda s: StepBudget(s, threshold))
    exec(compile('from __future__ import annotations\n' + transform(source), module.__file__, 'exec'), namespace)
    module.Scheduler.schedule = namespace['schedule']
    return True
