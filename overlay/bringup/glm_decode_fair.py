# SPDX-License-Identifier: Apache-2.0
"""Central, opt-in mixed-step budget; TP workers consume SchedulerOutput only."""
import json
from pathlib import Path
import sys

from glm_adaptive_chunk import StepBudget, LARGE

CHUNKS = (512, 1024, 2048, 4096)


def settings(env):
    mode = env.get('GLM_DECODE_FAIR', '0')
    if mode not in ('0', '1'):
        raise ValueError('GLM_DECODE_FAIR must be 0 or 1')
    chunk = int(env.get('GLM_DECODE_FAIR_CHUNK', '4096'))
    if chunk not in CHUNKS:
        raise ValueError('GLM_DECODE_FAIR_CHUNK must be 512/1024/2048/4096')
    boot_decode_steps(env)
    path = env.get('GLM_DECODE_FAIR_CONTROL', '')
    if path and (mode != '1' or not Path(path).is_absolute()):
        raise ValueError('GLM_DECODE_FAIR_CONTROL requires enabled mode and an absolute path')
    if mode == '1':
        from glm_adaptive_chunk import settings as adaptive_settings
        if not adaptive_settings(env)[0] or not env.get('GLM_W2_PREFILL_CONTROL'):
            raise ValueError('decode fairness requires adaptive prefill and GLM_W2_PREFILL_CONTROL')
    return mode == '1', chunk, path


def boot_decode_steps(env):
    value = int(env.get('GLM_DECODE_FAIR_DECODE_STEPS', '0'))
    if value < 0:
        raise ValueError('GLM_DECODE_FAIR_DECODE_STEPS must be nonnegative')
    return value


def control_chunk(scheduler, path, fallback):
    if not path:
        return fallback
    # Only EngineCore's scheduler reads this file. Atomic replace selects the
    # whole next step; workers never read it, even if their mounts differ.
    control = json.loads(Path(path).read_text())
    if (not isinstance(control, dict)
            or type(control.get('schema')) is not int or control['schema'] not in (1, 2)
            or set(control) != ({'schema', 'chunk', 'sequence'} |
                               ({'decode_steps'} if control['schema'] == 2 else set()))
            or type(control['chunk']) is not int or control['chunk'] not in (0, *CHUNKS)
            or (control['schema'] == 2 and (type(control['decode_steps']) is not int
                                          or control['decode_steps'] < 0))
            or type(control['sequence']) is not int or control['sequence'] < 0):
        raise RuntimeError('decode fair control schema/chunk/decode_steps/sequence')
    prior = getattr(scheduler, '_glm_decode_fair_control', None)
    if control != prior:
        if prior is not None and control['sequence'] <= prior['sequence']:
            raise RuntimeError('decode fair stale sequence')
        scheduler._glm_decode_fair_control = dict(control)
        # A new arm starts with a mixed chunk; subsequent chunks have an N-step floor.
        scheduler._glm_df_since_prefill = control.get('decode_steps', 0)
        print('GLM_DECODE_FAIR ' + json.dumps(control, sort_keys=True), file=sys.stderr, flush=True)
    return control['chunk']


def control_policy(scheduler, path, fallback, decode_steps=0):
    chunk = control_chunk(scheduler, path, fallback)
    return chunk, scheduler._glm_decode_fair_control.get('decode_steps', 0) if path else decode_steps


class FairStepBudget(StepBudget):
    def __init__(self, scheduler, threshold, chunk, decode_steps=0):
        super().__init__(scheduler, threshold)
        self.scheduler = scheduler
        self.chunk = chunk
        self.decode_steps = decode_steps
        self.since_prefill = getattr(scheduler, '_glm_df_since_prefill', decode_steps)
        self.prefills = {r.request_id for r in scheduler.running
                         if r.num_computed_tokens < r.num_prompt_tokens}
        spec = scheduler.vllm_config.speculative_config
        self.draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        self.decodes = {}
        if self.serving and chunk:
            for r in scheduler.running:
                if r.num_computed_tokens < r.num_prompt_tokens:
                    continue
                # Match native async eligibility/max-output guards. No worker
                # flags, memory readings, device values or acceptance estimates.
                if scheduler.current_step < r.next_decode_eligible_step:
                    continue
                if (r.num_output_placeholders > 0 and
                        r.num_computed_tokens + 2 - r.num_output_placeholders >=
                        r.num_prompt_tokens + r.max_tokens):
                    continue
                need = min(r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens,
                           scheduler.max_model_len - r.num_computed_tokens - scheduler.num_sampled_tokens_per_step)
                threshold_cap = scheduler.scheduler_config.long_prefill_token_threshold
                if threshold_cap > 0:
                    need = min(need, threshold_cap)
                if need > 0:
                    self.decodes[r.request_id] = need
        has_prefill = bool(self.prefills) or any(
            r.num_computed_tokens < r.num_prompt_tokens
            for queue in (scheduler.waiting, scheduler.skipped_waiting) for r in queue)
        self.active = bool(self.decodes) and has_prefill
        if self.active:
            # Keep allocation/capture config untouched. The aggregate prefill
            # allowance below bounds mixed work; use existing 4096 capacity to
            # leave room for decode rows and every native draft reservation.
            self.limit = self.input_limit = LARGE

    def request(self, request, computed, token_budget, input_budget, scheduled):
        if not self.active:
            return super().request(request, computed, token_budget, input_budget)
        if computed >= request.num_prompt_tokens:
            return token_budget, input_budget, LARGE
        self.prefills.add(request.request_id)
        if self.since_prefill < self.decode_steps:
            return token_budget, input_budget, 0
        used = sum(n for rid, n in scheduled.items() if rid in self.prefills)
        # Native preemption removes victims and refunds budgets. Recompute
        # reservations from live running + emitted rows, so both compose.
        pending = [self.decodes[r.request_id] for r in self.scheduler.running
                   if r.request_id in self.decodes and r.request_id not in scheduled]
        reserve = sum(pending)
        cap = max(0, min(self.chunk - used, token_budget - reserve,
                         input_budget - reserve - (len(pending) + 1) * self.draft_slots))
        return token_budget, input_budget, cap

    def finish(self, scheduled):
        # Count actual emitted steps, not eligibility or allocation attempts.
        # KV refusals/preemption/pause must not reset the floor or create credit.
        if not self.active:
            self.scheduler._glm_df_since_prefill = self.decode_steps
        elif any(scheduled.get(rid, 0) > 0 for rid in self.prefills):
            self.scheduler._glm_df_since_prefill = 0
        elif any(scheduled.get(rid, 0) > 0 for rid in self.decodes):
            self.scheduler._glm_df_since_prefill = min(self.decode_steps, self.since_prefill + 1)
