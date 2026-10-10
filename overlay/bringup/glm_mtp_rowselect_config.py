# SPDX-License-Identifier: Apache-2.0
"""Boot-only native MTP pass-1 tail selection; no tensor imports."""


def options(env):
    value = env.get('GLM_MTP_ROWSELECT', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_MTP_ROWSELECT must be 0 or 1 (boot only)')
    if value == '0':
        return False
    required = {'GLM_MTP_KSTOP': '1', 'GLM_MTP_FIX': '1',
                'VLLM_USE_V2_MODEL_RUNNER': '1', 'GLM_DRAFT_HEAD': 'nvfp4',
                'GLM_DRAFT_HEAD_INIT': '1', 'GLM_MOE_DET_ALIGN': '1'}
    if any(env.get(k) != v for k, v in required.items()):
        raise ValueError('rowselect requires det-align, cand3 V2 native K-stop and INIT NVFP4 draft head')
    if env.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT') != 'reuse':
        raise ValueError('rowselect requires qualified cand3 reuse capture layout')
    return True


def descriptor_rows(num_tokens, num_reqs, max_num_reqs=4):
    selected = num_reqs or min(num_tokens, max_num_reqs)
    if not 0 < selected <= num_tokens:
        raise ValueError('invalid rowselect descriptor')
    return dict(input_rows=num_tokens, kv_write_rows=num_tokens,
                attention_rows=num_tokens, tail_rows=selected,
                discarded_tail_rows=num_tokens-selected)
