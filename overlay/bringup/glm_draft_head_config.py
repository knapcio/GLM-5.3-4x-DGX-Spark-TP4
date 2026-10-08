# SPDX-License-Identifier: Apache-2.0
"""Cold launch validation and exact persistent head accounting."""
def options(env):
    value = env.get('GLM_DRAFT_HEAD', '0')
    if value not in ('0', 'nvfp4', 'int8'):
        raise ValueError('GLM_DRAFT_HEAD must be 0, nvfp4 or int8')
    if value != '0' and (env.get('VLLM_USE_V2_MODEL_RUNNER') != '1' or
                         env.get('GLM_MTP_KSTOP') != '1'):
        raise ValueError('draft head requires native V2 MTP kstop')
    initial_on(env)
    return value


def byte_cost(kind, n=38720, k=6144, sms=48):
    if n % 64 or k % 128 or kind not in ('nvfp4', 'int8'):
        raise ValueError('qualified head geometry/format required')
    packed = n*k//2 if kind == 'nvfp4' else n*k
    scales = n*k//16 if kind == 'nvfp4' else n*k//128*2
    metadata = 4 if kind == 'nvfp4' else 16
    return dict(packed=packed, scales=scales, metadata=metadata, workspace=4*sms,
                total=packed+scales+metadata+4*sms, bf16_retained=2*n*k)


def initial_on(env):
    value = env.get('GLM_DRAFT_HEAD_INIT', '0')
    if value not in ('0', '1'):
        raise ValueError('GLM_DRAFT_HEAD_INIT must be 0 or 1')
    if value == '1' and env.get('GLM_DRAFT_HEAD', '0') == '0':
        raise ValueError('draft head initial ON requires a quantized bank')
    return value == '1'
