# SPDX-License-Identifier: Apache-2.0
"""Immutable boot options and replicated projection accounting."""
def options(env):
    kind=env.get('GLM_DRAFT_EHPROJ','0')
    if kind not in ('0','fp8'):
        raise ValueError('GLM_DRAFT_EHPROJ must be 0 or fp8; NVFP4 failed agreement')
    init=env.get('GLM_DRAFT_EHPROJ_INIT','0')
    if init not in ('0','1') or (kind=='0') != (init=='0'):
        raise ValueError('eh_proj requires matching boot-only format and INIT=1')
    if kind!='0' and any(env.get(k)!=v for k,v in {
        'VLLM_USE_V2_MODEL_RUNNER':'1','GLM_MTP_KSTOP':'1',
        'GLM_DRAFT_HEAD':'nvfp4','GLM_DRAFT_HEAD_INIT':'1'}.items()):
        raise ValueError('eh_proj requires cand3 native V2 kstop and INIT NVFP4 head')
    return kind


def byte_cost(n=6144,k=12288,sms=48,retain=False):
    if (n,k)!=(6144,12288) or type(sms) is not int or sms<1:
        raise ValueError('qualified replicated eh_proj geometry required')
    packed=n*k;scales=2*n;workspace=4*sms;bf16=2*n*k
    total=packed+scales+workspace
    return dict(packed=packed,scales=scales,workspace=workspace,total=total,
        bf16_source=bf16,bf16_retained=bf16 if retain else 0,
        delta=total if retain else total-bf16)
