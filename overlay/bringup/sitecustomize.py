# SPDX-License-Identifier: Apache-2.0
"""Full GLM startup: opt-in param hash RPCs, SWA pool, fast loader, MLA, capture guard, draft loader, short DSA, dirty L2, glue-lite, MTP K-stop, MLA plan skip, carveout KV."""
import importlib.util
import os
import sys
try:
    if os.environ.get('GLM_PARAM_HASH', '0') != '0':
        import glm_param_hash
        glm_param_hash.register()
    tail = os.environ.get('GLM_FLASH_SITECUSTOMIZE', '/overlay/swa-pool/sitecustomize.py')
    if os.environ.get('GLM_DSA_SWA_POOL') == '1':
        spec = importlib.util.spec_from_file_location('_glm_pool_startup', tail)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif os.environ.get('GLM_FAST_LOAD') == '1':
        import glm_fast_load
        glm_fast_load.register()
    if os.environ.get('GLM_SPEC_SAMPLE', '0') != '0':
        import glm_spec_sample
        glm_spec_sample.register()
    if os.environ.get('GLM_MTP_FIX') == '1':
        import glm_mtp_fix
        glm_mtp_fix.register()
    if os.environ.get('GLM_FULL_MLA', '0') != '0':
        import glm_full_mla
        glm_full_mla.register()
    if os.environ.get('GLM_FULL_MLA') == 'triton':
        import glm_window_memory
        glm_window_memory.register()
    if os.environ.get('GLM_DRAFT_LOWMEM') == '1':
        import glm_draft_lowmem
        glm_draft_lowmem.register()
    if os.environ.get('GLM_W2_PREFILL_CONTROL'):
        import glm_prefill_switch
        glm_prefill_switch.register()
    if os.environ.get('GLM_INDEXER_SHORTCUT', '0') != '0':
        import glm_dsa_short
        glm_dsa_short.register()
    if os.environ.get('GLM_DIRTY_L2', '0') != '0':
        import glm_dirty_l2
        glm_dirty_l2.register()
    if os.environ.get('GLM_GLUE_LITE_BANKABLE', '0') != '0':
        raise RuntimeError('GLM_GLUE_LITE_BANKABLE needs an external in-boot A/B harness; not supported here')
    glue = [k for k in ('GLM_GLUE_ROUTER_BF16', 'GLM_GLUE_MOE_WS', 'GLM_GLUE_DSA_IDX_CACHE')
            if os.environ.get(k, '0') != '0']
    if glue:
        if any(os.environ[k] != '1' for k in glue):
            raise ValueError('GLM_GLUE_ROUTER_BF16, GLM_GLUE_MOE_WS and GLM_GLUE_DSA_IDX_CACHE must be 0 or 1')
        import glm_glue_lite
        glm_glue_lite.register()
    if os.environ.get('GLM_MTP_KSTOP', '0') != '0':
        # Kstop owns the three shared transforms when the shortcut is enabled.
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kstop'))
        import glm_mtp_kstop
        glm_mtp_kstop.register()
    if os.environ.get('GLM_SKIP_MLA_PLAN', '0') not in ('', '0'):
        import glm_skip_mla_plan
        glm_skip_mla_plan.register()
    if os.environ.get('GLM_DISPRAM_KV', '0') != '0':
        import glm_dispram_kv
        glm_dispram_kv.register()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(78)
