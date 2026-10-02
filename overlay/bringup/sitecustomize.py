# SPDX-License-Identifier: Apache-2.0
"""Full GLM startup: SWA pool, fast loader, MLA, capture guard, draft loader, short DSA, dirty L2."""
import importlib.util
import os
import sys
try:
    tail = os.environ.get('GLM_FLASH_SITECUSTOMIZE', '/overlay/swa-pool/sitecustomize.py')
    if os.environ.get('GLM_DSA_SWA_POOL') == '1':
        spec = importlib.util.spec_from_file_location('_glm_pool_startup', tail)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif os.environ.get('GLM_FAST_LOAD') == '1':
        import glm_fast_load
        glm_fast_load.register()
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
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(78)
