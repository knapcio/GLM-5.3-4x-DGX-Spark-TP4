# SPDX-License-Identifier: Apache-2.0
"""Fast checkpoint loader and opt-in real attention NVFP4 startup."""
import os
if os.environ.get('GLM_ATTN_WEIGHTS', 'int8') != 'int8':
    try:
        import glm_nvfp4_attn
        glm_nvfp4_attn.register()
    except BaseException:
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(78)
if os.environ.get('GLM_FAST_LOAD') == '1' or os.environ.get('GLM_LOADER'):
    try:
        import glm_fast_load
        glm_fast_load.register()
    except BaseException:
        import sys, traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(78)
