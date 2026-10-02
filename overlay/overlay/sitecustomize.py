# SPDX-License-Identifier: Apache-2.0
"""Only the fast checkpoint loader is enabled in this profile."""
import os
if os.environ.get('GLM_FAST_LOAD') == '1':
    import glm_fast_load
    glm_fast_load.register()
