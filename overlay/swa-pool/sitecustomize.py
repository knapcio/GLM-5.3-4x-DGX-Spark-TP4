# SPDX-License-Identifier: Apache-2.0
"""Independent drafter pool, followed by the loader startup."""
import importlib.util
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
if os.environ.get('GLM_DSA_SWA_POOL') == '1':
    import glm_dsa_swa_pool
    glm_dsa_swa_pool.register()
spec = importlib.util.spec_from_file_location('_glm_loader_startup', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'overlay/sitecustomize.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
