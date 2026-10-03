# SPDX-License-Identifier: Apache-2.0
"""Cold import hooks on/off in separate processes, using the real pinned modules."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def child():
    sys.path.insert(0, str(ROOT / 'overlay/bringup'))
    import glm_spec_sample as S
    active = S.register()
    modules = {n: importlib.import_module(n) for n in S.PINS}
    for mod in modules.values():
        S.check_source(mod)
    cfg = modules[S.CONFIG].SpeculativeConfig
    kernel = modules[S.REJECTION]._resample_kernel
    assert bool(getattr(cfg.__post_init__, '_glm_spec_sample', False)) == active
    assert kernel.__module__ == ('glm_spec_sample_kernel' if active else S.REJECTION)
    print(json.dumps({'import_PASS': True, 'flag': int(active), 'kernel': kernel.__module__}))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child()
    else:
        for flag in ('0', '1'):
            subprocess.run([sys.executable, '-B', __file__, '--child'], check=True,
                           env=dict(os.environ, GLM_SPEC_SAMPLE=flag))
