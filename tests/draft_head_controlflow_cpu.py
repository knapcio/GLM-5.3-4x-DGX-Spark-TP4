# SPDX-License-Identifier: Apache-2.0
"""CPU-only native controlflow adapter: allow V2 with substituted Triton math."""
from pathlib import Path
import runpy
import sys
assert Path('/.dockerenv').exists(), 'pinned CPU container only'
import vllm.config.vllm as config
# The historical harness replaces every CUDA/Triton execution with CPU math.
# The no-driver image disables Triton before its fake-SM121 metadata is set.
config.HAS_TRITON = True
print('CPU harness only: V2 Triton availability probe substituted', flush=True)
ROOT=Path(__file__).resolve().parents[1]
sys.argv=sys.argv[1:]
runpy.run_path(str(ROOT/sys.argv[0]),run_name='__main__')
