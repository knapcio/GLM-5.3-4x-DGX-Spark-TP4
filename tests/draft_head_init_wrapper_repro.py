# SPDX-License-Identifier: Apache-2.0
"""CPU reproducer: production DSA/DH wrapper order, baseline and fixed factory."""
import json, subprocess, sys, types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop')]
import glm_draft_head as fixed
import glm_dsa_short as dsa
baseline=types.ModuleType('dh_baseline')
source=Path(sys.argv[1]).read_text() if len(sys.argv)>1 else subprocess.check_output(['git','show','9cd0116:overlay/bringup/glm_draft_head.py'],cwd=ROOT,text=True)
exec(compile(source,'9cd0116/glm_draft_head.py','exec'),baseline.__dict__)
@dataclass(frozen=True)
class Desc:
    short_context:bool=True

def check(dh,order):
    seen=[]
    def native(self,factory):factory(Desc(),False)('FULL')
    cls=type('CudaGraphManager',(),dict(capture=native,dispatch=lambda *a:None))
    mod=NS(CudaGraphManager=cls,graph_capture=lambda **kw:None)
    if order=='dsa-first':dsa.install_cg(mod);dh.install_graph(mod)
    else:dh.install_graph(mod);dsa.install_cg(mod)
    manager=object.__new__(cls)
    factory=lambda desc,warmup:lambda mode:seen.append(dsa.SHORT.get())
    with dsa.context(False):
        manager.capture(factory)
        manager._k4_drafthead_factory(Desc(),False)('NONE')
    return dict(captured_short=seen[0],eager_short=seen[1],equal=seen[0]==seen[1])
rows=[dict(order=order,baseline=check(baseline,order),fixed=check(fixed,order))
      for order in ('dsa-first','dh-first')]
assert rows[0]['baseline']['equal'] is False
assert all(r['fixed']['equal'] for r in rows)
print(json.dumps(dict(baseline='9cd0116',scope='actual production DSA/DH callbacks; CPU forward substitutes',results=rows),indent=2))
