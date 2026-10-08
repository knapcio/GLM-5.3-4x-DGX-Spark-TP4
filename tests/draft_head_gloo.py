# SPDX-License-Identifier: Apache-2.0
"""Four real ARM64 CPU Gloo ranks; production control with CPU graph fixtures."""
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_draft_head import runner,prepare_cpu
import glm_draft_head as d


def worker(rank,path,fault,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4,
        timeout=datetime.timedelta(seconds=25))
    r=runner();prepare_cpu(r)
    if rank==2:
        if fault=='busy':r.execute_model_state=object()
        if fault=='guard':r.speculator._kstop.guard_pending=object()
        if fault=='ready':r._draft_head_ready=False
        if fault=='reset':
            class BadGraph:
                def reset(self):raise RuntimeError('rank2 reset')
            r.speculator.decode_cudagraph_manager.graphs['M1']=BadGraph()
        if fault=='capture':r.speculator.capture=lambda:(_ for _ in ()).throw(ValueError('rank2'))
    result=[]
    try:
        for epoch,on in enumerate([1,0]*10):
            d.switch(r,0 if fault=='payload' and rank==2 else on,epoch,True,
                vote=lambda payload,valid:d.agree(payload,valid,dist.group.WORLD),
                sync=lambda:None,mem=lambda:5*(1<<30) if fault=='memory' and rank==2 else 8*(1<<30),
                collect=lambda:None,empty=lambda:None)
            result.append(on)
        verdict='PASS'
    except RuntimeError:verdict='REFUSED'
    Path(output,f'{rank}.json').write_text(json.dumps(dict(rank=rank,fault=fault,
        verdict=verdict,rounds=result,on=r._draft_head.on,ready=r._draft_head_ready)))
    dist.destroy_process_group()


if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    for fault in ('none','busy','guard','ready','payload','memory','capture','reset'):
        p=out/fault;p.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(worker,args=(rank_path:=temp+'/gloo',fault,str(p)),nprocs=4,join=True)
        rows=[json.loads((p/f'{r}.json').read_text()) for r in range(4)]
        assert all(r['verdict']==('PASS' if fault=='none' else 'REFUSED') for r in rows),rows
        if fault in ('capture','reset'):assert all(not r['ready'] for r in rows)
        print(json.dumps(dict(fault=fault,results=rows)),flush=True)
