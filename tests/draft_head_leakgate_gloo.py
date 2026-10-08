# SPDX-License-Identifier: Apache-2.0
"""Four CPU ranks execute the installed worker RPC; no CUDA/HTTP substitutes."""
import datetime
import copy
from functools import partial
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_draft_head import runner, prepare_cpu
import glm_draft_head as dh


def worker(rank, path, fault, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=40))
    r=runner();prepare_cpu(r)
    original_bank=r._draft_head.bank
    r._draft_head=NS(on=False, bank=original_bank, original=NS(weight=NS(device=NS(type='cuda'))))
    calls=[]
    def change(r,on,epoch,drained):
        calls.append(on);r._draft_head.on=bool(on);r._draft_head_epoch=epoch
        if fault=='transition' and rank==2 and len(calls)==1:
            valid=False
        else:valid=True
        dh.agree({'transition':len(calls)},valid,dist.group.WORLD)
        if fault=='weights' and rank==2 and len(calls)==5:original_bank.weight.add_(1)
    def sample(r):
        checked=len(calls)==5 and rank==2
        if checked and fault=='bank':r._draft_head.bank=copy.deepcopy(original_bank)
        if checked and fault=='bank_storage':original_bank.workspace=torch.zeros(99,dtype=torch.int32)
        if checked and fault=='target':r.cudagraph_manager.graphs['target']=object()
        return dict(allocated=None if checked and fault=='sample' else (100<<20)+((9<<20) if checked and fault=='allocated' else 0),
                    reserved=(120<<20)+((17<<20) if checked and fault=='reserved' else 0),
                    mem_available=(4<<30) if checked and fault=='memory' else (8<<30))
    def vote(payload,valid):
        if fault=='payload' and rank==2 and payload.get('leakcheck_step')==2:
            payload=dict(payload,on=0)
        dh.agree(payload,valid and not (fault=='initial' and rank==2),dist.group.WORLD)
    callback=partial(dh.leakcheck,vote=vote,sync=lambda:None,collect=lambda:None,
                     empty=lambda:None,sample=sample)
    cls=type('Worker',(),{});dh.install_worker(NS(Worker=cls));w=cls();w.model_runner=r
    fake=ModuleType('vllm.distributed');fake.get_tp_group=lambda:NS(rank_in_group=rank,cpu_group=dist.group.WORLD)
    with patch.dict(sys.modules,{'vllm.distributed':fake}),patch.object(dh,'switch',change),patch.object(dh,'leakcheck',callback),patch.object(dh,'telemetry',lambda r:{}):
        # Exact registered callback is invoked as by WorkerProc. No exception may
        # escape it: the pinned executor treats callback errors as engine-fatal.
        receipt=w.draft_head_leakcheck('true')
        alive=w.draft_head_status()['rank']==rank
    if fault=='none':
        assert receipt['ok'] and len(receipt['samples'])==22 and len(calls)==24 and not r._draft_head.on
    else:
        assert receipt['refused'] and not r._draft_head_ready and not r._k4_drafthead_ready and alive
        if fault not in ('initial','transition'):
            assert len(receipt['all_rank_samples'])==4 and len(receipt['samples'])==3
            assert any(f['rank']==2 for f in receipt['failed_conditions'])
            assert {s['rank'] for s in receipt['all_rank_samples']}=={0,1,2,3}
    Path(output,f'{rank}.json').write_text(json.dumps(dict(fault=fault,alive=alive,receipt=receipt),indent=2)+'\n')
    dist.destroy_process_group()

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    for fault in ('none','allocated','reserved','sample','payload','memory','bank','bank_storage','weights','target','initial','transition'):
        p=out/fault;p.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:mp.spawn(worker,args=(tmp+'/vote',fault,str(p)),nprocs=4,join=True)
        print('LEAKGATE '+fault+' PASS ranks=4 alive=4',flush=True)
