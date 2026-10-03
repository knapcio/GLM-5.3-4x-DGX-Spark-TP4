# SPDX-License-Identifier: Apache-2.0
"""Offline four-rank Gloo check of count-only K2 decisions and prepare guards."""
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/kstop'))
import kstop_runtime as rt


def worker(rank, rendezvous, control, mismatch):
    os.environ.update(GLM_MTP_KSTOP_CONTROL=control,GLM_MTP_KSTOP_CONTROL_MODE='local',
        GLM_PAD_HYGIENE='1' if mismatch=='on' or (mismatch=='padding' and rank) else '0',
        GLM_MTP_KSTOP_UNIFORM_BATCH='1' if mismatch=='policy' and rank else 'k2',GLM_INDEXER_SHORTCUT='0',GLM_MTP_KSTOP_CAPTURE_LAYOUT='m12' if mismatch=='layout' and rank else 'reuse')
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=datetime.timedelta(seconds=30))
    rt.tp_group=lambda:NS(rank_in_group=rank,ranks=list(range(4)),cpu_group=dist.group.WORLD)
    try:
        r=rt.Runtime(NS(max_num_reqs=4,device='cpu'))
        if mismatch not in (False,'on'):
            try:r.begin(NS(req_ids=['a'],num_reqs=1))
            except RuntimeError as e:assert 'collective-safe kstop refusal at control' in str(e)
            else:raise AssertionError('different uniform policies, capture layouts or padding flags agreed')
            return
        for n in (1,2,3,4):
            ids=list('abcd'[:n]);r.begin(NS(req_ids=ids,num_reqs=n))
            for step in (1,2):
                if not r.can_advance(step,n):break
                # c1 stops normally; multi-request confidences differ across
                # ranks and are deliberately ignored by the count-only rule.
                r.confidence[:n].fill_(.5 if n==1 else .1+rank*.2)
                r.arm(step,n);r.packed_host(torch.arange(n));r.after_metadata(step,n)
            r.finish(torch.ones(n,3,dtype=torch.int64))
            want=1 if n==1 else 2
            assert r.lengths==[want]*n
            runner=NS(speculator=NS(_kstop=r),device='cpu')
            output=NS(finished_req_ids=set(),preempted_req_ids=set(),scheduled_new_reqs=[],
                scheduled_spec_decode_tokens={rid:[-1]*3 for rid in ids},
                num_scheduled_tokens=dict.fromkeys(ids,4),total_num_scheduled_tokens=4*n)
            view=rt.select_inputs(runner,output)
            assert view.total_num_scheduled_tokens==(want+1)*n
            assert not runner._kstop_needs_remap
            rt.STATE=dict(runtime=r,rows=torch.arange(16),src=torch.arange(16),dead=torch.zeros(16,dtype=torch.bool))
            batch=NS(req_ids=ids,num_reqs=n,num_tokens=(want+1)*n,query_start_loc_np=list(range(0,(want+1)*n+1,want+1)),
                num_tokens_after_padding=(want+1)*n if n<=2 else 12,is_prefilling_np=[False]*n)
            rt.guard_inputs(runner,batch,False);r.packed_host(torch.arange(n));rt.prepare(runner,batch,False)
            assert rt.STATE['identity'] and r.bad is None and r.guard_pending is None and r.fallback_syncs==0
            snapshot=[*r.lengths,r.guards,r.folds]
            tensor=torch.tensor(snapshot,dtype=torch.int64)
            peers=[torch.empty_like(tensor) for _ in range(4)];dist.all_gather(peers,tensor)
            assert all(torch.equal(tensor,x) for x in peers),'rank decisions or guards differed'
    finally:dist.destroy_process_group()


if __name__=='__main__':
    assert Path('/.dockerenv').exists(),'pinned CPU container required'
    with tempfile.TemporaryDirectory() as tmp:
        control=Path(tmp)/'control.json';control.write_text(json.dumps(dict(schema=1,mode='k3-stop',epoch=0,tau=.74)))
        for mismatch in (False,'on','policy','layout','padding'):
            mp.spawn(worker,args=(str(Path(tmp)/f'group-{mismatch}'),str(control),mismatch),nprocs=4,join=True)
    print('K2 FOUR-RANK PASS: c1/c2/c3/c4, exact c2 M6 and c3 M12 padding, prepare guards, pad off/on, divergent-policy/layout/padding refusal')
