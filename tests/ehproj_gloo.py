# SPDX-License-Identifier: Apache-2.0
"""Four local CPU peers: identical debit, mismatched owner, failed preparation."""
import os,sys,tempfile
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/overlay')]
from test_draft_ehproj import fixture,bank,prepare
import glm_draft_ehproj as eh
import glm_draft_head as dh

def peer(rank,path):
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4,timeout=timedelta(seconds=30))
    vote=lambda payload,valid:dh.agree(payload,valid,group=dist.group.WORLD)
    try:
        r,_=fixture();prepare(r,factory=bank,vote=vote)
        assert r._draft_ehproj_receipt['delta']==-75484992
        for case in ('owner','prepare'):
            r,original=fixture(shared=case=='owner' and rank==2)
            def factory(o):
                if case=='prepare' and rank==1:raise RuntimeError('local preparation error')
                return bank(o)
            try:prepare(r,factory=factory,vote=vote)
            except RuntimeError:pass
            else:raise AssertionError('collective refusal required')
            assert r.speculator.model.model.layers['78'].eh_proj is original
        if rank==0:print('Four-peer Gloo prepare PASS; owner drift/local failure refuse all before attachment',flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':
    if sys.platform=='darwin':os.environ.update(GLOO_SOCKET_IFNAME='lo0',MASTER_ADDR='127.0.0.1')
    with tempfile.TemporaryDirectory() as d:mp.spawn(peer,args=(str(Path(d)/'join'),),nprocs=4,join=True)
