# SPDX-License-Identifier: Apache-2.0
"""Four real Gloo ranks run FP8 refusal rollback and single-rank refusal."""
import os,sys,tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop')]
import glm_draft_head as dh
import glm_draft_ehproj as eh
from test_draft_ehproj_rollback import RollbackTests

def peer(rank,join):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+join,world_size=4,rank=rank,timeout=timedelta(seconds=60))
    agree=dh.agree
    vote=lambda payload,valid:agree(payload,valid,group=dist.group.WORLD)
    try:
        RollbackTests.setUpClass()
        for case in ('restored','reload_hash','memory'):
            test=RollbackTests();test.setUp()
            try:
                runner=test.runner()
                # Every rank sees the same checkpoint bytes at the same path.
                shared=Path(join).parent/'shared.safetensors'
                if rank==0:shared.write_bytes(test.path.read_bytes())
                dist.barrier()
                runner._draft_ehproj_source['path']=str(shared)
                original_sha=eh.tensor_sha
                def sha(weight):
                    result=original_sha(weight)
                    return '0'*64 if case=='reload_hash' and rank==2 else result
                def memory():
                    return int(4.5*(1<<30))+runner._draft_ehproj_source['bytes'] if case=='memory' and rank==2 else 8<<30
                with patch.object(eh,'tensor_sha',sha):
                    try:test.boot(runner,vote=vote,mem=memory)
                    except RuntimeError:
                        assert case!='restored'
                        assert not runner._draft_head_ready and runner.speculator._kstop.bad
                    else:
                        assert case=='restored'
                        test.assert_native(runner)
                vote({'rollback_case':case},True)
                if rank==0:print('TP4 rollback '+case+' PASS',flush=True)
            finally:test.doCleanups()
    finally:dist.destroy_process_group()

if __name__=='__main__':
    if sys.platform=='darwin':os.environ.update(GLOO_SOCKET_IFNAME='lo0',MASTER_ADDR='127.0.0.1')
    with tempfile.TemporaryDirectory() as d:mp.spawn(peer,args=(str(Path(d)/'join'),),nprocs=4,join=True)
