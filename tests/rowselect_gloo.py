# SPDX-License-Identifier: Apache-2.0
"""Real four CPU peers: rowselect attachment and exact INIT votes."""
from datetime import timedelta
from pathlib import Path
import sys
import os
import socket
import tempfile
from types import SimpleNamespace as NS
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/bringup'),str(ROOT/'overlay/kstop'),str(ROOT/'overlay/overlay')]
import glm_draft_head as dh
import kstop_runtime as kr


def peer(rank,path):
    # file:// rendezvous plus an explicitly bound loopback device on Mac.
    if sys.platform == 'darwin':
        os.environ['GLOO_SOCKET_IFNAME']='lo0'
        os.environ['MASTER_ADDR']='127.0.0.1'
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4,timeout=timedelta(seconds=30))
    kr.tp_group=lambda:NS(ranks=list(range(4)),cpu_group=dist.group.WORLD)
    vote=lambda payload,valid:dh.agree(payload,valid,group=dist.group.WORLD)
    gather=dh.gather_receipts
    try:
        for case in ('same','descriptor_drift','prepare_failure'):
            ready=False
            try:
                vote({'rowselect':1,'tail_rows':3 if case=='descriptor_drift' and rank==2 else 4},
                     not(case=='prepare_failure' and rank==1))
                ready=True
            except RuntimeError:pass
            assert ready==(case=='same')
        # Execute the actual shared INIT full-M/selected gate on four ranks.
        # One bad rank must prevent every READY, including the final vote.
        from unittest.mock import patch
        from test_rowselect_init import setup
        from test_draft_head_init import fixture as head_fixture
        for case in ('good', 'flip', 'hidden_flip', 'proposal', 'unwritten', 'kv', 'logits', 'stale', 'final_vote'):
            check,_ = setup(case if rank == 2 and case in ('flip','hidden_flip','proposal','unwritten','kv','logits','stale') else None,
                            wide_hidden=case=='hidden_flip')
            head = head_fixture();head.speculator = check.speculator
            def init_vote(payload, valid):
                assert not head._draft_head_ready
                vote(payload, valid and not (case=='final_vote' and rank==2 and 'initial_qualified' in payload))
            with patch.dict(sys.modules, {'vllm.config.compilation':NS(CUDAGraphMode=NS(FULL='FULL',NONE='NONE'))}), \
                 patch.object(dh,'gather_receipts',lambda value,group=None:gather(value,dist.group.WORLD)):
                try:
                    dh.qualify_initial(head,vote=init_vote,sync=lambda:None,mem=lambda:8<<30,
                        collect=lambda:None,empty=lambda:None)
                except RuntimeError:
                    assert case not in ('good',)
                else:
                    assert case in ('good',)
                assert head._draft_head_ready == (case in ('good',))
        if rank == 0:
            print('TP4 real INIT: full-M/selected + two-state flip refusal; token/unwritten/KV/logits/stale/final-vote refusal; READY false until completion on4/4 PASS', flush=True)
        if rank==0:print('TP4 Gloo: attachment agreement and exact INIT/refusal PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    if sys.platform == 'darwin':
        os.environ['GLOO_SOCKET_IFNAME']='lo0'
        os.environ['MASTER_ADDR']='127.0.0.1'
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0))
                probe.listen(4)
        except OSError as exc:
            print(f'SKIP TP4 Gloo: loopback sockets unavailable; GLOO_RULE: {exc}',flush=True)
            sys.exit(0)
    with tempfile.TemporaryDirectory() as d:mp.spawn(peer,args=(str(Path(d)/'join'),),nprocs=4,join=True)
