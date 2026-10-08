# SPDX-License-Identifier: Apache-2.0
"""Four CPU Gloo ranks execute INIT; rank2 faults produce collective OFF fallback."""
import datetime,json,sys,tempfile
from unittest.mock import patch
from contextlib import ExitStack
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_draft_head_init import fixture,boot
from dataclasses import replace
from test_draft_head_replay_criterion import descriptor_fixture, PrefillDesc
import glm_draft_head as dh
AGREE=dh.agree
GATHER=dh.gather_receipts

def worker(rank,path,fault,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4,
        timeout=datetime.timedelta(seconds=25))
    if fault.startswith('prefill_'):
        # SHORT override on all ranks; only rank 2 omits the feedback store.
        descriptor = replace(PrefillDesc(), short_context=True) if fault in ('prefill_short_sparse_partial_write', 'prefill_short_one_omit') else PrefillDesc()
        local_fault = fault.removeprefix('prefill_').removeprefix('short_') if rank == 2 else 'two_state_flip'
        r = descriptor_fixture(local_fault, descriptor)
        if 'logits' in fault:
            r.speculator.draft_logits = torch.full((4, 3, 64), float('nan'))
    else:
        r = fixture(fault if rank == 2 else None)
    votes=[]
    equal=torch.equal
    def vote(payload,valid):
        votes.append(dict(stage=next(iter(payload)),valid=bool(valid),ready=r._draft_head_ready))
        if rank==2 and fault=='final_vote' and 'initial_qualified' in payload:valid=False
        AGREE(payload,valid,dist.group.WORLD)
    def compare(a,b):
        if rank==2 and fault=='compare_exception' and r._draft_head.on and a.shape==(1,256):
            raise RuntimeError('injected output comparison error')
        return equal(a,b)
    def memory():
        terms=dh._qualification_terms.get() or []
        return 4<<30 if rank==2 and fault=='final_memory' and r._draft_head.on and terms and terms[-1]['check']=='initial_target_equality' else 8<<30
    def cleanup():
        if rank==2 and fault=='cleanup' and dh._qualification_terms.get() is not None:
            raise RuntimeError('injected cleanup error')
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(dh,'gather_receipts',
                lambda value, group=None: GATHER(value, dist.group.WORLD)))
            stack.enter_context(patch.object(torch,'equal',compare))
            stack.enter_context(patch.object(dh.gc,'collect',cleanup))
            boot(r,vote,mem=memory);verdict='PASS'
    except RuntimeError:verdict='FATAL'
    else:
        if not r._draft_head.on:verdict='FALLBACK'
    if not r._draft_head.on:
        assert r.speculator._kstop.bad is None
        # Cold status and a normal OFF draft forward remain callable.
        manager=dh.managers(r)[0]
        desc=next(iter(manager.graphs))
        manager._k4_drafthead_factory(desc,False)(None)
        assert torch.isfinite(r.speculator.hidden_states).all()
    Path(output,f'{rank}.json').write_text(json.dumps(dict(rank=rank,fault=fault,verdict=verdict,
        failure=dh.status(r)['init_failure'], dh_gate=dh.status(r)['dh_gate_passed'], on=r._draft_head.on,epoch=r._draft_head_epoch,ready=r._draft_head_ready,votes=votes)))
    dist.destroy_process_group()

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    passing = {'none', 'prefill_two_state_flip', 'prefill_float_drift'}
    for fault in ('none','mismatch','nonfinite','missing','target','inputs_exception','execution_exception',
                  'compare_exception','final_memory','cleanup','final_vote',
                  'prefill_two_state_flip','prefill_float_drift',
                  'prefill_short_one_omit','prefill_short_sparse_partial_write',
                  'prefill_noop','prefill_stale_hidden','prefill_stale_all','prefill_stale_row',
                  'prefill_nondeterministic','prefill_stateful','prefill_nonfinite','prefill_tokens',
                  'prefill_eager_tokens','prefill_graph2_tokens','prefill_decode_tokens_unwritten',
                  'prefill_row_not_written','prefill_element_not_written','prefill_feedback_error',
                  'prefill_float_seventeen_ulp','prefill_diff_fraction','prefill_confidence_error',
                  'prefill_hidden_nonfinite','prefill_logits_error','prefill_logits_nonfinite',
                  'prefill_logits_nondeterministic','prefill_inactive_logits',
                  'prefill_graph2_excess','prefill_roundtrip_tokens','prefill_roundtrip_error'):
        p=out/fault;p.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:mp.spawn(worker,args=(temp+'/gloo',fault,str(p)),nprocs=4,join=True)
        rows=[json.loads((p/f'{r}.json').read_text()) for r in range(4)]
        assert all(r['verdict']==('PASS' if fault in passing else 'FALLBACK') for r in rows),rows
        assert all(r['ready']==(fault in passing) and r['on']==(fault in passing) and r['epoch']==-1 for r in rows),rows
        assert all(r['dh_gate']==(fault in passing) for r in rows),rows
        if fault not in passing:assert all(r['failure']['votes'] for r in rows),rows
        assert all(not v['ready'] for r in rows for v in r['votes']),rows
        print(json.dumps(dict(fault=fault,results=rows)),flush=True)
