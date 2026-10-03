# SPDX-License-Identifier: Apache-2.0
"""Four TP states with native pinned SamplingStates and the real prepare guard.

Always runs four fresh interpreters with distinct Python hash seeds and native
warmup cleanup/allocation. Default: MAX/all_gather emulated through pipes on
Mac (no sockets). --gloo also runs real Gloo in the pinned offline image.
No vLLM/CUDA imports or GPU arithmetic.
"""
import argparse
import ast
import __future__
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/kstop'))
import kstop_runtime as rt
import glm_mtp_kstop as hook

CASES=('legacy-t0','raw-t0','native-t0','native-t1','explicit-t1','mixed','signed-zero',
       'temperature','seed','mapping','width','length','invalid-mapping','shortcut-bound','shortcut-padding','divergent-t1')
FAIL={'legacy-t0':'sampling.seeds','temperature':'sampling.temperature','seed':'sampling.seeds','divergent-t1':'sampling.seeds',
      'width':'widths','length':'sampling.lengths','invalid-mapping':'valid','shortcut-bound':'short-dsa.lengths',
      'shortcut-padding':'short-dsa.padding'}


class Uva:
    """Replace only UVA storage; execute stock initialization/writes unchanged."""
    def __init__(self,n,dtype):
        self.np=torch.zeros(n,dtype=dtype).numpy();self.copy_to_uva()
    def copy_to_uva(self):self.gpu=torch.from_numpy(self.np.copy())


def native_states(source):
    name='vllm.v1.worker.gpu.sample.states'
    raw=(source/(name.replace('.','/')+'.py')).read_text()
    pins=json.loads((ROOT/'overlay/kstop/compat_source_pins.json').read_text())
    assert hashlib.sha256(raw.encode()).hexdigest()==pins[name]
    tree=ast.parse(raw);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    ns=dict(np=np,torch=torch,SamplingParams=NS,UvaBackedTensor=Uva)
    exec(compile(tree,name,'exec'),ns)
    # Keep native request admission and generated seeds unchanged.
    raw=(source/(hook.RUNNER.replace('.','/')+'.py')).read_text()
    cooked=hook.transform(hook.RUNNER,raw)
    assert '_kstop.seed_request' not in cooked
    def admission(text):
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
        return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='add_requests'))
    assert admission(raw)==admission(cooked)
    return ns['SamplingStates']


def make_case(rank,case,States):
    ss=States(4,32)
    temp=0. if case in ('legacy-t0','raw-t0','native-t0','signed-zero','width','invalid-mapping','shortcut-bound','shortcut-padding') else 1.
    owner=NS(max_num_reqs=4,device='cpu',speculative_config=NS(draft_sample_method='probabilistic'),use_fp64_gumbel=False)
    r=rt.Runtime(owner);r.mode='k3-stop';r.epoch=0
    runner=NS(device='cpu',speculator=NS(_kstop=r),sampler=NS(sampling_states=ss),model_config=NS(seed=1234))
    ids=['unseeded-warm-28315']+(['greedy-neighbor'] if case=='mixed' else [])
    mapping=[2]+([0] if case=='mixed' else [])
    raw_seeds=[]
    np.random.seed(1234)
    if case in ('legacy-t0','raw-t0','native-t0','divergent-t1'):
        np.random.random(rank)
    for i,(rid,idx) in enumerate(zip(ids,mapping)):
        params=NS(temperature=temp if i==0 else 0.,top_p=1.,min_p=0.,top_k=0,
                  seed=777 if case=='explicit-t1' else None,logprobs=None)
        # Native worker resets NumPy to the common model seed after warmup.
        # Equal draw histories agree; divergence is a guarded T>0 refusal.
        ss.add_request(idx,params)
        raw_seeds.append(int(ss.seeds.np[idx]))
        assert bool(ss.seeds_set[idx])==(case=='explicit-t1')
        if case=='explicit-t1':assert ss.seeds.np[idx]==777
    # Unused mirror and padding rows vary, including filters ignored by drafts.
    ss.temperature.np[3]=np.nan;ss.seeds.np[3]=rank*911
    ss.top_p.np[mapping]=.1+rank*.1;ss.min_p.np[mapping]=rank*.1;ss.top_k.np[mapping]=rank+1
    if rank==3:
        if case=='temperature':ss.temperature.np[2]=.7
        if case=='seed':ss.seeds.np[2]+=1
        if case=='mapping':
            ss.temperature.np[1]=ss.temperature.np[2];ss.seeds.np[1]=ss.seeds.np[2];mapping[0]=1
        if case=='signed-zero':ss.temperature.np[2]=-0.
        if case=='invalid-mapping':mapping[0]=-1
    ss.apply_staged_writes()
    for key in ('temperature','seeds'):
        assert np.array_equal(getattr(ss,key).np,getattr(ss,key).gpu.numpy(),equal_nan=True)
    n=len(ids);widths=[2048+(case=='width' and rank==3)]+([1] if n==2 else [])
    starts=np.cumsum([0,*widths,rank+19])
    batch=NS(req_ids=ids,num_reqs=n,query_start_loc_np=starts,is_prefilling_np=[True]*n+[False],
             idx_mapping_np=np.array([*mapping,-1,3]),seq_lens_cpu_upper_bound=torch.tensor([2048]*n+[rank]),
             has_prefill=True,num_tokens_after_padding=2048+n-1)
    if case=='length' and rank==3:batch.seq_lens_cpu_upper_bound[0]+=1
    if case.startswith('shortcut'):
        r.shortcut=True
        if rank==3:
            if case=='shortcut-bound':batch.seq_lens_cpu_upper_bound[0]+=1
            else:batch.num_tokens_after_padding+=1
    output=NS(finished_req_ids=set(),preempted_req_ids=set(),scheduled_new_reqs=[NS(req_id=rid) for rid in ids],
              scheduled_spec_decode_tokens={},num_scheduled_tokens=dict(zip(ids,[2048]+([1] if n==2 else []))))
    rt.select_inputs(runner,output)
    state=dict(runtime=r,runner=runner,rows=torch.arange(2052),src=torch.arange(2052),dead=torch.zeros(2052,dtype=torch.bool))
    return r,runner,batch,state,raw_seeds


def consume(rank,case,r,runner,batch,state):
    rt.STATE=state
    stderr=io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:r.packed_host(torch.tensor([2048]))
        except RuntimeError as e:
            assert case in FAIL,(case,str(e))
            assert 'collective-safe kstop refusal at prepare' in str(e)
            lines=stderr.getvalue().splitlines();assert len(lines)==1
            record=json.loads(lines[0].removeprefix('KSTOP_GUARD_FIELDS '))
            assert record['rank']==rank and len(record['peers'])==4
            assert any(FAIL[case] in key for key in record['mismatched']),(case,record)
            assert all('sampling' in ' '.join(p) for p in record['peers'])
        else:
            assert case not in FAIL,case
            rt.prepare(runner,batch,False)
            assert not stderr.getvalue() and not r.guard_inputs_log
            assert state['identity'] and r.bad is None
    assert r.guard_pending is None and r.fallback_syncs==0


def enqueue(case,runner,batch):
    if case=='legacy-t0':
        original=rt._inputs
        def legacy(r,b):
            widths,valid,payload=original(r,b)
            ss=runner.sampler.sampling_states
            payload['sampling.seeds']=[int(ss.seeds.np[x]) for x in b.idx_mapping_np[:b.num_reqs]]
            return widths,valid,payload
        with patch.object(rt,'_inputs',legacy):rt.guard_inputs(runner,batch,False)
    else:rt.guard_inputs(runner,batch,False)


def simulated(States):
    for case in CASES:
        instances=[];packets=[]
        for rank in range(4):
            instance=make_case(rank,case,States);r,runner,batch,state,_=instance
            rt.STATE=state
            with patch.object(rt,'tp_group',lambda:NS(rank_in_group=rank,ranks=list(range(4)),cpu_group=None)), \
                 patch.object(dist,'all_reduce',lambda payload,**kw:packets.append(payload.clone())):
                enqueue(case,runner,batch)
            instances.append(instance)
        if case=='raw-t0':
            # Old prepare folded these four differing active seeds unconditionally.
            assert len({json.dumps(x[4]) for x in instances})==4
            assert len({x[0].trail for x in instances})==1
        reduced=torch.stack(packets).amax(0);k=rt.GUARD_WORDS+rt.GUARD_SLOTS
        ok=(reduced[:k]==-reduced[k+1:]).all()&(reduced[k]==0)
        for r,*_ in instances:r.guard_pending=('prepare',ok)
        peers=[x[0].guard_field_digests() for x in instances] if case in FAIL else None
        def gather(out,fields,**kw):out[:]=peers
        for rank,(r,runner,batch,state,_) in enumerate(instances):
            with patch.object(rt,'tp_group',lambda:NS(rank_in_group=rank,ranks=list(range(4)),cpu_group=None)), \
                 patch.object(dist,'all_gather_object',gather):
                consume(rank,case,r,runner,batch,state)
        print('SIMULATED TP4 PASS '+case,flush=True)


def worker(rank,rendezvous,source):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,
                            timeout=datetime.timedelta(seconds=60))
    rt.tp_group=lambda:NS(rank_in_group=rank,ranks=list(range(4)),cpu_group=dist.group.WORLD)
    States=native_states(Path(source))
    try:
        for case in CASES:
            r,runner,batch,state,_=make_case(rank,case,States);rt.STATE=state
            enqueue(case,runner,batch)
            consume(rank,case,r,runner,batch,state)
            dist.barrier()
            if rank==0:print('GLOO TP4 PASS '+case,flush=True)
    finally:dist.destroy_process_group()


class Staged:
    """CPU token storage substitute; native allocation/free-list code runs."""
    def __init__(self,shape,dtype,device,**kw):self.gpu=torch.zeros(shape,dtype=dtype)
    def stage_write_elem(self,idx,value):self.gpu[idx]=value
    def stage_write(self,idx,start,values):self.gpu[idx,start:start+len(values)]=torch.tensor(values)
    def apply_write(self):pass


def native_pool(source):
    pins={'vllm.v1.worker.gpu.states':'99418f5df43ca612ded72609fee011620b065b2cab2f249ccb387096bf4ae71a',
          'vllm.v1.worker.gpu.warmup':'5b0964afbdab59a0a1fb619a3caf8611504549d4ef50643583da7b2963d34eab',
          hook.RUNNER:hook.PINS[hook.RUNNER]}
    trees={}
    for name,want in pins.items():
        raw=(source/(name.replace('.','/')+'.py')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==want,name
        trees[name]=ast.parse(raw)
    def execute(nodes,ns):
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'native-pool','exec',
                     flags=__future__.annotations.compiler_flag),ns)
    ns=dict(np=np,torch=torch,UvaBackedTensor=Uva,StagedWriteTensor=Staged)
    execute([n for n in trees['vllm.v1.worker.gpu.states'].body if isinstance(n,ast.ClassDef)],ns)
    pool=ns['RequestState'](4,32,32,3,32,'cpu')
    cls=next(n for n in trees[hook.RUNNER].body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
    execute([n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_remove_request','finish_requests')],ns)
    runner=NS(req_states=pool,model_state=NS(remove_request=lambda rid:None),lora_state=NS(remove_request=lambda rid:None),pooling_runner=None,
              pp_handler=None,encoder_cache=None,prompt_logprobs_worker=None)
    runner._remove_request=lambda rid:ns['_remove_request'](runner,rid)
    runner.finish_requests=lambda output:ns['finish_requests'](runner,output)
    # Execute the native warmup ID construction and set cleanup assignment.
    warmup=trees['vllm.v1.worker.gpu.warmup'];env=dict(num_reqs=4,cleanup_output=NS())
    assignments=[n for n in ast.walk(warmup) if isinstance(n,ast.Assign)]
    ids=next(n for n in assignments if any(isinstance(t,ast.Name) and t.id=='req_ids' for t in n.targets))
    cleanup=next(n for n in assignments if any(isinstance(t,ast.Attribute) and t.attr=='finished_req_ids' for t in n.targets)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='set')
    execute([ids,cleanup],env)
    for rid in env['req_ids']:pool.add_request(rid,8,[0]*8,0,8)
    env['cleanup_output'].preempted_req_ids=set()
    runner.finish_requests(env['cleanup_output'])
    assert pool.num_reqs==0 and len(pool.free_indices)==4
    return pool,runner


def hashseed_worker(rank,source):
    """Fresh interpreter: PYTHONHASHSEED must be set BEFORE process startup."""
    assert os.environ['PYTHONHASHSEED']==str(rank)
    pool,native_runner=native_pool(source);States=native_states(source)
    ss=States(4,32)
    first_slot=None
    for phase in ('first-t0','first-t1','finish-preempt-mixed','swapped-values'):
        if phase in ('first-t0','finish-preempt-mixed'):
            if phase=='finish-preempt-mixed':
                # Execute native union/iteration for subsequent finishes and
                # preemptions too; admission still pops this local free-list.
                native_runner.finish_requests(NS(finished_req_ids={'live-a'},preempted_req_ids={'live-b'}))
            for rid in ('live-a','live-b'):pool.add_request(rid,8,[0]*8,0,8)
        ids=['live-b','live-a']  # Deliberately different from admission order.
        mapping=[pool.req_id_to_index[rid] for rid in ids]
        if first_slot is None:first_slot=pool.req_id_to_index['live-a']
        np.random.seed(1234)  # Native post-warmup model seed reset.
        for rid in ('live-a','live-b'):
            temp=0. if phase=='first-t0' or (phase!='first-t1' and rid=='live-b') else (1. if rid=='live-a' else .5)
            params=NS(temperature=temp,top_p=1.,min_p=0.,top_k=0,seed=None,logprobs=None)
            ss.add_request(pool.req_id_to_index[rid],params)
        if phase=='swapped-values' and rank==3:
            # Valid indices alone cannot prove values belong to the request.
            mapping.reverse()
        ss.apply_staged_writes()
        owner=NS(max_num_reqs=4,device='cpu',speculative_config=NS(draft_sample_method='probabilistic'),use_fp64_gumbel=False)
        r=rt.Runtime(owner);r.mode='k3-stop';r.epoch=0
        runner=NS(speculator=NS(_kstop=r),sampler=NS(sampling_states=ss))
        batch=NS(req_ids=ids,num_reqs=2,query_start_loc_np=np.array([0,2048,2050,2053]),
                 is_prefilling_np=[True,True,False],idx_mapping_np=np.array([*mapping,-1]),
                 seq_lens_cpu_upper_bound=torch.tensor([1024,2048,rank]),has_prefill=True,num_tokens_after_padding=2050)
        state=dict(runtime=r,runner=runner,rows=torch.arange(2052),src=torch.arange(2052),dead=torch.zeros(2052,dtype=torch.bool))
        rt.STATE=state
        rt.select_inputs(runner,NS(finished_req_ids=set(),preempted_req_ids=set(),scheduled_new_reqs=[NS(req_id=x) for x in ids],
                                  scheduled_spec_decode_tokens={},num_scheduled_tokens=dict(zip(ids,[2048,2]))))
        packets=[]
        group=lambda:NS(rank_in_group=rank,ranks=list(range(4)),cpu_group=None)
        with patch.object(rt,'tp_group',group),patch.object(dist,'all_reduce',lambda p,**kw:packets.append(p.clone())):
            # Round-1 negative control: reinstate precisely the erroneous
            # local-address field, keeping all request values unchanged.
            saved=(r.trail,list(r.guard_inputs_log),r.bad)
            original=rt._inputs
            def legacy(runtime,b):
                widths,valid,payload=original(runtime,b)
                payload['sampling.mapping']=mapping
                return widths,valid,payload
            with patch.object(rt,'_inputs',legacy):rt.guard_inputs(runner,batch,False)
            r.trail,r.guard_inputs_log,r.bad=saved;r.guard_pending=None;r.inputs_guarded=False
            rt.guard_inputs(runner,batch,False)
        print(json.dumps(dict(phase=phase,rank=rank,first_slot=first_slot,mapping=mapping,
                              legacy=packets[0].tolist(),packet=packets[1].tolist(),fields=r.guard_field_digests())),flush=True)
        reply=json.loads(sys.stdin.readline())
        r.guard_pending=('prepare',torch.tensor(reply['ok']))
        def gather(out,fields,**kw):out[:]=reply['peers']
        case='temperature' if phase=='swapped-values' else 'mapping'
        with patch.object(rt,'tp_group',group),patch.object(dist,'all_gather_object',gather):
            consume(rank,case,r,runner,batch,state)


def hashseed_processes(source):
    processes=[subprocess.Popen([sys.executable,'-B',__file__,'--source',str(source),'--hashseed-worker',str(rank)],
               env=dict(os.environ,PYTHONHASHSEED=str(rank)),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
               stderr=subprocess.PIPE,text=True) for rank in range(4)]
    def agrees(packets):
        reduced=torch.tensor(packets,dtype=torch.int64).amax(0);k=rt.GUARD_WORDS+rt.GUARD_SLOTS
        return bool((reduced[:k]==-reduced[k+1:]).all()&(reduced[k]==0))
    try:
        for phase in ('first-t0','first-t1','finish-preempt-mixed','swapped-values'):
            records=[]
            for p in processes:
                line=p.stdout.readline()
                assert line,(phase,p.stderr.read())
                records.append(json.loads(line))
            assert all(x['phase']==phase and x['rank']==rank for rank,x in enumerate(records))
            slots=[x['first_slot'] for x in records]
            assert slots==[0,1,0,2],slots
            if phase!='swapped-values':
                assert not agrees([x['legacy'] for x in records]),'round-1 mapping must refuse'
            ok=agrees([x['packet'] for x in records])
            assert ok==(phase!='swapped-values'),(phase,'guard agreement',ok,'first slots',slots)
            reply=json.dumps(dict(ok=ok,peers=[x['fields'] for x in records]))+'\n'
            for p in processes:p.stdin.write(reply);p.stdin.flush()
            print('HASHSEED TP4 PASS '+phase+' first_slots='+str(slots),flush=True)
        for p in processes:
            _,err=p.communicate(timeout=60)
            assert p.returncode==0,err
    finally:
        for p in processes:
            if p.poll() is None:p.kill()
        for p in processes:p.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gloo',action='store_true')
    parser.add_argument('--source',type=Path,default=Path(os.environ.get('GLM_IMAGE_SRC','/image-source')))
    parser.add_argument('--hashseed-worker',type=int,choices=range(4),help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.environ.update(GLM_MTP_KSTOP_CONTROL_MODE='local',GLM_MTP_KSTOP_UNIFORM_BATCH='0',GLM_INDEXER_SHORTCUT='0')
    if args.hashseed_worker is not None:
        hashseed_worker(args.hashseed_worker,args.source);sys.exit(0)
    hashseed_processes(args.source)
    if args.gloo:
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=(str(Path(tmp)/'rendezvous'),str(args.source)),nprocs=4,join=True)
    else:simulated(native_states(args.source))
    print('KSTOP SPECSAMPLE TP4 PASS ('+('Gloo' if args.gloo else 'simulated reduction')+')',flush=True)
