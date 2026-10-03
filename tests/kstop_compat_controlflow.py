# SPDX-License-Identifier: Apache-2.0
"""Run the campaign's real fix19 harness with the recipe overlay under test.

Only CPU tensor scaffolding is adapted: native gather's substitute now supplies
its CPU length mirrors; fake dispatch records the shortcut descriptor; MTP's
actual index-sharing hooks run against CPU top-k buffers on the meta-built model.
The engine/scheduler/runner/propose/stop/planner bodies are the original harness.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def replace(s,old,new):
    assert s.count(old)==1, 'campaign harness anchor drift: '+old[:80]
    return s.replace(old,new)


def cpu_gumbel(z,idx,temp,seeds,pos,**kw):
    """T=0 CPU substitute for the stock Gumbel kernel's raw-logit store."""
    col=int(kw['logits_cache_col'].reshape(-1)[0]);good=idx>=0
    cache=kw['logits_cache']
    # Stock tl.store casts to the cache pointer dtype, before temperature.
    # Torch indexed assignment instead requires an explicit matching dtype.
    cache[idx[good].long(),col]=z[good].to(cache.dtype)
    return z.argmax(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True,type=Path);p.add_argument('--out',required=True)
    p.add_argument('--uniform',choices=['0','1','k2'],default='0');p.add_argument('--capture-layout',choices=['m12','reuse'],default='reuse');p.add_argument('--pad-hygiene',choices=['0','1'],default='0');p.add_argument('--sample',choices=['0','1'],default='0');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert a.prepare_only or Path('/.dockerenv').exists(),'pinned CPU container required'
    campaign=a.campaign.resolve();s=(campaign/'tests/controlflow_cpu.py').read_text()
    original_sha=hashlib.sha256(s.encode()).hexdigest()
    assert original_sha=='dda443c1a817d3a33e7690dc44d69dbd9a8bbc922dedde4ca9d9029cbaa40012','campaign harness source drift'
    overlay=Path(tempfile.mkdtemp())/'overlay';overlay.mkdir()
    for f in (campaign/'overlay').iterdir():
        if f.suffix in ('.py','.json'):shutil.copy(f,overlay/f.name)
    for parent in ('overlay/kstop','overlay/bringup'):
        for f in (ROOT/parent).iterdir():
            if f.suffix in ('.py','.json'):shutil.copy(f,overlay/f.name)
    # Meta fixtures do not run weight post-processing: configure the actual
    # draft Marlin kernel too, so padding's boot validation covers both models.
    s=replace(s,'        for m in target.modules():','        for m in [*target.modules(), *draft.modules()]:')
    s=replace(s,"overrides=[('--max-num-seqs','2')]","overrides=[('--max-num-seqs','4')]")
    s=replace(s,"        scenario('two_requests',","        scenario('four_requests',[('k',20,12,0),('l',30,12,0),('m',40,12,0),('n',50,12,0)])\n        scenario('two_requests',")
    s=replace(s,"        scenario('two_requests',","        scenario('three_requests',[('o',20,12,0),('p',30,12,0),('q',40,12,0)])\n        scenario('two_requests',")
    # Ensure this is the ORIGINAL harness and fixture root, not a reduced model.
    s=replace(s,"ROOT=Path(__file__).resolve().parents[1]",f"ROOT=Path({str(campaign)!r})")
    # Embed the same host-tested substitute in standalone prepared harnesses.
    s=replace(s,'NS=types.SimpleNamespace\n','NS=types.SimpleNamespace\n\n'+inspect.getsource(cpu_gumbel))
    s=replace(s,'    glm_mtp_fix.register();assert glm_mtp_kstop.register();cycle_receipts.register()',
        '    import glm_dsa_short as D,glm_spec_sample\n'
        '    if os.environ.get("GLM_SPEC_SAMPLE")=="1":glm_spec_sample.register()\n'
        '    assert D.register()\n'
        '    glm_mtp_fix.register();assert glm_mtp_kstop.register();cycle_receipts.register()')
    s=replace(s,'z=torch.full((n,32),-30.,dtype=torch.float32);step=',
        'z=torch.full((n,sp.draft_logits.shape[-1] if sp.draft_logits is not None else 32),-30.,dtype=torch.float32);step=')
    s=replace(s,'    sp.model=draft','''    sp.model=draft
    sp.share_mtp_topk_indices=bool(draft.config.index_share_for_mtp_iteration)
    assert sp.share_mtp_topk_indices
    mla=[m for m in draft.modules() if hasattr(m,'skip_topk') and hasattr(m,'topk_indices_buffer')]
    assert mla
    for m in mla:m.topk_indices_buffer=torch.full((fleet.scheduler_config.max_num_batched_tokens,2048),-1,dtype=torch.int32)
    shared={}
    native_compact=draft.model.compact_topk_indices
    def compact(slots):
        expected=[m.topk_indices_buffer[slots].clone() for m in mla]
        native_compact(slots)
        for m,e in zip(mla,expected):assert torch.equal(m.topk_indices_buffer[:len(slots)],e)
        shared['indices']=[m.topk_indices_buffer[:len(slots)].clone() for m in mla]
    draft.model.compact_topk_indices=compact
    # Gumbel tensor math is CPU scaffolding; call kstop's actual sample_draft,
    # write the same sparse request/cache column, keep T=0 stock argmax.
    import vllm.v1.worker.gpu.spec_decode.speculator as BASE
    BASE.gumbel_sample=cpu_gumbel''')
    s=replace(s,"        hs=torch.zeros(max(num_tokens,1),H,dtype=torch.bfloat16);return hs,hs",'''        step=int(sp.current_draft_step.reshape(-1)[0])
        for i,m in enumerate(mla):
            assert m.skip_topk==(step>0)
            if step==0:
                m.topk_indices_buffer[:num_tokens]=torch.arange(num_tokens,dtype=torch.int32)[:,None]+100*i
            else:
                e=shared['indices'][i]
                assert torch.equal(m.topk_indices_buffer[:len(e)],e),'shared indices changed across MTP steps'
        hs=torch.zeros(max(num_tokens,1),H,dtype=torch.bfloat16);return hs,hs''')
    s=replace(s,"    R.gather_batch_req_state=lambda so,dummy:(NS(num_tokens=so.total_num_scheduled_tokens),None)",'''    def gathered(so,dummy):
        if dummy:return NS(num_tokens=so.total_num_scheduled_tokens),None
        b=batch_for(so)
        uniform=int(b.num_scheduled_tokens[0]) if len(set(b.num_scheduled_tokens.tolist()))==1 and not b.has_prefill else None
        ok=D.eligible(so,b,[STATE['reqs'][r]['computed'] for r in b.req_ids],uniform)
        D.SHORT.set(ok);D.COUNTS['eligible' if ok else 'fallback']+=1
        # Gather tensor math is replaced by the original harness; these are
        # the actual CPU mirrors consumed by the new guard payload.
        ss=R.sampler.sampling_states
        for k in ('temperature','top_p','min_p','top_k','seeds'):
            obj=getattr(ss,k,None)
            if obj is None:obj=NS();setattr(ss,k,obj)
            if not hasattr(obj,'np'):obj.np=np.full(8,{'temperature':0.,'top_p':1.,'min_p':0.,'top_k':32,'seeds':0}[k])
        return b,uniform
    R.gather_batch_req_state=gathered''')
    s=replace(s,"cg_mode=CUDAGraphMode.NONE,max_query_len=kw.get('max_query_len')", "cg_mode=CUDAGraphMode.NONE,short_context=D.active(),max_query_len=kw.get('max_query_len')")
    if a.uniform == 'k2':
        s=replace(s,"    scenario('one_request',[('a',40,12,0)])", "    scenario('one_request',[('a',40,12,0)])\n    CONF.update({('u',None):.8})\n    scenario('one_request_k2',[('u',40,12,0)])")
        s=replace(s,'    R.prepare_inputs=lambda so,state,desc:batch_for(so)', '''    def prepared(so,state,desc):
        batch=batch_for(so)
        batch.num_tokens_after_padding=desc.num_tokens
        batch.num_reqs_after_padding=desc.num_reqs
        assert batch.num_tokens <= desc.num_tokens
        return batch
    R.prepare_inputs=prepared''')
        # Keep warmup synthetic/eager as in the original harness. Real request
        # dispatch uses the pinned candidate builder and dispatcher; only CUDA
        # capture/replay tensor math is substituted by the CPU target output.
        s=replace(s,"    def fake_dispatch(manager,num_reqs,num_tokens,uniform_token_count=None,dp_size=1,dp_rank=0,**kw):\n",
            "    def fake_dispatch(manager,num_reqs,num_tokens,uniform_token_count=None,dp_size=1,dp_rank=0,**kw):\n"
            "        if manager is not None and uniform_token_count in (2,3,4):\n"
            "            assert not kw.get('need_eager',False), 'uniform K2 target went eager'\n"
            "        if manager is not None and not kw.get('need_eager',False):\n"
            "            desc=manager.dispatch(num_reqs,num_tokens,uniform_token_count,0,kw.get('max_query_len'))\n"
            "            if uniform_token_count in (2,3,4):\n"
            "                target_dispatches.append((num_reqs,num_tokens,desc.num_reqs,desc.num_tokens,uniform_token_count))\n"
            "                assert desc.cg_mode==CUDAGraphMode.FULL, 'target went eager'\n"
            "            if num_reqs>1 and uniform_token_count==3:\n"
            "                expected=(2,6) if num_reqs==2 and os.environ.get(\"GLM_MTP_KSTOP_CAPTURE_LAYOUT\")==\"reuse\" else (4,12)\n"
            "                assert (desc.num_reqs,desc.num_tokens)==expected\n"
            "            return desc,None\n")
        s=replace(s,"    # ---- scenarios: real AsyncScheduler + EngineCore.step_with_batch_queue ----", '''    import vllm.v1.worker.gpu.cudagraph_utils as CG
    fleet.compilation_config.cudagraph_capture_sizes=[1,4,12,16]
    fleet.compilation_config.max_cudagraph_capture_size=16
    platform=CG.current_platform
    CG.current_platform=NS(get_global_graph_pool=lambda:None)
    try:
        R.cudagraph_manager=CG.CudaGraphManager(fleet,torch.device('cpu'),CUDAGraphMode.FULL_DECODE_ONLY,4)
    finally:CG.current_platform=platform
    manager=R.cudagraph_manager
    manager.graphs={d:None for ds in manager._capture_descs.values() for d in ds}
    manager._graphs_captured=True
    graph_calls=[]
    target_dispatches=[]
    def replay(desc):
        assert desc in manager.graphs
        graph_calls.append(dict(n=desc.num_reqs,rows=desc.num_tokens,q=desc.uniform_token_count,short=desc.short_context))
        return torch.zeros(desc.num_tokens,H,dtype=torch.bfloat16)
    manager.run_fullgraph=replay
    # ---- scenarios: real AsyncScheduler + EngineCore.step_with_batch_queue ----''')
        s=replace(s,"    finish(a,out)\n\n\ndef finish", "    out['target_graph_calls']=graph_calls\n    out['target_dispatches']=target_dispatches\n    assert set(target_dispatches) >= {(1,2,1,2,2),(1,3,1,3,3),(1,4,1,4,4),(2,6,*((2,6) if os.environ.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT')=='reuse' else (4,12)),3),(3,9,4,12,3),(4,12,4,12,3)}\n    finish(a,out)\n\n\ndef finish")
    s=replace(s,"    finish(a,out)\n\n\ndef finish", "    out['shortcut']=dict(counts=D.COUNTS,sharing=sp.share_mtp_topk_indices)\n    assert D.COUNTS['eligible']>0\n    assert all(not m.skip_topk for m in mla),'reuse flag leaked'\n    finish(a,out)\n\n\ndef finish")
    # Record every emitted token ID, preserving the original scheduler loop.
    s=replace(s,"    STATE=dict(reqs={},steps=[])","    STATE=dict(reqs={},steps=[])\n    token_ids={}")
    s=replace(s,"                            done.setdefault(o.request_id,0);done[o.request_id]+=len(o.new_token_ids)",
        "                            done.setdefault(o.request_id,0);done[o.request_id]+=len(o.new_token_ids)\n"
        "                            token_ids.setdefault(name,{}).setdefault(o.request_id,[]).extend(o.new_token_ids)")
    s=replace(s,"    finish(a,out)\n\n\ndef finish", "    out['token_ids']=token_ids\n    finish(a,out)\n\n\ndef finish")
    compile(s,str(campaign/'tests/controlflow_cpu.py'),'exec')
    if a.prepare_only:
        Path(a.out).write_text(s)
        print('HARNESS ADAPTATION/COMPILE PASS '+original_sha);return
    os.environ.update(GLM_PAD_HYGIENE=a.pad_hygiene,GLM_INDEXER_SHORTCUT='1',GLM_MTP_KSTOP_UNIFORM_BATCH=a.uniform,GLM_MTP_KSTOP_CAPTURE_LAYOUT=a.capture_layout,GLM_SPEC_SAMPLE=a.sample)
    sys.argv=[str(campaign/'tests/controlflow_cpu.py'),'--overlay',str(overlay),'--out',a.out]
    exec(compile(s,str(campaign/'tests/controlflow_cpu.py'),'exec'),{'__name__':'__main__','__file__':str(campaign/'tests/controlflow_cpu.py')})
    result=json.loads(Path(a.out).read_text())
    assert result['initialize']=='PASS' and result['warmup']['status']=='PASS',result
    for name,v in result['scenarios'].items():
        if name=='missing_proposal_fails_closed':assert v['status']=='FAIL' and 'local: select' in v['error'],v
        else:assert v['status']=='PASS' and v['fallback_syncs']==0,(name,v)
    result['harness_sha256']=original_sha
    result['compat']=dict(pad_hygiene=a.pad_hygiene,uniform=a.uniform,capture_layout=a.capture_layout,sample=a.sample,shortcut='1',status='PASS_CPU_CONTROL_FLOW')
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
