# SPDX-License-Identifier: Apache-2.0
"""Actual shared INIT body + pinned native _prefill/decoder/MLA on CPU.

Kernel substitutes test gate coverage/restoration/refusal, never CUDA numerics.
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from dataclasses import dataclass
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'overlay/bringup'), str(ROOT/'overlay/kstop'), str(ROOT/'overlay/overlay')]
import glm_mtp_rowselect as R
import glm_draft_head_qual as Q
from test_mtp_rowselect import fixture, native_method, Linear
from test_draft_head import local_vote

@dataclass(frozen=True)
class Desc:
    num_tokens: int
    num_reqs: int
    short_context: bool = False
    cg_mode: str = 'FULL'


def setup(fault=None, before_attach=None, wide_hidden=False):
    layer, block, mla = fixture()
    # Preserve a changed-input signal in this tiny kernel substitute. The
    # original 4-column RMSNorm rounded all three probe inputs to equal output.
    # Native MLA/decoder/MTP/prefill bodies still execute unchanged.
    layer.hnorm = torch.nn.Identity()
    cache = torch.zeros(2,64,4,dtype=torch.bfloat16)
    mla.mla_attn.register_buffer('kv_cache', cache)
    attention = mla.mla_attn.forward
    def write(*a, **kw):
        out = attention(*a, **kw)
        m = len(out)
        cache[0,:m].copy_(mla.mla_attn.kv)
        if fault == 'kv' and block._glm_rowselect_indices is not None:
            cache[0,0,0] += 1
        return out
    mla.mla_attn.forward = write
    model = torch.nn.Module();model.layer = layer
    model.config = NS(vocab_size=11);head = Linear(4,11,False)
    model.compute_logits = lambda h:head(h[:,:4])
    sp = NS(model=model, max_num_reqs=4, num_speculative_steps=3,
        input_buffers=NS(input_ids=torch.zeros(16,dtype=torch.long),positions=torch.zeros(16,dtype=torch.long)),
        target_input_buffers=NS(),block_tables=NS(slot_mappings=torch.full((1,16),-1,dtype=torch.long)),
        hidden_states=torch.zeros(16,4,dtype=torch.bfloat16),draft_tokens=torch.zeros(4,3,dtype=torch.long),
        _kstop=NS(confidence=torch.zeros(4),capture_layout='reuse'),temperature=torch.zeros(4),idx_mapping=torch.arange(4),
        last_token_indices=torch.zeros(4,dtype=torch.long),current_draft_step=torch.zeros(1,dtype=torch.long),
        seeds=None,draft_logits=None)
    def sample(h,*a):
        logits=sp.model.compute_logits(h)
        sp._kstop.confidence[:len(h)].copy_(logits.float().softmax(-1).amax(-1))
        if fault == 'confidence' and block._glm_rowselect_indices is not None:
            sp._kstop.confidence[0] += .1
        if fault == 'logits' and block._glm_rowselect_indices is not None:
            sp._glm_rowselect_logits[0,0] += 1
        return logits.argmax(-1) + (1 if fault == 'proposal' and block._glm_rowselect_indices is not None else 0)
    sp.sample_draft=sample
    if fault == 'hidden_flip' or wide_hidden:
        sp.hidden_states = torch.zeros(16,128,dtype=torch.bfloat16)
    sp._run_model=lambda m,*a,**kw:layer(sp.input_buffers.input_ids[:m],sp.input_buffers.positions[:m],
        sp.hidden_states[:m,:4],torch.zeros(m,4,dtype=torch.bfloat16))
    if fault == 'hidden_flip' or wide_hidden:
        run = sp._run_model
        sp._run_model = lambda *a,**kw:tuple(t.repeat(1,32) for t in run(*a,**kw))
    fn=native_method('vllm.v1.worker.gpu.spec_decode.autoregressive.speculator',
        'AutoRegressiveSpeculator','_prefill',dict(torch=torch,CUDAGraphMode=NS(NONE='NONE')))
    import types
    native=types.MethodType(fn,sp)
    if before_attach is not None:
        sp._prefill = native
        before_attach(sp)
    R.attach(sp,dict(block=block,mla=mla,native_block=block.forward,native_mla=mla.forward,native_prefill=native,
        block_fn=R.clone_method(block.forward.__func__,'block'),mla_fn=R.clone_method(mla.forward.__func__,'mla'),
        prefill_fn=R.clone_method(fn,'prefill')))
    R.capture_logits(sp)
    if fault == 'constant':
        original_prefill = sp._prefill
        def constant_feedback(n,m,*args,**kwargs):
            original_prefill(n,m,*args,**kwargs)
            sp.hidden_states[:n].fill_(.25)
        sp._prefill = constant_feedback
    for name in ('on_prefill_begin','on_prefill_end','on_multi_step_decode_begin','on_multi_step_decode_end'):
        setattr(sp,name,lambda n:None)
    first=NS(graphs={})
    later=NS(graphs={})
    def factory(d,warmup=False):
        sp.block_tables.slot_mappings.fill_(-1)
        return lambda mode:sp._prefill(d.num_reqs,d.num_tokens,None,None,None)
    flips = [0]
    for d in [Desc(m,n,short) for m,n in ((2,1),(3,1),(4,1),(6,2),(12,4),(16,4)) for short in (False,True)]:
        def replay(d=d):
            if fault=='stale':sp.last_token_indices.zero_()
            sentinel = sp.hidden_states[:d.num_reqs].clone()
            sp._prefill(d.num_reqs,d.num_tokens,None,None,None)
            if fault == 'unwritten':sp.hidden_states[0].copy_(sentinel[0])
            if fault == 'token_unwritten':sp.draft_tokens[0,0] = -1
            if fault == 'nonfinite':sp._kstop.confidence[0] = float('nan')
            if fault == 'hidden_flip':
                flips[0] += 1
                if flips[0] % 2:sp.hidden_states[0,-1:].view(torch.int16).add_(1)
            if fault == 'flip':
                flips[0] += 1
                if flips[0] % 2:
                    sp._kstop.confidence[0] += 2e-6
                    sp._glm_rowselect_logits[0,0] += 1/32
            if fault=='feedback':sp.hidden_states[0,0] += 1
        first.graphs[d]=NS(replay=replay)
    first._k4_drafthead_factory=factory
    # Later-stage native body uses no row selection in production; fixture uses
    # same forward solely to cover untouched shared qualification control flow.
    d=Desc(4,4)
    def later_forward(mode):
        sp._prefill(d.num_reqs,d.num_tokens,None,None,None)
        sp.draft_tokens[:d.num_reqs, sp.num_speculative_steps-1].copy_(sp.draft_tokens[:d.num_reqs,0])
    def later_factory(desc,warmup=False):
        factory(desc,warmup)
        return later_forward
    later.graphs[d]=NS(replay=lambda:later_forward('NONE'))
    later._k4_drafthead_factory=later_factory
    sp.prefill_cudagraph_manager=first;sp.decode_cudagraph_manager=later
    return NS(speculator=sp,cudagraph_manager=NS(graphs={})), cache


class Tests(unittest.TestCase):
    def run_qual(self,r,vote=local_vote,scratch_limit=Q.SCRATCH_LIMIT):
        with patch.dict(sys.modules,{'vllm.config.compilation':NS(CUDAGraphMode=NS(FULL='FULL',NONE='NONE'))}):
            return Q.qualify(r,sync=lambda:None,vote=vote,mem=lambda:8<<30,scratch_limit=scratch_limit)

    def test_all_positions_full_selected_outputs_and_restore(self):
        import glm_draft_head as dh
        r,cache=setup();saved=Q.tensor_state(r.speculator);before=cache.clone()
        scopes = set()
        term = dh.qualification_term
        def qualification_term(check, conditions, measured, error=None):
            if check == 'replay_compare':
                for name, result in measured['pairs'].items():
                    enabled = False
                    hidden = result['outputs'][1]
                    self.assertNotIn('near_zero_relief_enabled', hidden)
                    self.assertEqual('rows_over_diff_fraction' in hidden, enabled)
                    scopes.add((measured['manager'], name))
            return term(check, conditions, measured, error)
        with patch.object(dh, 'qualification_term', qualification_term):
            report=self.run_qual(r)
        for name in ('full_vs_eager', 'full_vs_graph1', 'full_vs_graph2',
                     'eager_vs_graph1', 'eager_vs_graph2', 'graph1_vs_graph2',
                     'zero_roundtrip_full', 'zero_roundtrip_eager',
                     'zero_roundtrip_graph1', 'zero_roundtrip_graph2'):
            self.assertIn((0, name), scopes)
        self.assertIn((1, 'graph1_vs_graph2'), scopes)
        self.assertTrue(R.qualification_valid(report))
        cases=report['rowselect']['cases']
        self.assertEqual(len(cases),380)
        self.assertEqual({x['position'] for x in cases},{0,1,2,3})
        for t,s in saved:self.assertTrue(torch.equal(t,s))
        self.assertTrue(torch.equal(cache,before))
        self.assertIsNone(r.speculator._glm_rowselect_native[0]._glm_rowselect_indices)

    def shared_fault_qual(self, fault, target=Desc(16,4,True), position=3,
                          pattern=1, return_probe=4, accepted=False):
        """Write the same bad output in all arms: pair agreement cannot catch it."""
        import glm_draft_head as dh
        r,cache=setup(wide_hidden=True);sp=r.speculator
        saved=Q.tensor_state(sp);before=cache.clone();terms=[];stage={}
        native=sp._prefill
        def forward(n,m,*args,**kwargs):
            native(n,m,*args,**kwargs)
            if (stage.get('manager') != 0 or stage.get('descriptor') != repr(target)
                    or stage.get('position') != position
                    or stage.get('sentinel_pattern') != pattern):
                return
            if fault == 'constant' or (fault == 'constant_after_b' and stage.get('probe') != 1):
                sp.hidden_states[:n].fill_(.25)
            if fault == 'constant_row':sp.hidden_states[n-1].fill_(.25)
            if fault == 'hidden_cumulative' and stage.get('probe') in (2,4):
                sp.hidden_states[0,-1:].view(torch.int16).add_(8 if stage['probe']==2 else 17)
            if stage.get('probe') != return_probe:return
            if fault.startswith('hidden_') and fault != 'hidden_cumulative':
                steps,count=map(int,fault.removeprefix('hidden_').split('_'))
                sp.hidden_states[:n].flatten()[:count].view(torch.int16).add_(steps)
            if fault == 'confidence_drift':sp._kstop.confidence[:n].add_(2e-6)
            if fault == 'confidence_error':sp._kstop.confidence[:n].add_(2e-5)
            if fault == 'logits_drift':sp._glm_rowselect_logits[:n,0].add_(1/16)
            if fault == 'logits_error':sp._glm_rowselect_logits[:n,0].add_(1/8)
            if fault == 'tokens':sp.draft_tokens[:n,0].add_(1)
            if fault == 'kv':cache[0,0,0].add_(1)
            if fault == 'selection':sp.idx_mapping[:n].add_(1)
        sp._prefill=forward
        def vote(payload,valid):
            if 'replay_inputs' in payload:stage.update(payload['replay_inputs'])
            local_vote(payload,valid)
        token=dh._qualification_terms.set(terms)
        try:
            if accepted:
                self.assertTrue(R.qualification_valid(self.run_qual(r,vote=vote)))
            else:
                with self.assertRaises(RuntimeError):self.run_qual(r,vote=vote)
        finally:
            dh._qualification_terms.reset(token)
        for t,saved_tensor in saved:self.assertTrue(torch.equal(t,saved_tensor))
        self.assertTrue(torch.equal(cache,before))
        return [t for t in terms if t['check']=='replay_compare' and
                t['measured']['descriptor']==repr(target) and
                t['measured']['position']==position and
                t['measured']['sentinel_pattern']==pattern]

    def test_constant_feedback_across_entire_bank_refuses(self):
        import glm_draft_head as dh
        r,cache=setup('constant');saved=Q.tensor_state(r.speculator);before=cache.clone();terms=[]
        token=dh._qualification_terms.set(terms)
        try:
            with self.assertRaises(RuntimeError):self.run_qual(r)
        finally:
            dh._qualification_terms.reset(token)
        failed=next(t for t in terms if t['check']=='replay_compare' and
                    not t['conditions']['changed_input_sensitive'])
        self.assertEqual(failed['measured']['probe'],1)
        self.assertTrue(all(v for k,v in failed['conditions'].items()
                            if k != 'changed_input_sensitive'))
        executions=[t for t in terms if t['check']=='replay_execution']
        self.assertTrue(all(t['conditions']['hidden_feedback_written'] for t in executions))
        self.assertEqual(set(failed['measured']['changed_rows_by_arm']),
                         {'full','eager','graph1','graph2'})
        for t,s in saved:self.assertTrue(torch.equal(t,s))
        self.assertTrue(torch.equal(cache,before))

    def test_constant_feedback_written_by_every_arm_refuses(self):
        for fault in ('constant','constant_row','constant_after_b'):
            with self.subTest(fault=fault):
                terms=self.shared_fault_qual(fault)
                failed=terms[-1]
                self.assertEqual(failed['measured']['probe'],3 if fault=='constant_after_b' else 1)
                self.assertFalse(failed['conditions']['changed_input_sensitive'])
                self.assertTrue(all(v for k,v in failed['conditions'].items()
                                    if k != 'changed_input_sensitive'))
                rows=failed['measured']['changed_rows_by_arm']
                self.assertEqual(set(rows),{'full','eager','graph1','graph2'})
                self.assertTrue(all(not changes[-1] for changes in rows.values()))
                if fault == 'constant_row':
                    self.assertTrue(all(all(changes[:-1]) for changes in rows.values()))

    def test_zero_roundtrip_shared_state_requires_exact_tokens_and_bounded_floats(self):
        for return_probe in (2,4):
            for fault,accepted in (('hidden_16_8',False),('hidden_17_1',False),
                    ('hidden_1_9',False),('confidence_drift',False),('confidence_error',False),
                    ('logits_drift',False),('logits_error',False),('tokens',False),('kv',False)):
                with self.subTest(fault=fault,return_probe=return_probe):
                    terms=self.shared_fault_qual(fault,return_probe=return_probe,accepted=accepted)
                    term=next(t for t in terms if t['measured']['probe']==return_probe)
                    self.assertEqual(term['conditions']['zero_input_roundtrip'],accepted)
                    pairs=term['measured']['pairs']
                    self.assertTrue(all(p['tokens_exact'] and p['drift_bounded'] and
                        p['kv_selection_exact'] for name,p in pairs.items()
                        if not name.startswith('zero_roundtrip_')))
                    self.assertEqual({name for name in pairs if name.startswith('zero_roundtrip_')},
                        {'zero_roundtrip_'+arm for arm in ('full','eager','graph1','graph2')})
                    if fault == 'tokens':
                        self.assertFalse(pairs['zero_roundtrip_graph1']['tokens_exact'])
                    if fault == 'hidden_16_8':
                        hidden=pairs['zero_roundtrip_graph1']['outputs'][1]
                        self.assertEqual(hidden['max_bf16_steps'],16)
                        self.assertEqual(hidden['mismatches'],8)

    def test_roundtrip_uses_first_zero_not_previous_return(self):
        terms=self.shared_fault_qual('hidden_cumulative')
        first_return=terms[-1]
        self.assertEqual(first_return['measured']['probe'],2)
        self.assertFalse(first_return['conditions']['zero_input_roundtrip'])
        hidden=first_return['measured']['pairs']['zero_roundtrip_graph1']['outputs'][1]
        self.assertEqual(hidden['max_bf16_steps'],8)
        # Even the first eight-step drift now refuses, before cumulative drift.

    def test_roundtrip_sequence_and_baselines_cover_every_position_and_pattern(self):
        # A healthy bank must return to the FIRST A after both B and C, for
        # each position/pattern separately (opposite signs are different inputs).
        r,_=setup();compares=[]
        def vote(payload,valid):
            if 'replay_compare' in payload:compares.append(payload['replay_compare'])
            local_vote(payload,valid)
        report=self.run_qual(r,vote=vote)
        for case in report['cases']:
            self.assertEqual(case['input_sequence'],[0.,.125,0.,-.125,0.])
        keys={(s['manager'],s['descriptor'],s['position'],s['sentinel_pattern']) for s in compares}
        self.assertEqual(len(keys),78) # 38 prefill positions + one decode, two patterns.
        for key in keys:
            trials=[s for s in compares if (s['manager'],s['descriptor'],
                s['position'],s['sentinel_pattern'])==key]
            self.assertEqual([s['probe'] for s in trials],list(range(5)))
            self.assertEqual([s['value'] for s in trials],[0.,.125,0.,-.125,0.])

    def test_each_output_and_changed_selection_refuses_and_restores(self):
        for fault in ('kv','logits','confidence','feedback','stale','proposal','unwritten','token_unwritten','nonfinite'):
            with self.subTest(fault=fault):
                r,cache=setup(fault);saved=Q.tensor_state(r.speculator);before=cache.clone()
                with self.assertRaises(RuntimeError):self.run_qual(r)
                for t,s in saved:self.assertTrue(torch.equal(t,s))
                self.assertTrue(torch.equal(cache,before))

    def test_missing_kv_and_scratch_cap_refuse(self):
        r,_=setup()
        with patch.object(R,'kv_scratch',side_effect=RuntimeError('missing KV')), self.assertRaises(RuntimeError):self.run_qual(r)
        with self.assertRaises(RuntimeError):self.run_qual(r,scratch_limit=16)

    def test_actual_init_withholds_ready_until_all_checks_and_peer_vote(self):
        import glm_draft_head as dh
        from test_draft_head_init import fixture as head_fixture
        for fault in (None, 'kv', 'peer'):
            r,cache=setup(None if fault=='peer' else fault)
            head=head_fixture();head.speculator=r.speculator
            def vote(payload,valid):
                self.assertFalse(head._draft_head_ready)
                local_vote(payload,valid and not (fault=='peer' and 'replay_compare' in payload))
            with patch.dict(sys.modules,{'vllm.config.compilation':NS(CUDAGraphMode=NS(FULL='FULL',NONE='NONE'))}):
                kwargs=dict(vote=vote,sync=lambda:None,mem=lambda:8<<30,collect=lambda:None,empty=lambda:None)
                if fault:
                    with self.assertRaises(RuntimeError):dh.qualify_initial(head,**kwargs)
                    self.assertFalse(head._draft_head_ready)
                else:
                    report=dh.qualify_initial(head,**kwargs)
                    self.assertTrue(head._draft_head_ready)
                    self.assertTrue(R.qualification_valid(report))

    def test_two_state_flip_refuses_even_inside_old_bounds(self):
        for fault in ('flip', 'hidden_flip'):
            r,cache=setup(fault);before=cache.clone()
            with self.assertRaises(RuntimeError):self.run_qual(r)
            self.assertTrue(torch.equal(cache,before))

    def test_actual_capture_wrapper_rowselect_refusal_boots_off(self):
        import glm_draft_head as dh
        from test_draft_head_init import fixture as head_fixture, completed_vote
        for fault in ('kv', 'unwritten', 'proposal', 'peer', 'outer'):
            r,_=setup(None if fault in ('peer','outer') else fault)
            head=head_fixture();head.speculator=r.speculator
            target=dict(head.cudagraph_manager.graphs)
            graphs=[dict(m.graphs) for m in dh.managers(head)]
            factories=[m._k4_drafthead_factory for m in dh.managers(head)]
            def capture(self):
                for m,g,f in zip(dh.managers(self),graphs,factories):
                    m.graphs=dict(g);m._k4_drafthead_factory=f
                return 'captured'
            head.speculator.capture=lambda:capture(head)
            cls=type('Runner',(),dict(load_model=lambda s:None,capture_model=capture))
            for m in dh.managers(head):m._capture_descs={'FULL':list(m.graphs)}
            dh.install_runner(NS(GPUModelRunner=cls))
            R.install_runner(NS(GPUModelRunner=cls))
            def vote(payload,valid):
                completed_vote(payload,valid and not
                    (fault=='peer' and 'replay_compare' in payload) and not
                    (fault=='outer' and 'rowselect_descriptors' in payload))
            with patch.dict(sys.modules,{'vllm.config.compilation':NS(CUDAGraphMode=NS(FULL='FULL',NONE='NONE'))}), \
                 patch.object(dh,'agree',vote),patch.object(dh,'memory',lambda:8<<30), \
                 patch.object(torch.cuda,'synchronize',lambda:None),patch.object(torch.cuda,'empty_cache',lambda:None), \
                 patch.object(dh,'renew_draft_pool',lambda r:None):
                self.assertEqual(cls.capture_model(head),'captured')
            self.assertFalse(head._draft_head.on)
            self.assertFalse(head._draft_head_ready)
            self.assertFalse(head.speculator._glm_rowselect['ready'])
            self.assertTrue(head.speculator._glm_rowselect['disabled'])
            self.assertIsNone(head.speculator._kstop.bad)
            self.assertEqual(head.cudagraph_manager.graphs,target)
            self.assertTrue(all(m.graphs for m in dh.managers(head)))
            block,mla=head.speculator._glm_rowselect_native[:2]
            self.assertEqual((block.forward,mla.forward),head.speculator._glm_rowselect_native[5:7])
            # The retained outer prefill dispatches to native full-M on OFF.
            head.speculator._prefill(1,2,None,None,None)
            self.assertIsNone(block._glm_rowselect_indices)
            self.assertIsNone(mla._glm_rowselect_indices)

    def test_active_logits_written_and_inactive_nan_columns_preserved(self):
        for fault in (None, 'unwritten', 'inactive'):
            r,_=setup();sp=r.speculator
            sp.draft_logits=torch.full((4,3,11),float('nan'))
            original=sp.sample_draft
            def sample(h,*a):
                tokens=original(h,*a)
                step=int(sp.current_draft_step[0])
                if fault != 'unwritten':sp.draft_logits[:len(h),step].copy_(sp._glm_rowselect_logits[:len(h)])
                if fault == 'inactive':sp.draft_logits[0,(step+1)%3,0]=1
                return tokens
            sp.sample_draft=sample
            if fault:
                with self.assertRaises(RuntimeError):self.run_qual(r)
            else:
                self.assertTrue(R.qualification_valid(self.run_qual(r)))
            self.assertTrue(torch.isnan(sp.draft_logits).all())

    def test_shared_hidden_policy_boundaries(self):
        import glm_rowselect_qual as rq
        base=[torch.zeros((1,3),dtype=torch.long),torch.ones((1,128),dtype=torch.bfloat16),
              torch.zeros(1),torch.zeros(1,11),torch.zeros(64,4,dtype=torch.bfloat16),
              torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),
              torch.zeros(1,2,dtype=torch.long),torch.zeros(2,dtype=torch.long)]
        for steps,count,passed in ((0,2,True),(16,2,False),(17,1,False),(1,1,False),(1,3,False)):
            right=[t.clone() for t in base]
            right[1][0,:count].view(torch.int16).add_(steps)
            pair=rq.bounded_pair(base,right,False,True)
            self.assertEqual(pair['drift_bounded'],passed)
            self.assertTrue(pair['tokens_exact'] and pair['kv_selection_exact'])

    def test_ready_receipt_is_mandatory(self):
        for report in (None,{}, {'rowselect':dict(full_m_selected=True,changed_positions=True,logits_confidence_feedback_kv=True,cases=[])}):
            self.assertFalse(R.qualification_valid(report))
        from glm_mtp_rowselect_config import options
        from test_mtp_rowselect import ENV
        with self.assertRaises(ValueError):options(ENV|{'GLM_MTP_KSTOP_CAPTURE_LAYOUT':'m12'})
        without_layout=dict(ENV);without_layout.pop('GLM_MTP_KSTOP_CAPTURE_LAYOUT')
        with self.assertRaises(ValueError):options(without_layout)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
