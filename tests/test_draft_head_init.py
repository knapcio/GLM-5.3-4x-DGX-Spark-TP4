# SPDX-License-Identifier: Apache-2.0
"""Actual INIT capture wrapper and shared replay body, CPU native-buffer fixture."""
import sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'overlay/kstop'), str(ROOT/'overlay/bringup')]
from test_draft_head import runner, prepare_cpu, local_vote
import glm_draft_head as dh
import glm_draft_head_qual as qual

@dataclass(frozen=True)
class Desc:
    cg_mode: str = 'FULL'
    num_tokens: int = 1
    num_reqs: int = 1

COMPILATION=NS(CUDAGraphMode=NS(FULL='FULL', NONE='NONE'))

def fixture(fault=None):
    r=runner()
    with patch.object(dh,'INITIAL_ON',True):prepare_cpu(r)
    sp=r.speculator
    buffers=NS(input_ids=torch.ones(1,dtype=torch.long), positions=torch.ones(1,dtype=torch.long))
    sp.input_buffers=buffers;sp.target_input_buffers=buffers;sp.block_tables=NS(slot=torch.ones(1))
    sp._kstop.confidence=torch.ones(1);sp._kstop.bad=None
    sp.hidden_states=torch.ones(1,256,dtype=torch.bfloat16)
    sp.draft_tokens=torch.ones(1,4,dtype=torch.long);sp.draft_logits=None;sp.max_num_reqs=1;sp.num_speculative_steps=3
    sp.temperature=torch.ones(1);sp.idx_mapping=torch.zeros(1,dtype=torch.long)
    sp.last_token_indices=torch.zeros(1,dtype=torch.long);sp.current_draft_step=torch.zeros(1,dtype=torch.long)
    for name in ('on_prefill_begin','on_prefill_end','on_multi_step_decode_begin','on_multi_step_decode_end'):
        setattr(sp,name,lambda n:None)
    def forward(mode):
        if fault=='execution_exception' and r._draft_head.on:raise RuntimeError('injected draft execution error')
        assert int(sp.current_draft_step[0]) in (0,sp.num_speculative_steps-1)
        logits=r._draft_head.bank(sp.hidden_states)
        sp.draft_tokens.fill_(int(logits.argmax(-1)[0]))
        sp.hidden_states.mul_(2).add_(1);sp._kstop.confidence.fill_(.9)
    def factory(desc,warmup):
        if fault=='inputs_exception' and r._draft_head.on:raise RuntimeError('injected draft input error')
        sp.block_tables.slot.fill_(-1)
        return forward
    def capture():
        for index,m in enumerate(dh.managers(r)):
            def replay(index=index):
                forward(None)
                if fault=='mismatch' and r._draft_head.on and index==1:sp.hidden_states.add_(10)
                if fault=='nonfinite' and r._draft_head.on and index==1:sp._kstop.confidence.fill_(float('nan'))
                if fault=='target' and r._draft_head.on and index==1:r.model.lm_head.weight.add_(1)
            m.graphs={Desc():NS(replay=replay)}
            m._k4_drafthead_factory=factory
            m._capture_descs={'FULL':[Desc()]}
        if fault=='missing' and r._draft_head.on:dh.managers(r)[1].__dict__.pop('_k4_drafthead_factory')
    sp.capture=capture;capture()
    return r


def completed_vote(payload, valid):
    if not valid:
        raise dh.VoteRefused(dict(votes=[dict(rank=0, valid=False, measured=payload,
                                             terms=list(dh._qualification_terms.get() or []))]))


def boot(r,vote=completed_vote,mem=lambda:8<<30):
    cls=type('GPUModelRunner',(),{'load_model':lambda self:None,
        'capture_model':lambda self:'capture-result' if not self.speculator.capture() else None})
    dh.install_runner(NS(GPUModelRunner=cls))
    with patch.dict(sys.modules,{'vllm.config.compilation':COMPILATION}), \
         patch.object(dh,'agree',vote),patch.object(dh,'memory',mem), \
         patch.object(torch.cuda,'synchronize',lambda:None), \
         patch.object(torch.cuda,'empty_cache',lambda:None), \
         patch.object(dh,'renew_draft_pool',lambda r:None),patch.object(dh,'switch',side_effect=AssertionError('setter called')):
        return cls.capture_model(r)

class InitTests(unittest.TestCase):
    def test_init_real_capture_shared_qualification_and_restoration(self):
        r=fixture();before=qual.tensor_state(r.speculator);target=dict(r.cudagraph_manager.graphs)
        votes=[]
        def vote(payload,valid):
            votes.append((payload,valid,r._draft_head_ready));completed_vote(payload,valid)
        self.assertEqual(boot(r,vote),'capture-result')
        self.assertTrue(r._draft_head_ready);self.assertTrue(r._draft_head.on)
        self.assertEqual(r._draft_head_epoch,-1)
        self.assertEqual(r.cudagraph_manager.graphs,target)
        self.assertTrue(all(not ready for _,_,ready in votes))
        report=r._draft_head_qualification
        self.assertEqual(len(report['cases']),2)
        self.assertTrue(all(row['changed_inputs']==3 for row in report['cases']))
        self.assertTrue(report['target_fixed_hidden']['bit_exact'])
        for t,saved in before:self.assertTrue(torch.equal(t,saved))

    def test_reserved_k4_boot_publishes_combined_ready_after_qualification(self):
        r=fixture();r.speculator.num_speculative_steps=4;r.speculator._kstop.k4_capture=True
        boot(r)
        self.assertTrue(r._k4_drafthead_ready)
        self.assertIs(r._k4_drafthead_qualification,r._draft_head_qualification)
        self.assertTrue(r._draft_head.on);self.assertEqual(r._draft_head_epoch,-1)

    def test_mismatch_nonfinite_missing_factory_boot_off_failed_dh_gate(self):
        for fault in ('mismatch','nonfinite','missing'):
            with self.subTest(fault=fault):
                r=fixture(fault);before=qual.tensor_state(r.speculator)
                self.assertEqual(boot(r),'capture-result')
                self.assertFalse(r._draft_head_ready);self.assertFalse(r._k4_drafthead_ready)
                self.assertFalse(r._draft_head.on);self.assertEqual(r._draft_head_epoch,-1)
                self.assertIsNone(r.speculator._kstop.bad)
                self.assertFalse(dh.status(r)['dh_gate_passed'])
                self.assertIsNotNone(dh.status(r)['init_failure'])
                for t,saved in before:self.assertTrue(torch.equal(t,saved))

    def test_target_fixed_hidden_mismatch_abort(self):
        r=fixture();original=qual.qualify
        def corrupt(*a,**kw):
            report=original(*a,**kw);r.model.lm_head.weight.add_(1);return report
        with patch.object(qual,'qualify',corrupt):boot(r)
        self.assertFalse(r._draft_head.on)
        self.assertFalse(r._draft_head_ready)

    def test_final_collective_refusal_and_cleanup_failure_fallback(self):
        for fault in ('vote','cleanup'):
            r=fixture()
            def vote(payload,valid):completed_vote(payload,valid and not ('initial_qualified' in payload and fault=='vote'))
            calls=[]
            def cleanup():
                calls.append(1)
                if fault=='cleanup' and dh._qualification_terms.get() is not None:raise RuntimeError('cleanup')
            with patch.object(dh.gc,'collect',cleanup):boot(r,vote)
            self.assertFalse(r._draft_head.on)
            self.assertFalse(r._draft_head_ready)

    def test_native_capture_factory_hook(self):
        cls=type('CudaGraphManager',(),{'capture':lambda self,*a,**kw:42})
        dh.install_graph(NS(CudaGraphManager=cls,graph_capture=lambda **kw:None));m=cls();factory=lambda desc,warmup:lambda mode:None
        self.assertEqual(m.capture(factory),42);self.assertTrue(callable(m._k4_drafthead_factory))
        self.assertEqual(m.capture(create_forward_fn=factory),42)


    def test_log_terms_before_final_vote_and_receipt_contains_inner_exception(self):
        r=fixture('missing');rows=[]
        def vote(payload,valid):
            if 'initial_qualified' in payload:
                rows.extend(dh._qualification_terms.get())
            completed_vote(payload,valid)
        boot(r,vote)
        self.assertTrue(any(t['check']=='replay_prepare' and
            t['error']['text']=='missing native captured draft factory' for t in rows))
        self.assertTrue(any(v['terms'] for v in r._draft_head_init_failure['votes']))

    def test_runtime_errors_and_failed_off_recapture_remain_fatal(self):
        r=fixture()
        with patch.object(qual,'qualify',side_effect=RuntimeError('device fault')):
            # Local exception is converted by the completed final vote.
            boot(r)
        self.assertFalse(r._draft_head.on)
        r=fixture('mismatch')
        original=r.speculator.capture
        def capture():
            if not r._draft_head.on:raise RuntimeError('OFF capture failed')
            return original()
        r.speculator.capture=capture
        with self.assertRaises(dh.VoteRefused):boot(r)
        self.assertFalse(r._draft_head_ready)

    def test_fallback_refuses_changed_target_graphs_or_method(self):
        for fault in ('graphs','method'):
            r=fixture();original=qual.qualify
            def corrupt(*a,**kw):
                report=original(*a,**kw)
                if fault=='graphs':r.cudagraph_manager.graphs['changed']=object()
                else:r.model.lm_head.quant_method=type(r.model.lm_head.quant_method)()
                return report
            with self.subTest(fault=fault),patch.object(qual,'qualify',corrupt),self.assertRaises(dh.VoteRefused):
                boot(r)
            self.assertFalse(r._draft_head_ready)

    def test_real_descriptor_wrapper_order_eager_matches_capture(self):
        import glm_dsa_short as dsa
        @dataclass(frozen=True)
        class ShortDesc:
            short_context: bool=True
        for order in ('dsa-first', 'dh-first'):
            seen=[]
            def native(self,factory):
                factory(ShortDesc(),False)('FULL')
            cls=type('CudaGraphManager',(),dict(capture=native,dispatch=lambda *a:None))
            mod=NS(CudaGraphManager=cls,graph_capture=lambda **kw:None)
            if order=='dsa-first':
                dsa.install_cg(mod);dh.install_graph(mod)
            else:
                dh.install_graph(mod);dsa.install_cg(mod)
            # Initialization is irrelevant to factory wrapping; real DSA
            # install_cg capture callback is exercised in both hook orders.
            manager=object.__new__(cls)
            factory=lambda desc,warmup:lambda mode:seen.append(dsa.SHORT.get())
            with dsa.context(False):
                manager.capture(factory)
                manager._k4_drafthead_factory(ShortDesc(),False)('NONE')
            self.assertEqual(seen,[True,True],order)

    def test_sparse_output_columns_exclude_unwritten_nan_sentinels(self, corrupt=False):
        r=fixture();sp=r.speculator
        sp.draft_logits=torch.full((1,3,64),float('nan'))
        native=sp.capture
        def capture():
            native()
            for index,m in enumerate(dh.managers(r)):
                factory=m._k4_drafthead_factory
                def wrapped(desc,warmup,factory=factory):
                    forward=factory(desc,warmup)
                    def run(mode):
                        forward(mode)
                        sp.draft_logits[:,int(sp.current_draft_step)].fill_(1.)
                        if corrupt and r._draft_head.on:sp.draft_logits[:,1].fill_(3.)
                    return run
                m._k4_drafthead_factory=wrapped
                m.graphs[Desc()].replay=lambda run=wrapped(Desc(),False):run(None)
        sp.capture=capture;capture()
        boot(r)
        self.assertEqual(r._draft_head.on,not corrupt)
        self.assertEqual(r._draft_head_ready,not corrupt)
        if not corrupt:self.assertTrue(torch.isnan(sp.draft_logits[:,1]).all())

    def test_inactive_logit_column_corruption_refuses(self):
        self.test_sparse_output_columns_exclude_unwritten_nan_sentinels(corrupt=True)

if __name__=='__main__':unittest.main()
