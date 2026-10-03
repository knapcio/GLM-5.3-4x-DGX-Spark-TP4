# SPDX-License-Identifier: Apache-2.0
"""Reuse perf/spec-sample conditional distribution/poison oracles with K-stop K3.

Executes install_mtp's sample_draft, substituting Torch noise math for the native
Triton gumbel sampler; exact Triton distribution still requires a GPU receipt.
"""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'overlay/kstop'))
import kstop_runtime as rt
from kstop_policy import stop
from test_spec_sample import simulate_batch, SOURCE
from spec_sample_distribution import batch_cases,processor,ConditionalCounts


class KstopDistribution(unittest.TestCase):
    def test_k3_variable_stop_cached_laws_and_mixed_t0(self):
        profiles=batch_cases(processor(SOURCE),k=3)
        # Alongside native test laws (K1), add K2/K3 confidence regimes.
        for maximum in (.82,.90):
            c=copy.deepcopy(profiles[0]);c.name+='-qmax'+str(maximum)
            raw=torch.tensor([maximum,(1-maximum)*.5,(1-maximum)*.3,(1-maximum)*.2]).log()
            c.proposal=torch.stack([raw.roll(s) for s in range(3)])
            profiles.append(c)
        class Spec:
            def __init__(self,*a):pass
            def propose(self,*a):pass
            def _configure_fused_multi_step_decode(self):pass
        rt.install_mtp(Spec)
        for dtype in (torch.float32,torch.bfloat16):
            for c in profiles:
                c.q=(c.proposal.to(dtype).float().double()/(c.temperature or 1.)).softmax(-1)
                maximum=c.proposal.to(dtype).float().softmax(-1).amax(-1).tolist()
                lens,cum=stop([3],[1.],[maximum[0]],1,.74)
                if lens[0]>1:lens,cum=stop(lens,cum,[maximum[1]],2,.74)
                # Stop K1 -> placeholder column 1; K2 -> column 2; K3 -> bonus.
                bound=lens[0] if lens[0]<3 else -1
                original=getattr(c,'original_placeholder',c.placeholder)
                c.original_placeholder=original
                c.placeholder=bound if original<0 else original if bound<0 else min(original,bound)
            tally=ConditionalCounts(profiles)
            rngs=[torch.Generator().manual_seed(s) for s in (20261002,91239,20261002^0x53504543)]
            def sample(z,idx,temps,cache,step,noise):
                owner=NS(model=NS(compute_logits=lambda h:z),use_fp64_gumbel=False,_kstop=NS(confidence=torch.zeros(len(z))))
                def native(logits,m,t,seeds,pos,**kw):
                    live=m>=0;kw['logits_cache'][m[live].long(),step]=logits[live]
                    temperature=t[m.clamp(min=0).long()].double()
                    value=logits.double()/temperature.clamp(min=1e-10)[:,None]+noise
                    return torch.where(temperature==0,logits.argmax(-1),value.argmax(-1))
                with patch.dict(sys.modules,{'vllm.v1.worker.gpu.spec_decode.speculator':NS(gumbel_sample=native)}):
                    x=Spec.sample_draft(owner,None,torch.zeros(len(z),dtype=torch.int64),idx,temps,None,step,cache)
                expected=(z.float().amax(-1)-z.float().logsumexp(-1)).exp()
                assert torch.equal(owner._kstop.confidence,expected),'stop read sampled-token q instead of max q'
                return x
            draws=1_000_000*len(profiles)
            for offset in range(0,draws,20_000):
                tally.add(*simulate_batch(profiles,min(20_000,draws-offset),offset,rngs,dtype=dtype,draft_sampler=sample))
            results=tally.check()
            print('KSTOP K3 conditional distributions PASS',dtype,len(results))


if __name__=='__main__':unittest.main(verbosity=2)
