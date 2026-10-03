# SPDX-License-Identifier: Apache-2.0
"""Async TP4 MTP stop, piggybacked on the existing exact-length planner D2H.

Control modes (GLM_MTP_KSTOP_CONTROL_MODE):
  local (default)  every TP rank decides from its own copy of the
                   vocab-parallel MTP logits (LogitsProcessor all-gathers the
                   full vocabulary). No Gloo status all-gather, no rank0
                   broadcast and no per-cycle control-file read.
                   Fix3 guard: before the target forward and before each
                   further MTP pass decision, every rank enqueues ONE fixed
                   13 x int64 (104 B) MAX all-reduce on the existing vLLM
                   PyNccl communicator: [w0,w1,c0..c3,bad,-w0,-w1,-c0..-c3]
                   with w = 124 bits of a running SHA-256 over every
                   decision-relevant host input (probabilistic), c = the exact
                   FP32 bits of the probabilities the decision will read and
                   bad = local refusal. The device-side ok flag rides in the
                   EXISTING planner D2H (packed_host); the host checks it
                   before the decision / before the forward launch. Every rank
                   reads the same reduced flag at the same host point, so a
                   mismatch raises the same Python error on all ranks before
                   any divergent launch: no device assert (no CUDA context
                   corruption), no early one-rank raise. A rank that detects
                   invalid local state folds it into bad and continues in
                   lockstep until that common check point.
  broadcast        OFFLINE CPU TESTS ONLY (round-1 rank0 authority with fixed
                   Gloo all-gathers/broadcasts). Prohibited on the fleet: one
                   small Gloo all-gather costs ~3.3 ms on these Sparks, 5-7 per
                   stop cycle. Refused unless GLM_MTP_KSTOP_BROADCAST_OFFLINE_TEST=1.
  GLM_MTP_KSTOP_UNIFORM_BATCH=1 (default 0): with more than one request in
                   the draft batch, every request drafts all three tokens (no
                   stop decision, no decision guard, no confidence read), so the
                   next verify rectangle is uniform q4 and replays a graph
                   instead of the eager heterogeneous path. The choice depends
                   only on num_reqs, which every rank holds identically (bound
                   by the begin fold and the prepare guard); the flag itself is
                   bound by the cold control agreement. One request keeps the
                   confidence stop.
                   k2: the same count-only policy drafts exactly two tokens,
                   yielding uniform q3 verify rectangles. 1 retains full K3.
Credits: knapcio full-GLM campaign cross-domain review, mechanism #1 (rank-replicated deterministic decisions).
"""
import collections
import copy
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import torch
from kstop_policy import stop, sources

STATE=None
CONTROL_MODES=('local','broadcast')
MODES=('k2','k3-stop','k3')
BROADCAST_TEST_FLAG='GLM_MTP_KSTOP_BROADCAST_OFFLINE_TEST'
UNIFORM_FLAG='GLM_MTP_KSTOP_UNIFORM_BATCH'
GUARD_SLOTS=4   # install_mtp refuses max_num_seqs>4
GUARD_WORDS=2   # 2 x 62-bit words of the host-state SHA-256 (124 bits)
GUARD_LEN=2*(GUARD_WORDS+GUARD_SLOTS)+1


def broadcast_allowed(env=None):
    env=os.environ if env is None else env
    return env.get(BROADCAST_TEST_FLAG)=='1'


def control_mode(env=None):
    env=os.environ if env is None else env
    mode=env.get('GLM_MTP_KSTOP_CONTROL_MODE','local') or 'local'
    if mode not in CONTROL_MODES:raise ValueError('GLM_MTP_KSTOP_CONTROL_MODE must be local')
    if mode=='broadcast' and not broadcast_allowed(env):
        raise ValueError('broadcast kstop control is prohibited on the fleet (~3.3 ms per Gloo all-gather); offline CPU tests only')
    return mode


def uniform_batch_policy(env=None):
    env=os.environ if env is None else env
    value=env.get(UNIFORM_FLAG,'0') or '0'
    if value not in ('0','1','k2'):raise ValueError(UNIFORM_FLAG+' must be 0, 1 or k2')
    return value


def pad_hygiene(env=None):
    env=os.environ if env is None else env
    value=env.get('GLM_PAD_HYGIENE','0')
    if value not in ('0','1'):raise ValueError('GLM_PAD_HYGIENE must be 0 or 1')
    return value=='1'


def uniform_batch(env=None):
    return uniform_batch_policy(env)!='0'


def _is_cuda(device):
    return torch.device(device).type=='cuda'


def tp_group():
    from vllm.distributed.parallel_state import get_tp_group
    return get_tp_group()


def check_control(c, current_epoch=-1):
    if set(c)!={'schema','mode','epoch','tau'} or c['schema']!=1:raise ValueError('control schema')
    if type(c['epoch']) is not int or not 0<=c['epoch']<2**53 or c['epoch']<current_epoch:raise ValueError('epoch regression')
    if c['mode'] not in MODES:raise ValueError('mode')
    if type(c['tau']) not in (int,float) or not .6<=c['tau']<=.85:raise ValueError('tau')
    return c


class Runtime:
    def __init__(self, owner, control=None):
        self.owner=owner;self.round=0;self.epoch=-1;self.mode=None;self.tau=.74
        self.used=set();self.proposals={};self.synthetic=False;self.pending=None;self.ready=None
        self.confidence=torch.zeros(owner.max_num_reqs,dtype=torch.float32,device=owner.device)
        self.lengths=[];self.cumulative=[];self.ids=[]
        self.control=control or control_mode()
        if self.control not in CONTROL_MODES:raise ValueError('control mode')
        if self.control=='broadcast' and not broadcast_allowed():
            raise ValueError('broadcast kstop control is prohibited on the fleet; offline CPU tests only')
        # Local-mode guard state. trail is a running SHA256 over every
        # decision-relevant host input; bad is a sticky local refusal reason.
        # guard_pending = (stage, device ok flag) until a host check consumes it.
        self.trail=bytes(32);self.bad=None;self.guards=0;self.folds=0
        self.guard_pending=None;self.fallback_syncs=0;self.inputs_guarded=False
        # Keep host fold inputs only until a successful guard. Field hashes and
        # peer exchange are computed exclusively on the collective failure path.
        self.guard_inputs_log=[]
        self.history=collections.deque(maxlen=512);self.events=collections.deque(maxlen=256)
        # Optional uniform-batch policy (default off: no state, payload or code-path change).
        self.uniform=uniform_batch();self.uniform_k=2 if uniform_batch_policy()=='k2' else 3
        self.capture_layout=os.environ.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT','m12')
        if self.capture_layout not in ('m12','reuse'):raise ValueError('GLM_MTP_KSTOP_CAPTURE_LAYOUT must be m12 or reuse')
        self.pad_hygiene=pad_hygiene()
        self.uniform_now=False;self.uniform_cycles=0
        self.sample_mode=getattr(getattr(owner,'speculative_config',None),'draft_sample_method','greedy')
        self.shortcut=os.environ.get('GLM_INDEXER_SHORTCUT','0')=='1'

    def forget(self, ids):
        for rid in ids:self.proposals.pop(rid,None);self.used.discard(rid)

    # ---- broadcast-mode primitives (also used on cold, drained paths) ----
    def agree(self, stage, payload, valid=True):
        """Fixed-size CPU all-gather BEFORE any rank can choose another graph.

        Four 63-bit SHA256 words bind host-only state (no device wait). A rank
        refusal is data until all peers have acknowledged it. Missing/crashed
        ranks are bounded by the group timeout and external coordinator abort.
        """
        group=tp_group()
        digest=hashlib.sha256(json.dumps([stage,self.round,self.epoch,self.mode,
            self.ids,payload],sort_keys=True).encode()).digest()
        words=[int.from_bytes(digest[i:i+8],'little') & ((1<<63)-1) for i in range(0,32,8)]
        row=torch.tensor([int(valid),*words],dtype=torch.int64)
        rows=[torch.empty_like(row) for _ in group.ranks]
        torch.distributed.all_gather(rows,row,group=group.cpu_group)
        if any(x[0]!=1 or not torch.equal(x[1:],row[1:]) for x in rows):
            raise RuntimeError('collective-safe kstop refusal at '+stage+
                ' (planner/proposal/request order/epoch); no extra-sync fallback')

    def broadcast(self, values):
        # One 16-double CPU broadcast: error/mode/epoch/tau/n + 4 lengths +
        # 4 cumulative probabilities. No pickle, object broadcast, GPU transfer.
        group=tp_group();packet=torch.zeros(16,dtype=torch.float64)
        if group.rank_in_group==0:packet[:len(values)]=torch.tensor(values,dtype=torch.float64)
        torch.distributed.broadcast(packet,src=group.ranks[0],group=group.cpu_group)
        return packet.tolist()

    # ---- local-mode primitives ----
    def fold(self, stage, payload, valid=True):
        """Host-only: extend the running digest; never communicates."""
        meta=[stage,self.round,self.epoch,self.mode,self.tau,list(self.ids)]
        self.guard_inputs_log.append((stage,meta,payload,bool(valid)))
        self.trail=hashlib.sha256(self.trail+json.dumps([*meta,payload],sort_keys=True,allow_nan=True).encode()).digest()
        self.folds+=1
        if not valid and self.bad is None:self.bad=stage

    def guard(self, stage, n=0):
        """Enqueue one fixed 13 x int64 MAX all-reduce; never waits, never asserts.

        [w0,w1,c0..c3,bad,-w0,-w1,-c0..-c3]: ok iff every max equals minus the
        max of its negation (all ranks equal) and max(bad)==0. c are the exact
        FP32 bits of confidence[:n] (zero padded), so a decision is checked on
        its INPUTS before it is made. The device ok flag is checked on the host
        by the next packed_host (existing planner D2H) or, if no planner copy
        precedes the next launch, by a counted synchronous read. Same size on
        every rank whatever the local state (no mismatched-count collective).
        """
        if self.guard_pending is not None:self.check_guard()  # never stack guards
        if not 0<=n<=GUARD_SLOTS:self.fold('guard-slots',[n],False);n=0
        group=tp_group();cuda=_is_cuda(self.owner.device);device=torch.device(self.owner.device)
        words=[int.from_bytes(self.trail[i:i+8],'little')&((1<<62)-1) for i in range(0,8*GUARD_WORDS,8)]
        host=torch.tensor([*words,1 if self.bad else 0],dtype=torch.int64,pin_memory=cuda)
        if cuda:
            comm=getattr(getattr(group,'device_communicator',None),'pynccl_comm',None)
            if comm is None or comm.disabled:raise RuntimeError('local kstop guard requires the PyNccl TP communicator')
            host=host.to(device,non_blocking=True)
        conf=torch.zeros(GUARD_SLOTS,dtype=torch.int64,device=device)
        if n:conf[:n]=self.confidence[:n].view(torch.int32).to(torch.int64)
        self.guard_confidence=conf
        payload=torch.cat((host[:GUARD_WORDS],conf,host[GUARD_WORDS:],-host[:GUARD_WORDS],-conf))
        if cuda:out=comm.all_reduce(payload,op=torch.distributed.ReduceOp.MAX)
        else:
            torch.distributed.all_reduce(payload,op=torch.distributed.ReduceOp.MAX,group=group.cpu_group);out=payload
        k=GUARD_WORDS+GUARD_SLOTS
        ok=(out[:k]==-out[k+1:]).all()&(out[k]==0)
        self.guard_pending=(stage,ok);self.guards+=1

    def check_guard(self, value=None):
        """Host check of the pending guard. value comes from the shared planner
        D2H; None means a counted synchronous fallback read. Every rank holds
        the same reduced flag, so every rank raises here, at the same point."""
        stage,ok=self.guard_pending;self.guard_pending=None
        if value is None:
            value=float(ok.to(torch.float64).reshape(1).cpu()[0]);self.fallback_syncs+=1
        if value!=1.0:
            self.diagnose_guard(stage)
            raise RuntimeError('collective-safe kstop refusal at '+stage+' (guard: digest/probability-bit/validity '
                'mismatch across ranks; planner/proposal/request order/epoch/probability'+
                ('; local: '+self.bad if self.bad else '')+'); every rank refuses at this host point')
        self.guard_inputs_log.clear()

    def guard_field_digests(self):
        """Failure-only host rendering; also usable by simulated TP tests."""
        def digest(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=True).encode()).hexdigest()
        fields={}
        for i,(name,meta,payload,valid) in enumerate(self.guard_inputs_log):
            prefix=str(i)+':'+name+'.'
            for key,value in zip(('stage','round','epoch','mode','tau','request_ids'),meta):
                fields[prefix+'context.'+key]=digest(value)
            fields[prefix+'valid']=digest(valid)
            labels={'begin':('num_reqs','used'), 'select':('request_order','token_counts','spec_widths','proposals'),
                'arm':('step','num_reqs','lengths','cumulative'),
                'decision':('step','num_reqs','lengths','cumulative','probabilities')}.get(name)
            values=payload if isinstance(payload,dict) else dict(zip(labels,payload)) if labels else {'payload':payload}
            for key,value in values.items():fields[prefix+key]=digest(value)
        fields['trail']=self.trail.hex();fields['validity']=digest(self.bad)
        fields['confidence']=digest(self.guard_confidence.cpu().tolist())
        return fields

    def diagnose_guard(self, stage):
        """All ranks have the same failed flag: cold Gloo exchange before raise.

        No device read/field hashing/extra collective on the success path.
        The prior trail already passed a guard; differences must be in these
        folds, live confidence bits, or sticky validity. Full trail is included
        to distinguish an unexpected history mismatch.
        """
        fields=self.guard_field_digests()
        group=tp_group();peers=[None for _ in group.ranks]
        try:torch.distributed.all_gather_object(peers,fields,group=group.cpu_group)
        except Exception as e:
            # Diagnostics must not replace the original collective refusal.
            print('KSTOP_GUARD_FIELDS '+json.dumps(dict(stage=stage,rank=group.rank_in_group,
                subdigests=fields,diagnostic_error=str(e)),sort_keys=True),file=sys.stderr,flush=True)
            return
        keys=sorted(set().union(*(p.keys() for p in peers)))
        mismatched=[key for key in keys if len({p.get(key) for p in peers})>1]
        print('KSTOP_GUARD_FIELDS '+json.dumps(dict(stage=stage,rank=group.rank_in_group,
            mismatched=mismatched,subdigests=fields,peers=peers),sort_keys=True),file=sys.stderr,flush=True)

    def load_control(self):
        """Local mode, once per process: own control copy + one cold agree."""
        valid=True
        try:
            c=check_control(json.loads(Path(os.environ['GLM_MTP_KSTOP_CONTROL']).read_text()),self.epoch)
        except Exception:
            valid=False;c=dict(mode='k2',epoch=0,tau=.74)
        self.agree('control',[c['mode'],c['epoch'],c['tau']]+(['uniform-batch'] if self.uniform else [])+
            (['k2'] if self.uniform and self.uniform_k==2 else [])+
            (['reuse-m6'] if self.uniform and self.uniform_k==2 and self.capture_layout=='reuse' else [])+
            (['probabilistic',bool(getattr(self.owner,'use_fp64_gumbel',False))] if self.sample_mode=='probabilistic' else [])+
            (['short-dsa'] if self.shortcut else [])+
            (['pad-hygiene'] if self.pad_hygiene else []),valid)
        self.mode,self.epoch,self.tau=c['mode'],c['epoch'],float(c['tau'])

    def set_control(self, mode, epoch, tau):
        """Drained switch on every rank (local mode). Cold CPU agreement."""
        valid=True
        try:check_control(dict(schema=1,mode=mode,epoch=epoch,tau=tau),self.epoch+1)
        except Exception:valid=False
        self.agree('switch',[mode,epoch,tau],valid)
        self.mode,self.epoch,self.tau=mode,epoch,float(tau)
        self.trail=bytes(32);self.bad=None;self.guard_pending=None;self.inputs_guarded=False
        self.guard_inputs_log.clear()

    # ---- per-cycle protocol ----
    def begin(self, batch, synthetic=False):
        self.synthetic=bool(synthetic or getattr(self.owner,'_kstop_synthetic',False))
        self.ids=list(batch.req_ids);self.pending=self.ready=None;self.uniform_now=False
        if self.synthetic:
            self.lengths=[3]*batch.num_reqs;self.cumulative=[1.]*batch.num_reqs;return 3
        if self.control=='local':
            if self.mode is None:self.load_control()
            self.fold('begin',[batch.num_reqs,sorted(self.used)],0<batch.num_reqs<=4)
        else:
            self.agree('begin', [batch.num_reqs,sorted(self.used),self.tau],0<batch.num_reqs<=4)
            group=tp_group();packet=[]
            if group.rank_in_group==0:
                try:
                    c=check_control(json.loads(Path(os.environ['GLM_MTP_KSTOP_CONTROL']).read_text()),self.epoch)
                    changed=self.mode is not None and (c['mode'],c['tau'])!=(self.mode,self.tau)
                    if changed and (c['epoch']==self.epoch or self.used):raise ValueError('switch requires drained/new epoch')
                    packet=[0,MODES.index(c['mode']),c['epoch'],c['tau']]
                except Exception:packet=[1]
            packet=self.broadcast(packet)
            if packet[0]:raise RuntimeError('rank0 kstop control failed: schema/epoch/drained control')
            self.mode=MODES[int(packet[1])];self.epoch=int(packet[2]);self.tau=packet[3]
        if self.uniform and self.mode=='k3-stop':
            # Rank-identical: num_reqs is bound by the begin fold / agree and the prepare guard.
            self.uniform_now=batch.num_reqs>1;self.uniform_cycles+=self.uniform_now
            if self.control=='local':self.fold('uniform',[batch.num_reqs,self.uniform_now])
        k=self.uniform_k if self.uniform_now else (2 if self.mode=='k2' else 3)
        self.lengths=[k]*batch.num_reqs;self.cumulative=[1.]*batch.num_reqs
        self.round+=1;self.used.update(self.ids);return k

    def can_advance(self, step, n):
        # Do not build prospective third-pass metadata or copy planner lengths
        # when an earlier decision already capped every request at two.
        if self.synthetic:return True
        # Local: lengths are the last guarded decision; no communication.
        if self.control=='broadcast':self.agree('advance', [step,n,self.lengths,self.cumulative])
        return any(k>step for k in self.lengths)

    def arm(self, step, n):
        self.pending=None;self.ready=None
        if not self.synthetic and self.mode=='k3-stop' and not self.uniform_now:
            self.pending=(step,n)
            if self.control=='local':
                # Check the decision INPUTS (host state + exact probability
                # bits) before the decision; result rides the planner copy.
                self.fold('arm',[step,n,self.lengths,self.cumulative])
                self.guard('decision-'+str(step),n)

    def packed_host(self, values):
        """Replace ONE native .cpu(): lengths and confidence share its wait.

        Float64 exactly preserves integer positions and FP32 probabilities.
        No .cpu(), .item() or synchronize elsewhere on the hot policy path.
        Every TP rank reaches this planner; in local mode each rank keeps its
        own (bit-identical) probabilities, in broadcast mode rank0's rule.
        """
        g=self.guard_pending
        if self.pending is None and g is None:return values.cpu()
        parts=[values.to(torch.float64)];m=values.numel()
        if self.pending is not None:step,n=self.pending;parts.append(self.confidence[:n].to(torch.float64))
        if g is not None:parts.append(g[1].to(torch.float64).reshape(1))
        host=torch.cat(parts).cpu()
        # Guard first: the symmetric refusal point precedes any use of values.
        if g is not None:self.check_guard(float(host[-1]))
        if self.pending is not None:
            self.ready=(step,n,host[m:m+n].tolist());self.pending=None
        return host[:m].to(values.dtype)

    def after_metadata(self, step, n):
        if self.synthetic or self.mode!='k3-stop' or self.uniform_now:return True
        valid=self.ready is not None and self.ready[:2]==(step,n)
        if self.control=='local':
            if valid:probs=self.ready[2]
            else:
                # Planner copy bypassed (deterministic code path in serving;
                # injected on one rank in tests). Counted synchronous fallback:
                # check the pending guard, read the same verified probability
                # bits, mark the state invalid and stay in lockstep; the next
                # guard refuses on every rank. No local early raise.
                if self.guard_pending is not None:self.check_guard()
                probs=self.confidence[:n].to(torch.float64).cpu().tolist();self.fallback_syncs+=1
                self.fold('decision-planner-missing',[step,n],False)
            try:lens,cum=stop(self.lengths,self.cumulative,probs,step,self.tau)
            except ValueError as e:
                # Inputs were verified bit-identical on every rank, so every
                # rank raises this same error at the same point.
                raise RuntimeError('collective-safe kstop refusal at decision-'+str(step)+' (invalid probability on every rank: '+str(e)+')')
            self.fold('decision',[step,n,lens,cum,probs]);self.ready=None
            self.history.append(dict(round=self.round,epoch=self.epoch,step=step,
                p=[float(x).hex() for x in probs],lengths=list(lens)))
            self.lengths=list(lens);self.cumulative=list(cum)
            return any(k>step for k in self.lengths)
        self.agree('planner', [step,n,self.lengths,self.cumulative],valid)
        group=tp_group();packet=[]
        if group.rank_in_group==0:
            try:
                lens,cum=stop(self.lengths,self.cumulative,self.ready[2],step,self.tau)
                packet=[0,n,*lens,*([0]*(4-n)),*cum,*([0]*(4-n))]
            except Exception:packet=[1]
        packet=self.broadcast(packet);self.ready=None
        if packet[0]:raise RuntimeError('rank0 kstop confidence failed')
        self.lengths=[int(x) for x in packet[2:2+n]];self.cumulative=packet[6:6+n]
        return any(k>step for k in self.lengths)

    def finish(self, result):
        self.pending=self.ready=None
        if self.synthetic:return result
        # Persistent device drafts remain valid embedding IDs. Sampler masks
        # physically dead rows separately; never feed -1 into MTP embeddings.
        for i,(rid,k) in enumerate(zip(self.ids,self.lengths)):
            result[i,k:]=0
            self.proposals[rid]=k
        return result

    def after_warmup(self):
        """Fix19: called by the worker right after warmup_kernels. Warmup is
        synthetic for kstop (no proposals, folds, guards or events); any state
        it left behind would surface at the first real request, so refuse at
        startup instead (deterministic, identical on every rank)."""
        left=dict(bad=self.bad,guard_pending=self.guard_pending is not None,inputs_guarded=self.inputs_guarded,
            proposals=sorted(self.proposals),used=sorted(self.used),pending=self.pending,ready=self.ready)
        if self.bad is not None or self.guard_pending is not None or self.inputs_guarded or self.proposals or self.used or self.pending or self.ready:
            raise RuntimeError('kstop state not clean after warmup: '+json.dumps(left,sort_keys=True,default=str))
        self.ids=[];self.lengths=[];self.cumulative=[];self.synthetic=False
        return left

    def audit(self):
        out=dict(control=self.control,trail=self.trail.hex(),guards=self.guards,folds=self.folds,
            bad=self.bad,mode=self.mode,epoch=self.epoch,tau=self.tau,round=self.round,
            fallback_syncs=self.fallback_syncs,guard_pending=self.guard_pending is not None)
        if self.uniform:out.update(uniform_batch=True,uniform_cycles=self.uniform_cycles)
        if self.uniform and self.uniform_k==2:out['uniform_k']=2
        return out


def packed_host(values):
    if STATE is None:return values.cpu()
    return STATE['runtime'].packed_host(values)


def check_local(runner, group):
    """Startup preconditions of local control; identical on every rank."""
    comm=getattr(getattr(group,'device_communicator',None),'pynccl_comm',None)
    if comm is None or comm.disabled:
        raise RuntimeError('local kstop control requires the existing PyNccl TP communicator (broadcast control is prohibited on the fleet; no fallback)')
    # fix18: the drafter is DeepSeekMTP; its only LogitsProcessor is
    # DeepSeekMultiTokenPredictor.logits_processor (deepseek_mtp.py), applied
    # to the shared head of the MTP layer in compute_logits.
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    lps=[x for x in runner.speculator.model.modules() if isinstance(x,LogitsProcessor)]
    if not lps or any(not x.use_all_gather or x.logits_as_input for x in lps):
        raise RuntimeError('local kstop control requires the all-gather MTP LogitsProcessor (rank-identical logits)')
    return True


# Fix18 (window 6 stop boot: 'full GLM 256-expert model required'): the
# served checkpoint is the FULL GLM-5.3 (architectures GlmMoeDsaForCausalLM,
# model_type glm_moe_dsa), which the pinned vLLM registry serves from
# vllm.model_executor.models.deepseek_v2 (GlmMoeDsaForCausalLM subclasses
# DeepseekV2ForCausalLM; MoE layers are DeepseekV2MoE). Its MTP drafter is
# DeepSeekMTP (vllm.model_executor.models.deepseek_mtp; speculative.py
# rewrites glm_moe_dsa -> deepseek_mtp / DeepSeekMTPModel) loaded by the
# generic MTPSpeculator. The fix7-fix17 code looked for the GLM-5.3-FLASH class
# vllm.models.glm5next.nvidia.model.Glm5NextMoE, found none and failed closed.
TARGET_ARCH='GlmMoeDsaForCausalLM';DRAFT_ARCH='DeepSeekMTP'
FULL_GLM=dict(model_type='glm_moe_dsa',n_routed_experts=256,num_experts_per_tok=8,hidden_size=6144,moe_intermediate_size=2048,
    kv_lora_rank=512,vocab_size=154880,num_nextn_predict_layers=1,first_k_dense_replace=3)


def model_binding(runner):
    """Fix18: the live target and drafter are the full GLM-5.3 classes the
    pinned vLLM builds; returns the target's DeepseekV2MoE layers. Fails closed
    on any other model family (e.g. GLM-5.3-Flash glm5next)."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE,GlmMoeDsaForCausalLM
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    # These unmodified bodies define MTP compaction/reuse and native sampling.
    # Validate them before capture alongside the transformed-source pins.
    pins=json.loads(Path(__file__).with_name('compat_source_pins.json').read_text())
    for name,want in pins.items():
        module=__import__(name,fromlist=['__file__'])
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=want:
            raise RuntimeError('kstop compatibility source drift: '+name)
    model=runner.model
    if not isinstance(model,GlmMoeDsaForCausalLM) or type(model).__name__!=TARGET_ARCH:
        raise RuntimeError('kstop requires the full GLM-5.3 target %s (vllm deepseek_v2), got %s'%(TARGET_ARCH,type(model).__module__+'.'+type(model).__name__))
    cfg=model.config;bad={k:getattr(cfg,k,None) for k,v in FULL_GLM.items() if getattr(cfg,k,None)!=v}
    if bad:raise RuntimeError('kstop requires the full GLM-5.3 config; differs: '+json.dumps(bad,sort_keys=True,default=str))
    freq=getattr(cfg,'moe_layer_freq',1) or 1
    want=sum(1 for i in range(cfg.num_hidden_layers) if i>=cfg.first_k_dense_replace and i%freq==0)
    moes=[m for m in model.modules() if isinstance(m,DeepseekV2MoE)]
    if not moes or len(moes)!=want or any(m.n_routed_experts!=256 for m in moes):
        raise RuntimeError('full GLM 256-expert model required (%d DeepseekV2MoE layers, config expects %d)'%(len(moes),want))
    sp=getattr(runner,'speculator',None);draft=getattr(sp,'model',None)
    if not isinstance(sp,MTPSpeculator) or not isinstance(draft,DeepSeekMTP) or type(draft).__name__!=DRAFT_ARCH:
        raise RuntimeError('kstop requires the native %s drafter under MTPSpeculator, got %s / %s'%(DRAFT_ARCH,type(sp).__name__,type(draft).__name__))
    return moes


def initialize(runner):
    global STATE
    if STATE is not None:raise RuntimeError('one runner per worker required')
    moes=model_binding(runner)
    r=runner.speculator._kstop
    if r.pad_hygiene:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
        draft_moes=[m for m in runner.speculator.model.modules() if isinstance(m,DeepseekV2MoE)]
        if not draft_moes:raise RuntimeError('pad hygiene requires native draft MoE hook')
        moes=moes+draft_moes
    for m in moes:
        q=m.experts._quant_method;kernel=getattr(q,'moe_kernel',None);impl=getattr(kernel,'impl',kernel)
        if q.is_monolithic or 'Marlin' not in type(getattr(impl,'fused_experts',None)).__name__:
            raise RuntimeError('kstop dead rows require modular Marlin topk hook')
    if r.control=='local':check_local(runner,tp_group())
    import deadrow_ops  # existing unchanged opaque custom op registered before capture
    m=runner.max_num_tokens
    STATE=dict(runner=runner,runtime=r,
        rows=torch.arange(m,device=runner.device,dtype=torch.int64),
        src=torch.arange(m,device=runner.device,dtype=torch.int64),
        dead=torch.zeros(m,device=runner.device,dtype=torch.bool),identity=True)


def select_inputs(runner, output):
    """Worker-local schedule view; scheduler retains original optimistic budget.

    Native async scheduler rolls back ALL unsampled budget (including skipped
    proposals), while worker post_update uses actual physical query lengths.
    No changed scheduler output object is returned to its owner.
    """
    r=runner.speculator._kstop
    # Fix19 (window 6 rerun: 'local: select' on the first request). The V2
    # worker's warmup_kernels (gpu/warmup.py) runs NON-dummy execute_model steps
    # whose decode steps carry spec tokens ([0]*K) for _warmup_i_ requests,
    # while their propose is synthetic (never recorded). select_inputs used to
    # fold those steps as invalid; no guard runs during warmup, so the sticky
    # local refusal surfaced at the first real request's prepare guard on every
    # rank. Warmup steps are synthetic here too: no fold, no event, no forget,
    # identity widths (prepare/guard_inputs already skip them).
    if getattr(r.owner,'_kstop_warmup',False):
        runner._kstop_needs_remap=False;return output
    preempted=output.preempted_req_ids or set()
    if preempted:r.events.append(dict(event='preempted',round=r.round,ids=sorted(preempted)))
    if output.finished_req_ids:r.events.append(dict(event='finished',round=r.round,ids=sorted(output.finished_req_ids)))
    r.forget(output.finished_req_ids | preempted)
    new={req.req_id for req in output.scheduled_new_reqs}
    r.forget(new)
    ds=output.scheduled_spec_decode_tokens
    # Fix19: a request scheduled NEW (or, on the V2 runner, resumed after
    # preemption) in THIS step can carry spec tokens only as the scheduler's
    # uniform-decode padding (scheduler.py pad_spec_decode: [-1]*K for a
    # 1-token remainder while decodes run). No propose has run for it, by
    # construction: verify it at the full scheduled width (no dead rows), the
    # same on every rank, folded below. A missing proposal for any OTHER request
    # stays invalid (fail closed).
    for rid in sorted(new):
        if rid in ds and output.num_scheduled_tokens.get(rid,1)>1:r.proposals[rid]=len(ds[rid])
    # Bind order, current epoch and proposal contents before physical dispatch.
    ordered=list(output.num_scheduled_tokens)
    proposal=[(rid,r.proposals.get(rid)) for rid in ordered]
    valid=all(r.proposals.get(rid) in (1,2,3) for rid in ds
        if output.num_scheduled_tokens.get(rid,1)>1)
    payload=[ordered,list(output.num_scheduled_tokens.values()),
        [(rid,len(tokens)) for rid,tokens in ds.items()],proposal]
    # Local: folded here and checked by the prepare guard, which runs before
    # the target forward and before any locally raised refusal.
    if r.control=='local':r.fold('select',payload,valid)
    else:r.agree('select',payload,valid)
    runner._kstop_needs_remap=any(r.proposals.get(rid,3)<len(tokens) for rid,tokens in ds.items())
    # Uniform rectangles select the actual q2/q3/q4 graph. Heterogeneous
    # rectangles retain their physical width and suppress only dead routes.
    if not ds or set(ds)!=set(output.num_scheduled_tokens):return output
    ks={rid:r.proposals.get(rid) for rid in ds}
    if any(k is None or output.num_scheduled_tokens[rid]!=len(ds[rid])+1 for rid,k in ks.items()):return output
    ks={rid:min(k,len(ds[rid])) for rid,k in ks.items()}
    if len(set(ks.values()))!=1:return output
    view=copy.copy(output);view.num_scheduled_tokens=dict(output.num_scheduled_tokens)
    view.scheduled_spec_decode_tokens=dict(ds)
    for rid,k in ks.items():
        view.num_scheduled_tokens[rid]=k+1;view.scheduled_spec_decode_tokens[rid]=ds[rid][:k]
    view.total_num_scheduled_tokens=sum(view.num_scheduled_tokens.values())
    runner._kstop_needs_remap=False
    return view



def _inputs(r,batch):
    widths=[int(b-a) for a,b in zip(batch.query_start_loc_np,batch.query_start_loc_np[1:])][:batch.num_reqs]
    valid=all(batch.is_prefilling_np[i] or w==1 or
        (r.proposals.get(rid) in (1,2,3) and 2<=w<=4)
        for i,(rid,w) in enumerate(zip(batch.req_ids,widths)))
    payload=dict(request_ids=list(batch.req_ids),widths=widths,
        prefilling=[bool(x) for x in batch.is_prefilling_np[:batch.num_reqs]],
        proposals=[(rid,r.proposals.get(rid)) for rid in batch.req_ids])
    # Host UVA mirrors: no device read or additional synchronization. These
    # determine the Gumbel draft law and every later sampled-prefix stop input.
    if r.sample_mode=='probabilistic':
        try:
            ss=STATE['runner'].sampler.sampling_states
            idx=[int(x) for x in batch.idx_mapping_np[:batch.num_reqs]]
            if (len(idx)!=batch.num_reqs or len(batch.req_ids)!=batch.num_reqs or
                len(widths)!=batch.num_reqs or any(x<0 or x>=len(ss.temperature.np) or
                    x>=len(ss.seeds.np) for x in idx)):raise ValueError('live sampling mapping')
            temps=[float(ss.temperature.np[x]) for x in idx]
            if any(not math.isfinite(t) or t<0 for t in temps):raise ValueError('temperature')
            # Slots are worker-local: native set-based cleanup changes the
            # free-list order across PYTHONHASHSEED values. Bind VALUES to
            # request IDs in batch order, never the storage addresses.
            ids=batch.req_ids
            payload['sampling.temperature']=[(rid,(t if t else 0.).hex()) for rid,t in zip(ids,temps)]
            # Native T=0 does not load seeds/positions. Never hash an unused
            # random seed, nor top-k/p/min-p: stock drafts ignore those.
            # CPU upper bounds do feed draft attention metadata, so retain
            # their live prefix even though Gumbel reads positions instead.
            payload['sampling.seeds']=[(rid,int(ss.seeds.np[x]) if t else None) for rid,x,t in zip(ids,idx,temps)]
            lengths=batch.seq_lens_cpu_upper_bound[:batch.num_reqs].tolist()
            if len(lengths)!=batch.num_reqs:raise ValueError('live sampling lengths')
            payload['sampling.lengths']=list(zip(ids,lengths))
        except (AttributeError,KeyError,TypeError,IndexError,ValueError,OverflowError):valid=False;payload['sampling.missing']=True
    if r.pad_hygiene:
        payload['pad-hygiene']=[int(batch.num_tokens),int(batch.num_tokens_after_padding)]
    if r.shortcut:
        try:
            payload['short-dsa.lengths']=batch.seq_lens_cpu_upper_bound[:batch.num_reqs].tolist()
            payload['short-dsa.prefill']=bool(batch.has_prefill)
            payload['short-dsa.padding']=int(batch.num_tokens_after_padding)
        except (AttributeError,TypeError,IndexError,ValueError,OverflowError):valid=False;payload['short-dsa.missing']=True
    return widths,valid,payload


def guard_inputs(runner,batch,dummy_run):
    """Fix3 hook after prepare_inputs and BEFORE prepare_attn: fold the select
    and prepare payloads and enqueue the prepare guard, whose flag then rides
    the target's existing SM90 planner D2H inside prepare_attn. The host check
    therefore precedes every width-dependent collective of the target forward."""
    s=STATE
    if s is None:raise RuntimeError('dead-row state not initialized before profile/capture')
    r=s['runtime']
    if dummy_run or getattr(r.owner,'_kstop_warmup',False) or r.control!='local':return
    _,valid,payload=_inputs(r,batch)
    r.fold('prepare',payload,valid);r.guard('prepare');r.inputs_guarded=True


def prepare(runner,batch,dummy_run):
    s=STATE
    if s is None:raise RuntimeError('dead-row state not initialized before profile/capture')
    s['src'].copy_(s['rows']);s['dead'].zero_();s['identity']=True
    r=s['runtime']
    # Check the boot flag before evaluating any padding-only batch fields.
    if r.pad_hygiene:prepare_tail(batch.num_tokens,batch.num_tokens_after_padding)
    if dummy_run or getattr(r.owner,'_kstop_warmup',False):return
    widths,valid,payload=_inputs(r,batch)
    if r.control=='local':
        # Normally guard_inputs ran and the planner copy already checked the
        # flag. Otherwise guard here and check synchronously (counted): the
        # check still precedes the forward launch on every rank.
        if not r.inputs_guarded:r.fold('prepare',payload,valid);r.guard('prepare')
        r.inputs_guarded=False
        if r.guard_pending is not None:r.check_guard()
    else:r.agree('prepare',payload,valid)
    src=[];dead=[];start=0
    for i,(rid,w) in enumerate(zip(batch.req_ids,widths)):
        k=r.proposals.pop(rid,None)
        if batch.is_prefilling_np[i] or w==1:
            src.extend(range(start,start+w));dead.extend([False]*w)
        else:
            if k is None or not 2<=w<=4:raise RuntimeError('stale/partial verify proposal')
            k=min(k,w-1)
            src.extend(start+j if j<=k else start for j in range(w))
            dead.extend(j>k for j in range(w))
        start+=w
    s['identity']=not any(dead)
    if not s['identity']:
        # Pinned caching allocator records async copies, keeping host storage
        # alive until DMA completes. No reusable staging buffer can be raced.
        pin=torch.device(runner.device).type=='cuda'
        hs=torch.tensor(src,dtype=torch.int64,pin_memory=pin)
        hd=torch.tensor(dead,dtype=torch.bool,pin_memory=pin)
        s['src'][:len(src)].copy_(hs,non_blocking=True)
        s['dead'][:len(dead)].copy_(hd,non_blocking=True)



def prepare_tail(num_tokens,num_tokens_padded,reset=False):
    """Fill the stable device source/mask buffers before replay; no D2H.

    Native host counts are already known. CUDA fill_/copy_ are stream ordered;
    the captured remap reads these buffers anew on every replay. No Python
    tail-count branch is captured, and no token/position/KV input is changed.
    """
    s=STATE
    if s is None:raise RuntimeError('padding before initialization')
    if not s['runtime'].pad_hygiene:return
    if reset:reset_draft_rows()
    if not 0<num_tokens<=num_tokens_padded<=len(s['src']):
        raise RuntimeError('invalid padding bounds')
    s['padding']=num_tokens<num_tokens_padded
    s['src'][num_tokens:num_tokens_padded].zero_()
    s['dead'][num_tokens:num_tokens_padded].fill_(True)


def dispatch_tokens(num_reqs,num_tokens,uniform_token_count):
    """Same shape-only lookup key for capture analysis and native dispatch."""
    return (12 if os.environ.get('GLM_MTP_KSTOP_UNIFORM_BATCH')=='k2'
            and os.environ.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT','m12')=='m12'
            and 1<num_reqs<4 and uniform_token_count==3
            and num_tokens==3*num_reqs else num_tokens)


def capture_can_pad(manager,desc,compatible):
    """Inspect the real priority table, never a hard-coded bank of sizes.

    Uniform inputs have exactly requests * query width tokens. Short DSA
    dispatch only selects exact shapes. Varlen/mixed inputs can use any token
    count with their native query bound. No tensor values or rank state enter
    this decision; even unreachable descriptors safely carry no padding op.
    """
    if getattr(desc,'short_context',False):return False
    for tokens in range(1,desc.num_tokens):
        for reqs in range(1,min(manager.max_num_reqs,tokens)+1):
            uniform=tokens//reqs if tokens%reqs==0 else None
            for q in {None,uniform}:
                key=(dispatch_tokens(reqs,tokens,q),desc.num_active_loras)
                candidates=manager._candidates.get(key,())
                # max_query_len is bounded only on native varlen graphs.
                for bound in {None,manager.decode_query_len}:
                    selected=next((d for d in candidates if compatible(
                        d,reqs,tokens,q,desc.num_active_loras,bound)),None)
                    if selected==desc:return True
    return False


def capture_forward(manager,desc,compatible,fn):
    """Scope the capture decision around both warmup and captured forwards."""
    if STATE is None or not STATE['runtime'].pad_hygiene:return fn
    can_pad=capture_can_pad(manager,desc,compatible)
    def run(mode):
        prior=STATE.get('capture_pad')
        STATE['capture_pad']=can_pad
        try:return fn(mode)
        finally:STATE['capture_pad']=prior
    return run


def remap(weights,ids):
    if STATE is None:raise RuntimeError('remap before initialization')
    # Potentially padded captures must include the op even at identity capture.
    # Eager forwards use the actual host counts. Heterogeneous K-stop rows
    # retain their original eager remap independently of tail padding.
    capture=STATE.get('capture_pad')
    needed=(capture if capture is not None else
            STATE.get('padding',False) or not STATE.get('identity',True))
    if (STATE['runtime'].pad_hygiene and needed) or (
            not STATE['runtime'].pad_hygiene and not STATE.get('identity',True)):
        torch.ops.glm_deadrow.remap_(weights,ids,STATE['src'][:ids.shape[0]])


def mask_drafts(tokens, indices):
    if STATE is None:raise RuntimeError('sampler before initialization')
    return torch.where(STATE['dead'].index_select(0,indices.to(torch.int64)),-1,tokens)


def reset_draft_rows():
    # Target map is not appropriate for later M1/M4 MTP passes.
    if STATE is not None:
        STATE['src'].copy_(STATE['rows']);STATE['dead'].zero_();STATE['identity']=True
        STATE['padding']=False


def install_mtp(cls):
    init=cls.__init__;propose=cls.propose;configure=cls._configure_fused_multi_step_decode
    @functools.wraps(init)
    def initialize_mtp(self,config,device):
        sc=config.speculative_config;pc=config.parallel_config
        if (sc.method!='mtp' or sc.num_speculative_tokens!=3 or sc.draft_sample_method!=('probabilistic' if os.environ.get('GLM_SPEC_SAMPLE','0')=='1' else 'greedy') or
            sc.rejection_sample_method!='standard' or sc.enable_adaptive_verification or
            not config.scheduler_config.async_scheduling):raise RuntimeError('native Kmax3 selected draft sampler/standard async required')
        if (pc.tensor_parallel_size,pc.pipeline_parallel_size,pc.data_parallel_size,pc.decode_context_parallel_size,pc.prefill_context_parallel_size)!=(4,1,1,1,1):raise RuntimeError('TP4 only')
        if (config.cache_config.block_size!=64 or config.scheduler_config.max_num_seqs>4 or config.lora_config or config.kv_transfer_config or pc.enable_eplb or getattr(pc,'enable_expert_parallel',False) or getattr(pc,'use_sequence_parallel_moe',False)):raise RuntimeError('unsupported layout')
        if str(config.compilation_config.cudagraph_mode).split('.')[-1]!='FULL_DECODE_ONLY':raise RuntimeError('FULL_DECODE_ONLY required')
        init(self,config,device)
        if self.use_local_argmax_reduction:raise RuntimeError('native full-logit head required')
        self._kstop=Runtime(self)
    def unfused(self):configure(self);self.use_fused_multi_step_decode=False
    def sample(self,hidden_states,positions,idx_mapping,temperature,seeds,draft_step,draft_logits):
        logits=self.model.compute_logits(hidden_states);z=logits.float()
        self._kstop.confidence[:len(z)].copy_((z.amax(-1)-z.logsumexp(-1)).exp())
        if draft_logits is not None:
            # Stock sampler/cache column and temperature-zero argmax behavior.
            # Confidence above is ALWAYS max softmax(raw logits), never q(x).
            from vllm.v1.worker.gpu.spec_decode.speculator import gumbel_sample
            return gumbel_sample(logits,idx_mapping,temperature,seeds,positions+1,
                apply_temperature=True,logits_cache=draft_logits,
                logits_cache_col=draft_step,use_fp64=self.use_fp64_gumbel)
        return logits.argmax(-1)
    @functools.wraps(propose)
    def selected(self,*args,**kwargs):
        a=inspect.signature(propose).bind(self,*args,**kwargs);a.apply_defaults()
        synthetic=a.arguments.get('dummy_run') or a.arguments.get('is_profile') or getattr(self,'_kstop_synthetic',False)
        prior=self.num_speculative_steps
        try:
            self.num_speculative_steps=self._kstop.begin(a.arguments['input_batch'],synthetic)
            reset_draft_rows()
            result=propose(self,*args,**kwargs)
            return self._kstop.finish(result)
        finally:
            self.num_speculative_steps=prior;self._kstop.pending=self._kstop.ready=None
            self.on_multi_step_decode_end(0)
    cls.__init__=initialize_mtp;cls._configure_fused_multi_step_decode=unfused
    cls.sample_draft=sample;cls.propose=selected
