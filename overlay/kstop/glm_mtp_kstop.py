# SPDX-License-Identifier: Apache-2.0
"""Default-off source-pinned native MTP K3 cumulative stop overlay."""
import contextlib,hashlib,importlib.abc,importlib.machinery,importlib.util,json,os,sys,threading
from pathlib import Path
ROOT=Path(__file__).parent;PINS=json.loads((ROOT/'source_pins.json').read_text())
PREFIX='vllm.v1.worker.gpu.'
RUNNER=PREFIX+'model_runner';AR=PREFIX+'spec_decode.autoregressive.speculator'
MTP=PREFIX+'spec_decode.mtp.speculator';CG=PREFIX+'cudagraph_utils'
WORKER='vllm.v1.worker.gpu_worker';MOE='vllm.model_executor.layers.fused_moe.runner.moe_runner'
SAM=PREFIX+'spec_decode.rejection_sampler';MLA='vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90'


def once(s,old,new):
    if s.count(old)!=1:raise RuntimeError('kstop source anchor drift: '+old[:100])
    return s.replace(old,new,1)


def transform(name,raw):
    if hashlib.sha256(raw.encode()).hexdigest()!=PINS[name]:raise RuntimeError('kstop source drift: '+name)
    try:
        import kstop_gate;kstop_gate.LOADED[name]=PINS[name]
    except ImportError:pass
    s=raw
    if name in (RUNNER,AR,MOE,SAM,MLA):
        s=once(s,'import torch\n','import torch\nimport kstop_runtime as _kstop\n')
    if name==RUNNER:
        s=once(s,'        get_offloader().post_init()\n','        get_offloader().post_init()\n        _kstop.initialize(self)\n')
        s=once(s,'        if not dummy_run:\n            # Update the request states.',
            '        self.speculator._kstop_synthetic = bool(dummy_run or is_profile or getattr(self.speculator, "_kstop_warmup", False))\n'
            '        if not dummy_run and not is_profile:\n'
            '            scheduler_output = _kstop.select_inputs(self, scheduler_output)\n'
            '        if not dummy_run:\n            # Update the request states.')
        s=once(s,'            need_eager=is_profile or skip_compiled,',
            '            need_eager=is_profile or skip_compiled or (not dummy_run and getattr(self, "_kstop_needs_remap", False)),')
        # Fix3: the prepare guard is enqueued BEFORE prepare_attn, so its flag
        # rides the target's existing SM90 planner D2H (host check before the
        # forward launch, no extra synchronization).
        s=once(s,'            block_tables, slot_mappings = self.prepare_attn(input_batch)\n',
            '            _kstop.guard_inputs(self, input_batch, is_profile)\n'
            '            block_tables, slot_mappings = self.prepare_attn(input_batch)\n')
        s=once(s,'        # Update the EPLB meta.\n',
            '        _kstop.prepare(self, input_batch, dummy_run or is_profile)\n\n        # Update the EPLB meta.\n')
    elif name==AR:
        s=once(s,'        self.on_prefill_begin(num_reqs)\n',
            '        if self._kstop.pad_hygiene:\n'
            '            _kstop.prepare_tail(num_tokens, prefill_batch_desc.num_tokens, reset=True)\n'
            '        self.on_prefill_begin(num_reqs)\n')
        s=once(s,'        for step in range(1, self.num_speculative_steps):\n            # Rebuild every step',
            '        for step in range(1, self.num_speculative_steps):\n'
            '            if not self._kstop.can_advance(step, num_reqs):\n                break\n'
            '            self._kstop.arm(step, num_reqs)\n            # Rebuild every step')
        s=once(s,'            self.current_draft_step.fill_(step)\n\n            if batch_desc.cg_mode',
            '            if not self._kstop.after_metadata(step, num_reqs):\n                break\n'
            '            self.current_draft_step.fill_(step)\n'
            '            if self._kstop.pad_hygiene:\n'
            '                _kstop.prepare_tail(num_reqs, batch_desc.num_tokens, reset=True)\n\n            if batch_desc.cg_mode')
    elif name==CG:
        s=once(s,'import gc\n','import gc\nimport kstop_runtime as _kstop\n')
        s=once(s,'            decode_query_lens = [self.decode_query_len]\n',
            '            decode_query_lens = ([2,3,4] if self.decode_query_len > 1 else [1])\n')
        # Size 4 already captures M6 q3. Reuse its exact c2 graph without
        # adding a capture bank; m12 retains the previous dispatch policy.
        s=once(s,'        key = (num_tokens, effective_loras)\n',
            '        dispatch_tokens = _kstop.dispatch_tokens(num_reqs, num_tokens, uniform_token_count)\n'
            '        key = (dispatch_tokens, effective_loras)\n')
        for warmup in ('True','False'):
            old=f'forward_fn = create_forward_fn(desc, warmup={warmup})'
            s=once(s,old,f'forward_fn = _kstop.capture_forward(self, desc, _is_compatible, create_forward_fn(desc, warmup={warmup}))')
    elif name==MLA:
        s=once(s,'ctx = positions[:num_rows].cpu().to(torch.int64) + 1',
            'ctx = _kstop.packed_host(positions[:num_rows]).to(torch.int64) + 1')
        s=once(s,'seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int32)',
            'seq_lens = _kstop.packed_host(cam.seq_lens[:num_reqs]).to(torch.int32)')
    elif name==MOE:
        s=once(s,'            fused_out = self.routed_experts.forward_modular(\n',
            '            _kstop.remap(topk_weights, topk_ids)\n'
            '            fused_out = self.routed_experts.forward_modular(\n')
    elif name==SAM:
        s=once(s,'        draft_sampled = input_batch.input_ids[input_batch.logits_indices]\n',
            '        draft_sampled = input_batch.input_ids[input_batch.logits_indices]\n'
            '        draft_sampled = _kstop.mask_drafts(draft_sampled, input_batch.logits_indices)\n')
    elif name==WORKER:
        s=once(s,'            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)',
            '            sp = self.model_runner.speculator\n'
            '            prior = getattr(sp, "_kstop_warmup", False)\n'
            '            if sp is not None: sp._kstop_warmup = True\n'
            '            try:\n'
            '                warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)\n'
            '            finally:\n'
            '                if sp is not None: sp._kstop_warmup = prior\n'
            '            if sp is not None and getattr(sp, "_kstop", None) is not None: sp._kstop.after_warmup()')  # fix19
    return s


# Fix12 (window 4 stop boot: 'kstop requires source loader'). The served
# bringup chain puts wrapper loaders in front of the file loader (glm_full_mla
# returns its own Loader with only create_module/exec_module, whose
# exec_module calls the ORIGINAL SourceFileLoader it captured). kstop
# therefore reads the source from the pinned file itself (PathFinder, the same
# path the chain loads, hash pin unchanged) and hands the transformed code
# object to whichever SourceFileLoader executes exactly (name, path) inside
# the chain's own exec_module. The chain (full-MLA installer, after-import
# hooks) still runs; kstop fails closed unless its code ran exactly once.
_PENDING={};_LOCK=threading.RLock();_SCOPE=dict(depth=0,saved=None)


def _pinned_get_code(self,fullname):
    slot=_PENDING.get((fullname,getattr(self,'path',None)))
    if slot is not None and getattr(self,'name',None)==fullname:
        slot[1]+=1;return slot[0]
    return _SCOPE['saved'][1](self,fullname)


@contextlib.contextmanager
def pinned_code(fullname,origin,code):
    """While the loader chain executes `fullname`, the SourceFileLoader for
    exactly (fullname, origin) returns `code`. Yields [code, times used]."""
    key=(fullname,origin)
    with _LOCK:
        if key in _PENDING:raise RuntimeError('kstop re-entrant pinned import: '+fullname)
        if _SCOPE['depth']==0:
            cls=importlib.machinery.SourceFileLoader
            _SCOPE['saved']=('get_code' in cls.__dict__,cls.get_code);cls.get_code=_pinned_get_code
        _SCOPE['depth']+=1;slot=_PENDING[key]=[code,0]
    try:yield slot
    finally:
        with _LOCK:
            del _PENDING[key];_SCOPE['depth']-=1
            if _SCOPE['depth']==0:
                cls=importlib.machinery.SourceFileLoader;had,original=_SCOPE['saved'];_SCOPE['saved']=None
                if had:cls.get_code=original
                else:del cls.get_code


def pinned_source(fullname,path,spec):
    """Source of the file the loader chain will execute, read through a plain
    PathFinder spec; refuses anything that is not that exact source file."""
    if spec is None or spec.loader is None or not spec.origin:raise RuntimeError('kstop requires a loadable spec: '+fullname)
    search=path
    if search is None and '.' in fullname:search=sys.modules[fullname.rpartition('.')[0]].__path__
    plain=importlib.machinery.PathFinder.find_spec(fullname,search)
    if plain is None or type(plain.loader) is not importlib.machinery.SourceFileLoader:raise RuntimeError('kstop requires a source file: '+fullname)
    if plain.origin!=spec.origin:raise RuntimeError('kstop source origin differs from the loader chain: %s %s != %s'%(fullname,plain.origin,spec.origin))
    return plain.loader.get_source(fullname)


def shortcut_module():
    mod=sys.modules.get('glm_dsa_short')
    return mod if mod is not None and mod.ENABLED and mod.KSTOP else None


def combined_transform(fullname,raw):
    # Both validators see the original bytes. Only one loader supplies code;
    # neither finder can replace the other's transformed code object.
    s=transform(fullname,raw)
    d=shortcut_module()
    if d is not None and fullname in d.SHARED:
        if PINS[fullname]!=d.PINS[fullname]:raise RuntimeError('kstop/DSA pin disagreement: '+fullname)
        d.transform(fullname,raw)  # validate raw pin and anchors
        s=d.TRANSFORMS[fullname](s)
    return s


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in PINS:return None
        index=sys.meta_path.index(self);sys.meta_path.remove(self)
        try:spec=importlib.util.find_spec(fullname)
        finally:sys.meta_path.insert(index,self)
        origin=spec.origin if spec else None
        source=combined_transform(fullname,pinned_source(fullname,path,spec));code=compile(source,origin,'exec',dont_inherit=True)
        original=spec.loader.exec_module
        def execute(module):
            with pinned_code(fullname,origin,code) as slot:original(module)
            if slot[1]!=1:raise RuntimeError('kstop pinned source not executed by the loader chain: %s (used %d times)'%(fullname,slot[1]))
            if getattr(module,'__file__',None)!=origin:raise RuntimeError('kstop module origin drift: '+fullname)
            d=shortcut_module()
            if d is not None and fullname in d.SHARED:
                if fullname in d.INSTALL:d.INSTALL[fullname](module)
            if fullname==MLA:
                host=getattr(module.FlashInferMLASparseSM90Builder._kv_lens_host,'__code__',None)
                if host is None or 'packed_host' not in host.co_names:raise RuntimeError('kstop MLA planner transform replaced after import')
            if fullname==MTP:
                from kstop_runtime import install_mtp
                install_mtp(module.MTPSpeculator)
        spec.loader.exec_module=execute;return spec


def register(env=None):
    env=os.environ if env is None else env;flag=env.get('GLM_MTP_KSTOP','0')
    pad=env.get('GLM_PAD_HYGIENE','0')
    if pad not in ('0','1'):raise ValueError('GLM_PAD_HYGIENE must be 0 or 1')
    if pad=='1' and flag!='1':raise ValueError('GLM_PAD_HYGIENE=1 requires GLM_MTP_KSTOP=1')
    if flag=='0':return False
    if flag!='1':raise ValueError('GLM_MTP_KSTOP must be 0 or 1')
    if env.get('GLM_MTP_KSTOP_CAPTURE_LAYOUT','m12') not in ('m12','reuse'):raise ValueError('GLM_MTP_KSTOP_CAPTURE_LAYOUT must be m12 or reuse')
    if env.get('GLM_MTP_KSTOP_UNIFORM_BATCH','0') not in ('0','1','k2',''):raise ValueError('GLM_MTP_KSTOP_UNIFORM_BATCH must be 0, 1 or k2')
    # fix12: GLM_KVLENS_EXACT / GLM_EARLY_PLAN replace the MLA _kv_lens_host that kstop transforms.
    for incompatible in ('GLM_MTP_PHASE_K','GLM_MTP_K_BANK','GLM_HYBRID_V2','GLM_HYBRID_V3','GLM_HYBRID_ASYNC','GLM_DSA_SWA_POOL','GLM_DSA_DRAFT_FOLD','GLM_DEADROW','GLM_KVLENS_EXACT','GLM_EARLY_PLAN'):
        if env.get(incompatible,'0') not in ('0','','off'):raise RuntimeError('incompatible '+incompatible)
    if env.get('GLM_MTP_FIX')!='1' or not env.get('GLM_MTP_KSTOP_CONTROL'):raise RuntimeError('native loader fix/control required')
    if any(n in sys.modules for n in PINS):raise RuntimeError('kstop registration too late')
    sys.meta_path.insert(0,Hook());return True
