# SPDX-License-Identifier: Apache-2.0
"""Draft-only head copy; drained runtime controls recapture native draft graphs.

The target head and logits processor are never patched. No new CUDA kernel.
"""
import functools
from contextlib import contextmanager
from contextvars import ContextVar
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from glm_draft_head_config import options, byte_cost, initial_on

FORMAT = '0'
INITIAL_ON = False
RUNNER = 'vllm.v1.worker.gpu.model_runner'
WORKER = 'vllm.v1.worker.gpu_worker'
PINS = {
    'vllm.distributed.parallel_state': 'c6d65b96a260ea0d779da1073d1799392813eee74c9ed44753dfa56f4e7a9e4c',
    RUNNER: 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415',
    WORKER: 'b2e580d74e7259ff2cbc82dabf38a43409ea5584d143880436583d9fc1ceedf1',
    'vllm.model_executor.models.deepseek_mtp': '88521bcf3bfec6c773998dbefe30448504640907a4ac98f4e822d0ed38e2436f',
    'vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16': '15550ed16f9931937fc662d69f2e190401b7c9ef9ce6c6ba6001bbd03d527562',
    'vllm.model_executor.kernels.linear.mixed_precision.marlin': 'ec18b17815f5114cdc8ba72779095b8ae3d4d1369508a0ef1d30029904cd329c',
    'vllm.model_executor.layers.quantization.utils.marlin_utils': '6f91cbf2b0ae276abd4b57b80b20f260307f9df3e24d40ee2b0abf76f4c7f100',
    'vllm.model_executor.layers.quantization.utils.marlin_utils_fp4': 'a7c61d4f67b5d23c4236645ffc0f6f1fe6f8ef2577e83229c479b311c1cddd8b',
    'vllm.model_executor.layers.logits_processor': '7112cdda465ae3c9d996bcf6c7c3d46f21c23934f0c4d0ecbd527ed699e971b6',
    'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator': '575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1',
    'vllm.v1.worker.gpu.cudagraph_utils': 'c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f',
    'vllm.v1.worker.gpu.spec_decode.eagle.utils': '65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28',
    'vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils': '13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5',
}


def quantize_int8(w):
    if w.ndim != 2 or w.shape[1] % 128 or not torch.isfinite(w).all():
        raise ValueError('finite group128 weights required')
    z = w.float().reshape(len(w), -1, 128)
    scales = (z.abs().amax(-1) / 127).to(torch.bfloat16)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(z / scales.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes.reshape_as(w), scales


def int8_pack(q):
    u = (q.to(torch.int64) + 128).reshape(len(q), -1, 4)
    return (u[..., 0] | u[..., 1] << 8 | u[..., 2] << 16 | u[..., 3] << 24).to(torch.int32)


def storage_bytes(module):
    tensors = list(module.parameters()) + list(module.buffers())
    # vLLM Marlin keeps workspace as a plain attribute.
    tensors += [m.workspace for m in module.modules() if hasattr(m, 'workspace')]
    stores = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return sum(stores.values())


class Bank(torch.nn.Module):
    def __init__(self, head, kind):
        super().__init__()
        w = head.weight.detach()
        n, k = w.shape
        if w.dtype != torch.bfloat16 or n % 64 or k % 128 or not w.is_cuda:
            raise ValueError('CUDA BF16 unpadded Marlin-compatible head required')
        if kind == 'int8':
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16
            scheme = CompressedTensorsWNA16(strategy='group', num_bits=8,
                group_size=128, symmetric=True, layer_name='draft_head')
            scheme.create_weights(self, n, k, [n], k, torch.bfloat16, lambda *a, **kw: None)
            if type(scheme.kernel).__name__ != 'MarlinLinearKernel':
                raise RuntimeError('draft INT8 requires Marlin W8A16')
            self.to(w.device)
            for start in range(0, n, 1024):
                q, s = quantize_int8(w[start:start+1024])
                self.weight_packed.data[start:start+len(q)].copy_(int8_pack(q))
                self.weight_scale.data[start:start+len(q)].copy_(s)
            self.weight_shape.data.copy_(torch.tensor([n,k], device=w.device))
        else:
            import glm_nvfp4_format as fmt
            from glm_nvfp4_attn import new_scheme
            scheme = new_scheme()
            self.input_size_per_partition = k
            self.output_size_per_partition = n
            self.logical_widths = [n]
            self.params_dtype = torch.bfloat16
            self.weight_packed = torch.nn.Parameter(torch.empty((n,k//2),dtype=torch.uint8,device=w.device),False)
            self.weight_scale = torch.nn.Parameter(torch.empty((n,k//16),dtype=torch.float8_e4m3fn,device=w.device),False)
            maximum = max(float(w[i:i+1024].abs().amax()) for i in range(0,n,1024))
            scale = maximum / (6*448)
            self.weight_global_scale = torch.nn.Parameter(torch.tensor([scale],dtype=torch.float32,device=w.device),False)
            for start in range(0,n,1024):
                packed, scales = fmt.encode(w[start:start+1024].float().cpu().numpy(), scale)
                self.weight_packed.data[start:start+len(packed)].copy_(torch.from_numpy(packed).to(w.device))
                self.weight_scale.data[start:start+len(scales)].copy_(torch.from_numpy(scales).view(torch.float8_e4m3fn).to(w.device))
        scheme.process_weights_after_loading(self)
        self.scheme = scheme
        if kind == 'int8':
            self.workspace = scheme.kernel.workspace
        self.kind = kind
        self.resident_bytes = storage_bytes(self)
        expected = byte_cost(kind,n,k,torch.cuda.get_device_properties(w.device).multi_processor_count)['total']
        if self.resident_bytes != expected:
            raise RuntimeError(f'draft bank storage mismatch {self.resident_bytes} != {expected}')

    def forward(self,x,bias=None):
        if x.dtype != torch.bfloat16:
            raise ValueError('draft head requires BF16 activations')
        return self.scheme.apply_weights(self,x,bias)


class DraftMethod:
    def apply(self, layer, x, bias=None):
        if layer.on:
            return layer.bank(x,bias)
        return layer.original.quant_method.apply(layer.original,x,bias=bias)


class DraftHead(torch.nn.Module):
    """Only the draft model owns this facade; metadata/gather stay native."""
    def __init__(self, original, bank):
        super().__init__()
        self.original = original
        self.bank = bank
        self.on = False
        self.quant_method = DraftMethod()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('original'), name)


def memory():
    fields = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])*1024


class VoteRefused(RuntimeError):
    """Expected control refusal; runtime RPC boundaries return its receipt."""
    def __init__(self, receipt):
        super().__init__('draft head all-rank refusal')
        self.receipt = receipt


def gather_receipts(value, group=None):
    if group is None:
        from vllm.distributed import get_tp_group
        group = get_tp_group().cpu_group
    rows = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(rows, value, group=group)
    return rows


def runtime_refusal(runner, exc):
    runner._draft_head_ready = False
    runner._k4_drafthead_ready = False
    runner.speculator._kstop.bad = 'control refused; whole candidate stop required'
    return dict(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                ok=False, refused=True, ready=False, resume_allowed=False,
                samples=getattr(runner, '_draft_head_leak_samples', []), refusal=exc.receipt)


def runtime_rpc(method):
    @functools.wraps(method)
    def call(worker, *a, **kw):
        try:
            return method(worker, *a, **kw)
        except VoteRefused as exc:
            return runtime_refusal(worker.model_runner, exc)
        except RuntimeError as exc:
            # The unchanged K-stop implementation raises for its cold K4
            # switch vote. Convert only this named completed control vote;
            # hot lifecycle/admission/device errors remain fatal.
            if not str(exc).startswith('collective-safe kstop refusal at k4-switch '):
                raise
            runner = worker.model_runner
            policy = runner.speculator._kstop
            votes = gather_receipts(dict(
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                condition='k4_switch_vote', measured=dict(error=str(exc),
                    epoch=policy.epoch, k4_on=policy.k4_on,
                    mem_available=memory())))
            return runtime_refusal(runner, VoteRefused(dict(votes=votes)))
    return call


_qualification_terms = ContextVar('draft_head_qualification_terms', default=None)


def qualification_term(check, conditions, measured=None, error=None):
    row = dict(rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               check=check, conditions={k: bool(v) for k, v in conditions.items()},
               measured=measured or {}, error=None if error is None else
               dict(type=type(error).__name__, text=str(error)))
    rows = _qualification_terms.get()
    if rows is not None:
        rows.append(row)
    print('draft-head-qualification '+json.dumps(row, sort_keys=True), flush=True)
    return all(row['conditions'].values())


def agree(payload, valid, group=None):
    if group is None:
        from vllm.distributed import get_tp_group
        group = get_tp_group().cpu_group
    raw = json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    t = torch.tensor(list(hashlib.sha256(raw).digest())+[int(valid)],dtype=torch.int32)
    lo, hi = t.clone(), t.clone()
    torch.distributed.all_reduce(lo,op=torch.distributed.ReduceOp.MIN,group=group)
    torch.distributed.all_reduce(hi,op=torch.distributed.ReduceOp.MAX,group=group)
    if not torch.equal(lo,hi) or int(lo[-1]) != 1:
        mismatch = not torch.equal(lo[:-1], hi[:-1])
        rows = gather_receipts(dict(rank=torch.distributed.get_rank(group),
            condition='payload_mismatch' if mismatch else 'local_validity',
            valid=bool(valid), measured=payload, terms=_qualification_terms.get()), group)
        raise VoteRefused(dict(votes=rows, payload_mismatch=mismatch))


def prepare(runner, factory=Bank, vote=agree):
    error = None
    descriptor = {}
    try:
        draft = runner.speculator.model
        if type(draft).__name__ != 'DeepSeekMTP' or draft is runner.model:
            raise ValueError('pinned DeepSeekMTP required')
        layers = list(draft.model.layers.values())
        if len(layers) != 1:
            raise ValueError('one native MTP layer required')
        head = layers[0].shared_head.head
        target = runner.model.lm_head
        if head is not target or head.weight.dtype != torch.bfloat16:
            raise ValueError('MTP must alias target BF16 head before detaching')
        lp = draft.model.logits_processor
        if lp.head_dtype not in (None,torch.bfloat16) or lp.logits_as_input or not lp.use_all_gather:
            raise ValueError('native BF16 full-logit draft processor required')
        if type(head.quant_method).__name__ != 'UnquantizedEmbeddingMethod':
            raise ValueError('target head must be unquantized')
        bank = factory(head,FORMAT)
        facade = DraftHead(head,bank)
        facade.on = INITIAL_ON
        descriptor = dict(format=FORMAT,initial_on=INITIAL_ON,shape=list(head.weight.shape),bytes=bank.resident_bytes)
    except Exception as exc:
        error = exc
    vote({'prepare':descriptor},error is None)
    if error is not None:
        raise RuntimeError('draft head preparation failed') from error
    layers[0].shared_head.head = facade
    if runner.model.lm_head is not target or target.quant_method is facade.quant_method:
        raise RuntimeError('target head ownership changed')
    runner._draft_head = facade
    runner._draft_head_epoch = -1
    runner._draft_head_ready = False
    runner.model_memory_usage += bank.resident_bytes


def managers(runner):
    return [runner.speculator.prefill_cudagraph_manager,runner.speculator.decode_cudagraph_manager]


def release_draft_graphs(runner):
    """Called only after the drained all-rank vote and device synchronization.

    FULL draft graphs own a separate pool: target graphs cannot pin retired
    draft activations/head GEMM outputs. The bank is prepared once and reused;
    no quantized weights or workspaces are copied during a toggle.
    """
    for manager in managers(runner):
        for graph in manager.graphs.values():
            if hasattr(graph, 'reset'):
                graph.reset()
        manager.graphs.clear()
        manager._graphs_captured = False
        # Integration factory closes over dummy capture inputs/metadata.
        manager.__dict__.pop('_k4_drafthead_factory', None)
        manager.pool = None


def renew_draft_pool(runner, pool_factory=torch.cuda.graph_pool_handle):
    # Both draft routines execute serially and share one pool, never target's.
    pool = pool_factory()
    for manager in managers(runner):
        manager.pool = pool


def full_draft_graphs(runner):
    # Pinned serving layout is FULL_DECODE_ONLY. PIECEWISE caches would require
    # model-scoped wrapper teardown; refuse them before destroying any graph.
    from vllm.config.compilation import CUDAGraphMode
    return all(all(mode == CUDAGraphMode.FULL or not descs
                   for mode, descs in manager._capture_descs.items())
               for manager in managers(runner))


def descriptors(runner):
    return [[repr(k) for k in m.graphs] for m in managers(runner)]


def status(runner):
    head = getattr(runner,'_draft_head',None)
    return dict(format=FORMAT,on=bool(head and head.on),epoch=getattr(runner,'_draft_head_epoch',-1),
        ready=getattr(runner,'_draft_head_ready',False),
        qualification=getattr(runner,'_draft_head_qualification',None),side_bytes=head.bank.resident_bytes if head else 0,
        confidence_source='active draft raw logits',graphs=descriptors(runner) if head else [],
        init_failure=getattr(runner, '_draft_head_init_failure', None),
        dh_gate_passed=bool(head and head.on and runner._draft_head_ready and
                            getattr(runner, '_draft_head_init_failure', None) is None))


def telemetry(runner):
    head=getattr(runner,'_draft_head',None)
    if head is None or head.original.weight.device.type!='cuda':return {}
    return dict(mem_available=memory(),allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())


def capture_draft_arm(runner, cuda):
    # Shared with runtime OFF/ON: target capture is already complete.
    import kstop_runtime as rt
    rt.reset_draft_rows()
    if cuda:
        renew_draft_pool(runner)
    runner.speculator.capture()


def switch(runner,on,epoch,scheduler_drained=False,vote=agree,sync=torch.cuda.synchronize,mem=memory,collect=gc.collect,empty=torch.cuda.empty_cache):
    head = getattr(runner,'_draft_head',None)
    r = getattr(getattr(runner,'speculator',None),'_kstop',None)
    valid = (head is not None and getattr(runner,'_draft_head_ready',False) and
        type(on) is int and on in (0,1) and type(epoch) is int and
        runner._draft_head_epoch < epoch < 2**53 and scheduler_drained is True and
        runner.execute_model_state is None and r is not None and
        r.pending is None and r.ready is None and r.guard_pending is None and
        mem() >= 6*(1<<30) and
        (head.original.weight.device.type != 'cuda' or full_draft_graphs(runner)))
    prior = status(runner)
    vote({'on':on,'epoch':epoch,'prior':prior},valid)
    sync()
    if head.on != bool(on):
        old_desc = descriptors(runner)
        # Destroy obsolete draft graphs before recapture; do not retain a
        # second graph pool at the 262144 layout. Target graphs remain intact.
        runner._draft_head_ready = False
        error = None
        try:
            release_draft_graphs(runner)
            collect()
            empty()
            sync()
        except Exception as exc:
            error = exc
        vote({'released': old_desc, 'on': on, 'epoch': epoch}, error is None)
        if error is not None:
            raise RuntimeError('draft graph release failed') from error
        cuda=head.original.weight.device.type=='cuda'
        head.on = bool(on)
        if cuda:torch.cuda.reset_peak_memory_stats()
        before=telemetry(runner)
        error = None
        try:
            capture_draft_arm(runner, cuda)
            sync()
            collect()
            empty()
            runner._draft_head_last_capture=dict(before=before,after=telemetry(runner))
            valid = descriptors(runner) == old_desc and mem() >= 4.5*(1<<30)
        except Exception as exc:
            valid = False
            error = exc
        # Capture errors fail the candidate collectively. Never restore only
        # one rank, or publish READY after a partial capture.
        vote({'captured':descriptors(runner),'on':on},valid)
        if error is not None:
            raise RuntimeError('draft graph recapture failed') from error
        runner._draft_head_ready = True
    r.forget(list(r.used))
    r.proposals.clear()
    r.trail = bytes(32)
    r.ids=[];r.lengths=[];r.cumulative=[];r.bad=None
    r.inputs_guarded=False
    if hasattr(r,'guard_inputs_log'):
        r.guard_inputs_log.clear()
    runner._draft_head_epoch = epoch
    return status(runner)


def bank_identity(bank):
    tensors = list(bank.named_parameters()) + list(bank.named_buffers())
    tensors += [(name+'.workspace', m.workspace) for name, m in bank.named_modules()
                if hasattr(m, 'workspace')]
    layout = [(name, id(t), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes(),
               str(t.dtype), list(t.shape), list(t.stride())) for name, t in tensors]
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def bank_weight_hash(bank):
    """Immutable bank parameters/buffers; exclude mutable Marlin lock workspace.

    One tensor at a time on CPU, no owning device clone or retained host copy.
    """
    digest = hashlib.sha256()
    for name, tensor in list(bank.named_parameters()) + list(bank.named_buffers()):
        if name.endswith('workspace'):
            continue
        digest.update(name.encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        del raw
    return digest.hexdigest()


def graph_identity(manager):
    return [(repr(k), id(g), list(g.pool()) if hasattr(g, 'pool') else None)
            for k, g in manager.graphs.items()]


def leakcheck(worker, scheduler_drained=False, vote=agree,
              sync=torch.cuda.synchronize, collect=gc.collect,
              empty=torch.cuda.empty_cache, sample=telemetry, gather=gather_receipts):
    """Next-window GPU gate: two warmed arm baselines, 20 actual transitions.

    Caller owns coordinator lock and keeps scheduler paused with APC retained.
    Combined workers use the complete transaction including replay qualification.
    Success restores the starting arm; failure leaves candidate paused/unready.
    """
    runner = worker.model_runner
    head = getattr(runner, '_draft_head', None)
    vote({'leakcheck': status(runner)}, scheduler_drained is True and head is not None
         and head.original.weight.device.type == 'cuda' and runner._draft_head_ready)
    starting = int(head.on)
    # Keep original graph objects alive so their Python identities cannot recycle.
    target = dict(runner.cudagraph_manager.graphs)
    bank = head.bank
    bank_stores = storage_bytes(bank)
    bank_signature = bank_identity(bank)
    weight_hash = bank_weight_hash(bank)
    target_signature = graph_identity(runner.cudagraph_manager)
    baselines, rows = {}, []
    combined = getattr(runner.speculator._kstop, 'k4_capture', False)
    def change(on):
        epoch = max(runner._draft_head_epoch, getattr(runner.speculator._kstop, 'epoch', -1)) + 1
        if combined:
            from glm_k4_drafthead import transaction
            policy = runner.speculator._kstop
            transaction(worker, int(policy.k4_on), on, policy.k4_settings.tau4, epoch, True)
        else:
            switch(runner, on, epoch, True)
    # Warm both formats/capture paths before charging any allocator drift.
    for on in (1-starting, starting):
        change(on)
    for index in range(22):
        on = (1-starting) if index % 2 == 0 else starting
        change(on)
        current_weight_hash = bank_weight_hash(head.bank)
        sync(); collect(); empty(); sync()
        current = sample(runner)
        conditions = dict(
            sample_values=all(type(current.get(k)) is int and current[k] >= 0
                              for k in ('allocated', 'reserved', 'mem_available')),
            memory_floor=type(current.get('mem_available')) is int and current['mem_available'] >= 4.5*(1 << 30),
            bank_identity=head.bank is bank and bank_identity(head.bank) == bank_signature,
            bank_storage=storage_bytes(head.bank) == bank_stores,
            bank_weights=current_weight_hash == weight_hash,
            target_graph_identity=graph_identity(runner.cudagraph_manager) == target_signature)
        if index < 2:
            baselines[on] = current
        baseline = baselines[on]
        deltas = {k: current[k]-baseline[k]
                  if type(current.get(k)) is int and type(baseline.get(k)) is int else None
                  for k in ('allocated', 'reserved')}
        conditions.update(allocated_drift=deltas['allocated'] is not None and abs(deltas['allocated']) <= 8 << 20,
                          reserved_drift=deltas['reserved'] is not None and abs(deltas['reserved']) <= 16 << 20)
        row = dict(index=index, on=on, **current, deltas=deltas, conditions=conditions,
                   limits=dict(allocated=8 << 20, reserved=16 << 20, mem_available=int(4.5*(1 << 30))),
                   bank_hash=bank_identity(head.bank), baseline_bank_hash=bank_signature,
                   bank_weight_hash=current_weight_hash, baseline_bank_weight_hash=weight_hash,
                   bank_bytes=storage_bytes(head.bank), baseline_bank_bytes=bank_stores,
                   target_graphs=graph_identity(runner.cudagraph_manager),
                   baseline_target_graphs=target_signature,
                   rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                   draft_pools=[list(m.pool) if isinstance(getattr(m, 'pool', None), tuple)
                                else repr(getattr(m, 'pool', None)) for m in managers(runner)],
                   draft_graphs=[graph_identity(m) for m in managers(runner)],
                   capture_streams=[getattr(getattr(m, '_draft_capture_context', None),
                                            'stream', None).cuda_stream
                                    if getattr(m, '_draft_capture_context', None) else None
                                    for m in managers(runner)])
        rows.append(row)
        runner._draft_head_leak_samples = rows
        # Flush before collective voting: even a broken collective retains evidence.
        print('draft-head-leak-sample '+json.dumps(row, sort_keys=True), flush=True)
        try:
            vote({'leakcheck_step': index, 'on': on}, all(conditions.values()))
        except VoteRefused as exc:
            histories = gather(dict(rank=row['rank'], samples=rows, baselines=baselines))
            failures = [history['samples'][-1] for history in histories]
            report = runtime_refusal(runner, exc)
            report.update(baselines=baselines, samples=rows, all_rank_histories=histories,
                          all_rank_samples=failures,
                          transitions_checked=max(0, index-1),
                          failed_conditions=[dict(rank=r.get('rank'), index=r['index'],
                              condition=k, measured=r) for r in failures
                              for k, passed in r['conditions'].items() if not passed])
            if exc.receipt.get('payload_mismatch'):
                report['failed_conditions'].extend(dict(rank=v['rank'], condition='payload_mismatch',
                    measured=v['measured']) for v in exc.receipt['votes'])
            return report
    return dict(ok=True, refused=False, baselines=baselines, samples=rows, transitions_checked=20,
                allocated_tolerance_bytes=8 << 20, reserved_tolerance_bytes=16 << 20,
                scope='actual loaded native draft graph transitions; per-rank CUDA allocator')



@torch.inference_mode()
def qualify_initial(runner, vote=agree, sync=torch.cuda.synchronize, mem=memory,
                    collect=gc.collect, empty=torch.cuda.empty_cache):
    """No setter/toggle: qualify the captured ON arm before collective READY."""
    from glm_draft_head_qual import qualify, graph_key
    head = runner._draft_head
    runner._draft_head_ready = False
    runner._draft_head_qualification = None
    target = runner.model.lm_head
    method = target.quant_method
    target_graphs = dict(runner.cudagraph_manager.graphs)
    runner._draft_head_init_target_graphs = target_graphs
    runner._draft_head_init_target_method = method
    keys = graph_key(runner)
    saved = []
    report = None
    error = None
    terms = []
    token = _qualification_terms.set(terms)
    try:
        conditions = dict(head_on=head.on, epoch=runner._draft_head_epoch == -1,
            target_alias=head.original is target, target_method=method is not head.quant_method,
            target_bf16=type(method).__name__ == 'UnquantizedEmbeddingMethod',
            idle=runner.execute_model_state is None)
        valid = qualification_term('initial_qualification', conditions,
            dict(epoch=runner._draft_head_epoch, method=type(method).__name__, graphs=keys))
        vote({'initial_qualification': keys}, valid)
        # Fixed hidden target logits before and after all native replay checks.
        # ON and epoch -1 remain untouched; never call a runtime setter.
        try:
            for m in (1, 4, 12, 16):
                for value in (0., .125, -.125):
                    x = torch.full((m, target.weight.shape[1]), value,
                        device=target.weight.device, dtype=torch.bfloat16)
                    logits = method.apply(target, x, bias=None).clone()
                    finite = bool(torch.isfinite(logits).all())
                    qualification_term('initial_target_logits', dict(finite=finite), dict(m=m, value=value, nonfinite=int((~torch.isfinite(logits)).sum())))
                    if not finite:
                        raise RuntimeError('nonfinite fixed-hidden target logits')
                    saved.append((x, logits))
            sync()
        except Exception as exc:
            error = exc
        available = mem()
        valid = qualification_term('initial_target_prepare', dict(no_exception=error is None, memory_floor=available >= 4.5*(1 << 30)), dict(mem_available=available, floor_bytes=int(4.5*(1 << 30))), error)
        vote({'initial_target_prepare': [1, 4, 12, 16]}, valid)
        if error is not None:
            raise RuntimeError('initial target qualification preparation failed') from error
        report = qualify(runner, sync=sync, vote=vote, mem=mem)
        conditions = dict(target_identity=runner.model.lm_head is target,
            target_method=target.quant_method is method,
            target_graphs=runner.cudagraph_manager.graphs == target_graphs,
            graph_keys=graph_key(runner) == keys, head_on=head.on,
            epoch=runner._draft_head_epoch == -1)
        valid = qualification_term('initial_identity', conditions,
            dict(graphs=graph_key(runner), epoch=runner._draft_head_epoch, head_on=head.on))
        for i, (x, logits) in enumerate(saved):
            actual = method.apply(target, x, bias=None)
            equal = bool(torch.equal(logits, actual))
            qualification_term('initial_target_equality', dict(bit_exact=equal), dict(case=i, m=len(x), max_abs=float((logits.float()-actual.float()).abs().max()), mismatches=int((logits != actual).sum())))
            valid = equal and valid
        actual = None
        sync()
        if not valid:
            raise RuntimeError('initial target fixed-hidden equality or graph/state mismatch')
        report['target_fixed_hidden'] = dict(ms=[1, 4, 12, 16], changed_inputs=3, bit_exact=True)
    except Exception as exc:
        error = exc
        # Retain scalar diagnostics, not the failed replay frame's scratch tensors.
        error.__traceback__ = None
        qualification_term('initial_exception', dict(no_exception=False), error=exc)
    finally:
        saved.clear()
        # Release the last loop temporaries too before sampling steady memory.
        x = logits = actual = None
        try:
            collect()
            empty()
        except Exception as exc:
            error = exc
            qualification_term('initial_cleanup', dict(no_exception=False), error=exc)
    try:
        available = mem()
        valid = qualification_term('initial_qualified', dict(no_exception=error is None, memory_floor=available >= 4.5*(1 << 30)), dict(mem_available=available, floor_bytes=int(4.5*(1 << 30))), error)
        vote({'initial_qualified': report}, valid)
        if error is not None:
            raise RuntimeError('initial draft graph qualification failed') from error
    except BaseException:
        runner._draft_head_ready = False
        runner._k4_drafthead_ready = False
        raise
    finally:
        runner._draft_head_init_terms = terms
        _qualification_terms.reset(token)
    runner._draft_head_qualification = report
    if getattr(runner.speculator._kstop, 'k4_capture', False):
        runner._k4_drafthead_qualification = report
        runner._k4_drafthead_ready = True
    runner._draft_head_ready = True
    return report


_capture_manager = ContextVar('draft_head_capture_manager', default=None)


def install_graph(mod):
    """Draft-only stable streams; native pool/TP/PP capture contexts retained."""
    original = mod.CudaGraphManager.capture
    # Callback is installed on the speculator module, which imports the base
    # class. The native method resolves graph_capture in the BASE module.
    capture_module = mod
    if not hasattr(capture_module, 'graph_capture'):
        from vllm.v1.worker.gpu import cudagraph_utils as capture_module
    native_context = capture_module.graph_capture
    @contextmanager
    def context(device, graph_capture_context=None):
        manager = _capture_manager.get()
        if manager is not None and graph_capture_context is None:
            from vllm.distributed.parallel_state import GraphCaptureContext
            if not hasattr(manager, '_draft_capture_context'):
                manager._draft_capture_context = GraphCaptureContext(torch.cuda.Stream(device=device))
            graph_capture_context = manager._draft_capture_context
        with native_context(device=device, graph_capture_context=graph_capture_context) as ctx:
            yield ctx
    capture_module.graph_capture = context
    @functools.wraps(original)
    def capture(self, *a, **kw):
        # Speculator subclass alone gets stable streams. Target behavior is native.
        draft = type(self).__name__ == 'SpeculatorCudaGraphManager'
        token = _capture_manager.set(self if draft else None)
        try:
            result = original(self, *a, **kw)
            factory = a[0] if a else kw['create_forward_fn']
            def qualified_factory(desc, warmup):
                forward = factory(desc, warmup)
                def run(mode):
                    import glm_dsa_short as dsa
                    with dsa.context(bool(getattr(desc, 'short_context', False))):
                        return forward(mode)
                return run
            self._k4_drafthead_factory = qualified_factory
            return result
        finally:
            _capture_manager.reset(token)
    mod.CudaGraphManager.capture = capture


def initial_fallback(runner, refusal):
    target = runner._draft_head_init_target_graphs
    keys = descriptors(runner)
    runner._draft_head_ready = runner._k4_drafthead_ready = False
    runner._draft_head_qualification = None
    runner._k4_drafthead_qualification = None
    runner._draft_head_init_failure = refusal.receipt
    error = None
    try:
        torch.cuda.synchronize()
        release_draft_graphs(runner)
        gc.collect(); torch.cuda.empty_cache()
        runner._draft_head.on = False
        capture_draft_arm(runner, True)
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        error = exc
    available = memory()
    conditions = dict(no_exception=error is None, head_off=not runner._draft_head.on,
        graph_set=descriptors(runner) == keys, target_graphs=runner.cudagraph_manager.graphs == target,
        target_identity=runner.model.lm_head is runner._draft_head.original,
        target_method=runner.model.lm_head.quant_method is runner._draft_head_init_target_method,
        memory_floor=available >= 4.5*(1 << 30))
    valid = qualification_term('initial_fallback_off', conditions,
        dict(graphs=descriptors(runner), mem_available=available, floor_bytes=int(4.5*(1 << 30))), error)
    agree({'initial_fallback_off': keys}, valid)
    # No poison: native OFF engine can decode. DH controls stay unready and
    # status retains the failed gate, so candidate acceptance must fail.
    runner.speculator._kstop.bad = None
    print('draft-head INIT qualification REFUSED; boot continues with DH OFF; '
          'DH gate FAILED '+json.dumps(refusal.receipt, sort_keys=True), flush=True)


def install_runner(mod):
    cls = mod.GPUModelRunner
    load, capture = cls.load_model, cls.capture_model
    @functools.wraps(load)
    def loaded(self,*a,**kw):
        result = load(self,*a,**kw)
        prepare(self)
        return result
    @functools.wraps(capture)
    def captured(self,*a,**kw):
        self._draft_head_ready = False
        self._draft_head_qualification = None
        self._draft_head_init_failure = None
        agree({'initial_pool': status(self)}, full_draft_graphs(self))
        error = None
        try:
            torch.cuda.synchronize()
            release_draft_graphs(self)
            gc.collect()
            torch.cuda.empty_cache()
            renew_draft_pool(self)
        except Exception as exc:
            error = exc
        agree({'initial_pool_released': status(self)}, error is None)
        if error is not None:
            raise RuntimeError('initial draft pool preparation failed') from error
        result = capture(self,*a,**kw)
        agree({'initial':status(self)},all(m.graphs for m in managers(self)))
        if self._draft_head.on:
            # Reuse the successful toggle order after target capture, with
            # fresh draft-only pool and the same stable capture streams.
            error = None
            keys = descriptors(self)
            try:
                torch.cuda.synchronize()
                release_draft_graphs(self)
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.synchronize()
                capture_draft_arm(self, True)
                torch.cuda.synchronize()
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                error = exc
            qualification_term('initial_recapture', dict(no_exception=error is None,
                graph_set=descriptors(self) == keys), dict(graphs=descriptors(self), draft_pools=[repr(m.pool) for m in managers(self)],
                     capture_streams=[getattr(getattr(getattr(m, '_draft_capture_context', None),
                         'stream', None), 'cuda_stream', None) for m in managers(self)]), error)
            agree({'initial_recaptured': keys}, error is None and descriptors(self) == keys)
            try:
                qualify_initial(self, vote=agree, sync=torch.cuda.synchronize, mem=memory,
                                collect=gc.collect, empty=torch.cuda.empty_cache)
            except VoteRefused as exc:
                initial_fallback(self, exc)
        else:
            self._draft_head_ready = True
        return result
    cls.load_model,cls.capture_model = loaded,captured


def install_worker(mod):
    def get(self):
        from vllm.distributed import get_tp_group
        return dict(rank=get_tp_group().rank_in_group,**status(self.model_runner),
            memory=telemetry(self.model_runner),last_capture=getattr(self.model_runner,'_draft_head_last_capture',None))
    @runtime_rpc
    def set_head(self,on,epoch,scheduler_drained=False):
        # The pinned private RPC exposes serialized string arguments.
        values=[]
        for value in (on,epoch,scheduler_drained):
            try:values.append(json.loads(value) if isinstance(value,str) else value)
            except (ValueError,TypeError):values.append(None)
        on,epoch,scheduler_drained=values
        switch(self.model_runner,on,epoch,scheduler_drained)
        return get(self)
    @runtime_rpc
    def check(self,scheduler_drained=False):
        if isinstance(scheduler_drained,str):
            try:scheduler_drained=json.loads(scheduler_drained)
            except ValueError:scheduler_drained=False
        return microcheck(self.model_runner,scheduler_drained)
    mod.Worker.draft_head_status,mod.Worker.draft_head_set = get,set_head
    mod.Worker.draft_head_microcheck = check
    @runtime_rpc
    def check_leak(self, scheduler_drained=False):
        if isinstance(scheduler_drained, str):
            try: scheduler_drained = json.loads(scheduler_drained)
            except ValueError: scheduler_drained = False
        from vllm.distributed import get_tp_group
        return dict(leakcheck(self, scheduler_drained), rank=get_tp_group().rank_in_group)
    mod.Worker.draft_head_leakcheck = check_leak


@torch.inference_mode()
def microcheck(runner,scheduler_drained=False):
    """Future all-rank fixed-input CUDA screen on the actual loaded heads.

    Not a production MTP/KV replay qualification. Caller keeps scheduler paused.
    """
    head=getattr(runner,'_draft_head',None)
    policy=getattr(getattr(runner,'speculator',None),'_kstop',None)
    valid=(head is not None and not head.on and runner._draft_head_ready and
        runner.execute_model_state is None and scheduler_drained is True and policy is not None and
        policy.pending is None and policy.ready is None and policy.guard_pending is None and memory()>=6*(1<<30))
    agree({'microcheck':status(runner)},valid)
    torch.cuda.synchronize()
    rows=[];error=None
    try:
        target=runner.model.lm_head
        target_method=target.quant_method
        k=target.weight.shape[1]
        generator=torch.Generator(device=target.weight.device).manual_seed(20261007)
        for m in (1,4,12,16):
            x=torch.randn((m,k),device=target.weight.device,dtype=torch.bfloat16,generator=generator)
            changed=x.neg().clone()
            baseline=target_method.apply(target,x,bias=None).clone()
            head.on=True
            if not torch.equal(baseline,target.quant_method.apply(target,x,bias=None)):
                raise RuntimeError('target fixed-hidden logits changed with draft head ON')
            if target.quant_method is not target_method or runner.model.lm_head is not target:
                raise RuntimeError('target head ownership changed during microcheck')
            eager=head.bank(x).clone()
            graph=torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                replay=head.bank(x)
            for value in (x.clone(),changed,torch.zeros_like(x)):
                x.copy_(value)
                expected=head.bank(x).clone()
                graph.replay();torch.cuda.synchronize()
                if not torch.isfinite(replay).all() or not torch.equal(replay,expected):
                    raise RuntimeError('draft GEMM changed-input CUDA replay differs')
                if torch.count_nonzero(head.bank.workspace).item():
                    raise RuntimeError('draft Marlin workspace locks not idle')
            rows.append(dict(m=m,target_bit_exact=True,changed_input_replays=3,
                drift=float((eager.float()-baseline.float()).abs().max()),locks_idle=True))
            head.on=False
            del graph,replay,eager,baseline,expected,x,changed
        valid=True
    except Exception as exc:
        error=exc;valid=False
    finally:
        head.on=False
    agree({'microcheck_ms':[1,4,12,16]},valid)
    if error is not None:raise RuntimeError('draft CUDA microcheck failed') from error
    from vllm.distributed import get_tp_group
    return dict(rank=get_tp_group().rank_in_group,cases=rows,
        scope='actual loaded head CUDA GEMM; production MTP/KV replay still required')


def register(env=None):
    global FORMAT, INITIAL_ON
    env = os.environ if env is None else env
    FORMAT = options(env)
    INITIAL_ON = initial_on(env)
    if FORMAT == '0':
        return False
    from glm_skip_mla_plan import Hooks
    hooks = Hooks()
    import importlib.util
    root = Path(next(iter(importlib.util.find_spec('vllm').submodule_search_locations)))
    for name,expected in PINS.items():
        path = root/(name.removeprefix('vllm.').replace('.','/')+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('draft head source drift '+name)
    # Quantization and Marlin helper source closure already pinned by NVFP4.
    from glm_nvfp4_attn import check_sources
    check_sources()
    import sys
    sys.meta_path.insert(0,hooks)
    hooks.after_import('vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils',install_graph)
    hooks.after_import(RUNNER,install_runner)
    hooks.after_import(WORKER,install_worker)
    return True
