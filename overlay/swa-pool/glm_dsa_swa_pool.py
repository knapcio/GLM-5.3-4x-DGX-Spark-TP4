# SPDX-License-Identifier: Apache-2.0
"""Independent paged drafter pools for vLLM 487ecf187. Default off.

Target specs and tensor geometry pass through unchanged. Drafter SWA uses its
original window/block geometry; full attention retains the configured context.
Import hooks validate whole source files before applying anchored source edits.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys

ENV = 'GLM_DSA_SWA_POOL'
TAG = 'glm-dsa-swa-pool'
PINS = json.loads(Path(__file__).with_name('source_pins.json').read_text())
U_NAME = 'vllm.v1.core.kv_cache_utils'
C_NAME = 'vllm.v1.core.kv_cache_coordinator'
M_NAME = 'vllm.v1.core.kv_cache_manager'
I_NAME = 'vllm.v1.kv_cache_interface'
D_NAME = 'vllm.v1.worker.gpu.spec_decode.dflash.speculator'
R_NAME = 'vllm.v1.worker.gpu.model_runner'


def enabled():
    value = os.environ.get(ENV, '0').strip()
    if value not in ('0', '1'):
        raise ValueError(f'{ENV} must be 0 or 1')
    if value == '1' and os.environ.get('GLM_DSA_DRAFT_FOLD', '0') != '0':
        raise ValueError(f'{TAG}: disable GLM_DSA_DRAFT_FOLD before arming')
    return value == '1'


def check_source(name, source):
    got = hashlib.sha256(source.encode()).hexdigest()
    if got != PINS[name]:
        raise RuntimeError(f'{TAG}: {name} sha {got} != {PINS[name]}; refusing source')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'{TAG}: anchor count {source.count(old)} != 1: {old[:70]}')
    return source.replace(old, new, 1)


def transform_source(name, source):
    check_source(name, source)
    if name == M_NAME:
        source = replace_once(source,
            'if required_blocks > self.block_pool.get_num_free_blocks():',
            'if not _glm_pool_fits(self.coordinator, num_blocks_to_allocate, watermark_blocks):')
        source = replace_once(source,
            'if required_blocks > available_blocks:',
            'if not _glm_pool_fits(self.coordinator, num_blocks_to_allocate, watermark_blocks + reserved_blocks):')
        source += '\nfrom glm_dsa_swa_pool import pool_fits as _glm_pool_fits\n'
    elif name == D_NAME:
        # Rejected target rows must never become committed drafter context.
        # Null entries are logical holes, not writable physical page zero.
        source = replace_once(source,
            'ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)',
            'ctx_slot = tl.where((ctx_block_id != 0) & (ctx_start + j < valid_ctx_end),\n'
            '                        ctx_block_id * block_size + (ctx_pos % block_size), PAD_SLOT_ID)')
        source = replace_once(source,
            'q_slot = q_block_id * block_size + (query_pos % block_size)',
            'q_slot = tl.where((q_block_id != 0) & (query_pos < max_model_len),\n'
            '                      q_block_id * block_size + (query_pos % block_size), PAD_SLOT_ID)')
    elif name == R_NAME:
        for field in ('new_block_ids_to_zero', 'kv_cache_block_copies'):
            source = replace_once(source,
                f'if scheduler_output.{field}:',
                f'if scheduler_output.{field} and _glm_is_pool(self.kv_cache_config):\n'
                f'            raise RuntimeError("{TAG}: unqualified {field} is unsupported")\n'
                f'        if scheduler_output.{field}:')
        source += '\nfrom glm_dsa_swa_pool import is_pool as _glm_is_pool\n'
    return source


def inner(group):
    spec = group.kv_cache_spec
    return getattr(spec, 'kv_cache_specs', {n: spec for n in group.layer_names})


def is_pool(config):
    return hasattr(config, 'glm_pool_blocks')


def pattern(mod, config, specs):
    """Fail closed for unsupported layouts; unrelated/no-draft models use stock."""
    target = {n: s for n, s in specs.items() if type(s) is mod.MLAAttentionSpec}
    draft = {n: s for n, s in specs.items() if type(s) in (mod.SlidingWindowSpec, mod.FullAttentionSpec)}
    if not target or not draft:
        return None
    if len(target) + len(draft) != len(specs):
        raise ValueError(f'{TAG}: unsupported spec types')
    pc = config.parallel_config
    if (pc.pipeline_parallel_size, pc.decode_context_parallel_size, pc.prefill_context_parallel_size) != (1, 1, 1):
        raise ValueError(f'{TAG}: requires PP1/DCP1/PCP1')
    if getattr(config, 'kv_transfer_config', None) is not None:
        raise ValueError(f'{TAG}: KV connectors require group-qualified transport; unsupported')
    if config.scheduler_config.disable_hybrid_kv_cache_manager or not config.use_v2_model_runner:
        raise ValueError(f'{TAG}: requires hybrid allocator and V2 runner')
    if config.model_config.original_max_model_len == -1:
        raise ValueError(f'{TAG}: explicit max_model_len required')
    if any(s.page_size_padded is not None or s.compress_ratio != 1 for s in target.values()):
        raise ValueError(f'{TAG}: target padding/compression unsupported')
    if mod.UniformTypeKVCacheSpecs.from_specs(target) is None:
        raise ValueError(f'{TAG}: target must remain one uniform group')
    if {s.block_size for s in target.values()} != {64}:
        raise ValueError(f'{TAG}: target block must be 64')
    # Restrict the overlay to the saved full-model geometry and BF16 drafters.
    import torch
    if len(target) != 99 or sorted(s.head_size for s in target.values()) != [132]*21 + [576]*78:
        raise ValueError(f'{TAG}: expected 78 MLA + 21 indexer layers')
    if any(s.dtype is not torch.uint8 for s in target.values()):
        raise ValueError(f'{TAG}: target must keep fp8/indexer uint8 storage')
    if any(s.dtype is not torch.bfloat16 or s.page_size_padded is not None for s in draft.values()):
        raise ValueError(f'{TAG}: unpadded BF16 drafter required')
    sw = {n: s for n, s in draft.items() if type(s) is mod.SlidingWindowSpec}
    full = {n: s for n, s in draft.items() if type(s) is mod.FullAttentionSpec}
    if not sw or len({s.sliding_window for s in sw.values()})!=1 or any(s.sliding_window not in (2048,4096) or s.block_size != 16 for s in sw.values()):
        raise ValueError(f'{TAG}: expected drafter SWA2048/4096 with block16')
    if len(full) > 1 or any(s.block_size != 64 or s.sliding_window is not None for s in full.values()):
        raise ValueError(f'{TAG}: at most one true full-attention/block64 drafter layer')
    return target, sw, full


def make_groups(mod, config, specs):
    parts = pattern(mod, config, specs)
    if parts is None:
        return None
    groups = []
    for i, part in enumerate(parts):
        if not part:
            continue
        uniform = mod.UniformTypeKVCacheSpecs.from_specs(part)
        if uniform is None:
            raise ValueError(f'{TAG}: each pool must have one allocation type')
        groups.append(mod.KVCacheGroupSpec(list(part), uniform, is_eagle_group=i != 0))
    return groups


def pool_counts(config, groups, available):
    """Conservative S * per-request peak, including speculative tail and null page."""
    seqs = config.scheduler_config.max_num_seqs
    lookahead = config.speculative_config.num_speculative_tokens + 1
    max_len = config.model_config.max_model_len
    flight = config.max_in_flight_tokens + lookahead
    counts = [0]
    for group in groups[1:]:
        spec = next(iter(inner(group).values()))
        window = getattr(spec, 'sliding_window', None)
        if window is not None:
            peak = math.ceil(min(window - 1 + flight, max_len) / spec.block_size) + 1
        else:
            peak = math.ceil(max_len / spec.block_size)
        counts.append(1 + seqs * peak)
    reserve = sum(n * g.kv_cache_spec.page_size_bytes for n, g in zip(counts[1:], groups[1:]))
    counts[0] = (available - reserve) // groups[0].kv_cache_spec.page_size_bytes
    if counts[0] < 1 + math.ceil(max_len / groups[0].kv_cache_spec.block_size):
        raise ValueError(f'{TAG}: KV budget insufficient for one full target request plus drafter pools')
    return tuple(counts)


def build_config(mod, config, groups, available, counts=None):
    override = config.cache_config.num_gpu_blocks_override
    if override is not None:
        # Only the V2 runner's dry-capture initializer uses zero-budget overrides.
        if available != 0 or counts is not None:
            raise ValueError(f'{TAG}: production num_gpu_blocks_override unsupported')
        counts = (max(1, override),) * len(groups)
    elif counts is None:
        counts = pool_counts(config, groups, available)
    tensors = [mod.KVCacheTensor(size=s.page_size_bytes * count, shared_by=[name])
               for group, count in zip(groups, counts) for name, s in inner(group).items()]
    result = mod.KVCacheConfig(num_blocks=counts[0], kv_cache_tensors=tensors, kv_cache_groups=groups)
    result.glm_pool_blocks = tuple(counts)
    result.glm_pool_max_seqs = config.scheduler_config.max_num_seqs
    result.glm_pool_flight = config.max_in_flight_tokens + config.speculative_config.num_speculative_tokens + 1
    sys.stderr.write(f'{TAG}: pool blocks={counts}, bytes={sum(t.size for t in tensors)}, '
                     f'target physical tokens={counts[0]*64}, usable={(counts[0]-1)*64}\n')
    return result


def install_utils(mod):
    if getattr(mod.get_kv_cache_groups, '__glm_swa_pool__', False):
        return
    stock_groups = mod.get_kv_cache_groups
    stock_configs = mod.get_kv_cache_configs
    stock_from = mod.get_kv_cache_config_from_groups
    stock_capacity = mod.get_max_concurrency_for_kv_cache_config

    def groups(config, specs):
        return make_groups(mod, config, specs) or stock_groups(config, specs)

    def configs(config, specs, available):
        gs = make_groups(mod, config, specs[0]) if specs else None
        if gs is None:
            return stock_configs(config, specs, available)
        if len(specs) != len(available) or any(s != specs[0] for s in specs):
            raise ValueError(f'{TAG}: TP ranks must expose identical specs/budgets')
        if config.cache_config.num_gpu_blocks_override is not None:
            raise ValueError(f'{TAG}: production block override unsupported')
        counts = pool_counts(config, gs, min(available))
        return [build_config(mod, config, copy.deepcopy(gs), a, counts) for a in available]

    def from_groups(config, gs, available):
        if gs and len(gs) > 1 and all(type(s) is mod.MLAAttentionSpec for s in inner(gs[0]).values()):
            return build_config(mod, config, gs, available)
        return stock_from(config, gs, available)

    def capacity(config, cache):
        if not is_pool(cache):
            return stock_capacity(config, cache)
        length = config.model_config.max_model_len
        limits = [cache.glm_pool_max_seqs]
        for group, count in zip(cache.kv_cache_groups, cache.glm_pool_blocks):
            s = next(iter(inner(group).values()))
            if getattr(s, 'sliding_window', None) is None:
                limits.append((count - 1) / math.ceil(length / s.block_size))
        return min(limits)

    groups.__glm_swa_pool__ = True
    mod.get_kv_cache_groups = groups
    mod.get_kv_cache_configs = configs
    mod.get_kv_cache_config_from_groups = from_groups
    mod.get_max_concurrency_for_kv_cache_config = capacity


class Demand(int):
    def __new__(cls, values):
        obj = super().__new__(cls, sum(values))
        obj.values = tuple(values)
        return obj


class Pools:
    """Lifecycle dispatch by object identity; numerical IDs are local to each pool."""
    def __init__(self, pools):
        self.pools = pools
        self.hash_block_size = pools[0].hash_block_size
        self.null_block = pools[0].null_block
        self.num_gpu_blocks = sum(p.num_gpu_blocks for p in pools)
        self.owners = {id(b): p for p in pools for b in p.blocks}

    def free_blocks(self, blocks):
        # Preserve stock reversed/deferred free order within each pool.
        for block in blocks:
            self.owners[id(block)].free_blocks((block,))

    def touch(self, blocks):
        for block in blocks:
            self.owners[id(block)].touch((block,))

    def get_num_free_blocks(self):
        return sum(p.get_num_free_blocks() for p in self.pools)

    def get_usage(self):
        return max(p.get_usage() for p in self.pools)

    def evict_blocks(self, ids):
        # The public API lacks group IDs: invalidate in every pool conservatively.
        for p in self.pools:
            p.evict_blocks({i for i in ids if i < p.num_gpu_blocks})

    def reset_prefix_cache(self):
        # No partial reset while any pool has live request references.
        if any(p.get_num_free_blocks() != p.num_gpu_blocks - 1 for p in self.pools):
            return False
        return all(p.reset_prefix_cache() for p in self.pools)

    def take_events(self):
        return [e for p in self.pools for e in p.take_events()]

    def emit_cached_block_events(self, request, count, block_size, group_id):
        self.pools[group_id].emit_cached_block_events(request, count, block_size, group_id)


def pool_fits(coordinator, demand, reserve=0):
    if not isinstance(coordinator.block_pool, Pools):
        return int(demand) + reserve <= coordinator.block_pool.get_num_free_blocks()
    if not isinstance(demand, Demand):
        raise RuntimeError(f'{TAG}: lost group-qualified allocation demand')
    return all(n + reserve <= p.get_num_free_blocks()
               for n, p in zip(demand.values, coordinator.block_pool.pools))


def install_coordinator(mod):
    base = mod.KVCacheCoordinator
    if getattr(base, '__glm_swa_pool__', False):
        return
    init = base.__init__
    init_sig = inspect.signature(init)
    demand_fn = base.get_num_blocks_to_allocate
    demand_sig = inspect.signature(demand_fn)
    hybrid_hit = mod.HybridKVCacheCoordinator.find_longest_cache_hit

    def initialize(self, *args, **kwargs):
        bound = init_sig.bind(self, *args, **kwargs); bound.apply_defaults()
        cfg = bound.arguments['kv_cache_config']
        if is_pool(cfg) and bound.arguments['metrics_collector'] is not None:
            raise ValueError(f'{TAG}: detailed KV metrics use unqualified IDs; unsupported')
        init(self, *args, **kwargs)
        if not is_pool(cfg):
            return
        from vllm.v1.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
        pools, managers = [], []
        for gid, (group, count) in enumerate(zip(cfg.kv_cache_groups, cfg.glm_pool_blocks)):
            pool = mod.BlockPool(count, self.enable_caching, bound.arguments['hash_block_size'],
                                 bound.arguments['enable_kv_cache_events'], metrics_collector=None)
            pools.append(pool)
            manager = get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec, max_in_flight_tokens=cfg.glm_pool_flight,
                max_model_len=self.max_model_len, block_pool=pool, enable_caching=self.enable_caching,
                kv_cache_group_id=gid, scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=False)
            manager.use_eagle = gid in self.eagle_group_ids
            managers.append(manager)
        self.block_pool = Pools(pools)
        self.single_type_managers = tuple(managers)

    def demand(self, *args, **kwargs):
        if not is_pool(self.kv_cache_config):
            return demand_fn(self, *args, **kwargs)
        a = demand_sig.bind(self, *args, **kwargs); a.apply_defaults(); a = a.arguments
        return Demand([m.get_num_blocks_to_allocate(
            a['request_id'], a['num_tokens'], a['new_computed_blocks'][i],
            a['total_computed_tokens'], a['num_local_computed_tokens'], a['num_tokens_main_model'],
            apply_admission_cap=a['apply_admission_cap'])
            for i, m in enumerate(self.single_type_managers)])

    def hit(self, hashes, maximum):
        if not is_pool(self.kv_cache_config):
            return hybrid_hit(self, hashes, maximum)
        candidate = maximum // self.scheduler_block_size * self.scheduler_block_size
        # Every miss shrinks the candidate. Re-evaluate ALL groups, including
        # both full-attention groups, at the final boundary before returning.
        while True:
            results = []
            smallest = candidate
            for gid, manager in enumerate(self.single_type_managers):
                eagle = gid in self.eagle_group_ids
                margin = manager.block_size if eagle else 0
                blocks, length = manager.find_longest_cache_hit(
                    block_hashes=hashes, max_length=min(candidate + margin, maximum),
                    kv_cache_group_ids=[gid], block_pool=manager.block_pool,
                    kv_cache_spec=manager.kv_cache_spec, drop_eagle_block=eagle,
                    alignment_tokens=self.scheduler_block_size)
                results.append(blocks[0])
                smallest = min(smallest, length)
            smallest = smallest // self.scheduler_block_size * self.scheduler_block_size
            if smallest == candidate:
                return tuple(results), candidate, 0
            candidate = smallest

    base.__init__ = initialize
    base.get_num_blocks_to_allocate = demand
    mod.HybridKVCacheCoordinator.find_longest_cache_hit = hit
    base.__glm_swa_pool__ = True


def install_interface(mod):
    # Independent pools cannot reinterpret a target block as BF16 drafter KV.
    # Group-blind zero/copy messages would corrupt equal-numbered live blocks.
    stock = mod.KVCacheConfig.needs_kv_cache_zeroing.fget
    mod.KVCacheConfig.needs_kv_cache_zeroing = property(
        lambda cfg: False if is_pool(cfg) else stock(cfg))


def install_runner(mod):
    init = mod.GPUModelRunner.initialize_kv_cache
    dummy = mod.set_dummy_context

    def initialize(self, cfg):
        init(self, cfg)
        if is_pool(cfg):
            self.block_tables.glm_pool_blocks = cfg.glm_pool_blocks

    def dummy_context(batch, tables, context_len, num_blocks, max_len):
        # Profiling contexts must not index a smaller drafter pool with a
        # target-sized modulus. Capture's normal zero dummy tables stay valid.
        counts = getattr(tables, 'glm_pool_blocks', (num_blocks,))
        return dummy(batch, tables, context_len, min(counts), max_len)

    mod.GPUModelRunner.initialize_kv_cache = initialize
    mod.set_dummy_context = dummy_context


INSTALLERS = {U_NAME: install_utils, C_NAME: install_coordinator,
              I_NAME: install_interface, R_NAME: install_runner}


def register():
    if not enabled():
        return
    if any(n in sys.modules for n in PINS):
        raise RuntimeError(f'{TAG}: must register before vllm imports')
    if any(getattr(f, '__glm_swa_pool__', False) for f in sys.meta_path):
        return

    class Hook(importlib.abc.MetaPathFinder):
        __glm_swa_pool__ = True

        def find_spec(self, fullname, path, target=None):
            if fullname not in PINS:
                return None
            # Remove temporarily to resolve the actual source loader.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None or not hasattr(spec.loader, 'get_source'):
                raise RuntimeError(f'{TAG}: source loader required for {fullname}')
            source = transform_source(fullname, spec.loader.get_source(fullname))

            original_exec = spec.loader.exec_module
            if not hasattr(spec.loader, 'get_code'):
                raise RuntimeError(f'{TAG}: source-code loader required for {fullname}')
            code = compile(source, spec.origin, 'exec', dont_inherit=True)
            spec.loader.get_code = lambda name: code

            def execute(module):
                # Preserve pre-existing post-import hooks (RoCE and memory guard).
                original_exec(module)
                if fullname in INSTALLERS:
                    INSTALLERS[fullname](module)
                sys.stderr.write(f'{TAG}: pinned {fullname}\n')

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Hook())
    sys.stderr.write(f'{TAG}: registered (487ecf187)\n')
