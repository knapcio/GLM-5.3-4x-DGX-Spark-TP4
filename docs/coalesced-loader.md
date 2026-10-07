# Opt-in coalesced checkpoint loader

Base: `perf/adaptive-chunk` `84a38e06f16ee09e0521a653851fe574bbf57fae`.
Set `GLM_LOADER=coalesced` to select the new iterator. Unset it to keep the
existing `GLM_FAST_LOAD` policy; `GLM_LOADER=fast` explicitly selects that policy.
The checked profile uses lazy safetensors. Other strategies continue through
the original iterator and are outside the coalesced budget/receipt contract.
Unknown loader values and excessive budgets fail before placement.

## Provenance and measured comparison

The modified port is from Allan Clark / ajclark's
[coalesced.py](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/blob/f0b64af5ac6028624e5ae255d998faa1e0a9324a/runtime/nvme_loader/spark_nvme/coalesced.py)
and `streaming.py`, Apache-2.0. Original LICENSE and NOTICE are retained
verbatim in `LICENSES/ajclark-*`; source hashes are in
`docs/ajclark-loader-provenance.json`. The pinned fetched snapshot on the Mac
is `/srv/projects/ext/ajclark-f0b64af`, sealed read-only. Shell DNS was
unavailable, so the files were fetched through GitHub instead of git clone.
The repository was identified through the tonyd2wild/ajclark attribution trail.

Ajclark's [implementation receipt](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/blob/f0b64af5ac6028624e5ae255d998faa1e0a9324a/docs/COALESCED-LOADER-IMPLEMENTATION.md)
reports 41.8–42.4 s target loading and 114.68 s health, from fresh workers on
already booted hosts with compiler/driver caches. There is one activation
per configuration. It reads 405.220 GB / 177,569 tensors per rank, using
32 readers, 128 MiB batches, about 256 MiB staging and 3.82 GB peak owning
storage. Its minimum sampled available memory was about 1.68 GB. These
conditions differ from this recipe's 10 GiB pre-capture and 8 GiB warm floors.

| Property | Existing fast loader | Upstream coalesced | Local port |
|---|---|---|---|
| Reads | Concurrent per-tensor `preadv`, buffered, consumed-page eviction | Concurrent aligned direct ranges | Concurrent aligned direct ranges, packed within stock-order groups |
| Batching | 64 MiB host slabs, 256 MiB read-ahead | 128 MiB CUDA batches | At most 128 MiB per normal batch |
| Placement input | Host views, native slice/uploads | Owning CUDA views | Owning CUDA views; CPU views in Mac tests |
| Pipeline | Reader futures ahead of native consumption | Producer uploads overlap reads and placement | One queued batch plus producer and consumer; event fences |
| Staging | Pinned caching allocator slabs | Two reusable pinned tiles | Two tiles: 256 MiB pinned requests; +8 KiB alignment space in CPU tests |
| Owning outputs | Slab lifetime monitored | Default 6 GiB budget | Hard maximum/default 4 GiB; retained aliases charged |
| Order/filters | Stock order, MTP/EP before reads | Physical order; skips at consumption | Stock order; MTP/EP before range planning |
| Prepared conversion | None | None | None; no dtype conversion or quantizer change |

Concurrency alone is not the explanation: the old path already has four
readers. The material changes are page-cache bypass, fewer/larger uploads,
and explicit overlap with native placement. The upstream synchronous-to-async
64 MiB change cut target loading from 53.4–54.9 to 44.9–45.7 s. Stock ordering,
16 readers and tighter memory accounting may reduce the transferable gain.

## Memory and correctness contract

- `GLM_COALESCED_BATCH_MB`: 1..128, default 128.
- `GLM_COALESCED_THREADS`: 1..32, default 16.
- `GLM_COALESCED_OWNED_MB`: 1..4096, default 4096; must fit three normal batches.
- `GLM_COALESCED_DIRECT`: 0 or 1, default 1. Mac tests use 0.
- `GLM_COALESCED_EMERGENCY_MB`: 1..9216, default 3072 (3 GiB).
- `GLM_COALESCED_MARGIN_MB`: 1..4096, default 1024 (1 GiB).
- Header inventory: at most 64 MiB encoded headers and 200,000 selected tensors.
  These metadata limits bound planning independently of payload size. Decoded
  metadata has Python object overhead and is separate from the staging budget.

The 2026-10-06 13:59 fleet attempt failed on all four ranks: the old 10 GiB
absolute check charged unified-memory destination commit as loader pressure.
The loader now snapshots `MemAvailable` at each iterator's start. Its live
staging plus owning batches plus the next allocation must fit that baseline
minus the margin. Pinned tiles are the staging allocation, counted once.
Pressure checks require `available + destination_write_credit + accounted
loader_storage >= baseline - margin`. Accounted owning storage keeps its
bounded high-water mark through teardown to allow allocator-cached batches.
Separately, `available - next_allocation >= emergency_floor` always applies,
even after 95 GiB of destination credit. Missing Linux meminfo fails closed.

`DefaultModelLoader.load_weights` registers destination parameter/buffer
storages. A dispatch observer credits unique byte ranges of native `copy_`
destinations, after TP/EP slicing, before enqueue (covering concurrent commit).
Repeated writes and aliases are deduplicated; strided writes use exact ranges.
Unknown storage and more than 200,000 fragments receive zero credit. The
observer does not credit full source tensors or a guessed TP ratio. Written
bytes bound possible new destination commit; already-resident destination
pages can make that credit conservative. The absolute emergency check and
independent storage bounds remain necessary throughout the load. No capacity
credit is given to KV, and existing fleet/capture/admission guards remain.

Direct-I/O support is probed before native placement. Any coalesced exception
before the first handoff selects the ordinary fast loader automatically;
all ranks agree on that fallback. Temporary batches/caches are drained first,
and the rest of the same model load stays on fast. Once any tensor has been
handed to placement, a coalesced failure raises and aborts the whole load.
Buffered mode
evicts only consumed extents. No global cache flushing is added.

Large tensors are assembled through the two fixed host tiles into bounded
owning output storage. A large tensor or native fusion retaining too much
storage fails rather than waiting for later weights to release it. Storage
weak references count entire backing allocations, even for tiny aliases.
The producer fences each pinned slot before overwrite; consumers wait on
upload events and record their stream on output storage. Each batch also
finishes native consumer work before retiring temporary storage, limiting
allocator accumulation while the next batch is read/uploaded.

Source stamps are checked before reading each batch and after completion.
Short reads finish safely; required EOF aborts. Header checks are not payload
hashes. Names, order, shapes, dtypes and bytes are preserved; native TP slicing,
packed weight mapping and post-processing remain in vLLM. `GLM_MTP_ONLY_LOAD` and
`GLM_TARGET_SKIP_MTP` use the existing selection context and its index/audit
checks. Exceptions never switch transport after partial placement.

Multi-rank loads use a separately connected TCPStore client with a 2 s RPC
timeout. All ranks rendezvous before first placement (one common backend
decision) and after model loading. A monitor polls a shared abort key every
100 ms, independent of tensor counts/order; there is no per-tensor collective.
Local failures publish that key, and progressing peers raise and drain their
read/upload pipeline. A peer stuck in native/CUDA/NCCL code is terminated
with SIGTERM after 30 s and SIGKILL after another 5 s. Missing ranks time out
at rendezvous after 120 s. Unsupported distributed stores fail closed.

Pinned requests use a power-of-two allocation bin (charged in full); the allocator must
return page-aligned storage or the opt-in load fails. This avoids requesting
128 MiB + 4 KiB and potentially rounding each host block to 256 MiB. Staging
metrics describe tensor storage, not allocator bookkeeping. Outstanding read
futures are capped at twice the configured thread count.

Load teardown drains readers/uploads; `release` synchronizes and empties
temporary CUDA and pinned allocator caches before KV profiling. The ordinary
KV head, dispram lease/carveout, null block, FP4x pool size, prefill banks and
capture guards remain the base recipe's. No loader-memory saving is credited
to increased KV capacity in this port.

## NVFP4 attention composition

At `perf/nvfp4-attn` `3488797`, the sidecar transform wraps the fast iterator
and swaps selected target attention companions, while bypassing the draft.
Keep that transform outside the coalesced iterator when merging the branches;
resolve the shared `glm_fast_load.py` hunk by retaining its NVFP4 wrapper and
this port's backend dispatch. The `_glm_fast_load` marker is intentionally
retained for that composition.

The real converter/manifest/transform passes a local two-matrix composition
test: stock + sidecar equals coalesced + the same sidecar (six tensor hashes).
This proves iterator composition on CPU, not GPU qualification or equivalence
between INT8 and NVFP4. Sidecar `load_file(..., device='cpu')` reads and native
NVFP4 repacking allocate outside this loader's budgets. A combined branch
must measure those peaks and retain its own admission check; it cannot reuse
this port's 4 GiB budget as a bound for sidecar memory. Hold the sidecar off
for the first loader boot. No quantized attention source is discarded early
by this port, so it claims no sidecar read-time saving.

## Forecast and next gate

The five saved boots have median main load 188.72 s and mean health 282.44 s;
mean main/MTP loads are 188.76/6.96 s. Discounted forecast: **60 s main load**,
with a **45–90 s** planning range, conditional on direct-I/O support and the
memory/retention gates. Holding all other phases fixed predicts **153.68 s
health**, saving **128.76 s/boot**. A 42 s main load predicts 135.68 s health,
not 115 s. GPU load time and actual memory savings are unmeasured locally.

The exact one-candidate-boot procedure is in `docs/coalesced-loader-fleet-plan.md`.
