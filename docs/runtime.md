# Runtime

`start.sh` loads `profiles/current.env`, then `.env.example`, then an optional `.env` and invokes the
workstation launcher. The profile records every fixed explicit container env key from the selected
Boot B launch after removing its two instrumentation keys. Five fabric/rank keys are filled from `.env`.
`profiles/serve-args.json` records the effective ordered serving vector with exactly one speculative config.

The default `RECIPE_PROFILE=native-mtp-k2` selects native K2 async. Set `RECIPE_PROFILE=dspark-k3`
to apply `profiles/dspark-k3.env` and `profiles/dspark-k3-args.json`: MTP fix off, independent SWA
pool and draft low-memory loader on, unmodified Red Hat drafter K3, synchronous scheduling.
The alternative mounts and verifies `DRAFT_DIR`; native MTP needs only the target checkpoint.
Stop the current deployment before changing profiles and use a fresh runtime path.

The host runtime is mounted at `/overlay`; it contains these modules in the same container namespaces
used by the measured launch:

| startup order | module | active gate / source guard |
|---|---|---|
| 1 | `bringup/sitecustomize.py` | first on PYTHONPATH; aborts process on registration failure |
| 2 | `overlay/glm_fast_load.py` | `GLM_FAST_LOAD=1`; called directly when the external draft pool is off |
| 3 | `bringup/glm_mtp_fix.py` | `GLM_MTP_FIX=1`; pinned native MTP module, packed quant mapping |
| 4 | `bringup/glm_full_mla.py` | `GLM_FULL_MLA=triton`; SM90 backend SHA256 |
| 5 | `bringup/glm_window_memory.py` | full-MLA gate; pinned V1/V2 runner capture boundary |
| 6 | `bringup/glm_prefill_switch.py` | drained cap2048, constructor capacity4096 |
| 7 | `bringup/glm_dsa_short.py` | `GLM_INDEXER_SHORTCUT=1` (native K2 only); five pinned module sources |
| 8 | `bringup/glm_dirty_l2.py` | `GLM_DIRTY_L2=discard`; needs `GLM_FULL_MLA=triton` and `GLM_MLA_SPLIT_K=32`; split-kernel SHA256; registered last |

The native K2 profile disables the DSpark SWA pool and draft low-memory adapter. It loads MTP from `/model`, shares the checkpoint's target embedding/head and preserves the image's native MTP index compaction/reset. Fused QKV, gate-up and indexer packed-module mappings are supplied before compressed-tensors configuration. Explicit draft quantization is `compressed-tensors`; target and draft KV are FP8, block size64. No weight bytes, norm semantics or kernel arithmetic change. The .pth-installed `glm_roce.boot` registers transport independently.

The sparse-MLA adapter uses the pinned image's sparse slot conversion and metadata; it does not replace
DSA selection with dense attention. It handles 16 heads/rank, 512 latent dimensions and RoPE 0 or 64,
BF16 queries and plain BF16/FP8 KV. This model uses RoPE64 and FP8 target KV. At up to 36 rows it calls
`glm_full_mla_split_kernel.sparse_mla` with 32 splits; larger prefill/mixed batches use
`glm_full_mla_kernel.sparse_mla`. The arithmetic and launch bodies of both kernels match the deployed
source; only module descriptions changed. Kernel AST hashes excluding the top-level docstring are in
`docs/results/source-provenance.json`. Unsupported layouts and kernel failures propagate.

Native MTP K2 verifies three rows per stream (M3 at c1, M12 at c4), with graph captures1/3/6/12 and native MTP passes M1/M4. Greedy drafting, standard rejection and async scheduling are explicit. The target cache remains2GiB/rank, context32768 and four slots; scheduler cap2048 is selected after idle admission, with4096 constructor capacity.

## Short-context DSA shortcut

GLM-5.3's DSA indexer keeps the top 2048 tokens per row. When a row's context has at most 2048 tokens,
the pinned `persistent_topk` takes its all-selected branch: it writes 0..L-1 in order with -1 padding and
never reads the indexer logits. `glm_dsa_short` skips the indexer query GEMM (`wq_b`), the query
RoPE/FP8 quantisation and the paged MQA logits for exactly those rows. The key path (`wk_weights_proj`,
`k_norm`, key RoPE, index-cache write) is unchanged, so the selected tokens and both caches are the same
bytes as the stock path. There is no new GPU kernel.

Eligibility is decided on the CPU before graph dispatch, with no device synchronisation: target verify
batches of one or four requests with three tokens each (M3/M12) and both native MTP passes (M1/M4), every
row's upper-bound length at most 2048, no prefill, new, resumed, structured-output or mixed-width request.
Eligible batches replay separately captured FULL graphs (`short_context` in the graph descriptor); all other
batches, and every row past 2048 tokens, use the stock graphs. The module refuses to start unless the run is
native MTP K2, FULL_DECODE_ONLY, TP4 with no context parallelism and the pinned GLM-5.3 indexer geometry.
It logs `glm-dsa-short: armed`, the captured short widths and the first short dispatch. `GLM_INDEXER_SHORTCUT=0`
returns to the stock path; the DSpark profile sets it to 0.

## Dirty-L2 fix

The split32 MLA writes t x 32 x 16 x 512 fp32 partials (1 MiB per row), reduces them once and never reads
them again. Left dirty in L2, they are written back to DRAM while the absorbed UV bmm and the INT8 o_proj
that follow stream their weights; on one GB10 that write-back added 13 us (M3) to 38 us (M12) to each o_proj.
`glm_dirty_l2` wraps `glm_full_mla_split_kernel.sparse_mla` after import (the module's SHA256 is pinned):
the released `_mla_partial` runs unchanged, and `_mla_reduce_discard` is the released reduce body followed
by a CTA barrier and one predicated `discard.global.L2` per consumed 128-byte line. Each reduce program is
the only reader of its lines and discards them only after all of its loads, so the output is bit-identical.
Buffers, launch grids and captured graphs are unchanged; the larger-row unsplit kernel is not touched.
It logs `glm-dirty-l2: armed (discard, split 32)` and the first dispatch. It refuses any mode other than
`0` or `discard`, and any split count other than 32. `GLM_DIRTY_L2=0` returns to the released reduce; the
DSpark profile sets it to 0. The three kernel source hashes are in `docs/results/dirty-l2-profile.json`.

## Transport and loading

Both data rails use `rocep1s0f0,roceP2p1s0f0`, GID index 3, RoCE v2, `NCCL_NET=IB`, no NIC merging,
no cross-NIC, NVLS or cuMem allocations. Eligible collectives use RoCEnante; its size/type gates leave
other operations on NCCL. `--disable-custom-all-reduce` disables vLLM's separate local custom reducer.
NCCL buffer size is 1,048,576 bytes and the channel cap is eight. The library is preloaded from
`/opt/nccl/libnccl.so.2.30.7`. The base transport's spin limit/cache defaults remain image settings.

The target fast loader uses 64 MiB slabs, 256 MiB ahead and four reader threads, drops consumed cache
pages and checks two tensors per shard against the stock mmap source. The historical `/draft/` low-memory override is disabled for native MTP. Loader bytes/order/dtypes/shapes are tested against safetensors on CPU.

The recipe starts with a fresh per-deployment JIT cache built from the pinned image before the model containers start: after creating `OVERLAY_REMOTE/cache` and before compaction and the 110 GiB preboot check, the launcher runs one `docker run --rm` per node in parallel (same image and container environment with `PYTHONPATH` empty so no overlay hook runs, `/cache` and `/opt/nccl` mounted as in serving, no GPU, no network, 16 GiB memory limit, 600 s in-container timeout) that builds FlashInfer's `sampling` and `batch_mla_attention_…_e4m3_…_ckv_512_kpe_64` modules. Any failure stops the launch before compaction. The serving process then runs the same ninja build, finds it up to date and compiles nothing; before this step the cold sampling build (three nvcc processes, about 2-2.7 GiB of host memory for about 37 s) ran right after graph capture and pushed rank 0 below the launcher's 8 GiB floor. Sources, flags and the resulting modules are the ones the server would build itself. E2b built separate immutable prepared-weight snapshots for sync/async native K2, bound to constructor configuration as well as source/image/runtime/CLI. The measured fleet uses those caches; the fresh-clone launcher uses the native loader and has not been re-timed for K2. HF/transformers run offline inside serving containers.

## Watchdog

The foreground serving process owns the four containers and samples all ranks every 10 seconds.
It stops its deployment on any exited/OOM rank, SSH loss, MemAvailable below 6 GiB, swap usage at least
64 MiB above the initial sample for three consecutive samples, CUDA/worker/numeric errors, GPU use
above 20 percent without changing progress for 180 seconds, or a boot/admission deadline over 900 seconds.
During boot progress is log changes; after admission it is prompt/generated-token counters. GPU use is
sampled on every rank. The optional import_utils DeepEP WARNING is excluded; other error patterns remain fatal.

At capture, the source-pinned hook requires at least 10 GiB MemAvailable before it starts; while graphs are
being captured the launcher keeps an 8 GiB floor on every rank (native MTP K2 dips to about 9.5 GiB on the
head during capture). After health 200, admission
requires at least 8 GiB on all four ranks for 60 seconds. Monitoring remains active during benchmarks
and serving. Ctrl-C stops the owned deployment. A failed stop retains the lock. Detailed samples stay in
local `logs/`; passwords, tokens, cookies and private keys are never requested or written.
