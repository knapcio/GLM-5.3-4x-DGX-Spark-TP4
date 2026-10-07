# Runtime

`start.sh` loads `profiles/current.env`, then `.env.example`, then an optional `.env` and invokes the
workstation launcher. The profile records every fixed explicit container env key from the selected
Boot B launch after removing its two instrumentation keys. Five fabric/rank keys are filled from `.env`.
`profiles/serve-args.json` records the effective ordered serving vector with exactly one speculative config.

The default `RECIPE_PROFILE=native-mtp-k2` selects native MTP async with K-stop (up to three drafts, confidence
stop, uniform multi-request batches) and prefix caching; `GLM_MTP_KSTOP=0` (with `GLM_MTP_KSTOP_UNIFORM_BATCH=0`
and, if wanted, `GLM_INDEXER_SHORTCUT=1`) in `.env` returns to the released native K2 vector apart from prefix
caching and shard selection. Set `RECIPE_PROFILE=dspark-k3`
to apply `profiles/dspark-k3.env` and `profiles/dspark-k3-args.json`: MTP fix off, independent SWA
pool and draft low-memory loader on, unmodified Red Hat drafter K3, synchronous scheduling.
The alternative mounts and verifies `DRAFT_DIR`; native MTP needs only the target checkpoint.
Stop the current deployment before changing profiles and use a fresh runtime path.

The host runtime is mounted at `/overlay`; it contains these modules in the same container namespaces
used by the measured launch:

| startup order | module | active gate / source guard |
|---|---|---|
| 1 | `bringup/sitecustomize.py` | first on PYTHONPATH; aborts process on registration failure |
| 2 | `overlay/glm_param_hash.py` | opt-in tool, never in a profile: `GLM_PARAM_HASH=1` with the dev API (campaign weight-equality windows) |
| 3 | `overlay/glm_fast_load.py` (+ `glm_mtp_select.py`) | `GLM_FAST_LOAD=1`; called directly when the external draft pool is off; shard selection with `GLM_MTP_ONLY_LOAD=1` / `GLM_TARGET_SKIP_MTP=1` (on) |
| 4 | `bringup/glm_mtp_fix.py` | `GLM_MTP_FIX=1`; pinned native MTP module, packed quant mapping |
| 5 | `bringup/glm_full_mla.py` | `GLM_FULL_MLA=triton`; SM90 backend SHA256 |
| 6 | `bringup/glm_window_memory.py` | full-MLA gate; pinned V1/V2 runner capture boundary |
| 7 | `bringup/glm_prefill_switch.py` | drained cap2048, constructor capacity4096 |
| 8 | `bringup/glm_dsa_short.py` | `GLM_INDEXER_SHORTCUT=1` (off in the default: K-stop is on); five pinned module sources |
| 9 | `bringup/glm_dirty_l2.py` | `GLM_DIRTY_L2=discard`; needs `GLM_FULL_MLA=triton` and `GLM_MLA_SPLIT_K=32`; split-kernel SHA256 |
| 10 | `bringup/glm_glue_lite.py` | staged, off: any of `GLM_GLUE_ROUTER_BF16`, `GLM_GLUE_MOE_WS`, `GLM_GLUE_DSA_IDX_CACHE` = `1`; eleven pinned module sources; registered after dirty L2 |
| 11 | `kstop/glm_mtp_kstop.py` | on: `GLM_MTP_KSTOP=1`; composes with `GLM_INDEXER_SHORTCUT=1`; appends `/overlay/kstop` to `sys.path`; eight pinned module sources |
| 12 | `bringup/glm_skip_mla_plan.py` | optional, off: `GLM_SKIP_MLA_PLAN=1` or `ab`; needs `GLM_FULL_MLA=triton`; SM90 backend and worker SHA256; registered last |

`bringup/glm_spec_sample.py` (optional, off) registers right after the fast loader when `GLM_SPEC_SAMPLE=1`; it
needs `GLM_MTP_KSTOP=0` (the launcher refuses the pair, and K-stop's MTP install refuses any draft sampler other
than greedy).

The native K2 profile disables the DSpark SWA pool and draft low-memory adapter. It loads MTP from `/model`, shares the checkpoint's target embedding/head and preserves the image's native MTP index compaction/reset. Fused QKV, gate-up and indexer packed-module mappings are supplied before compressed-tensors configuration. Explicit draft quantization is `compressed-tensors`; target and draft KV are FP8, block size64. No weight bytes, norm semantics or kernel arithmetic change. The .pth-installed `glm_roce.boot` registers transport independently.

The sparse-MLA adapter uses the pinned image's sparse slot conversion and metadata; it does not replace
DSA selection with dense attention. It handles 16 heads/rank, 512 latent dimensions and RoPE 0 or 64,
BF16 queries and plain BF16/FP8 KV. This model uses RoPE64 and FP8 target KV. At up to 36 rows it calls
`glm_full_mla_split_kernel.sparse_mla` with 32 splits; larger prefill/mixed batches use
`glm_full_mla_kernel.sparse_mla`. The arithmetic and launch bodies of both kernels match the deployed
source; only module descriptions changed. Kernel AST hashes excluding the top-level docstring are in
`docs/results/source-provenance.json`. Unsupported layouts and kernel failures propagate.

With K-stop on (default) the launcher selects K3 and capture sizes `[1, 4, 16]` with a 544-block KV pool; see
the K-stop section. With `GLM_MTP_KSTOP=0`, native MTP K2 verifies three rows per stream (M3 at c1, M12 at c4), with graph captures1/3/6/12 and native MTP passes M1/M4. Greedy drafting, standard rejection and async scheduling are explicit. The target cache remains2GiB/rank, context32768 and four slots; scheduler cap2048 is selected after idle admission, with4096 constructor capacity.

## Launch-shape switches

`RECIPE_*` lines in `profiles/current.env` are read by the launcher and never become container variables.
Their defaults (`RECIPE_*` at `0` / `32768`) leave the vector as the profile selects it; values other than those
listed are refused, and the DSpark profile resets them. With K-stop on, `RECIPE_C4_PASS2_GRAPH` has no effect
(`[1, 4, 16]` already gives the later pass M4), and `RECIPE_KV_PIN_L1=1` and `RECIPE_MAX_MODEL_LEN=44288` are refused. `rank_args()` applies them to a copy of `profiles/serve-args.json`.

| Switch | Default | On |
|---|---|---|
| `RECIPE_C4_PASS2_GRAPH` | `0` | `1`: `cudagraph_capture_sizes` becomes `[1, 3, 4, 6, 12]` (same JSON spelling, nothing else changes) |
| `RECIPE_MAX_MODEL_LEN` | `32768` | `44288`: `--max-model-len 44288`; `44224` only with `GLM_MTP_KSTOP=1` (its full-pool layout) |
| `RECIPE_KV_PIN_L1` | `0` | `1`: `--kv-cache-memory-bytes 2543549952` (821 blocks, 2.369 GiB) instead of 2 GiB |
| `RECIPE_NCCL_NO_LL128` | `0` | `1`: `NCCL_PROTO=^LL128` added to every container's environment (model and JIT prep) |
| `GLM_MTP_KSTOP` (also a container switch) | `1` | `1`: `num_speculative_tokens` 3, capture sizes `[1, 4, 16]` (max 16) and one of the two K-stop KV layouts below; refused with `GLM_INDEXER_SHORTCUT=1` or `RECIPE_KV_PIN_L1=1` |
| `GLM_MTP_KSTOP_UNIFORM_BATCH` (container switch) | `1` | `1` (needs `GLM_MTP_KSTOP=1`): batches of two or more requests draft all three tokens, see the K-stop section |

The later native MTP pass runs with one token per request, and its graph manager drops capture sizes above the
four request slots, so the released list gives it M1 and M3 only and a four-request later pass runs eager.
Width 4 adds that graph (and, with the short-context shortcut, its short-context twin). Target and first-MTP
graphs are the same under both lists. A changed capture list is a new launch vector, so a prepared-weight
snapshot keyed on the launch arguments does not carry over.

One 64-token block of FP8 target KV is 3,098,112 bytes per rank (79 MLA layers x 576 B and 22 indexer
layers x 132 B per token). The released 2 GiB pin is 693 blocks; vLLM keeps one as its null block and needs
`cdiv(max_model_len, 64)` blocks for one request, so 44,288 is the largest context the released pool can
serve. The launcher refuses a context that does not fit the selected pin. The L1 pin (821 blocks) adds
8,192 tokens of pool for concurrent requests; the context stays as selected.

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

## Glue-lite (staged, off by default)

`glm_glue_lite` removes small launches whose results are unused or recomputed identically; it changes no
arithmetic and adds no kernel. Each switch is `0` or `1`; any other value, and the bank-switching mode
`GLM_GLUE_LITE_BANKABLE` (it needs an external in-boot A/B harness), stops startup. Strict mode (default)
refuses source drift in any of the eleven pinned vLLM files and requires the arming counts in
`GLM_GLUE_ROUTER_EXPECT=target:75,mtp:1` and `GLM_GLUE_IDX_EXPECT=57`.

- `GLM_GLUE_ROUTER_BF16=1` (F1): the GLM gate's last tier returns the BF16 router logits instead of casting
  them to FP32. The fused CUDA `grouped_topk` single-group kernel widens each BF16 logit to FP32 before any
  arithmetic, so the selected experts and weights are those of the cast input. Armed per MoE layer only for
  the covered contract (256 experts, one group, top-8, sigmoid with an FP32 correction bias, renormalize,
  scaling 1.0 or 2.5, modular Marlin experts, no fused shared-expert gate); anything else is refused.
- `GLM_GLUE_MOE_WS=1` (F2): `fused_marlin_moe` gets one persistent zeroed lock workspace per device instead
  of `torch.zeros` per call. Marlin resets every lock it used before a call completes, so stream-ordered reuse
  sees the same zeros. It is allocated at model load; a capture that finds none fails closed; a caller's own
  workspace is never touched; eager calls on a second stream use the stock allocation.
- `GLM_GLUE_DSA_IDX_CACHE=1` (F3): on the 57 static skip-top-k layers of the target model, the index
  conversion (`triton_convert_req_index_to_global_index` and its `full_like(-1)`) reuses the result of the
  last computing layer of the same forward when every input pointer, shape, stride, dtype and flag is the
  same. Any other MLA forward invalidates it; the MTP layer is never armed.

It logs `glm-glue-lite: registered`, the F1 and F3 arming counts and, in strict mode, refuses a partial
arming. The single-GPU gates in `tests/gpu/` (`run_gpu_gates.py` runs F1, F2, F3 in order on one idle GPU in
the pinned image) check bit equality against the stock path, including CUDA-graph replay with inputs changed
in place; they have not been run yet, and the switches stay off until they pass and a fleet A/B shows the gain.
The DSpark profile sets all three to 0.

## Native MTP K-stop (on by default)

`overlay/kstop/` runs the checkpoint's MTP layer with up to three drafts and a confidence stop. The draft
probability of each pass is `exp(max - logsumexp)` of the native MTP head's logits (greedy drafts unchanged).
Pass 1 always runs; pass 2 runs only for requests with p1 >= tau and pass 3 only for p1·p2 >= tau, with
tau 0.74 for every kind of traffic (`overlay/kstop/control.json`, read once per process). A stopped request
keeps the draft it just computed; the verifier then checks only the drafted tokens plus the bonus row. At c1
the runner selects the physical M2/M3/M4 graph from lengths it already knows on the host. In a mixed four-request
batch the rectangle keeps its width and the rows after each request's bonus row reuse that request's root
experts after top-k (`deadrow_ops`, one small Triton launch), so they add no expert reads; they are masked from
the sampler. Probabilities travel in the existing async MLA planner copy, so there is no extra device sync.

Control is local: each rank decides from its own copy of the all-gathered MTP logits (vLLM's logits processor
all-gathers the full vocabulary, so all ranks hold the same bits). Before the target forward and before each
further MTP pass, every rank enqueues one 13 x int64 (104 B) PyNccl MAX all-reduce carrying 124 bits of a
running SHA-256 of the decision inputs, the exact FP32 bits of the probabilities and a local-refusal flag. The
result rides the planner copy, and a mismatch raises the same error on every rank at the same host point.
Broadcast control is refused outside offline CPU tests. The campaign's isolated canary measured about 96 us per
guard (one per K2-shaped cycle, two to three per K3-stop cycle); the serving cost is unmeasured.

The overlay source-pins eight vLLM modules and transforms them at import (runner, autoregressive and MTP
speculator, graph manager, worker warm-up, MoE runner, rejection sampler, SM90 MLA planner); it reads the pinned
file through `PathFinder`, so it composes with the full-MLA wrapper loader, and fails closed unless its code ran
exactly once. It refuses phase-K, K-bank, hybrid, SWA-pool, draft-fold, dead-row and early-plan hooks, and the
launcher and startup refuse it with the short-context shortcut, which transforms the same runner, speculator and
graph sources. It requires native MTP with greedy drafts and standard rejection, async scheduling, TP4,
block 64, at most four slots, FULL_DECODE_ONLY graphs and modular Marlin MoE. The correctness gate, guard canary,
coordinator and A/B switch used on the fleet are campaign tools and are not part of the served path.

The served runtime and hook are the ones that booted on the fleet at c1 on 2026-10-02 after two fixes: vLLM's
warm-up runs real decode steps with spec tokens whose drafts are synthetic, so warm-up is now synthetic for
K-stop as well (and checked clean right after it); and a new or resumed request padded to the decode width by the
scheduler carries spec tokens without any draft, so it is verified at its full scheduled width. The runtime binds
the full GLM-5.3 classes explicitly (`GlmMoeDsaForCausalLM` target, `DeepSeekMTP` drafter) and fails closed on
anything else. The four-slot `[1, 4, 16]` layout of this release has not booted yet.

tau was chosen offline from assumed costs (K2 79.6 ms, K3 91.2 ms, 11.75 ms saved per skipped position); it
should be refit once the skip cost is measured.

Launch layouts. With `GLM_MTP_KSTOP=1` the launcher accepts only two layouts, both K3 with four slots and
FULL_DECODE_ONLY graphs at capture sizes `[1, 4, 16]` (14 descriptors, from the pinned graph manager: target
and first MTP pass M2/M3/M4 for one request, M4 (q2) and M6 (q3) for two, M16 for four; later passes M1 and M4):

| `RECIPE_MAX_MODEL_LEN` | KV pool | NCCL LL128 | Rank-0 warmed MemAvailable, predicted | Status |
|---|---|---|---|---|
| `32768` (default) | 544 blocks (`--kv-cache-memory-bytes 1685372928`) | on or off | 8.78 [8.68, 8.85] GiB | memory-admitted by the campaign sim; never booted |
| `44224` | 693 blocks (2 GiB) | off (`RECIPE_NCCL_NO_LL128=1` required) | 8.70 [8.60, 8.77] GiB | unmeasured: holds only if NCCL without LL128 frees its estimated 0.44 GiB |

One 44,224-token request plus the K3 lookahead plus vLLM's null block fills the 693 blocks exactly; 44,288 needs
one more and is refused. The earlier capture list 1..16 at 693 blocks (28 descriptors, predicted 7.82 GiB, under
the launcher's 8 GiB floor) cannot be launched. Every c1 shape replays a graph. c2 and c3 pad q4 rectangles to
M16 and later passes to M4; uniform q2/q3 rectangles at c3/c4 run eager (`docs/results/kstop-memory-admission.json`).

Batches of more than one request. When the requests of a batch drafted different lengths, the target pass runs
eager: captured graphs carry no dead-row remap. Draft lengths differ in most c2-c4 cycles. With
`GLM_MTP_KSTOP_UNIFORM_BATCH=1` (default 1 in this release), every request of a draft batch with two or more requests drafts all
three tokens: there is no stop decision, decision guard or probability read in that cycle, and the next verify
rectangle is uniform q4 and replays the M16 graph. A lone request keeps the confidence stop. The choice depends
only on the number of requests, which every rank holds identically (bound by the per-cycle guard), and the flag
is part of the startup control agreement. With the flag off nothing changes. It is on by default because with it
off every c2-c4 cycle in which the requests stopped at different lengths runs the target pass eager, and that is
most of them. Whether K-stop at c2-c4 with this policy is faster than native K2 is unmeasured.

## Prefix caching

On in the native profile: `profiles/serve-args.json` carries `--enable-prefix-caching` in place of
`--no-enable-prefix-caching` (one token, the same flag and position as the APC deployment
`glm53full-apcq-20261002-121736-b5`); worker ranks keep `--headless` immediately before it. vLLM's automatic
prefix caching reuses full 64-token KV blocks of an identical prompt prefix from the same pinned pool; there is
no extra memory reservation. The DSpark K3 profile keeps prefix caching off. Requests that must be cold (the
release gate's prefill and teacher probes) carry a nonce or a fresh `cache_salt`.

## Optional switches (off by default)

MLA plan skip (`overlay/bringup/glm_skip_mla_plan.py`). Under the Triton MLA forward nothing runs the FlashInfer
SM90 plan, yet the stock builder plans on every build and reads exact positions with blocking `.cpu()` copies.
`GLM_SKIP_MLA_PLAN=1` returns the parent metadata without the plan. `GLM_SKIP_MLA_PLAN=ab` arms the same switch
off (or `GLM_SKIP_MLA_PLAN_AB_INIT=1` on) and adds two worker RPCs (`skip_mla_plan_set`, `skip_mla_plan_status`)
for in-boot A/B over `/collective_rpc`; it needs `VLLM_SERVER_DEV_MODE=1` on the private loopback API and is a
window mode, never a serving mode. These three keys are not in the profile: the launcher passes them to the
containers only when set to a value other than 0, refuses bad values, refuses the dev API without `ab`, and
requires `GLM_FULL_MLA=triton`. With K-stop on, K-stop carries its decision probabilities and guard flag in this
planner's host copy, so a build that would skip while K-stop has such a payload pending plans as stock instead
(`counts['kstop']` in the status); only builds without a K-stop payload skip.

Spec-sample (`GLM_SPEC_SAMPLE=1`, [docs/spec-sample](spec-sample/README.md)): probabilistic native MTP drafts for
T>0 with an independent residual noise stream. Boot-time; needs `GLM_MTP_KSTOP=0`.

Persistent compile cache (`RECIPE_PERSISTENT_CACHE=1`): see Transport and loading. Off.

Parameter hash (`GLM_PARAM_HASH=1`, `overlay/overlay/glm_param_hash.py`): a campaign tool that hashes the
final GPU weights per rank over worker RPCs for weight-equality windows. It is never set by a profile.

## Display-carveout KV hook (off)

`RECIPE_DISPRAM` (`.env.example`, default `0`) is the launcher hook for [dispram](https://github.com/kindlingai/kindling-spark-os/tree/main/dispram)
by kindlingai, which lends the GB10's unused display carveout to CUDA processes. The integration itself
(`scripts/dispram_recipe.py`, `scripts/dispram.sh`, `overlay/bringup/glm_dispram_kv.py`, `service/`) is not part
of this release; until it is added, `1`/`auto` and `require` are refused before anything starts. The hook calls
it in four places only: extra container mounts and environment, the lender check after JIT prep and before
compaction, the lease postcheck after `stop`, and `./start.sh dispram-setup`. The planned split once its TP4
window passes is 1.5 GiB of ordinary KV per rank plus the 2,046 MiB carveout; how that combines with the K-stop
KV layout is decided with the integration.

## Transport and loading

Both data rails use `rocep1s0f0,roceP2p1s0f0`, GID index 3, RoCE v2, `NCCL_NET=IB`, no NIC merging,
no cross-NIC, NVLS or cuMem allocations. Eligible collectives use RoCEnante; its size/type gates leave
other operations on NCCL. `--disable-custom-all-reduce` disables vLLM's separate local custom reducer.
NCCL buffer size is 1,048,576 bytes and the channel cap is eight. Each of the 96 NCCL connections per rank
holds about 6.19 MiB of pinned host memory, 4.69 MiB of it for the LL128 protocol; the staged
`RECIPE_NCCL_NO_LL128=1` sets `NCCL_PROTO=^LL128` so those buffers are not allocated (about 0.44 GiB per rank). The library is preloaded from
`/opt/nccl/libnccl.so.2.30.7`. The base transport's spin limit/cache defaults remain image settings.

The target fast loader uses 64 MiB slabs, 256 MiB ahead and four reader threads, drops consumed cache
pages and checks two tensors per shard against the stock mmap source. The historical `/draft/` low-memory override is disabled for native MTP. Loader bytes/order/dtypes/shapes are tested against safetensors on CPU.

Native MTP shard selection (`overlay/overlay/glm_mtp_select.py`). The checkpoint's one MTP layer
(`model.layers.78.*`, 2,343 tensors, 9.35 GiB) lives in shards 270-274 of 282. Both pinned consumers
classify each name with `get_spec_layer_idx_from_weight_name` before any side effect: `DeepSeekMTP.load_weights`
drops every other name (including `model.embed_tokens` and `lm_head`, which the proposer then takes from the
target), `DeepseekV2Model.load_weights` drops the MTP names. `GLM_MTP_ONLY_LOAD=1` reads only the MTP tensors
for the draft and `GLM_TARGET_SKIP_MTP=1` leaves them out of the target read; the draft previously re-read all
282 shards (365.4 GiB per rank) to keep them. The shard headers decide what is read, the index must agree with
them, every MTP tensor must arrive exactly once, and after the load every draft parameter whose kind the
checkpoint provides must be in the loaded set; any failure stops the boot. `GLM_MTP_ONLY_LOAD=audit` keeps
the full scan and runs the same checks. `tests/test_mtp_select.py` drives the image's own loader, iterator and
`DeepSeekMTP.load_weights`; on the real shards the weight-loader calls and bytes equal the stock full scan's.

The recipe starts with a fresh per-deployment JIT cache built from the pinned image before the model containers start: after creating `OVERLAY_REMOTE/cache` and before compaction and the 110 GiB preboot check, the launcher runs one `docker run --rm` per node in parallel (same image and container environment with `PYTHONPATH` empty so no overlay hook runs, `/cache` and `/opt/nccl` mounted as in serving, no GPU, no network, 16 GiB memory limit, 600 s in-container timeout) that builds FlashInfer's `sampling` and `batch_mla_attention_…_e4m3_…_ckv_512_kpe_64` modules. Any failure stops the launch before compaction. The serving process then runs the same ninja build, finds it up to date and compiles nothing; before this step the cold sampling build (three nvcc processes, about 2-2.7 GiB of host memory for about 37 s) ran right after graph capture and pushed rank 0 below the launcher's 8 GiB floor. Sources, flags and the resulting modules are the ones the server would build itself. E2b built separate immutable prepared-weight snapshots for sync/async native K2, bound to constructor configuration as well as source/image/runtime/CLI. The measured fleet uses those caches; the fresh-clone launcher uses the native loader and has not been re-timed for K2. HF/transformers run offline inside serving containers.

Opt-in persistent compile cache: `RECIPE_PERSISTENT_CACHE=1` (with `PERSISTENT_CACHE_DIR`, a node-local path
outside every release tree and runtime path). Before JIT prep, one `docker run --rm` per node (the image by its
local ID, no GPU, no network) computes the key from the image ID, the FlashInfer version, the CUDA arch lists,
the overlay tree hash and the cache layout, and copies a ready generation for exactly that key into the empty
deployment cache (`overlay/tools/glm_persistent_cache.py`, as root inside the container). Another key, a missing
`READY` or a `key.json` that differs is a miss and the boot builds fresh. Each generation carries a manifest
(size and SHA-256 of every file, every symlink target) bound to `READY` by its SHA-256; the payload is verified
before the copy and the copied deployment cache again after it. A changed, missing or extra file, or a
manifest/`READY` that does not verify, stops the launch (helper exit 3, partial copy removed) until the
generation is removed. After admission the save helpers run in background threads beside the watchdog,
which keeps sampling memory, swap, errors and progress; they publish the deployment cache (without the
prefill control file) as that key's generation by one rename, unless it exists. Helpers still running
after 630 s, or at any abort, are stopped with `docker stop` and publish nothing; a save failure is printed
and never stops serving. Root-owned cache files are never copied by the SSH user.

## Watchdog

The foreground serving process owns the four containers. While booting it polls the loopback `/health` every
second and samples all ranks every 5 seconds (the admission window is sampled at that rate); after admission
it samples every 10 seconds.
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

K-stop compatibility is opt-in: `GLM_INDEXER_SHORTCUT=1` captures separate short graphs
for exact c1/c4 q2/q3/q4 rectangles and draft q1. Mixed widths, prefill, padding
and long contexts retain the stock indexer. Draft eligibility reserves the full
Kmax3 cycle before computing and compacting the shared MTP top-k rows; native
steps 1+ reuse those rows unchanged. Defaults remain shortcut off. Extra graphs
need a new GPU memory admission receipt before serving qualification.

`GLM_SPEC_SAMPLE=1` also supports K-stop. The stop rule still reads max softmax
of raw draft logits, independent of temperature and the sampled token; the native
Gumbel sampler writes the logits cache used by standard rejection. Host UVA
sampling parameters/seeds and shortcut length bounds are included in the prepare
guard digest. The existing residual-noise correction is required. T=0 keeps
argmax sampling and the same confidence. CPU checks cannot qualify GPU distributions.
