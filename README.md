# GLM-5.3 on 4x NVIDIA DGX Spark (vLLM TP4, Int4-Int8Mix, native MTP K2)

Serve the **full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (743B total, about 40B active) on four
DGX Spark boxes (GB10, SM121, 128 GB unified memory each) behind a RoCE switch. One chat-completions
endpoint with tool calling and reasoning, 32,768-token context, up to four concurrent sequences.

- **Weights:** [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/tree/206507bbb047d8223964a0414cd83230c59428f9),
  revision `206507bb`, 282 safetensors shards; W4A16/W8A16 compressed tensors, Marlin MoE. No local requantization.
- **Decode:** Z.ai's native MTP layer from the target checkpoint, fixed K=2, with `GLM_MTP_FIX=1`,
  FP8 target/draft KV, block size 64 and async scheduling. Split32 Triton sparse MLA keeps NoPE and
  RoPE separate; RoCEnante collectives use both ConnectX-7 rails. No separate drafter is needed.
  While a context is at most 2048 tokens the DSA indexer's top-k keeps every token, so the
  short-context shortcut skips the indexer query and logits there (same selection and cache bytes).
  After the split MLA reduce, its consumed fp32 partials are discarded from L2 instead of being written
  back to DRAM (`GLM_DIRTY_L2=discard`; output bits unchanged).
- **Prefill:** 2048-token scheduling chunks with fixed 4096 constructor capacity; prefix caching off.
  FULL_DECODE_ONLY target graphs capture 1/3/6/12 rows (M3 at c1, M12 at c4); native MTP passes use M1/M4.
- **Alternative:** qualified Red Hat DSpark K3 with an independent BF16 draft SWA pool, synchronous
  scheduling and 4/8/12/16 target graph widths: `RECIPE_PROFILE=dspark-k3 ./start.sh serve`.

## Current results

The default profile (`profiles/current.env`), measured on 2026-10-02 from a fresh clone of this release's code
booted with `./start.sh serve` as documented below, on four DGX
Spark with the existing 2200 MHz GPU cap (the recipe does not change clocks). Every number below comes from that
one boot ([boot receipt](docs/results/readme-1002-boot.json)). Its RigMark rows replace those of the
release-gate boot earlier the same day (prose 24.9, code 34.4, structured 36.2; see [docs/history.md](docs/history.md)),
so that sparkDash and RigMark come from the same boot.

### sparkDash, thinking off

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 |
|---|---:|---:|---:|---:|
| prose | **30.0** | 42.4 (22.2) | 61.1 (15.9) | 58.9 (15.4) |
| code | 34.1 | 49.3 (24.7) | 65.0 (16.7) | 62.2 (16.4) |
| structured | 36.6 | 56.9 (28.5) | 91.8 (23.2) | 89.3 (23.1) |
| json | 35.5 | 48.7 (25.6) | 76.9 (19.6) | 67.5 (17.9) |

**Prefill, cold, tok/s**

| 4k | 8k | 16k | 32k |
|---:|---:|---:|---:|
| 1000 | 942 | 890 | 839 |

The 32k cell uses a 32,256-token prompt: the 32,768-token context cannot hold a 32,768-token prompt plus output.

### RigMark 1.0.0, thinking on, effort low

**Decode, c1, tok/s**

| workload | median | range |
|---|---:|---:|
| prose | **24.9** | 24.1-25.7 |
| code | 33.4 | 32.9-33.8 |
| structured | 35.4 | 34.6-36.6 |

**Concurrency, code workload, aggregate end-to-end tok/s (per-stream decode in brackets)**

| workload | C1 | C2 | C4 |
|---|---:|---:|---:|
| code | 30.6 (32.6) | 46.5 (25.5) | 68.4 (18.8) |

**Prefill 8K:** 951 tok/s cold (warm replay 843; prefix caching is off).

**How these were measured.** sparkDash 1.8.8 DecodeBench via [`bench/sparkdash.py`](bench/sparkdash.py) `full`:
256 new tokens, temperature 0, thinking off, idle endpoint; one discarded c1 warm-up per prompt type and one
discarded prose c4 warm-up; each cell is the median of 5 runs at c1, 3 at c2 and c4 and 2 at c8 (aggregate and
per-stream medians are taken separately); decode excludes prefill and first-token latency. At c > 1 sparkDash
sends the same prompt to every stream, and at c8 four streams queue behind the server's four slots. The prefill
table is sparkDash's prefill bench: prompt tokens divided by time to first token, one discarded 4k warm-up, median
of three rounds per length; prefix caching is off, so every prompt is cold. RigMark 1.0.0 (pinned
`d8353e93b274`): 4096-token budget, five decode runs per workload with all 15 completion gates passed (range =
slowest to fastest run); the concurrency rows are RigMark's separate 256-token code workload, three rounds each,
timed end to end, so they are not comparable with the sparkDash cells; prefill is effective input throughput to
the first token, median of three 8192-token prompts. Raw output:
[sparkDash](docs/results/readme-1002-sparkdash.jsonl), [RigMark](docs/results/readme-1002-rigmark.json).

The selectable Red Hat DSpark K3 profile (`RECIPE_PROFILE=dspark-k3`) was not re-measured for this release; its
last numbers and the comparison that made native MTP the default are in [docs/history.md](docs/history.md), as are
the earlier releases and same-boot A/B checks.

### Serving and quality

The default is native MTP K2 async with the short-context DSA shortcut and the dirty-L2 fix on `GLM-5.3`,
loopback port 8095, 32,768-token context and four slots. On the fresh-clone boot qeval scored **73/72/73**, mean
**72.67/75**: **PASS**, zero truncation and no new recurring failures ([quality receipt](docs/results/dirty-l2-qeval-admission.json)).
Health 200, an exact `OK` reply, four running ranks without OOM or restarts, the shortcut and the discard armed and
dispatched on all ranks, and all-rank MemAvailable at least 9.03 GiB for 60 s after warm-up (8.95 GiB minimum over
the whole gate) are in the [gate receipt](docs/results/dirty-l2-release-gate.json). The results boot above (same launch vector
after container-name and runtime-path normalisation; qeval not repeated) passed the same serving and arming checks,
held all ranks at or above 9.01 GiB for 60 s after warm-up and stayed at or above 8.65 GiB (rank 0) through the
benchmarks ([boot receipt](docs/results/readme-1002-boot.json)).

The operational screen uses the checker-matched mean threshold 70.553/75, zero truncation and no new
recurring failure among historically always-passing tasks; there is no per-run 71 veto. It is not a global
intelligence-equivalence proof. The ≥50 tok/s prose-c1 target and multi-day stability remain unqualified.
The fresh-clone launcher uses the native loader: health 200 came 467 s after launch (470 s on the results boot). Preflight hashes all 290
target files on every node (about 20 minutes) unless the node's record of its last full hash still matches every
file's size, times, inode and device and is less than seven days old (`VERIFY_WEIGHTS=full` always rehashes); the
2026-10-02 gate reused those records. The launcher starts from a fresh per-deployment cache built from the pinned
image before the model containers start: one short
`docker run --rm` per node (same image and environment, no GPU or network) compiles FlashInfer's sampling and
batch MLA modules into it, so no nvcc build runs inside the serving process after graph capture. It is not a
cache seed from an earlier boot and ships no binaries.

## What is in the stack

| Piece | Where | Purpose | Credit |
|---|---|---|---|
| Full-model sparse MLA | `overlay/bringup/glm_full_mla*.py`, `GLM_FULL_MLA=triton`, split K=32 through 36 rows | avoids the stock FP8/RoPE launch's 101,888-byte shared-memory request, 512 bytes above GB10's 99 KiB opt-in cap; keeps NoPE/RoPE separate | CosmicRaisins sm12x split-layout work; Matt Mastracci's sparse-MLA ideas; local implementation |
| Switched RoCE collectives | `Dockerfile.roce`, `roce/`, `GLM_ROCE_ALLREDUCE=1` | eligible TP collectives over RDMA on both rails; larger or unsupported operations use NCCL | b12x/RoCEnante, Luke Alonso, Jason Cook, tonyd2wild, rhys101 |
| Drained prefill-cap switch | `overlay/bringup/glm_prefill_switch.py`, `GLM_W2_PREFILL_CHUNK=2048` | initial 512 scheduler cap, then all-rank atomic 2048 control after steady idle admission; constructor capacity 4096 stays fixed | knapcio; vLLM scheduler contributors (Apache-2.0) |
| Native MTP K2 async | `profiles/serve-args.json`, `GLM_MTP_FIX=1` | checkpoint MTP, packed quant mapping, FP8 KV block64; two proposals | Z.ai, vLLM; keys/@u1tra_instinct (TensorFold results pointed us to MTP) |
| Short-context DSA shortcut | `overlay/bringup/glm_dsa_short.py`, `GLM_INDEXER_SHORTCUT=1` | for rows of at most 2048 tokens, top-k keeps every token, so the indexer query GEMM and logits are skipped; stock key/cache path and selection bytes unchanged; separate FULL graphs for M1/M3/M4/M12 | [Guess-Verify-Refine](https://arxiv.org/pdf/2604.22312) authors (all-selected case when the length is at most top-k); DeepSeek DSA; vLLM indexer/persistent_topk contributors; local implementation (knapcio) |
| Dirty-L2 fix | `overlay/bringup/glm_dirty_l2*.py`, `GLM_DIRTY_L2=discard` | the split32 MLA leaves 1 MiB of dead fp32 partials per row dirty in L2, and their write-back slowed the following INT8 o_proj by 13 µs (M3) to 38 µs (M12) per layer; the reduce now ends with a CTA barrier and `discard.global.L2` of the lines it consumed. Released partial kernel and reduce arithmetic, bit-identical output, no buffer or graph change; split 32 only, off in the DSpark profile | own diagnosis and fix (knapcio); NVIDIA PTX `discard.global.L2`; Triton contributors; split32 kernel lineage as above |
| Red Hat DSpark K3 alternative | `profiles/dspark-k3-args.json` | three proposals per verification step, fixed graph widths; adaptive verification off | Red Hat DSpark |
| Alternative draft SWA pool | `overlay/swa-pool/`, `GLM_DSA_SWA_POOL=1` for DSpark only | preserves target geometry and the drafter's 2048-token sliding window without folding draft KV into target pages | local allocator hooks on vLLM |
| NCCL memory settings | `NCCL_BUFFSIZE=1048576`, `NCCL_MAX_NCHANNELS=8` | 1 MiB buffers and at most eight channels keep transport shared memory within the host budget | NVIDIA NCCL |
| Fast loading | `overlay/overlay/glm_fast_load.py`, `overlay/bringup/glm_draft_lowmem.py` | target: 64 MiB slabs, 256 MiB read-ahead, four threads; draft: 16/32 MiB, one thread, pageable staging; consumed pages dropped | Willian-Zhang's GB10 loading report; local loader |
| Launcher JIT prep and hash records | `scripts/cluster.py`, `scripts/weights.py` | FlashInfer sampling and batch MLA modules are built into the fresh per-deployment cache before the model containers start; the full weight rehash is skipped while the per-host record of the last full hash still matches and is less than seven days old | FlashInfer JIT build; local launcher (knapcio) |

The runtime modules are env-gated and source-pinned. A drift or unsupported layout stops startup.
Default and alternative hooks are env-gated; no draft folding, adaptive K, draft conversion, certified head or profiling is enabled.
[Exact launch comparison](docs/dry-vs-best.md), [runtime details](docs/runtime.md).

## Requirements

- Four NVIDIA DGX Sparks, ARM64 Linux, GB10/SM121, 128 GB unified memory each. The measured fleet used
  kernel `6.17.0-1031-nvidia` and driver `580.173.02`; the pinned image contains CUDA 13.0.1 and
  vLLM `0.1.dev20051+g487ecf187` from tonyd2wild's v11 build.
- Docker with NVIDIA Container Toolkit, `linux/arm64` images, passwordless SSH from the workstation,
  and Python 3, bash and rsync on the workstation. CPU tests use uv and Python 3.12.
- A switched RoCE fabric configured with NVIDIA Sync, two reachable data rails and working management
  Ethernet/Wi-Fi or Tailscale. Set the exact interface, GID index, HCA names and IPs in `.env`.
- Local SSD space on **each** node: 405.28 GB target files (plus 2.50 GB drafter for DSpark), image/JIT caches and
  partial-download space. About 500 GB free is practical. Model weights are present in full on every rank.
- NCCL 2.30.7 at the same host path on all nodes, pinned by SHA256 in `.env`, and the
  `/usr/local/sbin/spark-compact-mem.sh` helper with noninteractive sudo. Installation: [docs/install.md](docs/install.md).

## Build

1. **Image**, on each idle node:
   ```bash
   docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
   ```
   Pinned tonyd2wild v11 base digest plus the Apache-2.0 b12x subset at `b58f34ea`; no full CUDA rebuild.
   The build checks 20 image-source pins and runs the RoCE CPU/image wiring tests.
2. **Weights**, CPU download on each node:
   ```bash
   scripts/download_weights.sh "$HOME/models"
   ```
   Both immutable revisions and every shipped file's SHA256 are in `manifests/`.
   The downloader resumes `.part` files and refuses mismatched existing files. [docs/weights.md](docs/weights.md).
3. **NCCL** 2.30.7 in `NCCL_HOST_DIR`; record its binary hash in `.env` on all four nodes.

## Quick start

```bash
cp .env.example .env             # hosts, data rails, image, model/NCCL paths; loads profiles/current.env
DRY=1 ./start.sh serve           # four JIT prep (docker run --rm) and four model docker run lines; no SSH
./start.sh preflight             # CPU checks, all weight hashes, image architecture, fabric, conflicts
./start.sh serve                 # JIT prep, compaction, workers first, then head; foreground watchdog
```

In a second terminal:

```bash
./start.sh status                # all ranks running/no OOM; health 200 and memory admission in serving terminal
./start.sh logs 0
ssh -L 18095:127.0.0.1:8095 Spark_01
```

The endpoint binds to loopback on the head. Use the tunnel or Tailscale with your own authenticated proxy.
Do not expose serving, SSH, Dashboard or Jupyter ports through the public router.

```bash
curl http://127.0.0.1:18095/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3", "messages": [{"role": "user", "content": "What is 19 + 23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

`reasoning_effort` is `low`, `high` or `max`; the default is `high`. Tools use `glm47`, reasoning uses `glm45`.
For a thinking-off request, use `chat_template_kwargs: {"enable_thinking": false}`.
`./start.sh stop` stops and preserves all four containers. Container names are unique per boot; a deployment
refuses occupied fleet locks, names, GPU workloads and existing runtime directories. After stopping, select a
fresh `OVERLAY_REMOTE` for another boot. The launcher does not stop unrelated containers or change clocks/sysctls.

## Benchmarks and gates

```bash
SPARKDASH_API=http://127.0.0.1:5555/api/sparks/spark-01/llm python3 bench/sparkdash.py full 32768   # on the head
python3 bench/qeval.py run mtp-k2-q1 --url http://127.0.0.1:18095/v1/chat/completions
GLM_IMAGE_SRC=/path/to/image-source tests/run_cpu_tests.sh
python3 scripts/compare_dry.py
```

Repeat qeval three times serially with unchanged task budgets. A change is released after a fresh-clone gate
and a speed comparison in the same boot where a runtime switch exists, otherwise on at least two boots per arm
([release rule](docs/validation.md#release-rule)). Its code checks execute returned Python;
use an isolated benchmark account. The CPU tests exercise the pinned image's real allocator, page boundaries,
rejection/null-slot masking, loader byte equality and four-process Gloo collective wiring. The image source
path is the directory containing `vllm/`; extraction and the tested retained path are in [docs/validation.md](docs/validation.md).
CPU passes and dry parity are packaging evidence; fleet qualification has a separate scope in that document.

## Known limits

- The operational qeval/long-context regression screen passes; the unstable A/A/long-logprob reference remains under investigation. No full-model
  intelligence-equivalence or multi-day stability claim is made.
- Tight unified-memory headroom: 2 GiB target KV per rank, 32k context and four slots. Four simultaneous
  32k contexts exceed the reported KV capacity. The watchdog stops for low memory, sustained swap growth,
  runtime errors, stalled GPU progress or boot timeout; the capture hook requires 10 GiB available.
- Benchmarks use an idle endpoint, short public prompts and the named fixed-K configurations. Native images and full-context mixed
  traffic have not been qualified by this offline package.
- Target and RedHat DSpark weights use the **GLM-5.3 licence**, separately from the recipe's MIT/Apache licences.
  **incoai DFlash2 is CC BY-NC-ND 4.0 and is not used by default or distributed here.**
- The 2026-10-02 fresh-clone gate used the existing per-node builds of the pinned `Dockerfile.roce` image;
  the image was not rebuilt for this release. RoCEnante failures stop the job; they do not silently switch a
  poisoned transport to a different implementation.

## More documentation

- [docs/install.md](docs/install.md): prerequisites, image/NCCL installation and launch ownership
- [docs/weights.md](docs/weights.md): immutable downloads, manifests and verification
- [docs/runtime.md](docs/runtime.md): used modules, loading order and transport settings
- [docs/validation.md](docs/validation.md): CPU evidence and fleet validation scope
- [docs/dry-vs-best.md](docs/dry-vs-best.md): exact four-rank env/argument comparison
- [docs/results/](docs/results/): current measurement and packaging receipts
- [docs/files.md](docs/files.md): complete source-file inventory

See [CREDITS.md](CREDITS.md) for authors and source references, and [NOTICE](NOTICE) with [LICENSES/](LICENSES/)
for licence boundaries. The recipe is MIT; third-party code and model weights retain their own terms.
