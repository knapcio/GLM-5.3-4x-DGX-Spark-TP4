# Installation

The supported topology is four DGX Sparks behind a RoCE switch, with NVIDIA Sync managing the data rails.
Management SSH remains on LAN/Wi-Fi or Tailscale. Configure the fabric through Sync before using this recipe;
the launcher consumes existing addresses and does not write network configuration.

Copy this repository to each node for the image build and CPU downloads. Docker/NVIDIA Container Toolkit
must support `--gpus all` and ARM64. Build on an idle fleet, never while another owner is serving or measuring.

```bash
docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
docker image inspect glm53-roce:v11-b58f34ea --format '{{.Architecture}} {{.Id}}'
scripts/download_weights.sh "$HOME/models"
```

The base is `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
It contains the v11 vLLM `487ecf187` build and its model support. The final stage adds only the pinned b12x
transport subset, the v11 import shim and a CPU-built libibverbs proxy. Image source checks refuse all 20
pinned files if their bytes differ. Per-node Docker image IDs can differ after local builds.

## NCCL

The measured configuration preloads NCCL 2.30.7 from a host mount. A source build on each Spark is:

```bash
git clone https://github.com/NVIDIA/nccl.git nccl-2.30.7-src
cd nccl-2.30.7-src
git checkout 73cf112295c33aee2b895f329f592f2a9b4b0f97
make -j4 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
mkdir -p "$HOME/nccl-2.30.7"
cp -a build/lib/. "$HOME/nccl-2.30.7/"
sha256sum "$HOME/nccl-2.30.7/libnccl.so.2.30.7"
```

That commit is NVIDIA's `v2.30.7-1` tag. Toolchains can produce different binary hashes. Pin the installed
binary in `.env` as `NCCL_SHA256`: one hash when all four hosts have identical bytes, otherwise four
comma-separated hashes in `HOSTS` order. The example values are the libraries read from the measured fleet,
whose second host has a separately built 2.30.7 binary; they are not a claim that every source build produces them. The precise flags of that
historical binary build are unknown. NCCL source and binaries retain NVIDIA's own licence.

## Compaction and ownership

The launcher requires `/usr/local/sbin/spark-compact-mem.sh`, installed by the administrator. Its minimal
operation is `sync`, then `echo 1 > /proc/sys/vm/compact_memory`; it does not drop caches or persist tuning.
Allow noninteractive sudo for this exact helper, rather than putting passwords in `.env`. The helper must
return success; compaction failures stop launch. No clock command or persistent sysctl tuning is included.

From the workstation, set `.env` hostnames, IPs, paths and fabric names. All four target/draft directories
must be identical to their manifests. `./start.sh preflight` hashes every file, checks ARM64 images and
NCCL bytes, requires at least 110 GiB MemAvailable per node, verifies the primary data address, refuses
partial downloads, GPU containers and existing runtime directories, and checks the compaction sudo rule.

`serve` acquires `$HOME/fleet_busy` on rank 0 with noclobber before staging. It never steals a lock.
It saves the exact ownership token locally, copies the runtime to a fresh `OVERLAY_REMOTE`, creates a
private cache, compacts all nodes and launches workers 3/2/1 before the head. It keeps the lock while the
foreground watchdog is active. `stop` targets only the recorded four container names; it releases the lock
only after all four are verified stopped and only if the lock contents match its token. An unreachable node
leaves ownership retained for recovery through management SSH or a local console.

Container names include time and a random suffix. Existing names are never reused. Containers are stopped
and preserved. Select another fresh runtime path after stopping; keep old caches and logs for diagnosis.
The recipe makes no automatic model restart, power-cycle or unrelated-container stop.

The API is loopback port 8095 on the head. For remote use, an SSH tunnel is sufficient:

```bash
ssh -L 18095:127.0.0.1:8095 Spark_01
```

Image build and fresh runtime qualification were not executed during the offline packaging job.
