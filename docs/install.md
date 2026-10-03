# stack-1003 installation

Prepare four DGX Sparks with Docker/NVIDIA Container Toolkit, an existing switched
RoCE fabric and complete full-model weights at identical operator-selected paths.
The administrator configures private addresses, rail identifiers, image IDs and
paths in `.env`; published examples contain template fields only.

```bash
docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
docker image inspect glm53-roce:v11-b58f34ea --format '{{.Architecture}} {{.Id}}'
scripts/download_weights.sh "$MODEL_ROOT"
```

The base image and b12x subset remain digest/source pinned. Model revisions,
file sizes and hashes are in `manifests/`; [weights](weights.md) describes the
resumable download and verification procedure. Install the pinned NCCL library,
record its binary hash privately and provide the administrator's compaction helper
at `/usr/local/sbin/spark-compact-mem.sh`. Keep the existing ownership/watchdog rules.

## Pinned, unmodified dispramd

The external lender is kindling's
[dispramd at the pinned revision](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram).
It is an AGPL-3.0 tool fetched and built separately; no daemon source is vendored here.
The wrapper pins commit `5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e`, hashes the
source and resulting library, and preserves upstream licence/bundling notices.
NVIDIA headers are separately pinned to `20e4e6e19cc26ba47b5cbe23130a396be100c427`.
The script expects its pinned driver/toolchain; a hash mismatch is a refusal.

On each idle node, install the wrapper in the generic service directory and
fetch/build the upstream tool unchanged. These commands are operator setup,
not actions performed by the offline docs preparation.

```bash
sudo install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" /srv/glm-dispram
sudo install -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
  scripts/dispram.sh /srv/glm-dispram/dispram.sh
export DISPRAM_HOME=/srv/glm-dispram
/srv/glm-dispram/dispram.sh fetch
/srv/glm-dispram/dispram.sh build
```

Run fetch/build as the chosen service user, with Docker access. Keep the standard
`/srv/glm-dispram` location consistent across the launcher, lender and lock helper.
For another location, render all three configurations consistently before install.
The local wrapper is separate from the external AGPL implementation.

## Start and persist the lender

Prefer the included persistent units: render the service user in
`service/dispramd.service` and `service/sudoers-dispram`, review them, then follow
[service setup](../service/README.md). The DRM lock is ordered before Docker;
the daemon runs after Docker and the lock. No borrower may start before preborrow.

For a manual session on an idle node:

```bash
/srv/glm-dispram/dispram.sh drm-lock
/srv/glm-dispram/dispram.sh start
/srv/glm-dispram/dispram.sh watch
/srv/glm-dispram/dispram.sh preborrow
```

The DRM-lock command uses sudo authentication on stdin. Use the local
administrator's normal authentication; store no credential in this repository.
`start` refuses GPU contexts, borrowers, DRM users or unreadable checks.
The daemon is never automatically restarted. [Recovery](../service/README.md#recovery)
requires borrower shutdown and verified release before stopping the lender.

## Select the release layout

Build the included ARM64 copy guard with the locally cached pinned runtime image:

```bash
bash overlay/guard/build_guard.sh
```

This runs the CPU interposition test, builds
`overlay/guard/libdispram_copy_guard.so`, and records source/binary hashes in
`overlay/guard/SHA256SUMS`. The source is the same guard used in the fleet gate.
Without the built library, the launcher refuses dispram. The guard catches the
dynamic CUDA runtime copy and memset paths; its source documents the interception
limits. GPU positive-control evidence remains in the dispram qualification.

Configure the private `.env` values shown in [runtime](runtime.md), including
`RECIPE_DISPRAM=require`, `RECIPE_MAX_MODEL_LEN=66112`, short DSA on,
uniform K2, c2 reuse and descriptor-scoped pad hygiene v2 on.
The example selects these switches; the primary v2 gate passed.
Keep spec-sample off. Then perform preflight, guarded boot and the
[release gate](validation-release-RUN.md) before publishing filled measurements.

W4 used the narrowly scoped `DISPRAM_ALLOW_RM_ALLOC_OOM=1` lender setting
described in [runtime](runtime.md#display-carveout-kv). Use it consistently in
the private `.env` and manual `watch`, `preborrow`, `postcheck` and `verified-free`
checks. It does not suppress unrelated GPU faults.
