# stack-1003 runtime

The public measurements describe the full target with native MTP, confidence
stopping, uniform K2 batches, short DSA, APC, dispram KV and c2 capture reuse.
`start.sh` loads the base profile, then `.env.example` and private `.env`. The
example selects the following stack-1003 layout; configure deployment paths and
addresses privately before booting.

```bash
GLM_MTP_FIX=1
GLM_MTP_KSTOP=1
GLM_MTP_KSTOP_UNIFORM_BATCH=k2
GLM_INDEXER_SHORTCUT=1
GLM_MTP_KSTOP_CAPTURE_LAYOUT=reuse
RECIPE_DISPRAM=require
RECIPE_MAX_MODEL_LEN=66112
DISPRAM_ALLOW_RM_ALLOC_OOM=1
GLM_SPEC_SAMPLE=0
GLM_PAD_HYGIENE=1
```

The candidate example selects `GLM_PAD_HYGIENE=1` with descriptor-scoped v2.
The primary v2 gate passed; receipts are linked from the README. The underlying base
profile retains the conservative flag-off layout for explicit alternatives.
The pinned native profile already enables prefix caching, async scheduling,
FP8 target/draft KV, block size 64, split-layout MLA, dirty-L2 discard,
switched RoCE, slab loading and MTP shard selection.

## Native MTP and capture layout

A lone request drafts at most three tokens, stopping when confidence falls:
the next proposal requires the configured cumulative draft probability threshold.
Multi-request batches draft two tokens per request. The verifier checks the
actual proposals; dead rows reuse live expert routes. Rank-replicated decisions
and the request-keyed guard must agree before dispatch.

The K2 capture list is `[1, 4, 12, 16]`. At c2, `reuse` dispatches the already
captured M6 q3 rectangle; c3 pads to M12 and c4 uses M12 exactly. The source
pins and compatible short-DSA transform retain c1 captures and the admitted
memory charge. `m12` is a diagnostic alternative, outside the published table.
Pad hygiene remaps graph-tail MoE routes to a live row; it leaves live routes intact.
V2 includes the remap only in graph descriptors that can receive padding and in
padded eager forwards. Exact c1 graphs omit it; a graph shared by c3 and c4
retains it. See [descriptor analysis](pad-hygiene-v2.md).

The short-DSA path skips indexer query/logits only where top-k selects all tokens.
Stock key/cache operations remain. Mixed widths, prefill, padding and longer
contexts retain the stock indexer. Record arming and dispatch on every rank.

## Display-carveout KV

`RECIPE_DISPRAM=require` selects the K-stop layout with `RECIPE_MAX_MODEL_LEN=66112`.
The source geometry is a 1 GiB ordinary KV head plus the display carveout per rank,
with one mapped KV range and per-layer storage boundaries. Cross-rank negotiation
must agree on size and successful mapping. The display memory is not extra weight
storage. Admitted context 66112, KV blocks 1039
and minimum available memory 8.96 GiB come from the primary v2 gate.

The launcher requires `overlay/guard/libdispram_copy_guard.so`. Build the included
Apache-2.0 source with `bash overlay/guard/build_guard.sh`; this runs the CPU
interposition test and records source/binary hashes in `overlay/guard/SHA256SUMS`.
The reproduced ARM64 binary matches the binary used for W4. This prerequisite is
separate from kindling's unmodified daemon. The launcher refuses a missing guard.

The W4 lender checks use `DISPRAM_ALLOW_RM_ALLOC_OOM=1`. This existing opt-in
counts and permits only the exact handled `_memdescAllocInternal` allocation
failure defined in `scripts/dispram.sh`; Xid, SMMU, IOMMU faults and other NVRM
failure lines still refuse lending. Apply the same setting to manual lender
checks and the launcher. W4 shutdown passed on all four ranks with the complete
carveout returned and no CUDA contexts or borrowers remaining.

The lender must precede every borrower. `auto` can fall back to plain KV in other
layouts; the release uses `require`, so unavailable or unsafe lending refuses boot.
The hook maps via NVIDIA VMM and zeroes with an SM kernel. Unsupported copy,
sleep, transfer/offload and layout operations refuse. See [setup](install.md) and
[persistent service/recovery](../service/README.md).

## Load, prefill and APC

Fast load uses anonymous/pinned slabs and read-ahead, discarding consumed file
pages. Native MTP shard selection reads only the checkpoint's MTP tensors; the
target skips them. Header/index/stream/parameter checks retain the weight bytes.
Boot through health: 283.8 s; admission: 343.9 s.

The drained prefill-cap switch preserves fixed constructor capacity. APC reuses
identical full prefix blocks inside the same pinned KV pool. Concurrent requests
share that pool; maximum context describes one admitted request, not four full
context requests simultaneously. Shared-prefix sweeps need warmth labels.

## Watchdog and optional levers

JIT preparation runs before model capture. Hash records reuse a recent full
verification only while file records match; `VERIFY_WEIGHTS=full` forces rehashing.
The foreground watchdog owns only its deployment and monitors rank exits, OOM,
progress, available memory, swap and numeric/runtime errors. Source admission
floors and dwell periods remain unchanged. Failed stops retain ownership locks.
The 180-second busy-without-progress timer resets while the GPUs are idle, so
a new request after a long quiet period receives a fresh stall allowance.
Continuous GPU activity without token/load progress still aborts after the
threshold.

`GLM_SPEC_SAMPLE=0`: experimental probabilistic drafts stay off. Its source
supports the composed K-stop path, but CPU checks do not qualify serving
sampling distributions. [Experiment](spec-sample/README.md).
Other opt-in levers and the Red Hat DSpark recipe have separate qualification;
the README tables report only the selected release layout.
