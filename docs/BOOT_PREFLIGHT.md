# CPU boot preflight

`scripts/boot_preflight.py` consumes the launcher's saved DRY Docker commands.
It runs CPU-only containers with `--pull never --network none`, preserving each
rank's feature env and vLLM argument vector. It never invokes SSH. The default
budget is 120 seconds across all ranks. The standalone diagnostic exits 1 on failure. The release launcher treats it as advisory: default ON only with an existing header cache; otherwise warning + skip. Failure, timeout, missing image or malformed receipt also warns and continues. It never substitutes for memory, payload-hash or fleet admission checks.
Reports include per-check errors, tracebacks, source hashes, rank, rendered image,
actual image selection, duration, and an explicit list of hardware/value fakes.

The pinned v11 base on Colima is:
`ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
Its Python sources are the serving image's pinned sources; the RoCE image adds
transport code that this CPU check does not exercise. Use `--image` to select
this already-installed base on the Mac. Without it, the rendered image is used.

## Real header cache

No complete real header cache was available in the local cand2 receipts on
2026-10-06. An operator can later run this standard-library-only exporter on a
node using **the exact model and sidecar paths from DRY**, then copy its output
to the Mac. The preflight itself does not contact a node:

```sh
python3 scripts/checkpoint_headers.py \
  --model /srv/glm/models/Tech2wild/GLM-5.3-Int4-Int8Mix \
  --sidecar /srv/glm/models/GLM-5.3-attn-nvfp4-87cf357 \
  --sidecar /srv/glm/models/GLM-5.3-more-nvfp4-cand2-20261006-w1435 \
  --out headers.json
```

The exporter reads only each shard's 8-byte length and JSON header, plus
`config.json`, `model.safetensors.index.json`, and sidecar `manifest.json`.
It does not read, map, or hash tensor payloads. Its no-clobber JSON cache retains
raw headers, header hashes, sizes, original documents and source paths.
Preflight validates index/header coverage, offsets/byte counts, pinned target
metadata hashes/shard sizes, DRY mount identity, sidecar source hashes, group
inventory, companion coverage and NVFP4 leaf shapes/dtypes. Payload SHA256,
scale values and weight_shape values cannot be verified from headers.

```sh
python3 scripts/boot_preflight.py --dry dry.txt --headers headers.json \
  --image ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 \
  --out boot-preflight.json
```

The real full GLM target and native MTP constructors select real quant methods
on meta. The actual target/draft `load_weights` implementations map source names
and call the real per-parameter loaders for expert, fused projection and TP
slices. A structural `copy_` checker rejects dtype casts, wrong shapes, unknown
names and unloaded parameters. Native intentionally omitted MTP/indexer tensors
and synthesized activation globals retain their serving-loader semantics.

The probe runs the real selected quant `process_weights_after_loading` hooks,
real MLA constructor and post-load shape assertions, FP4x/recent-KV startup
registration, adaptive scheduler source transformation/capacity assertions,
and the kstop runtime's exact method guard. It also executes the NVFP4 MTP apply
path with tiny CPU activations to verify pinned API reachability.

Explicit fakes cover SM121/TP4/EP1 metadata, GPU MLA dispatch, CUDA streams/events,
Marlin repacks/workspaces/GEMMs, KV-b dequantization, meta value reductions,
weight_shape payloads and CUDA control/remap operations. All fakes are local to
an ephemeral worker. No numeric correctness, CUDA execution, memory admission,
collective, quality or throughput claim follows from PASS.

`--config tests/fixtures/boot_preflight_glm53_config.json` permits construction
and historical diagnostics without a cache. Header checks still fail and the
command still exits 1. `--rank 0` is a diagnostic subset; default covers all DRY
ranks. The fixture config is byte-identical to the pinned target config.

## Drivers and regressions

`integ_1006.py run` defaults to advisory `--boot-preflight`; supply `--preflight-headers`
and optionally `--preflight-image`. The gate runs before fleet inspection,
lock acquisition or serving stop. `--no-boot-preflight` records an explicit skip.
The release launcher writes `logs/<container>/cpu-boot-preflight.json`. Its standalone diagnostic remains strict;
its launcher/integ advisory wrapper records missing-cache skips and failure warnings. The historical cand2 patch
is retained as provenance only; do not apply its blocking policy to this release.

```sh
python3 -m unittest discover -s tests -p test_boot_preflight.py
# CPU container only:
docker run --rm --pull never --network none -v "$PWD:/pkg:ro" \
  --entrypoint python3 IMAGE -B /pkg/tests/boot_preflight_image_contracts.py
python3 tests/run_boot_preflight_image.py --image IMAGE --out /mac/path/results
```

The historical runner uses local git archives with the same cand2 DRY/config:

| Source | Quant/MLA hooks | Boot-c guard | Boot-d API |
|---|---|---|---|
| `88f27fb` | PASS | exact modular Marlin rejection | missing `_custom_ops.silu_and_mul` |
| `613bf73` | PASS | PASS | missing `_custom_ops.silu_and_mul` |
| `9fa41c8` plus tooling | PASS | PASS | PASS |

Thus the boot-c check passes on `613bf73`, while preflight also correctly flags
that revision's independently discovered boot-d failure. With no real cache,
none of these diagnostic runs grants header admission.

Verified on 2026-10-06 with the local ARM64 Colima v11 base: 82 Mac unit tests
and 3 serving-image loader tests passed. Historical rank-0 probes took 14.07 s
and 13.164 s; the fixed candidate's four-rank construction/hooks probe took
53.192 s. The image matched all 23 release and 7 kstop compatibility source
pins. Full-cache processing time remains unmeasured until real headers arrive.

Launcher errors now retain the first traceback and first terminal exception
for each failing rank, including exited ranks. Complete logs are fetched only
when the normal 150-line telemetry tail indicates failure; steady polling keeps
its bounded tail. Caret/frame lines no longer become the failure summary.
