# NVFP4 sidecars: commands and identities

Run on each rank with the pinned, already installed ARM64 serving image ID. Original checkpoint mounts are
read-only. Outputs must be distinct new directories outside the checkpoint. No GPU is used. Existing verified
sidecars are reused for the release gate; conversion and the full QDQ audit are preparation outside its 75 minutes.

Converter source SHA256:

| File | SHA256 |
|---|---|
| `scripts/convert_attn_nvfp4.py` | `c093587051b02a95cb7f594273f3c2a511faf97a67881485ff7a5678d86139be` |
| `scripts/convert_more_nvfp4.py` | `db08d857849d35d7136ea9c61c2357ee48f6d543915be72e6365805e5ba697d7` |
| `scripts/nvfp4_canonical_hash.py` | `061a0cf976a77a663f4c406e8d7370ca56ca460017599175ed4d01274e9de334` |

```bash
# Set MODEL_DIR, ATTN_OUT, MORE_OUT, IMAGE_ID, CONVERT_TAG to local absolute paths/identities.
# PWD is this release checkout. Refuse an existing output at initial creation.
test ! -e "$ATTN_OUT" && test ! -e "$MORE_OUT"
mkdir "$ATTN_OUT" "$MORE_OUT"
for group in attn more; do
  if [ "$group" = attn ]; then
    output="$ATTN_OUT"; converter=convert_attn_nvfp4.py; args=()
  else
    output="$MORE_OUT"; converter=convert_more_nvfp4.py; args=(--groups shared,dense,mtp)
  fi
  time docker run --pull never --network none --platform linux/arm64 \
    --name "glm53-$CONVERT_TAG-$group-convert" --cpus 2 --memory 8g --memory-swap 8g \
    -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -v "$PWD:/code:ro" -v "$MODEL_DIR:/model:ro" -v "$output:/output" \
    --entrypoint bash "$IMAGE_ID" -c \
    'exec nice -n 19 ionice -c 3 python3 "/code/scripts/$1" /model /output --rows 64 "${@:2}"' \
    converter "$converter" "${args[@]}"
  (cd "$output" && sha256sum -c SHA256SUMS)
  sha256sum "$output/SHA256SUMS"
done
python3 scripts/nvfp4_canonical_hash.py "$ATTN_OUT"
time docker run --pull never --network none --platform linux/arm64 \
  --name "glm53-$CONVERT_TAG-more-audit" --cpus 2 --memory 8g --memory-swap 8g \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD:/code:ro" -v "$MODEL_DIR:/model:ro" -v "$MORE_OUT:/more:ro" -v "$RECEIPTS:/receipts" \
  --entrypoint bash "$IMAGE_ID" -c \
  'exec nice -n 19 ionice -c 3 python3 /code/scripts/nvfp4_more_audit.py /model /more --out /receipts/more-audit.json'
```

Require `complete=true`, attention 385 matrices, shared 225 + dense 6 + MTP 776 = **1007 matrices**, no indexer.
All output checksums must pass; the full extra-group CPU audit must be bit-exact and source hashes unchanged.
Record container exit code, wall time, cgroup peak/events, source index/config/shard hashes and all-rank identities.
Preserve containers and partial outputs; resume only the converter's validated same-input output, never boot partial data.

October 6 measured conversion: attention **7.5 min/rank**, 7,147,557,956 data bytes, 82 source shards;
extra groups **5.0 min/rank**, 7,397,610,724 bytes. Full extra-group QDQ audit passed 1007/1007 on all four ranks
(window interval about 13 minutes, including collection); exact standalone per-rank audit duration unknown.

Extra-group `SHA256SUMS` SHA256, byte-identical x4:
`c26bfb42c5743e025e702c7bb4600bb673bd350e9522fd24f96255bcf7ba10bb`.

Attention canonical aggregate, identical x4:
`af8fd1af03a4b89cddb2febf5f0d16b25de34bbd7c1edb0f3a602ddf39910f50`.
Source index SHA256: `c935795467984516f24a64d2dc2ce07bed74a8f83ada76950f28b847786eeddf`.
The original attention files predate the deterministic single-key metadata fix in `3488797`. Their tensors are
identical; metadata ordering produced four raw `SHA256SUMS` identities, in rank order:

```text
b829dc824e342c98517bc4eb2398122049ff852b8ce11a8b5db381fd53e9cc5f
047f6cba4a2de9912e200ae236683349dc217b379910f01f1c50477504a06d7a
97e7813d05a0391dcc9f78da0a4dfbf51d42ee218c3e8d4cc270ab8e97b30653
f5069b206d5402f42323327b660a226c42a7464fc8141823b4246925f8d97f87
```

Fresh deterministic conversion does not inherit an unmeasured raw or canonical checksum identity: the canonical
algorithm includes sorted header metadata, and the single-key metadata fix changes that header. Both new identities
are unknown until generated. Save them, verify every file, require equal identities across fresh replicas, and retain
the independent full audit before accepting a replacement. The hash-reuse gate uses the existing verified sidecars. Credit: Tech2wild/tonyd2wild checkpoint,
Z.ai model, NVIDIA Marlin/vLLM packing and quantization, knapcio converters; model licence remains applicable.
