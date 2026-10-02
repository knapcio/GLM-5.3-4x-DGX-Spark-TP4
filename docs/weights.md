# Weights and immutable manifests

Every node stores a full target checkpoint at identical absolute paths. Native MTP is included in
that checkpoint. The unmodified Red Hat drafter is required only for the DSpark K3 alternative.
TP4 shards the loaded model, not the files on disk. No weights are included in this repository.

| checkpoint | revision | safetensors files | total downloaded files |
|---|---|---:|---:|
| Tech2wild/GLM-5.3-Int4-Int8Mix | `206507bbb047d8223964a0414cd83230c59428f9` | 282 | 290 |
| RedHatAI/GLM-5.3-speculator.dspark | `b374b95663447ea0e935151be4f3d6666e36e6d7` | 1 | 6 |

`manifests/target.json` and `drafter.json` record full revisions, file names, byte sizes and SHA256.
The companion `.sha256` files use the standard sha256sum format. The target is about 405.28 GB and
the drafter about 2.50 GB. `.gitattributes` and the local backup template are not serving assets and are excluded.

```bash
scripts/download_weights.sh "$HOME/models"
python3 scripts/weights.py verify target "$HOME/models/Tech2wild/GLM-5.3-Int4-Int8Mix"
# Only for RECIPE_PROFILE=dspark-k3:
python3 scripts/weights.py download drafter "$HOME/models/RedHatAI/GLM-5.3-speculator.dspark"
python3 scripts/weights.py verify drafter "$HOME/models/RedHatAI/GLM-5.3-speculator.dspark"
```

The downloader uses public immutable resolve URLs, streams to `.part`, resumes with Range requests,
checks size/SHA256 before rename and refuses a mismatched existing final file. It does not need a token
or import a GPU framework. Large-file hashes come from the saved immutable checkpoint manifests;
small-file hashes were completed from read-only deployed-file reads. The target's large-file manifest
was verified on all four ranks in Boot B. This packaging job did not rehash 1.6 TB of target weights.

`./start.sh preflight` and `serve` hash every file on the first run on each host and keep a receipt in
`~/.cache/glm53-full-verify/`, keyed by the manifest and directory, with each file's size, mtime, ctime,
inode and device. Later runs skip hashing only if all of these are unchanged; any change rehashes every
file. A receipt older than seven days is ignored, so every host rehashes everything at least once a week.
`VERIFY_WEIGHTS=full ./start.sh serve` always hashes everything and does not use the receipt.

The pinned target template already supports `enable_thinking=false` by appending an empty think block
at generation time; no template patch is needed. Its SHA256 is
`9687489b1c7eddac80c78c7d57a11e791601013c46dfd7202241fed85e340a74`, matching the Boot B saved template.
The target config and drafter config were checked against their pinned public resolve URLs.

The target and RedHat drafter use the GLM-5.3 model licence; see their pinned model cards and the
[target licence](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/blob/206507bbb047d8223964a0414cd83230c59428f9/LICENSE).
The recipe's code licence does not grant rights to model weights. No drafter fine-tune, head conversion,
Markov modification or quantization is applied. Incoai DFlash2 is CC BY-NC-ND 4.0 and is not downloaded
or used by this recipe. CPU-only allocator fixtures exercise alternate synthetic layouts; they do not
install alternative models or select alternative serving profiles.
