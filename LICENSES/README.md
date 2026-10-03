# Licence boundaries

`../LICENSE` covers repository-authored recipe scripts and documentation under MIT.
`Apache-2.0.txt` covers files bearing that SPDX label. The unmodified b12x transport has its own
`../roce/b12x/LICENSE`, source headers and provenance. `MIT-sparkDash.txt` preserves the external
measurement tool's attribution; its collector implementation is not vendored.

Models are downloaded separately. The full target and RedHat DSpark use the GLM-5.3 licence:
[pinned target licence](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/blob/206507bbb047d8223964a0414cd83230c59428f9/LICENSE),
[pinned RedHat card](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark/blob/b374b95663447ea0e935151be4f3d6666e36e6d7/README.md).
Incoai DFlash2's CC BY-NC-ND licence is not replaced by this repository's code licence;
that checkpoint is not included or used by the launcher.

The referenced container, system libraries and GPU toolchain retain their own terms.
`../NOTICE` records attribution and the limits of the inherited source grants.

## External display-carveout tool and recipe references

[kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram)
is an external AGPL-3.0 tool used unmodified at the pinned commit. It is not vendored;
`scripts/dispram.sh` retains upstream `LICENSE`, `LICENSE-GPL` and
`BUNDLING-EXCEPTION` when staging the external installation. Consult those
pinned upstream terms for the daemon/library boundary. The local recipe licence
does not relicense them. The separately authored copy guard and its build/test
scripts in `overlay/guard/` carry Apache-2.0 SPDX headers; the licence text is
included in `Apache-2.0.txt`.

[Ash Hart / ashhart, TensorFold](https://github.com/ashhart/TensorFold): MIT;
[Jay Leaton, GLM TensorFold Spark recipe](https://github.com/jayleaton/glm53-tensorfold-spark):
Apache-2.0. They are ideas/measurement references; no source is vendored from
either, so no copied-source licence text is claimed here. Matt Mastracci's
contribution is ideas only. vLLM and Red Hat DSpark retain their existing
component/model terms. See `../CREDITS.md` for all preserved attribution.
