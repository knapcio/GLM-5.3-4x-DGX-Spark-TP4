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

`ajclark-Apache-2.0.txt` and `ajclark-NOTICE.txt` retain the original terms and
attribution for the modified coalesced loader port from Allan Clark / ajclark
at `f0b64af5ac6028624e5ae255d998faa1e0a9324a`. Its source provenance and local
changes are recorded in `../docs/ajclark-loader-provenance.json` and
`../docs/coalesced-loader.md`.

Mia (MiaAI-Lab) is credited for the FP4 KV idea only. Its source licence is unverified; no code was copied.
External [kindling dispramd](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram)
is used unmodified under its AGPL-3.0 terms and is fetched separately. No lender code or binary is bundled.
NVFP4 sidecars are model derivatives generated locally from the pinned Tech2wild/tonyd2wild checkpoint; its Z.ai model licence continues to apply.

The local FP8 draft eh_proj, qualification/rollback and glue-lite adapters retain
Apache-2.0 SPDX headers. Their underlying vLLM, Marlin, NVIDIA and PyTorch
dependencies keep their own terms; these adapters vendor no external source.
