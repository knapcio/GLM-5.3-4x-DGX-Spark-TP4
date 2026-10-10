# Public-export audit

## October 10 full-model candidate

Scope: the tracked candidate tip based on dev release `36373c6`; private ancestor history is not rewritten.
Only the tip tree is intended for public export. Preparation is offline and performs no push.
`scripts/public_export_audit.py` reports zero unresolved findings. The broader location-only pattern
review found no local home paths, tailnet references, tool/session names or key material. Private receipt
roots were replaced with portable relative references; historical receipts are sanitized copies, not
exact original identity evidence.

Broad matches for generic experiment labels, coordination and ownership were reviewed: historical
provenance and an external test CLI flag; benchmark scheduling roles and vLLM coordinator namespaces;
licence terminology, lock/rank/tensor ownership and synthetic test fields. They contain no private
endpoints, credentials or tool attribution. Required dependency and credited organization names remain.
The detector's tool-name alternatives use equivalent escaped spellings and were checked against positive
fixtures; their detection is unchanged.

Offline evidence: [release-1010-offline.json](results/release-1010-offline.json), **670 cases: 644 passed,
26 skipped**, no unresolved test failure. The stock runner reached a native Gloo bind denied by the
sandbox. K-stop's 29 cases were checked with mocked one-rank collectives; five native RoCE Gloo cases
and one TCPStore case remain skipped. Other explicit skips require unavailable image/GPU/platform support.
Native frozen-config import ordering and full-size FP8 preparation also passed with explicit CUDA
library/repack substitutes. None of these CPU checks is GPU or live release qualification.
The serving-source hashes are in [release-1010-source.sha256](results/release-1010-source.sha256).

## Historical release 1006 public-export audit

Scope: tracked release-tip snapshot, not private ancestor commits. Original unsanitized receipts remain in the private local campaign archive. Historical exported receipts have portable substitutions and are not exact original identity evidence.

Initial tracked-tree scan found 188 home-path occurrences, 308 deployed-hostname occurrences, 98 IPv4 spellings, one tailnet name and one non-owner email (the coordinator destination). Seven broad name hits were software dependency/API namespaces, not authorship. No key material or session links were found.

Removed site home paths, deployed host/tailnet names and coordinator email; blanked site configuration in `.env.example`; replaced address fixtures with RFC 5737 TEST-NET values and made the stress sampler consume configured management hosts. Service templates require an explicitly rendered service user. `knapcio@gmail.com` is the only permitted email. Standard loopback and synthetic TEST-NET/invalid-address validation fixtures remain.

Reproduce after staging:

```bash
python3 scripts/public_export_audit.py --out /tmp/release-export-audit.json
git archive HEAD > /tmp/release-export.tar
```

## October 7 finalization audit

The 371-file would-be public snapshot was scanned again after the completed gate was added.
Initial findings: **two deployed-hostname references**, in `GATE-RESULT.md` lines 59 and 184.
Both were replaced with rank labels; no runtime source was changed by this finalization.
There were no site IPs, tailnet names, home paths, key material, non-owner emails or session links.

Final audit: **0 unresolved findings** across 371 text files. Allowed fixtures: 110 loopback/TEST-NET
spellings, two deliberately invalid IPv4 values, and three occurrences of `knapcio@gmail.com`.
A supplemental IPv6, broader hostname/home-path, key and session scan found eight regex artifacts:
four `threading.local()` namespace uses and four Python strided slices. Inspection confirmed that none
is an address or hostname. Software package names and credited organizations remain where needed for
working code, attribution and licence terms.

The prepared public snapshot is a separate local repository with one root commit and no remote.
It contains the sanitized tracked tree, with no private ancestor commits or site configuration.
The public README clone was read only for the comparison. Public publication remains owner-controlled;
no public push is part of this finalization.

## October 10 reduced-gate result finalization

All README result cells are generated offline by `scripts/release_table.py`. Fresh measurements and
carried diagnostics are distinguished; missing 250K and fresh qeval x3 measurements are explicit.
Receipt paths and SHA256s are in [the gate notes](release-1010-gate.md#recorded-reduced-gate-and-cell-provenance).
The source manifest changes only the offline formatter hash; serving source and launch selectors are unchanged.

`tests/run_cpu_tests.sh`: 533 cases, 513 passed, 20 skipped, exit 0. Release/recipe rerun: 120 passed, no skips or failures. Receipt integrity checks reject changed carried bytes, wrong phase identity, failed workload gates and wrong T0 report hashes.
The tracked export audit has zero findings. This finalization is local and unpushed.
