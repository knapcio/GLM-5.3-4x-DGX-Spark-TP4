# Changelog

## 2026-10-03 — stack-1003

Primary release gate: PASS. Measurements come from W4 v2 primary, inference
source `0f24383`, pad hygiene on and spec-sample off. An independent confirmation boot is supplemental.

- Full GLM-5.3 native MTP with confidence-stop K1–3 at c1 and uniform K2 batches.
- Composed short-context DSA and K-stop integration; APC on.
- Display-carveout KV through pinned, unmodified kindling dispramd; required
  release layout and persistent service/recovery instructions.
- Include the qualified Apache-2.0 copy-guard source with a pinned ARM64 build
  command and CPU interposition test.
- Fast load and MTP shard selection; boot approximately 285 s.
- c2 capture-layout reuse; descriptor-scoped padding hygiene v2 on, omitting
  the remap from exact c1 graphs; experimental spec-sample off.
- sparkDash-first release tables, separate cold/warm prefill, aggregate concurrency
  with per-stream brackets and decode at 0/16K/30K/60K preceding context.
- qeval receipt fields and the long64k-v2 probe with its paired-repeat PASS rule.
- Attribution and licence boundaries refreshed without dropping existing credits.

Earlier release entries and their measurements are archived in
[measurement history](docs/history.md#archived-changelogmd-before-stack-1003).
