# Coalesced-loader offline receipt

2026-10-06, Europe/Warsaw. Worktree `/srv/projects/glm53-full-loader`,
branch `perf/coalesced-loader`, base `84a38e0`. Commit identity: knapcio.
The final commit ID is recorded in the requested Mac diagnostic receipt:
`sparks/diagnostics/glm53-full-20261006-loader/RESULT.md`.

Implemented an opt-in modified port of Allan Clark / ajclark's Apache-2.0
reader at `f0b64af5ac6028624e5ae255d998faa1e0a9324a`. The read-only fetched
source snapshot is `ext/ajclark-f0b64af`; verbatim license and NOTICE are
retained in `LICENSES/ajclark-*` and attribution added to CREDITS/NOTICE.
Shell DNS was unavailable; GitHub fetch supplied the pinned files. No SSH,
fleet endpoint, deployment, push, GPU execution or checkpoint mutation occurred.

`GLM_LOADER=coalesced` selects aligned concurrent reads into two reusable
staging tiles, asynchronous owning CUDA batches and native placement. It
preserves stock tensor order and filters MTP/EP tensors before read planning.
Defaults: batch128 MiB, threads16, direct1, owned4096 MiB. Pinned requests are
at most 256 MiB total; power-of-two allocation bins are charged in full and
page alignment is required. CPU staging adds at most 8 KiB alignment space.
Headers/tensor count are separately bounded. Linux allocation admission and
consumption checks enforce 10 GiB available memory; existing capture/dispram
guards and KV allocation remain in place. Temporary allocator caches are
released before KV profiling. Default launch vectors are unchanged.

Verification used the Mac's cached Python/PyTorch2.14.1 environment and the
saved release-dirtyl2 image source. Results:

| Check | Result |
|---|---|
| Coalesced byte/lifetime/failure suite | 12/12 pass |
| Per-rank synthetic SHA-256 comparisons | 3142 tensors x4, hash-EQUAL; exact order/dtype/shape, NaNs, empties, mixed dtypes, oversized tensor |
| Retention/cancellation/read gates | Backing-alias budget, consumer retention failure, early close, read errors, short reads, EOF, direct page alignment, pending tile fence, allocator-bin charge, source mutation, memory floor pass |
| Launcher suite | 58/58 pass |
| Recipe / release-gate suites | 24/24 and 18/18 pass |
| Existing MTP suite | 12 pass, 1 real-shard test skipped: no GLM_MTP_LAB input |
| Existing parameter-hash suite | 15 pass, 2 skips: Linux meminfo and installed worker attach unavailable on Mac |
| Total unittest cases | 142 run: 139 pass, 3 explicit platform/input skips |
| Existing fast-loader regression | All 324 fixture tensors byte-EQUAL; wiring pass |
| Actual NVFP4 source composition | 6 tensor hashes EQUAL across two toy matrices using `perf/nvfp4-attn` 3488797 converter/manifest/transform |
| Dry launch | Four ranks; sole candidate delta `GLM_LOADER=coalesced` |
| Source pins | 23+7 pass; unchanged |
| Additional checks | Startup selector/budget refusals exit78; host sampler parser; upstream hashes/read-only modes/verbatim attribution; AST and whitespace pass |

The complete test logs are in `docs/results/coalesced-loader-tests.txt`.
Full-checkpoint GPU weight hashes, Linux direct-I/O transport and boot speed
remain unmeasured. `tests/gpu/coalesced_transport.py` provides the future
delayed-upload/non-default-stream gate. The synthetic 3,142 cases do not
substitute for a real final-weight manifest on each fleet rank.

The real NVFP4 sidecar transform composes outside the iterator and bypasses
the draft. Its CPU reads and native repacking are outside this loader's
budgets. Merge the two branches' shared fast-loader wiring carefully, measure
sidecar peaks separately, and retain the memory guard. No sidecar/KV capacity
credit is assumed; keep sidecars off for the first loader-only boot.

**Forecast, not a Mac measurement:** main load **60 s**, planning range45–90 s,
against median188.72 s. Keeping all other saved phases fixed gives mean
health **153.68 s**, admission213.68 s, and **128.76 s saved per boot**.
The upside42 s main-load case gives135.68 s health. Ajclark's114.68 s total
is not transferred wholesale: his receipt has different warm-up and much
less available-memory headroom (about1.68 GB minimum).

Design comparison: `docs/coalesced-loader.md`. Exact one-candidate-boot plan:
`docs/coalesced-loader-fleet-plan.md`. It requires the unchanged launch/memory
contract, 100 ms all-rank host samples clipped to loader timestamps, boot phase
ledger, four final GPU weight manifests against a compatible qualified baseline,
and restoration of the recorded serving containers. An absent compatible
baseline makes the final hash verdict INCOMPLETE; no extra control boot is
implicitly authorized by this offline result.
