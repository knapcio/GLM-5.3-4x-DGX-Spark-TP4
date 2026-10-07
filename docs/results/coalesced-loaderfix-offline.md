# Coalesced loader memory guard fix — 2026-10-06

Mac-only fix from `44cf298`, branch `perf/coalesced-loader`, identity knapcio.
The 13:59 fleet failure is recorded in the workspace NIGHT_LOG; this task
did not contact the fleet or perform another boot.

The relative guard counts two staging/pinned tiles once, live owning batches
and their bounded allocator high-water mark. Native destination writes earn
unique byte-range credit after TP/EP slicing, rather than charging destination
commit to loader pressure. The separate configurable emergency floor defaults
to 3 GiB; the relative margin defaults to 1 GiB. Unknown destination storage
earns zero credit. No KV capacity or later-phase guard changes.

All ranks choose ordinary-fast fallback together before the first placement
handoff; fallback remains active for the rest of that model load. Later
exceptions raise and publish a shared abort. The independent TCPStore channel
uses bounded RPC/rendezvous waits and a 30 s teardown grace plus 5 s kill grace
for a peer stuck in native code. Reader/upload teardown remains fenced.

| Mac check | Result |
|---|---|
| Coalesced memory/hash/lifetime/fallback suite | 20/20 |
| Rank failure/fallback suite | 12/12, including real four-client loopback TCPStore |
| Launcher | 58/58 |
| Recipe / release gates | 24/24 + 18/18 |
| MTP selection | 12 pass, 1 real-shard input skip |
| Parameter hashes | 15 pass, 2 platform/worker skips |
| Total unittest cases | 162 run, 159 pass, 3 explicit skips |
| Synthetic per-rank hashes | 3142 tensors x4 EQUAL |
| Existing fast loader | 324 tensors byte-EQUAL + wiring |
| NVFP4 sidecar composition | 6 hashes EQUAL |
| Image-source pins / AST / shell / whitespace | 23+7 / PASS / PASS / PASS |

The simulation starts at 18 GiB, reproduces the first 8 GiB destination commit
dropping availability through 10 GiB, and credits 95 GiB total destination
writes with 640 MiB bounded staging/owning storage and a 128 MiB next request.
Later commits reclaim other resident memory; availability bottoms at
5.375 GiB rather than an impossible negative value. External 2 GiB loss trips
the relative guard early; availability below 3 GiB trips even with 95 GiB
credit. Hash checks run on cached Mac PyTorch 2.14.1; no GPU speed claim.

One fixed-loader boot, phase breakdown, compatible normal-loader hashes for
every rank and sampled per-rank MemAvailable minima remain pending:
`docs/coalesced-loader-fleet-plan.md`. A fast-fallback boot does not qualify
coalesced performance. Logs and the final commit/push receipt live in
`sparks/diagnostics/glm53-full-20261006-loaderfix/`.
