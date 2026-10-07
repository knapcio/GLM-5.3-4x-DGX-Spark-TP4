# cand-1006b Mac validation

Branch `perf/cand-1006b`, worktree `/srv/projects/glm53-full-cand2`.
Cached `origin/perf/integ-1006` base `fa89c1627dfdde4b9effd392b349e1544ffdb532`;
merge parents `e303cb3e65069cb4b2d8df9d51b969914726cedf` and
`f99b96cb1c4cb31914680b10a4c848a2dd1ae434` are both ancestors.

Fetch was attempted twice, but github.com DNS resolution failed. This checkout
includes fa89c16; later remote integration fixes are unverified and must be
reconciled before a fleet package is frozen. No SSH, fleet access, image job,
GPU execution or push occurred.

The sole merge conflict in start.sh was resolved by preserving both sets of
switch exports. Recent precision defaults to0; extra groups default toattn
when attention NVFP4 is selected; default weight selection remains int8.
Indexer remains opt-in. The launcher regression now covers recent reservation
combined with attn,shared,dense,mtp on all four ranks.

Mac CPU evidence: 387 unit passes plus13 RoCE pure passes,21 explicit
platform/input skips. Full K-stop Gloo initialization aborted at uv_bind
operation-not-permitted; five RoCE distributed cases also failed at the local
socket boundary. These are blocked, not passed. Full inventory was attempted.
Recent-suite invocation initially lacked FP4_DUMPS; a correction then lacked
GLM_IMAGE_SRC. Both logs remain; final run with both saved inputs passed12/12,
including18 groups/54 original BF16 rows reconstructed bit-exactly as FP8.
Launcher59/59 includes the combined all-rank subtest; NVFP4 suites38/38 pass.
Base23/compatibility7 and additional NVFP4 source22 pins verified; all Python
AST, start.sh shell syntax and whitespace checks pass. Real CUDA/Triton
compile/replay/model performance and quality remain future fleet checks.

A saved serving-shape reference .env/guard was applied identically to base and
candidate offline clones. Four-rank default DRY output is byte-identical after
normalizing only the clone path (identical CTN). Normalized SHA256:
`e95cd4c6964a61a03507cf6d9421ae6d66b3f6eda798694abc38dcc076de766e`.
Combined opt-in DRY preserves maxlen98176 and ordinary1GiB per rank, carries
all recent selectors and attn,shared,dense,mtp, and mounts the new sidecar.

Evidence root:
`/srv/campaign/diagnostics/glm53-full-20261006-cand2`.
`VERIFICATION.json`, `mac-tests/` and `dry-validation/` retain receipts.
The single coordinator-window driver plan is `PLAN.md` there: online bounded
CPU conversion on all4, one candidate boot, off/on/off/on, paired cycle and
serving comparisons, qevalx3/needles/RigMark, original-source serving restore.
It is a written plan, not an executed fleet driver.
