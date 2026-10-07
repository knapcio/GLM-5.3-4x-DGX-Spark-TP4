# Mac validation — 2026-10-06

Offline only. No SSH, fleet endpoint, push, GPU execution, image pull/build,
service installation or host-policy change. Existing cached Python3.12.14 /
PyTorch2.14.1 / NumPy / safetensors was used; nothing was downloaded.

Every `tests/test_*.py` CPU file and the RoCE CPU file was attempted. Final
unique unit evidence: **364 passed, 21 explicit skips**, including the12
socket-independent K-stop cases. Additionally13 RoCE pure checks passed.
Full K-stop process-group initialization aborted at `uv_bind: operation not
permitted`; five RoCE multiprocess cases failed because loopback sockets could
not resolve/bind in this sandbox. Those collective checks remain BLOCKED.
Colima context exists, but `docker ps -q` was denied access to its socket,
so the Linux pinned-image/Triton interpreter suite was not executed. The21
skips are Triton/vLLM/Linux or missing real-shard/probe inputs; source extracts
are not represented as an installed vLLM image.

The first admission test attempt lacked `FP4_SIM_DIR`. The corrected invocation
passed2/2 with the saved release-stack simulator. The failed first log remains
in the receipt directory. Changed integration suites were rerun after final
edits, not silently substituted for distributed or GPU validation.

| Check | Result |
|---|---|
| Base / compatibility / NVFP4 source pins | 23 /7 /13 PASS |
| New geometry / coordinator refusal / failure recovery tests | 7 PASS; half-GiB grid, exact blocks, numeric sim boundary/source binding, unknown lock, loader receipts, exclusive boot ledger, failed candidate restore and failed-stop refusal |
| KV headroom suite | 10 PASS; existing cache/loader exclusion and physical arena gates |
| Launcher suite | 58 PASS |
| NVFP4 suite | 25 PASS |
| Registered stock/fast/coalesced NVFP4 transform | 6 tensors hash-EQUAL; real converter/manifest/transform on two toy matrices; coalesced actually selected with GLM_FAST_LOAD unset; registration idempotent |
| Coalesced regression | 12 PASS; synthetic3142 tensors per rank ×4 hash-EQUAL, reader lifetime/budgets/errors/direct-range planning |
| Fast-loader regression | 324 fixture tensors byte-EQUAL |
| Full unit inventory | Raw suite/count/skip/failure ledger in `validation/SUMMARY.json` |
| Full-model GPU weights / CUDA transport / fleet boot and stress | Not executed; mandatory future driver gates |
| Fresh preparation | Control0 +4 half-GiB steps, each4 equal ranks;5 fresh original-B restore packages; all legacy admissions REFUSED |
| Syntax / whitespace | Python AST, shell syntax, `git diff --check` PASS |

Source fixture: `validation/image-source/` assembled from existing local
extracts. `validation/source-components.json` records exact source roots and
hashes; this is component-source provenance, not a complete container image
claim. `validation/nvfp4-pins.json` retains the additional pinned NVFP4 sources.

All receipts are below:
`/srv/campaign/diagnostics/glm53-full-20261006-integ`.
`prepared-mac-ready` is the final source-bound local preparation. Earlier
`prepared-mac-*` directories are intermediate validation receipts and must
not be used as the final coordinator package. No guard binary is committed.

Reproduce the Mac CPU inventory using an already-installed suitable Python:

```bash
python3 tests/run_integ_mac.py \
  --source /path/to/saved-pinned-source-parent \
  --sim-dir /path/to/day3/release-stack \
  --out /path/to/fresh-mac-validation
```

The runner preserves every suite's output and returns failure for blocked or
failed suites; it does not relabel socket failures as passed. Linux image/
interpreter checks use the existing `tests/run_fp4_kv_cpu.sh` on a Mac with
accessible Colima, `--pull never --network none`, not an online fallback.
The exact stress/restore procedure is in `PLAN.md`.
