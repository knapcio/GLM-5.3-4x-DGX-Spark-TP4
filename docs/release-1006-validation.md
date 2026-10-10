# Offline release validation — 2026-10-06 Europe/Warsaw

Historical CPU validation below preceded the completed 262144 fleet gate. Current performance, capacity and floors
are recorded in [GATE-RESULT](../GATE-RESULT.md) and the [portable summary](results/release-1006-summary.json).

Release from cand2 `d048bd5`; merge parents include preflight `2b4699d` and coalesced fix `28af4f9`. All commits use knapcio.

Mac: every `tests/test_*.py`, source pins and RoCE CPU suite: **38 suite entries green**, 464 unittest cases, 20 platform skips (443 passes in the first complete run plus one added guard case). Distributed K-stop/Gloo and RoCE checks passed. Fast loader: 324 tensors + wiring; coalesced: 3142 hashes per rank equal, x4; NVFP4 composition: six toy-sidecar hashes equal. Recent opt-in writer/readers still bit-exact against saved BF16 rows.

ARM64 pinned image `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`: corresponding 38 suites green after harness corrections, 464 unittest cases /5 explicit skips, plus three real vLLM CPU loader contract tests. Runtime image lacks git; clone boundary passed on Mac. Corrected image environment used a writable ephemeral copy, complete saved sim fixtures, two CPUs, 8 GiB cap with no extra swap, no visible GPU and network none. The initial 3 GiB cap killed four-worker checks; its failures are retained in private receipts and were not mistaken for implementation verdicts.

The inherited CPU copy guard also compiled and passed four detected write-range hits /six forwarded calls. No new or changed GPU kernel, CUDA execution, fleet transport, fleet boot or serving action ran.

Base 23 and compatibility seven source pins, NVFP4 source pins, all Python AST, shell syntax and whitespace checks pass. Fresh-default DRY checks four 165312/2622-block ranks, all NVFP4 groups, fast loader, no recent bank or dev API; explicit WINDOW=2048/AB=1/INIT=0 opt-in remains accepted. Cached-header diagnostic missing-cache, timeout and malformed-result paths warn without blocking. Guard deadline stops only its owned deployment; verified handoff preserves it.

Reproduce all Mac suites with an already installed CPU Python/environment and saved fixtures (no dependency/network installation):

```bash
FP4_DUMPS="$SAVED_FP4_PROBE" python3 -B tests/run_integ_mac.py --source "$SAVED_IMAGE_SOURCE" --sim-dir "$SAVED_SIM_DIR" --out "$NEW_CPU_RECEIPTS"
```

No throughput or numeric GPU claim is added by these checks. At this offline checkpoint the release gate was pending. The later 262144 gate passed; the coalesced loader was refused at load and stays opt-in. [Memory decision](memory-1006.md), [export audit](publication-audit.md), [GATE](../GATE.md).
