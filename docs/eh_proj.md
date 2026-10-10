# FP8 draft eh_proj

The native GLM-5.3 MTP layer combines embedding and hidden-state feedback with a replicated,
bias-free 6144x12288 `eh_proj`. This recipe replaces only that draft projection with FP8 W8A16
Marlin; the target's weights, BF16 LM head and verification path stay unchanged.
The implementation is ported from the `81e8863` serving code, alongside deterministic alignment
from the `6da108c` lineage. Defaults are `GLM_DRAFT_EHPROJ=fp8` and `GLM_DRAFT_EHPROJ_INIT=1`.

Per-output-channel BF16 scales use the E4M3 range of 448. The bank is 75,509,952 bytes per rank
at 48 SMs, replacing a 150,994,944-byte BF16 source (75,484,992 bytes saved).
The original weight is not retained. Only its local checkpoint identity and hash are saved for rollback.
The projection is replicated, not tensor-parallel sharded.

Paired c1 (`c1pair`) measurements against det-align showed **+3.6-3.9 % prose** and
**+5.0-5.4 % code** throughput. The qeval panel was **within calibrated noise**.
These observations supported the lever. The exact combined gen-12 configuration subsequently passed the
owner-approved reduced release gate: sparkDash prose/code c1 39.38/45.74 tok/s, combined INIT ready on all ranks,
and 30/30 T=0 outputs byte/token-identical to stack-2. The gate carried the preregistered 116-item quality panel;
qeval x3 was not rerun and is not claimed as a fresh pass. See the README and `GATE-RESULT.md` for scope.

The head's inner INIT qualification runs on the attached FP8 bank before readiness: write coverage,
exact draft tokens, bounded finite float drift, and a bit-exact target fixed-hidden-state head check.
All ranks vote on each boundary. A refusal drains draft graphs, turns the head off, reloads only the
replicated BF16 projection from the local safetensors checkpoint, checks host/device hashes and memory,
and recaptures/qualifies native OFF graphs. Failed rollback stops startup. A successful fallback also
fails release acceptance: every rank must report both the draft head and eh_proj ready.

To disable the projection at boot, set both keys in `.env`; they must agree. Runtime projection
switching is not supported. Row selection remains default off and is excluded from the release gate.

```bash
GLM_DRAFT_EHPROJ=0
GLM_DRAFT_EHPROJ_INIT=0
```

Source pins cover the native model runner, Marlin FP8 utility and checkpoint loader.
`tests/test_draft_ehproj.py`, `tests/test_draft_ehproj_rollback.py` and the native preparation/import
checks cover accounting, isolation, all-rank refusal and rollback. CUDA execution and the full release
gate require the pinned fleet image; CPU tests do not qualify GPU arithmetic.

Credit: **knapcio** for the local adapter, qualification, rollback and measurements;
**vLLM FP8 Marlin contributors**, **NVIDIA FP8** and **PyTorch contributors** for the underlying implementation.
Local runtime files retain Apache-2.0 SPDX headers. [CREDITS](../CREDITS.md), [NOTICE](../NOTICE).
