# FP4x KV kernel performance pre-review packet

Status: implementation staged. No GPU run has been performed. The Mac
Docker socket denied the pinned-image CPU/static attempts, so resource
receipts are unavailable. Complete those checks before submitting this
packet for pre-review or running a GPU fixture.

Target: pinned v11 ARM64 image digest
`4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`,
GB10 SM121, 48 SMs, 100 KiB physical shared/SM, 99 KiB opt-in/CTA,
65,536 registers/SM. No compiler/image substitution.

Inputs: `tests/fp4_kv_gb10_manifest.json` (93 explicit launch
specializations), `overlay/bringup/glm_fp4_kv_kernel.py`,
`glm_fp4_mla_kernel.py`, `glm_fp4_mla_split_kernel.py`, the unchanged
FP8 controls and dirty-L2 reduce. All launches use 4 warps, one stage;
writer debug assertions and exact divide are explicit. The manifest
contains actual absorbed-query strides and the release 2,048 sparse slots.

Critical cell: `_mla_partial`, H16/D512/R64, BN16, split32/SEG64,
small target/MTP widths1/3/6/12/16. A packed tile reconstructs BF16 once
for both QK and PV. At top2048 a row reads 729,088 useful bytes/layer
(356 payload bytes/slot, padding skipped), versus1,179,648 FP8 bytes.
Dots and the reduce are unchanged; dequant adds nibble selection and two
FP32 multiplies. The baseline is already register-heavy; spills or lower
occupancy can erase the traffic benefit. Expected speed is unknown.

Writer: one CTA/row owns both nibbles, scales and padding, with a scalar
negative/OOB slot guard. Reductions compute32 block maxima and one row
maximum; exact division matches the reference. RoPE retains E4M3.
Gather: BN16 direct paged read, bounded BF16 workspace; no full-cache
expansion. Scratch: one36-row partial bank plus metadata per device,
shared serially by target/MTP, warm allocation only. Dynamic shared comes
from the compiler and must be reported per specialization, not inferred
from tensor sizes. No separate decode dequant launch is added.

Attach final-package/image-bound `report.json`, metadata, PTX/cubin,
ptxas/cuobjdump/link logs and pinned CPU results. Compare FP8/FP4 partial
registers, stack/local/spills, dynamic/static shared and occupancy ceiling.
Require zero partial spills. Review coalescing, redundant scale loads,
BF16 MMA layouts, bank conflicts and writer assertion cost. Any tuning
invalidates the old receipt and requires recompilation. Static checks do
not establish correctness, capture replay or zero decode regression.

GPU fixture (after pre-review): `tests/gpu/fp4_kv_qualification.py`.
Full two-boot qualification and memory sizing: [docs/fp4-kv.md](docs/fp4-kv.md).
