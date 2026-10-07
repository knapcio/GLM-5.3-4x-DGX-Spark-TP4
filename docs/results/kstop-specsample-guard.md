# Prepare digest threat model and final round-2 correction

Scope: isolated fix/kstop-specsample-guard; round 2 starts at d03fb47.
Offline Mac only; no SSH, fleet actions or pushes.
W3 B failed at its first real prepare on all TP4 ranks. The scheduler dump in
B/r0.log shows 28,315 prompt tokens, first chunk 2,048, T=0, seed=None.
A ran with spec-sample off. RESULT_FLEET1.md is absent locally; section B of
kstop-compat/RESULT.md and the supplied B/r0..r3.log paths are available.

Source evidence is the stock extraction at day3/release-dirtyl2/image-source
for image sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6.
All seven checked dependencies also match the apc-prep extraction; the stock
extraction was independently verified against compat_source_pins.json.

| Field bound at prepare (including the preceding trail) | TP invariance and relevance |
|---|---|
| Stage, round, epoch, mode, tau, runtime request IDs | Protocol state; must agree. Cold agreement binds control. Divergence is real, even during prefill. |
| Select: ordered scheduled request IDs and token counts | Same scheduler output reaches each TP worker; order affects row layout and must agree. Preserve insertion order for row order. |
| Select: speculative request IDs/token widths, proposal lengths | Must agree; widths determine physical dispatch. Dictionary traversal can theoretically differ without changing rows; schedule order is common in this pinned executor. Proposal list follows scheduled order. |
| Prepare: live request IDs, query widths, prefill flags, proposal mapping | Must agree; affects dead row remapping, graph shape and compaction. Padded rows are outside num_reqs. |
| Host request-state mapping idx_mapping_np[:num_reqs] | Worker-local storage addresses, **not TP-invariant**. Native warmup cleans up set(req_ids); finish_requests iterates that set, remove_request appends each freed index, and admission pops the last index. Different PYTHONHASHSEED values yield different slots for the same request. Use indices only to read each request's local values; never hash indices. Retain local bounds validation. Padding -1 must never index a stale mirror row. |
| Host UVA temperature | add_request writes every live slot on every last-PP TP worker; FP32 mirror is the source of truth. Used by drafts, including T=0 branch selection, so retain. float.hex is stable; signed zeros have identical sampler semantics and should canonicalize. Nonfinite/negative values must refuse. |
| Host UVA top_p/min_p/top_k | add_request writes live values identically; top_k normalizes <=0 to vocab_size. These affect target sampling, but stock Speculator._copy_request_inputs explicitly ignores them for draft sampling. Exclude from draft digest; they are not the cause of differing draft tokens. |
| Host UVA seeds | SamplingStates.add_request generates a local NumPy random int64 for seed=None, even T=0. Workers reset NumPy to the common model seed after warmup (gpu_worker.py:821); matching draw histories produce identical generated seeds regardless of slot. Local generation alone does not demonstrate different TP streams. T=0 Gumbel never loads seed or position, so omit its unused seed value. For T>0, hash the native seed by request ID and refuse a real mismatch collectively. Explicit seeds remain native. |
| CPU sequence length upper bounds | Same batch construction and scheduler counters; must agree when used for short-DSA eligibility. Native Gumbel reads actual positions+1, while autoregressive Speculator uses live upper bounds for max_seq_len and draft attention metadata. Retain the live prefix for spec-sample as well as shortcut; dropping it would weaken the guard. |
| Draft sampler and fp64 noise mode (cold control) | Configuration must agree; probabilistic vs greedy selects cache/noise, fp64 selects RNG math. Preserve cold binding. |
| Short DSA: length bounds, has_prefill, padded token count | Must agree; selects cache/graph route. Retain. |
| Exact FP32 confidence bits at decision guard | Not part of prepare's n=0 probability slots. Decision guards retain exact live bits; a mismatch is real. |
| Sticky validity and prior trail | Must agree; missing mirrors/proposals fail collectively. A past divergence cannot be erased by sanitizing a new payload. |

UvaBackedTensor initializes its CPU source with zeros and stages copies into
round-robin UVA buffers. add_request updates each live parameter before
apply_staged_writes on every last-PP worker (TP4/PP1 means all four). There is
no rank-0-only sampling mirror update. Inactive slots may contain old requests;
they are not sampler inputs and must stay outside the digest. Padded draft
mapping is filled with -1 and masked by Gumbel. There is no per-request Torch
generator in this V2 path: int64 seeds + positions feed Triton Philox directly.

The concrete reproduced first-prefill hazard is hashing worker-local slot
indices. Four fresh CPU interpreters with PYTHONHASHSEED=0/1/2/3 execute the
native warmup cleanup and allocate the first real request in slots **0/1/0/2**.
Identical scheduler output therefore does not imply identical local mappings.
This is the likely explanation for the fleet refusal, but existing logs lack
per-field measurements and do not prove the incident's exact mismatched field.
The earlier diagnosis based on independent generated seeds is withdrawn:
equal post-warmup RNG histories give equal seeds. Divergent RNG histories are
a latent T>0 hazard, tested as a collective refusal rather than claimed as the
incident's cause.

# Implementation and validation

Round 2 removes seed_request and its admission hook. Normalization is not
needed to correct slot-dependent false refusals, and replacing the native
random int64 with a truncated request-ID hash cannot establish the exact native
seed distribution or independence for reused IDs. The final implementation
does not write seeds or alter RNG draw histories. Generated and explicit seeds,
seeds_set, target Gumbel, draft Gumbel, acceptance and salted residual sampling
retain native behavior, including realized outputs whenever the guard passes.
The test compares the transformed add_requests AST with the exact stock AST
structure; the native sampler/sampling-state source pins still match.
This preserves the output law by leaving its seed generation and sampling
arithmetic unchanged, rather than introducing a new random-stream assumption.

Prepare binds named live fields. Sampling temperature, effective seed and CPU
lengths are ordered (request_id, value) pairs in batch order. The digest never
contains the local slot indices. Any permutation of local slots with equal
values per request therefore produces identical payloads and guard packets.
An incorrect mapping that associates a different temperature/seed with a
request still refuses. Native allocation, cleanup and preemption are unmodified.
T=0 seeds are represented as null, signed zero
temperatures canonicalize, invalid active indices/temperatures refuse, and
inactive/padded rows stay outside the digest. Draft-irrelevant target filters
are omitted. Live CPU length bounds remain guarded because stock autoregressive
Speculator.propose uses them for max_seq_len and draft attention metadata.
The sampling-state, sampler and UVA-buffer sources now have startup SHA pins
alongside the original four compatibility dependencies.

The existing fixed-size MAX reduction, exact probability bits and host refusal
point are preserved. Host fold inputs are retained only until a successful
guard. Failure computes per-field SHA-256 sub-digests, exchanges those on the
cold CPU group, and writes KSTOP_GUARD_FIELDS on every rank with local/peer
hashes and differing field names. No field hashing, peer exchange or additional
device reads occurs on success. Diagnostic exchange errors still log the local
field hashes and preserve the original refusal.

Round-2 Mac checks against the stock extraction (2026-10-03):

- Four-process CPU regression PASS, with PYTHONHASHSEED=0/1/2/3 set before
  interpreter startup, no sockets. Executes pinned RequestState initialization,
  add_request/remove_request, GPUModelRunner.finish_requests/_remove_request,
  native warmup request-ID construction and set cleanup; only token/UVA storage
  and irrelevant model callbacks are CPU substitutes. First slots are 0/1/0/2.
  Pipes carry fixed guard packets to the parent for the same MAX reduction,
  then each worker executes the actual planner host check and prepare path.
  First T=0 and T>0 batches and subsequent finish/preempt/re-admission with mixed
  temperatures pass despite different slots. Swapped request values refuse on
  all four workers with field diagnostics. Restoring the old sampling.mapping
  field refuses for every otherwise-valid phase.
- Before/after check: copied the final harness into a temporary checkout layout
  and substituted the actual d03fb47 runtime. It fails first-t0 with guard
  agreement=False and slots 0/1/0/2. The final runtime passes the same harness.
  Receipts: /srv/projects/kstop-specsample-guard-receipts/round2/.
- 16 simulated TP4 scenarios PASS. Execute pinned SamplingStates initialization,
  add_request and staged writes with CPU-backed UVA storage. Deliberately give
  workers different RNG draw histories for unused T=0 seeds and a negative T>0
  seed case. Equal model seeds/draw histories pass natively. Reintroducing the old unconditional
  T=0 seed fold makes every rank refuse at prepare; the fixed payload passes.
  Covers unseeded T=0 first 2,048-token chunk of a 28,315-token prompt, T>0 seed
  agreement/refusal, explicit seeds, mixed T=0/T>0, signed zero, stale mirrors,
  padding -1, and per-field collective refusals for meaningful differences.
- Compatibility 14 tests, recipe 24 tests, launcher 54 tests, short DSA 9 tests,
  spec-sample 6 tests:
  PASS. Recipe negative controls reject drift in all seven compatibility pins.
- Kstop 29 tests PASS with only one-rank all_reduce/all_gather transport
  substituted; original test bodies and tensor operations run. Unmodified
  test_kstop.py aborts in module setup because Gloo cannot bind a socket here.
- Kstop K3 conditional distribution test PASS (FP32 and BF16, 119 conditions
  each). CPU noise arithmetic substitutes for native Triton; this is not a GPU
  Philox qualification.
- Source pins: 23 recipe and 7 compatibility sources PASS.
- Python AST parsing, shell syntax, git diff --check: PASS.
- Final fix round 2; no additional review round. Retained CPU length bounds for
  autoregressive draft attention. Removed the unsupported slot-invariance
  assumption and unnecessary seed normalization.
- Real Gloo runtime: unavailable in this Mac sandbox (uv_bind: operation not
  permitted), confirmed again during round 2. Docker socket was unavailable in
  round 1; no Docker run attempted in round 2. No pinned-image or GPU run is
  claimed. The former one-rank control-flow harness stubs add_requests entirely;
  the new regression executes the native seed population body explicitly.

# Coordinator Docker command (prepared, not executed)

Use the isolated checkout as the host working directory. Requires cached pinned
ARM64 image and an empty local Colima queue. Output stays under Projects.

```sh
cd /srv/projects/glm53-full-ksspec
set -o pipefail
IMAGE=ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6
OUT=/srv/projects/kstop-specsample-guard-receipts
[[ $(uname -s) == Darwin && $(docker context show) == colima ]] || exit 2
running=$(docker ps -q) || exit 2
[[ -z $running ]] || exit 2
docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}' || exit 2
mkdir -p "$OUT"
docker run --pull never --network none --rm --platform linux/arm64 --cpus 2 --memory 8g \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,src=$PWD,dst=/pkg,readonly" --workdir /pkg \
  --entrypoint /usr/bin/nice "$IMAGE" -n 10 \
  python3 -B /pkg/tests/kstop_specsample_rank_cpu.py --gloo \
  --source /usr/local/lib/python3.12/dist-packages \
  2>&1 | tee "$OUT/tp4-gloo.log"
```

Expected: four HASHSEED TP4 PASS lines, sixteen GLOO TP4 PASS lines, followed by
KSTOP SPECSAMPLE TP4 PASS (Gloo), zero exit. The legacy negative case must
refuse collectively, while the fixed T=0/T>0 cases must pass.
The full existing offline matrix now includes this four-rank regression:

```sh
cd /srv/projects/glm53-full-ksspec
./tests/run_kstop_compat_cpu.sh \
  /srv/campaign/diagnostics/glm53-full-20260929/day3 \
  /srv/projects/kstop-specsample-guard-receipts/full-suite
```
