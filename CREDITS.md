# Credits

- **Allan Clark / ajclark**: coalesced original-checkpoint reader, two event-fenced pinned tiles,
  owning CUDA batches, producer queue and storage-lifetime budgets, ported from
  [GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/tree/f0b64af5ac6028624e5ae255d998faa1e0a9324a/runtime/nvme_loader).
  Apache-2.0; original license and NOTICE retained in `LICENSES/ajclark-*`.
  The modified port preserves stock ordering and local MTP selection, packs disjoint page ranges,
  caps transient storage and enforces the recipe's memory floor. **tonyd2wild**: upstream serving
  recipe and the `dcp/` attribution trail used to locate this implementation. Checkpoint terms remain separate.

- **Z.ai**: GLM-5.3 and the checkpoint's native multi-token prediction (MTP) layer; the current recipe reuses that layer at K2 without a separate drafter checkpoint. [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3).
- **keys / @u1tra_instinct**: TensorFold results that pointed this campaign toward native MTP. **Ash Hart and TensorFold contributors**, and the [GLM TensorFold Spark recipe](https://github.com/jayleaton/glm53-tensorfold-spark): measured speculation and execution evidence that informed the investigation. [TensorFold](https://github.com/ashhart/TensorFold). No TensorFold kernel is copied by the MTP mapping fix.
- **vLLM contributors**: native MTP loading, shared target embedding/head, index compaction/reset, greedy drafting and standard rejection sampling. The source-pinned local packed-module fix supplies the fused QKV/gate-up/indexer mapping required by the existing compressed-tensors checkpoint; it does not change weights or kernel arithmetic.
- **Guess-Verify-Refine authors**: the short-sequence DSA case, where a context no longer than top-k selects
  every token and the indexer's scoring can be skipped, which the short-context shortcut applies
  ([paper](https://arxiv.org/pdf/2604.22312)). No code from it is used.
- **DeepSeek**: the DSA lightning indexer with top-2048 selection used by GLM-5.3. **vLLM contributors**: the
  sparse indexer, `persistent_topk` and FULL graph manager the short-context shortcut relies on; its all-selected
  branch for rows of at most 2048 tokens is what makes skipping the query/logits path exact. The source-pinned
  local `glm_dsa_short` adapter (knapcio full-GLM campaign) adds no kernel and keeps every stock key/cache operation.
- **NVIDIA**: the PTX `discard.global.L2` instruction used by the dirty-L2 fix. **Triton contributors**: inline
  PTX and the compiler the fix is built with. The diagnosis (dirty split-K partials slowing the next INT8 o_proj),
  the discard reduce and its tests are our own work (knapcio full-GLM campaign, from its own profiling); the partial kernel is the
  repository's own split32 MLA, unchanged, whose split-layout design credits CosmicRaisins (below).
- **Native MTP K-stop** (on by default): stopping the draft early when its confidence falls follows the
  speculative-decoding literature on dynamic draft length, notably SpecDec++ (Huang, Guo and Wang: adaptive
  candidate lengths from a predicted acceptance probability) and DISCO (Mamou et al.: dynamic speculation
  lookahead); no code from either is used. The mechanism and its rank-replicated control came out of the
  campaign's think rounds (knapcio full-GLM campaign). Z.ai's native MTP layer and shared head; **vLLM contributors**: the MTP and
  autoregressive speculators, V2 model runner and graph manager, async scheduler rollback, rejection sampler,
  all-gather logits processor and PyNccl communicator it builds on. The dead-row expert reuse follows the
  DeepSeek recipe's verify-cap idea (rows after the bonus row reuse the request root; no DeepSeek code); the
  remap custom op is carried unchanged from the campaign's dead-row work. The uniform cumulative stop, local
  rank-replicated decisions with the digest guard, the source pins and their tests are local work (knapcio
  full-GLM campaign, including its cross-domain review that proposed rank-replicated decisions). Source excerpts
  keep their Apache-2.0 SPDX headers.
- **vLLM contributors**: automatic prefix caching (block hashing, the prefix-cache manager and `cache_salt`),
  which the native profile turns on with one serving flag; no local code.
- **Native MTP shard selection**: vLLM's `get_spec_layer_idx_from_weight_name` and the pinned `DeepSeekMTP` /
  `DeepseekV2Model` weight loaders it relies on (vLLM contributors); the header/index-driven shard selection and
  its checks are local work (knapcio full-GLM campaign).
- **MLA plan skip** (optional): the vLLM FlashInfer SM90 sparse-MLA backend authors and the FlashInfer
  `BatchMLAPagedAttentionWrapper` authors; the diagnosis (two blocking host reads per cycle feeding a plan nothing
  runs under the Triton MLA forward) and the switch are local work (knapcio full-GLM campaign).
- **Spec-sample** (optional): the standard speculative-sampling theorem (Leviathan, Kalman and Matias, "Fast
  Inference from Transformers via Speculative Decoding"; Chen et al., "Accelerating Large Language Model Decoding
  with Speculative Sampling"): accepting with min(1, p/q) and resampling from the normalised residual max(p - q, 0)
  leaves the target distribution unchanged. vLLM contributors: the probabilistic draft sampler and rejection
  sampler it switches to; the independent residual noise salt and tests are local work.
- **vLLM contributors**: the fused CUDA `grouped_topk` kernels (their BF16 instantiation widens each logit before
  any arithmetic), the Marlin MoE kernel's lock reset (`barrier_release`) and the sparse-MLA index conversion that
  the staged glue-lite switches rely on. The launch ledger, the exactness arguments, the source-pinned
  `glm_glue_lite` adapter, its tests and GPU gates are local work (knapcio full-GLM campaign); no kernel is added.
- **vLLM contributors**: the V2 model runner's CUDA graph manager and capture-size handling that the staged
  c4 later-MTP graph switch relies on. The diagnosis (the released capture list leaves the four-request later MTP
  pass eager) is local work (knapcio full-GLM campaign); the switch only adds width 4 to the capture list.

- **tonyd2wild (Tony)**: v11 ARM64 container and the vLLM `487ecf187` model-support build, including the
  model/backend adaptations required for this target and DSpark. The recipe references
  `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
  [Full-model recipe](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark).
- **Tech2wild**: the full GLM-5.3 Int4-Int8Mix compressed-tensors checkpoint, immutable revision
  `206507bbb047d8223964a0414cd83230c59428f9`.
  [Pinned model card](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/blob/206507bbb047d8223964a0414cd83230c59428f9/README.md).
- **Red Hat DSpark / Speculators contributors**: unmodified GLM-5.3 DSpark drafter, immutable revision
  `b374b95663447ea0e935151be4f3d6666e36e6d7`, the proposal/confidence/Markov model and its training.
  The historical DSpark K3 fallback uses the GLM-5.3 model licence.
  [Pinned drafter card](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark/blob/b374b95663447ea0e935151be4f3d6666e36e6d7/README.md).
- **CosmicRaisins**: sm12x sparse-MLA split-layout work, especially keeping the 512-dimensional latent
  and 64-dimensional RoPE dot products separate to avoid padding 576 to 1024. This design informed the
  local full-model Triton implementation. No CosmicRaisins source file is vendored.
  [Pinned kernel](https://github.com/CosmicRaisins/glm-5.2-gb10/blob/8a41f551f0b8097b7e9659a3e4dfe58aa0a51f86/kernels/sm12x_sparse_mla_attn.py),
  [change record](https://github.com/CosmicRaisins/glm-5.2-gb10/blob/8a41f551f0b8097b7e9659a3e4dfe58aa0a51f86/CHANGES.md).
  Upstream is Apache-2.0.
- **Matt Mastracci (mmastrac)**: sparse-MLA prefill design discussions and the Triton sparse-gather/softmax
  approach developed in the Flash recipe lineage. The full-model adapter adds RoPE and capture-safe split-K;
  no KDA, FlashKDA, convolution-split or other unrelated Flash optimization is enabled here.
  [GLM recipe](https://github.com/mmastrac/glm-5.3-flash-4x-gx10),
  [vLLM sparse-MLA discussion](https://github.com/vllm-project/vllm/pull/58454).
- **Luke Alonso (@lukealonso), Jason Cook (@original-el8), Local Inference Lab**: b12x/RoCEnante
  one-shot RDMA collectives and the vLLM integration lineage.
  [b12x #295](https://github.com/local-inference-lab/b12x/pull/295),
  [vLLM shim #597](https://github.com/local-inference-lab/vllm/pull/597).
  The unmodified vendored transport subset is pinned to `b58f34eaf978277621efced6678e6713fd7122e4`;
  file hashes and the Apache-2.0 licence are in `roce/b12x/`.
- **tonyd2wild and rhys101**: the SM121 v11 RoCEnante port and runtime work that informed the local
  `roce/glm_roce` wrapper. The wrappers preserve the Flash recipe's attribution and import-time integration.
- **Willian-Zhang**: the GB10 file-backed safetensors copy-cost report
  [vLLM #58726](https://github.com/vllm-project/vllm/issues/58726), motivating the anonymous/pinned-slab
  loader port from the local DeepSeek recipe. The loader does not change weight bytes.
- **FlashInfer contributors**: sparse-MLA backend, metadata conversion, sparse selection and other runtime
  dependencies in the external image.
  The full-model Triton kernel replaces attention computation while retaining the image's sparse metadata path.
- **vLLM, Triton, PyTorch, Marlin, NVIDIA CUDA/NCCL**: serving, compilation, tensor runtime,
  mixed-precision operators and network libraries. Their licences remain attached to those components.
- **sparkDash contributors**: DecodeBench measurement protocol; MIT notice retained in `LICENSES/MIT-sparkDash.txt`.
- **Alex Ellis and RigMark contributors**: RigMark 1.0.0 benchmark protocol and prompts, pinned to
  `d8353e93b274e8d880ab14a5df2a55c87d7bee16`; the RigMark results retain its workload settings and
  completion gates. [RigMark](https://github.com/alexellis/rigmark). Measurements only; no RigMark code is vendored.
- **knapcio**: public fresh-clone release coordinator, teacher-logprob/Qeval checks and shared-prefix
  ground-truth scan, following the local Flash and DeepSeek release protocols; four-node integration, source-pinned full-model MLA adapter, independent draft pool,
  loader policy, guarded launcher, CPU tests and recorded qualification work.

The GLM Flash and DeepSeek repositories supplied recipe conventions and local wrappers:
[GLM Flash](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4),
[DeepSeek](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4).
This repository does not claim a blanket relicense of either project or of externally referenced images.

- **knapcio campaign day2/w2; vLLM scheduler contributors (Apache-2.0)**: qualified drained prefill-cap switch, fixed constructor capacity and all-rank control readback.

- **Draft-only NVFP4 LM head** (on by default): **Marlin** (IST-DASLab; Elias Frantar, Dan Alistarh and
  contributors): the W4A16 mixed-precision GEMM, as integrated in vLLM; **NVIDIA**: the NVFP4 format the draft head
  bank uses. **vLLM contributors**: the Marlin linear integration and NVFP4 schemes, the V2 model
  runner's native MTP speculator and draft CUDA graph manager, the shared LM head and logits processor (left unpatched
  for the target), and `moe_align_block_size` with the Marlin MoE kernel, whose atomic token ordering and
  block-partitioned fp32 reduction explain the few-ULP draft-state variation the INIT criterion bounds (no change to
  either is made). **Z.ai**: GLM-5.3's native MTP layer and shared head. The quantized draft-only copy, the all-rank
  INIT qualification with collective fallback, the write-coverage / exact-token / bounded-float criterion, the
  instrumented localisation of the variation and the CPU/Gloo tests are local work (knapcio).

- **Decode/prefill time-slicing** (on by default): **vLLM contributors**: the chunked-prefill scheduler, request
  eligibility, native preemption and the `SchedulerOutput` broadcast the policy runs inside; the pinned scheduler
  source is transformed, not replaced (Apache-2.0). Related prior work: Sarathi-Serve stall-free batching
  (Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024), which
  bounds prefill work per step so decodes keep progressing; no code from it is used. The aggregate prefill cap with
  decode reservations, the N pure-decode steps between chunks, the mixed-step cost model, the schema-2 runtime
  sidecar and the CPU tests are local work (knapcio).

- **FP4x KV storage**: knapcio's original dc5ad9e E2M1/block-E4M3/power-of-two-row
  encoder, packed ABI, fused writer/readers and CPU tests. NVIDIA and Triton contributors:
  datatype conversions/compiler; vLLM contributors: MLA/MTP cache dispatch, paged indexer,
  specs, prefix hashing and graph lifecycle. The packed sparse readers retain the local
  split-layout design credited to CosmicRaisins and Matt Mastracci. **Mia (MiaAI-Lab)** supplied the FP4 KV idea (ideas only; licence unverified; no code copied).
  Tech2wild/tonyd2wild supplied the checkpoint and recipe lineage; no FP4 kernel or ABI implementation from them is copied.

## Experimental KV headroom planning

The conditional cache policy uses Linux kernel memory interfaces and the
Tech2wild/tonyd2wild serving comparison as motivation. The memory-pool extension
retains kindlingai's dispram integration and the existing FP4x implementation.
The future stress harness follows the campaign's fp4x-cal clone/launcher,
think10 GateMetrics and sparkDash benchmark conventions. It is default-off;
expanded memory budgets and performance remain unqualified.

- **kindling dispramd**: external, unmodified AGPL-3.0 lender at
  [5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e](https://github.com/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e/kindling/dispram). Downloaded separately; no relicensing or binary bundled here.
- **Marlin kernels via vLLM (IST-DASLab and vLLM contributors); NVIDIA NVFP4 format**: NVFP4 W4A16 packing, scales, TP slicing and native-MTP dispatch used by the independently written sidecar converters. **Z.ai**: architecture, tokenizer, native MTP and model licence. **Tech2wild/tonyd2wild**: original Int4-Int8Mix checkpoint.
