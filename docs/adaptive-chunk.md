# Adaptive FP4x prefill

Base: `perf/fp4x-kv` at `64269fe`. `GLM_PREFILL_CHUNK_ADAPTIVE=1`
uses a 4096-token request budget during prefill when **total context** tokens
(computed tokens including APC hits + remaining prompt tokens) are at least
`GLM_PREFILL_CHUNK_THRESHOLD` (default 16384); otherwise 2048.
The threshold accepts a positive token count; lower values need separate
qualification because boot h showed a cold 4K regression. FP4x defaults to 1; FP8 defaults
to 0. Explicit 0 keeps the original fixed `GLM_W2_PREFILL_CHUNK` behavior.
The launcher passes both optional variables, including an explicit zero,
to every container. Settings are read at scheduler import, not live.

Keep the existing drained serving control at 2048. With adaptive enabled,
4096 as a serving control also uses 2048 until eligible; 512 remains the
unchanged boot/admission phase. The existing control schema and drain rules
are unchanged. This feature requires the existing 4096 constructor capacity.

## Scheduling and equivalence

The source pin is SHA256
`4c38a32c7405eb95eb9dd3b3d04cbfe5d0cb4ebc0b18efbaa4adc68c7a9bca5a`
for `vllm.v1.core.sched.scheduler` (vLLM `487ecf187`). The transform changes
only `Scheduler.schedule`, before the existing drained-control wrapper.
Running requests calculate remaining prompt tokens as
`num_prompt_tokens - num_computed_tokens`, then compare computed + remaining
to the threshold. Waiting requests use local-plus-external `num_computed_tokens`
**after** the native APC lookup in the same calculation. Thus total context
includes the cached prefix, earlier prefill chunks and remaining prompt; it
equals the full prompt length throughout prefill. Decode-only requests never
expand the budget. Remote asynchronous loads do not expand the budget.

Each step starts with the native 2048 total/input budgets. Discovering an
eligible request extends both remaining budgets once, to a maximum of 4096;
every prefill request also has its own 2048/4096 cap. All requests share these
budgets. The native priority order, early exits, preemption budget refunds,
draft reservations, encoder limits and model-length limits remain in place.
An eligible request behind an exhausted budget waits for the next step,
as in native scheduling; this change does not scan or reorder queues.

Native K3 reserves two input slots per scheduled request: a solitary prefill
normally schedules 2046 or 4094 tokens, not the nominal 2048 or 4096. For a
20,480-token cold prompt the tested splits are five chunks of 4094, then 10.
The choice is recalculated every step using total context, so long prompts
retain the 4096 budget through their final prefill chunk. A 61,376-token APC
hit plus 3072 new tokens uses 4096 and schedules the whole delta in one step.
Decode tokens and their
speculative shapes use the existing budget/reservation logic.

If no scheduled request qualifies, the budgets and all downstream native
arguments are identical to the 2048 control. Tests compare complete scheduler
outputs and KV allocation arguments for small total contexts, decode
and mixed batches, including 63 combinations of sizes and draft reservations.
This equivalence supports default 1 for FP4x. It does not assert identical
generated text for long prompts with changed chunk boundaries.

The decision is rank-invariant: `v1/engine/core.py` calls `self.scheduler.schedule`
once, then sends its `SchedulerOutput` to `execute_model`; the multiprocess
executor invokes the collective RPC with `args=(scheduler_output,)` and enqueues
that tuple in `rpc_broadcast_mq`. Only the DP leader constructs this queue;
all TP workers consume the same schedule. No worker independently selects a
prefill chunk. CPU tests check this dispatch path in the extracted pinned image.

## Memory and offline validation

The scheduler config, 4096 constructor allocation, KV pool, max sequences,
full-sequence admission rule, lookahead and FULL_DECODE_ONLY capture sizes
are unchanged. The transform extends local scheduling counters, never the
constructor/config values. Its output assertion checks the chosen total cap;
native nonnegative input-budget assertions preserve the draft-slot ceiling.
No CUDA graph, buffer, kernel or memory-ledger change is introduced.

Supplied fleet evidence (2026-10-06, same boot, A2048/B4096/A2048):

| Case | Cap 4096 versus cap 2048 |
| --- | --- |
| APC: 61,376 cached tokens + approximately 3K new tokens | 7.2% faster; median 4.995 s versus 5.38 s |
| Cold 4K | 9% slower |
| Cold 8K | 0.7% faster |
| Cold 16K | 3.6% faster |
| Cold 32K | 5.3% faster |
| Cold 64K | 5.6% faster |
| Cold 96K | 6.4% faster |

This supports selecting by total attention context, including APC hits, with
16384 as the default threshold. Boot-h memory was identical (rank-0 minimum
9.22–9.27 GiB per cap), and decode cycle unchanged. These supplied fixed-cap
measurements support the existing maximum 4096 allocation; adaptive v2 still
needs coordinator GPU qualification. The local change and validation are offline.

Run locally with the parent of an extracted pinned `vllm/` tree:

```sh
GLM_IMAGE_SRC=/path/to/extracted-image python3 -B tests/test_adaptive_chunk.py
python3 -B tests/test_launcher.py
GLM_IMAGE_SRC=/path/to/extracted-image python3 -B tests/test_recipe.py
```

The adaptive tests execute the complete native/transformed scheduling function
with inert KV and executor boundaries; no torch, CUDA, SSH or fleet connection
is required. They cover cold 4K/20K, 61,376 cached tokens with 3K APC delta, exact threshold,
running and waiting requests, concurrent budgets, mixed decode, admission
refusal, paused steps, the 512 boot phase, disabled mode, and wrapper composition.

Original v1 Mac validation on 2026-10-06: adaptive scheduler 14/14, launcher 56/56,
recipe 24/24, release gate 18/18 and persistent cache 10/10 passed; all 23
base and 7 compatibility image-source pins passed. Adaptive, launcher and
recipe tests also ran under Python 3.12. Shell/Python syntax and whitespace
checks passed. CUDA execution and adaptive TTFT are left to the coordinator.

V2 offline validation on 2026-10-06: adaptive scheduler 14/14, launcher
56/56, recipe 24/24, release gate 18/18 and persistent cache 10/10 passed;
all 23 base and 7 compatibility image-source pins passed. Python AST and
whitespace checks passed. Rank dispatch, full-sequence admission, constructor
capacity and capture behavior remain covered by the existing tests. No fleet
or GPU action was run.

## Coordinator verification plan

The env is boot-time only, so use **two boots** of the same commit and exact
boot-h FP4x recipe (same pinned image, 98176 context, 1573 blocks, graphs,
weights and other switches). No deployment or measurement was run for this
change on the Mac.

1. Boot A: `GLM_PREFILL_CHUNK_ADAPTIVE=0`, serving cap 2048. Boot B:
   `GLM_PREFILL_CHUNK_ADAPTIVE=1`, threshold 16384, serving cap 2048. Save both
   complete launch vectors. They must differ only in the adaptive env.
   In B, save the scheduler's `GLM_PREFILL_CHUNK_ADAPTIVE armed` receipt;
   retain the normal 512 admission receipt and serving-cap acknowledgement.
2. On each boot, verify exact `OK`, no restarts/fatals, all-rank pool/capture
   receipts, admission floors, and steady memory recovery. Repeat the same
   cold 4K/8K/16K/32K/64K/96K ladder with at least three unique prompts per
   cell (96K must fit the 98176 limit including output). Put a unique salt at
   the **start** of every measured prompt; require zero prefix hits and save
   actual token counts. Compare medians and spread by cell. Small cells should
   match A within repeat noise; assess the boot-h >=3% gain target at 32K+,
   including the final prefill chunks under the total-context rule.
3. Seed a 61,376-token prompt, then append approximately 3K while keeping
   total prompt plus output below 98176. Record actual APC hits and delta.
   Compare warm TTFT to A: total context exceeds 16384, so B must use the
   4096 budget even with only approximately 3K uncached tokens. Repeat with
   total contexts just below and at 16384 to exercise both branches.
4. Run two concurrent cold long requests that fit the unchanged shared KV pool
   (e.g. 20K each), two short requests, and a long prefill alongside an active
   decode. Compare scheduler budget validity, completion, preemption and
   latency to A. Run the same c1 prose/code cycle-ms pairs and decode-only
   c2/c4 checks; retain the boot-h +1% cycle guard.
5. Retain all-rank minimum memory, swap deltas, recovery and fatal/restart
   counts across the ladder and mixed tests; rank 0 must remain >=8.3 GiB.
   Finish with APC, needle and the existing quality smoke gates. Promote only
   after these receipts; rollback is boot A's explicit adaptive=0 env.

For a **one-boot functional screening** when a second boot is unavailable,
boot B alone and run steps 2–5, checking the receipt and expected small/large
branch behavior. That verifies admission and operation, but cannot establish
a same-boot adaptive A/B TTFT comparison. Changing a shell env or worker RPC
does not toggle the central scheduler's captured boot setting.
