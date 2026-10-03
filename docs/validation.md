# stack-1003 validation and receipts

Primary gate **PASS**, date **2026-10-03**, inference source **0f24383**,
receipt set **W4 v2 primary (pad=1, spec-sample=0)**. Pad hygiene v2 is on.
This document records the receipt contract for the same admitted primary boot.
Historical validation narrative and measurements are in [history](history.md).

## Serving admission

Retain the selected image IDs, model/manifests, actual environment on every rank,
boot-through-health/admission timing, exact `OK` reply, running ranks, no OOM or
restarts, unchanged watchdog admission and all-rank lever arming/dispatch.
Record native MTP K-stop, uniform K2, short DSA, APC, required dispram mapping,
copy guard, fast load, c2 reuse and the gate-selected pad-hygiene value.

The copy guard is an included local-build prerequisite; build instructions are in
[installation](install.md#select-the-release-layout).
Missing guard, daemon, readable monitoring, lease release or rank agreement is
a refusal; it cannot become a qualified release by filling a table.

## qeval

Use `bench/qeval.py` with three serial runs and the unchanged checker and budgets.
Freeze the checker-matched mean threshold before examining results, from the
qualified reference mean minus its established noise allowance. Preserve task
results, raw finish reasons, truncations and recurring-failure classification.
PASS requires the mean to meet that fixed threshold, zero unexpected truncation
outside known cap items, and no new recurring failure on historically stable tasks.
Recorded scores 71, 72, 72, mean 71.7/75,
threshold 70.553/75 and verdict PASS. Failed tasks across these
runs: `code_camel_to_snake`, `code_interval_intersect`, `code_word_wrap`, `json_manifest`.
Code checkers execute returned Python; use the isolated benchmark account.

## long64k-v2

[`bench/long_retrieval_v2.py`](../bench/long_retrieval_v2.py) is copied from the
seeded retrieval probe with portable CLI paths. The scoring algorithm and PASS
rule are unchanged. It creates 16,384- and 63,488-token nominal inputs using the
checkpoint tokenizer/chat template, with two identical requests per length.
Actual rendered inputs fit just below each target; answer generation has the
original budget. Seven registry fields, alias resolution, operator repair and
archive count give ten typed scalar checks. An independent serialized-record
oracle verifies the generated ground truth.

**PASS rule:** for EACH paired repeat, long score >= control score - 1 and all
seven long registry fields must be correct. Both pairs pass. All four transports
must succeed and responses must finish with `stop`. Transport/protocol failures
produce ERROR, never PASS. Do not loosen the rule after viewing receipts.
An offline scorer/oracle self-check is not a served quality result. A low control
score does not establish absolute quality even if the relative rule passes.

```bash
python3 bench/long_retrieval_v2.py --offline --tokenizer-dir "$TOKENIZER_DIR" \
  --out "$OFFLINE_PROBE_RECEIPTS"
python3 bench/long_retrieval_v2.py --execute --endpoint "$SERVING_URL" \
  --tokenizer-dir "$TOKENIZER_DIR" --out "$LONG_PROBE_RECEIPTS"
```

Keep `panel-*.json`, `prompt-*.txt`, `self-check.json`, `run-*.json` and `summary.json`
in the gate workspace. Retain prompt/template/tokenizer hashes, local/server token
counts, finish reasons, transport errors, timing and per-field scores. Second
repeats may use warm APC; label them. Verdict **PASS**.

## Performance protocol

Fill sparkDash first, prose/code/structured/json order, aggregate tok/s with
per-stream rates in brackets at c1/c2/c4/c8. c8 can queue behind the four slots.
Discard warmups, keep complete runs and preserve collector timing semantics.
RigMark thinking-on low-effort measurements remain separately labelled.
Cold prefill uses independent uncached prefixes; warm replay has its own row.
Record actual input lengths. The collector's built-in sweep ends at 32K; separate
64K prefill receipts and the 0/16K/30K/60K context sweep are required.

All current table cells come from the same admitted release boot. A runtime
lever's causal claim needs its predeclared same-boot comparison and uncertainty;
without a runtime switch, use repeated independent boots. Historical table rows
cannot establish a causal speed claim. Do not publish incomplete requests as rates.

## Offline packaging checks

Use local standard-library checks for docs, links, JSON, shell syntax and the
probe scorer/oracle. The full pinned-image CPU suite needs an extracted `vllm/`
tree and installed dependencies; do not confuse that prerequisite with a new
serving run. Packaging verification performs no remote operations.
Full intelligence equivalence and multi-day stability remain unqualified.
