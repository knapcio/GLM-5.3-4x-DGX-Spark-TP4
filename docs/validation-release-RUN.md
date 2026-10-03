# stack-1003 gate receipt runbook

Run only in the operator's scheduled serving window. W4 R measurements are
recorded in the accompanying results; publication and production promotion remain
held. The commands below reproduce the protocols with operator-selected paths.

1. Freeze the release revision, image IDs, private deployment config and qeval
   threshold. Select the [release layout](runtime.md), reviewed copy guard and
   pinned dispramd. Keep spec-sample off; retain the pad-hygiene gate decision.
2. Complete the existing health, exact reply, every-rank arming, OOM/restart and
   watchdog memory-admission checks. Preserve boot-through-health/admission times.
3. Run the sparkDash sweep first with thinking off. Record prose/code/structured/json
   at c1/c2/c4/c8, aggregate and per-stream rates; keep complete raw jobs and warmups.
4. Measure cold prefill and warm replay separately. Use fresh prefixes for cold
   cells; record actual input tokens and APC state. The standard collector covers
   4K/8K/16K/32K; run a separately labelled near-64K prefill cell in the admitted
   context. Do not infer it from long-probe TTFT or silently label a shorter request 64K.
5. Run c1 prose decode at added context 0/16K/30K/60K, thinking off,
   with matched output budgets and actual prompt lengths in every receipt.
6. Run the pinned RigMark thinking-on low-effort protocol. Preserve completion
   gates, median/range and concurrency timing definitions; do not mix its rates
   with sparkDash rates.
7. Run qeval three times serially with distinct labels and unchanged budgets.
   Preserve checker scores, mean, frozen threshold, truncation and recurring failures.
8. Run long64k-v2 against the same endpoint and tokenizer; keep all four responses
   and apply its exact [PASS rule](validation.md#long64k-v2).
9. Verify APC repeated-prefix reuse and correct completion. Verify shutdown/lease
   return and recovery through the existing lender contract when stopping the gate.
10. Fill every `{{...}}` field from a named completed receipt. Missing data stays
    visibly pending; a failure remains FAIL/ERROR/HOLD. Preserve raw receipts privately
    and scrub deployment details before exporting any derived public summary.

```bash
# Set SPARKDASH_SPARK_ID to the configured dashboard node identifier.
SPARKDASH_API="$DASH_API" python3 bench/sparkdash.py full 66112 > "$SPARKDASH_RECEIPT"
python3 bench/decode_context.py --endpoint "$SERVING_URL" \
  --tokenizer-json "$TOKENIZER_DIR/tokenizer.json" --out "$CONTEXT_RECEIPT"
python3 bench/qeval.py run stack-1003-q1 --url "$CHAT_COMPLETIONS_URL"
python3 bench/qeval.py run stack-1003-q2 --url "$CHAT_COMPLETIONS_URL"
python3 bench/qeval.py run stack-1003-q3 --url "$CHAT_COMPLETIONS_URL"
python3 bench/long_retrieval_v2.py --execute --endpoint "$SERVING_URL" \
  --tokenizer-dir "$TOKENIZER_DIR" --out "$LONG_PROBE_RECEIPTS"
```

The historical `scripts/release_gate.py` packet has an older measurement scope;
it is not a complete substitute for this receipt checklist. In particular, the
new context sweep and long64k-v2 are distinct requirements. Preserve its applicable
serving/admission guards without treating unsupported or omitted cells as passes.
