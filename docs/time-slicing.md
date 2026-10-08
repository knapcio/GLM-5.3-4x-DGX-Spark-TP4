# Decode/prefill time-slicing

On by default: `GLM_DECODE_FAIR=1`, `GLM_DECODE_FAIR_CHUNK=4096`, `GLM_DECODE_FAIR_DECODE_STEPS=40`
(`profiles/current.env`). Source: `overlay/bringup/glm_decode_fair.py`, wired into the adaptive prefill
scheduler transform in `overlay/bringup/glm_adaptive_chunk.py`.

## Problem

vLLM's chunked prefill mixes a prefill chunk into every scheduler step while another request decodes.
On this cluster a mixed step with a 4096-token chunk takes about 5.2 s (`0.443 + 0.001164 x C` seconds,
C = chunk tokens), so a request that is decoding while another agent sends a cold 100K-token prompt
gets one token batch every ~5 s, about 0.44 tok/s, for the whole ~131 s of that prefill.

## Policy

Only when there is both prefill work and at least one eligible decoding request:

1. The **sum** of prefill tokens in a step is capped at the chunk (4096). All outstanding decode rows
   and their native draft slots are reserved first; running order, queue order, KV admission and
   native preemption are unchanged.
2. After a step that scheduled prefill, the next **40 steps are pure decode** (prefill gets a zero
   budget). Pure-decode steps keep the captured CUDA graph shapes (~0.08-0.09 s per step).
3. Steps that emit nothing, or are paused or refused, do not count towards the 40.

With no eligible decoding request, the unchanged adaptive rule applies (2048 below 16,384 total prompt
tokens, otherwise 4096) and a lone prefill runs at full speed. Decode-only and c1 steps schedule exactly
what the adaptive scheduler schedules. The policy lives in the central EngineCore scheduler only; its decision is
broadcast to all tensor-parallel ranks with the normal `SchedulerOutput`, so ranks cannot disagree.

## Measured (boot T2, October 8, 2026)

| Check | Result |
|---|---|
| A decodes (60,043-token prefix-cached prompt) while B prefills a cold ~100K prompt | A **12.46 tok/s** (14.15 / 10.77); policy off on the same October 7 window boot: 0.44 tok/s |
| B time to first token | **212.0 s** (216.1 / 208.0); policy off: 130.9 s, so **+62 %** |
| A's token gaps during B's prefill | median 0.08-0.09 s; 25 gaps of ~5.3 s, one per prefill chunk |
| c1 decode, same boot, policy ON vs OFF (ABBA x2, 48 pairs) | **+0.83 %** [-0.36, +2.05], consistent with unchanged; prose +1.51 %, code +0.29 % |
| Preemptions / errors in the scenario | 0 / 0 |

After B's first token, A returns to its solo rate. In real agent use, A pauses for tool calls, and B's
prefill runs at full speed during those pauses, so the realized TTFT cost may be below the synthetic +62 %.
On the October 7 boot, N=20 instead of 40 gave A about 7.1 tok/s for +30 % TTFT; 4096/N40 was chosen for
multi-agent use.

## Runtime control without a reboot

The profile boots the policy with no file dependency. To be able to change or switch it off while serving,
boot with `GLM_DECODE_FAIR_CONTROL=/cache/decode-fair.json`. The file lives on the rank-0 host at
`$OVERLAY_REMOTE/cache/decode-fair.json` (mounted as `/cache/decode-fair.json`) and must exist before the
first request is scheduled. The launcher requires a fresh `$OVERLAY_REMOTE`, creates `$OVERLAY_REMOTE/cache` during
`./start.sh serve` and writes `d2w2-prefill-control.json` there just before the containers start. Write the sidecar
once that file appears; the engine then loads weights for about five minutes before health 200. Use one writer:

```bash
# during the boot: policy ON, 4096 / 40, sequence 0
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 4096 --decode-steps 40 --sequence 0

# emergency OFF while serving
SEQ=1   # one more than the sequence currently in the file
python3 scripts/decode_fair_control.py "$OVERLAY_REMOTE/cache/decode-fair.json" --chunk 0 --decode-steps 0 --sequence "$SEQ"
```

The writer replaces the file atomically and refuses a sequence that is not higher than the one in the file.
The scheduler reads the file every step: an unchanged file is accepted silently, and a changed file takes effect
on the next whole scheduler step, with no drain, only if its sequence is higher than the last accepted one.
Rank 0's log acknowledges each accepted change with a `GLM_DECODE_FAIR {...}` line carrying the new values.
A missing, malformed or stale (changed but not higher sequence) file fails closed: the central scheduler raises
an error, which stops the engine; it does not switch the policy off. Without a sidecar, switch it off by booting with `GLM_DECODE_FAIR=0`.

## Tests

`tests/test_decode_fair.py` and `tests/test_decode_timeslice.py` run the complete pinned vLLM scheduler on CPU:
both running orders, many-request permutations, the aggregate cap, decode reservations, APC deltas, KV refusal,
priority preemption refunds, four state replicas, unchanged c1/decode-only outputs, the default-off path,
OFF/ON/OFF/ON sidecar changes with running requests, and the 4096/N40 boot policy through the real installer.
