# Historical pre-gate plan — October 6

**Superseded plan, retained for provenance.** The completed 262144 qualification and current floors are in
[GATE-RESULT.md](GATE-RESULT.md). The 165312 layout, floors and 75-minute budget below describe the original
plan; they are not the current release instructions.

## Fresh-clone release gate — one boot, at most 75 minutes

Not executed by release prep. One coordinator owns the fleet and the existing heartbeat throughout. No concurrent
clients or benchmark jobs. Use the management recovery path; preserve the current deployment, config, image IDs,
watch drop-in and restoration command before handoff. Do not delete containers, alter clocks, sysctls or services.
Stage the image, weights, guard, tokenizer, sidecars and RigMark checkout before the timed gate. Conversion commands,
full audits and sidecar identities are in [NVFP4-SIDECARS](docs/NVFP4-SIDECARS.md). Reuse those verified sidecars.

| Serial phase | Maximum minutes |
|---|---:|
| Fresh clone, pins and pre-stop checks | 3 |
| One 165312 boot and 60 s admission | 7 |
| Shared-pool c4 + single maximum stress | 10 |
| Needle 16K/160K, twice | 5 |
| qeval c1 x3 | 15 |
| sparkDash c1 x5 and concurrency/prefill sweep | 19 |
| RigMark 1.0.0 | 14 |
| Receipt validation and handoff | 2 |
| **Total** | **75** |

Budgets are hard caps, not promised completion times. A timeout is INCOMPLETE/PARKED-IMPL, never a pass or an idea
rejection. No retries or second boot in this gate. Leave the qualified candidate under the persistent serving watch
only after all gates pass; on failure stop this deployment, preserve receipts, and hand the documented restoration
procedure to the coordinator. Restoration is recovery, outside this one-candidate gate.

## Fresh clone and exact identities

The operator sets `PRIVATE_ORIGIN`, `RELEASE_SHA`, `RELEASE_DIR`, `SITE_CONFIG`, `RECEIPTS`, `RIGMARK_DIR` and
`BASE`/`DASH_API` from the private deployment record. `SITE_CONFIG` is outside git and contains no credentials.
`RELEASE_SHA` is the prepared release tip. Validate private origin before cloning; no public repository push.

```bash
test "$PRIVATE_ORIGIN" = https://github.com/knapcio/GLM-5.3-4x-DGX-Spark-TP4-dev.git
git clone --single-branch --branch release/glm53-1006 "$PRIVATE_ORIGIN" "$RELEASE_DIR"
cd "$RELEASE_DIR"
test "$(git rev-parse HEAD)" = "$RELEASE_SHA"
test -z "$(git status --porcelain)"
cp "$SITE_CONFIG" .env
source .env
export RECIPE_HOSTS="${HOSTS[*]}"
mkdir "$RECEIPTS"
git rev-parse HEAD > "$RECEIPTS/commit.txt"
sha256sum scripts/convert_attn_nvfp4.py scripts/convert_more_nvfp4.py \
  scripts/nvfp4_canonical_hash.py bench/qeval.py bench/qeval_tasks.py > "$RECEIPTS/sources.sha256"
bash overlay/guard/build_guard.sh
DRY=1 ./start.sh serve > "$RECEIPTS/dry.txt"
```

Fill site config explicitly: `CTN=glm53full-release-1006-UNIQUE`, unique `OVERLAY_REMOTE`, four immutable `IMAGES`,
`GLM_KV_FORMAT=fp4x`, `RECIPE_DISPRAM=require`, `RECIPE_KV_HEAD_BYTES=3221225472`, `RECIPE_MAX_MODEL_LEN=165312`,
`GLM_ATTN_WEIGHTS=nvfp4`, `GLM_NVFP4_GROUPS=attn,shared,dense,mtp`, both read-only sidecar paths,
`GLM_FP4_RECENT_WINDOW=0`, `GLM_FP4_RECENT_AB=0`, `GLM_LOADER=fast`, `DISPRAM_ALLOW_RM_ALLOC_OOM=1`.
No dev API for the default arm. DRY must show four equal pools: **2622 blocks /167808 tokens**, no recent bank,
no indexer conversion, APC, pad hygiene, K-stop K3/c1 +K2/batches, graphs [1,4,12,16], adaptive prefill.

Before stopping serving, the coordinator verifies the saved memlab `6e7cded` receipt for these selectors:
admission lower rank0 **10.132 GiB**, stressed lower **9.063 GiB**, allowance **0.787713 GiB** (LOO MAE 0.265286).
Zero page-cache credit. Bind the saved prediction to the actual clone/DRY hashes; changed selectors need a new sim.
182080 projects 8.545 GiB but is outside the measured range; do not enlarge the default in this gate.

On each node, using the configured paths, verify read-only source and sidecars, not just their directory names:

```bash
(cd "$GLM_ATTN_NVFP4_DIR" && sha256sum -c SHA256SUMS)
(cd "$GLM_NVFP4_MORE_DIR" && sha256sum -c SHA256SUMS)
sha256sum "$GLM_ATTN_NVFP4_DIR/SHA256SUMS" "$GLM_NVFP4_MORE_DIR/SHA256SUMS"
python3 scripts/nvfp4_canonical_hash.py "$GLM_ATTN_NVFP4_DIR"
```

For the existing reused sidecars require exact per-rank raw attention identities and canonical
attention hash `af8fd1af...`, extra manifest hash `c26bfb42...` (full values in the sidecar document), complete
inventory and saved full CPU audits. A deterministic attention replacement changes header metadata, so its raw and
canonical hashes require separate verified preparation and a recorded new identity before this gate. Verify model
manifest, image/source pins and NCCL hashes. Do the full sidecar
rehash before the timer if it cannot finish within three minutes; inside the timer confirm unchanged file identities.
Use `RECIPE_BOOT_PREFLIGHT_HEADERS` only for an existing exact-path header cache. The cached-header preflight is
default on with that cache, otherwise skip with a warning; failure never vetoes boot. Payload and memory checks remain mandatory.

## Boot, admission and guard

Coordinator handoff: quiesce requests, drain, stop the previous serving watch, stop its ranks through its recorded
launcher, verify clean borrower/lender postcheck and token release, and keep the expiring operator hold/heartbeat.
One fleet owner only. The release launcher acquires its deployment lock and performs compaction on every node before
boot. Start locally on the management coordinator under its user systemd manager; the Mac is only a client.

```bash
systemd-run --user --unit="$CTN-serve" --property=RuntimeMaxSec=4500 \
  --working-directory="$RELEASE_DIR" /bin/bash ./start.sh serve
# Once state/deployment.json exists, start the all-rank 1 Hz guard, continuing through every probe.
systemd-run --user --unit="$CTN-guard" --property=RuntimeMaxSec=4510 \
  --working-directory="$RELEASE_DIR" --setenv="RECIPE_HOSTS=$RECIPE_HOSTS" \
  /usr/bin/python3 scripts/release_1006_probe.py guard --execute --boot "$CTN" \
  --out "$RECEIPTS/guard" --seconds 4500 --handoff-file "$RECEIPTS/handoff.json"
journalctl --user -u "$CTN-serve" --no-pager > "$RECEIPTS/serve.log"
```

Require health 200, exact `OK`, all four ranks running/no restarts or fatal errors, 60 continuous seconds >=8 GiB
all ranks, correct pool/cap, attn385/shared225/dense6/mtp776 loaded exactly once per rank, no recent reservation,
and both adaptive prefill paths. Guard floor **8.05 GiB all ranks**, no >=64 MiB sustained swap growth,
fresh samples <=3 s, kernel/CUDA/numeric/worker errors stop the owned deployment. Existing watchdog also uses idle
reset, 600 s busy-without-progress and 1800 s boot deadline; this gate's seven-minute boot cap is stricter.
Save image/checkpoint/tokenizer/guard/overlay/graph identities and admission timestamps. Any failed unit stops the gate.

## Probes, serial on that same boot

At entry save `GATE_STARTED=$(date +%s)` and export it through the coordinator's existing environment. Use this
wrapper for each payload; it caps the phase by both its budget and the remaining 75 minutes. Commands are serial.
Any failure stops the owned candidate immediately and ends this gate. No next phase after failure.

```bash
phase() {
  local label=$1 budget=$2 remaining
  shift 2
  remaining=$((GATE_STARTED + 4500 - $(date +%s)))
  if [ "$remaining" -le 0 ]; then ./start.sh stop; return 124; fi
  if [ "$remaining" -lt "$budget" ]; then budget=$remaining; fi
  systemd-run --user --wait --pipe --unit="$CTN-$label" --property="RuntimeMaxSec=$budget" \
    --working-directory="$RELEASE_DIR" --setenv="RECIPE_HOSTS=$RECIPE_HOSTS" "$@" \
    > "$RECEIPTS/$label.log" 2>&1 || { ./start.sh stop; return 1; }
}
```

Payloads below use `phase stress 600`, `phase needles 300`, `phase qeval 900`, `phase dash 1140`, and
`phase rigmark 840`. Prefix env assignments with `/usr/bin/env` inside the wrapper (for example
`phase dash 1140 /usr/bin/env SPARKDASH_API="$DASH_API" python3 bench/sparkdash.py full 165312`). The wrapper's
phase log is the complete collector output. Set `RECEIPTS` outside the checkout. Boot remains capped at 420 seconds,
including admission; use journal admission timestamps and abort via the owned launcher if that cap is exceeded.

```bash
python3 scripts/release_1006_probe.py stress --execute --boot "$CTN" --base "$BASE" \
  --out "$RECEIPTS/stress"
NEEDLE_TARGETS=16384,160000 python3 bench/long_retrieval_v2.py --execute \
  --endpoint "$BASE" --tokenizer-dir "$MODEL_DIR" --boot-label "$CTN" --out "$RECEIPTS/needles"
python3 bench/qeval_window.py --fleet-window --url "$BASE/v1/chat/completions" \
  --label "$CTN" --meta "$RECEIPTS/metadata.json" --out "$RECEIPTS/qeval" --budget-s 900 --timeout 45
python3 bench/nvfp4_more_gate.py qeval "$RECEIPTS/qeval" --out "$RECEIPTS/qeval-gate.json"
SPARKDASH_API="$DASH_API" python3 bench/sparkdash.py full 165312 > "$RECEIPTS/sparkdash.jsonl"
test "$(git -C "$RIGMARK_DIR" rev-parse HEAD)" = d8353e93b274e8d880ab14a5df2a55c87d7bee16
test -z "$(git -C "$RIGMARK_DIR" status --porcelain)"
python3 "$RIGMARK_DIR/rigmark" run --base-url "$BASE" --model GLM-5.3 --label "$CTN" \
  --comparison-id "$CTN-rigmark" --metadata "$RECEIPTS/metadata.json" \
  --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --prefill-depths 8192 --output "$RECEIPTS/rigmark.json"
```

Write `metadata.json` before probing with the recorded commit, boot/container/image identities, layout selectors,
source/checker hashes and no credentials/site addresses in its publishable copy. The qeval task checker SHA256 is
`54719522d26996198c870264dfe5a93e2dd23f33436626c2477f1ac71206ffd2`; the recipe source differs from the operational
Flash source only in its docstring/model alias. Raw responses, reasoning, finish reasons, all 225 tasks and repeated
failures stay in the private receipts. Require mean >=70.553/75, c1, no request failures and no new recurring quality
regression against boot-e receipts; retain `reason_r10` truncation and the recorded failures as the reference.

Stress must observe four resident decoding streams, four distinct 40832-token prompts +1024 outputs each, retained
APC and zero preemptions; then one 164288-token prompt +1024 outputs. Require rank-0 stressed minimum >=**8.4 GiB**,
all ranks above guard floor and no swap/error trip. This qualifies the shared pool, not four maximum contexts.
Needles run twice at 16K and 160K: every long score >= paired 16K score-1, all seven registry fields correct,
finish_reason=stop, no transport errors. sparkDash requires prose/code c1 x5 and its full c1/c2/c4/c8 sweep
(plus prose c3), all streams complete; separate cold-prefill table. RigMark must identify protocol 1.0.0, thinking
on/effort low, and all English prose/code/structured completion gates pass; save c4 code and cold/warm 8K prefill.

## Optional coalesced arm and final handoff

Choose the arm **before** this single boot. Opt-in config: `GLM_LOADER=coalesced`, `GLM_PARAM_HASH=1`,
`VLLM_SERVER_DEV_MODE=1`. Keep the same 165312/no-recent/NVFP4 vector. A fast-loader manifest from the exact same
weight layout and image is required; the old attention-only manifest is insufficient. If unavailable, use default
fast and leave coalesced TBD-gate. No second boot and no automatic enablement. Reserve two minutes from slack for
the existing `glm_phash_run` RPC (32 MiB chunks, floor >=8.05, 90 s maximum) on all ranks. Run this payload in a
120-second `phase hashes` unit with `BASE`, `CTN`, `FAST_MANIFEST_DIR` and `RECEIPTS` exported:

```python
import json, os, pathlib, urllib.request
base=os.environ['BASE']; tag=os.environ['CTN']
opts=json.dumps(dict(chunk_mb=32, segment_mb=256, threads=6, max_seconds=90, mem_floor_gib=8.05, attrs=True))
request=urllib.request.Request(base+'/collective_rpc', data=json.dumps(dict(method='glm_phash_run', args=[tag,opts])).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request, timeout=95) as response: rows=json.load(response)['results']
pathlib.Path(os.environ['RECEIPTS'],'hash-rpc.json').write_text(json.dumps(rows,indent=2))
assert len(rows)==4 and sorted(r.get('tp_rank') for r in rows)==list(range(4))
for row in rows:
    assert row.get('done') is True and not row.get('error')
    ref=json.loads(pathlib.Path(os.environ['FAST_MANIFEST_DIR'],f"rank{row['tp_rank']}.json").read_text())
    assert row['digest_params_buffers']==ref['digest_params_buffers']
```

Copy each returned manifest path from its recorded read-only diagnostic cache mapping into private receipts;
compare parameters/buffers with `glm_param_hash.compare`, runtime scratch separately. Require every verdict EQUAL.
A partial hash is INCOMPLETE. If no slack remains, this optional arm is not qualified.

At <=75 minutes validate every receipt against the same boot, collect guard minima/swap/error summaries and
repeat health/exact OK/APC. Only then fill README's 165K TBD-gate cells and commit as knapcio to the private branch.
Hand the unchanged admitted clone to the existing persistent serving-watch/heartbeat, verify both active and fresh,
preserve the deployment token ownership. Write `handoff.json` with exact `ctn`, deployment `token`, `watch_active:true` and `heartbeat_verified:true` only after those checks; the guard exits normally on that receipt. A deadline or signal without it stops the deployment. Stop the temporary serve unit only after that ownership transfer.
Public publication and any larger context remain for the owner.
