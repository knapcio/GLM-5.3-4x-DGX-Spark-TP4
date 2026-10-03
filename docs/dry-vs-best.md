# Release launch-vector inspection

Generate the configured release vector locally:

```bash
DRY=1 ./start.sh serve
```

It must select full GLM-5.3 native MTP, K-stop, uniform K2, short DSA, APC,
required dispram KV, c2 reuse and spec-sample off. Pad hygiene uses the gate-selected
value **1 (descriptor-scoped v2)**. The launcher refuses dispram vectors
without the required copy guard; [runtime](runtime.md) describes that prerequisite.

Keep raw vectors with private gate receipts. Public tables identify their gate
source through 0f24383 and W4 v2 primary (pad=1, spec-sample=0). Private addresses, machine
names and filesystem layout are not public release data. Earlier vector evidence
and validation narrative are archived in [history](history.md).
