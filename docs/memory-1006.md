# Release memory decision

The measured release is now **262144 context /4135 blocks /264640 KV tokens**, with a 6,318,718,976-byte
ordinary head per rank. Boot H passed stress at 6.69 GiB rank-0 minimum and a 40.9-minute soak with no errors
or preemptions. Ballast showed no observed harm down to 3.5 GiB under load. Floors: live 5.5, stress/capture 6.0,
admission 6.5 for 60 s, pre-capture 7.5 GiB. [Completed gate](../GATE-RESULT.md).

## Historical offline projection (superseded by fleet measurements)

Frozen memlab `6e7cdedab0565daf1dcb08bd9742bb5c874ff186`, LOO MAE 0.265286 GiB; empirical allowance 0.787713 GiB.
Selectors: FP4x, attn/shared/dense/mtp, no recent bank, graphs [1,4,12,16], four slots, LL128 on, zero page-cache credit.

| Context | Ordinary head | rank0 admission lower GiB | rank0 stressed lower GiB | Scope |
|---|---:|---:|---:|---|
| 165312 | 3 GiB | 10.131957 | 9.062957 | Within frozen measured range |
| 182080 | 3.5 GiB | 9.613904 | 8.544904 | Outside measured admission range; projection only |
| 186752 | 3.638671875 GiB (3906994176 B) | 9.470202 | 8.401202 | Largest projected head on 2 MiB grid clearing rank0 stress >=8.4 |

The next 2 MiB head fails the 8.4 criterion. Expanded heads remain REFUSED by memlab because they are outside its measured range. No new admission authority follows from a projected lower bound. At this historical checkpoint the default remained 165312 pending fleet proof; the completed gate above supersedes it.

The 2048 recent reservation costs 986.2 MiB/rank even with INIT=0. Setting WINDOW=0 removes it; the runtime still permits an explicit opt-in reservation. The retained 165K layout changes no KV head bytes or prefill capacity. Its recorded boot-g stressed minimum was 8.42 GiB with the reservation; the predicted freed margin is not a measurement.

Reproduce from the frozen memlab branch (CPU only):

```bash
python3 scripts/memlab.py predict --head-gib 3 --groups attn,shared,dense,mtp --recent-window 0 --max-model-len 165312 --out memlab-165312.json
python3 scripts/memlab.py predict --head-gib 3.5 --groups attn,shared,dense,mtp --recent-window 0 --max-model-len 182080 --out memlab-182080.json
```

The largest projection sweeps aligned head bytes from 3 GiB upward, calling the same `memlab.predict` and stopping when its rank0 `per_rank_lower_GiB` falls below 8.4. Saved vectors are in `docs/results/memlab-*.json`.

Frozen source SHA256: `6365a046e9dbaf745750cddbb3f32cb085f68f67fdbed7c7a238db8a79df266a` (`scripts/memlab.py`).
Frozen data SHA256: `63157a89ca5064415d7a1ceb18924506f44a0535e9de53a7a4122c4c1e364853` (`docs/memlab/data.json`).
