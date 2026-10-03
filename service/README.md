# Persistent dispramd service and recovery

These units wrap the pinned, unmodified kindling AGPL lender described in
[installation](../docs/install.md). They are installation templates; this offline
package does not claim that any unit has been deployed or reboot-qualified.

| File | Role |
|---|---|
| `dispram-drm-lock.service` + `dispram-drm-lock.sh` | Root oneshot locks DRM nodes after udev and before Docker. Unlock refuses while lender/context/lease state is live or unverifiable. |
| `dispramd.service` | Runs the selected service user after Docker and the DRM lock, using `dispram.sh service` and fail-closed monitoring. |
| `sudoers-dispram` | Grants that same service user only the required unit-start command. |
| `install.sh` | Installs, enables and starts both units only with no GPU process; refuses unrendered user templates. |

## Installation

Install/fetch/build the wrapper and external lender at `/srv/glm-dispram` first.
Select an existing service account with Docker and system journal access. Render
`SERVICE_USER` in both `dispramd.service` and `sudoers-dispram` to that account.
The wrapper directory, launcher `DISPRAM_HOME`, unit environment and lock-helper
run directory must agree. Review the rendered units and sudo rule before install.

For the W4 layout, also add `Environment=DISPRAM_ALLOW_RM_ALLOC_OOM=1` to the
rendered lender's `[Service]` section. This is the exact handled allocation-line
exception documented in [runtime](../docs/runtime.md#display-carveout-kv).
Use `export DISPRAM_ALLOW_RM_ALLOC_OOM=1` for manual wrapper commands, and set
the same value in the launcher's private `.env`; otherwise the service and manual
checks would classify the same kernel history differently. Other fault checks
remain active.

```bash
sudo bash service/install.sh
export DISPRAM_ALLOW_RM_ALLOC_OOM=1
systemctl is-enabled dispram-drm-lock.service dispramd.service
systemctl --no-pager status dispram-drm-lock.service dispramd.service
/srv/glm-dispram/dispram.sh watch
/srv/glm-dispram/dispram.sh preborrow
```

Both units are enabled for the next boot. The DRM lock precedes Docker; the
lender itself is a Docker container and cannot start before Docker. Engines use
no automatic restart and must pass preborrow before a lease. Reboot survival
must be verified with the operator's gate receipts, not inferred from `enable`.

## Recovery

`Restart=no` and the lender container's no-restart policy are deliberate. On an
INVALID watch (daemon crash, Xid/SMMU, DRM use or unreadable monitoring), the
supervisor writes a per-boot invalid marker, stops borrowers and releases the
lender only after verified release. It then exits failed. A timeout or unknown
state retains the lender/DRM lock; do not force release.

1. Stop the owned serving deployment with `./start.sh stop`; preserve logs.
2. Check `dispram.sh status`, borrowers, CUDA contexts and the lease record.
   `dispram.sh postcheck` must confirm release. If the daemon is unavailable,
   retain the invalid state and use local administrative recovery/reboot rather
   than assuming that a missing response means an empty lease.
3. If the lender remains running, stop it only when `dispram.sh verified-free`
   succeeds. No borrower, CUDA context or lease may remain; unknown is a refusal.
4. Investigate the INVALID cause. A deliberate `dispram.sh reset-invalid` is
   allowed only after the daemon is stopped and GPU state is idle. The reset
   archives the marker and records the operator; it does not make a live lease safe.
5. Reset the failed unit and start only after the DRM lock and all idle checks
   are valid. Run watch and preborrow again before admitting a new borrower.

```bash
/srv/glm-dispram/dispram.sh reset-invalid
sudo systemctl reset-failed dispramd.service
sudo systemctl start dispram-drm-lock.service dispramd.service
/srv/glm-dispram/dispram.sh watch
/srv/glm-dispram/dispram.sh preborrow
```

For permanent rollback, stop borrowers, verify release, stop the lender, then
restore DRM through the guarded unlock helper. Never restart the lender under
live borrowers or delete lease/invalid records to bypass a refusal.

## Keep a qualified manual deployment running

For an already healthy manual lender with active borrowers, `dispram.sh monitor`
adopts monitoring without starting or restarting the daemon. Its failure path
matches the service: record INVALID, stop borrowers, verify release, then stop
the lender. Keep it independent of an interactive SSH session and use the same
`DISPRAM_ALLOW_RM_ALLOC_OOM=1` setting as the launcher.

The separate `dispram.sh deadman none` mode removes the test window's absolute
deadline while retaining shutdown on a stale or missing controller heartbeat.
`none` is for an explicitly selected serving deployment; numeric deadlines retain
their previous test behavior. Keep refreshing `DISPRAM_HOME/heartbeat` while the
launcher watchdog is active. Do not leave the old numeric-deadline process
running alongside the serving guard. Persistent units remain the documented
reboot setup; manual supervision alone does not establish reboot survival.
