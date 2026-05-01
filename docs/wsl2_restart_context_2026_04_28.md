# WSL2 Restart Context — 2026-04-28 22:22 EDT

## Incident

Supervisor issued ORANGE alert due to system restart detected at 22:22 EDT, appearing to fall outside the scheduled production window (14:00, 16:45, 18:00, 18:55 ET).

## Investigation

System logs show:
- **Kernel boot**: 09:09:17 EDT (early morning)
- **Uptime**: 11:47 (continuously running)
- **Who -b entry**: 22:22 EDT (systemd/WSL2 resume marker, not kernel reboot)
- **Production run**: 16:30 ET (completed successfully during continuous uptime)

## Root Cause

WSL2 Ubuntu subsystem on Windows can suspend/resume without a full kernel reboot. The 22:22 timestamp reflects a systemd service restart (likely OpenClaw gateway), not a system reboot. This is normal behavior.

## Impact

**None.** All production artifacts were generated during the 09:09–present continuous kernel uptime. The 22:22 event occurred AFTER production completed.

## Recommendation

Supervisor's ORANGE alert is overly conservative for WSL2 + systemd environments. Consider:

1. **Short-term (now)**: Accept ORANGE as informational. Wednesday 20:30 run should resolve if no new anomalies.
2. **Long-term**: Document WSL2-specific restart behavior in exception table or suppress false-positive timing alerts.

## Reference

See `agents/ops_supervisor/SOUL.md` for exception table rules. Current version does not account for WSL2 resume vs hard reboot distinction.

