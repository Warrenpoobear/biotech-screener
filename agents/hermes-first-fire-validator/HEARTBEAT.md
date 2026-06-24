# HEARTBEAT.md — Hermes First-Fire Validator

## Schedule

On-demand or post-snapshot when `artifacts/ops/first_fire_ledger/latest.json` updates.

## Checklist

- [ ] First-fire ledger exists and parses
- [ ] `python3 agents/hermes-first-fire-validator/run_job.py` exits 0
- [ ] Town bridge event constructed (dry-run or live per `OPERATOR_DELIVERY_DRY_RUN`)

Reply `HEARTBEAT_OK` when the job completes without FAIL events.
