# HEARTBEAT.md — Hermes Ruleset Integrity

## Schedule

On-demand or post-snapshot when ruleset manifest changes.

## Checklist

- [ ] Active ruleset manifest readable (`production_data/decision_rulesets/manifest.json`)
- [ ] `python3 agents/hermes-ruleset-integrity/run_job.py` exits 0
- [ ] Integrity result routed to Town when FAIL

Reply `HEARTBEAT_OK` when validation PASS or expected dry-run INFO.
