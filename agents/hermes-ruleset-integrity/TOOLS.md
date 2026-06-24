# TOOLS.md — Hermes Ruleset Integrity

## Run job

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
export PYTHONPATH=.

export OPERATOR_DELIVERY_DRY_RUN=1
python3 agents/hermes-ruleset-integrity/run_job.py
```

## Input / output

- **Reads**: `production_data/decision_rulesets/manifest.json`, pinned ruleset JSON
- **Env**: `TOWN_EMAIL`, `OPERATOR_DELIVERY_DRY_RUN`
