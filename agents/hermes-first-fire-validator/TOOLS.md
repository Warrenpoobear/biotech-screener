# TOOLS.md — Hermes First-Fire Validator

## Run job

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
export PYTHONPATH=.

# Dry-run (default)
export OPERATOR_DELIVERY_DRY_RUN=1
python3 agents/hermes-first-fire-validator/run_job.py

# Live Town delivery
export OPERATOR_DELIVERY_DRY_RUN=0
python3 agents/hermes-first-fire-validator/run_job.py
```

## Input / output

- **Input**: `artifacts/ops/first_fire_ledger/latest.json`
- **Audit**: `artifacts/audit/first_fire_validations/`
- **Env**: `TOWN_EMAIL`, `OPERATOR_DELIVERY_DRY_RUN`
