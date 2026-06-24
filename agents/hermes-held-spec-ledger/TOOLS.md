# TOOLS.md — Hermes Held Spec Ledger

## Run job

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
export PYTHONPATH=.

export OPERATOR_DELIVERY_DRY_RUN=1
python3 agents/hermes-held-spec-ledger/run_job.py
```

## Prerequisites

Build knowledge layer first when ledger is stale:

```bash
python3 tools/build_hermes_knowledge_layer.py
```

## Input / output

- **Input**: `artifacts/ops/held_spec_ledger/latest.json`
- **Env**: `TOWN_EMAIL`, `OPERATOR_DELIVERY_DRY_RUN`
