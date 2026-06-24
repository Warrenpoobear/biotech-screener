# TOOLS.md — Hermes Contradiction Detector

## Run job

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
export PYTHONPATH=.

# After knowledge layer build
python3 tools/build_hermes_knowledge_layer.py
python3 agents/hermes-contradiction-detector/run_job.py --from-build

# Standalone
export OPERATOR_DELIVERY_DRY_RUN=1
python3 agents/hermes-contradiction-detector/run_job.py
```

## Input / output

- **Input**: `artifacts/ops/knowledge_layer/latest_state.json`
- **Env**: `TOWN_EMAIL`, `OPERATOR_DELIVERY_DRY_RUN`
