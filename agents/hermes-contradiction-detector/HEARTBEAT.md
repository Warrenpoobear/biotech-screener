# HEARTBEAT.md — Hermes Contradiction Detector

## Schedule

On-demand after `tools/build_hermes_knowledge_layer.py` or with `--from-build`.

## Checklist

- [ ] `artifacts/ops/knowledge_layer/latest_state.json` exists
- [ ] `python3 agents/hermes-contradiction-detector/run_job.py` exits 0
- [ ] HARD contradictions routed to Town when present

Reply `HEARTBEAT_OK` when no HARD contradictions or job completes cleanly.
