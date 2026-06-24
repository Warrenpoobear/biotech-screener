# Hermes Contradiction Detector

**Role:** Route `HARD_CONTRADICTION` items from the knowledge layer to Town (Spec 090).

**Entry:** `agents/hermes-contradiction-detector/run_job.py`

**Inputs:** `artifacts/ops/knowledge_layer/latest_state.json` (`warnings` array)

**Outputs:** Town email via `send_operator_event(event_type=contradiction_detected)`

**Cadence:** After `tools/build_hermes_knowledge_layer.py` (on-demand / daily on operator host)

**Authority:** observe_only — read-only; no repo mutations

**LLM:** none (Lane A)

**Dry-run:** `OPERATOR_DELIVERY_DRY_RUN=1` (default) logs without sending email.

## Skills

| Skill | Use when |
|-------|----------|
| `town-operator-bridge` | Route HARD_CONTRADICTION to Town |
| `self-improving` | Recurring contradiction class → LRN |
