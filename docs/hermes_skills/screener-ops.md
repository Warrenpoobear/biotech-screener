---
name: screener-ops
triggers:
  - daily pipeline
  - production run
  - screener operations
  - Hermes knowledge layer
  - Town-Hermes bridge
  - OpenClaw fleet
  - agent fleet
  - spec lifecycle
  - cron
  - ops supervisor
  - heartbeat
  - herald pipeline
  - export contract
  - backfill tooling
description: >
  Daily production operations, Hermes knowledge layer (Spec 089), agent fleet
  monitoring, and spec/governance lifecycle for the Wake Robin biotech screener.
  13-step pipeline orchestrator, four monitoring layers, anomaly classification
  (new/carried/resolved), Town-Hermes bridge architecture, OpenClaw fleet model
  config, SOUL.md ruleset system, spec states and active spec registry.
---

# Screener Ops & Governance Skill

## Purpose

Reference for daily production operations, the Hermes knowledge layer, agent fleet monitoring, and the spec/governance lifecycle that governs all changes to the biotech screener.

This skill is organized into two sections:

1. **Framework Reference** — Stable pipeline architecture, processes, and governance
2. **Operational State** — Volatile infrastructure status snapshots

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Daily Production Pipeline

**Runner**: `tools/run_daily_production.py` (13-step orchestrator)
**Cron**: 5:30 PM ET weekdays + `@reboot` catch-up for missed runs
**Timeout**: 6000s (100 min) — covers worst-case AACT + tail steps

### Pipeline Steps (in order)

1. Price refresh
2. Cache warm (including FDA)
3. Screen (with `--inputs-manifest write`)
4. Audit
5. Gates
6. Manifest + promotion
7. Drift report
8. Action packet
9. Shadow portfolio
10. Trade plan
11. Portfolio report
12. Readiness scorecard
13. Ops digest + PIT backfill (optional)

**Key Rule**: Always warm 8-K cache BEFORE running screen.

---

## Hermes Knowledge Layer (Spec 089)

**Generator**: `tools/build_hermes_knowledge_layer.py`

Repo-native "ops brain" that continuously answers:

1. What is the current operational state?
2. What changed since the last good state?
3. What is held, blocked, or awaiting first-fire validation?
4. What contradictions exist across specs, audit memos, cron, and registry?
5. What is the next allowed operator action?
6. What is explicitly not allowed?

### Four Layers

| Layer | Purpose | Output |
|-------|---------|--------|
| Capture | Read-only from specs, artifacts, registry, git, cron | Raw state |
| Normalize | Structured ledgers | `artifacts/ops/knowledge_layer/` |
| Reason | Drift, contradiction, missed-run detection | Alerts |
| Deliver | Operator briefs | Daily/weekly summaries |

### Output Artifacts

| Artifact | Location |
|---------|----------|
| Latest state | `artifacts/ops/knowledge_layer/latest_state.{json,md}` |
| Held spec ledger | `artifacts/ops/held_spec_ledger/latest.{json,md}` |
| First fire ledger | `artifacts/ops/first_fire_ledger/latest.{json,md}` |
| Contradiction ledger | `artifacts/ops/contradiction_ledger/latest.md` |
| Operator briefs | `artifacts/ops/operator_brief/daily/YYYY-MM-DD.md` |

---

## Town-Hermes Bridge (Spec 090)

**Module**: `common/operator_delivery.py`

Routes Hermes Knowledge Layer events to Town via email trigger. Town does NOT control Hermes.

### Architecture

```
Hermes job completes
  -> write ledger artifact (repo)
  -> send_operator_event(channel="town", ...)
    -> structured email to TOWN_EMAIL (djschulz@gmail.com)
    -> Town routine triggers on [Hermes] subject prefix
    -> Town creates task / DMs operator
```

### What Town is NOT

- NOT a scheduler or cron controller
- NOT a repo mutator or spec approver
- NOT allowed to reactivate bioshort_watch LLM
- NOT the authoritative source for any production state

---

## OpenClaw Agent Fleet

### Agent Registry

**File**: `agents/AGENT_REGISTRY.json`

### Model Configuration

- **Primary**: `deepseek/deepseek-v4-flash:free` (OpenRouter) — fleet-wide migration 2026-05-20; 27 agents migrated
- **Fallback**: Anthropic Claude SDK (for Claude-specific models)
- **Auto-routing**: "deepseek" models → OpenRouter (OpenAI-compatible), "claude" → Anthropic SDK

### Inference Tuning

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Frequency penalty | 0.1 |
| Top_p | 0.95 |
| Repetition penalty | 1.2 |
| API timeout | 2400s |
| Retry strategy | Exponential backoff (500ms–8000ms) |

### Uncertainty Handling (all agents)

- ops_supervisor: missing artifacts → RED (not GUESS); confidence < 0.7 → escalate
- sentinel: missing drift → FAIL; boundary cases → WARN
- data_auditor: missing snapshot → FAIL; specific ticker counts (not "some")
- ic_health_monitor: missing dashboard → UNKNOWN; threshold boundaries → ALERT (conservative)
- fleet_steward: unreachable status → MEDIUM; missing last_run → anomalous

### Monitoring Layers

| Layer | Tool | Purpose |
|-------|------|---------|
| Heartbeat | `tools/agent_heartbeat_checks.py` | Per-agent health |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly classification |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline task orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
|---------------|----------|---------|
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops_supervisor) are intentionally unsupervised — no HEARTBEAT.md.

### Herald Pipeline

Done predicate requires BOTH:
- `data/press_releases/deduped/deduped_{date}.jsonl`
- `data/press_releases/classified/classified_{date}.jsonl`

If classification failed but dedupe exists, next supervisor run retries classification.

---

## SOUL.md / Ruleset System

**SOUL.md**: Per-agent operating manual defining boundaries, tools, and heartbeat checks. Located in `agents/{name}/SOUL.md`.

**Ruleset Health Monitor** (`tools/ruleset_health_monitor.py`): JSONL history grows with each evaluation date. Tracks consecutive WARN days by active ruleset ID. Recommends rollback after sustained degradation.

---

## Spec Lifecycle

### Spec States

| State | Meaning |
|-------|---------|
| DRAFT | Under development |
| IN PROGRESS | Active work, phased |
| HELD | Blocked on dependency |
| RESOLVED | All acceptance criteria met |
| SUPERSEDED / MITIGATED | Failure modes neutralized via different route |
| CLOSED | Formally closed |

Each spec has: acceptance criteria with section references, phase gates (A/B/C/D), blocking dependencies, closure memos in `artifacts/audit/`.

### Held-Spec Ledger

Tracks all specs that are held/blocked with: what is held and why, first-fire validation status, alert deadlines, next operator action.

---

## Export Contract Registry (Spec 101)

Status: CLOSED (commits eaa4ea87 + cba4ee0f). `ev_severity_score` now exported.

**Runway severity fields exported (post-Spec 101):**
`runway_severity_score`, `ev_severity_score`, `runway_buffer_months`, `financing_truth_gate`, `dilution_haircut`, `size_multiplier`, `severity_bucket`, `severity_notes`

`check_severity_formulas()` QA validation runs on every snapshot.

---

## Backfill Tooling (Spec 102)

Research-enablement tooling for backfilling expectation fields into historical snapshots.

Target fields: `short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct` (required); `insider_net_buy_value_90d` (optional)

Key rules:
- Default: additive-only (`recompute=False`). Original ranks/actions preserved.
- Every backfill emits a structured manifest
- `_backfill_version` metadata column added to all backfilled snapshots (null for originals)
- Research scripts must filter on `_backfill_version` to avoid silent pre/post mixing

---

## Source Files

| Component | File |
|----------|------|
| Daily Production Runner | `tools/run_daily_production.py` |
| Knowledge Layer Builder | `tools/build_hermes_knowledge_layer.py` |
| Operator Delivery | `common/operator_delivery.py` |
| Agent Heartbeat Checks | `tools/agent_heartbeat_checks.py` |
| Ops Supervisor | `agents/ops_supervisor/supervisor.py` |
| Post-Snapshot Supervisor | `tools/run_post_snapshot_supervisor.py` |
| Ruleset Health Monitor | `tools/ruleset_health_monitor.py` |
| Ops Digest Builder | `tools/build_ops_digest.py` |
| Readiness Scorecard | `tools/weekly_readiness_scorecard.py` |
| Cron Wrapper | `tools/cron_daily_production.sh` |
| Agent Registry | `agents/AGENT_REGISTRY.json` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** — Verify against current pipeline or infrastructure before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-24*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Architecture freeze**: ACTIVE — h20d DEFERRED (Path B, 2026-05-24)

## Governance Freeze Status

*Last reviewed: 2026-05-24*

- **Architecture Freeze**: ACTIVE — no selector/ranker/sizing/KG changes
- **13F Q1 2026 Quarantine**: ACTIVE — Jaccard 0.364 (gate ≥ 0.70); attribution-only
- **Phase 2 Step 5 (KG gating)**: blocked on quarantine clearance + h20d decision
- Gate results: `artifacts/audit/13f_q1_2026_refresh_gates_2026_05_24.md`

## Infrastructure

*Last reviewed: 2026-05-25*

- **Platform**: WSL2 on Windows host
- **Agent model**: `deepseek/deepseek-v4-flash:free` (OpenRouter) since 2026-05-20
- Daily cron: 5:30 PM ET weekdays
- Sleep-cliff risk: Windows host suspend kills crons silently
- Stopgap: `powercfg /change standby-timeout-ac 0`

### Cursor Cloud / CI / Hermes Notes

- Cursor Cloud agents need Python deps from `requirements.txt` plus `pytest-xdist` before running `run_screen.py` or pytest on main.
- GitHub Actions failures that say `The job was not started because an Actions budget is preventing further use.` are provider budget/quota blocks, not PR code failures.
- PR #304 is Track B draft/spec-test-only. Expected red fail-closed tests must not be made green without explicit governance clearance.
- Repo-native Hermes MCP can work in Cursor Cloud while production Hermes/Hermes Link runtime is absent; stale cloud knowledge artifacts require local/production refresh before triage.
