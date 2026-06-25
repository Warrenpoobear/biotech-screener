# Agent Fleet Architecture Index

**Purpose:** Top-level index for the three-tier agent-fleet documentation and shared governance invariants. This is the front door for understanding the fleet structure, authority boundaries, and self-improvement workflow.

**Status:** Three-tier fleet documentation complete and codegraph-validated (2026-05-26).

---

## Fleet Architecture Overview

The biotech-screener agent fleet is organized into **three tiers**, each with distinct roles, authority levels, and governance constraints:

### Layer A: Data Ingestion (Deterministic, Premarket)
**Read:** [`docs/DATA_INGESTION_AGENT_MAP.md`](DATA_INGESTION_AGENT_MAP.md)

- **Role:** Polling external APIs, fetching market data, normalizing, deduplicating, staging for pipeline ingestion
- **Agents (6):** company_news_ingest (deprecated), ctgov_poller, earnings_calendar_sync, herald, aact_trial_ingest, universe_maintenance
- **Cadence:** Daily premarket (14:00 ET) or weekly (Monday 10:00 ET)
- **Authority:** `write_artifacts` or `observe_only` (deterministic, no LLM variability)
- **Key constraint:** Staging-artifact model — all outputs written to `artifacts/`, `cache/`, `data/press_releases/` only; NO direct production_data mutations
- **LLM policy:** herald classification is deterministic-deterministic split (fetch → classify, separate cached steps)

### Layer B: Signal Monitors (Anomaly/Watch, Post-Production)
**Read:** [`docs/SIGNAL_MONITOR_AGENT_MAP.md`](SIGNAL_MONITOR_AGENT_MAP.md)

- **Role:** Continuous monitoring, anomaly detection, decision-support context, no ranking/scoring/execution mutations
- **Agents (8):** biotech_news_digest, catalyst_delta, price_action_watch, options_watch, review_queue_steward, ic_health_monitor, intraday_mover_watch, grok_biotech_watch
- **Cadence:** Staggered post-production (18:30–21:30 ET) or continuous
- **Authority:** `observe_only` (all)
- **Key constraint:** Lane B policy — deterministic pre-filter first, LLM invoked only on anomaly detection; routine days may produce zero LLM calls
- **LLM policy:** Direct_llama_on_anomaly — thresholds trigger interpretation, not cadence

### Layer C: Control Plane (Diagnostics, Triage, Proposal)
**Read:** [`docs/CONTROL_PLANE_AGENT_MAP.md`](CONTROL_PLANE_AGENT_MAP.md)

- **Role:** Health assessment, drift detection, quality assurance, fleet coordination, operational verdicts
- **Agents (7):** data_auditor, fleet_steward, ops, ops_supervisor, production_qa, qa, sentinel
- **Cadence:** Staggered post-production (18:30–20:30 ET)
- **Authority:** `observe_only` (3) + `observe_and_propose` (4) — no auto-execution
- **Key constraint:** Proposal-first governance — all changes proposed to operator; no agent mutations of production state
- **Suppressed-agent ledger:** Explicit reasons for each suppression (shadow_watch, company_news_ingest, bioshort_watch); documented in supervisor.py:121–125

---

## Shared Invariants

**All three tiers enforce these principles:**

### 1. Registry-Driven Discovery
- Agent identity sourced from `agents/AGENT_REGISTRY.json` (canonical source of truth)
- Role, cadence, authority, status defined in registry
- No agent hardcoding or duplication in code

### 2. Explicit Suppression Reasons
- Suppressed agents listed with documented reasons in `ops_supervisor/supervisor.py:SUPPRESSED_AGENTS`
- Reasons must cite Spec disposition (e.g., "Spec 085 Path C 2026-05-06")
- Reactivation requires new spec (cannot resume without specification)

### 3. Staging-Artifact Model
- **Data-ingestion:** All outputs written to `artifacts/`, `cache/`, or `data/` (staging)
- **Signal-monitors:** All outputs written to `artifacts/{monitor_name}/` (informational)
- **Control-plane:** All outputs written to `artifacts/{agent_name}/` (reports only)
- **Never:** Direct writes to `production_data/`, `production_snapshots/`, or ranking/scoring/sizing surfaces

### 4. Deterministic Prefilters
- **Data-ingestion:** Schema validation, deduplication, staleness checks, PIT filtering
- **Signal-monitors:** Threshold-based anomaly gates (>3σ moves, signal IC drift >0.05, option IV spikes)
- **Control-plane:** Contract tests, QA checks, drift thresholds
- **No random selection or external state mutations** (idempotent, repeatable)

### 5. Safe I/O Wrappers & Atomic Writes
- All file operations use `safe_mkdir_with_error()`, `safe_file_write()`, `safe_json_read/write()`
- Atomic write pattern: temp file + `os.replace()` (prevents partial writes / corruption)
- Permission enforcement: umask 0o077, file mode 0o600
- Error classes: `FileOperationError`, `FileReadError`, `FileWriteError`, `DirectoryError`

### 6. Authority Boundaries
**Agents CAN:**
- Read all production data, agent outputs, market data
- Write to designated artifact/cache/memory directories
- Detect and classify anomalies
- Propose improvements via Town/OpenClaw workflow
- Flag issues and suggest next actions

**Agents CANNOT:**
- Mutate `production_data/`, `production_snapshots/`, or decision_rulesets/
- Change ranking weights, scoring coefficients, or model behavior
- Execute trades or rebalance portfolios
- Auto-trigger portfolio changes
- Modify rulesets or selector/ranker parameters
- Bypass the proposal-first governance model

### 7. Proposal-First Governance for All Changes
- Changes to agent behavior, anomaly thresholds, or outputs require Spec or proposal
- Use `docs/templates/SELF_IMPROVEMENT_PROPOSAL.md` for suggestions
- Check `docs/FAILURE_PATTERN_LIBRARY.md` for known failure modes before proposing
- No automatic promotion or recursive self-improvement

---

## Relationship to Town/OpenClaw Governance

**All three tiers feed into the Town/OpenClaw proposal workflow:**

1. **Data-ingestion agent change** → Submit `SelfImprovementProposal` (new source, schema, frequency)
2. **Signal-monitor enhancement** → Submit proposal (new anomaly type, LLM context, threshold)
3. **Control-plane improvement** → Submit proposal (new health check, verdict rule, suppression reason)
4. **Operator reviews** → OpenClaw preflight validates (authority, scope, safety)
5. **Approved** → New spec issued; implementation separate ticket

**Reference documents:**
- `docs/TOWN_OPENCLAW_AGENT_FLOW_CONTRACT.md` — agent communication and authority model
- `docs/FAILURE_PATTERN_LIBRARY.md` — known failures and recovery patterns
- `docs/templates/SELF_IMPROVEMENT_PROPOSAL.md` — proposal template and acceptance criteria

---

## Runtime Identity Caveat

**Important:** This documentation captures **architectural design and governance policy**, not runtime identity.

- Registry and wrapper docs establish **intended** authority and responsibility
- **True agent identity** is determined by:
  - Local `SOUL.md` / `HEARTBEAT.md` / `AGENTS.md` files in each agent workspace
  - Real Hermes runtime validation (if running via Hermes platform)
  - Actual cron entries and heartbeat signals

**Do NOT synthesize agent identity from this documentation alone.** Use this as a map of the governance layer, not as ground truth about what any agent is actually doing right now. For live identity verification, run:

```bash
python tools/agent_heartbeat_checks.py --agent <name>  # Check live heartbeat
python3 tools/fleet_ops_status.py --write              # Operator triage snapshot
python3 tools/fleet_completion_audit.py              # Verify deterministic cron wiring
python tools/run_agent_direct.py --agent <name> --message "IDENTITY_CHECK"  # Lane C only
```

---

## Validation Summary

**Codegraph validation (2026-05-26) confirms alignment between documentation and implementation:**

| Dimension | Finding | Status |
|-----------|---------|--------|
| **Agent discovery** | `AGENT_REGISTRY.json` + `SPECIALIZED_CHECKS` dispatch matches registry-driven model | ✅ Clean |
| **Suppression tracking** | `SUPPRESSED_AGENTS` dict (supervisor.py:121–125) contains explicit Spec dispositions | ✅ Clean |
| **Control-plane verdict** | `ValidationReport` class with flag aggregation matches triage logic | ✅ Clean |
| **Data-ingestion staging** | Atomic writes, cache validation, PIT filtering match staging-artifact model | ✅ Clean |
| **Signal-monitor policy** | Threshold pre-filters + conditional LLM match Lane B implementation | ✅ Clean |
| **Authority boundaries** | Safe I/O wrappers and path constraints match isolation model | ✅ Clean |
| **Audit trail** | `AuditLog` with deterministic serialization present | ✅ Clean |

**Conclusion:** Three-tier fleet documentation is **high-fidelity** representation of production code. No discrepancies detected.

---

## Index: Three Agent-Fleet Maps

Use this table to navigate the full architecture:

| Map | Layer | Purpose | Agents |
|-----|-------|---------|--------|
| [DATA_INGESTION_AGENT_MAP](DATA_INGESTION_AGENT_MAP.md) | A (Premarket) | Polling, fetching, normalizing, staging artifacts | company_news_ingest (deprecated), ctgov_poller, earnings_calendar_sync, herald, aact_trial_ingest, universe_maintenance |
| [SIGNAL_MONITOR_AGENT_MAP](SIGNAL_MONITOR_AGENT_MAP.md) | B (Post-production, continuous) | Monitoring, anomaly detection, decision-support context | biotech_news_digest, catalyst_delta, price_action_watch, options_watch, review_queue_steward, ic_health_monitor, intraday_mover_watch, grok_biotech_watch |
| [CONTROL_PLANE_AGENT_MAP](CONTROL_PLANE_AGENT_MAP.md) | C (Post-production) | Health assessment, triage, coordination, proposals | data_auditor, fleet_steward, ops, ops_supervisor, production_qa, qa, sentinel |

---

## Non-Goals

This documentation does **not** authorize or enable:

- Live agent changes (use Spec + proposal workflow)
- Authority escalation (observe_only remains observe_only)
- Automatic portfolio actions (all proposal-first)
- LLM policy changes (Lane B gates remain in place)
- Production-data mutations by agents (staging-artifact model enforced)
- Self-improvement without spec approval (proposal-first frozen)

---

## Deterministic Cron Consolidation (2026-06)

Phases 2–9 retired scheduled `run_agent_direct` from production cron and added operator triage artifacts. **Phase 10 (code-complete):** completion audit runs before `fleet_ops_status --write` so `status.json` embeds `registry_coverage`; weekly digest surfaces registry coverage from `completion_audit.json`.

| Phase | Focus | Status |
|-------|-------|--------|
| 2–3 | Outcome feedback, herald recovery, evening catchup builders | Code-complete |
| 4–5 | Artifact escalation, fleet_ops, ops_supervisor, Rule 12 gates | Code-complete |
| 6 | Watchdog recovery, digest integration, skillpatch defense | Code-complete |
| 7 | `fleet_completion_audit.py` + operator runbook | Code-complete |
| 8 | Host onboarding checklist + Telegram fleet surface | Code-complete |
| 9 | Registry coverage audit + daily completion artifact | Code-complete |
| 10 | Audit→fleet_ops ordering + digest registry coverage + wiring contract | Code-complete |
| 11 | Live crontab verify vs install reference (`fleet_crontab_verify.py`) | Code-complete |
| 12 | Crontab verify in evening catchup, digest, Telegram | Code-complete |
| 13 | Watchdog herald health + F-2026-005 recovery | Code-complete |
| 14 | Migration closure contract + Rule 12 host onboarding | Code-complete |
| 15 | Unified host onboarding script + docs/skills sweep | Code-complete |

**Migration arc (phases 2–15) is code-complete.** Host execution: `bash tools/run_fleet_host_onboarding.sh`

Operator reference:

| Concern | Tool / script |
|---------|----------------|
| Host onboarding (one command) | `bash tools/run_fleet_host_onboarding.sh` |
| Install WSL crontab | `bash tools/install_agent_fleet_crontab.sh` |
| Verify live crontab | `python3 tools/fleet_crontab_verify.py --write` |
| Evening safety net | `tools/cron_evening_catchup.sh` |
| Missed production / monitoring | `tools/cron_watchdog.sh` |
| Wiring verification (run first) | `python3 tools/fleet_completion_audit.py --write` |
| Fleet triage (run after audit) | `python3 tools/fleet_ops_status.py --write` |
| Host onboarding | `bash tools/run_fleet_operator_checklist.sh` |
| Registry coverage | `artifacts/fleet_ops/{date}_completion_audit.json` |
| Rule 12 self-improve gate | `docs/governance/RULE_12_PROMOTION_CHECKLIST.md` |

Host blockers **F-2026-005** (Herald) and **F-2026-006** (CI) must close before `SELFIMPROVE_GATES_MET=1`. Crontab install on WSL host remains operator action.

---

## Next Steps for Improvements

If you identify an issue or improvement in the fleet:

1. **Check failure patterns first:** `docs/FAILURE_PATTERN_LIBRARY.md`
2. **Draft proposal:** Use `docs/templates/SELF_IMPROVEMENT_PROPOSAL.md`
3. **Submit via Town/OpenClaw:** See `docs/TOWN_OPENCLAW_AGENT_FLOW_CONTRACT.md`
4. **Await operator approval:** No agent has `mutate_config` authority
5. **Spec issued + implemented:** Once approved, work proceeds via separate spec ticket

---

**Document version:** 2026-05-26  
**Validation:** Codegraph (2026-05-26, no discrepancies)  
**Status:** Architecture documented, governance established, ready for operational use  
**Next review:** Upon any agent proposal or governance change
