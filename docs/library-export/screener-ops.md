# Screener Ops & Governance Skill

## Purpose

Reference for daily production operations, the Hermes knowledge layer, agent fleet monitoring, and the spec/governance lifecycle that governs all changes to the biotech screener.

This skill is organized into two sections.

## Operator Profile

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **Credentials**: CFA (CFA Institute), CAIA (CAIA Association). 30+ years institutional investment management.
- **Domain expertise**: Asset allocation, portfolio construction, manager research & selection, biotech equity research & due diligence, 13F/13D/13G filing analysis, clinical trial analysis, SEC EDGAR, derivatives/options.
- **Technical skills**: AI agent architecture & development (designed and built the Hermes/OpenClaw fleet), LLM prompt engineering, Python scripting, WSL2/cron administration, API integration.
- **Escalation authority**: All QUARANTINE, PRODUCER_AUDIT_REQUIRED, and spec approval decisions route to this operator. The operator is the sole authority for promotion approvals, spec closures, architecture changes, and pipeline governance decisions.
- **Town-Hermes bridge target**: Operator briefs and alerts deliver to djschulz@gmail.com (personal) and dschulz@wakerobin.co (work).
- **Wake Robin context**: Wake Robin is a real estate investment and community development company. The DEM biotech screener is a parallel investment research capability operated by the Director of Investments.

1. **Framework Reference** - Stable pipeline architecture, processes, and governance (changes only with code updates)
2. **Operational State** - Volatile infrastructure and status snapshots that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Daily Production Pipeline

**Runner**: `tools/run_daily_production.py` (13-step orchestrator)
**Cron**: 5:30 PM ET weekdays + `@reboot` catch-up for missed runs

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

### Key Rule

Always warm 8-K cache BEFORE running screen.

### Pipeline Timeout

6000s (100 min) to cover worst-case AACT + tail steps. Previous 4500s was killing mid-AACT on Mondays.

**Monday timeout rule (origin: failure F-2026-004, PT):** Monday runs ingest the weekend AACT batch and are the longest of the week — they are the binding case for this timeout. Keep dedicated monitoring on Monday pipeline duration, and validate ANY future timeout change against the trailing 4-week Monday duration distribution (not a single representative run). Treat a Monday run approaching the timeout as an ALERT even when mid-week runs finish comfortably.

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
| --- | --- | --- |
| Capture | Read-only from specs, artifacts, registry, git, cron | Raw state |
| Normalize | Structured ledgers | `artifacts/ops/knowledge_layer/` |
| Reason | Drift, contradiction, missed-run detection | Alerts |
| Deliver | Operator briefs | Daily/weekly summaries |

### Output Artifacts

| Artifact | Location |
| --- | --- |
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

- Schema v1.0, as-of 2026-05-17 (per agent_governance.md, authoritative source): 30 total agents — 27 active, 1 suppressed (bioshort_watch), 1 retired (company_news_ingest), 1 shadow (shadow_watch). Other documents may show older counts (17, 26, 27, 28) — always cite agent_governance.md with a dated reference.
- Authority levels: observe_only, observe_and_propose, write_artifacts, mutate_data, mutate_config
- Only crt_resolution_watcher holds mutate_data authority (writes to catalyst resolution tables under orchestrator supervision)

### Model Configuration (updated 2026-05-20)

- **Primary model**: DeepSeek v4 flash (Together AI) - all agents default to this
- **Fallback**: Anthropic Claude SDK (for Claude-specific models)
- **Auto-routing**: "deepseek" models -> Together API (OpenAI-compatible), "claude" -> Anthropic SDK
- **Previous**: Llama 3.3 70B Instruct Turbo (switched 2026-05-13 to 2026-05-20), OpenRouter (out of credits as of 2026-05-13)

### Inference Tuning (DeepSeek v4-optimized, 2026-05-20)

| Parameter | Value | Rationale |
| --- | --- | --- |
| Temperature | 0.2 | Stronger governance determinism |
| Frequency penalty | 0.1 | Reduce repetition loops |
| Top_p | 0.95 | Tighter nucleus sampling |
| Repetition penalty | 1.2 | Anti-loop guard |
| API timeout | 2400s | DeepSeek inference variance (Together can spike 8-12s cold) |
| Retry strategy | Exponential backoff | 500ms-8000ms delays |
| Compression threshold | 0.5 | Less aggressive for 131K context |

### Uncertainty Handling (all agents, 2026-05-13)

All agents tuned with explicit uncertainty escalation rules:

- ops_supervisor: missing artifacts -> RED (not GUESS); confidence < 0.7 -> escalate
- sentinel: missing drift -> FAIL; boundary cases -> WARN; ambiguous rollback -> both commands
- data_auditor: missing snapshot -> FAIL; specific ticker counts (not "some")
- ic_health_monitor: missing dashboard -> UNKNOWN; threshold boundaries -> ALERT (conservative)
- fleet_steward: unreachable status -> MEDIUM; missing last_run -> anomalous (not healthy)

### Monitoring Layers

| Layer | Tool | Purpose |
| --- | --- | --- |
| Heartbeat | `tools/agent_heartbeat_checks.py` | Per-agent health |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly classification |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline task orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
| --- | --- | --- |
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops_supervisor) are intentionally unsupervised and do not carry HEARTBEAT.md.

### Herald Pipeline

Done predicate requires BOTH deduped AND classified JSONL:

- `data/press_releases/deduped/deduped_{date}.jsonl`
- `data/press_releases/classified/classified_{date}.jsonl`

If classification failed but dedupe exists, the next supervisor run retries classification.

---

## SOUL.md / Ruleset System

### SOUL.md

Per-agent operating manual defining boundaries, tools, and heartbeat checks. Located in each agent workspace under `agents/{name}/SOUL.md`.

### Ruleset Health Monitor

**Tool**: `tools/ruleset_health_monitor.py`

- JSONL history grows with each new evaluation date (idempotent on same-day reruns)
- Tracks consecutive WARN days by active ruleset ID
- Recommends rollback after sustained degradation

## Governance Artifacts (PR #286, merged May 16, 2026)

### governance/AGENT_ROUTING_POLICY.md

Tier 0-4 routing policy classifying every part of the codebase by governance sensitivity. Defines allowed tools, review requirements, and merge rules per tier. The policy itself is Tier 4. Changes require a memo, not a direct edit.

### governance/STATUS.md

Enforcement status: AGENT_ROUTING_POLICY.md is live. Enforcement layers pending: agent_registry.yml (PR 2), AGENT_DIRECTORY_MAP.md, CI registry validation, import-graph validation.

### governance/HASH_ROTATIONS.md

Required landing zone for any Tier 3 production-hash rotation. Each entry requires: old hash, new hash, effective date, affected surface, reason, downstream impact, reviewer.

### Operational Routing

docs/ops/hermes_openclaw_routing_policy.md (v1.0, effective 2026-05-15) defines three execution lanes:

- Lane A (Deterministic Production): No LLM. Scripts, cron, tests only.
- Lane B (Cheap Monitoring): File/JSON checks first. LLM on anomaly only via run_agent_direct.py.
- Lane C (High-Token Manual): Manual sessions for synthesis, audits, refactoring. No autonomous cron.

Critical constraint: no cron job may depend on a gateway token.

---

## Spec Lifecycle

### Spec States

| State | Meaning |
| --- | --- |
| DRAFT | Under development |
| IN PROGRESS | Active work, phased |
| HELD | Blocked on dependency |
| RESOLVED | All acceptance criteria met |
| SUPERSEDED / MITIGATED | Failure modes neutralized via different route |
| CLOSED | Formally closed |

### Held-Spec Ledger

Tracks all specs that are held/blocked with: what is held and why, first-fire validation status, alert deadlines, next operator action.

---

## Source Files

| Component | File |
| --- | --- |
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

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline or infrastructure before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-24*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Prior ruleset**: `2a3e79eb` (v1.13.0) - RETIRED 2026-05-04
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Manifest**: 36+ entries, no duplicate IDs
- **Architecture freeze**: LIFTED 2026-05-26 (h20d checkpoint passed). Post-h20d sequence can begin after operator approval.

## Hermes Skills Sync

*Last reviewed: 2026-06-24 (path corrected)*

- **CORRECTION (2026-06-24):** The prior `docs/hermes_skills/` path cited previously does NOT exist in the `Warrenpoobear/hermes-agent` repo (verified via direct directory listing). The actual synced knowledge docs live at **`docs/knowledge/`**. Do not reference `docs/hermes_skills/`.
- **Knowledge-layer docs at `docs/knowledge/` (Town -> Hermes sync, merged 2026-06-24):** financial-health, clinical-scoring, biotech-validation, selector-ranker, pe-pacing, sfo-liquidity-architecture, spending-liquidity, institutional-signal, screener-ops (9 files). Synced as reference docs a SOUL.md may cite — NOT agents.
- **Ops/meta-skills NOT synced (reconciliation 2026-06-24):** memory-steward, self-improving, openclaw-agent-optimize were reviewed and intentionally NOT copied — they overlap the repo's own self-management governance.

## Infrastructure

*Last reviewed: 2026-06-25*

- **Current**: WSL2 on Windows host (BCM-LPT-012)
- **Agent model**: DeepSeek v4 flash via Together AI (switched 2026-05-20, was Llama 3.3 70B)
- **Agent fleet**: 30 agents (27 active, 1 suppressed, 1 retired, 1 shadow). 696 recent executions, 0 failures.
- **CI status**: RED (budget exhaustion, ~48 days as of Jun 25; Jun 1 recovery target unconfirmed — verify terminal output)
- **Herald Digest**: RECOVERED 2026-06-24 - full digests delivered Jun 24-25, confirmed via operator inbox. Residual issue: intermittent morning-slot misses. No longer a multi-week outage.
- **Planned**: $15/mo Linux VPS (DigitalOcean / Hetzner). No timeline set. WSL2 remains dev environment.
