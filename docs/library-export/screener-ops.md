# Screener Ops & Governance Skill

## Purpose

Reference for daily production operations, the Hermes knowledge layer, agent fleet monitoring, and the spec/governance lifecycle that governs all changes to the biotech screener.

This skill is organized into two sections:

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

### Model-Specific Prompting

Agent AGENTS.md docs updated with structured prompting procedures (originally Llama-optimized, carried forward to DeepSeek v4):

- IF/THEN chains instead of open-ended reasoning
- Step numbering for multi-step workflows
- Schema-first output format
- No inferred data; report missing explicitly

### Gateway Monitoring

- `~/.hermes/monitor_together_latency.py` tracks latency trends
- Alerts on success rate <80% or avg latency >5s
- Logs to `together_latency.log`

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

### Compliance Memo

"Why the DEM 27-Agent Fleet Is Insulated from Model-Output-as-Control-Signal Failures" - Final version, repo-verified. Cites Texas A&M security taxonomy (470 advisories). Available in Content Library at ai-projects/.

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

### Active Spec Numbering

Specs numbered sequentially (currently 071-105 range active). Each spec has:

- Acceptance criteria with explicit section references
- Phase gates (A/B/C/D typical)
- Blocking dependencies on other specs
- Closure memos in `artifacts/audit/`

**Spec numbering collision resolved (2026-05-14):** Original expectation coverage spec was drafted as Spec 100, which collided with the existing IC tooling correction spec. Renumbered to Spec 105 in commit cb242311.

**Schema/coverage/export specs (commits cb242311 through b310671a, 2026-05-14):**

| Spec | Title | Status | Commit |
| --- | --- | --- | --- |
| 105 | Expectation Layer Coverage Verification | CODE-CLOSED / pending live QA | 0ddbb509 |
| 101 | Runway Severity v1.1 Export Contract | CLOSED | eaa4ea87 + cba4ee0f |
| 104 | Insider Diagnostic Stabilization | MEASURED / pending 2026-05-15 | b310671a |
| 102 | Historical Backfill for Expectation Research | DRAFT | -- |

**Other specs shipped 2026-05-14:**

| Spec | Title | Status | Commit |
| --- | --- | --- | --- |
| 087 B2 | Dashboard Freshness Envelope | CLOSED | 400a6cd9 |
| 087 B0 | Stale-Propagation Guard | CLOSED (formal closure memo) | 0f0c7952 |
| 087C A | Bioshort Alpha Research Design | DESIGN (memo only) | 7628b9c6 |
| 088 B | Catalyst Delta v2 Filter Companion | SHIPPED | 5ca4b033 |

All schema/coverage/export specs are correctness work. No new model, no new alpha.

### Held-Spec Ledger

Tracks all specs that are held/blocked with:

- What is held and why
- First-fire validation status
- Alert deadlines
- Next operator action

---

## Expectation Layer Coverage Gate (Spec 105)

**QA file**: `production_qa_check.py`
**Status:** CODE-CLOSED (commit 0ddbb509). Pending live production snapshot QA via `python tools/production_qa_check.py --as-of-date YYYY-MM-DD`.

Production pipeline hard-fails if market-expectation fields are missing or under-covered in `rankings.csv`. Thresholds sourced from `FEATURE_COVERAGE_REQUIREMENTS` (not hardcoded).

### Required Expectation Fields

| Field | Required Coverage | Source |
| --- | --- | --- |
| `short_interest_pct` | 0.90 | Market data provider |
| `close_price` | 0.99 | Market data provider |
| `market_cap_mm` | 0.95 | Market data provider |
| `priced_move_pct` | 0.80 | Derived (catalyst pricing model) |
| `insider_net_buy_value_90d` | 0.30 | Form 4 (tracked nonblocking / diagnostic only) |

### Gate Behavior

- Runs every pipeline execution at Step 5 (Gates)
- Hard fail if any required field is missing from DataFrame
- Hard fail if any field falls below its per-field threshold
- Error message includes: field name, actual coverage, required threshold
- Coverage stats logged every run regardless of pass/fail
- Expectation model must consume these columns from `rankings.csv`, not from a parallel source

### Key Rule

`FEATURE_COVERAGE_REQUIREMENTS` is the single source of truth. If thresholds change, the gate inherits automatically. Do NOT hardcode coverage floors in pipeline scripts.

---

## Export Contract Registry (Spec 101)

**Status:** CLOSED (commits eaa4ea87 + cba4ee0f, 2026-05-14). `ev_severity_score` now exported. Build gap resolved.

Tracks which computed fields are exported to CSV and snapshots.

### Runway Severity Export (v1.1, RESOLVED)

**All exported (post-Spec 101):**

- `runway_severity_score`, `ev_severity_score`, `runway_buffer_months`, `financing_truth_gate`
- `dilution_haircut`, `size_multiplier`, `severity_bucket`, `severity_notes`
- `check_severity_formulas()` QA validation runs on every snapshot
- Validates finiteness before formula checks; fails explicitly on blank/NaN/Inf

**Derived field contracts (must hold for all non-null rows):**

```
dilution_haircut == 0.35 * ev_severity_score       (tolerance 1e-6)
size_multiplier == max(0.40, 1 - 0.60 * ev_severity_score)  (tolerance 1e-6)
```

Pre-v1.1 snapshot readers default `ev_severity_score` to NaN (not fail).

---

## Diagnostic Fields Registry (Spec 104)

Fields tracked for observability but explicitly excluded from scoring, ranking, and selection.

### Current Diagnostic Fields

| Field | Status | Meaning of Null | Meaning of 0.0 |
| --- | --- | --- | --- |
| `insider_net_buy_value_90d` | DIAGNOSTIC ONLY | Not fetched / no Form 4 coverage | Fetched, no insider buy activity in 90d |

### Insider Model Isolation Guard (CRITICAL)

`insider_net_buy_value_90d` must NOT enter the expectation model's `market_features` input. The model has an `insider_net_buy_z` weight that activates silently if the field flows upstream. Guard with at least one of:

1. **Input exclusion (preferred):** Runtime assert that `insider_net_buy_value_90d` is NOT in `market_features` DataFrame at inference
2. **Weight zeroing:** `insider_net_buy_z` weight = 0.0 with test
3. **Drop guard:** Pre-inference step that drops the field if present, with logged warning

### Rules

- Never collapse blank (NaN) and zero (0.0) -- they have different semantics
- Never impute zero for missing or blank for zero
- CI check: flag suspicious if column is ALL zero or ALL null
- Field must remain in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Does not affect ranks, actions, or position sizing
- Promotion requires: 20+ stable snapshots, >= 60% coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval

---

## Backfill Tooling (Spec 102)

Research-enablement tooling for backfilling expectation fields into historical snapshots.

### Target Fields

`short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct` (required); `insider_net_buy_value_90d` (optional)

### Key Rules

- Default: additive-only (`recompute=False`). Original ranks/actions preserved.
- Every backfill emits a structured manifest (snapshot_date, fields_added, coverage before/after, recompute flag, timestamp, version)
- `_backfill_version` metadata column added to all backfilled snapshots (null for originals)
- Research scripts must filter on `_backfill_version` to avoid silent pre/post mixing
- Default scope: 30 trading days, configurable

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
- **Model status**: FROZEN (BLOCKED_LEVEL_0) as of 2026-07-10 — canonical authority `docs/model_documentation.md`. The 2026-05-26 h20d freeze lift was Spec-100-scoped and is spent; the binding window now is the DEM candidate freeze + NO_MODEL_CHANGE forward-validation window (mandate SM-20260629-001, 0/20 eligible LIVE windows, code-enforced). No selector/ranker/sizing change is authorized without explicit operator authorization.

## Knowledge Graph Implementation (Spec 110 / Spec 089)

*Last reviewed: 2026-05-24*

- **KG Phase 2 Step 4**: COMPLETE (68/68 tests PASS)
- **Spec 110 Phase 1 PoC**: COMPLETE (56 nodes, 16 edges)
- **Phase 2 Step 4d**: Deferred post-h20d
- **Phase 2 Step 5**: BLOCKED on 13F clearance + h20d decision
- **13F Q1 2026 cohort**: HOLD — h20d re-eval gate (2026-07-01) not cleared as of Jul 4. The 0.875 Jaccard cited previously was the Q4/49-manager comparison, NOT the Q1 55-manager test. Actual Q1 55-manager Jaccard = 0.463 (FAIL vs 0.70 threshold). Q1 13Fs promoted (PR #429, Jun 28) so deferred quarantine script is runnable — requires local execution of `tools/check_13f_cohort_quarantine.py` + operator sign-off. Source: repo commit 31ebb017 (2026-07-04).
- **Forward monitor**: ~35 trading days accumulated

## Codegraph Integration

*Added: 2026-05-24*

- **Version**: v0.9.4 (npm global, Node 22)
- **Index**: 1,667 files / 50,259 nodes / 114,016 edges / 108 MB
- **Cursor MCP**: Registered globally (10 tools, auto-allow)
- **Claude Code MCP**: Registered globally
- **Hermes MCP**: DEFERRED (pending disambiguation wrapper policy)
- **Runbook**: `docs/CODEGRAPH_RUNBOOK.md` (committed)
- **Decision**: D-2026-008 — interactive use approved, autonomous deferred

## Hermes Skills Sync

*Last reviewed: 2026-06-24 (path corrected)*

- **CORRECTION (2026-06-24):** The prior `docs/hermes_skills/` path cited here does NOT exist in the `Warrenpoobear/hermes-agent` repo (verified via direct directory listing). The actual synced knowledge docs live at **`docs/knowledge/`**. Do not reference `docs/hermes_skills/`.
- **Knowledge-layer docs at `docs/knowledge/` (Town -> Hermes sync, merged 2026-06-24):** financial-health, clinical-scoring, biotech-validation, selector-ranker, pe-pacing, sfo-liquidity-architecture, spending-liquidity, institutional-signal, screener-ops (9 files). Synced as reference docs a SOUL.md may cite — NOT agents.
- **Ops/meta-skills NOT synced (reconciliation 2026-06-24):** memory-steward, self-improving, openclaw-agent-optimize were reviewed and intentionally NOT copied — they overlap the repo's own self-management governance (CLAUDE.md / AGENTS.md / authority model) and self-improving's `.learnings/` write loop remains governance-gated. Flag-not-copy decision.
- **Older deploy claims (2026-05-24, byte-level history unverified):** prior note asserted "7 new skills deployed" + "7 canonical preserved" + openclaw-agent-optimize to devops/. Counts/locations not re-verified this cycle; treat the `docs/knowledge/` listing above as current ground truth.

## BioShort Research (Spec 092)

*Last reviewed: 2026-05-13*

| Phase | Status | Key Output |
| --- | --- | --- |
| A (inventory) | COMPLETE | 142/162 usable snapshots, 18 with decision_portfolio.csv fallback |
| B (research-mode isolation) | COMPLETE | --research-mode flag, archive redirect, mode tagging |
| C (historical panel) | COMPLETE | 146 rows x 16 features, 100% success, 0 live path mutations |
| D (forward returns) | COMPLETE | DEFER verdict 60.5% hit T+5, median T+5 +0.63%, T+20 +2.49% |

Key findings (pseudo-PIT):

- DEFER verdict: 129 samples, 60.5% accuracy at T+5 (forward_5d >= 0)
- Median T+5 return: +0.63%
- Median T+20 return: +2.49%
- Median drawdown: -2.86% over 20d post-recommendation
- Pseudo-PIT caveat: features computed with current logic on historical snapshots. No promotion claims supported.

## Town-Hermes Bridge Status

*Last reviewed: 2026-06-26*

- Phase A complete (dry-run mode, `OPERATOR_DELIVERY_DRY_RUN=1`) — superseded by Phase B live delivery
- **Phase B LIVE (2026-06-25):** `OPERATOR_DELIVERY_DRY_RUN=0` confirmed in `.env`; email delivery active to operator Town inbox for all hard-failure event types
- Phase B wiring covers all event types: held-spec, first-fire, ruleset-integrity, snapshot-missing, contradiction_detected, cron_missed
- **Note (2026-06-24):** `cron_missed` Town alerts may trace to Class P (cron `sys.path` isolation) — see `openclaw-data-pipeline-debug` Class P and `town-operator-bridge` triage table
- Direction: Hermes→Town delivery only. Town→Hermes write-back remains design-only / operator-mediated — no repo or runtime mutation from Town
- Cursor/Town project memory: `~/.hermes/hermes-agent/.learnings/projects/cursor-town.md` (created 2026-05-24)
- Skill doc: `docs/hermes_skills/town-operator-bridge.md` · Spec 090

## Infrastructure

*Last reviewed: 2026-07-12*

- **Current**: WSL2 on Windows host (BCM-LPT-012)
- **Agent model**: DeepSeek v4 flash via Together AI (switched 2026-05-20, was Llama 3.3 70B)
- **Agent fleet**: 30 agents (27 active, 1 suppressed, 1 retired, 1 shadow). 696 recent executions, 0 failures.
- **Preflight timeout**: Fixed 10s -> 20s (2026-05-24)
- **Hermes Link**: v0.6.5, paired mode, relay-connected, local API port 52379
  - Binary: `~/.npm-global/bin/hermeslink`
  - Package: `@hermespilot/link` (npm global)
  - Runtime data: `~/.hermeslink/` (config, staging, conversations)
- Daily cron runs 4:30-7:30 PM ET on weekdays (critical window 14:30-18:55)
- universe_maintenance cron: 10:00 AM ET (fixed race condition)
- Sleep-cliff risk: Windows host suspend kills crons silently
- Stopgap: `powercfg /change standby-timeout-ac 0`
- Missed cron signature: 24-48h gap in `data/snapshots/`
- **CI status**: RED (see operational-health-baselines Outage Anchors — anchor 2026-05-08, compute elapsed at read time; ~65d as of Jul 12; pytest 3.10, pytest 3.12, lint, smoke, replay failing; only dependabot PRs #476-480 open as of Jul 12; governance-critical gates semgrep + protected-paths remain GREEN)
- **Cron guard fix (PR #489, merged Jul 10):** `set -e` bug in `cron_daily_production.sh` was silently skipping the forward-validation tail on WARN-status (exit 2) runs since ~2026-04-17. Fixed — forward-validation now runs post-pipeline. Unblocks SM-20260629-001 mandate.

- **Forward-validation mandate detail (RECONCILED 2026-07-12, PR #492/#489):** candidate `model_hash=827c35a9ed3ee6e1` (`ast-v1` behavioral fingerprint; legacy raw-bytes `a9983a67c6954813`; registered 2026-06-26, unchanged). Every capture is classified LIVE or REPLAY (`fv_capture.v2`); a window counts toward SM-20260629-001 only if LIVE + quality-PASS + hash-match + XBI benchmark + realized 5d excess. The 10 legacy captures (Jun 12–26) are backfill and do NOT count — **0/20 eligible LIVE windows, code-enforced**; ~mid-Nov 2026 gate, 2026-09-30 status checkpoint. Liveness monitor `tools/forward_validation_liveness_monitor.py` + wrapper `tools/cron_forward_validation_liveness.sh` are authored but **not yet installed in crontab — operator deploy step pending**. See ic-evaluation for the four-way evaluator distinction.
- **Test trust audit (PR #491, merged Jul 12):** `tools/test_trust_audit.py` v1 (L0+L1 read-only, advisory). Trust score 70.68/100, 1,701 findings. Advisory-only; do NOT gate CI on it yet.
- **Herald Digest**: RECOVERED 2026-06-24 - full digests delivered Jun 24-25 (pre-market/midday/EOD), confirmed via operator inbox. Residual issue: intermittent morning-slot misses (Jun 26 pre-market missed); on a miss run herald_health_check.py --recover. No longer a multi-week outage.
- **Planned**: $15/mo Linux VPS (DigitalOcean / Hetzner). No timeline set. WSL2 remains dev environment.
