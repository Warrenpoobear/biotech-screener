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

1. **Framework Reference** - Stable pipeline architecture, processes, and governance (changes only with code updates)
2. **Operational State** - Volatile infrastructure and status snapshots that require periodic refresh

---

## Repo scale (agent orientation)

Use this to calibrate blast-radius and where complexity lives — not to justify skipping CodeGraph preflight.

| Layer | Approx. Python LOC | Character |
| --- | ---: | --- |
| Scoring engines | ~4,000 | `decision_engine` + `selector_engine` + `ranker_engine` — **compact, high leverage** |
| Screen orchestrator | ~13,000 | `run_screen.py` — daily production path |
| Production stack | ~176,000 | `src/`, `common/`, `tools/`, agents, governance, engines |
| Tests | ~254,000 | Contract, leakage, regression — largest layer |
| Research scripts | ~90,000 | `scripts/research/` backfills and studies |
| All Python (excl. data dirs) | ~750,000 | Full repo |

**Summary:** small scoring core inside a medium institutional pipeline with a **heavy verification shell**. Edits to engines or `final_score` paths are gated regardless of how “small” the diff looks.

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

### Host authority (operator WSL vs Cloud)

*Last reviewed: 2026-06-07 · plumbing baseline `main` @ `ec4b2726`*

| Host | Can validate | Cannot validate |
| --- | --- | --- |
| **Operator WSL** | `crontab -l`, `biotech_hedge_report.py` schedule, `output/hedge_report/`, `BIOSHORT_VERDICT.json`, producer logs, B1b/B2/B3 governance, `~/.hermes/config.yaml` | OpenClaw job history (separate checks) |
| **Cursor Cloud** | Repo plumbing, registry/MCP, CodeGraph, ledger **build** (read-only) | Authoritative cron, hedge artifacts, gateway model |

**Contradiction severities**

- `HARD_CONTRADICTION` — only when `crontab` is available and B1b producer line is missing.
- `UNKNOWN_CLOUD_ENV` — C1/C3 when `crontab` binary is absent (Cloud Agent VMs). Not a governance pass/fail.
- First-fire `FAIL_ARTIFACT_MISSING_PAST_DEADLINE` — emitted when hedge artifacts are missing past deadline. **Expected on Cloud** (no `output/hedge_report/` in checkout). Authoritative only after WSL rebuild.

**Standing rule:** No further cloud cleanup for registry/MCP/CodeGraph/knowledge-layer plumbing unless a new failure appears.

---

## Path C Governance Monitoring (2026-05-28 to 2026-06-03)

**Status**: ACTIVE governance window  
**Full Operational Guide**: See **`path-c-operational-runbook.md`** skill (daily monitoring, window close decision automation, emergency conditions)  
**Monitoring Framework**: See **`path-c-governance-monitoring.md`** skill (IC floor guardrail, forward eval monitoring, decision logic)  

Temporary catalyst timing policy override with hard exit conditions (floor threshold, drawdown trigger, cohort stability check). Daily cron at 10:15 AM ET. Window close decision on 2026-06-03.


### Operator WSL acceptance gate (authoritative)

Run on **operator WSL only** after every `git pull` that touches Hermes, skills, or bioshort/hedge producer paths.

#### Phase 0 — Prep

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
git pull origin main
git log -1 --oneline
```

#### Phase 1 — Repo plumbing

```bash
python3 tools/build_hermes_knowledge_layer.py
python3 tools/audit_hermes_skills.py
python3 tools/audit_learnings.py
cat artifacts/ops/contradiction_ledger/latest.md
cat artifacts/ops/first_fire_ledger/latest.md
```

**Healthy on WSL:** `0 hard` contradictions; crontab surface **available** (not `UNKNOWN_CLOUD_ENV`).

#### Phase 2 — Gateway model

```bash
python3 -c "
import yaml, pathlib
p = pathlib.Path.home() / '.hermes' / 'config.yaml'
if not p.exists():
    print('MISSING', p)
else:
    d = yaml.safe_load(p.read_text())
    m = d.get('model', {})
    print('model.default:', m.get('default'))
    print('model.provider:', m.get('provider'))
    for fb in d.get('fallback_providers', [])[:3]:
        print('fallback:', fb.get('provider'), fb.get('model'))
"
```

**Healthy (2026-05-20+ intent):** `openrouter` primary; `deepseek/deepseek-v4-flash:free` default/active. Together Llama as fallback is OK.

Optional smoke: `hermes gateway status` · `hermes chat -q "Reply with only the model id you are using." -Q`

**Do not** change `run_agent_direct.py` until this gate passes — it bypasses the gateway and defaults to Together Llama.

Docs: `docs/HERMES_GATEWAY_SETUP.md` · `docs/hermes_agents/hermes_tools_map.md` §5.

#### Phase 3 — Spec 087 B1b (bioshort hedge producer)

```bash
crontab -l | grep biotech_hedge_report | grep -v '^#'
crontab -l | grep bioshort_watch
ls -lt output/hedge_report/ | head -10
tail -50 logs/biotech_hedge_report.log
python3 tools/build_hermes_knowledge_layer.py
cat artifacts/ops/first_fire_ledger/latest.md
```

**Expected cron:** one active Friday `0 18 * * 5` line calling `tools/biotech_hedge_report.py` with explicit `--portfolio-csv data/snapshots/$(date +\%F)/portfolio_positions.csv`.

**Pass criteria:** recent `hedge_report_YYYY-MM-DD.json`; `BIOSHORT_VERDICT.json` with matching `as_of_date`; clean log (no traceback, no `MASSIVE_API_KEY` warnings); bioshort_watch LLM cron **suppressed** only.

**Seed note:** `FIRST_FIRE_SEED` in `build_hermes_knowledge_layer.py` is pinned to `hedge_report_2026-05-08.json`. Spec 087 B2 records formal closure 2026-05-14. If newer Friday artifacts exist but the ledger still FAILs on the May 8 filename, **operator evidence** (cron + recent artifacts + log) governs closure; update the seed in a separate docs PR if needed.

| WSL finding | Meaning |
| --- | --- |
| Cron + hedge artifacts present | B1b may be closable on operator evidence |
| Cron missing or hedge artifacts missing | Narrow **Spec 087 B1b ops repair** — not model/ranker/governance redesign |
| Cron present, artifacts missing | Inspect `logs/biotech_hedge_report.log`; check snapshot preflight |
| Gateway still Claude/Llama-only primary | Update `~/.hermes/config.yaml`; `hermes gateway restart` |

#### Phase 4 — Optional runtime skills sync

Only if the gateway reads stale copies from `~/.hermes/skills/` (see `docs/hermes_agents/operator_host_skills.md`). Do **not** overwrite `memory-steward`.

#### Printable checklist

```text
[ ] git pull main
[ ] build_hermes_knowledge_layer.py  → 0 HARD on WSL
[ ] audit_hermes_skills.py           → 32/32 clean
[ ] ~/.hermes/config.yaml            → openrouter + deepseek-v4-flash default
[ ] crontab B1b line                 → one Friday 18:00 biotech_hedge_report
[ ] crontab bioshort_watch           → suppressed only
[ ] output/hedge_report/             → recent JSON + BIOSHORT_VERDICT
[ ] logs/biotech_hedge_report.log    → no traceback / no MASSIVE_API_KEY warn
[ ] first_fire_ledger                → PASS or operator evidence closes B1b
```

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
- NOT allowed to reactivate bioshort\_watch LLM
- NOT the authoritative source for any production state

---

## OpenClaw Agent Fleet

### Agent Registry

**File**: `agents/AGENT_REGISTRY.json`

### Model Configuration (updated 2026-05-20)

- **Primary model**: `deepseek/deepseek-v4-flash:free` (OpenRouter) - fleet-wide migration 2026-05-20
- **Registry**: 34 registry entries (29 active, 1 suppressed, 4 deprecated; 31 directories on disk) — includes 4 Hermes governance jobs in `agents/` (held-spec-ledger, first-fire-validator, ruleset-integrity, contradiction-detector) — see `agents/AGENT_REGISTRY.json`
- **Fallback**: Anthropic Claude SDK (for Claude-specific models)
- **Auto-routing**: "deepseek" models -> OpenRouter (OpenAI-compatible), "claude" -> Anthropic SDK
- **Previous**: Llama 3.3 70B Instruct Turbo (Together AI, 2026-05-13 to 2026-05-20)

### Hermes model routing (surface-specific, 2026-05-31)

Do not assume one model for all Hermes paths:

| Layer | Truth source | LLM? |
| --- | --- | --- |
| Hermes MCP (Cursor) | `mcp_server/hermes_server.py` | No — repo-native read-only MCP; do not substitute upstream `hermes mcp serve` |
| Lane A `hermes-*` jobs | `agents/hermes-*/run_job.py` | No — `llm_policy: none` |
| Hermes Gateway / CLI | `~/.hermes/config.yaml` on operator WSL | Yes — verify live after `git pull` |
| Fleet SOUL intent | `agents/*/SOUL.md`, this skill | `deepseek/deepseek-v4-flash:free` (OpenRouter) |
| `run_agent_direct.py` | `tools/run_agent_direct.py` | Bypasses gateway; defaults to **Together Llama** |

Docs: `docs/hermes_agents/hermes_tools_map.md` §5, `docs/HERMES_GATEWAY_SETUP.md` (WSL acceptance gate). Do not change `run_agent_direct.py` routing without operator WSL verification.

### Inference Tuning (Llama-optimized, 2026-05-13)

| Parameter | Value | Rationale |
| --- | --- | --- |
| Temperature | 0.2 | Stronger governance determinism |
| Frequency penalty | 0.1 | Reduce repetition loops |
| Top\_p | 0.95 | Tighter nucleus sampling |
| Repetition penalty | 1.2 | Anti-loop guard |
| API timeout | 2400s | Llama inference variance (Together can spike 8-12s cold) |
| Retry strategy | Exponential backoff | 500ms-8000ms delays |
| Compression threshold | 0.5 | Less aggressive for 131K context |

### Uncertainty Handling (all agents, 2026-05-13)

All agents tuned with explicit uncertainty escalation rules:

- ops\_supervisor: missing artifacts -> RED (not GUESS); confidence < 0.7 -> escalate
- sentinel: missing drift -> FAIL; boundary cases -> WARN; ambiguous rollback -> both commands
- data\_auditor: missing snapshot -> FAIL; specific ticker counts (not "some")
- ic\_health\_monitor: missing dashboard -> UNKNOWN; threshold boundaries -> ALERT (conservative)
- fleet\_steward: unreachable status -> MEDIUM; missing last\_run -> anomalous (not healthy)

### Llama-Specific Prompting

Agent AGENTS.md docs updated with Llama-specific procedures:

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
| Fleet ops | `tools/fleet_ops_status.py` | One-shot triage + `artifacts/fleet_ops/` |
| Fleet audit | `tools/fleet_completion_audit.py` | Verify deterministic cron wiring |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly classification |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline task orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
| --- | --- | --- |
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops\_supervisor) and deterministic Lane A `hermes-*` jobs are intentionally unsupervised/on-demand and do not carry HEARTBEAT.md.

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
- Every backfill emits a structured manifest (snapshot\_date, fields\_added, coverage before/after, recompute flag, timestamp, version)
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
| Fleet Ops Status | `tools/fleet_ops_status.py` |
| Fleet Completion Audit | `tools/fleet_completion_audit.py` |
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

*Last reviewed: 2026-05-13*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Prior ruleset**: `2a3e79eb` (v1.13.0) - RETIRED 2026-05-04
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Manifest**: 36+ entries, no duplicate IDs
- **Architecture freeze**: In effect until post-h20d checkpoint (\~2026-05-26)

## BioShort Research (Spec 092)

*Last reviewed: 2026-05-13*

| Phase | Status | Key Output |
| --- | --- | --- |
| A (inventory) | COMPLETE | 142/162 usable snapshots, 18 with decision\_portfolio.csv fallback |
| B (research-mode isolation) | COMPLETE | --research-mode flag, archive redirect, mode tagging |
| C (historical panel) | COMPLETE | 146 rows x 16 features, 100% success, 0 live path mutations |
| D (forward returns) | COMPLETE | DEFER verdict 60.5% hit T+5, median T+5 +0.63%, T+20 +2.49% |

Key findings (pseudo-PIT):

- DEFER verdict: 129 samples, 60.5% accuracy at T+5 (forward\_5d >= 0)
- Median T+5 return: +0.63%
- Median T+20 return: +2.49%
- Median drawdown: -2.86% over 20d post-recommendation
- Pseudo-PIT caveat: features computed with current logic on historical snapshots. No promotion claims supported.

## Town-Hermes Bridge Status

*Last reviewed: 2026-06-24*

- Phase A complete (dry-run mode, `OPERATOR_DELIVERY_DRY_RUN=1`)
- Phase B wiring complete (2026-05-30): all event types in repo — held-spec, first-fire, ruleset-integrity, snapshot-missing, contradiction_detected, cron_missed
- **2026-06-24:** `cron_missed` Town alerts may trace to Class P (cron `sys.path` isolation) — see `openclaw-data-pipeline-debug` Class P and `town-operator-bridge` triage table
- Phase B live delivery: pending operator sign-off to set `OPERATOR_DELIVERY_DRY_RUN=0` in `.env` (see `docs/hermes_skills/town-operator-bridge.md`)
- Skill doc: `docs/hermes_skills/town-operator-bridge.md` · Spec 090

## Knowledge Graph Implementation Status

*Last reviewed: 2026-05-24*

- **Phase 2 Step 4 (KG implementation)**: COMPLETE (2026-05-21) — 68/68 tests PASS; 4a loader + 4b queries + 4c contradictions + 4e integration; h20d evidence package ready
- **Spec 110 Phase 1 PoC**: COMPLETE (2026-05-21) — 56 nodes, 16 edges, 5 query patterns, 22 tests PASS; pipeline provenance graph; no production wiring
- **Phase 2 Step 4d (CLI)**: deferred post-h20d
- **Phase 2 Step 5 (KG gating)**: blocked on 13F quarantine clearance + h20d decision (~2026-05-26)

## Infrastructure

*Last reviewed: 2026-05-31*

- **Production host**: WSL2 on Windows (operator authority for cron + artifacts)
- **Agent model**: `deepseek/deepseek-v4-flash:free` via OpenRouter (2026-05-20)
- Daily production cron: 5:30 PM ET weekdays
- `universe_maintenance`: 10:00 AM ET
- Sleep-cliff risk: Windows host suspend kills crons silently — `powercfg /change standby-timeout-ac 0`
- Missed cron signature: 24-48h gap in `data/snapshots/`
- **Planned**: Linux VPS migration; WSL2 remains dev environment

### Pipeline recovery patterns (2026-06-24)

Cross-ref `openclaw-data-pipeline-debug` Classes M–P and `town-operator-bridge` triage table:

| Class | Signature | Fix |
| --- | --- | --- |
| M | yfinance `T00:00:00` parse error | `strftime("%Y-%m-%d")` not `isoformat()` |
| N | Delisted ticker still in screen | Patch **all** universe loaders, not one |
| O | Cache warm 1800s timeout | Align `--warm-sources` CLI default to function default |
| P | `ModuleNotFoundError: No module named 'tools'` in cron | Insert `PROJECT_ROOT` on `sys.path` before imports |

### Repo plumbing baseline (Cloud + Cursor)

*Last reviewed: 2026-06-16 · `cursor/update-openclaw-ed92`*

| Area | Status |
| --- | --- |
| Skills + knowledge recursion | `self-improving`, `.learnings/`, `audit_learnings.py` on `main` |
| CodeGraph | v0.9.9 pinned; MCP + CLI; bounded proof model |
| Hermes fleet | 29 active agents + registry tombstones; WSL acceptance gate for cron/gateway |
| CI / tests | Full suite green; Track B skips intentional; Actions budget ≠ code failure |

### Cursor Cloud Agent Environment

- Install hook: `.cursor/environment.json` — `npm install -g @colbymchenry/codegraph@0.9.9 --prefix "$HOME/.local"`, then `pip install -r requirements.txt`, then `codegraph sync` or `codegraph index`.
- MCP: `.cursor/mcp.json` — `codegraph` + repo-native `hermes` (`mcp_server/hermes_server.py`, read-only). Upstream `hermes mcp serve` is write-capable and is not approved for Cursor on this repo.
- Python deps from `requirements.txt` only; **`pytest-xdist` not required** (`pyproject.toml` uses `-q -m 'not network'`).
- Cloud can run registry tests and build knowledge-layer ledgers; **cannot** authoritatively validate operator cron or `output/hedge_report/`.
- Expected cloud build summary: `0 hard / N cloud-env / M possible`; first-fire FAIL may still appear if hedge artifacts are absent in the checkout.

### CI Budget Failures

- GitHub Actions annotation `The job was not started because an Actions budget is preventing further use.` means provider budget/quota blocked job startup.
- Treat this as CI infrastructure, unrelated to PR diffs unless logs show actual job steps ran.
- Do not patch code for this signal; restore/wait for Actions budget and rerun.

### Track B Governance Contracts

- PR #304 is draft/spec-test-only for fail-closed production-governance contracts.
- Expected red tests are evidence of current gaps, not CI failures to fix.
- Do not implement ranker/final_score behavior, snapshot writer/promotion semantics, selector, sizing, or KG production changes from that PR without explicit governance clearance.

### Hermes Runtime vs Repo MCP

- Repo-native Hermes MCP works in Cursor Cloud (registry, SOUL, knowledge artifacts when built locally).
- Production Hermes gateway, scheduled jobs, and OpenClaw runtime are **operator-host only** — not visible in Cloud.
- Triage knowledge-layer alerts on **operator WSL** after `git pull` and `build_hermes_knowledge_layer.py`; see **Host authority** above.

### Cursor skills knowledge (sync workflow)

| Step | Command / path |
| --- | --- |
| Edit skill source | `skills/<dir>/SKILL.md` or `REFERENCE.md` |
| Sync to Hermes mirror | `python3 tools/sync_hermes_skills.py` |
| Audit mirrors + `_meta.json` | `python3 tools/audit_hermes_skills.py` |
| Audit learnings tiers | `python3 tools/audit_learnings.py` |
| Commit | `docs/hermes_skills/` + `skills/` when mirrors change |
| Optional WSL runtime | Copy to `~/.hermes/skills/` only if gateway reads stale copies |

**Recursive self-improvement:** after significant ops/hermes sessions, run the loop in `skills/self-improving/SKILL.md` (log → promote → skill-patch → sync → `harvest_log.md`). Ops discoveries for this repo usually land in `screener_ops` or `codegraph`.

Runbook: `docs/hermes_agents/operator_host_skills.md` · History: `docs/hermes_skills/harvest_log.md`