---
name: hermeslink-state-capture
triggers:
  - "phase 2 step 5 deployment"
  - "weekly knowledge layer update"
  - "check infrastructure state"
  - "audit Hermes governance"
  - "hermeslink status"
description: >
  Deterministic Hermes ops state snapshot (Spec 089). Builder: build_hermes_knowledge_layer.py.
  Captures git, crontab, agent registry, held specs, contradictions, first-fire. Operator-host
  authoritative for C1/C3 cron checks. Phase B routes hard contradictions to Town (dry-run default).
  Post-build: hermes-contradiction-detector. Separate from build_knowledge_graph.py (governance KG).
---

# Hermeslink State Capture — Knowledge Layer (Spec 089)

## Status Update (2026-06-07)

**ACTIVE** — Ops knowledge layer + Town-Hermes Phase B egress on `main`. Baseline `ec4b2726`.

| Component | Tool / path | Notes |
| --- | --- | --- |
| **Hermeslink builder** | `tools/build_hermes_knowledge_layer.py` | Spec 089 ops brain (this skill) |
| **Learnings audit** | `tools/audit_learnings.py` | HOT/WARM tier hygiene; read-only |
| **Skills telemetry** | `tools/skills_execution_logger.py` | JSONL under `artifacts/skills_learning/` |
| **Skills learning report** | `tools/hermes_skills_learning_loop_v2.py` | Monthly advisory report; no auto-routing |
| **Governance KG** | `tools/build_knowledge_graph.py` + `query_knowledge_graph.py` | Separate graph; not Hermeslink |
| **Town bridge** | `common/town_bridge_events.py` | Auto `contradiction_detected` after build |
| **Follow-up job** | `agents/hermes-contradiction-detector/run_job.py` | Re-reads `latest_state.json` warnings |
| **MCP read** | `knowledge_read(artifact=...)` | `contradiction_ledger` → `latest.md` |
| **Cron (operator)** | Weekdays ~5:45 PM ET | Run on **operator WSL** after `git pull` |

**Agent fleet (registry):** **29 active** (includes 4 Hermes governance jobs), plus 1 suppressed and 4 deprecated registry entries. Lint: `pytest tests/test_agent_registry.py`.

## Purpose

Provide a deterministic snapshot of Hermes-managed **operational** infrastructure state (not production rankings). Answers:

- "What's the current git state and code version?"
- "Which cron jobs are active vs suppressed?"
- "What specs are held and why? What's released?"
- "Which first-fire schedules are at risk?"
- "Are there infrastructure contradictions?"
- "Is the system ready for major decisions?"
- "What governance gates are blocking work?" (NEW: KG-driven)

**Scope:** Read-only capture + validation + governance queries. No modifications, no enforcement.

---

## How It Works

### Four-Layer Pipeline

**Layer 1: Capture**
- Git state (HEAD, branch, uncommitted files)
- Crontab (active jobs, suppressed markers)
- Agent registry status
- Key artifacts (snapshots, models, reports)
- Held specs and blockers
- First-fire schedules

**Layer 2: Normalize**
- Merge all sources into unified graph
- Compute metadata (freshness, status, counts)
- Extract contradictions

**Layer 3: Validate**
- Contradiction checks **C1–C5** (see table below)
- On Cloud without `crontab`: C1/C3 emit `UNKNOWN_CLOUD_ENV` (not hard failures)

**Layer 4: Emit**
- `artifacts/ops/knowledge_layer/latest_state.{json,md}`
- `artifacts/ops/first_fire_ledger/latest.{json,md}`
- `artifacts/ops/contradiction_ledger/latest.md` (+ dated copy)
- `artifacts/ops/held_spec_ledger/latest.json`
- **Phase B:** if any `HARD_CONTRADICTION`, `send_operator_event(contradiction_detected)` (respects `OPERATOR_DELIVERY_DRY_RUN`)

---

## Host authority (operator WSL vs Cloud)

| Check | Operator WSL | Cursor Cloud |
| --- | --- | --- |
| C1 bioshort cron vs registry | Authoritative | `UNKNOWN_CLOUD_ENV` |
| C3 biotech_hedge_report cron | Authoritative | `UNKNOWN_CLOUD_ENV` |
| C2 watchlist freshness | Partial (file may exist) | Same |
| C4 git clean | Yes | Yes |
| C5 BIOSHORT_VERDICT date | Yes if artifact present | Often MISSING in checkout |
| First-fire FAIL past deadline | Yes if hedge artifacts present | Often MISSING — do not treat as prod failure |

**Rule:** Re-run Hermeslink on operator host before acting on cron or hedge first-fire findings from Cloud builds.

---

## Run Hermeslink

### Runtime Reference

Hermes Link runtime details confirmed 2026-05-25:

| Component | Location / Value |
| --- | --- |
| Package | `@hermespilot/link` |
| Version | Hermes Link v0.7.1 |
| Mode | Paired, relay-connected |
| Host | BCM-LPT-012 \(WSL2\) |
| Local API port | `52379` |
| Binary | `~/.npm-global/bin/hermeslink` |
| Symlink target | `~/.npm-global/lib/node_modules/@hermespilot/link/dist/cli/index.js` |
| Runtime data | `~/.hermeslink/` \(config, staging, conversations\) |

Cursor Cloud limitation: the repo-native Hermes MCP server may work while the
production Hermes/Hermes Link runtime and local port are absent. Treat this as an
environment limitation, not a repo failure.

### Command (full Hermeslink cycle)

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
git pull   # on operator host

# 1. Build ledgers (Spec 089)
python3 tools/build_hermes_knowledge_layer.py

# 2. Optional explicit Town route (same as step 1 when hard contradictions exist)
python3 agents/hermes-contradiction-detector/run_job.py

# 3. Review
cat artifacts/ops/knowledge_layer/latest_state.md
cat artifacts/ops/contradiction_ledger/latest.md
cat artifacts/ops/first_fire_ledger/latest.md

# 4. Hermes governance jobs (Phase B — dry-run unless OPERATOR_DELIVERY_DRY_RUN=0)
python3 agents/hermes-held-spec-ledger/run_job.py
```

### Example output (2026-05-31, Cloud VM)

```
[build_hermes_knowledge_layer] 2026-05-31
  crontab surface:    UNKNOWN_CLOUD_ENV
  held items:         6
  first-fire status:  FAIL_ARTIFACT_MISSING_PAST_DEADLINE
  contradictions:     0 hard  /  2 cloud-env  /  2 possible
  Cron checks skipped (non-authoritative host) — verify on operator machine.
```

On operator WSL with crontab + hedge artifacts, expect `crontab surface: OPERATOR_HOST` and authoritative C1/C3.

---

## Interpreting Results

### Git State

| Field | Meaning |
|-------|---------|
| `git head` | Current commit hash (check if latest spec landed) |
| `branch` | Current branch (should be `main` for production) |
| `uncommitted files` | Any changes not committed (should be 0 for deployment) |

**Action:** If uncommitted files > 0, review `git status --porcelain` before major decisions.

### Cron Jobs

| Metric | Expected | Action |
|--------|----------|--------|
| `active_job_count` | 55–65 | Alert if <50 (jobs may have been removed) |
| `suppressed_job_count` | 0–2 | Alert if >3 (too many suppressed) |

**Action:** Check which jobs are suppressed; verify they're intentional (e.g., bioshort_watch LLM suppressed since May 6).

### Agents

| Status | Count (2026-06-19 registry) | Action |
|--------|-------------------|--------|
| `active` | 29 | Includes 4 Hermes governance jobs |
| `suppressed` | 1 | `bioshort_watch`; reactivation requires operator approval and a separate spec |
| `deprecated` | 4 | Historical tombstones/workspaces owned by `shadow_monitor` or `herald` |
| **total registry entries** | **34** | From `AGENT_REGISTRY.json`; 31 directories currently exist on disk |

**Action:** If `active` drops significantly, check for failed deployments. Lint: `pytest tests/test_agent_registry.py`.

### Held Specs (6 Items)

| Spec | Status | Blocker | Action |
|------|--------|---------|--------|
| spec_087_b1b | AWAITING_FIRST_FIRE | First-fire deadline 2026-05-09 09:00 ET | Check if deadline passed |
| spec_087_b2 | HELD | B1b must pass first | Wait for B1b closure |
| spec_087c | HELD | ≥4 fresh weekly hedge reports required | Monitor report count |
| spec_088_phase_b | HELD | Spec 087 active branch must close | Wait for 087 decision |
| bioshort_watch_llm | HELD_SUPPRESSED | Separate reactivation decision required | Do NOT reactivate without approval |
| score_rank_pct | SPEC_REQUIRED | Streak monitor fires nightly 22:00 ET | Write spec if streak continues |

**Action:** Verify no unexpected holds have been added.

### Contradictions (5 Checks)

| Check | Expected Status | Details |
|-------|---|---|
| **C1** | OK | bioshort_watch suppressed in registry, no active cron line |
| **C2** | OK | watchlist_current.json fresh (≤3d old) |
| **C3** | OK | biotech_hedge_report.py cron line active (Spec 087 B1b) |
| **C4** | OK | Working tree clean (0 uncommitted files) |
| **C5** | WARN or OK | BIOSHORT_VERDICT as_of_date vs first-fire date (may be pre-fire) |

**Action checklist:**
- ✓ If C1 status = HARD_CONTRADICTION: escalate (suppressed but cron active)
- ✓ If C2 status = NEEDS_OPERATOR_DECISION: rerun catalyst_resolution_tracker
- ✓ If C3 status = HARD_CONTRADICTION: verify cron install for Spec 087 B1b
- ✓ If C4 status = POSSIBLE_DRIFT: review `git status` before committing
- ✓ If C5 status = WARN_DATE_MISMATCH and deadline passed: review first-fire status

### First-Fire Schedules

**Current Schedule (1 item):**

| Job | Expected Fire | Status | Deadline | Action |
|-----|---|---|---|---|
| biotech_hedge_report | 2026-05-08T18:00 ET | WARN_DATE_MISMATCH | 2026-05-09T09:00 ET | Pre-first-fire (expected artifact from 2026-05-08, check after deadline) |

**Action:** If deadline has passed and artifact missing, escalate.

---

## Decision-Readiness Checklist

Use before major decisions (13F validation, freeze lift, Phase 2 unlock):

```
PRE-DECISION HERMESLINK AUDIT

□ Git state clean (uncommitted files = 0)
□ No hard contradictions (C1, C3 not HARD_CONTRADICTION)
□ Cron jobs stable (active_job_count > 50)
□ Held specs consistent (no new unexpected holds)
□ No stale artifacts (first-fire deadlines not missed)
□ Contradiction actions resolved (C2, C4, C5 either OK or in progress)

Status: READY / NOT_READY_ISSUES: __________
```

---

## When to Run Hermeslink

### Mandatory (before major decisions)
- Before invoking 13f-validation-coordinator
- Before invoking phase-2-step-4-readiness
- Before governance-spec-enforcement audit
- Before freeze lift decision (May 26 h20d)

### Recommended (routine checks)
- Daily 08:00 ET (after morning snapshot, post-cron-run)
- After any cron job modification
- After any agent deployment or update
- After spec hold status changes

### Optional
- On-demand if operator suspects infrastructure issues
- During debugging (verify state before/after changes)

---

## Integration with Other Skills

**Feeds into:**
- `governance-spec-enforcement` — Uses state as baseline for freeze/gate audit
- `13f-validation-coordinator` — Ensures clean state before validation trigger
- `phase-2-step-4-readiness` — Pre-launch verification checklist

**Data lineage:**
- Source: crontab, git, AGENT_REGISTRY, artifacts/, memory files
- Transform: `build_hermes_knowledge_layer.py`
- Sink: `artifacts/ops/` (knowledge_layer/, first_fire_ledger/, contradiction_ledger/, held_spec_ledger/)
- Egress (Phase B): `common/town_bridge_events` → `common/operator_delivery` → Town email
- **Operator triage:** `town-operator-bridge.md` maps `event_type` → root cause (Classes M–P, 2026-06-24)

**Related (not Hermeslink):**
- `tools/build_knowledge_graph.py` — governance spec graph (Spec 089 KG pilot / Spec 110)
- `docs/hermes_skills/town-operator-bridge.md` — event types and live-email checklist

---

## Troubleshooting

### If Hermeslink Fails

1. **Check git repo:**
   ```bash
   cd /mnt/c/Projects/biotech_screener/biotech-screener
   git status
   ```

2. **Verify crontab readable:**
   ```bash
   crontab -l | wc -l  # Should return >50
   ```

3. **Check artifacts directory:**
   ```bash
   ls -la artifacts/ops/knowledge_layer/
   ```

4. **Run with verbose output:**
   ```bash
   python3 -u tools/build_hermes_knowledge_layer.py 2>&1 | tail -50
   ```

### If Contradictions Spike

- C1 HARD_CONTRADICTION → bioshort_watch cron active despite suppression
- C2 NEEDS_OPERATOR_DECISION → watchlist_current.json missing or stale
- C3 HARD_CONTRADICTION → biotech_hedge_report cron missing
- C4 POSSIBLE_DRIFT → uncommitted changes (review before commit)
- C5 WARN_DATE_MISMATCH → BIOSHORT_VERDICT timestamp mismatch (may be expected)

**Resolution:** Address contradictions in order (C1/C3 are blockers, C2/C4/C5 are informational).

### If Cloud Artifacts Are Stale

If `latest_state.json` records a different branch/head than the current checkout,
do not treat C2/C3/C5 or first-fire warnings as fresh production failures.

Safe sequence:

1. Refresh Hermes knowledge artifacts on the intended local/production runtime.
2. Confirm whether the same warnings persist on the fresh branch/head.
3. Triage only persistent warnings as operator items.
4. Do not use stale cloud artifacts to justify production code or cron changes.

---

## Implementation Notes

- **Tool location:** `tools/build_hermes_knowledge_layer.py`
- **Language:** Python 3.12+
- **Dependencies:** stdlib only (subprocess for git/crontab)
- **Runtime:** ~2–5 seconds on operator host
- **Output format:** JSON + Markdown ledgers under `artifacts/ops/`
- **Town import:** builder inserts repo root on `sys.path` before `town_bridge_events`
- **Cron scheduling:** Operator weekdays ~5:45 PM ET (after daily production window); not wired into `run_daily_production.py`
- **Tests:** `tests/test_build_hermes_knowledge_layer_contradictions.py`, `tests/test_town_bridge_events.py`

---

## See Also

- `artifacts/ops/knowledge_layer/` — State outputs
- `.claude/agents/hermes-operator.md` — Operator runbook
- `tools/build_hermes_knowledge_layer.py` — Implementation
- Spec 089 (Governance KG) — Related governance infrastructure graph

