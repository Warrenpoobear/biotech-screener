---
name: hermeslink-state-capture
triggers:
  - "update hermeslink"
  - "check infrastructure"
  - "audit Hermes"
  - "hermeslink status"
description: >
  Real-time deterministic state snapshot of Hermes infrastructure: cron jobs, 
  agents, held specs, contradictions, first-fire schedules, git state. 
  Read-only monitoring tool; outputs to artifacts/ops/.
---

# Hermeslink State Capture — Infrastructure Health Monitoring

## Purpose

Provide real-time, deterministic snapshot of Hermes-managed infrastructure state. Answers:

- "What's the current git state and code version?"
- "Which cron jobs are active vs suppressed?"
- "What specs are held and why?"
- "Which first-fire schedules are at risk?"
- "Are there infrastructure contradictions?"
- "Is the system ready for major decisions?"

**Scope:** Read-only capture + validation. No modifications, no enforcement.

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
- 5 infrastructure contradiction checks (C1-C5)
- Verify cron consistency with registry
- Check artifact freshness thresholds

**Layer 4: Emit**
- `artifacts/ops/knowledge_layer/latest_state.json` (full state)
- `artifacts/ops/first_fire_ledger/latest.json` (schedules)
- `artifacts/ops/contradiction_ledger/latest.md` (issues)
- `artifacts/ops/held_spec_ledger/latest.json` (blockers)

---

## Run Hermeslink

### Runtime Reference

Hermes Link runtime details confirmed 2026-05-25:

| Component | Location / Value |
| --- | --- |
| Package | `@hermespilot/link` |
| Version | Hermes Link v0.6.5 |
| Mode | Paired, relay-connected |
| Host | BCM-LPT-012 \(WSL2\) |
| Local API port | `52379` |
| Binary | `~/.npm-global/bin/hermeslink` |
| Symlink target | `~/.npm-global/lib/node_modules/@hermespilot/link/dist/cli/index.js` |
| Runtime data | `~/.hermeslink/` \(config, staging, conversations\) |

Cursor Cloud limitation: the repo-native Hermes MCP server may work while the
production Hermes/Hermes Link runtime and local port are absent. Treat this as an
environment limitation, not a repo failure.

### Command

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
python3 tools/build_hermes_knowledge_layer.py
```

### Output

```
[build_hermes_knowledge_layer] 2026-05-19
  repo: /mnt/c/Projects/biotech_screener/biotech-screener

Layer 1: capture...
Layer 2: normalize...
Layer 3: contradiction scan...
Layer 4: write outputs...
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/knowledge_layer/latest_state.json
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/knowledge_layer/latest_state.md
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/first_fire_ledger/latest.json
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/first_fire_ledger/latest.md
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/contradiction_ledger/latest.md
  wrote /mnt/c/Projects/biotech_screener/biotech-screener/artifacts/ops/held_spec_ledger/latest.json

=== Summary ===
  git head:           be0d26fb
  uncommitted files:  0
  held items:         6
  first-fire status:  WARN_DATE_MISMATCH
  contradictions:     0 hard  /  1 possible
```

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

| Status | Count | Action |
|--------|-------|--------|
| `active` | 25–30 | Core + specialty agents |
| `deprecated` | 1–3 | Old agents (OK to exist) |
| `shadow` | 1–2 | Trial agents (OK if limited) |

**Action:** If `active` drops significantly, check for failed deployments.

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
- Transform: build_hermes_knowledge_layer.py
- Sink: artifacts/ops/ (knowledge_layer/, first_fire_ledger/, contradiction_ledger/, held_spec_ledger/)

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

- **Tool location:** `tools/build_hermes_knowledge_layer.py` (600 lines)
- **Language:** Python 3.10+
- **Dependencies:** pathlib, json, subprocess, datetime
- **Runtime:** ~2–5 seconds (reads crontab, git, ~1000 files)
- **Output format:** JSONL (one record per line, newline-delimited JSON)
- **Cron scheduling:** Optional (manual trigger preferred for now; can enable with `schedule build_hermes_knowledge_layer.py daily 08:00 ET`)

---

## See Also

- `artifacts/ops/knowledge_layer/` — State outputs
- `.claude/agents/hermes-operator.md` — Operator runbook
- `tools/build_hermes_knowledge_layer.py` — Implementation
- Spec 089 (Governance KG) — Related governance infrastructure graph

