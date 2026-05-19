# Hermes Phase 2B Agent Jobs — Runbook

**Branch**: `spec-089-phase-2b-alerting-prep-2026-05-19`
**Status**: PREP ONLY — no live cron entries, no autonomous alerts, no Town delivery
**Spec**: `specs/changes/spec_089_hermes_knowledge_layer.md` (Phase 2B)

---

## Overview

Phase 2B adds LLM-driven analysis on top of the Phase 2A deterministic ledgers. Two agents are defined:

| Agent | Nickname | Purpose | Intended Cadence |
|-------|----------|---------|------------------|
| `hermes-held-spec-ledger` | Keeper | Analyze held-spec blockers, deadline proximity, status transitions | Mon 08:45 ET + daily while specs open |
| `hermes-first-fire-validator` | Scout | Validate first-fire artifacts against pass/fail criteria | One-shot after new cron job's first run |

Both agents are **observe-only** (read ledgers, write analysis to `memory/` — never modify ledgers, never send alerts).

---

## Agent Definitions

Both agents are defined in:

```
agents/<agent-name>/
  IDENTITY.md    — short identity card
  SOUL.md        — core principles, boundaries, output format
  HEARTBEAT.md   — detailed checklist and status codes
  memory/        — analysis outputs (YYYY-MM-DD.md)
```

They are registered in `agents/AGENT_REGISTRY.json` with status `shadow` and `llm_policy: manual_only`.

---

## Manual Invocation

Both agents require inline-data **wrapper scripts** because `run_agent_direct.py` provides no file-access tools to the LLM. The wrappers read the ledger files, embed the data in the message, then invoke the LLM.

### Prerequisites

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
```

**Provider note**: `run_agent_direct.py` supports two routes:
- `claude-*` (e.g. `claude-sonnet-4-20250514`) → Anthropic SDK via `ANTHROPIC_API_KEY`
- `meta-llama/*` (default) → Together AI via `TOGETHER_API_KEY`

Together AI has exhausted free-tier credits (402). Use `--model claude-sonnet-4-20250514` for now.

### hermes-held-spec-ledger (wrapper)

```bash
# Dry-run (recommended first)
python3 tools/run_hermes_held_spec_ledger.py --dry-run --model claude-sonnet-4-20250514

# Standard analysis run (writes to memory/)
python3 tools/run_hermes_held_spec_ledger.py --model claude-sonnet-4-20250514
```

### hermes-first-fire-validator (wrapper)

```bash
# Dry-run (recommended first)
python3 tools/run_hermes_first_fire_validator.py --dry-run --model claude-sonnet-4-20250514

# Standard validation run (writes to memory/)
python3 tools/run_hermes_first_fire_validator.py --model claude-sonnet-4-20250514
```

### Direct invocation (without wrapper — ledger data NOT inlined)

Only useful if the agent already has the data in memory from a previous run:

```bash
python3 tools/run_agent_direct.py --agent hermes-held-spec-ledger --message DRY_RUN --skip-preflight
python3 tools/run_agent_direct.py --agent hermes-first-fire-validator --message DRY_RUN --skip-preflight
```

### Combined run (both agents sequentially)

```bash
python3 tools/run_hermes_held_spec_ledger.py --model claude-sonnet-4-20250514 && \
python3 tools/run_hermes_first_fire_validator.py --model claude-sonnet-4-20250514
```

---

## Inputs & Outputs

### hermes-held-spec-ledger

| | Path |
|---|---|
| Input | `artifacts/ops/held_spec_ledger/latest.json` (inlined by wrapper) |
| Input | `artifacts/ops/knowledge_layer/latest_state.json` (inlined by wrapper) |
| Output | `agents/hermes-held-spec-ledger/memory/YYYY-MM-DD.md` |
| Log | `logs/agents_direct/hermes-held-spec-ledger_YYYYMMDD_HHMMSS_UUID.json` |

### hermes-first-fire-validator

| | Path |
|---|---|
| Input | `artifacts/ops/first_fire_ledger/latest.json` (inlined by wrapper) |
| Input | `artifacts/ops/knowledge_layer/latest_state.json` (inlined by wrapper) |
| Input | Referenced artifacts (per `expected_artifacts[]` in ledger) |
| Input | Referenced log files (per `expected_log` in ledger) |
| Output | `agents/hermes-first-fire-validator/memory/YYYY-MM-DD.md` |
| Log | `logs/agents_direct/hermes-first-fire-validator_YYYYMMDD_HHMMSS_UUID.json` |

---

## Intended Cadence (not yet enabled)

These are the planned cron entries (documented here for future implementation):

```cron
# hermes-held-spec-ledger — Mon 08:45 ET (weekly kickoff)
45 8 * * 1 cd /mnt/c/Projects/biotech_screener/biotech-screener && /usr/bin/python3 tools/run_hermes_held_spec_ledger.py --model claude-sonnet-4-20250514 >> logs/hermes-held-spec-ledger.log 2>&1

# hermes-first-fire-validator — initially manual only, add cron after:
#   - First fire occurs for a job
#   - Agent has been dry-run verified successfully 3+ times
#   - Expected cron line: 45 9 * * 1-5
```

### Activation criteria (for enabling cron):

1. Both agents have been dry-run successfully 3+ times with no errors
2. Analysis output in `memory/` has been reviewed by operator for quality
3. `hermes-held-spec-ledger` triggers no unexpected or hallucinated status proposals
4. `hermes-first-fire-validator` correctly validates the existing `biotech_hedge_report` WARN_DATE_MISMATCH state
5. No API exhaustion issues during dry runs

**Dry-run status as of 2026-05-19**: Both agents pass. held-spec-ledger exit 0 (6 items, correct spec_087_b1b flag). first-fire-validator exit 0 (correct PARTIAL_PASS_WITH_DATE_ISSUE verdict). See dry-run verification below.

---

## Failure Modes

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| `TOGETHER_API_KEY not found` | `.env` missing or key not set | Check `grep TOGETHER_API_KEY .env` |
| `ANTHROPIC_API_KEY not found` | `.env` missing or key not set | Check `grep ANTHROPIC_API_KEY .env` |
| `No context loaded for agent hermes-*` | Agent directory missing | Verify `agents/hermes-held-spec-ledger/` exists |
| `[PREFLIGHT BLOCKED]` | Preflight governance blocking | Run with `--skip-preflight` for prep, or review preflight config |
| `ERROR: 429` | Rate limit | Wait 30-60 min, retry |
| `ERROR: 402` | Exhausted API credits | Add credits, then `hermes auth reset <provider>` |
| Agent produces empty/trivial output | Ledger has zero items (no specs held) | This is valid — report `ALL_CLEAR`, not an error |
| Agent says "I cannot read files" or similar | Using direct `run_agent_direct.py` without wrapper | Use the wrapper scripts instead |
| LLM hallucinates artifact paths or dates | Insufficient context or model confusion | Check INPUTS above — ensure the ledgers are recent and complete |

---

## Disable / Rollback Procedure

Since there are **no live cron entries** yet, rollback is simply:

```bash
# Revert the entire prep branch
git checkout main
git branch -D spec-089-phase-2b-alerting-prep-2026-05-19

# Or, soft reset if merged
git revert <commit-hash>
```

To remove agent definition files without reverting the branch:

```bash
git rm -r agents/hermes-held-spec-ledger/
git rm -r agents/hermes-first-fire-validator/
git rm references/hermes-phase-2b-agent-jobs.md
git rm tools/run_hermes_held_spec_ledger.py
git rm tools/run_hermes_first_fire_validator.py
# Then revert AGENT_REGISTRY.json changes
git checkout main -- agents/AGENT_REGISTRY.json
```

---

## Phase 2B → Phase 2 Transition

When this prep is validated and you're ready to move to full Phase 2:

1. Add cron entries (see Intended Cadence above)
2. Set `"status": "active"` and `"llm_policy": "direct_llama_on_anomaly"` in `AGENT_REGISTRY.json`
3. Wire agent outputs into Town/alerts (see spec_089.md for deferred scope)
4. Implement contradiction escalation logic (contradiction-detector agent)
5. Implement daily operator brief (uses outputs from both agents)

---

## Verification Checklist (for each dry run)

```
[ ] hermes-held-spec-ledger: exit code 0
[ ] hermes-held-spec-ledger: memory/YYYY-MM-DD.md written with analysis (dry-run skips memory)
[ ] hermes-held-spec-ledger: log written to logs/agents_direct/
[ ] hermes-held-spec-ledger: analysis correctly identifies all N held items
[ ] hermes-held-spec-ledger: no proposals to modify ledger files

[ ] hermes-first-fire-validator: exit code 0
[ ] hermes-first-fire-validator: memory/YYYY-MM-DD.md written with validation (dry-run skips memory)
[ ] hermes-first-fire-validator: log written to logs/agents_direct/
[ ] hermes-first-fire-validator: correctly validates WARN_DATE_MISMATCH or PASS
[ ] hermes-first-fire-validator: no proposals to re-execute jobs

[ ] No live cron entries added (verify: crontab -l | grep hermes)
[ ] No autonomous alerts or Town delivery active
[ ] AGENT_REGISTRY.json shows status: shadow for both agents
```

---

## Dry-Run Verification (2026-05-19)

### hermes-held-spec-ledger

```
Model:     claude-sonnet-4-20250514
Exit:      0
Items:     6 held specs analyzed
Findings:  spec_087_b1b flagged 11 days overdue (30k/50k deadline alert at 2026-05-09 09:00 ET)
           Proposed transition: AWAITING_FIRST_FIRE → NEEDS_OPERATOR_DECISION
Proposals: None to modify ledgers — observe-only respected
```

### hermes-first-fire-validator

```
Model:     claude-sonnet-4-20250514
Exit:      0
Jobs:      1 (biotech_hedge_report)
Verdict:   PARTIAL_PASS_WITH_DATE_ISSUE
Findings:  Job executed, hedge_report artifact exists and fresh (61KB, mtime 2026-05-08 19:23)
           BIOSHORT_VERDICT as_of_date=2026-05-01 vs expected 2026-05-08 — matches WARN_DATE_MISMATCH
           Recommendation line "IBB Straight put 15% OTM" present
Proposals: None to re-execute jobs — observe-only respected
```
