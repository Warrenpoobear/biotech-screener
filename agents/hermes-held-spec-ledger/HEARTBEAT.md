# HEARTBEAT.md — Hermes Held-Spec Ledger Agent

On invocation (documented as `--message DAILY` or `--message CHECK`), run the full held-spec ledger analysis.

## Checklist

### 1. Load inputs

- Read `artifacts/ops/held_spec_ledger/latest.json`
- Read `artifacts/ops/knowledge_layer/latest_state.json`
- If today's `latest.json` is missing or >24h old → flag `STALE_LEDGER`

### 2. Per-item analysis

For each item in `.items[]`:

| Check | What to evaluate |
|-------|-----------------|
| Blocker staleness | Compare `last_evidence` date to today. >7d → stale |
| Deadline proximity | Parse `alert_condition` for dates. ≤3 business days → flag |
| Blocked-by chain | Is the `blocker` referencing another held item? Check if that item may have resolved |
| Artifact existence | Check each `related_artifacts[]` path exists. Missing → flag |
| Not_allowed relevance | Are any `not_allowed` items moot due to external change? |

### 3. Cross-reference with knowledge layer

- Read `key_artifacts` section of latest_state.json for freshness of referenced artifacts
- Check `warnings[]` for any that affect held items

### 4. Write analysis

Write structured markdown to `agents/hermes-held-spec-ledger/memory/YYYY-MM-DD.md`

## Output format

```
HELD-SPEC LEDGER ANALYSIS — YYYY-MM-DD
Ledger as_of: YYYY-MM-DD
Items: N

{item_id}: {CURRENT|REVIEW_NEEDED|DEADLINE_APPROACHING}
  blocker staleness: Xd
  deadline: {none|approaching}
  proposal: {none|<transition suggestion>}

Overall: {ALL_CLEAR|REVIEW|ACTION_NEEDED}
```

## Status codes

- `ALL_CLEAR` — all items current, no approaching deadlines, no stale blockers
- `REVIEW` — at least one item with stale blocker or approaching deadline
- `ACTION_NEEDED` — deadline within 24h, blocker condition likely satisfied, artifact missing

## Dry-run mode

`run_agent_direct.py --agent hermes-held-spec-ledger --message DRY_RUN --skip-preflight`

In dry-run mode: perform the full analysis but preface output with `[DRY RUN]` and do not write to memory even if `--write-memory` is set. (The `run_agent_direct.py --skip-preflight` flag bypasses preflight checks for development; the DRY_RUN message tells this agent to suppress memory writes.)
