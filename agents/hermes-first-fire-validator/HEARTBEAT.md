# HEARTBEAT.md — Hermes First-Fire Validator Agent

On invocation (`--message VALIDATE` or `--message CHECK`), run the full first-fire validation.

## Checklist

### 1. Load inputs

- Read `artifacts/ops/first_fire_ledger/latest.json`
- Read `artifacts/ops/knowledge_layer/latest_state.json`

### 2. Per-job validation

For each job in `.jobs[]`:

| Check | Method |
|-------|--------|
| Expected first-fire | Has `expected_first_fire` passed? Parse as datetime |
| Artifact presence | Check each path in `expected_artifacts[]` exists and is non-empty |
| Artifact date | Read each artifact's embedded `as_of_date` or mtime vs expected |
| Log file | Check log in `expected_log` exists, has recent timestamp, no ERROR/EXCEPTION |
| Pass criteria | Evaluate each criterion against file evidence |
| Fail criteria | Evaluate each criterion against file evidence |
| Deadline | If past `alert_deadline` with criteria unmet → flag as OVERDUE |

### 3. Cross-reference with knowledge layer

- Check `key_artifacts` section for referenced artifacts
- Check `warnings[]` for any that affect this job

### 4. Write validation report

Write structured markdown to `agents/hermes-first-fire-validator/memory/YYYY-MM-DD.md`

## Output format

```
FIRST-FIRE VALIDATION — YYYY-MM-DD
Jobs: N

{job_name}: {PASS|FAIL|NEEDS_INVESTIGATION|NOT_YET_DUE}
  criteria passed: X/N
  artifacts: {ALL_FOUND|MISSING|STALE}
  deadline: {OK|OVERDUE}

Overall: {ALL_CLEAR|ATTENTION|OVERDUE}
```

## Status codes

- `ALL_CLEAR` — all jobs passed all criteria, or waiting for not-yet-due first fires
- `ATTENTION` — at least one job has FAIL criteria or partial pass
- `OVERDUE` — past `alert_deadline` with criteria unmet

## Dry-run mode

`run_agent_direct.py --agent hermes-first-fire-validator --message DRY_RUN --skip-preflight`

In dry-run mode: perform full validation but preface output with `[DRY RUN]` and suppress memory writes.
