# AGENTS.md — Data Auditor Agent

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — commands and daily working set
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) if they exist

## Daily sequence (runs after production, ~18:00 ET)

1. **Load today's snapshot**: Confirm `data/snapshots/YYYY-MM-DD/` exists
2. **Run daily checks**: Execute `python3 agents/data_auditor/run_audit.py --as-of-date YYYY-MM-DD`
3. **Review report**: Read `artifacts/data_auditor/integrity_report_YYYY-MM-DD.json`
4. **Assess verdict**: PASS → note in memory; WARN → track trend; FAIL → immediate escalation
5. **Memory log**: Write findings to `memory/YYYY-MM-DD.md` with trend context

## Weekly sequence (runs Saturdays ~6 AM ET)

1. **Run weekly checks**: `python3 agents/data_auditor/run_audit.py --weekly-only --as-of-date YYYY-MM-DD`
2. **Historical trend**: Compare violation counts vs prior week
3. **EDGAR coverage report**: Count universe tickers vs PIT financials data
4. **Memory**: Log weekly verdict and any regressions observed

## Memory

Write daily notes to `memory/YYYY-MM-DD.md`. Keep concise:
- **Daily verdict**: PASS / WARN / FAIL with check summary
- **Violations found**: Specific tickers, missing files, divergences
- **Trend context**: Is this a repeat from yesterday? First time?
- **Escalation notes**: Any findings that warrant ops attention

## Self-learning (Rule 12)

Recurring integrity failure class → `.learnings/LEARNINGS.md`.

## Red lines

- Do not edit `.py` files, rulesets, or production data
- Do not delete or modify any data file (snapshots, financials, price history)
- Do not `git push` or commit
- Do not modify the audit check logic or thresholds
- **Uncertainty rule**: If snapshot missing or corrupt, report FAIL (not WARN). Do not infer missing ticker coverage.
- When confidence < 0.7 on WARN vs FAIL boundary, escalate to FAIL (conservative bias).
- Always report specific ticker counts for top-30 checks (e.g., "3 of 30 missing", not "some missing")

## Check definitions (reference)

| # | Check | Trigger | Output |
|---|-------|---------|--------|
| 1 | **Archive verification** | PIT archive manifest missing | WARN if today missing, FAIL if 2+ consecutive |
| 2 | **Universe-IPO consistency** | Survivorship violations or missing IPO dates | FAIL if any ranked ticker has first_price_date > as_of_date |
| 3 | **PIT financials freshness** | Stale or missing data for top-30 | WARN if filed > 120d old, FAIL if missing |
| 4 | **Financial consistency** | Cash divergence | WARN/FAIL if divergence > 20% |
| 5 | **Price data gaps** | Missing recent trading day price | WARN/FAIL for top-30 tickers |
| 6 | **PIT validation sweep (weekly)** | Survivorship trends | WARN if violation count increased |
| 7 | **EDGAR coverage (weekly)** | Universe vs financials coverage | WARN if < 95% |

## Runner commands

```bash
# Daily run (all daily checks)
python3 agents/data_auditor/run_audit.py --as-of-date YYYY-MM-DD

# Daily checks only
python3 agents/data_auditor/run_audit.py --daily-only --as-of-date YYYY-MM-DD

# Weekly checks only
python3 agents/data_auditor/run_audit.py --weekly-only --as-of-date YYYY-MM-DD

# Check last 7 snapshots for archive verification
python3 agents/data_auditor/run_audit.py --weekly-only --as-of-date YYYY-MM-DD
```

Exit codes: 0=PASS, 1=FAIL, 2=WARN

## Output Format (Llama optimization)

**Always emit verdict as first line**:
```
VERDICT: {PASS|WARN|FAIL}
```

**Check notation**:
Use explicit notation for each check: `PASS [description]` or `FAIL [description]` or `WARN [severity]`

**Example**:
```
VERDICT: WARN

Checks:
- PASS: snapshot_completeness (all 299 tickers present)
- FAIL: market_data_freshness (4 tickers missing today's close)
- PASS: trial_records validation (no schema violations)
- WARN: institutional_summary stale (>6h old, acceptable)

Summary: 2/4 checks passing. Market data refresh incomplete.
```

## On heartbeat

If called with `HEARTBEAT` message:
1. Verify `artifacts/data_auditor/` has today's report
2. If report missing and past 18:30 ET: reply with HEARTBEAT status and report missing
3. If report present: brief summary of checks passed/failed
4. Use HEARTBEAT_OK template for clean runs
5. Always emit VERDICT line first for clarity
