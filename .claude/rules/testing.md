---
paths:
  - tests/**
  - .github/**
---

# Testing & CI

## Test Commands
```bash
pytest -p no:warnings                          # full suite
ruff check src tests scripts tools             # lint
ruff format --check src tests scripts tools    # format check
```

## CI Pipeline
- GitHub Actions workflow
- Currently RED since ~May 8 (PR #285 open/unmerged)
- phase2-daily-production cron is dark while CI is red

## Test Requirements for New Signals
1. Unit test with known fixture input -> expected output
2. Leakage test confirming `data_available_timestamp` compliance
3. Ablation test stub showing Sharpe contribution >= 0.1

## Test Conventions
- Use PIT fixtures — never fetch live data in tests
- Tests asserting Tier 3 behavior are themselves Tier 3 (per governance policy)
- All outputs: `encoding='utf-8'`, `lineterminator='\n'`, `quoting=csv.QUOTE_MINIMAL`

## Known Failure Patterns (from failure-patterns skill)
- **F-2026-004**: AACT Pipeline Timeout — Monday runs may take longer (weekend batch). Timeout is 6000s.
- **F-2026-006**: CI Extended Red — CI red > 5 days should trigger merge block and operator escalation.
- **F-2026-002**: IC Tooling Scope Conflation — `run_rank_ic_backtest.py` measured composite_score IC, not production final_score IC. All prior ranker IC claims invalidated.

Check the failure-patterns skill catalog before investigating new test failures.
