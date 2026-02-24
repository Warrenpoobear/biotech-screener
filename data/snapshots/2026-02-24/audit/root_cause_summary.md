# Data Integrity Audit — Root Cause Summary

Generated: 2026-02-24T14:40:09.387217Z

## 1. Price-Derived Cross-Validation

- **Drawdown (252d)**: 319 OK, 0 FAIL, 1 missing/skip (of 320)
- **RSI (14d)**: 320 OK, 0 FAIL, 0 missing/skip (of 320)
- **Beta XBI (60d)**: 320 OK, 0 FAIL, 0 missing/skip (of 320)
- **Alpha (60d)**: 320 OK, 0 FAIL, 0 missing/skip (of 320)
- **Drawdown XBI**: 320 OK, 0 FAIL, 0 missing/skip (of 320)
- **Drawdown Relative**: 319 OK, 0 FAIL, 1 missing/skip (of 320)

## 2. Root Cause Classification

No FAIL rows found — all price-derived fields within tolerance.

## 3. Invariant Violations

No invariant violations found.

## 4. Catalyst Spot-Check

- Sampled 25 tickers
- Days mismatch: 0
- Mode consistency failures: 0
- Window consistency issues: 0
- **All spot-checks passed.**

## 5. Recommendations

### Source Tagging (recommended for all price-derived fields)
- Add `de_drawdown_source`, `de_beta_source`, `de_rsi_source`, `de_alpha_source` columns
- Values: 'price_csv', 'cache', 'pipeline_fallback', 'missing'

### Regression Tests
- Add test for 'stale-but-present' beta overwrite (mirrors test_stale_drawdown_overwrite)
- Add test for 'stale-but-present' RSI overwrite
- Add invariant assertion in CI (fast mode: load rankings.csv, run check_invariants)
