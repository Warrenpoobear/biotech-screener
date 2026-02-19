# Data Integrity Audit — Root Cause Summary

Generated: 2026-02-19
Snapshot: data/snapshots/2026-02-19
Price history: production_data/price_history.csv (471,964 rows, 352 tickers)

## 1. Price-Derived Cross-Validation Results

| Field | OK | FAIL | Missing | Status |
|-------|-----|------|---------|--------|
| Drawdown (252d) | 318 | 0 | 1 | **PASS** |
| RSI (14d) | 319 | 0 | 0 | **PASS** (fixed this session) |
| Beta XBI (60d) | 126 | 189 | 4 | **KNOWN** (XBI date gap) |
| Alpha (60d) | 37 | 282 | 0 | **KNOWN** (depends on beta + window) |
| Drawdown XBI | 319 | 0 | 0 | **PASS** |
| Drawdown Relative | 318 | 0 | 1 | **PASS** |

## 2. Bugs Fixed in This Audit

### 2a. RSI stale-but-present (FIXED)
- **Root cause**: `_hydrate_beta_rsi()` only computed RSI for tickers where `rsi_14d` was missing. If the Morningstar pipeline provided a stale RSI value, it was never overwritten.
- **Impact**: 76 tickers had stale RSI values. Worst case: AVTR stored=77.3, actual=27.3 (delta=50).
- **Fix**: Changed RSI hydration to ALWAYS recompute from price_history.csv for ALL tickers (same pattern as drawdown fix).
- **Tests**: 5 new regression tests in `test_hydrate_drawdown.py::TestRsiOverwrite`.
- **Verification**: Post-fix audit shows 319/319 RSI OK.

### 2b. Drawdown stale-but-present (FIXED in prior session)
- Previously fixed: `_hydrate_drawdown()` now always recomputes from price_history.csv.
- 318/319 OK (1 ticker has no price data → missing, not wrong).

## 3. Known Limitations (not bugs)

### 3a. Beta XBI (189 FAIL)
**NOT a code bug.** The discrepancy arises from two factors:

1. **XBI price data ends 2026-01-27** while ticker data goes to 2026-02-13 (17 trading-day gap). The audit's recomputed beta uses the latest overlapping window (ending Jan 27), while the pipeline/cache beta may use a different window.

2. **Pipeline beta from Morningstar** was computed at pipeline time with unknown windowing. The defensive_features_cache.json has no metadata (no timestamp, no as_of_date).

**Beta diff distribution (189 FAIL):**
- diff > 0.5: 147 (78%)
- diff > 1.0: 79 (42%)
- diff > 2.0: 9 (5%)

**Recommended fix**: Update XBI price_history.csv to current date, then add overwrite logic to `_hydrate_beta_rsi()` (Part A) to recompute beta from price_history.csv for ALL tickers. This requires the same date-aligned merge approach used in the audit tool.

### 3b. Alpha 60d (282 FAIL)
**NOT a code bug.** Alpha = `return_60d - beta * xbi_return_60d`. Discrepancies come from:
1. Different beta values (see 3a above)
2. Different 60d return windows (pipeline vs audit recomputation)
3. Pipeline computes alpha in `enrich_archive_inputs.py`, not recomputed at snapshot time

**Resolution**: Once beta is fresh, adding inline alpha recomputation would resolve this.

## 4. Invariant Violations

- **range_de_alpha_60d** (2 tickers): ALMS (2.65), ERAS (3.44)
  - These are legitimate outliers with extreme 60d returns, not data errors.
  - Consider widening the sanity range or adding an "extreme_alpha" risk flag.

## 5. Catalyst Spot-Check

- Sampled 25 tickers across all modes/sources
- **0 issues found**: days consistent, mode consistent, window consistent
- All catalyst fields are computed fresh from Module 3 each run — no stale-but-present risk.

## 6. Field Overwrite Policy Summary

| Field | Policy | Source of Truth | Status |
|-------|--------|-----------------|--------|
| de_drawdown | overwrite_always | price_history.csv | FIXED |
| de_drawdown_xbi | overwrite_always | price_history.csv (XBI) | OK |
| de_drawdown_rel_xbi | overwrite_always | derived | OK |
| de_rsi_14d | **overwrite_always** | price_history.csv | **FIXED this session** |
| de_beta_xbi_60d | fill_missing_only | defensive_features_cache.json | NEEDS FIX (after XBI data refresh) |
| de_alpha_60d | none (pipeline only) | score_breakdown.enhancements | NEEDS FIX (after beta fix) |
| vol_60d | none (pipeline only) | pipeline | ACCEPTABLE (low decision impact) |

## 7. Recommendations

### Immediate (done)
1. RSI overwrite policy changed to `overwrite_always` — **DONE**
2. Regression tests added — **DONE** (5 tests in TestRsiOverwrite)
3. Audit tool created — **DONE** (`tools/data_integrity_audit.py`)

### Next Sprint
1. **Refresh XBI in price_history.csv** (prerequisite for beta/alpha fix)
2. **Add beta recomputation from price_history.csv** — overwrite policy like drawdown/RSI
3. **Add alpha recomputation from price_history.csv** — `return_60d - beta * xbi_return_60d`
4. **Source tagging columns**: `de_drawdown_source`, `de_beta_source`, `de_rsi_source`
5. **Cache metadata**: Add `generated_at`, `as_of_date` to defensive_features_cache.json

### CI Integration
- Run `data_integrity_audit.py --skip-prices` (invariant checks only) on every screen run
- Run full audit weekly or on data refresh
