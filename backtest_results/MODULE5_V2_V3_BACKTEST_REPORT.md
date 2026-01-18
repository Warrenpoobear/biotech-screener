# Module 5 v2 vs v3 Backtest Comparison Report (v3 - Production Grade)

**Date:** 2026-01-18
**Backtest Period:** 2023-01-01 to 2024-12-31
**Test Frequency:** 30 days (25 test dates, 22 with return data)
**Universe:** 24 biotech tickers
**Top/Bottom N:** 5 tickers for spread calculation
**Cost Model:** 50 bps per turnover

## Executive Summary

**v3 is the clear winner** after adding costed performance and concentration analysis:

| Metric | v2 | v3 | Winner |
|--------|----|----|--------|
| IC Mean | +0.107 | +0.102 | Tie |
| IC t-stat | 2.49 | 2.98 | **v3** |
| Gross Return | +275% | +404% | **v3** |
| **Net Return (after costs)** | +263% | +398% | **v3** |
| Cost Drag | 3.2% | 1.2% | **v3** |
| Turnover | 35% | 12% | **v3** |
| **Top-1 Contribution** | 207.6% | 6.8% | **v3** |
| Median Top-5 Return | +5.1% | +2.3% | v2 |
| Win Rate | 62.7% | 60.9% | Tie |
| Max Drawdown | 22.8% | 25.8% | v2 |

**Critical Finding:** v2's returns are NOT robust - 207% top-1 contribution means a single outlier name is driving all performance. v3's 6.8% top-1 contribution shows balanced, diversified returns.

## Production-Grade Validations

### A) Costed Performance

| Version | Gross Return | Net Return | Cost Drag |
|---------|--------------|------------|-----------|
| v2 | +275.1% | +263.1% | 3.2% |
| v3 | +404.3% | +398.0% | 1.2% |

**v3 wins decisively after costs.** The 3x lower turnover (12% vs 35%) translates to 2.7x lower cost drag.

### B) Concentration / Robustness Check

| Metric | v2 | v3 | Interpretation |
|--------|----|----|----------------|
| Median Top-5 Return | +5.12% | +2.30% | v2 has outliers |
| Top-1 Contribution | **207.6%** | **6.8%** | v2 is one-name-driven |
| Win Rate | 62.7% | 60.9% | Similar |

**CRITICAL:** v2's 207% top-1 contribution means:
- One name contributed more than the total return
- Other names had net negative contribution
- Performance is NOT robust to that name being removed

v3's 6.8% top-1 contribution means:
- Returns are evenly distributed across top picks
- No single-name dependency
- **Robust to removing any single name**

## Sanity Check: IC vs Spread Consistency

- v2: 86.4% periods consistent
- v3: 77.3% periods consistent

Both pass the sanity check (IC sign matches spread sign), though v2 has slightly better consistency.

## IC Analysis

```
Version | Mean   | Std    | t-stat | 95% CI           | Positive %
--------|--------|--------|--------|------------------|----------
v2      | +0.107 | 0.202  | 2.49   | [+0.02, +0.19]   | 72.7%
v3      | +0.102 | 0.160  | 2.98   | [+0.04, +0.16]   | 72.7%
```

- Both t-stats > 2.0 (statistically significant)
- CIs overlap → no statistical difference in IC
- v3 has tighter CI (more reliable)

## Why v2's Higher Median Return is Misleading

v2 shows +5.12% median vs v3's +2.30%, which might seem like v2 is better. But:

1. **Top-1 contribution of 207%** means one outlier is masking poor picks
2. **Mean vs Median divergence** in v2 confirms outlier-driven returns
3. **v3's lower median with 6.8% top-1** means more names are contributing

This is the classic "one lucky pick" vs "systematic edge" distinction.

## Recommendation: Deploy v3 as Default

Based on production-grade validation:

| Criterion | v2 | v3 | Decision |
|-----------|----|----|----------|
| IC | +0.107 | +0.102 | Tie |
| Net Return | +263% | +398% | **v3** |
| Concentration | 207% | 6.8% | **v3** (robust) |
| Cost Drag | 3.2% | 1.2% | **v3** |
| Drawdown | 22.8% | 25.8% | v2 (slight) |

**v3 should be the default scorer.** The combination of:
- Higher net returns after costs
- Diversified returns (not dependent on one name)
- Lower turnover and cost drag
- Similar IC with better t-stat

...makes it clearly superior for production use.

### Deployment Recommendation

1. **Deploy v3 as primary scorer immediately**
2. **Log v2 scores alongside for 4-8 cycles** (monitoring, not gating)
3. **Add divergence alerts** for cases where v2 and v3 top-10 differ significantly
4. **Consider removing v2 after validation period**

### Risk to Monitor

v3's slightly higher max drawdown (25.8% vs 22.8%) suggests it may hold positions longer during adverse moves. Monitor for:
- Drawdown events exceeding 30%
- Correlation with XBI drawdowns
- Consider adding a circuit breaker if drawdown exceeds threshold

---

## Files

- `backtest/compare_module5_versions.py` - Production-grade comparison harness
- `backtest_results/module5_v2_v3_comparison_2023-01-01_2024-12-31.json` - Full results

## v3 Correctness Fixes Verified

The following v3 correctness fixes are in place (verified in code):
- ✅ `_coalesce()` correctly handles `Decimal("0")` as valid (not None)
- ✅ `runway_gate` derived from `runway_months` directly (not liquidity)
- ✅ Uses `is not None` checks throughout breakdown/hash
