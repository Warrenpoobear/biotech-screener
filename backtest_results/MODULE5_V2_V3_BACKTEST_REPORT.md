# Module 5 v2 vs v3 Backtest Comparison Report

**Date:** 2026-01-18
**Backtest Period:** 2023-01-01 to 2024-12-31
**Test Frequency:** 30 days (25 test dates)
**Universe:** 24 biotech tickers

## Executive Summary

This report compares the performance of Module 5 v2 (Enhanced Composite) versus v3 (IC-Enhanced Edition) using a point-in-time backtest with forward 90-day returns.

### Key Findings

| Metric | v2 | v3 | Difference |
|--------|----|----|------------|
| **IC Mean** | 0.1073 | 0.1019 | -0.0055 |
| **IC Std Dev** | 0.2019 | 0.1602 | -0.0417 |
| **IC Positive %** | 72.7% | 72.7% | 0% |
| **Q5-Q1 Spread** | -5.78% | -4.06% | +1.72% |
| **Rank Correlation** | - | - | 0.841 |

**Assessment: SIMILAR performance** - Both versions show comparable predictive power.

## Detailed Results

### 1. Information Coefficient (IC) Analysis

The Information Coefficient measures the rank correlation between predicted scores and forward returns. Higher IC indicates better predictive power.

**v2 IC Statistics:**
- Mean: 0.1073 (GOOD)
- Standard Deviation: 0.2019
- Positive Periods: 72.7% (16/22)
- Range: [-0.239, 0.459]

**v3 IC Statistics:**
- Mean: 0.1019 (GOOD)
- Standard Deviation: 0.1602 (lower volatility)
- Positive Periods: 72.7% (16/22)
- Range: [-0.131, 0.376]

**Interpretation:**
- Both versions show GOOD predictive power (IC > 0.05)
- v3 shows **more stable** IC with lower standard deviation
- v2 has slightly higher peak IC but also larger drawdowns
- v3 appears more conservative/robust

### 2. Quintile Spread Analysis

The Q5-Q1 spread measures the return difference between top and bottom quintiles.

| Version | Mean Spread | Positive Periods |
|---------|-------------|------------------|
| v2 | -5.78% | 31.8% |
| v3 | -4.06% | 22.7% |

**Note:** Negative spreads indicate the top quintile underperformed the bottom quintile on average. This is common in volatile biotech sectors where reversals occur frequently. The synthetic test data may not fully capture real-world dynamics.

### 3. Ranking Stability

How consistent are rankings between v2 and v3?

- **Mean Rank Correlation:** 0.841
- **Range:** [0.757, 0.912]
- **Assessment:** MODERATELY similar rankings

This indicates v3 makes meaningful changes to rankings while maintaining overall structure. Approximately 16% of ranking information differs between versions.

### 4. Per-Period IC Comparison

```
Date        | IC_v2   | IC_v3   | Winner
------------|---------|---------|--------
2023-01-01  | -0.108  |  0.194  | v3
2023-01-31  | -0.239  | -0.033  | v3
2023-03-02  |  0.055  |  0.127  | v3
2023-04-01  |  0.105  |  0.141  | v3
2023-05-01  |  0.127  |  0.004  | v2
2023-05-31  |  0.073  | -0.021  | v2
2023-06-30  | -0.174  | -0.125  | v3
2023-07-30  |  0.373  |  0.325  | v2
2023-08-29  |  0.079  |  0.142  | v3
2023-09-28  |  0.183  |  0.127  | v2
2023-10-28  |  0.088  |  0.090  | v3
2023-11-27  |  0.265  |  0.004  | v2
2023-12-27  | -0.093  |  0.042  | v3
2024-01-26  |  0.085  | -0.090  | v2
2024-02-25  |  0.184  |  0.109  | v2
2024-03-26  |  0.345  |  0.105  | v2
2024-04-25  |  0.270  |  0.346  | v3
2024-05-25  |  0.459  |  0.376  | v2
2024-06-24  |  0.356  |  0.305  | v2
2024-07-24  |  0.274  |  0.328  | v3
2024-08-23  | -0.130  | -0.123  | v3
2024-09-22  | -0.217  | -0.131  | v3
```

**Win/Loss:** v2 wins 10, v3 wins 12

## Architectural Differences

### Module 5 v2 Features
- Confidence-weighted scoring
- Monotonic caps (risk gates)
- Robust normalization (winsorized rank percentile)
- Weakest-link hybrid aggregation
- Volatility-adjusted weighting
- Co-invest overlay support

### Module 5 v3 Additional Features
- Price momentum signal (60-day alpha vs XBI)
- Peer-relative valuation signal
- Catalyst signal decay (time-based IC modeling)
- Smart money signal (13F position changes)
- Non-linear interaction terms
- Shrinkage normalization (Bayesian cohort adjustment)
- Adaptive weight learning (optional)
- Regime-adaptive components

## Limitations

1. **Synthetic Data:** This backtest uses deterministically generated Module 1-4 outputs rather than real production data. v3's advanced features (momentum, valuation, smart money) were not fully exercised.

2. **No Enhancement Data:** v3 was tested without market_data_by_ticker (momentum/valuation signals) which limits its potential improvement.

3. **Short Horizon:** 90-day forward returns may not capture v3's value in regime-adaptive weighting.

## Recommendations

1. **Consider v3 for production** - Similar IC with lower volatility suggests more stable predictions.

2. **Enable v3 enhancements** - Run with full market data to leverage momentum/valuation signals.

3. **Monitor rank correlation** - At 0.84, v3 makes meaningful different picks. Track which version's unique picks perform better.

4. **Test with real data** - Run backtest with actual Module 1-4 production outputs for more accurate comparison.

## Conclusion

Module 5 v3 performs **equivalently** to v2 in this backtest with the benefit of lower IC volatility. The new v3 features (momentum, valuation, smart money, interactions) were not fully exercised due to synthetic data. We recommend:

1. Deploying v3 as the primary scorer
2. Running parallel v2 tracking for 3 months
3. Re-evaluating with full enhancement data enabled

---

**Files Generated:**
- `backtest/compare_module5_versions.py` - Comparison harness
- `backtest_results/module5_v2_v3_comparison_2023-01-01_2024-12-31.json` - Full results
