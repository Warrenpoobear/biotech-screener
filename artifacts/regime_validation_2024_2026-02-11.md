# 2024 Regime Validation — Ruleset 131800e4

**Date**: 2026-02-11
**Ruleset**: 131800e4 (a_floor=0.60, catalyst_near=120, mid=180,
             dd_rel=-0.25, enable_cost_haircut=True, cap=1000)
**Archives**: 12 snapshots (2024-01-31 to 2024-12-31)
**Baseline**: 2025 regime (10 snapshots) for comparison

## Executive Summary

**Result: NO NEW BREAKAGE** from cost-aware sizing. The 2024 regime is hostile
as previously established (broken catalyst feed, negative alpha), but the
cost haircut does not make it worse. Membership and tier assignments are
identical with and without the haircut. The drift guardrail system correctly
flags 4/12 dates as FAIL (A%<2%), which is the desired behavior.

## Regime Comparison (2024 vs 2025)

| Metric                 | 2024        | 2025        | Delta       |
|------------------------|-------------|-------------|-------------|
| AB-CD separation (60d) | -6.4pp      | +7.3pp      | -13.7pp     |
| A-tier mean return     | +5.9%       | +18.4%      | -12.5pp     |
| B-tier mean return     | -5.1%       | +39.2%      | -44.3pp     |
| A-tier hit rate        | 49.0%       | 59.5%       | -10.5pp     |
| B-tier hit rate        | 34.7%       | 72.3%       | -37.6pp     |
| Eligible mean return   | -4.5%       | +23.4%      | -27.9pp     |
| Gross residual (mean)  | -3.7pp      | +24.6pp     | -28.3pp     |
| Mean turnover          | 33.6%       | 34.6%       | -1.0pp      |
| Median cost (port.)    | 602 bps     | 567 bps     | +35 bps     |
| Portfolio count        | 20/20       | 16-20       | —           |
| Rescued by rel. gate   | 8           | 49          | -41         |

## Tier Separation (60d)

| Tier | 2024 N | 2024 Mean | 2024 Hit | 2025 N | 2025 Mean | 2025 Hit |
|------|--------|-----------|----------|--------|-----------|----------|
| A    | 53     | +5.9%     | 49.0%    | 43     | +18.4%    | 59.5%    |
| B    | 257    | -5.1%     | 34.7%    | 143    | +39.2%    | 72.3%    |
| C    | 407    | -5.5%     | 36.5%    | 201    | +12.8%    | 59.7%    |
| D    | 1397   | +5.7%     | 37.8%    | 1421   | +29.0%    | 65.5%    |

Key: In 2024, tier signal is **inverted** (D outperforms B, C nearly matches B).
This is the known catalyst-broken regime where tier assignments carry noise.

## Per-Snapshot Detail

| Date       | Elig% | A cnt | A% dev | Cat miss% | Gross resid |
|------------|-------|-------|--------|-----------|-------------|
| 2024-01-31 | 32.6  | 3     | 1.7    | 67.9      | -14.2       |
| 2024-02-29 | 33.5  | 3     | 1.7    | 72.4      | +5.1        |
| 2024-03-29 | 35.3  | 3     | 1.7    | 75.4      | +0.4        |
| 2024-04-30 | 36.0  | 5     | 2.9    | 66.7      | +5.9        |
| 2024-05-31 | 30.1  | 6     | 3.4    | 71.7      | +15.5       |
| 2024-06-28 | 29.4  | 5     | 2.8    | 63.5      | +2.4        |
| 2024-07-31 | 32.2  | 6     | 3.4    | 54.4      | -13.2       |
| 2024-08-30 | 33.9  | 5     | 2.8    | 51.7      | -14.1       |
| 2024-09-30 | 36.5  | 5     | 2.8    | 47.7      | -26.4       |
| 2024-10-31 | 38.2  | 4     | 2.2    | 48.5      | -8.9        |
| 2024-11-29 | 36.3  | 3     | 1.7    | 47.7      | -4.6        |
| 2024-12-31 | 33.0  | 5     | 2.8    | 45.8      | +7.0        |

## Cost Haircut Impact

**Membership**: Identical with and without haircut (same 20 names, same order).
**Tier assignments**: Identical (cost haircut is L3-only, no eligibility/tier effect).
**Turnover**: Identical (66.4% overlap, 33.6% turnover in both cases).
**Weight distribution**: Haircut shifts weight from illiquid to liquid names.

Cost bucket distribution (portfolio positions, N=240):

| Bucket        | Count | Share |
|---------------|-------|-------|
| <=400bps 1.0x | 38    | 15.8% |
| <=1000bps 0.85x | 124 | 51.7% |
| <=2000bps 0.70x | 71  | 29.6% |
| >2000bps 0.55x | 7    | 2.9%  |

Median round-trip cost: 602 bps (vs 567 bps in 2025 — slightly higher due to
smaller 2024 universe having fewer liquid names in A+B).

## Drift Guardrail Evaluation

| Guardrail               | Threshold | Dates tripped | Verdict |
|--------------------------|-----------|---------------|---------|
| A% < 2% (FAIL)          | 2.0%      | 4/12          | EXPECTED |
| Catalyst missing > 85%  | 85.0%     | 0/12          | PASS    |
| Top-25 overlap < 50%    | 50.0%     | 1/11          | MARGINAL |
| Optionality std < 0.10  | 0.10      | (not computed)| —       |

The 4 FAIL dates (Jan-Mar + Nov 2024) correspond to A-count=3 (1.7% of dev).
This is the **correct behavior** — the guardrail system detects the hostile
regime and would have triggered INVESTIGATE or ROLLBACK_RECOMMENDED.

## Catalyst Regime

| Strength | 2024 N | 2024 Mean | 2025 N | 2025 Mean |
|----------|--------|-----------|--------|-----------|
| near     | 262    | +2.7%     | 301    | +33.2%    |
| mid      | 104    | +9.6%     | 128    | +25.8%    |
| far      | 336    | +5.0%     | 308    | +31.9%    |
| missing  | 1412   | +1.4%     | 1071   | +25.4%    |

In 2024, catalyst_missing drops from 68% (Jan) to 46% (Dec) as the feed
improves through the year. But even with catalyst data, the signal is weak
(near: +2.7% vs missing: +1.4%, spread of only 1.3pp, vs 7.8pp in 2025).

## Conclusions

1. **Cost-aware sizing introduces NO new failure modes** in 2024.
2. **2024 remains a hostile regime** with negative alpha and inverted tiers —
   this is structural (broken catalyst feed) and unrelated to cost haircut.
3. **Guardrail system works as designed**: 4/12 FAIL triggers on A%<2%
   would correctly flag the need for investigation.
4. **No parameter change needed**: the cost haircut is neutral-to-slightly-positive
   in 2024 (identical membership, weight redistribution toward liquid names).
5. **The modeling fork for 2024** is catalyst feed quality, not cost structure.
   Regime-conditional thresholds won't help without better underlying data.
