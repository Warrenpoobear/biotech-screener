# Cost-Aware Sizing: ADV Coverage Audit

**Date**: 2026-02-11
**Production market_data**: collected 2026-02-03
**Panel source**: biotech-buckets backtest (10 snapshots, 2025-01-31 to 2025-10-31)
**Ruleset**: 34bb662d (a_floor=0.60, catalyst_near=120, catalyst_mid=180)
**Cost model**: cap=2000 bps, biotech buckets (400/1000/2000 round-trip bps)

---

## 1. Universe ADV Coverage

| Metric | Value |
|--------|-------|
| Universe tickers | 353 |
| market_data.json records | 350 |
| **Tickers with ADV > 0** | **350 / 353 (99.2%)** |
| Missing | BLUE (delisted), VERV (delisted), _XBI_BENCHMARK_ (synthetic) |

**Verdict**: No real-universe tickers lack ADV data. The 3 missing entries are
delisted securities and a benchmark placeholder — none would appear in a live
portfolio.

## 2. Cost Distribution (full universe, N=350)

| Percentile | Round-trip bps |
|------------|----------------|
| P5  | 227 |
| P10 | 269 |
| P25 | 425 |
| P50 (median) | 746 |
| P75 | 1,409 |
| P90 | 2,210 |
| P95 | 2,980 |
| Max | 4,050 |

**Bucket distribution** (biotech thresholds):

| Bucket | cost_mult | Count | Pct |
|--------|-----------|-------|-----|
| <=400 bps | 1.00x | 82 | 23.4% |
| 401-1000 bps | 0.85x | 142 | 40.6% |
| 1001-2000 bps | 0.70x | 83 | 23.7% |
| >2000 bps | 0.55x (floor) | 43 | 12.3% |

All four buckets are well-populated. No degeneracy.

## 3. Dev-Stage Coverage by Tier

| Tier | Total | With ADV | Coverage |
|------|-------|----------|----------|
| A | 3 | 3 | **100%** |
| B | 16 | 16 | **100%** |
| C | 23 | 23 | 100% |
| D | 141 | 141 | 100% |
| **All dev** | **183** | **183** | **100%** |

## 4. A+B Portfolio Coverage

**19 / 19 eligible A+B positions have cost data (100%)**

Cost profile (round-trip bps): min=264, median=688, max=4036, mean=1146

| Bucket | cost_mult | Count | Pct |
|--------|-----------|-------|-----|
| <=400 bps | 1.00x (no haircut) | 5 | 26% |
| 401-1000 bps | 0.85x | 6 | 32% |
| 1001-2000 bps | 0.70x + band step-down | 6 | 32% |
| >2000 bps | 0.55x + band step-down | 2 | 11% |

### Per-Ticker Detail (A+B, sorted by cost descending)

| Ticker | Tier | ADV ($M) | Cost (RT bps) | Bucket | Effect |
|--------|------|----------|---------------|--------|--------|
| NAUT | B | 0.6 | 4,036 | >2000 | 0.55x + band down |
| IKT | B | 0.8 | 3,609 | >2000 | 0.55x + band down |
| TENX | B | 2.8 | 1,896 | 1001-2000 | 0.70x + band down |
| ACRV | B | 2.9 | 1,888 | 1001-2000 | 0.70x + band down |
| ELDN | B | 4.3 | 1,543 | 1001-2000 | 0.70x + band down |
| ARTV | B | 4.4 | 1,533 | 1001-2000 | 0.70x + band down |
| CLYM | B | 5.9 | 1,314 | 1001-2000 | 0.70x + band down |
| ABEO | B | 8.7 | 1,085 | 1001-2000 | 0.70x + band down |
| PVLA | B | 21.0 | 696 | 401-1000 | 0.85x only |
| KALV | A | 21.5 | 688 | 401-1000 | 0.85x only |
| XENE | B | 36.2 | 532 | 401-1000 | 0.85x only |
| DNTH | B | 41.7 | 496 | 401-1000 | 0.85x only |
| VRDN | A | 54.5 | 434 | 401-1000 | 0.85x only |
| NUVL | B | 63.3 | 404 | 401-1000 | 0.85x only |
| PTGX | B | 75.9 | 369 | <=400 | no haircut |
| LQDA | B | 83.8 | 352 | <=400 | no haircut |
| CELC | B | 94.4 | 331 | <=400 | no haircut |
| MRUS | B | 112.8 | 304 | <=400 | no haircut |
| AKRO | A | 150.0 | 264 | <=400 | no haircut |

## 5. Archive vs Production Cross-Check

Production market_data.json and archive market_data.json (2025-10-31) produce
**identical ADV values** for all checked tickers. Archives snapshot the same
market_data.json at collection time, so there is no pipeline-specific data gap.

## 6. Missing Tickers

| Ticker | Status | In Screen | In market_data.json |
|--------|--------|-----------|---------------------|
| BLUE | Delisted | No | No |
| VERV | Delisted | No | No |
| _XBI_BENCHMARK_ | Synthetic | No | No |

No actionable gaps. No backfill needed.

---

## Promotion Assessment

| Gate | Threshold | Observed | Status |
|------|-----------|----------|--------|
| A+B coverage | >= 60% costed | 100% (19/19) | **PASS** |
| Bucket spread | all 4 populated | 26/32/32/11% | **PASS** |
| No missing A/B | 0 missing | 0 | **PASS** |
| Eligibility unchanged | same with/without | confirmed (backtest) | **PASS** |

**Conclusion**: ADV coverage is not a blocker. All A+B portfolio positions have
cost data. The bucket distribution produces meaningful differentiation (8 names
get 0.70x+band-down, 2 names get 0.55x+band-down, 6 names get 0.85x, 5 names
get no haircut). The cost-aware sizing feature can be promoted from a data
availability standpoint.
