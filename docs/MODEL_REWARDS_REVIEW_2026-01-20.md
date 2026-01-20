# Model Rewards Review - 2026-01-20

## Executive Summary

**Critical Finding**: The biotech screener is producing a portfolio skewed toward commercial-stage cashflow names rather than early-stage catalyst biotech because **multiple data pipelines are completely broken**, causing the model to default to neutral scores (50) for key signals.

## Key Issues Identified

### 1. CRITICAL: Catalyst Pipeline Failure

**Diagnostic Evidence** (from `checkpoints/module_5_2026-01-18.json`):
```
events_detected_total: 0
tickers_with_events: 0
in_catalyst_window: 0 / 308
```

**Impact**:
- ALL 308 tickers receive `catalyst_score = 50.00` (neutral default)
- `catalyst_confidence = 0.25` (low) for all tickers
- `catalyst_window_bucket = "UNKNOWN"` for all tickers
- Catalyst proximity and delta bonuses = 0 for all tickers

The entire Module 3 catalyst signal is non-functional. Every ticker gets the same neutral score.

### 2. CRITICAL: Market Data Pipeline Failure

**Diagnostic Evidence**:
```
with_market_data: 0 / 308
with_momentum_signal: 0 / 308
```

**Impact**:
- ALL tickers have `momentum: raw=None, normalized=50, confidence=0.3`
- No volatility adjustments being applied
- Flag `momentum_data_incomplete` on every single security

### 3. CRITICAL: Smart Money Pipeline Failure

**Diagnostic Evidence**:
```
with_smart_money: 0 / 308
```

**Impact**:
- 13F holdings data not being loaded
- No institutional sentiment signal

---

## Per-Ticker Score Breakdown (Top 10)

| Rank | Ticker | Composite | Clinical | Financial | Catalyst | Momentum | PoS | Defaults Used |
|------|--------|-----------|----------|-----------|----------|----------|-----|---------------|
| 1 | NVAX | 88.49 | 100.00 | 97.84 | 50.00 | 50 | 100.00 | momentum |
| 2 | SIGA | 80.61 | 90.94 | 96.78 | 50.00 | 50 | 58.77 | momentum, pos_cap |
| 3 | INDV | 78.30 | 100.00 | 71.60 | 50.00 | 50 | 100.00 | momentum |
| 4 | KROS | 77.48 | 90.90 | 100.00 | 50.00 | 50 | 38.43 | momentum |
| 5 | IMCR | 76.90 | 97.84 | 100.00 | 50.00 | 50 | 14.51 | momentum |
| 6 | ALKS | 76.50 | 100.00 | 79.24 | 50.00 | 50 | 58.77 | momentum, pos_cap |
| 7 | CADL | 75.93 | 85.49 | 99.38 | 50.00 | 50 | 38.43 | momentum |
| 8 | BBOT | 72.67 | 100.00 | 76.23 | 50.00 | 50 | 100.00 | momentum |
| 9 | MIRM | 72.01 | 94.75 | 73.15 | 50.00 | 50 | 57.72 | momentum |
| 10 | TGTX | 64.01 | 90.90 | 36.11 | 50.00 | 50 | 100.00 | momentum |

**Key Observation**: Every single top 10 name has:
- Catalyst = 50.00 (default)
- Momentum = 50 (default)

---

## Why Commercial-Stage Names Are Winning

With catalyst and momentum signals broken (all defaulting to 50):

**Effective Scoring Drivers** (by actual contribution):
| Component | Effective Weight | Working? |
|-----------|-----------------|----------|
| Clinical | ~30% | YES |
| Financial | ~23% | YES |
| PoS | ~15% | PARTIAL (late-stage capped at 100) |
| Catalyst | ~8% | NO (all defaults) |
| Momentum | ~5% | NO (all defaults) |
| Valuation | ~3% | YES |

**Commercial-stage companies win because:**
1. High financial scores (cash flow positive, low burn rate)
2. High clinical scores (approved drugs = maximum stage score)
3. PoS scores capped at 100% for approved/commercial stage

**Early-stage catalyst biotech SHOULD win on:**
1. Catalyst proximity bonuses (BROKEN)
2. Catalyst event momentum (BROKEN)
3. Price momentum around catalysts (BROKEN)

---

## Default Value Masking Problem

The current system hides data pipeline failures by:

```python
# From module_5_composite_v3.py lines 757-760
clin_norm = _coalesce(..., default=Decimal("50"))
fin_norm = _coalesce(..., default=Decimal("50"))
cat_norm = _coalesce(..., default=Decimal("50"))  # <-- PROBLEM
```

**Issues:**
1. Missing catalyst data → neutral score (50), not penalty
2. Missing momentum data → neutral score (50), not penalty
3. Confidence penalty only reduces weight by 50% at worst
4. Uncertainty penalty caps at 30% even with multiple missing components

**Recommendation**: Consider adding a `data_coverage_gate` that EXCLUDES or HEAVILY PENALIZES securities when:
- Catalyst data missing (current: neutral score)
- Market data missing (current: neutral score)
- Multiple components defaulting

---

## Position Sizing Analysis

**Current Settings:**
```
top_n_cutoff: 60
with_nonzero_weight: 60
total_allocated_weight: 0.8979 (90% target)
```

**Average weight** = 90% / 60 = 1.5%

**Dynamic floor logic** (from `defensive_overlay_adapter.py`):
- ≤50 securities: 1.0% floor
- 51-100 securities: 0.5% floor
- 101-200 securities: 0.3% floor

With 60 securities, the 0.5% floor is likely binding for low-scoring names, explaining why 8/10 are at minimum weight in your observation.

---

## Recommended Actions

### Immediate (Data Pipeline Fixes)

1. **Debug Catalyst Pipeline**: Zero events detected across 308 tickers is clearly broken
   - Check `production_data/catalyst_events_*.json` generation
   - Verify trial snapshot comparison is working
   - Check AACT data freshness

2. **Add Market Data Feed**: Momentum signals completely missing
   - Verify `production_data/market_data.json` is populated
   - Check price history data pipeline

3. **Add Smart Money Feed**: 13F holdings data not being loaded
   - Check `production_data/holdings_snapshots.json` integration

### Model Risk Mitigations

4. **Add Data Coverage Gate**: Don't let missing data hide behind neutral scores
   ```python
   # Proposed: Penalize or exclude when critical signals missing
   if n_defaulted_signals >= 2:
       severity = "sev2"  # 50% penalty
   if n_defaulted_signals >= 3:
       severity = "sev3"  # Exclude
   ```

5. **Add Pipeline Health Check**: Pre-flight validation before scoring
   ```python
   assert catalyst_events_detected > 0, "Catalyst pipeline broken"
   assert market_data_coverage > 0.5, "Market data missing"
   ```

---

## Files Reviewed

- `checkpoints/module_5_2026-01-18.json` - Full scoring output
- `production_data/catalyst_events_vnext_2026-01-18.json` - Zero events
- `module_5_composite_v3.py` - Default value handling (lines 757-760)
- `defensive_overlay_adapter.py` - Position sizing logic (lines 161-182)

---

## Conclusion

The model is NOT broken in its logic - the weights and scoring formulas are sound. However, **multiple upstream data pipelines have failed silently**, and the model's default-to-neutral behavior is masking these failures, causing the portfolio to drift toward commercial-stage names where the working signals (clinical + financial) dominate.

**Priority**: Fix the catalyst and market data pipelines before trusting the ranking output.
