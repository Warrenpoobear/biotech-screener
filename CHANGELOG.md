# Changelog

All notable changes to the biotech screener will be documented in this file.

## [v4.0.1] - 2026-01-30

### Summary
Robustness improvements to cluster threshold computation and better diagnostics.

### Fixed

#### Cluster Threshold Score-Key Drift (P0)
- **Bug**: `compute_cluster_percentile_thresholds()` always used `composite_score`, even after defensive overlay modified it
- **Impact**: Thresholds computed on post-defensive scores caused circular dependency
- **Fix**: Prefer `composite_score_before_defensive` when present
- **Result**: Cluster thresholds now computed on true pre-defensive scores

#### Percentile Gating Inactive Warning (P0)
- **Bug**: No alert when floor dominated ALL clusters, making percentile gating ineffective
- **Impact**: Silent degradation of boost eligibility gate
- **Fix**: Emit warning when all clusters are floor_dominated; add `all_clusters_floor_dominated` to diagnostics
- **Result**: Operators alerted when percentile gating has no effect

### Files Modified
- `defensive_overlay_adapter.py` - Score-key drift fix, floor-dominated warning
- `tests/test_defensive_overlay_adapter.py` - 4 new P0 regression tests

---

## [v4.0.0] - 2026-01-30

### Summary
Major scoring fixes to eliminate artificial clustering and improve signal differentiation.
88.6% of rankings now driven by pure alpha signals.

### Fixed

#### Smart Money Score Compression (P0)
- **Bug**: `_saturating_bonus()` was computed but never used in actual score calculation
- **Impact**: 9 tier1 holders → score of 54 (should be 80)
- **Root cause**: Score used raw overlap weight sums instead of saturated bonus
- **Fix**: Apply saturation to combined overlap before splitting into elite_core/conditional contributions
- **Result**: PTCT (9 holders) now scores 80 instead of 54

**Parameters changed** (`src/modules/ic_enhancements.py`):
```python
SMART_MONEY_OVERLAP_SATURATION_THRESHOLD: 1.5 → 4.0
SMART_MONEY_OVERLAP_LINEAR_BONUS: 12 → 8
SMART_MONEY_OVERLAP_MAX_BONUS: 20 → 32
```

#### Momentum Score Ceiling Clustering (P0)
- **Bug**: 62 tickers clustered at identical 92.75 score
- **Impact**: Loss of differentiation among top momentum names
- **Root cause**: Linear slope (150) caused 30%+ alphas to hit ceiling before shrinkage
- **Fix**:
  1. Added `_log_dampen_alpha()` for extreme alpha dampening
  2. Reduced slope from 150 → 80
  3. Raised ceiling from 95 → 99
- **Result**: Only 20 tickers at new ceiling (96.55), scores spread across 70-95 range

**Parameters changed** (`src/modules/ic_enhancements.py`):
```python
MOMENTUM_SLOPE: 150 → 80
MOMENTUM_SCORE_MAX: 95 → 99
MOMENTUM_LOG_DAMPEN_THRESHOLD: (new) 0.30
```

#### Boost Eligibility Gate Calibration
- **Bug**: V4 slope reduction improved OPK's pre-defensive score from 48.6 → 50.0, pushing it over the floor=49 gate
- **Impact**: OPK jumped to rank 4 despite weak alpha (negative momentum, zero smart money)
- **Fix**: Raised eligibility floor from 49 → 51
- **Result**: Only INCY (55.7) and VTRS (51.2) qualify for elite boosts

**Parameters changed** (`defensive_overlay_adapter.py`):
```python
boost_eligibility_floor: 49 → 51
```

### Metrics

| Metric | Before V4 | After V4 |
|--------|-----------|----------|
| Momentum @ ceiling | 62 @ 92.75 | 20 @ 96.55 |
| Smart money (PTCT 9 holders) | 54 | 80 |
| Elite boosts | 5 | 2 |
| Alpha-driven rankings | 85.3% | 88.6% |
| OPK rank | 4 (boosted) | 63 (gated) |

### Rank Driver Distribution (Final)
- `alpha`: 272 (88.6%)
- `suppressed`: 25 (8.1%)
- `defensive_penalty`: 8 (2.6%)
- `defensive_boost`: 2 (0.7%)

### Files Modified
- `src/modules/ic_enhancements.py` - Scoring parameter tuning, log-dampening, saturation fix
- `defensive_overlay_adapter.py` - Boost eligibility floor adjustment

### Testing Notes
- Sanity comparison shows only 3 names moved ≥20 ranks (the correctly gated names)
- Median rank churn: 0.0, Mean: 1.3
- No collateral damage to alpha-driven rankings

---

## [v3.0.0] - 2026-01-29

### Added
- Fundamental Red-Flag Suppressor v1.0
- Boost eligibility gate (percentile-within-cluster + floor)
- `rank_driver` field for IC audit (alpha | defensive_boost | defensive_penalty | suppressed)
- Diversification proof columns in CSV export (corr_xbi, beta_xbi)

### Fixed
- Clustering now runs BEFORE defensive overlay (was after, causing empty thresholds)
- Cache merge uses `overwrite=True` to replace stale values

---

## [v2.0.0] - 2026-01-15

### Added
- Defensive overlay with elite/penalty buckets
- Smart money V3 with Elite Core vs Conditional separation
- Multi-window momentum blending (20d/60d/120d)

---

## [v1.0.0] - 2026-01-01

### Initial Release
- Module 1-5 pipeline
- Clinical, financial, catalyst scoring
- Basic composite ranking
