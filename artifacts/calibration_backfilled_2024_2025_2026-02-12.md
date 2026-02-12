# Calibration Sweep: Backfilled vs Original Archives (2024+2025)

**Date**: 2026-02-12
**Panels**: `walkforward_panel__backfilled_combined.csv` / `walkforward_panel__original_combined.csv`
**Grid**: 8 a_floor (0.40..0.70) x 5 catalyst_near (60..180) = 40 combos
**Archives**: 22 snapshots (12 x 2024, 10 x 2025), 3922 panel rows each

## Decision

**PARK backfill.** Do not promote `trial_active` into the production enrichment pipeline.

The backfill adds +24pp catalyst coverage but has **zero effect** on tier assignment,
eligibility, or calibration outcomes. The 2024 regime remains structurally inverted
regardless of backfill status. The next leverage point is stronger catalyst sources
(SEC 8-K / conference calendars) or eligibility gate investigation.

---

## Summary Table (4 calibration sweeps)

| Panel | Date Range | Winner | Sep | Score | cat_miss% | Pass | AB DD |
|-------|-----------|--------|-----|-------|-----------|------|-------|
| Backfilled | 2024+2025 | a=0.58 cn=60 | +1.95pp | -5.22 | 31.7% | 30/40 | -23.91 |
| Original | 2024+2025 | a=0.58 cn=60 | +1.95pp | -5.22 | 55.2% | 30/40 | -23.91 |
| Backfilled | 2025-only | a=0.70 cn=60 | +8.05pp | +2.10 | 26.6% | 40/40 | -19.83 |
| Original | 2025-only | a=0.70 cn=60 | +8.05pp | +2.10 | 47.3% | 40/40 | -19.83 |

**Every metric is identical** between backfilled and original. The only difference is
the DQ catalyst_missing percentage (cosmetic; both pass the 80% gate).

## Panel-Level Deltas (Backfilled vs Original)

| Metric | Count | % of 3922 |
|--------|------:|----------:|
| Tier changed | 0 | 0.0% |
| Eligibility changed | 0 | 0.0% |
| Catalyst_mode changed | 946 | 24.1% |
| Weight changed | 360 | 9.2% |

All 946 catalyst_mode changes are `missing -> specific_days` (with days=366, strength=far).
Weight changes are within-tier reordering effects; they do not propagate to calibration.

## Catalyst Coverage (backfill effect)

| Regime | missing (orig) | missing (bkf) | Delta |
|--------|----------------|---------------|-------|
| 2024 | 1412 (66.8%) | 893 (42.2%) | -24.6pp |
| 2025 | 1071 (59.2%) | 644 (35.6%) | -23.6pp |
| All | 2483 (63.3%) | 1537 (39.2%) | -24.1pp |

Coverage improves substantially, but ALL moved rows land in the "far" band (366 days),
which cannot promote to A-tier (requires near <= 120d or mid <= 180d).

## Per-Year Tier Separation (baseline, a_floor=0.55)

| Year | Tier | n | Mean 60d | Median 60d | Hit% |
|------|------|--:|:--------:|:----------:|-----:|
| 2024 | A | 50 | +6.02% | +0.96% | 50.0% |
| 2024 | B | 231 | -5.35% | -8.88% | 34.6% |
| 2024 | C | 340 | -5.44% | -7.54% | 36.8% |
| 2024 | D | 1405 | +5.60% | -9.75% | 37.7% |
| 2025 | A | 38 | +18.17% | +12.61% | 60.5% |
| 2025 | B | 122 | +36.28% | +17.17% | 71.3% |
| 2025 | C | 176 | +13.32% | +8.57% | 60.8% |
| 2025 | D | 1457 | +29.16% | +14.99% | 65.5% |

**AB-CD separation**:
- 2024: median +1.02pp (weak), mean **-6.78pp** (deeply inverted)
- 2025: median +1.34pp, mean **+4.53pp** (healthy)
- Combined: median +0.96pp baseline, +1.95pp at a_floor=0.58 (best)

## Top 10 Configs (2024+2025 combined, identical across backfilled/original)

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 | **0.58** | 60 | 9.0% | +1.95pp | -23.91 | -5.22 | Y |
| 2 | 0.58 | 90 | 9.0% | +1.95pp | -23.91 | -5.22 | Y |
| 3 | 0.58 | 120 | 9.0% | +1.95pp | -23.91 | -5.22 | Y |
| 4 | 0.58 | 150 | 9.0% | +1.95pp | -23.91 | -5.22 | Y |
| 5 | 0.58 | 180 | 9.0% | +1.95pp | -23.91 | -5.22 | Y |
| 6 | 0.70 | 60 | 7.4% | +1.97pp | -24.05 | -5.24 | Y |
| 7 | 0.70 | 90 | 7.4% | +1.97pp | -24.05 | -5.24 | Y |
| 8 | 0.60 | 60 | 8.6% | +1.69pp | -23.85 | -5.46 | Y |
| 9 | 0.60 | 90 | 8.6% | +1.69pp | -23.85 | -5.46 | Y |
| 10 | 0.60 | 120 | 8.6% | +1.69pp | -23.85 | -5.46 | Y |

**Ridge**: catalyst_near is completely flat (zero discriminating power on combined data).
The a_floor parameter does all the work.

**Failing configs**: a_floor=0.45 and 0.50 (negative separation).

## Why Backfill Has Zero Effect

1. `trial_active` emits `days_to_catalyst = window_days + 1` (366 days)
2. 366d maps to `catalyst_strength = "far"` in the decision engine
3. A-tier requires `near` (<=120d) or `mid` (<=180d) — "far" cannot promote
4. B/C tier boundaries depend on optionality + eligibility, not catalyst proximity
5. Therefore: moving 946 rows from "missing" to "far" changes **no tier assignments**

## Why 2024 Remains Inverted

The 2024 inversion (mean AB-CD = -6.78pp) is NOT a catalyst coverage problem:
- **B-tier underperformance** (-5.35% mean): B selection depends on optionality/eligibility
- **D-tier outperformance** (+5.60% mean): ineligible names that outperform
- Catalyst strength is already monotonically correct in 2024 (mid > far > near > missing)

The problem is in the eligibility + optionality layer, not catalyst coverage.

## Production Ruleset Implications

Current production ruleset **68b2c45e** (a_floor=0.60, cat_near=120) is close to the
combined-panel winner (a_floor=0.58, cat_near=any). The 2025-only winner (a_floor=0.70)
diverges significantly, suggesting 2024 data pulls the optimum toward lower thresholds.

**No ruleset change warranted from this analysis.** The combined sweep validates that
the current production config sits in the passing region (a_floor=0.60: sep=+1.69pp, passes).

## Next Steps

1. **PARK** trial_active backfill — no production integration
2. **Investigate eligibility gate**: which D-tier (ineligible) names drive 2024 outperformance?
3. **Stronger catalyst sources**: SEC 8-K filings or conference calendars could produce
   near/mid signals that trial_active cannot
4. **Accept 2024 as hostile**: use 2025-only calibration for parameter tuning, treat 2024
   as out-of-distribution validation only

## Files Produced

- `artifacts/walkforward_panel__backfilled_combined.csv` (3922 rows, 22 snapshots)
- `artifacts/walkforward_panel__original_combined.csv` (3922 rows, 22 snapshots)
- `artifacts/calibration_report__backfilled_2024_2025.md` (2D sweep, 40 combos)
- `artifacts/calibration_report__original_2024_2025.md` (2D sweep, 40 combos)
- `artifacts/calibration_report__backfilled_2025_only.md` (2D sweep, 40 combos)
- `artifacts/calibration_report__original_2025_only.md` (2D sweep, 40 combos)
- `artifacts/calibration_backfilled_2024_2025_2026-02-12.md` (this memo)
