# SEC 8-K Catalyst Enrichment Evaluation

**Date**: 2026-02-12
**Panels**: `walkforward_panel__secadcom_combined.csv` / `walkforward_panel__original_combined.csv`
**Grid**: 8 a_floor (0.40..0.70) x 5 catalyst_near (60..180) = 40 combos
**Archives**: 22 snapshots (12 x 2024, 10 x 2025), 3922 panel rows each

## Decision

**PARK SEC 8-K enrichment.** Do not wire `sec_8k_catalyst_collector` into the production
enrichment pipeline. The additional NEAR catalyst signals are real but do not improve
(and slightly degrade) tier separation and calibration outcomes.

---

## What Was Tested

Existing SEC 8-K and FDA ADCOM collectors (`sec_8k_catalyst_collector.py`,
`fda_adcom_collector.py`) were wired into `enrich_archive_inputs.py` with
`--sec-8k-mode live --adcom-mode live`. All 22 archives (2024+2025) were
re-enriched with SEC event data into `data/archives_secadcom/`. Both the
original and SEC-enriched archives were then run through the full walkforward
panel + 2D calibration sweep pipeline.

## Coverage Impact (all tickers, 6883 ticker-dates)

| Metric | Original | SEC-enriched | Delta |
|--------|----------|-------------|-------|
| NEAR | 1527 (22.2%) | 1661 (24.1%) | +134 (+1.9pp) |
| MID | 446 (6.5%) | 446 (6.5%) | 0 |
| FAR | 996 (14.5%) | 957 (13.9%) | -39 |
| MISSING | 3914 (56.9%) | 3819 (55.5%) | -95 (-1.4pp) |

Source distribution (SEC-enriched):
- `trial_pcd`: 2805 (40.8%, was 42.4%)
- `sec_8k`: 210 (3.1%, new)
- `pdufa`: 49 (0.7%)
- `none`: 3819 (55.5%, was 56.9%)

SEC contributed 210 catalyst assignments. Of the 210:
- 134 added **new NEAR** catalysts (tickers previously MISSING or FAR)
- 76 displaced existing trial_pcd entries with closer SEC dates

Per date: ~6-17 SEC-sourced tickers (avg ~10), with 2024 Q1-Q3 seeing highest impact.

## Panel-Level Deltas (SEC vs Original)

| Metric | Count | % of 3922 |
|--------|------:|----------:|
| Tier changed | 19 | 0.5% |
| Eligibility changed | 0 | 0.0% |
| Catalyst_mode changed | 45 | 1.1% |
| Weight changed | 151 | 3.9% |

Tier transitions (all positive direction):
- **B -> A**: 13 (SEC provided NEAR catalyst to high-optionality B-tier names)
- **C -> B**: 6

A-tier grew: 91 -> 104 (+14.3%), primarily in 2024 (52 -> 64).

## Per-Year Tier Performance

### 2024 (2114 rows)

| Tier | n (orig) | n (SEC) | Mean 60d (orig) | Mean 60d (SEC) | Median (orig) | Median (SEC) |
|------|----------|---------|-----------------|----------------|---------------|--------------|
| A | 50 | 62 | +6.02% | +6.11% | +0.96% | +1.37% |
| B | 231 | 224 | -5.35% | -6.10% | -8.88% | -9.57% |
| C | 340 | 335 | -5.44% | -5.37% | -7.54% | -7.50% |
| D | 1405 | 1405 | +5.60% | +5.60% | -9.75% | -9.75% |

**2024 AB-CD separation**: median +1.02pp -> +0.98pp (**-0.05pp**, marginal worse)
**2024 AB-CD mean**: -6.78pp -> -6.94pp (**still deeply inverted**)

### 2025 (1808 rows)

| Tier | n (orig) | n (SEC) | Mean 60d (orig) | Mean 60d (SEC) | Median (orig) | Median (SEC) |
|------|----------|---------|-----------------|----------------|---------------|--------------|
| A | 38 | 39 | +18.17% | +18.49% | +12.61% | +15.22% |
| B | 122 | 122 | +36.28% | +36.02% | +17.17% | +16.18% |
| C | 176 | 175 | +13.32% | +13.39% | +8.57% | +8.76% |
| D | 1457 | 1457 | +29.16% | +29.16% | +14.99% | +14.99% |

**2025 AB-CD separation**: median +1.34pp -> +0.99pp (**-0.35pp**, worse)

## Calibration Sweep Comparison

### Combined 2024+2025

| Metric | Original | SEC-enriched | Delta |
|--------|----------|-------------|-------|
| Winner | a=0.58, cn=60 | a=0.70, cn=60 | different |
| Best separation | +1.95pp | +1.67pp | **-0.28pp** |
| Best score | -5.22 | -5.49 | -0.27 |
| cat_near flat? | YES | YES | no change |
| Passing combos | 30/40 | 30/40 | same |
| DQ cat_missing% | 55.2% | 54.2% | -1.0pp |

### 2025-only

| Metric | Original | SEC-enriched | Delta |
|--------|----------|-------------|-------|
| Winner | a=0.70, cn=60 | a=0.70, cn=60 | same |
| Best separation | +8.05pp | +7.38pp | **-0.67pp** |
| Best score | 2.10 | 1.44 | -0.66 |
| cat_near flat? | YES | YES | no change |
| Passing combos | 40/40 | 40/40 | same |
| DQ cat_missing% | 47.3% | 47.3% | 0 |

## Success Criteria Assessment

1. **Does 2024 stop being inverted?** NO. Mean AB-CD = -6.94pp (was -6.78pp).
2. **Does catalyst_near regain discriminating power?** NO. Completely flat across all values.
3. **Does separation improve?** NO. Worse by 0.28-0.67pp depending on date range.

**All three success criteria FAIL.**

## Why SEC Made Things Slightly Worse

1. **B-tier dilution**: The 13 B->A upgrades removed *relatively better* B names into A,
   making remaining B-tier worse (2024 B mean: -5.35% -> -6.10%)
2. **Noisy promotion**: SEC NEAR catalysts promoted names where a readout date exists
   in filings, but filing-derived dates don't predict returns better than trial-based dates
3. **trial_pcd displacement**: 114 tickers lost trial_pcd source (replaced by closer SEC date),
   but the SEC-sourced dates didn't produce better forward returns
4. **Fundamental problem unchanged**: 2024 inversion is driven by eligibility/optionality
   layer (B underperformance, D outperformance), not catalyst coverage

## Production Ruleset Implications

Current production ruleset **68b2c45e** (a_floor=0.60, cat_near=120) remains valid.
No change warranted from this analysis.

## What This Rules Out

- SEC 8-K filing-derived catalyst dates are not a useful signal for the decision engine
  tier system in its current architecture
- The catalyst coverage problem in 2024 is NOT a "missing dates" problem — it's a
  "the eligibility/optionality layer doesn't predict returns in that regime" problem
- Adding more catalyst sources (ADCOM, conference calendars) is unlikely to help
  unless they provide *qualitatively different* information (e.g., signal quality, not just dates)

## Next Steps

1. **PARK** SEC 8-K enrichment for production archives
2. **Keep the wiring code** in `enrich_archive_inputs.py` (committed, tested) for future experiments
3. **Investigate eligibility gate**: which D-tier (ineligible) names drive 2024 outperformance?
4. **Accept 2024 as hostile**: continue using 2025-only for parameter calibration
5. **Consider**: could SEC signals be useful as a *confidence* overlay rather than
   proximity input? (e.g., "SEC confirms trial timeline" vs "SEC provides new date")

## Files Produced

- `data/archives_secadcom/` — 22 SEC-enriched archive copies (gitignored)
- `artifacts/walkforward_panel__secadcom_combined.csv` (3922 rows)
- `artifacts/walkforward_report__secadcom_combined.md`
- `artifacts/calibration_report__secadcom_2024_2025.md` (2D sweep, 40 combos)
- `artifacts/calibration_report__secadcom_2025_only.md` (2D sweep, 40 combos)
- `artifacts/sec_enrichment_evaluation_2026-02-12.md` (this memo)
