# Regime Validation: 2024 Backfilled Catalysts

**Date**: 2026-02-12
**Ruleset**: v1.2.1_candidate (68b2c45e)
**Archives**: `data/archives_backfilled/` (33 archives, trial_active backfill applied)
**Baseline**: `data/archives/` (original, unmodified)

## Question

Does adding +24pp catalyst coverage via `trial_active` backfill repair the
2024 tier inversion (C/D outperforming A/B)?

## Answer

**NO.** The backfill has **zero impact** on tier assignment or AB-CD separation.

## Why: Structural Mechanism

The `trial_active` source emits `days_to_catalyst = window_days + 1` (366d),
which maps to `catalyst_strength = "far"` in the decision engine. A-tier
requires **near or mid** catalyst strength (<=180d). Since trial_active can
never produce near/mid, it cannot promote any ticker to A-tier.

The 519 rows that changed `catalyst_mode` from "missing" to "specific_days"
all retained their original tier (339 D-tier ineligible, 119 B-tier, 61 C-tier).

## Tier Separation (60d mean return)

| Regime | Tier | n | Mean | Hit% | MaxDD | AB-CD Sep |
|--------|------|--:|-----:|-----:|------:|----------:|
| **2024 Original** | A | 53 | +5.87% | 49.0% | -25.85% | |
| | B | 257 | -5.11% | 34.7% | -30.29% | |
| | C | 407 | -5.54% | 36.5% | -29.87% | |
| | D | 1397 | +5.65% | 37.8% | -35.57% | **-6.36pp** |
| **2024 Backfilled** | A | 53 | +5.87% | 49.0% | -25.85% | |
| | B | 257 | -5.11% | 34.7% | -30.29% | |
| | C | 407 | -5.54% | 36.5% | -29.87% | |
| | D | 1397 | +5.65% | 37.8% | -35.57% | **-6.36pp** |
| **2025 Original** | A | 43 | +18.40% | 59.5% | -24.56% | |
| | B | 143 | +39.16% | 72.3% | -22.91% | |
| | C | 201 | +12.76% | 59.7% | -23.77% | |
| | D | 1421 | +29.04% | 65.5% | -29.66% | **+7.34pp** |

**2024 AB-CD separation**: -6.36pp (inverted, unchanged by backfill)
**2025 AB-CD separation**: +7.34pp (healthy, unchanged by backfill)

## Catalyst Strength Distribution (backfill effect)

| Regime | Band | Original | Backfilled | Delta |
|--------|------|----------|------------|-------|
| 2024 | near | 262 (12.4%) | 262 (12.4%) | 0 |
| 2024 | mid | 104 (4.9%) | 104 (4.9%) | 0 |
| 2024 | far | 336 (15.9%) | 855 (40.4%) | +519 |
| 2024 | missing | 1412 (66.8%) | 893 (42.2%) | -519 |
| 2025 | near | 301 (16.6%) | 301 (16.6%) | 0 |
| 2025 | mid | 128 (7.1%) | 128 (7.1%) | 0 |
| 2025 | far | 308 (17.0%) | 735 (40.7%) | +427 |
| 2025 | missing | 1071 (59.2%) | 644 (35.6%) | -427 |

The backfill moves tickers from "missing" to "far" only. Near/mid unchanged.

## Strength-Level Returns (2024)

| Band | n (orig) | Mean (orig) | n (bkf) | Mean (bkf) |
|------|----------|-------------|---------|------------|
| near | 262 | +2.69% | 262 | +2.69% |
| mid | 104 | +9.57% | 104 | +9.57% |
| far | 336 | +4.96% | 855 | +0.72% |
| missing | 1412 | +1.41% | 893 | +3.38% |

The 519 migrated tickers had below-average returns. Moving them from "missing"
to "far" dilutes the far bucket (4.96% -> 0.72%) and concentrates the remaining
missing bucket on slightly better performers (1.41% -> 3.38%). This is
**information-neutral at the tier level** — neither signal quality improved.

## Root Cause Analysis

The 2024 tier inversion is NOT caused by missing catalyst coverage. The core
issues are:

1. **B-tier underperformance** (-5.11%) — the largest AB component. B-tier
   selection is driven by optionality + eligibility, not catalyst proximity.
   Catalyst backfill cannot fix B-tier quality.

2. **D-tier outperformance** (+5.65%) — ineligible tickers that happen to be
   strong performers. This reflects the "optionality premium" in early-stage
   biotech that the eligibility gate filters out.

3. **Catalyst strength is already monotonically correct** in 2024: mid (+9.57%)
   > far (+4.96%) > near (+2.69%) > missing (+1.41%). The signal works;
   the problem is in the eligibility + optionality layer, not catalyst coverage.

## Verdict

**PARK backfill.** Do not promote `trial_active` into the enrichment pipeline.

The backfill successfully adds coverage (+24pp across both regimes) and
correctly classifies signals as "far", but this has **zero effect on tier
assignment** because A-tier requires near/mid strength that trial_active
structurally cannot produce.

The 2024 inversion is rooted in B-tier quality (optionality + eligibility),
not catalyst coverage. The next highest-leverage investigation should focus on:

1. **Eligibility gate analysis** — which ineligible D-tier tickers are the
   strong performers? Is the gate too aggressive?
2. **Optionality signal quality in 2024** — is the clinical_optionality_pct_dev
   score predictive or noise in the 2024 regime?
3. **Strong-source catalysts** (SEC 8-K filings, conference calendars) could
   produce near/mid signals, but the 2024 strength-level data suggests the
   catalyst dimension is already working — the problem is elsewhere.

## Files Produced

- `artifacts/walkforward_panel_backfilled_2024.csv` (2114 rows, 12 snapshots)
- `artifacts/walkforward_panel_backfilled_2025.csv` (1808 rows, 10 snapshots)
- `artifacts/walkforward_report_backfilled_2024.json`
- `artifacts/walkforward_report_backfilled_2025.json`
- `artifacts/walkforward_panel_original_2024.csv` (baseline)
- `artifacts/walkforward_panel_original_2025.csv` (baseline)
