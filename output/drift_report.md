# Daily Drift Report
Date: 2026-02-20 | Ruleset: 8f99d47e | Window: 5 snapshots

## Current Snapshot

| Metric                    | Value  |
|---------------------------|--------|
| A-tier count (dev)        |     33 |
| A-tier % (dev)            |  18.0% |
| B-tier count (dev)        |     40 |
| Eligible % (dev)          |  72.7% |
| Catalyst missing (elig)   |   0.0% |
| Backfill share (elig)     |   0.0% |
| Drawdown coverage (dev)   |   0.0% |
| Rel DD coverage (dev)     |  99.5% |
| Top-25 overlap (vs prior) |  92.0% |
| Optionality std           |   0.29 |
| Composite IQR             |    0.1 |

## Cost Context

| Metric                    | Value  |
|---------------------------|--------|
| Cost coverage (dev)       | 99.5% |
| Est cost P10 / P50 / P90  | 351.7 / 926.9 / 2020.0 bps |
| Mean cost mult            | 0.7754 |
| Bucket no/mild/heavy/floor | 14.2% / 38.3% / 31.1% / 16.4% |
| Cap binding               | 47.8% |

## Returns Source Mix

| Source                 | Count | Share |
|------------------------|-------|-------|
| morningstar            |   183 | 100.0% |
| csv                    |     0 |  0.0% |
| csv_outlier_override   |     0 |  0.0% |
| unknown                |     0 |  0.0% |

## Catalyst Coverage

Dev tickers: 183 | Eligible: 183 (100.0%)
Eligible = dev tickers with non-empty `tier_dev`.

| Catalyst Mode     | Count | Share (elig) |
|-------------------|-------|--------------|
| specific_days     |   169 |        92.3% |
| blended_window    |     0 |         0.0% |
| far_window        |     0 |         0.0% |
| no_upcoming       |    14 |         7.7% |
| missing           |     0 |         0.0% |

> **Note:** Coverage here is *specific_days* (dated catalysts within the actionable near/mid window), not audit "any catalyst". See `docs/CATALYST_COVERAGE_CROSSWALK.md` for definitions.

## Catalyst Source Mix

Eligible dev tickers: 183

| Source                 | Count | Share (elig) |
|------------------------|-------|--------------|
| CTGOV_CALENDAR         |   110 |        60.1% |
| FDA_CALENDAR           |     0 |         0.0% |
| SEC_8K_FILING          |     7 |         3.8% |
| SEC_10Q_FILING         |     0 |         0.0% |
| SEC_10K_FILING         |     0 |         0.0% |
| SEC_6K_FILING          |     0 |         0.0% |
| FEDERAL_REGISTER       |     0 |         0.0% |
| CORPORATE_CALENDAR     |     0 |         0.0% |
| none                   |    14 |         7.7% |
| unknown                |    52 |        28.4% |

### Unknown Source — Top Offenders

Dated-catalyst tickers with missing or unrecognized source.

| Ticker | Catalyst Mode  | Days | Event Type     | Reason         |
|--------|----------------|------|----------------|----------------|
| ABEO   | specific_days  |  496 | —              | missing        |
| ABSI   | specific_days  |  527 | —              | missing        |
| ACLX   | specific_days  |  284 | —              | missing        |
| ALDX   | specific_days  |  284 | —              | missing        |
| ALLO   | specific_days  |  550 | —              | missing        |
| ATXS   | specific_days  |  374 | —              | missing        |
| AURA   | specific_days  |  283 | —              | missing        |
| AVBP   | specific_days  |  725 | —              | missing        |
| BBOT   | specific_days  |  527 | —              | missing        |
| BCAX   | specific_days  |  314 | —              | missing        |
| ...    | (42 more)               |      |                |                |

_Non-CTGOV dated catalysts: 7 ticker(s). CTGOV floor not breached (60.1% >= 37.0%)._

## Catalyst Event Type Mix

Eligible dev tickers: 183

| Event Type             | Count | Share (elig) |
|------------------------|-------|--------------|
| DATA_READOUT           |     7 |         3.8% |
| FDA_DECISION           |     0 |         0.0% |
| FDA_ADCOM              |     0 |         0.0% |
| FDA_PDUFA_DATE         |     0 |         0.0% |
| FDA_APPROVAL           |     0 |         0.0% |
| FDA_CRL                |     0 |         0.0% |
| FDA_RTF                |     0 |         0.0% |
| FDA_WARNING_LETTER     |     0 |         0.0% |
| FDA_SUBMISSION         |     0 |         0.0% |
| CT_PRIMARY_COMPLETION  |    92 |        50.3% |
| CT_STUDY_COMPLETION    |    18 |         9.8% |
| CT_RESULTS_POSTED      |     0 |         0.0% |
| TRIAL_ONGOING          |     0 |         0.0% |
| none                   |    14 |         7.7% |
| unknown                |    52 |        28.4% |

### Unknown Event Type — Top Offenders

Dated-catalyst tickers with missing or unrecognized event type.

| Ticker | Catalyst Mode  | Days | Catalyst Source      | Reason         |
|--------|----------------|------|----------------------|----------------|
| ABEO   | specific_days  |  496 | —                    | missing        |
| ABSI   | specific_days  |  527 | —                    | missing        |
| ACLX   | specific_days  |  284 | —                    | missing        |
| ALDX   | specific_days  |  284 | —                    | missing        |
| ALLO   | specific_days  |  550 | —                    | missing        |
| ATXS   | specific_days  |  374 | —                    | missing        |
| AURA   | specific_days  |  283 | —                    | missing        |
| AVBP   | specific_days  |  725 | —                    | missing        |
| BBOT   | specific_days  |  527 | —                    | missing        |
| BCAX   | specific_days  |  314 | —                    | missing        |
| ...    | (42 more)               |      |                      |                |

## Catalyst Priority Distribution

Eligible dev tickers: 133

| Priority | Label    | Count | Share (elig) |
|----------|----------|-------|--------------|
|        1 | FDA      |     0 |         0.0% |
|        2 | Readout  |     6 |         4.5% |
|        3 | Ongoing  |    80 |        60.2% |
|        9 | None     |    47 |        35.3% |
|       99 | Unknown  |     0 |         0.0% |

## Rolling Window (last 5 runs)

| Metric              | Min   | Max   | Mean  | Median | IQR   | Delta | Current |
|---------------------|-------|-------|-------|--------|-------|-------|---------|
| A Pct               |  15.8 |  20.2 |  17.9 |   18.0 |   2.5 |   0.0 |    18.0 |
| Catalyst Missing Pct |   0.0 |   0.0 |   0.0 |    0.0 |   0.0 |   0.0 |     0.0 |
| Top25 Overlap Pct   |  52.0 |  96.0 |  81.0 |   88.0 |  35.0 |   4.0 |    92.0 |
| Optionality Std     |  0.29 |  0.29 |  0.29 |    0.3 |   0.0 |  -0.0 |     0.3 |
| Catalyst Strength Near Pct |  35.8 |  56.6 | 44.08 |   42.7 |  10.7 |   0.2 |    42.9 |

## Warning: Mixed Rulesets in Window
Rulesets observed: 8f99d47e, aa0aaf28, e1be5370

## Adaptive Warnings

- Median cost = 926.9 bps > 60.0 bps ceiling
- Cap binding = 47.8% > 20.0% ceiling
- Catalyst source unknown = 28.4% (missing=28.4%) > 5.0% ceiling
- Catalyst event type unknown = 28.4% (missing=28.4%) > 5.0% ceiling
- inst_delta_nonzero_pct=0.0% < 5.0% floor

## Drift Attribution
Comparing 2026-02-19 → 2026-02-20

### Eligibility Gate Changes

| Gate | Prior | Current | Delta |
|------|-------|---------|-------|
| fundamental_red_flag | 10 | 10 | 0 |
| sev3 | 0 | 0 | 0 |
| deep_drawdown | 46 | 44 | -2 |
| adv_fail | 0 | 0 | 0 |

### Catalyst Strength Shifts

| Band | Prior | Current | Delta |
|------|-------|---------|-------|
| near | 56 | 57 | +1 |
| mid | 14 | 14 | 0 |
| far | 55 | 55 | 0 |
| missing | 6 | 7 | +1 |

### Portfolio Churn
Top-25 overlap: 23/25

**Dropped:**

| Ticker | Tier | Band | Reason |
|--------|------|------|--------|
| NUVL | A | L | high_opt+catalyst_near |
| ZBIO | A | L | high_opt+catalyst_near |

**Added:**

| Ticker | Tier | Band | Reason |
|--------|------|------|--------|
| CCCC | A | M | high_opt+catalyst_near |
| CRSP | A | L | high_opt+catalyst_near |

### Gate Margin Shifts

| Metric | Prior | Current | Delta |
|--------|-------|---------|-------|
| Median dd_abs_margin | N/A | N/A | N/A |
| P10 dd_abs_margin | N/A | N/A | N/A |
| Median dd_rel_margin | N/A | N/A | N/A |
| Rescued count | N/A | N/A | N/A |
| DD rel-margin rescue count | 0 | 0 | 0 |

### Gate Pressure

Share of dev tickers within ±5pp of each gate threshold.

| Metric | Prior | Current | Delta |
|--------|-------|---------|-------|
| DD abs near gate % | N/A | N/A | N/A |
| DD rel near gate % | N/A | N/A | N/A |
| Optionality near A-floor % | N/A | N/A | N/A |
| Rescued share % | N/A | N/A | N/A |
| DD rel-margin rescue share % | 0.0% | 0.0% | +0.0pp |

## Guardrails: WARN
Recommended action: INVESTIGATE


## Guardrails Config
ID: cb6bcabe
- fail_a_pct_low: 0.0%
- warn_a_pct_low: 1.5%
- fail_a_pct_high: 25.0%
- fail_catalyst_missing_high: 85.0%
- fail_overlap_low: 50.0%
- fail_dispersion_low: 0.1
- warn_median_cost_bps_high: 60.0 bps
- warn_cost_coverage_low: 80.0%
- warn_cap_binding_high: 20.0%
- warn_backfill_share_high: 60.0%
- warn_dd_rel_margin_rescue_share_high: 5.0%
- warn_rs_unknown_share_high: 8.5%
- warn_rs_csv_outlier_override_share_high: 1.0%
- warn_rs_morningstar_share_low: 85.0%
- warn_cat_eligible_share_low: 95.0%
- warn_cat_specific_days_share_low: 40.0%
- warn_cs_ctgov_share_low: 37.0%
- warn_cs_unknown_share_high: 5.0%
- warn_ct_unknown_share_high: 5.0%
- warn_ct_fda_share_spike: 30.0%
- warn_iqr_k: 2.0
- warn_iqr_floor: 1.0
- warn_min_window: 3
- fail_corroboration_count: 2

## Suggested Guardrails

Data-driven suggestions from prior snapshots (excludes current).
These are **informational only** and do not affect drift status.

| Metric | N | Median | IQR | Suggested | Current | Tighter? |
|--------|---|--------|-----|-----------|---------|----------|
| A-tier % | 4 | 17.8% | 3.4 | 10.9% | 1.5% | **yes** |
| Catalyst eligible share | 4 | 100.0% | 0.0 | 100.0% | 95.0% | **yes** |
| Specific-days share | 4 | 92.3% | 11.0 | 70.2% | 40.0% | **yes** |
| Morningstar share | 4 | 100.0% | 0.0 | 100.0% | 85.0% | **yes** |
| Returns unknown share | 4 | 0.0% | 0.0 | 0.0% | 8.5% | **yes** |
| CTGOV calendar share | 4 | 60.1% | 10.6 | 38.8% | 37.0% | **yes** |
| Unknown source share | 4 | 28.7% | 4.7 | 38.2% | 5.0% |  |
| Unknown event type share | 4 | 28.7% | 4.7 | 38.2% | 5.0% |  |
