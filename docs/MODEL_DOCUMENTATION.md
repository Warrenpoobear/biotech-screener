# Biotech Screener Model Documentation

**Version:** 2.0.0
**Last Updated:** February 3, 2026
**System:** Wake Robin Capital Biotech Screening Pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Architecture](#data-architecture)
4. [Module Descriptions](#module-descriptions)
5. [Output Format: JSON](#output-format-json)
6. [Output Format: CSV](#output-format-csv)
7. [Field Definitions](#field-definitions)
8. [Scoring Methodology](#scoring-methodology)
9. [PIT (Point-in-Time) Compliance](#pit-point-in-time-compliance)
10. [Configuration](#configuration)

---

## Overview

The Biotech Screener is a deterministic, production-grade screening system that ranks biotech investment opportunities. It processes clinical trial data, financial filings, market data, and catalyst events to produce weekly ranked portfolios.

### Key Principles

- **Determinism**: No `datetime.now()` calls; all dates are explicit via `--as-of-date`
- **PIT Compliance**: Point-in-time filtering prevents lookahead bias
- **Decimal Arithmetic**: All financial calculations use `Decimal` for precision
- **Stable Ordering**: Output is deterministically sorted for reproducibility
- **Content Hashing**: Inputs are hash-verified for integrity

### Typical Usage

```bash
python run_screen.py \
    --as-of-date 2026-01-30 \
    --data-dir production_data \
    --output production_data/results.json \
    --enable-enhancements \
    --enable-clustering
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA SOURCES                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  universe.json  │ trial_records   │ financial_data  │  defensive_features   │
│  (ticker list)  │    .json        │    .json        │      .json            │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUN_SCREEN.PY                                   │
│                         (Orchestrator Pipeline)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   Module 1   │───▶│   Module 2   │───▶│   Module 3   │                 │
│   │   Universe   │    │  Financial   │    │   Catalyst   │                 │
│   │  Filtering   │    │    Health    │    │  Detection   │                 │
│   └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                          │
│   ┌──────────────┐    ┌──────────────┐          │                          │
│   │   Module 4   │◀───│   Module 5   │◀─────────┘                          │
│   │   Clinical   │───▶│  Composite   │                                     │
│   │ Development  │    │   Ranking    │                                     │
│   └──────────────┘    └──────────────┘                                     │
│                              │                                              │
│                              ▼                                              │
│                    ┌──────────────────┐                                    │
│                    │    Defensive     │                                    │
│                    │     Overlay      │                                    │
│                    └──────────────────┘                                    │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT FILES                                      │
├─────────────────────────────┬───────────────────────────────────────────────┤
│       results.json          │              results.csv                       │
│   (complete pipeline data)  │        (flattened for analysis)               │
└─────────────────────────────┴───────────────────────────────────────────────┘
```

---

## Data Architecture

### Input Files

| File | Description | Source | Update Frequency |
|------|-------------|--------|------------------|
| `universe.json` | Investable ticker list with metadata | XBI ETF holdings + manual additions | Weekly |
| `trial_records.json` | Clinical trial data per ticker | ClinicalTrials.gov API | Weekly |
| `financial_data.json` | SEC EDGAR financial metrics | SEC XBRL API | Quarterly |
| `defensive_features.json` | Market data: vol, correlation, drawdown | Yahoo Finance | Daily |
| `short_interest.json` | Short interest data | FINRA / exchanges | Bi-weekly |
| `institutional_holdings.json` | 13F institutional positions | SEC EDGAR | Quarterly |

### Input File Schemas

#### universe.json
```json
[
  {
    "ticker": "VRTX",
    "company_name": "Vertex Pharmaceuticals",
    "market_cap_usd": 120000000000,
    "sector": "Biotechnology",
    "archetype": "commercial_biotech",
    "status": "ACTIVE"
  }
]
```

#### trial_records.json
```json
[
  {
    "ticker": "VRTX",
    "nct_id": "NCT04058392",
    "phase": "Phase 3",
    "status": "Recruiting",
    "conditions": ["Cystic Fibrosis"],
    "start_date": "2019-08-01",
    "first_posted": "2019-08-15",
    "last_update_posted": "2025-12-01",
    "primary_completion_date": "2026-06-30",
    "primary_completion_type": "ESTIMATED"
  }
]
```

#### financial_data.json
```json
[
  {
    "ticker": "VRTX",
    "cik": "0000875320",
    "Cash": 12500000000,
    "Cash_date": "2025-09-30",
    "Assets": 25000000000,
    "Liabilities": 8000000000,
    "Revenue": 8500000000,
    "CFO": 3200000000,
    "R&D": 2100000000,
    "LongTermDebt": 0,
    "TotalDebt": 500000000,
    "collected_at": "2026-01-15"
  }
]
```

#### defensive_features.json
```json
{
  "VRTX": {
    "vol_60d": 0.28,
    "vol_252d": 0.32,
    "corr_xbi": 0.65,
    "beta_xbi_60d": 0.85,
    "drawdown": -0.12,
    "max_drawdown_252d": -0.25,
    "rsi_14d": 55
  }
}
```

---

## Module Descriptions

### Module 1: Universe Filtering

**File:** `module_1_universe.py`
**Purpose:** Filter the investable universe based on status gates

**Filters Applied:**
- Minimum market cap: $50M
- Status gates: Exclude DELISTED, ACQUIRED, SHELL, SUSPENDED
- Shell company detection via keyword matching

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `active_securities` | list | Tickers passing all filters |
| `excluded_securities` | list | Tickers failing filters with reasons |
| `diagnostic_counts.active` | int | Count of active tickers |
| `diagnostic_counts.excluded` | int | Count of excluded tickers |

---

### Module 2: Financial Health

**File:** `module_2_financial.py`
**Purpose:** Score tickers on financial health (0-100)

**Scoring Components:**
| Component | Weight | Description |
|-----------|--------|-------------|
| Cash Runway | 45% | Quarters of cash remaining at current burn rate |
| Dilution Risk | 25% | ATM offerings, shelf registrations, debt load |
| Liquidity | 15% | Trading volume adequacy |
| Revenue Score | 15% | Revenue presence and scale |

**Severity Levels:**
| Level | Runway | Meaning |
|-------|--------|---------|
| SEV3 | <6 months | Critical - imminent dilution risk |
| SEV2 | 6-12 months | Warning - may need financing |
| SEV1 | 12-18 months | Caution - monitor closely |
| NONE | ≥18 months | Healthy runway |

**Burn Rate Hierarchy:**
1. CFO or FCF (preferred - actual cash flow)
2. Net Income if negative (accounting proxy)
3. R&D × 1.5 (last resort proxy)

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `financial_score` | float | Normalized score 0-100 |
| `runway_quarters` | float | Estimated quarters of cash |
| `burn_rate` | float | Quarterly cash burn |
| `burn_source` | string | CFO, NET_INCOME, or RD_PROXY |
| `severity` | string | SEV1, SEV2, SEV3, or NONE |
| `data_state` | string | FULL, PARTIAL, MINIMAL, NONE |

---

### Module 3: Catalyst Detection

**File:** `module_3_catalyst.py`
**Purpose:** Detect and score clinical and corporate catalyst events

**Event Types (Delta-Based):**
| Event Type | Description | Impact | Confidence |
|------------|-------------|--------|------------|
| CT_STATUS_SEVERE_NEG | Trial terminated/withdrawn | -3 | 0.95 |
| CT_STATUS_DOWNGRADE | Status worsened | -1 to -3 | 0.85 |
| CT_STATUS_UPGRADE | Status improved | +1 to +3 | 0.80 |
| CT_TIMELINE_PUSHOUT | Completion delayed | -1 to -3 | 0.75 |
| CT_TIMELINE_PULLIN | Completion accelerated | +1 to +3 | 0.70 |
| CT_DATE_CONFIRMED_ACTUAL | Date confirmed | +1 | 0.85 |
| CT_RESULTS_POSTED | Results published | +1 | 0.90 |

**Calendar-Based Events (Forward-Looking):**
| Window | Event | Confidence |
|--------|-------|------------|
| 0-30 days | UPCOMING_PCD/SCD | 0.90 |
| 31-60 days | UPCOMING_PCD/SCD | 0.80 |
| 61-90 days | UPCOMING_PCD/SCD | 0.70 |
| 91-180 days | UPCOMING_PCD/SCD | 0.55 |
| 181-270 days | UPCOMING_PCD/SCD | 0.45 |

**Scoring Formula:**
```
event_score = impact × confidence × proximity
catalyst_score_net = sum(positive_events) - sum(negative_events)
```

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `catalyst_score_net` | float | Net catalyst score |
| `catalyst_score_pos` | float | Sum of positive events |
| `catalyst_score_neg` | float | Sum of negative events |
| `severe_negative_flag` | bool | Kill switch if critical negative |
| `n_events_upcoming` | int | Count of upcoming catalysts |
| `catalyst_window_days` | int | Days to nearest catalyst |
| `top_3_events` | list | Highest-impact events |

---

### Module 4: Clinical Development

**File:** `module_4_clinical_dev.py`
**Purpose:** Score clinical pipeline quality (0-100)

**Scoring Components:**
| Component | Max Points | Description |
|-----------|------------|-------------|
| Phase Advancement | 30 | Most advanced phase score |
| Phase Progress | 5 | Progression within phase |
| Trial Count | 5 | Number of active trials |
| Indication Diversity | 5 | Unique therapeutic areas |
| Recency | 5 | Recent trial activity |
| Design Quality | 25 | Randomization, blinding, endpoints |
| Execution Track Record | 25 | Completion vs termination rate |
| Endpoint Strength | 20 | OS, PFS vs biomarker endpoints |

**Phase Scores:**
| Phase | Points |
|-------|--------|
| Approved | 30 |
| Phase 3 | 25 |
| Phase 2/3 | 22 |
| Phase 2 | 18 |
| Phase 1/2 | 12 |
| Phase 1 | 8 |
| Preclinical | 3 |

**PIT Date Priority:**
1. `first_posted` (most reliable)
2. `last_update_posted`
3. `results_first_posted`
4. `source_date`
5. `start_date`

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `clinical_score` | float | Normalized score 0-100 |
| `lead_phase` | string | Most advanced phase |
| `trial_count` | int | Unique trials (PIT-filtered) |
| `n_trials_raw` | int | Raw trial count before PIT |
| `pit_filtered_count_ticker` | int | Trials excluded by PIT |
| `recency_days` | int | Days since last update |
| `date_coverage_pct` | float | % of trials with PIT dates |

---

### Module 5: Composite Ranking

**File:** `module_5_composite_with_defensive.py`
**Purpose:** Combine all signals into final ranking with defensive overlay

**Default Weights (v3 Scoring):**
| Component | Weight | Description |
|-----------|--------|-------------|
| Financial Health | 15% | Module 2 score |
| Clinical Development | 35% | Module 4 score |
| Catalyst Score | 10% | Module 3 net score |
| Momentum | 10% | Price momentum signal |
| Smart Money | 10% | Institutional signal |
| Valuation | 10% | Market cap per trial |
| PoS (Probability of Success) | 5% | Indication-based success rate |
| Short Interest | 5% | Crowding/squeeze potential |

**Defensive Overlay:**
The defensive multiplier adjusts scores based on risk characteristics:

| Bucket | Multiplier | Criteria |
|--------|------------|----------|
| Elite | 1.30-1.40x | Low vol, low correlation, low drawdown |
| Diversifier | 1.10-1.20x | Moderate risk, negative correlation |
| Core | 1.00x | Average characteristics |
| Risky | 0.85-0.95x | High vol or high correlation |
| Avoid | 0.70-0.80x | Extreme risk metrics |

**Output Fields:** See [Field Definitions](#field-definitions) section.

---

## Output Format: JSON

### Top-Level Structure

```json
{
  "run_metadata": {
    "as_of_date": "2026-01-30",
    "run_id": "abc123",
    "version": "3.0.0"
  },
  "summary": {
    "total_evaluated": 353,
    "active_universe": 326,
    "final_ranked": 318,
    "excluded": 12,
    "catalyst_events": 480,
    "severe_negatives": 0
  },
  "module_1_universe": { ... },
  "module_2_financial": { ... },
  "module_3_catalyst": { ... },
  "module_4_clinical": { ... },
  "module_5_composite": {
    "ranked_securities": [ ... ],
    "diagnostic_counts": { ... },
    "weights_used": { ... }
  },
  "production_validation": {
    "passed": true,
    "config": { ... }
  }
}
```

### ranked_securities Record

Each security in `module_5_composite.ranked_securities` contains:

```json
{
  "ticker": "VRTX",
  "composite_rank": 1,
  "composite_score": 84.57,
  "z_score": 1.69,
  "expected_excess_return": 0.135,
  "volatility": 0.316,
  "drawdown": -0.091,
  "defensive_multiplier": 1.40,
  "defensive_bucket": "elite",
  "rankable": true,
  "severity": "none",
  "stage_bucket": "late",
  "market_cap_bucket": "large",
  "momentum_signal": { ... },
  "catalyst_effective": { ... },
  "valuation_signal": { ... },
  "smart_money_signal": { ... },
  "short_interest_signal": { ... },
  "defensive_features": { ... }
}
```

---

## Output Format: CSV

The CSV export (`export_results_csv.py`) flattens the JSON into analysis-friendly columns.

### Column Groups

#### Core Columns (Position 1-25)
Primary ranking and identification fields.

#### Signal Columns (Position 26-60)
Flattened signal scores from enhancement modules.

#### Confidence Columns (Position 61-65)
Data quality confidence scores.

#### Clinical PIT Columns (Position 66-73)
Module 4 observability for PIT audit.

#### Catalyst Debug Columns (Position 74-80)
Module 3 catalyst details for debugging.

---

## Field Definitions

### Core Output Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `ticker` | string | - | Stock ticker symbol |
| `composite_rank` | int | 1-N | Final ranking position |
| `composite_score` | float | 0-100+ | Final weighted score (may exceed 100 with defensive boost) |
| `z_score` | float | -3 to +3 | Standard deviations from mean |
| `expected_excess_return` | float | -0.5 to +0.5 | Predicted alpha vs benchmark |
| `volatility` | float | 0-2 | Annualized volatility (60-day) |
| `drawdown` | float | -1 to 0 | Current drawdown from 52-week high |
| `cluster_id` | int | 0-N | Correlation cluster assignment |
| `corr_xbi` | float | -1 to +1 | Correlation with XBI ETF |
| `beta_xbi` | float | 0-3 | Beta vs XBI benchmark |

### Defensive Overlay Fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `defensive_multiplier` | float | 0.7-1.4 | Score adjustment factor |
| `defensive_bucket` | string | elite, diversifier, core, risky, avoid | Risk classification |
| `defensive_notes` | string | - | Explanation of bucket assignment |
| `rank_driver` | string | alpha, defensive_boost, defensive_penalty, suppressed | What drove the final rank |

### Risk Data State Fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `risk_data_state_vol` | string | live, stale, missing | Volatility data freshness |
| `risk_data_state_drawdown` | string | live, stale, missing | Drawdown data freshness |
| `risk_data_state_beta` | string | live, stale, missing | Beta data freshness |
| `confidence_risk` | float | 0-1 | Confidence in risk metrics |

### Signal Fields

#### Momentum Signal
| Field | Type | Description |
|-------|------|-------------|
| `momentum_score` | float | 0-100 momentum contribution |
| `momentum_window` | string | Window used (20d, 60d, etc.) |
| `momentum_alpha` | float | Excess return vs benchmark |

#### Catalyst Signal
| Field | Type | Description |
|-------|------|-------------|
| `catalyst_score` | float | Raw catalyst score |
| `catalyst_effective_score` | float | Time-decayed effective score |
| `catalyst_proximity_score` | float | Bonus for near-term catalysts |

#### Valuation Signal
| Field | Type | Description |
|-------|------|-------------|
| `valuation_score` | float | 0-100 cheapness score |
| `valuation_method` | string | mcap_per_trial, ev_cfo, ev_revenue |
| `valuation_raw_metric` | float | Raw valuation metric |
| `valuation_pct_overall` | float | Percentile within method |
| `valuation_pct_in_size_bucket` | float | Percentile within size peer group |

#### Smart Money Signal
| Field | Type | Description |
|-------|------|-------------|
| `smart_money_score` | float | Institutional signal score |
| `smart_money_overlap` | int | Count of top-tier holders |
| `smart_money_tier1_holders` | string | Names of tier-1 holders |

#### Short Interest Signal
| Field | Type | Description |
|-------|------|-------------|
| `short_interest_score` | float | SI-based signal |
| `short_interest_crowding` | string | low, medium, high |
| `short_interest_squeeze` | float | Squeeze potential score |

### Clinical PIT Observability Fields

| Field | Type | Description |
|-------|------|-------------|
| `pit_trials_total` | int | Raw trial count before any filtering |
| `pit_trials_eligible` | int | Trials after PIT filtering |
| `pit_trials_filtered` | int | Trials excluded by PIT date check |
| `pit_trials_no_date` | int | Trials excluded for missing dates |
| `clinical_lead_phase` | string | Most advanced phase |
| `clinical_lead_nct_id` | string | NCT ID of lead trial |
| `clinical_recency_days` | int | Days since last trial update |
| `clinical_score_raw` | float | Raw Module 4 score |

### Catalyst Debug Fields

| Field | Type | Description |
|-------|------|-------------|
| `catalyst_event_count_upcoming` | int | Number of upcoming events |
| `catalyst_confidence_m3` | float | Module 3 confidence level |
| `catalyst_window_days_m3` | int | Days to nearest catalyst |
| `catalyst_window_bucket_m3` | string | 30d, 60d, 90d, none |
| `catalyst_next_date` | string | Date of next catalyst |
| `catalyst_top_event_type` | string | Type of highest-impact event |
| `catalyst_top_event_date` | string | Date of highest-impact event |

### Confidence Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `confidence_overall` | float | 0-1 | Overall data quality |
| `confidence_financial` | float | 0-1 | Financial data quality |
| `confidence_clinical` | float | 0-1 | Clinical data quality |
| `confidence_catalyst` | float | 0-1 | Catalyst data quality |
| `confidence_pos` | float | 0-1 | PoS mapping confidence |

### Flags and Status Fields

| Field | Type | Description |
|-------|------|-------------|
| `rankable` | bool | Whether security is eligible for investment |
| `severity` | string | none, sev1, sev2, sev3 |
| `stage_bucket` | string | early, mid, late |
| `market_cap_bucket` | string | micro, small, mid, large |
| `fundamental_red_flag` | bool | Material concerns identified |
| `fundamental_red_flag_reasons` | string | Semicolon-separated reasons |

---

## Scoring Methodology

### Composite Score Calculation

```python
composite_score = sum(
    weight_i * normalized_score_i
    for each component
) * defensive_multiplier
```

### Z-Score Calculation

```python
z_score = (composite_score - cohort_mean) / cohort_std
```

### Expected Excess Return

```python
expected_excess_return = IC * z_score * volatility
```

Where IC (Information Coefficient) is calibrated from backtests.

### Defensive Multiplier

```python
if vol < 0.25 and corr < 0.5 and drawdown > -0.15:
    bucket = "elite"
    multiplier = 1.30 + bonus
elif corr < 0.3:
    bucket = "diversifier"
    multiplier = 1.10
elif vol > 0.6 or drawdown < -0.40:
    bucket = "avoid"
    multiplier = 0.75
else:
    bucket = "core"
    multiplier = 1.00
```

---

## PIT (Point-in-Time) Compliance

### Validation Rules

1. **Trial Date Gate**: `last_update_posted <= as_of_date`
2. **Financial Data Gate**: `filing_date <= as_of_date`
3. **Market Data Gate**: `price_date <= as_of_date`
4. **Effective Trading Date**: Events disclosed after market close are effective next trading day

### PIT Date Priority for Trials

```python
PIT_PRIORITY = [
    "first_posted",        # Most reliable - CT.gov posting date
    "last_update_posted",  # Most recent update
    "results_first_posted", # Results posting
    "source_date",         # Data collection date
    "start_date",          # Trial start (least reliable)
]
```

### Date Coverage Diagnostics

Module 4 tracks date coverage to detect PIT issues:

```python
diagnostic_counts = {
    "trials_with_dates": 450,
    "date_coverage_pct": 97.2,
    "date_coverage_warning": None  # or warning message if <90%
}
```

---

## Configuration

### Command-Line Arguments

```bash
python run_screen.py \
    --as-of-date 2026-01-30 \        # Required: analysis date
    --data-dir production_data \      # Input data directory
    --output results.json \           # Output file
    --enable-enhancements \           # Enable momentum, SI, smart money
    --enable-clustering \             # Enable correlation clustering
    --top-n 60 \                      # Limit to top N positions
    --cash-target 0.10 \              # 10% cash allocation
    --defensive-config aggressive     # Elite boost configuration
```

### Defensive Configuration Presets

**Default:**
```json
{
  "elite_multiplier": 1.30,
  "diversifier_multiplier": 1.10,
  "risky_multiplier": 0.90,
  "avoid_multiplier": 0.75
}
```

**Aggressive:**
```json
{
  "elite_multiplier": 1.40,
  "diversifier_multiplier": 1.20,
  "risky_multiplier": 0.85,
  "avoid_multiplier": 0.70
}
```

### Module 5 Weights Configuration

```json
{
  "financial_health": 0.15,
  "clinical_development": 0.35,
  "catalyst": 0.10,
  "momentum": 0.10,
  "smart_money": 0.10,
  "valuation": 0.10,
  "pos": 0.05,
  "short_interest": 0.05
}
```

---

## Appendix: File Locations

| File | Path | Description |
|------|------|-------------|
| Main orchestrator | `run_screen.py` | Pipeline coordinator |
| Module 1 | `module_1_universe.py` | Universe filtering |
| Module 2 | `module_2_financial.py` | Financial health |
| Module 3 | `module_3_catalyst.py` | Catalyst detection |
| Module 4 | `module_4_clinical_dev.py` | Clinical development |
| Module 5 | `module_5_composite_with_defensive.py` | Final ranking |
| CSV export | `export_results_csv.py` | JSON to CSV conversion |
| Production validation | `production_validation.py` | Output validation |
| Date backfill | `backfill_ctgov_dates.py` | PIT date enhancement |

---

## Changelog

- **2026-02-03 v2.0.0**: Comprehensive documentation with field definitions
- **2026-01-30 v1.1.0**: Added PIT diagnostics, production validation
- **2026-01-20 v1.0.0**: Initial documentation

---

*For questions or issues, see the project repository or contact the development team.*
