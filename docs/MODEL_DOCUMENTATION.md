# Biotech Screener Model Documentation

**Version:** 2.4.2
**Last Updated:** February 22, 2026
**System:** Wake Robin Capital Biotech Screening Pipeline

---

## Table of Contents

1. [Overview](#overview)
2. [Investment Committee Guide](#investment-committee-guide)
3. [System Architecture](#system-architecture)
4. [Data Architecture](#data-architecture)
5. [Module Descriptions](#module-descriptions)
6. [Output Format: JSON](#output-format-json)
7. [Output Format: CSV](#output-format-csv)
8. [Field Definitions](#field-definitions)
9. [Scoring Methodology](#scoring-methodology)
10. [Decision Engine (Phase-2)](#decision-engine-phase-2)
11. [PIT (Point-in-Time) Compliance](#pit-point-in-time-compliance)
12. [Monitoring & Reporting](#monitoring--reporting)
13. [Configuration](#configuration)
14. [Appendix: File Locations](#appendix-file-locations)
15. [Changelog](#changelog)

---


## Overview

The Biotech Screener is a deterministic, production-grade screening system that ranks biotech investment opportunities. It processes clinical trial data, financial filings, market data, and catalyst events to produce a ranked universe plus audit-friendly diagnostics (portfolio construction, position counts, and sizing are downstream policy layers).

### Key Principles

- **Determinism**: No `datetime.now()` calls; all dates are explicit via `--as-of-date`
- **PIT Compliance**: Point-in-time filtering prevents lookahead bias
- **Decimal Arithmetic**: All financial calculations use `Decimal` for precision
- **Stable Ordering**: Output is deterministically sorted for reproducibility
- **Content Hashing**: Inputs are hash-verified for integrity

### Typical Usage

```bash
python run_screen.py \
    --as-of-date 2026-02-14 \
    --data-dir production_data \
    --output production_data/screen_output.json \
    --snapshot-dir data/snapshots \
    --decision-mode phase2 \
    --enable-enhancements \
    --enable-smart-money \
    --log-level INFO
```


## Investment Committee Guide

### What this model is (and is not)

**Designed for**
- Systematic *ranking* of biotech tickers using point‑in‑time (PIT) clinical, financial, market, and catalyst data.
- Repeatable, audit-friendly screening with deterministic outputs and diagnostics.

**Not designed for**
- Automatic security selection counts, position sizing, or risk budgeting (those are *downstream policy layers* on top of the ranked universe).

### Recommended IC workflow

1. **Portfolio**: Start with `decision_portfolio.csv` (Decision Engine output: A+B tier, top-K, sized). Under ruleset v1.4.0, `composite_score` is driven by alpha cohort signals (not Module 5 linear combination). This supersedes legacy Module 5 `composite_rank` for all investment decisions.
2. **Risk gates**: Review `severity`, `risk_flags`, and `tier_reason` before any investment decision.
3. **Thesis build**: Use catalyst provenance (`catalyst_source`, `catalyst_days`, `catalyst_strength`) + Module 4 (pipeline quality) as the thesis backbone.
4. **Health check**: Review `phase2_health.json` — FAIL means do not act, WARN means review before acting.
5. **Implementation**: Apply portfolio constraints (cash target, max weight, sector/cluster caps) on top of Decision Engine weights; document any overrides.
6. **Monitoring**: Track health gate status, catalyst coverage, and tier distribution each run (see Monitoring & Reporting).

### What to show the IC each run (one page)

- **Health gate**: `phase2_health.json` status (OK/WARN/FAIL) and reasons.
- **Portfolio**: `decision_portfolio.csv` — top-20 positions with tiers, weights, catalyst proximity.
- **Delta**: `phase2_run_delta_report.txt` — entries/exits, turnover, L1 weight change vs prior run.
- **Catalyst coverage**: % dev tickers with `specific_days` mode, source mix distribution.
- **Tier distribution**: A/B/C/D counts and any tier migrations from prior run.
- **Exceptions**: ineligible tickers and reasons, any health gate warnings.

---


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
| `institutional_holdings.json` | 13F institutional positions (legacy) | SEC EDGAR | Quarterly |
| `data/caches/sec_13f/PIT/{date}/` | PIT-safe 13F holdings cache | SEC EDGAR via `warm_13f_cache.py` | Daily (warm step) |

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

**Data Sources:**
| Source | Mode | Description |
|--------|------|-------------|
| ClinicalTrials.gov Calendar | Always on | Forward-looking PCD/SCD dates from trial records |
| ClinicalTrials.gov Delta | Always on | Status changes, timeline shifts detected between snapshots |
| SEC 8-K Filings | Always on | Timing phrases extracted from 8-K filings (primary SEC source) |
| SEC Multi-Form (10-Q/10-K/6-K) | `cache_only` | Timing phrases from quarterly/annual/foreign filings; quality-gated |
| FDA Calendar (PDUFA) | Always on | PDUFA action dates |
| FDA Regulatory (Federal Register) | `cache_only` | FDA approvals, CRLs, RTFs, warning letters |

**Quality Gating (Multi-Form Events):**

Multi-form events pass through a three-layer filter before reaching the scoring pipeline:

1. **Source triage**: 10-Q/10-K fetch exhibit documents only (press releases); 6-K/8-K fetch full text
2. **Relevance filter**: Boilerplate blocking (8 keywords: "foreseeable future", "going concern", etc.) + biopharma context requirement (15 keywords: "fda", "phase 1/2/3", "topline", etc.) within ±300 chars of match
3. **Hard gate at merge**: Only MED/HIGH confidence + DAY/WEEK/MONTH/QUARTER precision survive; LOW confidence and HALF_YEAR/YEAR precision events are blocked

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

**SEC Filing Event Types:**
| Event Type | Source Forms | Description |
|------------|-------------|-------------|
| DATA_READOUT | 8-K, 6-K | Topline data, interim analysis |
| FDA_PDUFA_DATE | 8-K, 6-K | PDUFA action date disclosed |
| SAFETY_SIGNAL | 8-K, 6-K | Serious adverse events, clinical holds |
| CLINICAL_HOLD | 8-K, 6-K | FDA clinical hold |

**Scoring Formula:**
```
event_score = impact × confidence × proximity
catalyst_score_net = sum(positive_events) - sum(negative_events)
```

**Source Mix Sidecar:**

Each snapshot writes `catalyst_source_mix.json` alongside `rankings.csv`, containing:
- `total_events`, `unique_tickers_with_events`
- `by_source`: event count per source (CTGOV_CALENDAR, SEC_8K_FILING, FDA_CALENDAR, etc.)
- `by_confidence`: HIGH/MED/LOW distribution
- `by_date_precision`: DAY/WEEK/MONTH/QUARTER/HALF_YEAR/RANGE distribution

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

### Module 5: Composite Ranking (Legacy — being superseded by Decision Engine)

> **Deprecation notice:** Module 5's fixed-weight composite scoring is being replaced by the [Decision Engine (Phase-2)](#decision-engine-phase-2), which provides explicit tier assignments, catalyst-aware eligibility gates, cost-aware sizing, and externalized rulesets. Module 5 continues to produce the `composite_score` and `composite_rank` used as inputs to the Decision Engine, but portfolio decisions (tier, actionable rank, target weight) are now driven by the Decision Engine. See the [architectural transition](#architectural-transition-module-5--decision-engine) note below.

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

#### Component signal definitions (Module 5 inputs)

Module 5 aggregates *normalized* sub-scores (generally 0–100) and then applies the defensive multiplier. The weights below are a policy choice; the key IC question is whether each sub-score is behaving as intended.

- **Financial Health (Module 2)**  
  Normalized 0–100 score based on cash runway, dilution risk, liquidity, and revenue presence. Includes **severity** levels that act as hard/soft gates.  

- **Clinical Development (Module 4)**  
  Normalized 0–100 score from phase advancement/progress, trial count & diversity, recency, and design/execution quality. Provides PIT observability fields to audit what was included vs excluded.

- **Catalyst (Module 3)**  
  Uses net catalyst score plus time-decayed effective score and a proximity bonus (`catalyst_proximity_score`) driven by forward-looking windows and confidence. Includes kill switches via `severe_negative_flag`.

- **Momentum**  
  `momentum_alpha` is excess return vs a benchmark over `momentum_window`; mapped to `momentum_score` (0–100). Treat as a *timing* input, not a thesis anchor.

- **Smart Money**  
  Institutional signal (0–100) derived from overlap counts and holder quality; `smart_money_overlap` and `smart_money_tier1_holders` provide the audit trail.

- **Valuation**  
  Cheapness score (0–100) computed from a raw metric (`valuation_raw_metric`) under a stated method (`valuation_method`). Percentiles are reported both overall and within size buckets.

- **PoS (Probability of Success)**  
  A prior probability (phase × indication) mapped into a 0–100 contribution; `confidence_pos` indicates mapping quality.

- **Short Interest**  
  Crowding/squeeze signal (0–100) with explicit crowding bucket and squeeze potential diagnostic fields.


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

#### Decision Engine Columns (rankings.csv)
Tier assignments, catalyst provenance, and portfolio decisions.

| Field | Type | Description |
|-------|------|-------------|
| `eligible` | 0/1 | Passes eligibility gates |
| `ineligible_reasons` | string | Pipe-delimited reasons if ineligible |
| `tier_dev` | A/B/C/D | Dev-stage tier assignment (drug_developer only) |
| `tier_commercial` | A/B/C/D | Commercial tier assignment (commercial_* only) |
| `tier_any` | A/B/C/D | Unified tier: tier_dev for dev, tier_commercial for commercial |
| `tier_reason` | string | Why tier_dev was assigned |
| `tier_any_reason` | string | Why tier_any was assigned (dev or commercial reason) |
| `actionable_rank` | int | Rank within eligible+tiered universe |
| `target_weight_pct` | float | Target portfolio weight |
| `catalyst_mode` | string | specific_days, blended_window, far_window, no_upcoming, missing |
| `catalyst_days` | int | Days to nearest catalyst |
| `catalyst_strength` | string | NEAR, MID, FAR, MISSING |
| `catalyst_decay_w` | float | Catalyst time-decay weight (1.0 = hard cutoff mode) |
| `catalyst_source` | string | Source of nearest catalyst (CTGOV_CALENDAR, SEC_8K_FILING, etc.) |
| `catalyst_event_type` | string | Type of nearest catalyst event |
| `cat_priority` | int | Catalyst source priority (1=highest) |
| `mom_state` | string | Momentum regime |
| `risk_flags` | string | Pipe-delimited risk flag labels |
| `size_band` | string | L, M, S, XS |
| `size_reasons` | string | Pipe-delimited sizing rationale |
| `cost_mult` | float | Cost-aware sizing multiplier |
| `cost_bucket` | string | Cost classification |
| `missing_components` | string | Pipe-delimited missing data feeds (catalyst\|sponsor\|drawdown) |
| `missingness_penalty` | int | Count of missing components (0..N) |
| `commercial_quality` | float | Raw quality composite (commercial_* only) |
| `commercial_quality_pct` | float | Quality percentile within commercial cohort |
| `clinical_score_z` | float | PIT-safe cross-sectional z-score of Module 4 clinical_score within drug_developer universe (population std) |
| `clinical_score_z_tier` | float | PIT-safe tier-local z-score of clinical_score within (tier_dev × drug_developer); used by clinical sort signal |
| `de_drawdown_missing_reason` | string | Why drawdown is missing: no_price_series, series_too_short, or empty |
| `top_3_drivers` | string | Top 3 composite score drivers with contributions |

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

## Decision Engine (Phase-2)

**File:** `decision_engine.py`
**Purpose:** Actionable portfolio construction layer that supersedes Module 5's composite scoring for all investment decisions

The Decision Engine replaces Module 5's fixed-weight composite ranking as the authoritative source of portfolio decisions. It consumes Module 5's `composite_score` as one input among several, then applies explicit eligibility gates, catalyst-aware tier assignments, and cost-aware sizing to produce a filtered, sized portfolio with full audit trail.

### Architectural Transition: Module 5 → Decision Engine

| Concern | Module 5 (Legacy) | Decision Engine (Current) |
|---------|-------------------|--------------------------|
| Ranking method | Fixed-weight linear combination | Multi-layer: eligibility → overlays → tier → sizing |
| Catalyst integration | 10% weight in composite score | Explicit catalyst strength bands (NEAR/MID/FAR/MISSING) gate tier assignment |
| Portfolio construction | None — ranking only | Full: tier filter, top-K selection, target weights |
| Configuration | Hardcoded weights | Externalized frozen rulesets with content-hash IDs |
| Reproducibility | Implicit via code version | Explicit via pinned `ruleset_id` + health gate |
| Health monitoring | None | Automated: FAIL/WARN/OK with turnover, coverage, and ruleset drift checks |
| Catalyst sources | Single Module 3 net score | Source-aware priority (FDA > CTGOV/SEC > corporate) with quality gating |

Module 5 remains in the pipeline to produce `composite_score` and `composite_rank`, which the Decision Engine uses as a base signal. All downstream consumers (IC reports, portfolio construction, drift monitoring) should use Decision Engine outputs (`tier_dev`, `actionable_rank`, `target_weight_pct`) rather than Module 5's `composite_rank`.

### Processing Layers

| Layer | Name | Description |
|-------|------|-------------|
| Pre-DE | Composite Engine | If `composite_engine="alpha_cohort"`: overwrite composite_score with alpha_cohort_raw, recompute rank/pct |
| L0 | Eligibility | No disqualifying flags (severity, red flags) |
| L2 | Overlays | Catalyst tilt, momentum tilt, cost haircut |
| L4a | Dev Tier | A/B/C/D tier for `drug_developer` archetype based on optionality + catalyst strength |
| L4b | Commercial Tier | A/B/C/D tier for `commercial_*` archetypes based on quality composite + catalyst strength |
| L3 | Sizing | Target weight allocation within tier/band constraints |

### Tier Assignment

#### Dev Tier (`tier_dev`) — drug_developer archetype

| Tier | Criteria | Description |
|------|----------|-------------|
| A | optionality >= a_floor (0.60) + catalyst NEAR/MID | Highest conviction, actionable catalyst |
| B | optionality >= a_floor OR catalyst NEAR/MID | One strong signal present |
| C | Eligible but neither criterion met | Watchlist |
| D | Eligible with adverse flags | Deprioritized |

#### Commercial Tier (`tier_commercial`) — commercial_biotech / commercial_pharma archetypes

| Tier | Criteria | Description |
|------|----------|-------------|
| A | quality_pct >= 0.85 + catalyst NEAR/MID | High-quality commercial with actionable catalyst |
| B | quality_pct >= 0.60 OR catalyst NEAR/MID | Moderate quality or catalyst present |
| C | quality_pct < 0.60 or missing | Low quality or no data |
| D | Ineligible | Deprioritized |

**Commercial quality composite**: 45% financial_score + 35% valuation_score + 20% momentum_score, percentile-ranked within the commercial cohort. Stored as `commercial_quality` (raw) and `commercial_quality_pct` (percentile).

**Unified tier (`tier_any`)**: `tier_dev` for drug developers, `tier_commercial` for commercial names. Always populated; the other tier column is empty.

#### Tiering Priority Mode

| Mode | Sort Key Prefix | Portfolio Filter | Default |
|------|----------------|------------------|---------|
| `dev_first` | (eligible, is_dev, tier_ord, ...) | drug_developer + tier_dev in [A,B] | Yes |
| `tier_first` | (eligible, tier_ord, is_dev, ...) | tier_any in [A,B] (cross-archetype) | No |

In `tier_first` mode, commercial A-tier names compete with dev A-tier for portfolio slots. Within the same tier, dev names sort first. Commercial names do **not** receive the dev optionality sizing boost (`tier_a_dev` band upgrade) — commercial quality is a different metric from dev optionality.

### Catalyst Strength Bands

| Band | Days to Catalyst | Description |
|------|-----------------|-------------|
| NEAR | <= 120 days | Imminent catalyst |
| MID | 121-180 days | Approaching catalyst |
| FAR | > 180 days | Distant catalyst |
| MISSING | No catalyst found | No dated catalyst event |

### Far-Window Catalyst Mode

When `far_window_days > 0` in the ruleset, the pipeline scans trial_records for INTERVENTIONAL trials with future PCD beyond the normal catalyst horizon. Tickers with `catalyst_mode` of `no_upcoming` or `missing` are overridden to `far_window` if a qualifying PCD is found.

| Field | Value |
|-------|-------|
| `catalyst_mode` | `"far_window"` |
| `catalyst_source` | `"CTGOV_PCD_FAR"` |
| `catalyst_event_type` | `"CT_PRIMARY_COMPLETION"` |
| `catalyst_strength` | `"far"` |
| `catalyst_decay_w` | `far_window_decay_mult` (default 0.15) |

**Good catalyst modes** (for health gating): `{"specific_days", "blended_window", "far_window"}`

### Catalyst Priority (Tiebreaker Mode)

When tickers share the same tier/rank, catalyst source priority breaks ties:

| Priority | Sources |
|----------|---------|
| 1 | FDA_CALENDAR, FDA_ADCOM_CALENDAR, FDA_PDUFA |
| 2 | CTGOV_CALENDAR, SEC_10Q/10K/6K_FILING |
| 3 | FEDERAL_REGISTER (procedural notices), corporate/ongoing events |
| 9 | No catalyst |
| 99 | Unknown source |

### Clinical Sort Signal (Within-Tier Reordering)

**Purpose:** Nudge tickers with stronger clinical profiles higher within their existing tier, without changing tier membership.

**Design guarantee:** Clinical affects **within-tier ordering only**. Tier assignment is unchanged — validated across 24 snapshots with zero tier churn.

**Pipeline columns:**

| Column | Scope | Description |
|--------|-------|-------------|
| `clinical_score_z` | All drug_developer | Cross-sectional z-score of Module 4 `clinical_score` (population std, ddof=0) |
| `clinical_score_z_tier` | All drug_developer with tier_dev | Tier-local z-score within (tier_dev × drug_developer). This is the column consumed by the sort key. |

**Sort key integration:**

The clinical signal is **blended into the anchor term** of the sort key (not a separate tuple position, which would be unreachable behind continuously-varying terms):

| Sort Mode | Anchor Term | Blend |
|-----------|-------------|-------|
| tiebreaker | `comp_rank` | `effective_comp_rank = comp_rank - clin_adj` |
| blended | `effective_comp_rank` | `effective_comp_rank = comp_rank - bonus - clin_adj` |
| off | `opt_neg` | `effective_opt_neg = opt_neg - clin_adj` |

**Clinical adjustment formula:**

```
cz_tier = clinical_score_z_tier (from rankings.csv)
stage_mult = {early: 0.0, mid: 1.0, late: 1.5}[stage_bucket]
cz_eff = clamp(max(0, cz_tier), 0, 2.0)     # positive-only + safety clamp
clin_adj = clinical_sort_weight * cz_eff * stage_mult
```

**Gating:**

| Gate | Effect | Rationale |
|------|--------|-----------|
| Stage gate | `early=0` (no effect), `mid=1.0`, `late=1.5` | Clinical signal is most predictive at 6–12m horizons (recall study); late-stage names get stronger boost |
| Positive-only | Only boosts high-clinical; never penalizes low | Avoids "inverted gradient" where early-stage high-optionality names are dragged down by low clinical scores |
| ±2.0 clamp | Bounds tier-local z to prevent spiky outliers in small cohorts | Safety: small tier groups (N<10) can produce extreme z values |

**DecisionRuleset fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_clinical_sort_signal` | bool | `True` (v1.4.0) | Master switch (ON in active ruleset) |
| `clinical_sort_weight` | float | `1.0` | Multiplier for clinical adjustment |
| `clinical_positive_only` | bool | `True` | Only boost positive z, never penalize |
| `clinical_stage_mults` | tuple | `(early=0, mid=1, late=1.5)` | Stage gating multipliers |

**Signal interpretation:** `cz_eff` is a mid/late-stage within-tier preference signal; it is *not* designed to capture extreme binary event winners. Top returners in biotech are often driven by M&A, FDA surprises, or short squeezes on low-clinical-score names — the signal produces lift and above-baseline recall at 6m/12m horizons, but unconditional mean cz_eff among the top-50 can appear inverted. The correct metrics are recall vs baseline, lift, and precision@K.

**Replay evidence (24 snapshots, Jan 15 – Feb 17, 2026, w=1.0 pos_only):**

| Metric | Result |
|--------|--------|
| Top-20 overlap | 100% on 23/24 dates (96%) |
| Top-60 overlap | 100% on all 24 dates |
| Tier churn | Zero |
| Max rank churn | 2 (single date) |
| Up-mover mean cz_tier | +1.31 |
| Down-mover mean cz_tier | -0.71 |
| Direction delta | +2.02 (signal always nudges correctly) |

**Recall evidence (t0=2024-01-31, 285 survivors, top K=50):**

| Flag | N | Baseline | R@50 6m | Lift 6m | R@50 12m | Lift 12m | Prec@50 24m |
|------|---|----------|---------|---------|----------|----------|-------------|
| `clinical_cz_eff_top20` | 31 | 10.9% | 16.0% | 2.11 | 16.0% | 2.91 | 19.35% |
| `clinical_score_z_top20` | 56 | 19.7% | 22.0% | 1.21 | 22.0% | 2.57 | 12.50% |

### Alpha Cohort Scoring (Alternative Ranking Signal)

**File:** `module_5_alpha_cohort.py`
**Purpose:** Table-driven ranking signal keyed on `stage_bucket × catalyst_horizon_band × clinical_z_sign`. Activated via `sort_anchor="alpha_cohort"`.

**Design:** Each ticker maps to one of 36 cohort cells (3 stages × 6 horizons × 2 signs). The cell's historical mean excess return (6m) is shrunk toward zero via James-Stein shrinkage and clipped to prevent extreme values. The resulting `alpha_cohort_raw` is percentile-ranked to produce `alpha_cohort_pct`, which replaces `composite_rank` as the actionable sort anchor when enabled.

**Grid dimensions:**

| Dimension | Values |
|-----------|--------|
| Stage | `early`, `mid`, `late` (empty/unknown → `early`) |
| Horizon | `near_0_30`, `near_31_90`, `near_91_180`, `near_181_270`, `far_271_540`, `none` |
| Sign | `pos` (clinical_score_z_tier > 0), `nonpos` (≤ 0 or missing) |

**Horizon mapping:**

| catalyst_mode | catalyst_days | Horizon band |
|--------------|---------------|--------------|
| `specific_days` / `blended_window` | 0-30 | `near_0_30` |
| `specific_days` / `blended_window` | 31-90 | `near_31_90` |
| `specific_days` / `blended_window` | 91-180 | `near_91_180` |
| `specific_days` / `blended_window` | 181-270 | `near_181_270` |
| `specific_days` / `blended_window` | 271+ | `far_271_540` |
| `far_window` | any | `far_271_540` |
| `no_upcoming` / `missing` | any | `none` |

**Shrinkage formula:** `w = n / (n + shrink_k)`, `alpha = clamp(mean × w, clip_min, clip_max)`

**Percentile:** `pct = (rank - 0.5) / N` where rank 1 = highest alpha. Ties broken by ticker (alphabetical ascending).

**Pipeline columns:**

| Column | Description |
|--------|-------------|
| `alpha_cohort_key` | Pipe-delimited key: `stage\|horizon\|sign` |
| `alpha_cohort_raw` | Shrunk + clipped alpha estimate |
| `alpha_cohort_pct` | Deterministic percentile rank (0, 1) |

**DecisionRuleset fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `composite_engine` | str | `"legacy"` | `"alpha_cohort"` to overwrite composite_score with alpha_cohort_raw |
| `sort_anchor` | str | `"composite_rank"` | `"alpha_cohort"` to use alpha_cohort_pct as primary sort key |
| `alpha_cohort_table_path` | str | `production_data/alpha_cohort_tables/v1.json` | Path to cohort lookup table |
| `alpha_cohort_shrink_k` | float | 50.0 | Shrinkage strength parameter |
| `alpha_cohort_clip_min` | float | -0.10 | Floor for clipped alpha |
| `alpha_cohort_clip_max` | float | 0.10 | Ceiling for clipped alpha |

**Relationship to Module 5:** Module 5 continues to run and produce `composite_score`, `composite_rank`, and all component scores consumed by the Decision Engine. Alpha cohort scoring is an alternative ranking signal that replaces `composite_rank` in the sort key when enabled. Both `composite_score` and `alpha_cohort_pct` are written to the snapshot for diagnostic comparison.

**Active status:** In ruleset v1.4.0 (ID=`aa0aaf28`), `composite_engine="alpha_cohort"` and `sort_anchor="alpha_cohort"` are both enabled, making alpha cohort the authoritative ranking signal for all portfolio decisions.

### Composite Engine Override

When `composite_engine="alpha_cohort"`, the pipeline performs a **pre-DE override**:

1. For each ticker with an `alpha_cohort_raw` value, `composite_score` is overwritten with `alpha_cohort_raw`
2. `composite_rank`, `score_rank_pct`, and `score_z` are recomputed from the new composite_score
3. The Decision Engine then operates on these overwritten values

This means tier assignment (which uses `score_rank_pct` for the optionality floor) is driven by alpha cohort signals rather than the legacy Module 5 linear combination.

**Pipeline ordering invariant:**
```
alpha cohort scoring → composite engine override → alpha signal contract validation
→ far-horizon catalyst hydration → DE loop → clinical_score_z_tier computation
```

### Alpha Signal Contract (`alpha_signal_contract.py`)

**Version:** v1.1.0

Validates required and recommended fields at the Decision Engine boundary to catch data gaps early.

| Scope | Required Fields |
|-------|----------------|
| rec dict | `catalyst_decay.days_to_catalyst`, `catalyst_decay.in_optimal_window`, `defensive_features`, `severity` |
| csv_row | `clinical_score_z` (for drug_developer, commercial_biotech, commercial_pharma) |
| alpha output | `alpha_cohort_key`, `alpha_cohort_raw`, `alpha_cohort_pct` (when alpha enabled) |

```python
from alpha_signal_contract import validate_alpha_inputs, validate_alpha_outputs
validate_alpha_inputs(rec_by_ticker, csv_rows, schema_mode="warn")
validate_alpha_outputs(csv_rows, schema_mode="warn", alpha_cohort_enabled=True)
```

### Ruleset Configuration

Decision rules are externalized as frozen `DecisionRuleset` dataclass instances, serialized to JSON with content-hash IDs for reproducibility.

- **Active ruleset**: `v1.4.0_alpha_cohort_candidate.json` (ID=`aa0aaf28`) — alpha cohort composite engine ON + clinical sort signal ON + tiebreaker priority
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Previous**: `v1.3.4_clinical_sort_candidate.json` (ID=`f9842e1f`)
- **Candidate**: `v1.4.1_tier_first_candidate.json` (ID=`054bc5cc`) — commercial tier promotion with tier_first mode, pending replay

### Phase-2 Health Gate

**File:** `run_phase2_snapshot_delta.py`

The health gate compares current vs prior snapshots and reports OK/WARN/FAIL:

| Check | FAIL Threshold | WARN Threshold |
|-------|---------------|----------------|
| Ruleset identity | != pinned ID | - |
| Portfolio empty | 0 positions | - |
| A-tier count | < 1 | < 2 |
| Optionality coverage | < 80% (when A=0) | - |
| Name turnover | - | > 50% |
| Weight L1 delta | - | > 55% |
| Catalyst coverage drop | - | > 5pp |
| Drawdown coverage | < 95% | < 99% |
| Sponsor coverage | - | < 90% |
| Catalyst component coverage | - | < 85% |
| Portfolio missing data | - | count > 0 |

### Snapshot Outputs

Each Phase-2 run saves to `data/snapshots/{as_of_date}/`:

| File | Description |
|------|-------------|
| `rankings.csv` | Full universe with 40+ decision engine columns |
| `catalyst_source_mix.json` | Event distribution by source, confidence, precision |
| `catalyst_shadow_metrics.json` | Transition KPIs, overlap metrics, source attribution vs prior snapshot |
| `decision_ruleset.json` | Frozen ruleset used for this run |
| `decision_portfolio.csv` | Filtered A+B tier, top-K positions |
| `decision_portfolio.json` | Same as CSV, JSON format |
| `phase2_health.json` | Health gate result + metrics |
| `phase2_run_delta_report.txt` | Human-readable delta report |
| `phase2_run_delta.csv` | Per-ticker delta details |
| `phase2_run_delta_details.json` | Machine-readable delta details (portfolio turnover, catalyst coverage, top catalysts) |
| `ic_onepager.md` | IC one-pager summary (health, portfolio, delta, tier distribution, exceptions) |
| `metadata.json` | Run metadata, version, timestamps |

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

## Monitoring & Reporting

Track these metrics each run to detect data issues, model drift, and unintended behavior changes:

- **Universe health**: `summary.total_evaluated`, `summary.active_universe`, `summary.final_ranked`, exclusions with reasons.
- **PIT health**: Module 4 `date_coverage_pct` distribution; count of PIT-filtered trials.
- **Catalyst coverage**: % tickers with `catalyst_proximity_score > 0`, plus `catalyst_window_bucket_m3` distribution.
- **Risk & overlay**: distribution of `defensive_bucket`, `volatility`, `drawdown`, and `confidence_risk` states.
- **Red flags / kill switches**: counts of `severe_negative_flag`, `severity=SEV3`, and `fundamental_red_flag`.

### PIT-Safe Coinvest Features Builder

**File:** `scripts/build_coinvest_features_from_13f.py`
**Purpose:** Produce deterministic, PIT-safe per-ticker coinvest features from quarterly 13F caches, eliminating lookahead bias from the smart-money factor during historical simulation.

**Design:** Reads ONLY from PIT 13F caches (`data/caches/sec_13f/PIT/{as_of_date}/`). Replicates the `run_screen.py` conviction formula exactly. Compares current vs prior quarter holdings for position change classification (NEW/INCREASE/HOLD/DECREASE/EXIT).

**Conviction formula** (replicates `run_screen.py:802-967`):
```
holder_conviction = tier_w × pos_w × chg_w × recency_w
```

| Component | Formula | Constants |
|-----------|---------|-----------|
| `tier_w` | Manager tier weight | `{1: 1.0, 2: 0.6, 3: 0.2, 0: 0.2}` |
| `pos_w` | `clamp(sqrt(position_pct), 0.5, 2.0)` | `position_pct = holding_value / total_value × 100` |
| `chg_w` | Position change weight | `{NEW: 1.25, INCREASE: 1.25, HOLD: 1.0, DECREASE: 0.75, EXIT: 0.5}` |
| `recency_w` | `clamp(1.5 - days_since_filing / 180, 0.7, 1.5)` | Change threshold: 10% |

**Prior quarter detection:** Scans cache dirs to find one whose dominant `period_of_report` matches the prior quarter-end (Dec→Sep, Sep→Jun, Jun→Mar, Mar→Dec). Override with `--prior-cache-dir` or skip with `--no-prior`.

**Ticker resolution:** Embedded `ticker` field preferred; falls back to `production_data/cusip_static_map.json` (CUSIP→ticker). Non-universe tickers excluded from output but count toward manager's total portfolio value.

**PIT safety:** Future filings (`filed_at > as_of_date`) are skipped. EXIT positions tracked in provenance `change_summary` but excluded from per-ticker features.

**Output schema** (`coinvest_features.v1`):

| Top-level field | Description |
|----------------|-------------|
| `as_of_date`, `cache_as_of_date` | Date alignment |
| `period_of_report`, `prior_period_of_report` | Quarterly report periods |
| `tickers_with_signal`, `signal_coverage_pct` | Coverage metrics |
| `tickers` | Per-ticker features (see below) |
| `provenance` | Builder version, managers used, change summary |

Per-ticker fields: `tier1_count`, `sponsor_tier1_count`, `sponsor_overlap_count`, `coinvest_overlap_count`, `conviction_overlap`, `tier1_conviction_overlap`, `max_tier1_position_pct`, `days_since_latest_filing`, `coinvest_recency_state` (fresh ≤90d / stale), `coinvest_holders`, `holder_tiers`, `position_changes`.

**CLI:**
```bash
python3 scripts/build_coinvest_features_from_13f.py \
  --as-of-date 2025-12-31 \
  --cache-root data/caches/sec_13f/PIT \
  --out production_data/coinvest_features/2025-12-31.json \
  --universe production_data/universe.json
```

Additional flags: `--cusip-map`, `--prior-cache-dir`, `--no-prior`

**Live run (2026-02-21):** 29/29 managers, 277/353 tickers (78.5% coverage), prior Q3 2025 auto-detected. Change summary: 193 NEW, 512 INCREASE, 179 HOLD, 197 DECREASE, 236 EXIT.

**Tests:** 39 tests in `tests/test_build_coinvest_features.py` covering cache loading, per-ticker features, position change classification, prior quarter finding, schema validation, and edge cases.

### 13F Cache Health Gate

The daily production runner includes a WARN-only gate for the PIT-safe 13F institutional holdings cache:

| Check | Result | Condition |
|-------|--------|-----------|
| PASS | Coverage adequate | `coverage_pct >= sec_13f_coverage_warn_pct` (default 80%) |
| WARN | Low coverage | `coverage_pct < 80%` or cache missing/malformed |
| WARN | Schema violation | `validate_sec_13f_index_schema()` fails any of 12 invariants |

The gate **never returns FAIL** — 13F data is informational (not consumed by the screen scoring pipeline). Schema validation checks: required fields, version match (`sec_13f_pit_index.v1`), type correctness, count/coverage consistency, per-manager contract (selected managers must have paths, rejected managers must have rejection_reason), and CIK sort determinism.

**Cache warming:**
```bash
# Standalone
python tools/warm_13f_cache.py --as-of-date 2026-02-19 --elite-only --max-workers 4

# Via dispatcher
python warm_caches.py --as-of-date 2026-02-19 --sources sec_13f
```

**PIT selection algorithm:** Filter filings where `filing_date <= as_of_date` → latest `report_date` → prefer `13F-HR/A` over `13F-HR` → latest `filing_date` among ties.

### Catalyst Shadow Metrics (Telemetry)

Each snapshot writes `catalyst_shadow_metrics.json` with KPIs comparing current vs prior snapshot (dev tickers only):

| Metric | Description |
|--------|-------------|
| `bad_to_good` / `good_to_bad` | Catalyst mode transition counts + ticker lists (good = specific_days or blended_window) |
| `bad_to_good_in_top60` / `bad_to_good_in_top100` | Decision-relevant flips (portfolio touch zone) |
| `median_catalyst_days_good` | Median days-to-catalyst for tickers with good catalyst mode |
| `p10/p50/p90_catalyst_days` | Distribution of catalyst days across all dev tickers |
| `top60_overlap` / `top100_overlap` | Jaccard similarity on composite_rank vs prior snapshot |
| `sec_8k_events` / `ctgov_events` / `fda_events` | Source attribution counts |
| `A_tier_count` | Dev A-tier count |

Rollup: `scripts/rollup_shadow_metrics.py` aggregates per-snapshot JSON into `output/catalyst_shadow_timeseries.csv`.

Suggested artifacts:
- A small "run-to-run diff" report (top movers by rank and by catalyst window).
- A weekly chart of coverage and confidence metrics (should be stable absent upstream data changes).

### Signal Robustness Backtest (`scripts/backtest_signal_robustness.py`)

Out-of-sample cross-sectional IC evaluation across archived snapshots. Tests whether clinical/catalyst/alpha signals predict forward returns.

**Key features:**
- Per-date Spearman IC and top-minus-bottom spread computation
- Forward-return coverage diagnostics (`n_fwd_rets / n_price_rows`)
- Data freshness metadata: `price_end_date`, `max_archive_date`, `price_gap_days`, `fwd_returns_stale`
- Training modes: `expanding`, `trailing-N`, `decay-H` (exponential weighting)
- Price extension: `--extend-prices` auto-fetches missing price data via yfinance before backtest

**CLI flags:**

| Flag | Description |
|------|-------------|
| `--min-fwd-coverage` | Minimum forward-return coverage threshold |
| `--fail-if-stale` | Exit 2 if forward returns are stale |
| `--no-warn-if-stale` | Suppress staleness warnings |
| `--extend-prices` | Auto-fetch missing prices before running |
| `--prices-through` | Extend prices through this date |

**Skip reasons:** `EMPTY_RANKINGS` (no data), `NO_FWD_RET` (no forward returns), `LOW_COVERAGE` (below threshold)

### Diagnostic: Flipper Return Attribution (`scripts/diag_flipper_returns.py`)

#### Purpose

Quantify whether **catalyst-driven state changes** (specifically **bad→good catalyst flips** recorded in `catalyst_shadow_metrics.json`) show **subsequent excess returns** relative to the same-day universe. This is an attribution tool to answer:

> "When the pipeline says a name improved due to catalysts, did it actually outperform afterward?"

This diagnostic is **separate from portfolio construction**. It measures *signal efficacy*, not final sizing.

#### Inputs

* **Snapshots directory** (e.g., `data/snapshots/`), containing per-date outputs.
* Shadow metrics file per snapshot date (e.g., `catalyst_shadow_metrics.json`) that records:
  * flip dates
  * flipped tickers
  * flip type (bad→good)
* Price data:
  * baseline: `production_data/price_history.csv`
  * optional backfill via **yfinance** for missing tickers/dates

#### Outputs

Written to `data/diag/` (diagnostic, transient):

* `flipper_returns_<date_range>.csv` — one row per flip event
* `flipper_returns_<date_range>.md` — summary tables, cohort breakdowns, and top/bottom movers
* `price_cache.csv` — cached fetched prices to avoid refetching

> `data/diag/` is gitignored.

#### Methodology

**1) Collect flip events (PIT-aligned)**

For each snapshot date `t`:

* Read the flip list from shadow metrics.
* Treat `t` as the **signal timestamp** (what the model "knew" as of that snapshot).
* Record each flip as an event `(ticker, flip_date=t, cohort=organic/regime)`.

**2) Define cohorts (to isolate rollout artifacts)**

* **Organic flips:** flip_count ≤ 10 on date `t` (normal behavior; currently small-N)
* **Regime flips:** flip_count > 10 on date `t` (pipeline rollout / backfill days; large-N)

Rationale: bulk flips can be dominated by **data availability changes**, not true information arrival, so they are analyzed separately.

**3) Trading-day forward return alignment**

Using sorted unique trading dates from price data:

* Map each flip date `t` to the next available trading date (if needed).
* Compute forward returns at: **t+1d**, **t+5d**, **t+20d** *(trading days)*
* Skip a horizon if `t+N` is not available (insufficient forward history).

**4) Excess returns vs same-day universe**

For each flip event and each horizon:

* Compute **flipper return**
* Compute **universe median return** on the same flip date and horizon
* Define **excess return = flipper − universe_median**

Also report **hit rate** = % of events where excess return > 0.

**5) Robust summaries**

* Overall + cohort-specific summary stats
* Optional tier breakdowns (A/B/C/D) when sample size supports it
* Outlier-trimmed stats (e.g., drop top/bottom 2 by excess 5d) to reduce single-name distortion
* Top/bottom movers table for inspection

#### Design Decisions

* **No test file:** This is a diagnostic script (like `diag_topn_flippers.py`); arithmetic is straightforward and validated via report inspection.
* **Price cache:** `data/diag/price_cache.csv` prevents repeated downloads across runs.
* **Batch fetching:** Uses yfinance per-ticker fetch with local caching for efficiency.
* **Trading-day alignment:** Uses actual trading dates from the price series to define t+N.
* **Regime threshold:** `flip_count > 10` to separate "rollout days" from organic flips (organic max is low in current telemetry; rollout days can be 70+).

#### Verification / Runbook

1. **No network / coverage check**
   ```bash
   python3 scripts/diag_flipper_returns.py --snapshot-dir data/snapshots --no-fetch
   ```
   Expected: runs using existing prices, reports missing coverage.

2. **Full run with backfill**
   ```bash
   python3 scripts/diag_flipper_returns.py --snapshot-dir data/snapshots
   ```
   Expected: writes CSV + markdown report, updates `price_cache.csv`.

3. **Inspect report** — cohort separation present, coverage % reported, stats tables populated, top/bottom movers plausible.

4. **Sanity expectations** — organic flips: small sample, noisy; regime flips: dominates sample, interpret carefully due to overlapping windows and rollout artifacts.

#### Interpretation Notes / Limitations

* **Overlapping windows:** if the same ticker flips across multiple nearby dates (e.g., staggered cache rollout), return windows overlap; treat as non-independent observations.
* **Horizon availability:** t+20 may be unavailable for recent flip dates (insufficient forward data).
* **Regime flips are not "true events":** they can reflect improved detection rather than real-world information arrival; their return attribution is still useful, but should be interpreted as "signal added by coverage expansion," not necessarily "market reaction to new info."

#### Connection to Continuous Alpha Columns

The flipper return attribution diagnostic is the **empirical validation harness** for the two continuous alpha columns (e.g., `catalyst_alpha_z` and `clinical_alpha_z`): a "flip" is simply a **discrete manifestation of a meaningful upward change** in the underlying catalyst/clinical state, and the diagnostic tests whether those state improvements are followed by **positive forward excess returns** versus the same-day universe. In practice, you can treat the flipper cohorts as **event labels** for "large positive deltas" in `catalyst_alpha_z` (and later `clinical_alpha_z`), then use the same framework to evaluate (a) **level vs delta** predictive power (does high `*_alpha_z` outperform, or do **jumps** in the score outperform?), (b) **horizon fit** (t+1/t+5/t+20), and (c) **artifact isolation** (regime flips indicate coverage rollouts; organic flips are closer to real information arrival). This keeps the columns continuous for ranking/diagnostics, while giving you a PIT-safe way to prove they behave like alpha rather than noise.

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
| Module 3 | `module_3_catalyst.py` | Catalyst detection + scoring |
| Module 4 | `module_4_clinical_dev.py` | Clinical development |
| Module 5 | `module_5_composite_with_defensive.py` | Final ranking |
| Decision Engine | `decision_engine.py` | Post-processing: tiers, sizing, portfolio |
| Phase-2 Delta/Health | `run_phase2_snapshot_delta.py` | Health gate + delta report |
| SEC 8-K Collector | `wake_robin_data_pipeline/collectors/sec_8k_catalyst_collector.py` | SEC filing catalyst extraction |
| FDA Collector | `wake_robin_data_pipeline/collectors/fda_adcom_collector.py` | FDA calendar + regulatory notices |
| Drift Report | `scripts/run_drift_report.py` | Daily guardrails + rollback triggers |
| Shadow Metrics Rollup | `scripts/rollup_shadow_metrics.py` | Aggregate per-snapshot telemetry into timeseries |
| IC One-Pager | `scripts/make_ic_onepager.py` | Generate IC summary from snapshot artifacts |
| Ruleset Compare/Replay | `scripts/compare_rulesets_replay.py` | Re-sort rankings with baseline vs candidate ruleset |
| Cache Warmer | `warm_caches.py` | Pre-build SEC 8-K, FDA, and 13F caches before screen run |
| 13F Cache Builder | `tools/warm_13f_cache.py` | PIT-safe 13F holdings cache + schema validator |
| Manager Registry | `elite_managers.py` | Elite (Tier 1) and full manager CIK lists |
| Flipper Return Attribution | `scripts/diag_flipper_returns.py` | Forward return analysis for catalyst flips |
| Top Returners Recall | `scripts/diag_top_returners_recall.py` | Multi-horizon signal recall study (clinical + catalyst vs realized returns) |
| Ablation Comparison | `scripts/compare_ablation_snapshots.py` | Snapshot A/B comparison |
| Alpha Cohort Scoring | `module_5_alpha_cohort.py` | Table-driven alternative ranking signal |
| Alpha Signal Contract | `alpha_signal_contract.py` | DE boundary validation (v1.1.0) |
| Signal Robustness Backtest | `scripts/backtest_signal_robustness.py` | Out-of-sample IC + coverage diagnostics |
| PIT Coinvest Features | `scripts/build_coinvest_features_from_13f.py` | PIT-safe coinvest features from 13F cache |
| CSV export | `export_results_csv.py` | JSON to CSV conversion |
| Production validation | `production_validation.py` | Output validation |
| Date backfill | `backfill_ctgov_dates.py` | PIT date enhancement |

---

## Changelog

- **2026-02-22 v2.4.2**: Added PIT-safe coinvest features builder (`scripts/build_coinvest_features_from_13f.py`). Standalone script reads ONLY from quarterly PIT 13F caches to produce deterministic per-ticker coinvest features (conviction formula, position changes, tier counts). Eliminates lookahead bias from smart-money factor during historical simulation. Output schema `coinvest_features.v1` with provenance tracking. Live run: 29/29 managers, 277/353 tickers (78.5% coverage). 39 tests.
- **2026-02-20 v2.4.1**: Added PIT-safe 13F institutional holdings warm cache (`tools/warm_13f_cache.py`). Schema-versioned index (`sec_13f_pit_index.v1`) with 12-invariant pure validator. WARN-only `sec_13f_cache` gate in daily production runner. Integrated into `warm_caches.py` dispatcher (`--sources sec_13f`) and CI workflow. 29 elite managers, 100% coverage on 2026-02-19 snapshot.
- **2026-02-18 v2.4.0**: Alpha cohort composite engine promoted — pinned ruleset `f9842e1f` → `aa0aaf28` (`composite_engine="alpha_cohort"`, `sort_anchor="alpha_cohort"`). Added composite engine override (pre-DE rewrite of composite_score/rank/pct). Added `far_window` catalyst mode for far-horizon PCD detection. Added alpha signal contract v1.1.0 (`alpha_signal_contract.py`). Added PIT event ledger sidecar. Added signal robustness backtest (`backtest_signal_robustness.py`) with forward-return coverage diagnostics, data freshness metadata, and `--extend-prices` auto-fetch. Added `sort_anchor="optionality_pct"` option. Added catalyst coverage bucket telemetry to shadow metrics.
- **2026-02-17 v2.3.1**: Promoted clinical sort signal — pinned ruleset `e1be5370` → `f9842e1f` (`enable_clinical_sort_signal=True`). Extended clinical z-scores to commercial cohorts. Added `clinical_sort_telemetry` to snapshot metadata (`n_nonzero_clin_adj_dev`, `n_nonzero_clin_adj_comm`). Fixed ctgov cache masking integration PIT tests.
- **2026-02-16 v2.3.0**: Clinical sort signal — tier-local z-score (`clinical_score_z_tier`) blended into sort key anchor, stage-gated (early=0, mid=1.0, late=1.5), positive-only with ±2.0 clamp. Candidate v1.3.4 (`f9842e1f`). Added `clinical_score_z` (cross-universe) and top returners recall diagnostic.
- **2026-02-16 v2.2.1**: Added flipper return attribution diagnostic documentation (methodology, runbook, interpretation notes)
- **2026-02-16 v2.2.0**: Commercial tier promotion (tier_commercial, tier_any, quality composite, tiering_priority_mode), missingness penalty columns + health guardrails, catalyst shadow metrics telemetry, IC one-pager, updated pinned ruleset to e1be5370, corrected catalyst priority table (FEDERAL_REGISTER demoted to pri=3), expanded Decision Engine columns table
- **2026-02-15 v2.1.0**: Added Decision Engine (Phase-2) section, expanded Module 3 with SEC multi-form sources, quality gating, source mix sidecar, updated file locations
- **2026-02-03 v2.0.0**: Comprehensive documentation with field definitions
- **2026-01-30 v1.1.0**: Added PIT diagnostics, production validation
- **2026-01-20 v1.0.0**: Initial documentation

---

*For questions or issues, see the project repository or contact the development team.*
