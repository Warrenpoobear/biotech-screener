# Financial Health Scoring Skill

## Purpose

Score a biotech company's financial survivability to produce a normalized 0-100 financial health score. This skill encodes the exact rules from Wake Robin's Module 2 pipeline (v1 + v2), the Dilution Risk Engine, Liquidity Scoring, and Short Interest Engine so that scoring is reproducible and auditable.

## Preconditions

- All arithmetic MUST use `Decimal` (never `float`). Initialize from strings: `Decimal("500000000")`.
- All dates MUST be ISO 8601 (`YYYY-MM-DD`). Never call `datetime.now()`.
- PIT safety: only use data where `source_date <= as_of_date - 1`.
- Rounding: `ROUND_HALF_UP`. Scores to 2 dp (`0.01`), rates to 4 dp (`0.0001`).

---

## Step 1: Determine Cash Burn Rate

Use a hierarchical priority to select the best available burn rate proxy. Stop at the first available source.

### Burn Rate Source Priority

| Priority | Source | Confidence | Notes |
|----------|--------|-----------|-------|
| 1 | CFO quarterly (explicitly quarterly) | HIGH | Preferred |
| 2 | CFO YTD (with quarter differencing) | HIGH | Derive quarterly |
| 3 | CFO annual (divide by months in period) | HIGH | Annualized |
| 4 | Trailing 4Q average | HIGH | If quarterly history |
| 5 | FCF quarterly/annual (same hierarchy as CFO) | HIGH | Fallback |
| 6 | Net Income (if negative, divide by 3) | MEDIUM | Proxy |
| 7 | R&D * 1.5 / months_in_period | LOW | Last resort |

### YTD Period Detection (by filing date month)

| Filing Month Range | Period | Months |
|-------------------|--------|--------|
| Jan-Mar | Q1 | 3 |
| Apr-Jun | Q2 | 6 |
| Jul-Sep | Q3 | 9 |
| Oct-Dec | Q4/Annual | 12 |

---

## Step 2: Compute Cash Runway

```
runway_months = current_cash / abs(quarterly_burn) * 3
```

If burn rate is zero or positive (cash-generating), `runway_months = 1200` (100 years, effectively infinite).

### Runway Severity Classification

| Runway | Severity | Consequence |
|--------|----------|-------------|
| < 6 months | SEV3 | **Hard gate** - exclude from screening |
| 6-12 months | SEV2 | 50% penalty (soft gate) |
| 12-18 months | SEV1 | 10% penalty (caution) |
| >= 18 months | NONE | No penalty |

### Runway Score (v1, tier-based)

| Runway | Score |
|--------|-------|
| >= 24 months | 100.0 |
| 18-24 months | 90.0 |
| 12-18 months | 70.0 |
| 6-12 months | 40.0 |
| < 6 months | 10.0 |

### Runway Score (v2, piecewise linear)

| Breakpoint | Score |
|-----------|-------|
| 0 months | 5 |
| 6 months | 40 |
| 12 months | 70 |
| 18 months | 90 |
| 24+ months | 100 |

Linear interpolation between breakpoints.

---

## Step 3: Burn Acceleration Analysis (v2 only)

Detect quarter-over-quarter changes in burn rate.

| Condition | Threshold | Action |
|----------|-----------|--------|
| Accelerating burn | QoQ change >= +10% | Penalty up to 30% |
| Decelerating burn | QoQ change <= -10% | Bonus up to +10% |
| Stable | -10% to +10% | No adjustment |

### Acceleration Penalty Formula

```
penalty_pct = min(0.30, avg_qoq_change / 100 * 0.5)
adjusted_runway_score = runway_score * (1.0 - penalty_pct)
```

---

## Step 4: Dilution Risk Scoring

### Cash-to-Market-Cap Sigmoid (v1)

```
sigmoid = 100 / (1 + exp(-15 * (cash_to_mcap - 0.15)))
```

- Inflection point (midpoint): 15% cash/mcap
- Steepness parameter (k): 15.0
- Clamp exp input to [-50, 50] to avoid overflow
- Final score clamped to [0, 100]

### Runway-Based Penalty (v1)

If `runway_months < 12`:
```
penalty_factor = clamp(0.5 + (runway_months / 24), 0.5, 1.0)
dilution_score = dilution_score * penalty_factor
```

### Dilution Risk Buckets (v2)

| Cash/Market Cap Ratio | Risk Level |
|----------------------|-----------|
| >= 30% | LOW |
| 15-30% | MODERATE |
| 5-15% | HIGH |
| < 5% | SEVERE |
| No data | UNKNOWN |

### Financing Pressure Score (v2, 0-100)

Average of three components:

**Runway Component:**
| Runway | Score |
|--------|-------|
| >= 24 months | 0 |
| 12-24 months | 30 |
| 6-12 months | 60 |
| < 6 months | 90 |

**Cash/Mcap Component:**
| Cash/Mcap | Score |
|----------|-------|
| > 30% | 10 |
| 15-30% | 30 |
| 5-15% | 60 |
| <= 5% | 90 |

**Share Dilution Component (if available):**
| Annual Dilution | Score |
|----------------|-------|
| <= 5% | 10 |
| 5-10% | 30 |
| 10-20% | 50 |
| > 20% | 80 |

---

## Step 5: Dilution Risk Engine (Forced-Raise Probability)

Used for catalyst-aware dilution analysis. Computes probability that a company will need to raise capital before its next catalyst.

### Core Parameters

- **RISK_SCORE_MIN**: 0.0
- **RISK_SCORE_MAX**: 1.0
- **USABLE_CAPACITY_FACTOR**: 0.70 (only 70% of shelf/ATM capacity is realistically accessible)

### Cash Gap Calculation

```
monthly_burn = abs(quarterly_burn) / 3
months_to_catalyst = days_to_catalyst / 30.44
cash_needed = monthly_burn * months_to_catalyst
usable_capacity = (shelf_capacity + atm_remaining) * 0.70
total_available = current_cash + usable_capacity
cash_gap = cash_needed - total_available
```

### Raise Feasibility

```
dilution_pct_mcap = cash_gap / market_cap
days_to_raise = cash_gap / (avg_daily_volume * share_price * 0.10)

cap_penalty = min(1.0, dilution_pct_mcap / 0.20)
volume_penalty = min(1.0, days_to_raise / 30)
raise_feasibility = 1.0 - ((cap_penalty + volume_penalty) / 2)
raise_feasibility = clamp(raise_feasibility, 0.0, 1.0)
```

### Hard Limits

- **DILUTION_PCT_MCAP_HARD_THRESHOLD**: 0.20 (>20% of mcap is very difficult)
- **DAYS_TO_RAISE_HARD_THRESHOLD**: 30 days
- **DAILY_VOLUME_UTILIZATION**: 0.10 (10% of daily volume)

### Risk Bucketing

| Condition | Risk Score | Bucket |
|----------|-----------|--------|
| cash_gap <= 0 | 0.0 | NO_RISK |
| raise_feasibility > 0.70 | <= 0.40 | LOW_RISK |
| raise_feasibility 0.40-0.70 | 0.40 + 0.30*(1-f) | MEDIUM_RISK |
| raise_feasibility <= 0.40 | 0.70 + 0.30*(1-f) | HIGH_RISK |

### Confidence Factors (sum, clamped to [0, 1])

| Data Available | Contribution |
|---------------|-------------|
| Base (required fields) | 0.50 |
| shelf_capacity | +0.20 |
| atm_remaining | +0.15 |
| avg_volume | +0.15 |

---

## Step 6: Liquidity Assessment

### Market Cap Tier Boundaries

| Tier | Range |
|------|-------|
| MICRO | < $300M |
| SMALL | $300M - $2B |
| MID | $2B - $10B |
| LARGE | >= $10B |

### ADV (Average Daily Volume) Thresholds by Tier

| Tier | ADV Threshold |
|------|--------------|
| Micro | $750K |
| Small | $2M |
| Mid | $5M |
| Large | $10M |

### ADV Scoring (0-70 pts)

```
ratio = adv / (2 * tier_threshold)
adv_score = clamp(int(ratio * 70), 0, 70)
```

### Spread Scoring (0-30 pts)

| Spread (bps) | Score |
|-------------|-------|
| <= 50 | 30 |
| >= 400 | 0 |
| Between | Linear interpolation |

### Penny Stock Penalty

- **Threshold**: Price < $2.00
- **Effect**: Max liquidity score capped at 10

### Dollar ADV Tiers (v1 scoring)

| Dollar ADV | Score |
|-----------|-------|
| >= $50M | 100 |
| $20-50M | 90 |
| $10-20M | 80 |
| $5-10M | 70 |
| $1-5M | 55 |
| $500K-1M | 40 |
| $100K-500K | 25 |
| < $100K | 10 |

### Liquidity Hard Gates

| Gate | Threshold | Action |
|------|-----------|--------|
| ADV FAIL | < $100K daily | Hard exclusion |
| ADV WARN | $100K-500K | Warning flag |
| ADV PASS | >= $500K | Green light |

### Liquidity Risk Flags

- `FLAG_WIDE_SPREAD`: spread >= 400 bps
- `FLAG_LOW_LIQUIDITY`: ADV < tier threshold
- `FLAG_PENNY_STOCK`: price < $2.00

---

## Step 7: Revenue Scoring (v1)

### Pre-Revenue Baseline

Companies with no revenue start at 50 pts (neutral, no penalty).

### Three Components

**Presence Bonus (binary):**
- Revenue >= $10M: +40 pts
- Revenue < $10M: 0 pts

**Scale Bonus (log-bucketed):**
| Revenue | Bonus |
|---------|-------|
| >= $1B | 40 |
| $100M-1B | 30 |
| $10M-100M | 15 |
| < $10M | 0 |

**Coverage Penalty (if burning despite revenue):**
| Coverage (Revenue/Burn) | Penalty |
|-------------------------|---------|
| < 0.25 | -20 |
| 0.25-0.5 | -10 |
| >= 0.5 | 0 |

Maximum revenue score: 80 pts (40 + 40 + 0).

---

## Step 8: Short Interest Signal

### Squeeze Potential Classification

| Level | SI % of Float | Days-to-Cover |
|-------|--------------|---------------|
| EXTREME | >= 40% | >= 10 |
| HIGH | >= 20% | >= 7 |
| MODERATE | >= 10% | >= 5 |
| LOW | < 10% | < 5 |

### Signal Components (base score = 50)

| Component | Weight | Details |
|----------|--------|---------|
| Squeeze Potential | 40% | EXTREME: +25, HIGH: +15, MOD: +8, LOW: 0 |
| Trend (SI change) | 30% | Covering: up to +15, Building: down to -12 |
| Institutional Support | 20% | >=70%: +10, >=50%: +6, >=30%: +3 |
| Days-to-Cover | 10% | >=15d: +8, >=10d: +5, >=7d: +3, >=5d: +1 |

### Trend Contributions (from SI % change)

**Covering (bullish):**
- <= -20%: +15, <= -10%: +8, <= -5%: +4

**Building (bearish):**
- >= +20%: -12, >= +10%: -6, >= +5%: -3

### Signal Direction

| Score | Direction |
|-------|----------|
| >= 60 | BULLISH |
| 40-60 | NEUTRAL |
| <= 40 | BEARISH |

### Crowding Risk

| SI % of Float | Risk Level |
|--------------|-----------|
| >= 30% | HIGH |
| >= 15% | MEDIUM |
| < 15% | LOW |

---

## Step 9: Compute Composite Financial Score

### V1 Weights

| Component | Weight |
|----------|--------|
| Runway | 45% |
| Dilution | 25% |
| Liquidity | 15% |
| Revenue | 15% |

### V2 Weights

| Component | Weight |
|----------|--------|
| Runway | 50% |
| Dilution | 30% |
| Liquidity | 20% |

```
financial_score = sum(component_score * weight for each component)
financial_score = clamp(financial_score, 0, 100)
```

---

## Step 10: Apply Severity Penalties

| Severity | Multiplier |
|----------|-----------|
| NONE | 1.0 |
| SEV1 | 0.90 |
| SEV2 | 0.50 |
| SEV3 | 0.00 (excluded) |

```
final_score = financial_score * severity_multiplier
```

---

## Data Quality Requirements

### Minimum Field Coverage

- Financial Field Coverage >= 50%: proceed with scoring
- Financial Field Coverage < 50%: issue warning, score with available data
- Market Field Coverage >= 80%: required for inclusion
- Market Field Coverage < 80%: exclude ticker

### Data Quality States

| State | Definition |
|-------|-----------|
| FULL | All key fields present |
| PARTIAL | Some fields present, no critical missing |
| MINIMAL | Only basic fields available |
| NONE | No data available |

---

## Composite Integration

The financial score enters Module 5 composite with these weights:

| Weight Set | Financial Weight |
|-----------|-----------------|
| V3 Enhanced (all signals) | 24% |
| V3 Default (no enhancements) | 35% |
| V3 Partial (some enhancements) | 28% |
| Baker-Style Fundamental | 22% |

---

## Pre-Flight Checks

Before scoring, verify:

1. Financial records are PIT-admissible (`source_date <= as_of_date - 1`)
2. Cash and market cap are present and positive (Decimal type)
3. At least one burn rate source is available (otherwise SEV1)
4. Runway is computed before dilution scoring (dependency)
5. Liquidity data (ADV) is available for gating check
6. All score components use `Decimal` arithmetic
7. Output includes both `financial_score` and legacy `financial_normalized` fields (same value)
8. Output includes `_governance` block with `score_version`, `schema_version`, `pit_cutoff`

---

## Source Files

| Component | File |
|----------|------|
| Financial Scoring (v1) | `module_2_financial.py` |
| Financial Scoring (v2) | `module_2_financial_v2.py` |
| Dilution Risk Engine | `dilution_risk_engine.py` |
| Liquidity Scoring | `liquidity_scoring.py` |
| Short Interest Engine | `short_interest_engine.py` |
| Integration Contracts | `common/integration_contracts.py` |
