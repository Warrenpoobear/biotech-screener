# Spec 117 — EES Scoring Methodology for Broad Public Equity Screening

**Status:** RESEARCH / DESIGN  
**Created:** 2026-11-15  
**Governance Tier:** 4 (Research Judgment)  
**Predecessor:** Spec 116 (Expectation Error Score, biotech overlay)  
**Scope:** Methodology only — no implementation, no production change

---

## 1. Purpose

This document specifies a scoring methodology for identifying asymmetric return
opportunities in public equity, adapted from the Expectation Error Score (EES) framework
validated in biotech event-driven screening. It translates the two empirically confirmed
phenomena — **trap detection** and **structure quality** — from a narrow biotech
catalyst universe to a broad equity context.

This is a design document. It does not propose changes to the DEM biotech screener. It
describes how the EES intellectual framework would be implemented as a standalone scoring
system for a general equity universe.

---

## 2. Foundational Principles

### 2.1 The Inversion Principle

EES's central empirical finding: **signals designed to find underpriced opportunities are
more valuable when inverted — used to identify and exclude positions that appear
attractive but systematically underperform.**

This is not contrarianism. It is the recognition that in reasonably efficient markets,
apparent mispricings visible to a quantitative screen are usually visible to everyone
else, and the names that screen as "cheap" are cheap for reasons the screen does not
capture. The informational content is in the negative space: the screen can reliably
identify which "opportunities" are illusory, even when it cannot reliably identify which
are real.

**Implication for methodology:** Every scoring dimension in this system is oriented as a
penalty (identify drag) rather than a signal (identify alpha). The primary model outputs
are exclusion gates, not ranking scores.

### 2.2 Gate-First, Rank-Second

EES demonstrated that percentile-based hard gates (exclude the bottom N%) outperform
linear signal blending. The threshold sweep across 81 operating points showed:

- Hard filter at the 20th percentile: Sharpe 2.4× baseline
- Linear blend of the same signals as ranking weights: Sharpe ~1.0× baseline

The mechanism: linear blending dilutes the signal by spreading it across the full
distribution, including names where the signal is noise. Hard gates concentrate the
benefit where the signal is strongest — the tails.

**Implication for methodology:** The EES broad equity system produces gate decisions
(pass/fail per dimension) as its primary output. Ranking within the surviving universe is
delegated to whatever alpha model the operator runs. EES does not compete with the alpha
model; it cleans the alpha model's opportunity set.

### 2.3 Orthogonal Dimensions

EES's two surviving overlays (quality and trap) had a correlation of 0.179 — near-
orthogonal. This means they filter different names for different reasons, and applying
both gates removes more drag than either alone without excessive overlap.

**Implication for methodology:** Each scoring dimension must be empirically tested for
independence from every other dimension. Correlated dimensions are redundant and should be
merged or dropped, not stacked.

---

## 3. Universe Classification

Before scoring, every name in the universe must be classified along two axes that
determine which base-rate tables and scenario trees apply.

### 3.1 Situation Type

The first axis is the **situation type** — what kind of return-generating event or regime
the name is in. This replaces the biotech `catalyst_family|phase` classification with a
general equity taxonomy.

| Situation Type | Description | Examples |
|---|---|---|
| `EARNINGS_REVISION` | Consensus estimates shifting; name is in an active revision cycle | Pre-earnings drift, estimate momentum |
| `CATALYST_BINARY` | Discrete upcoming event with binary outcome | FDA decision, regulatory ruling, patent verdict, M&A vote |
| `CATALYST_CONTINUOUS` | Ongoing operational catalyst with gradual information release | Product launch ramp, subscriber growth, store openings |
| `RESTRUCTURING` | Corporate transformation in progress | Spin-off, activist campaign, management change, cost restructuring |
| `MEAN_REVERSION` | Name has deviated significantly from intrinsic value metrics | Deep value, distressed recovery, post-selloff bounce thesis |
| `MOMENTUM_CONTINUATION` | Name is in an established trend with positive feedback | Earnings acceleration, sector rotation, breakout |
| `YIELD_CARRY` | Position generates income while waiting for re-rating | Dividend growth, credit spread compression, buyback yield |
| `EVENT_ARB` | Spread trade on an announced corporate event | Merger arb, tender offer, rights offering |

A name may qualify for multiple types. Use the **primary driver** of the investment thesis
— the one the position would be exited on if it resolved. If genuinely ambiguous, score
under each type and take the worst (most conservative) gate outcome.

### 3.2 Quality Tier

The second axis is the **quality tier** — the structural quality of the company, which
determines how to interpret valuation signals. A "cheap" high-quality name is a different
animal from a "cheap" low-quality name.

| Quality Tier | Definition | Sorting Criteria |
|---|---|---|
| `HIGH` | Profitable, positive FCF, investment-grade or near, low leverage | ROE > cost of equity, interest coverage > 5×, positive FCF 3+ years |
| `MEDIUM` | Operationally sound but with leverage, cyclicality, or execution risk | Profitable but cyclical, moderate leverage, mixed FCF |
| `LOW` | Pre-profit, distressed, or structurally challenged | Negative FCF, high leverage, going-concern risk, secular decline |

Quality tier is not itself a scoring dimension — it is a conditioning variable that
changes the interpretation of the trap and structure scores. A low-quality name that
screens as cheap on valuation metrics is far more likely to be a trap than a high-quality
name at the same valuation discount.

---

## 4. Scoring Dimensions

The broad equity EES has five scoring dimensions, organized into two overlay groups
(trap and structure) plus one standalone diagnostic. Each dimension produces a score
normalized to [-1, +1] or [0, 1] and a corresponding percentile gate.

### 4.1 Trap Overlay (T)

The trap overlay identifies names where apparent cheapness is adversely selected —
the market is not mispricing, it is correctly discounting something the screen does not
see.

#### 4.1.1 Valuation Gap Score (VG)

**Analogous to:** EES `base_rate_gap_score`

**Purpose:** Measure how far the name's current valuation deviates from the historical
distribution for names of the same situation type and quality tier.

**Inputs:**
- `valuation_metric`: The primary valuation ratio appropriate to the situation type (see
  table below)
- `historical_median`: Median valuation for the name's {situation_type, quality_tier} cell
  over the trailing 10-year window
- `historical_iqr`: Interquartile range of the same distribution

**Valuation metric by situation type:**

| Situation Type | Primary Metric | Fallback Metric |
|---|---|---|
| EARNINGS_REVISION | Forward P/E (NTM) | EV/EBITDA (NTM) |
| CATALYST_BINARY | EV/Revenue (NTM) | Market cap / pipeline value |
| CATALYST_CONTINUOUS | EV/Revenue (NTM) | PEG ratio |
| RESTRUCTURING | EV/EBITDA (NTM) | Sum-of-parts vs market cap |
| MEAN_REVERSION | P/B or P/TBV | EV/EBITDA (LTM) |
| MOMENTUM_CONTINUATION | Forward P/E (NTM) | EV/Sales growth-adjusted |
| YIELD_CARRY | Dividend yield or FCF yield | Spread to benchmark |
| EVENT_ARB | Announced deal spread | Implied probability vs base rate |

**Formula:**
```
raw_vg = (current_valuation - historical_median) / (historical_iqr + ε)
VG = tanh(raw_vg / 2)
```

**Interpretation:** Negative VG = name trades below median (appears "cheap"). **As a trap
component, negative VG is penalized** — names that look cheap relative to their own
history, conditional on situation type and quality, tend to underperform. The market's
discount reflects information the valuation metric does not capture.

**Conditioning on quality tier:** This is where quality tier matters most. The same -0.5
VG score means very different things:
- HIGH quality at -0.5 VG: Possibly a genuine opportunity (quality floor limits downside)
- LOW quality at -0.5 VG: Likely a trap (cheap and getting cheaper)

The trap gate threshold should be more aggressive (higher cutoff) for LOW quality names.

#### 4.1.2 Consensus Divergence Score (CD)

**Analogous to:** EES `conditional_misprice_score`

**Purpose:** Measure whether scenario-weighted expected returns diverge from the market-
implied return, conditioned on the situation type.

**Inputs:**
- `implied_return`: The market's priced-in expectation, derived from:
  - For options-available names: Implied move from at-the-money straddle
  - For earnings-driven names: Forward P/E × consensus EPS growth rate implied return
  - For event arb: Deal spread (implied probability of completion)
  - Fallback: Consensus target price vs current price
- `scenario_ev`: Probability-weighted expected return across scenarios
- `p_bull`, `p_base`, `p_bear`: Scenario probabilities

**Scenario probability defaults by situation type:**

| Situation Type | p_bull | p_base | p_bear | Bull Move | Base Move | Bear Move |
|---|---|---|---|---|---|---|
| EARNINGS_REVISION | 0.30 | 0.50 | 0.20 | +15% | +3% | -12% |
| CATALYST_BINARY | 0.45 | 0.10 | 0.45 | +25% | +2% | -20% |
| CATALYST_CONTINUOUS | 0.35 | 0.40 | 0.25 | +20% | +5% | -10% |
| RESTRUCTURING | 0.25 | 0.35 | 0.40 | +30% | +5% | -15% |
| MEAN_REVERSION | 0.30 | 0.30 | 0.40 | +25% | 0% | -20% |
| MOMENTUM_CONTINUATION | 0.40 | 0.35 | 0.25 | +18% | +5% | -15% |
| YIELD_CARRY | 0.25 | 0.55 | 0.20 | +12% | +4% | -8% |
| EVENT_ARB | 0.80 | 0.05 | 0.15 | +5% | +1% | -25% |

These are starting defaults. The tables must be empirically calibrated to the actual
universe's realized distributions. Uncalibrated tables are a known weakness of the
biotech EES (Spec 116 §14, Limitation 1).

**Formula:**
```
scenario_ev = p_bull × bull_move + p_base × base_move + p_bear × bear_move
raw_cd = (scenario_ev - implied_return) / (|implied_return| + ε)
CD = tanh(raw_cd / 2)
```

**Interpretation:** Positive CD = scenario EV exceeds implied return (appears
underpriced). **As a trap component, positive CD is penalized** — names where the naive
scenario analysis says "this is too cheap" are the ones most likely to be correctly priced
by the informed market.

#### 4.1.3 Trap Overlay Composite

```
trap_overlay = -(w_vg × VG + w_cd × CD)
```

Default weights: `w_vg = 0.50`, `w_cd = 0.50` (equal weight, following EES precedent
where both components had near-identical IC).

Higher trap_overlay = safer (less likely to be a value trap).

---

### 4.2 Structure Overlay (S)

The structure overlay identifies positions where the thesis may be correct but the
position structure creates negative carry, timing drag, or execution disadvantage.

#### 4.2.1 Timing Decay Score (TD)

**Analogous to:** EES `timing_decay_risk_score`

**Purpose:** Penalize positions where the cost of holding scales with catalyst timing
uncertainty. This is EES's strongest standalone signal.

**Inputs:**
- `implied_cost`: The annualized cost of maintaining the position, derived from:
  - For options positions: Theta decay (annualized) / premium paid
  - For levered positions: Borrowing cost + margin interest
  - For unlevered equity: Opportunity cost proxy = max(0, risk_free_rate - dividend_yield)
  - For short positions: Borrow cost + dividend liability
- `timing_precision`: How well the catalyst timing is known

**Timing precision categories:**

| Precision | Factor | Examples |
|---|---|---|
| `EXACT` | 0.00 | Announced earnings date, scheduled FDA decision, shareholder vote date |
| `WEEK` | 0.15 | "Expected this week" guidance, imminent regulatory decision |
| `MONTH` | 0.35 | "Expected in Q4" narrowed to a month, typical earnings pre-announcement |
| `QUARTER` | 0.60 | "Expected in 2H26", "next quarter", typical restructuring milestones |
| `HALF_YEAR` | 0.80 | "Expected in 2027", vague management guidance |
| `YEAR_PLUS` | 1.00 | No stated timeline, thesis depends on "eventually" |
| `UNKNOWN` | 1.00 | No catalyst identified; pure valuation thesis |

**Formula:**
```
TD = clamp(implied_cost × timing_factor / cost_normalizer, 0.0, 1.0)
```

Where `cost_normalizer` scales by situation type (higher for situation types where
waiting is inherent, e.g. RESTRUCTURING; lower for CATALYST_BINARY where timing
resolution is typically faster):

| Situation Type | cost_normalizer | Rationale |
|---|---|---|
| CATALYST_BINARY | 10.0 | Fast resolution expected |
| EVENT_ARB | 8.0 | Deal timeline usually known |
| EARNINGS_REVISION | 12.0 | Quarterly cadence bounds the wait |
| CATALYST_CONTINUOUS | 15.0 | Gradual, multi-quarter |
| MOMENTUM_CONTINUATION | 15.0 | Trend duration uncertain |
| RESTRUCTURING | 20.0 | Multi-year transformation |
| MEAN_REVERSION | 25.0 | Reversion timing is famously uncertain |
| YIELD_CARRY | 30.0 | Carry offsets waiting cost |

**Interpretation:** High TD = expensive position with uncertain timing. The structure
overlay inverts this: `structure = -(1.0 × TD)`. Names with certain timing and low
holding cost score highest.

**Range:** [0, 1] hard-clamped.

#### 4.2.2 Liquidity Friction Score (LF)

**Analogous to:** EES `slippage_penalty_score` (which was deprecated for look-ahead bias
in biotech, but the concept is sound — the *implementation* was contaminated, not the
idea)

**Purpose:** Penalize positions where execution friction erodes the edge. In broad equity
(unlike biotech), market-cap data is reliable and contemporaneous, so this dimension is
viable.

**Inputs:**
- `avg_daily_volume_20d`: 20-day average daily dollar volume
- `bid_ask_spread_bps`: Median quoted spread in basis points (or estimated from
  tick-size / price)
- `position_size_pct_adv`: Intended position size as % of ADV (for institutional
  screens; omit for retail)

**Formula:**
```
volume_penalty = clamp(1.0 - log10(adv_dollars / 1e6) / 3.0, 0.0, 1.0)
spread_penalty = clamp(spread_bps / 50.0, 0.0, 1.0)
size_penalty = clamp(position_pct_adv / 10.0, 0.0, 1.0)  # 0 for retail screens

LF = 0.40 × volume_penalty + 0.40 × spread_penalty + 0.20 × size_penalty
```

**Interpretation:** High LF = expensive to enter/exit, wide spreads, thin books. This is
a real execution drag, not an alpha signal — it erodes whatever edge the primary model
finds.

**Range:** [0, 1].

**Critical implementation note:** All inputs must be contemporaneous (same-day or
previous-day). The biotech EES slippage penalty died because `market_cap_mm` contained
future prices. Any liquidity metric must be verified as point-in-time. If ADV data lags
by more than 5 business days, substitute a degraded estimate (trailing 60d ADV × 0.80)
and flag the score as `low_confidence`.

#### 4.2.3 Structure Overlay Composite

```
structure_overlay = -(w_td × TD + w_lf × LF)
```

Default weights: `w_td = 0.70`, `w_lf = 0.30`. Timing decay gets more weight because it
showed stronger standalone IC in the biotech context and is harder to arbitrage away.
Liquidity friction is more mechanical and better understood — the market prices it more
efficiently.

Higher structure_overlay = better structure (lower holding cost, better liquidity).

---

### 4.3 Crowding Diagnostic (CR) — Non-Gating

**Analogous to:** EES `crowding_bias_score` (which had IC ~0 in biotech)

**Purpose:** Measure positioning crowding as a **diagnostic** signal. Not gating — the
biotech evidence showed crowding has no predictive power as a hard filter. But extreme
crowding is useful for operator awareness and position-sizing discretion.

**Inputs:**
- `short_interest_pct`: Short interest as % of float
- `institutional_ownership_change_qoq`: Quarter-over-quarter change in institutional
  ownership %
- `analyst_consensus_skew`: (# buy ratings - # sell ratings) / total ratings
- `options_put_call_ratio`: Put/call open interest ratio

**Formula:**
```
si_z = (short_interest - cross_sectional_median_si) / (cross_sectional_iqr_si + ε)
inst_chg_z = (inst_ownership_chg - median_chg) / (iqr_chg + ε)
consensus_z = (analyst_skew - median_skew) / (iqr_skew + ε)
pcr_z = (put_call_ratio - median_pcr) / (iqr_pcr + ε)

CR = tanh((0.30 × si_z + 0.25 × inst_chg_z + 0.25 × consensus_z + 0.20 × pcr_z) / 2)
```

All cross-sectional statistics are computed per batch (same-day universe), not from
historical lookup tables. This follows EES's approach of computing SI percentiles per
snapshot.

**Interpretation:** High CR = crowded long (everyone agrees). Low CR = contrarian or
heavily shorted. This is reported but does not gate. Extreme values (|CR| > 0.7) are
flagged in the operator diagnostic.

**Range:** [-1, +1].

**Why non-gating:** Crowding had IC ~0 in the biotech universe. In broader equity,
crowding may carry signal (short squeezes, herding unwinds), but the evidence from EES
says: do not gate on it until you have a positive Checklist v2 result. Report it. Let the
operator decide. Do not automate a filter on an unvalidated signal.

---

## 5. Gate System

### 5.1 Gate Architecture

Each gating dimension (trap and structure) operates as an independent percentile-based
filter. Gates are computed cross-sectionally per scoring run — the thresholds are relative
to the current universe, not fixed absolute values.

```
For dimension D with scores S_1 ... S_n (sorted ascending):

    threshold = S[floor(n × cutoff_pct / 100)]

    gate_pass(name_i) = S_i > threshold

    eligible(name_i) = trap_gate_pass AND structure_gate_pass
```

### 5.2 Default Operating Modes

| Mode | Trap Cutoff | Structure Cutoff | Use Case |
|---|---|---|---|
| `permissive` | 10% | 0% (OFF) | Maximum coverage; trap filter only |
| `normal` | 20% | 10% | Balanced — primary operating point |
| `conservative` | 25% | 20% | Tighter filtering; lower frequency, higher quality |
| `aggressive` | 30% | 25% | Maximum filtering; fewest names, strongest structure |

**Recommended starting point:** `normal` mode (trap 20%, structure 10%). This follows the
EES precedent where trap was the primary gate and structure (timing) was secondary.

### 5.3 Degeneracy Protection

When a gate distribution cannot differentiate names, the gate automatically bypasses:

1. **All identical:** If `min(scores) == max(scores)`, all names pass
2. **Mode-dominated:** If >75% of scores are identical (single value dominates), all pass
3. **Insufficient universe:** If fewer than 20 names are scored, all gates bypass

Degeneracy protection prevents false filtering when input data is sparse, missing, or
uniformly distributed. The threshold is set at 75% (slightly more sensitive than EES's
80%) because the broader equity universe has more heterogeneous data quality.

### 5.4 Quality-Tier-Adjusted Thresholds

Gates can be tightened for lower-quality tiers, reflecting the empirical observation that
value traps are concentrated in low-quality names:

| Quality Tier | Trap Cutoff Adjustment | Structure Cutoff Adjustment |
|---|---|---|
| HIGH | -5pp (more permissive) | No adjustment |
| MEDIUM | No adjustment (default) | No adjustment |
| LOW | +10pp (more aggressive) | +5pp (more aggressive) |

Under `normal` mode with quality adjustment:
- HIGH quality names: trap gate at 15%, structure at 10%
- MEDIUM quality names: trap gate at 20%, structure at 10%
- LOW quality names: trap gate at 30%, structure at 15%

This is a design choice to be validated empirically. The biotech EES did not differentiate
by quality tier because the biotech universe is almost entirely low-quality (pre-profit).

---

## 6. Composite Score

### 6.1 EES Broad Equity Score

```
ees_broad = w_trap × trap_overlay + w_structure × structure_overlay
```

Default weights: `w_trap = 0.55`, `w_structure = 0.45`.

Trap gets slightly more weight because:
- The biotech trap overlay had marginally higher IC than quality (t=18.5 vs t=18.1)
- Trap detection is the more novel insight; structure quality is better understood by the
  market and therefore less likely to persist as edge

### 6.2 Score Interpretation

| Score Range | Interpretation |
|---|---|
| > 0.0 | Above-average structure and low trap risk |
| -0.25 to 0.0 | Average — neither particularly safe nor risky |
| -0.50 to -0.25 | Elevated trap or structure risk; review recommended |
| < -0.50 | High risk on one or both dimensions; likely gated out |

### 6.3 Usage Modes

The EES broad equity score can be deployed in three ways:

**Mode A: Post-Filter (Recommended)**
Run the primary alpha model (whatever it is), then apply EES gates to the output. Names
that fail either gate are excluded from the final portfolio. EES does not change the
ranking of survivors — the primary model's order is preserved.

This is the recommended deployment because it:
- Preserves the primary model's structure and interpretability
- Requires no retraining or recalibration of the primary model
- Adds value through exclusion, not through ranking
- Can be toggled on/off without changing anything else

**Mode B: Overlay Adjustment**
Add the EES broad equity score as a penalty term to the primary model's composite score:

```
adjusted_score = primary_score + lambda × ees_broad
```

Where `lambda` is calibrated to produce the desired filtering intensity. Start with
`lambda = 0.20` (EES explains 20% of the variation in the adjusted score). This is more
parametric than Mode A and requires backtesting to calibrate `lambda`.

**Mode C: Standalone Screen**
Use EES gates alone as a negative screen on the full universe, without a primary alpha
model. The surviving names are the "opportunity set" — structurally sound names that are
not obvious traps. An operator or discretionary PM then selects from this set.

This is viable for fundamental/discretionary investors who want a quantitative pre-filter
but do not use a quantitative alpha model.

---

## 7. Data Requirements

### 7.1 Required Inputs (Gate System Fails Without These)

| Field | Source | Frequency | PIT Requirement |
|---|---|---|---|
| Valuation metric (P/E, EV/EBITDA, etc.) | Financial data vendor | Daily or weekly | Must be as-of-date; no forward-looking |
| Historical valuation distribution | Computed from archive | Trailing 10y, quarterly | Must use archived snapshots per date |
| Consensus estimates (EPS, revenue) | Sell-side consensus | Weekly | Must be consensus as-of snapshot date |
| Implied volatility / option-implied move | Options data vendor | Daily | Same-day or T-1 |
| Catalyst date / event calendar | Corporate calendar + filings | Daily | Disclosed-date only; no private knowledge |
| Catalyst timing precision | Derived from event source | Per-event | — |

### 7.2 Recommended Inputs (Improve Accuracy But Not Required)

| Field | Source | Used In |
|---|---|---|
| Average daily dollar volume (20d) | Market data | Liquidity friction |
| Bid-ask spread | Level 1 quotes | Liquidity friction |
| Short interest % of float | Lending data (bi-monthly) | Crowding diagnostic |
| Institutional ownership change (QoQ) | 13F filings | Crowding diagnostic |
| Analyst consensus distribution | Sell-side data | Crowding diagnostic |
| Put/call open interest ratio | Options data | Crowding diagnostic |

### 7.3 PIT Discipline

Every input must satisfy point-in-time (PIT) constraints:

1. **`data_available_timestamp <= as_of_date`** — no future information
2. **Financial statements:** Use the filing date (10-K/10-Q accepted date), not the period
   end date. A company reporting Q3 results on November 5 has Q3 data available on
   November 5, not September 30.
3. **Consensus estimates:** Use the consensus snapshot as of the scoring date. Do not use
   today's consensus to evaluate yesterday's scoring decision.
4. **Options data:** Same-day close or T-1. Options surfaces change daily; stale data
   (>2 business days) must be flagged as low confidence.
5. **Institutional holdings (13F):** Available 45 days after quarter end. Do not use Q2
   13F data in May — it does not exist until mid-August.

**The biotech EES lost its slippage signal to a PIT violation.** The market_cap field
contained current prices, not historical. This contaminated the entire signal. PIT
discipline is not optional — it is the difference between a real signal and a mirage.

---

## 8. Calibration Requirements

### 8.1 Base-Rate Tables Must Be Empirically Derived

The biotech EES used hardcoded base-rate tables from academic literature (Wong et al.
clinical trial success rates). This is a known weakness (Spec 116 §14, Limitation 1).

For the broad equity implementation, base-rate tables must be built from the actual
universe's realized return distributions:

1. **Collect:** For each {situation_type, quality_tier} cell, gather all realized returns
   for names that were in that cell at scoring time, measured at 5d, 20d, and 63d horizons
2. **Compute:** Median, IQR, P10, P90 of the realized absolute return distribution
3. **Require:** Minimum 50 observations per cell. Cells with fewer observations use the
   situation_type marginal (collapsed across quality tiers). Cells with fewer than 20
   observations at the marginal level use the unconditional distribution.
4. **Refresh:** Quarterly, using a rolling 10-year window
5. **Version:** Every table version is timestamped and archived. The scoring function logs
   which table version was used for each run.

### 8.2 Scenario Probability Tables

The scenario probability defaults (§4.1.2) are starting points only. Calibration requires:

1. **Event resolution data:** For each situation type, collect realized outcomes
   (bull/base/bear) and compare to the default probabilities
2. **Move magnitude calibration:** Compare realized moves in each scenario to the default
   magnitudes
3. **Annual re-estimation:** Probabilities shift with market regimes. A table calibrated
   in 2023 may not reflect 2026 dynamics.
4. **Minimum calibration sample:** 100 resolved events per situation type before
   overriding defaults

### 8.3 Gate Threshold Calibration

Follow the EES threshold sweep methodology:

1. **Grid search:** Score the universe historically, then sweep gate thresholds from 0% to
   40% in 5pp increments for both trap and structure
2. **Metrics per cell:** Sharpe ratio, mean excess return, hit rate, trade frequency
3. **Efficient frontier:** Plot Sharpe vs frequency. The optimal operating point maximizes
   Sharpe subject to frequency >= 50% (the screen must trade more than half the time)
4. **Out-of-sample validation:** The sweep identifies the threshold on the first half of
   history; confirm stability on the second half
5. **Annual re-sweep:** Market structure changes. Thresholds may drift.

---

## 9. Validation Framework

### 9.1 Required Before Deployment

Following the Checklist v2 standard from the biotech screener:

| Test | Requirement | Purpose |
|---|---|---|
| **Fama-MacBeth regression** | EES dimensions significant after controlling for known factors (market, size, value, momentum, quality, low-vol) | Confirms EES captures something factors miss |
| **Bootstrap confidence interval** | 95% CI on Sharpe improvement > 0 | Confirms economic significance |
| **FDR correction** | Benjamini-Hochberg across all tested dimensions; only dimensions surviving FDR at 5% are used in gates | Corrects for multiple testing |
| **LOSO (Leave-One-Subperiod-Out)** | Positive Sharpe improvement in >= 75% of subperiods | Confirms temporal stability |
| **Year stability** | No single year drives more than 40% of cumulative performance | Confirms not a regime artifact |

### 9.2 Required Before Increasing Gate Aggressiveness

| Test | Requirement |
|---|---|
| Forward shadow (30+ trading days) | Gate-filtered portfolio outperforms unfiltered |
| Coverage monitoring | Gate does not exclude >40% of the opportunity set |
| Regime conditioning | Gate adds value in both risk-on and risk-off environments |

### 9.3 Ongoing Monitoring

| Metric | Frequency | Alert Threshold |
|---|---|---|
| Trap overlay IC (cross-sectional Spearman) | Weekly | IC < 0 for 4 consecutive weeks |
| Structure overlay IC | Weekly | IC < 0 for 4 consecutive weeks |
| Gate-eligible vs gate-excluded spread | Monthly | Spread < 0 for 2 consecutive months |
| Trap-structure correlation | Monthly | Correlation > 0.50 (losing orthogonality) |
| Degeneracy rate | Daily | >30% of runs trigger degeneracy bypass |
| Coverage (% eligible) | Daily | <50% of universe eligible |

---

## 10. Implementation Pseudocode

```python
class EESBroadEquity:
    """Expectation Error Score for broad public equity screening.

    Produces trap and structure gates as primary output.
    Ranking within survivors is delegated to the operator's alpha model.
    """

    def score_universe(
        self,
        universe: List[Dict],     # rows with required fields
        as_of_date: str,
        gate_mode: str = "normal",
        quality_adjust: bool = True,
    ) -> List[Dict]:
        """Score all names and compute gates.

        Returns enriched rows with:
            valuation_gap_score, consensus_divergence_score,
            timing_decay_score, liquidity_friction_score,
            crowding_diagnostic_score,
            trap_overlay, structure_overlay, ees_broad_score,
            trap_gate, structure_gate, ees_eligible
        """

        # 1. Classify each name
        for row in universe:
            row["_situation_type"] = classify_situation(row)
            row["_quality_tier"] = classify_quality(row)

        # 2. Compute cross-sectional anchors (for crowding)
        si_vals = [r["short_interest_pct"] for r in universe if r.get("short_interest_pct")]
        si_median = median(si_vals)
        si_iqr = iqr(si_vals)
        # ... similar for other crowding inputs

        # 3. Score each name
        for row in universe:
            sit = row["_situation_type"]
            qt = row["_quality_tier"]

            # Trap overlay components
            row["valuation_gap_score"] = self._valuation_gap(row, sit, qt)
            row["consensus_divergence_score"] = self._consensus_divergence(row, sit)
            row["trap_overlay"] = -(0.50 * row["valuation_gap_score"]
                                  + 0.50 * row["consensus_divergence_score"])

            # Structure overlay components
            row["timing_decay_score"] = self._timing_decay(row, sit)
            row["liquidity_friction_score"] = self._liquidity_friction(row)
            row["structure_overlay"] = -(0.70 * row["timing_decay_score"]
                                       + 0.30 * row["liquidity_friction_score"])

            # Crowding diagnostic (non-gating)
            row["crowding_diagnostic_score"] = self._crowding(
                row, si_median, si_iqr, ...
            )

            # Composite
            row["ees_broad_score"] = (0.55 * row["trap_overlay"]
                                    + 0.45 * row["structure_overlay"])

        # 4. Compute gates
        mode_thresholds = GATE_MODES[gate_mode]
        trap_cut = mode_thresholds["trap_cut_pct"]
        struct_cut = mode_thresholds["structure_cut_pct"]

        trap_scores = sorted(r["trap_overlay"] for r in universe)
        struct_scores = sorted(r["structure_overlay"] for r in universe)
        n = len(universe)

        trap_thresh = trap_scores[int(n * trap_cut / 100)]
        struct_thresh = struct_scores[int(n * struct_cut / 100)]

        # Degeneracy check
        trap_degenerate = is_degenerate(trap_scores, threshold=0.75)
        struct_degenerate = is_degenerate(struct_scores, threshold=0.75)

        for row in universe:
            qt = row["_quality_tier"]

            # Quality-adjusted thresholds
            adj_trap_cut = trap_cut + QUALITY_ADJUSTMENTS[qt]["trap"]
            adj_struct_cut = struct_cut + QUALITY_ADJUSTMENTS[qt]["structure"]

            adj_trap_thresh = trap_scores[int(n * adj_trap_cut / 100)]
            adj_struct_thresh = struct_scores[int(n * adj_struct_cut / 100)]

            row["trap_gate"] = (True if trap_degenerate
                               else row["trap_overlay"] > adj_trap_thresh)
            row["structure_gate"] = (True if struct_degenerate
                                   else row["structure_overlay"] > adj_struct_thresh)
            row["ees_eligible"] = row["trap_gate"] and row["structure_gate"]

        return universe
```

---

## 11. Differences from Biotech EES

| Dimension | Biotech EES (Spec 116) | Broad Equity EES (this spec) |
|---|---|---|
| Universe | ~300 biotech names | Full public equity universe |
| Situation classification | catalyst_family \| phase | 8 situation types × 3 quality tiers |
| Valuation gap | priced_move_pct vs base-rate table | Valuation metric vs historical distribution |
| Conditional misprice | Scenario EV from clinical outcome priors | Scenario EV from consensus-derived probabilities |
| Timing decay | clinical_days_precision × priced_move | Catalyst timing precision × implied holding cost |
| Liquidity friction | DEPRECATED (market_cap look-ahead) | Rebuilt with ADV + spread + position size |
| Crowding | Short interest only (IC ~0) | Multi-factor (SI + inst chg + consensus + PCR); still non-gating |
| Quality conditioning | None (biotech is uniformly low-quality) | Quality tier adjusts trap gate aggressiveness |
| Base-rate tables | Hardcoded from literature | Must be empirically derived from universe history |
| Gate structure | 2 modes (normal/conservative) | 4 modes with quality-tier adjustment |
| Validation standard | IC-only (Checklist v2 not run) | Full Checklist v2 required before deployment |

---

## 12. Risks and Mitigations

### 12.1 The Inversion May Not Generalize

**Risk:** The trap inversion (obvious cheap underperforms) was validated in biotech, where
information asymmetry is extreme (clinical trial results are insider-knowable). In broader
equity, where information is more evenly distributed, apparent value may actually be value.

**Mitigation:** The Fama-MacBeth test (§9.1) directly tests this. If the trap overlay has
no power after controlling for standard value factors, the inversion does not generalize.
Do not deploy without a positive FM result.

### 12.2 Situation Classification Errors

**Risk:** Misclassifying a name's situation type leads to wrong base rates and wrong
scenario trees. A RESTRUCTURING name misclassified as MEAN_REVERSION gets inappropriate
valuation benchmarks.

**Mitigation:** Use the primary-driver rule (§3.1) and validate classification against a
human-labeled holdout set. Target >85% agreement between automated classification and
expert judgment. Where classification is uncertain, score under multiple types and take
the most conservative gate outcome.

### 12.3 Stale or Look-Ahead Data

**Risk:** The single largest threat. The biotech EES lost its slippage signal to look-
ahead contamination, and the broader DEM system lost its entire backtest narrative to
market-cap look-ahead.

**Mitigation:**
1. Every input carries a `data_available_timestamp`
2. The scoring function asserts `data_available_timestamp <= as_of_date` for every input
3. Liquidity data (ADV, spread) must be same-day or T-1; anything older is flagged
4. Financial data uses filing date, not period date
5. Quarterly PIT audits: re-derive scores using archived-only inputs and compare to
   production scores. Any divergence > 0.01 in mean score triggers investigation.

### 12.4 Overfitting Gate Thresholds

**Risk:** The threshold sweep (§8.3) optimizes on historical data. The optimal thresholds
may not persist out-of-sample.

**Mitigation:**
1. Split-sample validation (calibrate on first half, confirm on second)
2. Prefer the center of a stable plateau over the sharp optimum
3. Forward shadow before deployment (30+ trading days)
4. Annual re-sweep with explicit comparison to prior thresholds
5. Use simple round numbers (20%, not 18.7%) — false precision invites overfitting

### 12.5 Regime Dependence

**Risk:** The EES biotech results were strongest in bear/neutral markets and weakest in
strong bull. In bull markets, even "trap" names can rally, making the gate look like it is
removing winners.

**Mitigation:**
1. Report gate performance by regime (bull/neutral/bear, defined by 6-month market return)
2. Accept that the system will underperform in strong bull markets — it is designed to
   remove drag, and drag matters less when everything rallies
3. Do not abandon the system in a bull market; its value compounds through the full cycle

---

## 13. Relationship to Existing DEM Infrastructure

This spec describes a standalone methodology. However, if implemented within the DEM
infrastructure, the following constraints apply:

1. **Overlay-only policy** — same as biotech EES. Does not touch selector, ranker,
   decision engine, or portfolio construction.
2. **Governance tier** — Tier 4 for design decisions, Tier 3 for implementation, Tier 2
   for daily operation once validated.
3. **Forward validation mandate** — any deployment within DEM requires the same 20-window
   out-of-sample validation as the biotech model.
4. **No shared state** — broad equity EES does not read or write the same artifacts as
   biotech EES. Separate CSV columns, separate sidecar files, separate shadow tracker.

---

## 14. Open Questions

1. **Does the trap inversion survive Fama-MacBeth controls?** This is the existential
   question. If apparent value in broad equity is actually value (unlike in biotech), the
   trap overlay has no power and the methodology reduces to structure-only.

2. **What is the right valuation metric per situation type?** The table in §4.1.1 is a
   starting proposal. The wrong metric produces noise, not signal.

3. **How should the crowding diagnostic be weighted if it ever gains predictive power?**
   Keep it non-gating until there is a positive Checklist v2 result, but design the
   interface to accommodate promotion if evidence materializes.

4. **Should quality-tier-adjusted thresholds be additive (§5.4) or multiplicative?**
   Additive is simpler and more interpretable. Multiplicative may be more appropriate if
   the trap concentration in low-quality names is nonlinear.

5. **What is the minimum universe size for meaningful cross-sectional gates?** EES uses
   n < 5 as its bypass threshold. Broader equity may need n < 50 per {situation_type,
   quality_tier} cell.

---

## Appendix A: Scoring Output Schema

```python
@dataclass
class EESBroadEquityScore:
    ticker: str
    as_of_date: str
    situation_type: str
    quality_tier: str

    # Trap overlay components
    valuation_gap_score: float          # [-1, +1]
    consensus_divergence_score: float   # [-1, +1]
    trap_overlay: float                 # higher = safer

    # Structure overlay components
    timing_decay_score: float           # [0, 1]
    liquidity_friction_score: float     # [0, 1]
    structure_overlay: float            # higher = better structure

    # Crowding diagnostic (non-gating)
    crowding_diagnostic_score: float    # [-1, +1]

    # Composite
    ees_broad_score: float

    # Gates
    trap_gate: bool
    structure_gate: bool
    ees_eligible: bool

    # Confidence
    confidence: float                   # [0, 1]
    missing_inputs: List[str]

    # Audit
    base_rate_table_version: str
    scenario_table_version: str
    gate_mode: str
    quality_adjusted: bool
```

## Appendix B: Gate Mode Definitions

```python
GATE_MODES = {
    "permissive":    {"trap_cut_pct": 10, "structure_cut_pct": 0},
    "normal":        {"trap_cut_pct": 20, "structure_cut_pct": 10},
    "conservative":  {"trap_cut_pct": 25, "structure_cut_pct": 20},
    "aggressive":    {"trap_cut_pct": 30, "structure_cut_pct": 25},
}

QUALITY_ADJUSTMENTS = {
    "HIGH":   {"trap": -5, "structure": 0},
    "MEDIUM": {"trap": 0,  "structure": 0},
    "LOW":    {"trap": 10, "structure": 5},
}
```

## Appendix C: Lessons from Biotech EES

These are the empirical lessons from the biotech implementation (Spec 116) that directly
informed this methodology. They are not theoretical — they are things that were tried,
measured, and either worked or failed:

1. **4 of 6 original sub-scores died.** Build fewer dimensions and validate each before
   adding more. The biotech EES's six-mistake framework was intellectually elegant but
   empirically over-specified.

2. **Linear blending underperformed hard gating.** The threshold sweep demonstrated this
   clearly: Sharpe 2.4× for gates vs ~1.0× for blending. Design for exclusion, not
   ranking.

3. **The inversion was the discovery.** Signals designed to find alpha found drag when
   flipped. This was not in the hypothesis — it emerged from the ablation study.

4. **PIT contamination is the single biggest threat.** The slippage penalty had IC +0.106
   and looked like the strongest signal in the model. It was 100% look-ahead. Later, the
   entire DEM backtest history was deprecated for the same contamination vector.

5. **Orthogonality matters.** Quality and trap had correlation 0.179 and both added value.
   If two dimensions correlate above 0.40, one of them is redundant.

6. **Conservative modes sacrifice frequency.** The best Sharpe came at 63% frequency —
   sitting out a third of dates. Accept the tradeoff or run in normal mode.

7. **Forward shadow is the only real evidence.** 421 snapshots of in-sample IC look
   impressive (t=18+). But those t-stats reflect many non-independent observations, and
   the slippage penalty's t=18.5 was completely spurious. Forward shadow with actual
   completed exits is the only test that matters.
