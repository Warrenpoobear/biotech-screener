# Biotech Alpha Screener — Model Documentation

**Version**: 1.0.0
**Effective Date**: 2026-01-28
**Classification**: Internal Use
**Owner**: Wake Robin Capital Management

---

## Executive Summary

The Biotech Alpha Screener is a quantitative ranking system designed to identify asymmetric risk/reward opportunities in the biotechnology sector. The model combines clinical, financial, catalyst, and market signals into a single composite score, with explicit governance controls to prevent common failure modes.

**Design Philosophy**: Conservative, hard to impress, easy to disappoint, and explicit about why.

---

## 1. Model Architecture

### 1.1 Pipeline Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Module 1   │     │  Module 2   │     │  Module 3   │     │  Module 4   │
│  Universe   │────▶│  Financial  │────▶│  Catalyst   │────▶│  Clinical   │
│  Selection  │     │  Health     │     │  Scoring    │     │  Pipeline   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                              │                 │                 │
                              └────────────────┼────────────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │      Module 5       │
                                    │  Composite Scoring  │
                                    │  (with Enhancements)│
                                    └─────────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   Ranked Output     │
                                    │   + Diagnostics     │
                                    └─────────────────────┘
```

### 1.2 Data Flow

| Stage | Input | Output | Coverage |
|-------|-------|--------|----------|
| M1 Universe | Market data | 303 active securities | 100% |
| M2 Financial | SEC filings, price data | Financial scores + severity | 100% |
| M3 Catalyst | Event calendars, FDA data | Catalyst scores + decay | 100% |
| M4 Clinical | Trial registries | Clinical scores + phase | 100% |
| Clinical Filter | M4 output | 280 eligible (excl. preclinical) | 92% |
| M5 Composite | All modules | 248 ranked securities | 82% |

### 1.3 Technical Requirements

- **Determinism**: Identical inputs produce identical outputs (hash-verifiable)
- **PIT Safety**: No future data leakage; all calculations use as-of-date
- **Decimal Arithmetic**: All scoring uses Python Decimal to prevent float drift
- **Auditability**: Full provenance chain with explicit flags

---

## 2. Component Definitions

### 2.1 Clinical Score (26% base weight)

**Purpose**: Assess pipeline strength and clinical execution quality.

| Factor | Description | Weight |
|--------|-------------|--------|
| Lead Phase | Most advanced program stage | 30% |
| Trial Count | Active trial breadth | 20% |
| Pipeline Diversity | Therapeutic area spread | 15% |
| Trial Quality | Design, endpoints, enrollment | 20% |
| Competitive Position | vs. peers in same indication | 15% |

**Normalization**: Cohort-relative (early/mid/late stage peers)

### 2.2 Financial Score (24% base weight)

**Purpose**: Assess balance sheet durability and financing risk.

| Factor | Description | Weight |
|--------|-------------|--------|
| Cash Runway | Months of cash at current burn | 50% |
| Dilution Risk | Cash/market cap ratio | 30% |
| Liquidity | Dollar ADV for position sizing | 20% |

**Severity Levels**:
- SEV3 (Critical): Runway < 6 months
- SEV2 (Warning): Runway 6-12 months
- SEV1 (Caution): Runway 12-18 months
- None (Healthy): Runway ≥ 18 months

### 2.3 Catalyst Score (16% base weight)

**Purpose**: Assess near-term event optionality with time decay.

| Factor | Description |
|--------|-------------|
| Event Proximity | Days to next material catalyst |
| Event Materiality | Binary vs. incremental readouts |
| Decay Function | Exponential decay (τ = 30 days) |
| FDA Designations | BTD, Fast Track, Priority Review |

**Decay Formula**: `score × exp(-days / τ)`

### 2.4 Probability of Success (14% base weight)

**Purpose**: Phase-appropriate success probability based on historical rates.

| Phase | Base PoS | Adjustment Factors |
|-------|----------|-------------------|
| Preclinical | 5-10% | Modality, target validation |
| Phase 1 | 15-25% | Safety signals, PK/PD |
| Phase 2 | 25-40% | Efficacy signal, endpoint hit |
| Phase 3 | 50-70% | Prior phase strength |
| Approved | 80-95% | Commercial execution |

### 2.5 Momentum Score (9% base weight)

**Purpose**: Capture institutional positioning and price trends.

| Signal | Window | Description |
|--------|--------|-------------|
| Price Momentum | 60-day | Risk-adjusted returns |
| Volume Trend | 20-day | Accumulation/distribution |
| Smart Money | 13F filings | Institutional flow direction |

**Confidence Gating**: Requires minimum price history; low-confidence signals dampened.

#### 2.5.1 Smart Money: Elite 13F Manager Tracking

The smart money signal tracks 13F filings from **17 elite biotech-focused institutional managers** (~$112B combined AUM), classified into Elite Core and Conditional tiers.

**Selection Criteria**:
- Biotech/healthcare specialist (>50% portfolio in life sciences)
- Long-term track record (10+ years)
- Significant AUM ($1B+ in 13F securities)
- Known for deep scientific/clinical due diligence

**Elite Core — Primary Signal Source** (13 managers, ~$76B AUM, Weight: 1.5×)

| Manager | CIK | AUM ($B) | Style |
|---------|-----|----------|-------|
| Baker Bros Advisors | 0001263508 | 13.8 | Concentrated Conviction |
| RA Capital Management | 0001346824 | 8.1 | Clinical Probability Engine |
| Perceptive Advisors | 0001224962 | 3.5 | Event-Driven |
| Deerfield Management | 0001009258 | 7.2 | Multi-Strategy |
| Ally Bridge Group | 0001822947 | 2.5 | Asia Biotech |
| Redmile Group | 0001425738 | 4.8 | Growth Equity |
| OrbiMed Advisors | 0001055951 | 17.2 | Diversified Healthcare |
| Venbio Partners | 0001776382 | 2.3 | Venture Crossover |
| Bain Capital Life Sciences | 0001703031 | 5.5 | Private Equity Crossover |
| RTW Investments | 0001493215 | 3.2 | Clinical-Stage Specialists |
| Tang Capital Partners | 0001232621 | 2.8 | Oncology Focused |
| Farallon Capital | 0000909661 | 36.0 | Multi-Strategy Macro |
| Suvretta Capital Management | 0001569064 | 5.1 | Healthcare Long/Short |

**Conditional — Secondary Breadth Signal** (4 managers, ~$36B AUM, Weight: 1.0×)

| Manager | CIK | AUM ($B) | Style |
|---------|-----|----------|-------|
| Venrock | 0001005477 | 2.5 | Venture Capital |
| Cormorant Asset Management | 0001398659 | 4.1 | Event-Driven |
| Deep Track Capital | 0001631282 | 1.8 | Fundamental Long/Short |
| Viking Global | 0001103804 | 27.0 | Multi-Strategy Macro |

**Position Change Scoring**:

| Change Type | Score | Description |
|-------------|-------|-------------|
| NEW | +10 pts | Fresh position (wasn't held prior quarter) |
| ADD | +5 pts | Increased position by >10% |
| HOLD | +2 pts | Position unchanged (±10%) |
| TRIM | -3 pts | Decreased position by >10% |
| EXIT | -8 pts | Completely exited position |

**Coordinated Activity Thresholds**:

| Signal | Threshold | Interpretation |
|--------|-----------|----------------|
| Coordinated Buying | ≥3 managers adding | Strong bullish signal |
| Fresh Conviction | ≥2 managers initiating | New opportunity identified |
| Coordinated Selling | ≥3 managers trimming/exiting | Risk signal |
| Crowded Position | ≥6 managers holding | Crowding risk |

**Signal Calculation**:
```
position_score = change_score × tier_weight
ticker_momentum = Σ(position_score for all managers)
smart_money_score = normalize(ticker_momentum, lookback=4 quarters)
```

**Crowding Levels**:
- LOW: 1-2 managers holding
- MODERATE: 3-5 managers holding
- HIGH: 6+ managers holding (crowding risk flag)

**Data Freshness**: 13F filings have 45-day lag from quarter-end. Signal is marked stale if most recent filing is >60 days old.

### 2.6 Short Interest (6% base weight)

**Purpose**: Assess crowding risk and squeeze potential.

| Factor | Description |
|--------|-------------|
| SI % Float | Short interest as % of float |
| Days to Cover | SI / average daily volume |
| Borrow Cost | Implied from options skew |

### 2.7 Valuation Score (5% base weight)

**Purpose**: Constrain extreme narrative valuations (not drive rankings).

| Approach | Application |
|----------|-------------|
| Development-stage | mcap/pipeline, peer relative |
| Commercial-stage | Revenue multiples, cashflow |

**Design Note**: Valuation is intentionally low-weighted. It acts as a constraint on extreme outliers, not a ranking driver.

---

## 3. Scoring Methodology

### 3.1 Pipeline Execution Order

```
1.  Normalize by cohort (early/mid/late)
2.  Apply regime weights (BULL/BEAR/NEUTRAL)
3.  Apply hard regime gating
4.  Apply confidence gating
5.  Compute confidence-weighted contributions
6.  Apply asymmetric transform
7.  Compute weighted contributions
8.  Hybrid aggregation (85% weighted / 15% geometric)
9.  Apply interaction adjustments (±3 pts max)
10. Apply contradiction detector
11. Apply uncertainty penalty (30% max)
12. Apply severity gate
13. Apply monotonic caps
14. Apply dynamic score ceilings
15. Apply existential flaw caps
16. Final adjustments and rounding
```

### 3.2 Key Enhancements

#### Regime Gating
Adjusts component influence based on market regime:

| Regime | Momentum | Financial | Catalyst |
|--------|----------|-----------|----------|
| BEAR | Capped 30% | Penalties +25% | Normal |
| BULL | Full | Penalties -15% | Boosted +15% |
| NEUTRAL | Normal | Normal | Normal |

#### Asymmetric Transform
Implements convex downside / concave upside:
- Upside dampening: +10 → +6 (0.6x)
- Downside amplification: -10 → -12 (1.2x)

#### Existential Flaw Detection
Hard caps for critical risks:
- Runway < 9 months → Score capped at 65
- Binary clinical risk (single asset, early phase) → Score capped at 65

#### Contradiction Detector
Penalizes conflicting signals:
- High momentum + liquidity gate failure → -5 pts
- High valuation + low runway → -3 pts
- High clinical + severe dilution → -4 pts

### 3.3 Aggregation Formula

```
Base Score = 0.85 × Σ(weight_i × normalized_i × confidence_i)
           + 0.15 × Π(normalized_i ^ weight_i)

Final Score = Base Score
            - interaction_penalty
            - contradiction_penalty
            - uncertainty_penalty
            → apply severity gate
            → apply caps and ceilings
```

---

## 4. Regime Detection Methodology

The regime detection system determines the current market environment and adjusts component weights accordingly. This section provides comprehensive documentation of the regime engine architecture.

### 4.1 Regime Definitions

The system recognizes seven distinct market regimes:

| Regime | Description | Primary Triggers |
|--------|-------------|------------------|
| **BULL** | Risk-on environment with positive momentum | VIX < 18, XBI outperforming SPY, positive fund flows |
| **BEAR** | Risk-off environment with negative momentum | VIX > 25, XBI underperforming SPY, negative fund flows |
| **NEUTRAL** | Balanced market conditions | VIX 18-25, mixed signals |
| **VOLATILITY_SPIKE** | Acute volatility event | VIX > 30, VIX rate of change > 25% |
| **SECTOR_ROTATION** | Capital rotating between sectors | XBI vs SPY divergence > 2σ |
| **RECESSION_RISK** | Macro deterioration signals | Yield curve inversion, credit spreads widening |
| **CREDIT_CRISIS** | Systemic credit stress | Credit spreads > 500bps, fund flow collapse |

### 4.2 Input Signals

The regime engine ingests multiple market signals with defined thresholds:

#### 4.2.1 Volatility Signals

| Signal | Source | Thresholds |
|--------|--------|------------|
| VIX Level | CBOE | < 18 (low), 18-25 (normal), > 25 (elevated), > 30 (spike) |
| VIX Rate of Change | Calculated | > 25% daily = spike trigger |
| VIX Term Structure | VIX futures | Contango/backwardation state |
| Biotech Vol (RVOL) | XBI options | Realized vs implied divergence |

#### 4.2.2 Relative Performance Signals

| Signal | Calculation | Interpretation |
|--------|-------------|----------------|
| XBI vs SPY (20-day) | Rolling beta-adjusted return | Sector momentum |
| XBI vs SPY (60-day) | Rolling beta-adjusted return | Sector trend |
| Biotech vs Healthcare | XBI vs XLV relative | Subsector rotation |
| Small vs Large Cap | IWM vs SPY | Risk appetite proxy |

#### 4.2.3 Macro Signals

| Signal | Source | Thresholds |
|--------|--------|------------|
| 2s10s Yield Curve | Treasury rates | < 0 = inversion warning |
| Fed Funds Rate | FRED | Rate change direction |
| Credit Spreads (HY) | ICE BofA HY Index | > 400bps = stress, > 500bps = crisis |
| Investment Grade Spreads | ICE BofA IG Index | > 150bps = stress |

#### 4.2.4 Fund Flow Signals

| Signal | Source | Interpretation |
|--------|--------|----------------|
| Biotech ETF Flows | XBI, IBB, ARKG | 20-day cumulative flow direction |
| Healthcare Sector Flows | XLV | Sector allocation proxy |
| Equity Fund Flows | ICI data | Broad risk appetite |

### 4.3 VIX Kalman Filter

The regime engine applies a Kalman filter to smooth VIX observations and reduce noise:

```
Kalman Filter Parameters:
├── Process Noise (Q):     0.01
├── Measurement Noise (R): 0.1
├── Initial Estimate:      Current VIX
└── Initial Error:         1.0

State Update:
1. Predict: x_pred = x_prev (random walk assumption)
2. Predict error: P_pred = P_prev + Q
3. Kalman gain: K = P_pred / (P_pred + R)
4. Update: x_new = x_pred + K × (measurement - x_pred)
5. Update error: P_new = (1 - K) × P_pred
```

**Purpose**: The Kalman filter prevents regime whipsaw on single-day VIX spikes by smoothing the signal while remaining responsive to genuine regime shifts.

**Smoothing Factor**: Effective smoothing of ~0.9 (90% prior, 10% new observation), resulting in 3-5 day lag for regime transitions.

### 4.4 Hidden Markov Model (HMM)

The regime engine uses a Hidden Markov Model to estimate regime probabilities:

#### 4.4.1 State Space

```
States = {BULL, BEAR, NEUTRAL, VOLATILITY_SPIKE,
          SECTOR_ROTATION, RECESSION_RISK, CREDIT_CRISIS}
```

#### 4.4.2 Transition Probability Matrix

The HMM uses empirically-derived transition probabilities:

```
                   To:
           BULL  BEAR  NEUT  VSPK  SROT  RESK  CRIS
From:     ┌─────────────────────────────────────────┐
BULL      │ 0.85  0.03  0.08  0.02  0.01  0.01  0.00│
BEAR      │ 0.03  0.82  0.10  0.03  0.01  0.01  0.00│
NEUTRAL   │ 0.15  0.15  0.60  0.05  0.03  0.02  0.00│
VOL_SPIKE │ 0.05  0.25  0.20  0.45  0.03  0.02  0.00│
SECT_ROT  │ 0.20  0.15  0.40  0.05  0.18  0.02  0.00│
RECESS    │ 0.02  0.15  0.10  0.08  0.05  0.55  0.05│
CREDIT    │ 0.01  0.10  0.05  0.10  0.02  0.12  0.60│
          └─────────────────────────────────────────┘
```

**Key Properties**:
- High diagonal values (persistence): Regimes tend to persist
- NEUTRAL acts as transition hub (highest off-diagonal mass)
- Crisis regimes (RECESSION_RISK, CREDIT_CRISIS) are sticky once entered
- VOLATILITY_SPIKE is transient (low diagonal)

#### 4.4.3 Emission Probabilities

Each regime has associated emission distributions for observable signals:

| Regime | VIX Mean | XBI-SPY Mean | Credit Spread Mean |
|--------|----------|--------------|-------------------|
| BULL | 15 ± 3 | +0.5% ± 1% | 350bps ± 50 |
| BEAR | 28 ± 5 | -1.5% ± 2% | 450bps ± 75 |
| NEUTRAL | 20 ± 4 | 0% ± 1.5% | 380bps ± 60 |
| VOL_SPIKE | 35 ± 8 | -2% ± 3% | 500bps ± 100 |
| RECESSION | 25 ± 6 | -1% ± 2% | 480bps ± 80 |
| CREDIT_CRISIS | 40 ± 10 | -3% ± 4% | 600bps ± 150 |

### 4.5 Ensemble Classification

The regime engine uses ensemble classification combining multiple methods:

#### 4.5.1 Score-Based Classifier

Deterministic rules based on signal thresholds:

```python
def score_based_regime(signals):
    vix = signals["vix_smoothed"]
    xbi_rel = signals["xbi_vs_spy_20d"]
    credit = signals["hy_spread"]

    # Crisis detection (highest priority)
    if credit > 500 and vix > 30:
        return "CREDIT_CRISIS"

    # Volatility spike detection
    if vix > 30 and signals["vix_roc"] > 0.25:
        return "VOLATILITY_SPIKE"

    # Recession risk detection
    if signals["yield_curve"] < 0 and credit > 400:
        return "RECESSION_RISK"

    # Bull/Bear/Neutral classification
    bull_score = sum([
        vix < 18,
        xbi_rel > 0.01,
        signals["fund_flows"] > 0,
        credit < 380,
    ])

    if bull_score >= 3:
        return "BULL"
    elif bull_score <= 1:
        return "BEAR"
    else:
        return "NEUTRAL"
```

#### 4.5.2 Ensemble Combination

The final regime is determined by combining classifiers:

```
Final Regime = weighted_vote([
    (score_based_regime, weight=0.4),
    (hmm_viterbi_regime, weight=0.35),
    (hmm_forward_regime, weight=0.25),
])
```

**Confidence Score**: The ensemble also outputs a confidence score (0-1) based on classifier agreement:
- All three agree: confidence = 1.0
- Two agree: confidence = 0.7
- All disagree: confidence = 0.4 (defaults to NEUTRAL)

### 4.6 Staleness Gating

The regime engine implements staleness checks on input data:

#### 4.6.1 Staleness Thresholds

| Signal Category | Max Age | Fallback Behavior |
|-----------------|---------|-------------------|
| VIX | 1 trading day | Use last known + uncertainty haircut |
| XBI/SPY prices | 1 trading day | Use last known + uncertainty haircut |
| Credit spreads | 2 trading days | Use last known + 10% confidence reduction |
| Fund flows | 5 trading days | Ignore signal, rely on others |
| Fed rates | 30 days | Use last known (slow-moving) |

#### 4.6.2 Confidence Haircuts

When data is stale, confidence is reduced:

```
haircut = min(0.3, staleness_days × 0.05)
adjusted_confidence = base_confidence × (1 - haircut)
```

### 4.7 Signal Weight Adjustments by Regime

Once the regime is detected, component weights are adjusted:

#### 4.7.1 BEAR Regime Adjustments

```python
BEAR_ADJUSTMENTS = {
    "momentum": {
        "weight_multiplier": Decimal("0.6"),    # Reduce momentum influence
        "max_contribution_cap": Decimal("30"),   # Cap at 30% of normal
    },
    "financial": {
        "severity_multiplier": Decimal("1.25"), # Amplify runway penalties
        "liquidity_threshold_mult": Decimal("1.5"),  # Tighter liquidity
    },
    "catalyst": {
        "weight_multiplier": Decimal("1.0"),    # Unchanged
    },
    "valuation": {
        "upside_cap": Decimal("55"),            # Cap optimistic valuations
    },
}
```

#### 4.7.2 BULL Regime Adjustments

```python
BULL_ADJUSTMENTS = {
    "momentum": {
        "weight_multiplier": Decimal("1.0"),    # Full momentum
    },
    "financial": {
        "severity_multiplier": Decimal("0.85"), # Soften penalties
    },
    "catalyst": {
        "weight_multiplier": Decimal("1.15"),   # Boost catalysts
    },
    "valuation": {
        "upside_cap": Decimal("100"),           # No cap
    },
}
```

#### 4.7.3 VOLATILITY_SPIKE Adjustments

```python
VOLATILITY_SPIKE_ADJUSTMENTS = {
    "momentum": {
        "weight_multiplier": Decimal("0.3"),    # Heavy dampening
        "confidence_floor": Decimal("0.5"),     # Force low confidence
    },
    "financial": {
        "severity_multiplier": Decimal("1.5"),  # Heavy penalty amplification
    },
    "catalyst": {
        "weight_multiplier": Decimal("0.8"),    # Slightly reduce
    },
    "short_interest": {
        "weight_multiplier": Decimal("1.3"),    # Squeeze risk elevated
    },
}
```

### 4.8 Momentum Health Kill Switch

The regime engine includes a kill switch for momentum when signal health degrades:

#### 4.8.1 Information Coefficient (IC) Monitoring

```python
def check_momentum_health(momentum_signals, lookback=60):
    """
    Calculate rolling IC between momentum signal and forward returns.
    Kill switch triggers if IC drops below threshold.
    """
    ic = calculate_ic(
        momentum_signals["score"],
        momentum_signals["forward_5d_return"],
        lookback=lookback
    )

    if ic < IC_KILL_THRESHOLD:  # -0.05
        return MomentumHealth.KILLED
    elif ic < IC_WARNING_THRESHOLD:  # 0.02
        return MomentumHealth.DEGRADED
    else:
        return MomentumHealth.HEALTHY
```

#### 4.8.2 Kill Switch Effects

| Health State | Momentum Weight | Flag |
|--------------|-----------------|------|
| HEALTHY | Base weight | None |
| DEGRADED | Base × 0.5 | `momentum_ic_degraded` |
| KILLED | 0 (redistributed) | `momentum_ic_killed` |

When momentum is killed, its weight is redistributed pro-rata to Clinical (50%) and Financial (50%).

### 4.9 Regime Output Specification

The regime engine outputs a structured result:

```json
{
  "regime": "BEAR",
  "confidence": 0.85,
  "signals": {
    "vix_raw": 27.5,
    "vix_smoothed": 26.2,
    "xbi_vs_spy_20d": -0.012,
    "xbi_vs_spy_60d": -0.025,
    "hy_spread": 425,
    "yield_curve_2s10s": -0.15,
    "fund_flows_20d": -250000000
  },
  "classifier_votes": {
    "score_based": "BEAR",
    "hmm_viterbi": "BEAR",
    "hmm_forward": "NEUTRAL"
  },
  "staleness_flags": [],
  "adjustments_applied": {
    "momentum_weight_mult": 0.6,
    "financial_severity_mult": 1.25,
    "valuation_upside_cap": 55
  },
  "momentum_health": "HEALTHY",
  "as_of_date": "2026-01-28T16:00:00Z"
}
```

### 4.10 Regime Transition Logging

All regime transitions are logged for audit:

```
[2026-01-15 16:00:00] REGIME_TRANSITION: NEUTRAL → BEAR
  - Trigger: VIX crossed 25 threshold (25.3)
  - Confidence: 0.78
  - Duration in prior regime: 12 days
  - Classifier agreement: 2/3

[2026-01-22 16:00:00] REGIME_TRANSITION: BEAR → BEAR (reconfirmed)
  - Confidence increased: 0.78 → 0.91
  - VIX: 28.1, XBI-SPY: -1.8%
```

### 4.11 Backtesting Considerations

The regime engine is designed for PIT-safe backtesting:

1. **No Look-Ahead**: All signals use as-of-date data only
2. **Staleness Simulation**: Historical staleness patterns replicated
3. **Transition Lag**: 1-day lag enforced between signal observation and regime change
4. **HMM Warm-Up**: Requires 30 days of history before regime output is valid

---

## 5. Output Specification

### 5.1 Ranked Securities Output

| Field | Type | Description |
|-------|------|-------------|
| ticker | string | Security identifier |
| composite_rank | int | 1 = best |
| composite_score | decimal | 0-100 scale |
| score_breakdown | object | Component details |
| flags | array | All applied adjustments |
| effective_weights | object | Final component weights |
| regime | string | Active regime at scoring time |
| regime_confidence | decimal | Regime classification confidence |

### 5.2 Score Breakdown Structure

```json
{
  "components": [
    {
      "name": "clinical",
      "raw": 75.26,
      "normalized": 79.08,
      "contribution": 21.82,
      "confidence": 0.95,
      "weight_effective": 0.2859
    }
  ],
  "regime_info": {
    "regime": "BEAR",
    "confidence": 0.85,
    "adjustments": ["momentum_gated", "financial_amplified"]
  },
  "final": {
    "pre_penalty_score": 68.01,
    "post_cap_score": 68.01,
    "composite_score": 68.25
  }
}
```

### 5.3 Diagnostic Flags

Flags provide full transparency on score adjustments:

| Flag Category | Examples |
|---------------|----------|
| Severity | `sev2_penalty_applied`, `sev3_gated` |
| Caps | `liquidity_cap_applied`, `runway_cap_applied` |
| Ceilings | `stage_ceiling_preclinical`, `no_catalyst_12mo_ceiling` |
| Existential | `existential_runway`, `existential_binary_clinical_risk` |
| Asymmetric | `clinical_asymmetric_upside_dampened` |
| Contradiction | `momentum_liquidity_conflict` |
| Regime | `regime_bear_active`, `regime_momentum_gated`, `regime_financial_amplified` |
| Momentum Health | `momentum_ic_degraded`, `momentum_ic_killed` |
| Staleness | `vix_stale_haircut`, `credit_spread_stale` |

---

## 6. Expected Behavior

### 6.1 Distribution Characteristics

| Metric | Expected | Rationale |
|--------|----------|-----------|
| Mean score | < 45 | Conservative baseline |
| % above 60 | < 15% | Hard to impress |
| % below 40 | > 45% | Easy to disappoint |
| Max score | < 80 | Ceiling enforcement |

### 6.2 Acceptable Failure Modes

These behaviors are **by design**, not bugs:

1. **Great companies ranked 50-100**: Screener optimizes for incremental alpha, not company quality
2. **High momentum % in tail**: Small denominator effect in low-score securities
3. **Zero financial contribution (~15%)**: SEV2/SEV3 severity gating working correctly
4. **Sparse data → low scores**: Confidence dampening on uncertain signals

### 6.3 Red Flags (Investigate Immediately)

- Mean score > 50
- Any score > 85
- SEV3 in top 50 ranks
- Phase 1 names dominating top 20
- Hash mismatch on identical inputs
- Regime stuck for > 90 days without signal support
- Momentum IC negative for > 30 days

---

## 7. Testing Framework

### 7.1 Test Coverage

| Test Type | Count | Description |
|-----------|-------|-------------|
| Unit tests | 143 | Individual function validation |
| Integration | 12 | Cross-module data flow |
| Determinism | 8 | Reproducibility verification |
| Invariant | 15 | Behavioral constraint checks |
| Regime tests | 18 | Regime detection and gating |

### 7.2 Regime-Specific Tests

| Test | Description |
|------|-------------|
| HMM transition | Verify transition probabilities sum to 1 |
| Kalman convergence | VIX smoother converges in < 10 iterations |
| Staleness handling | Stale data reduces confidence appropriately |
| BEAR gating | Momentum capped to 30% in BEAR regime |
| BULL boost | Catalyst boosted 15% in BULL regime |
| Kill switch | Momentum zeroed when IC < -0.05 |

### 7.3 Sanity Check Framework

Nine-point validation performed on every production run:

1. **Distribution Sanity**: Score shape validation
2. **Regime Behavior**: Correct gating per regime
3. **Weakest-Link**: Existential flaws penalized
4. **Financial M2**: Friction not propulsion
5. **Valuation**: Not dominating rankings
6. **Confidence**: Sparse data dampened
7. **Rank Stability**: No whipsaw behavior
8. **Feel Right**: Known names explainable
9. **Health Check**: Conservative, explicit

---

## 8. Governance

### 8.1 Change Control

| Class | Description | Approval |
|-------|-------------|----------|
| A | Weight changes > ±2% | IC + Quant Lead |
| B | New enhancement/cap | IC + Quant Lead |
| C | Threshold adjustment | Quant Lead |
| D | Bug fix (invariant-preserving) | Quant Lead |
| E | Documentation/logging | Self-approve |
| R | Regime threshold changes | IC + Quant Lead |

### 8.2 Prohibited Changes

Without explicit IC override:
- Removing severity gates
- Increasing max score ceiling above 80
- Removing existential flaw detection
- Changing to float arithmetic
- Allowing valuation to exceed 10% weight
- Removing regime gating mechanisms
- Disabling momentum kill switch
- Modifying HMM transition probabilities without backtest

### 8.3 Monitoring Requirements

| Frequency | Metrics |
|-----------|---------|
| Daily | Mean drift, top 10 composition, invariants, regime state |
| Weekly | Distribution shape, severity effectiveness, regime transitions |
| Monthly | Post-catalyst attribution, known-name review, IC health |
| Quarterly | HMM calibration review, regime threshold validation |

---

## 9. Limitations & Assumptions

### 9.1 Known Limitations

1. **Data Dependency**: Quality bounded by source data (SEC, trial registries)
2. **Biotech-Specific**: Not applicable to other healthcare subsectors
3. **US-Centric**: ADR/foreign listings may have data gaps
4. **Backward-Looking**: Financial data has reporting lag
5. **Regime Lag**: 1-3 day lag in regime detection due to smoothing
6. **HMM Cold Start**: Requires 30-day warm-up for stable regime probabilities

### 9.2 Key Assumptions

1. Historical phase transition rates are predictive
2. Cash burn rates are relatively stable near-term
3. Catalyst dates from public sources are accurate
4. Market regime classification is correct
5. VIX is a valid proxy for biotech risk environment
6. HMM transition probabilities remain stable over time

### 9.3 Regime-Specific Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| VIX-centric | May miss biotech-specific stress | XBI relative performance as secondary signal |
| US market hours | Non-US events may lag | Pre-market VIX futures monitoring |
| Credit spread staleness | Corporate bond data delayed | 2-day tolerance with confidence haircut |
| Rapid regime shifts | Kalman smoother introduces lag | VIX ROC override for spike detection |

### 9.4 Not Designed For

- Intraday trading signals
- Options strategy construction
- Macro/sector timing
- Position sizing (separate module)
- Non-biotech healthcare (pharma, devices, services)

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-28 | Initial production release |
| 1.0.1 | 2026-01-28 | Expanded regime detection documentation |
| 1.0.2 | 2026-01-28 | Added 13F elite manager tracking (17 managers incl. Suvretta) |

---

## 11. Contact & Support

**Model Owner**: Wake Robin Capital Management
**Documentation**: `/docs/MODEL_DOCUMENTATION.md`
**Definition of Done**: `/docs/MODULE_5_DEFINITION_OF_DONE.md`
**Source Code**: Private repository

---

*This document is intended for internal use and should accompany any model output shared for investment committee review.*

**Document Hash**: Generated on demand
**Last Validated**: 2026-01-28
