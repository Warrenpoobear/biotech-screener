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

## 4. Output Specification

### 4.1 Ranked Securities Output

| Field | Type | Description |
|-------|------|-------------|
| ticker | string | Security identifier |
| composite_rank | int | 1 = best |
| composite_score | decimal | 0-100 scale |
| score_breakdown | object | Component details |
| flags | array | All applied adjustments |
| effective_weights | object | Final component weights |

### 4.2 Score Breakdown Structure

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
  "final": {
    "pre_penalty_score": 68.01,
    "post_cap_score": 68.01,
    "composite_score": 68.25
  }
}
```

### 4.3 Diagnostic Flags

Flags provide full transparency on score adjustments:

| Flag Category | Examples |
|---------------|----------|
| Severity | `sev2_penalty_applied`, `sev3_gated` |
| Caps | `liquidity_cap_applied`, `runway_cap_applied` |
| Ceilings | `stage_ceiling_preclinical`, `no_catalyst_12mo_ceiling` |
| Existential | `existential_runway`, `existential_binary_clinical_risk` |
| Asymmetric | `clinical_asymmetric_upside_dampened` |
| Contradiction | `momentum_liquidity_conflict` |

---

## 5. Expected Behavior

### 5.1 Distribution Characteristics

| Metric | Expected | Rationale |
|--------|----------|-----------|
| Mean score | < 45 | Conservative baseline |
| % above 60 | < 15% | Hard to impress |
| % below 40 | > 45% | Easy to disappoint |
| Max score | < 80 | Ceiling enforcement |

### 5.2 Acceptable Failure Modes

These behaviors are **by design**, not bugs:

1. **Great companies ranked 50-100**: Screener optimizes for incremental alpha, not company quality
2. **High momentum % in tail**: Small denominator effect in low-score securities
3. **Zero financial contribution (~15%)**: SEV2/SEV3 severity gating working correctly
4. **Sparse data → low scores**: Confidence dampening on uncertain signals

### 5.3 Red Flags (Investigate Immediately)

- Mean score > 50
- Any score > 85
- SEV3 in top 50 ranks
- Phase 1 names dominating top 20
- Hash mismatch on identical inputs

---

## 6. Testing Framework

### 6.1 Test Coverage

| Test Type | Count | Description |
|-----------|-------|-------------|
| Unit tests | 143 | Individual function validation |
| Integration | 12 | Cross-module data flow |
| Determinism | 8 | Reproducibility verification |
| Invariant | 15 | Behavioral constraint checks |

### 6.2 Sanity Check Framework

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

## 7. Governance

### 7.1 Change Control

| Class | Description | Approval |
|-------|-------------|----------|
| A | Weight changes > ±2% | IC + Quant Lead |
| B | New enhancement/cap | IC + Quant Lead |
| C | Threshold adjustment | Quant Lead |
| D | Bug fix (invariant-preserving) | Quant Lead |
| E | Documentation/logging | Self-approve |

### 7.2 Prohibited Changes

Without explicit IC override:
- Removing severity gates
- Increasing max score ceiling above 80
- Removing existential flaw detection
- Changing to float arithmetic
- Allowing valuation to exceed 10% weight

### 7.3 Monitoring Requirements

| Frequency | Metrics |
|-----------|---------|
| Daily | Mean drift, top 10 composition, invariants |
| Weekly | Distribution shape, severity effectiveness |
| Monthly | Post-catalyst attribution, known-name review |

---

## 8. Limitations & Assumptions

### 8.1 Known Limitations

1. **Data Dependency**: Quality bounded by source data (SEC, trial registries)
2. **Biotech-Specific**: Not applicable to other healthcare subsectors
3. **US-Centric**: ADR/foreign listings may have data gaps
4. **Backward-Looking**: Financial data has reporting lag

### 8.2 Key Assumptions

1. Historical phase transition rates are predictive
2. Cash burn rates are relatively stable near-term
3. Catalyst dates from public sources are accurate
4. Market regime classification is correct

### 8.3 Not Designed For

- Intraday trading signals
- Options strategy construction
- Macro/sector timing
- Position sizing (separate module)

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-28 | Initial production release |

---

## 10. Contact & Support

**Model Owner**: Wake Robin Capital Management
**Documentation**: `/docs/MODEL_DOCUMENTATION.md`
**Definition of Done**: `/docs/MODULE_5_DEFINITION_OF_DONE.md`
**Source Code**: Private repository

---

*This document is intended for internal use and should accompany any model output shared for investment committee review.*

**Document Hash**: Generated on demand
**Last Validated**: 2026-01-28
