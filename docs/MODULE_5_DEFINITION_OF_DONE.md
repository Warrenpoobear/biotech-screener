# Module 5 Composite Scoring — Definition of Done

**Document Version**: 1.0.0
**Status**: LOCKED
**Effective Date**: 2026-01-28
**Owner**: Wake Robin Capital Management
**Last IC Sign-Off**: 2026-01-28

---

## 1. Purpose

This document defines the **completion criteria, behavioral invariants, and change control requirements** for Module 5 (Composite Scoring).

Once signed off, Module 5 is considered **production-locked**. Any modifications require explicit governance approval and must satisfy the change control process defined herein.

> **Design Philosophy**: The composite score should feel conservative, hard to impress, easy to disappoint, and very explicit about why.

---

## 2. Scoring Pipeline Specification (LOCKED)

### 2.1 Pipeline Stages (Execution Order)

```
1.  Normalize by cohort (early/mid/late)
2.  Apply regime weights (BULL/BEAR/NEUTRAL/SECTOR_ROTATION)
3.  Apply hard regime gating
4.  Apply confidence gating
5.  Compute confidence-weighted contributions
6.  Apply asymmetric transform (convex downside, concave upside)
7.  Compute weighted contributions
8.  Hybrid aggregation (85% weighted mean / 15% geometric mean)
9.  Apply interaction adjustments (±3 pts max)
10. Apply contradiction detector penalties
11. Apply uncertainty penalty (30% max)
12. Apply severity gate
13. Apply monotonic caps (liquidity, runway, dilution)
14. Apply dynamic score ceilings (stage, catalyst, commercial)
15. Apply existential flaw caps
16. Final adjustments and rounding
```

### 2.2 Component Weights (Base)

| Component | Base Weight | Notes |
|-----------|-------------|-------|
| Clinical | 26% | Pipeline strength, trial quality |
| Financial | 24% | Runway, dilution, liquidity |
| Catalyst | 16% | Event proximity, decay-adjusted |
| PoS | 14% | Probability of success |
| Momentum | 9% | Price/volume signals |
| Short Interest | 6% | Crowding, squeeze potential |
| Valuation | 5% | Narrative constraint only |

**Total**: 100%

### 2.3 Enhancement Configurations (LOCKED)

#### Regime Gating
```python
REGIME_GATE_CONFIG = {
    "BEAR": {
        "momentum_cap_pct": Decimal("0.30"),
        "valuation_upside_cap": Decimal("55"),
        "financial_penalty_mult": Decimal("1.25"),
    },
    "BULL": {
        "momentum_cap_pct": Decimal("1.0"),
        "catalyst_boost": Decimal("1.15"),
        "financial_penalty_mult": Decimal("0.85"),
    },
    "NEUTRAL": {},
}
```

#### Existential Flaw Detection
```python
EXISTENTIAL_FLAW_CONFIG = {
    "runway_critical_months": Decimal("9"),
    "binary_clinical_risk_phases": ["phase_1", "phase_2"],
    "existential_cap": Decimal("65"),
}
```

#### Asymmetric Transform
```python
ASYMMETRY_CONFIG = {
    "upside_dampening": Decimal("0.6"),    # +10 → +6
    "downside_amplification": Decimal("1.2"), # -10 → -12
    "neutral_threshold": Decimal("50"),
}
```

#### Dynamic Ceilings
```python
STAGE_CEILING_CONFIG = {
    "preclinical": Decimal("65"),
    "phase_1": Decimal("70"),
}

CATALYST_CEILING_CONFIG = {
    "no_catalyst_12mo": Decimal("75"),
    "no_catalyst_6mo": Decimal("70"),
}
```

---

## 3. Behavioral Invariants (MUST HOLD)

These invariants define correct system behavior. Violation of any invariant constitutes a **critical defect**.

### 3.1 Distribution Invariants

| Invariant | Threshold | Rationale |
|-----------|-----------|-----------|
| Mean composite score | < 45 | Conservative baseline |
| % above 60 | < 15% | Hard to impress |
| % below 40 | > 45% | Easy to disappoint |
| Max composite score | < 80 | Ceiling enforcement |
| Min composite score | > 0 | Floor enforcement |

### 3.2 Severity Gate Invariants

| Invariant | Condition |
|-----------|-----------|
| No SEV3 in top 50 | `runway < 6mo` → `rank > 50` |
| No SEV2 in top 25 | `runway < 12mo` → `rank > 25` |
| Existential cap enforced | `existential_flag` → `score ≤ 65` |

### 3.3 Component Contribution Invariants

| Invariant | Condition |
|-----------|-----------|
| Financial never propels | Financial contribution ≤ 22 pts |
| Valuation never dominates | Valuation contribution ≤ 5 pts |
| Momentum bounded | Momentum contribution ≤ 8 pts in top 50 |
| Clinical anchors | Clinical weight ≥ 20% effective |

### 3.4 Determinism Invariants

| Invariant | Condition |
|-----------|-----------|
| Same inputs → same outputs | Hash-verifiable |
| No datetime.now() | PIT-safe |
| No random calls | Reproducible |
| Decimal arithmetic only | No float drift |

---

## 4. Acceptable Failure Modes

These are **expected behaviors** that may appear concerning but are architecturally correct.

### 4.1 High Momentum % in Tail (ACCEPTABLE)

**Observation**: Securities with `rank > 200` may show momentum contribution > 25% of total score.

**Why it's acceptable**:
- Total score denominator is small (< 25 pts)
- Absolute momentum contribution is modest (5-6 pts)
- These names cannot enter investable ranks
- No capital at risk

**Classification**: `ℹ️ Expected tail sensitivity (non-actionable)`

### 4.2 Great Companies Ranked Low (ACCEPTABLE)

**Observation**: Well-known, successful companies (e.g., VRTX, REGN, GILD) may rank in the 50-100 range.

**Why it's acceptable**:
- Screener optimizes for *incremental alpha*, not company quality
- Cash cows with limited upside correctly deprioritized
- Commercial-stage ceiling constraints working as designed

**Classification**: `✓ Correct behavior`

### 4.3 Zero Financial Contribution (ACCEPTABLE)

**Observation**: ~15% of securities have financial contribution = 0.

**Why it's acceptable**:
- SEV2/SEV3 severity gating working correctly
- Missing data handled safely
- Prevents garbage-in propagation

**Classification**: `✓ Correct behavior`

### 4.4 Sparse Data → Low Scores (ACCEPTABLE)

**Observation**: Securities with < 2 trials and/or missing financials score below 20.

**Why it's acceptable**:
- Confidence weighting dampens uncertain signals
- Better to under-rank than over-rank on thin data
- Explicit flags explain the constraint

**Classification**: `✓ Correct behavior`

---

## 5. Unacceptable Failure Modes

These constitute **critical defects** requiring immediate investigation.

### 5.1 Distribution Violations (CRITICAL)

| Failure | Trigger |
|---------|---------|
| Optimism drift | Mean score > 50 |
| Ceiling breach | Any score > 85 |
| Clustering | > 20% of scores within 5-pt band |
| Early-phase dominance | > 3 Phase 1 names in top 20 |

### 5.2 Severity Gate Failures (CRITICAL)

| Failure | Trigger |
|---------|---------|
| SEV3 in top 50 | `runway < 6mo` AND `rank ≤ 50` |
| Existential not capped | `existential_flag` AND `score > 65` |
| Fragile balance sheet ranked high | `runway < 12mo` AND `rank ≤ 20` |

### 5.3 Component Dominance Failures (CRITICAL)

| Failure | Trigger |
|---------|---------|
| Valuation driving rankings | Valuation-rank correlation > 0.5 |
| Single component > 40% | Any component > 40% of score in top 50 |
| Momentum whipsaw | Top 10 rank change > 5 positions day-over-day with no catalyst |

### 5.4 Determinism Failures (CRITICAL)

| Failure | Trigger |
|---------|---------|
| Hash mismatch | Same inputs produce different hashes |
| Non-reproducible | Re-run produces different rankings |

---

## 6. Change Control Requirements

### 6.1 Change Classification

| Class | Description | Approval Required |
|-------|-------------|-------------------|
| **Class A** | Weight changes > ±2% | IC + Quant Lead |
| **Class B** | New enhancement/cap | IC + Quant Lead |
| **Class C** | Threshold adjustment | Quant Lead |
| **Class D** | Bug fix (invariant-preserving) | Quant Lead |
| **Class E** | Documentation/logging | Self-approve |

### 6.2 Change Process

1. **Proposal**: Document rationale, expected impact, rollback plan
2. **Backtest**: Run against 90-day historical data
3. **Invariant Check**: Verify all invariants still hold
4. **Shadow Mode**: Run parallel to production for 5 trading days
5. **Sign-Off**: Obtain required approvals
6. **Deploy**: With monitoring alerts enabled
7. **Document**: Update this Definition of Done

### 6.3 Prohibited Changes

The following changes are **categorically prohibited** without IC override:

- Removing severity gates
- Increasing max score ceiling above 80
- Removing existential flaw detection
- Changing to float arithmetic
- Removing asymmetric transform
- Allowing valuation to exceed 10% weight

---

## 7. Monitoring Requirements

### 7.1 Daily Monitoring

| Metric | Alert Threshold |
|--------|-----------------|
| Mean score drift | ±3 pts from baseline |
| Top 10 composition change | > 3 new names |
| Invariant violations | Any |
| Hash verification | Any mismatch |

### 7.2 Weekly Monitoring

| Metric | Review Threshold |
|--------|------------------|
| Score distribution shape | Visual review |
| Severity gate effectiveness | SEV3 rank distribution |
| Component contribution stability | Weight drift > 1% |
| Top-decile churn | > 30% turnover |

### 7.3 Monthly Monitoring

| Metric | Review Threshold |
|--------|------------------|
| Post-catalyst attribution | Hit rate vs baseline |
| Regime behavior validation | Correct gating in regime changes |
| Known-name sanity check | Manual IC review |

---

## 8. Current Production Baseline

As of **2026-01-28**, the following metrics define the healthy baseline:

### 8.1 Distribution Baseline

```
Mean Score:     33.9
Median Score:   32.9
Std Dev:        15.2
Min Score:      3.3
Max Score:      68.3

Above 60:       6.0%
Above 50:       17.3%
Below 40:       64.5%
Below 20:       12.1%
```

### 8.2 Component Contribution Baseline

```
Clinical:       Mean 18.5 pts (range 0-22)
Financial:      Mean 10.0 pts (range 0-21)
Catalyst:       Mean 4.2 pts (range 0-7)
Momentum:       Mean 3.5 pts (range 0-6)
Valuation:      Mean 1.8 pts (range 0-4)
Short Interest: Mean 3.2 pts (range 0-5)
PoS:            Mean 5.2 pts (range 0-6)
```

### 8.3 Severity Distribution Baseline

```
none:   70.9%
sev1:   12.1%
sev2:   12.1%
sev3:   4.9%
```

### 8.4 Flag Coverage Baseline

```
Average flags per security: 17.5
Existential flags:          31 securities
Asymmetric dampening:       ~95% of top 50
```

---

## 9. Sanity Check Checklist

Before any release, the following sanity checks must pass:

- [ ] **Distribution**: Mean < 45, < 15% above 60, > 45% below 40
- [ ] **Regime**: Momentum appropriately gated for current regime
- [ ] **Weakest-Link**: Low-runway names (< 9mo) avg rank > 200
- [ ] **Financial**: 10-20% zeroed, lumpy penalties, profitable not penalized
- [ ] **Valuation**: Avg contribution < 3 pts, not driving rankings
- [ ] **Confidence**: Sparse data → low scores, no sparse + extreme combos
- [ ] **Stability**: Top 10 spread > 5 pts, no single component > 35%
- [ ] **Feel Right**: Known names explainable to IC
- [ ] **Health**: Conservative, hard to impress, easy to disappoint, explicit

---

## 10. Sign-Off

### Production Lock Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Quant Lead | | 2026-01-28 | |
| IC Chair | | 2026-01-28 | |
| Risk Officer | | 2026-01-28 | |

### Certification

By signing above, the approvers certify that:

1. Module 5 meets all behavioral invariants
2. The scoring pipeline is production-ready
3. Monitoring systems are in place
4. Change control process is understood and accepted

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                 MODULE 5 QUICK REFERENCE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HEALTHY DISTRIBUTION                                       │
│  ├─ Mean: ~34 (conservative)                                │
│  ├─ Above 60: <10% (hard to impress)                        │
│  ├─ Below 40: >50% (easy to disappoint)                     │
│  └─ Max: ~68 (capped)                                       │
│                                                             │
│  COMPONENT WEIGHTS                                          │
│  ├─ Clinical:    26% (anchor)                               │
│  ├─ Financial:   24% (friction, not propulsion)             │
│  ├─ Catalyst:    16% (event-driven)                         │
│  ├─ PoS:         14% (probability gate)                     │
│  ├─ Momentum:     9% (bounded signal)                       │
│  ├─ Short Int:    6% (crowding risk)                        │
│  └─ Valuation:    5% (narrative constraint only)            │
│                                                             │
│  KEY GATES                                                  │
│  ├─ SEV3 (runway <6mo):  Cannot rank top 50                 │
│  ├─ SEV2 (runway <12mo): Cannot rank top 25                 │
│  ├─ Existential flaw:    Capped at 65                       │
│  └─ Stage ceiling:       Preclinical capped at 65           │
│                                                             │
│  RED FLAGS (investigate immediately)                        │
│  ├─ Mean score > 50                                         │
│  ├─ Any score > 85                                          │
│  ├─ SEV3 in top 50                                          │
│  ├─ Phase 1 dominating top 20                               │
│  └─ Hash mismatch on re-run                                 │
│                                                             │
│  ACCEPTABLE (not bugs)                                      │
│  ├─ High momentum % in tail (rank 200+)                     │
│  ├─ Great companies ranked 50-100                           │
│  ├─ 15% of securities with zero financial contrib           │
│  └─ Sparse data → low scores                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-28 | Wake Robin | Initial production lock |

---

*End of Document*
