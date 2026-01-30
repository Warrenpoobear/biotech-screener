# Defensive Overlay Spec v1.0

**Status:** Production default
**Scope:** Risk control overlay applied after signal aggregation
**Applies to:** Ranking, research outputs, and downstream portfolio construction
**Design goal:** Shape tails, preserve signal integrity, remain deterministic

---

## 1. Purpose & Philosophy

The Defensive Overlay is a **risk-control layer**, not an alpha engine.

It is designed to:

- Reduce tail risk
- Penalize structurally fragile names
- Modestly reward *high-quality* defensive profiles
- Preserve cross-sectional ranking integrity
- Remain stable, auditable, and point-in-time safe

The overlay **does not**:

- Predict returns
- Replace alpha signals
- Perform portfolio construction (unless explicitly enabled)

---

## 2. Price Data Requirements (Authoritative)

### 2.1 Price Type

All calculations **must use adjusted close prices**.

**Rationale**

- Adjusted prices correctly account for:
  - Stock splits
  - Dividends
  - Corporate actions
- Unadjusted closes introduce **false volatility and drawdowns**

### 2.2 Required Fields

Minimum required schema:

```
date, ticker, adj_close
```

Optional (stored but not required for overlay):

```
volume
open, high, low
```

### 2.3 History Stored

- **2–3 years of daily adjusted closes** per ticker
- History stored rolling forward
- All calculations slice trailing windows as of `as_of_date`

### 2.4 Benchmarks

- **XBI** – primary biotech risk anchor
- Optional future: SPY (macro diagnostics only)

---

## 3. Return Construction

All returns are:

```
daily log returns = ln(adj_close_t / adj_close_{t-1})
```

Rules:

- Calendar-aligned on overlapping dates only
- Missing days are dropped (no forward/backward filling)
- If insufficient overlapping observations exist → metric is skipped and flagged

---

## 4. Core Risk Metrics (Exact Definitions)

### 4.1 Volatility (Primary Risk Magnitude)

**Metric**

```
realized_volatility_63d
```

**Definition**

```
σ = sqrt(252) × stdev(log_returns[-63:])
```

**Notes**

- 63 trading days ≈ 3 months (institutional convention)
- Long enough to smooth noise
- Responsive to regime changes
- Used as the primary volatility proxy

**Current implementation:** `vol_60d` (functionally equivalent)

---

### 4.2 Drawdown (Path Dependency)

**Metric (Primary)**

```
current_drawdown_63d
```

**Definition**

```
current_drawdown = (current_price / max_price_last_63d) - 1
```

**Interpretation**

- Measures *current damage*
- Answers: "Is this name already impaired?"

**Optional Diagnostic (Future)**

```
max_drawdown_63d
```

*Reported only; not required for v1 behavior*

**Current implementation:** `drawdown_current` (functionally equivalent)

---

### 4.3 Correlation to XBI (Secondary)

**Metric**

```
corr_xbi_126d
```

**Definition**

```
corr(log_returns_stock[-126:], log_returns_XBI[-126:])
```

**Rules**

- Requires ≥ 126 overlapping trading days
- Uses demeaned returns
- If unavailable → omitted and explicitly flagged

**Current implementation:** `corr_xbi_120d` (functionally equivalent)

---

### 4.4 Beta to XBI (Secondary)

**Metric**

```
beta_xbi_63d
```

**Definition**

```
cov(stock, XBI) / var(XBI)
```

**Notes**

- Diagnostic only
- Not required for defensive eligibility

**Current implementation:** `beta_xbi_60d`

---

## 5. Clustering (Exposure Control)

### 5.1 Default Clustering (Always On)

**Purpose**

- Control exposure concentration
- Provide risk attribution
- Support diversification analysis

**Method**
Categorical / fundamental clustering

**Hierarchical Inputs (in order)**

1. Indication
2. Stage bucket
3. Market cap bucket

**Output**

```
cluster_id  (always populated)
```

**Important Clarification**

> `cluster_id` represents an **exposure cohort**, not a correlation cluster.

This distinction is explicit and intentional.

---

### 5.2 Correlation-Based Clustering (Optional Mode)

**Status**

- Disabled by default

**Spec (if enabled)**

- 252 trading days preferred (minimum 126)
- Demeaned daily log returns
- Hierarchical clustering (average linkage)
- Deterministic ordering

Used only for portfolio construction research, not default ranking.

---

## 6. Defensive Classification Logic

Each security is assigned a **defensive bucket**:

| Bucket | Description |
|--------|-------------|
| `elite` | Low vol, shallow drawdown, low XBI correlation |
| `good` | Good diversifier, moderate metrics |
| `neutral` | Default bucket (majority of universe) |
| `penalty` | High vol and/or deep drawdown |

### Inputs

- Volatility (63d)
- Drawdown (63d)
- Correlation to XBI (if available)
- Cluster-relative comparisons (where applicable)

### Thresholds

| Condition | Threshold | Multiplier |
|-----------|-----------|------------|
| Elite diversifier | corr < 0.30, vol < 0.40 | 1.40 |
| Good diversifier | corr < 0.40, vol < 0.50 | 1.10 |
| High correlation | corr > 0.80 | 0.95 |
| High volatility | vol > 0.80 | 0.97 |
| Deep drawdown | drawdown < -0.40 | 0.92 |
| Momentum bonus | ret_21d > 0.10 | 1.05 |
| Momentum penalty | ret_21d < -0.20 | 0.95 |
| RSI oversold | RSI < 30 | 1.03 |
| RSI overbought | RSI > 70 | 0.98 |

---

## 7. Quality Gate (Critical Control)

### 7.1 Boost Eligibility Rule

Defensive **boosts are conditional**.

```
Elite status requires:
  pre_defensive_composite_score >= quality_threshold
```

**Default threshold:** 50

**Rationale**

- Prevents low-quality, low-volatility names from being promoted
- Preserves alpha integrity
- Aligns with institutional diversification practice

### 7.2 Penalties

- Penalties may apply regardless of quality
- Risk control supersedes signal strength on the downside

### 7.3 Gate Notes

When a security is boost-gated, the output includes:

```
def_boost_gated_below_<threshold>
```

---

## 8. Defensive Multiplier Application

### 8.1 Formula

```
adjusted_composite_score = raw_composite_score × defensive_multiplier
```

### 8.2 Multiplier Bounds

```
mult_floor = 0.75
mult_ceiling = 1.60
```

### 8.3 Properties

- Mean multiplier ≈ 1.0
- Bounded (no extreme amplification)
- Most names remain neutral
- Tail reshaping only

### 8.4 Ordering Guarantee

```
expected_excess_return → defensive_multiplier → final score
```

Risk adjustment never contaminates signal estimation.

---

## 9. Required Output Columns (Always Emitted)

Every run must emit:

| Column | Description |
|--------|-------------|
| `composite_score` | Final risk-adjusted score |
| `z_score` | Cross-sectional standardized score |
| `expected_excess_return` | Linear mapping from z_score |
| `volatility` | 63d realized volatility |
| `drawdown` | Current drawdown from 63d high |
| `cluster_id` | Exposure cohort identifier |
| `defensive_multiplier` | Applied multiplier |
| `defensive_bucket` | Classification (elite/good/neutral/penalty) |
| `defensive_notes` | Audit trail of applied adjustments |

---

## 10. Metadata & Diagnostics

Each output includes explicit provenance:

```json
"defensive_overlay_config": {
  "defensive_overlay_applied": true,
  "portfolio_weights_computed": false,
  "boost_eligibility_threshold": "50",
  "enabled_factors": {
    "correlation": true,
    "volatility": true,
    "momentum": true,
    "rsi": true,
    "drawdown": true,
    "boost_eligibility_gate": true
  }
}
```

Partial data cases (e.g., IPOs) must include:

- Missing metrics
- Explicit skip reasons (e.g., `def_cache_ipo_insufficient_history`)

---

## 11. Non-Goals (Explicit)

The Defensive Overlay does **not** include:

- Intraday data
- GARCH / EWMA volatility
- ML-based clustering
- Volatility-scaled expected returns
- Implicit portfolio construction

---

## 12. Summary

This Defensive Overlay:

- Is institutionally standard
- Reshapes tails without distorting the universe
- Separates alpha from risk cleanly
- Is deterministic, PIT-safe, and auditable
- Is suitable as a **default research artifact**

**Spec v1.0 is production-locked.**

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial production spec with boost eligibility gate |
