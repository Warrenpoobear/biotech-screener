# Defensive Overlay Spec v1.0

**Purpose**
The Defensive Overlay is a **risk-control layer** applied *after* signal aggregation. Its goal is to **reshape tails, not the center**, by penalizing fragile names and modestly rewarding genuinely defensive ones — without distorting the universe or embedding hidden alpha assumptions.

This overlay is **deterministic, PIT-safe, and auditable**.

---

## 1. Design Principles

1. **Separation of concerns**
   - Alpha signals → determine *expected excess return*
   - Defensive overlay → adjusts *risk exposure*, not signal quality

2. **Stability over cleverness**
   - Long enough windows to suppress noise
   - Deterministic rules, no stochastic clustering

3. **Tail shaping, not ranking domination**
   - Mean multiplier ≈ 1.0
   - Most names remain neutral
   - Only tails are meaningfully affected

4. **Point-in-time safety**
   - All calculations use trailing data as of `as_of_date`
   - No forward-looking data or survivorship leakage

---

## 2. Required Price History

**Frequency**
- Daily only (no intraday)

**Fields**
- `date`
- `ticker`
- `adjusted_close`
- `volume` (optional, for future liquidity screens)

**History Stored**
- ≥ 2 years rolling daily history per ticker

**Benchmarks**
- XBI (biotech beta/correlation anchor)

---

## 3. Core Risk Metrics (Exact Definitions)

All returns are **daily log returns**.

### 3.1 Volatility

**Metric**
```
realized_volatility_60d
```

**Definition**
```
σ = sqrt(252) × stdev(log_returns[-60:])
```

**Notes**
- 60 trading days ≈ 3 months
- Annualized using √252
- Used as the primary risk magnitude proxy

---

### 3.2 Drawdown

**Metric**
```
current_drawdown_60d
```

**Definition**
```
drawdown = (current_price / max_price_last_60d) - 1
```

**Notes**
- Measures *current damage*, not historical worst case
- Captures path dependency ignored by volatility
- Particularly important for biotech financing risk

---

### 3.3 Correlation (Secondary / Optional)

**Metric**
```
corr_xbi_120d
```

**Definition**
```
corr(log_returns_stock[-120:], log_returns_XBI[-120:])
```

**Rules**
- Computed only if ≥ 120 overlapping trading days exist
- Otherwise omitted and explicitly flagged as missing

---

### 3.4 Beta (Secondary)

**Metric**
```
beta_xbi_60d
```

**Definition**
```
cov(stock, XBI) / var(XBI)
```

**Notes**
- Used for diagnostics and diversification awareness
- Not required for defensive eligibility

---

## 4. Clustering (Risk Exposure Control)

### 4.1 Default Clustering (Always On)

**Method**
- Categorical / fundamental clustering

**Inputs (hierarchical preference)**
1. Indication
2. Stage bucket
3. Market cap bucket

**Output**
```
cluster_id (always populated)
```

**Rationale**
- Stable
- Deterministic
- Auditable
- Does not require long return history

Clusters represent **exposure cohorts**, not correlation regimes.

---

### 4.2 Correlation-Based Clustering (Optional)

**Status**
- Disabled by default

**Spec (if enabled)**
- 252 trading days
- Demeaned log returns
- Hierarchical clustering (average linkage)
- No KMeans

Used only for portfolio construction research, not default ranking.

---

## 5. Defensive Classification Logic

Each ticker is assigned a **defensive bucket** based on risk metrics.

### Buckets

| Bucket | Description |
|--------|-------------|
| `elite` | Low vol, shallow drawdown, low XBI correlation |
| `good` | Good diversifier, moderate metrics |
| `neutral` | Default bucket (majority of universe) |
| `penalty` | High vol and/or deep drawdown |

### Eligibility Rules

**Elite**
- Low volatility: `vol_60d < 0.40` (40% annualized)
- Low XBI correlation: `corr_xbi_120d < 0.30`
- *Must pass minimum quality gate* (see §6)

**Good Diversifier**
- Low volatility: `vol_60d < 0.50`
- Moderate correlation: `corr_xbi_120d < 0.40`
- *Must pass minimum quality gate*

**Penalized**
- High volatility: `vol_60d > 0.80`
- High correlation: `corr_xbi_120d > 0.80`
- Deep drawdown: `drawdown < -0.40`

---

## 6. Quality Gate (Critical)

Defensive **boosts** require baseline signal quality.

**Rule**
```
boost eligibility requires:
  pre_defensive_composite_score >= boost_eligibility_threshold
```

**Default threshold:** 50

This prevents low-quality, low-volatility names from being artificially promoted.

**Penalties may apply regardless of quality.**

### Rationale

Without this gate, names like OPK (pre-defensive score 48.61) would receive elite 1.40x boosts purely from low correlation, jumping above fundamentally stronger names.

---

## 7. Defensive Multiplier Application

### Multiplier Formula
```
adjusted_score = raw_composite_score × defensive_multiplier
```

### Multiplier Values

| Condition | Multiplier | Note |
|-----------|------------|------|
| Elite diversifier | 1.40 | Requires quality gate |
| Good diversifier | 1.10 | Requires quality gate |
| Momentum bonus | 1.05 | 21d return > 10% |
| RSI oversold | 1.03 | RSI < 30 |
| RSI overbought | 0.98 | RSI > 70 |
| High correlation | 0.95 | corr > 0.80 |
| High volatility | 0.97 | vol > 0.80 |
| Deep drawdown | 0.92 | drawdown < -40% |
| Momentum penalty | 0.95 | 21d return < -20% |

### Multiplier Bounds
```
mult_floor = 0.75
mult_ceiling = 1.60
```

### Properties
- Mean multiplier ≈ 1.0
- Bounded (no extreme amplification)
- Applied *after* expected excess return is computed

---

## 8. Output Fields (Always Emitted)

These columns are **always present** in output, even if null:

| Column | Description |
|--------|-------------|
| `composite_score` | Final risk-adjusted score |
| `z_score` | Cross-sectional standardized score |
| `expected_excess_return` | Linear mapping from z_score |
| `volatility` | 60d realized volatility |
| `drawdown` | Current drawdown from 60d high |
| `cluster_id` | Exposure cohort identifier |
| `defensive_multiplier` | Applied multiplier |
| `defensive_bucket` | Classification (elite/good/neutral/penalty) |
| `defensive_notes` | Audit trail of applied adjustments |

---

## 9. Diagnostics & Metadata

Each run explicitly records:

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

Partial data availability (e.g., IPOs) is surfaced via:
- Missing metrics flagged explicitly
- `def_cache_*` skip reasons in notes

---

## 10. Non-Goals (By Design)

- No intraday data
- No GARCH / EWMA volatility
- No ML-based clustering
- No volatility-scaling of expected returns
- No hidden portfolio construction logic

---

## 11. Summary

The Defensive Overlay is a **risk shaping layer**, not an alpha engine.

It:
- Reduces tail risk
- Improves diversification awareness
- Preserves ranking integrity
- Remains stable across runs
- Is defensible to ICs, PMs, and auditors

**This spec is production-locked.**

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial spec with boost eligibility gate |
