# Fundamental Red-Flag Suppressor v1.0

**Status:** Production default
**Scope:** Risk sanity override applied AFTER defensive overlay
**Applies to:** Final composite ranking
**Design goal:** Prevent structurally broken companies from ranking above median

---

## 1. Purpose & Philosophy

The Fundamental Red-Flag Suppressor is a **risk sanity override**, not an alpha engine.

It prevents companies that are:
- Financially distressed
- Heavily diluted
- Clinically impaired

...from ranking above median **even when** they appear:
- Low-vol
- Low-correlation
- "Defensive"

### Design Principles

1. **Suppress, don't reshuffle** - Never boost, only cap or penalize
2. **Hard rules only** - No ML, no subjective scoring, deterministic thresholds
3. **Fundamentals > defensiveness** - A stock can be defensive *and still bad*
4. **Explainable in one sentence** - Every suppression emits a reason code

---

## 2. Pipeline Position (Locked)

```
signals → expected_excess_return
        → defensive overlay (multiplier)
        → fundamental red-flag suppressor  ← HERE
        → final composite_score
```

**Rationale:**
- Defensive overlay adjusts *risk profile*
- Red-flag suppressor enforces *business viability*
- Alpha is never recomputed

---

## 3. Red-Flag Criteria (v1.0)

A security is **red-flagged** if **ANY** of the following are true:

### 3.1 Financial Viability

| Condition | Threshold | Reason Code |
|-----------|-----------|-------------|
| Cash runway | < 6 months | `cash_runway_lt_6m` |
| Cash burn trajectory | `critical` | `cash_burn_critical` |
| Revenue | = 0 AND stage ≥ Phase 3 | `no_revenue_late_stage` |

### 3.2 Capital Structure / Dilution

| Condition | Threshold | Reason Code |
|-----------|-----------|-------------|
| Dilution risk | `HIGH` | `dilution_risk_high` |
| Share count growth | > 40% YoY (if available) | `share_dilution_gt_40pct` |

### 3.3 Clinical Credibility

| Condition | Threshold | Reason Code |
|-----------|-----------|-------------|
| Phase momentum | `strong_negative` | `phase_momentum_strong_neg` |
| Pipeline diversity | `single_asset` AND stage < Phase 3 | `single_asset_early_stage` |

---

## 4. Suppression Logic

### 4.1 Classification

```python
fundamental_red_flag: bool  # true if ANY criterion triggered
fundamental_red_flag_reasons: List[str]  # all triggered reason codes
```

### 4.2 Action (Median Cap)

```python
if fundamental_red_flag:
    final_score = min(adjusted_score, median_score)
```

**Guarantees:**
- No red-flagged name ranks above median
- No rank reshuffling among "good" names
- No hidden boosts

### 4.3 Bounds

```
suppression_floor = 0.0  # Can suppress to zero in extreme cases
suppression_ceiling = median_score  # Never above median
```

---

## 5. Required Output Columns

Every run must emit:

| Column | Type | Description |
|--------|------|-------------|
| `fundamental_red_flag` | bool | True if any criterion triggered |
| `fundamental_red_flag_reasons` | List[str] | All triggered reason codes |
| `composite_score_pre_suppression` | Decimal | Score before suppression |

---

## 6. Metadata & Diagnostics

```json
"fundamental_red_flag_config": {
  "suppressor_enabled": true,
  "suppressor_version": "1.0.0",
  "median_score_used": "42.5",
  "flagged_count": 15,
  "suppressed_count": 8
}
```

---

## 7. Non-Goals (Explicit)

The Red-Flag Suppressor does **not** include:

- Subjective "fundamental quality score"
- Analyst opinions
- Discretionary blacklists
- ML classifiers
- Reweighting alpha components

---

## 8. Summary

This suppressor:

- Is institutionally standard
- Prevents "technically correct but PM-untrustworthy" rankings
- Separates business viability from risk profile cleanly
- Is deterministic, PIT-safe, and auditable
- Is suitable as a **production default**

**Spec v1.0 is production-locked.**

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial production spec |
