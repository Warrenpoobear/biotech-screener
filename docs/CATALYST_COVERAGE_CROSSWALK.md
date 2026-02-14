# Catalyst Coverage Crosswalk

Two systems report catalyst coverage with different definitions. This doc
prevents IC from misreading one number against the other.

---

## The Two Metrics

### 1. Audit Coverage ("any catalyst")

**Source:** `scripts/audit_archive_catalyst.py`, vNext enrichment summary

**Definition:** Fraction of dev-stage tickers that have *any* catalyst record
in the enrichment pipeline — regardless of whether it survives decision-engine
filters (near/mid/far windows, strength bands, etc.).

**Typical value:** 70-85% of dev tickers

**What it measures:** Data pipeline health. "Did the CTGOV/SEC/FDA collectors
find *something* for this ticker?" A low number here means a data gap (broken
feed, missing CIK mapping, CTGOV API downtime), not a portfolio problem.

### 2. Drift Coverage ("specific_days")

**Source:** `scripts/run_drift_report.py` → Catalyst Coverage table

**Definition:** Fraction of eligible dev tickers whose `catalyst_mode` is
`specific_days` (i.e., the decision engine found a dated catalyst within the
configured near/mid window and the ticker has a concrete days-to-catalyst
value).

**Typical value:** 40-50% of eligible dev tickers

**What it measures:** Actionable catalyst density in the portfolio universe.
"How many tickers have a *dated, actionable* catalyst event?" This is always
lower than audit coverage because:

- Catalysts outside the near/mid window (>180 days) → `catalyst_mode = "far"` or `"missing"`
- Tickers with only `blended_window` proximity (no specific date) are excluded
- The denominator is eligible dev tickers (with non-empty `tier_dev`), not all dev tickers

---

## Why They Differ

```
Audit "any catalyst" (78.5%)
  └─ Tickers with ANY catalyst record in vNext enrichment
      ├─ Within near/mid window? ──YES──→ specific_days (47%)  ← drift metric
      ├─ Within far window? ────────────→ far (not counted in drift)
      ├─ Blended/no specific date? ─────→ blended_window (not counted)
      └─ Past all windows? ────────────→ no_upcoming / missing
```

The gap between 78.5% (audit) and 47% (drift) is explained entirely by
window filtering. It does NOT indicate missing data.

---

## Guardrail Thresholds

| Metric | Warn | Fail | Rationale |
|--------|------|------|-----------|
| Drift: `catalyst_missing` | — | > 85% | Entire catalyst feed broken |
| Drift: `specific_days` share | < 40% | — | Sparse catalyst regime |
| Drift: CTGOV share | < 37% | — | Primary source dropping |
| Drift: unknown sources | > 5% | — | Provenance gap |
| Audit: dev coverage | — | < 40% | Pipeline-level data loss |

---

## Quick Reference: Where to Look

| Question | Tool | Metric |
|----------|------|--------|
| "Is the data pipeline collecting catalysts?" | `audit_archive_catalyst.py` | Any-catalyst coverage % |
| "Do we have actionable dated catalysts?" | `run_drift_report.py` | specific_days share % |
| "Which sources are feeding catalysts?" | `run_drift_report.py` | Catalyst Source Mix table |
| "Are new event types being tracked?" | `run_drift_report.py` | Catalyst Event Type Mix table |
| "Why did a ticker lose its catalyst?" | `run_drift_report.py --verbose-offenders` | Portfolio Churn table |

---

## Example: 2026-02-07 Snapshot

| Layer | Metric | Value |
|-------|--------|-------|
| Audit (pipeline) | Dev tickers with any catalyst | ~78.5% |
| Drift (engine) | Eligible with specific_days | 47.0% |
| Drift (engine) | Catalyst missing | 53.0% |
| Drift (engine) | CTGOV share | 45.9% |
| Drift (engine) | Unknown sources | 0.0% |

**Reading:** Pipeline health is strong (78.5% have data). Of those, 47% have
dated catalysts within the actionable window — this is the number that drives
portfolio composition. The 31.5pp gap is window filtering, not data loss.
