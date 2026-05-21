---
paths:
  - src/wake_robin_screener/modules/**
  - src/wake_robin_screener/features/**
  - src/wake_robin_screener/common/**
  - config/**
---

# Signals, Specs & Feature Status

## Signal Naming (Current)
| Current Name | Legacy Name | Notes |
|-------------|-------------|-------|
| `coinvest_score_z` | `sponsorship_score_z` | Renamed v1.14.0 |
| `inst_delta_z` | `momentum_delta_z` | Renamed v1.14.0 |

Always use current names in new code. Legacy names may appear in older docs — treat as identical.

## Insider Diagnostic (Spec 104)
`insider_net_buy_value_90d` is **DIAGNOSTIC ONLY**. Tracked in `DIAGNOSTIC_FIELDS`, explicitly excluded from `ALPHA_FEATURE_REGISTRY`. Does NOT enter scoring model, ranker, or selector.

**CRITICAL**: The expectation model has an `insider_net_buy_z` weight that activates silently if the field flows into `market_features`. Spec 104 R4a requires an explicit isolation guard.

**Blank vs. Zero**: NaN/None/blank = not fetched. 0.0 = fetched, no insider buy activity. Never collapse blank and zero.

**Promotion requires ALL of**: 20+ stable snapshots, >= 60% non-null coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval.

## Expectation Layer Coverage Gate (Spec 105)
Production hard-fails if market-expectation fields are missing or under-covered in `rankings.csv`.

| Field | Threshold | Blocking |
|-------|-----------|----------|
| `short_interest_pct` | 0.90 | Yes |
| `close_price` | 0.99 | Yes |
| `market_cap_mm` | 0.95 | Yes |
| `priced_move_pct` | 0.80 | Yes |
| `insider_net_buy_value_90d` | 0.30 | No (diagnostic) |

Thresholds sourced from `FEATURE_COVERAGE_REQUIREMENTS` (single source of truth).

## Active Spec Status (071-105)

### Recently Resolved
| Spec | Title | Status |
|------|-------|--------|
| 093 | financial_score sign direction | RESOLVED (INTENTIONAL_STRESS_UPSIDE) |
| 101 | Runway Severity v1.1 Export Contract | CLOSED |
| 087 B0 | Stale-Propagation Guard | CLOSED |
| 087 B2 | Dashboard Freshness Envelope | CLOSED |
| 088 B | Catalyst Delta v2 Filter Companion | SHIPPED |

### Active / Blocked
| Spec | Title | Status | Blocker |
|------|-------|--------|---------|
| 094 | Selector-only comparator | RANKER_UNPROVEN | Rerun target 2026-05-27 |
| 095 | Evaluation scope (IC tooling gap) | CURRENT_TOOLS_CONFLATED | Blocks ranker IC claims |
| 100 | Ranker IC tooling correction | Spec written, no impl | Architecture freeze (~May 26) |
| 104 | Insider diagnostic stabilization | MEASURED | Isolation guard (R4a) |
| 105 | Expectation layer coverage verification | CODE-CLOSED | Pending live QA |
| 102 | Historical backfill for expectation research | DRAFT | — |

### Monitoring
| Spec | Purpose | Gate | Review |
|------|---------|------|--------|
| 096 | Gate/ranker separation doctrine | Promotion paths | Ongoing |
| 097 | Event-EV prospective monitoring | Brier <= 0.08, n >= 30 | Monthly |
| 098 | Catalyst timing prospective monitor | Correlation > 0.15 | Monthly |
| 099 | Clinical orthogonality audit | Pre-promotion gate | Before clinical promotion |
