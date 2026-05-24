---
name: selector-ranker
triggers:
  - selector engine
  - ranker engine
  - coinvest selector
  - pairwise ranker
  - EW Top-30 construction
  - decision engine
  - ruleset
  - final_score
  - production stack
  - dead lanes
description: >
  Reference for the production two-stage selector/ranker architecture and EW
  Top-30 construction. Covers v1.14.0 stack (coinvest-only selector, pairwise
  minimal ranker), decision engine layers, dead lanes, and promotion governance.
  financial_score negative weight is intentional stress-upside (Spec 093).
---

# Selector / Ranker / Construction Skill

## Purpose

Reference for the production two-stage selector/ranker architecture and EW Top-30 construction. This is how the screener turns scored tickers into an actionable ranked portfolio.

This skill is organized into two sections:

1. **Framework Reference** - Stable architecture, engines, dead lanes, and governance (changes only with code updates)
2. **Operational State** - Volatile research status and metrics that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Production Stack (v1.14.0)

```
Modules 1-5 (scoring)
  -> Decision Engine (L0 gates -> L2 overlays -> L4 tiers -> L3 sizing -> sort key)
  -> Selector Engine (B6: coinvest_score_z 100%)
  -> Ranker Engine (pairwise_minimal: 2 deployed features, top-60 cohort, ordinal-only)
  -> Sort by final_score -> EW Top-30 -> rankings.csv
```

---

## Selector Engine

**File**: `selector_engine.py`

### B6 Selector (Production)

- **v1.14.0**: coinvest_score_z at 100% weight (coinvest-only)
- **Prior (v1.13.0)**: coinvest 65% + inst_delta_z 35%
- inst_delta_z zeroed 2026-05-04 (ALERT: mean IC = -0.097 over 36 dates, two-frame confirmed)
- Reinstatement conditions documented in governance log

### Selector Validation

- Checklist v2 (2026-04-04): bootstrap +2.42pp/mo, 95% CI [1.25%, 3.70%], P(>0) = 99.99%
- LOSO: ROBUST across all dimensions
- Neither component survives standalone, but the bundle is real
- Sort anchor: `selector_score`

---

## Ranker Engine

**File**: `ranker_v2_pairwise.py`

### Pairwise Minimal Ranker (Production)

- 2 deployed features, ordinal-only (no rank-weighting, no confidence sizing)
- Live artifact: `production_data/ranker_v2_model.json`
- Deployed weights: `coinvest_score_z` +0.02 (capped Family C live pilot), `financial_score` -0.0533
- ECE = 0.129 (POOR calibration - confirms ordinal-only is correct)
- Top-60 cohort scope
- inst_delta_z excluded from production ranker and zeroed in selector since v1.14.0

### Within-Top-30 Feature Roles

| Feature | Role | Evidence / Weight |
|---------|------|-------------------|
| coinvest_score_z | Bounded institutional support within top-60 | +0.02 deployed (+0.0613 trained) |
| financial_score | True negative penalty (stress-upside) | -0.0533 deployed; NW t-stat -3.41 |
| inst_delta_z | Diagnostic/export only | 0% selector weight; absent from ranker |

### financial_score Sign Direction (RESOLVED, Spec 093)

- Weight: -0.0533 in `production_data/ranker_v2_model.json`
- **Confirmed intentional**: stress-upside thesis (Spec 074, reconfirmed Spec 093 2026-05-13)
- Classification: INTENTIONAL_STRESS_UPSIDE
- Negative weight means financially safe names are penalized (more catalytic, less safe names preferred)
- Six-diagnostic audit confirmed TRUE PENALTY in both bull (NW-t=-3.42) and bear (-3.38) regimes

---

## Construction

### EW Top-30

- Equal-weight, top 30 names by final_score
- K=30 validated by PIT sweep (stable K=25-35 plateau, net-of-cost peak)
- RW-EW delta = -0.09pp, t = -0.95 (rank-weighting does NOT help)

### Production Evidence

- True PIT backtest: +2.34pp/mo net-of-cost, t = 2.57, 69% hit rate, 67 monthly periods (Jun 2020 - Apr 2026)
- Bear/neutral alpha engine: Bear +3.37pp (75% hit), Neutral +6.23pp (93% hit), Bull -0.37pp (50% hit)
- Regime caveat: expect bounded underperformance in strong bull markets

---

## Decision Engine

**File**: `decision_engine.py`

### Pipeline Layers

| Layer | Purpose |
|-------|---------|
| L0 | Hard gates (liquidity, price, data quality) |
| L2 | Overlays (event_type_score as diagnostic) |
| L4 | Tier classification |
| L3 | Position sizing |

### EV/Sizing Severity Consumption (Spec 101, RESOLVED)

```
dilution_haircut = 0.35 * ev_severity_score
size_multiplier = max(0.40, 1.0 - 0.60 * ev_severity_score)
```

`ev_severity_score` exported to `rankings.csv` and `SNAPSHOT_COLUMNS`. `check_severity_formulas()` QA validation runs every snapshot.

---

## Dead Lanes (Do Not Reopen Without New Evidence)

| Lane | Status | Why |
|------|--------|-----|
| Options surface-shape as ranker | DEAD | 50-month IC negative all horizons |
| Options-as-alpha (Spec 053) | CLOSED | 37 signals tested, ALL fail |
| Static execution features (Spec 054) | CLOSED | All noise/destructive |
| Clinical composites as ranker (Spec 055) | CLOSED | Negative across ALL slices |
| total_volume_z | DEAD | IC = -0.10 on PIT data |
| Always-on rank-weighting | NOT PROMOTED | RW-EW = -0.09pp |
| insider_exec_buy_value_90d | SHADOW ONLY | 1/5 Checklist v2 |
| aact_execution_score | SHADOW ONLY | 1/5 Checklist v2 |
| cal_alpha | REMOVED v1.12.0 | Confirmed no-op |
| Clinical sort signal | OFF | Insufficient IC |
| Fixed sleeve budgets | RETIRED | Primary construction damage (+153.6pp drag) |

---

## Promotion Governance

| Component | File |
|----------|------|
| Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Promote Script | `scripts/promote_ruleset.py` (blocks unless battery PASS) |
| Health Monitor | `tools/ruleset_health_monitor.py` (post-promotion drift) |
| Rollback | `scripts/promote_ruleset.py --rollback --reason "..."` |

---

## Source Files

| Component | File |
|----------|------|
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker v2 Pairwise | `ranker_v2_pairwise.py` |
| Main Orchestrator | `run_screen.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Checklist v2 | `scripts/research/checklist_v2_rerun.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-24*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Architecture freeze**: ACTIVE — h20d DEFERRED (Path B, 2026-05-24). No changes until gate clears.

## Governance Freeze & Quarantine Status

*Last reviewed: 2026-05-24*

- **Architecture Freeze**: ACTIVE — no selector/ranker/sizing changes authorized
- **13F Q1 2026 Quarantine**: ACTIVE — Jaccard 0.364 (gate requires ≥ 0.70)
- **Re-decision**: Condition-based; earliest plausible 2026-06-15, more likely 2026-07-01+
- **Next eligible action**: Spec 100 corrected final_score IC evaluation + Checklist v2 battery once gate clears
