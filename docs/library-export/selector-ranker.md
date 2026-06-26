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
  -> Ranker Engine (pairwise_minimal: 6 features, top-60 cohort, ordinal-only)
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

> **Signal naming cross-reference (CON-1):** The repo model documentation and .docx files use legacy signal names: `sponsorship_score_z` (= `coinvest_score_z`) and `momentum_delta_z` (= `inst_delta_z`). These are the SAME signals under different names. The "sponsorship/momentum" naming predates the v1.14.0 renaming to "coinvest/inst_delta." All current skill documents and production code use `coinvest_score_z` and `inst_delta_z`.

### Selector Validation

- Checklist v2 (2026-04-04): bootstrap +2.42pp/mo, 95% CI [1.25%, 3.70%], P(>0) = 99.99%
- LOSO: ROBUST across all dimensions
- Neither component survives standalone, but the bundle is real
- Sort anchor: `selector_score`

### What the Selector Learns

Coinvest selects WHICH 30 names enter the portfolio. It captures institutional co-investment conviction from elite biotech managers.

---

## Ranker Engine

**File**: `ranker_v2_pairwise.py`

### Pairwise Minimal Ranker (Production)

- **6 input features** enter the ranker at runtime, but the **deployed artifact stores only 2 non-zero trained weights** (coinvest_score_z and financial_score) in `production_data/ranker_v2_model.json`. The remaining 4 features have near-zero or washed-out coefficients but are retained in the feature vector for forward compatibility and diagnostic logging.
- **Note on document discrepancies:** The repo model documentation files and the .docx Executive Overview describe the ranker as "2-feature" because they reference the stored weight artifact. This skill describes "6 features" because it references the full runtime input vector. Both are correct descriptions of different layers. The deployed artifact's `provenance` block is authoritative for production weights.

> **OPEN ISSUE N1 (Run 8, 2026-05-25):** `coinvest_score_z` **deployed ranker weight = +0.02** (capped Family C live pilot, per model_documentation.md v1.7.2). The **trained basis weight = +0.0613** (stored in artifact). These differ. The +0.02 cap reflects a deliberate live-pilot ceiling, not the trained coefficient. Always cite +0.02 as the deployed weight in production context. Cite +0.0613 only when describing the trained artifact.

- ECE = 0.129 (POOR calibration - confirms ordinal-only is correct)
- Top-60 cohort scope
- inst_delta_z zeroed in **selector** since v1.14.0 (2026-05-04), but **remains active in ranker** as a feature

> **Fix applied 2026-05-16 (Code Review H3):** Corrected "excluded from ranker since Spec 051" to clarify that inst_delta_z was zeroed in the SELECTOR (not the ranker). Cross-reference: institutional-signal skill confirms "Active in ranker: dominant positive discriminator (NW-t = +3.32)".

### Within-Top-30 Feature Roles

| Feature | Role | NW t-stat |
| --- | --- | --- |
| inst_delta_z | Dominant positive discriminator | +3.32 |
| financial_score | True negative penalty (stress-upside) | -3.41 |
| coinvest_score_z | Washes out within cohort | +0.49 |

### financial_score Sign Direction (RESOLVED, Spec 093)

- Weight: -0.0533 in `production_data/ranker_v2_model.json`
- **Confirmed intentional**: stress-upside thesis (Spec 074, reconfirmed Spec 093 2026-05-13)
- Classification: INTENTIONAL_STRESS_UPSIDE
- Negative weight means financially safe names are penalized (more catalytic, less safe names preferred)
- Evidence: correct higher_is_better=True encoding; six-diagnostic audit confirmed TRUE PENALTY in both bull (NW-t=-3.42) and bear (-3.38) regimes
- Raw components: 50% runway + 30% dilution + 20% liquidity (all directional: higher = better health)
- Rank-normalized within stage x size cohort (direction preserved)
- t-statistic significant (-3.41), persists across cohorts and regimes

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
| --- | --- |
| L0 | Hard gates (liquidity, price, data quality) |
| L2 | Overlays (event_type_score as diagnostic) |
| L4 | Tier classification |
| L3 | Position sizing |

### EV/Sizing Severity Consumption (Spec 101, RESOLVED)

The L3 position sizing layer consumes `ev_severity_score` (from runway severity v1.1) to compute:

```
dilution_haircut = 0.35 * ev_severity_score
size_multiplier = max(0.40, 1.0 - 0.60 * ev_severity_score)
```

`ev_severity_score` is now exported to `rankings.csv` and `SNAPSHOT_COLUMNS` (Spec 101, commits eaa4ea87 + cba4ee0f). `check_severity_formulas()` QA validation runs every snapshot.

---

## Dead Lanes (Do Not Reopen Without New Evidence)

| Lane | Status | Why |
| --- | --- | --- |
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
| --- | --- |
| Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Promote Script | `scripts/promote_ruleset.py` (blocks unless battery PASS) |
| Health Monitor | `tools/ruleset_health_monitor.py` (post-promotion drift) |
| Rollback | `scripts/promote_ruleset.py --rollback --reason "..."` |

---

## Source Files

| Component | File |
| --- | --- |
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker v2 Pairwise | `ranker_v2_pairwise.py` |
| Ranker Legacy | `ranker_engine.py` |
| Main Orchestrator | `run_screen.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Checklist v2 | `scripts/research/checklist_v2_rerun.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** - The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-13*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Architecture freeze**: LIFTED 2026-05-26 (h20d checkpoint passed). Post-h20d sequence can begin after operator approval.
- **ranker_active_contract.py**: Exists on unmerged branch (`hygiene/ranker-active-contract-2026-04-30`), deferred to post-freeze.

## Ranker Alternatives Research (Key Findings, 2026-05-13 audit cycle)

- **Spec 093 (financial_score sign direction)**: RESOLVED as INTENTIONAL_STRESS_UPSIDE. Not an artifact.
- **Spec 094 (selector-only comparator)**: Classification RANKER_UNPROVEN. Jaccard overlap between selector-only and production top-30 is 42.7% (significant churn). Rerun target was 2026-05-27 — status PENDING operator confirmation.
- **Spec 095 (evaluation scope)**: CURRENT_TOOLS_CONFLATED. IC backtest measures composite_score, NOT production final_score. Ranker IC is UNMEASURED.
- **Spec 100 (ranker IC tooling correction)**: Commit 2faa88e6. Highest-priority code change post-architecture-freeze.

**Promotion eligibility horizon**: April 2027 (one-year stability gate)

### Monitoring Specs (2026-05-13)

| Spec | Purpose | Gate | Next Review |
| --- | --- | --- | --- |
| 096 | Gate/ranker separation doctrine | Defines promotion paths | Ongoing |
| 097 | Event-EV prospective monitoring | Brier <= 0.08, n >= 30 | Monthly |
| 098 | Catalyst timing prospective monitor | Correlation > 0.15 | Monthly |
| 099 | Clinical orthogonality audit | Pre-promotion gate | Before any clinical signal promotion |
