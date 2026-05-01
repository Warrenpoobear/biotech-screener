# Ranker Active Contract — 2026-04-30

Hygiene pass to make the production ranker path match what is actually active,
and to reduce attribution confusion from inert/dead scoring lanes.

**Scope:** documentation, registry, and tests only. No alpha promotion, no
retrain, no feature-weight change, no coinvest removal, no selector change,
no portfolio construction change. Produced from clean `main` (`5079519e`),
isolated from unmerged `hotfix/iv-quarantine-2026-04-25`.

**Companion code:** `common/ranker_active_contract.py` (registry),
`tests/test_ranker_active_contract.py` (drift gates).

---

## 1. Active production scoring path

### Ranker — `production_data/ranker_v2_model.json`

Pairwise logistic, `feature_set="minimal_v2"`, **2 features only**:

| Feature             | Deployed weight | Trained weight | Note                                     |
| ------------------- | --------------: | -------------: | ---------------------------------------- |
| `coinvest_score_z`  |          +0.020 |        +0.0613 | Capped vector (Family C live-pilot)      |
| `financial_score`   |        −0.05332 |       (unchanged) | Penalizes "safe" names                |

Bias = +0.5019. Cohort = top 60 by `selector_score`. Outside the cohort,
`final_score = selector_score × 0.0001` (sorts non-cohort eligible *below*
the cohort but preserves their relative order). Non-eligible rows have
`ranker_v2_score = ""`.

The artifact's `provenance` block is authoritative:

> coinvest_score_z weight capped from 0.0613 (trained) to 0.02 (deployed) —
> Family C live-pilot vector. financial_score weight unchanged.

### Selector — `run_screen.A4_SELECTOR_CONFIG`

Spec 050 v1.1, validated true-PIT backtest +2.34pp/mo net (t=2.60).

Block weights:

| Block             | Weight |
| ----------------- | -----: |
| `clinical`        |   0.00 |
| `catalyst`        |   0.15 |
| `survivability`   |   0.10 |
| `institutional`   |   0.65 |
| `market_structure`|   0.10 |

Institutional signal weights (override of `SelectorConfig` defaults):

| Signal                  | Weight | Note                                  |
| ----------------------- | -----: | ------------------------------------- |
| `coinvest_score_z`      |   0.65 | Active                                |
| `inst_delta_z`          |   0.35 | Active                                |
| `coinvest_recency_state`|   0.00 | Categorical, A4 override zeroed it    |

Other blocks inherit `SelectorConfig` defaults (catalyst, survivability,
market_structure signals — see `selector_engine.SelectorConfig`).

### Construction — `decision_engine.compute_decision_fields()`

Takes `final_score` and produces `actionable_rank`, `target_weight_pct`,
`tier_*`. No further scoring. Multiplicative haircuts only
(`cost_haircut_mult`, `catalyst_tilt_mult`, etc.) — these are construction
controls, not alpha.

---

## 2. Where coinvest is currently used

| Site                         | Field                  | Weight | Effective contribution |
| ---------------------------- | ---------------------- | -----: | ---------------------: |
| Ranker v2 (active)           | `coinvest_score_z`     | +0.020 |                  +0.020 |
| Selector institutional block | `coinvest_score_z`     |   0.65 | 0.65 × 0.65 = **0.4225** |
| Selector institutional block | `coinvest_recency_state`|  0.00 |                       0 |

Coinvest in raw / non-z form (e.g. `coinvest_score`, `coinvest_conviction`,
`coinvest_max_position_pct`, `sponsor_*`) is **diagnostic pass-through** —
written to `rankings.csv` for inspection but not consumed by ranker or
selector. See `coinvest_score_z` is the *only* coinvest-derived field in
either active path.

Standing direction (per memory `policy_coinvest_context_layer_2026_04_25`):
coinvest is a context layer, not a ranker feature — but **do NOT strip
coinvest from the live ranker without an audited replacement + forward
attribution**. Today's ranker is frozen at the 2-feature minimal_v2.

---

## 3. Computed-but-inactive fields (categorized)

### DEPRECATED_INERT — was active before, now inert

| Field                    | Why inert                                                                |
| ------------------------ | ------------------------------------------------------------------------ |
| `clinical_score_v2_z`    | Removed from ranker minimal_v2 (Spec 055 ablation -0.35pp, destructive). Selector clinical block weight = 0.00. Computation kept for research. |
| `clinical_quality_score` | Spec 057 monitor-only. Sizing tilt tested + REJECTED. Use for attribution only. |
| `clinical_alpha_z`       | Selector clinical block zeroed; computation kept.                       |
| `coinvest_recency_state` | A4 override zeros it (default SelectorConfig was 0.20).                 |

### SIDECAR_RESEARCH — written to a separate sidecar artifact

| Field                          | Sidecar location                          |
| ------------------------------ | ----------------------------------------- |
| `ees_v2_score`                 | `expectation_error_overlay.json`          |
| `ees_v3_score`                 | `ees_v3_overlay.json`                     |
| `expectation_error_score`      | `expectation_error_overlay.json`          |
| `conditional_misprice_score`   | EES v3 overlay (true alpha source — 4/5 Checklist v2) |
| `base_rate_gap_score`          | EES v3 overlay (NEGATIVE control only — IC -0.090, anti-predictive; never promote) |

Promotion rule: EES requires Checklist v2 5/5 (currently 4/5 — WS4 fail
t_adj=1.53 vs 1.65 threshold) AND ≥30 live days. See Spec 064 promotion
battery.

### DIAGNOSTIC_PASS_THROUGH — written to rankings.csv but not in active path

| Field                          | Purpose                                  |
| ------------------------------ | ---------------------------------------- |
| `insider_net_buy_value_90d`    | Form 4 Pass B (2026-04-24). Scoring lane CLOSED 2026-04-05. `required=True` in `production_qa_check.py` is a data-integrity gate ONLY — does not promote to selector / ranker / sizing. See Spec 065. |

The rankings.csv contains many additional fields not enumerated in the
registry (display-only metadata, debug columns, options chain pass-throughs,
etc.). The registry is **not exhaustive** — it focuses on fields that have
caused or could cause attribution confusion. Add new entries when:

- closing a scoring lane (mark `DEPRECATED_INERT`)
- promoting a sidecar to active (move `SIDECAR_RESEARCH` → `ACTIVE_*`)
- a new pass-through field is mistaken for active driver

---

## 4. Boundaries

```
                         eligible universe (from gates)
                                    │
                                    ▼
                          ┌──────────────────┐
                          │     SELECTOR     │  A4_SELECTOR_CONFIG
                          │  selector_score  │  (block × signal)
                          └──────────────────┘
                                    │
                              top-60 cohort
                                    │
                                    ▼
                          ┌──────────────────┐
                          │      RANKER      │  ranker_v2_model.json
                          │ ranker_v2_score  │  (coinvest_score_z, financial_score)
                          └──────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
       cohort: final_score = ranker_v2_score   non-cohort eligible:
                                                final_score = selector_score × 0.0001
                                    │
                                    ▼
                          ┌──────────────────┐
                          │  CONSTRUCTION    │  decision_engine
                          │ actionable_rank  │  target_weight_pct
                          │   tier_*         │  multiplicative haircuts
                          └──────────────────┘
                                    │
                                    ▼
                              rankings.csv
                          decision_portfolio.{csv,json}
```

Sidecars (EES overlays, options snapshot, expectation_error_overlay,
ees_gate_diagnostics, etc.) are written *alongside* this flow from
`save_validation_snapshot` but do not feed back into `final_score` or
`target_weight_pct`.

---

## 5. Drift gates

`tests/test_ranker_active_contract.py` enforces:

1. `ACTIVE_RANKER_FEATURES` == `production_data/ranker_v2_model.json` `feature_names`
2. `ACTIVE_SELECTOR_BLOCKS` == `run_screen.A4_SELECTOR_CONFIG.block_weights`
3. `ACTIVE_INSTITUTIONAL_SIGNALS` == A4 institutional signal weights
4. Clinical block weight remains 0.00 (Spec 055 protection)
5. No field tagged `SIDECAR_RESEARCH` / `DEPRECATED_INERT` / `DIAGNOSTIC_PASS_THROUGH` appears in the deployed ranker artifact's `feature_names`
6. Per-field status guards for known-dead lanes (`clinical_score_v2_z`,
   `clinical_quality_score`, `clinical_alpha_z`, `coinvest_recency_state`,
   `ees_v2_score`, `ees_v3_score`, `expectation_error_score`,
   `conditional_misprice_score`, `base_rate_gap_score`,
   `insider_net_buy_value_90d`)

If you change any active weight or close any lane, you MUST update
`common/ranker_active_contract.py` *and* the corresponding ground truth, or
this test will fail in CI.

---

## 6. Before/after — proof of zero numeric output change

This pass adds three files, modifies zero existing files:

```
A  artifacts/ranker_active_contract_2026_04_30.md   (audit artifact, this file)
A  common/ranker_active_contract.py                 (registry — pure declarations)
A  tests/test_ranker_active_contract.py             (drift + status guards)
```

`common/ranker_active_contract.py` has no executable side effects on import
(no module-level code beyond dataclass declarations and tuple constants).
It is **not imported** by any production scoring code. Zero risk of
behavior change.

`production_data/ranker_v2_model.json` is unchanged
(SHA-256 prefix `3cd76916e58cebb2`, captured before and after).

---

## Non-goals (not done by this pass)

- Coinvest stays in the live ranker.
- Ranker is not retrained.
- Clinical / EES / catalyst / options / Form 4 are not promoted.
- No model weights changed.
- IV quarantine branch not touched.
- No unrelated runtime files cleaned.
