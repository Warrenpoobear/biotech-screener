# Decision Engine Improvement Recommendations

**Based on**: 23 snapshots, 2026-01-15 to 2026-02-16
**Diagnostics source**: `output/decision_engine_diagnostics_2026-01-15_2026-02-16.json`
**Engine version**: v1.3.x (DecisionRuleset, catalyst_priority_mode)

---

## Executive Summary

The Decision Engine is stable: avg tier churn 0.7%, top-20 Jaccard 0.929, top-60 Jaccard 0.975.
However, diagnostics reveal structural issues worth addressing:

1. **Clinical score paradox**: A-tier names have the _lowest_ mean clinical score (16.6) of any tier. C-tier has the highest (63.4). Clinical quality does not drive tier assignment.
2. **Catalyst gating is binary**: 25% of names lack `catalyst_days` entirely. The hard near/mid/far cutoff creates cliff effects at the boundary.
3. **136 names (~43%) have no tier**: These are commercial archetypes (exempt from clinical gate) but still represent a missed ranking opportunity.
4. **Tier churn concentrates at regime transitions**: Jan-30→Feb-02 had 8 changes (5 A-upgrades) from SEC 8-K cache warming, not fundamental change.

---

## Recommendation 1: Smooth Catalyst Gating with Logistic Decay

**Current**: Hard cutoffs at `catalyst_near=120d` and `catalyst_mid=180d` create cliff effects. A name at day 119 gets "near" (A-eligible); at day 121 it drops to "mid" (B at best).

**Change**: Replace step function with logistic decay:
```
catalyst_proximity_weight = 1 / (1 + exp((days_to_catalyst - catalyst_midpoint) / decay_rate))
```
Where `catalyst_midpoint=150d`, `decay_rate=30d`. This maps: 60d → ~0.95, 120d → ~0.73, 150d → 0.50, 180d → ~0.27, 240d → ~0.05.

**Benefit**: Eliminates cliff effects. Names near boundaries transition smoothly rather than flipping tiers overnight.

**Risk**: Changes tier assignment for names near the 120d/180d boundaries. Could increase A-count if decay is too generous.

**Validation metric**: Tier churn rate at dates where no cache change occurred (currently 0.7% avg — target: <0.5%). Monitor A-count stability (current range: 31-38).

**Guardrail**: A-count must stay in [25, 45] band; if violated, tighten `catalyst_midpoint`.

---

## Recommendation 2: Percentile-Based A-Floor (Regime-Adaptive)

**Current**: `a_floor=0.60` is a fixed optionality threshold. A-count swings 31-38 depending on how many names happen to exceed 0.60.

**Change**: Set `a_floor` dynamically as the P_k percentile of dev-cohort optionality, targeting a stable A-count band.
```python
a_floor_adaptive = np.percentile(dev_optionality_scores, a_floor_percentile)
a_floor = max(a_floor_adaptive, a_floor_hard_minimum)  # e.g., hard min = 0.45
```
With `a_floor_percentile=60` (i.e., top 40% of dev cohort are A-eligible, subject to catalyst gate).

**Benefit**: A-count stabilizes regardless of optionality distribution drift. Prevents regime-dependent A-count swings.

**Risk**: Makes A-tier a relative measure rather than absolute. In a weak market, poor names could get A-tier.

**Validation metric**: A-count coefficient of variation across dates (current: ~7% — target: <3%). A-count should be in [33, 40].

**Guardrail**: Hard minimum `a_floor >= 0.45` prevents degraded-quality A-tier.

---

## Recommendation 3: Clinical Score Integration into Tier

**Current**: Clinical score has _negative_ correlation with tier quality (A-tier mean=16.6, C-tier mean=63.4). This happens because the tier gate is purely {optionality + catalyst}, and high-optionality names tend to be early-stage (low clinical scores).

**Change**: Add a clinical-quality overlay that penalizes A-tier names with very low clinical scores:
```python
if tier_dev == "A" and clinical_score < clinical_floor:
    tier_dev = "B"
    tier_reason += "+clinical_weak"
```
Where `clinical_floor = P25(A-tier clinical) ≈ 8.2`. This catches the worst ~25% of A-tier clinical profiles.

**Benefit**: Prevents names with no meaningful clinical data from reaching A-tier purely on catalyst timing + optionality.

**Risk**: Reduces A-count by ~9 names (25% of 37). May conflict with the optionality-first philosophy.

**Validation metric**: Mean clinical_score of A-tier should rise from 16.6 to >25. Monitor for lost alpha if demoted names actually outperform.

**Guardrail**: Only apply as a soft flag initially (add `clinical_weak` flag without changing tier), then promote to hard gate after 4-week forward test.

---

## Recommendation 4: Explainability Columns

**Current**: `tier_reason` gives a single string (e.g., `high_opt+catalyst_near`). Users can't quickly see _why_ a name is ranked where it is.

**Change**: Add three columns to rankings.csv:
- `top_3_drivers`: Sorted list of the 3 highest z-score component signals (e.g., "momentum_z=2.1, smart_money_z=1.8, clinical_z=-0.4")
- `catalyst_detail`: Human-readable catalyst info (e.g., "FDA PDUFA in 45d" or "Phase 3 readout in 120d")
- `risk_summary`: Compact flags (e.g., "high_vol+low_runway" or "clean")

**Benefit**: Portfolio manager can scan rankings.csv and immediately understand each name's positioning without opening per-ticker artifacts.

**Risk**: Minimal — purely additive columns, no model change.

**Validation metric**: N/A (UX improvement). Verify columns are populated for >95% of ranked names.

**Guardrail**: None needed — purely informational.

---

## Recommendation 5: Missing-Data Penalty for Catalyst

**Current**: 25% of names have missing `catalyst_days`. These names get `catalyst_mode=missing` and are ineligible for A-tier, but their composite_rank is not explicitly penalized.

**Change**: Apply a composite score haircut when key signals are missing:
```python
if catalyst_mode in ("missing", "no_upcoming"):
    composite_score *= (1 - missing_catalyst_penalty)  # e.g., 0.95 → 5% haircut
```

**Benefit**: Names with complete data surface above names with comparable raw scores but missing catalyst. Incentivizes data completeness.

**Risk**: Could disadvantage names where catalyst data legitimately doesn't exist (e.g., commercial companies). Must be archetype-aware.

**Validation metric**: Rank of missing-catalyst names should decrease by 5-10 positions on average. Top-60 missing-catalyst rate should drop from 6.4% to <3%.

**Guardrail**: Only apply to `archetype=drug_developer`. Skip for commercial archetypes.

---

## Recommendation 6: Separate Rank from Actionability

**Current**: `actionable_rank` is derived from `composite_rank` filtered by tier. This conflates "how good is this name" with "is it actionable now."

**Change**: Maintain two independent rankings:
1. `quality_rank`: Pure fundamental quality (clinical + financial + smart_money), time-stable
2. `actionability_rank`: Time-varying (catalyst proximity + momentum + vol regime)

Final `portfolio_rank = f(quality_rank, actionability_rank)` with configurable blend weight.

**Benefit**: Separates long-term conviction from short-term timing. Enables "watchlist" view (high quality, low actionability) vs "trade now" view.

**Risk**: Adds complexity. Two-rank system requires clear documentation and UX guidance.

**Validation metric**: Quality rank should have higher Jaccard stability (>0.95) than current composite. Actionability rank should correlate with forward 1-month returns better than composite alone.

**Guardrail**: Start with quality_weight=0.6, actionability_weight=0.4 and calibrate via walk-forward panel.

---

## Recommendation 7: Tier D Refinement — Separate "Weak" from "Uninvestable"

**Current**: D-tier conflates "ineligible" (1260 instances, 30% of all tier reasons) with names that simply score poorly. Low momentum is the D-tier signature (mean=39.6 vs A-tier 61.6).

**Change**: Split D into:
- `D_momentum`: Fundamentally okay but in momentum downtrend (recoverable)
- `D_ineligible`: Failed hard gates (not recoverable without data change)
- `D_weak`: Low across multiple signals

**Benefit**: Reduces D-tier false equivalence. "D_momentum" names are watchlist candidates when momentum turns.

**Risk**: More tiers = more complexity. May confuse rather than clarify if not well-documented.

**Validation metric**: D_momentum names should have higher forward 3-month return than D_ineligible names. If not, the split is not informative.

**Guardrail**: Implement as a `tier_detail` sub-column, keeping `tier_dev` unchanged for backward compatibility.

---

## Recommendation 8: Monotonic Safety Constraint on Financial Survivability

**Current**: A name with very low financial_score (e.g., <12 months runway) can still reach A-tier if optionality + catalyst are strong enough.

**Change**: Add a financial safety floor:
```python
if financial_score < financial_safety_floor and tier_dev == "A":
    tier_dev = "B"
    tier_reason += "+financial_risk"
```
Where `financial_safety_floor` targets names in bottom quartile of financial scores (P25 of A-tier financial = 9.7).

**Benefit**: Prevents allocating capital to names with severe cash/runway risk, even if their catalyst timing is perfect.

**Risk**: Excludes some legitimate pre-revenue biotechs that are viable through upcoming catalysts.

**Validation metric**: Count of A-tier names with financial_score < 10 should go to zero. Monitor for lost alpha in excluded names.

**Guardrail**: Only apply when `runway_bucket = "critical"`. Do not apply broadly to all low-financial names.

---

## Priority Matrix

| # | Recommendation | Effort | Impact | Priority |
|---|---------------|--------|--------|----------|
| 1 | Smooth catalyst gating | Medium | High | **P1** — directly reduces cliff-driven churn |
| 4 | Explainability columns | Low | Medium | **P1** — no model risk, immediate UX value |
| 3 | Clinical score overlay | Low | Medium | **P2** — addresses the A-tier paradox |
| 5 | Missing-data penalty | Low | Medium | **P2** — improves data hygiene incentives |
| 2 | Percentile-based A-floor | Medium | Medium | **P2** — stabilizes A-count across regimes |
| 8 | Financial safety floor | Low | Low | **P3** — edge case protection |
| 6 | Separate rank/actionability | High | High | **P3** — architectural change, defer |
| 7 | Tier D refinement | Low | Low | **P4** — nice-to-have diagnostic split |

---

## Implementation Sequence

1. **Week 1**: #4 (explainability columns) + #5 (missing-data penalty) — low risk, additive
2. **Week 2**: #1 (logistic catalyst decay) — needs walk-forward validation
3. **Week 3**: #3 (clinical overlay) + #8 (financial floor) — behind config flags
4. **Week 4+**: #2 (adaptive A-floor) after observing regime behavior
5. **Backlog**: #6 and #7 for v2.0 architectural review
