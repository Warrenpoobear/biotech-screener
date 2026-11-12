# Weekly Signal Regime Sweep — 2026-10-26

**Source dashboard:** `artifacts/ic_dashboard/2026-10-20_dashboard.json` (latest available; no 2026-10-26 dashboard yet generated)
**Mode:** Diagnose-only, read-only. No production files modified.
**Scope rule:** Per `signal-shared-regime-check` skill, only signals flagged WARN or ALERT in the latest IC dashboard are paired with a comparator and probed.

## Fleet Status at Latest Dashboard

| Signal | Mean IC | Hit Rate | Latest IC | Health | Dates | In Scope? |
|---|---|---|---|---|---|---|
| score_rank_pct | +0.0264 | 87% | +0.0765 | WEAK | 52 | No (WEAK ≠ WARN/ALERT) |
| inst_delta_z | -0.0010 | 59% | +0.0137 | **WARN** | 41 | **Yes** |

Only one load-bearing signal (`inst_delta_z`) met the WARN/ALERT threshold this week. `score_rank_pct` is WEAK, which is below the sweep's WARN/ALERT pairing threshold and was not probed.

## Comparator Check: inst_delta_z vs coinvest_score_z

**Rationale:** Both are institutional-flow lane signals (same comparator pairing used in the 2026-05-04 precedent case).

### Methodology fidelity
- Rerun `inst_delta_z` mean_ic = **-0.0010** vs dashboard mean_ic = **-0.0010** → delta = 0.0000 (exact match)
- Fidelity check: **PASSED**

### Results (identical window: 2026-06-03 → 2026-08-14, horizon=20d, n_snapshots=50)

| Signal | n_dates | Mean IC | Hit Rate | Latest IC |
|---|---|---|---|---|
| inst_delta_z (flagged) | 41 | -0.0010 | 58.54% | +0.0137 |
| coinvest_score_z (comparator) | 50 | +0.0294 | 58.00% | -0.2928 |

### Co-trajectory
- Common dates: 41
- Spearman ρ = **+0.9216** (p < 0.0001)
- Pearson r = +0.9003
- **Interpretation: shared_regime**

## Diagnosis

Per the skill's interpretation rule (both signals' IC trajectories move together, ρ > +0.5 → shared regime):

- `inst_delta_z` degraded sharply from late-July through mid-August (IC trough -0.26 on 2026-08-06), then recovered to +0.0137 by 2026-08-14.
- `coinvest_score_z` traced the **same** decay-and-partial-recovery pattern over the identical window (trough -0.31 on 2026-08-06), with high co-movement (ρ=0.92).
- This indicates the institutional-flow lane as a whole experienced a broad regime shift in the Jun–Aug window, not an isolated failure of `inst_delta_z` alone. Both signals went deeply negative together, consistent with "shared regime" rather than "signal-specific."
- Note: `coinvest_score_z`'s own latest-date reading (-0.2928 on 2026-08-14) is itself negative and worse than `inst_delta_z`'s latest (+0.0137), so neither signal looks clean at window-end — this is NOT the clean "comparator healthy, flagged broken" pattern seen in the 2026-05-04 inst_delta_z case (where coinvest_score_z probe was +0.097 and ρ was -0.33, anti-correlated/signal-specific).

## Fleet-Level Summary

| Signal Pair | Finding | Status |
|---|---|---|
| inst_delta_z ↔ coinvest_score_z | shared_regime (ρ=+0.92) | Checked — institutional lane broadly degraded in Jun–Aug window, not inst_delta_z-specific |
| score_rank_pct | — | Not checked (health=WEAK, below WARN/ALERT sweep threshold) |

**Signals healthy:** none in-scope this week (only one signal met the WARN/ALERT bar, and it returned a shared-regime finding rather than a clean-health comparator)
**Signals with findings:** inst_delta_z (shared_regime)
**Signals with insufficient data:** none

## What This Does NOT Prove

- Does not establish causality for the Jun–Aug institutional-lane degradation.
- Does not recommend a weight change, signal removal, or ruleset edit — this is evidence only; governance decision belongs to the operator.
- Does not cover `score_rank_pct` (WEAK, not WARN/ALERT) or any signal outside the two tracked in the current dashboard (`clinical_optionality_pct_dev`, `clinical_score_v2_z` not in dashboard's signals set this cycle).

## Artifacts

- `artifacts/ic_dashboard/2026-10-26_shared_regime_check.json` (full per-date IC series + co-trajectory stats)
- Source dashboard: `artifacts/ic_dashboard/2026-10-20_dashboard.json`
- Script used: `~/.hermes/skills/data-science/signal-shared-regime-check/scripts/shared_regime_check.py` (unmodified, executed via system `/usr/bin/python3` — scipy 1.11.4)

## Provenance

- Run: 2026-10-26, Hermes cron job (`weekly-signal-regime-sweep`)
- Skill: `signal-shared-regime-check` v (SKILL.md unmodified)
- No writes to rankings.csv, ranker_engine.py, decision_engine.py, configs, or rulesets.
