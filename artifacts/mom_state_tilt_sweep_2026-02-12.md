# Momentum State Weight Tilt — Sweep Results

**Date**: 2026-02-12
**Base ruleset**: a021df60 (v1.2.1_candidate, a_floor=0.60, catalyst_near=120, catalyst_mid=180)
**Tilt configs**: neutral penalty = {0.80, 0.85, 0.90, 0.95}, tailwind=1.0, headwind=1.0

## Context

The 2024 inversion attribution found `mom_state=neutral` as the #1 driver of AB-D
spread inversion (-40pp in 2024, flips to +8pp in 2025). This sweep tests whether
penalizing neutral-state weights at the L3 sizing layer improves portfolio-weighted
returns without changing membership (tier assignments are unaffected).

## 2025-Only Results (10 snapshots, 1808 panel rows)

| Config           | Mean Wt Ret 60d | TW wt  | NU wt  | HW wt  | Jaccard |
|------------------|----------------:|-------:|-------:|-------:|--------:|
| baseline (off)   |       +3682.40% |  6.11% |  5.71% |  3.85% |   0.654 |
| neutral=0.80     |       +3628.05% |  6.63% |  5.29% |  4.42% |   0.654 |
| neutral=0.85     |       +3643.12% |  6.49% |  5.40% |  4.26% |   0.654 |
| neutral=0.90     |       +3657.10% |  6.35% |  5.51% |  4.11% |   0.654 |
| neutral=0.95     |       +3670.04% |  6.23% |  5.61% |  3.98% |   0.654 |

## Full Panel Results (22 snapshots, 3922 panel rows, 2024+2025)

| Config           | Mean Wt Ret 60d | TW wt  | NU wt  | HW wt  | Jaccard |
|------------------|----------------:|-------:|-------:|-------:|--------:|
| neutral=0.80     |       +1552.08% |  6.61% |  4.85% |  4.17% |   0.656 |
| neutral=0.85     |       +1561.15% |  6.44% |  4.97% |  4.04% |   0.656 |
| neutral=0.90     |       +1569.59% |  6.29% |  5.09% |  3.91% |   0.656 |
| neutral=0.95     |       +1577.61% |  6.14% |  5.21% |  3.80% |   0.656 |

## Key Findings

1. **Baseline outperforms all tilt variants** on 2025-only weighted returns
   (+3682% vs +3628% at most aggressive n=0.80). The neutral penalty reduces
   total portfolio-weighted returns across all tested values.

2. **Weight redistribution works as designed**: neutral-state weights decrease
   (5.71% → 5.29% at n=0.80) and redistribute to tailwind (6.11% → 6.63%)
   and headwind (3.85% → 4.42%) via normalization.

3. **Membership is preserved**: Jaccard stability identical across all configs
   (0.654 for 2025, 0.656 for full). Tier assignments are unaffected.

4. **Same pattern on full panel**: higher neutral penalty → lower returns
   monotonically. The 2024 inversion was a *membership* problem (neutral
   tickers in wrong tiers), not a *weighting* problem. L3-only tilt cannot
   fix the membership issue.

5. **Tier separation identical**: all configs produce the same per-tier mean
   returns because the tilt only affects portfolio weights, not which tickers
   appear in each tier.

## Recommendation

**Do not enable momentum state tilt.** The infrastructure is in place
(`enable_mom_state_tilt` + `mom_state_tilt_mults`) but the sweep shows no
return improvement at any neutral penalty level tested. The 2024 neutral
inversion is a tier-membership problem better addressed by catalyst/optionality
parameters, not by L3 weight modifiers.

Keep `enable_mom_state_tilt=False` in production ruleset a021df60.

## Artifacts

- `walkforward_panel_baseline_2025.csv` — baseline panel (1808 rows)
- `walkforward_panel_mom_n0{80,85,90,95}_2025.csv` — 2025-only panels
- `walkforward_panel_mom_n0{80,85,90,95}_full.csv` — full panels (3922 rows)
- `walkforward_report_*_2025.json` / `.md` — per-config reports
- `walkforward_report_*_full.json` / `.md` — per-config full reports
- `mom_tilt_n0{80,85,90,95}.json` — candidate ruleset JSONs
