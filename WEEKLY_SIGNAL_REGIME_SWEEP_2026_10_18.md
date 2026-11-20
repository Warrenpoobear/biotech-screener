# Weekly Signal Regime Sweep — 2026-10-18

## Facts

- Latest IC dashboard: `artifacts/ic_dashboard/2026-10-16_dashboard.json` (as_of_date 2026-10-16, generated 2026-10-16T20:47:25Z)
- Dashboard window: 2026-05-28 → 2026-10-15, lookback=60, horizon=20
- Dashboard-level attention flag: **LOW**
- Load-bearing signals evaluated and their health classification (per `tools/build_ic_dashboard.py` thresholds: HEALTHY ≥ 0.03 implied by `IC_HEALTHY`, WEAK in [0.00, IC_HEALTHY), WARN in [-0.03, 0.00), ALERT < -0.03):

| Signal | mean_ic | hit_rate | n_dates | health |
|---|---|---|---|---|
| score_rank_pct | +0.0262 | 0.8704 | 54 | WEAK |
| inst_delta_z | +0.0039 | — | 42 | WEAK |

- No signal in the latest dashboard carries a `WARN` or `ALERT` health label.

## Inference

- Per the task mandate ("Run the signal-shared-regime-check skill against EACH load-bearing signal... flagged WARN or ALERT"), the trigger condition for running comparator probes is not met this week.
- Both tracked signals are in the WEAK band (non-negative mean IC, below the HEALTHY cutoff but above the WARN threshold of 0.00). WEAK is not a trigger condition per the skill's `when_to_use` criteria (SIGNAL_ALERT / WARN only).
- No shared-regime or anti-correlation comparator runs were executed this week — there is nothing to pair/diagnose.

## Options

Not applicable — no governance decision is pending. No signal requires a weight change, zero-out, or further isolation this cycle.

## What this sweep does NOT prove

- Does not assert the signals are healthy in an absolute sense — WEAK means marginal positive IC, not strong. This sweep only confirms neither signal crossed into WARN/ALERT territory requiring comparator diagnosis.
- Does not evaluate signals outside the two tracked in the current dashboard (`score_rank_pct`, `inst_delta_z`). If other load-bearing signals (e.g., `clinical_optionality_pct_dev`, `coinvest_score_z`) are active in production but absent from this dashboard snapshot, they were not assessed here.

## Next step

None required. Resume normal weekly cadence; re-run this sweep if `ic_health_monitor` raises a WARN/ALERT on any load-bearing signal before the next scheduled sweep.

## Provenance

- Source dashboard: `artifacts/ic_dashboard/2026-10-16_dashboard.json`
- Health thresholds: `tools/build_ic_dashboard.py` (`IC_WARN = 0.00`, `IC_ALERT = -0.03`)
- Skill reference: `signal-shared-regime-check` (`~/.hermes/skills/data-science/signal-shared-regime-check/SKILL.md`)
- Diagnose-only run. No production files (rankings.csv, run_screen.py, decision_rulesets, configs) were modified.
- No new artifact written to `artifacts/ic_dashboard/` this cycle (no comparator probe was executed — trigger condition not met).
