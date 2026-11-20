# Manager roster v4: backtest lead is in-sample — research record (2026-11-16)

**Status:** research record, recorded at the operator's instruction on 2026-11-16. It changes nothing in production: no roster, weight, signal, order, sizing or eligibility change, and no clock effect. `model_hash a9a430154568e242` unchanged.

**Why this is recorded:** registry v4.0 (the 30-manager `elite_core` roster, 2026-11-01) was chosen on trailing three-year returns. `docs/governance/2026-11-01-coinvest-13f-integrity-clock-reset.md` says backtests over that window are selection-biased. This record measures how much, so that later Top-30 backtest figures are quoted with the right caveat.

## Test

Two regenerated histories that differ **only** in the manager roster used to build `coinvest_signals.json`:

| History | Roster | Managers per date |
|---|---|---|
| `snapshots_pit_v7` | registry v4.0 (new, 30 `elite_core`) | 24–33 |
| `snapshots_pit_v8` | registry v3.2 (old, 49 `elite_core`) | 39–50 |

Same code (with PR #682 cash/securities fix), same staged inputs, same PIT 13F cache, same builder flags (`--nearest-prior-days 95`, per-date universe). Top-30 by `actionable_rank` (selector-only), monthly snapshot-to-snapshot holding, equal weight, split-adjusted closes, net of 50 bps one-way on replaced names. 77 monthly periods, 2020-03 → 2026-06. Average Top-30 overlap between the two rosters: about 21/30.

## Result (mean monthly excess return vs XBI, net)

| Window | New roster (v4) | Old roster (v3.2) | Difference |
|---|---|---|---|
| Mar 2020 – Oct 2023 (before the selection window, 43 months) | +0.61 pt (t 1.06) | +0.52 pt (t 0.93) | +0.09 pt (t 0.24) |
| Nov 2023 – Jun 2026 (selection window, 34 months) | +2.24 pt (t 2.52) | +1.07 pt (t 1.19) | +1.17 pt (t 1.73) |
| Full 77 months | +1.33 pt (t 2.60) | +0.76 pt (t 1.51) | — |

## Reading

- **Before the window it was picked on, the new roster is indistinguishable from the old one** (+0.09 pt/month, t 0.24).
- **Its whole lead is inside the window used to choose it.** That is the signature of selection on outcomes, not of a better roster.
- Every history regenerated since 2026-11-01 uses the v4 roster: `snapshots_pit_v5`, `v6`, `v7` (and the v7 copy written into `data/snapshots/` on 2026-11-16), and the low-vol sleeve blend charts built on them. **Their Top-30 figures for 2023–2026 are inflated by roster hindsight.**
- A fairer reading of the strategy's historical edge is the old-roster figure, roughly **+0.5 to +0.8 pt/month, not statistically significant**. This agrees with the 2026-11-15 manager-list test (the v4 list sat at the 98th percentile of random 30-manager lists; about 0.4–0.5 pt/month of hindsight).

## What this does and does not mean

- It does **not** say the v4 roster is worse going forward. Out of sample the two are the same, and the live roster is not changed by this record.
- Quote the old-roster figure alongside any v5/v6/v7 Top-30 backtest. Do not cite 2023–2026 Top-30 history on the v4 roster as evidence.
- The forward test (mandate SM-20260629-001, clock started 2026-11-16) remains the only out-of-sample evidence.

## Reproduction

- Histories: `data/snapshots_pit_v7` and `data/snapshots_pit_v8_old_roster` (copied from the research worktree on 2026-11-16; untracked).
- The v8 staging rebuilt `coinvest_signals.json` with `scripts/build_coinvest_features_from_13f.py` after swapping in registry v3.2 (worktree only, restored afterwards).
