# Split confirmation: one 15% window for momentum, drawdown, beta, and RSI (2026-11-20)

**Classification:** Tier 3 model-input bug fix (`DATA_CORRECTION` of a price adjustment). Not a weight, signal, ranker, selector-formula, sizing, or eligibility-rule change. Ruleset `8887576e` is unchanged.
**Status:** Open pull request against `main` (`ab0ad565`). Not merged. Merging this record is the operator's authorization to land the fix.
**Hash:** `model_hash` `a9a430154568e242` (`ast-v2`) is unchanged. The fingerprint covers `ranker_engine.py`, `selector_engine.py`, `decision_engine.py`, `common/ranker_mode.py`, and the `A4_SELECTOR_CONFIG` assignment in `run_screen.py`. This diff does not touch those. No entry in `governance/HASH_ROTATIONS.md`.
**Clock:** see "Clock" below. `oos_clock_start` is still **2026-11-17**. This PR does not edit `artifacts/forward_validation/CANDIDATE.json`.

## Decision

Confirm a registry split against the price series with one relative window, `abs(observed / factor - 1) <= 0.15`, and use that confirmed registry factor in momentum, drawdown, beta, alpha, RSI, and the 60-day volatility that shares the beta series.

## Evidence

- Momentum (`run_screen.py`, `_observed_registry_split_factor`) confirmed a split when `abs(log(observed) - log(factor)) <= log(2)`. For AZN's factor 2.0 (`production_data/corporate_actions.json`, ADR ratio change 2026-01-28), a flat already-adjusted series has `abs(log(1) - log(2)) == log(2)`, so the check matched and the lookback was multiplied by 2 again. Any 20/60/120-day window that crossed the effective date then showed about −50%.
- The price-file repair already uses 15% (`scripts/repair_price_history_splits.py`, previously `REGISTRY_FACTOR_TOLERANCE = 0.15`). `tests/test_split_registry_confirmation.py` only covered a 1:20 split, where `log(20)` is far outside `log(2)`, so the flat-series miss did not fail.
- Drawdown and beta/RSI only dropped days beyond +300% / −75%. AZN's documented +96% day (95.60 → 187.80) stayed in those series. Beta, alpha, RSI, and `vol_60d` are computed from that same series.

Red tests before the fix: flat factor-2.0 momentum returned −0.50; the drawdown fixture stayed at the raw −0.10 instead of the adjusted −0.55; beta hydration did not take a registry.

## Reasoning

A flat series is not evidence of a split. The repair script's 15% relative window already encoded that: ratio 1 versus factor 2.0 is a 50% error. Sharing that predicate (`registry_split_confirmed` in `common/corporate_actions.py`) keeps the repair and the scorer from drifting. When the adjacent bars do match, the scorer multiplies by the registry factor, which is what momentum already did for a 1:20 split. Drawdown, beta, RSI, and `vol_60d` need the same scale so a confirmed jump is not a return and an unconfirmed crash is still only dropped by the +300% / −75% filter.

Point-in-time: splits with `effective_date` after `as_of_date` stay invisible, and the price series is already cut at `as_of_date`. No `datetime.now()`.

## Boundary conditions

- The window is relative, not log. A stitched jump more than 15% away from the registry factor is not confirmed. SKYE 2026-08-17 is the current case: notes record 0.5015 → 3.32 (ratio 6.62) against factor 8.0, relative error 0.172. The old log check confirmed it (`abs(log(6.62) - log(8)) ≈ 0.19 <= log(2)`). Momentum no longer applies ×8 there. Drawdown for SKYE still uses the existing short-history fallback, which scales by the registry factor without this window, so that fallback is unchanged.
- Every other split whose notes record the jump stays inside 15%: AZN 95.60 → 187.80 (error 0.018), IMMP 0.26 → 4.71 (error 0.094), KALA 0.083 → 3.95 (error 0.048), GOSS 0.1542 → 11.20 (error 0.092). A flat series confirms none of them. AZN factor 2.0 is the only current registry factor for which a flat series used to confirm.
- The repair script still applies the observed ratio when it rewrites `price_history_split_adj.csv`. The scorer applies the registry factor, so a day that is 1.8% off (AZN) remains a small residual return. That matches the existing 1:20 momentum path. This PR does not retarget the repair onto the registry factor.
- Unconfirmed moves, including a real crash with no registry entry, are not scaled. The +300% / −75% drop still applies to whatever is left.
- This does not change selector block weights, the ranker, sizing, or the drawdown-gate thresholds. It changes the prices those gates and the market-structure block read.

## What changed

| File | Change |
|---|---|
| `common/corporate_actions.py` | `REGISTRY_FACTOR_TOLERANCE = 0.15`, `registry_split_relative_error`, `registry_split_confirmed`. One definition. |
| `scripts/repair_price_history_splits.py` | The small-ratio search calls that helper. The local `0.15` constant is gone. |
| `run_screen.py` | Momentum uses the helper. `_apply_confirmed_registry_splits` scales a series by each confirmed registry factor and is called from drawdown and from beta/RSI (which also feeds alpha and `vol_60d`) before the outlier filter. |
| `tests/test_split_registry_confirmation.py` | Factor 2.0, jumped and flat, for momentum, drawdown, and beta. |

## Measured impact

`production_data/price_history.csv` is gitignored and is not in this checkout, so the screen was not run with and without the fix. No live rank table. What follows is the registry arithmetic on the ratios written in `corporate_actions.json`, for an as-of date of 2026-11-20.

Windows from that date, counting weekdays: 60 trading days reach 2026-08-28; 120 trading days reach 2026-06-05; 252 trading days reach 2025-12-03. Momentum's own lookback is `int(days * 365 / 252)` calendar days (60 → 2026-08-26, 120 → 2026-05-31), which does not change who falls inside.

| Ticker | What moves on the raw file at 2026-11-20 | What does not |
|---|---|---|
| AZN | 252-day drawdown. The +96% day was inside the window and under the outlier thresholds, so it stayed in the peak path. It is now scaled by 2.0. `de_drawdown` is 0.25 of the market-structure block, and that block is 0.10 of `A4_SELECTOR_CONFIG`. The drawdown gate can also change eligibility if the adjusted value crosses it. | 20/60/120-day momentum and 60-day beta, alpha, RSI, and `vol_60d`. Those windows do not contain 2026-01-28. On the raw jump (ratio 1.964) both the old and new checks confirm, so a momentum window that does contain the date still adjusts once. The −50% path is the flat, already-adjusted series, which production notes say `price_history.csv` is not. |
| KALA | 252-day drawdown. Post-split bars from 2026-05-08 exceed 126 by mid-November, so the old path kept only the post-split regime. The new path keeps adjusted pre-split history. | 60-day beta, RSI, and vol. The split is before that window. Momentum 120-day also starts after 2026-05-08. |
| IMMP | 60-day beta, alpha, RSI, and `vol_60d`. 2026-09-28 sits inside that window. The old outlier filter dropped pre-split bars; the new series keeps them on a ×20 basis. | Momentum: 0.26 → 4.71 is inside 15% of 20, so both checks adjust. Drawdown: fewer than 126 post-split bars, so the old short-history fallback already scaled by the registry factor. |
| GOSS | Nothing expected at this as-of. 0.1542 → 11.20 is inside 15% of 80. Post-split history is still short, so drawdown stays on the registry-scaled fallback. The 60-day window starts after 2026-08-17. | |
| SKYE | `return_120d`, if that window is filled from price history. 0.5015 → 3.32 is 17% from factor 8, so momentum stops applying ×8. The unadjusted day is about +562%; the old adjustment left about −17%. | Drawdown stays on the short-history fallback (scales by 8 without this window). The 60-day beta window starts after 2026-08-17. |

Selector order is `selector_only`. The market-structure block is the path from these fields into `selector_score`. Institutional weight is 0.65 on `coinvest_score_z` alone, so a drawdown or beta change moves a name inside a 10% block, and moves the book only if that change reorders the top 30 or flips the drawdown gate. That reorder was not measured here.

Operator rerun, once `production_data/price_history.csv` is on disk: run `run_phase2_daily.py --as-of-date <date>` on `main` and on this branch into two empty snapshot directories, and diff `actionable_rank`. Do not overwrite an existing snapshot directory.

## Clock

`docs/FORWARD_VALIDATION_PROTOCOL.md` §1: an authorized bug fix during the NO_MODEL_CHANGE window resets the out-of-sample clock. This fix can change a selector input (drawdown, and beta/RSI/`vol_60d` when the split sits in the 60-day window) and can change eligibility through the drawdown gate. It is not the 2026-11-18 AZN case. That ruling was "no reset" because the repair changed no current score (`docs/governance/2026-11-18-azn-adr-ratio-split-fix.md`).

This PR does not move `oos_clock_start`. On merge, unless the operator rules otherwise, `oos_clock_start` becomes the first promoted production run on this code, and mandate SM-20260629-001 restarts at 0/20. Captures from 2026-11-17 through the day before that run stay in the ledger and stop counting.
