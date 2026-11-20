# `de_vol_60d` refreshed on every run — input fix before the selector-only clock starts (2026-11-15)

**Status:** operator-requested 2026-11-15. Merging the PR that carries this record is the authorization. It is a model-input change, so it is recorded here rather than shipped as a silent bug fix.

**Change:** `run_screen.py::_hydrate_beta_rsi`, Step C. `vol_60d` is now recomputed from `price_history.csv` on every run (overwrite-always, the policy beta, alpha and RSI already use). A stored value is kept only when the price series is too short to recompute (fewer than 31 closes).

No weight, signal, selection rule, order, sizing or eligibility rule changes. `model_hash` is unchanged (`a9a430154568e242`, `ast-v2`; checked on the branch and in the production checkout), so there is no hash rotation.

## What was wrong

- **Step C kept stale values.** It only *filled* a missing `vol_60d`. A value already present on the universe record's `defensive_features` was kept indefinitely. The comment gave the reason as "so we don't perturb the 272 tickers that already carry a value under the model freeze".
- **Live snapshots:** `de_vol_60d` is identical on 2026-08-03 and 2026-11-12 for **244 of 287** tickers. For example, PSNL reads 1.0777, AVTR 0.5928, AMPH 0.4149 and VRTX 0.2519 on every date. `de_beta_xbi_60d` changes daily for all of them.
- **Historical snapshots (`data/snapshots_pit_v5`):** the same field is identical for 150 of 158 tickers between 2020-01-31 and 2022-01-31, and for 230 of 272 between 2024-01-31 and 2026-06-26. Historical runs therefore used one fixed, later-dated volatility per stock, which is a small look-ahead in the selector's market-structure block.

## Where the field is used

| Consumer | Use | Effect |
|---|---|---|
| `selector_engine.py` market-structure block | `de_vol_60d`, weight 0.30, lower is better | The block has weight 0.10 in `A4_SELECTOR_CONFIG`, so about 3% of `selector_score` |
| `decision_engine.py` overlays | `high_vol` risk flag | Lowers the size tier only. With an equal-weight Top-30 it does not change holdings |
| `ranker_v2_pairwise.py` | Feature of the shadow ranker | Shadow record only |

## Measured impact

Method:
- Recompute `selector_score` with `A4_SELECTOR_CONFIG` on each snapshot's rows, with the stored `de_vol_60d` and with a fresh value computed by the Step C formula. The fresh value uses closes on or before the snapshot date, the production outlier filter, the last 60 simple returns and sample standard deviation.
- The recomputation reproduces production `selector_score` exactly on 2026-11-12 (rank correlation 1.000000, max difference 0).

**Live, selector-only order:**

| Sample | Top-30 overlap, fresh vs frozen |
|---|---|
| 2026-11-12 | 30/30 (all-name rank correlation 0.998) |
| 30 daily snapshots, 2026-10-01 → 2026-11-12 | mean 29.0, min 28. 0 names changed on 7 days, 1 on 17 days, 2 on 6 days |

The usual swap is AMLX out (19 of 30 days), replaced by ERAS, ORKA, ARWR or STOK.

**History, 84 monthly periods (v5 plus live month-ends to 2026-11-12, net of 50 bps):**

| | Frozen vol | Point-in-time vol | Difference |
|---|---|---|---|
| Top-30 overlap | — | mean 29.4/30, min 28 | — |
| Excess vs XBI, pt/month (t) | +1.23 (2.32) | +1.22 (2.28) | −0.01 (t −0.13) |

The historical look-ahead did not move the backtest. The selector's market-structure block taken on its own should not be cited from that history.

## Why now

- **No reset cost now.** `oos_clock_start` for the selector-only order has not been recorded; it is expected after the first promoted run on 2026-11-16 (see `2026-11-15-pre-clock-decisions.md`). A fix that lands before that run becomes part of the window's starting configuration and costs no reset. After the clock starts, the same fix would reset it.
- **If this merges after the 2026-11-16 run has started,** `oos_clock_start` is the first promoted run on the merged code (expected 2026-11-17), not 2026-11-16.
- **Drift:** one or two Top-30 names on top of the selector-only switch. The dated drift override already in force (2026-11-16 to 2026-11-20, against the 2026-11-12 baseline) covers it; no new override entry.

## Point-in-time check

Only closes dated on or before `as_of_date` are read. That date filter already exists in `_hydrate_beta_rsi` and is unchanged; a new test confirms that a price crash after `as_of_date` leaves `vol_60d` unchanged.

## Tests

`tests/test_hydrate_drawdown.py::TestVol60dHydration`:
- a stored stale value is overwritten, and equals a fresh fill;
- a stored value survives when the series is too short;
- the leakage guard.

Full suite: 18,836 passed. 2 local-only fleet-audit failures, the same on unmodified main and passing under `CI=true`.

## Rollback

Revert the commit. Step C then preserves stored values again.

## Not changed

The stored `defensive_features.vol_60d` values on the universe records are not rewritten. They are now overwritten in memory on every run where the price series allows it.
