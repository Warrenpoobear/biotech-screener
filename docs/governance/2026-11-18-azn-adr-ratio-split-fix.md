# AZN ADR ratio change missing from the split-adjusted price file (2026-11-18)

**Classification:** `DATA_CORRECTION` (Tier 3: a price input during the NO_MODEL_CHANGE window).
**Status:** PROPOSED in PR #712. Merging it is the operator's decision; the daily run then regenerates `price_history_split_adj.csv` from the corrected registry.
**Measured effect on the current ranking:** none (see "Effect").
**Clock:** operator decision pending (see "Clock").

## Finding

AstraZeneca changed its ADR ratio: one ADR went from half an ordinary share to one share. The quoted price doubled overnight. Sharadar `ACTIONS` records it as `adrratiosplit` 2 (`split` 0.5) dated 2026-02-02; in `price_history.csv` the doubling falls on 2026-01-28 (95.60 → 187.80).

`scripts/repair_price_history_splits.py` only looks at one-day moves beyond +300% or −75%, then keeps those confirmed by `corporate_actions.json`. A +96% move never reached the registry check, and AZN had no registry entry. So `price_history_split_adj.csv` records a +96% one-day return that did not happen.

Found while backtesting the LVB sleeve (`scripts/research/lvb_universe_backtest.py`, PR #714). LVB held AZN in January 2026, and the false return added 3.4 points to the sleeve's return that month.

## Change (PR #712)

| File | Change |
|---|---|
| `scripts/repair_price_history_splits.py` | New `find_registry_splits_below_threshold`. A registry split with no confirmed gap event is applied when a one-day move within the 5-day confirmation window matches its factor within 15%. |
| `production_data/corporate_actions.json` | AZN `reverse_split`, ratio 1:2, factor 2.0, effective 2026-01-28 (the date of the move in our feed; Sharadar dates the action 2026-02-02, inside the window). |
| `tests/test_split_registry_small_ratio.py` | Applied; registry entry with a mismatched move not applied; unregistered doubling left as a return; registry entry present. Written first and red; 50/50 split tests pass. |

Safeguards are unchanged:
- **No registry entry, no adjustment.** Unregistered moves of any size stay as returns, as set by the 2026-11-13 false-splits correction (`2026-11-13-corporate-actions-false-splits.md`).
- **No matching move, no adjustment.** A registry entry alone never rescales a series, so a source that is already adjusted is not adjusted twice.

## Effect

Regenerating from today's `price_history.csv` with the corrected registry:

| | Rows changed | Tickers |
|---|---|---|
| `price_history_split_adj.csv` | 1,525 (AZN, 2020-01-02 → 2026-01-27, ×1.964) | AZN only |

No other registry entry newly matches under the new rule.

**Current scoring is unchanged:**
- `run_screen.py` reads raw `price_history.csv` plus the registry for momentum (20/60/120-day windows, none reaching 2026-01-28) and for the 252-day drawdown. AZN's drawdown is −0.1299 with and without the entry.
- The split-adjusted file feeds the Path C drawdown (baseline 2026-05-29) and the forward shadows, whose windows start after the change.
- `model_hash` is unchanged (no engine file touched), so there is no hash rotation.

**Historical research:** any backtest that held AZN across 2026-01-28 on the split-adjusted file is overstated for that month. The LVB backtest is the one known case.

## Clock

The protocol resets the out-of-sample clock for any authorized bug fix. This fix changes no current score, order or holding, and the forward-validation captures store baskets inline. **Recommendation: record it as a data correction without a reset,** the same way the 2026-11-13 registry correction was proposed. The operator decides.

## Related: JBIO registry entry is mislabelled (not changed here)

`corporate_actions.json` lists JBIO as a `forward_split` 9:1 (factor 0.108) on 2025-04-29, with the note "Second split". Sharadar `ACTIONS` shows what happened that day: Aerovate (AVTE) paid a **$2.40 special cash dividend** on a $2.68 stock, did a **1:35 reverse split**, and became Jade Biosciences (JBIO).

- **Economic outcome:** a holder of 35 old shares (worth $93.80 on the adjusted scale) received $84.00 in cash plus one new share at $10.14, about 98% of the value. The true return is close to zero.
- **Production file:** the mislabelled "split" factor (0.108) happens to match 10.14 / 93.80, so the file shows about 0% for the move. The number is roughly right; the label is wrong.
- **Sharadar `closeadj`:** does not reflect the dividend and books a −87% loss. Research runs on Sharadar prices are understated for that month. LVB held JBIO in April 2025, which costs the Sharadar LVB backtest about 2.9 points that month (about 0.04 pt/month over 82 months).

**Proposed follow-up (separate change):** replace the JBIO entry with a `special_dividend` of $2.40 plus a `reverse_split` 1:35, both on 2025-04-29. Then check the result against the repair script's dividend rule before merging.

## Records

- PR #712 (origin). The mirror PR waits for the `main` sync: the two `main` branches have diverged and the mirror lacks the registry-confirmation logic this builds on.
- `docs/MODEL_DOCUMENTATION.md`, 2026-11-18 entry.
