# Corporate-actions registry: 26 real price moves registered as splits (2026-11-13)

**Classification:** `DATA_CORRECTION`. `production_data/corporate_actions.json` is read by the daily run (split-adjusted momentum and drawdown, the split-adjusted price file).
**Status:** PROPOSED. Not deployed. Merging and pulling is the operator's decision.
**Measured effect on the current ranking:** none (see "Effect").

## Finding

The registry was created on 2026-04-05 from detected one-day price discontinuities. 26 of its 37 split entries are not splits. They are trial and regulatory readouts: single-day moves of −75% to −93% or +330% to +1,446% on 40× to 8,000× normal volume. `scripts/repair_price_history_splits.py` adjusts only registry-confirmed discontinuities, so each false entry makes `price_history_split_adj.csv` rescale the ticker's earlier prices and erase the move.

Example: ABVX closed at $10.00 on 2025-07-22 and $68.60 on 2025-07-23 on 27.8M shares (46× its 20-day average). The registry called it a 1:7 reverse split "to maintain NASDAQ listing compliance". The adjusted series shows $68.60 on both days.

The same defect was found and fixed for one ticker on 2026-11-12 (IMMP 2026-03-13, commit `08400a1a`).

## Evidence

Checked against Sharadar `ACTIONS` (2019-06-01 → 2026-11-13) for every ticker with a split entry, and against the raw price and volume series.

| Ticker | Registry entry | Raw close before → after | Volume vs 20-day avg | Sharadar split on or near the date |
|---|---|---|---|---|
| ABVX | reverse 2025-07-23 | 10.00 → 68.60 | 47× | none |
| ACRS | forward 2023-11-13 | 4.76 → 0.64 | 67× | none (no rows) |
| AMLX | forward 2024-03-08 | 18.97 → 3.36 | 41× | none |
| BIOA | forward 2024-12-09 | 20.09 → 4.65 | 46× | none |
| COGT | reverse 2020-07-06 | 1.78 → 8.80 | 136× | none (real 1:4 on 2020-11-09) |
| DFTX | reverse 2020-03-03 | 0.75 → 3.45 | 2,641× | none (real split 2022-08-29) |
| DRUG | reverse 2024-10-15 | 2.49 → 38.49 | 7,880× | none (real split 2023-07-14) |
| GOSS | forward 2026-02-23 | 2.13 → 0.42 | 50× | none |
| IVVD | forward 2021-12-14 | 34.26 → 7.26 | 15× | none |
| JBIO | forward 2024-06-17 | 861.70 → 57.75 | 88× | none |
| KALA | forward 2025-09-29 | 19.05 → 2.05 | 109× | none |
| KOD | forward 2022-02-23 | 50.35 → 9.86 | 42× | none (no rows) |
| MCRB | reverse 2020-08-10 | 92.80 → 454.00 | 108× | none (real split 2025-04-22) |
| MENS | forward 2025-12-17 | 15.40 → 2.92 | 19× | none |
| MLTX | forward 2025-09-29 | 61.99 → 6.24 | 99× | none |
| NMRA | forward 2025-01-02 | 10.60 → 1.97 | 49× | none |
| ORKA | reverse 2020-05-28 | 25.49 → 123.98 | 5,867× | none (real split 2024-09-03) |
| PHVS | reverse 2022-12-08 | 2.51 → 11.46 | 1,117× | none |
| PRAX | forward 2022-06-06 | 128.85 → 28.20 | 76× | none (real split 2023-11-29) |
| PRQR | forward 2022-02-11 | 5.64 → 1.39 | 50× | none (no rows) |
| REPL | forward 2025-07-22 | 12.32 → 2.81 | 43× | none (no rows) |
| SRRK | reverse 2024-10-07 | 7.42 → 34.28 | 43× | none (no rows) |
| SYRE | reverse 2023-06-22 | 2.65 → 11.40 | 468× | none (real split 2023-09-08) |
| FGEN | forward 2023-06-26 | not in price file | — | none (no rows) |
| LYRA | forward 2024-05-06 | not in price file | — | none (real split 2025-05-28) |
| VTYX | forward 2023-11-07 | not in price file | — | none |

"No rows" means Sharadar holds no corporate action of any kind for the ticker in the window. Where Sharadar shows a real split on another date, the raw price series has no discontinuity there, so the vendor closes are already adjusted for it.

## Entries kept

AKTX 2026-03-31, IMMP 2026-09-28, KALA 2026-05-08, SKYE 2026-08-17, TECX 2024-06-21 and 2024-06-24, JBIO 2025-04-29, LYRA 2025-06-02: Sharadar shows a split within a week, and volume on the day is at or below normal.

Kept but not verified here:
- GOSS 2026-08-17: Sharadar dates the split 2026-09-11; the raw series jumps 73× on 2026-08-17 on near-zero volume.
- SUPN / INDV 2022-11-28: ticker reuse, a separate matter.
- JBIO 2025-04-29: Sharadar shows 1:35; the registry factor is 0.108.

## Effect

- **Current ranking: none.** The 2026-11-12 snapshot was run twice in an isolated copy, once with the current registry and adjusted price file and once with the corrected ones. The Top-30 is identical (30/30, and identical to the live 2026-11-12 Top-30), and no name's `de_drawdown`, eligibility, `de_alpha_60d` or `mom_state` changes.
- **Selection benchmark: none.** Rerun on `data/snapshots_pit_v4` with the corrected adjusted prices, every figure is unchanged, because the Top-30 did not hold any of these names in its event month. The closest were GOSS (rank 32, January 2026) and MLTX (rank 40, August 2025).
- **Not measured:** drawdown and momentum on historical dates close to each event, and any research panel that uses per-name returns across the full universe (IC panels, eligibility comparisons). Those include the erased moves.

## What changes

- `production_data/corporate_actions.json`: 26 entries removed (73 → 47).
- `tests/test_corporate_actions_registry.py`: the 26 pairs must not be registered as splits.

`production_data/price_history_split_adj.csv` is not tracked; the nightly repair step rebuilds it from the registry.

## Open

- Whether this counts as an input change under the forward-validation protocol. The measured effect on the 2026-11-12 ranking is zero, but the file feeds live price adjustment.
- The registry still has no check that a registered split is a split. A volume test on the day (a split does not come with 40× volume) or a Sharadar `ACTIONS` cross-check at registration would have caught all 26.
