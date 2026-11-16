# PIT financials: cash and marketable securities read from every XBRL tag — input fix before the selector-only clock starts (2026-11-16)

**Status:** operator-authorized 2026-11-16 ("fix before 16:30 today"). Merging the PR that carries this record is the authorization. It is a model-input change, recorded here rather than shipped as a silent bug fix. If merged and deployed before the 2026-11-16 16:30 ET run, it is part of the selector-only starting configuration and `oos_clock_start` is recorded from that run; otherwise the clock starts at the first promoted run on the merged code.

**Change:** how the point-in-time financial store (`production_data/pit_financials/`, read by `run_screen.py` whenever `pit_mode != off`) records liquid assets.

- `tools/build_pit_financials.py`: `cash` and `short_term_investments` are now collected from **every** tag in `CASH_TAGS` / `CURRENT_INVESTMENT_TAGS`, each fact carrying its `tag`. Before, the builder kept only the first tag that had any facts at all.
- `pit_financials.py`:
  - Cash = latest balance-sheet date across cash tags; on that date plain cash outranks cash incl. restricted cash (IFRS `CashAndCashEquivalents` outranks the narrower IFRS `Cash` line).
  - Current securities are added only from the **same** balance-sheet date as cash, taking the **max** across tags (different tags usually describe the same portfolio; the static collector's no-double-count rule).
- The 364 store files were rebuilt from the cached SEC companyfacts already on disk (fetched 2026-11-09/11). Only the `cash` and `short_term_investments` series changed; every other fact and `collected_at` is byte-identical (checked file by file).
- Also in this PR: `tools/fetch_pending_biotech_data.py` and `tools/refresh_eligible_biotech_universe.py` write `universe.json` without NaN. On 2026-11-16 08:30 the weekly refresh wrote `"price": NaN` for KALA; `run_screen.py` loads `universe.json` with orjson, which rejects NaN, so the 16:30 run could not have started. The live file was repaired by hand the same morning (that one value set to `null`; backup kept).

No weight, signal, selection rule, order, sizing or eligibility rule changes. `model_hash` is unchanged (`a9a430154568e242`, `ast-v2`), so there is no hash rotation.

## What was wrong

- **Securities never reached the store for most filers.** Companies report their treasury portfolio as `AvailableForSaleSecuritiesDebtSecuritiesCurrent` (CYTK, SYRE, COGT, VRDN, MRNA, KYMR) or `DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent` (GRAIL), which the builder did not read; where an older tag had any facts it shadowed the current one (DNTH's `ShortTermInvestments`). The store then held cash only:

  | Ticker | PIT liquid assets before | After | Static `financial_data.json` |
  |---|---|---|---|
  | CYTK | $203M | $1,130M | $1,130M |
  | SYRE | $121M | $1,072M | $1,072M |
  | COGT | $224M | $791M | $791M |
  | GRAL | $29M | $789M | $29M (static is wrong too) |

- **Stale cash for 10 filers** that changed cash tags: BIIB and ILMN read 2018–2019 balance sheets, GILD and NTRA 2022, ARWR and TXG 2024.
- Effect: 51 of 320 scored names were understated by more than 30%. Understated runway pushed names into `sev3` (excluded at module 5) or the `survivability_critical` red flag (ineligible). On 2026-11-12, 14 universe names were excluded by `sev3`, among them CYTK and SYRE; COGT, VRDN and MRNA were ineligible on `survivability_critical`.
- It is long-standing: snapshots from 2026-10-30 onward show `liquid_components=['Cash']` for these names. Historical regenerations that staged this store (v5/v6) likely carry it too; not measured here.

## Measured impact

Method: `run_screen.py` for 2026-11-13 on a copy of today's production inputs, run twice from the same code, once with the old store (control) and once with the rebuilt store (treatment), under `LiveStateGuard`. Same flags as production phase-2 except `--diagnostics none`; caches copied from the production checkout.

| | Control | Treatment |
|---|---|---|
| Eligible names | 151 | 163 |
| Excluded by `sev3` gate (module 5) | 31 | 5 |
| `sev2` names in rankings | 38 | 20 |
| `survivability_critical` red flags | 14 | 7 |
| Names losing eligibility | — | **none** |

**Top-30 overlap 25/30.**

- In: COGT (rank 2), VRDN (8), DNTH (9), CYTK (10), SYRE (17). These are high-conviction co-investment names (SYRE is 6th and COGT 5th in the universe on tier-1 conviction overlap) that were excluded only because their cash was understated.
- Out: RYTM, ORKA, VOR, ASND, DSGN, which were ranks 23–30 and become 31–37.
- Newly eligible: COGT, DNTH, VRDN, SYRE, CYTK, GRAL, QSI, AGIO, MRNA, PRME, OVID, DNA.

Of the 8 names whose cash went down, all move from a stale balance sheet (2018–2025) to the current one.

## Not done

- No historical backtest of the change.
- The static `financial_data.json` collector has the same tag gap for some names (GRAL). It is only a fallback when the PIT store is present, so it is not fixed here.
- The universe-maintenance report still flags delisted/acquired names as stale; cosmetic.
