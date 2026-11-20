# Out-of-sample clock start recorded: 2026-11-17

**Date:** 2026-11-17
**Classification:** Tier 4 governance record. No code, model, ranker, selector, eligibility, sizing or gate behavior changes.
**Authorization:** operator, 2026-11-17. PR #696 (coinvest CUSIP mappings plus RPRX, ILMN and STVN universe adds) is a model-input change that resets the clock, with `oos_clock_start` set to the first promoted run on the merged code (memo `2026-11-17-coinvest-cusips-and-universe-adds-clock-reset.md`). The operator approved this record in session ("do 1 through 7").
**Hash rotation:** none. `model_hash a9a430154568e242` (`ast-v2`) and ruleset `8887576e` are unchanged.

---

## 1. Out-of-sample clock start = 2026-11-17

`artifacts/forward_validation/CANDIDATE.json` now reads `oos_clock_start = 2026-11-17`. This supersedes the 2026-11-16 start, which moves to `prior_oos_clock_starts`. The reset is listed under `resolved_clock_resets`. Mandate SM-20260629-001 restarts at 0/20.

| Item | Value |
|---|---|
| Run | 2026-11-17, cron 16:30 ET |
| Run manifest commit | `fd731536` (merge of PR #696) |
| Result | WARN; snapshot promoted; 44 gates: 40 PASS, 4 WARN (audit, drift_monitoring, ruleset_health, regulatory_calendar), no FAIL |
| Forward-validation capture | LIVE, quality PASS, model hash match, `ranker_mode=selector_only`, eligible for the mandate |
| Eligible names | 151 |
| Top-30 | RVMD, COGT, TNGX, IRON, EWTX, TRVI, SVRA, ELVN, VRDN, DMRA, SYRE, CLDX, ERAS, PRAX, STOK, TARS, PHVS, IMCR, VOR, MIRM, IDYA, MAZE, DSGN, LBRX, DNTH, TRAX, ALKS, DFTX, BCRX, COAG |
| Against live 2026-11-16 | 25/30. In: COAG, DMRA, IMCR, MAZE, TRAX. Out: CYTK, DYN, KARD, MBX, SRZN |
| New universe names | ILMN and STVN are eligible (ranks 151 and 148). RPRX is not eligible on 2026-11-17 |

The membership change matches the 2026-11-17 Q3 13F refresh preview (#692), which made the same four swaps on 2026-11-16 inputs (DMRA, IMCR, MAZE, TRAX in; BCRX, SRZN, MBX, KARD out), plus one day of price movement.

## 2. The replaced 2026-11-16 snapshot does not affect the ledger

On 2026-11-17 the operator replaced the 21 dates in `data/snapshots` that the v9 historical rebuild covers, including the live 2026-11-16 directory. The live directories were moved to `data/snapshots_backup_pre_v9_2026-11-17/` (PR #698).

Each forward-validation capture stores its baskets inline (Top-30 and the comparison baskets). `tools/fill_forward_returns.py` reads only the captures and prices; `tools/run_forward_validation.py` reads `rankings.csv` only at capture time. The 2026-11-16 and 2026-11-17 captures are therefore unaffected. For the record, the live 2026-11-16 `rankings.csv` (sha256 `c2d1c1647524cdb7…`) is in the backup folder.

## 3. Ledgers persisted

The day's forward-validation ledger rows (`captures.jsonl`, `fills.jsonl`, `WEEKLY_SUMMARY.md`) are committed with this record, as on 2026-11-16.
