# Financial score: solvency by operating cash flow, deduped cash total — proposed bug fix

**Date:** 2026-11-12
**Status:** PROPOSED. Not merged. Merge needs operator authorization under protocol §1.
**Classification:** Tier 3. `module_2_financial_v2.py` produces `financial_score`, one of the two features the live `minimal_v2` ranker reads.
**Authorization:** operator asked for this PR to be built in session 2026-11-12. Authorization to merge, and the clock decision in §4, are still open.
**Protocol:** `docs/FORWARD_VALIDATION_PROTOCOL.md` §1 — an authorized bug fix during the NO_MODEL_CHANGE window resets the out-of-sample clock.
**Hash rotation:** none. `model_hash 827c35a9ed3ee6e1` covers `ranker_engine.py`, `selector_engine.py` and `decision_engine.py`, none of which change. Ruleset `8887576e` is unchanged.
**Supersedes:** PR #610 and PR #611 (see §5).

---

## 1. Defects

| # | Defect | Evidence (2026-11-11 screen, main `69323d70`) | Effect |
|---|---|---|---|
| F1 | Positive net income marks a company "profitable" even when operating cash flow is negative | `calculate_burn_rate_v2` returned `PROFITABLE` on `CFO > 0 OR NetIncome > 0`. 14 names have TTM net income > 0 and TTM CFO < 0 | Monthly burn 0 and runway 999 months for those 14 names. KYMR (rank 7) should be 15.6 months, BDTX 7.0, PGEN 7.6, NMRA 10.6 |
| F2 | Liquid assets = Cash + the first investment tag found | `calculate_liquid_assets` ignored the collectors' deduped `CashAndSecurities` total and stopped at the first positive tag, which can be years stale (DNTH `ShortTermInvestments_date` = 2022-12-31) | No name changes on 2026-11-11: liquid assets are identical on both arms. It is a latent error, most exposed on static fallback records |

## 2. Changes (branch `fix/financial-runway-and-cash-total`)

1. F1: operating cash flow decides solvency. Net income is used only when no cash-flow figure exists.
2. F2: `CashAndSecurities` is preferred when present and positive. Otherwise the largest investment tag is used, not the first.
3. Tests: `tests/test_financial_runway_and_cash_total.py` (10 tests; 6 fail on main).

Selector and ranker weights, the ranker's negative sign on `financial_score`, eligibility rules and sizing are untouched.

## 3. Measured impact

Isolated screens for as-of 2026-11-11 on copies of `production_data/`, main `69323d70` against this branch. Run without `--strict` and without a prior-snapshot directory. The main arm reproduced the production 2026-11-11 Top-30 exactly (30/30).

| Metric | main | this branch |
|---|---:|---:|
| Rows in `rankings.csv` | 296 | 296 |
| Top-30 membership overlap | — | 30/30 |
| Top-60 membership overlap | — | 60/60 |
| Eligibility differences | — | 0 |
| Names with a 999-month runway | 97 | 83 |
| Names failing the sev3 gate | 14 | 14 |
| Module 2 records changed | — | 14 of 320 |

Rank moves inside the Top-30: KYMR 7 → 4, ORKA 21 → 19, SVRA 16 → 15, ORIC 4 → 5, SMMT 5 → 6, TRAX 6 → 7, ANNX 15 → 16, MDGL 19 → 21. A lower `financial_score` moves a name up because the ranker weight is negative.

Module 2 `financial_score` for the four names that move most: BDTX 79.3 → 47.4, PGEN 69.7 → 42.8, NMRA 86.4 → 66.1, KYMR 74.5 → 65.4. BDTX, PGEN and NMRA have no actionable rank on either arm.

This is a single-date measurement. It is not performance evidence.

## 4. Clock

`oos_clock_start` is pending and expected to be the first post-merge production run of PR #614 (2026-11-12). This fix changes a ranker input, so under protocol §1:

- merged before `oos_clock_start` is recorded, it is part of the new window's starting configuration;
- merged after, it resets the clock again.

The operator decides which. The Top-30 basket is unchanged on 2026-11-11, so the equal-weight forward test would have captured the same names either way on that date.

## 5. Why not PR #611

PR #611 contains F1 and F2 plus further changes. Merged onto current main it changes the Top-30 to 26/30 and removes 22 names from `rankings.csv`, including DNTH and STOK (ranks 1 and 2).

- Cause: it divides `CFO` by 6 or 9 months when the statement date is June or September. Since PR #586 the PIT store supplies trailing-12-month figures (DNTH TTM CFO = −128.171M − 129.060M + 82.077M = −175.154M), so burn is overstated 1.33x to 2x for 228 of 320 names.
- The date rule is also wrong for non-calendar fiscal years: IMMP's fiscal year ends 30 June, so its June-dated annual figure would be halved.
- With that one piece neutralised, PR #611 leaves the Top-30 at 30/30 and the Top-60 at 60/60.

Not carried over from PR #611, each needing its own review:

- Market-cap unit conversion in code. PR #621 fixed the data and added a schema gate; market caps are identical on both arms.
- The 120-day stale-PIT-store gate. 54 stores are older than 120 days; for 24 of them the static record is newer, but its cash flow is year-to-date and would be divided by 12. The root cause is that those PIT stores are not being refreshed.
- Statement-date admissibility filter, removal of the `$10` / `100,000`-share defaults, clinical endpoint word boundaries, EDGAR collector period selection, CI and packaging changes.

## 6. Follow-ups

1. Operator: authorize or decline the merge and record the clock decision.
2. On merge: sync `docs/MODEL_DOCUMENTATION.md` and affected skills (model-doc-and-skill-sync).
3. Refresh the 24 PIT financial stores that lag their static records (ARCT, AUTL, CTKB, ENGN, ETON, KALA, LFCR, MGTX, MRVI, OBIO, PACB, PAHC, RPRX, TKNO, XERS, ILMN, DRUG, PRQR and the retired APLS, CNTA, ESPR, KALV, NUVL, SGMO).
4. The ranker's negative weight on `financial_score` was fit before PR #586 and PR #621 corrected its inputs. Re-fit or re-accept it under Checklist v2.
