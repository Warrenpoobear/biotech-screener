# Selector-only shadow capture (Spec 094) — observability only

**Date:** 2026-11-12
**Status:** PROPOSED. Not merged.
**Classification:** Tier 2 observability. Capture-only. No model, ranker, selector, eligibility, sizing or gate behavior changes, and no clock effect. `tools/run_daily_production.py` gains one non-blocking step; because that file is the production runner, review it as Tier 3.
**Authorization:** operator asked for this to be built in session 2026-11-12. Merge is the operator's decision.
**Hash rotation:** none.

---

## 1. Why

Spec 094 asks whether the pairwise ranker adds value over the selector alone, and has stood at `RANKER_UNPROVEN`. Nothing was accruing forward evidence on that question:

- `tools/coinvest_shadow_tracker.py` records daily selections for seven arms, but none is the live selector's own Top-30, its ranker arm uses the legacy ranker, and its `history.csv` forward-return columns are written once on capture day and never filled.
- A live-snapshot review on 2026-11-12 (71 live dates, 2026-04-03 → 08-13) found `financial_score` IC inside the ranker cohort positive in four of five months, against the ranker's negative weight, and the production order inside the cohort negative in the same four. Four non-overlapping windows; not significant.

The ranker matters: on live snapshots since 2026-05-13 the selector-only Top-30 and the production Top-30 share only 16 to 20 names (mean 18.3 of 30).

## 2. What it does

`tools/selector_only_shadow.py`:

- **Capture.** For a promoted snapshot, record the production Top-30 (`actionable_rank`) and the selector-only Top-30 (`selector_score` descending, ties on ticker). Both draw from the same pool: eligible rows with an `actionable_rank`. The row stores the `rankings.csv` SHA-256.
- **Discipline.** A row written on the snapshot date is `LIVE`; anything written later is `REPLAY`. A date already in the ledger is never rewritten, so a later regeneration of the snapshot cannot change the record.
- **Summary.** Equal-weight forward returns at 5 and 20 trading days from split-adjusted closes, production minus selector-only, on non-overlapping windows only. No reading is offered below 20 windows.
- **Wiring.** One non-blocking step in the daily run, after the coinvest shadow step. A failure logs a warning and nothing else.
- **Ledger.** `artifacts/selector_only_shadow/captures.jsonl` is un-ignored in `.gitignore`, the same way as the forward-validation ledger. It still has to be committed periodically; nothing does that automatically.

## 3. Replay context (not evidence)

Replaying the 67 live, non-regenerated snapshots from 2026-05-13 (written after the fact, so `REPLAY`; scratch output, not committed):

| Horizon | Non-overlapping windows | Mean diff (production − selector-only) | t | Ranker ahead |
|---|---:|---:|---:|---:|
| 5 days | 14 | +0.15 pts | +0.43 | 71% |
| 20 days | 4 | −0.23 pts | −0.34 | 50% |

On this history the ranker neither clearly helps nor clearly hurts. These snapshots used the financial inputs as they were before PR #586, #621 and #649.

## 4. Limits

- At one independent 5-day window a week, 20 LIVE windows takes about five months; 20-day windows take far longer.
- The comparison is equal-weight, gross of costs, and ignores turnover differences between the two baskets.
- It is not a promotion test. Any change to the ranker still needs Checklist v2 and operator sign-off.
