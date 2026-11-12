# Out-of-sample clock start recorded; Path C IC-gate closure re-recorded

**Date:** 2026-11-12
**Classification:** Tier 4 governance record. No code, model, ranker, selector, eligibility, sizing or gate behavior changes.
**Authorization:** operator (Darren Schulz), in session 2026-11-12: instructed that `oos_clock_start` be recorded, that the Path C record be made, and that the evening's ledgers be committed.
**Hash rotation:** none. `model_hash 827c35a9ed3ee6e1` and ruleset `8887576e` are unchanged.

---

## 1. Out-of-sample clock start = 2026-11-12

`artifacts/forward_validation/CANDIDATE.json` now reads `oos_clock_start = 2026-11-12`. This completes memo `2026-11-04-missing-fields-clock-reset.md` §4 step 4 and supersedes the 2026-11-02 start, which is kept under `prior_oos_clock_starts`.

The start is the first production run on the merged code:

| Item | Value |
|---|---|
| Run | 2026-11-12, cron 16:30 ET, finished 16:51 ET |
| Run manifest commit | `2770e533` |
| Result | WARN (audit, drift_monitoring, ruleset_health, regulatory_calendar); snapshot promoted; 43 gates; no errors |
| `ctgov_content_freshness` | PASS (newest `last_update_posted` 2026-11-12) |
| Forward-validation capture | LIVE, quality PASS, model hash match, benchmark available |
| Prices | through 2026-11-12 (Sharadar PARTIAL at run time, yfinance appended 350 rows) |

Changes that are part of the starting configuration because they were merged and deployed before this run:

- PR #614, #635, #636: CT.gov source refresh and blocking content gate; `inst_delta_z` blank semantics.
- PR #649: `financial_score` solvency by operating cash flow and deduped cash total.
- PR #622 (with #621): Sharadar as the primary price and market-cap source. Its first in-pipeline run is this run, so the window is not split across the cutover. The governance gap noted in the model documentation (no separate memo for #622) is closed by this record to that extent only.
- PR #654: `ctgov_content_freshness` added to `GATE_ALLOWLIST`. Without it this run would have raised at manifest build.
- Input changes merged 2026-11-02 → 2026-11-12 that the model documentation lists (PDUFA calendar additions, universe additions and status updates, INDV → SUPN, split-registry and special-dividend handling, SEC 8-K mislabel fix).

Consequences:

- Mandate SM-20260629-001 restarts at **0/20**. `WEEKLY_SUMMARY.md` regenerated under the new start shows 0 mandate-eligible windows.
- Captures dated 2026-11-03 → 2026-11-11 and the seven pre-2026-11-01 windows are history only.
- Split-adjusted price history was rewritten during this run for 44 tickers (21,351 rows), following the corporate-action fixes merged 2026-11-11/12. All of it predates the new start, so no counted window is affected. One effect is visible in the first snapshot: IMMP's 2026-03-13 crash is no longer smoothed out, so its drawdown reads −94% and it is ineligible.
- Any further authorized input or model fix resets the clock again.

## 2. Path C IC-gate closure re-recorded

Path C was formally closed on 2026-06-24 citing "`score_rank_pct` mean_ic +0.0432 > +0.03". Memo `2026-11-12-ic-dashboard-sign-correction.md` showed that figure had the wrong sign and measured the composite percentile, not the production order.

The closure is **re-recorded against the production order**:

| | Original record (2026-06-24) | Re-recorded (2026-11-12) |
|---|---|---|
| Signal | `score_rank_pct` | `actionable_rank` |
| Reading | +0.0432, hit 54.3%, N = 35 | +0.1534, hit 90%, N = 31 live dates (2026-04-11 → 2026-06-23) |
| Tool | dashboard / weekly sweep with the inverted sign | corrected `tools/build_ic_dashboard.py`, 20-day horizon, split-adjusted prices, regenerated snapshots excluded |
| Gate (mean IC > +0.03) | cited as met | met |

- The original citation is **retracted**. For the same dates the corrected `score_rank_pct` reading is −0.0201.
- The decision itself stands: Path C remains closed, the catalyst-timing override policy remains in production weights, and Path A durable gates apply.
- Caveat carried with the re-record: the 31 daily dates overlap heavily under a 20-day horizon and amount to a few independent observations. The gate was a threshold test, not a significance test, and this re-record does not turn it into performance evidence.

## 3. Ledgers committed with this record

- `artifacts/forward_validation/captures.jsonl` (+1: 2026-11-12), `fills.jsonl` (+6), `WEEKLY_SUMMARY.md` (regenerated).
- `artifacts/selector_only_shadow/captures.jsonl` (first row, LIVE, 2026-11-12, overlap 17/30) and `summary.md`.

They were copied from the production checkout after the run finished and are byte-identical to what the run wrote, except `WEEKLY_SUMMARY.md`, which was regenerated under the new clock start.
