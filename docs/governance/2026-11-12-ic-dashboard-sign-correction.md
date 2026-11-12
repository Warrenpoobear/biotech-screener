# IC dashboard and weekly sweep: `score_rank_pct` sign inverted — correction and restatement

**Date:** 2026-11-12
**Status:** PROPOSED. Not merged.
**Classification:** Tier 2 tooling (monitoring only). No model, ranker, selector, eligibility, sizing or production-gate behavior changes. The clock is not affected.
**Authorization:** operator asked for this fix to be built in session 2026-11-12. Merge is the operator's decision.
**Hash rotation:** none.

---

## 1. Defects

| # | Defect | Evidence | Effect |
|---|---|---|---|
| S1 | `score_rank_pct` negated before the IC | `tools/build_ic_dashboard.py` had `("score_rank_pct", False)`; `scripts/shared_regime_check.py` copied it. The percentile rises with `composite_score` (best rank → highest percentile, `common/score_to_er.py`; Spec 091 §1 says the same). Spearman of `score_rank_pct` against `composite_rank` is −1.000 on 2026-02-27, 03-27, 04-15, 06-23, 09-11 and 11-11. Present since the tool was written (`023114c6`, 2026-03-27) | Every `score_rank_pct` IC ever reported by the dashboard, `history.jsonl` and the weekly sweep has the wrong sign |
| S2 | The dashboard did not measure the production order | `score_rank_pct` is the composite percentile. Its Spearman with `selector_score` is about 0.27 on 2026-09-11 | "Primary signal HEALTHY" said nothing about what production trades. Same class as F-2026-002 |
| S3 | Forward returns from unadjusted prices | `production_data/price_history.csv` shows GOSS +7,163% and SKYE +562% on 2026-08-17, SION −91% and TENX −90% on 2026-08-10, ABOS −91% on 2026-10-30, EYPT −85% on 2026-10-19; none appear in `price_history_split_adj.csv` | Corrupted forward returns for the affected names |
| S4 | A healthy lookback mean hid a collapse | 2026-11-11 dashboard: mean +0.1764 "HEALTHY" while the latest reading was +0.0068 | No trend signal |

Separately, the 2026-11-11 production dashboard was built by code that predated the regenerated-snapshot exclusion (#628): all 40 of its dates (2026-08-18 → 2026-10-14) are in `config/snapshot_provenance_exclusions.json`.

## 2. Changes (branch `fix/ic-dashboard-sign-and-scope`)

1. S1: `score_rank_pct` is `higher_is_better=True` in both tools.
2. S2: new dashboard signal `actionable_rank` (lower is better). Rows with no rank (ineligible) are not observations.
3. S3: the default price file is `price_history_split_adj.csv`, falling back to `price_history.csv` only when it is absent. The file used is recorded as `price_source`.
4. S4: each signal gains `recent_ic` (mean of the last 5 dates) and `trend`. `trend` is `decaying` when the lookback mean is at or above the HEALTHY threshold and `recent_ic` is at or below 0. It never changes the health label; it raises `attention` from LOW to MEDIUM.
5. Tests: `tests/test_ic_dashboard_sign_and_scope.py` (8 tests, all fail on main); one existing fixture updated for the corrected sign.

Thresholds (HEALTHY ≥ +0.03, WARN/ALERT bands) are unchanged. No signal is renamed, suppressed or demoted (Spec 091 hard constraints).

## 3. Restated readings

Corrected tool, 20-day horizon, 60-snapshot lookback, split-adjusted prices, regenerated snapshots excluded unless stated.

| As of | Window (dates with realized returns) | Signal | Reported then | Corrected |
|---|---|---|---:|---:|
| 2026-06-24 | 2026-04-11 → 06-23, 31 dates | `score_rank_pct` | +0.0432 HEALTHY (2026-06-23 history row; 35 dates) | −0.0201 WARN |
| 2026-06-24 | same | `actionable_rank` | not measured | +0.1534, hit 90%, HEALTHY |
| 2026-06-24 | same | `coinvest_score_z` | not measured | +0.1478, hit 87%, HEALTHY |
| 2026-11-11 | 2026-08-18 → 10-14, 40 dates, **regenerated included** | `score_rank_pct` | +0.1764 HEALTHY | −0.1697 ALERT |
| 2026-11-11 | same, regenerated included | `actionable_rank` | not measured | −0.2949, hit 13%, ALERT |
| 2026-11-11 | 2026-06-02 → 08-13, 49 live dates | `score_rank_pct` | n/a | −0.0251 WARN |
| 2026-11-11 | same, live only | `actionable_rank` | n/a | +0.0318 HEALTHY, recent −0.2943, **decaying** |
| 2026-11-11 | same, live only | `coinvest_score_z` | n/a | +0.0419 HEALTHY, recent −0.2538, **decaying** |

Reading these:

- **Path C IC gate (closed 2026-06-24 on "+0.0432 > +0.03").** The cited number had the wrong sign and measured the composite percentile. On the production order over the same period the corrected reading is +0.1534. The gate's conclusion is not overturned, but the evidence cited for it was wrong. Whether to re-record the closure against the production-order reading is an operator decision.
- **Composite score.** Corrected, the composite percentile has been mildly negative in every window measured here. It is informational only (Spec 091); it is not the production order.
- **August to October.** The production order was strongly negative on snapshots dated 2026-08-18 → 10-14, but those are regenerations and are not point-in-time evidence. On live dates the last five readings (to 2026-08-13) average −0.29.
- **Staleness.** With regenerated snapshots excluded, the newest live date with a realized 20-day return is 2026-08-13. The dashboard cannot say anything about September onward until live snapshots from 2026-11-02 have 20 trading days of returns (early December).
- Daily snapshots with a 20-day horizon overlap. These windows contain only a handful of independent observations. None of this is performance evidence for or against the model.

## 4. Not changed, and follow-ups

- Historical artifacts (`artifacts/ic_dashboard/*`, `history.jsonl`, `WEEKLY_SIGNAL_REGIME_SWEEP_*.md`) are left as written. Their `score_rank_pct` values must be read with the sign flipped.
- `agents/ic_health_monitor/SOUL.md` still lists `clinical_optionality_pct_dev` and `inst_delta_z` as load-bearing. That is stale and out of scope here.
- `tools/forward_eval_gate.py` already measures negated `actionable_rank` on eligible rows and is not affected.
- The first production run after merge will append a `history.jsonl` row whose `score_rank_pct_ic` has the corrected sign; rows before it do not.
