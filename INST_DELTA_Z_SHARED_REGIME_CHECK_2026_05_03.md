# Shared-Regime Check: coinvest_score_z vs inst_delta_z — 2026-05-03

**Status:** Diagnose-only follow-up to `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md` §5 item 1.
**Type:** Read-only IC computation; no production code or config touched.
**Source artifact:** `artifacts/ic_dashboard/2026-05-03_shared_regime_check.json`

---

## Question

Is the inst_delta_z negative IC reading a SHARED REGIME affecting institutional-flow signals broadly, or is it SIGNAL-SPECIFIC?

## Method

Replicated `tools/build_ic_dashboard.py` methodology against `coinvest_score_z` over the identical 35-snapshot window already used by `inst_delta_z` in the 2026-05-01 dashboard.

- Window: 2026-02-19 → 2026-04-01 (35 snapshots, identical to the inst_delta_z `per_date` array)
- Horizon: 20 trading days (matches `DEFAULT_HORIZON` in build_ic_dashboard)
- Min observations per date: 10 (matches `compute_ic` cutoff)
- Spearman IC, higher_is_better=True for both signals (consistent with the SIGNALS spec for inst_delta_z, and the design of coinvest_score_z)
- 325 unique tickers across the window
- Price source: `production_data/price_history.csv`

## Results

| Signal | Mean IC | Hit Rate | n | Min/Max | Latest IC |
|---|---|---|---|---|---|
| coinvest_score_z | **+0.1033** | **91.4%** | 35 | -0.0689 / +0.1911 | -0.0689 (2026-04-01) |
| inst_delta_z | **-0.1006** | **8.6%** | 35 | -0.1936 / +0.0794 | -0.0092 (2026-04-01) |

The inst_delta_z line matches the dashboard exactly (mean -0.1006, latest -0.0092, n=35), confirming methodology fidelity.

### Co-trajectory

| Statistic | Value | p-value |
|---|---|---|
| Spearman ρ(coinvest_IC, inst_delta_IC) | **-0.5712** | 0.0003 |
| Pearson r | **-0.5742** | — |

35 common dates. Both correlations are strongly NEGATIVE and statistically significant.

## Interpretation

**Not a shared regime — signal-specific failure of inst_delta_z.**

Three findings:

1. **The institutional-flow lane is not broadly broken.** `coinvest_score_z` is HEALTHY by every cut: mean IC +0.10, hit rate 91% (matching `clinical_optionality_pct_dev` over the same window). Whatever is dragging inst_delta_z is not "13F-derived signals don't work right now."

2. **The two signals are anti-correlated, not co-moving.** The -0.57 Spearman ρ on IC trajectories is the opposite of what a shared-regime explanation would predict. On days when coinvest IC is strongly positive, inst_delta_z IC is strongly negative, and vice-versa. They are effectively opposing bets at the IC level.

3. **The negative co-trajectory is a structural property of the signal pair, not noise.** With p=0.0003, n=35, this is unlikely to reverse with one or two more dates of data. It is also consistent with the design — coinvest tracks AUM-weighted overlap with elite-manager holdings (a level signal); inst_delta_z tracks period-over-period overlap CHANGE (a delta signal). Levels and deltas can diverge for long stretches.

## Implications for the disposition

The shared-regime hypothesis (and the "wait for refresh, it'll fix itself" framing) becomes weaker:

- A 13F refresh on ~2026-05-15 will reset the delta input, but the structural negative correlation means even a "good" refresh would not necessarily flip inst_delta_z's IC sign — it might just reset the magnitude.
- "Drop or down-weight inst_delta_z" (Options A/B in the parent memo) cannot be defended-against by saying "the institutional lane is broken right now." The institutional lane IS working — through coinvest.
- The cleanest reading is: **the level signal is doing the work; the delta signal is contributing noise (or worse) over the recent window.** The +0.45 ranker weight on inst_delta_z is fighting the +X weight on coinvest_score_z.

This finding does NOT, on its own, change the parent memo's disposition. It removes one defense of Option C ("hold and wait for refresh") and slightly strengthens the case for Options A/B. The decision still belongs to the operator.

## What this check does NOT prove

- It does not show inst_delta_z has been broken historically — only over the 2026-02-19 → 2026-04-01 window.
- It does not isolate the mechanism (regime-driven contrarian flow, post-cohort-quarantine contamination, structural delta-vs-level divergence, model-misspecification).
- It does not evaluate Option B's specific weight choice. A separate IR-impact backtest sweep is still required for that.
- It does not address the SOUL.md / ranker_engine.py weight-string mismatch (0.30 vs 0.45). Documentation drift, separate ticket.

## Provenance

- Script: `/tmp/shared_regime_check.py` (transient; logic recorded here)
- Artifact JSON: `artifacts/ic_dashboard/2026-05-03_shared_regime_check.json` (full per-date IC arrays for both signals)
- Methodology source: `tools/build_ic_dashboard.py:117-173`
- Parent memo: `INST_DELTA_Z_GOVERNANCE_REVIEW_2026_05_03.md`
- Window selection rationale: parent memo §1.1 (dashboard date_range)
