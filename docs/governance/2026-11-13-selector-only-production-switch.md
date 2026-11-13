# Production order switched to selector-only (2026-11-13)

**Classification:** `OPERATOR_AUTHORIZED_MODEL_CHANGE`. Tier 3 code (`run_screen.py` ranker dispatch), Tier 4 decision.
**Authorized by:** the operator, in session on 2026-11-13 ("switch to selector only model").
**Takes effect:** with the first daily production run on the merged code. Not in effect until this PR is merged and the production checkout is pulled.
**Out-of-sample clock:** resets. The mandate SM-20260629-001 restarts at 0/20 from the first run on the merged code.

## What changes

`run_screen.py --ranker-mode` gains `selector_only` and it becomes the default (`common/ranker_mode.py`).

| | Before | After |
|---|---|---|
| Eligibility gates | unchanged | unchanged |
| Selector | coinvest-only (`coinvest_score_z` 100%), ruleset `8887576e` | unchanged |
| Order inside the eligible set | pairwise `minimal_v2` ranker reorders the top-60 selector cohort (`coinvest_score_z` +0.02, `financial_score` −0.0533) | `selector_score` order for the whole eligible set; no ranker |
| `final_score` | `ranker_v2_score` for the cohort, `selector_score × 0.0001` below it | `selector_score` |
| `ranker_active` | "1" for the 60 cohort names | "0" for every name |
| `ranker_v2_score`, `ranker_v2_rank` | production | still computed and written, as a shadow record only |
| Construction | equal-weight Top-30 | unchanged |

Not changed: `decision_engine.py`, `selector_engine.py`, `ranker_engine.py`, `ranker_v2_pairwise.py`, `production_data/ranker_v2_model.json`, the ruleset, eligibility, sizing.

Rollback: `--ranker-mode pairwise_minimal` (one flag; the ranker code and model file are untouched).

## Why

1. **The ranker's weights were fitted on contaminated history.** Historical runs read today's market cap on every past date, and `financial_score` divides cash and burn by market cap. The ranker's negative `financial_score` weight therefore rewarded companies whose market cap later grew (`docs/governance/2026-11-13-market-cap-lookahead-deprecation.md`).
2. **With point-in-time market cap the signal is gone.** `financial_score` monthly rank correlation with next-period return: −0.013 (t = −1.3) across eligible names, +0.018 inside the ranker cohort (`data/snapshots_pit_v5`).
3. **Selector order alone did better on the corrected histories.**

| History | Current rule | Selector-only | Difference |
|---|---|---|---|
| `snapshots_pit_v4` (filing-based market cap) | +0.78 pt/period (t = 1.67) | +1.41 (t = 2.68) | +0.62 (t = 1.64) |
| `snapshots_pit_v5` (Sharadar daily market cap, false splits removed) | +0.96 (t = 2.02) | +1.53 (t = 2.94) | +0.57 (t = 1.63), positive in 6 of 7 years |

Top-30, equal weight, 79 monthly periods 2020-01-31 → 2026-06-26, net of 50 bps one-way on actual turnover, XBI gross, JBIO excluded. Both arms replayed with the production selector and ranker stage on the recorded rows.

4. **Live data pointed the same way before the correction.** Inside the ranker cohort `financial_score` was positively related to returns in 4 of 5 live months (2026-04 → 2026-08), against the negative weight.

## What this decision does not rest on

- **Checklist v2 was not run.** The governed promotion path (FM, bootstrap, FDR, LOSO) was skipped by operator decision.
- **The difference is not statistically significant** (t ≈ 1.6 on both histories).
- **The history is pseudo-PIT:** no ticker that stopped trading before 2025 is in the price file, the selector settings were chosen with this period in view, and January and February 2020 use current 13F holdings.
- **The live selector-only shadow** (PR #653) had one LIVE capture when this was decided. It is not evidence for or against.

## Effect on the current ranking

2026-11-12 inputs, run both ways in an isolated copy:

- The new order equals `selector_score` descending exactly (ties by ticker), for all 162 eligible names.
- Top-30 overlap with the ranker order: 17 of 30.
- Leave: ABSI, ASND, BLTE, DNLI, ERAS, KYMR, MDGL, PCVX, REPL, SLDB, TARS, TRAX, TVTX.
- Enter: AMLX, AURA, BCRX, CLDX, DFTX, DMRA, ELVN, EWTX, IDYA, IMCR, SGMT, TNGX, TSHA.
- The new Top-30 is identical, name for name and in order, to the `selector_only` basket the live shadow ledger recorded for 2026-11-12.
- Eligibility, `selector_score`, `coinvest_score_z`, `financial_score` and `de_drawdown` are unchanged for every name.

The live account that tracks the Top-30 would trade 13 names at its next rebalance. No trade is made by this change.

## Hash and candidate identity

The candidate `model_hash` (`ast-v1`) covers `ranker_engine.py`, `selector_engine.py` and `decision_engine.py`. None of them changes, so **the hash stays `827c35a9ed3ee6e1` although behavior changes**. The ranker dispatch lives in `run_screen.py`, which the hash does not cover. This is recorded in `governance/HASH_ROTATIONS.md` as a behavior change without a hash change, and `CANDIDATE.json` now carries `ranker_mode`.

## Follow-ups

1. **Record the new `oos_clock_start`** in `artifacts/forward_validation/CANDIDATE.json` after the first production run on the merged code. Until it is recorded, captures from selector-only runs must not be counted toward the mandate.
2. **Selector-only shadow ledger:** each capture now also records the `ranker` basket, and the summary reports ranker minus selector-only, so the ranker's counterfactual keeps accruing.
3. **Extend the candidate hash** to cover the ranker dispatch, or add `ranker_mode` to the capture's identity check. Not done here.
4. **Checklist v2 on `snapshots_pit_v5`**, if the operator wants the decision backed after the fact.
5. **Spec 094** (selector-only comparator) and **Spec 093** (`financial_score` sign) need their status updated by the operator.
6. **Hermes skill mirrors** beyond `selector-ranker` still describe the pairwise ranker as production.
