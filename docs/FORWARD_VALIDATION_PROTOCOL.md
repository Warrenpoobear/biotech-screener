# Forward Validation Protocol — DEM Top-30 Candidate Model

**Generated:** 2026-06-28
**Classification:** `PRE_REGISTRATION_NO_MODEL_CHANGE`
**Status:** RATIFIED 2026-06-28 (operator) — test specification locked; not to be re-specified after forward data is seen
**Author scope:** Documentation-only. This file specifies a test. It does not create, wire, or modify any cron job, pipeline step, agent, ranker, selector, sizing rule, score, eligibility rule, or production artifact.

---

## 0. Why this document exists

The historical evidence for the DEM Top-30 basket is encouraging but not yet trustworthy as a statement about the *current* model:

- **YTD 2026 (corrected, split-adjusted):** DEM EW +38.52% vs XBI +27.43% = **+11.09pp** excess (commits `1288aae8`, `8ae5535a`).
- **Full-history 2020–2026:** portfolio +261.6% vs XBI +62.4% = +199.1pp; weekly non-overlapping **t=1.90**, n=328, no negative year (`artifacts/autopsy/top30_ytd_validation/TOP30_FULL_HISTORY_VALIDATION.md`, commit `409146b9`).

Both numbers are contaminated for the purpose of judging the production model:

1. **Model-version heterogeneity** — 2020–2025 archives reflect v1.0–v1.3. The current production model (v1.4, `actionable_rank`, locked April 2026) has only ~24 clean out-of-sample weeks (t=0.66).
2. **Development-time bias** — the model was iterated against this same history. It is in-sample by construction.
3. **Statistical immaturity** — weekly non-overlapping t has not cleared the two-tailed 95% threshold (1.96). The daily-overlapping t overstates significance via 5-day-window autocorrelation.

The correct next move is **not** more model improvement. It is to create a clean, pre-registered evidentiary record of the frozen candidate model **before** forward data arrives, so the verdict cannot be retrofitted. This document is that pre-registration.

---

## 1. The candidate model (freeze declaration)

The following is declared the **candidate** and is frozen for the validation period. This declaration is documentation of intent; the binding production freeze already in effect is the 2026-06-20 scoped architecture freeze recorded in `CLAUDE.md` / `.claude/rules/operational-state.md`. This protocol does not lift, extend, or alter that freeze.

| Component | Frozen value |
|-----------|--------------|
| Ruleset | `8887576e` (v1.14.0), pinned in `run_screen.py` + `run_phase2_snapshot_delta.py` |
| Model version | v1.4 `actionable_rank` |
| Basket | Equal-weight top-30 by `actionable_rank` |
| Rebalance | Weekly, Monday (per `production_data/AGENTIC_ACCOUNT_RULES.md`) |
| Price source | `price_history_split_adj.csv` (split-adjusted); raw only beyond split-adj cutoff, flagged per row |
| Benchmark | XBI, split-adjusted, same-day endpoints |

**No changes permitted during the validation window** unless a verified bug is identified and an operator explicitly authorizes the change (which resets the out-of-sample clock for affected periods):

- No ranker, selector, feature, eligibility, catalyst-bucket, or sizing change.
- No rebalance-policy change.
- No manual basket overrides except documented hard-exit rules already in `AGENTIC_ACCOUNT_RULES.md`.
- No price-source change.

The daily validation record exists to **observe** the frozen model. It must not become a reason to tune it. Any tuning is a *new* candidate with a *new* lock date.

---

## 2. Pre-registered primary test

> **H₀:** mean weekly excess return of the frozen DEM Top-30 basket over XBI ≤ 0.
> **H₁:** mean weekly excess return > 0.

- **Unit of evidence:** completed **non-overlapping 5-trading-day** forward windows from the candidate lock date.
- **Estimator:** mean weekly excess (DEM EW return − XBI return), split-adjusted, same-day endpoints, **no silent missing-price fallback** (a missing price fails the row, it does not substitute raw).
- **Significance basis:** weekly non-overlapping draws only. Daily-overlapping t-stats are reported as context but are **not** the gate (autocorrelation inflates them).
- **One-tailed** at 95% is the directional gate; two-tailed 95% (|t| ≥ 1.96) is the stronger confirmation milestone.

This test is fixed as of ratification (2026-06-28). It is not to be re-specified after data is seen.

---

## 3. Cadence

| Cadence | Purpose |
|---------|---------|
| Daily model run | Capture frozen Top-30, ranks, weights, data-quality status, corporate-action flags |
| Daily completed-return update | Fill in 1d / 5d / 20d forward returns as they become observable |
| Weekly non-overlapping 5d window | **Primary statistical validation gate** |
| Monthly chain | Portfolio-reality check (chain-linked basket return) |
| Quarterly review | Governance decision point |

---

## 4. Pass / fail gates

| Gate | Threshold | Meaning |
|------|-----------|---------|
| Minimum sample | 20 completed non-overlapping weekly windows | Below this, **verdict = INSUFFICIENT_DATA** regardless of t |
| Directional | weekly mean XS > 0 AND one-tailed t ≥ 1.65 | Candidate remains valid, directionally supported |
| Confirmation | two-tailed |t| ≥ 1.96 over ≥ 20 windows | Eligible for operator promotion review (not automatic) |
| Drawdown guard | no excess-return drawdown vs XBI worse than the worst observed YTD (≈ −2.4%/wk cluster, May 26 → Jun 8) without documented cause | A worse cluster triggers REVIEW_REQUIRED |
| Adversarial | control-basket excess (Section 6) must be materially below candidate excess | Guards against "any 30 biotech names" beta capture |

**Fail and rollback rules (added 2026-11-15, before the selector-only clock start; operator decision, `docs/governance/2026-11-15-pre-clock-decisions.md` §4).** The gates above define passing only. These define failing. They add to the §2 test and do not re-specify it.

| Rule | Threshold | Meaning |
|------|-----------|---------|
| Fail | at a checkpoint of 20, 40 or 60 completed windows: weekly mean XS < 0 AND t ≤ −1.65 | STOP_AND_REVIEW: the operator reviews the model before any further capital is committed |
| Not yet significant | fewer than 60 completed windows and neither the fail rule nor the directional gate is met | Keep accruing. This is not evidence against the model: with an edge near the no-hindsight backtest estimate (about +0.25 pt a week against a weekly standard deviation near 2.3 pt) the chance of clearing t ≥ 1.65 at 20 windows is roughly one in eight |
| Read-out | 60 completed windows without clearing the directional gate | Treated as evidence against the model; operator decision on continuing |
| Selector-only rollback | after at least 20 completed windows: selector-only minus the shadow `pairwise_minimal` basket has paired weekly t ≤ −1.65 (`tools/selector_only_shadow.py` ledger) | Operator reviews reverting to `--ranker-mode pairwise_minimal`. A smaller gap is noise at this sample size |

None of these acts automatically. Each is a trigger for an operator decision.

Clearing the confirmation gate does **not** promote the model. It makes the model *eligible* for an operator promotion decision. Promotion, unfreeze, and gate-clearance remain explicit operator actions.

**Estimated confirmation-eligibility date:** ~2026-10-31 (≈ 20 forward weeks from the v1.4 lock), consistent with the gate noted in the full-history validation report.

---

## 5. Daily truth card

Each trading day, produce one page: **DEM Daily Forward Validation Card**. Append-only; one immutable row per day.

- Date
- Candidate model hash + ruleset hash (`8887576e`)
- Current Top-30 (tickers + ranks)
- Rank changes from prior day
- XBI close
- Price coverage (n names with valid split-adj close / 30)
- Data-quality status
- Corporate-action flags
- Endpoint parity status (DEM and XBI use same as-of date)
- Last completed 1d / 5d / 20d return
- Cumulative candidate-period return vs XBI vs excess
- Hit rate
- Weekly non-overlapping t-stat (running)
- Worst 4-week excess drawdown
- Adversarial-control status
- Whether the candidate remains valid
- Whether operator action is required

Suggested artifact path (specification only — **NOT wired** by this document; wiring is deferred, see §10): `artifacts/live_shadow/forward_validation/<YYYY-MM-DD>/TRUTH_CARD.md`, alongside the existing `artifacts/live_shadow/go_nogo/<date>/GO_NOGO.md`.

---

## 6. Adversarial controls

To distinguish selection skill from biotech beta / rally participation, each weekly window also computes:

- **Random-30 control:** equal-weight random 30 from the eligible universe (seed pinned per week for reproducibility).
- **Inverse-rank control:** equal-weight bottom-30 by `actionable_rank`.
- **XBI-only:** pure benchmark.

The candidate's excess must remain materially above the random-30 control and above the inverse-rank control. If the random-30 control captures most of the candidate's excess in a window, that window's excess is flagged BETA_NOT_SKILL.

---

## 7. Weekly and monthly summaries

**DEM Weekly Validation Summary** (every Monday):
- Last completed 5d Top-30 return, XBI 5d, XS, IC
- Control-basket comparison (Section 6)
- Cumulative forward-validation result
- Data-quality exceptions
- Whether the candidate remains valid

**DEM Monthly Validation Summary** (month-end):
- Monthly chain-linked Top-30 return vs XBI vs excess
- Drawdown
- Regime context (from regime shadow)
- Rank efficacy
- Alpha-stream beta vs portfolio beta (per the beta decomposition in `MODEL_DOCUMENTATION.md`)
- Keep / freeze / re-spec recommendation

Daily cards form the evidence ledger; weekly and monthly summaries form the decision record.

---

## 8. Data-integrity discipline (carried from YTD audit)

The two YTD bugs (raw-vs-split-adjusted price source; asymmetric DEM/XBI endpoints) must not recur in forward data:

- Forward returns use `price_history_split_adj.csv` only; any raw fallback is row-flagged and counted against price coverage.
- DEM and XBI returns must use identical as-of endpoints every window (endpoint-parity check, fail-closed).
- Corporate-action dates flagged per window; spinout/split windows carry an explicit treatment tag.
- Missing prices fail the row — never silently substituted.

---

## 9. Honest framing language (required)

Use:
- "directionally supportive", "not yet statistically confirmed", "diagnostic only", "requires forward validation", "operator clearance still required", "no production behavior change".

Avoid:
- "proven alpha" (unless the confirmation gate is cleared AND operator promotes), "investable" (unless governance explicitly says so), "bug explains performance" (unless the counterfactual confirms it).

Current honest verdict: **"The model is behaving as if it has real cross-sectional skill, and the corrected data supports that — but the current locked version is not yet statistically confirmed out-of-sample."** Not: "the current locked version is proven."

---

## 10. Governance

- `model_change: False`
- `ranker_change: False`
- `selector_change: False`
- `sizing_change: False`
- `production_wiring_change: False`
- `trading_action: False`
- `classification: PRE_REGISTRATION_NO_MODEL_CHANGE`
- Freeze status: 2026-06-20 scoped architecture freeze remains in effect; unchanged by this document.

### Ratification log

| Date | Action | By | Note |
|------|--------|----|------|
| 2026-06-28 | DRAFT created | Town assistant (GitHub API) | Pre-registration authored; merged via PR #428 |
| 2026-06-28 | **RATIFIED** | Operator (Darren) | Test specification locked. The §2 primary test must not be re-specified after forward data is seen. |
| 2026-11-15 | §4 amended | Operator | Fail, read-out and selector-only rollback rules added before `oos_clock_start` for the selector-only order is recorded (0 windows observed). The §2 test and the existing pass gates are unchanged. Memo: `docs/governance/2026-11-15-pre-clock-decisions.md`. |

### Deferred — NOT yet authorized

- **Wiring the daily truth-card artifact (§5) into the pipeline is explicitly deferred.** The operator has NOT authorized it. Ratifying this protocol does **not** authorize building, scheduling, or cron-wiring the daily card, weekly/monthly summaries, or any artifact generation. That remains a separate, governed implementation task requiring its own explicit operator approval (and would touch `tools/run_daily_production.py` / cron — outside the observe-only/docs lane).

---

*This document pre-registers a test. It does not run one, wire one, or change the model. Implementation of the daily card / weekly / monthly artifacts is a separate, explicitly-authorized task that has NOT been approved.*
