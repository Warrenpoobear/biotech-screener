# Shadow Guard Promotion Gates

**Classification:** VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

**Branch:** `research/failure-mode-shadow-guards-2026-06-28`  
**Created:** 2026-06-28  
**Status:** ACCUMULATING — no gate met

---

## Overview

Three shadow monitors were deployed following the YTD Top-30 failure post-mortem
(2026-06-28). They are **observation-only** infrastructure: they log baskets,
compute forward returns, and surface warnings. None may cause a production model
change until the gates in this document are met and an operator explicitly authorizes.

Controlled promotion ladder: **shadow → evidence → gate → operator review → production**

No step may be skipped. No gate is waived by strong initial results.

---

## Shadow Monitor A — Repeat-Offender Persistence

**Script:** `scripts/research/repeat_offender_monitor.py`  
**Output:** `artifacts/failure_mode_guards/repeat_offender_{date}.json`  
**Seed:** YTD post-mortem `name_attribution.csv` (7 windows, 210 rows)

### What it tracks

Names currently in Top-30 that appeared in the bottom-5 contributors
(by `contrib_to_xs`) in ≥2 of the last 12 completed windows.

### Promotion gate

| criterion | required | status |
|-----------|----------|--------|
| Serial bottom-5 windows | ≥10 rolling windows of evidence (beyond YTD seed) | ACCUMULATING |
| Persistence signal | Flagged names must show negative forward return on average | UNTESTED |
| Non-degradation | Clean (non-flagged) names must not degrade when substituted | UNTESTED |
| Operator review | Written authorization required | NOT STARTED |

### What promotion means

Adding a **soft warning** (advisory only) to the weekly rebalance output listing
flagged names and suggested replacements. **Not a veto. Not a model change.**
The ranker, selector, sizing, and final_score are unchanged.

### What promotion does NOT authorize

- Automatic removal of flagged names from the portfolio
- Changes to `actionable_rank`, `final_score`, or any ranker weight
- Hard-veto of `build_window` or `less_binary` catalyst buckets
- Any change to selector blocks

---

## Shadow Monitor B — EES-Guarded Top-30

**Script:** `scripts/research/ees_guarded_shadow.py`  
**Output:** `artifacts/failure_mode_guards/ees_guarded_baskets.jsonl`, `ees_guarded_returns.jsonl`  
**Lead signal:** `raw_veto_core` (IC=0.064, t=2.36, LATE +7.1%)

### What it tracks

Two baskets logged each week:
- `raw_top30` — unmodified model selection
- `guarded_top30` — EES-fail names removed, filled from ranks 31–60

Forward XS is computed for each once prices are available.
`delta_pp = guarded_xs − raw_xs`.

### Promotion gate

| criterion | required | notes |
|-----------|----------|-------|
| Completed windows | ≥20 | Starting from 2026-06-28 |
| Avg delta | Positive (guarded > raw) | Must hold across regime types |
| Pct positive | ≥60% of windows where delta > 0 | Not just in stress weeks |
| Normal-week check | No degradation in non-failure weeks | Avoids over-exclusion |
| Statistical | t-stat ≥ 1.5 on delta series | Minimum credibility bar |
| Operator review | Written authorization required | |

### What promotion means

Adding `ees_v3_gate=False` as a **selector-level soft block** — advisory flag
in ranking output, not a hard exclusion. Production portfolio is unchanged until
a separate capital-authorization decision is made.

### What promotion does NOT authorize

- Hard removal of EES-fail names from production portfolio
- Changes to `ees_v3_score` weighting in final_score
- Changes to EES thresholds or `raw_veto_core` policy parameters
- Treating EES as "proven" or "alpha-generating"
- Claiming EES prevents all failure windows

---

## Shadow Monitor C — Rank-Depth Replacement

**Script:** `scripts/research/rank_depth_replacement_shadow.py`  
**Output:** `artifacts/failure_mode_guards/rank_depth_baskets.jsonl`, `rank_depth_returns.jsonl`

### What it tracks

Five baskets logged each week:

| basket | description |
|--------|-------------|
| `top30` | Unmodified model selection |
| `rank31_60` | Ranks 31–60 equal-weight (bench cohort) |
| `top60_ew` | Full top-60 equal-weight |
| `ees_guarded` | Top-30 with EES-fail excluded, bench fill |
| `ro_guarded` | Top-30 with repeat-offender flagged excluded, bench fill |

**Key question:** Is the bench (31–60) advantage *failure-specific* (a stress hedge)
or *persistent* (evidence the rank cutoff is wrong)?

### Promotion gate — threshold change

| criterion | required |
|-----------|----------|
| Completed windows | ≥20 |
| Bench avg delta vs Top-30 | > +0.5pp (trigger for "persistent" flag) |
| Pct positive | ≥65% of windows |
| Failure vs non-failure breakdown | Bench advantage concentrated in failure weeks |
| Operator review | Written authorization |

### What a "PERSISTENT" verdict authorizes

If the bench (31–60) consistently and significantly outperforms Top-30
(not just during stress), the operator may authorize:
- A **diagnostic review** of the rank cutoff boundary
- Running a counterfactual backtest on top-50 vs top-30

This does **not** authorize:
- Changing `actionable_rank` threshold in production
- Changing ranker weights to push ranks 31–60 higher
- Any portfolio position change

### Normal expected outcome

Bench advantage concentrated in failure windows → classified as **FAILURE-SPECIFIC**
(a sign that repeat offenders and EES-fail names degraded top-30 in those weeks,
not that the ranker is broken). This is the expected outcome.

---

## Rolling 4-Window Drawdown Trigger

**Script:** `scripts/research/rolling_drawdown_trigger.py`  
**Threshold:** −5.00pp rolling 4-window XS sum  
**Output:** `artifacts/failure_mode_guards/drawdown_trigger_{date}.json`

### What it triggers

When rolling 4-window XS ≤ −5.00pp, the trigger fires and:
1. Prints the repeat-offender summary
2. Prints the EES-guarded shadow summary
3. Lists 5 required remediation steps

### What it does NOT trigger

- Automatic liquidation or position change
- Suspension of the weekly rebalance
- Any model modification

Operator must explicitly decide next steps after reviewing all four monitors.

---

## Escalation Ladder

```
Shadow accumulating (0–9 windows)
  → No action. Log weekly. Read monthly.

Gate-approaching (10–19 windows, positive signal)
  → Flag in weekly card. Operator briefed. Still no action.

Gate-met (20+ windows, all criteria)
  → Written promotion request filed by research
  → Operator reads: all 20+ window results, per-window breakdown,
    regime breakdown, normal-week check
  → Operator decides: authorize, defer, or reject

Authorized
  → Research implements the specific authorized change only
  → Change is advisory output only (no production model modification)
  → New shadow monitor opened for the advisory output itself
```

---

## What Is Permanently Off-Limits

Regardless of shadow evidence:

| action | reason |
|--------|--------|
| Remove CELC/ABVX manually | Names may exit naturally; no discretionary overrides |
| Hard-veto `build_window` bucket | Bucket finding requires separate statistical validation |
| Promote EES into production selector | Requires capital-authorization from operator, not just shadow gate |
| Change `final_score` weighting | Outside scope of failure-mode shadow guards |
| Change `actionable_rank` cutoff | Requires full ranker re-validation |
| Treat options EP as explanatory | Options data coverage insufficient in 2026 failure windows |
| Use shadow evidence to claim "EES is proven" | Shadow delta ≠ alpha; confounded by regime, sample, survivorship |

---

## Weekly Card

**Script:** `scripts/research/weekly_failure_mode_card.py`

Runs all four monitors and produces:
- `artifacts/failure_mode_guards/FAILURE_MODE_CARD_{date}.md`
- `artifacts/failure_mode_guards/failure_mode_card_{date}.json`

Run every Monday after snapshot generation:
```bash
python3 scripts/research/weekly_failure_mode_card.py --snap-date YYYY-MM-DD
```

---

## Revision Log

| date | change |
|------|--------|
| 2026-06-28 | Initial version — 3 monitors + drawdown trigger deployed |
