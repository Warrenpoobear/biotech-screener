# Deprecation of backtest claims affected by market-cap look-ahead (2026-11-13)

**Status:** IN FORCE. Operator signed off by merging PR #660 on 2026-11-13.
**Classification:** `BACKTEST_CORRECTION / EVIDENCE_DEPRECATION`. Documentation only.
**Model effect:** none. No ranker, selector, eligibility, sizing, ruleset or production-run change. No out-of-sample clock effect. The model stays **FROZEN (BLOCKED_LEVEL_0)**.

## Finding

Historical `run_screen.py --as-of-date` runs read today's `production_data/market_data.json`. Every regenerated snapshot set therefore carries today's market cap on every past date. `module_2_financial_v2.py` divides cash and burn by market cap, so historical `financial_score` was lower for companies whose market cap later grew. The `minimal_v2` ranker weights `financial_score` negatively.

Mechanism, evidence tables and limits: `docs/governance/2026-11-13-historical-run-market-data-lookahead.md` (PR #659).

Same model (ruleset `8887576e`, commit `65829ced`), same inputs, 79 monthly periods 2020-01-31 → 2026-06-26, Top-30, net of 50 bps, XBI gross:

| | Today's market data | Point-in-time market data |
|---|---|---|
| Mean period excess | +1.64 pt, t = 2.99 | +0.68 pt, t = 1.47 |
| `financial_score` monthly IC, eligible names | −0.067 (t = −6.9) | −0.001 (t = −0.1) |
| Selector-only minus current rule | −0.60 pt (t = −1.69) | +0.62 pt (t = +1.64) |

## Claims deprecated

Each was computed on `data/snapshots_pit_v2`, an earlier regeneration, or a panel built from one.

| Claim | Where it is stated | Why |
|---|---|---|
| "+2.34pp/mo net-of-cost, t=2.57, 69% hit, 67 monthly periods" (Spec 050 true PIT backtest) | `research-backtest.md` Operating Truths 1, Trust Buckets, Promotion Story 1–2; `external-intel.md` developer profile | Portfolio selected with look-ahead `financial_score` |
| Checklist v2 bundle: bootstrap +2.42pp/mo, CI [1.25%, 3.70%], LOSO ROBUST | Operating Truths 2, Trust Buckets, Promotion Story 3 | Same history |
| Pairwise feature audit: `financial_score` NW-t = −3.41, `inst_delta_z` NW-t = +3.32, coinvest +0.49 | Operating Truths 3, Promotion Story 5 | Measured inside a Top-30 chosen with look-ahead; the `financial_score` relation disappears with point-in-time market cap |
| K=30 sweep, K=25–35 plateau | Operating Truths 6, Trust Buckets, Promotion Story 8 | Same history |
| Regime decomposition (bear +3.37pp, neutral +6.23pp, bull −0.37pp) | Operating Truths 7, Promotion Story 9 | Same history |
| `minimal_v2` ranker training evidence | `production_data/ranker_v2_model.json` (36 dates, 12,400 pairs) | Weights fitted on the same history |
| Spec 093: negative `financial_score` sign is intentional stress upside | `operational-state.md` Active Spec Status | The sign it explains is largely the look-ahead |
| "+0.33 pt/period, t = 0.73" rerun on `snapshots_pit_v2` (2026-11-13 morning) | `research-backtest.md` Canonical Benchmark Commands note; `MODEL_DOCUMENTATION.md` | Old model vintage on look-ahead history; not a clean reading in either direction |

## What is not deprecated

- The forward monitor and the live captures. They read each day's market data on that day.
- Live-date IC readings from 2026-04-03 onward that use live snapshots (for example the corrected `actionable_rank` reading in the 2026-11-12 Path C re-record). Regenerated dates inside that range (2026-08-17 → 2026-10-13, and the 2026-11-03 regenerations) are affected.
- Dead-lane conclusions. They were negative on contaminated history and are not relied on for anything.
- Engineering facts: tests, tooling, the gate set.

## What replaces them

Nothing is promoted in their place.

- The point-in-time regeneration (`data/snapshots_pit_v4`) is pseudo-PIT research: survivorship-biased, in-sample for the selector settings, with 20% of ticker-months still carrying today's share count. Its +0.68 pt/period (t = 1.47) is not statistically significant.
- The forward monitor remains the only out-of-sample evidence.

## Open decisions for the operator

1. ~~Sign off on this deprecation.~~ Done 2026-11-13 (PR #660 merged).
2. **Ranker.** The corrected history favours selector-only by +0.62 pt/period (t = 1.64), which is suggestive, not significant. The live selector-only shadow (PR #653) is accruing forward evidence. Removing or refitting the ranker is a governed model change with a clock reset and needs Checklist v2 on clean history.
3. **`run_screen.py`.** Whether a historical run should refuse to read a `market_data.json` dated after `--as-of-date`. Tier 3, frozen file; not changed here.
4. **Spec 093.** Reopen or annotate.

## Changes in this PR

- `.claude/rules/research-backtest.md`: banner, per-claim annotations, entries moved from "Safe to use now" to "Deprecated", evidence hierarchy reordered, benchmark commands updated.
- `.claude/rules/operational-state.md`: Governance Events (2026-11-13).
- `.claude/rules/external-intel.md`: one line in the developer profile.
- `docs/MODEL_DOCUMENTATION.md`: Recent Update entry.

Follow-up (2026-11-13): the selector-ranker skill, its Hermes mirror, `docs/library-export/*`, the root `model_documentation.md` and one Hermes debug skill were annotated. Left as written: ruleset descriptions in `production_data/decision_rulesets/manifest.json` (historical record of what each ruleset claimed when registered) and dated audit and spec artifacts.
