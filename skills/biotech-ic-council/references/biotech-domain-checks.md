# IC Council — Biotech Domain Checks

Companion reference for `skills/biotech-ic-council/SKILL.md`. Load this when a review touches event EV / the expectation layer, CT.gov/FDA catalyst logic, corporate actions, expectation-layer fields, Hermes/Wake Robin artifacts, or backtest validity. It expands the "Biotech-specific review anchors" section of the skill into concrete, checkable items.

These are checks, not authorizations. The council applies them read-only; any change they implicate to scoring, ranking, gates, or sizing still requires a separate model-change review.

## Event EV / expectation layer

The expectation layer estimates how much of a catalyst is already priced. It is NOT an alpha selector — keep the two separate in every finding.

- Confirm expectation features are used to answer "is this already priced?" — not "does this predict return?". A change that improves expected-move estimation is plumbing, not alpha, unless forward-validated.
- Verify the market-implied fields are present and sane: `short_interest_pct`, `priced_move_pct`, `market_cap_mm`, `close_price`, and any options/straddle-implied move fields.
- Watch for the recurring confusion (`IC_EXPECTATION_*`): an expectation feature being credited as ranker/selector alpha. Flag it.
- Insider net-buying coverage is the known weak spot — if a finding leans on it, confirm coverage exists rather than assuming it.

## Clinical / catalyst logic (CT.gov + FDA)

- Effective dating: a CT.gov change is observable via `lastUpdatePostDateStruct.date`; the feature must take effect on the **next trading day**, never the event day itself (no same-day leakage).
- Distinguish trial-status **upgrades vs downgrades**, timeline **push vs pull**, and confirm `ACTUAL` date confirmation vs estimated/anticipated.
- Treat "results posted" as a distinct event from a status flip — they have different severity.
- Severity / noise-band thresholds: confirm the change respects the documented bands rather than re-deriving them. Any change to the bands themselves is model-affecting (needs a Spec).
- Separate true clinical information from post-hoc market interpretation: a price move is not clinical evidence.

## Corporate actions

Every biotech price series must be checked for corporate-action contamination before any return is trusted:

- **Authoritative registry:** `production_data/corporate_actions.json` (loader: `common/corporate_actions.py`). YTD model-relevant M&A log: `artifacts/acquisitions/biotech_model_acquisitions_ytd_2026.md`.
- reverse splits and forward splits (split-adjusted vs raw price)
- spinouts and special distributions
- M&A (cash, stock, CVR) — confirm the **close** date (not announcement date) and whether the name is mid-deal. `deal_price` = cash component; CVR terms go in `notes`.
- delistings and ticker changes — confirm the last valid candle and `DATA_STALE` handling
- survivorship bias — confirm delisted names are retained in historical universes, not silently dropped
- **`pending_acquisition`:** documented in JSON but not loaded by `ACTION_TYPES`; does not trigger `is_dead()` until promoted to `acquisition` at close

If raw (unadjusted) prices could have entered a feature, that is a BLOCKER (`IC_CORP_ACTION_*`).

## PIT discipline

- Source dates must be observable on or before the snapshot date. A future-dated source in a historical snapshot is PIT leakage.
- Forward returns must never enter features under any circumstance.
- Generated outputs must not feed input hashes unless the freeze is intentional and documented.
- When provenance evidence (source_date, effective_date, hashes) is missing, mark `unobserved` — do not infer it.
- **Regenerated histories: check every input class separately, per date** (2026-11-17 v9 audit). A "PIT" rebuild can be clean on one input and leak on another:
  - 13F: coinvest built from filings available on that date (exact per-date cache), not the nearest quarter-end cache or today's `coinvest_signals.json` (silent fallback below a coverage gate).
  - Market data: per-date `market_data.json`; no `price_date` after the snapshot. Historical runs that read today's market cap carried look-ahead (2026-11-13 deprecation).
  - Financials: filed ≤ as-of date, and no fallback to the static file.
  - CT.gov: per-date trial caches admit trials by `first_posted` but carry the record **as captured later** — status, phase and results posted after the as-of date leak. Count `last_update_posted > as_of` per date; a date-based rollback is an approximation (phase and date revisions are not recoverable without version history).
  - Missing per-date caches fall back to a different file; list them.
- **External reference data carries its own as-of date.** ETF holdings, rosters, registries: read the file's as-of date, not the fetch time (a hard-coded `asOfDate` served June IBB holdings for five months).
- **Replacing a live snapshot directory:** before approving, confirm the forward-validation ledger stores baskets inline (or references a content hash), not a path. Keep the live directory in a backup folder and record its hash.

## Hermes / Wake Robin artifacts

Know which artifact a finding touches and its blast radius:

- `rankings.csv` (the production sort surface), snapshots, sidecars, provenance metadata
- `run_screen.py`, the production wrapper, the universe loader, price history, clinical cache, event outputs
- `final_score`, the selector, the ranker, and the gates — any change here is model-affecting, full council, Spec-gated.

Architecture reminder for context (do not re-litigate it in a review): since 2026-11-13 the production pipeline is `universe → A4 selector (selector_score; coinvest_score_z is the institutional block) → selector-only order (final_score = selector_score) → actionable_rank → DEM top-30`; the pairwise `minimal_v2` ranker is scored as a shadow only. "DEM" is the stored production top-30 of the current ranker, not a separate baseline algorithm. Do not present the production top-30 as a competitor "baseline" to its own selector overlay (failure mode `IC_BACKTEST_` / F-2026-010).

## Backtest validity

When a finding cites backtest evidence, check:

- **Regime mix** — is the result concentrated in a few rally months? Report ex-best-months performance, not just the headline.
- **Secondary windows** — does the edge survive outside the primary window and outside extraordinary XBI rallies?
- **XBI-relative** — are returns excess vs XBI, or is "alpha" actually beta/convexity?
- **Selection-count stability** — does the top-N count drift run to run?
- **IC with sample size** — an IC claim needs n and a t-stat; reject false precision and single-date cherry-picks.
- **Newey-West / t-stat** — confirm the standard-error treatment matches the overlap structure.
- Confirm the IC tool declares **which score field** it measures and that the field matches the production sort key (final_score), per failure mode F-2026-002.
- **Selection after a sweep is not evidence.** If a parameter (Top-N, cap, threshold) was picked from a sweep, say how many variants were tried and test the pick out of sample; check the shape is monotone (e.g. rank buckets 1-10 > 11-20 > 21-30) before believing a winner.
- **Roster / registry hindsight.** If the input set (13F manager roster, universe additions) was chosen on trailing performance, split the backtest at the selection window and quote both rosters (2026-11-16/17: v4 roster tied v3.2 before the window, all of its lead inside it).
- **Overlapping horizons.** Monthly starts with 3/6/12-month holds need Newey-West t with h-1 lags, and the effective sample is roughly years, not months.
- **Cost convention stated.** "50 bps × one-way turnover" is ~25 bps per trade. State per-trade cost and AUM; check impact scales with trade size vs ADV.
- **Paired comparisons.** Variant vs incumbent must be a paired difference on the same dates, with its own t; two separate t-stats are not a comparison.

## Production reliability

- Deterministic replay must hold: same inputs → same outputs.
- Schema compatibility and an output-diff projection for any change touching `rankings.csv` or snapshot outputs.
- Cache timeouts and fallback sources behave safely on miss.
- CI status is read and **stated explicitly** — never assumed green. A red or unobserved CI blocks any production-path approval above `hold`.
- **Check the same CI job on `main` before reading a red check as evidence about a PR.** A permanently red check (e.g. `replay` against a stale golden baseline, red 400+ runs) masks real regressions; the finding is "fix the baseline", not "this PR regressed".
- **Every writer of a file honours the same config.** When an operator config (e.g. coverage exemptions) is added, find every tool that rewrites the target file; one ignoring it silently reverts the operator's decision (2026-11-17: fetch step demoted the #696 names).
- A rollback command/path is named for every production change.

## Portfolio / risk

- ADV / liquidity and small-cap slippage realism.
- Crowding in elite-manager-owned names; short-interest squeeze/fragility.
- Options-implied move vs the model's expected move.
- Concentration and drawdown; whether the change should affect **sizing** (Spec-gated) or only ranking/review.
- **Single-factor check.** Measure the correlation of the selector's IC with its dominant input's IC (0.94 for coinvest). If ≈1, the portfolio is a one-factor bet: judge diversification at the allocation level (sleeves, mixes) rather than by capping the factor inside the selector (caps and crowding limits were rejected 2026-11-15 and 2026-11-17).
- **Judge a strategy on the yardstick its mandate uses.** A low-beta sleeve fails XBI-relative rules by design; say which measure applies (XBI-relative, beta-adjusted alpha, Sharpe, absolute drawdown) before reading a result.
- **Clock-reset churn.** Count out-of-sample clock resets in the window. Repeated small resets mean no forward evidence ever accrues; prefer batching changes into one governed reset.
