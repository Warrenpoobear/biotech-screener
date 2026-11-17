# Proposal: 50/50 Top-30 + low-risk (volatility–beta) sleeve

**Date:** 2026-11-17
**Status:** PROPOSED — not approved. Nothing in this memo takes effect until the operator approves it in writing (merging this memo is not approval of the change; approval is recorded in §9).
**Classification:** Tier 4 governance decision. Portfolio-construction model change. Resets the out-of-sample clock.
**Scope:** what the portfolio holds and how it is judged. The selector, ranker, scoring, eligibility and the production Top-30 itself are **unchanged**.

---

## 1. Decision requested

Replace "equal-weight production Top-30" with:

> **50% production Top-30 + 50% low-risk sleeve**, the sleeve being the 30 lowest-risk eligible names outside the Top-30, ranked on 60-day volatility and 120-day beta to XBI.

Together with the yardstick changes this requires (§5, §6), adopted as **one** governed change with **one** clock reset, followed by a **six-month no-change window** (§7).

## 2. Why

1. **The model is effectively one factor.** The selector's IC tracks the coinvest IC almost exactly (correlation 0.94 over 84 months). When the factor reverses the whole book goes with it: August–September 2026 cost the Top-30 −21.9% over three months against XBI −0.8%.
2. **Stock-selection add-ons do not fix it.** About 27 add-on and alternative signals were tested on 2026-11-15/17 (13F rebuilds, crowding caps, Top-N, 13D/G, Form 4, financings, FDA events, takeovers, exclusivity, momentum, timing rules, and others). None clears the pre-registered bar (t > 3). See the 2026-11-17 research entry in `docs/MODEL_DOCUMENTATION.md`.
3. **Diversification at the portfolio level does help, in-sample.** A low-risk sleeve is nearly uncorrelated with the Top-30's excess return and roughly halves drawdown while keeping the long-run excess over XBI.
4. **The edge of the Top-30 itself is small.** On the v9 history the roster-adjusted range is about +0.3 to +0.7 pt/month against XBI, not significant. Concentrating the whole book on it is the larger risk.

## 3. Specification

| Item | Specification |
|---|---|
| Half A | Production Top-30, exactly as today (`actionable_rank` 1–30, selector-only, ruleset `8887576e`), equal weight within the half |
| Half B | Low-risk sleeve: eligible names in that day's `rankings.csv` **not** in the Top-30; risk score = mean of percentile ranks of `vol_60d` (annualised std of daily log returns, last 60 days) and `beta_120d` (beta of daily log returns to XBI, last 120 days); lower risk = higher score; the 30 best-scoring names, equal weight within the half |
| Inputs | Split-adjusted closes dated on or before the snapshot date (`production_data/price_history_split_adj.csv`); no other inputs |
| Exclusions | Names with 20-day annualised volatility < 10% (price pinned near a pending cash-deal price, or stale feed) — as in the existing shadow tool |
| Split | 50% / 50% of account value |
| Reconstitution | Monthly, at the first rebalance after month-end, from that day's promoted snapshot (both halves) |
| Reweighting | Weekly (Monday), back to equal weight within each half and 50/50 between halves — the account's existing weekly practice (Top-30: +0.14 pt/month after costs, t 2.69; sleeve: +0.04, t 1.32 — optional for the sleeve, kept for one simple rule) |
| Industry cap (optional guardrail) | At most 15 of the 30 sleeve names from one industry (point-in-time `industry` field). Binds in about 2% of months historically and changes nothing; it only guards against a future pile-up. Operator choice (§9, decision 6) |
| Overlap | None by construction (the sleeve excludes the Top-30) |

This is the `vol_beta` variant registered as a capture-only shadow on 2026-11-16 (`tools/low_risk_sleeve_shadow.py`, ledger `artifacts/low_risk_sleeve_shadow_vol_beta/`), chosen **before** the 2026-11-17 measure search. The three-measure base sleeve (adds `max_gain_21d`) is not proposed: `max_gain_21d` was the weakest measure in both halves of the 2026-11-17 search.

## 4. Evidence (in-sample; research only)

`snapshots_pit_v9` (85 month-ends; 13F per date, point-in-time market data and financials, CT.gov date rollback), March 2020 – November 2026 (82 months), monthly, net of 50 bp × one-way turnover (≈ $1M per-trade costs):

| Portfolio | Ann. return | Volatility | Sharpe | Max drawdown | Beta | CAPM alpha/month (t) | Worst stretch vs XBI |
|---|---|---|---|---|---|---|---|
| Top-30 (today) | 18.4% | 31% | 0.70 | −50% | 1.00 | +0.72 (1.45) | −24% |
| Vol–beta sleeve alone | 21.9% | 15% | 1.39 | −11% | 0.34 | +1.40 (3.57) | −29% |
| **50/50 Top-30 + vol–beta (proposed)** | **21.1%** | **21%** | **1.03** | **−25%** | **0.67** | **+1.06 (3.10)** | **−15%** |
| 50/50 Top-30 + three-measure sleeve | 19.3% | 21% | 0.95 | −26% | 0.69 | +0.92 (2.78) | −14% |
| XBI | 9.8% | 26% | 0.49 | −57% | 1.00 | — | — |

- Mean excess over XBI is the same as the Top-30 alone (+0.72 pt/month) at two-thirds of the volatility and half the drawdown.
- By year against XBI: 2020 −27.4 pt, 2021 +22.1, 2022 +17.3, 2023 +13.2, 2024 +8.8, 2025 +22.8, 2026 to date −2.3. The cost is strong rallies (2020).
- 2026: year to date +21.7% (Top-30 alone +10.7%, XBI +19.6%); last three months −3.3% (Top-30 alone −21.9%).
- 2024–2026 only (after the measures were fixed in the 2026-11-17 search's selection period): Sharpe 1.40, alpha +1.10 pt/month (t 1.95).

### 4a. Weight between the halves

Same 82 months, blends of the two halves:

| Sleeve share | Ann. return | Sharpe | Max drawdown | Beta | Alpha/month (t) | Worst stretch vs XBI | Worst 12 months vs XBI | 12-month periods behind XBI |
|---|---|---|---|---|---|---|---|---|
| 0% (Top-30 only) | 18.4% | 0.70 | −50% | 1.00 | +0.72 (1.45) | −24% | −14% | 26% |
| 25% | 19.9% | 0.84 | −38% | 0.84 | +0.89 (2.21) | −16% | −9% | 21% |
| **50% (proposed)** | **21.1%** | **1.03** | **−25%** | **0.67** | **+1.06 (3.10)** | **−15%** | **−11%** | **13%** |
| 75% | 21.7% | 1.24 | −15% | 0.51 | +1.23 (3.61) | −17% | −28% | 20% |
| 100% (sleeve only) | 21.9% | 1.39 | −11% | 0.34 | +1.40 (3.57) | −29% | −49% | 23% |

Mean excess over XBI is the same at every split (+0.70 to +0.72 pt/month). 50/50 has the smallest XBI-relative risk of any split; more sleeve improves absolute and beta-adjusted figures but widens the XBI-relative risk sharply (the 100% sleeve trailed XBI by 49 pt over 12 months in the 2020 rally) and drops the coinvest model entirely. 50/50 is proposed because Rule 4 and the mandate are XBI-anchored; choosing 75% or 100% from this table would be choosing with hindsight.

### 4b. Sleeve design checks (2026-11-17)

Each design choice was varied one at a time on the same data and costs, with the selection period (March 2020 – 2023) kept separate from 2024–2026 where a choice was searched. The proposed specification came out best or tied on every dimension:

| Dimension | Tested | Result | Specification |
|---|---|---|---|
| Risk measures | 8 measures alone and in equal-weight pairs | Beta is the key ingredient; volatility + beta beats the three-measure base sleeve (alpha +1.40 vs +1.10 pt/month); `max_gain_21d` is the weakest measure in both periods; the selection-period winner did worse than volatility + beta in 2024–26 | 60-day volatility + 120-day beta, equal weight |
| Lookback windows | 6 volatility × 3 beta windows (18 cells) | All 18 have alpha t 2.9–3.8; the rank correlation between selection-period and test-period Sharpe across cells is −0.03 (picking a "best" window is noise); 252-day beta slightly weaker | 60 / 120 days (on the plateau; 2024–26 Sharpe 1.83) |
| Number of names | 10, 15, 20, 25, 30, 40, 50 | Smooth plateau from 25 to 40; 30 has the best Sharpe (1.39), lowest drawdown (−11%) and turnover; 20 names trail 30 by −0.29 pt/month (t −1.61), 6 of 7 years | 30 |
| Weighting | Equal, inverse volatility, inverse variance (10% cap) | Equal is best; inverse volatility −0.09 pt/month, inverse variance −0.17, both worse in 6 of 7 years, with more turnover and concentration | Equal weight |
| Market cap | Sleeve built within small (<$2B), mid ($2–10B), large (≥$10B) | Small caps earn most (+1.37 pt/month vs XBI) with twice the drawdown and a weaker 2024–26; mid caps weakest; large caps stabilise; the all-cap mix has the best Sharpe and lowest drawdown | All caps (no restriction) |
| Rebalance frequency | Monthly, every two weeks, weekly | Weekly +0.04 pt/month (t 1.32), all in the first half of the sample | Monthly reconstitution; weekly reweighting optional |
| Industry caps | None, 15, 10, 8 per industry | Uncapped sleeve averages 12 biotechnology, 8 specialty/generic, 6 large drug makers; cap 15 binds in 2% of months (no effect); cap 10 no gain; cap 8 −0.15 pt/month and drawdown −17% | None (15 as optional guardrail) |
| Rank buffer (hold a name until it falls past rank 40, 50 or 60) | Turnover 17%/month with no buffer, 11% at rank 40, 8% at rank 60; at 50 bp costs rank 40 adds +0.14 pt/month (t 0.92, better in 3 of 7 years) but maximum drawdown deepens from −11% to −16%; at 150 bp costs +0.19 (t 1.30, 4 of 7 years) | No buffer (scaling option, §8) |

The edge comes from preferring low-volatility, low-beta names at all, not from fine-tuning; further tuning would be fitting noise. Scripts are in the 2026-11-17 session scratchpad (`lowvol_opt.py`, `volbeta_windows.py`, `volbeta_n.py`, `volbeta_weights.py`, `volbeta_capbuckets.py`, `volbeta_freq.py`, `volbeta_sectorcap.py`, `volbeta_buffer.py`, `sleeve_vs_mix.py`).

**Limits.** In-sample; the history keeps only names that survived to 2025 (survivorship rebuild pending); the sleeve definition was informed by data covering the same period; the 13F roster carries hindsight; 82 months is a short sample; no forward evidence yet (the PR #697 and `vol_beta` shadows started 2026-11-16/17).

## 5. Account Rule 4 must change with it

Rule 4 today: review at −10 pp and liquidate at −15 pp of drawdown **against XBI**. A 0.67-beta book trails XBI in strong rallies by design — the proposed portfolio's worst historical stretch against XBI is −15%, which would have triggered a full liquidation while the portfolio was doing its job.

Options (operator to choose one):

| Option | Rule | Historical behaviour of the proposed portfolio |
|---|---|---|
| **A (recommended)** | Measure Rule 4 against a **beta-adjusted benchmark** (0.67 × XBI + 0.33 × cash), same −10 / −15 pp levels | Judges the portfolio on the risk it takes; needs a re-run of the touch frequency before adoption |
| B | Keep XBI-relative, widen to −20 / −25 pp | Simple; does not distinguish a beta gap from a model failure |
| C | Absolute drawdown: review at −20%, liquidate at −30% | Historical max drawdown −25% |

## 6. The forward-validation mandate must change with it

`docs/FORWARD_VALIDATION_PROTOCOL.md` §2 tests mean weekly excess of the DEM Top-30 over XBI. Proposed amendments (same structure, new primary statistic):

- **Primary:** CAPM alpha of the 50/50 portfolio against XBI on non-overlapping weekly windows (H₀: alpha ≤ 0).
- **Secondary:** Sharpe of the 50/50 portfolio vs the Top-30 alone (the existing shadow keeps tracking the Top-30), and mean excess over XBI (reported, not gated).
- **Fail / rollback:** keep the §4 structure — fail if alpha < 0 with t ≤ −1.65 at 20, 40 or 60 windows; roll back to Top-30-only if the 50/50 trails the Top-30 on a paired basis with t ≤ −1.65 after 20 windows.
- `CANDIDATE.json` gains the sleeve definition and a candidate hash that covers it (the current hash covers the selector files only; see `governance/HASH_ROTATIONS.md`).

## 7. Clock and change discipline

- The out-of-sample clock resets once, at the first promoted production run that publishes the 50/50 target portfolio; mandate restarts at 0/20.
- **Six-month no-change window** after that start: no selector, roster, sleeve, size or rule changes unless a fail or rollback rule fires, or an operator-declared incident requires a fix. The clock has reset six times in November 2026; this window exists so forward evidence can accumulate.

## 8. Implementation (after approval; separate PRs)

1. Production step that publishes `target_portfolio.csv` (both halves, weights, sleeve risk measures) next to `rankings.csv`, reusing `tools/low_risk_sleeve_shadow.py::build_sleeve` with `measures=["vol_60d", "beta_120d"]` (no new scoring code). Tests first.
2. `production_data/AGENTIC_ACCOUNT_RULES.md`: Rule 2 (rebalance to the target portfolio), Rule 4 (chosen option).
3. Rebalance skill (`biotech-rebalance`) reads `target_portfolio.csv` instead of the Top-30.
4. `FORWARD_VALIDATION_PROTOCOL.md` §2/§4 amendments; capture and evaluator handle the 50/50 basket; `CANDIDATE.json` and hash rotation entry.
5. `docs/MODEL_DOCUMENTATION.md` entry; `.claude/rules/operational-state.md` governance event.

**Scaling note (not part of this proposal).** At today's size (~$1M) the sleeve trades without a rank buffer: a buffer saves only ~0.03 pt/month in costs, its return gain is not significant, and it lets names that have drifted out of the low-risk set stay in the book (drawdown −16% vs −11%). If the account grows past roughly $20–50M, where costs are about three times higher, revisit a rank-40 buffer for both halves (it also cuts Top-30 turnover) as a separate proposal with its own evidence.

Rollback: publish Top-30-only targets again (one flag), revert Rule 4 to the XBI-relative levels.

## 9. Operator decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Adopt the 50/50 Top-30 + vol–beta portfolio | Adopt, as one governed change |
| 2 | Rule 4 option (A / B / C) | A, after re-running its touch frequency |
| 3 | Weekly reweighting of both halves vs monthly-only | Weekly (current practice) |
| 4 | Mandate amendments in §6 | Adopt |
| 5 | Six-month no-change window | Adopt |
| 6 | Optional industry cap of 15 names in the sleeve | Optional — no historical effect; adopt only as a guardrail |

Approval record: _(operator, date, decisions)_
