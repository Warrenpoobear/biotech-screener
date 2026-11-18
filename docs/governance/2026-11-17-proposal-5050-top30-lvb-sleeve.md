# Proposal: 50/50 Top-30 + LVB (low vol-beta) sleeve

**Date:** 2026-11-17
**Terminology:** LVB = low vol-beta sleeve (60-day volatility + 120-day beta to XBI), drawn from the **full eligible universe** (it may hold Top-30 names). The earlier ex-Top-30 version is called "LVB ex-Top-30" where it appears (2026-11-18: operator set LVB = full universe).
**Status:** PROPOSED — not approved. Nothing in this memo takes effect until the operator approves it in writing (merging this memo is not approval of the change; approval is recorded in §9).
**Classification:** Tier 4 governance decision. Portfolio-construction model change. Resets the out-of-sample clock.
**Scope:** what the portfolio holds and how it is judged. The selector, ranker, scoring, eligibility and the production Top-30 itself are **unchanged**.

---

## 1. Decision requested

Replace "equal-weight production Top-30" with:

> **50% production Top-30 + 50% LVB sleeve** (low vol-beta), the sleeve being the 30 lowest-risk names in the full eligible universe, ranked on 60-day volatility and 120-day beta to XBI. A name in both halves carries both weights.

Together with the yardstick changes this requires (§5, §6), adopted as **one** governed change with **one** clock reset, followed by a **six-month no-change window** (§7).

## 2. Why

1. **The model is effectively one factor.** The selector's IC tracks the coinvest IC almost exactly (correlation 0.94 over 84 months). When the factor reverses the whole book goes with it: August–September 2026 cost the Top-30 −21.9% over three months against XBI −0.8%.
2. **Stock-selection add-ons do not fix it.** About 27 add-on and alternative signals were tested on 2026-11-15/17 (13F rebuilds, crowding caps, Top-N, 13D/G, Form 4, financings, FDA events, takeovers, exclusivity, momentum, timing rules, and others). None clears the pre-registered bar (t > 3). See the 2026-11-17 research entry in `docs/MODEL_DOCUMENTATION.md`.
3. **Diversification at the portfolio level does help, in-sample.** An LVB sleeve is nearly uncorrelated with the Top-30's excess return and roughly halves drawdown while keeping the long-run excess over XBI. The case is risk reduction, not added alpha: much of the sleeve's alpha against XBI is exposure to large-cap healthcare and the broad market, and against SPY + XLV the 50/50 portfolio's alpha is not significant (§4c).
4. **The edge of the Top-30 itself is small.** On the v9 history the roster-adjusted range is about +0.3 to +0.7 pt/month against XBI, not significant. Concentrating the whole book on it is the larger risk.

## 3. Specification

| Item | Specification |
|---|---|
| Half A | Production Top-30, exactly as today (`actionable_rank` 1–30, selector-only, ruleset `8887576e`), equal weight within the half |
| Half B | LVB (low vol-beta) sleeve: all eligible names in that day's `rankings.csv`, **including** Top-30 names; risk score = mean of percentile ranks of `vol_60d` (annualised std of daily log returns, last 60 days) and `beta_120d` (beta of daily log returns to XBI, last 120 days); lower risk = higher score; the 30 best-scoring names, equal weight within the half |
| Inputs | Split-adjusted closes dated on or before the snapshot date (`production_data/price_history_split_adj.csv`); no other inputs |
| Exclusions | Names with 20-day annualised volatility < 10% (price pinned near a pending cash-deal price, or stale feed) — as in the existing shadow tool |
| Split | 50% / 50% of account value |
| Reconstitution | Monthly, at the first rebalance after month-end, from that day's promoted snapshot (both halves) |
| Reweighting | Weekly (Monday), back to equal weight within each half and 50/50 between halves — the account's existing weekly practice (Top-30: +0.14 pt/month after costs, t 2.69; sleeve: +0.04, t 1.32 — optional for the sleeve, kept for one simple rule) |
| Industry cap (optional guardrail) | At most 15 of the 30 sleeve names from one industry (point-in-time `industry` field). Binds in about 2% of months historically and changes nothing; it only guards against a future pile-up. Operator choice (§9, decision 6) |
| Overlap | Allowed. A name in both halves carries both weights (1/60 + 1/60 = 3.3%, the largest single-name weight). Historically the halves share 2.5 names on average (maximum 8; some overlap in 84% of months). Against the ex-Top-30 sleeve the 50/50 differs by −0.002 pt/month (t −0.03) |

This is the LVB standalone shadow (variant `lvb_full`, registered as `vol_beta_full`, pre-registered 2026-11-17; ledger `artifacts/low_risk_sleeve_shadow_vol_beta_full/`; first LIVE capture 2026-11-18, the 2026-11-17 capture is REPLAY). Its `combo` reading is this 50/50, with a name allowed in both halves. The ex-Top-30 shadow (variant `lvb`, registered as `vol_beta`, capture-only since 2026-11-16) continues as a comparison. The two measures were chosen **before** the 2026-11-17 measure search. The three-measure base sleeve (adds `max_gain_21d`) is not proposed: `max_gain_21d` was the weakest measure in both halves of the 2026-11-17 search.

## 4. Evidence (in-sample; research only)

`snapshots_pit_v9` (85 month-ends; 13F per date, point-in-time market data and financials, CT.gov date rollback), March 2020 – November 2026 (82 months), monthly, net of 50 bp × one-way turnover (≈ $1M per-trade costs):

| Portfolio | Ann. return | Volatility | Sharpe | Max drawdown | Beta | CAPM alpha/month (t) | Worst stretch vs XBI |
|---|---|---|---|---|---|---|---|
| Top-30 (today) | 18.4% | 31% | 0.70 | −50% | 1.00 | +0.72 (1.45) | −24% |
| LVB sleeve alone | 22.0% | 16% | 1.35 | −12% | 0.35 | +1.40 (3.42) | −30% |
| **50/50 Top-30 + LVB (proposed)** | **21.0%** | **21%** | **1.02** | **−26%** | **0.68** | **+1.06 (2.99)** | **−15%** |
| 50/50 Top-30 + three-measure sleeve | 18.8% | 21% | 0.93 | −26% | 0.69 | +0.89 (2.65) | −17% |
| XBI | 9.8% | 26% | 0.49 | −57% | 1.00 | — | — |

- Mean excess over XBI is about the same as the Top-30 alone (+0.71 vs +0.72 pt/month) at two-thirds of the volatility and half the drawdown.
- By year against XBI: 2020 −26.4 pt, 2021 +20.8, 2022 +16.0, 2023 +12.7, 2024 +11.2, 2025 +23.9, 2026 to date −3.1. The cost is strong rallies (2020).
- 2026: year to date +21.1% (Top-30 alone +10.7%, XBI +19.6%); last three months −4.3% (Top-30 alone −20.4%, XBI −10.2%).
- 2024–2026 only (after the measures were fixed in the 2026-11-17 search's selection period): Sharpe 1.43, alpha +1.14 pt/month (t 2.04).

### 4a. Weight between the halves

Same 82 months, blends of the two halves:

| Sleeve share | Ann. return | Sharpe | Max drawdown | Beta | Alpha/month (t) | Worst stretch vs XBI | Worst 12 months vs XBI | 12-month periods behind XBI |
|---|---|---|---|---|---|---|---|---|
| 0% (Top-30 only) | 18.4% | 0.70 | −50% | 1.00 | +0.72 (1.45) | −24% | −14% | 30% |
| 25% | 19.9% | 0.84 | −39% | 0.84 | +0.89 (2.19) | −16% | −9% | 25% |
| **50% (proposed)** | **21.0%** | **1.02** | **−26%** | **0.68** | **+1.06 (2.99)** | **−15%** | **−11%** | **18%** |
| 75% | 21.7% | 1.22 | −16% | 0.51 | +1.23 (3.46) | −17% | −27% | 24% |
| 100% (sleeve only) | 22.0% | 1.35 | −12% | 0.35 | +1.40 (3.42) | −30% | −48% | 32% |

Mean excess over XBI is the same at every split (+0.71 to +0.72 pt/month). 50/50 has the smallest XBI-relative risk of any split; more sleeve improves absolute and beta-adjusted figures but widens the XBI-relative risk sharply (the 100% sleeve trailed XBI by 48 pt over 12 months in the 2020 rally) and drops the coinvest model entirely. 50/50 is proposed because Rule 4 and the mandate are XBI-anchored; choosing 75% or 100% from this table would be choosing with hindsight.

### 4b. Sleeve design checks (2026-11-17)

These checks were run on LVB ex-Top-30 (before 2026-11-18); the full-universe LVB differs from it by +0.01 pt/month (t 0.09), so they carry over. Each design choice was varied one at a time on the same data and costs, with the selection period (March 2020 – 2023) kept separate from 2024–2026 where a choice was searched. The proposed specification came out best or tied on every dimension:

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
| Stop losses (sell on a daily close, execute next close, cash until the monthly rebalance) | 10–30% below purchase price, 10–25% below peak, whole sleeve to cash if down 5/8/10% within the month (12 rules) | Every rule lowered return and deepened maximum drawdown (−11.5% to −29% vs −10.6%); cost −0.03 to −0.57 pt/month, tighter stops worse (t −2.5 to −3.7), better in at most 2 of 7 years; stopped names gained a further 1–3% before the rebalance; no help in the five worst months. Survivor-only history understates the benefit of exiting names that collapse or delist | No stops |

The edge comes from preferring low-volatility, low-beta names at all, not from fine-tuning; further tuning would be fitting noise. Scripts are in `~/data/lvb_research_2026_11/scripts/` (`lowvol_opt.py`, `volbeta_windows.py`, `volbeta_n.py`, `volbeta_weights.py`, `volbeta_capbuckets.py`, `volbeta_freq.py`, `volbeta_sectorcap.py`, `volbeta_buffer.py`, `volbeta_stops.py`, `sleeve_vs_mix.py`).

### 4c. Alpha against broader benchmarks (2026-11-17)

XBI is equal-weighted small and mid-cap biotech. The sleeve holds mostly large, commercial drug makers, so alpha measured against XBI alone overstates it. Same 82 months, monthly net returns, regressed on ETF total returns (SPY, XLV, IWM, XBI; yfinance adjusted closes, research copy only):

| Portfolio | vs XBI | vs SPY (beta) | vs XLV (beta; R²) | vs SPY + XLV | vs SPY + XLV + IWM + XBI | 2024–26 vs SPY + XLV |
|---|---|---|---|---|---|---|
| Top-30 | +0.72 (t 1.43) | +0.41 (t 0.45; 0.91) | +0.65 (t 0.74; 0.99; 0.24) | +0.25 (t 0.29) | +0.65 (t 1.21) | +0.64 (t 0.45) |
| LVB sleeve | +1.40 (t 3.36) | +0.92 (t 2.06; 0.57) | +0.81 (t 2.58; 0.84; 0.64) | +0.73 (t 2.28) | +0.84 (t 2.81) | +1.33 (t 3.02) |
| 70/30 Top-30 + sleeve | +0.92 (t 2.31) | +0.56 (t 0.79; 0.81) | +0.70 (t 1.06; 0.95; 0.34) | +0.40 (t 0.60) | +0.70 (t 1.76) | +0.85 (t 0.84) |
| **50/50 (proposed)** | **+1.06 (t 2.94)** | **+0.67 (t 1.12; 0.74)** | **+0.73 (t 1.40; 0.92; 0.43)** | **+0.49 (t 0.94)** | **+0.74 (t 2.26)** | **+0.98 (t 1.30)** |

Alpha in pt/month (t).

- The sleeve behaves largely like a healthcare-sector portfolio: XLV explains 64% of its monthly variation, XBI 34%. Its beta to SPY is 0.57.
- After healthcare and the market are accounted for, the sleeve keeps about +0.7 to +0.9 pt/month (t 2.1–2.8): real but smaller and less certain than the XBI figure. Over the period it still beat every ETF on risk-adjusted terms (Sharpe 1.35 vs SPY 1.17, XLV 0.91; max drawdown −12% vs SPY −24%, XLV −16%).
- **The 50/50 portfolio's +1.06 (t 2.94) against XBI falls to +0.49 (t 0.94) against SPY + XLV.** Its benefit over the Top-30 alone is lower risk (drawdown −26% vs −50%, Sharpe 1.02 vs 0.70), not demonstrated stock-selection alpha. This memo should be read as a risk proposal.
- Building the sleeve on beta to SPY instead of XBI is slightly worse on every measure (−0.15 pt/month, t −0.8); beta to XBI stays.

Scripts: `lvb_full_memo.py` (all §4, §4a and §4c figures with full-universe LVB, 2026-11-18), `volbeta_spy.py`, `mix_factors.py` (`~/data/lvb_research_2026_11/scripts/`).

### 4d. Survivorship check on LVB (indirect 2026-11-17; partial direct 2026-11-18)

The v9 history holds only names still listed in 2025–26; no delisted-stock prices are available until the Sharadar account is unlocked (the 2026-11-15 delisted-price pull was not kept). A direct test is not possible yet. Indirect check: LVB rebuilt within a minimum market cap (point-in-time `market_cap_mm`, re-ranked within the pool), where delisting is mostly by takeover rather than failure. Same 82 months and costs:

| LVB universe | Ann. return | Sharpe | Max drawdown | Alpha vs XBI (t) | Alpha vs SPY + XLV (t) | Picks under $2B |
|---|---|---|---|---|---|---|
| All caps (specification) | 22.0% | 1.35 | −12% | +1.40 (3.4) | +0.73 (2.3) | 22% |
| ≥ $1B | 14.7% | 0.95 | −14% | +0.83 (2.2) | +0.18 (0.6) | 11% |
| ≥ $2B | 16.1% | 1.01 | −15% | +0.91 (2.4) | +0.21 (0.7) | 0% |
| ≥ $5B | 12.2% | 0.84 | −22% | +0.61 (1.9) | +0.04 (0.1) | 0% |
| ≥ $10B | 5.5% | 0.44 | −34% | +0.12 (0.4) | −0.36 (−1.1) | 0% |

(Full-universe LVB, as in the `lvb_full` shadow; the ex-Top-30 sleeve behaves the same, §4b.)

- **LVB's alpha comes mostly from its small-cap picks** (about 22% of names). Without them, alpha against SPY + XLV is about +0.2 pt/month and not significant.
- **Those are the names most exposed to survivorship bias:** a small biotech that failed and delisted is missing, so the small low-volatility names in the history are survivors by construction.
- **Mitigating:** failing biotechs usually become volatile before delisting and would tend to drop out of LVB first; for missing names making up 10% of the sleeve to erase the SPY + XLV alpha they would have to trail by about 7 pt/month (−60% a year). This is a plausibility argument, not a measurement.
- **What survives:** the drawdown reduction. Even large-cap-only LVB has far shallower drawdowns than XBI (−15% to −22% for ≥ $2B–5B vs −57%).
- **Consequence for this proposal:** LVB's alpha is treated as unproven until the delisted rebuild (about 400 names, prices, point-in-time financials, 13F CUSIPs, IPO dates; rerun of 85 dates). The case for adoption rests on risk reduction (§2, §4c). If the operator prefers, decision 1 can wait for that rebuild.

Script: `lvb_survivorship.py` (`~/data/lvb_research_2026_11/scripts/`).

**Partial direct check (2026-11-18, last 20 months only).** Delisted-name prices came from Massive (Polygon) grouped daily bars, 2024-11-19 to 2026-11-17; Massive holds only about 2 years on the current plan. Delisted names were found two ways:
- 13 were 13F-held common stocks absent from the price store, kept only when their Massive price matched the 13F implied price within 15%.
- 60 more came from EDGAR Form 25-NSE filings by healthcare companies (SIC 2834, 2835, 2836, 8731 and related). Each was mapped by CIK to the Massive common-stock ticker delisted within 45 days of the filing.

Of the 166 healthcare companies EDGAR shows delisting in the window, 70 mapped this way. The rest filed Form 25 for notes or warrants rather than common stock (63), had no delisted ticker in Massive (14), or are still in the price store (17). Two SPACs were excluded as not operating-company delistings; one of them (PCSC, Freenome) kept trading under a new ticker. Each added series stops at its own delisting date.

The 73 names were added to the eligible universe under the production rules (price ≥ $2, 20-day median dollar volume ≥ $1M), and full-universe LVB was rerun over the 20 v9 month-ends from 2025-05-30 to 2026-10-30:

| Universe | Cumulative | Mean pt/month | Alpha vs XBI | Beta |
|---|---|---|---|---|
| Survivors only | +67.6% | 2.69 | 1.95 | 0.23 |
| With 73 delisted names (exits at 0%, −30% or −100%) | +67.0% | 2.68 | 1.96 | 0.22 |

- **Difference −0.015 pt/month (t −0.4), the same under every exit assumption.** LVB held a delisted name in 4 of 20 months, at most two at a time: DVAX (2025-05), CRGX (2025-06 and 2025-07, a wind-down trading near cash), EXAS (2025-07) and XOMA (2026-05). None of them delisted during a month it was held.
- **The failures never reached the sleeve.** 38 of the 73 exits had a price pinned to a deal before delisting; the other 35 are wind-downs, failures and other exits. LVB held none of them in the month they left; it held XOMA only, two months before its exit.
- **The pinned-price exclusion worked as designed.** It removed takeover targets trading at a fixed deal price, including RGLS, VIGL, SCPH, CDTX, RAPT and DVAX in 2026-01.
- **Limits.** The test does not reach 2020–24, where §4d puts most of the small-cap exposure, and it does not reach the 2022 biotech drawdown. It strengthens the plausibility argument above, but it is not the full rebuild, and the treatment of LVB's alpha as unproven stands.

Script: `delist2/partial_survivorship_v2.py` (data and scripts in `~/data/lvb_research_2026_11/`).

### 4e. Calculation check (2026-11-17)

Run on LVB ex-Top-30 before the 2026-11-18 redefinition; the production code builds both versions with the same `build_sleeve` (`exclude_production=False` for the full universe), so the arithmetic and price checks apply to both.

- **Measures:** `vol_60d` and `beta_120d` from the production code (`tools/low_risk_sleeve_shadow.py::risk_measures`) match an independent recomputation from raw closes for 8 names on 2026-11-17 to the second decimal (e.g. AbbVie 19.59% / 0.230, Privia 28.81% / 0.083, Vericel 49.23% / 0.185).
- **Baskets:** on the same v9 `rankings.csv` files, the research backtest and the production `build_sleeve` pick 29.3 of 30 names on average (minimum 25); the Top-30 definitions agree 30/30. Differences come from simple vs log returns, sample vs population standard deviation, and recent listings without 120 days of history (neutral beta rank in research; volatility alone in production).
- **Results with production baskets:** LVB 22.0%/yr, Sharpe 1.37, max drawdown −11.9%, alpha +1.39 pt/month (t 3.52), vs 21.9%, 1.39, −10.6%, +1.40 (t 3.57) by the research method; 50/50 21.2%, 1.03, −25.0%, +1.06 (t 3.08) vs 21.1%, 1.03, −25.1%, +1.06 (t 3.10). Difference +0.007 pt/month (t 0.10). The §4 figures stand under either method.
- **Look-ahead:** corrupting every close after the as-of date leaves the basket and its measures unchanged.
- **Second price source (Stooq):** the split-adjusted price store was reconciled to Stooq's daily file for all 131 names LVB has held plus XBI (June 2019 – 2026-11-17, about 216,000 ticker-days). Latest closes match for every name; median daily-return correlation 0.995; 411 ticker-days differ by more than 5%, mostly single-day closing-print timing that cancels (27 day-equivalents over 21-day windows); LENZ and GOSS carry a different early split/merger adjustment. LVB rebuilt entirely on Stooq prices (selection and returns, production code): 21.5%/yr, Sharpe 1.39, max drawdown −11.0%, alpha +1.35 pt/month (t 3.62), vs 22.0%, 1.37, −11.9%, +1.39 (t 3.52) on the store; basket overlap 29.2/30; difference −0.04 pt/month (t −0.5). Price inputs are not a source of error.
- **Dividends:** both sources are split-adjusted only, so LVB's returns exclude dividends. Comparisons with XBI (§4, also price-only) are like for like; the §4c regressions on total-return ETFs understate LVB's alpha against SPY/XLV slightly (roughly 0.1 pt/month), which does not change the conclusions.
- These checks cover the arithmetic and the price inputs, not survivorship: Stooq, like the store, holds only names still listed (§4d).

Scripts: `lvb_verify.py`, `lvb_prodbaskets.py`, `stooq_check.py`, `stooq_impact.py` (`~/data/lvb_research_2026_11/scripts/`).

### 4f. LVB against a plain index as the second half (2026-11-18)

Is LVB better than simply holding an ETF in the second half? Same 82 months and costs. ETF halves carry no trading costs, and ETF returns include dividends while LVB's do not, so the comparison slightly favours the ETFs.

| Second half alone | Ann. return | Sharpe | Max drawdown | Beta to XBI |
|---|---|---|---|---|
| LVB | 22.0% | 1.35 | −12% | 0.35 |
| SPY | 18.2% | 1.17 | −24% | 0.31 |
| XLV | 13.4% | 0.91 | −16% | 0.29 |

| 50/50 with the Top-30 | Ann. return | Sharpe | Max drawdown | Alpha vs XBI (t) | Worst stretch vs XBI | 12-month periods behind XBI |
|---|---|---|---|---|---|---|
| **Top-30 + LVB (proposed)** | **21.0%** | **1.02** | **−26%** | **+1.06 (2.99)** | **−15%** | **18%** |
| Top-30 + SPY | 19.3% | 0.98 | −26% | +0.95 (2.89) | −16% | 13% |
| Top-30 + XLV | 16.8% | 0.88 | −23% | +0.78 (2.27) | −19% | 27% |

- **LVB beats XLV:** +0.63 pt/month (t 2.0), ahead in 5 of 7 years. Scaled to LVB's beta to XBI, the gap is +0.38 (t 1.1).
- **LVB does not clearly beat SPY:** +0.27 pt/month (t 0.6); in the 50/50 the gap is +0.13 (t 0.6).
- **The edge over both comes almost entirely from 2024–26.** Against SPY it is −0.18 pt/month in 2020–23 and +0.84 (t 1.1) in 2024–26; against XLV +0.22 (t 0.5) and +1.15 (t 2.8). Healthcare lagged in 2024–25, so XLV is a weak benchmark.
- **Most of the 50/50's improvement over the Top-30 alone comes from pairing it with any low-beta asset.** A plain SPY half halves the drawdown and keeps alpha against XBI at t 2.9.
- **What LVB adds over SPY is about +0.1 to +0.4 pt/month at the portfolio level, counting the dividend effect.** That is not statistically established, and it sits in 2024–26 and in the small-cap names whose survivorship is untested (§4d).
- **SPY has none of LVB's risks:** no model, no survivorship question, no turnover, no dependence on the price store. LVB's advantage is that the book stays in healthcare.

**Consequence for this proposal:** the baseline LVB has to beat is "50/50 Top-30 + SPY", not the Top-30 alone. Two defensible choices, for decision 1 (§9):

1. **Adopt the 50/50 with LVB, and judge the LVB half against SPY going forward.** The `lvb_full` shadow already records the sleeve; SPY needs adding to the daily price refresh (§8 step 5).
2. **Adopt the 50/50 with SPY now, and switch to LVB once its forward record beats SPY.** This is the conservative reading of the evidence. LVB's case beyond SPY rests mainly on the low-volatility effect being well documented outside this data.

Script: `lvb_vs_xlv.py` (`~/data/lvb_research_2026_11/scripts/`).

**Limits.** In-sample; the history keeps only names that survived to 2025 (survivorship rebuild pending; LVB's alpha sits in the small-cap names most exposed to it, §4d); the sleeve definition was informed by data covering the same period; the 13F roster carries hindsight; 82 months is a short sample; no forward evidence yet (the PR #697 and LVB shadows started 2026-11-16/17); alpha against XBI overstates the sleeve's edge (§4c).

## 5. Account Rule 4 must change with it

Rule 4 today: review at −10 pp and liquidate at −15 pp of drawdown **against XBI**. A 0.68-beta book trails XBI in strong rallies by design — the proposed portfolio's worst historical stretch against XBI is −15%, which would have triggered a full liquidation while the portfolio was doing its job.

Options (operator to choose one):

| Option | Rule | Historical behaviour of the proposed portfolio |
|---|---|---|
| **A (recommended)** | Measure Rule 4 against a **beta-adjusted benchmark** (0.68 × XBI + 0.32 × cash), same −10 / −15 pp levels | Judges the portfolio on the risk it takes; needs a re-run of the touch frequency before adoption |
| B | Keep XBI-relative, widen to −20 / −25 pp | Simple; does not distinguish a beta gap from a model failure |
| C | Absolute drawdown: review at −20%, liquidate at −30% | Historical max drawdown −26% |

## 6. The forward-validation mandate must change with it

`docs/FORWARD_VALIDATION_PROTOCOL.md` §2 tests mean weekly excess of the DEM Top-30 over XBI. Proposed amendments (same structure, new primary statistic):

- **Primary:** CAPM alpha of the 50/50 portfolio against XBI on non-overlapping weekly windows (H₀: alpha ≤ 0).
- **Secondary:** Sharpe of the 50/50 portfolio vs the Top-30 alone (the existing shadow keeps tracking the Top-30), mean excess over XBI, and alpha against SPY + XLV (§4c) — all reported, not gated. The SPY + XLV reading is the check that the result is more than healthcare-sector exposure.
- **Fail / rollback:** keep the §4 structure — fail if alpha < 0 with t ≤ −1.65 at 20, 40 or 60 windows; roll back to Top-30-only if the 50/50 trails the Top-30 on a paired basis with t ≤ −1.65 after 20 windows.
- `CANDIDATE.json` gains the sleeve definition and a candidate hash that covers it (the current hash covers the selector files only; see `governance/HASH_ROTATIONS.md`).

## 7. Clock and change discipline

- The out-of-sample clock resets once, at the first promoted production run that publishes the 50/50 target portfolio; mandate restarts at 0/20.
- **Six-month no-change window** after that start: no selector, roster, sleeve, size or rule changes unless a fail or rollback rule fires, or an operator-declared incident requires a fix. The clock has reset six times in November 2026; this window exists so forward evidence can accumulate.

## 8. Implementation (after approval; separate PRs)

1. Production step that publishes `target_portfolio.csv` (both halves, weights, sleeve risk measures) next to `rankings.csv`, reusing `tools/low_risk_sleeve_shadow.py::build_sleeve` with `measures=["vol_60d", "beta_120d"]` and `exclude_production=False` (no new scoring code); a name in both halves gets the sum of its weights. Tests first.
2. `production_data/AGENTIC_ACCOUNT_RULES.md`: Rule 2 (rebalance to the target portfolio), Rule 4 (chosen option).
3. Rebalance skill (`biotech-rebalance`) reads `target_portfolio.csv` instead of the Top-30.
4. `FORWARD_VALIDATION_PROTOCOL.md` §2/§4 amendments; capture and evaluator handle the 50/50 basket; `CANDIDATE.json` and hash rotation entry.
5. Add SPY and XLV to the daily price refresh so the §6 SPY + XLV reading (and the sleeve shadows) can use production prices.
6. `docs/MODEL_DOCUMENTATION.md` entry; `.claude/rules/operational-state.md` governance event.

**Scaling note (not part of this proposal).** At today's size (~$1M) the sleeve trades without a rank buffer: a buffer saves only ~0.03 pt/month in costs, its return gain is not significant, and it lets names that have drifted out of the low-risk set stay in the book (drawdown −16% vs −11%). If the account grows past roughly $20–50M, where costs are about three times higher, revisit a rank-40 buffer for both halves (it also cuts Top-30 turnover) as a separate proposal with its own evidence.

Rollback: publish Top-30-only targets again (one flag), revert Rule 4 to the XBI-relative levels.

## 9. Operator decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Adopt the 50/50, with LVB or with SPY as the second half (§4f) | Adopt the 50/50 as one governed change; LVB vs SPY for the second half is the operator's call (§4f, options 1 and 2) |
| 2 | Rule 4 option (A / B / C) | A, after re-running its touch frequency |
| 3 | Weekly reweighting of both halves vs monthly-only | Weekly (current practice) |
| 4 | Mandate amendments in §6 | Adopt |
| 5 | Six-month no-change window | Adopt |
| 6 | Optional industry cap of 15 names in the sleeve | Optional — no historical effect; adopt only as a guardrail |

Approval record: _(operator, date, decisions)_
