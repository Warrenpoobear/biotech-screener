# LVB sleeve backtest analysis (2026-11-18)

**Classification:** `RESEARCH_RECORD / NO_MODEL_CHANGE` (Tier 2). In-sample, pseudo-PIT history. Not evidence for a production change; the 50/50 Top-30 + LVB proposal (`docs/governance/2026-11-17-proposal-5050-top30-lvb-sleeve.md`, PROPOSED) remains a Tier 4 decision.
**Code:** `scripts/research/lvb_universe_backtest.py`, `lvb_factor_alpha.py`, `lvb_sharadar_prices.py`, `lvb_survivorship_check.py`, `lvb_5050_mix.py` (PR #714).

## Answer

After correcting the price source and the survivorship gap, LVB on the full eligible universe returned **17.6% a year** over 82 months, with Sharpe **1.12**, maximum drawdown **−17.8%** and CAPM alpha against XBI of **+1.08 pt/month (t 2.66)**. XBI returned 9.8% a year, with Sharpe 0.49 and maximum drawdown −57%.

The proposal memo's +1.40 pt/month (§4, t 3.42) overstates the edge by about a quarter. Against large-cap healthcare alone (XLV) the alpha is **+0.66 pt/month (t 2.09)** even before the survivorship correction. Most of LVB's advantage over XBI is owning large, steady healthcare names, not stock selection.

## Definition

LVB holds the 30 eligible names with the lowest risk score, equal weight. The score is the mean percentile rank of `vol_60d` (60-day annualised volatility) and `beta_120d` (120-day beta to XBI). Names with 20-day volatility below 10% are excluded (price pinned near a deal price, or a stale feed).

The sleeve is built with the shadow tool's own `tools/low_risk_sleeve_shadow.build_sleeve`, so backtest and forward shadows share one definition:
- `vol_beta`: pool excludes the production Top-30;
- `vol_beta_full`: pool is the full eligible universe.

**Data:** `data/snapshots_pit_v9`, holdings from 2020-03-31 to 2026-11-16 (82 monthly periods). Each holding runs from one snapshot close to the next. Results are net of 50 bp × one-way turnover; XBI is gross.

## From the memo figure to the corrected figure

| Step | Full-universe LVB: annual return | Sharpe | Max drawdown | Alpha vs XBI, pt/month (t) |
|---|---|---|---|---|
| Proposal memo §4 (full universe, production prices) | 22.0% | 1.35 | −12% | +1.40 (3.42) |
| Rerun, full universe, production prices | 21.8% | 1.35 | −12.3% | +1.38 (3.36) |
| Sharadar total-return prices | 19.4% | 1.23 | −12.6% | +1.22 (3.00) |
| + names delisted since, added back | 17.6% | 1.12 | −17.8% | +1.08 (2.66) |

### 1. Pool: full universe vs ex-Top-30

**Same within noise.** The paired difference, full minus ex-Top-30, is −0.02 pt/month (t −0.16). The full-universe sleeve holds 2.5 Top-30 names a month on average, and none in 16 of 82 months. The full-universe rerun on production prices reproduces the memo's §4 figures within small price differences; the ex-Top-30 rerun (22.0%, Sharpe 1.37, alpha +1.39) matches the memo's earlier research-method figures (21.9%, 1.39, +1.40).

### 2. Price source

Sharadar `closeadj` includes dividends, yet LVB earns less on it: 19.4% against 21.8%. Three corporate events explain most of the gap.

| Event | Production split-adjusted file | Sharadar `closeadj` | What actually happened |
|---|---|---|---|
| AZN, Jan 2026 | +96% one day | ≈ +1% | 2:1 ADR ratio change. **Production file wrong**; fix in PR #712 |
| JBIO, Apr 2025 | ≈ 0% | −87% | $2.40 special dividend + 1:35 reverse split; true return ≈ 0. **Sharadar wrong**; the registry label is wrong too (see memo) |
| LENZ, Mar 2024 | +98% for the month | +36% | 1:7 reverse split at the Graphite merger. Not resolved |

- **Net effect:** the AZN error inflates the production-price runs by 3.4 points in January 2026. The JBIO error deflates the Sharadar runs by about 2.9 points in April 2025, about 0.04 pt/month over the period.
- **Read the Sharadar row as slightly conservative.**
- **Coverage gap:** Sharadar lacks SUPN for 2021–2022 (the older history sits under SUPN1), so those months hold 29 names.

Records: `docs/governance/2026-11-18-azn-adr-ratio-split-fix.md`.

### 3. Survivorship

`data/snapshots_pit_v9` holds only names that survived into the current universe. Sharadar lists 1,076 SIC 2834–2836 common stocks that traded at a quarter-end since March 2020; v9 holds 290, and 402 of the missing names are delisted.

Each month, names missing from that day's rankings are added to the pool if they pass production-like eligibility:
- Sharadar point-in-time market cap at least the smallest eligible name's that day;
- price at least $5 and 20-day median dollar volume at least $500k (`common/data_quality.py`);
- not past the deep-drawdown gate: 252-day drawdown below −40% and more than 25 points worse than XBI's (ruleset `drawdown_gate`, `drawdown_rel_xbi_gate`);
- not the same company as a universe name under an older ticker (SUPN1, BHVN1 and INBX1 are matched to SUPN, BHVN and INBX).

The screen's fundamental red-flag and missing-financials gates are not replicated, so the drag below is probably a little overstated.

| Pool (full universe, Sharadar prices) | Annual return | Sharpe | Max drawdown | Alpha vs XBI (t) | Mean excess vs XBI, pt/month |
|---|---|---|---|---|---|
| Survivors only | 19.9% | 1.25 | −12.2% | +1.26 (3.05) | +0.55 |
| **+ names delisted since** | **17.6%** | **1.12** | **−17.8%** | **+1.08 (2.66)** | **+0.39** |
| + every missing drug/biotech name | 12.2% | 0.93 | −15.8% | +0.79 (2.03) | −0.02 |

- **Paired drag of the delisted add-back:** −0.16 pt/month (t −1.54). About 58 delisted names pass each month and LVB holds 3.5 of them. The most held are Taro, Seagen, Theravance, Abcam, Eagle, Horizon, Dynavax, Catalent, Intra-Cellular and Alexion: large, steady names later acquired.
- **Ex-Top-30 moves the same way:** alpha +1.29 → +1.12 (t 2.70), drawdown −12% → −20%.
- **The last row is a scope question, not survivorship.** It adds still-listed names outside the v9 universe, mostly big pharma (GSK, J&J, Novartis, Takeda, Pfizer, Merck). Drawn from that pool, LVB becomes a big-pharma basket that only matches XBI. LVB's edge depends on a biotech-centred universe.
- **Base differs slightly:** "Survivors only" is 19.9% rather than 19.4% because this run keeps Sharadar's dividend-adjusted prices for names such as ABBV, where the plain Sharadar run had fallen back to the split-adjusted file.

## Benchmarks

Full universe, monthly. The Sharadar column uses Sharadar on both sides of each regression and is before the survivorship correction.

| Benchmark | Production prices + yfinance ETFs | Sharadar stocks and ETFs | Beta (Sharadar) |
|---|---|---|---|
| XBI | +1.38 (3.36) | +1.22 (3.00) | 0.34 |
| SPY | +0.90 (2.06) | +0.75 (1.73) | 0.55 |
| XLV | +0.81 (2.56) | +0.66 (2.09) | 0.81 |

Alpha in pt/month, t in brackets. XLV explains about 62% of LVB's monthly variation; XBI explains about a third.

## Rebalance frequency

Full universe, Sharadar prices. Between rebalances the book drifts and pays no cost.

| Rebalance | Annual return | Sharpe | Max drawdown | Alpha vs XBI (t) | One-way turnover per month |
|---|---|---|---|---|---|
| Monthly | 19.4% | 1.23 | −12.6% | +1.22 (3.00) | 16.5% |
| Quarterly, end of Mar/Jun/Sep/Dec | 19.5% | 1.20 | −14.3% | +1.20 (2.92) | 10.0% |
| Quarterly, end of Jan/Apr/Jul/Oct | 18.5% | 1.20 | −11.9% | +1.13 (2.94) | 10.3% |
| Quarterly, end of Feb/May/Aug/Nov | 19.2% | 1.22 | −13.3% | +1.19 (2.98) | 9.5% |

- **Quarterly costs little:** 0.02–0.09 pt/month of alpha, for about 40% less turnover.
- **The ex-Top-30 version is more sensitive to timing** (17.4% to 22.8% across phases), because its pool shifts whenever the Top-30 changes.

## 50/50 Top-30 + LVB, recomputed

On corrected data the 50/50 mix still halves the Top-30's drawdown, but it now gives up some return: alpha against XBI is **+0.91 pt/month (t 2.58)**, down from the memo's +1.06 (t 2.99). Over this period it also trails a plain S&P 500 position on risk-adjusted terms.

| Portfolio | Annual return | Volatility | Sharpe | Max drawdown | Beta to XBI | Alpha vs XBI (t) | Mean excess vs XBI (t) | Worst 12 months vs XBI | 12-month periods behind XBI |
|---|---|---|---|---|---|---|---|---|---|
| Top-30 alone | 18.7% | 30.7% | 0.71 | −49.7% | 1.01 | +0.74 (1.46) | +0.75 (1.50) | −12% | 28% |
| LVB alone (corrected) | 17.6% | 15.6% | 1.12 | −17.8% | 0.35 | +1.08 (2.66) | +0.39 (0.59) | −31% | 38% |
| 50/50, memo inputs (production prices) | 21.1% | 20.9% | 1.02 | −26.1% | 0.68 | +1.06 (2.97) | +0.71 (1.61) | −9% | 18% |
| 50/50, Sharadar prices | 19.9% | 20.8% | 0.98 | −27.0% | 0.68 | +0.98 (2.76) | +0.63 (1.43) | −9% | 18% |
| **50/50, corrected** | **19.0%** | **20.9%** | **0.94** | **−27.9%** | **0.68** | **+0.91 (2.58)** | **+0.57 (1.30)** | **−10%** | **18%** |
| XBI | 9.8% | 26.2% | 0.49 | −56.7% | 1.00 | — | — | — | — |
| S&P 500 (SPY) | 18.2% | 15.4% | 1.17 | −23.9% | 0.31 | +1.17 (2.76) | +0.44 (0.61) | −34% | 31% |

Alpha and excess in pt/month. Same 82 months, net of 50 bp × one-way turnover, rebalanced to 50/50 monthly. "Corrected" = Sharadar total-return prices for both halves, with LVB drawn from the survivorship-corrected pool. The "memo inputs" row reproduces the proposal memo's 50/50 (21.0%, Sharpe 1.02, −26%, +1.06). Script: `scripts/research/lvb_5050_mix.py`.

- **Return:** the corrected mix trails the Top-30 alone by 0.18 pt/month (t −0.44). The memo's statement that the mix keeps the Top-30's full excess over XBI does not hold on corrected data; the shortfall is small and not significant.
- **Risk:** it halves volatility and maximum drawdown, and cuts the share of 12-month periods spent behind XBI from 28% to 18%. The halves' excess returns are nearly uncorrelated (0.10).
- **Against the S&P 500:** 19.0% a year against 18.2%, with more volatility (20.9% vs 15.4%) and a deeper drawdown (−28% vs −24%), so a lower Sharpe (0.94 vs 1.17). Alpha against SPY alone is +0.52 pt/month (t 0.87), against XLV alone +0.60 (t 1.14).
- **Limit:** only the LVB half is survivorship-corrected. The Top-30 half still comes from the survivor-only v9 universe; coinvest selection cannot be recomputed for names missing from the snapshots.

| Year | Top-30 | LVB (corrected) | 50/50 (corrected) | XBI |
|---|---|---|---|---|
| 2026 (to Nov 16) | +10.8% | +23.3% | +17.8% | +19.8% |
| 2025 | +76.9% | +23.4% | +49.4% | +35.9% |
| 2024 | +13.6% | +19.7% | +17.7% | +1.0% |
| 2023 | +15.3% | +5.4% | +10.7% | +7.6% |
| 2022 | −14.4% | −4.9% | −8.8% | −25.9% |
| 2021 | −23.2% | +14.9% | −5.7% | −20.5% |
| 2020 (from Mar 31) | +91.3% | +44.7% | +66.9% | +82.2% |

## Limits

- **In-sample.** The measures were chosen on the same period.
- **Approximate survivorship.** The add-back uses rules modelled on eligibility, not the screen itself. The proposal memo's partial check (§4d) calls price ≥ $2 and dollar volume ≥ $1M the production rules; `common/data_quality.py:81-82` says $5 and $500k, which this analysis uses. Which applies to the universe build is open.
- **Short sample.** 82 months, with the 2020 rally driving the worst relative stretch (−30% against XBI).
- **Price data.** The three events above; Sharadar `closeadj` misses at least one special dividend.
- **Not reproducible from git alone.** Sharadar data is licensed and kept local; the scripts re-fetch it with `SHARADAR_API_KEY`.

## Forward evidence

Unchanged by this analysis. `vol_beta` has captured since 2026-11-16 and `vol_beta_full` since 2026-11-17. Neither has a reading before 20 non-overlapping live windows.
