# Low-risk sleeve shadow — forward record (variant `lvb`, LVB)

Sleeve = the 30 lowest-risk names; pool: eligible names in rankings.csv, excluding production Top-30 (actionable_rank 1..30); measures: vol_60d, beta_120d; equal weight; names whose 20-day volatility is below 10% (price pinned near a deal price, or stale) are excluded. Compared with XBI and the production Top-30 on equal-weight forward returns, gross of costs. Non-overlapping windows only.

Pre-registered 2026-11-16, before the first capture. Variant `vol_beta`: the base sleeve without the max_gain_21d measure.
- Why: in the 2026-11-16 hold-out sensitivity test (77 months, net 50 bps, in-sample) max_gain_21d was the weak measure (alone: alpha +0.28 pt/month, t 0.65). vol+beta: alpha +1.16 (t 2.95), beta 0.37, Sharpe 1.27, max drawdown -12.6%; base: alpha +0.81 (t 2.07), beta 0.40, Sharpe 0.99, max drawdown -15.4%. The variant was chosen after seeing that table, so its backtest edge over the base is not evidence.
- Same readings and thresholds as the base sleeve: CAPM alpha on non-overlapping 21-day windows, the 50/50 Top-30 + sleeve Sharpe, and the paired difference against the base sleeve; no reading below 20 non-overlapping LIVE windows.
- Nothing here changes the model or the account.

Pre-registered 2026-11-17. Readings `top20` (production Top-20 = actionable_rank 1..20) and `combo20` (50/50 Top-20 + sleeve), derived from the existing captures. Only captures dated after 2026-11-17 count.
- Why: on the v9 rebuild (82-84 months, in-sample, $1M per-trade costs) Top-20 minus Top-30 was +0.40 pt/month (t 1.53) and the 50/50 Top-20 + sleeve had Sharpe 1.09 vs 0.97 for 50/50 Top-30, CAPM alpha +1.16 pt/month (t 3.33), beta 0.67. Top-20 was chosen after a size sweep (5..100) and its edge came from ranks 11-20 alone (ranks 1-10 were no better than 21-30), so the backtest is not evidence.
- Primary readings: Top-20 minus Top-30 (paired) on non-overlapping 21-trading-day windows; 50/50 Top-20 Sharpe against 50/50 Top-30 and Top-30, and its CAPM alpha against XBI.
- Secondary: the same on non-overlapping 5-day windows; 50/50 Top-20 minus 50/50 Top-30 (paired).
- No reading below 20 non-overlapping LIVE windows at a horizon. Gross of costs, like the other readings here.
- Nothing here changes the model, the portfolio size or the account. Adopting either is a governed model change with an out-of-sample clock reset.

Pre-registered 2026-11-17. Reading `rank_split`: production ranks 1-10 minus ranks 11-30, equal weight, paired, from the existing captures. Only captures dated after 2026-11-17 count.
- Hypothesis: the most crowded top picks (ranks 1-10: median 9 tier-1 managers, highest 12-month run-up) trail ranks 11-30, and the gap grows with holding period.
- Why: on the v9 rebuild (in-sample, gross) ranks 1-10 minus ranks 11-30 was -0.4 / -1.9 / -3.5 / -9.6% at 1 / 3 / 6 / 12 months (12-month Newey-West t -1.3; negative in both halves and 4 of 6 start years, but the 1-6 month sign flips between halves). Crowding variants of the selector did not beat the current model, so this is watched, not acted on.
- Primary reading: the 252- and 126-trading-day differences on non-overlapping windows. Secondary: 63 and 21 days. The 252-day reading needs about 20 years of LIVE captures to reach 20 windows, so read its sign and size at each annual review instead, alongside the shorter horizons, and treat it as descriptive.
- Nothing here changes the model or the account.

Captures: 3 live, 0 replay. Price source: `price_history_split_adj.csv`. **Only LIVE captures are evidence.**

## LIVE

| Horizon | Windows | Sleeve − XBI | t | Sleeve alpha | t | Beta | Combo − XBI | Top-30 − XBI | Sharpe sleeve / combo / Top-30 / XBI | Corr | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |

### LIVE — Top-20 and 50/50 Top-20 + sleeve (captures after 2026-11-17 only)

| Horizon | Windows | Top-20 − Top-30 | t | Top-20 − XBI | 50/50 Top-20 − XBI | 50/50 Top-20 − 50/50 Top-30 | t | 50/50 Top-20 alpha | t | Beta | Sharpe Top-20 / Top-30 / 50/50 Top-20 / 50/50 Top-30 / XBI | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a / n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a / n/a | insufficient_windows |

## REPLAY

| Horizon | Windows | Sleeve − XBI | t | Sleeve alpha | t | Beta | Combo − XBI | Top-30 − XBI | Sharpe sleeve / combo / Top-30 / XBI | Corr | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |

### REPLAY — Top-20 and 50/50 Top-20 + sleeve (captures after 2026-11-17 only)

| Horizon | Windows | Top-20 − Top-30 | t | Top-20 − XBI | 50/50 Top-20 − XBI | 50/50 Top-20 − 50/50 Top-30 | t | 50/50 Top-20 alpha | t | Beta | Sharpe Top-20 / Top-30 / 50/50 Top-20 / 50/50 Top-30 / XBI | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a / n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a / n/a | insufficient_windows |

## LIVE — Ranks 1-10 vs 11-30 (captures after 2026-11-17 only)

| Horizon | Windows | Ranks 1-10 − 11-30 | t | Ranks 1-10 − XBI | Ranks 11-30 − XBI | Reading |
|---|---|---|---|---|---|---|
| 21d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 63d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 126d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 252d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |

## REPLAY — Ranks 1-10 vs 11-30 (captures after 2026-11-17 only)

| Horizon | Windows | Ranks 1-10 − 11-30 | t | Ranks 1-10 − XBI | Ranks 11-30 − XBI | Reading |
|---|---|---|---|---|---|---|
| 21d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 63d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 126d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |
| 252d | 0 | n/a | n/a | n/a | n/a | insufficient_windows |

