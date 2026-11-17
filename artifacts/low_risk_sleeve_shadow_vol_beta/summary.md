# Low-risk sleeve shadow — forward record (variant `vol_beta`)

Sleeve = the 30 lowest-risk eligible names outside the production Top-30 (60-day volatility, largest one-day gain over 21 days, 120-day beta to XBI), equal weight; names whose 20-day volatility is below 10% (price pinned near a deal price, or stale) are excluded. Compared with XBI and the production Top-30 on equal-weight forward returns, gross of costs. Non-overlapping windows only.

Pre-registered 2026-11-16, before the first capture. Variant `vol_beta`: the base sleeve without the max_gain_21d measure.
- Why: in the 2026-11-16 hold-out sensitivity test (77 months, net 50 bps, in-sample) max_gain_21d was the weak measure (alone: alpha +0.28 pt/month, t 0.65). vol+beta: alpha +1.16 (t 2.95), beta 0.37, Sharpe 1.27, max drawdown -12.6%; base: alpha +0.81 (t 2.07), beta 0.40, Sharpe 0.99, max drawdown -15.4%. The variant was chosen after seeing that table, so its backtest edge over the base is not evidence.
- Same readings and thresholds as the base sleeve: CAPM alpha on non-overlapping 21-day windows, the 50/50 Top-30 + sleeve Sharpe, and the paired difference against the base sleeve; no reading below 20 non-overlapping LIVE windows.
- Nothing here changes the model or the account.

Captures: 1 live, 0 replay. Price source: `price_history_split_adj.csv`. **Only LIVE captures are evidence.**

## LIVE

| Horizon | Windows | Sleeve − XBI | t | Sleeve alpha | t | Beta | Combo − XBI | Top-30 − XBI | Sharpe sleeve / combo / Top-30 / XBI | Corr | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |

## REPLAY

| Horizon | Windows | Sleeve − XBI | t | Sleeve alpha | t | Beta | Combo − XBI | Top-30 − XBI | Sharpe sleeve / combo / Top-30 / XBI | Corr | Reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |
| 21d | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a / n/a / n/a / n/a | n/a | insufficient_windows |

