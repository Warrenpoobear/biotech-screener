# Low-risk sleeve shadow — forward record

Sleeve = the 30 lowest-risk eligible names outside the production Top-30 (60-day volatility, largest one-day gain over 21 days, 120-day beta to XBI), equal weight; names whose 20-day volatility is below 10% (price pinned near a deal price, or stale) are excluded. Compared with XBI and the production Top-30 on equal-weight forward returns, gross of costs. Non-overlapping windows only.

Pre-registered 2026-11-15, before the first capture.
- The sleeve is judged on risk-adjusted terms, not on excess return over XBI: by design it has a beta near 0.4 and will trail XBI in strong rallies.
- Primary reading: CAPM alpha of the sleeve against XBI on non-overlapping 21-trading-day windows, and whether the 50/50 Top-30 + sleeve mix has a higher Sharpe ratio than the Top-30 alone.
- Secondary: the same on non-overlapping 5-day windows; correlation of sleeve and Top-30 excess returns.
- No reading below 20 non-overlapping LIVE windows at a horizon (about 5 months at 5 days, about 20 months at 21 days).
- Backtest expectation for comparison (in-sample, 84 months): alpha +0.69 pt/month, beta 0.43, correlation with Top-30 excess 0.14.
- Nothing here changes the model or the account. Using the sleeve with real money is a separate governed decision.

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

