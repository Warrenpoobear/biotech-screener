# Catalyst Tilt Validation — 2026-02-12

## Summary

**Decision: PARK** — keep `enable_catalyst_tilt=False` in production.

The tilt is directionally correct (+0.18 to +0.37pp weighted 60d return) but
the effect is too small relative to noise to justify enabling. The underlying
signal (NEAR/MID outperform FAR/MISSING at the ticker level) is real but gets
diluted at the portfolio level because (a) the tiering system already
over-selects NEAR/MID tickers via A-tier gating, and (b) MISSING-catalyst
portfolio members perform well in 2025 (+34.8% mean), limiting the benefit of
down-weighting them. Infrastructure is validated and ready to revisit when
catalyst coverage improves.

## Bug Fix: Tilt Was Not Flowing Through

The initial runs (all 4 configs) produced **identical** strategy residuals.
Root cause: `build_strategy_portfolio()` and `build_all_dev_decisions()` in
`run_decision_strategy_backtest.py` extracted specific fields from
`compute_decision_fields()` output but omitted `catalyst_tilt_mult` and
`catalyst_tilt_applied`. As a result, `compute_target_weights()` defaulted
to `tilt_mult=1.0` for all positions.

**Fix**: Added `catalyst_tilt_mult` and `catalyst_tilt_applied` to both
functions' output dicts and to `PANEL_COLUMNS`. After the fix, weights
correctly diverge across configs (verified: 20/20 positions show weight
changes in aggressive vs baseline).

## Sweep Configuration

| Config       | Ruleset ID | enable_catalyst_tilt | NEAR | MID  | FAR  | MISSING |
|-------------|------------|---------------------|------|------|------|---------|
| Baseline    | 68b2c45e   | False               | 1.00 | 1.00 | 1.00 | 1.00    |
| Conservative| 7bbeeb29   | True                | 1.05 | 1.02 | 0.98 | 0.95    |
| Default     | c0b3ffcc   | True                | 1.10 | 1.05 | 0.95 | 0.90    |
| Aggressive  | b1bb6426   | True                | 1.20 | 1.10 | 0.90 | 0.80    |

**Window**: 2025-01-31 to 2025-10-31, 10 monthly snapshots, A+B filter, K=20.

## Portfolio-Weighted Metrics

| Config       | Wt Ret 60d | Wt Ret 20d | EqWt Ret 60d | Wt MaxDD 60d | Worst Snap |
|-------------|-----------|-----------|-------------|-------------|-----------|
| Baseline    | +35.81%   | +11.38%   | +34.27%     | -22.64%     | -6.99%    |
| Conservative| +35.89%   | +11.37%   | +34.27%     | -22.61%     | -7.00%    |
| Default     | +36.00%   | +11.34%   | +34.27%     | -22.58%     | -7.05%    |
| Aggressive  | +36.18%   | +11.31%   | +34.27%     | -22.51%     | -7.10%    |

### Deltas vs Baseline

| Config       | D WtRet60d | D WtRet20d | D MaxDD60d | D WorstSnap |
|-------------|-----------|-----------|-----------|------------|
| Conservative| +0.08pp   | -0.01pp   | +0.03pp   | -0.01pp    |
| Default     | +0.18pp   | -0.04pp   | +0.06pp   | -0.05pp    |
| Aggressive  | +0.37pp   | -0.07pp   | +0.13pp   | -0.11pp    |

**Key observations**:
- 60d weighted return monotonically improves (+0.37pp at aggressive)
- Max-DD improves (less drawdown) by +0.13pp at aggressive
- 20d return slightly deteriorates (-0.07pp) — negligible
- Worst snapshot slightly worsens (-0.11pp) — negligible
- Equal-weight return is **identical** — confirms tilt only affects weights, not membership

## Turnover & Stability

| Metric              | Baseline | Conservative | Default | Aggressive |
|--------------------|----------|-------------|---------|-----------|
| Jaccard Overlap    | 65.4%    | 65.4%       | 65.4%   | 65.4%     |
| Turnover           | 34.6%    | 34.6%       | 34.6%   | 34.6%     |

**Identical** — tilt changes weights, not membership.

## Catalyst Strength Weight Distribution

| Strength | Baseline wt% | Aggressive wt% | Delta  | Bucket Mean 60d |
|----------|-------------|----------------|--------|----------------|
| NEAR     | 27.4%       | 32.8%          | +5.4pp | +32.0%         |
| MID      | 19.2%       | 20.9%          | +1.7pp | +42.7%         |
| FAR      | 28.9%       | 26.2%          | -2.7pp | +31.6%         |
| MISSING  | 24.6%       | 20.1%          | -4.5pp | +34.8%         |

The tilt correctly shifts weight toward NEAR (+5.4pp) and MID (+1.7pp) and away
from FAR (-2.7pp) and MISSING (-4.5pp).

**Why the effect is small**: MISSING-catalyst portfolio members averaged +34.8%
mean 60d return — actually higher than NEAR (+32.0%) and FAR (+31.6%) in this
sample. The ticker-level catalyst signal (NEAR: +5.56pp median over MISSING)
doesn't translate cleanly to portfolio-level impact because the tiering system
already selects for NEAR/MID via A-tier gating. The remaining MISSING tickers in
the B-tier portfolio performed well in 2025.

## Weighted Cost Summary

| Config       | Median Cost | Mean Cost | P95 Cost  | P95 Participation |
|-------------|------------|----------|----------|------------------|
| Baseline    | 566.9 bps  | 733.9    | 1719.6   | 70.86%           |
| Conservative| 564.3 bps  | 731.6    | 1673.5   | 67.03%           |
| Default     | 562.6 bps  | 729.1    | 1626.0   | 64.06%           |
| Aggressive  | 581.5 bps  | 723.3    | 1562.5   | 59.37%           |

Tilt slightly reduces P95 cost and participation because NEAR/MID tickers tend
to be more liquid than MISSING tickers.

## Per-Snapshot Weighted 60d Returns

| Date       | Baseline | Conservative | Default  | Aggressive |
|-----------|---------|-------------|---------|-----------|
| 2025-01-31| -6.99%  | -7.00%      | -7.05%  | -7.10%    |
| 2025-02-28| +0.24%  | +0.10%      | -0.07%  | -0.39%    |
| 2025-03-31| +15.79% | +15.90%     | +16.03% | +16.27%   |
| 2025-04-30| +41.84% | +41.81%     | +41.89% | +41.94%   |
| 2025-05-30| +59.51% | +59.11%     | +58.79% | +58.08%   |
| 2025-06-30| +71.81% | +71.76%     | +71.68% | +71.59%   |
| 2025-07-31| +40.29% | +40.62%     | +41.12% | +41.95%   |
| 2025-08-29| +57.16% | +58.38%     | +59.67% | +62.25%   |
| 2025-09-30| +52.55% | +52.35%     | +52.13% | +51.69%   |
| 2025-10-31| +25.95% | +25.90%     | +25.76% | +25.56%   |

No consistent directionality per-snapshot — tilt helps in some months (Mar, Jul,
Aug) and hurts in others (Jan, Feb, May, Sep, Oct). Largest positive impact:
Aug-2025 (+5.09pp aggressive vs baseline).

## Decision Rationale

1. **Effect too small**: +0.37pp (aggressive) is within noise for 10 snapshots.
   Cannot distinguish from random weight reshuffling.

2. **No consistency**: Tilt helps in 4/10 months, hurts in 6/10. Not a stable
   signal at the portfolio level.

3. **Tail risk neutral**: Worst snapshot delta is -0.11pp (negligible).

4. **Turnover unchanged**: Good — no unintended churn from weight rebalancing.

5. **Cost structure slightly better**: P95 costs drop because NEAR/MID tickers
   tend to be more liquid. Modest benefit.

**Conclusion**: The infrastructure works correctly. The tilt is directionally
sensible but the portfolio-level alpha extraction is insufficient with current
data. Revisit when:
- Catalyst coverage improves (2024 archives are fixed, more NEAR/MID tickers)
- Larger sample window available (20+ snapshots)
- Ticker-level tilt signal strengthens (NEAR-MISSING spread > 10pp in portfolio)

## Artifacts

- `walkforward_panel_tilt_baseline.csv` (1808 rows)
- `walkforward_panel_tilt_conservative.csv` (1808 rows)
- `walkforward_panel_tilt_default.csv` (1808 rows)
- `walkforward_panel_tilt_aggressive.csv` (1808 rows)
- `walkforward_report_tilt_*.json` (4 JSON reports)
- `walkforward_report_tilt_*.md` (4 markdown reports)

## Provenance

- Code version: post-664240f (includes tilt passthrough fix)
- Price history SHA: d5ea922a48fbde4c0e829eae874ae2937b9947a558e1da29ea112633893e575d
- Base ruleset: 68b2c45e (v1.2.1_candidate.json, enable_catalyst_tilt=False)
- Panel baseline SHA: 57902c0b53d277f94f27f1887351dde90ba087a72916850cf9dee943597dda2e
