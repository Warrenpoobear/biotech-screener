# dd_rel_margin Rescue Flag — Walkforward Validation

**Date**: 2026-02-12
**Ruleset under test**: be13afeb (= b92f9338 + `enable_dd_rel_margin_rescue=True`)
**Baseline**: b92f9338 (active, rescue OFF)
**Threshold**: `dd_rel_margin_rescue_threshold = -0.05`

## Summary

**VERDICT: Leave rescue OFF in production.** The flag improves full-panel AB-CD separation
(+1.24pp) but **worsens 2025-only separation by -2.05pp** — the data regime that matters.
Turnover increases by +8.8pp (full) / +14.8pp (2025). The rescue is too aggressive at the
current -0.05 threshold: it admits 10-19 extra tickers per snapshot in 2025, diluting A-tier
quality without sufficient compensating signal.

---

## Rescued Counts Per Snapshot

| Date       | Rescued | Elig (base) | Elig (rescue) | Delta |
|------------|---------|-------------|---------------|-------|
| 2024-01-31 | 0       | 56          | 56            | +0    |
| 2024-02-29 | 0       | 58          | 58            | +0    |
| 2024-03-29 | 0       | 61          | 61            | +0    |
| 2024-04-30 | 13      | 55          | 76            | +21   |
| 2024-05-31 | 10      | 53          | 63            | +10   |
| 2024-06-28 | 0       | 52          | 52            | +0    |
| 2024-07-31 | 0       | 57          | 57            | +0    |
| 2024-08-30 | 0       | 60          | 60            | +0    |
| 2024-09-30 | 0       | 65          | 65            | +0    |
| 2024-10-31 | 0       | 68          | 68            | +0    |
| 2024-11-29 | 0       | 65          | 65            | +0    |
| 2024-12-31 | 7       | 59          | 66            | +7    |
| 2025-01-31 | 5       | 43          | 48            | +5    |
| 2025-02-28 | 12      | 40          | 52            | +12   |
| 2025-03-31 | 18      | 38          | 65            | +27   |
| 2025-04-30 | 11      | 23          | 44            | +21   |
| 2025-05-30 | 19      | 28          | 55            | +27   |
| 2025-06-30 | 11      | 25          | 44            | +19   |
| 2025-07-31 | 9       | 28          | 40            | +12   |
| 2025-08-29 | 6       | 42          | 48            | +6    |
| 2025-09-30 | 0       | 40          | 40            | +0    |
| 2025-10-31 | 0       | 42          | 42            | +0    |

Mean rescued (2025): 9.1 per snapshot. Peak: 19 (2025-05-30).
Rescue fires in 14 of 22 snapshots. Zero rescue in late 2025 when XBI drawdowns moderate.

---

## Tier Distribution (Eligible Only)

|            | A (base) | A (resc) | B (base) | B (resc) | C (base) | C (resc) |
|------------|----------|----------|----------|----------|----------|----------|
| Full       | 91 (8.6%)| 119 (9.7%)| 376 (35.5%)| 452 (36.9%)| 591 (55.9%)| 654 (53.4%)|
| 2025-only  | 39 (11.2%)| 65 (13.6%)| 124 (35.5%)| 178 (37.2%)| 186 (53.3%)| 235 (49.2%)|

Rescued tier breakdown (full): A=23, B=52, C=46 (of 121 rescued).
Rescued tier breakdown (2025): A=22, B=35, C=34 (of 91 rescued).

---

## AB-CD Separation (60d, Eligible Only)

|           | Base    | Rescue  | Delta    |
|-----------|---------|---------|----------|
| Full      | +8.51pp | +9.76pp | **+1.24pp** |
| 2025-only | +18.66pp| +16.61pp| **-2.05pp** |

The full-panel improvement (+1.24pp) is driven entirely by 2024, where rescue admits
names into a hostile catalyst regime. In 2025 — the well-formed regime and the one
that matters for forward-looking decisions — rescue dilutes A-tier from +18.17% to
+16.55% and worsens overall separation.

---

## Tier Performance (Mean 60d Return)

### Full (2024+2025)

| Tier | Base    | Rescue  | Delta    |
|------|---------|---------|----------|
| A    | +11.26% | +12.02% | +0.76pp  |
| B    | +9.03%  | +11.93% | +2.90pp  |
| C    | +0.96%  | +2.19%  | +1.23pp  |
| D    | +17.59% | +17.53% | -0.06pp  |

### 2025-Only

| Tier | Base    | Rescue  | Delta    |
|------|---------|---------|----------|
| A    | +18.17% | +16.55% | **-1.62pp** |
| B    | +36.28% | +36.21% | -0.07pp  |
| C    | +13.32% | +14.36% | +1.04pp  |
| D    | +29.16% | +29.58% | +0.42pp  |

A-tier dilution in 2025 (-1.62pp) is the primary concern: rescued names with shallow
relative drawdowns are not as strong as the incumbent A-tier population.

---

## Stability & Turnover

|           | Overlap (base) | Overlap (resc) | Turnover (base) | Turnover (resc) |
|-----------|----------------|----------------|-----------------|-----------------|
| Full      | 69.9%          | 61.1%          | 30.1%           | 38.9%           |
| 2025-only | 72.2%          | 57.4%          | 27.8%           | 42.6%           |

Turnover increases by +8.8pp (full) and +14.8pp (2025). Rescued names create
portfolio instability: they enter when XBI draws down (rescue fires) and exit when
it recovers (rescue no longer needed). This whipsaw pattern drives turnover.

---

## Rescued Cohort Performance

| Metric          | Full (n=121) | 2025 (n=91) |
|-----------------|--------------|-------------|
| Mean 60d return | +15.26%      | +21.21%     |
| Median 60d      | +8.18%       | +15.10%     |
| Hit rate        | 63.6%        | 69.2%       |
| P25             | -7.65%       | -2.67%      |
| P75             | +34.07%      | +37.31%     |

The rescued cohort is decent in isolation (69.2% hit rate in 2025). However, it
underperforms the D-tier baseline (+29.16% mean in 2025), suggesting these are NOT
the best D-tier names — they're the ones closest to the drawdown boundary, not
necessarily the ones with the strongest forward returns.

---

## Conclusion

1. **Rescue OFF in production** — the 2025 separation degradation (-2.05pp) and
   turnover spike (+14.8pp) outweigh the full-panel improvement.

2. **Infrastructure value preserved** — the flag, telemetry (`dd_rel_margin_rescued`),
   and drift monitoring (`dd_rel_margin_rescue_share_pct`) are in place. If a
   tighter threshold (e.g., -0.02) or a catalyst-conditioned rescue variant is
   developed later, the plumbing is ready.

3. **The -0.05 threshold is too generous** — it rescues 10-19 names per snapshot
   in 2025 (up to 71% increase in eligible count), flooding B/C tiers with
   marginal names.

4. **Potential future direction**: a tighter threshold (-0.02) or a composite
   condition (rel_margin > -0.02 AND catalyst_mode != "missing") would be more
   selective. This would require a separate calibration sweep.
