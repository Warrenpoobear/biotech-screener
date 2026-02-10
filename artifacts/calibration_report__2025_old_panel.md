# 2D Ruleset Calibration Report

Generated: 2026-02-10T09:44:09
Panel: artifacts/walkforward_panel.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.65** | 150 | 7.4 | +1.56 | -22.37 | -5.15 | Y |
| 2 | 0.40 | 150 | 13.3 | +1.09 | -22.37 | -5.62 | Y |
| 3 | 0.60 | 150 | 8.3 | +1.07 | -22.64 | -5.72 | Y |
| 4 | 0.58 | 150 | 8.5 | +0.90 | -22.64 | -5.89 | Y |
| 5 | 0.60 | 120 | 6.3 | +0.97 | -22.98 | -5.92 | Y |
| 6 | 0.65 | 120 | 5.7 | +0.97 | -23.08 | -5.95 | Y |
| 7 | 0.40 | 120 | 10.5 | +0.75 | -22.37 | -5.96 | Y |
| 8 | 0.40 | 90 | 7.9 | +0.38 | -22.60 | -6.40 | Y |
| 9 | 0.58 | 120 | 6.6 | +0.49 | -22.98 | -6.40 | Y |
| 10 | 0.60 | 90 | 4.8 | +0.49 | -23.29 | -6.50 | Y |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 2.2 | 1.0 | +0.12 | -22.87 | -6.74 | N |
| 60 | 0.40 | 5.0 | 2.3 | +0.23 | -22.83 | -6.62 | Y |
| 90 | 0.40 | 7.9 | 3.6 | +0.38 | -22.60 | -6.40 | Y |
| 120 | 0.60 | 6.3 | 2.9 | +0.97 | -22.98 | -5.92 | Y |
| 150 | 0.65 | 7.4 | 3.4 | +1.56 | -22.37 | -5.15 | Y |

## Neighbor Stability

Best candidate: a_floor=0.65, catalyst_near=150, score=-5.1523

Neighbors (±1 step): 3 checked, 3 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | 150 | +1.07 | -5.72 | Y |
| 0.60 | 120 | +0.97 | -5.92 | Y |
| 0.65 | 120 | +0.97 | -5.95 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.65`, `catalyst_near_days = 150`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 13.85 | 15.73 | +1.88 |
| CD median 60d ret % | 14.42 | 14.17 | -0.25 |
| Separation (AB-CD) % | -0.57 | 1.56 | +2.13 |
| AB median max-DD % | -23.03 | -22.37 | +0.65 |
| Mean A-count/date | 2.60 | 3.40 | +0.80 |
| Top-25 overlap % | 84.30 | 84.90 | +0.60 |
| Mean turnover % | 15.70 | 15.10 | -0.60 |

### Why Chosen

- Objective score: -5.1523 (best among 18 passing candidates out of 40 combos)
- All constraints satisfied

