# 2D Ruleset Calibration Report

Generated: 2026-02-10T08:11:42
Panel: artifacts/walkforward_panel_2025.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 120 | 6.3 | +0.97 | -22.98 | -5.92 | Y |
| 2 | 0.65 | 120 | 5.7 | +0.97 | -23.08 | -5.95 | Y |
| 3 | 0.40 | 120 | 10.5 | +0.75 | -22.37 | -5.96 | Y |
| 4 | 0.40 | 90 | 7.9 | +0.38 | -22.60 | -6.40 | Y |
| 5 | 0.58 | 120 | 6.6 | +0.49 | -22.98 | -6.40 | Y |
| 6 | 0.60 | 90 | 4.8 | +0.49 | -23.29 | -6.50 | Y |
| 7 | 0.65 | 90 | 4.2 | +0.49 | -23.41 | -6.53 | Y |
| 8 | 0.40 | 60 | 5.0 | +0.23 | -22.83 | -6.62 | Y |
| 9 | 0.50 | 120 | 8.3 | +0.02 | -22.30 | -6.67 | Y |
| 10 | 0.52 | 120 | 7.7 | -0.02 | -22.30 | -6.71 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 2.2 | 1.0 | +0.12 | -22.87 | -6.74 | N |
| 45 | 0.40 | 4.2 | 1.9 | +0.12 | -22.87 | -6.74 | Y |
| 60 | 0.40 | 5.0 | 2.3 | +0.23 | -22.83 | -6.62 | Y |
| 90 | 0.40 | 7.9 | 3.6 | +0.38 | -22.60 | -6.40 | Y |
| 120 | 0.60 | 6.3 | 2.9 | +0.97 | -22.98 | -5.92 | Y |

## Neighbor Stability

Best candidate: a_floor=0.60, catalyst_near=120, score=-5.9245

Neighbors (±1 step): 5 checked, 5 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.65 | 120 | +0.97 | -5.95 | Y |
| 0.58 | 120 | +0.49 | -6.40 | Y |
| 0.60 | 90 | +0.49 | -6.50 | Y |
| 0.65 | 90 | +0.49 | -6.53 | Y |
| 0.58 | 90 | +0.02 | -6.97 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.60`, `catalyst_near_days = 120`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 13.85 | 15.27 | +1.41 |
| CD median 60d ret % | 14.42 | 14.30 | -0.13 |
| Separation (AB-CD) % | -0.57 | 0.97 | +1.54 |
| AB median max-DD % | -23.03 | -22.98 | +0.05 |
| Mean A-count/date | 2.60 | 2.90 | +0.30 |
| Top-25 overlap % | 84.30 | 86.60 | +2.30 |
| Mean turnover % | 15.70 | 13.40 | -2.30 |

### Why Chosen

- Objective score: -5.9245 (best among 12 passing candidates out of 40 combos)
- All constraints satisfied

