# 2D Ruleset Calibration Report

Generated: 2026-02-10T09:48:41
Panel: artifacts/walkforward_panel.csv | Rows: 3922 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 120 | 5.8 | -0.50 | -25.09 | -8.03 | N |
| 2 | 0.58 | 150 | 7.6 | -0.53 | -25.00 | -8.03 | N |
| 3 | 0.58 | 120 | 6.0 | -0.50 | -25.11 | -8.03 | N |
| 4 | 0.60 | 150 | 7.3 | -0.60 | -24.91 | -8.07 | N |
| 5 | 0.65 | 120 | 5.1 | -0.50 | -25.39 | -8.12 | N |
| 6 | 0.65 | 150 | 6.5 | -0.53 | -25.33 | -8.13 | N |
| 7 | 0.40 | 120 | 8.2 | -1.18 | -23.97 | -8.37 | N |
| 8 | 0.40 | 150 | 10.2 | -1.18 | -23.97 | -8.37 | N |
| 9 | 0.55 | 150 | 8.1 | -0.84 | -25.12 | -8.38 | N |
| 10 | 0.58 | 90 | 4.6 | -0.79 | -25.38 | -8.40 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.60 | 1.6 | 1.0 | -1.13 | -25.75 | -8.85 | N |
| 60 | 0.60 | 3.0 | 1.8 | -0.86 | -25.59 | -8.54 | N |
| 90 | 0.58 | 4.6 | 2.7 | -0.79 | -25.38 | -8.40 | N |
| 120 | 0.60 | 5.8 | 3.5 | -0.50 | -25.09 | -8.03 | N |
| 150 | 0.58 | 7.6 | 4.5 | -0.53 | -25.00 | -8.03 | N |

## Neighbor Stability

Best candidate: a_floor=0.60, catalyst_near=120, score=-8.0283

Neighbors (±1 step): 8 checked, 0 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.58 | 150 | -0.53 | -8.03 | N |
| 0.58 | 120 | -0.50 | -8.03 | N |
| 0.60 | 150 | -0.60 | -8.07 | N |
| 0.65 | 120 | -0.50 | -8.12 | N |
| 0.65 | 150 | -0.53 | -8.13 | N |
| 0.58 | 90 | -0.79 | -8.40 | N |
| 0.60 | 90 | -0.79 | -8.40 | N |
| 0.65 | 90 | -0.79 | -8.51 | N |

**Stability: WEAK** — only 0/8 neighbors pass.

## Recommended Candidate

**No candidate passed all constraints.**

Consider relaxing constraints or expanding the search space.

