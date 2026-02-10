# 2D Ruleset Calibration Report

Generated: 2026-02-10T10:20:32
Panel: artifacts/walkforward_panel_full.csv | Rows: 3922 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.58** | 120 | 6.0 | -0.70 | -22.13 | -7.34 | N |
| 2 | 0.58 | 150 | 7.6 | -0.75 | -21.96 | -7.34 | N |
| 3 | 0.60 | 150 | 7.3 | -0.75 | -22.03 | -7.36 | N |
| 4 | 0.55 | 150 | 8.1 | -0.84 | -22.03 | -7.45 | N |
| 5 | 0.55 | 120 | 6.5 | -0.82 | -22.13 | -7.46 | N |
| 6 | 0.60 | 120 | 5.8 | -0.68 | -22.68 | -7.48 | N |
| 7 | 0.60 | 90 | 4.4 | -0.71 | -22.89 | -7.58 | N |
| 8 | 0.58 | 90 | 4.6 | -0.78 | -22.77 | -7.61 | N |
| 9 | 0.58 | 60 | 3.1 | -0.79 | -23.01 | -7.69 | N |
| 10 | 0.60 | 60 | 3.0 | -0.78 | -23.23 | -7.75 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 2.0 | 1.2 | -2.34 | -20.81 | -8.58 | N |
| 60 | 0.58 | 3.1 | 1.9 | -0.79 | -23.01 | -7.69 | N |
| 90 | 0.60 | 4.4 | 2.6 | -0.71 | -22.89 | -7.58 | N |
| 120 | 0.58 | 6.0 | 3.6 | -0.70 | -22.13 | -7.34 | N |
| 150 | 0.58 | 7.6 | 4.5 | -0.75 | -21.96 | -7.34 | N |

## Neighbor Stability

Best candidate: a_floor=0.58, catalyst_near=120, score=-7.3381

Neighbors (±1 step): 8 checked, 0 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.58 | 150 | -0.75 | -7.34 | N |
| 0.60 | 150 | -0.75 | -7.36 | N |
| 0.55 | 150 | -0.84 | -7.45 | N |
| 0.55 | 120 | -0.82 | -7.46 | N |
| 0.60 | 120 | -0.68 | -7.48 | N |
| 0.60 | 90 | -0.71 | -7.58 | N |
| 0.58 | 90 | -0.78 | -7.61 | N |
| 0.55 | 90 | -1.75 | -8.58 | N |

**Stability: WEAK** — only 0/8 neighbors pass.

## Recommended Candidate

**No candidate passed all constraints.**

Consider relaxing constraints or expanding the search space.

