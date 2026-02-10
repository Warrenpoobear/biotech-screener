# 2D Ruleset Calibration Report

Generated: 2026-02-10T09:36:55
Panel: artifacts/walkforward_panel_full.csv | Rows: 3922 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.58** | 150 | 6.2 | -3.15 | -24.26 | -10.43 | N |
| 2 | 0.60 | 150 | 5.8 | -3.19 | -24.23 | -10.46 | N |
| 3 | 0.60 | 120 | 4.3 | -3.15 | -24.43 | -10.48 | N |
| 4 | 0.58 | 120 | 4.7 | -3.15 | -24.44 | -10.48 | N |
| 5 | 0.40 | 120 | 7.7 | -3.55 | -23.11 | -10.48 | N |
| 6 | 0.40 | 150 | 9.9 | -3.55 | -23.11 | -10.48 | N |
| 7 | 0.40 | 90 | 5.6 | -3.80 | -23.15 | -10.74 | N |
| 8 | 0.40 | 60 | 3.6 | -3.85 | -23.18 | -10.80 | N |
| 9 | 0.55 | 150 | 7.1 | -3.51 | -24.35 | -10.82 | N |
| 10 | 0.45 | 120 | 7.1 | -3.89 | -23.29 | -10.88 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 1.6 | 0.6 | -3.96 | -23.20 | -10.92 | N |
| 60 | 0.40 | 3.6 | 1.3 | -3.85 | -23.18 | -10.80 | N |
| 90 | 0.40 | 5.6 | 2.1 | -3.80 | -23.15 | -10.74 | N |
| 120 | 0.60 | 4.3 | 1.6 | -3.15 | -24.43 | -10.48 | N |
| 150 | 0.58 | 6.2 | 2.3 | -3.15 | -24.26 | -10.43 | N |

## Neighbor Stability

Best candidate: a_floor=0.58, catalyst_near=150, score=-10.4270

Neighbors (±1 step): 5 checked, 0 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | 150 | -3.19 | -10.46 | N |
| 0.60 | 120 | -3.15 | -10.48 | N |
| 0.58 | 120 | -3.15 | -10.48 | N |
| 0.55 | 150 | -3.51 | -10.82 | N |
| 0.55 | 120 | -3.64 | -10.99 | N |

**Stability: WEAK** — only 0/5 neighbors pass.

## Recommended Candidate

**No candidate passed all constraints.**

Consider relaxing constraints or expanding the search space.

