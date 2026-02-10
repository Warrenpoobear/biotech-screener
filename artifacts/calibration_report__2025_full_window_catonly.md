# 2D Ruleset Calibration Report

Generated: 2026-02-10T09:49:12
Panel: artifacts/walkforward_panel_full.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 150 | 7.3 | -4.11 | -19.83 | -10.06 | N |
| 2 | 0.40 | 150 | 15.5 | -4.92 | -19.83 | -10.87 | N |
| 3 | 0.65 | 150 | 6.4 | -4.65 | -21.44 | -11.08 | N |
| 4 | 0.40 | 120 | 11.6 | -5.18 | -19.83 | -11.13 | N |
| 5 | 0.60 | 120 | 4.7 | -4.75 | -21.43 | -11.18 | N |
| 6 | 0.58 | 150 | 7.7 | -4.75 | -21.43 | -11.18 | N |
| 7 | 0.60 | 90 | 2.6 | -4.85 | -21.63 | -11.34 | N |
| 8 | 0.40 | 90 | 7.7 | -5.42 | -19.83 | -11.37 | N |
| 9 | 0.40 | 60 | 4.7 | -5.63 | -19.83 | -11.58 | N |
| 10 | 0.58 | 120 | 5.2 | -5.32 | -21.44 | -11.75 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 1.7 | 0.4 | -5.87 | -20.63 | -12.06 | N |
| 60 | 0.40 | 4.7 | 1.1 | -5.63 | -19.83 | -11.58 | N |
| 90 | 0.60 | 2.6 | 0.6 | -4.85 | -21.63 | -11.34 | N |
| 120 | 0.40 | 11.6 | 2.7 | -5.18 | -19.83 | -11.13 | N |
| 150 | 0.60 | 7.3 | 1.7 | -4.11 | -19.83 | -10.06 | N |

## Neighbor Stability

Best candidate: a_floor=0.60, catalyst_near=150, score=-10.0583

Neighbors (±1 step): 5 checked, 0 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.65 | 150 | -4.65 | -11.08 | N |
| 0.60 | 120 | -4.75 | -11.18 | N |
| 0.58 | 150 | -4.75 | -11.18 | N |
| 0.58 | 120 | -5.32 | -11.75 | N |
| 0.65 | 120 | -7.49 | -14.04 | N |

**Stability: WEAK** — only 0/5 neighbors pass.

## Recommended Candidate

**No candidate passed all constraints.**

Consider relaxing constraints or expanding the search space.

