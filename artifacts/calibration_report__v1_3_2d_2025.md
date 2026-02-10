# 2D Ruleset Calibration Report

Generated: 2026-02-10T11:32:23
Panel: artifacts/walkforward_panel_v1_3_full.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 30 | 9.9 | -4.65 | -19.83 | -10.60 | N |
| 2 | 0.60 | 60 | 9.9 | -4.65 | -19.83 | -10.60 | N |
| 3 | 0.60 | 90 | 9.9 | -4.65 | -19.83 | -10.60 | N |
| 4 | 0.60 | 120 | 9.9 | -4.65 | -19.83 | -10.60 | N |
| 5 | 0.60 | 150 | 9.9 | -4.65 | -19.83 | -10.60 | N |
| 6 | 0.58 | 30 | 10.3 | -4.85 | -20.63 | -11.04 | N |
| 7 | 0.58 | 60 | 10.3 | -4.85 | -20.63 | -11.04 | N |
| 8 | 0.58 | 90 | 10.3 | -4.85 | -20.63 | -11.04 | N |
| 9 | 0.58 | 120 | 10.3 | -4.85 | -20.63 | -11.04 | N |
| 10 | 0.58 | 150 | 10.3 | -4.85 | -20.63 | -11.04 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.60 | 9.9 | 2.3 | -4.65 | -19.83 | -10.60 | N |
| 60 | 0.60 | 9.9 | 2.3 | -4.65 | -19.83 | -10.60 | N |
| 90 | 0.60 | 9.9 | 2.3 | -4.65 | -19.83 | -10.60 | N |
| 120 | 0.60 | 9.9 | 2.3 | -4.65 | -19.83 | -10.60 | N |
| 150 | 0.60 | 9.9 | 2.3 | -4.65 | -19.83 | -10.60 | N |

## Neighbor Stability

Best candidate: a_floor=0.60, catalyst_near=30, score=-10.5983

Neighbors (±1 step): 5 checked, 0 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | 60 | -4.65 | -10.60 | N |
| 0.58 | 30 | -4.85 | -11.04 | N |
| 0.58 | 60 | -4.85 | -11.04 | N |
| 0.65 | 30 | -4.65 | -11.08 | N |
| 0.65 | 60 | -4.65 | -11.08 | N |

**Stability: WEAK** — only 0/5 neighbors pass.

## Recommended Candidate

**No candidate passed all constraints.**

Consider relaxing constraints or expanding the search space.

