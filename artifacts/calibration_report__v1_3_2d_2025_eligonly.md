# 2D Ruleset Calibration Report

Generated: 2026-02-10T11:36:58
Panel: artifacts/walkforward_panel_v1_3_full.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 30 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 2 | 0.60 | 60 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 3 | 0.60 | 90 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 4 | 0.60 | 120 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 5 | 0.60 | 150 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 6 | 0.58 | 30 | 10.3 | +2.41 | -20.63 | -3.78 | Y |
| 7 | 0.58 | 60 | 10.3 | +2.41 | -20.63 | -3.78 | Y |
| 8 | 0.58 | 90 | 10.3 | +2.41 | -20.63 | -3.78 | Y |
| 9 | 0.58 | 120 | 10.3 | +2.41 | -20.63 | -3.78 | Y |
| 10 | 0.58 | 150 | 10.3 | +2.41 | -20.63 | -3.78 | Y |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| 60 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| 90 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| 120 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| 150 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |

## Neighbor Stability

Best candidate: a_floor=0.60, catalyst_near=30, score=-3.4183

Neighbors (±1 step): 5 checked, 5 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | 60 | +2.53 | -3.42 | Y |
| 0.58 | 30 | +2.41 | -3.78 | Y |
| 0.58 | 60 | +2.41 | -3.78 | Y |
| 0.65 | 30 | +1.50 | -4.93 | Y |
| 0.65 | 60 | +1.50 | -4.93 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.60`, `catalyst_near_days = 30`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 9.41 | 10.00 | +0.58 |
| CD median 60d ret % | 7.98 | 7.47 | -0.52 |
| Separation (AB-CD) % | 1.43 | 2.53 | +1.10 |
| AB median max-DD % | -20.63 | -19.83 | +0.80 |
| Mean A-count/date | 3.00 | 2.30 | -0.70 |
| Top-25 overlap % | 71.40 | 69.50 | -1.90 |
| Mean turnover % | 28.60 | 30.50 | +1.90 |

### Why Chosen

- Objective score: -3.4183 (best among 30 passing candidates out of 40 combos)
- All constraints satisfied

