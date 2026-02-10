# 2D Ruleset Calibration Report

Generated: 2026-02-10T13:21:58
Panel: artifacts/walkforward_panel_v1_3_full.csv | Rows: 1808 | Baseline: a_floor=0.55, cat_mid=180
Grid: 7 a_floor × 7 cat_mid = 49 combos

## Top 10 Candidates

| # | a_floor | cat_mid | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | 150 | 7.3 | +2.78 | -19.83 | -3.17 | Y |
| 2 | 0.60 | 180 | 9.9 | +2.53 | -19.83 | -3.42 | Y |
| 3 | 0.62 | 180 | 9.9 | +2.21 | -19.83 | -3.74 | Y |
| 4 | 0.60 | 210 | 11.2 | +2.21 | -19.83 | -3.74 | Y |
| 5 | 0.62 | 210 | 10.7 | +2.21 | -19.83 | -3.74 | Y |
| 6 | 0.58 | 180 | 10.3 | +2.41 | -20.63 | -3.78 | Y |
| 7 | 0.58 | 210 | 12.0 | +2.41 | -20.63 | -3.78 | Y |
| 8 | 0.62 | 150 | 7.3 | +2.01 | -19.83 | -3.94 | Y |
| 9 | 0.58 | 150 | 7.7 | +2.21 | -21.43 | -4.22 | Y |
| 10 | 0.55 | 180 | 12.9 | +1.43 | -20.63 | -4.76 | Y |

## Ridge Summary (best a_floor per cat_mid)

| cat_mid | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 150 | 0.60 | 7.3 | 1.7 | +2.78 | -19.83 | -3.17 | Y |
| 180 | 0.60 | 9.9 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| 210 | 0.60 | 11.2 | 2.6 | +2.21 | -19.83 | -3.74 | Y |
| 240 | 0.60 | 11.6 | 2.7 | +1.46 | -21.43 | -4.97 | Y |
| 270 | 0.60 | 12.0 | 2.8 | +1.03 | -21.43 | -5.40 | Y |
| 300 | 0.60 | 12.4 | 2.9 | +1.03 | -21.43 | -5.40 | Y |
| 365 | 0.60 | 13.3 | 3.1 | +1.46 | -21.43 | -4.97 | Y |

## Neighbor Stability

Best candidate: a_floor=0.60, cat_mid=150, score=-3.1683

Neighbors (±1 step): 5 checked, 5 passing

| a_floor | cat_mid | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | 180 | +2.53 | -3.42 | Y |
| 0.62 | 180 | +2.21 | -3.74 | Y |
| 0.58 | 180 | +2.41 | -3.78 | Y |
| 0.62 | 150 | +2.01 | -3.94 | Y |
| 0.58 | 150 | +2.21 | -4.22 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.60`, `catalyst_mid_days = 150`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 9.41 | 10.50 | +1.09 |
| CD median 60d ret % | 7.98 | 7.73 | -0.26 |
| Separation (AB-CD) % | 1.43 | 2.78 | +1.35 |
| AB median max-DD % | -20.63 | -19.83 | +0.80 |
| Mean A-count/date | 3.00 | 1.70 | -1.30 |
| Top-25 overlap % | 71.40 | 70.10 | -1.30 |
| Mean turnover % | 28.60 | 29.90 | +1.30 |

### Why Chosen

- Objective score: -3.1683 (best among 36 passing candidates out of 49 combos)
- All constraints satisfied

