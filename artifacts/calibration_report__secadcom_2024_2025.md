# 2D Ruleset Calibration Report

Generated: 2026-02-12T15:04:26
Panel: artifacts/walkforward_panel__secadcom_combined.csv | Rows: 3922 | Baseline: a_floor=0.55, cat_near=90
Grid: 8 a_floor × 5 cat_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.70** | 60 | 7.8 | +1.67 | -23.88 | -5.49 | Y |
| 2 | 0.70 | 90 | 7.8 | +1.67 | -23.88 | -5.49 | Y |
| 3 | 0.70 | 120 | 7.8 | +1.67 | -23.88 | -5.49 | Y |
| 4 | 0.70 | 150 | 7.8 | +1.67 | -23.88 | -5.49 | Y |
| 5 | 0.70 | 180 | 7.8 | +1.67 | -23.88 | -5.49 | Y |
| 6 | 0.58 | 60 | 10.2 | +1.48 | -23.91 | -5.69 | Y |
| 7 | 0.58 | 90 | 10.2 | +1.48 | -23.91 | -5.69 | Y |
| 8 | 0.58 | 120 | 10.2 | +1.48 | -23.91 | -5.69 | Y |
| 9 | 0.58 | 150 | 10.2 | +1.48 | -23.91 | -5.69 | Y |
| 10 | 0.58 | 180 | 10.2 | +1.48 | -23.91 | -5.69 | Y |

## Ridge Summary (best a_floor per cat_near)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 60 | 0.70 | 7.8 | 3.8 | +1.67 | -23.88 | -5.49 | Y |
| 90 | 0.70 | 7.8 | 3.8 | +1.67 | -23.88 | -5.49 | Y |
| 120 | 0.70 | 7.8 | 3.8 | +1.67 | -23.88 | -5.49 | Y |
| 150 | 0.70 | 7.8 | 3.8 | +1.67 | -23.88 | -5.49 | Y |
| 180 | 0.70 | 7.8 | 3.8 | +1.67 | -23.88 | -5.49 | Y |

## Neighbor Stability

Best candidate: a_floor=0.70, cat_near=60, score=-5.4935

Neighbors (±1 step): 3 checked, 3 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.70 | 90 | +1.67 | -5.49 | Y |
| 0.65 | 60 | +0.39 | -6.79 | Y |
| 0.65 | 90 | +0.39 | -6.79 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.70`, `catalyst_near_days = 60`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | -1.60 | -0.51 | +1.09 |
| CD median 60d ret % | -1.96 | -2.18 | -0.21 |
| Separation (AB-CD) % | 0.36 | 1.67 | +1.31 |
| AB median max-DD % | -23.97 | -23.88 | +0.09 |
| Mean A-count/date | 5.30 | 3.80 | -1.50 |
| Top-25 overlap % | 73.80 | 71.50 | -2.30 |
| Mean turnover % | 26.20 | 28.50 | +2.30 |

### Tradeoffs

- Lower stability: overlap 71.5% vs baseline 73.8%

### Why Chosen

- Objective score: -5.4935 (best among 30 passing candidates out of 40 combos)
- All constraints satisfied

