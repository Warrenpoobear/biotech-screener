# 2D Ruleset Calibration Report

Generated: 2026-02-12T12:33:32
Panel: artifacts/walkforward_panel__original_combined.csv | Rows: 1808 | Baseline: a_floor=0.55, cat_near=90
Grid: 8 a_floor × 5 cat_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.70** | 60 | 9.7 | +8.05 | -19.83 | 2.10 | Y |
| 2 | 0.70 | 90 | 9.7 | +8.05 | -19.83 | 2.10 | Y |
| 3 | 0.70 | 120 | 9.7 | +8.05 | -19.83 | 2.10 | Y |
| 4 | 0.70 | 150 | 9.7 | +8.05 | -19.83 | 2.10 | Y |
| 5 | 0.70 | 180 | 9.7 | +8.05 | -19.83 | 2.10 | Y |
| 6 | 0.65 | 60 | 10.0 | +7.07 | -20.10 | 1.04 | Y |
| 7 | 0.65 | 90 | 10.0 | +7.07 | -20.10 | 1.04 | Y |
| 8 | 0.65 | 120 | 10.0 | +7.07 | -20.10 | 1.04 | Y |
| 9 | 0.65 | 150 | 10.0 | +7.07 | -20.10 | 1.04 | Y |
| 10 | 0.65 | 180 | 10.0 | +7.07 | -20.10 | 1.04 | Y |

## Ridge Summary (best a_floor per cat_near)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 60 | 0.70 | 9.7 | 3.4 | +8.05 | -19.83 | 2.10 | Y |
| 90 | 0.70 | 9.7 | 3.4 | +8.05 | -19.83 | 2.10 | Y |
| 120 | 0.70 | 9.7 | 3.4 | +8.05 | -19.83 | 2.10 | Y |
| 150 | 0.70 | 9.7 | 3.4 | +8.05 | -19.83 | 2.10 | Y |
| 180 | 0.70 | 9.7 | 3.4 | +8.05 | -19.83 | 2.10 | Y |

## Neighbor Stability

Best candidate: a_floor=0.70, cat_near=60, score=2.1017

Neighbors (±1 step): 3 checked, 3 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.70 | 90 | +8.05 | 2.10 | Y |
| 0.65 | 60 | +7.07 | 1.04 | Y |
| 0.65 | 90 | +7.07 | 1.04 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.70`, `catalyst_near_days = 60`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 14.43 | 16.93 | +2.50 |
| CD median 60d ret % | 9.59 | 8.88 | -0.71 |
| Separation (AB-CD) % | 4.84 | 8.05 | +3.21 |
| AB median max-DD % | -21.15 | -19.83 | +1.32 |
| Mean A-count/date | 4.60 | 3.40 | -1.20 |
| Top-25 overlap % | 72.10 | 68.60 | -3.50 |
| Mean turnover % | 27.90 | 31.40 | +3.50 |

### Tradeoffs

- Lower stability: overlap 68.6% vs baseline 72.1%

### Why Chosen

- Objective score: 2.1017 (best among 40 passing candidates out of 40 combos)
- All constraints satisfied

