# 2D Ruleset Calibration Report

Generated: 2026-02-12T15:04:33
Panel: artifacts/walkforward_panel__secadcom_combined.csv | Rows: 1808 | Baseline: a_floor=0.55, cat_near=90
Grid: 8 a_floor × 5 cat_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.70** | 60 | 10.0 | +7.38 | -19.80 | 1.44 | Y |
| 2 | 0.70 | 90 | 10.0 | +7.38 | -19.80 | 1.44 | Y |
| 3 | 0.70 | 120 | 10.0 | +7.38 | -19.80 | 1.44 | Y |
| 4 | 0.70 | 150 | 10.0 | +7.38 | -19.80 | 1.44 | Y |
| 5 | 0.70 | 180 | 10.0 | +7.38 | -19.80 | 1.44 | Y |
| 6 | 0.65 | 60 | 10.3 | +6.76 | -19.96 | 0.77 | Y |
| 7 | 0.65 | 90 | 10.3 | +6.76 | -19.96 | 0.77 | Y |
| 8 | 0.65 | 120 | 10.3 | +6.76 | -19.96 | 0.77 | Y |
| 9 | 0.65 | 150 | 10.3 | +6.76 | -19.96 | 0.77 | Y |
| 10 | 0.65 | 180 | 10.3 | +6.76 | -19.96 | 0.77 | Y |

## Ridge Summary (best a_floor per cat_near)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 60 | 0.70 | 10.0 | 3.5 | +7.38 | -19.80 | 1.44 | Y |
| 90 | 0.70 | 10.0 | 3.5 | +7.38 | -19.80 | 1.44 | Y |
| 120 | 0.70 | 10.0 | 3.5 | +7.38 | -19.80 | 1.44 | Y |
| 150 | 0.70 | 10.0 | 3.5 | +7.38 | -19.80 | 1.44 | Y |
| 180 | 0.70 | 10.0 | 3.5 | +7.38 | -19.80 | 1.44 | Y |

## Neighbor Stability

Best candidate: a_floor=0.70, cat_near=60, score=1.4392

Neighbors (±1 step): 3 checked, 3 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.70 | 90 | +7.38 | 1.44 | Y |
| 0.65 | 60 | +6.76 | 0.77 | Y |
| 0.65 | 90 | +6.76 | 0.77 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.70`, `catalyst_near_days = 60`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 14.43 | 16.39 | +1.96 |
| CD median 60d ret % | 9.59 | 9.01 | -0.58 |
| Separation (AB-CD) % | 4.84 | 7.38 | +2.54 |
| AB median max-DD % | -21.15 | -19.80 | +1.35 |
| Mean A-count/date | 4.80 | 3.50 | -1.30 |
| Top-25 overlap % | 72.10 | 68.10 | -4.00 |
| Mean turnover % | 27.90 | 31.90 | +4.00 |

### Tradeoffs

- Lower stability: overlap 68.1% vs baseline 72.1%

### Why Chosen

- Objective score: 1.4392 (best among 40 passing candidates out of 40 combos)
- All constraints satisfied

