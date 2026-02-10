# 2D Ruleset Calibration Report

Generated: 2026-02-10T10:20:40
Panel: artifacts/walkforward_panel_full.csv | Rows: 1808 | Baseline: a_floor=0.55, catalyst_near=90
Grid: 8 a_floor × 5 catalyst_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 | 0.40 | 30 | 2.2 | +0.30 | -20.56 | -5.87 | N |
| 2 |  **0.40** | 60 | 5.0 | +0.30 | -20.56 | -5.87 | Y |
| 3 | 0.40 | 90 | 7.9 | +0.30 | -20.56 | -5.87 | Y |
| 4 | 0.40 | 120 | 10.5 | +0.30 | -20.56 | -5.87 | Y |
| 5 | 0.40 | 150 | 13.3 | +0.30 | -20.56 | -5.87 | Y |
| 6 | 0.65 | 150 | 7.4 | +0.26 | -20.56 | -5.91 | Y |
| 7 | 0.60 | 120 | 6.3 | +0.19 | -20.56 | -5.98 | Y |
| 8 | 0.60 | 150 | 8.3 | +0.19 | -20.56 | -5.98 | Y |
| 9 | 0.65 | 120 | 5.7 | +0.19 | -21.43 | -6.24 | Y |
| 10 | 0.60 | 90 | 4.8 | -0.16 | -20.56 | -6.33 | N |

## Ridge Summary (best a_floor per catalyst_near_days)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 30 | 0.40 | 2.2 | 1.0 | +0.30 | -20.56 | -5.87 | N |
| 60 | 0.40 | 5.0 | 2.3 | +0.30 | -20.56 | -5.87 | Y |
| 90 | 0.40 | 7.9 | 3.6 | +0.30 | -20.56 | -5.87 | Y |
| 120 | 0.40 | 10.5 | 4.8 | +0.30 | -20.56 | -5.87 | Y |
| 150 | 0.40 | 13.3 | 6.1 | +0.30 | -20.56 | -5.87 | Y |

## Neighbor Stability

Best candidate: a_floor=0.40, catalyst_near=60, score=-5.8667

Neighbors (±1 step): 5 checked, 1 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.40 | 30 | +0.30 | -5.87 | N |
| 0.40 | 90 | +0.30 | -5.87 | Y |
| 0.45 | 90 | -2.53 | -8.70 | N |
| 0.45 | 30 | -2.61 | -8.78 | N |
| 0.45 | 60 | -2.61 | -8.78 | N |

**Stability: WEAK** — only 1/5 neighbors pass.

## Recommended Candidate

**`tier_a_optionality_floor = 0.40`, `catalyst_near_days = 60`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 11.74 | 14.43 | +2.69 |
| CD median 60d ret % | 14.42 | 14.13 | -0.29 |
| Separation (AB-CD) % | -2.69 | 0.30 | +2.99 |
| AB median max-DD % | -20.56 | -20.56 | +0.00 |
| Mean A-count/date | 2.60 | 2.30 | -0.30 |
| Top-25 overlap % | 84.30 | 84.50 | +0.20 |
| Mean turnover % | 15.70 | 15.50 | -0.20 |

### Why Chosen

- Objective score: -5.8667 (best among 8 passing candidates out of 40 combos)
- All constraints satisfied

