# 2D Ruleset Calibration Report

Generated: 2026-02-12T12:32:31
Panel: artifacts/walkforward_panel__backfilled_combined.csv | Rows: 3922 | Baseline: a_floor=0.55, cat_near=90
Grid: 8 a_floor × 5 cat_near = 40 combos

## Top 10 Candidates

| # | a_floor | cat_near | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.58** | 60 | 9.0 | +1.95 | -23.91 | -5.22 | Y |
| 2 | 0.58 | 90 | 9.0 | +1.95 | -23.91 | -5.22 | Y |
| 3 | 0.58 | 120 | 9.0 | +1.95 | -23.91 | -5.22 | Y |
| 4 | 0.58 | 150 | 9.0 | +1.95 | -23.91 | -5.22 | Y |
| 5 | 0.58 | 180 | 9.0 | +1.95 | -23.91 | -5.22 | Y |
| 6 | 0.70 | 60 | 7.4 | +1.97 | -24.05 | -5.24 | Y |
| 7 | 0.70 | 90 | 7.4 | +1.97 | -24.05 | -5.24 | Y |
| 8 | 0.70 | 120 | 7.4 | +1.97 | -24.05 | -5.24 | Y |
| 9 | 0.70 | 150 | 7.4 | +1.97 | -24.05 | -5.24 | Y |
| 10 | 0.70 | 180 | 7.4 | +1.97 | -24.05 | -5.24 | Y |

## Ridge Summary (best a_floor per cat_near)

| cat_near | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| 60 | 0.58 | 9.0 | 4.3 | +1.95 | -23.91 | -5.22 | Y |
| 90 | 0.58 | 9.0 | 4.3 | +1.95 | -23.91 | -5.22 | Y |
| 120 | 0.58 | 9.0 | 4.3 | +1.95 | -23.91 | -5.22 | Y |
| 150 | 0.58 | 9.0 | 4.3 | +1.95 | -23.91 | -5.22 | Y |
| 180 | 0.58 | 9.0 | 4.3 | +1.95 | -23.91 | -5.22 | Y |

## Neighbor Stability

Best candidate: a_floor=0.58, cat_near=60, score=-5.2229

Neighbors (±1 step): 5 checked, 5 passing

| a_floor | cat_near | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.58 | 90 | +1.95 | -5.22 | Y |
| 0.60 | 60 | +1.69 | -5.46 | Y |
| 0.60 | 90 | +1.69 | -5.46 | Y |
| 0.55 | 60 | +0.96 | -6.23 | Y |
| 0.55 | 90 | +0.96 | -6.23 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.58`, `catalyst_near_days = 60`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | -1.21 | -0.42 | +0.79 |
| CD median 60d ret % | -2.18 | -2.37 | -0.20 |
| Separation (AB-CD) % | 0.96 | 1.95 | +0.99 |
| AB median max-DD % | -23.96 | -23.91 | +0.05 |
| Mean A-count/date | 4.70 | 4.30 | -0.40 |
| Top-25 overlap % | 73.70 | 73.10 | -0.60 |
| Mean turnover % | 26.30 | 26.90 | +0.60 |

### Why Chosen

- Objective score: -5.2229 (best among 30 passing candidates out of 40 combos)
- All constraints satisfied

