# 2D Ruleset Calibration Report

Generated: 2026-02-11T06:06:37
Panel: artifacts/walkforward_panel_2025_v1_3_ddrel.csv | Rows: 1808 | Baseline: a_floor=0.55, dd_rel=90
Grid: 8 a_floor × 5 dd_rel = 40 combos

## Top 10 Candidates

| # | a_floor | dd_rel | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 | 0.50 | -0.05 | 0.0 | +16.15 | -14.36 | 11.84 | N |
| 2 | 0.52 | -0.05 | 0.0 | +16.15 | -14.36 | 11.84 | N |
| 3 | 0.55 | -0.05 | 0.0 | +16.15 | -14.36 | 11.84 | N |
| 4 | 0.58 | -0.05 | 0.0 | +16.15 | -14.36 | 11.84 | N |
| 5 | 0.60 | -0.05 | 0.0 | +16.15 | -14.36 | 11.84 | N |
| 6 | 0.65 | -0.2 | 4.9 | +8.01 | -18.88 | 2.35 | N |
| 7 | 0.65 | -0.15 | 2.0 | +6.12 | -18.88 | 0.46 | N |
| 8 | 0.50 | -0.1 | 0.0 | +6.77 | -22.39 | 0.05 | N |
| 9 | 0.52 | -0.1 | 0.0 | +6.77 | -22.39 | 0.05 | N |
| 10 | 0.55 | -0.1 | 0.0 | +6.77 | -22.39 | 0.05 | N |

## Ridge Summary (best a_floor per dd_rel)

| dd_rel | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| -0.25 | 0.60 | 7.8 | 2.4 | +3.11 | -21.44 | -3.32 | Y |
| -0.2 | 0.40 | 10.7 | 3.2 | +3.05 | -18.87 | -2.61 | Y |
| -0.15 | 0.65 | 2.0 | 1.7 | +6.12 | -18.88 | 0.46 | N |
| -0.1 | 0.50 | 0.0 | 0.0 | +6.77 | -22.39 | 0.05 | N |
| -0.05 | 0.50 | 0.0 | 0.0 | +16.15 | -14.36 | 11.84 | N |

## Neighbor Stability

Best candidate: a_floor=0.40, dd_rel=-0.2, score=-2.6103

Neighbors (±1 step): 5 checked, 1 passing

| a_floor | dd_rel | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.40 | -0.15 | +3.52 | -1.86 | N |
| 0.40 | -0.25 | +2.27 | -4.22 | Y |
| 0.45 | -0.2 | -2.04 | -7.89 | N |
| 0.45 | -0.25 | -2.21 | -8.81 | N |
| 0.45 | -0.15 | -5.91 | -11.85 | N |

**Stability: WEAK** — only 1/5 neighbors pass.

## Recommended Candidate

**`tier_a_optionality_floor = 0.40`, `drawdown_rel_xbi_gate = -0.2`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 10.00 | 11.42 | +1.43 |
| CD median 60d ret % | 7.73 | 8.37 | +0.65 |
| Separation (AB-CD) % | 2.27 | 3.05 | +0.78 |
| AB median max-DD % | -19.82 | -18.87 | +0.95 |
| Mean A-count/date | 3.30 | 3.20 | -0.10 |
| Top-25 overlap % | 64.50 | 53.00 | -11.50 |
| Mean turnover % | 35.50 | 47.00 | +11.50 |

### Tradeoffs

- Lower stability: overlap 53.0% vs baseline 64.5%

### Why Chosen

- Objective score: -2.6103 (best among 9 passing candidates out of 40 combos)
- All constraints satisfied

