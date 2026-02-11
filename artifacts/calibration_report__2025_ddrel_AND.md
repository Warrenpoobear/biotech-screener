# 2D Ruleset Calibration Report

Generated: 2026-02-11T06:06:24
Panel: artifacts/walkforward_panel_2025_v1_3_ddrel.csv | Rows: 1808 | Baseline: a_floor=0.55, dd_rel=90
Grid: 8 a_floor × 5 dd_rel = 40 combos

## Top 10 Candidates

| # | a_floor | dd_rel | A%(elig) | Sep | AB DD | Score | Pass |
|---|---------|----------|----------|-----|-------|-------|------|
| 1 |  **0.60** | -0.25 | 10.7 | +4.54 | -19.81 | -1.40 | Y |
| 2 | 0.60 | -0.2 | 10.7 | +4.54 | -19.81 | -1.40 | Y |
| 3 | 0.58 | -0.25 | 11.1 | +4.18 | -19.83 | -1.77 | Y |
| 4 | 0.58 | -0.2 | 11.1 | +4.18 | -19.83 | -1.77 | Y |
| 5 | 0.65 | -0.25 | 9.4 | +3.39 | -19.83 | -2.56 | Y |
| 6 | 0.65 | -0.2 | 9.4 | +3.39 | -19.83 | -2.56 | Y |
| 7 | 0.60 | -0.15 | 9.4 | +2.53 | -19.83 | -3.42 | Y |
| 8 | 0.60 | -0.1 | 9.4 | +2.53 | -19.83 | -3.42 | Y |
| 9 | 0.60 | -0.05 | 9.4 | +2.53 | -19.83 | -3.42 | Y |
| 10 | 0.40 | -0.25 | 20.1 | +2.27 | -19.82 | -3.68 | Y |

## Ridge Summary (best a_floor per dd_rel)

| dd_rel | best a_floor | A%(elig) | A count | Sep | AB DD | Score | Pass |
|----------|-------------|----------|---------|-----|-------|-------|------|
| -0.25 | 0.60 | 10.7 | 2.6 | +4.54 | -19.81 | -1.40 | Y |
| -0.2 | 0.60 | 10.7 | 2.6 | +4.54 | -19.81 | -1.40 | Y |
| -0.15 | 0.60 | 9.4 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| -0.1 | 0.60 | 9.4 | 2.3 | +2.53 | -19.83 | -3.42 | Y |
| -0.05 | 0.60 | 9.4 | 2.3 | +2.53 | -19.83 | -3.42 | Y |

## Neighbor Stability

Best candidate: a_floor=0.60, dd_rel=-0.25, score=-1.4034

Neighbors (±1 step): 5 checked, 5 passing

| a_floor | dd_rel | Sep | Score | Pass |
|---------|----------|-----|-------|------|
| 0.60 | -0.2 | +4.54 | -1.40 | Y |
| 0.58 | -0.25 | +4.18 | -1.77 | Y |
| 0.58 | -0.2 | +4.18 | -1.77 | Y |
| 0.65 | -0.25 | +3.39 | -2.56 | Y |
| 0.65 | -0.2 | +3.39 | -2.56 | Y |

**Stability: STRONG** — all neighbors pass constraints.

## Recommended Candidate

**`tier_a_optionality_floor = 0.60`, `drawdown_rel_xbi_gate = -0.25`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 10.00 | 11.42 | +1.43 |
| CD median 60d ret % | 7.73 | 6.88 | -0.84 |
| Separation (AB-CD) % | 2.27 | 4.54 | +2.27 |
| AB median max-DD % | -19.82 | -19.81 | +0.01 |
| Mean A-count/date | 3.30 | 2.60 | -0.70 |
| Top-25 overlap % | 64.50 | 62.90 | -1.60 |
| Mean turnover % | 35.50 | 37.10 | +1.60 |

### Why Chosen

- Objective score: -1.4034 (best among 32 passing candidates out of 40 combos)
- All constraints satisfied

