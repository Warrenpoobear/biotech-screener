# Ruleset Calibration Report

Generated: 2026-02-10T08:00:03
Panel: artifacts/walkforward_panel_2025.csv | Rows: 1808 | Baseline a_floor: 0.55

## Sweep Results

| a_floor | A%(elig) | B%(elig) | C%(elig) | AB med 60d | CD med 60d | Sep | AB DD | Overlap% | Score | Pass |
|---------|----------|----------|----------|------------|------------|-----|-------|----------|-------|------|
| **0.40** | 7.9 | 50.8 | 41.4 | +14.62 | +14.24 | +0.38 | -22.60 | 84.4 | -6.40 | Y |
| 0.45 | 7.7 | 47.9 | 44.4 | +13.85 | +14.42 | -0.57 | -22.93 | 85.9 | -7.45 | N |
| 0.50 | 6.3 | 42.9 | 50.8 | +14.11 | +14.42 | -0.31 | -22.37 | 83.1 | -7.02 | N |
| 0.52 | 5.9 | 41.4 | 52.7 | +13.85 | +14.42 | -0.57 | -22.64 | 82.3 | -7.36 | N |
| 0.55 | 5.7 | 39.8 | 54.5 | +13.85 | +14.42 | -0.57 | -23.03 | 84.3 | -7.48 | N |
| 0.58 | 5.0 | 38.5 | 56.5 | +14.44 | +14.41 | +0.02 | -23.29 | 83.8 | -6.97 | Y |
| 0.60 | 4.8 | 37.9 | 57.3 | +14.83 | +14.35 | +0.49 | -23.29 | 85.2 | -6.50 | Y |
| 0.65 | 4.2 | 33.7 | 62.1 | +14.83 | +14.35 | +0.49 | -23.41 | 83.6 | -6.53 | Y |

## Recommended Candidate

**`tier_a_optionality_floor = 0.40`**

### Baseline vs Candidate

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| AB median 60d ret % | 13.85 | 14.62 | +0.76 |
| CD median 60d ret % | 14.42 | 14.24 | -0.19 |
| Separation (AB-CD) % | -0.57 | 0.38 | +0.95 |
| AB median max-DD % | -23.03 | -22.60 | +0.43 |
| Mean A-count/date | 2.60 | 3.60 | +1.00 |
| Top-25 overlap % | 84.30 | 84.40 | +0.10 |
| Mean turnover % | 15.70 | 15.60 | -0.10 |

### Why Chosen

- Objective score: -6.4008 (best among 4 passing candidates)
- All constraints satisfied

