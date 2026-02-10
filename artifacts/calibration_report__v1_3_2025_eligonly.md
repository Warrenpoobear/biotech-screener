# Ruleset Calibration Report

Generated: 2026-02-10T11:36:48
Panel: artifacts/walkforward_panel_v1_3_full.csv | Rows: 1808 | Baseline a_floor: 0.55

## Sweep Results

| a_floor | A%(elig) | B%(elig) | C%(elig) | AB med 60d | CD med 60d | Sep | AB DD | Overlap% | Score | Pass |
|---------|----------|----------|----------|------------|------------|-----|-------|----------|-------|------|
| 0.40 | 19.7 | 35.6 | 44.6 | +9.65 | +7.73 | +1.92 | -19.83 | 77.4 | -4.03 | Y |
| 0.45 | 18.9 | 33.9 | 47.2 | +8.22 | +9.40 | -1.18 | -21.44 | 77.6 | -7.61 | N |
| 0.50 | 15.5 | 33.0 | 51.5 | +8.70 | +8.98 | -0.29 | -19.83 | 71.8 | -6.24 | N |
| 0.52 | 13.3 | 33.9 | 52.8 | +9.01 | +8.76 | +0.25 | -19.82 | 70.2 | -5.70 | Y |
| 0.55 | 12.9 | 31.8 | 55.4 | +9.41 | +7.98 | +1.43 | -20.63 | 71.4 | -4.76 | Y |
| 0.58 | 10.3 | 31.8 | 57.9 | +9.88 | +7.47 | +2.41 | -20.63 | 70.5 | -3.78 | Y |
| **0.60** | 9.9 | 30.5 | 59.7 | +10.00 | +7.47 | +2.53 | -19.83 | 69.5 | -3.42 | Y |
| 0.65 | 8.6 | 25.8 | 65.7 | +9.88 | +8.38 | +1.50 | -21.43 | 67.6 | -4.93 | Y |

## Recommended Candidate

**`tier_a_optionality_floor = 0.60`**

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

- Objective score: -3.4183 (best among 6 passing candidates)
- All constraints satisfied

