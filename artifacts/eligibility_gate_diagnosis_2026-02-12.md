# Eligibility Gate Diagnosis — 2026-02-12

Panel: walkforward_panel_diag.csv
Rows: 3922
Date range: 2024-01-31 to 2025-10-31
Ruleset params: a_floor=0.6, b_floor=0.3, catalyst_near=120d, catalyst_mid=180d

---
# Section 1: Tier Performance by Year

## 2024  (2114 rows)

Tier       N     Mean      Med      P25      P75    Hit%
----------------------------------------------------
A         50   +6.02%   +0.96%  -16.71%  +26.22%  50.00%
B        231   -5.35%   -8.88%  -23.18%   +7.90%  34.60%
C        340   -5.44%   -7.54%  -22.97%   +9.42%  36.80%
D       1405   +5.60%   -9.75%  -28.53%  +13.42%  37.70%

AB-CD separation (median): +1.02%
AB-CD separation (mean):   -6.78%

## 2025  (1808 rows)

Tier       N     Mean      Med      P25      P75    Hit%
----------------------------------------------------
A         38  +18.17%  +12.61%   -6.09%  +35.79%  60.50%
B        122  +36.28%  +17.17%   -0.95%  +59.53%  71.30%
C        176  +13.32%   +8.57%   -6.64%  +32.98%  60.80%
D       1457  +29.16%  +14.99%   -8.95%  +47.07%  65.50%

AB-CD separation (median): +1.34%
AB-CD separation (mean):   +4.53%


# Section 2: D-Tier Attribution (Ineligible Reasons)

Total D-tier rows: 2864

## 2024 — D-Tier (1405 rows)

### By ineligible_reasons
Reason                                  N     Mean      Med      P25      P75    Hit%
--------------------------------------------------------------------------------
deep_drawdown                        1405   +5.60%   -9.75%  -28.53%  +13.42%  37.70%

### By first_failed_gate
Gate                                    N     Mean      Med      P25      P75    Hit%
--------------------------------------------------------------------------------
deep_drawdown                        1405   +5.60%   -9.75%  -28.53%  +13.42%  37.70%

## 2025 — D-Tier (1459 rows)

### By ineligible_reasons
Reason                                  N     Mean      Med      P25      P75    Hit%
--------------------------------------------------------------------------------
deep_drawdown                        1457  +29.16%  +14.99%   -8.95%  +47.07%  65.50%

### By first_failed_gate
Gate                                    N     Mean      Med      P25      P75    Hit%
--------------------------------------------------------------------------------
deep_drawdown                        1457  +29.16%  +14.99%   -8.95%  +47.07%  65.50%


# Section 3: Surgical Ablations

Baseline tier distribution: {'C': 591, 'D': 2864, 'B': 376, 'A': 91}
Baseline AB-CD sep (mean): -5.57%
Baseline AB-CD sep (med):  -1.66%

## 2024  (2114 rows)

Baseline: {'C': 405, 'D': 1405, 'B': 252, 'A': 52}  sep_mean=-6.78%  sep_med=+1.02%

Ablation                       Flipped    A    B    C    D  Sep(mean)   Sep(med)    Δmean
------------------------------------------------------------------------------------------------
disable_all_gates                 1405  159  792 1163    0     -7.37%     +1.19%   -0.59%
disable_drawdown_only             1405  159  792 1163    0     -7.37%     +1.19%   -0.59%
disable_deep_drawdown             1405  159  792 1163    0     -7.37%     +1.19%   -0.59%
disable_sev3                         0   52  252  405 1405     -6.78%     +1.02%   +0.00%
disable_fundamental_red_flag         0   52  252  405 1405     -6.78%     +1.02%   +0.00%
disable_adv_fail                     0   52  252  405 1405     -6.78%     +1.02%   +0.00%

## 2025  (1808 rows)

Baseline: {'C': 186, 'D': 1459, 'B': 124, 'A': 39}  sep_mean=+4.53%  sep_med=+1.34%

Ablation                       Flipped    A    B    C    D  Sep(mean)   Sep(med)    Δmean
------------------------------------------------------------------------------------------------
disable_all_gates                 1459  210  622  976    0    +10.83%     +1.76%   +6.30%
disable_drawdown_only             1459  210  622  976    0    +10.83%     +1.76%   +6.30%
disable_deep_drawdown             1459  210  622  976    0    +10.83%     +1.76%   +6.30%
disable_sev3                         0   39  124  186 1459     +4.53%     +1.34%   +0.00%
disable_fundamental_red_flag         0   39  124  186 1459     +4.53%     +1.34%   +0.00%
disable_adv_fail                     0   39  124  186 1459     +4.53%     +1.34%   +0.00%


# Section 4: Drawdown Margin Analysis (D-Tier)

Total D-tier rows: 2864

## 2024 — D-Tier Drawdown Margins (1405 rows)

Rows with deep_drawdown reason: 1405

dd_abs_margin distribution (N=1405):
  Mean: -21.4pp  Median: -19.6pp
  P25: -31.5pp  P75: -9.9pp
  Barely ineligible (margin > -5pp): 165 / 1405 (11.7%)
  Very close (margin > -2pp): 69 / 1405 (4.9%)
  Histogram:
       0-2pp:    69 (4.9%)
       2-5pp:    96 (6.8%)
      5-10pp:   195 (13.9%)
     10-20pp:   359 (25.6%)
     20-50pp:   641 (45.6%)
       >50pp:    45 (3.2%)

dd_rel_margin distribution (N=1405):
  Mean: -33.6pp  Median: -31.7pp
  Barely ineligible (margin > -5pp): 8 / 1405 (0.6%)

Performance split (deep_drawdown rows only):
  Near gate  (margin > -10pp): N=360, mean=+2.76%, med=-10.42%, hit=33.90
  Far below  (margin <= -10pp): N=1045, mean=+6.58%, med=-9.51%, hit=39.00

## 2025 — D-Tier Drawdown Margins (1459 rows)

Rows with deep_drawdown reason: 1459

dd_abs_margin distribution (N=1459):
  Mean: -27.1pp  Median: -27.3pp
  P25: -37.7pp  P75: -15.0pp
  Barely ineligible (margin > -5pp): 89 / 1459 (6.1%)
  Very close (margin > -2pp): 27 / 1459 (1.9%)
  Histogram:
       0-2pp:    27 (1.9%)
       2-5pp:    62 (4.2%)
      5-10pp:   121 (8.3%)
     10-20pp:   269 (18.4%)
     20-50pp:   906 (62.1%)
       >50pp:    74 (5.1%)

dd_rel_margin distribution (N=1459):
  Mean: -32.1pp  Median: -32.1pp
  Barely ineligible (margin > -5pp): 38 / 1459 (2.6%)

Performance split (deep_drawdown rows only):
  Near gate  (margin > -10pp): N=209, mean=+12.46%, med=+3.64%, hit=54.10
  Far below  (margin <= -10pp): N=1248, mean=+31.95%, med=+17.11%, hit=67.40


# Section 5: D-Tier Reversal Filter (trail_ret_20d)

D-tier rows with trail_ret_20d: 2863 / 2864

## 2024 — D-Tier Reversal Split (1404 rows with trail_ret_20d)

Group                         N     Mean      Med      P25      P75    Hit%
----------------------------------------------------------------------
trail_ret_20d > 0           624   +9.33%   -9.68%  -27.35%  +13.44%  37.50%
trail_ret_20d <= 0          780   +2.66%   -9.80%  -29.34%  +13.42%  37.90%

Delta (up - down) mean: +6.67%  median: +0.12%

### trail_ret_20d quintiles
Quintile                    Range     N  Mean_60d   Med_60d    Hit%
----------------------------------------------------------------------
Q1              -65.4% to  -17.3%   280    -3.54%    -9.04%  37.90%
Q2              -17.3% to   -7.6%   280   +14.70%   -10.39%  38.60%
Q3               -7.5% to   +2.2%   280    -4.32%    -8.44%  37.50%
Q4               +2.3% to  +17.5%   280   +22.92%    -9.02%  36.80%
Q5              +17.5% to +465.2%   284    -1.55%   -10.70%  38.00%

## 2025 — D-Tier Reversal Split (1459 rows with trail_ret_20d)

Group                         N     Mean      Med      P25      P75    Hit%
----------------------------------------------------------------------
trail_ret_20d > 0           764  +33.36%  +18.67%   -5.56%  +51.55%  69.20%
trail_ret_20d <= 0          693  +24.52%  +10.08%  -11.71%  +38.69%  61.30%

Delta (up - down) mean: +8.84%  median: +8.59%

### trail_ret_20d quintiles
Quintile                    Range     N  Mean_60d   Med_60d    Hit%
----------------------------------------------------------------------
Q1              -87.8% to  -14.6%   291   +21.20%    +8.18%  58.10%
Q2              -14.6% to   -3.7%   291   +23.80%    +9.32%  62.20%
Q3               -3.7% to   +6.1%   290   +32.02%   +18.60%  66.60%
Q4               +6.1% to  +20.4%   290   +35.70%   +20.78%  74.50%
Q5              +20.4% to +824.9%   295   +33.05%   +16.75%  66.10%
