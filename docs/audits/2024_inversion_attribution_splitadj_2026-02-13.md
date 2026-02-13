# 2024 Inversion Attribution Report

**Date**: 2026-02-13
**Panel**: walkforward_panel__baseline_full.csv
**Rows**: 3922 (22 snapshots)

## Executive Summary

- 2024 AB-CD separation: **+2.38pp** (inverted)
- 2025 AB-CD separation: **+21.47pp** (healthy)

## Tier Performance by Year

### 2024

| Tier | N | Mean 60d | Median | Hit% | P25 | P75 |
|------|---|----------|--------|------|-----|-----|
| A | 51 | +5.87% | +0.00% | 49% | -16.70% | +25.54% |
| B | 236 | -5.11% | -8.83% | 35% | -22.68% | +7.90% |
| C | 342 | -5.54% | -7.62% | 36% | -23.18% | +9.26% |
| D | 1397 | +5.65% | -9.75% | 38% | -28.84% | +13.42% |

### 2025

| Tier | N | Mean 60d | Median | Hit% | P25 | P75 |
|------|---|----------|--------|------|-----|-----|
| A | 42 | +18.40% | +12.61% | 60% | -6.03% | +35.47% |
| B | 141 | +39.16% | +18.01% | 72% | -0.63% | +58.58% |
| C | 192 | +12.92% | +8.57% | 60% | -6.70% | +31.38% |
| D | 1418 | +29.03% | +14.88% | 66% | -9.47% | +47.12% |

## Top 3 Drivers of 2024 Inversion

Ranked by spread range (max AB-D spread minus min AB-D spread across
dimension values). Wider range = dimension explains more variance in
the AB vs D performance gap.

### #1: `optionality_bucket`

- Spread range (2024): **+101.78pp** (2025: +44.68pp)
- Worst cell: `b_floor_0.55-0.60` → AB-D = -100.21pp
- Best cell: `high_>0.75` → AB-D = +1.57pp

### #2: `mom_state`

- Spread range (2024): **+40.88pp** (2025: +9.66pp)
- Worst cell: `neutral` → AB-D = -40.46pp
- Best cell: `headwind` → AB-D = +0.42pp

### #3: `catalyst_strength`

- Spread range (2024): **+31.22pp** (2025: +28.87pp)
- Worst cell: `far` → AB-D = -24.82pp
- Best cell: `near` → AB-D = +6.40pp

---

## Detailed Dimension Breakouts

### band

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| XS | 4 | -40.75% | 0% | 1397 | +5.65% | 38% | **-46.40pp** |
| L | 210 | -2.63% | 36% | 0 | N/A | N/A | N/A |
| M | 53 | -3.42% | 40% | 0 | N/A | N/A | N/A |
| S | 20 | -0.48% | 55% | 0 | N/A | N/A | N/A |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| L | 140 | +36.97% | 71% | 0 | N/A | N/A | N/A |
| M | 38 | +24.34% | 66% | 0 | N/A | N/A | N/A |
| S | 5 | +38.90% | 60% | 0 | N/A | N/A | N/A |
| XS | 0 | N/A | N/A | 1418 | +29.03% | 66% | N/A |

### catalyst_mode

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| specific_days | 142 | -0.79% | 37% | 408 | +9.52% | 40% | **-10.31pp** |
| missing | 145 | -5.47% | 37% | 989 | +4.05% | 37% | **-9.52pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| specific_days | 126 | +34.64% | 69% | 532 | +34.26% | 67% | **+0.38pp** |
| missing | 57 | +33.86% | 70% | 886 | +25.89% | 65% | **+7.97pp** |

### catalyst_strength

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| far | 69 | -7.22% | 26% | 180 | +17.60% | 41% | **-24.82pp** |
| missing | 145 | -5.47% | 37% | 989 | +4.05% | 37% | **-9.52pp** |
| mid | 19 | +5.25% | 37% | 70 | +12.73% | 47% | **-7.48pp** |
| near | 54 | +5.29% | 52% | 158 | -1.11% | 37% | **+6.40pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| far | 44 | +31.60% | 66% | 220 | +37.83% | 71% | **-6.23pp** |
| near | 49 | +31.96% | 74% | 226 | +36.21% | 65% | **-4.25pp** |
| missing | 57 | +33.86% | 70% | 886 | +25.89% | 65% | **+7.97pp** |
| mid | 33 | +42.65% | 67% | 86 | +20.01% | 63% | **+22.64pp** |

### mom_state

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| neutral | 140 | -1.11% | 39% | 285 | +39.35% | 44% | **-40.46pp** |
| tailwind | 90 | -6.50% | 33% | 474 | -2.55% | 37% | **-3.95pp** |
| headwind | 57 | -2.90% | 40% | 638 | -3.32% | 36% | **+0.42pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| headwind | 34 | +30.34% | 71% | 660 | +28.16% | 63% | **+2.18pp** |
| tailwind | 46 | +40.32% | 74% | 494 | +34.35% | 70% | **+5.97pp** |
| neutral | 103 | +33.09% | 67% | 264 | +21.25% | 64% | **+11.84pp** |

### adv_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| mid_5-25M | 92 | -2.44% | 38% | 493 | +16.27% | 36% | **-18.71pp** |
| large_>25M | 133 | -3.80% | 35% | 431 | +7.56% | 48% | **-11.36pp** |
| micro_<1M | 11 | -15.50% | 9% | 36 | -14.35% | 22% | **-1.15pp** |
| small_1-5M | 51 | -0.09% | 49% | 437 | -6.58% | 32% | **+6.49pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| small_1-5M | 26 | +4.73% | 38% | 408 | +19.89% | 57% | **-15.16pp** |
| large_>25M | 102 | +35.74% | 76% | 444 | +38.90% | 71% | **-3.16pp** |
| mid_5-25M | 47 | +37.01% | 70% | 527 | +28.65% | 68% | **+8.36pp** |
| micro_<1M | 8 | +98.23% | 88% | 39 | +17.49% | 62% | **+80.74pp** |

### cost_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| low_<400 | 99 | -8.74% | 29% | 1397 | +5.65% | 38% | **-14.39pp** |
| extreme_>2k | 7 | -7.29% | 29% | 0 | N/A | N/A | N/A |
| high_1k-2k | 64 | -0.72% | 42% | 0 | N/A | N/A | N/A |
| mid_400-1k | 117 | +0.48% | 42% | 0 | N/A | N/A | N/A |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| extreme_>2k | 5 | +100.19% | 100% | 0 | N/A | N/A | N/A |
| high_1k-2k | 40 | +16.07% | 45% | 0 | N/A | N/A | N/A |
| mid_400-1k | 97 | +29.54% | 74% | 0 | N/A | N/A | N/A |
| low_<400 | 41 | +55.74% | 78% | 1418 | +29.03% | 66% | **+26.71pp** |

### dd_abs_margin_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| comfortable_>10pp | 123 | -6.63% | 31% | 0 | N/A | N/A | N/A |
| deep_<-20pp | 0 | N/A | N/A | 686 | +7.64% | 40% | N/A |
| missing | 8 | -26.80% | 12% | 0 | N/A | N/A | N/A |
| near_-10to-5pp | 0 | N/A | N/A | 195 | +16.77% | 42% | N/A |
| safe_0to10pp | 150 | +0.63% | 44% | 0 | N/A | N/A | N/A |
| stressed_-20to-10pp | 0 | N/A | N/A | 359 | +4.56% | 38% | N/A |
| borderline_-5to0pp | 6 | +4.99% | 33% | 157 | -14.41% | 25% | **+19.40pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| comfortable_>10pp | 85 | +39.29% | 69% | 0 | N/A | N/A | N/A |
| deep_<-20pp | 0 | N/A | N/A | 978 | +35.58% | 67% | N/A |
| missing | 0 | N/A | N/A | 0 | N/A | N/A | N/A |
| safe_0to10pp | 68 | +24.15% | 66% | 0 | N/A | N/A | N/A |
| stressed_-20to-10pp | 0 | N/A | N/A | 269 | +18.71% | 68% | N/A |
| near_-10to-5pp | 6 | +13.76% | 50% | 110 | +10.45% | 58% | **+3.31pp** |
| borderline_-5to0pp | 24 | +51.23% | 83% | 61 | +3.00% | 41% | **+48.23pp** |

### dd_rel_margin_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| borderline_-5to0pp | 68 | -9.15% | 29% | 30 | -2.78% | 47% | **-6.37pp** |
| comfortable_>10pp | 78 | -3.44% | 35% | 0 | N/A | N/A | N/A |
| deep_<-15pp | 0 | N/A | N/A | 1136 | +8.24% | 37% | N/A |
| missing | 8 | -26.80% | 12% | 0 | N/A | N/A | N/A |
| safe_0to10pp | 91 | +1.61% | 42% | 0 | N/A | N/A | N/A |
| stressed_-15to-5pp | 42 | +1.24% | 50% | 231 | -6.03% | 38% | **+7.27pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| borderline_-5to0pp | 5 | -4.71% | 20% | 91 | +21.21% | 69% | **-25.92pp** |
| comfortable_>10pp | 87 | +37.12% | 69% | 0 | N/A | N/A | N/A |
| deep_<-15pp | 0 | N/A | N/A | 1109 | +32.84% | 66% | N/A |
| missing | 0 | N/A | N/A | 0 | N/A | N/A | N/A |
| safe_0to10pp | 79 | +34.16% | 70% | 0 | N/A | N/A | N/A |
| stressed_-15to-5pp | 12 | +32.46% | 92% | 218 | +12.93% | 60% | **+19.53pp** |

### optionality_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| b_floor_0.55-0.60 | 5 | -8.05% | 40% | 72 | +92.16% | 35% | **-100.21pp** |
| mid_0.30-0.55 | 17 | +7.40% | 47% | 334 | +13.12% | 38% | **-5.72pp** |
| a_zone_0.60-0.75 | 77 | -2.28% | 39% | 231 | +2.35% | 41% | **-4.63pp** |
| low_<0.30 | 0 | N/A | N/A | 429 | -4.00% | 35% | N/A |
| high_>0.75 | 188 | -4.34% | 36% | 331 | -5.91% | 39% | **+1.57pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| b_floor_0.55-0.60 | 9 | +29.19% | 67% | 70 | +39.51% | 73% | **-10.32pp** |
| a_zone_0.60-0.75 | 46 | +22.59% | 56% | 224 | +31.78% | 66% | **-9.19pp** |
| low_<0.30 | 0 | N/A | N/A | 426 | +24.83% | 65% | N/A |
| high_>0.75 | 97 | +31.49% | 70% | 354 | +31.10% | 67% | **+0.39pp** |
| mid_0.30-0.55 | 31 | +62.53% | 87% | 344 | +28.17% | 63% | **+34.36pp** |

---

## D-Tier Outperformers Profile (2024)

D-tier top quartile (P75+): n=349, mean=+84.06%
D-tier bottom quartile (P25-): n=349, mean=-43.34%

**Size band:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| XS | 100.0% | 100.0% | +0.0pp |

**Catalyst mode:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| missing | 68.2% | 75.1% | -6.9pp |
| specific_days | 31.8% | 24.9% | +6.9pp |

**Momentum state:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| headwind | 41.5% | 48.4% | -6.9pp |
| neutral | 26.1% | 17.2% | +8.9pp |
| tailwind | 32.4% | 34.4% | -2.0pp |

**ADV bucket:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| large_>25M | 38.7% | 18.3% | +20.4pp |
| micro_<1M | 1.4% | 2.0% | -0.6pp |
| mid_5-25M | 33.5% | 37.8% | -4.3pp |
| small_1-5M | 26.4% | 41.8% | -15.4pp |

**DD abs margin:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| borderline_-5to0pp | 7.4% | 13.5% | -6.1pp |
| deep_<-20pp | 54.2% | 54.4% | -0.2pp |
| near_-10to-5pp | 14.9% | 9.2% | +5.7pp |
| stressed_-20to-10pp | 23.5% | 22.9% | +0.6pp |

---

## What to Test Next

Based on the attribution above, the recommended next step is a
**membership-preserving weight modifier** targeting the consistently
bad cell(s) identified in the top drivers. This stays in L3 (sizing)
and does not change eligibility or tier assignments.

Candidate modifiers to evaluate via walkforward:

1. **optionality_bucket penalty**: Weight *= penalty_mult for rows in the
   `b_floor_0.55-0.60` bucket (worst AB-D spread: -100.21pp).
   Test with mult = 0.70 and 0.85 via walkforward panel replay.
2. **mom_state penalty**: Same approach for `neutral` bucket.
3. **abs_shallow + rel_deep composite**: Weight penalty for rows near
   abs gate but crushed vs XBI (sector-wide drawdown + idiosyncratic weakness).
