# 2024 Inversion Attribution Report

**Date**: 2026-02-12
**Panel**: walkforward_panel__baseline_full.csv
**Rows**: 3922 (22 snapshots)

## Executive Summary

- 2024 AB-CD separation: **+2.11pp** (inverted)
- 2025 AB-CD separation: **+18.66pp** (healthy)

## Tier Performance by Year

### 2024

| Tier | N | Mean 60d | Median | Hit% | P25 | P75 |
|------|---|----------|--------|------|-----|-----|
| A | 50 | +6.02% | +0.96% | 50% | -16.70% | +25.88% |
| B | 231 | -5.35% | -8.88% | 35% | -22.85% | +7.56% |
| C | 340 | -5.44% | -7.54% | 37% | -23.04% | +9.40% |
| D | 1405 | +5.60% | -9.75% | 38% | -28.53% | +13.42% |

### 2025

| Tier | N | Mean 60d | Median | Hit% | P25 | P75 |
|------|---|----------|--------|------|-----|-----|
| A | 38 | +18.17% | +12.61% | 60% | -6.03% | +35.47% |
| B | 122 | +36.28% | +17.17% | 71% | -0.87% | +59.29% |
| C | 176 | +13.32% | +8.57% | 61% | -6.64% | +32.43% |
| D | 1457 | +29.16% | +14.99% | 66% | -8.95% | +47.07% |

## Top 3 Drivers of 2024 Inversion

Ranked by spread range (max AB-D spread minus min AB-D spread across
dimension values). Wider range = dimension explains more variance in
the AB vs D performance gap.

### #1: `optionality_bucket`

- Spread range (2024): **+101.15pp** (2025: +66.31pp)
- Worst cell: `b_floor_0.55-0.60` → AB-D = -100.21pp
- Best cell: `high_>0.75` → AB-D = +0.94pp

### #2: `mom_state`

- Spread range (2024): **+40.12pp** (2025: +19.75pp)
- Worst cell: `neutral` → AB-D = -40.20pp
- Best cell: `headwind` → AB-D = -0.08pp

### #3: `catalyst_strength`

- Spread range (2024): **+28.97pp** (2025: +11.67pp)
- Worst cell: `far` → AB-D = -24.57pp
- Best cell: `mid` → AB-D = +4.40pp

---

## Detailed Dimension Breakouts

### band

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| XS | 4 | -40.75% | 0% | 1405 | +5.60% | 38% | **-46.35pp** |
| L | 207 | -2.73% | 36% | 0 | N/A | N/A | N/A |
| M | 51 | -2.97% | 41% | 0 | N/A | N/A | N/A |
| S | 19 | -2.93% | 53% | 0 | N/A | N/A | N/A |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| L | 124 | +36.45% | 71% | 0 | N/A | N/A | N/A |
| M | 31 | +12.97% | 61% | 0 | N/A | N/A | N/A |
| S | 5 | +38.90% | 60% | 0 | N/A | N/A | N/A |
| XS | 0 | N/A | N/A | 1457 | +29.16% | 66% | N/A |

### catalyst_mode

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| specific_days | 139 | -0.67% | 38% | 412 | +9.33% | 40% | **-10.00pp** |
| missing | 142 | -5.93% | 37% | 993 | +4.06% | 37% | **-9.99pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| specific_days | 112 | +33.36% | 70% | 553 | +34.08% | 66% | **-0.72pp** |
| missing | 48 | +28.74% | 67% | 904 | +26.14% | 65% | **+2.60pp** |

### catalyst_strength

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| far | 68 | -7.14% | 26% | 181 | +17.43% | 40% | **-24.57pp** |
| missing | 142 | -5.93% | 37% | 993 | +4.06% | 37% | **-9.99pp** |
| near | 41 | +2.37% | 54% | 121 | +0.76% | 40% | **+1.61pp** |
| mid | 30 | +9.84% | 43% | 110 | +5.44% | 39% | **+4.40pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| near | 33 | +35.19% | 79% | 181 | +38.87% | 62% | **-3.68pp** |
| far | 41 | +32.84% | 66% | 229 | +36.38% | 70% | **-3.54pp** |
| missing | 48 | +28.74% | 67% | 904 | +26.14% | 65% | **+2.60pp** |
| mid | 38 | +32.33% | 66% | 143 | +24.34% | 66% | **+7.99pp** |

### mom_state

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| neutral | 139 | -1.03% | 39% | 286 | +39.17% | 44% | **-40.20pp** |
| tailwind | 88 | -6.94% | 33% | 478 | -2.57% | 37% | **-4.37pp** |
| headwind | 54 | -3.36% | 41% | 641 | -3.28% | 36% | **-0.08pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| headwind | 27 | +18.82% | 67% | 668 | +28.58% | 63% | **-9.76pp** |
| neutral | 92 | +30.61% | 65% | 284 | +22.43% | 64% | **+8.18pp** |
| tailwind | 41 | +43.69% | 78% | 505 | +33.70% | 69% | **+9.99pp** |

### adv_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| mid_5-25M | 89 | -2.17% | 39% | 498 | +15.96% | 35% | **-18.13pp** |
| large_>25M | 131 | -4.12% | 34% | 433 | +7.60% | 48% | **-11.72pp** |
| micro_<1M | 11 | -15.50% | 9% | 36 | -14.35% | 22% | **-1.15pp** |
| small_1-5M | 50 | -0.64% | 48% | 438 | -6.50% | 32% | **+5.86pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| small_1-5M | 24 | +4.57% | 38% | 412 | +19.84% | 57% | **-15.27pp** |
| large_>25M | 86 | +30.85% | 76% | 465 | +39.37% | 70% | **-8.52pp** |
| mid_5-25M | 44 | +37.72% | 70% | 539 | +28.25% | 68% | **+9.47pp** |
| micro_<1M | 6 | +115.64% | 83% | 41 | +18.88% | 63% | **+96.76pp** |

### cost_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| low_<400 | 111 | -8.50% | 29% | 1405 | +5.60% | 38% | **-14.10pp** |
| mid_400-1k | 170 | +0.05% | 43% | 0 | N/A | N/A | N/A |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| mid_400-1k | 124 | +28.61% | 64% | 0 | N/A | N/A | N/A |
| low_<400 | 36 | +43.57% | 83% | 1457 | +29.16% | 66% | **+14.41pp** |

### dd_abs_margin_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| borderline_-5to0pp | 0 | N/A | N/A | 165 | -13.79% | 25% | N/A |
| comfortable_>10pp | 123 | -6.63% | 31% | 0 | N/A | N/A | N/A |
| deep_<-20pp | 0 | N/A | N/A | 686 | +7.64% | 40% | N/A |
| missing | 8 | -26.80% | 12% | 0 | N/A | N/A | N/A |
| near_-10to-5pp | 0 | N/A | N/A | 195 | +16.77% | 42% | N/A |
| safe_0to10pp | 150 | +0.63% | 44% | 0 | N/A | N/A | N/A |
| stressed_-20to-10pp | 0 | N/A | N/A | 359 | +4.56% | 38% | N/A |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| comfortable_>10pp | 85 | +39.29% | 69% | 0 | N/A | N/A | N/A |
| deep_<-20pp | 0 | N/A | N/A | 979 | +35.59% | 67% | N/A |
| missing | 0 | N/A | N/A | 0 | N/A | N/A | N/A |
| near_-10to-5pp | 0 | N/A | N/A | 121 | +10.88% | 58% | N/A |
| safe_0to10pp | 68 | +24.15% | 66% | 0 | N/A | N/A | N/A |
| stressed_-20to-10pp | 0 | N/A | N/A | 269 | +18.71% | 68% | N/A |
| borderline_-5to0pp | 7 | +19.08% | 86% | 88 | +14.64% | 49% | **+4.44pp** |

### dd_rel_margin_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| stressed_-15to-5pp | 100 | -6.17% | 36% | 98 | -3.50% | 46% | **-2.67pp** |
| deep_<-15pp | 10 | +4.71% | 50% | 1299 | +6.34% | 37% | **-1.63pp** |
| comfortable_>10pp | 52 | -0.42% | 40% | 0 | N/A | N/A | N/A |
| missing | 8 | -26.80% | 12% | 0 | N/A | N/A | N/A |
| safe_0to10pp | 72 | -5.85% | 35% | 0 | N/A | N/A | N/A |
| borderline_-5to0pp | 39 | +7.48% | 44% | 8 | -1.63% | 25% | **+9.11pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| borderline_-5to0pp | 23 | +9.50% | 56% | 38 | +33.48% | 63% | **-23.98pp** |
| deep_<-15pp | 5 | +30.41% | 100% | 1212 | +31.30% | 66% | **-0.89pp** |
| comfortable_>10pp | 71 | +41.50% | 72% | 0 | N/A | N/A | N/A |
| missing | 0 | N/A | N/A | 0 | N/A | N/A | N/A |
| safe_0to10pp | 49 | +32.34% | 69% | 0 | N/A | N/A | N/A |
| stressed_-15to-5pp | 12 | +17.83% | 58% | 207 | +15.80% | 65% | **+2.03pp** |

### optionality_bucket

**2024:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| b_floor_0.55-0.60 | 5 | -8.05% | 40% | 72 | +92.16% | 35% | **-100.21pp** |
| mid_0.30-0.55 | 16 | +8.24% | 50% | 335 | +13.06% | 38% | **-4.82pp** |
| a_zone_0.60-0.75 | 75 | -2.01% | 40% | 233 | +2.23% | 41% | **-4.24pp** |
| low_<0.30 | 0 | N/A | N/A | 431 | -4.08% | 35% | N/A |
| high_>0.75 | 185 | -4.74% | 35% | 334 | -5.68% | 39% | **+0.94pp** |

**2025:**

| Value | N(AB) | AB mean | AB hit | N(D) | D mean | D hit | AB-D spread |
|-------|-------|---------|--------|------|--------|-------|-------------|
| b_floor_0.55-0.60 | 7 | +9.48% | 71% | 72 | +41.14% | 72% | **-31.66pp** |
| a_zone_0.60-0.75 | 41 | +23.83% | 56% | 229 | +31.36% | 66% | **-7.53pp** |
| high_>0.75 | 86 | +28.37% | 69% | 365 | +31.85% | 67% | **-3.48pp** |
| low_<0.30 | 0 | N/A | N/A | 436 | +24.58% | 65% | N/A |
| mid_0.30-0.55 | 26 | +62.81% | 88% | 355 | +28.16% | 63% | **+34.65pp** |

---

## D-Tier Outperformers Profile (2024)

D-tier top quartile (P75+): n=351, mean=+83.79%
D-tier bottom quartile (P25-): n=351, mean=-43.26%

**Size band:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| XS | 100.0% | 100.0% | +0.0pp |

**Catalyst mode:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| missing | 68.4% | 75.2% | -6.8pp |
| specific_days | 31.6% | 24.8% | +6.8pp |

**Momentum state:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| headwind | 41.6% | 48.1% | -6.5pp |
| neutral | 25.9% | 17.4% | +8.5pp |
| tailwind | 32.5% | 34.5% | -2.0pp |

**ADV bucket:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| large_>25M | 38.7% | 18.5% | +20.2pp |
| micro_<1M | 1.4% | 2.0% | -0.6pp |
| mid_5-25M | 33.3% | 37.6% | -4.3pp |
| small_1-5M | 26.5% | 41.9% | -15.4pp |

**DD abs margin:**
| Value | Top Q (winners) | Bottom Q (losers) | Delta |
|-------|-----------------|-------------------|-------|
| borderline_-5to0pp | 8.0% | 13.4% | -5.4pp |
| deep_<-20pp | 53.8% | 54.4% | -0.6pp |
| near_-10to-5pp | 14.8% | 9.4% | +5.4pp |
| stressed_-20to-10pp | 23.4% | 22.8% | +0.6pp |

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
