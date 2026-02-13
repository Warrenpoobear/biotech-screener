# Forward Return Integrity Audit

Scan date: 2026-02-13

## 1. Price Discontinuities Detected

Found **30** discontinuities across **26** tickers.

| Ticker | Date | Close Before | Close After | Pct Change | Type |
|--------|------|-------------|------------|------------|------|
| DRUG | 2024-10-15 | 2.49 | 38.49 | +1445.8% | reverse_split |
| TECX | 2024-06-24 | 1.40 | 16.10 | +1050.0% | reverse_split |
| ABVX | 2025-07-23 | 10.00 | 68.60 | +586.0% | reverse_split |
| INDV | 2022-11-28 | 3.13 | 21.03 | +571.9% | reverse_split |
| COGT | 2020-07-06 | 1.78 | 8.80 | +394.4% | reverse_split |
| MCRB | 2020-08-10 | 92.80 | 454.00 | +389.2% | reverse_split |
| ORKA | 2020-05-28 | 25.49 | 123.98 | +386.3% | reverse_split |
| SRRK | 2024-10-07 | 7.42 | 34.28 | +362.0% | reverse_split |
| PHVS | 2022-12-08 | 2.51 | 11.46 | +356.6% | reverse_split |
| SYRE | 2023-06-22 | 2.65 | 11.40 | +330.2% | reverse_split |
| LYRA | 2025-06-02 | 4.93 | 20.25 | +310.8% | reverse_split |
| PRQR | 2022-02-11 | 5.64 | 1.39 | -75.3% | forward_split |
| BIOA | 2024-12-09 | 20.09 | 4.65 | -76.8% | forward_split |
| REPL | 2025-07-22 | 12.32 | 2.81 | -77.2% | forward_split |
| PRAX | 2022-06-06 | 128.85 | 28.20 | -78.1% | forward_split |
| IVVD | 2021-12-14 | 34.26 | 7.26 | -78.8% | forward_split |
| KOD | 2022-02-23 | 50.35 | 9.86 | -80.4% | forward_split |
| VTYX | 2023-11-07 | 14.09 | 2.73 | -80.6% | forward_split |
| MENS | 2025-12-17 | 15.40 | 2.92 | -81.0% | forward_split |
| NMRA | 2025-01-02 | 10.60 | 1.97 | -81.4% | forward_split |
| AMLX | 2024-03-08 | 18.97 | 3.36 | -82.3% | forward_split |
| FGEN | 2023-06-26 | 401.25 | 67.75 | -83.1% | forward_split |
| INDV | 2022-11-23 | 19.64 | 3.13 | -84.1% | forward_split |
| ACRS | 2023-11-13 | 4.75 | 0.65 | -86.4% | forward_split |
| LYRA | 2024-05-06 | 201.50 | 26.10 | -87.1% | forward_split |
| JBIO | 2025-04-29 | 93.80 | 10.14 | -89.2% | forward_split |
| KALA | 2025-09-29 | 19.05 | 2.04 | -89.3% | forward_split |
| MLTX | 2025-09-29 | 61.99 | 6.25 | -89.9% | forward_split |
| TECX | 2024-06-21 | 16.80 | 1.40 | -91.7% | forward_split |
| JBIO | 2024-06-17 | 861.70 | 57.75 | -93.3% | forward_split |

## 2. Panel Tail Extremes (top/bottom 0.5%)

### 2024

| Ticker | As-Of | Return Col | Value | Tier | Eligible | Classification |
|--------|-------|-----------|-------|------|----------|----------------|
| DRUG | 2024-09-30 | fwd_ret_20d | +423091.33% | D | 0 | confirmed_split |
| DRUG | 2024-07-31 | fwd_ret_60d | +398695.57% | D | 0 | confirmed_split |
| DRUG | 2024-09-30 | fwd_ret_60d | +341727.63% | D | 0 | confirmed_split |
| DRUG | 2024-08-30 | fwd_ret_60d | +323301.54% | D | 0 | confirmed_split |
| CADL | 2024-02-29 | fwd_ret_60d | +52882.37% | D | 0 | suspicious_jump |
| SRRK | 2024-09-30 | fwd_ret_60d | +49006.71% | D | 0 | confirmed_split |
| JANX | 2024-01-31 | fwd_ret_60d | +46784.44% | D | 0 | suspicious_jump |
| JANX | 2024-01-31 | fwd_ret_20d | +44882.21% | D | 0 | suspicious_jump |
| CADL | 2024-01-31 | fwd_ret_60d | +35833.34% | D | 0 | suspicious_jump |
| SRRK | 2024-09-30 | fwd_ret_20d | +30845.63% | D | 0 | confirmed_split |
| MESO | 2024-02-29 | fwd_ret_60d | +29975.49% | D | 0 | suspicious_jump |
| VKTX | 2024-01-31 | fwd_ret_20d | +29457.20% | C | 1 | suspicious_jump |
| QURE | 2024-09-30 | fwd_ret_60d | +27727.27% | D | 0 | suspicious_jump |
| CADL | 2024-03-29 | fwd_ret_20d | +26514.28% | D | 0 | suspicious_jump |
| MESO | 2024-01-31 | fwd_ret_60d | +24748.60% | D | 0 | suspicious_jump |
| VTYX | 2024-01-31 | fwd_ret_20d | +24687.50% | D | 0 | suspicious_jump |
| SGMO | 2024-06-28 | fwd_ret_20d | +16209.06% | D | 0 | suspicious_jump |
| FHTX | 2024-01-31 | fwd_ret_20d | +15676.57% | D | 0 | suspicious_jump |
| SGMO | 2024-09-30 | fwd_ret_20d | +14760.31% | D | 0 | suspicious_jump |
| MESO | 2024-02-29 | fwd_ret_20d | +14558.82% | D | 0 | suspicious_jump |
| FULC | 2024-08-30 | fwd_ret_20d | -5590.64% | D | 0 | suspicious_jump |
| SLN | 2024-10-31 | fwd_ret_20d | -5661.76% | D | 0 | suspicious_jump |
| ABEO | 2024-03-29 | fwd_ret_20d | -5779.16% | B | 1 | suspicious_jump |
| TNYA | 2024-11-29 | fwd_ret_20d | -5810.06% | D | 0 | suspicious_jump |
| PRLD | 2024-08-30 | fwd_ret_20d | -5923.08% | C | 1 | suspicious_jump |
| TENX | 2024-01-31 | fwd_ret_20d | -6000.00% | D | 0 | suspicious_jump |
| ANRO | 2024-09-30 | fwd_ret_20d | -6222.02% | D | 0 | suspicious_jump |
| CTMX | 2024-04-30 | fwd_ret_20d | -6510.72% | D | 0 | suspicious_jump |
| RAPT | 2024-01-31 | fwd_ret_20d | -6549.80% | D | 0 | suspicious_jump |
| MRSN | 2024-10-31 | fwd_ret_60d | -7140.66% | D | 0 | suspicious_jump |
| CTMX | 2024-04-30 | fwd_ret_60d | -7192.98% | D | 0 | suspicious_jump |
| IMRX | 2024-01-31 | fwd_ret_60d | -7389.03% | D | 0 | suspicious_jump |
| MRSN | 2024-11-29 | fwd_ret_60d | -7681.74% | D | 0 | suspicious_jump |
| IMRX | 2024-02-29 | fwd_ret_60d | -7691.06% | D | 0 | suspicious_jump |
| PEPG | 2024-11-29 | fwd_ret_60d | -7705.19% | D | 0 | suspicious_jump |
| NMRA | 2024-10-31 | fwd_ret_60d | -8066.25% | D | 0 | confirmed_split |
| NMRA | 2024-12-31 | fwd_ret_20d | -8103.77% | D | 0 | confirmed_split |
| NMRA | 2024-11-29 | fwd_ret_60d | -8163.27% | D | 0 | confirmed_split |
| PRLD | 2024-08-30 | fwd_ret_60d | -8175.58% | C | 1 | suspicious_jump |
| NMRA | 2024-12-31 | fwd_ret_60d | -8924.53% | D | 0 | confirmed_split |

### 2025

| Ticker | As-Of | Return Col | Value | Tier | Eligible | Classification |
|--------|-------|-----------|-------|------|----------|----------------|
| ABVX | 2025-05-30 | fwd_ret_60d | +111794.87% | D | 0 | confirmed_split |
| ABVX | 2025-06-30 | fwd_ret_60d | +95151.52% | D | 0 | confirmed_split |
| VOR | 2025-05-30 | fwd_ret_60d | +82616.36% | D | 0 | suspicious_jump |
| ABVX | 2025-06-30 | fwd_ret_20d | +76363.64% | D | 0 | confirmed_split |
| ABVX | 2025-04-30 | fwd_ret_60d | +73796.30% | D | 0 | confirmed_split |
| ALMS | 2025-10-31 | fwd_ret_60d | +43771.94% | D | 0 | suspicious_jump |
| PRAX | 2025-09-30 | fwd_ret_60d | +42004.90% | D | 0 | suspicious_jump |
| TNGX | 2025-04-30 | fwd_ret_60d | +38794.34% | D | 0 | suspicious_jump |
| ANRO | 2025-09-30 | fwd_ret_60d | +38304.66% | D | 0 | suspicious_jump |
| VOR | 2025-05-30 | fwd_ret_20d | +35241.20% | D | 0 | suspicious_jump |
| PEPG | 2025-08-29 | fwd_ret_20d | +33706.92% | D | 0 | suspicious_jump |
| PRAX | 2025-09-30 | fwd_ret_20d | +27595.24% | D | 0 | suspicious_jump |
| RANI | 2025-09-30 | fwd_ret_20d | +27131.66% | D | 0 | suspicious_jump |
| OLMA | 2025-10-31 | fwd_ret_20d | +23634.21% | D | 0 | suspicious_jump |
| ANRO | 2025-09-30 | fwd_ret_20d | +23464.37% | D | 0 | suspicious_jump |
| QURE | 2025-08-29 | fwd_ret_20d | +23257.81% | D | 0 | suspicious_jump |
| ACRV | 2025-02-28 | fwd_ret_20d | -5642.02% | D | 0 | suspicious_jump |
| PRLD | 2025-10-31 | fwd_ret_20d | -5778.90% | C | 1 | suspicious_jump |
| ELDN | 2025-10-31 | fwd_ret_20d | -6261.15% | B | 1 | suspicious_jump |
| VOR | 2025-10-31 | fwd_ret_20d | -6282.90% | D | 0 | suspicious_jump |
| URGN | 2025-04-30 | fwd_ret_20d | -6450.48% | D | 0 | suspicious_jump |
| REPL | 2025-06-30 | fwd_ret_20d | -6535.52% | D | 0 | confirmed_split |
| AVXL | 2025-08-29 | fwd_ret_60d | -6618.26% | D | 0 | suspicious_jump |
| HUMA | 2025-01-31 | fwd_ret_60d | -6651.16% | D | 0 | suspicious_jump |
| RCKT | 2025-04-30 | fwd_ret_20d | -6844.78% | D | 0 | suspicious_jump |
| ACRV | 2025-01-31 | fwd_ret_60d | -7055.66% | D | 0 | suspicious_jump |
| VOR | 2025-04-30 | fwd_ret_20d | -7314.70% | D | 0 | suspicious_jump |
| ACRV | 2025-02-28 | fwd_ret_60d | -7723.74% | D | 0 | suspicious_jump |
| MLTX | 2025-08-29 | fwd_ret_60d | -7727.35% | B | 1 | confirmed_split |
| VOR | 2025-08-29 | fwd_ret_60d | -8109.14% | D | 0 | suspicious_jump |
| VOR | 2025-02-28 | fwd_ret_60d | -8144.97% | D | 0 | suspicious_jump |
| MLTX | 2025-07-31 | fwd_ret_60d | -8212.40% | B | 1 | confirmed_split |

## 3. Tier Separation Impact

### Treatment: raw

| Tier | N | Mean 60d | Median 60d |
|------|---|----------|------------|
| A | 88 | +1126.45% | +373.36% |
| B | 353 | +903.44% | -169.31% |
| C | 516 | +95.52% | -232.30% |
| D | 0 | N/A | N/A |

**AB mean**: +947.94%  |  **CD mean**: +95.52%  |  **Spread (AB−CD)**: +8.5242

### Treatment: excl_splits

| Tier | N | Mean 60d | Median 60d |
|------|---|----------|------------|
| A | 79 | +1164.72% | +357.44% |
| B | 336 | +1002.61% | -164.34% |
| C | 489 | +104.84% | -201.15% |
| D | 0 | N/A | N/A |

**AB mean**: +1033.47%  |  **CD mean**: +104.84%  |  **Spread (AB−CD)**: +9.2863

### Treatment: winsorized_200

| Tier | N | Mean 60d | Median 60d |
|------|---|----------|------------|
| A | 88 | +21.01% | +200.00% |
| B | 353 | -10.02% | -169.31% |
| C | 516 | -19.03% | -200.00% |
| D | 0 | N/A | N/A |

**AB mean**: -3.83%  |  **CD mean**: -19.03%  |  **Spread (AB−CD)**: +0.1520

## 4. Remediation Candidates

**26** tickers need split-adjusted prices:

- **ABVX** — 1 event(s): 2025-07-23 (reverse_split, +586.0%)
- **ACRS** — 1 event(s): 2023-11-13 (forward_split, -86.4%)
- **AMLX** — 1 event(s): 2024-03-08 (forward_split, -82.3%)
- **BIOA** — 1 event(s): 2024-12-09 (forward_split, -76.8%)
- **COGT** — 1 event(s): 2020-07-06 (reverse_split, +394.4%)
- **DRUG** — 1 event(s): 2024-10-15 (reverse_split, +1445.8%)
- **FGEN** — 1 event(s): 2023-06-26 (forward_split, -83.1%)
- **INDV** — 2 event(s): 2022-11-23 (forward_split, -84.1%); 2022-11-28 (reverse_split, +571.9%)
- **IVVD** — 1 event(s): 2021-12-14 (forward_split, -78.8%)
- **JBIO** — 2 event(s): 2024-06-17 (forward_split, -93.3%); 2025-04-29 (forward_split, -89.2%)
- **KALA** — 1 event(s): 2025-09-29 (forward_split, -89.3%)
- **KOD** — 1 event(s): 2022-02-23 (forward_split, -80.4%)
- **LYRA** — 2 event(s): 2024-05-06 (forward_split, -87.1%); 2025-06-02 (reverse_split, +310.8%)
- **MCRB** — 1 event(s): 2020-08-10 (reverse_split, +389.2%)
- **MENS** — 1 event(s): 2025-12-17 (forward_split, -81.0%)
- **MLTX** — 1 event(s): 2025-09-29 (forward_split, -89.9%)
- **NMRA** — 1 event(s): 2025-01-02 (forward_split, -81.4%)
- **ORKA** — 1 event(s): 2020-05-28 (reverse_split, +386.3%)
- **PHVS** — 1 event(s): 2022-12-08 (reverse_split, +356.6%)
- **PRAX** — 1 event(s): 2022-06-06 (forward_split, -78.1%)
- **PRQR** — 1 event(s): 2022-02-11 (forward_split, -75.3%)
- **REPL** — 1 event(s): 2025-07-22 (forward_split, -77.2%)
- **SRRK** — 1 event(s): 2024-10-07 (reverse_split, +362.0%)
- **SYRE** — 1 event(s): 2023-06-22 (reverse_split, +330.2%)
- **TECX** — 2 event(s): 2024-06-21 (forward_split, -91.7%); 2024-06-24 (reverse_split, +1050.0%)
- **VTYX** — 1 event(s): 2023-11-07 (forward_split, -80.6%)
