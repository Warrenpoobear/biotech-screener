# DEM Top-30 EW — Weekly Forward Validation Summary

**Updated:** 2026-11-11T21:52Z  
**Model hash:** `827c35a9ed3ee6e1`  
**Ruleset:** `8887576e`  
**Candidate registered:** 2026-06-26  
**Out-of-sample clock start:** 2026-11-02  
**Test:** Equal-weight Top-30 by `actionable_rank` vs XBI (5-day non-overlapping windows)  

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Non-overlapping 5d windows (all captured) | **12** |
| **Mandate-eligible live windows** | **1** |
| Mean weekly excess (basket − XBI) | **+1.197%** |
| Std weekly excess | 2.012% |
| t-stat | **2.06** |
| Hit rate (excess > 0) | **75%** |
| Cumulative excess | **+14.37%** |
| Cumulative basket | +52.00% |
| Cumulative XBI | +37.63% |

> Statistics above are **descriptive over all captured windows (incl. replay)** and are not mandate evidence. The mandate gate below counts only mandate-eligible live windows (capture_mode=LIVE, quality=PASS, model_hash_match, benchmark_available, 5d return realized).

## Gate Progress (mandate-eligible windows only)

- [1/20] Directional proof (XS>0, hit>50%)
- [1/40] Strong evidence (t>1.5)
- [1/52] Investable evidence (t near 2.0)

---

## Weekly Detail

| Week | Capture Date | Mode | Elig | Basket | XBI | Excess | Hit | Boot% | B30 | Quality |
|------|-------------|------|------|--------|-----|--------|-----|-------|-----|---------|
| 2026-W24 | 2026-06-12 | ? | — | +10.75% | +9.69% | +1.06% | + | 100% | +5.31% | DEGRADED |
| 2026-W25 | 2026-06-15 | ? | — | +9.76% | +8.31% | +1.45% | + | 98% | +5.60% | DEGRADED |
| 2026-W26 | 2026-06-22 | ? | — | +9.05% | +8.54% | +0.51% | + | 80% | +6.97% | DEGRADED |
| 2026-W29 | 2026-07-15 | LIVE | — | -0.72% | -1.11% | +0.39% | + | 96% | -4.45% | PASS |
| 2026-W30 | 2026-07-20 | LIVE | — | -0.32% | -2.00% | +1.67% | + | 95% | -2.28% | PASS |
| 2026-W31 | 2026-07-28 | LIVE | — | +5.76% | +3.02% | +2.74% | + | 96% | +3.23% | PASS |
| 2026-W32 | 2026-08-03 | LIVE | — | +5.63% | +7.05% | -1.42% | - | 9% | +7.41% | PASS |
| 2026-W33 | 2026-08-10 | LIVE | — | -1.13% | +0.03% | -1.16% | - | 28% | -2.89% | PASS |
| 2026-W42 | 2026-10-15 | LIVE | — | +3.42% | +1.24% | +2.18% | + | 92% | -1.58% | PASS |
| 2026-W43 | 2026-10-19 | LIVE | — | +1.47% | +1.77% | -0.30% | - | 69% | -2.04% | PASS |
| 2026-W44 | 2026-10-26 | LIVE | — | +2.08% | +1.05% | +1.03% | + | 88% | -2.38% | PASS |
| 2026-W45 | 2026-11-03 | LIVE | ✓ | +6.26% | +0.06% | +6.21% | + | 98% | +1.14% | PASS |

---

## Adversarial Controls (once ≥20 windows)

Boot% = fraction of 1000-sample random baskets the Top-30 EW beats in excess-vs-XBI.  
B30 = equal-weight bottom-30 basket **raw return** over the same window (not an excess — the §6 control compares B30 *excess* vs XBI, tabulated below).  

| Control | Current average |
|---------|----------------|
| Bootstrap percentile (avg) | 79% |
| Bottom-30 avg excess vs XBI | -1.97% |
| Bottom-30 avg raw return | +1.17% |

---

## Interpretation Notes

- 12 independent weekly 5d periods. One-tailed 95% threshold: ~1.65. Two-tailed 95%: 1.96.
  t=2.06 — two-tailed 95% cleared. Operator review of investability gate required.

- Do not promote EES, expectation-gap, or options-layer signals based on this record.
- Do not reinterpret historical evidence as forward evidence.
- A bad week is not refutation; a good week is not confirmation.
- Investability requires explicit operator clearance at the 52-window gate.

*Protocol: `docs/FORWARD_VALIDATION_PROTOCOL.md`*
