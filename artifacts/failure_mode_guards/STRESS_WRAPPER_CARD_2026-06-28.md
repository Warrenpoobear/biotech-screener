# Stress Wrapper Card — 2026-06-28

**Classification:** VALIDATION_INFRASTRUCTURE / STRESS_WRAPPER_SHADOW / NO_MODEL_CHANGE

---

## Investability Status

```
PILOT_INVESTABLE_WITH_SHADOW_GUARDS
```

Live pilot may proceed. Promotion question: does raw Top-30 plus monitored
stress guards produce better net forward outcomes without killing convexity?

---

## Regime Context

XBI trailing 20d return: **+6.8%**  →  regime: **MILD_RALLY**

> Model may lag momentum/junk rotation. Mild-rally lag is a KNOWN PATTERN, not model failure.

| regime | expected behavior |
|--------|------------------|
| BEAR/STRESS (<−3%) | Model outperforms — quality dispersion high |
| NEUTRAL (−3% to +3%) | Baseline alpha expected |
| MILD_RALLY (3–8%) | May lag momentum/junk — KNOWN PATTERN, not failure |
| STRONG_RALLY (>8%) | Watch beta/tail — XBI move may overwhelm selection |

---

## Conditional Risk Wrapper

Status: **ACTIVE**  |  Drawdown: +4.66pp (OK)

| condition | threshold | current | triggered |
|-----------|-----------|---------|-----------|
| (A) rolling_4w_xs | ≤ −5.00pp | +4.66pp | no |
| (B) repeat_offender_count | ≥ 2 | 3 | YES |
| (C) ees_false_top30 | ≥ 5 | 5 | YES |
| (D) build+less_binary | ≥ 55% | 50.0% | no |

### Stress Basket Comparison (current window, returns pending)

When returns are available, the table below will fill. Compare risk_guarded vs raw_top30.

No completed stress wrapper windows yet.

---

## Repeat-Offender Table

Flagged (≥2 bottom-5 windows): **3**  |  Watch: **9**

| ticker | rank | weeks_top30 | neg_windows | btm5_windows | cum_xs | ees | cat_bucket | replacement |
|--------|------|-------------|-------------|--------------|--------|-----|------------|-------------|
| ABVX **F** | 16 | 11 | 3 | 3 | -1.849pp | False | build_window | CELC (r31) |
| DRUG **F** | 11 | 11 | 3 | 3 | -1.264pp | True | less_binary | CELC (r31) |
| PRAX **F** | 14 | 11 | 5 | 2 | -1.368pp | True | binary_now | CELC (r31) |
| TYRA | 27 | 11 | 2 | 1 | -0.603pp | True | build_window | CELC (r31) |
| RVMD | 8 | 11 | 5 | 1 | -0.397pp | True | core | CELC (r31) |
| COGT | 1 | 11 | 3 | 1 | -0.379pp | True | binary_now | CELC (r31) |
| XENE | 19 | 11 | 5 | 1 | -0.371pp | False | build_window | CELC (r31) |
| ERAS | 15 | 11 | 1 | 1 | -0.223pp | True | less_binary | CELC (r31) |
| EWTX | 25 | 11 | 3 | 1 | +0.181pp | True | build_window | CELC (r31) |
| PHVS | 26 | 11 | 4 | 1 | +0.185pp | True | binary_now | CELC (r31) |
| NBIX | 30 | 11 | 2 | 1 | +0.264pp | True | core | CELC (r31) |
| TNGX | 12 | 11 | 1 | 1 | +1.628pp | False | build_window | CELC (r31) |

---

## EES-Guarded Shadow

EES delta by window quality (EES is insurance, not unconditional alpha):

| quality | n | avg_delta |
|---------|---|-----------|
| success | 30 | -0.15pp |
| neutral | 38 | -0.05pp |
| failure | 30 | +0.80pp |

Overall: n=115  avg_delta=+0.14pp  pct_positive=19%
EES-False in current Top-30: **5** (flag if ≥5)

Promotion gate: 20w, +delta in failure windows, does NOT reduce full-period net return.

---

## Rank-Depth Bench

Completed windows: **115**
Bench vs Top-30 avg delta: **+0.04pp** → FAILURE-SPECIFIC (expected — bench is a stress replacement pool)

---

## Current Catalyst Concentration

| bucket | n | flag |
|--------|---|------|
| build_window | 12 | ELEVATED |
| binary_now | 9 |  |
| core | 6 |  |
| less_binary | 3 |  |

---

## Promotion Gate Status

| gate | metric | windows | threshold | status |
|------|--------|---------|-----------|--------|
| EES guardrail | delta in failure windows | 115 | 20w | GATE_MET (count) |
| Rank-depth replacement | bench_delta | 115 | 20w | GATE_MET (count) |
| Stress wrapper | active-window delta | 0 | 20 active windows | ACCUMULATING |
| Repeat-offender | serial underperformance | 3 flagged | persistence evidence | ACCUMULATING |

No production selector/ranker/sizing change authorized until gates are met and operator reviews.

*Generated: 2026-06-28*