# Failure Mode Card — 2026-06-28

**Classification:** VALIDATION_INFRASTRUCTURE / FAILURE_MODE_SHADOW_GUARDS / NO_MODEL_CHANGE

---

## Rolling Drawdown Status

4-window rolling XS: **+4.66pp** (threshold: −5.00pp)  [OK]

| window | xs_pp |
|--------|-------|
| 2026-06-10 → 2026-06-17 | +0.52pp |
| 2026-06-11 → 2026-06-18 | +0.23pp |
| 2026-06-15 → 2026-06-22 | +2.21pp |
| 2026-06-16 → 2026-06-23 | +1.69pp |

## Repeat-Offender Flags

Flagged (≥2 recent bottom-5 windows): **3**  |  Watch (1 window): **9**

| ticker | rank | bad_windows | total_contrib | ees | replacement |
|--------|------|-------------|---------------|-----|-------------|
| ABVX | 16 | 3 | -2.190pp | False | CELC |
| DRUG | 11 | 3 | -1.264pp | True | CELC |
| PRAX | 14 | 2 | -1.425pp | True | CELC |

## EES-Guarded Shadow

No completed EES-guarded windows yet.

## Rank-Depth Replacement Shadow

No completed rank-depth windows yet.

## Current Catalyst Concentration

| bucket | n_names |
|--------|---------|
| build_window | 12 ⚠️ |
| binary_now | 9 |
| core | 6 |
| less_binary | 3 |

EES-False names in Top-30: **5** (flag if ≥5)

---

## Promotion Gate Status

| gate | evidence_needed | windows_completed | status |
|------|----------------|-------------------|--------|
| EES guardrail | 20w +delta ≥60% | 0 | ACCUMULATING |
| Repeat-offender warning | 10w serial underperformance | — | ACCUMULATING |
| Rank-depth replacement | 20w +delta | 0 | ACCUMULATING |

No production change authorized until gates are met and operator reviews.

*Generated: 2026-06-28 — VALIDATION_INFRASTRUCTURE / NO_MODEL_CHANGE*