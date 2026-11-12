---
name: research-backtest
description: Research history, dead lanes, evidence hierarchy, benchmark commands
metadata:
  type: research
  status: active
  paths:
    - scripts/research/**
    - tools/run_*benchmark*.py
    - tools/run_promotion_battery.py
---

# Research & Backtest Rules

---

## Current Operating Truths

Spec 050 (2026-04-03) replaced the old optionality-anchored selector with a two-stage
selector/ranker architecture. Checklist v2 rerun (2026-04-04) revalidated the live stack
under the Spec 055 statistical bar (FM, bootstrap, FDR, LOSO).

> **Production mental model: coinvest selects (sole institutional signal as of v1.14.0), financial penalizes
> "safe but less catalytic" names, and clinical is a weak/conditional feature under review.**
> inst_delta_z zeroed in selector 2026-05-04 (ALERT two-frame confirmed; governance log filed).

1. **B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30 is production.** True PIT backtest: +2.34pp/mo net-of-cost, t=2.57, 69% hit rate, 67 monthly periods (Jun 2020 — Apr 2026).
2. **B6 selector validated under Checklist v2.** Bootstrap: +2.42pp/mo, 95% CI [1.25%, 3.70%], P(>0)=99.99%. LOSO: ROBUST across all dimensions. Neither component survives standalone, but the bundle is real.
3. **Selector and ranker learn different structure.** B6 selector runs coinvest_score_z at 100% weight (coinvest-only since v1.14.0; inst_delta_z zeroed 2026-05-04, ALERT: mean IC = -0.097 over 36 dates). Within top-30: inst_delta is the dominant positive discriminator (NW-t=+3.32), financial_score is a true negative penalty (NW-t=-3.41), coinvest washes out (+0.49). inst_delta_z excluded from ranker since Spec 051.
4. **Pairwise ranker is ordinal-only.** ECE=0.129 (POOR calibration). No rank-weighting, no confidence sizing. Equal-weight is the correct construction.
5. **EW Top-30 is the correct construction.** RW-EW = -0.09pp, t=-0.95. Pairwise calibration confirms.
6. **K=30 validated by sweep.** Net-of-cost peak at +2.34pp, stable K=25-35 plateau.
7. **Bear/neutral alpha engine.** Bear: +3.37pp (75% hit), neutral: +6.23pp (93% hit), bull: -0.37pp (50% hit). Worst months are all bull regime.
8. **event_type_score is the only 5/5 Checklist v2 pass.** Use as overlay/diagnostic/sizer only — does NOT improve B6 bundle.
9. **insider_exec and aact_execution downgraded.** Both 1/5 under Checklist v2. Shadow only.
10. **Forward shadow accumulating daily** (7 arms in coinvest_shadow_tracker v2, wired into run_daily.py).
11. **score_rank_pct IC HEALTHY as of 2026-06-24**: mean_ic = +0.0432, hit_rate = 54.3%, N = 35 dates. This cleared the Path C IC gate and Day-5 Phase 2 checkpoint. Next IC review: 2026-07-01.
    *(Corrected 2026-11-12: this figure has the wrong sign and measures the composite percentile, not the production order. The IC tools negated `score_rank_pct`, which is higher-is-better. Corrected readings for the same period: `score_rank_pct` −0.0201, `actionable_rank` +0.1534, `coinvest_score_z` +0.1478. See `docs/governance/2026-11-12-ic-dashboard-sign-correction.md`.)*

---

## Trust Buckets

### Safe to use now (production-grade evidence)
- **B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30**: true PIT validated, t=2.57, 67 periods
- **B6 bundle revalidated under Checklist v2** (2026-04-04): bootstrap CI [1.25%, 3.70%], LOSO ROBUST
- **Pairwise ordinal-only policy**: ECE=0.129, no rank-weighting or confidence sizing
- Selector engine (`selector_engine.py`), ranker engines (`ranker_engine.py`, `ranker_v2_pairwise.py`): 48+ tests
- Statistical QA package (`common/stats/`): FM, bootstrap, FDR, LOSO, calibration — 36 tests
- PIT validation audit framework, PIT financial regeneration infrastructure
- K=30 validated by sweep (stable K=25-35 plateau)
- Forward shadow tracker (7 arms, wired into daily cron)
- event_type_score as overlay/diagnostic (5/5 Checklist v2 pass, but not selector weight)

### Deprecated (do not cite)
- **All survivorship-only benchmark numbers** (+93.7pp, +110.5pp, etc.)
- **Old optionality-anchored selector** — underwater on PIT data (-25pp cumulative)
- **DEFAULT selector weights** (clinical 35%, catalyst 25%) — destructive as selector (-0.53pp)
- **clinical_score_v2_z as selector anchor** — negative delta (-0.68pp), universally destructive
- **Pre-Checklist-v2 signal card t-stats** — superseded by FM/bootstrap/FDR/LOSO findings
- **insider_exec_buy_value_90d optimistic reads** — 1/5 under Checklist v2, FRAGILE
- **aact_execution_score optimistic reads** — 1/5 under Checklist v2, bear-unstable
- Any promotion memo citing pre-Spec-050 selector performance
- "Bear IR 3.35" regime story from contaminated data
- **Any ranker IC claim based on composite_score** (Spec 095, 2026-05-13) — measured the wrong score field, misattributed

### Current evidence hierarchy
1. **Checklist v2 rerun (2026-04-04)**: B6 bundle bootstrap+LOSO validated — STRONGEST (for signals)
2. **True PIT backtest (Spec 050)**: A4+ranker +2.34pp net, t=2.57 — STRONGEST (for portfolio)
3. **Pairwise feature audit (2026-04-04)**: within-top-30 FM on ranker features — SUPPORTING
4. **Forward shadow**: accumulating daily since 2026-04-03 — MONITORING
5. **Old PIT benchmark (Spec 048)**: optionality selector underwater — SUPERSEDED by new selector

---

## Do Not Reopen Without New Evidence

These lanes have been tested and either died or were superseded. Do not spend research
hours here unless genuinely new data or a structural model change creates a reason to revisit.

| Lane | Status | Why closed |
|------|--------|------------|
| Options surface-shape as systematic ranker | DEAD | 50-month backtest IC negative at all horizons |
| Options-as-alpha (Spec 053) | CLOSED | 37 signals tested, ALL fail as selector/ranker |
| Static execution features (Spec 054) | CLOSED | PCD overdue, update recency, pipeline velocity all noise/destructive |
| Clinical composites as ranker (Spec 055) | CLOSED | Negative across ALL robustness slices, universally destructive |
| `total_volume_z` | DEAD | IC=-0.10 on PIT-native data (109 obs) |
| Always-on rank-weighting (Top-20 or Top-30) | NOT PROMOTED | RW-EW = -0.09pp; pairwise ECE=0.129 confirms ordinal-only |
| Confidence/rank-weighted sizing | NOT JUSTIFIED | Pairwise scores not calibrated (ECE=0.129) |
| `insider_exec_buy_value_90d` | SHADOW ONLY | 1/5 Checklist v2, FRAGILE robustness |
| `aact_execution_score` | SHADOW ONLY | 1/5 Checklist v2, bear-unstable (-1.86pp) |
| Top-20 / pruner promotion story | DEPRECATED | PIT-financial correction shows both underwater vs XBI |
| Historical alpha narrative (+93pp / +110pp) | DEPRECATED | Inflated by financial look-ahead contamination |
| `cal_alpha` | REMOVED in v1.12.0 | Confirmed no-op, zero deltas at all horizons |
| Clinical sort signal | OFF | Insufficient IC, destructive as selector |
| Coinvest as standalone sort signal | SUPERSEDED | Now used as B6 selector anchor; standalone only 3/5 Checklist v2 |
| Quality tiebreaks (Specs 030/031) | EXHAUSTED | All economically immaterial |
| 91-180d drawdown gate | DEAD | Counterproductive at all thresholds |
| Dynamic caps | DEAD | Identical to plain EW |
| Fixed sleeve budgets | RETIRED | Primary construction damage mechanism (+153.6pp drag) |

---

## Current Promotion Story

1. **Coinvest-only selector + pairwise_minimal ranker (ordinal-only) + EW Top-30 is production (v1.14.0).** B6 selector now coinvest_score_z 100% (inst_delta_z zeroed 2026-05-04, ALERT: mean_ic=-0.097 over 36 dates; reinstatement conditions in governance log). Original B6 bundle (coinvest 65% + inst_delta 35%) validated in true PIT backtest: +2.34pp/mo net-of-cost, t=2.57, 69% hit rate, 67 monthly periods.
2. True PIT evidence: +2.34pp/mo net, t=2.57, 69% hit, beats XBI on return and risk.
3. **B6 bundle passes Checklist v2**: bootstrap +2.42pp/mo, CI [1.25%, 3.70%], LOSO ROBUST. Bundle > parts.
4. **Pairwise ordinal-only confirmed**: ECE=0.129. Do not rank-weight or confidence-size.
5. **Within-cohort roles clear**: coinvest selects (sole institutional signal, v1.14.0), financial penalizes safe names. inst_delta_z zeroed in selector — reinstatement pending IC recovery.
6. **event_type_score**: 5/5 Checklist v2 but overlay only — does not improve B6 bundle.
7. **Forward shadow is the validation layer.** 7 arms accumulating daily. Evaluate after 30 trading days.
8. **K=30 is validated** by PIT sweep (stable K=25-35 plateau, net-of-cost peak).
9. **Regime caveat**: this is a bear/neutral alpha engine. Expect bounded underperformance in strong bull.
10. The governance hold (Spec 048) **succeeded**: it prevented the old optionality selector from being institutionalized on contaminated data, which led to finding the better B6 selector.

---

## PIT Rules (Deep Dive)

1. **Never call the historical set "true PIT"** unless archived raw inputs, archived code, AND archived derived artifacts all exist as-of each date.
2. Historical benchmark outputs must carry `pseudo_pit_version` (1=contaminated, 2=cleaned).
3. Benchmark reruns must use the PIT-aware paths: `--pit-mode survivorship` or `--pit-mode full`.
4. Long-history conclusions are **provisional** until PIT-v2 financial rerun lands.
5. The forward monitor is the only true out-of-sample evidence. Accumulate it.

---

## Canonical Benchmark Commands

```bash
# Survivorship-cleaned selection benchmark (current baseline)
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30

# Monthly IC / selection benchmark
python3 scripts/research/selection_benchmark.py --pit-mode survivorship

# Ranker evaluation (inst_delta_z within top-30)
python3 scripts/research/ranker_evaluation_harness.py --signal inst_delta_z --pit-mode survivorship

# Construction v2 benchmark (all variants)
python3 scripts/research/construction_v2_benchmark.py --pit-mode survivorship

# PIT-financials snapshot regeneration (heavy lift, ~2h)
python3 scripts/research/regenerate_pit_v2_snapshots.py

# Run benchmarks on PIT-financial-corrected snapshots
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30 --snapshot-dir data/snapshots_pit_v2
```
