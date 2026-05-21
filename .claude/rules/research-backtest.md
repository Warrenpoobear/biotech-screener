---
paths:
  - scripts/research/**
  - tools/run_*benchmark*.py
  - tools/run_promotion_battery.py
  - scripts/run_signal_evidence.py
---

# Research, Backtests & Evidence

## PIT Rules
1. Never call the historical set "true PIT" unless archived raw inputs, archived code, AND archived derived artifacts all exist as-of each date.
2. Historical benchmark outputs must carry `pseudo_pit_version` (1=contaminated, 2=cleaned).
3. Benchmark reruns must use PIT-aware paths: `--pit-mode survivorship` or `--pit-mode full`.
4. Long-history conclusions are **provisional** until PIT-v2 financial rerun lands.
5. The forward monitor is the only true out-of-sample evidence. Accumulate it.

## Canonical Benchmark Commands
```bash
# Survivorship-cleaned selection benchmark
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30

# Monthly IC / selection benchmark
python3 scripts/research/selection_benchmark.py --pit-mode survivorship

# Ranker evaluation (inst_delta_z within top-30)
python3 scripts/research/ranker_evaluation_harness.py --signal inst_delta_z --pit-mode survivorship

# Construction v2 benchmark (all variants)
python3 scripts/research/construction_v2_benchmark.py --pit-mode survivorship

# PIT-financials snapshot regeneration (heavy lift, ~2h)
python3 scripts/research/regenerate_pit_v2_snapshots.py

# Benchmarks on PIT-financial-corrected snapshots
python3 scripts/research/build_selection_benchmark.py --pit-mode survivorship --top-n 20 --also-top30 --snapshot-dir data/snapshots_pit_v2
```

## Trust Buckets

### Safe to Use Now (production-grade evidence)
- B6 selector + pairwise_minimal ranker (ordinal-only) + EW Top-30: true PIT validated, t=2.57, 67 periods
- B6 bundle Checklist v2 validated: bootstrap CI [1.25%, 3.70%], LOSO ROBUST
- Pairwise ordinal-only policy: ECE=0.129, no rank-weighting
- Statistical QA package (`common/stats/`): FM, bootstrap, FDR, LOSO, calibration
- K=30 validated by sweep (stable K=25-35 plateau)
- Forward shadow tracker (7 arms, wired into daily cron)
- event_type_score as overlay/diagnostic (5/5 Checklist v2 pass, not selector weight)

### Deprecated (do not cite)
- All survivorship-only benchmark numbers (+93.7pp, +110.5pp)
- Old optionality-anchored selector (underwater on PIT data, -25pp cumulative)
- DEFAULT selector weights (clinical 35%, catalyst 25%) — destructive as selector
- clinical_score_v2_z as selector anchor — negative delta (-0.68pp)
- Pre-Checklist-v2 signal card t-stats
- insider_exec_buy_value_90d optimistic reads — 1/5 Checklist v2, FRAGILE
- aact_execution_score optimistic reads — 1/5 Checklist v2, bear-unstable
- Any ranker IC claim based on composite_score (Spec 095) — wrong score field
- "Bear IR 3.35" regime story from contaminated data

### Evidence Hierarchy
1. Checklist v2 rerun (2026-04-04): B6 bundle bootstrap+LOSO — STRONGEST (signals)
2. True PIT backtest (Spec 050): +2.34pp net, t=2.57 — STRONGEST (portfolio)
3. Pairwise feature audit (2026-04-04): within-top-30 FM — SUPPORTING
4. Forward shadow: accumulating daily since 2026-04-03 — MONITORING
5. Old PIT benchmark (Spec 048): optionality underwater — SUPERSEDED

## Dead Lanes (Do Not Reopen Without New Evidence)

| Lane | Status | Why Closed |
|------|--------|------------|
| Options surface-shape as systematic ranker | DEAD | 50-month backtest IC negative all horizons |
| Options-as-alpha (Spec 053) | CLOSED | 37 signals tested, ALL fail |
| Static execution features (Spec 054) | CLOSED | PCD overdue, update recency, pipeline velocity all noise |
| Clinical composites as ranker (Spec 055) | CLOSED | Negative across ALL robustness slices |
| `total_volume_z` | DEAD | IC=-0.10 on PIT-native (109 obs) |
| Always-on rank-weighting | NOT PROMOTED | RW-EW = -0.09pp; ECE=0.129 confirms ordinal-only |
| Confidence/rank-weighted sizing | NOT JUSTIFIED | Pairwise scores not calibrated |
| `insider_exec_buy_value_90d` | SHADOW ONLY | 1/5 Checklist v2, FRAGILE |
| `aact_execution_score` | SHADOW ONLY | 1/5 Checklist v2, bear-unstable |
| Top-20 / pruner promotion | DEPRECATED | Both underwater vs XBI on PIT-financial |
| Historical alpha (+93pp/+110pp) | DEPRECATED | Inflated by financial look-ahead contamination |
| `cal_alpha` | REMOVED v1.12.0 | Confirmed no-op, zero deltas |
| Clinical sort signal | OFF | Insufficient IC, destructive as selector |
| Coinvest standalone sort | SUPERSEDED | Now B6 selector anchor; standalone only 3/5 |
| Quality tiebreaks (Specs 030/031) | EXHAUSTED | Economically immaterial |
| 91-180d drawdown gate | DEAD | Counterproductive at all thresholds |
| Dynamic caps | DEAD | Identical to plain EW |
| Fixed sleeve budgets | RETIRED | Primary construction damage (+153.6pp drag) |

## Heavy-Lift Jobs
- PIT financial regeneration: COMPLETE. 76 dates in `data/snapshots_pit_v2/`, 72/72 OK.
- Historical alpha collapsed after correction. All pre-correction claims deprecated.
- Next heavy lift: forward monitor accumulation (no compute — just time).
- If forward positive: re-establish selector thesis from clean data. No backfill from historical.
- If forward negative: selector needs structural re-examination.
