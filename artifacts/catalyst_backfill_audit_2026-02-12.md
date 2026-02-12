# Catalyst Backfill v0 Coverage Audit

**Date**: 2026-02-12
**Window**: 365d forward
**Sources**: `trial_cd` (completion_date fallback), `trial_active` (active trial presence)
**Method**: Additive only — never overwrites existing catalyst signals

## Per-Archive Coverage

| Date       | Dev | Exist% | Bkfill% | Comb% | Delta   | CD | Active |
|------------|-----|--------|---------|-------|---------|---:|-------:|
| 2024-01-31 | 172 |  31.4% |  22.7%  | 54.1% | +22.7pp |  0 |     39 |
| 2024-02-29 | 173 |  29.5% |  24.3%  | 53.8% | +24.3pp |  0 |     42 |
| 2024-03-29 | 173 |  28.3% |  25.4%  | 53.8% | +25.5pp |  0 |     44 |
| 2024-04-30 | 175 |  28.0% |  26.3%  | 54.3% | +26.3pp |  0 |     46 |
| 2024-05-31 | 176 |  27.8% |  27.8%  | 55.7% | +27.9pp |  0 |     49 |
| 2024-06-28 | 177 |  27.1% |  28.2%  | 55.4% | +28.3pp |  0 |     50 |
| 2024-07-31 | 177 |  32.2% |  26.6%  | 58.8% | +26.6pp |  0 |     47 |
| 2024-08-30 | 177 |  35.0% |  24.9%  | 59.9% | +24.9pp |  0 |     44 |
| 2024-09-30 | 178 |  39.3% |  22.5%  | 61.8% | +22.5pp |  0 |     40 |
| 2024-10-31 | 178 |  39.3% |  23.6%  | 62.9% | +23.6pp |  0 |     42 |
| 2024-11-29 | 179 |  39.1% |  22.3%  | 61.5% | +22.4pp |  0 |     40 |
| 2024-12-31 | 179 |  40.8% |  20.1%  | 60.9% | +20.1pp |  0 |     36 |
| 2025-01-31 | 179 |  41.3% |  20.7%  | 62.0% | +20.7pp |  0 |     37 |
| 2025-02-28 | 179 |  41.9% |  21.8%  | 63.7% | +21.8pp |  0 |     39 |
| 2025-03-31 | 179 |  40.8% |  23.5%  | 64.2% | +23.4pp |  0 |     42 |
| 2025-04-30 | 179 |  40.8% |  22.3%  | 63.1% | +22.3pp |  0 |     40 |
| 2025-05-30 | 179 |  41.3% |  22.3%  | 63.7% | +22.4pp |  0 |     40 |
| 2025-06-30 | 181 |  42.0% |  21.5%  | 63.5% | +21.5pp |  0 |     39 |
| 2025-07-31 | 183 |  39.9% |  24.6%  | 64.5% | +24.6pp |  0 |     45 |
| 2025-08-29 | 183 |  38.8% |  26.8%  | 65.6% | +26.8pp |  0 |     49 |
| 2025-09-30 | 183 |  38.3% |  27.9%  | 66.1% | +27.8pp |  0 |     51 |
| 2025-10-31 | 183 |  42.6% |  24.6%  | 67.2% | +24.6pp |  0 |     45 |
| 2026-01-15 | 183 |  47.5% |  23.0%  | 70.5% | +23.0pp |  0 |     42 |
| 2026-01-20 | 183 |  47.5% |  23.0%  | 70.5% | +23.0pp |  0 |     42 |
| 2026-01-28 | 182 |  46.7% |  23.1%  | 69.8% | +23.1pp |  0 |     42 |
| 2026-01-30 | 182 |  46.7% |  23.1%  | 69.8% | +23.1pp |  0 |     42 |
| 2026-01-31 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-01 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-03 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-04 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-05 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-06 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| 2026-02-07 | 183 |  47.0% |  23.0%  | 69.9% | +22.9pp |  0 |     42 |
| **Average** |     |  40.1% |         | 63.8% | +23.7pp |    |        |

## Key Findings

1. **+23.7pp average coverage gain** across 33 archives
2. **All backfill is `trial_active`** (0 `trial_cd` hits) — the active-status filter
   correctly blocks COMPLETED trials with stale completion dates
3. 2024 coverage: 31% -> 54% (still ~46% missing)
4. 2025 coverage: 40% -> 65% (still ~35% missing)
5. Consistent ~39-51 active-trial tickers per archive

## Decision Fork Assessment

- 2024 early archives still have >40% missing after backfill
- This puts us in the **"still >60% missing in early 2024"** territory for
  those dates, suggesting primary enrichment improvement may yield more than
  additional layering
- Next step: re-run walkforward panel on backfilled archives to measure
  whether trial_active signal has tier-separation value
- Possible v0.1: relax COMPLETED filter for trials with CD in forward window
  (would add trial_cd hits at cost of lower signal confidence)
