# Decision Engine Data Accuracy Audit — 2026-02-19

**Audit Date**: 2026-02-20
**Snapshot Date**: 2026-02-19
**Git SHA**: `f7168b54`
**Python**: 3.12.3
**Overall Status**: **PASS**

---

## Step 0: Environment / Determinism

| Check | Result |
|-------|--------|
| Python version | 3.12.3 |
| Git HEAD | `f7168b54b945783fc7a223d244cb032f7659394d` |
| Working tree | Clean (untracked only: cache dirs, institutional summary) |

**Status: PASS**

## Step 1: Production Run (`run_daily_production.py --as-of-date 2026-02-19`)

| Check | Result |
|-------|--------|
| Exit code | 0 |
| Overall status | PASS |
| Gates total | 14 |
| Gates PASS | 14 |
| Gates WARN | 0 |
| Gates FAIL | 0 |

All 14 gates passed: `xbi_staleness`, `ctgov_cache`, `inputs_present`, `market_data_schema`, `market_data_staleness`, `market_data_coverage`, `audit`, `missing_reason_fraction`, `turnover`, `drift_monitoring`, `ctgov_pit_dates`, `sec_13f_cache`, `institutional_summary`, `institutional_delta`.

**Status: PASS**

## Step 2: Snapshot + Manifest

| Check | Result |
|-------|--------|
| Snapshot directory | 20 files |
| Manifest version | 1.3.0 |
| Ruleset hash | `8f99d47e` |
| Ruleset version | 1.6.0 |
| Git branch | main |

**Status: PASS**

## Step 3: Hard Contract Checks

| Check | Result |
|-------|--------|
| Gate allowlist complete | 14/14 (no missing, no extra) |
| Ruleset ID matches `PHASE2_PINNED_RULESET_ID` | `8f99d47e` = `8f99d47e` |
| Thresholds ID matches `PHASE2_PINNED_THRESHOLDS_ID` | `74457e8f` = `74457e8f` |
| All required manifest keys present | Yes |

**Status: PASS**

## Step 4: Rankings.csv Integrity

| Check | Result |
|-------|--------|
| Rows | 319 |
| Columns | 98 |
| Universe tickers | 353 (34 filtered by pipeline) |
| No duplicate tickers | Yes |
| `eligible` is binary | Yes (0/1) |
| Single DE version | `v1.3.0` |
| Single ruleset ID | `8f99d47e` |
| `actionable_rank` unique among A/B | Yes |
| `tier_any` consistent with `tier_dev` | Yes |

### Tier Distribution (dev)

| Tier | Count | Eligible |
|------|-------|----------|
| A | 32 | 32 |
| B | 41 | 41 |
| C | 58 | 58 |
| D | 52 | 0 |

### Archetype Distribution

| Archetype | Count |
|-----------|-------|
| drug_developer | 183 |
| commercial_biotech | 71 |
| commercial_pharma | 35 |
| platform_diagnostics | 13 |
| platform_devices | 12 |
| platform_services | 5 |

### Notes
- 14 negative `composite_score` values (expected: alpha_cohort mode produces signed scores)
- `inst_delta_z` all zero (expected: 2026-02-19 is the first snapshot with institutional delta — no prior available for this date)
- 34 tickers filtered by pipeline (353 universe → 319 ranked)

**Status: PASS**

## Step 5: Sidecar Schema / Invariant Checks

### decision_ruleset.json

| Field | Value | Expected | Match |
|-------|-------|----------|-------|
| composite_engine | alpha_cohort | alpha_cohort | PASS |
| catalyst_priority_mode | tiebreaker | tiebreaker | PASS |
| enable_coinvest_sort_signal | True | True | PASS |
| coinvest_sort_weight | 0.05 | 0.05 | PASS |
| enable_clinical_sort_signal | True | True | PASS |
| clinical_sort_weight | 1.0 | 1.0 | PASS |
| enable_institutional_sort_signal | False | False | PASS |
| institutional_sort_weight | 0.3 | 0.3 | PASS |
| catalyst_near_days | 120 | 120 | PASS |
| catalyst_mid_days | 180 | 180 | PASS |

### phase2_health.json

| Check | Result |
|-------|--------|
| status | OK |
| thresholds_id | `74457e8f` (matches pinned) |

### metadata.json

| Check | Result |
|-------|--------|
| as_of_date | 2026-02-19 |
| ticker_count | 319 |
| coinvest_coverage_pct | 82.4% |
| clinical_sort_telemetry | active (70 dev adj, 61 comm adj) |

### eligibility_debug.json

| Check | Result |
|-------|--------|
| Tickers | 319 |
| gate_mode | hard |
| abs_breach | 67 |
| rel_breach | 104 |
| rescued_by_rel | 0 |

### institutional_summary.json

| Check | Result |
|-------|--------|
| cache_as_of_date | 2026-02-19 |
| Tickers | 319 |
| Enriched (elite_holder_shares) | Yes |

### portfolio_positions.json

| Check | Result |
|-------|--------|
| n_positions | 20 |
| top_k | 20 |
| tier_filter | [A, B] |
| total_weight_pct | 99.97% |
| All positions tier A | Yes |
| Weight sum matches | Yes |

### Other Sidecars

| Sidecar | Present | Valid |
|---------|---------|-------|
| catalyst_shadow_metrics.json | Yes | Yes |
| drift_report.json | Yes | Yes |
| decision_portfolio.json | Yes | Yes (319 positions) |
| decision_portfolio.csv | Yes | Yes (319 rows, 29 cols) |
| portfolio_positions.csv | Yes | Yes (20 rows, 21 cols) |

**Status: PASS**

## Step 6: Determinism Spot-Check

| Check | Result |
|-------|--------|
| Input hashes present | Yes (274 entries) |
| Inputs manifest mode | off |
| Ruleset hash (file vs manifest) | Different serialization (expected) |

Note: `decision_ruleset.json` is the full flattened JSON. The `8f99d47e` hash in the manifest is computed by `DecisionRuleset.__post_init__` from frozen dataclass fields. The Step 3 pinned-ID check is authoritative.

**Status: PASS**

---

## Summary

| Step | Description | Status |
|------|-------------|--------|
| 0 | Environment | **PASS** |
| 1 | Production run | **PASS** (14/14 gates) |
| 2 | Snapshot + manifest | **PASS** |
| 3 | Hard contract checks | **PASS** |
| 4 | Rankings integrity | **PASS** |
| 5 | Sidecar checks | **PASS** |
| 6 | Determinism | **PASS** |

**Overall: PASS** — All invariants hold. No data accuracy issues found.
