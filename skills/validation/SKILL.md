# Validation & Governance Skill

## Purpose

Define the go/no-go gates, data quality checks, staleness windows, IC thresholds, and governance requirements that every pipeline run must satisfy before producing output. This skill encodes Wake Robin's fail-closed philosophy: uncertain or stale data triggers exclusion, not graceful degradation.

## Preconditions

- Pipeline runs MUST have an explicit `as_of_date` parameter (never `datetime.now()`).
- All validation uses `Decimal` arithmetic where scores are involved.
- PIT cutoff: `source_date <= as_of_date - 1` (standard) or `source_date < as_of_date - 2` (strict mode).

---

## Gate 1: Point-in-Time (PIT) Enforcement

| Rule | Formula | Consequence |
|------|---------|-------------|
| Standard PIT | `source_date <= as_of_date - 1 day` | Data admitted |
| Strict PIT | `source_date < as_of_date - 2 days` | Extra buffer for intraday data |
| Lookahead | `age_days < 0` (future data) | **Reject unconditionally** |

Every record must pass PIT admissibility before entering any scoring module. There are no exceptions.

---

## Gate 2: Data Staleness (Phase-Dependent)

### Financial Data

| Level | Age (days) | Action | Penalty |
|-------|-----------|--------|---------|
| PASS | <= 60 | Green light | 1.0x |
| WARN | 60-90 | Log warning | 1.0x |
| SOFT_GATE | 90-120 | Apply penalty | 0.5x |
| HARD_GATE | > 120 | **Exclude** | 0.0x |

### Trial Data - Phase 3

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 90 | 1.0x |
| WARN | 90-120 | 1.0x |
| SOFT_GATE | 120-180 | 0.6x |
| HARD_GATE | > 180 | **Exclude** |

### Trial Data - Phase 2

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 180 | 1.0x |
| WARN | 180-270 | 1.0x |
| SOFT_GATE | 270-365 | 0.7x |
| HARD_GATE | > 365 | **Exclude** |

### Trial Data - Phase 1

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 270 | 1.0x |
| WARN | 270-365 | 1.0x |
| SOFT_GATE | 365-545 | 0.8x |
| HARD_GATE | > 545 | **Exclude** |

### Market Data

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 3 | 1.0x |
| WARN | 3-5 | 1.0x |
| SOFT_GATE | 5-10 | 0.3x |
| HARD_GATE | > 10 | **Exclude** |

### Short Interest Data (FINRA 2-week lag built in)

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 20 | 1.0x |
| WARN | 20-30 | 1.0x |
| SOFT_GATE | 30-45 | 0.5x |
| HARD_GATE | > 45 | **Exclude** |

### 13F Holdings Data (45-day SEC filing lag)

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 60 | 1.0x |
| WARN | 60-90 | 1.0x |
| SOFT_GATE | 90-135 | 0.4x |
| HARD_GATE | > 135 | **Exclude** |

**SEC_13F_FILING_LAG_DAYS**: 45 (built-in constant).

---

## Gate 3: Data Quality Hard Gates

| Gate | Threshold | Action |
|------|-----------|--------|
| Financial data age | > 90 days | Exclude from scoring |
| Market data age | > 7 days | Exclude from scoring |
| Trial data age | > 30 days | Exclude from scoring |
| Liquidity (ADV) | < $500,000/day | Exclude ticker |
| Price (penny stock) | < $5.00 | Exclude ticker |
| Market field coverage | < 80% fields present | Exclude ticker |
| Financial field coverage | < 50% fields present | Issue warning |

---

## Gate 4: Circuit Breakers

| Condition | Threshold | Action |
|----------|-----------|--------|
| Records failing validation | > 20% | Log warning |
| Records failing validation | > 50% | **Fail entire pipeline** |
| Minimum records for check | < 10 | Skip circuit breaker check |

Circuit breakers prevent silent data corruption from propagating through the pipeline.

---

## Gate 5: Input Validation

| Validation | Rule | Default |
|-----------|------|---------|
| Ticker format | `^[A-Z]{1,5}$` | Max 5 uppercase alpha |
| Minimum date | >= 1990-01-01 | Historical cutoff |
| Cash | Non-negative | Required positive |
| Market cap | Positive | Required positive |
| Maximum runway | <= 1200 months | 100-year cap |
| Valid records % | >= 10% must pass | Minimum threshold |

---

## Gate 6: Score Bounds Validation

All scores must fall within [0, 100]:

| Score Field | Min | Max | Module |
|------------|-----|-----|--------|
| financial_score | 0.0 | 100.0 | Module 2 |
| clinical_score | 0.0 | 100.0 | Module 4 |
| catalyst_score | 0.0 | 100.0 | Module 3 |
| score_blended | 0.0 | 100.0 | Module 3 v2 |
| composite_score | 0.0 | 100.0 | Module 5 |

Any score outside [0, 100] is a pipeline error. Fail-closed.

---

## Gate 7: Weight Sum Validation

Module 5 component weights must sum to 1.0 within tolerance:

| Constraint | Expected | Tolerance |
|-----------|----------|-----------|
| Weight sum | 1.0 | +/- 0.01 |

Weights outside tolerance are a configuration error. Fail-closed.

---

## Gate 8: Module Coverage Minimums

| Module | Minimum Coverage | Action if Below |
|--------|-----------------|-----------------|
| Module 2 (Financial) | 80% of universe | Warning |
| Module 3 (Catalyst) | 80% of universe | Warning |
| Module 4 (Clinical) | 80% of universe | Warning |

---

## Gate 9: Severity System

### Severity Levels

| Level | Meaning | Score Multiplier | Action |
|-------|---------|-----------------|--------|
| NONE | Healthy | 1.0 | Include |
| SEV1 | Caution | 0.90 (10% penalty) | Include with flag |
| SEV2 | Warning | 0.50 (50% penalty) | Include, soft gate |
| SEV3 | Critical | 0.00 | **Exclude** |

SEV3 is a hard gate. The ticker is removed from the rankable universe.

---

## Gate 10: Pipeline Health Status

| Component | Coverage Threshold | Status if Below |
|----------|-------------------|----------------|
| catalyst_raw | 10% | DEGRADED |
| momentum | 0% | OPTIONAL |
| smart_money | 0% | OPTIONAL |
| market_data | 0% | OPTIONAL |

**Run Status Classification:**
- **OK**: All thresholds met
- **DEGRADED**: Optional components below threshold
- **FAIL**: Critical catalyst pipeline broken (< 5% with events)

---

## IC Quality Benchmarks

### Information Coefficient Thresholds

| Quality | IC Range | Classification | Action |
|---------|---------|---------------|--------|
| Excellent | IC > 0.05 | Institutional-grade | Deploy |
| Good | IC 0.03-0.05 | Tradeable | Use with confidence |
| Weak | IC 0.01-0.03 | Needs enhancement | Monitor |
| Noise | IC < 0.01 | No predictive power | Abandon signal |
| Negative | IC < 0 | Inverted signal | Investigate inversion |

### IC Measurement Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| MIN_OBS_IC | 10 | Minimum observations for IC calculation |
| MIN_OBS_TSTAT | 20 | Minimum for t-statistic |
| MIN_OBS_BOOTSTRAP | 30 | Minimum for bootstrap CI |
| MIN_ROLLING_WINDOW | 12 weeks | Minimum rolling window |
| BOOTSTRAP_ITERATIONS | 1000 | Bootstrap resampling count |
| TSTAT_THRESHOLD_95 | 2.0 | 95% confidence |
| TSTAT_THRESHOLD_99 | 2.58 | 99% confidence |

### Forward Return Horizons

| Horizon | Trading Days |
|---------|-------------|
| 1w | 5 |
| 2w | 10 |
| 1m | 20 |
| 1.5m | 30 |
| 3m | 60 |
| 4.5m | 90 |

### Market Cap Buckets (IC Analysis)

| Bucket | Range |
|--------|-------|
| MICRO | < $300M |
| SMALL | $300M - $1B |
| MID | $1B - $5B |
| LARGE | > $5B |

---

## Regime Data Staleness Haircuts

| Data Age | Confidence Multiplier |
|---------|----------------------|
| <= 2 days | 1.00 (full) |
| 3-5 days | 0.85 (15% haircut) |
| 6-10 days | 0.65 (35% haircut) |
| > 10 days | 0.00 (force UNKNOWN regime) |

---

## Production Hardening Limits

### File Size Limits

| File Type | Max Size |
|----------|---------|
| JSON files | 100 MB |
| Config files | 10 MB |
| Checkpoint files | 50 MB |

### Operation Timeouts

| Operation | Timeout |
|----------|---------|
| File read | 60 seconds |
| Module execution | 600 seconds (10 min) |
| Full pipeline | 3600 seconds (1 hour) |

### Logging Sanitization

| Limit | Value |
|-------|-------|
| List items logged | 10 max |
| String length logged | 200 chars max |
| Blocked patterns | `api_key`, `password`, `secret`, `token`, `credential`, `ssn`, `account_number`, `cusip` |

---

## Determinism Enforcement

| Setting | Required Value | Purpose |
|---------|---------------|---------|
| force_deterministic_timestamps | true | No `datetime.now()` |
| sort_output_keys | true | Reproducible JSON |
| include_content_hashes | true | Integrity verification |
| random_seed | 42 | Reproducible randomization |

### Determinism Rules

1. Same inputs MUST produce byte-identical outputs
2. All JSON serialization uses sorted keys
3. All list operations use deterministic sort keys
4. Content hashes (SHA256) included in every output for verification
5. No external API calls during scoring (stdlib only)
6. All timestamps derived from `as_of_date`, never from wall clock

---

## Governance Metadata Requirements

Every pipeline output MUST include:

```json
{
  "_governance": {
    "run_id": "<deterministic-hash>",
    "score_version": "<version>",
    "schema_version": "<version>",
    "parameters_hash": "sha256:<hash>",
    "pit_cutoff": "<ISO-date>",
    "as_of_date": "<ISO-date>"
  }
}
```

### Audit Stages

| Stage | When |
|-------|------|
| INIT | Pipeline initialization |
| LOAD | Data loading |
| ADAPT | Data transformation |
| FEATURES | Feature engineering |
| RISK | Risk calculation |
| SCORE | Scoring execution |
| REPORT | Report generation |
| FINAL | Pipeline completion |

### Audit Status Values

| Status | Meaning |
|--------|---------|
| OK | Stage passed |
| FAIL | Stage failed |
| SKIP | Stage skipped |

### Standard Error Codes

| Code | Description |
|------|-------------|
| MISSING_INPUT | Required input not found |
| SCHEMA_MISMATCH | Schema validation failed |
| HASH_ERROR | Integrity check failed |
| PARAMS_MISSING | Parameters incomplete |
| MAPPING_MISSING | Mapping not found |
| VALIDATION_ERROR | Data validation failed |
| UNKNOWN_ERROR | Unclassified error |

---

## Schema Version Support

| Module | Supported Versions |
|--------|-------------------|
| module_1 | 1.0.0 |
| module_2 | 1.0.0 |
| module_3 | dynamic, m3catalyst_vnext_20260111 |
| module_4 | 1.0.0 |
| module_5 | 1.0.0, 1.1.0 |

---

## Enhancement Engine Confidence Thresholds

| Engine | Confidence Gate | Effect Below Gate |
|--------|----------------|-------------------|
| PoS | 0.40 | PoS weight -> 0 |
| Momentum | 0.50 | Momentum not meaningful |
| Smart Money | 0.50 | Smart money signal excluded |
| Valuation | 0.40 | Valuation fallback to sector |

---

## Pre-Run Checklist

Before executing a pipeline run, verify:

1. `as_of_date` is explicitly provided (never derived from wall clock)
2. All input files exist and are within size limits
3. PIT cutoff is computed and logged
4. Schema versions match expected versions
5. Weight sums are within tolerance
6. No `float` arithmetic in scoring paths (only `Decimal`)
7. No `datetime.now()` calls in any module
8. No `random` module usage without explicit seed
9. Audit log writer is initialized
10. Run ID is deterministically generated

## Post-Run Checklist

After a pipeline run completes, verify:

1. All output scores are within [0, 100]
2. Governance metadata is present in every output file
3. Content hashes match recomputed hashes (determinism check)
4. No SEV3 tickers appear in ranked output
5. Coverage metrics are logged (per-module and per-signal)
6. Circuit breaker did not trip silently
7. Staleness penalties were applied where required
8. Audit log contains entries for all stages (INIT through FINAL)

---

## Source Files

| Component | File |
|----------|------|
| Data Quality Gates | `common/data_quality.py` |
| Staleness Gates | `common/staleness_gates.py` |
| PIT Enforcement | `common/pit_enforcement.py` |
| Input Validation | `common/input_validation.py` |
| Integration Contracts | `common/integration_contracts.py` |
| Schema Validation | `common/schema_validation.py` |
| Production Hardening | `common/production_hardening.py` |
| Robustness Utilities | `common/robustness.py` |
| IC Measurement | `backtest/ic_measurement.py` |
| Audit Log | `governance/audit_log.py` |
| Pipeline Config | `config.yml` |
