# Operator's Guide: biotech-screener Production Pipeline

## Overview

This guide covers safe operation of the biotech-screener investment ranking pipeline.
The system is designed for **determinism** (same inputs → identical outputs) and
**Point-in-Time (PIT) safety** (no lookahead bias).

---

## Quick Start

```bash
# Basic screening run
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results.json

# With all enhancements
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results.json \
    --enable-enhancements \
    --enable-coinvest \
    --enable-short-interest \
    --audit-log audit.jsonl
```

---

## Required Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--as-of-date` | **YES** | Analysis date (YYYY-MM-DD). No defaults! |
| `--data-dir` | **YES** | Directory with input data files |
| `--output` | For runs | Output JSON file path |

**CRITICAL**: `--as-of-date` has no default. You MUST provide it explicitly.
This prevents wall-clock dependencies that would break reproducibility.

---

## Input Data Requirements

The pipeline expects these files in `--data-dir`:

| File | Required | Description |
|------|----------|-------------|
| `universe.json` | YES | List of securities with metadata |
| `financial_records.json` | YES | Quarterly financials (cash, burn, etc.) |
| `trial_records.json` | YES | ClinicalTrials.gov trial data |
| `market_data.json` | YES | Price, volume, volatility |
| `holdings_snapshots.json` | For coinvest | 13F institutional holdings |
| `short_interest.json` | For SI signals | FINRA short interest data |
| `market_snapshot.json` | For regime | VIX, XBI performance, rates |

### Data Freshness Requirements

- **Financial data**: Must be ≤ 90 days old relative to as_of_date
- **Market data**: Must be ≤ 7 days old (prices, volume)
- **Trial data**: Must be ≤ 30 days old (CT.gov updates)

---

## Expected Outputs

### Success Indicators

```
[2026-01-15] Starting screening pipeline...
[1/7] Loading input data...
  Universe validation passed: 85 valid tickers
[2/7] Module 1: Universe filtering...
  Active: 72, Excluded: 13
[3/7] Module 2: Financial health...
  Scored: 72, Missing: 0
[4/7] Module 3: Catalyst detection...
  Events detected: 15, Tickers with events: 8/72
[5/7] Module 4: Clinical development...
  Scored: 72, Trials evaluated: 245
[6/7] Module 5: Composite ranking...
  Rankable: 68, Excluded: 4
  RUN STATUS: OK
```

### Warning Indicators

```
RUN STATUS: DEGRADED - Some components degraded: ['catalyst_coverage']
Component coverage: catalyst=0.65, momentum=0.82, smart_money=0.45
```

This means:
- Pipeline completed but some signals have low coverage
- Results are usable but some tickers may have incomplete scores

### Failure Indicators

```
RUN STATUS: FAIL - Pipeline health check FAILED
  ERROR: Circuit breaker tripped: 55% of financial records invalid
```

This means:
- Pipeline should NOT be used for trading
- Investigate data quality issues before re-running

---

## Checkpointing (Resume on Failure)

```bash
# Save checkpoints during run
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results.json \
    --checkpoint-dir ./checkpoints

# Resume from Module 3 (if Module 1-2 already completed)
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results.json \
    --checkpoint-dir ./checkpoints \
    --resume-from module_3
```

Checkpoints include integrity hashes - corrupted checkpoints are detected and ignored.

---

## Dry-Run Mode

Validate inputs without running the full pipeline:

```bash
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --dry-run
```

Output:
```
[DRY-RUN] Validating inputs for 2026-01-15...
Required files:
  universe.json: OK (85 records)
  financial_records.json: OK (312 records)
  trial_records.json: OK (1847 records)
  market_data.json: OK (85 records)

Content hashes:
  universe.json: abc123def456
  financial_records.json: 789ghi012jkl
  ...

Errors: None
```

---

## Failure Modes & Recovery

### 1. Missing Required Data

**Error**: `FileNotFoundError: Universe data file not found`

**Fix**: Ensure all required files exist in `--data-dir`. Run `--dry-run` first.

### 2. Invalid Tickers

**Error**: `ValueError: Universe contains 5 invalid tickers`

**Fix**: Run the ticker cleaning script:
```bash
python src/scripts/clean_universe.py --input production_data/universe.json
```

### 3. Circuit Breaker Tripped

**Error**: `CircuitBreakerError: 55% of records failed validation`

**Cause**: Data quality is too poor to produce reliable rankings.

**Fix**:
1. Check data collection pipeline for issues
2. Verify data sources are up-to-date
3. Review specific validation failures in logs
4. Consider running with `--strict-mode=False` (not recommended for production)

### 4. PIT Violation

**Error**: `PITViolationError: Record has source_date after PIT cutoff`

**Cause**: Data contains future information relative to `--as-of-date`.

**Fix**:
1. Verify `--as-of-date` is correct
2. Check data collection timestamps
3. Ensure data snapshot is older than `--as-of-date`

### 5. Checkpoint Integrity Failure

**Error**: `IntegrityError: Checkpoint corrupted or tampered`

**Fix**: Delete corrupted checkpoint and re-run from earlier module:
```bash
rm checkpoints/module_3/2026-01-15.json
python run_screen.py ... --resume-from module_2
```

---

## Audit Trail

Enable audit logging for compliance:

```bash
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results.json \
    --audit-log ./audit/2026-01-15.jsonl
```

Audit log contains:
- Input file content hashes
- Parameter snapshots
- Module versions
- Timestamps (deterministic, based on as_of_date)

---

## Reproducibility Verification

To verify two runs are identical:

```bash
# Run 1
python run_screen.py --as-of-date 2026-01-15 --data-dir prod --output run1.json

# Run 2 (same inputs)
python run_screen.py --as-of-date 2026-01-15 --data-dir prod --output run2.json

# Verify
python -c "
import json
import hashlib

with open('run1.json') as f:
    h1 = hashlib.sha256(f.read().encode()).hexdigest()
with open('run2.json') as f:
    h2 = hashlib.sha256(f.read().encode()).hexdigest()

print(f'Run 1 hash: {h1}')
print(f'Run 2 hash: {h2}')
print(f'Identical: {h1 == h2}')
"
```

Outputs should be byte-identical if:
- Same `--as-of-date`
- Same input data files
- Same code version

---

## Performance Guidelines

| Universe Size | Expected Time | Memory |
|--------------|---------------|--------|
| 50 tickers | ~30 seconds | 500MB |
| 100 tickers | ~1 minute | 800MB |
| 500 tickers | ~5 minutes | 2GB |

For large universes, consider:
- Increasing checkpoint frequency
- Running on machine with ≥4GB RAM
- Using `--timeout 3600` (1 hour)

---

## Configuration Files

### Key Settings in `config.yml`

```yaml
# Scoring weights
scoring_weights:
  clinical: 0.40
  financial: 0.25
  catalyst: 0.15
  other: 0.20

# Quality gates
module_params:
  market_cap_min_mm: 50
  burn_rate_threshold: -0.15
  min_adv_dollars: 500000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IC_VALIDATION_MODE` | `strict` | Schema validation: `strict`, `warn`, `off` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Contact & Support

- Issues: https://github.com/[org]/biotech-screener/issues
- Internal Slack: #biotech-screener-support

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.4.0 | 2026-01 | Ticker validation, PIT safety fixes |
| 1.3.0 | 2026-01 | V3 IC enhancements |
| 1.2.0 | 2025-12 | Defensive overlays |
