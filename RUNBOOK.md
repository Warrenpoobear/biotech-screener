# Biotech Screener Runbook

## Quick Reference

| Task | Command |
|------|---------|
| Health check | `python doctor.py` |
| Run pipeline | `python run_screen.py --as-of-date YYYY-MM-DD --data-dir production_data --output results.json` |
| Validate output | `python validate_pipeline.py --output results.json` |
| Run tests | `pytest tests/test_minimum_suite.py -v` |
| Create baseline | `pytest tests/test_golden_baseline.py::test_create_baseline -v` |

---

## 1. Prerequisites

### Check Your Environment

```bash
python doctor.py
```

This checks:
- Python version (3.10+ required)
- Dependencies installed
- Input files present
- Schemas valid
- Config file correct

**If FAIL**, fix the errors shown before proceeding.

### Required Input Files

All files should be in `production_data/`:

| File | Required | Description |
|------|----------|-------------|
| `universe.json` | Yes | List of tickers with metadata |
| `financial_records.json` | Yes | Financial data (Cash, NetIncome, etc.) |
| `trial_records.json` | Yes | Clinical trial data from CT.gov |
| `market_data.json` | Yes | Price, volume, market cap |
| `short_interest.json` | No | Short interest signals |
| `market_snapshot.json` | No | VIX, regime data |

---

## 2. Running the Pipeline

### Basic Run

```bash
python run_screen.py \
  --as-of-date 2026-01-20 \
  --data-dir production_data \
  --output results.json
```

### With All Features

```bash
python run_screen.py \
  --as-of-date 2026-01-20 \
  --data-dir production_data \
  --output results.json \
  --enable-enhancements \
  --checkpoint-dir checkpoints \
  --audit-log audit.jsonl
```

### Dry Run (Validate Only)

```bash
python run_screen.py \
  --as-of-date 2026-01-20 \
  --data-dir production_data \
  --dry-run
```

---

## 3. Validating Results

### Quick Validation

```bash
python validate_pipeline.py --output results.json
```

### Compare to Baseline

```bash
python validate_pipeline.py --output results.json --baseline golden/baseline_output.json
```

### Strict Mode (Warnings = Errors)

```bash
python validate_pipeline.py --output results.json --strict
```

---

## 4. Common Errors and Fixes

### Error: "PIT violation" / "Future data"

**Cause**: The `as_of_date` is before the data's last update date.

**Fix Options**:
1. Use a more recent `as_of_date`
2. The pipeline now filters future data automatically (1.4% tolerance)
3. Refresh your data files with older data for backtesting

### Error: "Missing required file"

**Cause**: One of the required input files is missing.

**Fix**: Ensure all required files exist in `production_data/`:
```bash
ls production_data/*.json
```

### Error: "All Module 2 scores are zero"

**Cause**: Financial data missing for all tickers.

**Fix**:
1. Check `financial_records.json` has data
2. Ensure ticker names match between universe and financial data
3. Check for uppercase/lowercase ticker mismatches

### Error: "Weights don't sum to target"

**Cause**: Position sizing issue in Module 5.

**Fix**: Check that there are enough rankable (non-excluded) tickers. SEV3 tickers are excluded.

### Warning: "Universe size changed"

**Cause**: The number of tickers changed from baseline.

**Why this happens**:
- Added/removed tickers from universe.json
- Market cap filtering changed
- Shell company detection changed

**Action**: Review if the change is expected.

---

## 5. Testing

### Run All Tests

```bash
pytest tests/test_minimum_suite.py -v
```

### Run Specific Test Categories

```bash
# Smoke tests only
pytest tests/test_minimum_suite.py -v -k smoke

# Schema tests
pytest tests/test_minimum_suite.py -v -k schema

# PIT tests
pytest tests/test_minimum_suite.py -v -k pit
```

### Create/Update Golden Baseline

```bash
pytest tests/test_golden_baseline.py::test_create_baseline -v
```

### Run Regression Test

```bash
pytest tests/test_golden_baseline.py::test_output_matches_baseline -v
```

---

## 6. Configuration

Configuration lives in `config.yml`. Key settings:

### Module 2 (Financial Health)

```yaml
module_2:
  weights:
    runway: 0.50      # Cash runway weight
    dilution: 0.30    # Dilution risk weight
    liquidity: 0.20   # Liquidity weight

  runway_thresholds:
    sev3_critical: 6  # Months for SEV3 (excluded)
    sev2_warning: 12  # Months for SEV2
    sev1_caution: 18  # Months for SEV1
```

### Module 5 (Composite)

```yaml
module_5:
  weights:
    clinical: 0.40    # Clinical development weight
    financial: 0.35   # Financial health weight
    catalyst: 0.25    # Catalyst momentum weight

  position_sizing:
    max_positions: 60
    target_weight_sum: 0.90
```

---

## 7. Output Interpretation

### Summary Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `total_evaluated` | Tickers in universe | 100-500 |
| `active_universe` | After status filtering | 80-90% of total |
| `final_ranked` | After severity filtering | 30-50% of active |
| `catalyst_events` | Events detected | 50-200 |
| `severe_negatives` | Bad trial news | 0-5 |

### Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| `none` | Healthy (18+ months runway) | Fully rankable |
| `sev1` | Caution (12-18 months) | Minor penalty |
| `sev2` | Warning (6-12 months) | Soft gate (capped) |
| `sev3` | Critical (<6 months) | Hard gate (excluded) |

### Exclusion Reasons

| Reason | Cause |
|--------|-------|
| `sev3_gate` | Critical financial health |
| `shell_company` | Detected as SPAC/shell |
| `delisted` | Ticker no longer active |

---

## 8. Troubleshooting Checklist

If the pipeline fails:

1. [ ] Run `python doctor.py` - fix any errors
2. [ ] Check `as_of_date` is not in the future
3. [ ] Verify all required files exist
4. [ ] Check file permissions (can write to output dir)
5. [ ] Review error message for specific module failure
6. [ ] Check `production_data/run_log_*.json` for details
7. [ ] Try with `--dry-run` to validate inputs first

If output looks wrong:

1. [ ] Run `python validate_pipeline.py --output results.json`
2. [ ] Check severity distribution in Module 2
3. [ ] Verify data freshness (staleness warnings)
4. [ ] Compare to baseline if available
5. [ ] Review excluded tickers and reasons

---

## 9. Data Refresh Workflow

### Weekly Refresh

```bash
# 1. Refresh trial data from CT.gov
python collect_ctgov_data.py --output production_data/trial_records.json

# 2. Refresh market data
python collect_market_data.py --output production_data/market_data.json

# 3. Run pipeline
python run_screen.py --as-of-date $(date +%Y-%m-%d) \
  --data-dir production_data \
  --output results_$(date +%Y-%m-%d).json

# 4. Validate
python validate_pipeline.py --output results_$(date +%Y-%m-%d).json
```

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `run_screen.py` | Main pipeline orchestrator |
| `doctor.py` | Health check |
| `validate_pipeline.py` | Output validation |
| `config.yml` | Configuration |
| `requirements.txt` | Dependencies |
| `module_1_universe.py` | Universe filtering |
| `module_2_financial.py` | Financial health |
| `module_3_catalyst.py` | Catalyst detection |
| `module_4_clinical_dev.py` | Clinical development |
| `module_5_composite_with_defensive.py` | Composite ranking |

---

## Contact

For issues, see: https://github.com/Warrenpoobear/biotech-screener/issues
