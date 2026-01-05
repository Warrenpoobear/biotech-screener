# Production Policy - Wake Robin Biotech Screener

## Overview

This document defines the recommended production policies for running the biotech screener in an automated, deterministic, and auditable manner.

---

## Data Ingestion Policy

### Sync Schedule

| Task | Frequency | Purpose |
|------|-----------|---------|
| **Daily Incremental** | 6:00 AM daily | Small delta sync, fast |
| **Weekly Full Refresh** | Sunday 5:00 AM | Catch corrections, eliminate drift |
| **Backtest Run** | 1st of month, 7:00 AM | Monthly measurement |

### Sync Modes

**Incremental (default)**
- Fetches only new dates since last sync
- Uses 5-day overlap to catch corrections
- Fast and bandwidth-efficient

**Full Refresh**
- Fetches complete history from start date
- Use weekly to ensure data integrity
- Use for initial setup or disaster recovery

### Data Files

| File | Purpose | Mutability |
|------|---------|------------|
| `data/raw/sharadar_sep/slice_*.csv` | Raw ingested slices | Immutable |
| `data/sharadar_sep.csv` | Curated, deduplicated | Rebuilt from raw |
| `data/curated/sharadar/state.json` | Sync state | Updated on success |
| `data/curated/sharadar/ingest_manifest.json` | Audit trail | Updated on success |

---

## Backtest Policy

### Run Cadence

- **Monthly**: 1st of each month after sync completes
- **Ad-hoc**: After significant data corrections or code changes

### Determinism Requirements

1. **Run ID**: Derived from config + data hashes (no timestamps)
   ```
   run_id = sharadar_{config_hash[:12]}_{data_hash[:12]}
   ```

2. **Reproducibility**: Same inputs → identical outputs
   - Same config → same config_hash
   - Same data file → same data_hash
   - Same run_id → same output folder

3. **What's hashed** (affects run_id):
   - Config parameters (start_year, end_year, delisting_policy, etc.)
   - Data file path (content hash in production)

4. **What's NOT hashed** (doesn't affect run_id):
   - Timestamps (stored as `created_at` in config.json)
   - Log messages
   - Cache hits/misses

### Gates (Must Pass Before Interpretation)

| Gate | Threshold | Action if Fail |
|------|-----------|----------------|
| Preflight | Schema + coverage | Stop, fix data |
| Return coverage | ≥ 80% | Stop, expand data |
| Fallback rate | ≤ 20% | Stop, expand universe |
| As-of alignment | 0 violations | Stop, fix calendar |
| Rank stability | ≥ 0.35 | Warn, investigate |
| Top-quintile churn | ≤ 65% | Warn, investigate |

---

## Observability

### Logs

| Location | Contents |
|----------|----------|
| `logs/sync_YYYYMMDD.log` | Sync execution logs |
| `logs/pipeline_YYYYMMDD.log` | Full pipeline logs |
| `output/manifests/run_manifest.jsonl` | Append-only run audit |

### Artifacts per Run

```
output/runs/{run_id}/
├── config.json               # Exact configuration
├── data_readiness.json       # Preflight results
├── run_summary.json          # Quality gates
├── sanity_metrics.json       # Research metrics
├── stability_attribution.json # Rank change analysis
└── backtest_results.json     # Full metrics
```

### Monitoring Checklist

Weekly:
- [ ] Check `state.json` for last_date progression
- [ ] Review sync logs for errors
- [ ] Verify `sharadar_sep.csv` file size is reasonable

Monthly (after backtest):
- [ ] All quality gates passed
- [ ] Stability gates in expected range
- [ ] IC sign consistent across delisting policies
- [ ] No unexpected stage IC patterns

---

## Failure Recovery

### Sync Failure

1. Check logs: `logs/sync_*.log`
2. Common causes:
   - API key expired → renew key
   - Rate limit hit → wait and retry
   - Network error → retry
3. Recovery: `scripts\run_sharadar_sync.bat` (incremental catches up)

### Backtest Failure

1. Check preflight: `output/runs/*/data_readiness.json`
2. Common causes:
   - Missing tickers → expand SEP export
   - Date range too short → extend data
   - Schema mismatch → check column mapping
3. Recovery: Fix data, re-run backtest

### Data Corruption

1. Delete curated file: `data\sharadar_sep.csv`
2. Full refresh: `scripts\run_sharadar_sync.bat --full-refresh`
3. Re-run backtest

---

## Security

### API Key Management

- Store as system environment variable (not in code)
- Never commit to version control
- Rotate periodically

### Access Control

- Limit write access to data directories
- Audit log access regularly
- Backup manifests and state files

---

## Version Control

### What to Commit

- All Python code
- Batch/PowerShell scripts
- Documentation
- Test files

### What NOT to Commit

- `data/` directory (large, sensitive)
- `output/runs/` (regeneratable)
- `logs/` (operational)
- API keys

### .gitignore Recommendations

```
data/
output/runs/
logs/
*.pyc
__pycache__/
.env
*api_key*
```

---

## Disaster Recovery

### Minimum Recovery Set

To rebuild from scratch, you need:
1. Code repository
2. Sharadar API access
3. Universe ticker list

### Recovery Steps

1. Clone repository
2. Set API key
3. Run full refresh: `--full-refresh --start-date 2020-01-01`
4. Run backtest
5. Verify gates pass

---

## Change Management

### Before Any Code Change

1. Run existing tests: `python -m pytest tests/`
2. Save current run_id for comparison

### After Code Change

1. Run tests again
2. Re-run backtest with same config
3. Compare run_ids (should be same if deterministic)
4. Compare key metrics (IC, spreads)
5. Document any expected differences

---

*Wake Robin Capital Management - Biotech Stock Screener v1.6.0*
