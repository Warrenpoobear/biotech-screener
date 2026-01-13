# Point-in-Time (PIT) Snapshot Architecture

## Purpose

Enable proper historical validation by capturing fundamentals at each screen date.
Eliminates look-ahead bias in backtesting.

## Current Problem

```
rankings_2024.csv + returns_2022.csv = INVALID (look-ahead bias)
rankings_2022.csv + returns_2022.csv = VALID (contemporaneous)
```

We only have current fundamentals, not historical snapshots.

## Proposed Architecture

### Directory Structure

```
data/
├── snapshots/
│   ├── 2022-01-15/
│   │   ├── universe.json       # Tickers active on this date
│   │   ├── financials.json     # Cash, debt, burn rate
│   │   ├── clinical.json       # Trial stages, phases
│   │   ├── catalysts.json      # Upcoming events
│   │   └── holdings_13f.json   # Institutional positions
│   ├── 2022-04-15/
│   │   └── ...
│   ├── 2023-01-15/
│   │   └── ...
│   └── latest/
│       └── ... (symlink to most recent)
├── returns/
│   └── returns_db_2020-01-01_2026-01-13.json
└── rankings/
    ├── rankings_2022-01-15.csv  # Generated from 2022-01-15 snapshot
    ├── rankings_2023-01-15.csv
    └── rankings_2024-01-15.csv
```

### Snapshot Schema

#### universe.json
```json
{
  "as_of_date": "2022-01-15",
  "tickers": ["VRTX", "REGN", "ALNY", ...],
  "count": 310,
  "source": "biotech_universe_filter"
}
```

#### financials.json
```json
{
  "as_of_date": "2022-01-15",
  "data_source": "SEC_10Q_10K",
  "tickers": {
    "VRTX": {
      "cash": 8500000000,
      "debt": 1200000000,
      "quarterly_burn": 150000000,
      "runway_months": 56,
      "market_cap": 65000000000,
      "filing_date": "2021-11-05"
    }
  }
}
```

#### clinical.json
```json
{
  "as_of_date": "2022-01-15",
  "data_source": "clinicaltrials_gov",
  "tickers": {
    "VRTX": {
      "lead_stage": "COMMERCIAL",
      "pipeline_count": 12,
      "phase3_count": 3,
      "phase2_count": 5,
      "phase1_count": 4
    }
  }
}
```

#### catalysts.json
```json
{
  "as_of_date": "2022-01-15",
  "tickers": {
    "VRTX": {
      "next_catalyst": "2022-03-15",
      "catalyst_type": "PDUFA",
      "days_to_catalyst": 59
    }
  }
}
```

#### holdings_13f.json
```json
{
  "as_of_date": "2022-01-15",
  "filing_quarter": "2021-Q3",
  "tickers": {
    "VRTX": {
      "institutional_holders": 1250,
      "institutional_ownership_pct": 92.5,
      "top_holders": ["Vanguard", "BlackRock", "State Street"]
    }
  }
}
```

## Data Sources

| Data Type | Source | Update Frequency | Historical Availability |
|-----------|--------|------------------|------------------------|
| Financials | SEC EDGAR 10-Q/10-K | Quarterly | 10+ years |
| Clinical | ClinicalTrials.gov API | Weekly | 2000-present |
| Catalysts | SEC 8-K, press releases | Daily | 5+ years |
| Holdings | SEC 13F filings | Quarterly | 10+ years |
| Prices | Yahoo Finance | Daily | 20+ years |

## Implementation Phases

### Phase 1: Capture Current (Week 1)
```python
# snapshot_collector.py
def capture_snapshot(as_of_date: str):
    """Capture current state as a snapshot."""
    snapshot = {
        'universe': collect_universe(),
        'financials': collect_financials(),
        'clinical': collect_clinical(),
        'catalysts': collect_catalysts(),
        'holdings': collect_13f_holdings()
    }
    save_snapshot(as_of_date, snapshot)
```

### Phase 2: Reconstruct Historical (Weeks 2-4)
```python
# historical_reconstructor.py
def reconstruct_snapshot(target_date: str):
    """Reconstruct point-in-time state from historical sources."""

    # Financials: Find most recent 10-Q/10-K before target_date
    financials = get_sec_filings_before(target_date)

    # Clinical: Query ClinicalTrials.gov archive
    clinical = get_trials_as_of(target_date)

    # Holdings: Find most recent 13F before target_date
    holdings = get_13f_before(target_date)

    return combine_snapshot(financials, clinical, holdings)
```

### Phase 3: Automated Pipeline (Weeks 4-6)
```python
# scheduled_capture.py
def weekly_snapshot():
    """Run every Sunday to capture weekly state."""
    today = date.today().isoformat()
    capture_snapshot(today)
    generate_rankings(today)
    validate_rankings(today)
```

## Validation Workflow

### With PIT Snapshots
```bash
# Generate rankings from historical snapshot
python generate_rankings.py --snapshot-date 2022-01-15

# Validate against forward returns
python validate_signals.py \
  --ranked-list rankings/rankings_2022-01-15.csv \
  --screen-date 2022-01-15 \
  --forward-months 6

# Result: Valid (no look-ahead bias)
```

### Walk-Forward Test
```bash
# Test across multiple periods
for date in 2022-01-15 2022-07-15 2023-01-15 2023-07-15 2024-01-15; do
    python generate_rankings.py --snapshot-date $date
    python validate_signals.py --ranked-list rankings/rankings_${date}.csv \
                               --screen-date $date --forward-months 6
done
```

## Priority Data Sources

### Must Have (Phase 1-2)
1. **SEC EDGAR** - Financials (cash, debt, burn)
2. **ClinicalTrials.gov** - Trial stages
3. **Yahoo Finance** - Prices (already have)

### Nice to Have (Phase 3)
4. **SEC 13F** - Institutional holdings
5. **FDA calendars** - PDUFA dates
6. **Press releases** - Catalyst announcements

## Estimated Effort

| Phase | Effort | Outcome |
|-------|--------|---------|
| Phase 1 | 1 week | Current snapshot capture working |
| Phase 2 | 3 weeks | 8 historical snapshots (quarterly 2022-2024) |
| Phase 3 | 2 weeks | Automated weekly capture + validation |

## Success Criteria

After implementation:
- [ ] Can generate rankings for any historical date
- [ ] Walk-forward validation shows consistent +5-15% Q1-Q5 spread
- [ ] No look-ahead bias in any validation
- [ ] Automated weekly snapshot capture running
