# Wake Robin Data Pipeline

**Production-grade data collection for biotech investment screening**

Systematically collects real-time data from free, public sources for your biotech universe. Built on the "governed over smart" philosophy with full auditability, determinism, and provenance tracking.

## Features

✅ **Three Data Sources** (all free, no API keys required):
- **Yahoo Finance**: Market prices, volume, valuation metrics
- **SEC EDGAR**: Quarterly financials, balance sheets, cash positions
- **ClinicalTrials.gov**: Trial status, phases, enrollment, conditions

✅ **Enterprise-Grade Infrastructure**:
- 24-hour caching to respect rate limits
- Automatic retry logic with exponential backoff
- Atomic file operations for data integrity
- Full provenance tracking (every data point sourced)
- Data quality scoring and reporting

✅ **Production Ready**:
- Tested on 20-ticker pilot universe
- Scales to 281+ ticker universe
- Weekly operational cadence
- Comprehensive error handling

## Quick Start

### 1. Installation

```bash
# Clone or download this directory
cd wake_robin_data_pipeline

# Install dependencies (Python 3.8+ required)
pip install -r requirements.txt
```

### 2. Run Data Collection

```bash
# Collect data for entire pilot universe (20 companies)
python collect_universe_data.py
```

Expected runtime: **~2-3 minutes** for 20 tickers (with caching: ~30 seconds)

### 3. View Results

```bash
# Check the latest snapshot
cat outputs/universe_snapshot_latest.json | head -100

# Check data quality report
cat outputs/quality_report_latest.json
```

## Directory Structure

```
wake_robin_data_pipeline/
├── collect_universe_data.py   # Main orchestrator
├── collectors/
│   ├── yahoo_collector.py     # Yahoo Finance integration
│   ├── sec_collector.py       # SEC EDGAR integration
│   └── trials_collector.py    # ClinicalTrials.gov integration
├── universe/
│   └── pilot_universe.json    # 20-ticker pilot configuration
├── cache/                      # 24-hour cached responses
│   ├── yahoo/
│   ├── sec/
│   └── trials/
└── outputs/                    # Timestamped snapshots
    ├── universe_snapshot_YYYYMMDD_HHMMSS.json
    ├── quality_report_YYYYMMDD_HHMMSS.json
    ├── universe_snapshot_latest.json (symlink)
    └── quality_report_latest.json (symlink)
```

## Data Quality Metrics

The pipeline generates comprehensive quality reports:

```json
{
  "universe_size": 20,
  "coverage": {
    "price_data": 20,      // 100% - Yahoo Finance is reliable
    "financial_data": 16,  // 80% - Some companies may not file with SEC
    "clinical_data": 18    // 90% - Varies by company pipeline
  },
  "avg_overall_coverage": 85.0,
  "tickers_by_quality": {
    "excellent": ["VRTX", "ALNY", "BMRN", ...],  // >80% coverage
    "good": ["GOSS", "RNA", ...],                // 60-80% coverage
    "fair": ["KYMR", ...],                       // 40-60% coverage
    "poor": ["EPRX"]                             // <40% coverage
  }
}
```

## Usage Examples

### Collect Data for Specific Tickers

```python
from collectors import yahoo_collector, sec_collector

# Single ticker
yahoo_data = yahoo_collector.collect_yahoo_data("VRTX")
sec_data = sec_collector.collect_sec_data("VRTX")

# Batch collection
tickers = ["VRTX", "ALNY", "BMRN"]
yahoo_results = yahoo_collector.collect_batch(tickers)
```

### Force Cache Refresh

```python
# Bypass 24-hour cache and fetch fresh data
data = yahoo_collector.collect_yahoo_data("VRTX", force_refresh=True)
```

### Add Your Own Tickers

Edit `universe/pilot_universe.json`:

```json
{
  "tickers": [
    {
      "ticker": "MYTICKER",
      "company": "My Biotech Company",
      "stage": "phase_2",
      "primary_indication": "Oncology",
      "market_cap_tier": "small"
    }
  ]
}
```

## Rate Limits & Respect

All collectors implement **1 request per second** rate limiting (conservative):

- **Yahoo Finance**: No official limit, but we respect their service
- **SEC EDGAR**: Official limit is 10 req/sec, we use 1 req/sec
- **ClinicalTrials.gov**: No stated limit, we use 1 req/sec

**Caching strategy**: 24-hour cache minimizes repeated requests. On subsequent runs within 24 hours, only changed data is re-fetched.

## Data Provenance Example

Every data point includes full provenance:

```json
{
  "ticker": "VRTX",
  "market_data": {
    "price": 425.67,
    "market_cap": 110500000000
  },
  "provenance": {
    "collection_timestamp": "2025-01-05T10:30:00",
    "sources": {
      "yahoo_finance": {
        "timestamp": "2025-01-05T10:29:45",
        "url": "https://finance.yahoo.com/quote/VRTX",
        "data_hash": "a1b2c3d4e5f6g7h8"
      },
      "sec_edgar": {
        "timestamp": "2025-01-05T10:29:50",
        "url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000875320.json",
        "cik": "0000875320"
      }
    }
  }
}
```

## Operational Cadence

**Weekly Update Schedule**:
1. **Tuesday**: Run `collect_universe_data.py` to get fresh data
2. **Wednesday**: Review quality report, fix any failures
3. **Thursday**: Feed snapshot into scoring model
4. **Friday**: Generate ranked lists and dossiers

## Scaling to Full Universe

To scale from 20-ticker pilot to full 281-ticker universe:

1. **Update universe file**: Replace `pilot_universe.json` with full list
2. **Adjust timing**: Expect ~20-30 minutes for full run (first time)
3. **Monitor quality**: Track coverage metrics, investigate failures
4. **Optimize caching**: 24-hour cache means subsequent runs take ~3-5 minutes

## Troubleshooting

### "yfinance not installed"
```bash
pip install yfinance
```

### "Could not resolve ticker to CIK"
Some tickers may not be in SEC database (foreign companies, recent IPOs). This is expected - the system handles it gracefully.

### "Network timeout"
Increase timeout in collector code or check internet connection. The system will retry automatically.

### "No data in cache"
First run always fetches fresh data. Subsequent runs within 24 hours use cache.

## Next Steps

After collecting real data:

1. **Feed into scoring model**: Use snapshot as input to Wake Robin scoring system
2. **Generate rankings**: Apply financial coverage gates, catalyst weighting, PoS models
3. **Create dossiers**: Build IC-ready investment packages for each ticker
4. **Track changes**: Compare snapshots over time to detect signal changes

## Technical Notes

- **Determinism**: Same inputs → same outputs (via content hashing)
- **Atomicity**: File operations are atomic (no partial writes)
- **Fail-loud**: Errors are explicit, not silent
- **Auditability**: Every claim traces to primary source
- **Windows Compatible**: All file operations work on Windows/Mac/Linux

## License

Internal use only for Wake Robin Capital Management.

## Support

Questions? Issues? Check the quality report first - it often reveals the root cause of data problems.
