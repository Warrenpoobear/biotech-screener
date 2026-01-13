# Morningstar Returns Integration - Phase 1
## Wake Robin Backtesting Framework

**Status**: Phase 1 Complete  
**Date**: January 12, 2026

This implementation provides historical returns data for backtesting Wake Robin's screening signals.

---

## What's Included

### Core Modules

1. **`morningstar_returns.py`** - Returns data fetcher
   - Fetches historical returns from Morningstar Direct
   - Tier-0 provenance tracking
   - Point-in-time discipline
   - Atomic caching with SHA256 verification
   - Uses `md.direct.returns()` (the working method)

2. **`build_returns_database.py`** - Database builder
   - Builds returns database for entire universe
   - Batch processing (20 tickers per batch)
   - Includes benchmark returns (XBI, IBB)
   - Complete provenance metadata

3. **`validate_signals.py`** - Signal validation
   - Tests if screening signals predict returns
   - Calculates hit rate, alpha, Sharpe ratio
   - Quintile analysis (top vs bottom)
   - Generates validation reports

---

## Installation

### Requirements

```bash
# Install Morningstar package
pip install morningstar-data --break-system-packages

# No other dependencies - uses stdlib only
```

### Authentication

```bash
# Set environment variable with your token
# Get token from: Morningstar Direct > Analytics Lab > Copy Authentication Token

export MD_AUTH_TOKEN="your-token-here"

# Or on Windows PowerShell:
$env:MD_AUTH_TOKEN="your-token-here"

# Token expires after 24 hours - refresh daily
```

---

## Quick Start

### Step 1: Build Returns Database

```bash
# Create a universe CSV file (ticker column required)
# Example: data/universe.csv with columns: ticker

python build_returns_database.py \
  --universe data/universe.csv \
  --start-date 2020-01-01 \
  --output-dir data/returns
```

This creates:
- `data/returns/returns_db_2020-01-01_YYYY-MM-DD.json` - Main database
- `data/returns/cache/batch_*.json` - Cached batch files

**Expected time**: ~5 minutes for 100 tickers

### Step 2: Validate Screening Signals

```bash
# You need a ranked list from a historical screen
# Format: CSV with 'ticker' column, JSON list, or text file

python validate_signals.py \
  --database data/returns/returns_db_2020-01-01_2024-12-31.json \
  --ranked-list historical_screen_2023-01-01.csv \
  --screen-date 2023-01-01 \
  --forward-months 6 \
  --output validation_results.json
```

**Output**:
```
VALIDATION REPORT
========================================
Screen Date: 2023-01-01
Forward Period: 6 months
Universe: 50 tickers
Valid Returns: 48

Performance Metrics:
  Hit Rate:           64.6%
  Average Return:     +8.32%
  Median Return:      +6.10%

Quintile Analysis:
  Top Quintile:       +14.20%
  Bottom Quintile:    +2.30%
  Spread:             +11.90%

Interpretation:
  ✓  GOOD: Hit rate >55% indicates useful signal
  ✅ STRONG: >10% spread indicates strong ranking
```

---

## Usage Examples

### Example 1: Test Module

```python
import morningstar_returns as mr

# Check availability
available, msg = mr.check_availability()
print(f"Morningstar: {available} - {msg}")

# Fetch returns for VRTX
data = mr.fetch_returns(
    sec_ids=['0P000005R7'],  # VRTX
    start_date='2024-01-01',
    end_date='2024-12-31',
    frequency='monthly',
    benchmark_sec_id='FEUSA04AER'  # XBI
)

print(f"Retrieved {data['provenance']['num_observations']} observations")

# Cache the data
path = mr.save_returns_cache(
    data,
    output_dir='data/returns/cache',
    filename='vrtx_2024.json'
)
```

### Example 2: Build Custom Database

```python
from build_returns_database import ReturnsDatabaseBuilder

# Initialize builder
builder = ReturnsDatabaseBuilder(output_dir='data/returns')

# Build for custom universe
tickers = ['VRTX', 'IONS', 'BMRN', 'RARE', 'REGN']
db_path = builder.build_database(
    ticker_universe=tickers,
    start_date='2020-01-01',
    end_date='2024-12-31'
)

print(f"Database: {db_path}")
```

### Example 3: Validate Historical Screen

```python
from validate_signals import BacktestValidator

# Load database
validator = BacktestValidator('data/returns/returns_db.json')

# Test ranked list
ranked_tickers = ['VRTX', 'IONS', 'BMRN', 'RARE']  # Your screen output
validation = validator.validate_ranked_list(
    ranked_tickers=ranked_tickers,
    screen_date='2023-01-01',
    forward_months=6
)

validator.print_validation_report(validation)
```

---

## Architecture

### Data Flow

```
1. Morningstar Direct API
   ↓
2. morningstar_returns.py (fetch + provenance)
   ↓
3. Cache (data/returns/cache/*.json)
   ↓
4. Database (data/returns/returns_db_*.json)
   ↓
5. Validation Framework
   ↓
6. Results (hit rate, alpha, etc.)
```

### Provenance Tracking

Every data point includes:

```json
{
  "provenance": {
    "source": "morningstar_direct",
    "fetch_timestamp": "2026-01-12T10:30:00Z",
    "data_start_date": "2020-01-01",
    "data_end_date": "2024-12-31",
    "frequency": "monthly",
    "benchmark_sec_id": "FEUSA04AER",
    "api_method": "md.direct.returns",
    "package_version": "1.13.0",
    "num_observations": 240,
    "sha256": "abc123..."
  }
}
```

### Point-in-Time Discipline

**Critical**: Never refetch historical data to prevent look-ahead bias.

```python
# ✅ CORRECT: Fetch once, cache forever
data = fetch_returns(start='2020-01-01', end='2024-12-31')
save_returns_cache(data, 'data/returns/historical_2020_2024.json')

# Later: Always load from cache
data = load_returns_cache('data/returns/historical_2020_2024.json')

# ❌ WRONG: Refetching introduces look-ahead bias
data = fetch_returns(start='2020-01-01', end='2024-12-31')  # DON'T DO THIS
```

---

## Testing

### Test Suite

```bash
# Test module directly
python morningstar_returns.py

# Expected output:
# ✅ Morningstar availability: True - Morningstar Direct available
# ✅ Retrieved 12 observations
# ✅ Saved to: /tmp/test_cache/vrtx_2024.json
# ✅ Loaded and verified SHA256
```

### Integration Test

Create `test_integration.py`:

```python
import os
os.environ['MD_AUTH_TOKEN'] = 'your-token-here'

import morningstar_returns as mr

# Test fetch
data = mr.fetch_returns(
    sec_ids=['0P000005R7'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

assert data['provenance']['num_observations'] > 0
print("✅ Integration test passed")
```

---

## Troubleshooting

### "MD_AUTH_TOKEN environment variable not set"

**Solution**: Set your authentication token:

```bash
# Get fresh token from Morningstar Direct
# Analytics Lab > Copy Authentication Token

export MD_AUTH_TOKEN="your-token-here"
```

### "Read timed out"

**Solution**: Reduce batch size or add delays:

```python
# In build_returns_database.py, line 129:
batch_size=10  # Reduce from 20 to 10
```

### "No securities found"

**Solution**: Check ticker symbols:
- Use exact ticker symbols (e.g., "VRTX" not "VERTEX")
- Some smaller biotech may not be in Morningstar
- Use SecId directly if known: `sec_ids=['0P000005R7']`

### "KRTX not found"

**Known issue**: Some clinical-stage biotech are not in Morningstar Direct.  
**Workaround**: Focus on larger/commercial biotech (>$1B market cap)

---

## Next Steps

### Phase 2: Basic Validation (Week 2)

Goal: Test if current screening signals work

**Tasks**:
1. Select 5 historical screen dates
2. Run validation for each date
3. Calculate aggregate metrics
4. Identify which signals work

**Questions to answer**:
- Do high-scoring tickers outperform?
- What's the overall hit rate?
- Is there alpha vs XBI benchmark?

### Phase 3: Pattern Analysis (Week 3)

Goal: Understand which patterns have alpha

**Tasks**:
1. Test each pattern independently (CASH_COW, RAMP, etc.)
2. Ablation testing (remove each pattern)
3. Optimize pattern weights

### Phase 4: Walk-Forward (Week 4)

Goal: Full historical simulation

**Tasks**:
1. Monthly rebalancing 2020-2024
2. Track cumulative returns
3. Drawdown analysis
4. Regime sensitivity

---

## Performance

### Benchmarks

**Database build** (100 tickers, 2020-2024):
- Fetch time: ~5 minutes
- Database size: ~5 MB
- Cache files: ~500 KB each

**Validation** (50 tickers):
- Load database: <1 second
- Calculate metrics: <1 second
- Total: ~1 second

### Scaling

**Supported universe sizes**:
- Small (50 tickers): 2-3 minutes
- Medium (200 tickers): 10-15 minutes
- Large (500 tickers): 30-45 minutes

**Rate limiting**:
- 0.1s delay between ticker searches
- 1.0s delay between batch fetches
- Adjust if hitting timeouts

---

## Support

### Common Issues

**Token expired**: Refresh from Analytics Lab (valid 24 hours)

**API timeouts**: Reduce batch size or add delays

**Missing tickers**: Not all biotech in Morningstar, focus on larger names

### Documentation

- Morningstar Direct: Help > Morningstar Data Python Reference
- Wake Robin: See PROJECT_README.md, SYSTEM_STATE.md

---

## License & Compliance

**Data Source**: Morningstar Direct (institutional license required)

**Usage**: Internal use only per Morningstar Direct license terms

**Redistribution**: Not permitted (raw Morningstar data cannot be redistributed)

**Audit Trail**: Complete provenance tracking for regulatory compliance

---

## Version History

**v1.0.0** (2026-01-12)
- Initial Phase 1 implementation
- Core returns fetching with Tier-0 provenance
- Database builder for universe
- Basic validation framework
- Stdlib-only dependencies (except morningstar-data)

---

## Contact

For questions about this implementation, refer to:
- RETURNS_INTEGRATION_ARCHITECTURE.md (design document)
- MORNINGSTAR_INTEGRATION_GUIDE.md (original planning)
- Wake Robin project documentation in /mnt/project/
