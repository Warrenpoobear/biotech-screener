# Morningstar Returns Integration - Phase 1

Production-ready backtesting framework for validating screening signals against actual forward returns.

## Overview

This integration allows you to:
1. **Build** a returns database from Morningstar Direct (requires token)
2. **Validate** screening signals against forward returns (NO token required)
3. **Measure** hit rate, alpha, and quintile performance

## Architecture: "Fetch Once, Use Forever"

```
Token Required          Token NOT Required
===============        ====================
↓                      ↓
Fetch from API    →    Cache to disk    →    Read from cache
(Once)                 (JSON files)          (Unlimited)
```

Key insight: **99% of your usage requires NO token**. You only need a token when building or updating the database.

## Quick Start

### Step 1: Verify Installation

```powershell
# Set your Morningstar Direct token
$env:MD_AUTH_TOKEN="your-token-here"

# Run the quickstart test
python quickstart_test.py
```

Expected output:
```
[PASS] MD_AUTH_TOKEN is set
[PASS] Module imported successfully
[PASS] morningstar-data package available
[PASS] Morningstar Direct initialized
[PASS] Retrieved 12 observations
[SUCCESS] All tests passed!
```

### Step 2: Build Returns Database (Token Required)

```powershell
# Build database for your universe
python build_returns_database.py \
  --universe example_universe.csv \
  --start-date 2020-01-01
```

This creates: `data/returns/returns_db_2020-01-01_YYYY-MM-DD.json`

**Time**: ~5-30 minutes depending on universe size

### Step 3: Validate Signals (NO Token Required)

```powershell
# Test if your screening signals predict returns
python validate_signals.py \
  --database data/returns/returns_db.json \
  --ranked-list your_screen_output.csv \
  --screen-date 2023-01-15 \
  --forward-months 6
```

Run this **unlimited times** without needing a token!

## Files Included

| File | Description | Token Required? |
|------|-------------|-----------------|
| `morningstar_returns.py` | Core module with fetcher and database classes | For fetching only |
| `build_returns_database.py` | CLI to build/update the returns database | Yes |
| `validate_signals.py` | CLI to validate signals against returns | **No** |
| `quickstart_test.py` | Verification script to test setup | Yes (for full test) |
| `example_universe.csv` | Sample 10-ticker universe for testing | N/A |

## Validation Output Example

```
============================================================
SIGNAL VALIDATION REPORT
============================================================

Screen Date:      2023-01-15
Forward Period:   6 months
Tickers Tested:   47/50

------------------------------------------------------------
PERFORMANCE METRICS
------------------------------------------------------------
  Hit Rate:          64.6%  (% profitable)
  Average Return:    +8.32%
  Median Return:     +6.10%
  Std Deviation:     +18.45%

------------------------------------------------------------
BENCHMARK COMPARISON (vs XBI)
------------------------------------------------------------
  XBI Return:        +4.21%
  Average Alpha:     +4.11%

------------------------------------------------------------
QUINTILE ANALYSIS
------------------------------------------------------------
  (Q1 = top-ranked, Q5 = bottom-ranked)

  Q1:     +14.20%  TOP
  Q2:      +9.30%
  Q3:      +7.50%
  Q4:      +4.80%
  Q5:      +2.30%  BOTTOM

  Q1-Q5 Spread:      +11.90%

------------------------------------------------------------
INTERPRETATION
------------------------------------------------------------
  [GOOD] Hit rate 64.6% > 55% indicates useful signal
  [STRONG] 11.9% Q1-Q5 spread indicates strong ranking power
  [ALPHA] 4.1% average alpha vs XBI
```

## Token Management

### When You Need a Token

| Activity | Token Needed? | Frequency |
|----------|---------------|-----------|
| Build initial database | Yes | Once |
| Run validation | **No** | Unlimited |
| Monthly database update | Yes | Monthly |
| Analysis/backtesting | **No** | Unlimited |
| Reading cached data | **No** | Always |

### Setting Your Token

**Windows PowerShell:**
```powershell
$env:MD_AUTH_TOKEN="your-token-here"
```

**Linux/Mac bash:**
```bash
export MD_AUTH_TOKEN="your-token-here"
```

### Token Expiration

Morningstar tokens typically expire after 24 hours. If your token expires mid-fetch:

1. Previous batches are already cached
2. Just refresh your token
3. Re-run the build command - it will continue from where it left off

## Data Quality Features

### Tier-0 Provenance
Every data point includes:
- SHA256 content hash
- Fetch timestamp
- Source version
- Complete audit trail

### Point-in-Time Discipline
- Only uses returns data AFTER the screen date
- Prevents look-ahead bias
- Ensures backtests reflect real-world conditions

### Atomic Operations
- All writes are atomic (temp file → rename)
- Prevents partial/corrupted files
- Safe for concurrent access

## API Reference

### MorningstarReturnsFetcher

```python
from morningstar_returns import MorningstarReturnsFetcher

# Requires MD_AUTH_TOKEN environment variable
fetcher = MorningstarReturnsFetcher(
    cache_dir=Path("data/returns"),  # Where to save database
    batch_size=20,                    # Tickers per API call
)

data = fetcher.fetch_returns(
    tickers=["VRTX", "BMRN", "ALNY"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    include_benchmark=True,  # Include XBI
)

fetcher.save_database(data)
```

### ReturnsDatabase (No Token Required)

```python
from morningstar_returns import ReturnsDatabase

# Works completely offline
db = ReturnsDatabase(Path("data/returns/returns_db.json"))

# Get available tickers
tickers = db.available_tickers  # ['ALNY', 'BMRN', 'VRTX', ...]

# Get date range
start, end = db.date_range  # ('2020-01-01', '2024-12-31')

# Get forward return
ret = db.get_forward_return("VRTX", "2023-01-15", forward_months=6)
# Returns: Decimal('0.082145') or None

# Get excess return vs XBI
alpha = db.get_excess_return("VRTX", "2023-01-15", forward_months=6)
# Returns: Decimal('0.041123') or None
```

## Troubleshooting

### "MD_AUTH_TOKEN not set"
Set your token before running:
```powershell
$env:MD_AUTH_TOKEN="your-token-here"
```

### "morningstar-data not installed"
Install the package:
```powershell
pip install morningstar-data
```

### "Token expired"
Get a fresh token from Morningstar Direct web interface and re-set it.

### "No returns data for period"
- Check that screen_date is within the database date range
- Check that forward period doesn't extend beyond database end date
- Verify tickers exist in the database

### "Connection failed"
- Check internet connectivity
- Verify token is valid
- Try again (may be transient network issue)

## Next Steps

### Phase 2: Multi-Date Validation
- Test signals across multiple historical dates
- Calculate aggregate statistics
- Identify regime-dependent performance

### Phase 3: Pattern Analysis
- Test each scoring pattern independently
- Ablation testing (remove each component)
- Optimize pattern weights based on evidence

### Phase 4: Walk-Forward Testing
- Monthly rebalancing simulation
- Full portfolio simulation vs XBI
- Drawdown and risk analysis

## Support

For issues with:
- **This integration**: Check the troubleshooting section above
- **Morningstar-data package**: See Morningstar documentation
- **Wake Robin screener**: See main project documentation
