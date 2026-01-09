# CUSIP Mapper Quick-Start Guide
## Wake Robin Biotech Screening System

**Purpose:** Map 13F filing CUSIPs to tickers in your biotech universe  
**Date:** 2026-01-09

---

## Quick Start (5 Minutes)

### Step 1: Initialize Mapper Files

```bash
cd /path/to/biotech-screener

# Create mapper files
python cusip_mapper.py init --data-dir production_data

# Output:
# Empty static map created: production_data/cusip_static_map.json
# Empty cache created: production_data/cusip_cache.json
```

### Step 2: Get OpenFIGI API Key (Free)

1. Visit: https://www.openfigi.com/api
2. Register for free API key
3. Check email for API key
4. Save to environment variable:

```bash
export OPENFIGI_API_KEY="your-api-key-here"
```

Or save to config file:
```bash
echo "your-api-key-here" > production_data/openfigi_api_key.txt
```

### Step 3: Test Single CUSIP

```bash
# Test with Apple (AAPL) CUSIP
python cusip_mapper.py query 037833100 \
    --data-dir production_data \
    --api-key $OPENFIGI_API_KEY

# Output:
# 037833100 → AAPL
#   Name: Apple Inc
#   Exchange: NASDAQ
#   Type: Common Stock
#   Source: openfigi
```

---

## Three-Tier Caching Strategy

```
Query CUSIP → Ticker
      ↓
[Tier 0: Static Map] ───→ Found? → Return instantly
      ↓ Miss
[Tier 1: Cache] ─────────→ Found? → Return instantly
      ↓ Miss                         (valid 90 days)
[Tier 2: OpenFIGI API] ──→ Query → Cache result → Return
      (rate-limited)
```

**Performance:**
- Tier 0/1 Hit: Instant (microseconds)
- Tier 2 Miss: 2.5 seconds per CUSIP (rate-limited)

**Strategy:**
- Pre-populate static map with known biotechs (one-time effort)
- Cache fills automatically via OpenFIGI
- 90-day TTL ensures stale data refreshes quarterly

---

## Populate Static Map (Recommended)

### Option 1: Manual Entry (Most Accurate)

Create `production_data/cusip_static_map.json`:

```json
{
  "037833100": {
    "cusip": "037833100",
    "ticker": "AAPL",
    "name": "Apple Inc",
    "exchange": "NASDAQ",
    "security_type": "Common Stock",
    "mapped_at": "2026-01-09T00:00:00",
    "source": "static"
  }
}
```

**Data Sources:**
1. **SEC EDGAR** - Company filings list CUSIP
2. **Yahoo Finance** - Look up ticker → CUSIP in profile
3. **Bloomberg Terminal** - If you have access
4. **QuantConnect** - Free CUSIP database
5. **Cusip.com** - Official but paid

### Option 2: Batch Import from CSV

If you have a CSV with ticker-CUSIP pairs:

```csv
ticker,cusip,name
AAPL,037833100,Apple Inc
NVAX,670002401,Novavax Inc
ARGX,03969T105,Argenx SE
```

Import script:
```python
import json
import csv
from datetime import datetime

with open('biotech_cusips.csv') as f:
    reader = csv.DictReader(f)
    
    static_map = {}
    for row in reader:
        cusip = row['cusip']
        static_map[cusip] = {
            'cusip': cusip,
            'ticker': row['ticker'],
            'name': row['name'],
            'exchange': 'NASDAQ',  # Adjust as needed
            'security_type': 'Common Stock',
            'mapped_at': datetime.now().isoformat(),
            'source': 'static'
        }

with open('production_data/cusip_static_map.json', 'w') as f:
    json.dump(static_map, f, indent=2)
```

### Option 3: Extract from Existing 13F Holdings

If you have past 13F data with CUSIP-Ticker pairs already matched:

```python
# Extract known mappings from historical data
# (Write custom script based on your data format)
```

---

## Batch Processing

### Prepare CUSIP List

Create `cusips_to_map.txt`:
```
037833100
670002401
03969T105
```

### Run Batch Mapper

```bash
python cusip_mapper.py batch cusips_to_map.txt \
    --data-dir production_data \
    --api-key $OPENFIGI_API_KEY

# Output:
# Loading 3 CUSIPs from cusips_to_map.txt
# Querying OpenFIGI batch 1 (3 CUSIPs)...
#   037833100 → AAPL
#   670002401 → NVAX
#   03969T105 → ARGX
# 
# CUSIP        Ticker   Source
# ----------------------------------------
# 037833100    AAPL     openfigi
# 670002401    NVAX     openfigi
# 03969T105    ARGX     openfigi
# 
# Found: 3/3 (100.0%)
# Saved 3 new mappings to cache
```

---

## Integration with EDGAR Extractor

### Update edgar_13f_extractor.py

The extractor is already configured to use CUSIPMapper:

```python
from cusip_mapper import CUSIPMapper

# Initialize mapper
mapper = CUSIPMapper(
    static_map_path=data_dir / 'cusip_static_map.json',
    cache_path=data_dir / 'cusip_cache.json',
    openfigi_api_key=os.getenv('OPENFIGI_API_KEY')
)

# Use in extraction
for holding in raw_holdings:
    ticker = mapper.get(holding.cusip)
    if ticker and ticker in universe_tickers:
        ticker_holdings[ticker] = holding

# Save cache after batch
mapper.save()
```

### Full Extraction with Mapper

```bash
# Extract 13F holdings (uses mapper automatically)
python edgar_13f_extractor.py \
    --quarter-end 2024-09-30 \
    --manager-registry production_data/manager_registry.json \
    --universe production_data/universe.json \
    --cusip-map production_data/cusip_static_map.json \
    --output production_data/holdings_snapshots.json

# Mapper stats
python cusip_mapper.py stats --data-dir production_data
```

---

## Rate Limits & Performance

### OpenFIGI Free Tier
- **Limit:** 25 requests per minute
- **Batch size:** 100 CUSIPs per request
- **Our setting:** 2.5s delay = ~24 requests/min (safe)

### Expected Performance

**Scenario 1: Cold Start (No Cache)**
- 12 managers × 2 quarters = 24 filings
- ~500 unique holdings per manager = 12,000 CUSIPs
- OpenFIGI batches: 12,000 / 100 = 120 batches
- Time: 120 × 2.5s = **5 minutes**

**Scenario 2: Warm Cache (After First Run)**
- Static map: ~200 common biotech CUSIPs
- Cache: ~5,000 CUSIPs from previous runs
- New CUSIPs: ~500 (4% of total)
- Time: 500 / 100 × 2.5s = **~12 seconds**

### Optimization Strategies

1. **Pre-populate static map** with your 322 universe tickers
2. **Run mapper once quarterly** to warm cache
3. **Share cache across team** (check into git)
4. **Get paid OpenFIGI tier** if needed (250 req/min = 10x faster)

---

## Troubleshooting

### "Rate limit exceeded" (429 error)

**Cause:** Exceeded 25 requests/minute  
**Solution:** Automatic retry with 60s wait

```python
# Already handled in cusip_mapper.py
if e.code == 429:
    print("Rate limit hit - waiting 60s...")
    time.sleep(60)
    return query_openfigi_batch(cusips, api_key)  # Retry
```

### "CUSIP not found"

**Possible reasons:**
1. Invalid CUSIP format (not 9 alphanumeric)
2. Security not in OpenFIGI database (rare)
3. Foreign security (try removing `"exchCode": "US"`)

**Fix:**
```python
# For foreign biotech (e.g., Argenx - Belgian ADR)
# Remove exchCode filter in query_openfigi_batch()
payload = [
    {
        "idType": "ID_CUSIP",
        "idValue": cusip,
        # "exchCode": "US"  # Remove this line
    }
]
```

### "Cache not updating"

**Cause:** Forgot to call `mapper.save()`  
**Fix:**

```python
mapper = CUSIPMapper(...)
results = mapper.get_batch(cusips)
mapper.save()  # Don't forget!
```

---

## File Structure

```
biotech-screener/
├── production_data/
│   ├── cusip_static_map.json       # Tier 0: Hand-curated
│   ├── cusip_cache.json            # Tier 1: Auto-populated
│   ├── openfigi_api_key.txt        # Optional: API key
│   ├── manager_registry.json       # Elite managers
│   └── universe.json               # 322 tickers
├── cusip_mapper.py                 # Mapper implementation
└── edgar_13f_extractor.py          # Uses mapper
```

---

## Testing Checklist

### Basic Functionality
- [ ] `cusip_mapper.py init` creates files
- [ ] `cusip_mapper.py query 037833100` returns AAPL
- [ ] Static map loads correctly
- [ ] Cache saves after queries

### Integration
- [ ] EDGAR extractor uses mapper
- [ ] Mappings cached between runs
- [ ] Coverage rate acceptable (~80%+ after static map)

### Edge Cases
- [ ] Invalid CUSIP format rejected
- [ ] Rate limit retry works (429 error)
- [ ] Foreign securities handled
- [ ] Cache TTL expires correctly (90 days)

---

## Next Steps

1. **Week 1 (Now):**
   - ✅ Mapper implemented
   - [ ] Get OpenFIGI API key
   - [ ] Populate static map with top 50 biotechs
   - [ ] Test with sample CUSIPs

2. **Week 1 (Later):**
   - [ ] Run full extraction for Q3 2024
   - [ ] Validate coverage rates
   - [ ] Tune static map based on gaps

3. **Week 2:**
   - [ ] Integrate into run_screen.py
   - [ ] Generate first institutional validation reports

---

## Production Checklist

Before running on real data:

- [ ] OpenFIGI API key configured
- [ ] Static map has 50+ entries (core biotechs)
- [ ] Cache directory exists and writable
- [ ] Rate limiting tested (doesn't exceed 25 req/min)
- [ ] Mapper.save() called after batch operations
- [ ] Cache TTL appropriate (90 days)

---

## Resources

- **OpenFIGI Docs:** https://www.openfigi.com/api
- **CUSIP Specification:** https://www.cusip.com/identifiers.html
- **SEC 13F Format:** https://www.sec.gov/files/form13f-nt.pdf
- **NASDAQ Ticker Lookup:** https://www.nasdaq.com/market-activity/stocks/screener

---

**Status:** Ready for Testing  
**Estimated Setup Time:** 30 minutes (including API key)  
**Estimated First Run:** 5-10 minutes (depending on cache state)
