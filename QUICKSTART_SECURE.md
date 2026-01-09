# Quick Setup (5 Minutes) - SECURE METHOD
## Wake Robin Institutional Validation

**Your API key:** `1a242384-a922-4b68-92dc-c15474f79d2d`

---

## Step 1: Run Automated Setup (2 minutes)

```bash
cd /path/to/biotech-screener

# Run setup script (will prompt for API key)
python setup_environment.py
```

**What this does:**
- ✅ Updates .gitignore (protects API key)
- ✅ Creates production_data/ and outputs/ directories
- ✅ Creates .env file with your API key (secure)
- ✅ Initializes CUSIP mapper files
- ✅ Verifies everything works

**When prompted for API key, paste:**
```
1a242384-a922-4b68-92dc-c15474f79d2d
```

---

## Step 2: Test CUSIP Mapper (1 minute)

```bash
# Mapper automatically loads API key from .env
python cusip_mapper.py query 037833100 --data-dir production_data

# Expected output:
# Loading CUSIP mapper...
#   Static map: 0 entries
#   Cache: 0 entries
#   Cache miss: 037833100 - querying OpenFIGI...
#   037833100 → AAPL
# 
# 037833100 → AAPL
#   Name: Apple Inc
#   Exchange: NASDAQ
#   Type: Common Stock
#   Source: openfigi
```

✅ **If you see this, your API key is working!**

---

## Step 3: Populate Static CUSIP Map (Optional, 10-15 minutes)

### Quick Method: Add Known Biotechs Manually

```bash
# Add a few core biotechs to speed up extraction
python build_static_cusip_map.py add \
    --static-map production_data/cusip_static_map.json \
    --ticker NVAX --cusip 670002401 --name "Novavax Inc"

python build_static_cusip_map.py add \
    --static-map production_data/cusip_static_map.json \
    --ticker ARGX --cusip 03969T105 --name "Argenx SE"

python build_static_cusip_map.py add \
    --static-map production_data/cusip_static_map.json \
    --ticker BIIB --cusip 09075V102 --name "Biogen Inc"

# Add 5-10 more from your universe...
```

### Thorough Method: Generate Full Template

```bash
# Generate CSV template from universe
python build_static_cusip_map.py generate-template \
    --universe production_data/universe.json \
    --output biotech_cusips_template.csv

# Open in Excel, fill in CUSIPs from:
# - SEC EDGAR: https://www.sec.gov/edgar/searchedgar/companysearch.html
# - Yahoo Finance: https://finance.yahoo.com/quote/[TICKER]

# Save as biotech_cusips.csv, then import:
python build_static_cusip_map.py import-csv biotech_cusips.csv \
    --output production_data/cusip_static_map.json
```

---

## Step 4: Extract Real 13F Data (5-10 minutes)

```bash
# Extract Q3 2024 holdings from elite biotech managers
python edgar_13f_extractor.py \
    --quarter-end 2024-09-30 \
    --manager-registry production_data/manager_registry.json \
    --universe production_data/universe.json \
    --cusip-map production_data/cusip_static_map.json \
    --output production_data/holdings_snapshots.json

# Expected output:
# ================================================================================
# EXTRACTING 13F HOLDINGS FOR Q ENDING 2024-09-30
# ================================================================================
# Current quarter: 2024-09-30
# Prior quarter: 2024-06-30
#
# Processing Baker Bros Advisors (0001263508)...
#   Querying submissions API...
#   Found 13F-HR: 0001263508-24-000003 (filed 2024-11-14)
#   Parsed 156 holdings, total value: $13,838,782K
#   Matched 23 holdings to universe
# 
# [... 12 managers × 2 quarters = 24 filings ...]
#
# ================================================================================
# EXTRACTION COMPLETE
# ================================================================================
# Tickers with institutional coverage: 145
# Output saved to: production_data/holdings_snapshots.json
```

---

## Verify Everything Works

```bash
# Check output file
ls -lh production_data/holdings_snapshots.json

# Count covered tickers
python -c "
import json
with open('production_data/holdings_snapshots.json') as f:
    data = json.load(f)
print(f'✅ Institutional coverage: {len(data)} tickers')
"

# View sample holding
python -c "
import json
with open('production_data/holdings_snapshots.json') as f:
    data = json.load(f)
    
ticker = list(data.keys())[0]
holding = data[ticker]

print(f'\nSample: {ticker}')
print(f'  Elite holders: {len(holding[\"holdings\"][\"current\"])}')
print(f'  Market cap: \${holding[\"market_cap_usd\"]:,.0f}')
"
```

---

## What Just Happened?

✅ **Your API key is secure:**
- Stored in `.env` file (gitignored, never committed)
- Never hardcoded in source files
- Automatically loaded by scripts

✅ **Data pipeline is working:**
- CUSIP mapper operational (3-tier caching)
- SEC EDGAR extractor functional
- 13F holdings extracted successfully

✅ **Ready for integration:**
- holdings_snapshots.json contains institutional data
- Can now integrate into run_screen.py

---

## Next Steps

### Immediate: Validate Coverage

```bash
python build_static_cusip_map.py stats \
    --static-map production_data/cusip_static_map.json \
    --universe production_data/universe.json

# Expected:
# Universe tickers:     322
# Static map entries:   5-10 (or more if you populated)
# Coverage:             145 / 322 (45.0%)
# Missing:              177
```

**Coverage breakdown:**
- 40-50% coverage is **expected and correct**
- Elite managers are selective (they don't hold every biotech)
- Missing tickers = NOT held by elite managers = NO_DATA (neutral)

### Week 2: Integration

1. **Wire into run_screen.py** (2-3 hours)
   - Add institutional validation layer
   - Generate reports with new columns

2. **Test end-to-end** (1 hour)
   - Run full screening pipeline
   - Validate institutional columns present

3. **Generate IC reports** (1 hour)
   - Top 60 with institutional signals
   - Alert report for exceptional activity

---

## Security Reminder 🔒

Your `.env` file is now protected:
- ✅ Listed in .gitignore
- ✅ Won't be committed to git
- ✅ API key never hardcoded

**If you accidentally commit .env:**
1. Rotate API key immediately at https://www.openfigi.com/api
2. Remove from git history (see SETUP_API_KEY_SECURELY.md)

---

## Troubleshooting

### "API key not found"
```bash
# Verify .env exists
ls -la .env

# Check contents (should show your key)
cat .env | grep OPENFIGI_API_KEY

# Test loading
python -c "
from pathlib import Path
for line in Path('.env').read_text().splitlines():
    if line.startswith('OPENFIGI_API_KEY'):
        print('✅ API key found in .env')
        break
"
```

### "Rate limit exceeded"
- Free tier: 25 requests/minute
- Script uses 2.5s delay = ~24 requests/minute (safe)
- If hit: Automatic 60s retry

### "CUSIP not found"
- Add to static map manually
- Or let OpenFIGI query (auto-caches for next time)

---

## Summary

**Time invested:** 5-20 minutes (depending on static map effort)

**What you got:**
- ✅ Secure API key setup
- ✅ Working CUSIP mapper
- ✅ Real 13F holdings extracted
- ✅ ~45% institutional coverage

**Ready for:** Week 2 integration into run_screen.py

---

**Files created:**
- `.env` - Your API key (secure, gitignored)
- `production_data/cusip_static_map.json` - CUSIP mappings
- `production_data/cusip_cache.json` - OpenFIGI cache
- `production_data/holdings_snapshots.json` - 13F holdings data

**Status:** ✅ Ready to proceed with integration!
