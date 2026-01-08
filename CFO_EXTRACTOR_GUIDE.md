# CFO Extractor Guide
**Production-Ready SEC Filing Parser for Operating Cash Flow**

---

## Overview

The CFO Extractor parses SEC 10-Q and 10-K filings to extract Operating Cash Flow (CFO) data with point-in-time integrity. It outputs data in the format Module 2 expects, handling the YTD conversion automatically.

**Key Features:**
- ✅ Extracts YTD CFO values from XBRL-tagged filings
- ✅ Captures fiscal_period metadata (Q1/Q2/Q3/FY)
- ✅ Handles non-calendar fiscal years
- ✅ Maintains filing dates for PIT validation
- ✅ Outputs directly to Module 2 format
- ✅ No manual YTD→quarterly conversion needed

---

## Quick Start

### 1. Get SEC Filings

**Option A: EDGAR Bulk Downloads**
```bash
# Download from SEC EDGAR
# Example: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001376985&type=10-&dateb=&owner=exclude&count=100

# Organize by ticker:
filings/
├── CVAC/
│   ├── 0001376985-24-000045.txt  # 10-Q Q3 2024
│   ├── 0001376985-24-000032.txt  # 10-Q Q2 2024
│   └── 0001376985-24-000018.txt  # 10-Q Q1 2024
├── RYTM/
│   ├── 0001564590-24-000056.txt
│   └── 0001564590-24-000042.txt
└── ...
```

**Option B: SEC EDGAR API**
```python
import requests

def download_filing(cik, accession_number, output_path):
    """Download a specific filing from SEC EDGAR"""
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}.txt"
    headers = {'User-Agent': 'Your Name your.email@example.com'}
    
    response = requests.get(url, headers=headers)
    with open(output_path, 'w') as f:
        f.write(response.text)
```

**Important:** SEC requires User-Agent header with contact info.

### 2. Run the Extractor

```bash
python cfo_extractor.py \
    --filings-dir ./filings \
    --as-of-date 2026-01-08 \
    --output cfo_data.json \
    --module-2-output financial_data_cfo.json
```

**Output:**
- `cfo_data.json` - Raw CFO records (all periods, audit trail)
- `financial_data_cfo.json` - Module 2 ready format

### 3. Integrate with Module 2

```python
from module_2_financial import run_module_2
import json

# Load extracted CFO data
with open('financial_data_cfo.json') as f:
    cfo_data = json.load(f)

# Merge with your existing financial data
for cfo_rec in cfo_data:
    ticker = cfo_rec['ticker']
    # Find matching record in your financial_data
    # Add CFO fields: fiscal_period, CFO_ytd_current, CFO_ytd_prev, etc.

# Run Module 2 (now with burn acceleration!)
results = run_module_2(
    universe=active_tickers,
    financial_data=financial_data,  # Now includes CFO fields
    market_data=market_data
)
```

---

## Data Output Format

### Raw CFO Records (cfo_data.json)
```json
{
  "CVAC": [
    {
      "ticker": "CVAC",
      "filing_date": "2024-11-07",
      "fiscal_year": 2024,
      "fiscal_period": "Q3",
      "period_end_date": "2024-09-30",
      "cfo_value": -285000000.0,
      "is_ytd": true,
      "form_type": "10-Q",
      "accession_number": "0001376985-24-000045",
      "source_tag": "NetCashProvidedByUsedInOperatingActivities"
    }
  ]
}
```

### Module 2 Format (financial_data_cfo.json)
```json
[
  {
    "ticker": "CVAC",
    "fiscal_period": "Q3",
    "CFO_ytd_current": -285000000.0,
    "CFO_ytd_prev": -190000000.0,
    "CFO_ytd_q3": -285000000.0,
    "CFO_fy_annual": null,
    "filing_date": "2024-11-07",
    "period_end_date": "2024-09-30"
  }
]
```

**Module 2 will automatically convert:**
- Q3 quarterly CFO = -285M - (-190M) = **-95M** (3 months)
- This is the correct quarterly burn, not the 9-month cumulative!

---

## How It Works

### 1. XBRL Tag Detection

The extractor searches for CFO using multiple standard XBRL tags (in priority order):

```python
CFO_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",  # Most common
    "CashProvidedByUsedInOperatingActivities",
    "NetCashFromOperatingActivities",
    "CashFromOperatingActivities",
    "OperatingCashFlow",
]
```

This handles variations across companies and filing formats.

### 2. Fiscal Period Extraction

Identifies fiscal period from multiple sources:
- **Explicit tags:** `<dei:FiscalPeriod>Q2</dei:FiscalPeriod>`
- **Context ID:** Contains "Q2", "6m", "SecondQuarter"
- **Duration:** `<duration>P6M</duration>` → 6 months → Q2
- **Form type:** 10-K → FY

### 3. YTD Handling

The extractor **preserves YTD values** and lets Module 2 convert them:

```
Filing: 10-Q Q2 2024
Extracted: CFO_ytd_current = -190M (6 months cumulative)
           CFO_ytd_prev = -95M (Q1, 3 months)

Module 2 calculates: Q2 CFO = -190M - (-95M) = -95M
```

**No conversion logic in extractor = simpler, more robust!**

### 4. Non-Calendar Fiscal Years

Automatically supported via fiscal_period metadata:

```
Company with June 30 FYE:
- Q1 = July-Sept (fiscal period = Q1)
- Q2 = Oct-Dec (fiscal period = Q2)
- Q3 = Jan-March (fiscal period = Q3)
- Q4 = April-June (fiscal period = FY)

Extractor uses fiscal_period, NOT calendar dates.
Module 2 uses fiscal_period for conversion.
Result: Perfect handling of any fiscal year!
```

---

## Point-in-Time Validation

The extractor enforces PIT by filtering on **filing_date**:

```python
# Only include filings available as of analysis date
as_of_date = date(2024, 11, 15)

# Filing dated 2024-11-07 → INCLUDED
# Filing dated 2024-11-20 → EXCLUDED (future leak!)
```

This ensures backtests use only information available at the time.

---

## Edge Cases Handled

### 1. Restatements
**Problem:** Company files amended 10-Q with restated CFO  
**Solution:** Extractor captures all filings; latest filing_date wins

### 2. Missing Prior Periods
**Problem:** Company's Q2 filing, but no Q1 on file  
**Solution:** `CFO_ytd_prev = None`; Module 2 sets `cfo_quality_flag = "MISSING"`

### 3. Multiple CFO Tags
**Problem:** Filing has multiple cash flow statements (continuing ops, discontinued ops)  
**Solution:** Uses primary tag (NetCashProvidedByUsedInOperatingActivities) first

### 4. Non-Standard Periods
**Problem:** Company reports "6 months ended June 30" instead of Q2  
**Solution:** Duration parsing (P6M → Q2) catches these

---

## Integration Checklist

### Phase 1: Setup (5-10 min)
- [ ] Download sample SEC filings for 2-3 tickers
- [ ] Organize in `filings/TICKER/` structure
- [ ] Run extractor to verify output
- [ ] Inspect `cfo_data.json` and `financial_data_cfo.json`

### Phase 2: Bulk Processing (1-2 hours)
- [ ] Download filings for full universe (100-200 tickers)
- [ ] Run batch extraction
- [ ] Validate filing_date PIT enforcement
- [ ] Check for missing data (log warnings)

### Phase 3: Module 2 Integration (30 min)
- [ ] Merge CFO data with existing financial_data.json
- [ ] Run Module 2 with burn acceleration enabled
- [ ] Compare scores before/after (should see differentiation)
- [ ] Validate burn_acceleration > 1.0 for known accelerators

### Phase 4: Production (Ongoing)
- [ ] Schedule weekly CFO updates (new filings)
- [ ] Monitor cfo_quality_flag distribution
- [ ] Track burn acceleration signals in scoring
- [ ] Add to daily/weekly screening pipeline

---

## Performance & Scaling

### Extraction Speed
- **Single filing:** ~50-100ms (XBRL parsing)
- **200 tickers × 4 filings each:** ~40-60 seconds
- **Bottleneck:** File I/O (use SSD)

### Optimization Tips
```python
# Parallel processing for large batches
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(parse_xbrl_filing, filing_paths)
```

### Storage Requirements
- **Raw filings:** ~500KB per 10-Q, ~1MB per 10-K
- **200 tickers × 4 quarters:** ~400MB
- **Extracted data:** ~50KB JSON (lightweight!)

---

## Troubleshooting

### Issue: "No CFO tag found"
**Cause:** Old filing format (pre-XBRL) or non-standard tag  
**Fix:** Add custom tag to CFO_TAGS list, or manually parse HTML tables

### Issue: "fiscal_period is None"
**Cause:** Filing lacks fiscal period metadata  
**Fix:** Infer from filing date + known FYE, or skip ticker

### Issue: "CFO value seems wrong (10x off)"
**Cause:** Scaling issue (value in thousands vs dollars)  
**Fix:** Check decimals attribute in XBRL: `decimals="-3"` means thousands

### Issue: "Module 2 shows 'MISSING' cfo_quality_flag"
**Cause:** Extractor didn't find required prior period (e.g., Q1 for Q2 calc)  
**Fix:** Download earlier filings to build full history

---

## Advanced Usage

### Custom Tag Mapping
```python
# Add industry-specific CFO tags
CFO_TAGS.append("CashFlowFromOperations")
CFO_TAGS.append("NetOperatingCashFlow")
```

### Filtering by Form Type
```python
# Only extract from 10-K (annual data)
records = [r for r in cfo_records if r.form_type == "10-K"]
```

### Historical Backfill
```python
# Extract all filings since 2020
for year in range(2020, 2025):
    for quarter in ["Q1", "Q2", "Q3", "FY"]:
        # Download and extract...
```

---

## Next Steps

1. **Test on 5 tickers** - Validate extractor works
2. **Scale to full universe** - Process 100-200 tickers
3. **Integrate with Module 2** - Enable burn acceleration
4. **Monitor signals** - Track acceleration flags in production
5. **Automate updates** - Weekly CFO refresh pipeline

---

## Example: End-to-End Workflow

```bash
# Step 1: Download filings (example using SEC EDGAR)
python download_sec_filings.py --tickers CVAC,RYTM,IMMP --forms 10-Q,10-K

# Step 2: Extract CFO data
python cfo_extractor.py \
    --filings-dir ./filings \
    --as-of-date 2026-01-08 \
    --module-2-output financial_data_cfo.json

# Step 3: Merge with existing data
python merge_financial_data.py \
    --base financial_data.json \
    --cfo financial_data_cfo.json \
    --output financial_data_complete.json

# Step 4: Run screening with burn acceleration
python run_screen.py \
    --as-of-date 2026-01-08 \
    --financial-data financial_data_complete.json
```

**Result:** Module 2 now detects burn acceleration and adjusts dilution scores accordingly!

---

## FAQ

**Q: Do I need all historical quarters?**  
A: For burn acceleration, you need at least 4 quarters. For basic CFO, just latest quarter works.

**Q: What if a company hasn't filed yet this quarter?**  
A: Extractor uses most recent available filing (PIT compliant). Module 2 works with whatever data exists.

**Q: Can this handle international companies?**  
A: Only if they file with SEC (ADRs, foreign issuers). IFRS filings may use different tags.

**Q: How often should I refresh CFO data?**  
A: Weekly is sufficient (10-Q/10-K have 40-45 day filing deadlines after quarter end).

---

**END OF GUIDE**
