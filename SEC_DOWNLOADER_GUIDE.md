# SEC Filing Downloader Guide
**Automated SEC EDGAR Data Feed**

---

## Quick Start (2 minutes)

### Step 1: Set Your Contact Info
**REQUIRED by SEC:** Edit `sec_filing_downloader.py` line 30:

```python
# Replace this:
USER_AGENT = "YourName/1.0 (your.email@example.com)"

# With your actual info:
USER_AGENT = "Darren Brooks/1.0 (darren@brookscapital.com)"
```

**Why?** SEC requires contact info to:
- Enforce rate limits fairly
- Contact you if there's an issue with your requests
- Block abusive scrapers

### Step 2: Download Filings

```bash
python sec_filing_downloader.py \
    --tickers CVAC,RYTM,IMMP \
    --count 8
```

**That's it!** Files are downloaded to `filings/TICKER/` automatically.

---

## How It Works

### Automatic Features

1. **CIK Lookup** ✅
   - Converts ticker → CIK automatically
   - Uses SEC's official company tickers list

2. **Rate Limiting** ✅
   - Enforces 10 requests/second limit
   - Adds delays automatically
   - SEC-compliant User-Agent

3. **Smart Organization** ✅
   ```
   filings/
   ├── CVAC/
   │   ├── 10-Q_2024-11-08_0001564590-24-000045.txt
   │   ├── 10-Q_2024-08-08_0001564590-24-000032.txt
   │   └── ...
   └── RYTM/
       └── ...
   ```

4. **Skip Duplicates** ✅
   - Only downloads new files
   - Use `--overwrite` to force re-download

5. **Download Log** ✅
   - Tracks all downloads in `download_log.json`
   - Useful for auditing

---

## Usage Examples

### Basic: 3 Tickers, 8 Filings Each
```bash
python sec_filing_downloader.py --tickers CVAC,RYTM,IMMP
```

### Large Batch: Full Universe
```bash
python sec_filing_downloader.py \
    --tickers CVAC,RYTM,IMMP,BMRN,SRPT,VRTX,ALNY,IONS,EXEL,NBIX \
    --count 12
```

### Only 10-K (Annual Reports)
```bash
python sec_filing_downloader.py \
    --tickers CVAC,RYTM \
    --forms 10-K \
    --count 5
```

### Custom Output Directory
```bash
python sec_filing_downloader.py \
    --tickers CVAC \
    --output-dir sec_data/raw_filings
```

### Update Existing (Download Only New)
```bash
# First run: Downloads 8 files
python sec_filing_downloader.py --tickers CVAC --count 8

# Later: Only downloads files after 2024-10-01
python sec_filing_downloader.py --tickers CVAC --count 8
# (Automatically skips duplicates)
```

---

## Integration with CFO Extractor

### End-to-End Pipeline

```bash
# Step 1: Download filings
python sec_filing_downloader.py --tickers CVAC,RYTM,IMMP --count 8

# Step 2: Extract CFO data
python cfo_extractor.py \
    --filings-dir ./filings \
    --as-of-date 2024-12-31 \
    --output cfo_data.json \
    --module-2-output financial_data_cfo.json

# Step 3: Use with Module 2
# (Merge with your financial_data.json and run screening)
```

### Automated Daily/Weekly Updates

Create a batch script (`update_filings.bat` or `update_filings.sh`):

```bash
#!/bin/bash
# update_filings.sh - Run daily/weekly

# Download latest filings
python sec_filing_downloader.py \
    --tickers $(cat universe_tickers.txt) \
    --count 4

# Extract CFO
python cfo_extractor.py \
    --filings-dir ./filings \
    --as-of-date $(date +%Y-%m-%d) \
    --output cfo_data_latest.json \
    --module-2-output financial_data_cfo_latest.json

echo "✅ Filings updated!"
```

---

## Performance

### Download Speed
- **Single ticker (8 filings):** ~10-15 seconds
- **10 tickers (8 each):** ~2-3 minutes
- **100 tickers (8 each):** ~20-30 minutes

**Bottleneck:** SEC rate limit (10 req/sec)

### Storage
- **10-Q filing:** ~300-800 KB each
- **10-K filing:** ~1-2 MB each
- **100 tickers × 8 filings:** ~500 MB total

---

## Troubleshooting

### Issue: "CIK not found for ticker"
**Cause:** Ticker not in SEC database or misspelled  
**Solutions:**
1. Verify ticker symbol on SEC.gov
2. Try company name search manually
3. Some tickers use different symbols on SEC

### Issue: "Rate limit exceeded"
**Cause:** Making requests too fast  
**Solution:** The downloader handles this automatically. If you see this, wait 60 seconds and retry.

### Issue: "403 Forbidden"
**Cause:** Invalid or missing User-Agent  
**Solution:** Set proper User-Agent with your contact info (see Step 1)

### Issue: "Downloaded file is empty or corrupted"
**Cause:** Network issue or SEC server error  
**Solution:** 
```bash
# Re-download with overwrite
python sec_filing_downloader.py --tickers CVAC --overwrite
```

### Issue: "SSL Certificate error"
**Cause:** Python SSL configuration  
**Solution:**
```bash
pip install --upgrade certifi
```

---

## Advanced Usage

### From Python Script

```python
from sec_filing_downloader import SECDownloader

# Create downloader
downloader = SECDownloader(
    output_dir=Path("filings"),
    user_agent="YourName/1.0 (your@email.com)"
)

# Download single ticker
files = downloader.download_ticker("CVAC", count=8)
print(f"Downloaded {len(files)} files")

# Download batch
tickers = ["CVAC", "RYTM", "IMMP"]
results = downloader.download_batch(tickers, count=8)

# Check what was downloaded
for ticker, paths in results.items():
    print(f"{ticker}: {len(paths)} files")
```

### Custom Filtering

```python
# Only download filings after a certain date
from datetime import date

downloader = SECDownloader()
filings = downloader.get_filings_list("0001564590", count=20)

# Filter by date
recent = [f for f in filings 
          if f['filing_date'] >= '2024-01-01']

# Download filtered list
for filing in recent:
    downloader.download_filing("CVAC", filing)
```

### Parallel Downloads (Advanced)

```python
from concurrent.futures import ThreadPoolExecutor

downloader = SECDownloader()
tickers = ["CVAC", "RYTM", "IMMP", "BMRN", "SRPT"]

with ThreadPoolExecutor(max_workers=3) as executor:
    # Download 3 tickers in parallel
    futures = [executor.submit(downloader.download_ticker, t) 
               for t in tickers]
    
    results = [f.result() for f in futures]
```

**Warning:** Don't exceed 10 requests/second total across all threads!

---

## File Format

### Downloaded File Structure

```
<FILING HEADER>
TYPE: 10-Q
...

<XBRL INSTANCE>
<us-gaap:NetCashProvidedByUsedInOperatingActivities ...>
-190000
</us-gaap:NetCashProvidedByUsedInOperatingActivities>

<context id="Q2_2024">
    <dei:FiscalYear>2024</dei:FiscalYear>
    <dei:FiscalPeriod>Q2</dei:FiscalPeriod>
    ...
</context>
```

The CFO extractor parses this automatically!

---

## SEC Compliance Best Practices

1. **Always include User-Agent** with contact info
2. **Respect rate limits** (10 req/sec max)
3. **Cache downloaded files** (don't re-download unnecessarily)
4. **Use reasonable counts** (8-12 filings usually sufficient)
5. **Add delays** between batch operations
6. **Monitor your usage** via download logs

### SEC Fair Access Policy

From SEC.gov:
> "To ensure fair access to all users, SEC systems employ technology to 
> limit excessive requests. Users making more than 10 requests per second 
> may have their access restricted."

Our downloader enforces this automatically! ✅

---

## Production Deployment

### Scheduled Updates

**Option 1: Windows Task Scheduler**
```
Action: Start a program
Program: python
Arguments: C:\path\to\sec_filing_downloader.py --tickers CVAC,RYTM --count 4
Schedule: Weekly, Mondays at 6 AM
```

**Option 2: Linux Cron**
```bash
# Edit crontab
crontab -e

# Add line (runs every Monday at 6 AM)
0 6 * * 1 cd /path/to/project && python sec_filing_downloader.py --tickers CVAC,RYTM --count 4
```

### Monitoring

Track download success rate:
```python
import json

with open('filings/download_log.json') as f:
    log = json.load(f)

for ticker, downloads in log.items():
    print(f"{ticker}: {len(downloads)} successful downloads")
```

---

## Cost & Resources

### Free!
- ✅ SEC EDGAR is completely free
- ✅ No API keys required
- ✅ No rate limit fees
- ✅ Unlimited downloads (within rate limits)

### Requirements
- Python 3.7+
- `requests` library
- Internet connection
- ~500 MB disk space per 100 tickers

---

## Next Steps

1. **Update User-Agent** in script (line 30)
2. **Test with 1-2 tickers** first
3. **Scale to full universe** once working
4. **Integrate with CFO extractor**
5. **Set up automated updates** (weekly/monthly)

---

## Support

### Useful Links
- [SEC EDGAR](https://www.sec.gov/edgar)
- [SEC Developer Resources](https://www.sec.gov/developer)
- [SEC Rate Limiting Policy](https://www.sec.gov/os/accessing-edgar-data)

### Common SEC Forms
- **10-Q:** Quarterly report
- **10-K:** Annual report
- **8-K:** Current event report (major announcements)
- **S-3:** Shelf registration (capital raises)
- **DEF 14A:** Proxy statement (shareholder meetings)

---

**Ready to automate your data feed!** 🚀

Run: `python sec_filing_downloader.py --tickers CVAC,RYTM,IMMP`
