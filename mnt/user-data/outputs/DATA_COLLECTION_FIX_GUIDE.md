# Data Collection Fix Guide

## 🔍 **Step 1: Run Diagnostics**

First, understand what's broken:

```powershell
python diagnose_data_collection.py
```

This will show you:
- How many stocks have each type of data
- Which stocks are complete vs missing data
- Sample comparison of complete vs incomplete stocks

---

## 🐛 **Common Issues & Fixes**

### **Issue 1: Data Collection Scripts Not Run**

**Symptoms:**
- Most stocks have 0 data
- Only a few stocks (4) have data
- Missing data across all modules

**Diagnosis:**
```powershell
# Check if data collection scripts exist
Get-ChildItem *collect*.py
Get-ChildItem *fetch*.py
Get-ChildItem *data*.py
```

**Fix:**
```powershell
# Run data collection for all stocks
python collect_financial_data.py
python collect_catalyst_data.py  
python collect_clinical_data.py
python collect_defensive_data.py
```

---

### **Issue 2: API Rate Limits**

**Symptoms:**
- First few stocks succeed, rest fail
- Exactly 4 stocks have data (typical rate limit)
- Errors like "429 Too Many Requests"

**Diagnosis:**
Check logs for rate limit errors:
```powershell
# Look for rate limit messages in recent runs
Select-String -Path *.log -Pattern "rate limit|429|throttle"
```

**Fix Option A: Add Delays**

Add sleep between API calls:
```python
import time

for ticker in tickers:
    fetch_data(ticker)
    time.sleep(1)  # 1 second delay between stocks
```

**Fix Option B: Implement Retry Logic**

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"Rate limited, waiting {delay}s...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def fetch_ticker_data(ticker):
    # Your API call here
    pass
```

**Fix Option C: Use Caching**

```python
import json
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_or_fetch(ticker, fetch_func):
    cache_file = CACHE_DIR / f"{ticker}.json"
    
    # Return cached if exists
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    # Fetch and cache
    data = fetch_func(ticker)
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    return data
```

---

### **Issue 3: Wrong Ticker Symbols**

**Symptoms:**
- API calls fail with "symbol not found"
- CUSIP works but ticker doesn't
- Some stocks have data, others don't

**Diagnosis:**
```python
# Test if ticker is valid
import yfinance as yf

ticker = "ARGX"
stock = yf.Ticker(ticker)
try:
    info = stock.info
    print(f"{ticker}: Valid - {info.get('longName', 'Unknown')}")
except:
    print(f"{ticker}: INVALID or API error")
```

**Fix:**

Map CUSIPs to correct tickers:
```python
# Your universe might have wrong ticker mappings
# Check and fix in universe.json or use CUSIP-to-ticker service

from openfigi import OpenFigiClient

client = OpenFigiClient()

def get_ticker_from_cusip(cusip):
    result = client.search(id_value=cusip, id_type='ID_CUSIP')
    if result:
        return result[0].get('ticker')
    return None
```

---

### **Issue 4: Missing API Keys / Credentials**

**Symptoms:**
- Authentication errors
- "Unauthorized" or "Forbidden" responses
- Works for some sources, not others

**Diagnosis:**
```powershell
# Check for environment variables
$env:SEC_API_KEY
$env:CLINICALTRIALS_API_KEY
$env:YAHOO_API_KEY
```

**Fix:**

Set required API keys:
```powershell
# In PowerShell
$env:SEC_API_KEY = "your-key-here"
$env:CLINICALTRIALS_API_KEY = "your-key-here"

# Or add to .env file
# Then load with python-dotenv
```

```python
# In Python
import os
from dotenv import load_dotenv

load_dotenv()

SEC_API_KEY = os.getenv('SEC_API_KEY')
```

---

### **Issue 5: Data Structure Mismatch**

**Symptoms:**
- Collection runs successfully
- But screening shows "no data"
- Data exists but in wrong format

**Diagnosis:**
```python
# Check structure of collected data
import json

with open('production_data/universe.json') as f:
    universe = json.load(f)

# Look at a stock with data
sample = next(s for s in universe if s.get('ticker') == 'ALNY')
print("ALNY structure:")
print(json.dumps(sample, indent=2))

# Look at a stock without data
sample2 = next(s for s in universe if s.get('ticker') == 'ARGX')
print("\nARGX structure:")
print(json.dumps(sample2, indent=2))
```

**Fix:**

Ensure data matches expected structure:
```python
# Expected structure for each module:

# Financial data (Module 2)
financial_data = {
    'market_cap': 12500000000,
    'cash': 2500000000,
    'debt': 500000000,
    'burn_rate': -100000000,  # Quarterly
    'revenue': 1000000000,
    # ... other fields
}

# Catalyst data (Module 3)
catalyst_data = {
    'next_catalyst_date': '2026-03-15',
    'catalyst_type': 'FDA_PDUFA',
    'catalyst_description': 'PDUFA date for Drug X',
    # ... other fields
}

# Clinical data (Module 4)
clinical_data = {
    'trials': [
        {
            'nct_id': 'NCT12345678',
            'phase': 'Phase 3',
            'status': 'Active',
            'enrollment': 500,
            # ... other fields
        }
    ]
}

# Defensive features
defensive_features = {
    'vol_60d': '0.2352',  # String format
    'corr_xbi': '0.27',
    'drawdown_60d': '-0.15',
}
```

---

## 🔧 **Step 2: Test Single Stock**

Before running full collection, test on one stock:

```python
# test_single_stock.py
import yfinance as yf
import json

def test_single_ticker(ticker):
    """Test data collection for a single ticker."""
    print(f"Testing data collection for {ticker}...")
    print("="*60)
    
    # Test Yahoo Finance
    print("\n1. Testing Yahoo Finance (Financial + Defensive)...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        print(f"  ✅ Name: {info.get('longName', 'N/A')}")
        print(f"  ✅ Market Cap: ${info.get('marketCap', 0):,.0f}")
        print(f"  ✅ Price history: {len(hist)} days")
        
        # Calculate volatility
        returns = hist['Close'].pct_change().dropna()
        vol = returns.std() * (252 ** 0.5)  # Annualized
        print(f"  ✅ Volatility: {vol:.2%}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test ClinicalTrials.gov
    print("\n2. Testing ClinicalTrials.gov (Clinical)...")
    try:
        import requests
        url = f"https://clinicaltrials.gov/api/v2/studies"
        params = {
            'query.term': ticker,
            'format': 'json',
            'pageSize': 10
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            trials = data.get('studies', [])
            print(f"  ✅ Found {len(trials)} trials")
            if trials:
                print(f"  ✅ Sample: {trials[0].get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'N/A')}")
        else:
            print(f"  ❌ Status code: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test SEC EDGAR (if applicable)
    print("\n3. Testing SEC EDGAR (Financial/Catalyst)...")
    try:
        # SEC requires company CIK, not ticker
        # This is just a connectivity test
        import requests
        headers = {
            'User-Agent': 'Your Company name@email.com'  # Required by SEC
        }
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {'action': 'getcompany', 'CIK': ticker, 'count': 1}
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            print(f"  ✅ SEC accessible")
        else:
            print(f"  ⚠️  Status code: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("Test complete!")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GILD"
    test_single_ticker(ticker)
```

Run it:
```powershell
python test_single_stock.py ARGX
python test_single_stock.py GILD
```

---

## 🚀 **Step 3: Fix & Re-Run Collection**

### **Option A: Sequential Collection (Slow but Safe)**

Run each module sequentially with delays:

```python
# collect_all_sequential.py
import json
import time
from pathlib import Path

def collect_all_data_sequential(universe_path="production_data/universe.json"):
    """Collect all data sequentially with delays to avoid rate limits."""
    
    with open(universe_path) as f:
        universe = json.load(f)
    
    print(f"Collecting data for {len(universe)} stocks...")
    print("This will take approximately {:.0f} minutes".format(len(universe) * 3 / 60))
    print()
    
    for i, security in enumerate(universe, 1):
        ticker = security.get('ticker')
        print(f"[{i}/{len(universe)}] Processing {ticker}...")
        
        try:
            # Collect financial data
            financial = collect_financial_for_ticker(ticker)
            security['financial_data'] = financial
            print(f"  ✅ Financial")
        except Exception as e:
            print(f"  ❌ Financial: {e}")
        
        time.sleep(1)  # Delay between modules
        
        try:
            # Collect catalyst data
            catalyst = collect_catalyst_for_ticker(ticker)
            security['catalyst_data'] = catalyst
            print(f"  ✅ Catalyst")
        except Exception as e:
            print(f"  ❌ Catalyst: {e}")
        
        time.sleep(1)
        
        try:
            # Collect clinical data
            clinical = collect_clinical_for_ticker(ticker)
            security['clinical_data'] = clinical
            print(f"  ✅ Clinical")
        except Exception as e:
            print(f"  ❌ Clinical: {e}")
        
        # Save progress after each stock
        if i % 10 == 0:
            with open(universe_path, 'w') as f:
                json.dump(universe, f, indent=2)
            print(f"  💾 Saved progress ({i}/{len(universe)})")
        
        time.sleep(2)  # Delay between stocks
    
    # Final save
    with open(universe_path, 'w') as f:
        json.dump(universe, f, indent=2)
    
    print()
    print("✅ Collection complete!")
    return universe
```

### **Option B: Parallel Collection (Fast but Risky)**

Use threading with rate limiting:

```python
# collect_all_parallel.py
import json
import time
import threading
from queue import Queue
from pathlib import Path

class RateLimitedCollector:
    def __init__(self, requests_per_second=2):
        self.delay = 1.0 / requests_per_second
        self.last_request = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request
            if time_since_last < self.delay:
                time.sleep(self.delay - time_since_last)
            self.last_request = time.time()

def worker(queue, collector, results, errors):
    while True:
        item = queue.get()
        if item is None:
            break
        
        ticker, security = item
        
        try:
            collector.wait_if_needed()
            
            # Collect all data
            financial = collect_financial_for_ticker(ticker)
            security['financial_data'] = financial
            
            collector.wait_if_needed()
            catalyst = collect_catalyst_for_ticker(ticker)
            security['catalyst_data'] = catalyst
            
            collector.wait_if_needed()
            clinical = collect_clinical_for_ticker(ticker)
            security['clinical_data'] = clinical
            
            results.append(security)
            print(f"✅ {ticker}")
            
        except Exception as e:
            errors.append((ticker, str(e)))
            print(f"❌ {ticker}: {e}")
        
        queue.task_done()

def collect_all_parallel(universe_path, num_workers=3):
    with open(universe_path) as f:
        universe = json.load(f)
    
    queue = Queue()
    results = []
    errors = []
    collector = RateLimitedCollector(requests_per_second=2)
    
    # Start workers
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, args=(queue, collector, results, errors))
        t.start()
        threads.append(t)
    
    # Add work to queue
    for security in universe:
        queue.put((security['ticker'], security))
    
    # Wait for completion
    queue.join()
    
    # Stop workers
    for _ in range(num_workers):
        queue.put(None)
    for t in threads:
        t.join()
    
    print(f"\n✅ Completed: {len(results)}")
    print(f"❌ Errors: {len(errors)}")
    
    return universe
```

---

## 📋 **Step 4: Validate Collection**

After re-running collection:

```powershell
# Run diagnostics again
python diagnose_data_collection.py

# Should see much better coverage:
# Financial data: 95+/98 (97%+)
# Catalyst data: 60+/98 (60%+)  
# Clinical data: 70+/98 (70%+)
# Defensive data: 97+/98 (99%+)
```

---

## ✅ **Step 5: Re-Run Screening**

Once data collection is fixed:

```powershell
# Re-run screening with complete data
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir production_data `
    --output screening_COMPLETE_DATA.json

# Check results
python -c "import json; data = json.load(open('screening_COMPLETE_DATA.json')); print('Screening complete with full data!')"
```

---

## 🎯 **Expected Results After Fix:**

### **Before (Broken):**
```
Financial: 4/98 (4%)
Catalyst: 4/98 (4%)
Clinical: 4/98 (4%)

Top rankings based purely on defensive overlay.
```

### **After (Fixed):**
```
Financial: 95/98 (97%)
Catalyst: 65/98 (66%)  
Clinical: 75/98 (77%)

Top rankings based on fundamentals + catalysts + clinical + defensive.
```

---

## 🚨 **Common Pitfalls:**

1. **Don't run collection too fast** - You'll hit rate limits
2. **Save progress frequently** - Don't lose work to crashes
3. **Use caching** - Don't re-fetch same data
4. **Handle errors gracefully** - One bad stock shouldn't kill everything
5. **Log everything** - You need to debug issues

---

## 📞 **Need Help?**

If collection still fails after these fixes, provide:
1. Error messages from collection logs
2. Output of diagnostic script
3. Your data collection code
4. APIs/services you're using

We'll debug further!
