# 🔧 Data Collection Fix - Action Plan

## 🎯 **Your Situation:**

**Current State:**
- Only 4/98 stocks (4%) have complete data: ALNY, BMRN, SRPT, VRTX
- 94/98 stocks (96%) have ONLY defensive data (volatility, correlation)
- Screening ranks stocks by defensive quality only (not fundamentals)

**What This Means:**
- ARGX ranks #1 because it's the most defensive (22% vol, low correlation)
- VRTX, BMRN, SRPT rank low despite having complete data
- System works correctly, but data collection pipeline is broken

---

## 📋 **Prioritized Action Plan**

### **Phase 1: Diagnose (15 minutes)**

#### **Step 1.1: Run Diagnostic Script**

```powershell
python diagnose_data_collection.py
```

**This will show:**
- Exactly how many stocks have each type of data
- Which stocks are complete vs incomplete
- Sample comparison of data structures

**Expected output:**
```
Financial data: 4/98 (4.1%)
Catalyst data: 4/98 (4.1%)
Clinical data: 4/98 (4.1%)
Defensive data: 97/98 (99.0%)

Stocks with complete data: ALNY, BMRN, SRPT, VRTX
Stocks with defensive only: ARGX, CVAC, GILD, ... (94 more)
```

#### **Step 1.2: Test Single Stock**

```powershell
# Test a stock that's missing data
python test_single_stock.py ARGX

# Test a stock that has data
python test_single_stock.py ALNY
```

**This will show:**
- Which APIs work vs fail
- Network/authentication issues
- Data structure problems

**Expected output:**
```
Testing Yahoo Finance for ARGX:
  ✅ Basic Info Retrieved
  ✅ Financial Data
  ✅ Price History

Testing ClinicalTrials.gov for ARGX:
  ✅ Found 12 studies

Testing SEC EDGAR for ARGX:
  ⚠️ No filings found (may be foreign company)

Summary: 2/3 tests passed
```

---

### **Phase 2: Identify Root Cause (10 minutes)**

Based on diagnostic results, identify which scenario you're in:

#### **Scenario A: Data Collection Never Run**

**Symptoms:**
- 0% or very low coverage (<10%)
- Missing data across all modules

**Fix:** Run your data collection scripts
```powershell
python collect_financial_data.py
python collect_catalyst_data.py
python collect_clinical_data.py
```

#### **Scenario B: API Rate Limits**

**Symptoms:**
- First 4-5 stocks succeed, rest fail
- Errors like "429 Too Many Requests"

**Fix:** Add delays between API calls
```python
import time
for ticker in tickers:
    collect_data(ticker)
    time.sleep(2)  # 2 second delay
```

#### **Scenario C: Missing API Keys**

**Symptoms:**
- "Unauthorized" or "Forbidden" errors
- Some APIs work, others don't

**Fix:** Set environment variables
```powershell
$env:SEC_API_KEY = "your-key"
$env:CLINICALTRIALS_API_KEY = "your-key"
```

#### **Scenario D: Wrong Ticker Format**

**Symptoms:**
- Some tickers work, others fail
- "Symbol not found" errors

**Fix:** Validate tickers in universe.json
```python
import yfinance as yf
ticker = yf.Ticker("ARGX")
print(ticker.info.get('longName'))  # Should work
```

#### **Scenario E: Data Structure Mismatch**

**Symptoms:**
- Collection runs successfully
- But screening shows "no data"

**Fix:** Check data structure
```python
# Your data should match expected format
financial_data = {
    'market_cap': 12500000000,
    'cash': 2500000000,
    # ... etc
}
```

---

### **Phase 3: Fix Data Collection (30-120 minutes)**

#### **Quick Fix (30 min):** Test & Collect 10 Stocks

```python
# test_collect_10.py
import json
import time

# Load universe
with open('production_data/universe.json') as f:
    universe = json.load(f)

# Test on first 10 defensive-only stocks
test_tickers = ['ARGX', 'CVAC', 'GILD', 'GLPG', 'HALO', 
                'INCY', 'OPCH', 'PCRX', 'AKRO', 'LGND']

for ticker in test_tickers:
    print(f"Collecting {ticker}...")
    try:
        # Your collection functions here
        financial = collect_financial(ticker)
        catalyst = collect_catalyst(ticker)
        clinical = collect_clinical(ticker)
        
        # Update universe
        for sec in universe:
            if sec['ticker'] == ticker:
                sec['financial_data'] = financial
                sec['catalyst_data'] = catalyst
                sec['clinical_data'] = clinical
        
        print(f"  ✅ Success")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    time.sleep(2)  # Rate limit protection

# Save
with open('production_data/universe.json', 'w') as f:
    json.dump(universe, f, indent=2)

print("\n✅ Test collection complete!")
```

**Run it:**
```powershell
python test_collect_10.py
```

**Then check:**
```powershell
python diagnose_data_collection.py
# Should show 14/98 (14%) instead of 4/98
```

#### **Full Fix (2 hours):** Collect All 98 Stocks

Once 10-stock test succeeds, scale up:

```python
# collect_all_with_progress.py
import json
import time
from datetime import datetime

with open('production_data/universe.json') as f:
    universe = json.load(f)

total = len(universe)
start_time = datetime.now()

for i, sec in enumerate(universe, 1):
    ticker = sec['ticker']
    
    # Skip if already has data
    if sec.get('financial_data') and sec.get('catalyst_data') and sec.get('clinical_data'):
        print(f"[{i}/{total}] {ticker} - Already complete, skipping")
        continue
    
    print(f"[{i}/{total}] {ticker} - Collecting...")
    
    try:
        if not sec.get('financial_data'):
            sec['financial_data'] = collect_financial(ticker)
        
        time.sleep(1)
        
        if not sec.get('catalyst_data'):
            sec['catalyst_data'] = collect_catalyst(ticker)
        
        time.sleep(1)
        
        if not sec.get('clinical_data'):
            sec['clinical_data'] = collect_clinical(ticker)
        
        print(f"  ✅ Complete")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Save progress every 10 stocks
    if i % 10 == 0:
        with open('production_data/universe.json', 'w') as f:
            json.dump(universe, f, indent=2)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = i / elapsed
        remaining = (total - i) / rate
        
        print(f"  💾 Progress saved ({i}/{total}, ~{remaining/60:.0f} min remaining)")
    
    time.sleep(2)  # Rate limit protection

# Final save
with open('production_data/universe.json', 'w') as f:
    json.dump(universe, f, indent=2)

print(f"\n✅ Collection complete! Processed {total} stocks")
```

---

### **Phase 4: Validate (5 minutes)**

```powershell
# Run diagnostics
python diagnose_data_collection.py
```

**Expected after fix:**
```
Financial data: 95/98 (97%)  ← Was 4%
Catalyst data: 65/98 (66%)   ← Was 4%
Clinical data: 75/98 (77%)   ← Was 4%
Defensive data: 97/98 (99%)  ← Already good

Complete data: 60+ stocks ← Was 4

✅ GOOD: Data coverage is sufficient for screening
```

---

### **Phase 5: Re-Run Screening (5 minutes)**

```powershell
# Re-run with complete data
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir production_data `
    --output screening_COMPLETE.json

# Validate
python -c "
import json
data = json.load(open('screening_COMPLETE.json'))
ranked = data['module_5_composite']['ranked_securities']

# Check top 10
print('Top 10 with complete data:')
for i, s in enumerate(ranked[:10], 1):
    ticker = s['ticker']
    score = s['composite_score']
    print(f'{i}. {ticker}: {score:.2f}')
"
```

**Expected results with complete data:**
```
Top 10 should now include fundamentally strong stocks:
1. VRTX: 52.30  (was rank 96!)
2. GILD: 51.80
3. ALNY: 51.20  (was rank 9)
4. BMRN: 48.50  (was rank 98!)
5. ARGX: 47.90  (still high due to defensive quality)
...
```

---

## 📊 **Expected Timeline**

| Phase | Duration | Cumulative |
|-------|----------|------------|
| **Phase 1: Diagnose** | 15 min | 15 min |
| **Phase 2: Root Cause** | 10 min | 25 min |
| **Phase 3a: Test 10** | 30 min | 55 min |
| **Phase 3b: Full Collection** | 120 min | 175 min (~3 hours) |
| **Phase 4: Validate** | 5 min | 180 min |
| **Phase 5: Re-Screen** | 5 min | 185 min |

**Total: ~3 hours** to go from 4% coverage to 95%+ coverage

---

## 🚨 **If You Get Stuck**

### **Problem: Rate Limits Hit Immediately**

**Solution:** Increase delays
```python
time.sleep(5)  # 5 seconds instead of 2
```

### **Problem: Collection Takes Too Long**

**Solution:** Run in parallel with rate limiting
```python
# Use ThreadPoolExecutor with rate limiter
from concurrent.futures import ThreadPoolExecutor
import threading

rate_limiter = threading.Semaphore(2)  # Max 2 concurrent

def collect_with_limit(ticker):
    with rate_limiter:
        return collect_data(ticker)
        time.sleep(1)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(collect_with_limit, t) for t in tickers]
```

### **Problem: Some APIs Keep Failing**

**Solution:** Skip and continue
```python
try:
    financial = collect_financial(ticker)
except Exception as e:
    print(f"Financial failed for {ticker}: {e}")
    financial = None  # Mark as missing but continue
```

### **Problem: Don't Have Collection Scripts**

**Solution:** I can help you build them! Just tell me:
1. What data sources you want to use
2. What fields you need for each module
3. Any API keys you have access to

---

## ✅ **Success Criteria**

After completing all phases, you should see:

1. ✅ **95%+ financial data coverage** (was 4%)
2. ✅ **60%+ catalyst data coverage** (was 4%)
3. ✅ **70%+ clinical data coverage** (was 4%)
4. ✅ **Top 10 rankings change** significantly
5. ✅ **VRTX, ALNY, BMRN move to top ranks** (from bottom)
6. ✅ **ARGX still in top 10** but for fundamentals + defensive (not just defensive)

---

## 🎯 **Next Steps Right Now**

```powershell
# Step 1: Download scripts
# (Already done - you have the files)

# Step 2: Run diagnostics
python diagnose_data_collection.py

# Step 3: Test single stock
python test_single_stock.py ARGX

# Step 4: Review output and identify root cause

# Step 5: Tell me what you found!
```

**Then we'll proceed with the appropriate fix from Phase 3!** 🚀

---

## 📁 **Files You Have:**

1. ✅ `diagnose_data_collection.py` - Shows data gaps
2. ✅ `test_single_stock.py` - Tests APIs for one stock
3. ✅ `DATA_COLLECTION_FIX_GUIDE.md` - Detailed fix guide

**Run Step 1 and 2 now, then share the output!** 📊
