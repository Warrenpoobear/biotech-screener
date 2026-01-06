# ETF Universe Expansion - Practical Implementation Guide

## Overview

You want to expand from 21 stocks to ~200 stocks (XBI/IBB/NBI constituents).

This guide provides **3 practical approaches** depending on your current infrastructure.

---

## 🎯 **Step 1: Generate Ticker List** (5 minutes)

```powershell
# Download and run the ETF constituents fetcher:
python fetch_etf_constituents.py --output etf_universe_template.json --template-only

# Review the output:
python -c "import json; d=json.load(open('etf_universe_template.json')); print(f'{len(d)} tickers')"
```

**Output:** `etf_universe_template.json` with ~200 tickers

---

## 🔀 **Step 2: Choose Your Data Collection Approach**

### **Approach A: Use Your Existing Data Pipeline** ⭐ RECOMMENDED

If you have `wake_robin_data_pipeline` or similar:

```powershell
# Extract ticker list
$tickers = (Get-Content etf_universe_template.json | ConvertFrom-Json).ticker -join ","

# Run your existing pipeline
cd wake_robin_data_pipeline
python collect_universe_data.py --tickers $tickers --as-of-date 2026-01-06 --output ../etf_universe_data

# Copy to production_data
cd ..
Copy-Item etf_universe_data/universe_snapshot_latest.json production_data/universe.json
```

**If this works → SKIP to Step 3**

---

### **Approach B: Phased Rollout** (Conservative)

Instead of jumping to 200 stocks, gradually expand:

#### **Phase 1: Top 50 (This Week)**

```python
# Create top-50 list (largest holdings from each ETF)
top_50 = [
    # Large caps (sure to have good data)
    "VRTX", "REGN", "AMGN", "GILD", "BIIB", "ALNY", "MRNA", "BNTX",
    "BMRN", "IONS", "INCY", "EXEL", "JAZZ", "UTHR", "SGEN", "LGND",
    "SRPT", "ARGX", "NBIX", "TECH", "RARE", "ACAD", "BGNE", "RGEN",
    
    # Mid caps
    "ROIV", "NTRA", "HZNP", "DAWN", "PCVX", "ARVN", "LEGN", "IMMU",
    "BLUE", "FATE", "CRSP", "NTLA", "EDIT", "BEAM", "VCYT", "NSTG",
    
    # Small caps
    "AXSM", "PTGX", "CDNA", "PRVA", "MYOV", "RETA", "FOLD", "BBIO",
    "EPRX", "GOSS"
]

# Save as smaller universe
import json
with open('top_50_tickers.json', 'w') as f:
    json.dump(top_50, f)
```

**Collect data for 50 stocks, validate, then expand**

#### **Phase 2: Top 100 (Next Week)**

Add next 50 tickers, validate again

#### **Phase 3: Full 200 (Week 3)**

Complete the universe

---

### **Approach C: Manual Data Collection** (If no automated pipeline)

For each ticker, you need 3 data points:

#### **1. Market Data (Defensive Features)**

Use Yahoo Finance (free):

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_market_data(ticker):
    """Collect defensive features for one ticker."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get 60 days of price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) < 30:
            return None
        
        # Calculate features
        current_price = hist['Close'].iloc[-1]
        price_60d_ago = hist['Close'].iloc[0]
        returns = hist['Close'].pct_change().dropna()
        
        # Get XBI for correlation
        xbi = yf.Ticker('XBI')
        xbi_hist = xbi.history(start=start_date, end=end_date)
        xbi_returns = xbi_hist['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(xbi_returns.index)
        
        features = {
            "price_current": f"{current_price:.2f}",
            "price_60d_ago": f"{price_60d_ago:.2f}",
            "return_60d": f"{(current_price/price_60d_ago - 1):.4f}",
            "vol_60d": f"{returns.std() * (252**0.5):.4f}",
            "drawdown_60d": f"{(hist['Close'].min() / hist['Close'].max() - 1):.4f}",
            "corr_xbi": f"{returns.loc[common_dates].corr(xbi_returns.loc[common_dates]):.4f}",
            "rsi_14d": "50.0",  # Simplified
            "vol_regime": "normal",
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        return features
    
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return None

# Collect for all tickers
for ticker in tickers:
    features = collect_market_data(ticker)
    if features:
        # Save to database/file
        pass
```

**Time: ~2 hours for 200 tickers (rate-limited)**

#### **2. Financial Data**

Use SEC EDGAR (free) or FinancialModelingPrep API ($30/month):

```python
import requests

def get_financial_data(ticker):
    """Get latest financials from FMP API."""
    api_key = "YOUR_FMP_API_KEY"  # Get from financialmodelingprep.com
    
    # Balance sheet
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=1&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()[0]
        return {
            "cash_usd": str(data.get('cashAndShortTermInvestments', 0)),
            "total_debt_usd": str(data.get('totalDebt', 0)),
            "quarterly_burn_usd": "50000000",  # Need to calculate from cash flow
            "runway_quarters": "10"  # Calculate: cash / quarterly_burn
        }
    return None
```

**Time: ~1 hour for 200 tickers (API)**

#### **3. Clinical Data**

Use ClinicalTrials.gov API (free):

```python
def get_clinical_data(ticker):
    """Get lead program from ClinicalTrials.gov."""
    # Map ticker to company name
    company_name = ticker_to_company[ticker]
    
    url = f"https://clinicaltrials.gov/api/query/study_fields"
    params = {
        "expr": company_name,
        "fields": "NCTId,Phase,Condition,OverallStatus",
        "max_rnk": 100,
        "fmt": "json"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        trials = response.json().get('StudyFieldsResponse', {}).get('StudyFields', [])
        
        if trials:
            # Find most advanced trial
            lead = trials[0]  # Simplified
            return {
                "phase": lead.get('Phase', ['Unknown'])[0],
                "indication": lead.get('Condition', ['Unknown'])[0],
                "trial_id": lead.get('NCTId', ['N/A'])[0]
            }
    
    return {
        "phase": "unknown",
        "indication": "unknown",
        "trial_id": "N/A"
    }
```

**Time: ~1 hour for 200 tickers**

---

## 🎯 **Step 3: Run Screener on Expanded Universe**

```powershell
# Once you have the full universe file:
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir etf_universe_data `
    --output etf_full_screening_results.json
```

**Expected runtime:** 3-5 minutes for 200 stocks

---

## 🎯 **Step 4: Analyze Results**

```powershell
python analyze_defensive_impact.py --output etf_full_screening_results.json
```

**Expected output:**
```
Total Securities: 200
Active (passed gates): ~150
Rankable: ~100

Position sizing:
  • 100 positions
  • Max weight: 0.0800 (8.00%)
  • Min weight: 0.0100 (1.00%)
  • Avg weight: 0.0090 (0.9%)

Defensive adjustments: 50-60% of securities
```

---

## 📊 **Recommended: Phased Approach**

### **Week 1: Top 50 Stocks**
- Easiest to collect data for (large caps)
- Validate full pipeline works
- Compare with your 21-stock results

### **Week 2: Expand to 100 Stocks**
- Add mid-caps
- Validate scaling
- Check computational performance

### **Week 3: Full 200 Stocks**
- Complete universe
- Final validation
- Production ready

---

## ⚠️ **Common Issues & Solutions**

### **Issue 1: Missing Data**
**Problem:** Not all 200 stocks have complete data  
**Solution:** Universe gates will filter them out (expected)

### **Issue 2: Rate Limiting**
**Problem:** Yahoo Finance limits API calls  
**Solution:** Add delays, cache data, use paid API

### **Issue 3: Computational Time**
**Problem:** 200 stocks takes too long  
**Solution:** Normal - 3-5 minutes is expected

### **Issue 4: Memory Usage**
**Problem:** System runs out of memory  
**Solution:** Process in batches (50 at a time)

---

## 💰 **Cost Breakdown**

### **Free Approach:**
- Market data: Yahoo Finance (rate-limited, slow)
- Financial: SEC EDGAR (requires parsing)
- Clinical: ClinicalTrials.gov (free API)
- **Total: $0/month**
- **Time: 4-6 hours first run, 1-2 hours updates**

### **Paid Approach:**
- Market data: Polygon.io ($200/month) or Alpha Vantage ($50/month)
- Financial: FinancialModelingPrep ($30/month)
- Clinical: ClinicalTrials.gov (free)
- **Total: $50-230/month**
- **Time: 1 hour first run, 15 minutes updates**

---

## 🎯 **Quick Start Recommendation**

**Start with Approach B (Phased Rollout):**

```powershell
# Week 1: Top 50
# Create list of top 50 large-cap biotech stocks
# These are easiest to get data for

# Run your existing data pipeline on just these 50:
python collect_universe_data.py --tickers VRTX,REGN,AMGN,... --as-of-date 2026-01-06

# Run screener:
python run_screen.py --as-of-date 2026-01-06 --data-dir top_50_data --output top_50_results.json

# Validate it works, then expand to 100, then 200
```

**This approach:**
- ✅ Lower risk (validate at each step)
- ✅ Faster feedback (see results weekly)
- ✅ Easier debugging (smaller batches)
- ✅ Builds confidence incrementally

---

## 📋 **Your Action Items**

### **Today:**
```
□ Run: python fetch_etf_constituents.py --output etf_template.json --template-only
□ Review the 200 tickers generated
□ Decide: Phased (50→100→200) or Full (200 immediately)
□ Check: Do you have wake_robin_data_pipeline working?
```

### **This Week:**
```
□ Collect data for first batch (21/50/200 stocks)
□ Run screener on expanded universe
□ Validate defensive overlays work at scale
□ Document any issues
```

### **Next Week:**
```
□ Expand to next batch if phased
□ Compare results across universe sizes
□ Finalize data collection approach
□ Document production process
```

---

## 🎊 **Ready to Start?**

**Recommended first command:**

```powershell
# Generate the full ticker list:
python fetch_etf_constituents.py --output etf_universe_template.json --template-only

# Then decide your approach based on what you see
```

---

**Which approach do you want to take?**
- **A:** Use existing pipeline (if you have wake_robin_data_pipeline)
- **B:** Phased rollout (50 → 100 → 200)
- **C:** Manual collection (build from scratch)

Let me know and I'll give you the exact commands for that approach! 🚀
