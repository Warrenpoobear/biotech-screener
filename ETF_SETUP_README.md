# ETF Universe Setup - Automatic vs Manual

---

## ⚡ **OPTION 1: Fully Automatic (Try This First)**

### **One Command:**
```powershell
python setup_complete_etf_universe.py
```

**What it does:**
1. Automatically downloads XBI, IBB, NBI holdings
2. Imports CSVs into JSON
3. Adds missing tickers to your universe
4. Verifies 100% coverage

**Time:** 2-3 minutes (fully automated)

### **Or Step-by-Step Automatic:**
```powershell
# Step 1: Auto-download
python auto_download_etf_holdings.py

# Step 2: Import
python import_etf_csvs.py

# Step 3: Add to universe
python add_etf_tickers_to_universe.py

# Step 4: Verify
python check_etf_coverage.py --universe production_data/universe.json
```

---

## 🖱️ **OPTION 2: Manual Download (If Automatic Fails)**

### **If auto-download doesn't work for all ETFs:**

```powershell
# Step 1: Create directory
mkdir etf_csvs

# Step 2: Manual downloads (use browser):
#   XBI: https://www.ssga.com/us/en/individual/etfs/funds/xbi
#   IBB: https://www.ishares.com/us/products/239699/
#   NBI: https://indexes.nasdaqomx.com/Index/Weighting/NBI
# Save all to etf_csvs/ directory

# Step 3: Import
python import_etf_csvs.py

# Step 4: Add to universe
python add_etf_tickers_to_universe.py

# Step 5: Verify
python check_etf_coverage.py --universe production_data/universe.json
```

**Time:** 10-15 minutes (with manual clicking)

---

## 📊 **What You'll Get**

### **Expected Results:**
```
XBI holdings: 165
IBB holdings: 267
NBI holdings: 262
Total unique: ~298 tickers

Your universe: 97 → 298 tickers
ETF coverage: 100% ✅
```

### **After Module 1 Filtering:**
```
Starting: 298 tickers
Filtered: ~148 (no data, illiquid, etc.)
Active: 150 tickers
```

---

## ⚠️ **Which Method Should You Use?**

### **Try Automatic First:**
```powershell
python setup_complete_etf_universe.py
```

**Why automatic works:**
- **IBB:** iShares has direct API (✅ high success rate)
- **XBI:** SPDR has predictable URLs (⚠️ medium success rate)
- **NBI:** Nasdaq API (⚠️ medium success rate)

### **If Automatic Fails:**
Some ETF providers use JavaScript-heavy sites or anti-scraping measures. The script will tell you which ones failed and provide manual download instructions.

**Fallback to manual for failed ETFs:**
- Script shows: "❌ XBI Failed - Download manually"
- Just download that one manually
- Re-run `import_etf_csvs.py`

---

## 🎯 **Quick Start (Recommended)**

```powershell
# Try automatic first
python setup_complete_etf_universe.py

# If it says "3/3 successful" → You're done! ✅
# If it says "2/3 successful" → Download missing one manually, then:
python import_etf_csvs.py
python add_etf_tickers_to_universe.py
```

---

## 📁 **Files You Have**

### **Automatic Download:**
- `auto_download_etf_holdings.py` - Downloads XBI, IBB, NBI
- `setup_complete_etf_universe.py` - One-command full setup

### **Import & Add:**
- `import_etf_csvs.py` - Import CSVs to JSON
- `add_etf_tickers_to_universe.py` - Add to universe

### **Verification:**
- `check_etf_coverage.py` - Verify coverage

---

## 🆘 **Troubleshooting**

### **"Connection timeout" or "403 Forbidden"**
Some ETF sites block automated downloads.

**Solution:** Use manual download for that ETF

### **"pandas not found" (for XBI Excel conversion)**
```powershell
pip install pandas openpyxl
```

Or manually open XBI Excel file and Save As CSV.

### **"SSL Certificate error"**
```powershell
pip install --upgrade certifi requests
```

---

## 🎊 **Bottom Line**

1. **Try automatic first:** `python setup_complete_etf_universe.py`
2. **If some fail:** Download those manually, then run import script
3. **Result:** 100% ETF coverage in 2-15 minutes

**Start now!** 🚀
