# Phase 2: Expand to 100 Stocks

## 🎯 Goal
Expand from 44 → 100 stocks to test institutional-scale portfolio construction.

---

## ✅ Current State
- 44 stocks validated
- Defensive overlays proven working
- Volatility range: 23% to 130%
- Top 10 concentration: 33%

---

## 🚀 Quick Start (5 Minutes)

### **Step 1: Run Extension Script (~30 seconds)**

```powershell
cd C:\Projects\biotech_screener\biotech-screener

# Download extend_to_100_stocks.py from above

# Run it:
python extend_to_100_stocks.py `
    --base top50_universe.json `
    --output universe_100stocks.json `
    --as-of-date 2026-01-06
```

**What happens:**
- Loads your 44 existing stocks
- Adds 56 new mid/small-cap stocks
- Collects real market data (Yahoo Finance)
- Creates 100-stock universe

**Expected runtime:** ~30 seconds (56 stocks × 0.5s delay)

---

### **Step 2: Use New Universe (5 seconds)**

```powershell
# Backup current:
Copy-Item production_data/universe.json production_data/universe_44stocks_backup.json

# Use 100-stock universe:
Copy-Item universe_100stocks.json production_data/universe.json
```

---

### **Step 3: Run Screener (~2 minutes)**

```powershell
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir production_data `
    --output screening_100stocks.json
```

**Expected runtime:** ~2 minutes (vs 1 minute for 44 stocks)

---

### **Step 4: Analyze Results (10 seconds)**

```powershell
python analyze_defensive_impact.py --output screening_100stocks.json
```

---

## 📊 Expected Results with 100 Stocks

### **Position Sizing:**
```
Active securities: ~90 (passed gates)
Rankable: ~80-85

Position weights:
  • Max: 4-5% (lowest vol stocks)
  • Min: 0.9-1.0% (highest vol stocks)
  • Average: ~1.1% (90% / 80 positions)
  • Range: 4-5:1 ratio

Top 10 concentration: ~20-25% (excellent!)
Top 20 concentration: ~40-45%
```

### **Defensive Adjustments:**
```
Securities adjusted: 30-40% (~25-35 stocks)

Low correlation bonuses: 15-25 stocks
High correlation penalties: 5-10 stocks
Drawdown warnings: 5-10 stocks
Volatility expansion: 5-10 stocks
```

### **Comparison with 44 Stocks:**

| Metric | 44 Stocks | 100 Stocks | Change |
|--------|-----------|------------|--------|
| Universe | 44 | ~95 | +115% |
| Avg position | 2.05% | ~1.1% | -46% |
| Max position | 4.09% | ~4.5% | +10% |
| Top 10 conc. | 32.9% | ~23% | -30% |
| Top 20 conc. | - | ~42% | - |

---

## 🎯 What We're Testing

### **1. Diversification at Scale**
- Can system handle 100 active positions?
- Does concentration continue to decrease?
- Are weights distributed sensibly?

### **2. Computational Performance**
- Does runtime scale linearly? (Should be ~2 minutes)
- Memory usage acceptable? (Should be <500MB)
- Any bottlenecks or slowdowns?

### **3. Defensive Overlay Behavior**
- Do extreme stocks still get extreme treatment?
- Does weight range stay reasonable (4-5:1)?
- Are adjustments proportional to universe size?

### **4. Robustness**
- How many stocks fail data validation?
- Does system handle missing data gracefully?
- Any edge cases or errors?

---

## 📋 56 New Stocks Being Added

### **By Market Cap:**

**Mid-Caps ($1-5B):**
- AGIO, ARWR, BCRX, CRBU, CRVS, CVAC
- CYTK, DRMA, DVAX, FGEN, GLPG, HRTX
- ICPT, ITCI, MDGL, NVAX, PCRX, SAGE
- SANA, SGMO, SNDX, TGTX

**Small-Caps ($500M-$1B):**
- AKRO, ALLO, APLS, ARCT, AVTR, BDTX
- BGXX, BPMC, CLDX, CRNX, DNLI, ETNB
- FIXX, FULC, HALO, IMCR, IMMP, IRWD
- KALA, KALV, KRYS, LQDA, MCRB, MRSN
- NVCR, OPCH, PRTA, RVMD, RXRX, RGNX
- SPRY, TARS, TBPH, TCDA

**Micro-Caps (<$500M):**
- IMVT, KYMR, MRVI, OCGN, PGEN, PTCT

These are all liquid, exchange-listed biotechs with clinical programs.

---

## ⚠️ Expected Issues

### **Normal/Expected:**
- **10-15% failures** (~5-8 stocks) due to delisted/merged companies
- **Some missing data** - normal for small-caps
- **Higher vol range** - micro-caps can be very volatile
- **Lower correlation on average** - more diversified names

### **Red Flags (Should NOT See):**
- Screener crashing or hanging
- Memory errors
- Weights not summing to 90%
- All positions getting equal weights
- Max position >8% or min <0.5%

---

## 🎯 Success Criteria

### **After running on 100 stocks, you should see:**

✅ **Pipeline completes** (~2 minutes)  
✅ **80-90 rankable securities**  
✅ **Average weight ~1.1%**  
✅ **Top 10 concentration 20-25%**  
✅ **Weight range 4-5:1**  
✅ **Defensive adjustments 30-40%**  
✅ **No crashes or errors**  

---

## 📈 Key Observations to Make

### **1. Weight Distribution**
```powershell
# Check if weights are well-distributed:
python analyze_defensive_impact.py --output screening_100stocks.json
```

Look for:
- Smooth gradient from max to min
- No clustering or gaps
- Reasonable concentration

### **2. Volatility Sorting**
- Lowest vol stocks should still get largest weights
- Highest vol stocks should get smallest weights
- Extreme cases (>100% vol) should hit minimum

### **3. System Performance**
- Runtime should be ~2x the 44-stock run
- Memory usage should be reasonable
- No errors or warnings

---

## 🔄 Comparison Commands

```powershell
# Compare 44 vs 100 stock results side-by-side:

Write-Host "44-Stock Results:" -ForegroundColor Cyan
python analyze_defensive_impact.py --output screening_44stocks.json | Select-String "Sum:|Max:|Top"

Write-Host "`n100-Stock Results:" -ForegroundColor Green  
python analyze_defensive_impact.py --output screening_100stocks.json | Select-String "Sum:|Max:|Top"
```

---

## 🎊 After Validation

Once 100 stocks works well:

### **Option 1: Continue to 200 (Full ETF Universe)**
- Add final 100 stocks
- Complete XBI/IBB/NBI coverage
- Full production deployment

### **Option 2: Start Forward Testing**
- Run weekly screenings
- Track portfolio changes
- Monitor performance
- Build deployment confidence

### **Option 3: Documentation & Review**
- Document methodology
- Create operations manual
- Prepare for IC review
- Plan live deployment

---

## 🚀 Ready to Run?

```powershell
# Full workflow (copy/paste):

# 1. Extend universe:
python extend_to_100_stocks.py --base top50_universe.json --output universe_100stocks.json

# 2. Use it:
Copy-Item universe_100stocks.json production_data/universe.json

# 3. Screen:
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_100stocks.json

# 4. Analyze:
python analyze_defensive_impact.py --output screening_100stocks.json
```

**Total time: ~3 minutes**

---

## 📝 Notes

### **Why These 56 Stocks?**
- All from XBI/IBB/NBI constituent lists
- Liquid enough for reliable data
- Clinical-stage companies (typical biotech)
- Diverse therapeutic areas
- Mix of market caps

### **Data Quality:**
- Market data (defensive features): REAL from Yahoo Finance
- Financial/Clinical: Placeholders (sufficient for testing)
- For production: Would need real SEC/ClinicalTrials.gov data

### **Performance:**
- 100 stocks is a "sweet spot" for testing
- Large enough to show diversification benefits
- Small enough to iterate quickly
- Representative of focused biotech fund

---

**Ready? Run the commands above!** 🚀
