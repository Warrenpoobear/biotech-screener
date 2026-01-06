# 📊 Historical Backtesting Guide

## Overview

Three tools for historical validation of defensive overlays:

1. **`analyze_defensive_impact.py`** - Analyze a single output file
2. **`run_historical_batch.ps1`** - Run pipeline across multiple dates
3. **`backtest_defensive_overlays.py`** - Full automated backtesting framework

---

## 🚀 Quick Start: Analyze Current Output

```powershell
# Analyze your current successful run:
python analyze_defensive_impact.py --output outputs/ranked_with_real_defensive_FINAL.json
```

**Expected Output:**
```
============================================================
DEFENSIVE OVERLAYS IMPACT ANALYSIS
============================================================

As of Date: 2026-01-06
Total Securities: 21

Position Sizing Analysis
  Weight Distribution:
    Sum:          0.9000 (target: 0.9000)
    Mean:         0.0429
    Min:          0.0100 (1.00%)
    Max:          0.0800 (8.00%)
    Range:        0.0700 (8.0:1 ratio)
  
  Concentration:
    Top 5:        0.3296 (33.0%)
    Top 10:       0.5844 (58.4%)

Defensive Adjustments
  Securities adjusted: 13/21 (61.9%)
  
  Adjustment Types:
    def_mult_low_corr_1.05           13 securities
    def_warn_drawdown_gt_30pct       1 security

Top 10 Positions by Weight
  1. ARGX    0.0800 (8.00%)   - Low vol, diversifier
  2. UTHR    0.0706 (7.06%)   - Low vol, diversifier
  ...
```

---

## 📅 Method 1: Simple Batch Runner (Recommended)

### **Step 1: Download the batch script**
Download **`run_historical_batch.ps1`** from above.

### **Step 2: Run batch backtest**
```powershell
# Run for 14 weekly dates (Oct 2025 - Jan 2026):
.\run_historical_batch.ps1 -DataDir production_data -OutputDir backtest_results

# Output:
# backtest_results/
#   ├── ranked_2025-10-07.json
#   ├── ranked_2025-10-14.json
#   ├── ...
#   ├── ranked_2026-01-06.json
#   └── batch_summary.csv
```

### **Step 3: Analyze each result**
```powershell
# Analyze specific dates:
python analyze_defensive_impact.py --output backtest_results/ranked_2025-10-07.json
python analyze_defensive_impact.py --output backtest_results/ranked_2025-11-15.json
python analyze_defensive_impact.py --output backtest_results/ranked_2026-01-06.json

# Or analyze all:
Get-ChildItem backtest_results\ranked_*.json | ForEach-Object {
    Write-Host "`n=== $($_.Name) ===" -ForegroundColor Cyan
    python analyze_defensive_impact.py --output $_.FullName
}
```

---

## 🔬 Method 2: Full Automated Backtest

### **For comprehensive analysis with comparisons:**

```powershell
# Download backtest_defensive_overlays.py

# Run automated backtest:
python backtest_defensive_overlays.py `
  --start-date 2025-10-01 `
  --end-date 2026-01-06 `
  --frequency weekly `
  --data-dir production_data `
  --output-dir backtest_results

# Generates:
#   backtest_results/
#     ├── backtest_report.txt        # Summary report
#     ├── backtest_results.json      # Detailed data
#     └── with_defensive_*.json      # Individual runs
```

---

## 📊 What to Look For

### **1. Weight Distribution Stability**
- **Good:** Weight range stays 0.06-0.08 (6-8:1 ratio)
- **Concern:** Range collapses to 0.01-0.02 (equal weight fallback)
- **Action:** If equal weighting occurs, check volatility data quality

### **2. Defensive Adjustment Frequency**
- **Expected:** 40-70% of securities adjusted
- **Good sign:** Consistent percentage across dates
- **Concern:** 0% or 100% adjusted (data issue or extreme market)

### **3. Position Sizing Patterns**
- **Top 5 concentration:** Should be 25-35%
- **Top 10 concentration:** Should be 50-65%
- **Look for:** Consistent inverse-vol relationship

### **4. Rank Stability**
- **Rank correlation:** Should be 0.90-0.95
- **Top 5 overlap:** Should be 3-4 out of 5
- **Means:** Defensive overlays make small, thoughtful adjustments

### **5. Score Impact**
- **Mean change:** ±2-3 points (5% adjustments)
- **Max change:** ±5 points (correlation outliers)
- **Distribution:** Most securities unchanged, outliers adjusted

---

## 📈 Sample Analysis Output

After analyzing multiple dates, you might see:

```
Date        Securities  Weight Range  Def Adj %  Top Position
--------------------------------------------------------------
2025-10-07  21          0.0720        57.1%      ARGX (7.8%)
2025-10-14  21          0.0695        61.9%      UTHR (7.5%)
2025-10-21  20          0.0680        60.0%      ARGX (7.9%)
2025-10-28  21          0.0710        52.4%      INCY (7.6%)
...
2026-01-06  21          0.0700        61.9%      ARGX (8.0%)

Average:    21.0        0.0701        58.7%      -
Std Dev:    0.4         0.0015        3.8%       -
```

**Interpretation:**
- ✅ Weight range stable (~0.07)
- ✅ Adjustment frequency consistent (53-62%)
- ✅ Diversified leadership (ARGX, UTHR, INCY rotate)

---

## 🎯 Expected Findings

### **What Success Looks Like:**

1. **Position Sizing Working:**
   - Clear weight variation (not all equal)
   - Low vol = larger positions
   - High vol = smaller positions
   - Range: 6-8:1 ratio

2. **Correlation Adjustments Applied:**
   - 40-70% securities get adjustments
   - Mostly low-correlation bonuses (1.05x)
   - Occasional high-correlation penalties (0.95x)

3. **Ranking Stability:**
   - Top 10 mostly stable
   - Rank correlation >0.90
   - Small, sensible reorderings

4. **Risk Flags Working:**
   - Drawdown warnings when appropriate
   - Volatility expansion flags
   - RSI extremes noted

### **Red Flags to Watch:**

- ⚠️ All weights equal → volatility data missing
- ⚠️ 0% adjustments → correlation data issue
- ⚠️ Rank correlation <0.70 → too much churn
- ⚠️ Weight range <0.02 → insufficient differentiation

---

## 🔧 Troubleshooting

### **Issue: Equal weights across all dates**
```
Max weight = Min weight = 0.0429
```
**Cause:** No volatility variation in data  
**Fix:** Ensure universe file has varying `vol_60d` values

### **Issue: 0% defensive adjustments**
```
Defensive adjustments: 0/21 (0.0%)
```
**Cause:** All correlations in neutral zone (0.40-0.80)  
**Status:** This is actually **normal and healthy**! No adjustments means no extreme crowding or outliers.

### **Issue: Pipeline fails on historical dates**
```
ERROR: Universe file not found
```
**Fix:** Ensure universe snapshots exist for historical dates, or use `production_data` with latest snapshot

---

## 📝 Quick Reference Commands

```powershell
# 1. Analyze single output:
python analyze_defensive_impact.py --output outputs/ranked_YYYY-MM-DD.json

# 2. Run batch backtest:
.\run_historical_batch.ps1 -OutputDir backtest_results

# 3. Analyze all results:
ls backtest_results\ranked_*.json | ForEach-Object {
    python analyze_defensive_impact.py --output $_.FullName
} > backtest_analysis_full.txt

# 4. Compare two dates:
python analyze_defensive_impact.py --output backtest_results/ranked_2025-10-07.json > oct.txt
python analyze_defensive_impact.py --output backtest_results/ranked_2026-01-06.json > jan.txt
code --diff oct.txt jan.txt
```

---

## 🎯 Next Steps

1. **Run analysis on current output** (verify it works)
2. **Run batch for last 3 months** (if historical data available)
3. **Review weight distribution patterns**
4. **Check adjustment frequency trends**
5. **Validate rank stability**

---

## 💡 Tips

- Start with **current output analysis** to verify tool works
- Use **batch runner** for systematic historical testing
- Save analysis outputs to text files for comparison
- Track key metrics over time (weight range, adjustment %, concentration)
- Compare periods of high/low volatility

---

**Ready to start? Run the analysis on your current output first!** 🚀
