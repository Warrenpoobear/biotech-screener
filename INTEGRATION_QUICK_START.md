# Defensive Overlays Integration - Quick Start

## 🎯 **Zero-Refactor Integration** (Recommended)

Your existing `module_5_composite.py` stays **unchanged**. Add defensive overlays with 3 files:

### **Files to Download:**
1. `defensive_overlay_adapter.py` - Helper functions
2. `module_5_composite_with_defensive.py` - Wrapper for your Module 5
3. Your existing `module_5_composite.py` - **No changes needed!**

### **Installation:**
```powershell
# Place all 3 files in the same directory as your module_5_composite.py
cd C:\Projects\biotech_screener\biotech-screener\wake_robin_scoring

# Copy the new files here (download from Claude)
# defensive_overlay_adapter.py
# module_5_composite_with_defensive.py
```

### **Integration (2-line change):**

**BEFORE:**
```python
from module_5_composite import rank_securities

output = rank_securities(
    scores_by_ticker=scores_by_ticker,
    active_tickers=active_tickers,
    as_of_date=as_of_date,
    normalization="cohort",
    cohort_mode="stage_only",
)
```

**AFTER:**
```python
from module_5_composite_with_defensive import rank_securities_with_defensive

output = rank_securities_with_defensive(
    scores_by_ticker=scores_by_ticker,
    active_tickers=active_tickers,
    as_of_date=as_of_date,
    normalization="cohort",
    cohort_mode="stage_only",
    validate=True,  # Optional: print validation diagnostics
)
```

That's it! 🎉

---

## **What You Get:**

### **1. Defensive Score Adjustments**
Each security gets correlation-based multiplier (0.95x - 1.05x):
```json
{
  "ticker": "VRTX",
  "composite_score": "89.78",  // After defensive adjustment
  "composite_score_before_defensive": "85.50",  // Original score
  "defensive_multiplier": "1.05",
  "defensive_notes": ["def_mult_low_corr_1.05"]
}
```

### **2. Position Weights**
Inverse-volatility sizing with caps:
```json
{
  "ticker": "VRTX",
  "position_weight": "0.0625",  // 6.25% position
  "composite_score": "89.78"
}
```

### **3. Validation Diagnostics**
When `validate=True`:
```
============================================================
DEFENSIVE OVERLAY VALIDATION
============================================================
✓ Weights sum: 0.9000 (target: 0.9000)
✓ All excluded securities have zero weight
✓ 18/20 securities have defensive adjustments

Position sizing:
  • 18 positions
  • Max weight: 0.0800 (8.00%)
  • Min weight: 0.0100 (1.00%)
  • Avg weight: 0.0500

Top 10 holdings:
Rank  Ticker  Score     Weight    Def Notes
------------------------------------------------------
1     VRTX    89.78     0.0625    def_mult_low_corr_1.05
2     ALNY    87.45     0.0580    -
3     INCY    85.12     0.0550    def_mult_high_corr_0.95
...
============================================================
```

---

## **Testing Your Integration:**

### **Test 1: Verify Files Work**
```python
# test_defensive_integration.py
from module_5_composite_with_defensive import rank_securities_with_defensive

# Use your existing scores_by_ticker dict
output = rank_securities_with_defensive(
    scores_by_ticker=scores_by_ticker,
    active_tickers=active_tickers,
    as_of_date="2026-01-06",
    validate=True,  # This prints diagnostics
)

# Check output
print(f"\nTotal securities: {len(output['ranked_securities'])}")
print(f"With defensive notes: {sum(1 for r in output['ranked_securities'] if r.get('defensive_notes'))}")
print(f"\nTop 3:")
for r in output['ranked_securities'][:3]:
    print(f"  {r['composite_rank']}. {r['ticker']}: {r['composite_score']} (weight: {r['position_weight']})")
```

### **Test 2: Compare Before/After**
```python
# Check which scores changed
for r in output['ranked_securities'][:10]:
    before = r.get('composite_score_before_defensive', r['composite_score'])
    after = r['composite_score']
    if before != after:
        delta = float(after) - float(before)
        print(f"{r['ticker']}: {before} → {after} ({delta:+.2f})")
```

---

## **Configuration Options:**

### **Disable Defensive Multiplier** (just do position sizing):
```python
output = rank_securities_with_defensive(
    ...,
    apply_defensive_multiplier=False,  # Skip score adjustments
    apply_position_sizing=True,         # Still calculate weights
)
```

### **Disable Position Sizing** (just do score adjustments):
```python
output = rank_securities_with_defensive(
    ...,
    apply_defensive_multiplier=True,    # Apply score adjustments
    apply_position_sizing=False,        # Skip weight calculation
)
```

### **Adjust Position Sizing Params:**
Edit `defensive_overlay_adapter.py` line 125:
```python
apply_caps_and_renormalize(
    ranked,
    cash_target=Decimal("0.15"),  # Change to 15% cash
    max_pos=Decimal("0.10"),      # Change max position to 10%
    min_pos=Decimal("0.01"),      # Keep min at 1%
)
```

---

## **Troubleshooting:**

### **Problem: ImportError**
```
ModuleNotFoundError: No module named 'module_5_composite'
```
**Fix:** Ensure all 3 files are in the same directory

### **Problem: Weights don't sum to 0.9000**
```
⚠️  Weights sum: 0.8523, expected 0.9000
```
**Fix:** Check that `defensive_features` exist in your `scores_by_ticker`
```python
# Debug:
print(scores_by_ticker["VRTX"].keys())
# Should see: 'defensive_features' in output
```

### **Problem: All weights are equal**
**Cause:** No volatility data available
**Fix:** Ensure `vol_60d` is in defensive_features:
```python
# Check:
print(scores_by_ticker["VRTX"]["defensive_features"]["vol_60d"])
```

### **Problem: Scores didn't change**
**Cause:** Correlations are all between 0.40-0.80 (neutral zone)
**Expected:** This is normal! Most stocks should be in neutral zone.
Only outliers get adjustments.

---

## **What's Happening Under the Hood:**

### **Flow:**
```
scores_by_ticker (with defensive_features)
    ↓
rank_securities() [your existing Module 5]
    ↓
enrich_with_defensive_overlays()
    ├─→ Apply correlation multiplier (0.95-1.05x)
    ├─→ Calculate inverse-vol weights
    ├─→ Apply caps (8% max, 1% min)
    └─→ Renormalize to 90% invested
    ↓
output (with position_weight and defensive_notes)
```

### **Key Functions:**

1. **defensive_multiplier()** - Correlation-based adjustment
   - High correlation (>0.80): 0.95x penalty (crowded trade)
   - Low correlation (<0.40): 1.05x bonus (diversification)
   - Neutral (0.40-0.80): 1.00x (no change)

2. **raw_inv_vol_weight()** - Inverse volatility sizing
   - weight = 1 / vol_60d
   - Lower volatility → larger position

3. **apply_caps_and_renormalize()** - Position sizing
   - Max 8% per position
   - Min 1% per position
   - Sum to 90% (10% cash reserve)
   - Excluded securities get 0%

---

## **Next Steps:**

1. ✅ Download 2 files (defensive_overlay_adapter.py, module_5_composite_with_defensive.py)
2. ✅ Place in same directory as module_5_composite.py
3. ✅ Change 1 import line in your pipeline
4. ✅ Run with validate=True to see diagnostics
5. ✅ Check output has position_weight field
6. ✅ Done!

---

## **Advanced: Direct Modification**

If you prefer to modify `module_5_composite.py` directly instead of using a wrapper, see `SURGICAL_PATCHES_module_5.txt` for exact line-by-line changes.

**Pros:** Single file, no wrapper
**Cons:** Requires modifying your existing Module 5 code

The wrapper approach is recommended for easy updates and testing.
