# Bug Fix - NaN/Infinity Handling

## 🐛 **The Bug:**

```
decimal.InvalidOperation: [<class 'decimal.InvalidOperation'>]
File "defensive_overlay_adapter.py", line 56, in sanitize_corr
    if corr < Decimal("-1") or corr > Decimal("1"):
       ^^^^^^^^^^^^^^^^^^^^
```

## 🔍 **Root Cause:**

Some stocks have **NaN (Not a Number)** or **Infinity** correlation values in the data. This happens when:
- Not enough overlapping price data with XBI
- Division by zero in correlation calculation
- Data collection edge cases

When Python's `Decimal` type encounters NaN or Infinity and you try to compare it (`corr < Decimal("-1")`), it raises `InvalidOperation` instead of returning True/False.

## ✅ **The Fix:**

**Added `.is_finite()` check BEFORE comparisons:**

```python
# BEFORE (Broken):
try:
    corr = Decimal(str(corr_s))
except Exception:
    flags.append("def_corr_parse_fail")
    return None, flags

# Treat placeholder as missing
if corr == PLACEHOLDER_CORR:
    flags.append("def_corr_placeholder_0.50")
    return None, flags

# Validate range ← CRASHES HERE if corr is NaN
if corr < Decimal("-1") or corr > Decimal("1"):
    flags.append("def_corr_out_of_range")
    return None, flags
```

```python
# AFTER (Fixed):
try:
    corr = Decimal(str(corr_s))
except Exception:
    flags.append("def_corr_parse_fail")
    return None, flags

# ✅ NEW: Check if finite BEFORE comparisons
if not corr.is_finite():
    flags.append("def_corr_not_finite")
    return None, flags

# Now safe to do equality check
if corr == PLACEHOLDER_CORR:
    flags.append("def_corr_placeholder_0.50")
    return None, flags

# Now safe to do range comparisons
if corr < Decimal("-1") or corr > Decimal("1"):
    flags.append("def_corr_out_of_range")
    return None, flags
```

## 📊 **What `.is_finite()` Does:**

```python
Decimal("0.5").is_finite()    # True
Decimal("-0.3").is_finite()   # True
Decimal("NaN").is_finite()    # False ← Catches this!
Decimal("Infinity").is_finite()  # False ← And this!
Decimal("-Infinity").is_finite() # False ← And this!
```

## 🎯 **Impact:**

Stocks with NaN/Infinity correlation will now:
- Get flagged with `"def_corr_not_finite"`
- Be treated as having **missing correlation data**
- **NOT get elite bonus** (correct behavior)
- **Still get position weights** based on volatility

## 🚀 **Next Steps:**

```powershell
# 1. Replace adapter (AGAIN, with NaN fix)
Copy-Item defensive_overlay_adapter_DYNAMIC_FLOOR.py defensive_overlay_adapter.py -Force

# 2. Re-run screener (should work now)
python run_screen.py `
    --as-of-date 2026-01-06 `
    --data-dir production_data `
    --output screening_100stocks_OPTIONC.json

# 3. Validate
python analyze_defensive_impact.py --output screening_100stocks_OPTIONC.json
```

## ✅ **Expected:**

```
[1/6] Loading input data...
[2/6] Module 1: Universe filtering...
  Active: 98, Excluded: 0
[3/6] Module 2: Financial health...
[4/6] Module 3: Catalyst analysis...
[5/6] Module 4: Clinical development...
[6/6] Module 5: Composite ranking...
DEBUG: Loading defensive features from production_data\universe.json
DEBUG: Extracted defensive_features for 97 tickers
✅ SUCCESS: Screening completed! ← Should see this
```

## 📋 **Sanity Check After Run:**

```powershell
python -c "
import json
data = json.load(open('screening_100stocks_OPTIONC.json'))
ranked = data['module_5_composite']['ranked_securities']

# Count different correlation issues
not_finite = sum(1 for s in ranked if 'def_corr_not_finite' in s.get('defensive_notes', []))
placeholder = sum(1 for s in ranked if 'def_corr_placeholder_0.50' in s.get('defensive_notes', []))
missing = sum(1 for s in ranked if 'def_corr_missing' in s.get('defensive_notes', []))

print(f'Correlation data issues:')
print(f'  NaN/Infinity: {not_finite}')
print(f'  Placeholder 0.50: {placeholder}')
print(f'  Missing field: {missing}')
print(f'  Total with issues: {not_finite + placeholder + missing}')
print(f'  Clean data: {len(ranked) - (not_finite + placeholder + missing)}')
"
```

## 💡 **Why This Happened:**

The `extend_universe_yfinance.py` script calculates correlation like this:

```python
corr = ticker_df['Close'].corr(xbi_df['Close'])
```

If there's insufficient overlapping data or other issues, pandas returns `NaN`. When this gets written to JSON and read back as a string, it becomes `"NaN"`, which Decimal can parse but can't compare.

## 🔮 **Future Prevention:**

In `extend_universe_yfinance.py`, you could add:

```python
import math

# After calculating correlation:
if pd.isna(corr) or math.isinf(corr) or not (-1 <= corr <= 1):
    features['corr_xbi'] = None  # Explicitly set to None
else:
    features['corr_xbi'] = str(corr)
```

But for now, the `sanitize_corr()` function handles it defensively!

---

**This is a one-line fix that makes the system robust to bad data.** ✅
