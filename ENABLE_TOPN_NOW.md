# 🎯 ENABLE TOP-N SELECTION (5 Minutes)

## What This Does:
Reduces portfolio from 98 stocks → 60 stocks (top performers only)
**Result:** Max weight increases from 3.69% → ~5.2% (better conviction)

---

## Step 1: Find The Line (2 min)

```powershell
# Search for where defensive overlay is applied
Select-String -Path *.py -Pattern "enrich_with_defensive_overlays" | Select-Object Filename, LineNumber
```

Look for the file that calls `enrich_with_defensive_overlays` with real data (not the adapter itself).

Likely files:
- `module_5_composite_with_defensive.py`
- `run_screen.py`
- `screening_pipeline.py`

---

## Step 2: Add One Parameter (2 min)

Find this code:
```python
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
)
```

Change to:
```python
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    top_n=60,  # ← ADD THIS LINE
)
```

---

## Step 3: Re-Run (1 min)

```powershell
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_TOP60_ENABLED.json
```

---

## Expected Changes:

### Before (All 98 stocks):
```
Position sizing:
  • 98 positions
  • Max weight: 0.0369 (3.69%)
  • Top 10 concentration: 24.7%
```

### After (Top 60 only):
```
Position sizing:
  • 60 positions  ← 38 excluded
  • Max weight: 0.0520 (5.20%)  ← +41%!
  • Top 10 concentration: 32.5%  ← Higher conviction
```

### Rankings Stay Same:
```
Top 10 (same order):
1. GILD  59.14  5.20%  ← Was 3.20%, now 5.20%!
2. VRTX  57.69  4.50%  ← Was 2.79%, now 4.50%!
3. PCRX  55.66  2.50%  ← Was 1.56%, now 2.50%!
...

Stocks 61-98: 0.00% (excluded, marked "NOT_IN_TOP_N")
```

---

## How To Check It Worked:

Look for in the output:
```
Top 10 holdings:
Rank  Ticker  Score     Weight    Def Notes
------------------------------------------------------------
1     GILD    59.14     0.0520    def_mult_elite_1.40  ← >5%!
```

If max weight is still ~3.69%, top-N wasn't applied. Check you added the parameter to the RIGHT call (the one with real data, not a test).

---

## Quick Validation:

```python
# Quick check
import json
data = json.load(open('screening_TOP60_ENABLED.json'))
ranked = data['module_5_composite']['ranked_securities']

invested = sum(1 for s in ranked if float(s['position_weight']) > 0)
excluded = sum(1 for s in ranked if 'NOT_IN_TOP_N' in s.get('position_flags', []))

print(f"Invested: {invested} (should be 60)")
print(f"Excluded: {excluded} (should be 38)")
print(f"Max weight: {max(float(s['position_weight']) for s in ranked)*100:.2f}% (should be ~5.2%)")
```

---

## If It Doesn't Work:

**Problem:** Still seeing 98 positions

**Solution:** You edited the adapter file, not the caller.

Find the ACTUAL call:
```powershell
# Show context around each call
Select-String -Path *.py -Pattern "enrich_with_defensive_overlays" -Context 5 | Out-String
```

Look for the call that has `composite_output` or similar real data variable.

---

## Alternative: Command Line Flag

If you want to toggle top-N easily, make it a CLI argument:

In `run_screen.py`:
```python
parser.add_argument("--top-n", type=int, default=None,
                   help="Select top N securities only")

# Then pass to defensive overlay:
output = enrich_with_defensive_overlays(
    ...,
    top_n=args.top_n,  # ← Use CLI arg
)
```

Usage:
```powershell
# No top-N
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output full.json

# With top-60
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --top-n 60 --output top60.json
```

This lets you compare both versions easily!
