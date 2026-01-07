# 🔒 PRODUCTION HARDENING CHECKLIST

## Your Diagnosis Is Spot-On - Here's The Fix Plan

You've identified 4 critical issues that separate "it runs" from "production-ready":

---

## 🚨 **1. PIT Filtering: "0 filtered" Is A Red Flag**

### **The Issue:**
```python
# In transform_clinical_for_module4.py, we set:
'last_update_posted': None,  # ← No date!
'source_date': None,         # ← No date!
```

**Result:** Module 4 can't filter anything → Potential lookahead bias!

### **The Fix:**

#### **Option A: Collect Real Dates (Recommended for production)**

Update `collect_all_universe_data.py` to fetch study dates:

```python
def collect_clinical_data(ticker, company_name=None):
    # ... existing code ...
    
    for study in studies:
        protocol = study.get('protocolSection', {})
        status_module = protocol.get('statusModule', {})
        
        trial = {
            'nct_id': id_module.get('nctId'),
            'title': id_module.get('briefTitle'),
            'status': status_module.get('overallStatus'),
            'phase': design_module.get('phases', ['N/A'])[0],
            
            # ADD THESE DATE FIELDS:
            'last_update_posted': status_module.get('lastUpdatePostDate'),
            'study_first_posted': status_module.get('studyFirstPostDate'),
            'results_first_posted': status_module.get('resultsFirstPostDate'),
        }
```

Re-run collection, transform, and screening.

#### **Option B: Use Current Date As Proxy (Quick fix)**

Update `transform_clinical_for_module4.py`:

```python
from datetime import datetime

# In transform loop:
trial_record = {
    # ... existing fields ...
    'last_update_posted': datetime.now().strftime('%Y-%m-%d'),  # Use today as proxy
    'source_date': as_of_date,  # Use screening date
}
```

**Trade-off:** All trials appear "current" → No historical filtering, but avoids future lookahead.

### **Add PIT Diagnostics to Module 4:**

Create `module_4_clinical_dev_ENHANCED.py` with additional logging:

```python
def compute_module_4_clinical_dev(trial_records, active_tickers, as_of_date):
    # ... existing code ...
    
    # ADD DIAGNOSTIC COUNTS:
    trials_with_dates = sum(1 for t in trial_records if t.get('last_update_posted'))
    date_values = [t.get('last_update_posted') for t in trial_records if t.get('last_update_posted')]
    
    print(f"\n  PIT DIAGNOSTIC:")
    print(f"    Trials with dates: {trials_with_dates}/{len(trial_records)} ({trials_with_dates/len(trial_records)*100:.1f}%)")
    if date_values:
        print(f"    Date range: {min(date_values)} to {max(date_values)}")
    print(f"    PIT cutoff: {pit_cutoff}")
    print(f"    Filtered (future): {pit_filtered_count}")
    
    # ... rest of existing code ...
```

### **Expected Output After Fix:**
```
[5/6] Module 4: Clinical development...
  PIT DIAGNOSTIC:
    Trials with dates: 464/464 (100.0%)
    Date range: 2020-01-15 to 2025-12-28
    PIT cutoff: 2026-01-06T00:00:00Z
    Filtered (future): 0
  Scored: 98, Trials evaluated: 464, PIT filtered: 0
```

---

## 🎯 **2. Enable Top-N Selection NOW**

### **The Math Says You Should:**
```
Current:  98 stocks, max 3.69%, top-10 24.7%
With N=60: 60 stocks, max 5.2%, top-10 32.5%  ← Better conviction!
```

### **The Fix (5 minutes):**

**File:** `module_5_composite_with_defensive.py` (or wherever composite is called)

Find this code (probably around line 200-300):

```python
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    # top_n=None,  ← Currently not used
)
```

**Change to:**

```python
output = enrich_with_defensive_overlays(
    output=composite_output,
    scores_by_ticker=defensive_scores,
    apply_multiplier=True,
    apply_position_sizing=True,
    top_n=60,  # ← ENABLE THIS
)
```

### **Add Top-N Validation:**

In `defensive_overlay_adapter_DYNAMIC_FLOOR.py`, add to the end of `apply_caps_and_renormalize()`:

```python
def apply_caps_and_renormalize(..., top_n: Optional[int] = None):
    # ... existing code ...
    
    # VALIDATION: Verify top-N was applied correctly
    if top_n is not None:
        invested_count = sum(1 for r in records if Decimal(r['position_weight']) > 0)
        excluded_count = sum(1 for r in records if 'NOT_IN_TOP_N' in r.get('position_flags', []))
        
        print(f"\n  TOP-N VALIDATION:")
        print(f"    Target: {top_n}")
        print(f"    Invested: {invested_count}")
        print(f"    Excluded: {excluded_count}")
        print(f"    Total: {invested_count + excluded_count}")
        
        if invested_count != top_n:
            print(f"    ⚠️  WARNING: Expected {top_n} positions, got {invested_count}")
```

---

## 🔧 **3. Fix Module 2 & 3 (Same Pattern as Module 4)**

### **Module 2 Shows "Scored: 0" - Field Mismatch**

**Check what Module 2 expects:**

```powershell
# Show Module 2's expected input format
Get-Content module_2_financial.py | Select-String -Pattern "def compute_module_2" -Context 0,20
```

**Likely issue:** Module 2 expects `financial.json` but you might need `financial_records.json`

**Quick diagnostic:**

```python
# Add to beginning of Module 2:
def compute_module_2_financial(financial_records, active_tickers, as_of_date):
    print(f"\n  MODULE 2 DIAGNOSTIC:")
    print(f"    Input records: {len(financial_records)}")
    print(f"    Active tickers: {len(active_tickers)}")
    
    # Check sample record structure
    if financial_records:
        sample = financial_records[0]
        print(f"    Sample keys: {list(sample.keys())[:10]}")
        print(f"    Has 'ticker': {('ticker' in sample)}")
        print(f"    Has 'cash': {('cash' in sample)}")
    
    # ... rest of existing code ...
```

**Same fix pattern as Module 4:**
1. Check if `financial.json` exists
2. If not, transform `universe.json` financial data → `financial_records.json`
3. Update `run_screen.py` line 162 to load correct file

### **Module 3 Shows "With catalyst: 0" - Missing Data**

**Likely issue:** You don't have a `catalyst.json` or equivalent.

**Options:**

**A. Create placeholder catalyst data:**
```python
# create_placeholder_catalyst.py
import json

# Generate minimal catalyst records
catalyst_records = [
    {
        'ticker': ticker,
        'has_catalyst': False,
        'catalyst_date': None,
        'catalyst_type': None,
    }
    for ticker in ['ARGX', 'GILD', 'VRTX', ...]  # Your universe
]

with open('production_data/catalyst_records.json', 'w') as f:
    json.dump(catalyst_records, f, indent=2)
```

**B. Derive catalysts from trials:**
```python
# Use trial readout dates as catalysts
# Extract from trial_records.json where status is "RECRUITING" or "ACTIVE"
```

---

## 📊 **4. Production Invariant Block (Critical!)**

### **Add to end of `run_screen.py`:**

```python
def validate_screening_output(result: Dict, as_of_date: str, config: Dict) -> None:
    """
    Production invariants - fail loud if something is wrong.
    
    This catches silent failures before they reach production.
    """
    print("\n" + "="*80)
    print("PRODUCTION VALIDATION")
    print("="*80)
    
    m5 = result.get('module_5_composite', {})
    ranked = m5.get('ranked_securities', [])
    
    # Invariant 1: Weight sum
    total_weight = sum(Decimal(s['position_weight']) for s in ranked)
    expected = Decimal('1.0') - config.get('cash_target', Decimal('0.10'))
    diff = abs(total_weight - expected)
    
    status = "✅" if diff < Decimal('0.001') else "❌"
    print(f"\n{status} Weight sum: {total_weight:.4f} (expected {expected:.4f}, diff {diff:.4f})")
    
    # Invariant 2: Excluded weights
    excluded_weight = sum(Decimal(s['position_weight']) for s in ranked if not s.get('rankable', True))
    status = "✅" if excluded_weight == 0 else "❌"
    print(f"{status} Excluded weight sum: {excluded_weight:.4f} (expected 0.0000)")
    
    # Invariant 3: Top-N if enabled
    top_n = config.get('top_n')
    if top_n:
        invested = sum(1 for s in ranked if Decimal(s['position_weight']) > 0)
        status = "✅" if invested == top_n else "❌"
        print(f"{status} Top-N count: {invested} (expected {top_n})")
    
    # Invariant 4: Module coverage
    m2_diag = result.get('module_2_financial', {}).get('diagnostic_counts', {})
    m3_diag = result.get('module_3_catalyst', {}).get('diagnostic_counts', {})
    m4_diag = result.get('module_4_clinical_dev', {}).get('diagnostic_counts', {})
    
    print(f"\n📊 Module Coverage:")
    print(f"  Module 2 (Financial): {m2_diag.get('scored', 0)}/{len(ranked)}")
    print(f"  Module 3 (Catalyst):  {m3_diag.get('with_catalyst', 0)}/{len(ranked)}")
    print(f"  Module 4 (Clinical):  {m4_diag.get('scored', 0)}/{len(ranked)}")
    
    # Invariant 5: PIT filtering sanity
    pit_filtered = m4_diag.get('pit_filtered', 0)
    trials_evaluated = m4_diag.get('total_trials', 0)
    
    if trials_evaluated > 100 and pit_filtered == 0:
        print(f"\n⚠️  WARNING: {trials_evaluated} trials but PIT filtered 0")
        print(f"   This suggests trials missing date fields")
    
    # Invariant 6: Data freshness
    print(f"\n📅 Data Dates:")
    print(f"  Analysis date: {as_of_date}")
    print(f"  Universe as_of: {result.get('module_1_universe', {}).get('as_of_date')}")
    print(f"  All modules match: {all(m.get('as_of_date') == as_of_date for m in [result.get(f'module_{i}_{n}') for i, n in [(1,'universe'), (2,'financial'), (3,'catalyst'), (4,'clinical_dev'), (5,'composite')]])}")
    
    print("\n" + "="*80)


# Call at end of run_screening_pipeline():
validate_screening_output(output, as_of_date, {
    'cash_target': Decimal('0.10'),
    'top_n': 60,  # Set if enabled
})
```

### **Expected Output:**
```
================================================================================
PRODUCTION VALIDATION
================================================================================

✅ Weight sum: 0.9000 (expected 0.9000, diff 0.0000)
✅ Excluded weight sum: 0.0000 (expected 0.0000)
✅ Top-N count: 60 (expected 60)

📊 Module Coverage:
  Module 2 (Financial): 95/98
  Module 3 (Catalyst):  48/98
  Module 4 (Clinical):  98/98

⚠️  WARNING: 464 trials but PIT filtered 0
   This suggests trials missing date fields

📅 Data Dates:
  Analysis date: 2026-01-06
  Universe as_of: 2026-01-06
  All modules match: True

================================================================================
```

---

## 🎯 **Priority Order (ROI-Ranked):**

### **1. Enable Top-N (5 min, high impact)**
- One line change
- Immediate improvement in conviction weights
- Ready to use right now

### **2. Fix Module 2 (30 min, highest ROI)**
- Financial data is core driver
- Will stabilize rankings significantly
- Same pattern as Module 4 fix

### **3. Add Production Validation (20 min)**
- Catches future regressions
- One-time investment, permanent benefit
- Prevents silent failures

### **4. Fix PIT Filtering (60 min)**
- Important for historical accuracy
- Less urgent if only screening current data
- Can phase in later

### **5. Fix Module 3 (60 min)**
- Catalysts are valuable but optional
- System works without it
- Lower priority

---

## 📋 **Files I'll Create For You:**

1. **production_validation.py** - Invariant checking module
2. **fix_module2_guide.md** - Step-by-step Module 2 fix
3. **enable_topn.md** - Quick top-N activation guide
4. **pit_diagnostic.py** - PIT date field checker

**Want me to create these?**
