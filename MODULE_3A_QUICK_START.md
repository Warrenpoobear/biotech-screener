# Module 3A Quick Start Guide
## Get Catalyst Detection Running in 15 Minutes

---

## ✅ **Prerequisites Check**

Before starting, verify you have:

1. ✅ **trial_records.json with dates** (you just backfilled this!)
   ```powershell
   python -c "import json; r = json.load(open('production_data/trial_records.json')); print('Has dates:', r[0].get('last_update_posted') is not None)"
   ```
   Expected: `Has dates: True`

2. ✅ **All Module 3A files downloaded:**
   - `ctgov_adapter.py`
   - `state_management.py`
   - `event_detector.py`
   - `catalyst_summary.py`
   - `module_3_catalyst.py`

3. ✅ **Python packages:**
   ```powershell
   # Already installed: requests, json, pathlib, datetime
   ```

---

## 🚀 **15-Minute Quick Start**

### **Step 1: Test Module 3 Standalone (5 minutes)**

Run Module 3 by itself first to verify it works:

```powershell
python module_3_catalyst.py --as-of-date 2026-01-06 --trial-records production_data/trial_records.json --state-dir production_data/ctgov_state --universe production_data/universe.json --output-dir production_data
```

**Expected Output:**
```
Starting Module 3 catalyst detection for 2026-01-06
Active tickers: 98
Processing 464 trials for active tickers
Converting to canonical format...
Converted 464 records successfully
No prior snapshot found - this is initial run
Detecting catalyst events...
Detected 0 events across 0 tickers
Aggregating events into ticker summaries...
Saving current snapshot...
Module 3 completed in 12.5 seconds

================================================================================
MODULE 3 CATALYST DETECTION COMPLETE
================================================================================
Events detected: 0
Tickers with events: 0/98
Severe negatives: 0
================================================================================
```

**Why 0 events?** This is expected on first run - no prior state to compare!

**Verify outputs created:**
```powershell
ls production_data/ctgov_state/
ls production_data/catalyst_events_*.json
```

You should see:
- `production_data/ctgov_state/state_2026-01-06.jsonl` (initial snapshot)
- `production_data/catalyst_events_2026-01-06.json` (empty events)

✅ **Success!** Module 3 standalone works!

---

### **Step 2: Integrate with run_screen.py (5 minutes)**

Add Module 3 to your screening pipeline:

**Option A: Manual Edit**

Open `run_screen.py` and add this after Module 2:

```python
# Module 3: Catalyst Detection
print("\n[3/7] Module 3: Catalyst detection...")

from module_3_catalyst import compute_module_3_catalyst, Module3Config
from event_detector import SimpleMarketCalendar

market_calendar = SimpleMarketCalendar()
m3_config = Module3Config()

m3_result = compute_module_3_catalyst(
    trial_records_path=data_dir / "trial_records.json",
    state_dir=data_dir / "ctgov_state",
    active_tickers=set(active_tickers),
    as_of_date=as_of_date,
    market_calendar=market_calendar,
    config=m3_config,
    output_dir=data_dir
)

catalyst_summaries = m3_result['summaries']
diag = m3_result.get('diagnostic_counts', {})
print(f"  Events detected: {diag.get('events_detected', 0)}, "
      f"Tickers with events: {diag.get('tickers_with_events', 0)}/{diag.get('tickers_analyzed', 0)}, "
      f"Severe negatives: {diag.get('severe_negatives', 0)}")
```

**Option B: Automatic Patch**

*(I can create a patch script if you want)*

---

### **Step 3: Run Full Screening (2 minutes)**

```powershell
python run_screen.py --as-of-date 2026-01-06 --data-dir production_data --output screening_with_catalyst.json
```

**Expected Output:**
```
[1/7] Module 1: Universe filtering...
  Active: 98, Excluded: 0

[2/7] Module 2: Financial health...
  Scored: 98, Missing: 0

[3/7] Module 3: Catalyst detection...
  Events detected: 0, Tickers with events: 0/98, Severe negatives: 0

[4/7] Module 4: Clinical development...
  Scored: 98, Trials evaluated: 464

[5/7] Module 5: Composite ranking...
  Rankable: 98, Excluded: 0

[COMPLETE] Pipeline finished successfully
```

✅ **Success!** Full pipeline with Module 3 works!

---

### **Step 4: Simulate Event Detection (3 minutes)**

To see events detected, run again 1 week later:

```powershell
# Simulate running next week
python run_screen.py --as-of-date 2026-01-13 --data-dir production_data --output screening_week2.json
```

**Expected Output:**
```
[3/7] Module 3: Catalyst detection...
  Events detected: 8, Tickers with events: 5/98, Severe negatives: 1
```

Now you'll see real events! (If any trials changed status in CT.gov)

---

## 📊 **Understanding the Output**

### **Catalyst Events File**

`production_data/catalyst_events_2026-01-06.json`:

```json
{
  "run_metadata": {
    "as_of_date": "2026-01-06",
    "tickers_analyzed": 98,
    "events_detected": 0,
    "severe_negatives": 0
  },
  "summaries": [
    {
      "ticker": "VRTX",
      "catalyst_score_pos": 0.0,
      "catalyst_score_neg": 0.0,
      "catalyst_score_net": 0.0,
      "severe_negative_flag": false,
      "events": []
    }
  ]
}
```

### **State Snapshot**

`production_data/ctgov_state/state_2026-01-06.jsonl`:

```jsonl
{"ticker":"ACAD","nct_id":"NCT001","overall_status":"RECRUITING","last_update_posted":"2026-01-05",...}
{"ticker":"ARWR","nct_id":"NCT002","overall_status":"COMPLETED","last_update_posted":"2026-01-04",...}
```

---

## 🎯 **Next Steps**

### **Weekly Production Use**

Run screening every Monday:

```powershell
# Get current date
$date = Get-Date -Format "yyyy-MM-dd"

# Run screening
python run_screen.py --as-of-date $date --data-dir production_data --output "screening_$date.json"
```

Module 3 will:
1. Load prior snapshot
2. Compare with current trial data
3. Detect changes (status, dates, results)
4. Score catalyst events
5. Save new snapshot

### **Review Catalyst Events**

Check for important events:

```powershell
# View severe negatives
python -c "import json; data = json.load(open('production_data/catalyst_events_2026-01-06.json')); print('Severe negatives:', [s['ticker'] for s in data['summaries'] if s['severe_negative_flag']])"
```

### **Integrate with Module 5**

See `MODULE_3A_INTEGRATION.md` for:
- Adding catalyst scores to composite ranking
- Implementing severe negative kill switch
- Accessing detailed event information

---

## ✅ **Verification Checklist**

After setup, verify:

- [ ] Module 3 runs standalone without errors
- [ ] State snapshot created in `production_data/ctgov_state/`
- [ ] Catalyst events file created
- [ ] run_screen.py includes Module 3
- [ ] Full pipeline runs successfully
- [ ] Module 3 shows in pipeline output

---

## 🚨 **Troubleshooting**

### **Issue: ModuleNotFoundError**

```
ModuleNotFoundError: No module named 'ctgov_adapter'
```

**Solution:** Ensure all Module 3A files are in your project directory:
```powershell
ls ctgov_adapter.py
ls state_management.py
ls event_detector.py
ls catalyst_summary.py
ls module_3_catalyst.py
```

### **Issue: "No prior snapshot found"**

```
No prior snapshot found - this is initial run
```

**Solution:** This is expected on first run! Events will be detected starting with second run.

### **Issue: Validation Gates Failed**

```
CRITICAL: Missing last_update_posted: 50
```

**Solution:** Re-run the backfill script:
```powershell
python backfill_ctgov_dates.py
```

---

## 📚 **Documentation Reference**

- **MODULE_3A_CONTRACT_SPEC.md** - Complete specification
- **MODULE_3A_IMPLEMENTATION_GUIDE.md** - 2-week plan
- **MODULE_3A_INTEGRATION.md** - run_screen.py integration
- **This guide** - Quick start

---

## 🎉 **Success Criteria**

You're done when:

✅ Module 3 runs standalone  
✅ State snapshot created  
✅ Integrated with run_screen.py  
✅ Full pipeline runs successfully  
✅ Ready for weekly production use  

---

## 📞 **Need Help?**

Common questions:

**Q: Why 0 events on first run?**  
A: No prior state to compare. Events detected starting run #2.

**Q: How often should I run?**  
A: Weekly recommended. More frequent = more sensitive to changes.

**Q: What if a trial is terminated?**  
A: Module 3 detects it as CT_STATUS_SEVERE_NEG and sets severe_negative_flag=True.

**Q: Can I backtest?**  
A: Yes! Run for historical dates to build state history. Ensure PIT discipline.

---

**You're ready to detect catalyst events!** 🚀
