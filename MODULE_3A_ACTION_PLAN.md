# Module 3A: Action Plan for Your Data
## Getting Catalyst Detection Working

---

## 🔍 **Current Situation:**

Your `trial_records.json` is **missing critical date fields** needed for Module 3A:

```json
{
  "ticker": "INCY",
  "nct_id": "NCT04205812",
  "status": "Active_Not_Recruiting",  // ✅ Good
  "last_update_posted": null,         // ❌ CRITICAL - Need this!
  "source_date": null                 // ❌ No dates at all
}
```

**Without dates, Module 3A cannot:**
- Timestamp when events occurred (PIT compliance)
- Detect changes with accurate timing
- Prevent lookahead bias

---

## ✅ **Solution: Backfill Dates from CT.gov**

I've created a script to automatically add the missing date fields.

### **Step 1: Install requests (if needed)**

```powershell
pip install requests --break-system-packages
```

### **Step 2: Run the backfill script**

```powershell
python backfill_ctgov_dates.py
```

**This will:**
1. Read your existing `production_data/trial_records.json`
2. Query CT.gov API for each trial (with rate limiting)
3. Add date fields:
   - `last_update_posted` (CRITICAL - PIT anchor)
   - `primary_completion_date` + `primary_completion_type`
   - `completion_date` + `completion_type`
   - `results_first_posted`
4. Save to `production_data/trial_records_with_dates.json`

**Expected time:** ~8 minutes for 464 trials (1 second per trial)

### **Step 3: Verify the output**

```powershell
# Check first record has dates
python -c "import json; records = json.load(open('production_data/trial_records_with_dates.json')); print(json.dumps(records[0], indent=2))"
```

**You should see:**
```json
{
  "ticker": "INCY",
  "nct_id": "NCT04205812",
  "status": "Active_Not_Recruiting",
  "last_update_posted": "2024-01-15",  // ✅ Now present!
  "primary_completion_date": "2024-06-01",
  "primary_completion_type": "ANTICIPATED"
}
```

### **Step 4: Replace original file**

```powershell
# Backup original
cp production_data/trial_records.json production_data/trial_records_backup.json

# Use enhanced version
cp production_data/trial_records_with_dates.json production_data/trial_records.json
```

---

## 📊 **Expected Results:**

After backfilling, you should have:

```
Date coverage: 95-100%
  ✅ last_update_posted: 460+ / 464 trials
  ✅ primary_completion_date: 400+ / 464 trials
  ✅ completion_date: 400+ / 464 trials
```

---

## 🚀 **Then: Module 3A Implementation**

Once dates are backfilled, I can immediately provide:

1. **Event Detector** - Detects status changes, timeline shifts
2. **Module 3 Orchestrator** - Main entry point
3. **run_screen.py Integration** - Plug into your pipeline
4. **Test Suite** - 30+ tests

**Timeline:** 2-3 hours after dates are confirmed

---

## 🔄 **Going Forward: Update Your Collector**

For future data collection, update your CT.gov collector to extract these fields:

```python
# Add to your collector
def extract_trial_data(ctgov_record):
    status_module = ctgov_record["protocolSection"]["statusModule"]
    
    return {
        "ticker": ticker,
        "nct_id": nct_id,
        "status": status_module["overallStatus"],
        
        # ADD THESE:
        "last_update_posted": status_module["lastUpdatePostDateStruct"]["date"],
        "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
        "primary_completion_type": status_module.get("primaryCompletionDateStruct", {}).get("type"),
        "completion_date": status_module.get("completionDateStruct", {}).get("date"),
        "completion_type": status_module.get("completionDateStruct", {}).get("type"),
        "results_first_posted": ctgov_record.get("resultsSection", {}).get("resultsFirstPostDateStruct", {}).get("date")
    }
```

---

## ⚠️ **Alternative: Store Full CT.gov Records**

Instead of flattening, you could store the full API response:

```json
{
  "ticker": "INCY",
  "nct_id": "NCT04205812",
  "ctgov_record": { /* full CT.gov v2 API response */ }
}
```

The adapter can extract what it needs. This is more future-proof.

---

## 📋 **Summary: Your Action Items**

### **Immediate (30 minutes):**
1. ✅ Download `backfill_ctgov_dates.py` (provided above)
2. ✅ Run: `python backfill_ctgov_dates.py`
3. ✅ Verify dates were added
4. ✅ Replace trial_records.json

### **Once Dates Confirmed:**
- Share one record from enhanced file
- I'll complete Module 3A implementation (2-3 hours)

### **Long-term (next data collection):**
- Update collector to extract date fields directly
- Consider storing full CT.gov records

---

## 💡 **Why This Matters:**

**With dates:**
- ✅ PIT-compliant catalyst detection
- ✅ Accurate event timing
- ✅ Market calendar integration
- ✅ Backtest-ready (no lookahead)

**Without dates:**
- ❌ Module 3A cannot work
- ❌ No way to timestamp events
- ❌ No delta detection possible

---

## 🚀 **Next Steps:**

**Run this now:**
```powershell
python backfill_ctgov_dates.py
```

**Then verify:**
```powershell
python -c "import json; records = json.load(open('production_data/trial_records_with_dates.json')); print(f\"Coverage: {sum(1 for r in records if r.get('last_update_posted'))}/{len(records)} have last_update_posted\")"
```

**Expected:** `Coverage: 450+/464 have last_update_posted`

Once confirmed, I'll complete the Module 3A implementation! 🎉
