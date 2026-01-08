# Module 2 Enhancement Summary
**Date:** 2026-01-08  
**Version:** 2.0 (Enhanced)

---

## What Changed

### New Features Added

1. **Burn Acceleration Detection**
   - Compares recent quarterly burn to 4Q trailing average
   - Amplifies dilution penalty when burn is accelerating (≥1.3x)
   - Caps at 3x to prevent numerical instability

2. **Catalyst Timing Integration**
   - Soft dependency on Module 3 catalyst summaries
   - Calculates "coverage ratio" = runway / time-to-catalyst
   - Applies additional penalty when coverage < 1.0 (need money before catalyst)
   - Gracefully degrades when catalyst data unavailable

3. **Recent Financing Dampener**
   - Reduces dilution penalty if capital raised in last 90 days
   - Maintains floor penalty for critically short runways (<9 months)
   - Only activates when `days_since_last_raise` data available

4. **Enhanced Diagnostic Output**
   - New fields: `burn_recent_m`, `burn_4q_avg_m`, `burn_acceleration`
   - Catalyst timing: `ttc_months`, `coverage`, `catalyst_timing_flag`
   - Financing context: `days_since_raise`
   - Data quality: `cfo_quality_flag`

---

## Key Functions Added

### `quarterly_cfo_from_ytd(fiscal_period, cfo_ytd_current, ...)`
- **Purpose:** Convert SEC 10-Q YTD CFO to true quarterly CFO
- **Logic:** 
  - Q1: YTD is already quarterly
  - Q2/Q3: Current YTD - Prior YTD
  - Q4: Annual FY - Q3 YTD
- **Output:** Quarterly CFO value (or None if insufficient data)
- **Handles:** Non-calendar fiscal years via fiscal_period metadata

### `calculate_burn_acceleration(cfo_recent_q, cfo_last_4q)`
- **Purpose:** Detect if burn rate is accelerating
- **Logic:** Recent monthly burn / 4Q average monthly burn
- **Output:** (burn_recent_m, burn_4q_avg_m, burn_acceleration)
- **Safeguards:** EPS floor to prevent division by zero, 3x cap

### `calculate_catalyst_coverage(runway_months, next_catalyst_date, as_of_date)`
- **Purpose:** Calculate if cash lasts until next catalyst
- **Logic:** runway_months / months_to_catalyst
- **Output:** (ttc_months, coverage, catalyst_timing_flag)
- **Flags:** KNOWN / UNKNOWN / FAR (>24 months out)

### `apply_dilution_penalty_enhanced(...)`
- **Purpose:** Apply all dilution modifiers in one place
- **Inputs:** Base score, runway, burn acceleration, coverage, days since raise
- **Logic:**
  - Base runway penalty (existing)
  - × 1.5x if burn_acceleration ≥ 1.3
  - × 1.75x if burn_acceleration ≥ 1.6
  - × 0.7x if coverage < 1.0
  - × 1.5x dampener if recent raise (with floor for critical runway)

### `score_financial_health_enhanced(...)`
- **Purpose:** Main scoring function with new features
- **New params:** catalyst_summary, as_of_date
- **Behavior:** Checks for optional data fields, computes enhancements only when available

---

## Data Requirements

### Required Fields (Existing - No Change)
```python
financial_data = {
    'ticker': str,
    'Cash': float,
    'NetIncome': float,
    'R&D': float
}

market_data = {
    'ticker': str,
    'market_cap': float,
    'price': float,
    'avg_volume': float
}
```

### Optional Enhancement Fields
```python
financial_data = {
    # ... existing fields ...
    
    # OPTION 1: Direct quarterly CFO (preferred if you have it)
    'CFO_recent_q': float,              # Most recent quarterly CFO
    'CFO_last_4q': List[float],         # Last 4 quarters of CFO
    
    # OPTION 2: YTD CFO (handles 10-Q reporting format)
    'fiscal_period': str,               # "Q1", "Q2", "Q3", "FY"/"Q4"
    'CFO_ytd_current': float,           # Current YTD CFO
    'CFO_ytd_prev': float,              # Prior YTD CFO (for Q2/Q3)
    'CFO_fy_annual': float,             # Full year CFO (for Q4 calculation)
    'CFO_ytd_q3': float,                # Q3 YTD CFO (for Q4 calculation)
    
    # Other optional fields
    'days_since_last_raise': int,       # Days since equity raise
}

# From Module 3 (passed as separate dict)
catalyst_summary = {
    'next_major_catalyst_date': str,    # Optional: ISO date
    # ... other Module 3 fields ...
}
```

### SEC 10-Q YTD CFO Handling

**Critical Enhancement:** Module 2 now deterministically converts YTD CFO to quarterly CFO.

**The Problem:**
- SEC 10-Qs report CFO as year-to-date (cumulative)
- Q2 CFO = 6 months cumulative, NOT just Q2
- Naively using YTD values inflates burn calculations by 2-3x

**The Solution:**
```python
def quarterly_cfo_from_ytd(fiscal_period, cfo_ytd_current, ...):
    # Q1: Already quarterly (YTD = Q1)
    # Q2: YTD(Q2) - YTD(Q1) 
    # Q3: YTD(Q3) - YTD(Q2)
    # Q4: FY(annual) - YTD(Q3)
```

**Supports Non-Calendar Fiscal Years:**
- Keys off `fiscal_period` metadata, not filing dates
- Handles companies with June 30, September 30, etc. fiscal year-ends

**Graceful Degradation:**
- If you provide `CFO_recent_q` + `CFO_last_4q`: Uses them directly (preferred)
- If you only provide YTD fields: Converts to quarterly automatically
- If neither: `cfo_quality_flag = "MISSING"`, burn acceleration stays neutral

### Data Pipeline Notes

**CFO History - Now Handles YTD Automatically!**

You have two options:

**Option 1 (Preferred):** Pre-compute quarterly CFO
- Extract CFO from 10-Q/10-K cash flow statements
- Convert YTD → quarterly yourself
- Maintain last 4-5 quarters of history per ticker
- Pass as `CFO_recent_q` and `CFO_last_4q`

**Option 2 (Easier):** Pass YTD CFO + fiscal_period
- Extract raw YTD CFO values from 10-Q
- Include `fiscal_period` metadata ("Q1", "Q2", "Q3", "FY")
- Module 2 converts to quarterly automatically
- Pass as `CFO_ytd_current`, `CFO_ytd_prev`, etc.

**Fiscal Year Boundaries:**
- Both options respect non-calendar fiscal years
- Option 1: Track by (fiscal_year, fiscal_period) tuples
- Option 2: fiscal_period metadata handles it automatically

**Key Advantage of Option 2:**
- No pre-processing pipeline required
- Just extract raw 10-Q values + metadata
- Module handles the YTD→quarterly math deterministically

**Catalyst Timing:** Automatically available when Module 3 runs
- Soft dependency: Module 2 works fine without it
- When available, coverage ratio provides additional signal

**Financing Events:** Optional future enhancement
- Would require parsing 8-K filings or S-3/424B documents
- Gated behind `days_since_last_raise` field existence

---

## Integration Steps

### Step 1: Update Module 2 File
Replace `module_2_financial.py` with the enhanced version:
```bash
cp module_2_financial_ENHANCED.py production_code/module_2_financial.py
```

### Step 2: Update Orchestrator
In `run_screen.py`, modify Module 2 call to pass catalyst summaries:

```python
# Run Module 3 first
module_3_result = compute_module_3_catalyst(...)
catalyst_summaries = module_3_result['summaries']

# Then run Module 2 with catalyst data
module_2_result = compute_module_2_financial(
    active_tickers=active_tickers,
    financial_data=financial_data,
    market_data=market_data,
    catalyst_summaries=catalyst_summaries,  # NEW
    as_of_date=as_of_date                    # NEW
)
```

### Step 3: Implement CFO History Extractor (Optional but Recommended)
Create `extract_cfo_history.py` to process 10-Q/10-K files:
- Parse cash flow statements
- Convert YTD to quarterly
- Store last 4-5 quarters per ticker
- Add fields to `financial_data.json`

### Step 4: Add Financing Event Tracker (Future)
When ready, implement:
- 8-K parser for capital raises
- Calculate `days_since_last_raise`
- Add to `financial_data.json`

---

## Backward Compatibility

**100% backward compatible!**

The enhanced module works with existing data:
- If CFO history missing → `burn_acceleration = 1.0` (no penalty amplification)
- If catalyst data missing → `catalyst_timing_flag = "UNKNOWN"` (no coverage penalty)
- If raise data missing → No dampening applied

All original functionality preserved.

---

## Testing

### Unit Test Example
```python
# Test burn acceleration
cfo_recent = -110e6  # $110M quarterly burn
cfo_history = [-90e6, -95e6, -100e6, -110e6]  # Accelerating

burn_recent_m, burn_4q_avg_m, accel = calculate_burn_acceleration(
    cfo_recent, cfo_history
)

assert accel > 1.15  # Should detect acceleration
```

### Integration Test
Run `python module_2_financial_ENHANCED.py` to see demo with:
- CVAC: Accelerating burn, near-term catalyst
- RYTM: Stable burn, recent raise

---

## Expected Impact

### Catches "Silent Killers"
- **Burn acceleration:** Company with "adequate" 15-month runway but accelerating burn → flagged
- **Catalyst timing:** Company with 14 months cash but catalyst 18 months out → penalized
- **Recent raise:** Company just raised $100M → temporarily dampened penalty

### Score Changes (Estimated)
- **Accelerating burn (1.5x):** Dilution score drops ~10-15 points
- **Coverage < 1.0:** Additional ~10 point drop
- **Recent raise:** +5-10 point boost (with floor)

### Real-World Example
**Before Enhancement:**
- Runway: 14 months
- Cash/Mcap: 20%
- Dilution Score: 70

**After Enhancement:**
- Burn acceleration: 1.4x (accelerating trial spend)
- Catalyst: 18 months out (coverage = 0.78)
- Dilution Score: 70 → 70 × 0.67 (accel) × 0.7 (coverage) = **33**

This properly flags the dilution trap!

---

## Next Steps

1. ✅ Review enhanced code  
2. ⬜ Extract YTD CFO + fiscal_period from 10-Q/10-K (Module 2 handles conversion!)
3. ⬜ Test on historical data
4. ⬜ Deploy to production
5. ⬜ Add financing event tracker (future)

**Note:** Step 2 is now much simpler - just extract raw YTD CFO values and metadata, no pre-processing needed!

---

## Questions?

**Q: Do I need CFO history immediately?**  
A: No. Module works without it, you just don't get burn acceleration signal.

**Q: Will this break existing scores?**  
A: Only improves them. Without new data fields, scores identical to v1.0.

**Q: How do I test it?**  
A: Run `python module_2_financial_ENHANCED.py` - includes demo with sample data.

**Q: Do I need to convert YTD to quarterly CFO myself?**  
A: No! Just pass YTD values + fiscal_period and Module 2 handles the conversion deterministically. This works for all fiscal year types (calendar and non-calendar).

---

**END OF SUMMARY**
