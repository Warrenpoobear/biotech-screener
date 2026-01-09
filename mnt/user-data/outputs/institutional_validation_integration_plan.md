# Institutional Validation Integration Plan
## Wake Robin Biotech Screening System v1.5

**Date:** 2026-01-09
**Status:** Design Complete, Ready for Implementation

---

## 1. Architecture Overview

### Current System (5 Modules)
```
Module 1: Universe Filtering
Module 2: Financial Health  
Module 3: Catalyst Detection
Module 4: Clinical Development
Module 5: Composite Ranking
```

### Enhanced System (5 + Validation Layer)
```
Module 1: Universe Filtering
Module 2: Financial Health
Module 3: Catalyst Detection  
Module 4: Clinical Development
Module 5: Composite Ranking
    ↓
[Institutional Validation Layer] ← Confirmation, NOT scoring
    ↓
Output: Rankings + Institutional Columns
```

**Critical Design Decision:**
- Institutional validation is **NOT** Module 6
- It's a **post-composite validation layer**
- Runs AFTER Module 5 completes
- Adds columns to output, does NOT modify composite score

---

## 2. Data Requirements

### Input: SEC 13F Filings

**Sources:**
- SEC EDGAR (primary): https://www.sec.gov/cgi-bin/browse-edgar
- 13F-HR filings from elite biotech hedge funds
- Quarterly filing cadence (45 days after quarter end)

**Required Data Structure:**
```python
# holdings_snapshots.json
{
  "ticker": "ARGX",
  "as_of_date": "2024-11-15",
  "current_quarter": "2024-09-30",
  "prior_quarter": "2024-06-30",
  "holdings": {
    "current": {
      "0001263508": {  # Baker Bros CIK
        "quarter_end": "2024-09-30",
        "state": "KNOWN",
        "shares": 1234567,
        "value_kusd": 85000,
        "put_call": ""
      },
      # ... other managers
    },
    "prior": {
      # ... prior quarter snapshots
    }
  },
  "filings_metadata": {
    "0001263508": {
      "quarter_end": "2024-09-30",
      "accession": "0001263508-24-000003",
      "total_value_kusd": 13838782,
      "filed_at": "2024-11-14T16:00:00",
      "is_amendment": false
    }
  }
}
```

### Manager Registry

**Elite Core (Priority 1):**
- Baker Bros Advisors: `0001263508`
- RA Capital Management: `0001346824`  
- Perceptive Advisors: `0001224962`
- Deerfield Management: `0001102854`
- Ally Bridge Group: `0001555280`
- Redmile Group: `0001496147`
- Orbimed Advisors: `0001107208`
- Venbio Partners: `0001655352`
- Bain Capital Life Sciences: `0001067983`
- RTW Investments: `0001713407`
- Tang Capital: `0001556308`
- Farallon Capital: `0001105977`

**Conditional Tier (for breadth only):**
- Venrock: `0001005477`
- Cormorant: `0001398659`
- Deep Track Capital: `0001631282`
- Viking Global: `0001103804`

---

## 3. Output Schema Changes

### Current CSV Output
```
ticker,rank,final_score,m1_universe,m2_financial,m3_catalyst,m4_clinical,market_cap,sector
```

### Enhanced CSV Output  
```
ticker,rank,final_score,m1_universe,m2_financial,m3_catalyst,m4_clinical,market_cap,sector,inst_score,inst_state,elite_holders,elite_new,elite_adds,elite_exits,crowding_flag
```

**New Columns:**
- `inst_score` (float): 0.00-1.00 validation score
- `inst_state` (str): "KNOWN", "INCOMPLETE", "STALE"  
- `elite_holders` (int): Number of elite funds holding
- `elite_new` (int): New initiations this quarter
- `elite_adds` (int): Material adds (>25% increase)
- `elite_exits` (int): Complete exits
- `crowding_flag` (bool): TRUE if crowding risk detected

### Top 60 Report Enhancement

**Current Format:**
```
   1. NVAX   (84.87  ) - early stage
   2. ALKS   (84.32  ) - late  stage
```

**Enhanced Format:**
```
   1. NVAX   (84.87  ) - early stage  │ INST: 0.72 [4H, 2N, 1A] ⚠️
   2. ALKS   (84.32  ) - late  stage  │ INST: 0.45 [2H, 0N, 0A]
   3. BBOT   (84.30  ) - early stage  │ INST: --- [NO DATA]
```

**Legend:**
- `0.72` = Institutional validation score
- `4H` = 4 elite holders
- `2N` = 2 new initiations  
- `1A` = 1 material add
- `⚠️` = Crowding flag

---

## 4. Implementation Steps

### Phase 1: Data Pipeline (Week 1)

**Step 1.1: Create 13F Data Extractor**
```python
# edgar_13f_extractor.py
# - Query SEC EDGAR for elite manager CIKs
# - Download 13F-HR XML filings
# - Parse holdings by CUSIP
# - Map CUSIP → Ticker (OpenFIGI API)
# - Store in holdings_snapshots.json
```

**Step 1.2: Build Manager Registry**
```python
# manager_registry.json
{
  "elite_core": [
    {"cik": "0001263508", "name": "Baker Bros Advisors", "aum_b": 13.8},
    {"cik": "0001346824", "name": "RA Capital Management", "aum_b": 8.1},
    # ...
  ],
  "conditional": [
    {"cik": "0001005477", "name": "Venrock", "aum_b": 2.5},
    # ...
  ]
}
```

**Step 1.3: Implement CUSIP Mapper**
```python
# cusip_mapper.py (reuse from existing SEC work)
# - Cache CUSIP → Ticker mappings
# - Handle splits/mergers
# - Validate against universe.json
```

### Phase 2: Module Integration (Week 2)

**Step 2.1: Create Validation Wrapper**
```python
# module_validation_institutional.py

def compute_institutional_validation(
    ticker: str,
    as_of_date: date,
    data_dir: Path
) -> Optional[InstitutionalValidationOutput]:
    """
    Wrapper to integrate institutional_validation_v1_2.py
    into Wake Robin pipeline.
    
    Returns None if insufficient data (ticker not tracked by elites).
    """
    # Load holdings snapshots
    holdings_path = data_dir / 'holdings_snapshots.json'
    if not holdings_path.exists():
        return None
    
    with open(holdings_path) as f:
        data = json.load(f)
    
    ticker_data = data.get(ticker)
    if not ticker_data:
        return None  # Ticker not tracked
    
    # Load manager registry
    registry = load_manager_registry(data_dir / 'manager_registry.json')
    
    # Convert to module's data structures
    current_holdings = build_snapshots(ticker_data['holdings']['current'])
    prior_holdings = build_snapshots(ticker_data['holdings']['prior'])
    filings_metadata = build_filings_metadata(ticker_data['filings_metadata'])
    
    # Call validation module
    from institutional_validation_v1_2 import validate_institutional_activity
    
    result = validate_institutional_activity(
        ticker=ticker,
        as_of_date=as_of_date,
        current_holdings=current_holdings,
        prior_holdings=prior_holdings,
        filings_metadata=filings_metadata,
        manager_tiers=registry['tiers'],
        elite_ciks=registry['elite_core'],
        market_cap_usd=ticker_data['market_cap_usd']
    )
    
    return result
```

**Step 2.2: Modify run_screen.py**
```python
# run_screen.py (after Module 5 completes)

# Existing: Modules 1-5 complete, composite scores computed
summaries = []
for ticker in ranked_tickers:
    # ... existing module scores ...
    
    # NEW: Add institutional validation
    inst_result = compute_institutional_validation(
        ticker=ticker,
        as_of_date=as_of_date,
        data_dir=production_data_dir
    )
    
    if inst_result:
        inst_dict = {
            'inst_score': inst_result.inst_validation_score,
            'inst_state': inst_result.inst_state.value,
            'elite_holders': inst_result.elite_holders_n,
            'elite_new': inst_result.elite_new_n,
            'elite_adds': inst_result.elite_material_add_n,
            'elite_exits': inst_result.elite_exit_n,
            'crowding_flag': inst_result.crowding_flag
        }
    else:
        # No institutional data for this ticker
        inst_dict = {
            'inst_score': None,
            'inst_state': 'NO_DATA',
            'elite_holders': 0,
            'elite_new': 0,
            'elite_adds': 0,
            'elite_exits': 0,
            'crowding_flag': False
        }
    
    summaries.append({
        **existing_summary,
        **inst_dict
    })
```

**Step 2.3: Update Report Generators**
```python
# generate_all_reports.py

# Modify CSV exporter to include new columns
def export_to_csv(summaries, output_path):
    fieldnames = [
        'ticker', 'rank', 'final_score',
        'm1_universe', 'm2_financial', 'm3_catalyst', 'm4_clinical',
        'market_cap', 'sector',
        # NEW FIELDS
        'inst_score', 'inst_state', 'elite_holders',
        'elite_new', 'elite_adds', 'elite_exits', 'crowding_flag'
    ]
    # ... write CSV with new columns

# Modify Top 60 report formatter
def format_institutional_summary(inst_dict):
    """
    Format institutional data for Top 60 report.
    
    Returns: " │ INST: 0.72 [4H, 2N, 1A] ⚠️"
    """
    if inst_dict['inst_state'] == 'NO_DATA':
        return " │ INST: --- [NO DATA]"
    
    score = inst_dict['inst_score']
    holders = inst_dict['elite_holders']
    new = inst_dict['elite_new']
    adds = inst_dict['elite_adds']
    crowding = " ⚠️" if inst_dict['crowding_flag'] else ""
    
    return f" │ INST: {score:.2f} [{holders}H, {new}N, {adds}A]{crowding}"
```

### Phase 3: Alert System (Week 3)

**Step 3.1: Generate Institutional Alerts**
```python
# institutional_alerts.py

def generate_weekly_alerts(summaries, output_dir):
    """
    Scan for exceptional institutional activity.
    Generate alert report for IC review.
    """
    alerts = []
    
    for s in summaries:
        if s['inst_state'] == 'NO_DATA':
            continue
        
        # High-priority alerts
        if s['elite_new'] >= 2:
            alerts.append({
                'priority': 'HIGH',
                'ticker': s['ticker'],
                'type': 'CLUSTER_BUY',
                'message': f"{s['elite_new']} elite funds initiated new positions"
            })
        
        if s['elite_exits'] >= 2:
            alerts.append({
                'priority': 'HIGH',
                'ticker': s['ticker'],
                'type': 'MULTIPLE_EXITS',
                'message': f"{s['elite_exits']} elite funds exited - negative signal"
            })
        
        # Medium-priority alerts  
        if s['elite_holders'] >= 3 and s['elite_adds'] >= 2:
            alerts.append({
                'priority': 'MEDIUM',
                'ticker': s['ticker'],
                'type': 'SUSTAINED_ACCUMULATION',
                'message': f"{s['elite_adds']} funds adding to existing positions"
            })
        
        if s['crowding_flag']:
            alerts.append({
                'priority': 'MEDIUM',
                'ticker': s['ticker'],
                'type': 'CROWDING_RISK',
                'message': f"{s['elite_holders']} funds in small cap - monitor exits"
            })
    
    # Write alert report
    with open(output_dir / f'institutional_alerts_{timestamp}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("INSTITUTIONAL ACTIVITY ALERTS\n")
        f.write("="*80 + "\n\n")
        
        for priority in ['HIGH', 'MEDIUM']:
            priority_alerts = [a for a in alerts if a['priority'] == priority]
            if priority_alerts:
                f.write(f"\n{priority} PRIORITY:\n")
                f.write("-" * 80 + "\n")
                for alert in priority_alerts:
                    f.write(f"  [{alert['ticker']}] {alert['type']}\n")
                    f.write(f"  → {alert['message']}\n\n")
```

---

## 5. Data Refresh Strategy

### Quarterly Refresh (Within 45 Days of Quarter End)

**Timeline:**
- **Q-End + 0 days**: Quarter ends (3/31, 6/30, 9/30, 12/31)
- **Q-End + 45 days**: SEC filing deadline
- **Q-End + 50 days**: Run institutional update

**Refresh Script:**
```bash
# refresh_institutional_data.sh

#!/bin/bash
# Run this 50 days after each quarter end

QUARTER_END="2024-09-30"
AS_OF_DATE=$(date +%Y-%m-%d)

echo "Refreshing institutional data for Q ending $QUARTER_END"

# Step 1: Download 13F filings
python edgar_13f_extractor.py \
    --quarter-end $QUARTER_END \
    --manager-registry production_data/manager_registry.json \
    --output production_data/holdings_snapshots.json

# Step 2: Validate data quality
python validate_institutional_data.py \
    --holdings production_data/holdings_snapshots.json \
    --universe production_data/universe.json

# Step 3: Re-run screen with institutional validation
python run_screen.py \
    --as-of-date $AS_OF_DATE \
    --output results_with_inst_validation.json

# Step 4: Generate institutional reports
python generate_all_reports.py results_with_inst_validation.json

echo "Institutional data refresh complete!"
```

### Weekly Operations (No Institutional Update)

**Institutional data remains static between quarter ends:**
- Module 3 catalyst detection runs weekly (AACT updates)
- Module 4 clinical development runs weekly  
- Modules 1, 2, 5 run weekly with fresh market data
- **Institutional validation uses most recent quarterly snapshot**

**Note in weekly reports:**
```
INSTITUTIONAL DATA: Based on Q3 2024 filings (as of 2024-11-14)
Next refresh: Q4 2024 filings (expected 2025-02-14)
```

---

## 6. Coverage Expectations

### Elite Manager Coverage by Market Cap

**Expected Coverage Rates:**

| Market Cap     | Elite Holders | Coverage Expectation |
|----------------|---------------|----------------------|
| Mega (>$10B)   | 0-2 managers  | 20% of tickers       |
| Large ($2-10B) | 2-5 managers  | 40% of tickers       |
| Mid ($500M-2B) | 3-8 managers  | 60% of tickers       |
| Small (<$500M) | 1-4 managers  | 30% of tickers       |

**Overall System Coverage:**
- ~45% of 322 tickers will have institutional data
- ~55% will show "NO_DATA" (not tracked by elite managers)
- This is **expected and correct** - elites are selective

### Interpretation Guide

**Institutional Score Ranges:**

| Score Range | Interpretation                          | Action                    |
|-------------|-----------------------------------------|---------------------------|
| 0.80-1.00   | Strong elite validation                 | High conviction candidate |
| 0.60-0.79   | Moderate institutional interest         | Consider for portfolio    |
| 0.40-0.59   | Neutral signal                          | Fundamentals-driven only  |
| 0.20-0.39   | Weak institutional validation           | Caution flag              |
| 0.00-0.19   | Negative signal (exits/no interest)     | Red flag                  |
| NO_DATA     | Not tracked by elite managers           | Neutral (not negative)    |

**Critical Distinction:**
- `inst_score = 0.15` = **BAD** (tracked but elites are exiting)
- `inst_state = NO_DATA` = **NEUTRAL** (not tracked, use fundamentals)

---

## 7. Integration Testing Plan

### Test Case 1: Strong Validation Signal
```python
# ARGX example (Argenx)
# Expected: High institutional score, multiple elite holders

ticker = "ARGX"
expected_holders = 4-6
expected_score = 0.65-0.85
expected_state = "KNOWN"
```

### Test Case 2: Negative Signal (Multiple Exits)
```python
# Hypothetical: FAKE ticker with 2+ elite exits
# Expected: Low score, exit flags, alert triggered

ticker = "FAKE"
expected_exits = 2
expected_score = 0.10-0.25
expected_alert = "MULTIPLE_EXITS"
```

### Test Case 3: No Institutional Coverage
```python
# Small cap not tracked by elites
# Expected: NO_DATA, institutional columns null

ticker = "SMOL"
expected_state = "NO_DATA"
expected_score = None
expected_holders = 0
```

### Test Case 4: Crowding Risk
```python
# Mid-cap with 7+ elite holders
# Expected: Crowding flag, medium score

ticker = "CRWD"
expected_holders = 7
expected_crowding = True
expected_score = 0.50-0.65
```

---

## 8. File Structure

```
biotech-screener/
├── production_data/
│   ├── holdings_snapshots.json         # NEW: 13F holdings by ticker
│   ├── manager_registry.json           # NEW: Elite manager CIKs
│   ├── universe.json                   # Existing
│   ├── trial_records.json              # Existing
│   └── ...
├── institutional_validation_v1_2.py    # NEW: Core validation logic
├── module_validation_institutional.py  # NEW: Pipeline wrapper
├── edgar_13f_extractor.py             # NEW: Data extraction
├── cusip_mapper.py                    # NEW: CUSIP→Ticker mapping
├── institutional_alerts.py            # NEW: Alert generation
├── run_screen.py                      # MODIFY: Add inst validation
├── generate_all_reports.py            # MODIFY: Add inst columns
└── refresh_institutional_data.sh      # NEW: Quarterly refresh
```

---

## 9. Constitutional Amendments Required

**Amendment #X: Institutional Validation Layer**

**Enacted:** 2026-01-09

**Rationale:**
Elite institutional activity provides market-validated confirmation signals but should never override fundamental analysis. This amendment establishes institutional validation as a **confirmation layer** that enriches output without distorting composite scores.

**Provisions:**

1. **Scoring Isolation**
   - Institutional validation scores are **NOT** part of composite ranking
   - Module 5 final_score remains unchanged
   - Institutional data appears as **additional columns** only

2. **Data Refresh Cadence**
   - Quarterly updates (within 50 days of quarter end)
   - Static between quarters (13F filings are quarterly)
   - Weekly screen uses most recent quarterly snapshot

3. **Coverage Philosophy**
   - ~45% ticker coverage is expected and correct
   - NO_DATA is neutral, not negative
   - Elite selectivity is signal, not system failure

4. **Alert Triggers**
   - 2+ elite initiations → HIGH priority review
   - 2+ elite exits → HIGH priority review  
   - Crowding flags → MEDIUM priority monitor

5. **Integration Pattern**
   - Confirmation-only (never discovery)
   - No gating (all candidates proceed to output)
   - No overlay (no score multiplication)
   - Alert generation only

---

## 10. Next Steps

### Week 1: Data Infrastructure
- [ ] Build `edgar_13f_extractor.py`
- [ ] Create `manager_registry.json` with elite CIKs
- [ ] Implement CUSIP→Ticker mapping with OpenFIGI
- [ ] Test extraction for Q3 2024 (most recent quarter)

### Week 2: Module Integration  
- [ ] Create `module_validation_institutional.py` wrapper
- [ ] Modify `run_screen.py` to call validation post-composite
- [ ] Update `generate_all_reports.py` with new columns
- [ ] Add institutional summary to Top 60 report

### Week 3: Alert System
- [ ] Build `institutional_alerts.py`
- [ ] Add alert report to output suite
- [ ] Test alert triggers on historical data
- [ ] Create `refresh_institutional_data.sh`

### Week 4: Validation
- [ ] Run integrated screen on full 322-ticker universe
- [ ] Validate institutional scores against known elite positions
- [ ] Verify NO_DATA handling for uncovered tickers
- [ ] Generate sample reports for IC review

---

## 11. Risk Mitigation

**Risk 1: Data Lag**
- 13F filings are 45-90 days behind real-time
- **Mitigation:** Clearly label data staleness in reports
- **Accept:** This is industry-standard limitation

**Risk 2: False Negatives (Good stocks with NO_DATA)**
- Elite managers don't track every good biotech
- **Mitigation:** NO_DATA ≠ negative signal
- **Accept:** Fundamentals-only approach for these tickers

**Risk 3: Crowding Reflexivity**
- High institutional ownership can create exit cascades
- **Mitigation:** Crowding flags + alerts
- **Accept:** Risk to monitor, not avoid entirely

**Risk 4: Look-Ahead Bias**
- Using filing_date instead of quarter_end for PIT
- **Mitigation:** Module already implements proper PIT discipline
- **Verify:** Test historical backtests with filed_at dates

---

## Summary

**What This Adds:**
- Institutional validation scores (0-1) as confirmation layer
- Elite activity columns (holders, initiations, adds, exits)
- Crowding risk flags for reflexivity protection
- Quarterly alert system for exceptional activity

**What This Doesn't Change:**
- Composite scoring methodology (Modules 1-5 unchanged)
- Ranking algorithm (still based on Module 5 scores)
- Weekly screening cadence (inst data static between quarters)
- Constitutional governance (confirmation-only philosophy)

**Implementation Timeline:** 4 weeks to production

**Success Criteria:**
- 40-50% institutional coverage achieved
- Zero false positives in alert system
- All reports include institutional columns
- IC members can trace any institutional claim to 13F filing

---

**Document Version:** 1.0  
**Status:** Ready for Implementation  
**Next Review:** After Phase 1 completion (Week 1)
