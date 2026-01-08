# SYSTEM_STATE.md
# Wake Robin Biotech Alpha System - System Architecture & State

**Last Updated:** 2026-01-08  
**Version:** 1.0.0-production  
**Owner:** Darren @ Brooks Capital Management

---

## System Overview

The Wake Robin Biotech Alpha System is a deterministic, point-in-time (PIT) investment screening framework designed to identify biotech co-investment opportunities by integrating:
- Universe filtering with status gates (Module 1)
- Financial health and cash runway analysis (Module 2)
- Clinical trial catalyst detection (Module 3)
- Clinical development quality scoring (Module 4)
- Composite aggregation and ranking (Module 5)

**Core Philosophy:** Every signal must be deterministic, backtestable, and use only information available at the evaluation date.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     run_screen.py                            │
│                  (Orchestration Layer)                       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────┬───────┼───────┬───────────┐
        │           │       │       │           │
        ▼           ▼       ▼       ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │Module 1 │ │Module 2 │ │Module 3 │ │Module 4 │ │Module 5 │
  │Universe │ │Financial│ │Catalyst │ │Clinical │ │Composite│
  │  ✅     │ │   ✅    │ │   ✅    │ │   ✅    │ │   ✅    │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
       │           │           │           │           │
       └───────────┴───────────┴───────────┴───────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │   Rankings   │
                   │   & Output   │
                   └──────────────┘
```

---

## Module Status

### Module 1: Universe Management (IMPLEMENTED ✅)
**Status:** PRODUCTION v1.0.0  
**File:** `module_1_universe.py`  
**Data Source:** Raw ticker list with company metadata

**Purpose:** Filter investable universe using status gates

**Components:**
1. **Status Classification** - Active vs excluded securities
2. **Shell Company Detection** - SPAC/blank check filtering
3. **Market Cap Minimum** - $50M threshold
4. **Delisting/M&A Filtering** - Remove inactive securities

**Output Schema:**
```python
{
    "as_of_date": str,
    "active_securities": [{"ticker": str, "status": str, "market_cap_mm": str}],
    "excluded_securities": [{"ticker": str, "reason": str}],
    "diagnostic_counts": {"total_input": int, "active": int, "excluded": int}
}
```

---

### Module 2: Financial Health (IMPLEMENTED ✅)
**Status:** ✅ PRODUCTION v1.0  
**File:** `module_2_financial.py`  
**Data Sources:** 
- SEC 10-Q/10-K filings (balance sheet, income statement)
- Market data (price, volume, market cap)

**Current Components:**
1. **Cash Runway (50% weight)** - Months of cash based on burn rate
2. **Dilution Risk (30% weight)** - Cash/market cap ratio with runway penalty
3. **Liquidity (20% weight)** - Market cap + trading volume tiers

**Input Data Schema:**
```python
financial_data = {
    "ticker": str,
    "Cash": float,           # Total cash + equivalents ($)
    "NetIncome": float,      # Quarterly net income ($)
    "R&D": float,            # Quarterly R&D expense ($)
}

market_data = {
    "ticker": str,
    "market_cap": float,     # Market capitalization ($)
    "price": float,          # Current stock price ($)
    "avg_volume": float,     # 30-day average volume (shares)
}
```

**Output Schema:**
```python
{
    "ticker": str,
    "financial_normalized": float,  # 0-100 composite score
    "runway_months": float | None,
    "runway_score": float | None,
    "dilution_score": float | None,
    "liquidity_score": float | None,
    "cash_to_mcap": float | None,
    "monthly_burn": float | None,
    "has_financial_data": bool
}
```

**Planned Enhancement (IN PROGRESS):**
- **Burn Acceleration Detection** - Track quarterly burn rate changes
- **Catalyst Timing Integration** - Coverage ratio (runway vs time-to-catalyst)
- **Recent Financing Dampener** - Adjust penalty after verified capital raises

---

### Module 3: Catalyst Detection (IMPLEMENTED ✅)
**Status:** PRODUCTION v3A.1.1  
**Files:** `module_3_catalyst.py`, `module_3_scoring.py`, `ctgov_adapter.py`, `event_detector.py`, `state_management.py`, `catalyst_summary.py`  
**Data Source:** ClinicalTrials.gov (CT.gov) trial records

**Purpose:** Detect and score clinical trial catalyst events through state-based change detection

**Architecture:**
1. **State Management** - Snapshot-based PIT validation
2. **Event Detection** - Compare current vs prior trial states
3. **Event Scoring** - Classify severity (CRITICAL_POSITIVE, POSITIVE, NEGATIVE, SEVERE_NEGATIVE)
4. **Aggregation** - Roll up events to ticker-level summaries

**Key Event Types:**
- **Critical Positive:** Phase advance (P2→P3, P3→NDA), FDA approval, breakthrough designation
- **Positive:** Enrollment complete/started, fast track, orphan designation
- **Negative:** Enrollment delay, trial delayed
- **Severe Negative:** Trial termination, suspension, FDA rejection, safety hold

**Event Detector Config:**
- `noise_band_days`: 7 (ignore date shifts within 7 days)
- `recency_threshold_days`: 730 (2 years)
- `decay_constant`: 30.0 (exponential decay for event recency)

**Output Schema:**
```python
{
    "summaries": {
        ticker: {
            "ticker": str,
            "events": [{"event_type": str, "nct_id": str, "detected_date": str}],
            "severe_negative_flag": bool,
            "score": float  # 0-100
        }
    },
    "diagnostic_counts": {
        "events_detected": int,
        "severe_negatives": int,
        "tickers_with_events": int
    }
}
```

---

### Module 4: Clinical Development Quality (IMPLEMENTED ✅)
**Status:** PRODUCTION v1.0.1-CUSTOM  
**File:** `module_4_clinical_dev.py`  
**Data Source:** ClinicalTrials.gov trial records

**Purpose:** Score clinical program quality and execution

**Components (120 pts, normalized to 0-100):**
1. **Phase Score (30 pts)** - Most advanced phase (Approved=30, P3=25, P2=18, P1=8)
2. **Phase Progress Bonus (5 pts)** - Continuous scoring by phase
3. **Trial Count Bonus (5 pts)** - Pipeline diversification (0 trials=0, 20+=5)
4. **Indication Diversity Bonus (5 pts)** - Multiple indications reduce binary risk
5. **Recency Bonus (5 pts)** - Most recent trial update (<30d=5, >2yr=1)
6. **Design Quality (25 pts)** - Randomization, blinding, endpoint strength
7. **Execution Track Record (25 pts)** - Completion vs termination rates
8. **Endpoint Portfolio (20 pts)** - Strong endpoints (OS, PFS, ORR) vs weak (biomarker, PK)

**Design Quality Indicators:**
- **Strong Endpoints:** Overall survival, PFS, complete response, ORR
- **Weak Endpoints:** Biomarker, pharmacokinetic, safety-only

**Output Schema:**
```python
{
    "scores": [{
        "ticker": str,
        "clinical_score": str,  # 0-100 normalized
        "phase_score": str,
        "trial_count_bonus": str,
        "diversity_bonus": str,
        "recency_bonus": str,
        "design_score": str,
        "execution_score": str,
        "endpoint_score": str,
        "lead_phase": str,
        "trial_count": int,
        "flags": List[str],
        "severity": str
    }],
    "diagnostic_counts": {"scored": int, "total_trials": int, "pit_filtered": int}
}
```

---

### Module 5: Composite Scoring (IMPLEMENTED ✅)
**Status:** PRODUCTION  
**File:** `module_5_composite.py`  
**Purpose:** Aggregate module scores into final rankings

**Scoring Framework:**
```python
final_score = (
    financial_score * 0.40 +      # Module 2
    catalyst_score * 0.25 +       # Module 3
    clinical_score * 0.25 +       # Module 4
    liquidity_score * 0.10        # Module 2 component
)
```

**Output:** Ranked list with composite scores and component breakdowns

---

### Future Modules (PLANNED)

### Module 6: Elite Manager 13F (PLANNED)
**Status:** Conceptual design complete  
**Data Source:** SEC 13F filings  
**Purpose:** Track positions from elite biotech hedge funds

**Target Managers:**
- Baker Bros Advisors
- RA Capital Management
- Perceptive Advisors
- Logos Capital
- RTW Investments

**Key Features (Planned):**
- Position initiation/increase/decrease tracking
- Elite manager consensus scoring
- Position size as % of manager portfolio
- Quarter-over-quarter flow analysis

---

### Module 7: Insider Transactions (PLANNED)
**Status:** Spec complete, implementation pending  
**Data Source:** SEC Form 4 filings  
**Purpose:** Score director/officer buying behavior

**Key Features (Planned):**
- Cluster buy detection (multiple insiders, short window)
- C-suite buying emphasis
- Size relative to salary
- Timing relative to blackout windows

---

## Data Pipeline Architecture

### Data Sources

#### 1. ClinicalTrials.gov (CT.gov)
**Access:** API or bulk downloads  
**Update Frequency:** Daily  
**Point-in-Time:** Uses `last_update_posted` field for PIT validation

**Data Fields:**
- Trial identification: `nct_id`, `ticker`
- Phase classification: `phase`
- Status: `status` (Active, Completed, Terminated, etc.)
- Dates: `start_date`, `primary_completion_date`, `last_update_posted`
- Design: `randomized`, `blinded`, `primary_endpoint`
- Enrollment: `enrollment`, `enrollment_type`
- Conditions: `conditions` (indication/disease areas)

**Processing Pipeline:**
1. **ctgov_adapter.py** - Converts raw CT.gov records to canonical format
2. **state_management.py** - Maintains historical snapshots for change detection
3. **event_detector.py** - Compares current vs prior state to identify events

#### 2. SEC EDGAR Filings
**Access:** Direct EDGAR API or bulk downloads  
**Rate Limit:** 10 requests/second with User-Agent  
**Point-in-Time:** Filing date determines data availability

**Parsing Requirements:**
- **10-Q/10-K:** Extract cash flow statement (CFO), balance sheet (cash), income statement
- **CRITICAL:** Handle YTD vs quarterly reporting
  - Q1: Already quarterly
  - Q2: YTD(Q2) - YTD(Q1)
  - Q3: YTD(Q3) - YTD(Q2)
  - Q4: Annual(10-K) - YTD(Q3)
- **Fiscal Year Handling:** Use `fiscalYear` and `fiscalPeriod` metadata, NOT filing dates
- **13F (planned):** Parse holdings with cusip → ticker mapping
- **Form 4 (planned):** Parse insider transaction tables

#### 3. Market Data
**Access:** Financial data provider API (specify vendor)  
**Frequency:** Daily EOD prices + volume  
**Fields:** price, volume, market_cap, shares_outstanding

---

## Data Flow Standards

### Point-in-Time Validation
**Every data point must include:**
```python
{
    "as_of_date": "YYYY-MM-DD",  # Data valid as of this date
    "source_date": "YYYY-MM-DD",  # Original filing/publication date
    "ticker": str,
    # ... data fields ...
}
```

### Missing Data Handling
- **Cash/burn data missing:** Return `None`, assign neutral score (50.0)
- **Market data missing:** Skip ticker or use last known value with staleness flag
- **Catalyst data missing:** Set `catalyst_timing_flag = "UNKNOWN"`

### Data Quality Flags
Every module output includes quality metadata:
```python
{
    "has_financial_data": bool,
    "cfo_quality_flag": str,  # "OK" | "NOISY" | "MISSING"
    "catalyst_timing_flag": str,  # "KNOWN" | "UNKNOWN" | "FAR"
}
```

---

## Module Integration Standards

### Input/Output Contracts
Each module follows this interface:

```python
def compute_module_X(
    # Required inputs
    active_tickers: List[str],     # From Module 1
    as_of_date: str,               # YYYY-MM-DD
    # Module-specific data sources
    **data_sources
) -> Dict:
    """
    Returns:
        {
            "scores" or "summaries": List[Dict],  # One per ticker
            "diagnostic_counts": Dict,
            "as_of_date": str,
            "provenance": Dict  # PIT validation metadata
        }
    """
```

### Soft Dependencies
Modules must gracefully handle missing upstream data:
- **Module 2** computes runway without catalyst timing (independent operation)
- **Module 2 enhanced** will compute coverage ratio only when Module 3 provides catalyst dates
- **Module 5** handles missing module scores by using neutral values (50.0)

### Point-in-Time Validation
**Common utilities (from common/):**
- `compute_pit_cutoff(as_of_date)` - Returns cutoff datetime
- `is_pit_admissible(source_date, cutoff)` - Validates data availability
- `create_provenance(version, inputs, cutoff)` - Generates audit trail

Every module filters data using PIT cutoff before processing.

---

## Production Pipeline Flow

```
1. run_screen.py orchestrator starts
   ↓
2. Load raw universe + market data
   ↓
3. Module 1: Filter to active securities
   ↓ (active_tickers)
4. Module 2: Score financial health
   ↓ (financial_scores)
5. Module 3: Detect catalyst events
   ↓ (catalyst_summaries)
6. Module 4: Score clinical development
   ↓ (clinical_scores)
7. Module 5: Compute composite scores
   ↓
8. Generate rankings + output files
   ↓
9. Write diagnostic reports
```

**Execution Time:** ~30-60 seconds for 200-ticker universe

---

## Known Limitations & Risks

### Data Quality
- **SEC filing delays:** 10-Q due 40-45 days after quarter end
- **Restatements:** Financial data can be revised
- **Delisted tickers:** Handle exchange removals gracefully
- **CT.gov update lag:** Trials may take 30+ days to reflect status changes

### Model Risks
- **Overfitting:** Keep feature count low, validate out-of-sample
- **Regime changes:** Backtest across multiple market cycles
- **Survivorship bias:** Include delisted/bankrupt companies in historical universe
- **Event detection noise:** 7-day noise band may miss legitimate quick changes

### Operational Risks
- **State corruption:** ctgov_state snapshots must be version-controlled
- **API rate limits:** CT.gov and SEC EDGAR have request throttling
- **Disk space:** State snapshots accumulate over time

---

## Backtest Framework (PLANNED)

### Expected File Locations
```
/production_data/
├── universe.json                # Module 1 output (active tickers)
├── financial_data.json          # SEC 10-Q/10-K extracts
├── market_data.json             # Daily price/volume data
├── trial_records.json           # CT.gov trial data
├── ctgov_state/                 # State snapshots for change detection
│   └── snapshot_YYYY-MM-DD.json
├── catalyst_events_YYYY-MM-DD.json  # Module 3 output
└── run_log_YYYY-MM-DD.json      # Execution logs
```

### Data Refresh Schedule
- **Market data:** Daily (EOD)
- **SEC filings:** Continuous monitoring, parse within 24h
- **Clinical trials:** Weekly refresh from CT.gov
- **State snapshots:** Created on each Module 3 run
- **13F filings (planned):** Quarterly (45 days post quarter-end)

---

## Scoring Framework

### Composite Score Calculation (Module 5)
```python
final_score = (
    financial_score * 0.40 +      # Module 2 (cash runway, dilution risk, liquidity)
    catalyst_score * 0.25 +       # Module 3 (event-based catalyst detection)
    clinical_score * 0.25 +       # Module 4 (clinical development quality)
    liquidity_score * 0.10        # Module 2 component (market cap + volume)
)
```

**Score Range:** 0-100 (normalized)

### Module Weights Rationale
- **Financial (40%):** Cash runway is the #1 killer in biotech - prevent dilution traps
- **Catalyst (25%):** Event-driven alpha from trial progression/setbacks
- **Clinical (25%):** Pipeline quality and execution track record
- **Liquidity (10%):** Ensures tradability, but secondary to fundamentals

### Threshold Rules
- **Strong Buy:** Final score ≥ 75
- **Buy:** Final score 65-75
- **Hold:** Final score 50-65
- **Avoid:** Final score < 50 OR severe negative catalyst flag

---

## Current Implementation Priority

### Module 2 Enhancement (IN PROGRESS - HIGHEST PRIORITY)
**Task:** Add burn acceleration + catalyst timing integration  
**Status:** Spec finalized, implementation pending  
**Deliverables:**
- [ ] `quarterly_from_ytd()` helper function (handles fiscal year edge cases)
- [ ] Burn acceleration calculator (4Q trailing avg vs recent quarter)
- [ ] Runway vs catalyst coverage ratio (soft dependency on Module 3)
- [ ] Recent financing dampener (if raise data available)
- [ ] Enhanced output schema with diagnostic fields

**Integration Points:**
- **Soft dependency on Module 3:** If catalyst timing available, compute coverage ratio
- **Graceful degradation:** Module 2 operates independently when catalyst data unavailable
- **Output enhancement:** Add `burn_acceleration`, `coverage`, `ttc_months`, `catalyst_timing_flag`

---

## Module Integration Standards

### Backtest Requirements
1. **Point-in-time data only** - No forward-looking bias
2. **Rebalance frequency** - Monthly or quarterly
3. **Transaction costs** - 20bps assumed
4. **Position sizing** - Equal weight or score-weighted
5. **Universe constraints** - Min market cap $50M, sufficient liquidity

### Performance Metrics
- Sharpe ratio
- Max drawdown
- Win rate
- Alpha vs XBI (biotech ETF benchmark)
- Information ratio

---

## Development Workflow

### Code Standards
- **Deterministic:** No randomness, same inputs → same outputs
- **Point-in-Time Enforced:** All data validated via PIT cutoff
- **Documented:** Every function has purpose + return value docstring
- **Testable:** Sample data in `if __name__ == "__main__"` block
- **Diff-only updates:** Changes delivered as patches (≤30 lines per file)

### Common Utilities (common/ directory)
- `provenance.py` - Audit trail generation
- `pit_enforcement.py` - Point-in-time validation
- `types.py` - Shared enums (StatusGate, Severity)

### Module Naming Convention
- `module_N_name.py` - Main module logic
- `module_N_*.py` - Supporting submodules (e.g., module_3_scoring.py)

### Git Workflow (When Implemented)
```bash
git checkout -b feature/module-2-burn-acceleration
# ... make changes ...
git commit -m "Add burn acceleration detector to Module 2"
git push origin feature/module-2-burn-acceleration
```

---

## Contact & Ownership

**System Owner:** Darren, Director of Investments  
**Organization:** Brooks Capital Management  
**Production Environment:** Local Python 3.9+ environment  
**Primary Use Case:** Biotech co-investment screening for $775M portfolio

---

## Document History

- **2026-01-08:** Complete system state documentation reflecting production Modules 1-5
- **Future:** Update after Module 2 burn acceleration enhancement

---

## Appendix: Module Interdependencies

```
Module 1 (Universe) → active_tickers → All other modules
                                      ↓
                        Module 2 (Financial) ← financial_data
                                      ↓        market_data
                        Module 3 (Catalyst) ← trial_records
                                      ↓        ctgov_state/
                        Module 4 (Clinical) ← trial_records
                                      ↓
                        Module 5 (Composite) ← All module scores
                                      ↓
                                  Final rankings
```

### Data Dependencies
- **Module 1** needs: raw_records (company metadata), market_data
- **Module 2** needs: financial_data (10-Q/10-K), market_data, [optional: catalyst timing from Module 3]
- **Module 3** needs: trial_records, ctgov_state/, as_of_date
- **Module 4** needs: trial_records, as_of_date
- **Module 5** needs: All module outputs

---

## Appendix: Data Field Mappings

### SEC → Module 2 Financial
```python
# From 10-Q/10-K Cash Flow Statement
CFO = "CashFromOperatingActivities"  # Can be negative (burn)
Cash = "CashAndEquivalents" + "MarketableSecurities"

# From Income Statement  
NetIncome = "NetIncomeLoss"  # Quarterly
R&D = "ResearchAndDevelopmentExpense"  # Quarterly
```

### CT.gov → Module 3 Catalyst
```python
# Trial identification
nct_id = "NCTId"
ticker = mapped_from_sponsor_or_manual

# Phase/Status
phase = "Phase"  # ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
status = "OverallStatus"  # ["Active, not recruiting", "Completed", "Terminated"]

# Dates (PIT validation)
last_update_posted = "LastUpdatePostDate"  # ISO format
primary_completion_date = "PrimaryCompletionDate"
```

### Module 1 → All Modules
```python
active_tickers = [sec["ticker"] for sec in active_securities]
# Used as filtering whitelist for Modules 2-5
```

---

**END OF SYSTEM_STATE.md**
