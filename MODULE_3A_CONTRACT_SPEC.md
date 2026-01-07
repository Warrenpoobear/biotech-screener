# Module 3A: CT.gov Catalyst Delta Detection
## Complete Contract Specification v1.1

**Status**: Ready for Implementation  
**Owner**: Wake Robin Capital Management  
**Integration**: Pre-Backtest Production Hardening  
**Timeline**: 2 weeks (compressed from 4)

---

## 1. Contract Overview

### 1.1 Purpose
Module 3A detects clinical trial status changes from ClinicalTrials.gov as PIT-compliant catalyst events. Unlike predictive models, it treats CT.gov as an "update-event feed" and timestamps events using CT.gov's own metadata.

### 1.2 Scope
**IN SCOPE:**
- Status change detection (RECRUITING → TERMINATED, etc.)
- Timeline changes (primary/study completion date shifts ≥14 days)
- Date confirmations (ANTICIPATED → ACTUAL)
- Results posting events
- PIT-clean delta detection with market calendar
- Deterministic output with audit hashing

**OUT OF SCOPE (Module 3B):**
- FDA AdCom calendar (different event shape)
- PDUFA dates (requires separate scraper)
- Company IR presentations (conference calendars)

### 1.3 Dependencies
- **Input**: `production_data/trial_records.json` (from existing Wake Robin collector)
- **State**: `production_data/ctgov_state/state_YYYY-MM-DD.jsonl` (managed by Module 3A)
- **Market Calendar**: Existing Wake Robin utility for `next_trading_day()`
- **Integration**: `run_screen.py` pipeline (Module 3 → Module 5)

---

## 2. Input/Output Contract

### 2.1 Input: trial_records.json

**Format**: List of trial records from CT.gov v2 API

**Required Fields Per Record:**
```python
{
  "ticker": "ACAD",  # May be at root or in wrapper
  "nct_id": "NCT01234567",  # May be nested in protocolSection
  "ctgov_record": {  # Optional wrapper
    "protocolSection": {
      "identificationModule": {
        "nctId": "NCT01234567"
      },
      "statusModule": {
        "overallStatus": "RECRUITING",
        "lastUpdatePostDateStruct": {
          "date": "2024-01-15"  # CRITICAL: PIT anchor
        },
        "primaryCompletionDateStruct": {
          "date": "2024-06-01",
          "type": "ANTICIPATED"
        },
        "completionDateStruct": {
          "date": "2024-12-01",
          "type": "ANTICIPATED"
        }
      },
      "resultsSection": {  # Optional
        "resultsFirstPostDateStruct": {
          "date": "2024-07-15"
        }
      }
    }
  }
}
```

**Adapter Contract:**
- Must handle 3 input variants: raw CT.gov v2, flattened, or hybrid
- Must extract `last_update_posted` (fail hard if missing)
- Must validate `last_update_posted ≤ as_of_date` (PIT compliance)

### 2.2 Output: catalyst_events_YYYY-MM-DD.json

**Format**: Deterministic catalyst summary per ticker

```json
{
  "run_metadata": {
    "as_of_date": "2024-01-15",
    "prior_snapshot_date": "2024-01-08",
    "tickers_analyzed": 98,
    "events_detected": 12,
    "severe_negatives": 2,
    "module_version": "3A.1.1"
  },
  "summaries": [
    {
      "ticker": "ACAD",
      "as_of_date": "2024-01-15",
      "catalyst_score_pos": 0.0,
      "catalyst_score_neg": 2.85,
      "catalyst_score_net": -2.85,
      "nearest_positive_days": null,
      "nearest_negative_days": 1,
      "severe_negative_flag": true,
      "events": [
        {
          "source": "CTGOV",
          "nct_id": "NCT01234567",
          "event_type": "CT_STATUS_SEVERE_NEG",
          "direction": "NEG",
          "impact": 3,
          "confidence": 0.95,
          "disclosed_at": "2024-01-14",
          "fields_changed": {
            "overallStatus": ["RECRUITING", "TERMINATED"]
          },
          "actual_date": null
        }
      ],
      "audit_hash": "7f3e9a2b1c4d5e6f"
    }
  ]
}
```

### 2.3 Output: run_log_YYYY-MM-DD.json (Non-Deterministic)

**Format**: Operational metadata (separate from deterministic output)

```json
{
  "run_timestamp": "2024-01-15T06:45:00Z",
  "execution_time_seconds": 127.3,
  "host": "prod-runner-01",
  "git_commit": "abc123def",
  "config": {
    "noise_band_days": 14,
    "recency_threshold_days": 90
  },
  "warnings": ["Unknown status string for NCT999: 'PAUSED' → UNKNOWN"],
  "errors": []
}
```

---

## 3. Event Type Taxonomy

### 3.1 Event Types

```python
class EventType(Enum):
    CT_STATUS_SEVERE_NEG = "CT_STATUS_SEVERE_NEG"       # SUSPENDED/TERMINATED/WITHDRAWN
    CT_STATUS_DOWNGRADE = "CT_STATUS_DOWNGRADE"         # Status worsened
    CT_STATUS_UPGRADE = "CT_STATUS_UPGRADE"             # Status improved
    CT_TIMELINE_PUSHOUT = "CT_TIMELINE_PUSHOUT"         # Completion date +14 days
    CT_TIMELINE_PULLIN = "CT_TIMELINE_PULLIN"           # Completion date -14 days
    CT_DATE_CONFIRMED_ACTUAL = "CT_DATE_CONFIRMED_ACTUAL"  # ANTICIPATED → ACTUAL
    CT_RESULTS_POSTED = "CT_RESULTS_POSTED"             # Results appeared
```

### 3.2 Detection Rules

**Status Changes:**
```python
# Status ordering (higher = better)
WITHDRAWN(0) < TERMINATED(1) < SUSPENDED(2) < UNKNOWN(3) < 
ENROLLING_BY_INVITATION(4) < NOT_YET_RECRUITING(5) < RECRUITING(6) < 
ACTIVE_NOT_RECRUITING(7) < COMPLETED(8)

# Classification
if new_status in {SUSPENDED, TERMINATED, WITHDRAWN}:
    → CT_STATUS_SEVERE_NEG (impact=3)
elif new_status.value < old_status.value:
    → CT_STATUS_DOWNGRADE (impact=1-3 based on delta)
elif new_status.value > old_status.value:
    → CT_STATUS_UPGRADE (impact=1-3 based on delta)
```

**Timeline Changes:**
```python
delta_days = (new_date - old_date).days

if abs(delta_days) < 14:
    → Ignore (noise band)
elif abs(delta_days) < 60:
    → impact = 1
elif abs(delta_days) < 180:
    → impact = 2
else:
    → impact = 3

if delta_days >= 14:
    → CT_TIMELINE_PUSHOUT (negative)
else:  # delta_days <= -14
    → CT_TIMELINE_PULLIN (positive)
```

**Date Confirmations:**
```python
if old_type == "ANTICIPATED" and new_type == "ACTUAL":
    days_since_actual = (as_of_date - actual_date).days
    if days_since_actual <= 90:
        → CT_DATE_CONFIRMED_ACTUAL (impact=1)
    else:
        → Ignore (stale)
```

### 3.3 Event Scoring

```python
def compute_event_score(event, as_of_date, calendar):
    """
    Score = impact × confidence × proximity
    """
    if event.event_type == CT_DATE_CONFIRMED_ACTUAL:
        # Decay from actual_date (not disclosed_at)
        days_since_actual = (as_of_date - event.actual_date).days
        proximity = 1.0 / (1.0 + days_since_actual / 30.0)
    else:
        # Full proximity for "now" events
        proximity = 1.0
    
    return event.impact * event.confidence * proximity
```

---

## 4. PIT Compliance Protocol

### 4.1 Disclosure Date Rule

**CRITICAL**: Use CT.gov's `lastUpdatePostDateStruct.date` + market calendar

```python
def effective_trading_date(disclosed_at: date, calendar: MarketCalendar) -> date:
    """
    Conservative: treat disclosed_at as effective next trading day
    Prevents same-day lookahead and handles weekends/holidays
    """
    return calendar.next_trading_day(disclosed_at)
```

**Example:**
- CT.gov updates on Friday 2024-01-12
- Effective date: Monday 2024-01-15
- Prevents weekend bias in event studies

### 4.2 Validation Gates

```python
# Gate 1: Future data detection (leakage)
if last_update_posted > as_of_date:
    raise FutureDataError("Leakage detected")

# Gate 2: Missing PIT anchor (fail hard)
if last_update_posted is None:
    raise MissingRequiredFieldError("last_update_posted required")

# Gate 3: Status coverage (warn/fail)
if pct_missing_overall_status > 0.10:
    raise ValidationError("Status coverage too low")
```

---

## 5. State Management

### 5.1 State Snapshot Schema

```python
@dataclass
class TrialStateSnapshot:
    """Minimal state for delta detection"""
    nct_id: str
    ticker: str
    snapshot_date: date
    overall_status: str
    primary_completion_date: date | None
    primary_completion_type: str | None
    completion_date: date | None
    completion_type: str | None
    results_first_posted: date | None
    last_update_posted: date
```

### 5.2 Storage Format

**JSONL** (one trial per line, sorted by ticker|nct_id):

```
production_data/ctgov_state/
├── state_2024-01-15.jsonl     # One file per snapshot
├── state_2024-01-22.jsonl
├── state_2024-01-29.jsonl
└── manifest.json              # Snapshot inventory
```

**Benefits:**
- Single open/close per snapshot (10× faster than per-file)
- Sorted keys enable binary search
- Git-friendly (no 1000+ file sprawl)
- Deterministic diffing

---

## 6. Integration with run_screen.py

### 6.1 Pipeline Position

```
[1/7] Module 1: Universe filtering
[2/7] Module 2: Financial health
[3/7] Module 3: Catalyst detection  ← NEW
[4/7] Module 4: Clinical development
[5/7] Module 5: Composite ranking  ← Consumes catalyst scores
[6/7] Defensive overlay
[7/7] Top-N selection
```

### 6.2 Module 3 Call Signature

```python
# In run_screen.py
from module_3_catalyst import compute_module_3_catalyst

m3_result = compute_module_3_catalyst(
    trial_records_path=data_dir / "trial_records.json",
    prior_state_path=data_dir / "ctgov_state",
    active_tickers=set(active_tickers),
    as_of_date=as_of_date,
    market_calendar=market_calendar,
    config=module_3a_config
)

# Returns
{
    "summaries": {ticker: TickerCatalystSummary},
    "diagnostic_counts": {
        "events_detected": 12,
        "severe_negatives": 2,
        "tickers_with_events": 8
    },
    "as_of_date": "2024-01-15"
}
```

### 6.3 Module 5 Integration

```python
# Module 5 consumes catalyst scores
composite_score = (
    0.25 * module_2_financial_score +
    0.15 * module_3_catalyst_score +  # NEW
    0.40 * module_4_clinical_score +
    0.20 * other_factors
)

# Kill switch
if catalyst_summary.severe_negative_flag:
    exclude_from_ranking = True
```

---

## 7. Configuration

### 7.1 module_3a_config.yaml

```yaml
# Rate limiting (adjust based on API behavior)
requests_per_second: 1.0
max_retries: 3
backoff_factor: 2.0

# Signal parameters
timeline_noise_band_days: 14
date_confirmation_recency_days: 90
date_confirmation_decay_constant: 30.0

# Impact thresholds (days)
timeline_thresholds: [60, 180]

# Confidence scores
confidence_scores:
  CT_STATUS_SEVERE_NEG: 0.95
  CT_STATUS_DOWNGRADE: 0.85
  CT_STATUS_UPGRADE: 0.80
  CT_TIMELINE_PUSHOUT: 0.75
  CT_TIMELINE_PULLIN: 0.70
  CT_DATE_CONFIRMED_ACTUAL: 0.85
  CT_RESULTS_POSTED: 0.90

# Validation thresholds
max_missing_overall_status: 0.05  # 5% warning threshold
fail_on_future_data: true
```

---

## 8. Testing Requirements

### 8.1 Unit Tests

```python
class TestModule3A:
    def test_status_downgrade_detection(self):
        """RECRUITING → TERMINATED = SEVERE_NEG"""
        
    def test_effective_trading_date_skips_weekend(self):
        """Friday disclosure → Monday effective"""
        
    def test_date_confirmation_proximity_uses_actual_date(self):
        """Proximity decays from actual_date, not disclosed_at"""
        
    def test_nearest_days_unsigned(self):
        """nearest_*_days ≥ 1 for delta events"""
        
    def test_deterministic_output_excludes_timestamps(self):
        """No run_timestamp in catalyst_events.json"""
        
    def test_jsonl_storage_roundtrip(self):
        """Snapshot serialization/deserialization"""
```

### 8.2 Integration Tests

```python
def test_end_to_end_pipeline():
    """Full pipeline from trial_records.json → catalyst_events.json"""
    
def test_run_screen_integration():
    """Module 3 output consumed by Module 5"""
    
def test_historical_validation():
    """Detect known events (2020-2024)"""
```

### 8.3 Validation Metrics (30-Day Post-Launch)

**Data Quality:**
- Coverage: ≥95% of tracked trials have valid state
- Freshness: State lags CT.gov by ≤7 days
- Completeness: Required fields present for ≥90%

**Signal Quality:**
- Precision: Severe negatives align with known failures (≥90%)
- Recall: Capture ≥80% of known setbacks
- Timeliness: Events detected within 1 week

**System Quality:**
- Determinism: 100% audit hash match on re-runs
- PIT Compliance: Zero leakage violations
- Latency: Weekly update completes in <30 minutes

---

## 9. Deliverables

### 9.1 Week 1 (Engine)

**Day 1-2: Core Components**
- `ctgov_adapter.py` - Field extraction with validation
- `state_management.py` - JSONL storage with sorted keys
- Unit tests (30+ tests)

**Day 3-4: Event Detection**
- `event_detector.py` - Delta detection + classification
- `event_scoring.py` - Proximity-weighted scoring
- Market calendar integration

**Day 5: Pipeline**
- `module_3_catalyst.py` - Orchestrator
- `run_screen.py` integration
- End-to-end test

### 9.2 Week 2 (Production)

**Day 1-2: Validation**
- Historical backtest (2020-2024)
- Precision/recall metrics
- PIT compliance audit

**Day 3: Integration**
- Module 5 composite scoring
- BQS worksheet columns
- Health checks

**Day 4: Documentation**
- Operational runbook
- Example notebook
- Video walkthrough

**Day 5: Deployment**
- Production deployment
- First weekly run
- IC sign-off

---

## 10. Success Criteria

### 10.1 Go/No-Go Gates

**Gate 1: Data Coverage** (Day 5)
- ✅ Adapter extracts ≥95% of trial records
- ✅ Zero leakage violations
- ✅ PIT anchor present for 100%

**Gate 2: Signal Quality** (Day 10)
- ✅ Precision ≥90% on known events
- ✅ Recall ≥80% on known events
- ✅ Determinism verified (byte-identical re-runs)

**Gate 3: Integration** (Day 15)
- ✅ run_screen.py pipeline working end-to-end
- ✅ Module 5 consumes catalyst scores correctly
- ✅ Health checks operational

---

## 11. Risk Mitigation

### 11.1 Schema Drift

**Risk**: CT.gov API changes field names/structure  
**Mitigation**: Adapter validation gates detect coverage drops

### 11.2 Rate Limiting

**Risk**: CT.gov throttles requests  
**Mitigation**: Configurable rate limiting + exponential backoff

### 11.3 PIT Violations

**Risk**: Lookahead bias in backtests  
**Mitigation**: Strict validation + leakage detector scans

---

## 12. Next Steps

1. **Validate Input Format**: Provide sample from `trial_records.json`
2. **Build Adapter**: Verify field extraction paths
3. **Implement Engine**: Delta detection + scoring
4. **Integration Test**: End-to-end with run_screen.py
5. **Deploy**: First production run

---

## Appendix A: Confidence Score Rationale

| Event Type | Confidence | Rationale |
|-----------|-----------|-----------|
| CT_STATUS_SEVERE_NEG | 0.95 | Official registry fact, unambiguous |
| CT_STATUS_DOWNGRADE | 0.85 | Clear direction, some interpretation |
| CT_STATUS_UPGRADE | 0.80 | Positive but less reliable predictor |
| CT_TIMELINE_PUSHOUT | 0.75 | Can be noisy (bureaucratic delays) |
| CT_TIMELINE_PULLIN | 0.70 | Rare, often administrative |
| CT_DATE_CONFIRMED_ACTUAL | 0.85 | Opens analysis window, high signal |
| CT_RESULTS_POSTED | 0.90 | Fact-based, but direction unknown |

---

## Appendix B: Field Extraction Determinism

**Try Order** (stops at first non-null):

```python
# ticker
1. record.get("ticker")
2. record.get("symbol")

# nct_id
1. record.get("nct_id")
2. root["protocolSection"]["identificationModule"]["nctId"]
3. root.get("nctId")

# last_update_posted (CRITICAL)
1. record.get("last_update_posted")
2. root["protocolSection"]["statusModule"]["lastUpdatePostDateStruct"]["date"]
3. root["statusModule"]["lastUpdatePostDateStruct"]["date"]
```

This ensures deterministic extraction regardless of input variant.
