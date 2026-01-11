# Module 3 Catalyst Upgrade vNext - Migration Guide

## Version Information

| Version | Value |
|---------|-------|
| Schema Version | `m3catalyst_vnext_20260111` |
| Score Version | `m3score_vnext_20260111` |
| Module Version | `3A.2.0` |

## Overview

Module 3 vNext is a deterministic, PIT-safe upgrade to the catalyst detection system. It provides:

- **Deterministic outputs**: Same inputs → byte-identical outputs
- **Point-in-time safety**: No wall-clock usage, as_of_date is required
- **Dual scoring**: Both `score_override` (hierarchical) and `score_blended` (confidence-weighted)
- **Integration hooks**: `next_catalyst_date`, `catalyst_window_bucket`, `catalyst_confidence`
- **Expanded taxonomy**: New event types for protocol amendments, enrollment changes

## Breaking Changes

### 1. `as_of_date` is now REQUIRED

```python
# OLD (allowed defaults)
compute_module_3_catalyst(trial_records_path, state_dir, active_tickers)

# NEW (as_of_date required)
compute_module_3_catalyst(trial_records_path, state_dir, active_tickers, as_of_date=date(2024, 1, 15))
```

### 2. Legacy wrapper removed

The `compute_module_3_catalyst_legacy()` function with hardcoded fallback dates has been removed. Use the canonical `compute_module_3_catalyst()` function.

### 3. Output format changes

The vNext output file is written to `catalyst_events_vnext_YYYY-MM-DD.json` with a new schema. The legacy format is still written to `catalyst_events_YYYY-MM-DD.json` for backwards compatibility.

## Schema Versions

### Event Schema (`CatalystEventV2`)

```json
{
  "event_id": "abc123...",
  "ticker": "AAPL",
  "nct_id": "NCT00000001",
  "event_type": "CT_STATUS_UPGRADE",
  "event_severity": "POSITIVE",
  "event_date": "2024-01-15",
  "field_changed": "overallStatus",
  "prior_value": "RECRUITING",
  "new_value": "ACTIVE_NOT_RECRUITING",
  "source": "CTGOV",
  "confidence": "HIGH",
  "disclosed_at": "2024-01-15"
}
```

### Summary Schema (`TickerCatalystSummaryV2`)

```json
{
  "_schema": {
    "schema_version": "m3catalyst_vnext_20260111",
    "score_version": "m3score_vnext_20260111"
  },
  "ticker": "AAPL",
  "as_of_date": "2024-01-15",
  "scores": {
    "score_override": "60.00",
    "score_blended": "58.50",
    "score_mode_used": "blended"
  },
  "flags": {
    "severe_negative_flag": false
  },
  "integration": {
    "next_catalyst_date": "2024-03-31",
    "catalyst_window_days": 75,
    "catalyst_window_bucket": "31_90",
    "catalyst_confidence": "HIGH"
  },
  "event_summary": {
    "events_total": 5,
    "events_by_severity": {"POSITIVE": 3, "NEUTRAL": 2},
    "events_by_type": {"CT_STATUS_UPGRADE": 2, "CT_TIMELINE_PULLIN": 1},
    "weighted_counts_by_severity": {"POSITIVE": "2.5000"}
  },
  "top_3_events": [...],
  "events": [...]
}
```

## Determinism Guarantees

1. **Event IDs**: SHA256 hash of `(ticker, nct_id, event_type, event_date, field_changed, prior_value, new_value, source)`
2. **Sorting**: Events sorted by `(event_date, event_type, nct_id, field_changed, prior_value, new_value)`
3. **JSON**: Canonical format with sorted keys, no trailing whitespace, UTF-8 encoding
4. **Floats**: All numeric values use `Decimal` for stable representation

## Scoring Modes

### Override Score (Hierarchical)

Priority rules:
1. If ANY severe negative → **20**
2. If ANY critical positive → **75**
3. Otherwise, blend positive/negative counts

### Blended Score (Confidence + Recency Weighted)

Formula:
```
contribution = severity_score × confidence_weight × recency_weight
score = base(50) + Σ(contributions) × staleness_factor
```

Weights:
- HIGH confidence: 1.0
- MED confidence: 0.6
- LOW confidence: 0.3

Recency decay: Exponential with 90-day half-life
Staleness penalty: 0.8x if no events within 180 days

## Event Taxonomy

### Existing Events (unchanged)
- `CT_STATUS_SEVERE_NEG`: Trial terminated/suspended/withdrawn
- `CT_STATUS_UPGRADE`: Status improvement
- `CT_STATUS_DOWNGRADE`: Status regression
- `CT_TIMELINE_PUSHOUT`: Date moved later
- `CT_TIMELINE_PULLIN`: Date moved earlier
- `CT_DATE_CONFIRMED_ACTUAL`: ANTICIPATED → ACTUAL
- `CT_RESULTS_POSTED`: Results first posted

### New Events
- `CT_PRIMARY_COMPLETION`: Primary endpoint completed
- `CT_STUDY_COMPLETION`: Study completed
- `CT_PROTOCOL_AMENDMENT`: Protocol amended
- `CT_ARM_ADDED`: New treatment arm
- `CT_ARM_REMOVED`: Arm removed
- `CT_ENDPOINT_CHANGED`: Endpoint modified
- `CT_ENROLLMENT_STARTED`: Enrollment began
- `CT_ENROLLMENT_COMPLETE`: Enrollment finished
- `CT_ENROLLMENT_PAUSED`: Enrollment paused
- `CT_ENROLLMENT_RESUMED`: Enrollment resumed
- `UNKNOWN`: Catch-all (zero score contribution)

## Integration Hooks

The following fields are available for downstream consumers:

| Field | Type | Description |
|-------|------|-------------|
| `next_catalyst_date` | ISO date or null | Next future catalyst date |
| `catalyst_window_days` | int or null | Days until next catalyst |
| `catalyst_window_bucket` | enum | `0_30`, `31_90`, `91_180`, `181_365`, `>365`, `UNKNOWN` |
| `catalyst_confidence` | enum | `HIGH`, `MED`, `LOW` |
| `top_3_events` | array | Top 3 events by severity/recency/confidence |

## Backwards Compatibility

### Loading Legacy Data

```python
from module_3_schema import TickerCatalystSummaryV2

# Legacy dict (old format)
legacy_data = {...}

# Automatically migrates
summary = TickerCatalystSummaryV2.from_dict(legacy_data)
```

The migration:
1. Detects missing `_schema.schema_version`
2. Maps legacy event types to new `EventType` enum
3. Computes severity from event type
4. Sets default confidence levels

### Output Files

Both formats are written:
- `catalyst_events_vnext_YYYY-MM-DD.json` - New schema
- `catalyst_events_YYYY-MM-DD.json` - Legacy schema

## Running Tests

```bash
# Run all Module 3 vNext tests
pytest tests/test_module_3_vnext.py -v

# Run specific test categories
pytest tests/test_module_3_vnext.py::TestDeterminism -v
pytest tests/test_module_3_vnext.py::TestPITSafety -v
pytest tests/test_module_3_vnext.py::TestSchemaEvolution -v
```

## Example Output

```json
{
  "_schema": {
    "schema_version": "m3catalyst_vnext_20260111",
    "score_version": "m3score_vnext_20260111"
  },
  "run_metadata": {
    "as_of_date": "2024-01-15",
    "prior_snapshot_date": "2024-01-08",
    "module_version": "3A.2.0"
  },
  "diagnostics": {
    "events_detected_total": 42,
    "events_deduped": 3,
    "events_by_type": {"CT_STATUS_UPGRADE": 15, "CT_TIMELINE_PULLIN": 10},
    "tickers_with_severe_negative": 2,
    "tickers_analyzed": 180
  },
  "summaries": {
    "AAPL": {...},
    "BIIB": {...}
  }
}
```
