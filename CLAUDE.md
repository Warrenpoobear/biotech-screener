# CLAUDE.md - AI Assistant Guide for biotech-screener

## Project Overview

**biotech-screener** is a deterministic, point-in-time (PIT) safe biotech investment screening system developed by Wake Robin Capital Management. It implements a multi-module pipeline that combines financial health analysis, clinical trial catalysts, and clinical development metrics to produce ranked investment opportunities.

**Key Principles:**
- **Determinism**: Same inputs always produce byte-identical outputs (no `random`, no `datetime.now()`)
- **Point-in-Time Safety**: Prevents lookahead bias by enforcing strict PIT cutoffs (`source_date <= as_of_date - 1`)
- **Fail-Closed**: Validates data and stops on errors rather than gracefully degrading
- **Audit Trail**: Complete machine-readable governance metadata for reproducibility

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Project Structure

```
biotech-screener/
├── src/                          # Core packaged modules
│   ├── common/                   # Shared utilities (hash, dates, types)
│   ├── providers/                # Clinical trial data providers (AACT)
│   ├── validators/               # Ticker validation and data quality
│   ├── history/                  # 13F holdings snapshot management
│   └── modules/                  # Scoring module definitions
│
├── common/                       # Root-level utilities
│   ├── date_utils.py            # ISO date handling
│   ├── pit_enforcement.py       # PIT cutoff computation
│   ├── provenance.py            # Audit trail and hashing
│   ├── types.py                 # Shared enums (Severity, StatusGate)
│   └── data_quality.py          # Data quality assessment
│
├── tests/                        # Test suite
│   ├── conftest.py              # Shared fixtures
│   ├── integration/             # End-to-end pipeline tests
│   ├── providers/               # Data provider tests
│   └── test_*.py                # Module-specific tests
│
├── data/                        # Reference data and caches
│   ├── aact_snapshots/          # ClinicalTrials.gov snapshots
│   └── pos_benchmarks_bio_2011_2020_v1.json  # PoS reference data
│
├── production_data/             # Production run outputs
├── backtest/                    # Backtesting framework
└── wake_robin_data_pipeline/    # Standalone data pipeline
```

## Core Modules (Pipeline)

| Module | File | Purpose |
|--------|------|---------|
| **Module 1** | `module_1_universe.py` | Universe filtering and classification |
| **Module 2** | `module_2_financial.py` | Financial health scoring (burn rate, dilution, liquidity) |
| **Module 3** | `module_3_catalyst.py` | CT.gov catalyst event detection |
| **Module 4** | `module_4_clinical_dev_v2.py` | Clinical development scoring |
| **Module 5** | `module_5_composite_with_defensive.py` | Composite ranking with defensive overlays |

### Enhancement Engines

- `pos_engine.py` - Indication-specific probability of success
- `short_interest_engine.py` - Squeeze potential and crowding risk
- `regime_engine.py` - Market regime classification (VIX, XBI momentum)
- `defensive_overlay_adapter.py` - Correlation sanitization and position sizing

## Module Integration Contracts

### Data Flow Architecture

```
Module 1 (Universe)
    ↓ outputs: active_securities[] + excluded_securities[]

Module 2 (Financial)  ← inputs: TickerCollection, financial_records, market_data
    ↓ outputs: scores[] {ticker, financial_score, market_cap_mm, severity, flags}

Module 3 (Catalyst)   ← inputs: TickerCollection, trial_records_path, DateLike
    ↓ outputs: summaries{} {ticker: TickerCatalystSummaryV2}

Module 4 (Clinical)   ← inputs: TickerCollection, trial_records
    ↓ outputs: scores[] {ticker, clinical_score, lead_phase, severity}

Module 5 (Composite)  ← inputs: ALL module results
    ↓ outputs: ranked_securities[], excluded_securities[]
```

### Type Conventions

All modules accept flexible input types for consistency:

```python
# Ticker collections - accept both Set and List
from typing import Union, Set, List
TickerCollection = Union[Set[str], List[str]]

# Date inputs - accept both date object and ISO string
from datetime import date
DateLike = Union[str, date]
```

### Standardized Score Field Names

| Module | Primary Field | Legacy Alias |
|--------|--------------|--------------|
| Module 2 | `financial_score` | `financial_normalized` |
| Module 3 | `catalyst_score` | `score_blended`, `catalyst_score_net` |
| Module 4 | `clinical_score` | - |
| Module 5 | `composite_score` | - |

### Module Output Schemas

**Module 1 Output:**
```python
{
    "active_securities": [{"ticker": str, "status": str, "market_cap_mm": float}],
    "excluded_securities": [{"ticker": str, "reason": str}],
    "diagnostic_counts": {...}
}
```

**Module 2 Output:**
```python
{
    "scores": [{
        "ticker": str,
        "financial_score": float,      # Standardized name
        "financial_normalized": float,  # Legacy alias (same value)
        "market_cap_mm": float,         # For Module 5 cohort analysis
        "runway_months": float,
        "severity": str,
        "flags": [str]
    }],
    "diagnostic_counts": {"scored": int, "missing": int}
}
```

**Module 3 Output:**
```python
{
    "summaries": {ticker: TickerCatalystSummaryV2},  # Primary (use this)
    "summaries_legacy": {...},  # DEPRECATED - will be removed in v2.0
    "diagnostic_counts": {...},
    "as_of_date": str,
    "schema_version": str,
    "score_version": str
}
```

**Module 4 Output:**
```python
{
    "as_of_date": str,
    "scores": [{
        "ticker": str,
        "clinical_score": str,  # Decimal as string
        "lead_phase": str,
        "severity": str,
        "flags": [str]
    }],
    "diagnostic_counts": {...},
    "provenance": {...}
}
```

### Schema Validation

Use `common/integration_contracts.py` for validation between modules:

```python
from common.integration_contracts import (
    validate_module_2_output,
    validate_pipeline_handoff,
    extract_financial_score,
    normalize_date_input,
    normalize_ticker_set,
)

# Validate module output
validate_module_2_output(m2_result)

# Validate handoff between modules
validate_pipeline_handoff("module_2", "module_5", m2_result)

# Extract scores with backwards compatibility
score = extract_financial_score(score_record)  # Handles both field names
```

### Deprecation Warnings

The following are deprecated and will be removed in v2.0:

- `summaries_legacy` in Module 3 output - use `summaries` instead
- `diagnostic_counts_legacy` in Module 3 output
- `financial_normalized` field name - use `financial_score` instead

## Coding Conventions

### Use Decimal for All Financial Calculations

```python
from decimal import Decimal

# CORRECT
cash = Decimal("500000000")
runway = (cash / abs(burn)).quantize(Decimal("0.01"))

# WRONG - Never use floats for money
cash = 500000000.0
```

### PIT Safety Pattern

```python
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible

def process_data(records, as_of_date: str):
    pit_cutoff = compute_pit_cutoff(as_of_date)  # as_of_date - 1

    for record in records:
        if not is_pit_admissible(record.get("source_date"), pit_cutoff):
            continue  # Skip future data
        # Process PIT-safe record
```

### Deterministic Hashing

```python
import hashlib
import json
from datetime import date
from decimal import Decimal

def stable_json_dumps(obj):
    """Sorted keys + custom serializers for reproducibility."""
    def default_serializer(o):
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Decimal):
            return str(o)
        raise TypeError(f"Cannot serialize {type(o)}")

    return json.dumps(obj, sort_keys=True, default=default_serializer)

def compute_hash(data) -> str:
    """SHA256 of canonical JSON."""
    return f"sha256:{hashlib.sha256(stable_json_dumps(data).encode()).hexdigest()}"
```

### Fail-Loud Validation

```python
# CORRECT - Explicit tracking of failures
def validate_tickers(tickers: list[str]) -> ValidationResult:
    valid = []
    invalid = {}
    for ticker in tickers:
        is_valid, reason = is_valid_ticker(ticker)
        if is_valid:
            valid.append(ticker)
        else:
            invalid[ticker] = reason  # Track why it failed
    return ValidationResult(valid=valid, invalid=invalid)

# WRONG - Silent failures
def validate_tickers(tickers):
    return [t for t in tickers if is_valid_ticker(t)]  # Lost error info
```

### Type Annotations

All functions should have type hints. Use `Protocol` for duck typing:

```python
from typing import Protocol, Dict, List
from datetime import date

class ClinicalTrialsProvider(Protocol):
    def load_trials(self, as_of_date: date) -> Dict[str, List[TrialRow]]: ...
```

### Enums for Constants

```python
from enum import Enum

class Severity(Enum):
    NONE = "none"     # No issue
    SEV1 = "sev1"     # 10% penalty
    SEV2 = "sev2"     # 50% penalty (soft gate)
    SEV3 = "sev3"     # Hard gate (excluded)

class StatusGate(Enum):
    ACTIVE = "active"
    EXCLUDED_SHELL = "excluded_shell"
    EXCLUDED_DELISTED = "excluded_delisted"
    EXCLUDED_ACQUIRED = "excluded_acquired"
```

## Testing Patterns

### Standard Test Fixtures

Tests use shared fixtures from `tests/conftest.py`:

```python
def test_financial_scoring(as_of_date, sample_financial_records):
    """as_of_date is always date(2026, 1, 15) for deterministic tests."""
    result = compute_financial_score(sample_financial_records, as_of_date)
    assert_score_bounded(result.score)
```

### Golden Output Tests

Ensure determinism with hash-based regression tests:

```python
def test_catalyst_v2_determinism():
    """Same inputs produce byte-identical outputs."""
    output1 = compute_module_3_catalyst(...)
    output2 = compute_module_3_catalyst(...)

    assert compute_hash(output1) == compute_hash(output2)
```

### Test Utilities

```python
from tests.conftest import assert_decimal_equal, assert_score_bounded

assert_decimal_equal(actual, expected, precision=2)
assert_score_bounded(score, min_val=Decimal("0"), max_val=Decimal("100"))
```

## Running the Pipeline

### Standalone Module Testing

```bash
python module_3_catalyst.py \
  --as-of-date 2026-01-15 \
  --trial-records production_data/trial_records.json \
  --state-dir production_data/ctgov_state \
  --universe production_data/universe.json \
  --output-dir production_data
```

### Full Pipeline

```bash
python run_screen.py \
  --as-of-date 2026-01-15 \
  --data-dir production_data \
  --output screening_results.json
```

## Key Design Decisions

### No External API Calls in Core Modules

Core scoring modules use only stdlib. Data fetching happens in separate collectors/providers.

### Governance Metadata in Every Output

```python
output = {
    "_governance": {
        "run_id": "abc123...",
        "score_version": "v1",
        "schema_version": "1.0.0",
        "parameters_hash": "sha256:...",
        "pit_cutoff": "2026-01-14",
    },
    "results": {...}
}
```

### Score Normalization

All scores are normalized to 0-100 range using rank-based normalization:

```python
score = min(max(raw_score, Decimal("0")), Decimal("100"))
```

## Common Gotchas

1. **Never use `datetime.now()`** - Pass explicit `as_of_date` parameter
2. **Never use `float` for money** - Use `Decimal` with string initialization
3. **Always check PIT admissibility** before using data
4. **Hash outputs for reproducibility** - Use `stable_json_dumps()` for deterministic serialization
5. **Don't silently drop invalid data** - Track and report validation failures

## File Naming Conventions

- `module_N_*.py` - Pipeline modules (1-5)
- `*_engine.py` - Enhancement/scoring engines
- `*_adapter.py` - Data format adapters
- `*_provider.py` - Data source providers
- `test_*.py` - Test files (in `tests/`)

## Dependencies

**Core (zero dependencies):**
- Python 3.10+ stdlib only

**Development:**
- pytest >= 7.0.0
- pytest-cov >= 4.0.0

**Data analysis (optional):**
- pandas >= 2.0.0
- numpy >= 1.24.0

## Scoring Weights (Module 5 Composite)

Default weighting for final ranking:
- 40%: Clinical development (Module 4)
- 25%: Financial health (Module 2)
- 15%: Catalyst momentum (Module 3)
- 20%: Other factors (market, momentum, regime)

## Event Types (Module 3 Catalyst)

| Event Type | Description | Impact |
|------------|-------------|--------|
| CT_STATUS_SEVERE_NEG | Trial stopped | 3 |
| CT_STATUS_DOWNGRADE | Status worsened | 1-3 |
| CT_STATUS_UPGRADE | Status improved | 1-3 |
| CT_TIMELINE_PUSHOUT | Completion delayed | 1-3 |
| CT_TIMELINE_PULLIN | Completion accelerated | 1-3 |
| CT_DATE_CONFIRMED_ACTUAL | Date confirmed | 1 |
| CT_RESULTS_POSTED | Results published | 1 |

## Important Files Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package configuration, dependencies |
| `tests/conftest.py` | Shared test fixtures (~500 lines) |
| `common/pit_enforcement.py` | PIT cutoff utilities |
| `common/provenance.py` | Audit trail and hashing |
| `common/integration_contracts.py` | Module boundary types and schema validation |
| `run_screen.py` | Main pipeline orchestrator |
