# CLAUDE.md - AI Assistant Guide for biotech-screener

## Project Overview

**biotech-screener** is a deterministic, point-in-time (PIT) safe biotech investment screening system developed by Wake Robin Capital Management. It implements a multi-module pipeline that combines financial health analysis, clinical trial catalysts, and clinical development metrics to produce ranked investment opportunities.

**Key Principles:**
- **Determinism**: Same inputs always produce byte-identical outputs (no `random`, no `datetime.now()`)
- **Point-in-Time Safety**: Prevents lookahead bias by enforcing strict PIT cutoffs (`source_date <= as_of_date - 1`)
- **Fail-Closed**: Validates data and stops on errors rather than gracefully degrading
- **Audit Trail**: Complete machine-readable governance metadata for reproducibility
- **Decimal-Only Arithmetic**: All financial calculations use `Decimal` (never floats)
- **Stdlib-Only Core**: Zero external dependencies in scoring modules

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run the full screening pipeline
python run_screen.py --as-of-date 2026-01-15 --data-dir production_data --output results.json
```

## Project Structure

```
biotech-screener/
├── Core Pipeline Modules
│   ├── module_1_universe.py           # Universe filtering & classification
│   ├── module_2_financial.py          # Financial health scoring
│   ├── module_2_financial_v2.py       # Enhanced financial metrics
│   ├── module_3_catalyst.py           # CT.gov catalyst event detection
│   ├── module_3_scoring.py            # Catalyst event scoring
│   ├── module_3_scoring_v2.py         # Enhanced catalyst scoring
│   ├── module_4_clinical_dev.py       # Clinical development metrics
│   ├── module_4_clinical_dev_v2.py    # Enhanced clinical scoring
│   ├── module_5_composite_v3.py       # Composite ranking (current)
│   ├── module_5_composite_with_defensive.py  # With defensive overlays
│   ├── module_5_scoring_v3.py         # Scoring types & utilities
│   └── module_5_diagnostics_v3.py     # V3 diagnostics & observability
│
├── Enhancement Engines
│   ├── pos_engine.py                  # Probability of Success (indication-specific)
│   ├── pos_prior_engine.py            # Prior PoS engine
│   ├── short_interest_engine.py       # Squeeze potential & crowding
│   ├── regime_engine.py               # Market regime classification
│   ├── dilution_risk_engine.py        # Forced-raise probability scoring
│   ├── competitive_pressure_engine.py # Competitive landscape analysis
│   ├── time_decay_scoring.py          # Time-based signal decay
│   ├── timeline_slippage_engine.py    # Trial timeline delay detection
│   ├── liquidity_scoring.py           # Trading liquidity assessment
│   ├── momentum_health_monitor.py     # IC-based momentum signal health
│   ├── indication_mapper.py           # Trial condition → indication mapping
│   └── manager_momentum_v1.py         # Institutional manager momentum
│
├── Data Adapters & Providers
│   ├── ctgov_adapter.py               # ClinicalTrials.gov format conversion
│   ├── accuracy_enhancements_adapter.py  # Accuracy improvement layer
│   ├── defensive_overlay_adapter.py   # Correlation sanitization & position sizing
│   ├── edgar_13f_extractor.py         # SEC 13F filing parsing
│   ├── finra_short_interest_feed.py   # FINRA short interest data
│   ├── cusip_mapper.py                # CUSIP ↔ ticker mapping
│   └── cusip_resolver.py              # CUSIP ↔ ticker resolution
│
├── common/                            # Core utilities (~450KB)
│   ├── __init__.py                    # Centralized exports (~200 items)
│   ├── accuracy_improvements.py       # Accuracy enhancement utilities
│   ├── constants.py                   # Global constants & thresholds
│   ├── data_consistency.py            # Cross-module data consistency
│   ├── data_integration_contracts.py  # Data integration schemas
│   ├── data_quality.py                # Data quality gates
│   ├── date_utils.py                  # ISO date handling & validation
│   ├── forward_looking_separation.py  # Forward-looking data isolation
│   ├── hash_utils.py                  # Deterministic SHA256 hashing
│   ├── input_validation.py            # Pipeline input validation
│   ├── integration_contracts.py       # Module boundary types & validation
│   ├── logging_config.py              # Structured logging configuration
│   ├── null_safety.py                 # Defensive null handling
│   ├── pit_enforcement.py             # PIT cutoff computation
│   ├── production_hardening.py        # Production stability utilities
│   ├── provenance.py                  # Audit trails & hashing
│   ├── random_state.py                # Determinism enforcement
│   ├── robustness.py                  # Data staleness, consistency checks
│   ├── robustness_extended.py         # Extended robustness utilities
│   ├── run_manifest.py                # Run metadata tracking
│   ├── run_summary.py                 # Run summary utilities
│   ├── schema_validation.py           # JSON schema validation
│   ├── score_utils.py                 # Score clamping & normalization
│   ├── staleness_gates.py             # Data staleness thresholds
│   └── types.py                       # Severity, StatusGate enums
│
├── src/                               # Package infrastructure
│   ├── common/hash_utils.py           # Re-exported hashing
│   ├── modules/
│   │   ├── ic_enhancements.py         # IC optimization layer (V1.1.0)
│   │   ├── ic_pit_validation.py       # PIT safety for IC lookback
│   │   └── intelligent_governance.py  # Smart governance
│   ├── providers/
│   │   ├── protocols.py               # Provider interfaces
│   │   ├── aact_provider.py           # AACT clinical trials provider
│   │   └── stub_provider.py           # Test stub provider
│   ├── validators/ticker_validator.py # Ticker validation
│   ├── history/snapshots.py           # 13F holdings snapshots
│   └── snapshot_generator.py          # Universe snapshot generation
│
├── governance/                        # Audit & compliance
│   ├── audit_log.py                   # JSONL audit trail writer
│   ├── canonical_json.py              # Deterministic JSON serialization
│   ├── hashing.py                     # Content hashing
│   ├── mapping_loader.py              # Configuration loading
│   ├── output_writer.py               # Output file writing
│   ├── params_loader.py               # Parameter loading
│   ├── pipeline_runner.py             # Pipeline execution
│   ├── run_id.py                      # Run ID generation
│   └── schema_registry.py             # Schema registration
│
├── backtest/                          # Backtesting framework
│   ├── __init__.py                    # Module exports
│   ├── compare_module5_versions.py    # Version comparison
│   ├── data_readiness.py              # Data readiness checks
│   ├── factor_attribution.py          # Factor-level attribution analysis
│   ├── ic_measurement.py              # Information Coefficient measurement
│   ├── metrics.py                     # Backtest metrics (IC, returns)
│   ├── portfolio_metrics.py           # Portfolio-level metrics
│   ├── returns_provider.py            # Historical returns
│   ├── sanity_metrics.py              # Sanity check metrics
│   ├── sharadar_provider.py           # Sharadar data provider
│   └── stability_attribution.py       # Attribution analysis
│
├── validation/                        # Validation framework
│   ├── validate_momentum_signal.py    # Momentum IC validation
│   └── validate_production_momentum.py # Production momentum check
│
├── tests/                             # Test suite (101 files, 55k+ lines)
│   ├── conftest.py                    # Shared fixtures (~500 lines)
│   ├── integration/                   # End-to-end pipeline tests
│   │   ├── test_data_enrichment_integration.py
│   │   ├── test_pipeline_robustness.py
│   │   ├── test_run_screen.py
│   │   └── test_snapshot_integration.py
│   ├── providers/                     # Data provider tests
│   └── test_*.py                      # Module-specific tests (97 files)
│
├── data/                              # Reference data & caches
│   ├── 13f_cache/                     # 13F filing cache
│   ├── aact_snapshots/                # ClinicalTrials.gov snapshots
│   ├── cache/                         # General data cache
│   ├── universe/                      # Universe snapshots
│   ├── pos_benchmarks_bio_2011_2020_v1.json  # PoS reference data
│   ├── indication_mapping.json        # Indication category mapping
│   ├── cusip_cache.json               # CUSIP resolution cache
│   ├── market_snapshot.json           # Market state snapshot
│   ├── short_interest.json            # Reference short interest
│   ├── daily_prices.csv               # Historical daily prices
│   └── trial_mapping.csv              # Trial-to-ticker mapping
│
├── production_data/                   # Production run data
│   ├── universe.json                  # Investable universe (~1MB)
│   ├── financial_records.json         # Financial metrics
│   ├── financial_data.json            # Extended financial data (~200KB)
│   ├── trial_records.json             # Clinical trials (~5.8MB, largest)
│   ├── market_data.json               # Market metrics (~307KB)
│   ├── short_interest.json            # Short interest data (~85KB)
│   ├── holdings_snapshots.json        # 13F snapshots (~363KB)
│   ├── catalyst_events_*.json         # Catalyst events per date
│   ├── run_log_*.json                 # Run logs per date
│   ├── screening_results_*.json       # Screening outputs
│   ├── ctgov_state/                   # State management
│   └── params_archive/                # Parameter snapshots
│
├── Pipeline Orchestration
│   ├── run_screen.py                  # Main pipeline orchestrator
│   ├── run_backtest.py                # Backtest orchestrator
│   ├── event_detector.py              # Catalyst event classification
│   ├── catalyst_summary.py            # Event aggregation & scoring
│   └── state_management.py            # JSONL state snapshots
│
├── patches/                           # Hotfix patches
│   ├── patch_001_pit_safe_ic_validation.py
│   ├── patch_002_deterministic_collection.py
│   ├── patch_003_schema_validation.py
│   └── patch_004_exception_handling.py
│
└── Configuration
    ├── pyproject.toml                 # Package configuration
    ├── config.yml                     # Pipeline configuration (v1.4.0)
    └── config/v3_production_integration.py  # V3 integration config
```

## Core Modules (Pipeline)

| Module | File | Purpose |
|--------|------|---------|
| **Module 1** | `module_1_universe.py` | Universe filtering, status gates, shell company detection |
| **Module 2** | `module_2_financial.py` | Financial health scoring (burn rate, dilution, liquidity) |
| **Module 3** | `module_3_catalyst.py` | CT.gov catalyst event detection and scoring |
| **Module 4** | `module_4_clinical_dev_v2.py` | Clinical development scoring with PoS integration |
| **Module 5** | `module_5_composite_with_defensive.py` | Composite ranking with defensive overlays |

## Enhancement Engines

Strategic scoring components that augment the core pipeline:

| Engine | File | Purpose | Version |
|--------|------|---------|---------|
| **Probability of Success** | `pos_engine.py` | Indication-specific trial success rates (BIO benchmarks) | 1.2.0 |
| **Short Interest** | `short_interest_engine.py` | Squeeze potential & crowding detection | 1.0.0 |
| **Market Regime** | `regime_engine.py` | BULL/BEAR/VOLATILITY classification | 1.0.0 |
| **Dilution Risk** | `dilution_risk_engine.py` | Forced-raise probability before catalyst | 1.0.0 |
| **Competitive Pressure** | `competitive_pressure_engine.py` | Competitive landscape analysis | - |
| **Timeline Slippage** | `timeline_slippage_engine.py` | Trial timeline delay detection | - |
| **Time Decay** | `time_decay_scoring.py` | Signal aging & decay modeling | - |
| **Liquidity Scoring** | `liquidity_scoring.py` | Trading liquidity assessment | - |
| **Momentum Health** | `momentum_health_monitor.py` | IC monitoring & signal health checks | - |
| **Indication Mapper** | `indication_mapper.py` | Trial conditions → indication categories | 2.0.0 |
| **Manager Momentum** | `manager_momentum_v1.py` | Institutional 13F position tracking | 1.0.0 |

### Dilution Risk Engine

```python
from dilution_risk_engine import DilutionRiskEngine

engine = DilutionRiskEngine()
result = engine.calculate_dilution_risk(
    ticker="ACME",
    quarterly_cash=Decimal("100000000"),
    quarterly_burn=Decimal("-15000000"),
    next_catalyst_date="2026-07-15",
    market_cap=Decimal("500000000"),
    as_of_date=date(2026, 1, 15)
)
# result["dilution_risk_score"]  # 0.00 - 1.00
# result["risk_bucket"]          # NO_RISK, LOW_RISK, MEDIUM_RISK, HIGH_RISK
```

### Regime Engine

```python
from regime_engine import RegimeEngine

engine = RegimeEngine()
regime = engine.classify_regime(
    vix_level=Decimal("22.5"),
    xbi_momentum=Decimal("0.05"),
    as_of_date=date(2026, 1, 15)
)
# regime.state  # "BULL", "BEAR", "VOLATILITY"
```

## IC-Based Enhancements

The `src/modules/ic_enhancements.py` module (~98KB) provides advanced signal processing for improved Information Coefficient (IC):

**Features:**
- Adaptive weight learning (historical IC optimization)
- Non-linear signal interactions (cross-factor synergies/penalties)
- Peer-relative valuation signal
- Catalyst signal decay (time-based IC modeling)
- Price momentum signal (relative strength)
- Shrinkage normalization (Bayesian cohort adjustment)
- Smart money signal (13F position changes with tier weighting)
- Volatility-adjusted scoring
- Regime-adaptive component selection

**Usage:**
```python
from src.modules.ic_enhancements import (
    compute_momentum_score,
    compute_smart_money_signal,
    apply_shrinkage_normalization,
)

# Momentum scoring with volatility adjustment
momentum = compute_momentum_score(
    alpha=Decimal("0.05"),
    volatility=Decimal("0.45"),
    confidence=Decimal("0.8")
)
```

### IC Measurement Framework

The `backtest/ic_measurement.py` module (~65KB) provides comprehensive IC analysis:

```python
from backtest.ic_measurement import (
    compute_information_coefficient,
    compute_ic_by_cohort,
    compute_factor_ic_decomposition,
)

# Compute overall IC
ic_result = compute_information_coefficient(
    predictions=score_predictions,
    actuals=forward_returns,
    method="spearman"
)

# Decompose by factor
factor_ics = compute_factor_ic_decomposition(
    predictions=predictions,
    actuals=actuals,
    factors=["clinical", "financial", "catalyst"]
)
```

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

Module 5 (Composite)  ← inputs: ALL module results + enhancement signals
    ↓ outputs: ranked_securities[], excluded_securities[], position_sizes[]
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

## Governance Framework

The `governance/` directory provides audit trail and compliance infrastructure:

### Audit Logging

```python
from governance.audit_log import AuditLogWriter, AuditStage, AuditStatus

writer = AuditLogWriter(output_path)
writer.log(
    stage=AuditStage.SCORE,
    status=AuditStatus.OK,
    inputs={"universe_hash": "sha256:..."},
    outputs={"scores_hash": "sha256:..."},
    version="1.0.0"
)
```

### Audit Stages

| Stage | Description |
|-------|-------------|
| `INIT` | Pipeline initialization |
| `LOAD` | Data loading |
| `ADAPT` | Data transformation/adaptation |
| `FEATURES` | Feature engineering |
| `RISK` | Risk calculation |
| `SCORE` | Scoring execution |
| `REPORT` | Report generation |
| `FINAL` | Pipeline completion |

### Canonical JSON Serialization

```python
from governance.canonical_json import canonical_dumps

# Deterministic JSON output (sorted keys, consistent formatting)
json_str = canonical_dumps(data)
```

## Robustness Utilities

The `common/robustness.py` module provides production-grade utilities:

### Data Freshness Validation

```python
from common.robustness import validate_data_freshness, DataFreshnessConfig

config = DataFreshnessConfig(
    warning_threshold_days=3,
    critical_threshold_days=7
)
result = validate_data_freshness(
    records=financial_records,
    as_of_date=date(2026, 1, 15),
    config=config
)
# result.is_fresh, result.stale_count, result.severity
```

### Retry with Exponential Backoff

```python
from common.robustness import retry_with_backoff, RetryConfig

config = RetryConfig(
    max_retries=4,
    base_delay_seconds=2,
    max_delay_seconds=16
)

@retry_with_backoff(config)
def fetch_data():
    return api.get_data()
```

### Memory Guards

```python
from common.robustness import chunk_universe, estimate_memory_usage

# Process large universe in chunks
for chunk in chunk_universe(large_ticker_list, chunk_size=100):
    process_tickers(chunk)
```

### Correlated Logging

```python
from common.robustness import CorrelatedLogger, with_correlation_id

@with_correlation_id
def process_pipeline():
    logger = CorrelatedLogger("pipeline")
    logger.info("Processing started")  # Includes correlation_id
```

## State Management

Trial state snapshots use JSONL format with sorted keys for stable diffs:

```python
from state_management import StateSnapshot, StateManager

# Load snapshot
snapshot = StateManager.load_snapshot(state_dir, as_of_date)

# Access records (sorted by ticker, nct_id)
for record in snapshot.records:
    print(record.ticker, record.nct_id, record.overall_status)

# Binary search for specific record
record = snapshot.get_record("ACME", "NCT12345678")
```

## Backtest Framework

The `backtest/` directory provides tools for historical validation:

| Module | Purpose |
|--------|---------|
| `metrics.py` | IC calculation, returns analysis, rank correlation |
| `ic_measurement.py` | Comprehensive IC measurement framework |
| `factor_attribution.py` | Factor-level attribution analysis |
| `portfolio_metrics.py` | Portfolio-level metrics (Sharpe, turnover, etc.) |
| `sanity_metrics.py` | Sanity checks (coverage, turnover, concentration) |
| `stability_attribution.py` | Attribution analysis for score changes |
| `data_readiness.py` | Data availability checks for backtest dates |
| `returns_provider.py` | Historical price returns |
| `sharadar_provider.py` | Sharadar fundamentals data provider |
| `compare_module5_versions.py` | A/B comparison between Module 5 versions |

### Running Backtests

```bash
python run_backtest.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --data-dir production_data \
  --output-dir backtest_results
```

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

### Test Coverage by Category

| Category | Test Files | Coverage |
|----------|------------|----------|
| Module Tests | 20 files | Core pipeline modules (M1-M5) |
| Enhancement Engine Tests | 15 files | All scoring engines |
| Integration Tests | 5 files | End-to-end pipeline tests |
| Utility Tests | 25 files | Common utilities |
| Provider Tests | 5 files | Data providers |
| Governance Tests | 4 files | Audit framework |
| Regression Tests | 15+ files | Golden baseline comparisons |
| Robustness Tests | 12 files | Error handling, edge cases |

**Total: 101 test files (~55,000 lines of test code)**

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

### With Defensive Overlays

```bash
python run_screen.py \
  --as-of-date 2026-01-15 \
  --data-dir production_data \
  --enable-defensive-overlay \
  --output screening_results.json
```

## Configuration Management

### Pipeline Configuration (`config.yml` v1.4.0)

```yaml
version: "1.4.0"

paths:
  data_dir: "production_data"
  input_files:
    universe: "universe.json"
    financial_records: "financial_records.json"
    trial_records: "trial_records.json"
    market_data: "market_data.json"

module_5:
  weights:
    clinical: 0.40
    financial: 0.35
    catalyst: 0.25
  position_sizing:
    max_positions: 60
    max_single_position: 0.10
  use_v3_scoring: true

enhancements:
  pos_engine:
    enabled: false
  regime_engine:
    enabled: false
  short_interest:
    enabled: false

determinism:
  force_deterministic_timestamps: true
  sort_output_keys: true
  include_content_hashes: true
```

### Parameter Archives

Historical parameter snapshots are stored in `production_data/params_archive/` for reproducibility and sensitivity analysis.

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

### Content-Addressable Snapshots

State management uses content hashing for snapshot identification, enabling efficient change detection and audit trails.

## Common Gotchas

1. **Never use `datetime.now()`** - Pass explicit `as_of_date` parameter
2. **Never use `float` for money** - Use `Decimal` with string initialization
3. **Always check PIT admissibility** before using data
4. **Hash outputs for reproducibility** - Use `stable_json_dumps()` for deterministic serialization
5. **Don't silently drop invalid data** - Track and report validation failures
6. **Never use `random` module** - Use explicit seed or deterministic alternatives
7. **Always validate module handoffs** - Use `validate_pipeline_handoff()` between modules
8. **Handle both legacy and new field names** - Use `extract_financial_score()` for backwards compatibility
9. **Include governance metadata** - Every output must have `_governance` block
10. **Test determinism** - Same inputs must produce byte-identical outputs
11. **Use retry with backoff for API calls** - Use `retry_with_backoff()` from `common/robustness.py`
12. **Validate data freshness** - Use `validate_data_freshness()` before processing
13. **Check schema versions** - Use `check_schema_version()` for backwards compatibility

## File Naming Conventions

- `module_N_*.py` - Pipeline modules (1-5)
- `*_engine.py` - Enhancement/scoring engines
- `*_adapter.py` - Data format adapters
- `*_provider.py` - Data source providers
- `*_validator.py` - Validation utilities
- `test_*.py` - Test files (in `tests/`)
- `*_v2.py`, `*_v3.py` - Version-specific implementations

## Dependencies

**Core (zero dependencies):**
- Python 3.10+ stdlib only

**Development:**
- pytest >= 7.0.0
- pytest-cov >= 4.0.0

**Data analysis (optional):**
- pandas >= 2.0.0
- numpy >= 1.24.0

**Optimization (optional):**
- scipy >= 1.11.0
- numpy >= 1.24.0

## Scoring Weights (Module 5 Composite)

Default weighting for final ranking (v1.4.0):
- 40%: Clinical development (Module 4)
- 35%: Financial health (Module 2)
- 25%: Catalyst momentum (Module 3)

**Position Sizing Defaults:**
- Maximum positions: 60
- Maximum single position: 10%
- Minimum single position: 0.5%
- Target weight sum: 100% (no cash reserve)

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
| `config.yml` | Pipeline configuration (v1.4.0) |
| `run_screen.py` | Main pipeline orchestrator (~108KB, 2800+ lines) |
| `run_backtest.py` | Backtest orchestrator (~52KB) |
| `tests/conftest.py` | Shared test fixtures (~500 lines) |
| `common/pit_enforcement.py` | PIT cutoff utilities |
| `common/provenance.py` | Audit trail and hashing |
| `common/integration_contracts.py` | Module boundary types and schema validation (~45KB) |
| `common/robustness.py` | Data staleness, consistency checks, retry logic |
| `common/data_quality.py` | Data quality gates |
| `common/production_hardening.py` | Production stability utilities (~52KB) |
| `common/staleness_gates.py` | Data freshness thresholds |
| `common/schema_validation.py` | JSON schema validation |
| `governance/audit_log.py` | JSONL audit trail writer |
| `state_management.py` | Trial state snapshot management |
| `src/modules/ic_enhancements.py` | IC optimization layer (~98KB) |
| `dilution_risk_engine.py` | Forced-raise probability scoring (~30KB) |
| `backtest/metrics.py` | IC and backtest metrics |
| `backtest/ic_measurement.py` | Comprehensive IC measurement framework (~65KB) |

## Recent Changes

### v1.4.0 (January 2026 - Latest)

- **Data Flow Improvements**: Fixed data flow gaps and enabled enhancements by default
- **13F Parsing Fixes**: Fixed 13F parsing and corrected all elite manager CIKs
- **YTD-Corrected Runway**: Fixed YTD date field passthrough in Module 2 wrapper
- **Correlation Data**: Achieved 100% coverage on correlation data collection
- **Cash Target Removal**: Removed 10% cash target; weights now sum to 100%
- **Production Hardening**: Added `common/production_hardening.py` for stability
- **Schema Validation**: Added `common/schema_validation.py` for JSON schema validation
- **Staleness Gates**: Added `common/staleness_gates.py` for data freshness thresholds
- **Hotfix Patches**: Added `patches/` directory with 4 production hotfix patches

### v1.1.0 (January 2026)

- **IC Enhancements Module**: Added `src/modules/ic_enhancements.py` with smart money signal, tier weighting, volatility adjustment
- **Dilution Risk Engine**: New `dilution_risk_engine.py` for forced-raise probability scoring
- **Indication Mapper v2.0**: Enhanced condition→indication mapping
- **Comprehensive Test Coverage**: Added tests for 11 previously untested modules (now 101 total)
- **Data Integration Fixes**: Fixed pipeline module data integration bugs
- **Determinism Improvements**: Eliminated `datetime.now()` violations
- **Module 5 Diagnostics**: Added `module_5_diagnostics_v3.py` for observability
