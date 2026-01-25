# Production Code Review: Biotech Screening System

**Review Date**: 2026-01-25
**Reviewer**: Claude (Production Engineer + Quant Reviewer)
**Codebase Version**: 1.5.0
**Scope**: Full pipeline review (296 Python files, ~75,000 LOC)

---

## A. Architecture Summary

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ universe.json    │  │ financial_       │  │ trial_records    │              │
│  │ (investable      │  │ records.json     │  │ .json            │              │
│  │  universe)       │  │ (cash, burn)     │  │ (CT.gov trials)  │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
│           │                      │                      │                       │
│  ┌────────┴─────────┐  ┌────────┴─────────┐  ┌────────┴─────────┐              │
│  │ market_data.json │  │ short_interest   │  │ holdings_        │              │
│  │ (price, volume)  │  │ .json            │  │ snapshots.json   │              │
│  └────────┬─────────┘  └──────────────────┘  └──────────────────┘              │
│           │                                                                     │
└───────────┼─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE ORCHESTRATOR                               │
│                            (run_screen.py - 1,851 LOC)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                    as_of_date (REQUIRED, no defaults)                       ││
│  │                    PIT Cutoff = as_of_date - 1 day                         ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORE PIPELINE MODULES                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐        │
│  │   MODULE 1       │     │   MODULE 2       │     │   MODULE 3       │        │
│  │   Universe       │────▶│   Financial      │────▶│   Catalyst       │        │
│  │   Filtering      │     │   Health         │     │   Detection      │        │
│  │                  │     │                  │     │   (Delta-based)  │        │
│  │  • Shell detect  │     │  • Runway calc   │     │  • State compare │        │
│  │  • Status gates  │     │  • Dilution risk │     │  • Event scoring │        │
│  │  • MCap filter   │     │  • Liquidity     │     │  • Calendar cats │        │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘        │
│           │                        │                        │                   │
│           ▼                        ▼                        ▼                   │
│  active_securities[]      financial_score[]         catalyst_summaries{}       │
│  excluded_securities[]    severity, flags           events, scores              │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐                                 │
│  │   MODULE 4       │     │   ENHANCEMENT    │                                 │
│  │   Clinical       │     │   LAYER          │                                 │
│  │   Development    │     │   (Optional)     │                                 │
│  │                  │     │                  │                                 │
│  │  • Phase scoring │     │  • PoS Engine    │                                 │
│  │  • Trial design  │     │  • Regime Detect │                                 │
│  │  • Indication    │     │  • SI Signals    │                                 │
│  │  • PIT filtering │     │  • Dilution Risk │                                 │
│  └────────┬─────────┘     └────────┬─────────┘                                 │
│           │                        │                                            │
│           └────────────┬───────────┘                                            │
│                        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                           MODULE 5                                          ││
│  │                   Composite Ranking + Defensive                             ││
│  │  ┌─────────────────────────────────────────────────────────────────┐       ││
│  │  │  V3 Scorer (IC-Enhanced)                                         │       ││
│  │  │  • Catalyst decay (time-based)                                   │       ││
│  │  │  • Price momentum (60d alpha vs XBI)                             │       ││
│  │  │  • Smart money (13F tier-weighted)                               │       ││
│  │  │  • Volatility adjustment                                         │       ││
│  │  │  • Shrinkage normalization                                       │       ││
│  │  │  • Sanity override (fallback to V2)                              │       ││
│  │  └─────────────────────────────────────────────────────────────────┘       ││
│  │  ┌─────────────────────────────────────────────────────────────────┐       ││
│  │  │  Defensive Overlay                                               │       ││
│  │  │  • Correlation sanitization                                      │       ││
│  │  │  • Inverse-volatility position sizing                            │       ││
│  │  │  • Top-N selection                                               │       ││
│  │  └─────────────────────────────────────────────────────────────────┘       ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT + GOVERNANCE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐           │
│  │ results.json      │  │ audit.jsonl       │  │ checkpoints/      │           │
│  │ • ranked_         │  │ • run_id          │  │ • module_1.json   │           │
│  │   securities[]    │  │ • params_hash     │  │ • module_2.json   │           │
│  │ • excluded_       │  │ • input_hashes    │  │ • ...             │           │
│  │   securities[]    │  │ • content_hash    │  │                   │           │
│  │ • position_sizes  │  │                   │  │                   │           │
│  │ • provenance      │  │                   │  │                   │           │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### State Management

| Component | Storage | Purpose |
|-----------|---------|---------|
| Trial Snapshots | `ctgov_state/{hash}/` | JSONL snapshots for delta detection |
| Checkpoints | `checkpoints/` | Resumable module outputs |
| Audit Trail | `audit.jsonl` | JSONL with parameter hashes |
| Run Manifests | Embedded in output | Full provenance metadata |

### Key Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Orchestrator | `run_screen.py` | 1,851 | Pipeline coordination |
| M1 Universe | `module_1_universe.py` | ~400 | Universe filtering |
| M2 Financial | `module_2_financial.py` | 992 | Financial health |
| M3 Catalyst | `module_3_catalyst.py` | 1,136 | Event detection |
| M4 Clinical | `module_4_clinical_dev_v2.py` | ~600 | Clinical scoring |
| M5 Composite | `module_5_composite_v3.py` | ~1,200 | IC-enhanced ranking |
| PIT Enforcement | `common/pit_enforcement.py` | 55 | Lookahead prevention |
| Data Quality | `common/data_quality.py` | 628 | Quality gates |

---

## B. Findings List (by Severity)

### CRITICAL (Immediate Fix Required)

#### C1. Look-Ahead Bias in IC Validation
**File**: `validation/validate_production_momentum.py:174-183`
**Impact**: IC validation uses FUTURE returns to validate CURRENT signals - textbook lookahead bias.

```python
# PROBLEM: Uses future returns for IC calculation
forward_date = self.calendar.add_trading_days(calc_date, forward_horizon_days)  # Future!
ret = (prices_df[ticker].loc[forward_date] /   # Future price
       prices_df[ticker].loc[calc_date]) - 1   # Current price
```

**Risk**: False confidence in signal quality. Production IC may be significantly lower than validated.

#### C2. datetime.now() in Data Collectors
**Files**: Multiple collector files
- `collect_market_data.py:80` - `end_date = datetime.now()`
- `collect_market_data.py:130` - `"collected_at": date.today().isoformat()`
- `extend_universe_yfinance.py:130,213`
- `wake_robin_data_pipeline/collectors/time_series_collector.py:34`

**Impact**: Non-deterministic data collection. Running the same script on different days produces different data even for the same `as_of_date`.

#### C3. Unvalidated JSON Loading at Critical Points
**File**: `module_3_catalyst.py:398-399`
```python
with open(trial_records_path) as f:
    trial_records_raw = json.load(f)  # No schema validation!
```

**Impact**: Malformed trial records could corrupt scoring silently.

### HIGH (Fix Within 1 Week)

#### H1. Silent Exception Swallowing
**File**: `defensive_overlay_adapter.py:47,89,126`
```python
except Exception:
    pass  # Swallows ALL exceptions silently
```

**Impact**: Hidden failures in financial calculations, hard to debug production issues.

#### H2. CUSIP Mapping Bug
**File**: `map_tickers_to_cusips.py:26`
```python
cusip = item.get('compositeFIGI', '')[-9:]  # Empty string if missing!
if cusip:  # Always passes since '' is falsy but this happens AFTER slice
    cusip_map[cusip] = {...}
```

**Impact**: Empty CUSIPs could pollute mapping table.

#### H3. Float/Decimal Mixing in Financial Calculations
**Files**: `module_2_financial_v2.py:1163-1217`
- Scores converted to float via `_to_float()` at output
- Intermediate calculations use Decimal
- Module 5 receives floats but may expect Decimals

**Impact**: Precision loss at module boundaries, potential type errors.

#### H4. as_of_date Defaulting to Today
**Files**:
- `audit_framework/orchestrator.py:71` - `as_of_date or datetime.now().date().isoformat()`
- `wake_robin_data_pipeline/collectors/time_series_collector.py:34` - `as_of = date.today()`

**Impact**: Data collected on different days will have different dates, breaking reproducibility.

#### H5. Array Access Without Length Check
**File**: `collect_market_data.py:94-95`
```python
hist['Close'].iloc[-1]   # No check if hist has >= 1 row
hist['Close'].iloc[-21]  # No check if hist has >= 21 rows
```

**Impact**: IndexError on tickers with insufficient history.

### MEDIUM (Fix Within 1 Month)

#### M1. Data Quality Gates Not Activated at Entry Points
**Files**: `validation/validate_data_integration.py:618-650`, `run_screen.py`

The comprehensive `DataQualityGates` class exists in `common/data_quality.py` but is not called at pipeline entry points. Data staleness and quality issues are not caught early.

#### M2. PIT Filter Not Applied Consistently
**Files**: Multiple files load data without checking `source_date <= pit_cutoff`

While `common/pit_enforcement.py` exists, many data loading paths don't use it:
- Financial records in `module_2_financial.py` wrapper (lines 870-880 do filter but only in legacy wrapper)
- Market data loading in collectors

#### M3. Missing Bounds Checks on Composite Scores
**Files**: `module_5_composite.py:470-480`
```python
cat_score_val = float(cat.score_blended)  # Could be outside 0-100
```

Component scores should be clamped before combination.

#### M4. Schema Validation at Module Boundaries
While `common/integration_contracts.py` provides validators, they're not consistently called:
- `validate_pipeline_handoff()` is used in `run_screen.py` but not in standalone module runs
- No validation on data at ingestion time

#### M5. Timestamp Metadata Non-Deterministic
**Files**: All collector outputs include `datetime.now()` timestamps

This makes output hashes unstable even for identical inputs.

### LOW (Technical Debt)

#### L1. Deprecated Fields Still in Use
- `summaries_legacy` in Module 3 output
- `financial_normalized` as alias for `financial_score`

#### L2. Test Fixtures Use date.today()
**File**: `tests/test_sec_13f.py:294,299,319,358`

Tests should use fixed dates from conftest.py.

#### L3. Magic Numbers Without Constants
Various threshold values scattered across modules without centralized configuration.

---

## C. Patch Set

### Patches Created

| Patch | File | Issue Fixed |
|-------|------|-------------|
| 001 | `patches/patch_001_pit_safe_ic_validation.py` | Look-ahead bias in IC validation |
| 002 | `patches/patch_002_deterministic_collection.py` | datetime.now() in collectors |
| 003 | `patches/patch_003_schema_validation.py` | Missing JSON schema validation |
| 004 | `patches/patch_004_exception_handling.py` | Silent exception swallowing |

### Patch 1: Fix IC Validation Look-Ahead Bias

**Severity**: CRITICAL
**File**: `patches/patch_001_pit_safe_ic_validation.py`

Provides `PITSafeICValidator` class that:
- Only uses historical IC data (no future returns)
- Implements walk-forward validation
- Enforces strict PIT cutoffs

### Patch 2: Deterministic Data Collection

**Severity**: CRITICAL
**File**: `patches/patch_002_deterministic_collection.py`

Provides:
- `DeterministicMarketDataCollector` that requires explicit as_of_date
- `@enforce_explicit_as_of_date` decorator
- `validate_collection_date()` function

### Patch 3: Schema Validation at Load Points

**Severity**: HIGH
**File**: `patches/patch_003_schema_validation.py`

Provides:
- `load_and_validate_trial_records()` with schema checking
- `load_and_validate_financial_records()`
- `safe_get()` and `safe_list_access()` for safe data access

### Patch 4: Exception Handling

**Severity**: HIGH
**File**: `patches/patch_004_exception_handling.py`

Provides:
- `ExceptionAccumulator` for batch error tracking
- `safe_decimal_convert()` and `safe_divide()`
- `@log_exception` and `@with_fallback` decorators

---

## D. Hardening Checklist

### Immediate (Day 1)

- [ ] **Apply Patch 001**: Replace IC validation with PIT-safe version
  - Risk: Current validation gives false confidence
  - Test: `pytest tests/test_patches.py::TestPITSafeICValidation -v`

- [ ] **Apply Patch 004**: Add exception handling to `defensive_overlay_adapter.py`
  - Replace `except Exception: pass` with `ExceptionAccumulator`
  - Risk: Silent failures are currently invisible

- [ ] **Add length checks**: `collect_market_data.py:94-95`
  ```python
  if len(hist) >= 21:
      returns_1m = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1)
  else:
      returns_1m = None
  ```

### Week 1

- [ ] **Apply Patch 003**: Add schema validation at data load points
  - `module_3_catalyst.py:398` - validate trial records
  - `run_screen.py` - validate all input files

- [ ] **Apply Patch 002**: Update collectors for determinism
  - Make `as_of_date` required parameter
  - Remove `datetime.now()` from metadata

- [ ] **Fix CUSIP mapping bug** in `map_tickers_to_cusips.py:26`:
  ```python
  figi = item.get('compositeFIGI')
  if figi and len(figi) >= 9:
      cusip = figi[-9:]
      cusip_map[cusip] = {...}
  ```

- [ ] **Add data quality gates** at pipeline entry:
  ```python
  gates = DataQualityGates()
  for record in financial_records:
      result = gates.validate_ticker_data(record["ticker"], record, as_of_date=as_of_date)
      if not result.passed:
          logger.warning(f"Quality gate failed: {result.warnings}")
  ```

### Week 2-4

- [ ] **Standardize Decimal/float handling**:
  - All module outputs should use consistent types
  - Add type guards at module boundaries

- [ ] **Add monitoring metrics**:
  - Error rates by module
  - Data staleness tracking
  - Score distribution statistics

- [ ] **Implement circuit breaker** for batch processing:
  ```python
  from common.data_quality import check_circuit_breaker

  result = check_circuit_breaker(total=len(records), failed=error_count)
  if result.tripped:
      raise CircuitBreakerError(result.message)
  ```

- [ ] **Remove deprecated code**:
  - `summaries_legacy` from Module 3
  - `financial_normalized` alias

---

## E. Model Accuracy & Methodology Improvements

### Feature Engineering Improvements

#### 1. Catalyst Timing Features
**Current Issue**: Catalyst dates are used but timing uncertainty isn't modeled.

**Recommendation**:
```python
# Add catalyst timing confidence
def compute_catalyst_timing_features(trial):
    # Historical slippage rate by phase/indication
    avg_slippage_days = get_historical_slippage(trial.phase, trial.indication)

    # Expected days to readout with uncertainty
    expected_days = (trial.expected_completion - as_of_date).days
    uncertainty = avg_slippage_days * 1.5  # Conservative

    return {
        "days_to_catalyst": expected_days,
        "catalyst_uncertainty_days": uncertainty,
        "catalyst_confidence": 1.0 / (1 + uncertainty / expected_days),
    }
```

#### 2. PoS Conditioning
**Current Issue**: PoS is static by indication/phase.

**Recommendation**: Condition on trial characteristics:
- Enrollment size relative to historical benchmarks
- Number of endpoints
- Prior trial results for same drug
- Company track record

#### 3. Short Interest Decay
**Current Issue**: Short interest signal doesn't account for crowding dynamics.

**Recommendation**:
```python
def compute_short_interest_signal(si_data, as_of_date):
    # Raw short interest
    si_pct = si_data.get("short_percent")

    # Days to cover (crowding indicator)
    dtc = si_data.get("days_to_cover")

    # Velocity (change in short interest)
    si_change_20d = si_data.get("si_change_20d")

    # Crowding risk: high SI + high DTC = squeeze potential
    crowding_score = si_pct * min(dtc / 5.0, 1.0)

    # But extreme crowding can reverse
    if crowding_score > 0.3:  # Very crowded
        crowding_score *= 0.7  # Reduce weight

    return crowding_score
```

### Validation Methodology

#### 1. Walk-Forward Validation
**Current Issue**: Validation may use future data inadvertently.

**Recommendation**: Implement strict walk-forward:
```python
def walk_forward_validation(
    model,
    data,
    train_window_months=24,
    test_window_months=1,
):
    """
    True out-of-sample validation.

    At each test date:
    1. Train on data from [test_date - 24mo, test_date)
    2. Predict on [test_date, test_date + 1mo)
    3. Measure IC on that month
    4. Move forward 1 month
    """
    results = []

    for test_date in test_dates:
        train_start = test_date - timedelta(days=train_window_months * 30)
        train_end = test_date - timedelta(days=1)

        # Train with ONLY historical data
        train_data = data[(data.date >= train_start) & (data.date <= train_end)]
        model.fit(train_data)

        # Test on future (but now historical) data
        test_data = data[(data.date >= test_date) &
                        (data.date < test_date + timedelta(days=30))]
        predictions = model.predict(test_data)

        # Measure IC
        ic = spearmanr(predictions, test_data.forward_returns)[0]
        results.append({"date": test_date, "ic": ic})

    return results
```

#### 2. Regime-Stratified Validation
**Current Issue**: IC may vary by market regime but this isn't measured.

**Recommendation**:
```python
def regime_stratified_ic(ic_series, regime_series):
    """Report IC by market regime."""
    regimes = ["BULL", "BEAR", "VOLATILITY"]

    for regime in regimes:
        mask = regime_series == regime
        regime_ic = ic_series[mask].mean()
        regime_std = ic_series[mask].std()
        regime_n = mask.sum()

        print(f"{regime}: IC={regime_ic:.3f} ± {regime_std:.3f} (n={regime_n})")
```

### Leakage Controls

#### 1. Event Window Isolation
**Recommendation**: Ensure features don't use data from event windows:
```python
def compute_features_with_event_isolation(ticker, as_of_date, events):
    """
    Compute features excluding data around known events.

    This prevents learning patterns that are only visible
    because we know an event happened.
    """
    # Get recent events for this ticker
    recent_events = [e for e in events
                    if e.ticker == ticker
                    and abs((as_of_date - e.date).days) < 10]

    if recent_events:
        # Use features from before the event window
        feature_cutoff = min(e.date for e in recent_events) - timedelta(days=5)
        return compute_features(ticker, feature_cutoff)

    return compute_features(ticker, as_of_date)
```

#### 2. Universe Survivorship Correction
**Current Issue**: Backtest universe may use current tickers (survivorship bias).

**Recommendation**:
```python
def get_historical_universe(as_of_date):
    """
    Get the universe as it existed on as_of_date.

    Must include:
    - Companies that were subsequently delisted
    - Companies that merged/acquired
    - Companies that went bankrupt
    """
    # Load from historical snapshot, not current universe
    snapshot_file = f"universe_snapshots/{as_of_date}.json"
    if not os.path.exists(snapshot_file):
        raise ValueError(f"No universe snapshot for {as_of_date}")

    return json.load(open(snapshot_file))
```

### Confidence Estimates

**Recommendation**: Add prediction confidence:
```python
def compute_score_with_confidence(ticker, features):
    # Base score
    score = model.predict(features)

    # Confidence based on:
    # 1. Data completeness
    data_coverage = sum(1 for f in features if f is not None) / len(features)

    # 2. Feature stability (how unusual are these features?)
    feature_zscore = compute_feature_zscore(features)
    feature_stability = 1.0 / (1 + np.abs(feature_zscore).mean())

    # 3. Historical accuracy for similar scores
    historical_accuracy = get_decile_accuracy(score)

    confidence = (data_coverage * 0.3 +
                 feature_stability * 0.3 +
                 historical_accuracy * 0.4)

    return {
        "score": score,
        "confidence": confidence,
        "data_coverage": data_coverage,
    }
```

---

## F. Test Plan

### Tests Created

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `tests/test_patches.py` | Verify patches fix identified issues | All 4 patches |

### Regression Tests Added

1. **PIT Safety Tests**: Verify no future data leakage
2. **Determinism Tests**: Same inputs produce same outputs
3. **Schema Validation Tests**: Malformed data is rejected
4. **Exception Handling Tests**: Errors are logged, not swallowed

### Suggested Future Tests

#### 1. End-to-End Pipeline Determinism
```python
def test_pipeline_determinism():
    """Same inputs on same as_of_date produce identical outputs."""
    result1 = run_screen(as_of_date='2026-01-15', data_dir='test_data')
    result2 = run_screen(as_of_date='2026-01-15', data_dir='test_data')

    # Hash comparison
    hash1 = compute_content_hash(result1)
    hash2 = compute_content_hash(result2)
    assert hash1 == hash2, "Pipeline is not deterministic"
```

#### 2. PIT Violation Detection
```python
def test_no_pit_violations():
    """No data should have source_date > pit_cutoff."""
    as_of_date = '2026-01-15'
    pit_cutoff = compute_pit_cutoff(as_of_date)

    for record in load_all_data():
        source_date = record.get('source_date')
        if source_date:
            assert source_date <= pit_cutoff, \
                f"PIT violation: {record['ticker']} has source_date={source_date}"
```

#### 3. Score Bounds Validation
```python
def test_scores_within_bounds():
    """All scores should be in [0, 100]."""
    result = run_screen(as_of_date='2026-01-15')

    for sec in result['ranked_securities']:
        score = float(sec['composite_score'])
        assert 0 <= score <= 100, f"{sec['ticker']} has invalid score {score}"
```

#### 4. Data Staleness Check
```python
def test_data_freshness():
    """Financial data should not be too stale."""
    as_of_date = date(2026, 1, 15)
    max_age_days = 90

    for record in load_financial_records():
        data_date = record.get('source_date')
        if data_date:
            age = (as_of_date - date.fromisoformat(data_date)).days
            assert age <= max_age_days, \
                f"{record['ticker']} has stale data ({age} days old)"
```

#### 5. IC Stability Test
```python
def test_ic_stability_across_regimes():
    """IC should be positive across all market regimes."""
    for regime in ['BULL', 'BEAR', 'VOLATILITY']:
        ic = compute_ic_for_regime(regime)
        assert ic > 0, f"Negative IC in {regime} regime: {ic}"
        assert ic > 0.03, f"Very low IC in {regime} regime: {ic}"
```

---

## G. Operator's Guide

### Running the Pipeline Safely

#### Prerequisites

```bash
# Verify environment
python --version  # Should be 3.10+
pip install -e ".[dev]"
pytest tests/ -v --tb=short  # Verify tests pass
```

#### Standard Production Run

```bash
# Always specify as_of_date explicitly - NEVER use defaults
python run_screen.py \
    --as-of-date 2026-01-15 \
    --data-dir production_data \
    --output results_2026-01-15.json \
    --checkpoint-dir ./checkpoints \
    --enable-enhancements \
    --audit-log audit_2026-01-15.jsonl
```

#### Collecting Fresh Data

```bash
# Step 1: Collect market data (requires network)
python collect_market_data.py \
    --universe production_data/universe.json \
    --output production_data/market_data.json

# Step 2: Verify data freshness
python -c "
import json
from datetime import date
data = json.load(open('production_data/market_data.json'))
for r in data[:5]:
    print(f'{r[\"ticker\"]}: collected {r.get(\"collected_at\", \"unknown\")}')
"

# Step 3: Run pipeline with fresh data
python run_screen.py --as-of-date 2026-01-15 ...
```

### Expected Outputs

| File | Contents | Size Range |
|------|----------|------------|
| `results.json` | Ranked securities, scores, positions | 100KB - 500KB |
| `audit.jsonl` | Line-by-line audit trail | 10KB - 50KB |
| `checkpoints/module_*.json` | Module outputs for debugging | 50KB - 200KB each |

### Failure Modes & Recovery

#### 1. Missing Data Files
```
Error: FileNotFoundError: production_data/universe.json not found
```
**Fix**: Ensure all input files exist. Run collectors if needed.

#### 2. Stale Data
```
Warning: Financial data is 95 days old (max: 90)
```
**Fix**: Run `collect_financial_data.py` to refresh.

#### 3. Schema Validation Failure
```
SchemaValidationError: Trial records validation failed: 5 errors
```
**Fix**: Check `trial_records.json` for malformed entries. Look at error details.

#### 4. Circuit Breaker Tripped
```
CircuitBreakerError: 60% of records failed validation (threshold: 50%)
```
**Fix**: Major data issue. Check data sources. Do NOT proceed.

#### 5. Network Issues During Collection
```
ConnectionError: Yahoo Finance API unreachable
```
**Fix**: Use `--use-cache` to use existing cached data:
```bash
python collect_market_data.py --use-cache
```

### Monitoring Recommendations

#### Daily Checks
1. **Score Distribution**: Should be roughly normal, mean ~50
2. **Coverage**: >90% of universe should have scores
3. **Error Rate**: <1% of tickers should fail scoring

#### Weekly Checks
1. **IC Trend**: Should be stable or improving
2. **Data Freshness**: All data <30 days old
3. **Turnover**: <30% rank changes week-over-week

#### Monthly Checks
1. **Walk-Forward IC**: Run backtest validation
2. **Regime IC**: Check IC by market regime
3. **Audit Log Review**: Check for recurring errors

### Configuration Files

| File | Purpose | Key Settings |
|------|---------|--------------|
| `config.yml` | Pipeline configuration | Weights, thresholds, paths |
| `pyproject.toml` | Package dependencies | Python version, deps |
| `CLAUDE.md` | Developer guide | Architecture, conventions |

### Contact & Escalation

For production issues:
1. Check this guide first
2. Review audit.jsonl for error details
3. Run `pytest tests/` to verify environment
4. Escalate to development team with:
   - as_of_date used
   - Error messages
   - audit.jsonl contents
   - `git log -1` (current version)

---

## Summary

### Critical Issues (Fix Immediately)
1. **C1**: Look-ahead bias in IC validation → Use `patches/patch_001_pit_safe_ic_validation.py`
2. **C2**: datetime.now() in collectors → Use `patches/patch_002_deterministic_collection.py`
3. **C3**: Missing JSON validation → Use `patches/patch_003_schema_validation.py`

### Strengths Identified
- Comprehensive PIT enforcement in core modules
- Deterministic hashing and provenance tracking
- Well-documented CLAUDE.md with clear conventions
- Extensive test coverage (104 test files)
- Proper Decimal arithmetic in financial calculations (mostly)

### Key Recommendations
1. Apply all 4 patches immediately
2. Add data quality gates at pipeline entry
3. Implement walk-forward IC validation
4. Add monitoring for error rates and data freshness
5. Consider regime-stratified IC reporting
