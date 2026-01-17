# Test Coverage Analysis

**Date**: 2026-01-17
**Author**: Claude Code Analysis
**Branch**: `claude/analyze-test-coverage-0RttJ`

## Executive Summary

The biotech-screener codebase has **218+ test functions across 39 test files**, with strong coverage for the core scoring modules (v2) but significant gaps in pipeline orchestration, data collection, and auxiliary engines.

### Coverage Highlights

| Category | Status | Coverage |
|----------|--------|----------|
| Core Scoring Modules (v2) | Strong | ~90% |
| Providers & Validators | Good | ~75% |
| Risk & Governance | Good | ~70% |
| Pipeline Orchestration | Critical Gap | ~0% |
| Data Collection | Critical Gap | ~0% |
| Auxiliary Engines | Gap | ~20% |

---

## Current Test Coverage

### Well-Tested Modules (Strong Coverage)

| Module | Test File | Tests | LOC |
|--------|-----------|-------|-----|
| `module_2_financial_v2` | `test_financial_v2_golden.py` | 42 | 1,071 |
| `module_5_composite_v2` | `test_composite_v2_golden.py` | 40 | 1,324 |
| `module_4_clinical_dev_v2` | `test_clinical_v2_golden.py` | 19 | 649 |
| `module_3_scoring_v2` | `test_catalyst_v2_golden.py`, `test_module_3_vnext.py` | 11 | 1,437 |
| Risk Gates | `test_risk_gates.py` | 8 | 465 |
| Governance | `test_governance.py` | 7 | 515 |
| Momentum Health | `test_momentum_health_monitor.py` | 7 | 317 |
| AACT Provider | `test_aact_provider.py` | 4 | ~500 |
| Snapshots | `test_snapshots.py` | 6 | 482 |

**Testing Patterns Used:**
- Golden fixture tests with determinism verification
- Edge case coverage (boundary conditions, missing data)
- PIT (Point-in-Time) discipline enforcement
- Content hash verification for reproducibility

### Partially Tested Modules (Moderate Coverage)

| Module | Existing Tests | Gap Analysis |
|--------|---------------|--------------|
| `liquidity_scoring.py` | `test_liquidity_scoring.py` (7 tests) | Missing edge cases for extreme values |
| `cfo_extractor.py` | `test_cfo_extractor.py` | Limited coverage of parsing edge cases |
| `module_3a_catalyst.py` | `test_module_3a.py` | Delta detection not fully tested |
| `backtest.py` | `test_backtest.py` (7 tests) | Missing multi-period scenarios |

---

## Critical Test Gaps (Priority Recommendations)

### Priority 1: Pipeline Orchestration (Critical)

**File**: `run_screen.py` (1,170 lines)
**Current Coverage**: 0%
**Risk Level**: HIGH

This is the main entry point that orchestrates all modules. No integration tests exist.

**Recommended Tests:**

```python
# tests/test_run_screen.py

class TestRunScreenPipeline:
    """Integration tests for the main screening pipeline."""

    def test_full_pipeline_determinism(self, tmp_path, sample_data_dir):
        """Verify identical inputs produce identical outputs."""

    def test_dry_run_validation(self, tmp_path, sample_data_dir):
        """Test dry-run mode validates inputs without running pipeline."""

    def test_checkpointing_save_resume(self, tmp_path, sample_data_dir):
        """Test checkpoint save/load for pipeline resumption."""

    def test_invalid_as_of_date_raises(self):
        """Test that invalid as_of_date format raises ValueError."""

    def test_missing_required_files_raises(self, tmp_path):
        """Test FileNotFoundError when required data files missing."""

    def test_ticker_validation_rejects_invalid(self, tmp_path, universe_with_invalid_tickers):
        """Test that invalid tickers in universe are rejected."""

    def test_enhancement_modules_integration(self, tmp_path, full_sample_data):
        """Test pipeline with enhancement modules enabled."""

    def test_audit_log_creation(self, tmp_path, sample_data_dir):
        """Test audit log JSONL file is created with correct structure."""
```

**Test Fixtures Needed:**
- `sample_data_dir`: Directory with minimal valid JSON files
- `universe_with_invalid_tickers`: Universe data with problematic tickers
- `full_sample_data`: Complete dataset for enhancement module testing

---

### Priority 2: Probability of Success Engine

**File**: `pos_engine.py` (677 lines)
**Current Coverage**: 0%
**Risk Level**: HIGH

Core scoring engine with complex indication mapping and benchmark lookups.

**Recommended Tests:**

```python
# tests/test_pos_engine.py

class TestProbabilityOfSuccessEngine:
    """Tests for PoS scoring engine."""

    def test_stage_score_mapping(self):
        """Verify stage names map to correct scores."""
        engine = ProbabilityOfSuccessEngine()

        # Test all stage mappings
        assert engine.calculate_pos_score("phase_1")["stage_score"] == Decimal("20")
        assert engine.calculate_pos_score("phase_3")["stage_score"] == Decimal("65")
        assert engine.calculate_pos_score("nda_bla")["stage_score"] == Decimal("80")

    def test_indication_normalization_word_boundaries(self):
        """Test indication matching uses word boundaries (no false positives)."""
        engine = ProbabilityOfSuccessEngine()

        # "dose" should NOT match "oncology" patterns
        result = engine._normalize_indication("dose escalation study")
        assert result != "oncology"

        # "cancer" should match oncology
        result = engine._normalize_indication("breast cancer treatment")
        assert result == "oncology"

    def test_oncology_vs_rare_disease_differentiation(self):
        """Verify rare disease has higher PoS than oncology at same stage."""
        engine = ProbabilityOfSuccessEngine()

        oncology = engine.calculate_pos_score("phase_3", indication="oncology")
        rare = engine.calculate_pos_score("phase_3", indication="rare disease")

        assert rare["pos_score"] > oncology["pos_score"]

    def test_benchmarks_fallback_when_file_missing(self, tmp_path):
        """Test fallback benchmarks when external file unavailable."""
        engine = ProbabilityOfSuccessEngine(benchmarks_path=str(tmp_path / "nonexistent.json"))

        assert engine.benchmarks_metadata["source"] == "FALLBACK_HARDCODED"

    def test_score_universe_content_hash_determinism(self):
        """Verify universe scoring produces deterministic content hash."""
        engine = ProbabilityOfSuccessEngine()
        universe = [
            {"ticker": "ACME", "base_stage": "phase_2", "indication": "oncology"},
            {"ticker": "BETA", "base_stage": "phase_3", "indication": "rare disease"},
        ]

        result1 = engine.score_universe(universe, date(2026, 1, 1))
        result2 = engine.score_universe(universe, date(2026, 1, 1))

        assert result1["provenance"]["content_hash"] == result2["provenance"]["content_hash"]

    def test_data_quality_state_classification(self):
        """Test data quality state assessment logic."""
        engine = ProbabilityOfSuccessEngine()

        # Full data
        result = engine.calculate_pos_score(
            "phase_2", indication="oncology",
            trial_design_quality=Decimal("1.1"),
            competitive_intensity=Decimal("0.9")
        )
        assert result["data_quality_state"] == "FULL"

        # Missing optional fields
        result = engine.calculate_pos_score("phase_2")
        assert result["data_quality_state"] == "PARTIAL"

    def test_adjustment_clamping(self):
        """Test that adjustments are clamped to valid ranges."""
        engine = ProbabilityOfSuccessEngine()

        # Out-of-range trial_design_quality should be clamped
        result = engine.calculate_pos_score(
            "phase_3",
            trial_design_quality=Decimal("2.0")  # Above 1.30 max
        )
        # Result should use clamped value of 1.30
        assert result["audit_entry"]["calculation"]["adjustments_applied"]["trial_design_quality"] == "1.30"
```

---

### Priority 3: Short Interest Engine

**File**: `short_interest_engine.py` (~300+ lines)
**Current Coverage**: 0%
**Risk Level**: MEDIUM-HIGH

Squeeze potential and crowding detection for supplementary signals.

**Recommended Tests:**

```python
# tests/test_short_interest_engine.py

class TestShortInterestSignalEngine:
    """Tests for short interest signal scoring."""

    def test_squeeze_potential_thresholds(self):
        """Test squeeze potential classification based on SI% and DTC."""
        engine = ShortInterestSignalEngine()

        # Extreme squeeze potential
        result = engine.calculate_short_signal(
            ticker="ACME",
            short_interest_pct=Decimal("45"),
            days_to_cover=Decimal("12")
        )
        assert result["squeeze_potential"] == "EXTREME"

        # Moderate
        result = engine.calculate_short_signal(
            ticker="BETA",
            short_interest_pct=Decimal("15"),
            days_to_cover=Decimal("6")
        )
        assert result["squeeze_potential"] == "MODERATE"

    def test_insufficient_data_handling(self):
        """Test graceful handling when no SI data available."""
        engine = ShortInterestSignalEngine()

        result = engine.calculate_short_signal(
            ticker="ACME",
            short_interest_pct=None,
            days_to_cover=None
        )
        assert result["status"] == "INSUFFICIENT_DATA"

    def test_crowding_risk_assessment(self):
        """Test crowding risk classification."""
        engine = ShortInterestSignalEngine()

        # High crowding (>30% SI)
        result = engine.calculate_short_signal(
            ticker="ACME",
            short_interest_pct=Decimal("35"),
            days_to_cover=Decimal("5")
        )
        assert result["crowding_risk"] == "HIGH"

    def test_signal_direction_bullish_on_covering(self):
        """Rapidly covering shorts should generate bullish signal."""
        engine = ShortInterestSignalEngine()

        result = engine.calculate_short_signal(
            ticker="ACME",
            short_interest_pct=Decimal("20"),
            days_to_cover=Decimal("8"),
            short_interest_change_pct=Decimal("-25")  # SI dropping
        )
        assert result["signal_direction"] == "BULLISH"

    def test_score_universe_coverage_tracking(self):
        """Test universe scoring tracks data coverage correctly."""
        engine = ShortInterestSignalEngine()
        universe = [
            {"ticker": "ACME", "short_interest_pct": Decimal("25")},
            {"ticker": "BETA"},  # No data
            {"ticker": "GAMMA", "short_interest_pct": Decimal("10")},
        ]

        result = engine.score_universe(universe, date(2026, 1, 1))
        assert result["diagnostic_counts"]["data_coverage_pct"] == "66.7%"
```

---

### Priority 4: Defensive Overlay Adapter

**File**: `defensive_overlay_adapter.py` (~200+ lines)
**Current Coverage**: 0%
**Risk Level**: MEDIUM

Position sizing with inverse-volatility weighting and correlation bonuses.

**Recommended Tests:**

```python
# tests/test_defensive_overlay_adapter.py

class TestDefensiveOverlay:
    """Tests for defensive overlay position sizing."""

    def test_sanitize_corr_handles_placeholder(self):
        """Test that 0.50 placeholder correlation is treated as missing."""
        corr, flags = sanitize_corr({"corr_xbi": "0.50"})
        assert corr is None
        assert "def_corr_placeholder_0.50" in flags

    def test_sanitize_corr_handles_nan(self):
        """Test NaN correlation is handled gracefully."""
        corr, flags = sanitize_corr({"corr_xbi": "NaN"})
        assert corr is None
        assert "def_corr_not_finite" in flags

    def test_elite_diversifier_bonus(self):
        """Test elite diversifier gets 1.40x multiplier."""
        mult, notes = defensive_multiplier({
            "corr_xbi": "0.25",  # Low correlation
            "vol_60d": "0.35"   # Low volatility
        })
        assert mult == Decimal("1.40")
        assert "def_mult_elite_1.40" in notes

    def test_high_correlation_penalty(self):
        """Test high correlation stocks get penalized."""
        mult, notes = defensive_multiplier({"corr_xbi": "0.85"})
        assert mult == Decimal("0.95")
        assert "def_mult_high_corr_0.95" in notes

    def test_inverse_vol_weight_scaling(self):
        """Test inverse-volatility weights scale correctly."""
        # Low volatility = high weight
        low_vol = raw_inv_vol_weight({"vol_60d": "0.20"})
        high_vol = raw_inv_vol_weight({"vol_60d": "0.60"})

        assert low_vol > high_vol
        # With power=2.0: 1/(0.20^2) = 25, 1/(0.60^2) = 2.78
        assert low_vol / high_vol > Decimal("5")

    def test_missing_volatility_returns_none(self):
        """Test missing volatility data returns None."""
        weight = raw_inv_vol_weight({})
        assert weight is None
```

---

### Priority 5: Elite Position Aggregator

**File**: `aggregator.py` (511 lines)
**Current Coverage**: 0%
**Risk Level**: MEDIUM

13F holdings aggregation for co-investment signals.

**Recommended Tests:**

```python
# tests/test_aggregator.py

class TestElitePositionAggregator:
    """Tests for 13F holdings aggregation."""

    def test_conviction_score_calculation(self):
        """Test conviction score formula components."""
        agg = ElitePositionAggregator()

        # Multiple Tier 1 holders with large positions
        positions = [
            ManagerPosition(
                manager_tier=1, manager_weight=1.2,
                position_weight=0.05, is_new_position=True
            ),
            ManagerPosition(
                manager_tier=1, manager_weight=1.2,
                position_weight=0.03, is_new_position=False
            ),
        ]

        score = agg._compute_conviction_score(positions)
        assert 0 <= score <= 100
        # 2 holders * 8 = 16 overlap + tier + concentration + momentum
        assert score > 30  # Significant score with 2 Tier 1 managers

    def test_aggregated_signal_tracking(self):
        """Test that all signal components are properly tracked."""
        signal = AggregatedSignal(
            cusip="12345678",
            ticker="ACME",
            issuer_name="ACME Corp",
            overlap_count=3,
            tier_1_count=2,
            total_value=10_000_000,
            conviction_score=65.0,
            holders=["Baker", "Perceptive", "RA Capital"],
            tier_1_holders=["Baker", "Perceptive"],
            managers_adding=["Baker"],
            managers_reducing=[],
            new_positions=["Baker"],
            exits=[]
        )

        result = signal.to_dict()
        assert result["overlap_count"] == 3
        assert result["tier_1_count"] == 2
        assert "Baker" in result["new_positions"]

    def test_point_in_time_filtering(self, mock_13f_filings):
        """Test that filings are filtered by as_of_date."""
        agg = ElitePositionAggregator()

        # Should only include filings before as_of_date
        positions = agg.fetch_all_holdings(
            quarters_back=1,
            as_of_date=date(2025, 6, 1)  # Historical date
        )

        for cik, pos_list in positions.items():
            for pos in pos_list:
                assert pos.filing_date <= date(2025, 6, 1)
```

---

### Priority 6: Data Collection Scripts

**Files**: `collect_all_data.py`, `collect_ctgov_data.py`, `collect_financial_data.py`, `collect_market_data.py`
**Current Coverage**: 0%
**Risk Level**: MEDIUM

**Recommended Approach:**

These scripts involve external API calls. Use mocking and integration test patterns:

```python
# tests/test_data_collection.py

class TestDataCollection:
    """Tests for data collection scripts (with mocked APIs)."""

    @pytest.fixture
    def mock_subprocess_run(self, mocker):
        return mocker.patch("subprocess.run")

    def test_collect_all_data_orchestration(self, mock_subprocess_run):
        """Test collect_all_data runs scripts in correct order."""
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # Simulate the main function logic
        from collect_all_data import run_collection

        run_collection("collect_market_data.py", "Market Data")
        run_collection("collect_ctgov_data.py", "Clinical Trials")
        run_collection("collect_financial_data.py", "Financial Data")

        assert mock_subprocess_run.call_count == 3

    def test_collection_failure_handling(self, mock_subprocess_run):
        """Test graceful handling of collection script failures."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "script")

        from collect_all_data import run_collection

        result = run_collection("failing_script.py", "Test")
        assert result is False
```

---

## Test Infrastructure Improvements

### 1. Shared Fixtures Library

Create `tests/conftest.py` with reusable fixtures:

```python
# tests/conftest.py

import pytest
from pathlib import Path
from decimal import Decimal
from datetime import date

@pytest.fixture
def sample_universe():
    """Minimal valid universe for testing."""
    return [
        {"ticker": "ACME", "sector": "Biotech", "market_cap": 5_000_000_000},
        {"ticker": "BETA", "sector": "Biotech", "market_cap": 2_000_000_000},
    ]

@pytest.fixture
def sample_financial_records():
    """Financial records with cash, burn, runway data."""
    return [
        {
            "ticker": "ACME",
            "cash": "500000000",
            "burn_rate": "50000000",
            "runway_months": 10,
            "as_of_date": "2025-12-31"
        }
    ]

@pytest.fixture
def sample_trial_records():
    """Clinical trial records for testing."""
    return [
        {
            "nct_id": "NCT12345678",
            "sponsor_ticker": "ACME",
            "phase": "Phase 3",
            "status": "Active, not recruiting",
            "primary_completion_date": "2026-06-15"
        }
    ]

@pytest.fixture
def sample_data_dir(tmp_path, sample_universe, sample_financial_records, sample_trial_records):
    """Create temporary data directory with sample files."""
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "universe.json").write_text(json.dumps(sample_universe))
    (data_dir / "financial_records.json").write_text(json.dumps(sample_financial_records))
    (data_dir / "trial_records.json").write_text(json.dumps(sample_trial_records))
    (data_dir / "market_data.json").write_text(json.dumps([]))

    return data_dir

@pytest.fixture
def as_of_date():
    """Standard as_of_date for deterministic tests."""
    return date(2026, 1, 15)
```

### 2. Property-Based Testing

Add `hypothesis` for property-based testing of scoring functions:

```python
# tests/test_scoring_properties.py

from hypothesis import given, strategies as st
from decimal import Decimal

@given(st.decimals(min_value=0, max_value=100))
def test_financial_score_always_bounded(score_input):
    """Financial scores are always between 0 and 100."""
    # Test implementation...
    assert Decimal("0") <= result <= Decimal("100")
```

### 3. Test Organization

Migrate root-level test files to organized structure:

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_pos_engine.py
│   ├── test_short_interest_engine.py
│   └── test_defensive_overlay.py
├── integration/
│   ├── test_run_screen.py
│   └── test_pipeline_determinism.py
├── providers/
│   ├── test_aact_provider.py
│   └── test_schema_validation.py
└── golden/
    ├── test_financial_v2_golden.py
    ├── test_composite_v2_golden.py
    └── test_catalyst_v2_golden.py
```

---

## Test Gap Summary Matrix

| Module | Priority | Effort | Impact | Recommendation |
|--------|----------|--------|--------|----------------|
| `run_screen.py` | P1 | Medium | Critical | Full integration test suite |
| `pos_engine.py` | P1 | Low | High | Unit tests for all public methods |
| `short_interest_engine.py` | P2 | Low | Medium | Unit tests with edge cases |
| `defensive_overlay_adapter.py` | P2 | Low | Medium | Unit tests for multipliers |
| `aggregator.py` | P2 | Medium | Medium | Mocked 13F data tests |
| `collect_*.py` | P3 | Medium | Low | Mocked subprocess tests |
| `cusip_mapper.py` | P3 | Low | Medium | Mapping validation tests |
| `regime_engine.py` | P3 | Low | Medium | Threshold boundary tests |

---

## Recommended Next Steps

1. **Immediate (This Sprint)**
   - Create `tests/conftest.py` with shared fixtures
   - Add `test_run_screen.py` with at least 5 integration tests
   - Add `test_pos_engine.py` with indication normalization tests

2. **Short-term (Next 2 Sprints)**
   - Add tests for `short_interest_engine.py`
   - Add tests for `defensive_overlay_adapter.py`
   - Migrate root-level tests to `tests/` directory

3. **Medium-term**
   - Add mocked tests for data collection scripts
   - Add property-based testing for scoring functions
   - Set up code coverage reporting (pytest-cov)

---

## Appendix: Test Count by File

| Test File | Test Count |
|-----------|------------|
| `test_composite_v2_golden.py` | 40 |
| `test_financial_v2_golden.py` | 42 |
| `test_clinical_v2_golden.py` | 19 |
| `test_module_3_vnext.py` | 10 |
| `test_risk_gates.py` | 8 |
| `test_governance.py` | 7 |
| `test_backtest.py` | 7 |
| `test_liquidity_scoring.py` | 7 |
| `test_momentum_health_monitor.py` | 7 |
| `test_integration_errors.py` | 7 |
| `test_data_integration.py` | 6 |
| `test_integration_determinism.py` | 6 |
| `test_snapshots.py` | 6 |
| `test_sec_13f.py` | 6 |
| `test_modules.py` | 6 |
| Other test files | ~34 |
| **Total** | **218+** |
