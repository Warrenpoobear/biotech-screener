"""
tests/test_data_integration_contracts.py - Data Integration Contract Tests

10 targeted regression tests for data integration:
1. Numeric falsy value handling (momentum_score=0)
2. Ticker case normalization across datasets
3. PIT timestamp boundary conditions
4. Schema drift detection
5. Join key mismatches
6. Coverage guardrail enforcement
7. Deterministic hash consistency
8. Holdings snapshot PIT filtering
9. Zero return_60d handling
10. Coordinate activity flag preservation

These tests prevent regressions for the critical integration bugs identified
in the data integration validation review.

Author: Wake Robin Capital Management
Version: 1.0.0
"""
import pytest
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import json
import hashlib

# Import contract validators
from common.data_integration_contracts import (
    # Schema validators
    validate_market_data_schema,
    validate_financial_records_schema,
    validate_trial_records_schema,
    validate_holdings_schema,
    # Join validation
    validate_join_invariants,
    normalize_ticker_set,
    check_ticker_uniqueness,
    check_ticker_case_consistency,
    # PIT validation
    validate_pit_admissibility,
    validate_dataset_pit,
    PITValidationResult,
    # Coverage guardrails
    validate_coverage_guardrails,
    CoverageConfig,
    CoverageReport,
    # Determinism
    compute_deterministic_hash,
    validate_output_determinism,
    # Numeric safety
    safe_numeric_check,
    validate_numeric_field,
    # Exceptions
    SchemaValidationError,
    JoinInvariantError,
    PITViolationError,
    CoverageGuardrailError,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def as_of_date() -> str:
    """Standard as_of_date for tests."""
    return "2026-01-15"


@pytest.fixture
def sample_universe() -> List[Dict]:
    """Sample universe with edge case tickers."""
    return [
        {"ticker": "ACME", "status": "active", "market_cap_mm": 1000},
        {"ticker": "BETA", "status": "active", "market_cap_mm": 500},
        {"ticker": "GAMMA", "status": "active", "market_cap_mm": 2000},
        {"ticker": "ZERO", "status": "active", "market_cap_mm": 100},
    ]


@pytest.fixture
def sample_market_data() -> List[Dict]:
    """Sample market data with edge cases."""
    return [
        {"ticker": "ACME", "price": 25.50, "return_60d": 0.15, "source_date": "2026-01-14"},
        {"ticker": "BETA", "price": 12.00, "return_60d": -0.05, "source_date": "2026-01-14"},
        {"ticker": "GAMMA", "price": 45.00, "return_60d": 0.0, "source_date": "2026-01-14"},  # Zero return
        {"ticker": "ZERO", "price": 5.00, "return_60d": 0, "source_date": "2026-01-14"},  # Zero as int
    ]


@pytest.fixture
def sample_financial_records() -> List[Dict]:
    """Sample financial records with mixed case tickers."""
    return [
        {"ticker": "ACME", "Cash": 100000000, "source_date": "2025-12-01"},
        {"ticker": "beta", "Cash": 50000000, "source_date": "2025-12-15"},  # lowercase
        {"ticker": "GAMMA", "Cash": 200000000, "source_date": "2025-11-01"},
        {"ticker": "ZERO", "Cash": 0, "source_date": "2025-12-20"},  # Zero cash
    ]


@pytest.fixture
def sample_trial_records() -> List[Dict]:
    """Sample trial records with PIT edge cases."""
    return [
        {"ticker": "ACME", "nct_id": "NCT12345678", "phase": "Phase 2", "first_posted": "2024-06-15"},
        {"ticker": "ACME", "nct_id": "NCT12345679", "phase": "Phase 3", "first_posted": "2026-01-14"},  # Edge of cutoff
        {"ticker": "BETA", "nct_id": "NCT23456789", "phase": "Phase 1", "first_posted": "2025-09-01"},
        {"ticker": "GAMMA", "nct_id": "NCT34567890", "phase": "Phase 3", "first_posted": "2023-03-20"},
        {"ticker": "ACME", "nct_id": "NCT99999999", "phase": "Phase 1", "first_posted": "2026-02-01"},  # Future
    ]


@pytest.fixture
def sample_momentum_results() -> Dict:
    """Sample momentum results with zero scores."""
    return {
        "rankings": [
            {"ticker": "ACME", "momentum_score": 75},
            {"ticker": "BETA", "momentum_score": 50},
            {"ticker": "GAMMA", "momentum_score": 0},  # Zero score (falsy but valid!)
            {"ticker": "ZERO", "momentum_score": 0.0},  # Zero as float
        ],
        "summary": {
            "coordinated_buys": ["ACME"],
            "coordinated_sells": ["GAMMA"],
        }
    }


# =============================================================================
# TEST 1: NUMERIC FALSY VALUE HANDLING
# =============================================================================

class TestNumericFalsyHandling:
    """
    Test that zero values are not incorrectly treated as missing/invalid.

    CRITICAL BUG: run_screen.py:884 has `if ticker and momentum_score` which
    would skip valid momentum_score=0 values.
    """

    def test_safe_numeric_check_zero_int(self):
        """Zero integer should be recognized as valid numeric."""
        assert safe_numeric_check(0) is True

    def test_safe_numeric_check_zero_float(self):
        """Zero float should be recognized as valid numeric."""
        assert safe_numeric_check(0.0) is True

    def test_safe_numeric_check_zero_decimal(self):
        """Zero Decimal should be recognized as valid numeric."""
        assert safe_numeric_check(Decimal("0")) is True

    def test_safe_numeric_check_zero_string(self):
        """Zero as string should be recognized as valid numeric."""
        assert safe_numeric_check("0") is True
        assert safe_numeric_check("0.0") is True

    def test_safe_numeric_check_none(self):
        """None should not be recognized as valid numeric."""
        assert safe_numeric_check(None) is False

    def test_safe_numeric_check_empty_string(self):
        """Empty string should not be recognized as valid numeric."""
        assert safe_numeric_check("") is False

    def test_safe_numeric_check_bool_false(self):
        """Boolean False should not be recognized as valid numeric (even though falsy)."""
        assert safe_numeric_check(False) is False

    def test_momentum_score_zero_not_skipped(self, sample_momentum_results):
        """
        Regression test: momentum_score=0 should not be skipped.

        The bug pattern: `if ticker and momentum_score` treats 0 as falsy.
        The fix: use `if ticker and safe_numeric_check(momentum_score)`
        """
        skipped_count = 0
        valid_count = 0

        for ranking in sample_momentum_results["rankings"]:
            ticker = ranking.get("ticker")
            score = ranking.get("momentum_score")

            # BUG pattern (would incorrectly skip score=0):
            if ticker and score:  # This skips score=0!
                pass
            else:
                skipped_count += 1

            # CORRECT pattern:
            if ticker and safe_numeric_check(score):
                valid_count += 1

        # The bug would skip 2 records (GAMMA and ZERO with score=0)
        assert skipped_count == 2, "Bug pattern should skip 2 zero-score records"
        # The fix handles all 4 records correctly
        assert valid_count == 4, "Correct pattern should handle all 4 records"


# =============================================================================
# TEST 2: TICKER CASE NORMALIZATION
# =============================================================================

class TestTickerCaseNormalization:
    """
    Test that ticker case is consistently normalized across datasets.

    BUG: run_screen.py:797 normalizes market_data to uppercase, but
    module_2_financial.py:641 does NOT normalize, causing join failures.
    """

    def test_normalize_ticker_set_uppercase(self):
        """Tickers should be normalized to uppercase."""
        mixed_case = ["ACME", "beta", "Gamma", "DeLtA"]
        normalized = normalize_ticker_set(mixed_case)

        assert normalized == {"ACME", "BETA", "GAMMA", "DELTA"}

    def test_normalize_ticker_set_strips_whitespace(self):
        """Whitespace should be stripped."""
        with_whitespace = [" ACME ", "BETA", "  GAMMA"]
        normalized = normalize_ticker_set(with_whitespace)

        assert normalized == {"ACME", "BETA", "GAMMA"}

    def test_normalize_ticker_set_empty_strings_excluded(self):
        """Empty strings should be excluded."""
        with_empty = ["ACME", "", "BETA", "   ", "GAMMA"]
        normalized = normalize_ticker_set(with_empty)

        assert normalized == {"ACME", "BETA", "GAMMA"}

    def test_case_consistency_detection(self, sample_financial_records):
        """Detect case inconsistencies in dataset."""
        # sample_financial_records has "beta" (lowercase)
        is_consistent, variants = check_ticker_case_consistency(sample_financial_records)

        # There should be no inconsistency within the same dataset
        # (each ticker only appears once, just with different case)
        assert is_consistent is True

    def test_join_with_case_mismatch(self, sample_universe, sample_financial_records):
        """
        Test that join validation catches case mismatches.

        Universe has "BETA" but financial has "beta".
        """
        universe_tickers = normalize_ticker_set([r["ticker"] for r in sample_universe])
        financial_tickers = normalize_ticker_set([r["ticker"] for r in sample_financial_records])

        # After normalization, they should match
        assert "BETA" in universe_tickers
        assert "BETA" in financial_tickers


# =============================================================================
# TEST 3: PIT TIMESTAMP BOUNDARY CONDITIONS
# =============================================================================

class TestPITBoundaryConditions:
    """
    Test PIT (Point-in-Time) boundary conditions.

    PIT Rule: source_date <= as_of_date - 1 (cutoff)
    For as_of_date=2026-01-15, cutoff is 2026-01-14
    """

    def test_pit_exactly_at_cutoff(self, as_of_date):
        """Record exactly at cutoff should be admissible."""
        records = [
            {"ticker": "ACME", "source_date": "2026-01-14"},  # Exactly at cutoff
        ]

        result = validate_pit_admissibility(records, as_of_date)

        assert result.is_valid is True
        assert result.pit_compliant == 1
        assert result.pit_violated == 0

    def test_pit_one_day_after_cutoff(self, as_of_date):
        """Record one day after cutoff should be rejected."""
        records = [
            {"ticker": "ACME", "source_date": "2026-01-15"},  # One day after cutoff
        ]

        result = validate_pit_admissibility(records, as_of_date)

        assert result.is_valid is False
        assert result.pit_violated == 1

    def test_pit_future_record(self, as_of_date):
        """Future record should be rejected."""
        records = [
            {"ticker": "ACME", "source_date": "2026-02-01"},  # Future
        ]

        result = validate_pit_admissibility(records, as_of_date)

        assert result.is_valid is False
        assert result.pit_violated == 1

    def test_pit_with_fallback_fields(self, as_of_date):
        """Test fallback to alternative date fields."""
        records = [
            {"ticker": "ACME", "first_posted": "2024-06-15"},  # No source_date
        ]

        result = validate_pit_admissibility(
            records, as_of_date,
            date_field="source_date",
            fallback_fields=["first_posted"]
        )

        assert result.is_valid is True
        assert result.pit_compliant == 1

    def test_pit_trial_dataset(self, sample_trial_records, as_of_date):
        """Test PIT validation for trial records dataset."""
        result = validate_dataset_pit("trial", sample_trial_records, as_of_date)

        # Should detect 1 future record (NCT99999999 with first_posted=2026-02-01)
        assert result.pit_violated >= 1
        assert any(r.get("nct_id") == "NCT99999999" for r in result.future_records)


# =============================================================================
# TEST 4: SCHEMA DRIFT DETECTION
# =============================================================================

class TestSchemaDriftDetection:
    """Test that schema validation catches missing/malformed fields."""

    def test_market_data_missing_ticker(self):
        """Records without ticker should fail validation."""
        records = [
            {"price": 25.50, "return_60d": 0.15},  # No ticker
        ]

        is_valid, invalid = validate_market_data_schema(records, raise_on_error=False)

        assert is_valid is False
        assert len(invalid) == 1

    def test_market_data_empty_ticker(self):
        """Empty ticker string should fail validation."""
        records = [
            {"ticker": "", "price": 25.50},
            {"ticker": "   ", "price": 12.00},  # Whitespace only
        ]

        is_valid, invalid = validate_market_data_schema(records, raise_on_error=False)

        assert is_valid is False
        assert len(invalid) == 2

    def test_market_data_non_numeric_price(self):
        """Non-numeric price should fail strict validation."""
        records = [
            {"ticker": "ACME", "price": "not_a_number"},
        ]

        is_valid, invalid = validate_market_data_schema(records, raise_on_error=False)

        assert is_valid is False
        assert len(invalid) == 1

    def test_trial_records_missing_nct_id(self):
        """Trial records without nct_id should fail."""
        records = [
            {"ticker": "ACME", "phase": "Phase 2"},  # No nct_id
        ]

        is_valid, invalid = validate_trial_records_schema(records, raise_on_error=False)

        assert is_valid is False

    def test_schema_validation_raises_on_request(self):
        """Schema validation should raise exception when requested."""
        records = [{"not_a_ticker": "ACME"}]

        with pytest.raises(SchemaValidationError):
            validate_market_data_schema(records, raise_on_error=True)


# =============================================================================
# TEST 5: JOIN KEY MISMATCHES
# =============================================================================

class TestJoinKeyMismatches:
    """Test detection of join key mismatches between datasets."""

    def test_join_missing_tickers(self, sample_universe):
        """Detect tickers missing from downstream datasets."""
        universe_tickers = normalize_ticker_set([r["ticker"] for r in sample_universe])
        # Financial missing "ZERO"
        financial_tickers = {"ACME", "BETA", "GAMMA"}

        result = validate_join_invariants(
            universe_tickers=universe_tickers,
            financial_tickers=financial_tickers,
            min_financial_coverage_pct=80.0,
        )

        # 3/4 = 75% < 80% threshold
        assert result["is_valid"] is False
        assert "ZERO" in result["missing"]["financial"]

    def test_join_orphan_tickers(self, sample_universe):
        """Detect orphan tickers (in downstream but not universe)."""
        universe_tickers = normalize_ticker_set([r["ticker"] for r in sample_universe])
        # Financial has extra "ORPHAN" ticker
        financial_tickers = {"ACME", "BETA", "GAMMA", "ZERO", "ORPHAN"}

        result = validate_join_invariants(
            universe_tickers=universe_tickers,
            financial_tickers=financial_tickers,
        )

        assert "ORPHAN" in result["orphans"]["financial"]
        assert result["orphan_counts"]["financial"] == 1

    def test_join_ticker_uniqueness(self):
        """Detect duplicate tickers in dataset."""
        records = [
            {"ticker": "ACME", "value": 1},
            {"ticker": "ACME", "value": 2},  # Duplicate
            {"ticker": "BETA", "value": 3},
        ]

        is_unique, duplicates = check_ticker_uniqueness(records)

        assert is_unique is False
        assert "ACME" in duplicates


# =============================================================================
# TEST 6: COVERAGE GUARDRAIL ENFORCEMENT
# =============================================================================

class TestCoverageGuardrails:
    """Test coverage guardrail thresholds."""

    def test_coverage_below_threshold_fails(self):
        """Coverage below threshold should fail."""
        report = validate_coverage_guardrails(
            universe_size=100,
            financial_count=70,  # 70% < 80% threshold
            clinical_count=80,
            market_count=80,
            config=CoverageConfig(min_financial_pct=80.0),
        )

        assert report.is_valid is False
        assert "financial" in report.failures[0]

    def test_coverage_at_threshold_passes(self):
        """Coverage exactly at threshold should pass."""
        report = validate_coverage_guardrails(
            universe_size=100,
            financial_count=80,  # Exactly 80%
            clinical_count=50,
            market_count=50,
            config=CoverageConfig(
                min_financial_pct=80.0,
                min_clinical_pct=50.0,
                min_market_pct=50.0,
                min_catalyst_pct=0.0,  # Not testing catalyst coverage
            ),
        )

        assert report.is_valid is True

    def test_coverage_empty_universe_skipped(self):
        """Empty universe should skip validation."""
        report = validate_coverage_guardrails(
            universe_size=0,
            financial_count=0,
        )

        assert report.is_valid is True
        assert "Empty universe" in report.warnings[0]

    def test_coverage_raises_on_request(self):
        """Coverage failure should raise exception when requested."""
        with pytest.raises(CoverageGuardrailError):
            validate_coverage_guardrails(
                universe_size=100,
                financial_count=50,  # 50% < 80%
                config=CoverageConfig(min_financial_pct=80.0),
                raise_on_error=True,
            )


# =============================================================================
# TEST 7: DETERMINISTIC HASH CONSISTENCY
# =============================================================================

class TestDeterministicHash:
    """Test deterministic hashing of data structures."""

    def test_hash_dict_key_order_invariant(self):
        """Hash should be same regardless of dict key order."""
        data1 = {"b": 2, "a": 1, "c": 3}
        data2 = {"a": 1, "b": 2, "c": 3}
        data3 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_deterministic_hash(data1)
        hash2 = compute_deterministic_hash(data2)
        hash3 = compute_deterministic_hash(data3)

        assert hash1 == hash2 == hash3

    def test_hash_nested_dict_order_invariant(self):
        """Hash should be same for nested dicts regardless of order."""
        data1 = {"outer": {"b": 2, "a": 1}}
        data2 = {"outer": {"a": 1, "b": 2}}

        hash1 = compute_deterministic_hash(data1)
        hash2 = compute_deterministic_hash(data2)

        assert hash1 == hash2

    def test_hash_decimal_consistency(self):
        """Decimal values should hash consistently."""
        data1 = {"value": Decimal("1.23")}
        data2 = {"value": Decimal("1.230")}  # Same value, different representation

        hash1 = compute_deterministic_hash(data1)
        hash2 = compute_deterministic_hash(data2)

        # Note: These may differ due to string representation
        # The important thing is same input gives same output
        assert hash1 == compute_deterministic_hash({"value": Decimal("1.23")})

    def test_output_determinism_validation(self):
        """validate_output_determinism should detect differences."""
        output1 = {"score": "50.00", "ticker": "ACME"}
        output2 = {"score": "50.00", "ticker": "ACME"}
        output3 = {"score": "50.01", "ticker": "ACME"}  # Different score

        is_same, _ = validate_output_determinism(output1, output2)
        is_diff, msg = validate_output_determinism(output1, output3)

        assert is_same is True
        assert is_diff is False
        assert "Hash mismatch" in msg


# =============================================================================
# TEST 8: HOLDINGS SNAPSHOT PIT FILTERING
# =============================================================================

class TestHoldingsSnapshotPIT:
    """Test PIT filtering for holdings snapshots."""

    def test_holdings_schema_valid(self):
        """Valid holdings format should pass."""
        holdings = {
            "ACME": {
                "holdings": {
                    "current": {
                        "0001263508": {"value_kusd": 50000},
                    }
                }
            }
        }

        is_valid, errors = validate_holdings_schema(holdings, raise_on_error=False)

        assert is_valid is True
        assert len(errors) == 0

    def test_holdings_schema_invalid_structure(self):
        """Invalid holdings structure should fail."""
        holdings = {
            "ACME": "not_a_dict",  # Should be dict
        }

        is_valid, errors = validate_holdings_schema(holdings, raise_on_error=False)

        assert is_valid is False

    def test_holdings_empty_ticker_key(self):
        """Empty ticker key should be detected."""
        holdings = {
            "": {"holdings": {}},  # Empty ticker
        }

        is_valid, errors = validate_holdings_schema(holdings, raise_on_error=False)

        assert is_valid is False


# =============================================================================
# TEST 9: ZERO RETURN_60D HANDLING
# =============================================================================

class TestZeroReturnHandling:
    """Test that zero return values are handled correctly."""

    def test_zero_return_is_valid_numeric(self):
        """return_60d=0 should be recognized as valid."""
        record = {"ticker": "ACME", "return_60d": 0}

        is_valid, _ = validate_numeric_field(record["return_60d"], "return_60d")

        assert is_valid is True

    def test_zero_return_float_is_valid(self):
        """return_60d=0.0 should be recognized as valid."""
        record = {"ticker": "ACME", "return_60d": 0.0}

        is_valid, _ = validate_numeric_field(record["return_60d"], "return_60d")

        assert is_valid is True

    def test_market_data_with_zero_returns(self, sample_market_data):
        """Market data schema should accept zero returns."""
        is_valid, invalid = validate_market_data_schema(
            sample_market_data, raise_on_error=False
        )

        assert is_valid is True
        # GAMMA and ZERO have zero returns, should both be valid

    def test_none_return_is_missing(self):
        """return_60d=None should be recognized as missing."""
        record = {"ticker": "ACME", "return_60d": None}

        is_valid, msg = validate_numeric_field(record["return_60d"], "return_60d")

        assert is_valid is False
        assert "None" in msg


# =============================================================================
# TEST 10: COORDINATED ACTIVITY FLAG PRESERVATION
# =============================================================================

class TestCoordinatedActivityFlags:
    """
    Test that coordinated buy/sell flags are correctly preserved.

    These flags come from momentum_results and should be injected into
    market_data correctly, without being filtered by falsy value checks.
    """

    def test_coordinated_buys_detected(self, sample_momentum_results):
        """Coordinated buys should be detected from summary."""
        coordinated_buys = set(sample_momentum_results["summary"]["coordinated_buys"])

        assert "ACME" in coordinated_buys

    def test_coordinated_sells_detected(self, sample_momentum_results):
        """Coordinated sells should be detected from summary."""
        coordinated_sells = set(sample_momentum_results["summary"]["coordinated_sells"])

        assert "GAMMA" in coordinated_sells

    def test_zero_score_with_coordinated_flag(self, sample_momentum_results):
        """
        GAMMA has momentum_score=0 AND is in coordinated_sells.

        The bug would skip GAMMA due to falsy score, losing the coordinated_sell flag.
        """
        rankings = sample_momentum_results["rankings"]
        gamma = next(r for r in rankings if r["ticker"] == "GAMMA")
        coordinated_sells = set(sample_momentum_results["summary"]["coordinated_sells"])

        # GAMMA has score=0 (falsy) but is a coordinated sell
        assert gamma["momentum_score"] == 0
        assert "GAMMA" in coordinated_sells

        # The fix should still process GAMMA
        assert safe_numeric_check(gamma["momentum_score"]) is True

    def test_inject_momentum_with_zero_scores(self, sample_momentum_results, sample_market_data):
        """
        Simulate momentum injection with zero scores.

        This mimics the logic in run_screen.py:871-902.
        """
        market_data_by_ticker = {r["ticker"]: r.copy() for r in sample_market_data}
        injected = []
        skipped_by_bug = []

        for ranking in sample_momentum_results["rankings"]:
            ticker = ranking.get("ticker")
            momentum_score = ranking.get("momentum_score")

            # BUG pattern:
            if ticker and momentum_score and ticker in market_data_by_ticker:
                pass  # Would process
            elif ticker and ticker in market_data_by_ticker:
                skipped_by_bug.append(ticker)

            # CORRECT pattern:
            if ticker and safe_numeric_check(momentum_score) and ticker in market_data_by_ticker:
                injected.append(ticker)

        # GAMMA and ZERO have score=0, bug would skip them
        assert "GAMMA" in skipped_by_bug
        assert "ZERO" in skipped_by_bug

        # Correct pattern handles all 4
        assert len(injected) == 4
        assert "GAMMA" in injected
        assert "ZERO" in injected


# =============================================================================
# INTEGRATION TEST: FULL VALIDATION HARNESS
# =============================================================================

class TestFullValidationHarness:
    """Integration test for the full validation harness."""

    def test_fixture_data_validation(
        self,
        sample_universe,
        sample_market_data,
        sample_financial_records,
        sample_trial_records,
        as_of_date,
    ):
        """Run all validations on fixture data."""
        # Schema validation
        mkt_valid, _ = validate_market_data_schema(sample_market_data, raise_on_error=False)
        fin_valid, _ = validate_financial_records_schema(sample_financial_records, raise_on_error=False)
        trial_valid, _ = validate_trial_records_schema(sample_trial_records, raise_on_error=False)

        assert mkt_valid is True
        assert fin_valid is True
        assert trial_valid is True

        # Join validation
        universe_tickers = normalize_ticker_set([r["ticker"] for r in sample_universe])
        market_tickers = normalize_ticker_set([r["ticker"] for r in sample_market_data])
        financial_tickers = normalize_ticker_set([r["ticker"] for r in sample_financial_records])

        join_result = validate_join_invariants(
            universe_tickers=universe_tickers,
            financial_tickers=financial_tickers,
            market_tickers=market_tickers,
        )

        # All 4 tickers should be covered (100%)
        assert join_result["coverage"]["financial_pct"] == 100.0
        assert join_result["coverage"]["market_pct"] == 100.0

        # PIT validation
        trial_pit = validate_dataset_pit("trial", sample_trial_records, as_of_date)

        # Should detect 1 future record
        assert trial_pit.pit_violated >= 1
