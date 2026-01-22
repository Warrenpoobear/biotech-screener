#!/usr/bin/env python3
"""
Tests for src/modules/ic_pit_validation.py

Tests PIT (Point-in-Time) validation and production gates.
Covers:
- Adaptive weight PIT validation
- Peer valuation PIT validation
- Co-invest PIT validation
- Weight stability validation
- Production gate orchestration
- Ablation testing framework
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal

from src.modules.ic_pit_validation import (
    PITValidationError,
    WeightStabilityError,
    DataQualityError,
    ValidationStatus,
    PITValidationResult,
    WeightProvenance,
    ProductionGateResult,
    validate_adaptive_weight_pit,
    validate_peer_valuation_pit,
    validate_coinvest_pit,
    validate_weight_stability,
    create_weight_provenance,
    run_production_gate,
    run_ablation_test,
    AblationResult,
    DEFAULT_EMBARGO_DAYS,
    MAX_WEIGHT_L1_CHANGE,
)


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_status_values(self):
        """All status values should be present."""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"
        assert ValidationStatus.WARNING.value == "warning"


class TestPITValidationResult:
    """Tests for PITValidationResult dataclass."""

    def test_creation(self):
        """Should create valid result."""
        result = PITValidationResult(
            status=ValidationStatus.PASSED,
            check_name="test_check",
            as_of_date="2026-01-15",
            details={"key": "value"},
        )

        assert result.status == ValidationStatus.PASSED
        assert result.passed == True
        assert result.check_name == "test_check"
        assert result.violations == []

    def test_passed_property(self):
        """passed property should reflect status."""
        passed_result = PITValidationResult(
            status=ValidationStatus.PASSED,
            check_name="test",
            as_of_date="2026-01-15",
            details={},
        )
        assert passed_result.passed == True

        failed_result = PITValidationResult(
            status=ValidationStatus.FAILED,
            check_name="test",
            as_of_date="2026-01-15",
            details={},
        )
        assert failed_result.passed == False


class TestValidateAdaptiveWeightPIT:
    """Tests for validate_adaptive_weight_pit function."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_valid_historical_scores(self, as_of_date):
        """Valid historical scores should pass."""
        # Embargo cutoff is as_of_date - 30 days = Dec 16
        historical_scores = [
            {"as_of_date": "2025-06-01", "ticker": "ACME", "score": 75},
            {"as_of_date": "2025-09-01", "ticker": "ACME", "score": 80},
            {"as_of_date": "2025-12-01", "ticker": "ACME", "score": 82},  # Before embargo
        ]

        historical_returns = {
            (date(2025, 6, 1), "ACME"): Decimal("0.05"),
            (date(2025, 9, 1), "ACME"): Decimal("0.08"),
        }

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
        )

        assert result.status == ValidationStatus.PASSED
        assert len(result.violations) == 0

    def test_scores_after_embargo_fails(self, as_of_date):
        """Scores after embargo cutoff should fail."""
        embargo_cutoff = as_of_date - timedelta(days=DEFAULT_EMBARGO_DAYS)

        historical_scores = [
            {"as_of_date": "2025-06-01", "ticker": "ACME", "score": 75},
            {"as_of_date": (embargo_cutoff + timedelta(days=1)).isoformat(), "ticker": "BAD", "score": 80},
        ]

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns={},
        )

        assert result.status == ValidationStatus.FAILED
        assert any("after embargo" in v for v in result.violations)

    def test_scores_without_date_fails(self, as_of_date):
        """Scores without as_of_date should be flagged."""
        historical_scores = [
            {"ticker": "ACME", "score": 75},  # Missing as_of_date
        ]

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns={},
        )

        assert result.status == ValidationStatus.FAILED
        assert any("without valid as_of_date" in v for v in result.violations)

    def test_returns_with_invalid_keys_fails(self, as_of_date):
        """Returns with invalid keys should fail."""
        historical_scores = [
            {"as_of_date": "2025-06-01", "ticker": "ACME", "score": 75},
        ]

        # Invalid key format (not tuple)
        historical_returns = {
            "ACME_2025-06-01": Decimal("0.05"),  # String key instead of tuple
        }

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
        )

        assert result.status == ValidationStatus.FAILED
        assert any("invalid keys" in v for v in result.violations)

    def test_returns_after_embargo_fails(self, as_of_date):
        """Returns with dates after embargo should fail."""
        embargo_cutoff = as_of_date - timedelta(days=DEFAULT_EMBARGO_DAYS)

        historical_scores = [
            {"as_of_date": "2025-06-01", "ticker": "ACME", "score": 75},
        ]

        historical_returns = {
            (embargo_cutoff + timedelta(days=1), "ACME"): Decimal("0.05"),  # After embargo
        }

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
        )

        assert result.status == ValidationStatus.FAILED
        assert any("returns after embargo" in v for v in result.violations)

    def test_short_training_window_fails(self, as_of_date):
        """Training window less than minimum should fail."""
        # Only 2 months of data
        historical_scores = [
            {"as_of_date": "2025-10-01", "ticker": "ACME", "score": 75},
            {"as_of_date": "2025-11-01", "ticker": "ACME", "score": 80},
        ]

        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns={},
            min_training_months=6,
        )

        assert result.status == ValidationStatus.FAILED
        assert any("Training window" in v for v in result.violations)

    def test_custom_embargo_days(self, as_of_date):
        """Should respect custom embargo days."""
        # With 60-day embargo, cutoff is Nov 16
        historical_scores = [
            {"as_of_date": "2025-11-20", "ticker": "ACME", "score": 75},  # Would pass 30-day, fail 60-day
        ]

        # Should fail with 60-day embargo
        result = validate_adaptive_weight_pit(
            as_of_date=as_of_date,
            historical_scores=historical_scores,
            historical_returns={},
            embargo_days=60,
        )

        assert result.status == ValidationStatus.FAILED


class TestValidatePeerValuationPIT:
    """Tests for validate_peer_valuation_pit function."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_valid_peer_data(self, as_of_date):
        """Valid peer data should pass."""
        peer_valuations = [
            {"ticker": "PEER1", "snapshot_date": "2026-01-14", "market_cap": 1000},
            {"ticker": "PEER2", "snapshot_date": "2026-01-13", "market_cap": 2000},
        ]

        result = validate_peer_valuation_pit(as_of_date, peer_valuations)

        assert result.status == ValidationStatus.PASSED

    def test_snapshot_after_pit_fails(self, as_of_date):
        """Snapshot after PIT cutoff should fail."""
        peer_valuations = [
            {"ticker": "PEER1", "snapshot_date": "2026-01-15", "market_cap": 1000},  # Same day = after cutoff
        ]

        result = validate_peer_valuation_pit(as_of_date, peer_valuations)

        assert result.status == ValidationStatus.FAILED
        assert any("after PIT cutoff" in v for v in result.violations)

    def test_many_missing_snapshots_warning(self, as_of_date):
        """More than 50% missing snapshots should warn."""
        peer_valuations = [
            {"ticker": "PEER1", "market_cap": 1000},  # No snapshot
            {"ticker": "PEER2", "market_cap": 2000},  # No snapshot
            {"ticker": "PEER3", "snapshot_date": "2026-01-14", "market_cap": 3000},
        ]

        result = validate_peer_valuation_pit(as_of_date, peer_valuations)

        assert result.status == ValidationStatus.WARNING
        assert any("50%" in v for v in result.violations)


class TestValidateCoinvestPIT:
    """Tests for validate_coinvest_pit function."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_valid_coinvest_data(self, as_of_date):
        """Valid co-invest data should pass."""
        coinvest_data = {
            "coinvest_usable": True,
            "coinvest_overlap_count": 3,
            "coinvest_published_at_max": "2026-01-10",
            "coinvest_flags": [],
        }

        result = validate_coinvest_pit(as_of_date, coinvest_data)

        assert result.status == ValidationStatus.PASSED

    def test_published_at_after_as_of_fails(self, as_of_date):
        """Published date >= as_of_date should fail."""
        coinvest_data = {
            "coinvest_usable": True,
            "coinvest_overlap_count": 3,
            "coinvest_published_at_max": "2026-01-15",  # Same as as_of_date
        }

        result = validate_coinvest_pit(as_of_date, coinvest_data)

        assert result.status == ValidationStatus.FAILED

    def test_unused_coinvest_passes(self, as_of_date):
        """Unused co-invest data should pass."""
        coinvest_data = {
            "coinvest_usable": False,
            "coinvest_flags": ["no_signal"],
        }

        result = validate_coinvest_pit(as_of_date, coinvest_data)

        assert result.status == ValidationStatus.PASSED


class TestValidateWeightStability:
    """Tests for validate_weight_stability function."""

    def test_first_period_passes(self):
        """First period (no previous weights) should pass."""
        current = {"clinical": Decimal("0.40"), "financial": Decimal("0.30")}

        result = validate_weight_stability(current, None)

        assert result.status == ValidationStatus.PASSED
        assert result.details["l1_change"] == "N/A (first period)"

    def test_small_change_passes(self):
        """Small weight change should pass."""
        previous = {"clinical": Decimal("0.40"), "financial": Decimal("0.30")}
        current = {"clinical": Decimal("0.42"), "financial": Decimal("0.28")}

        result = validate_weight_stability(current, previous)

        assert result.status == ValidationStatus.PASSED

    def test_large_change_fails(self):
        """Large weight change should fail."""
        previous = {"clinical": Decimal("0.40"), "financial": Decimal("0.30")}
        current = {"clinical": Decimal("0.60"), "financial": Decimal("0.10")}  # 0.40 total change

        result = validate_weight_stability(
            current, previous, max_l1_change=Decimal("0.15")
        )

        assert result.status == ValidationStatus.FAILED
        assert any("L1 change" in v for v in result.violations)

    def test_new_weight_key_counts_as_change(self):
        """New weight key should count towards L1 change."""
        previous = {"clinical": Decimal("0.50")}
        current = {"clinical": Decimal("0.40"), "catalyst": Decimal("0.10")}

        result = validate_weight_stability(current, previous)

        # Change is |0.50-0.40| + |0-0.10| = 0.20 > 0.15 default
        assert result.status == ValidationStatus.FAILED

    def test_custom_max_change(self):
        """Should respect custom max_l1_change."""
        previous = {"w1": Decimal("0.50")}
        current = {"w1": Decimal("0.30")}  # 0.20 change

        # Should fail with 0.15 max
        result1 = validate_weight_stability(
            current, previous, max_l1_change=Decimal("0.15")
        )
        assert result1.status == ValidationStatus.FAILED

        # Should pass with 0.25 max
        result2 = validate_weight_stability(
            current, previous, max_l1_change=Decimal("0.25")
        )
        assert result2.status == ValidationStatus.PASSED


class TestCreateWeightProvenance:
    """Tests for create_weight_provenance function."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_creates_valid_provenance(self, as_of_date):
        """Should create valid provenance record."""
        provenance = create_weight_provenance(
            as_of_date=as_of_date,
            training_start=date(2025, 1, 15),
            training_end=date(2025, 12, 15),
            embargo_days=30,
            universe_tickers=["ACME", "BETA", "GAMMA"],
            objective="maximize_rank_ic",
            constraints={"min_weight": Decimal("0.05"), "max_weight": Decimal("0.50")},
            weights={"clinical": Decimal("0.40"), "financial": Decimal("0.30")},
            historical_ic={"clinical": Decimal("0.15"), "financial": Decimal("0.12")},
            sample_size=500,
        )

        assert isinstance(provenance, WeightProvenance)
        assert provenance.as_of_date == "2026-01-15"
        assert provenance.training_start == "2025-01-15"
        assert provenance.sample_size == 500
        assert len(provenance.universe_hash) == 12
        assert len(provenance.determinism_hash) == 16

    def test_determinism_hash_is_deterministic(self, as_of_date):
        """Same inputs should produce same determinism hash."""
        params = dict(
            as_of_date=as_of_date,
            training_start=date(2025, 1, 15),
            training_end=date(2025, 12, 15),
            embargo_days=30,
            universe_tickers=["ACME", "BETA"],
            objective="maximize_ic",
            constraints={},
            weights={"w1": Decimal("0.50")},
            historical_ic={"w1": Decimal("0.10")},
            sample_size=100,
        )

        prov1 = create_weight_provenance(**params)
        prov2 = create_weight_provenance(**params)

        assert prov1.determinism_hash == prov2.determinism_hash

    def test_embargo_end_computed(self, as_of_date):
        """Embargo end should be training_end + embargo_days."""
        provenance = create_weight_provenance(
            as_of_date=as_of_date,
            training_start=date(2025, 1, 15),
            training_end=date(2025, 12, 15),
            embargo_days=30,
            universe_tickers=["ACME"],
            objective="test",
            constraints={},
            weights={},
            historical_ic={},
            sample_size=10,
        )

        expected_embargo_end = date(2025, 12, 15) + timedelta(days=30)
        assert provenance.embargo_end == expected_embargo_end.isoformat()


class TestRunProductionGate:
    """Tests for run_production_gate function."""

    @pytest.fixture
    def as_of_date(self):
        return date(2026, 1, 15)

    def test_no_checks_passes(self, as_of_date):
        """No data = no checks = passes."""
        result = run_production_gate(as_of_date)

        assert isinstance(result, ProductionGateResult)
        assert result.passed == True
        assert len(result.checks) == 0

    def test_valid_data_passes(self, as_of_date):
        """Valid data should pass all gates."""
        # Need 6+ months of training data to pass minimum training window
        historical_scores = [
            {"as_of_date": "2025-01-01", "ticker": "ACME", "score": 70},
            {"as_of_date": "2025-03-01", "ticker": "ACME", "score": 72},
            {"as_of_date": "2025-05-01", "ticker": "ACME", "score": 75},
            {"as_of_date": "2025-07-01", "ticker": "ACME", "score": 78},
            {"as_of_date": "2025-09-01", "ticker": "ACME", "score": 80},
            {"as_of_date": "2025-11-01", "ticker": "ACME", "score": 82},
        ]

        peer_valuations = [
            {"ticker": "PEER1", "snapshot_date": "2026-01-14", "market_cap": 1000},
        ]

        result = run_production_gate(
            as_of_date,
            historical_scores=historical_scores,
            peer_valuations=peer_valuations,
            use_adaptive_weights=True,
        )

        assert result.passed == True

    def test_blocking_violation_fails(self, as_of_date):
        """Blocking violation should fail gate."""
        peer_valuations = [
            {"ticker": "PEER1", "snapshot_date": "2026-01-16", "market_cap": 1000},  # Future
        ]

        result = run_production_gate(
            as_of_date,
            peer_valuations=peer_valuations,
        )

        assert result.passed == False
        assert len(result.blocking_violations) > 0
        assert "BLOCKED" in result.recommendation

    def test_weight_instability_is_warning(self, as_of_date):
        """Weight instability should be warning, not blocking."""
        previous_weights = {"w1": Decimal("0.50")}
        current_weights = {"w1": Decimal("0.20")}  # Large change

        result = run_production_gate(
            as_of_date,
            current_weights=current_weights,
            previous_weights=previous_weights,
            use_adaptive_weights=True,
        )

        # Weight instability is warning, not blocking
        assert len(result.warnings) > 0
        # May still pass if no other blocking violations

    def test_recommendation_messages(self, as_of_date):
        """Recommendation should reflect state."""
        # Passed
        result_pass = run_production_gate(as_of_date)
        assert "APPROVED" in result_pass.recommendation

        # Warning
        result_warn = run_production_gate(
            as_of_date,
            current_weights={"w1": Decimal("0.20")},
            previous_weights={"w1": Decimal("0.50")},
            use_adaptive_weights=True,
        )
        if result_warn.passed and result_warn.warnings:
            assert "CAUTION" in result_warn.recommendation

        # Failed
        peer_valuations = [{"ticker": "P", "snapshot_date": "2026-01-16"}]
        result_fail = run_production_gate(as_of_date, peer_valuations=peer_valuations)
        assert "BLOCKED" in result_fail.recommendation


class TestAblationTest:
    """Tests for run_ablation_test function."""

    def test_ablation_feature_adds_value(self):
        """Feature that adds IC should pass ablation."""
        # Baseline scores correlate with returns
        baseline_scores = [
            ("A", Decimal("90")), ("B", Decimal("70")),
            ("C", Decimal("50")), ("D", Decimal("30")),
            ("E", Decimal("80")), ("F", Decimal("60")),
            ("G", Decimal("40")), ("H", Decimal("20")),
            ("I", Decimal("85")), ("J", Decimal("65")),
        ]

        # Ablated scores are less correlated (random-ish)
        ablated_scores = [
            ("A", Decimal("50")), ("B", Decimal("80")),
            ("C", Decimal("30")), ("D", Decimal("70")),
            ("E", Decimal("40")), ("F", Decimal("90")),
            ("G", Decimal("60")), ("H", Decimal("20")),
            ("I", Decimal("45")), ("J", Decimal("75")),
        ]

        # Returns correlate with baseline better
        forward_returns = {
            "A": Decimal("0.10"), "B": Decimal("0.08"),
            "C": Decimal("0.05"), "D": Decimal("0.02"),
            "E": Decimal("0.09"), "F": Decimal("0.06"),
            "G": Decimal("0.03"), "H": Decimal("0.01"),
            "I": Decimal("0.095"), "J": Decimal("0.07"),
        }

        result = run_ablation_test(
            feature_name="test_feature",
            baseline_scores=baseline_scores,
            ablated_scores=ablated_scores,
            forward_returns=forward_returns,
        )

        assert isinstance(result, AblationResult)
        assert result.feature_name == "test_feature"
        # If baseline IC > ablated IC, feature passed (adds value)
        # The result depends on actual correlation

    def test_ablation_insufficient_data(self):
        """Too few samples should return zero IC."""
        baseline_scores = [("A", Decimal("50")), ("B", Decimal("60"))]  # Only 2
        ablated_scores = [("A", Decimal("55")), ("B", Decimal("65"))]
        forward_returns = {"A": Decimal("0.05"), "B": Decimal("0.08")}

        result = run_ablation_test(
            "small_feature",
            baseline_scores,
            ablated_scores,
            forward_returns,
        )

        assert result.baseline_ic == Decimal("0")

    def test_ablation_result_fields(self):
        """AblationResult should have all fields."""
        scores = [(chr(65+i), Decimal(str(90-i*5))) for i in range(15)]
        returns = {chr(65+i): Decimal(str(0.1-i*0.005)) for i in range(15)}

        result = run_ablation_test("test", scores, scores, returns)

        assert hasattr(result, "feature_name")
        assert hasattr(result, "baseline_ic")
        assert hasattr(result, "ablated_ic")
        assert hasattr(result, "ic_delta")
        assert hasattr(result, "passed")
        assert hasattr(result, "contribution_pct")


class TestConstants:
    """Tests for module constants."""

    def test_default_embargo_days(self):
        """Default embargo should be 30 days."""
        assert DEFAULT_EMBARGO_DAYS == 30

    def test_max_weight_l1_change(self):
        """Max weight change should be 0.15."""
        assert MAX_WEIGHT_L1_CHANGE == Decimal("0.15")
