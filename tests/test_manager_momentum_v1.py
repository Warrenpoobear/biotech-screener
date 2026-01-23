#!/usr/bin/env python3
"""
Tests for Manager Momentum Engine (v1)

Covers:
- Position change classification
- Momentum score computation
- Coordinated activity detection
- Crowding level analysis
- Determinism guarantees
"""

import pytest
from datetime import date
from pathlib import Path
from decimal import Decimal

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from manager_momentum_v1 import (
    ConvictionChange,
    CrowdingLevel,
    ManagerPosition,
    TickerMomentum,
    _to_decimal,
    _quantize_score,
    _compute_change_pct,
    _classify_change,
    _compute_determinism_hash,
    analyze_ticker_momentum,
    compute_manager_momentum,
    get_momentum_validation,
    SHARE_CHANGE_THRESHOLD,
    COORDINATED_ADD_MIN,
    CROWDING_THRESHOLD,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def manager_registry():
    """Sample manager registry."""
    return {
        "elite_core": [
            {"cik": "CIK001", "name": "Elite Fund A"},
            {"cik": "CIK002", "name": "Elite Fund B"},
        ],
        "conditional": [
            {"cik": "CIK003", "name": "Conditional Fund C"},
            {"cik": "CIK004", "name": "Conditional Fund D"},
        ],
    }


@pytest.fixture
def holdings_data_new_position():
    """Holdings data with new position."""
    return {
        "ACME": {
            "holdings": {
                "current": {
                    "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-12-31"},
                },
                "prior": {},
            },
        },
    }


@pytest.fixture
def holdings_data_add_position():
    """Holdings data with position increase."""
    return {
        "ACME": {
            "holdings": {
                "current": {
                    "CIK001": {"shares": 15000, "value_kusd": 750, "quarter_end": "2025-12-31"},
                },
                "prior": {
                    "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-09-30"},
                },
            },
        },
    }


@pytest.fixture
def holdings_data_trim_position():
    """Holdings data with position decrease."""
    return {
        "ACME": {
            "holdings": {
                "current": {
                    "CIK001": {"shares": 5000, "value_kusd": 250, "quarter_end": "2025-12-31"},
                },
                "prior": {
                    "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-09-30"},
                },
            },
        },
    }


@pytest.fixture
def holdings_data_exit_position():
    """Holdings data with position exit."""
    return {
        "ACME": {
            "holdings": {
                "current": {},
                "prior": {
                    "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-09-30"},
                },
            },
        },
    }


@pytest.fixture
def holdings_data_coordinated(manager_registry):
    """Holdings data with coordinated buying."""
    return {
        "ACME": {
            "holdings": {
                "current": {
                    "CIK001": {"shares": 15000, "value_kusd": 750, "quarter_end": "2025-12-31"},
                    "CIK002": {"shares": 12000, "value_kusd": 600, "quarter_end": "2025-12-31"},
                    "CIK003": {"shares": 8000, "value_kusd": 400, "quarter_end": "2025-12-31"},
                },
                "prior": {
                    "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-09-30"},
                    "CIK002": {"shares": 8000, "value_kusd": 400, "quarter_end": "2025-09-30"},
                    "CIK003": {"shares": 5000, "value_kusd": 250, "quarter_end": "2025-09-30"},
                },
            },
        },
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_decimal_from_int(self):
        """Converts int to Decimal."""
        assert _to_decimal(100) == Decimal("100")

    def test_to_decimal_from_float(self):
        """Converts float to Decimal."""
        result = _to_decimal(3.14)
        assert isinstance(result, Decimal)
        assert result == Decimal("3.14")

    def test_to_decimal_from_string(self):
        """Converts string to Decimal."""
        assert _to_decimal("123.45") == Decimal("123.45")

    def test_to_decimal_from_decimal(self):
        """Returns Decimal unchanged."""
        d = Decimal("99.99")
        assert _to_decimal(d) is d

    def test_to_decimal_none_returns_default(self):
        """None returns default value."""
        assert _to_decimal(None) is None
        assert _to_decimal(None, Decimal("0")) == Decimal("0")

    def test_to_decimal_invalid_returns_default(self):
        """Invalid value returns default."""
        assert _to_decimal("not a number", Decimal("0")) == Decimal("0")

    def test_quantize_score(self):
        """Quantizes score to 2 decimal places."""
        result = _quantize_score(Decimal("12.3456789"))
        assert result == Decimal("12.35")

    def test_compute_change_pct_positive(self):
        """Computes positive change percentage."""
        result = _compute_change_pct(150, 100)
        assert result == Decimal("50.00")

    def test_compute_change_pct_negative(self):
        """Computes negative change percentage."""
        result = _compute_change_pct(50, 100)
        assert result == Decimal("-50.00")

    def test_compute_change_pct_zero_prior(self):
        """Returns None for zero prior."""
        result = _compute_change_pct(100, 0)
        assert result is None


# ============================================================================
# CHANGE CLASSIFICATION
# ============================================================================

class TestChangeClassification:
    """Tests for position change classification."""

    def test_classify_new_position(self):
        """Classifies new position."""
        change, share_pct, value_pct = _classify_change(
            current_shares=10000, prior_shares=None,
            current_value=500, prior_value=None
        )
        assert change == ConvictionChange.NEW

    def test_classify_exit_position(self):
        """Classifies exited position."""
        change, share_pct, value_pct = _classify_change(
            current_shares=0, prior_shares=10000,
            current_value=0, prior_value=500
        )
        assert change == ConvictionChange.EXIT
        assert share_pct == Decimal("-100")

    def test_classify_add_position(self):
        """Classifies added position (>10% increase)."""
        change, share_pct, value_pct = _classify_change(
            current_shares=12000, prior_shares=10000,  # 20% increase
            current_value=600, prior_value=500
        )
        assert change == ConvictionChange.ADD

    def test_classify_trim_position(self):
        """Classifies trimmed position (>10% decrease)."""
        change, share_pct, value_pct = _classify_change(
            current_shares=8000, prior_shares=10000,  # 20% decrease
            current_value=400, prior_value=500
        )
        assert change == ConvictionChange.TRIM

    def test_classify_hold_position(self):
        """Classifies held position (<10% change)."""
        change, share_pct, value_pct = _classify_change(
            current_shares=10500, prior_shares=10000,  # 5% increase
            current_value=525, prior_value=500
        )
        assert change == ConvictionChange.HOLD

    def test_classify_unknown_with_zero_prior(self):
        """Classifies as unknown when prior is zero but current is also zero."""
        change, share_pct, value_pct = _classify_change(
            current_shares=0, prior_shares=0,
            current_value=0, prior_value=0
        )
        assert change == ConvictionChange.UNKNOWN


# ============================================================================
# TICKER MOMENTUM ANALYSIS
# ============================================================================

class TestTickerMomentumAnalysis:
    """Tests for analyze_ticker_momentum function."""

    def test_new_position_analysis(self, holdings_data_new_position, manager_registry):
        """Analyzes new position correctly."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_new_position["ACME"],
            manager_registry
        )

        assert result.ticker == "ACME"
        assert result.new_count == 1
        assert result.momentum_score > 0

    def test_add_position_analysis(self, holdings_data_add_position, manager_registry):
        """Analyzes added position correctly."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_add_position["ACME"],
            manager_registry
        )

        assert result.add_count == 1
        assert result.momentum_score > 0

    def test_trim_position_analysis(self, holdings_data_trim_position, manager_registry):
        """Analyzes trimmed position correctly."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_trim_position["ACME"],
            manager_registry
        )

        assert result.trim_count == 1
        assert result.momentum_score < 0

    def test_exit_position_analysis(self, holdings_data_exit_position, manager_registry):
        """Analyzes exited position correctly."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_exit_position["ACME"],
            manager_registry
        )

        assert result.exit_count == 1
        assert result.momentum_score < 0

    def test_coordinated_buying_detection(self, holdings_data_coordinated, manager_registry):
        """Detects coordinated buying."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_coordinated["ACME"],
            manager_registry
        )

        assert result.coordinated_buying is True

    def test_crowding_level_calculation(self, holdings_data_coordinated, manager_registry):
        """Calculates crowding level."""
        result = analyze_ticker_momentum(
            "ACME",
            holdings_data_coordinated["ACME"],
            manager_registry
        )

        assert result.total_managers == 3
        assert result.crowding_level in [CrowdingLevel.LOW, CrowdingLevel.MODERATE]


# ============================================================================
# COMPUTE MANAGER MOMENTUM
# ============================================================================

class TestComputeManagerMomentum:
    """Tests for compute_manager_momentum function."""

    def test_computes_for_all_tickers(self, holdings_data_coordinated, manager_registry):
        """Computes momentum for all tickers."""
        result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        assert result["tickers_analyzed"] == 1
        assert "ACME" in result["signals"]

    def test_result_structure(self, holdings_data_coordinated, manager_registry):
        """Result has expected structure."""
        result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        assert "as_of_date" in result
        assert "schema_version" in result
        assert "summary" in result
        assert "rankings" in result
        assert "signals" in result
        assert "provenance" in result

    def test_summary_aggregation(self, holdings_data_coordinated, manager_registry):
        """Summary aggregates signals correctly."""
        result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        summary = result["summary"]
        assert "coordinated_buys" in summary
        assert "coordinated_sells" in summary
        assert "fresh_convictions" in summary
        assert "crowded_positions" in summary

    def test_target_tickers_filter(self, holdings_data_coordinated, manager_registry):
        """Respects target_tickers filter."""
        result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15",
            target_tickers=["ACME"]
        )

        assert result["tickers_analyzed"] == 1

    def test_rankings_sorted_by_momentum(self, holdings_data_coordinated, manager_registry):
        """Rankings are sorted by momentum score descending."""
        result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        rankings = result["rankings"]
        if len(rankings) > 1:
            for i in range(1, len(rankings)):
                assert Decimal(rankings[i-1]["momentum_score"]) >= Decimal(rankings[i]["momentum_score"])


# ============================================================================
# MOMENTUM VALIDATION
# ============================================================================

class TestMomentumValidation:
    """Tests for get_momentum_validation function."""

    def test_validation_structure(self, holdings_data_coordinated, manager_registry):
        """Validation result has expected structure."""
        momentum_result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        validation = get_momentum_validation("ACME", momentum_result)

        assert "momentum_score" in validation
        assert "momentum_confidence" in validation
        assert "momentum_flags" in validation
        assert "momentum_adjustment" in validation

    def test_unknown_ticker_returns_zero(self, holdings_data_coordinated, manager_registry):
        """Unknown ticker returns zero values."""
        momentum_result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        validation = get_momentum_validation("UNKNOWN", momentum_result)

        assert validation["momentum_score"] == Decimal("0")
        assert validation["momentum_confidence"] == Decimal("0")

    def test_adjustment_bounded(self, holdings_data_coordinated, manager_registry):
        """Adjustment is bounded to +/- 5."""
        momentum_result = compute_manager_momentum(
            holdings_data_coordinated,
            manager_registry,
            "2026-01-15"
        )

        validation = get_momentum_validation("ACME", momentum_result)

        assert Decimal("-5") <= validation["momentum_adjustment"] <= Decimal("5")


# ============================================================================
# DATA CLASSES
# ============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_manager_position_to_dict(self):
        """ManagerPosition serializes to dict."""
        position = ManagerPosition(
            manager_cik="CIK001",
            manager_name="Test Fund",
            manager_tier="elite_core",
            quarter_end="2025-12-31",
            shares=10000,
            value_kusd=500,
            prior_shares=8000,
            prior_value_kusd=400,
            change=ConvictionChange.ADD,
            share_change_pct=Decimal("25.00"),
            value_change_pct=Decimal("25.00"),
        )

        d = position.to_dict()

        assert d["manager_cik"] == "CIK001"
        assert d["change"] == "ADD"
        assert d["share_change_pct"] == "25.00"


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_momentum_analysis_deterministic(self, holdings_data_coordinated, manager_registry):
        """Momentum analysis is deterministic."""
        result1 = analyze_ticker_momentum("ACME", holdings_data_coordinated["ACME"], manager_registry)
        result2 = analyze_ticker_momentum("ACME", holdings_data_coordinated["ACME"], manager_registry)

        assert result1.momentum_score == result2.momentum_score
        assert result1.determinism_hash == result2.determinism_hash

    def test_compute_momentum_deterministic(self, holdings_data_coordinated, manager_registry):
        """Full momentum computation is deterministic."""
        result1 = compute_manager_momentum(holdings_data_coordinated, manager_registry, "2026-01-15")
        result2 = compute_manager_momentum(holdings_data_coordinated, manager_registry, "2026-01-15")

        assert result1["signals"]["ACME"]["momentum_score"] == result2["signals"]["ACME"]["momentum_score"]

    def test_determinism_hash_stable(self, holdings_data_coordinated, manager_registry):
        """Determinism hash is stable across runs."""
        result1 = analyze_ticker_momentum("ACME", holdings_data_coordinated["ACME"], manager_registry)
        result2 = analyze_ticker_momentum("ACME", holdings_data_coordinated["ACME"], manager_registry)

        assert result1.determinism_hash == result2.determinism_hash


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_holdings(self, manager_registry):
        """Handles empty holdings."""
        holdings_data = {
            "ACME": {
                "holdings": {
                    "current": {},
                    "prior": {},
                },
            },
        }

        result = analyze_ticker_momentum("ACME", holdings_data["ACME"], manager_registry)

        assert result.total_managers == 0
        assert result.momentum_score == Decimal("0")

    def test_non_elite_managers_ignored(self, manager_registry):
        """Non-elite managers are ignored."""
        holdings_data = {
            "ACME": {
                "holdings": {
                    "current": {
                        "CIK999": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-12-31"},
                    },
                    "prior": {},
                },
            },
        }

        result = analyze_ticker_momentum("ACME", holdings_data["ACME"], manager_registry)

        assert result.total_managers == 0

    def test_all_exits(self, manager_registry):
        """Handles all managers exiting."""
        holdings_data = {
            "ACME": {
                "holdings": {
                    "current": {},
                    "prior": {
                        "CIK001": {"shares": 10000, "value_kusd": 500, "quarter_end": "2025-09-30"},
                        "CIK002": {"shares": 8000, "value_kusd": 400, "quarter_end": "2025-09-30"},
                    },
                },
            },
        }

        result = analyze_ticker_momentum("ACME", holdings_data["ACME"], manager_registry)

        assert result.exit_count == 2
        assert result.momentum_score < 0

