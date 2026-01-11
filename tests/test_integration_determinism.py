"""
Integration Tests for Determinism and End-to-End Pipeline

Tests that verify:
1. Same inputs produce identical outputs (determinism)
2. Risk gates integrate correctly with institutional scoring
3. Checkpointing produces consistent results
4. Parameter hashes are stable
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from risk_gates import (
    apply_all_gates,
    calculate_adv,
    calculate_runway_months,
    get_parameters_snapshot as get_risk_params,
    compute_parameters_hash as risk_params_hash,
    FLAG_ADV_UNKNOWN,
    FLAG_LOW_LIQUIDITY,
    FLAG_PENNY_STOCK,
    FLAG_MICRO_CAP,
    FLAG_CASH_RISK,
)

from liquidity_scoring import (
    compute_liquidity_score,
    score_all_tickers,
    get_parameters_snapshot as get_liq_params,
    compute_parameters_hash as liq_params_hash,
    LIQUIDITY_SCORING_VERSION,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "GOOD": {
            "ticker": "GOOD",
            "price": 50.0,
            "avg_volume": 1_000_000,  # ADV = $50M
            "market_cap": 5_000_000_000,  # $5B = mid
            "spread_bps": 50.0,
        },
        "LOW_LIQ": {
            "ticker": "LOW_LIQ",
            "price": 10.0,
            "avg_volume": 10_000,  # ADV = $100K
            "market_cap": 500_000_000,  # $500M = small
        },
        "PENNY": {
            "ticker": "PENNY",
            "price": 1.50,  # Below $2
            "avg_volume": 5_000_000,
            "market_cap": 100_000_000,
        },
        "MICRO": {
            "ticker": "MICRO",
            "price": 5.0,
            "avg_volume": 500_000,
            "market_cap": 30_000_000,  # Below $50M
        },
    }


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "GOOD": {
            "ticker": "GOOD",
            "Cash": 500_000_000,
            "NetIncome": 50_000_000,  # Profitable
        },
        "CASH_RISK": {
            "ticker": "CASH_RISK",
            "Cash": 50_000_000,
            "NetIncome": -100_000_000,  # 1.5 month runway
        },
        "PROFITABLE": {
            "ticker": "PROFITABLE",
            "Cash": 100_000_000,
            "NetIncome": 20_000_000,  # Profitable
        },
    }


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests that verify deterministic behavior."""

    def test_risk_gates_deterministic(self, sample_market_data, sample_financial_data):
        """Same inputs produce identical risk gate results."""
        for _ in range(5):
            result1 = apply_all_gates("GOOD", sample_market_data, sample_financial_data)
            result2 = apply_all_gates("GOOD", sample_market_data, sample_financial_data)

            assert result1 == result2

    def test_liquidity_score_deterministic(self, sample_market_data):
        """Same inputs produce identical liquidity scores."""
        for _ in range(5):
            result1 = compute_liquidity_score("GOOD", sample_market_data)
            result2 = compute_liquidity_score("GOOD", sample_market_data)

            assert result1 == result2

    def test_parameters_hash_stable(self):
        """Parameter hashes are stable across calls."""
        hashes_risk = [risk_params_hash() for _ in range(10)]
        hashes_liq = [liq_params_hash() for _ in range(10)]

        assert len(set(hashes_risk)) == 1, "Risk gates hash not stable"
        assert len(set(hashes_liq)) == 1, "Liquidity scoring hash not stable"

    def test_risk_flags_sorted(self, sample_market_data, sample_financial_data):
        """Risk flags are always sorted for determinism."""
        # Create ticker with multiple flags
        sample_market_data["MULTI"] = {
            "ticker": "MULTI",
            "price": 1.50,  # Penny stock
            "avg_volume": 1000,  # Low liquidity
            "market_cap": 30_000_000,  # Micro cap
        }
        sample_financial_data["MULTI"] = {
            "ticker": "MULTI",
            "Cash": 10_000_000,
            "NetIncome": -50_000_000,  # Cash risk
        }

        for _ in range(5):
            result = apply_all_gates("MULTI", sample_market_data, sample_financial_data)
            assert result["risk_flags"] == sorted(result["risk_flags"])

    def test_score_all_tickers_sorted(self, sample_market_data):
        """score_all_tickers returns sorted results."""
        tickers = ["PENNY", "GOOD", "LOW_LIQ", "MICRO"]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit_path = f.name

        try:
            results = score_all_tickers(
                tickers, sample_market_data, "2024-01-15", audit_path
            )

            # Results should be sorted by ticker
            result_tickers = [r["ticker"] for r in results]
            assert result_tickers == sorted(result_tickers)

            # Audit log should exist and have entries
            audit_path_obj = Path(audit_path)
            assert audit_path_obj.exists()

            lines = audit_path_obj.read_text().strip().split('\n')
            assert len(lines) == len(tickers)

            # Each audit entry should be valid JSON with sorted keys
            for line in lines:
                record = json.loads(line)
                assert "ticker" in record
                assert "score_version" in record
        finally:
            Path(audit_path).unlink(missing_ok=True)


# =============================================================================
# RISK GATE INTEGRATION TESTS
# =============================================================================

class TestRiskGateIntegration:
    """Tests for risk gate integration."""

    def test_adv_calculation_priority(self):
        """ADV calculation uses correct field priority."""
        # Direct ADV field takes priority
        data1 = {"TEST": {"ticker": "TEST", "adv_usd_20d": 5_000_000}}
        assert calculate_adv("TEST", data1) == 5_000_000

        # Fallback to volume * price
        data2 = {"TEST": {"ticker": "TEST", "avg_volume": 100_000, "price": 50.0}}
        assert calculate_adv("TEST", data2) == 5_000_000

    def test_runway_profitable_company(self, sample_financial_data):
        """Profitable companies get high runway."""
        runway = calculate_runway_months("PROFITABLE", sample_financial_data)
        assert runway == 999.0

    def test_runway_cash_burn(self, sample_financial_data):
        """Cash burn companies get correct runway."""
        runway = calculate_runway_months("CASH_RISK", sample_financial_data)
        # $50M / ($100M * 4/12) = 50 / 33.33 = 1.5 months
        assert runway is not None
        assert 1.0 < runway < 2.0

    def test_gate_composition(self, sample_market_data, sample_financial_data):
        """Gates correctly compose multiple checks."""
        # GOOD ticker passes all gates
        result = apply_all_gates("GOOD", sample_market_data, sample_financial_data)
        assert result["passes"] is True
        assert len(result["rejection_reasons"]) == 0

        # LOW_LIQ fails liquidity gate
        result = apply_all_gates("LOW_LIQ", sample_market_data, sample_financial_data)
        assert result["passes"] is False
        assert FLAG_LOW_LIQUIDITY in result["rejection_reasons"]

        # PENNY fails penny stock gate
        result = apply_all_gates("PENNY", sample_market_data, sample_financial_data)
        assert result["passes"] is False
        assert FLAG_PENNY_STOCK in result["rejection_reasons"]

        # MICRO fails micro cap gate
        result = apply_all_gates("MICRO", sample_market_data, sample_financial_data)
        assert result["passes"] is False
        assert FLAG_MICRO_CAP in result["rejection_reasons"]

        # CASH_RISK fails financial gate
        sample_market_data["CASH_RISK"] = {
            "ticker": "CASH_RISK",
            "price": 50.0,
            "avg_volume": 1_000_000,
            "market_cap": 1_000_000_000,
        }
        result = apply_all_gates("CASH_RISK", sample_market_data, sample_financial_data)
        assert result["passes"] is False
        assert FLAG_CASH_RISK in result["rejection_reasons"]


# =============================================================================
# LIQUIDITY SCORING INTEGRATION TESTS
# =============================================================================

class TestLiquidityScoringIntegration:
    """Tests for liquidity scoring integration."""

    def test_tier_classification_boundaries(self):
        """Market cap tiers have correct boundaries."""
        from liquidity_scoring import classify_market_cap_tier

        # Micro cap: < $300M
        assert classify_market_cap_tier(299_000_000) == "micro"
        assert classify_market_cap_tier(300_000_000) == "small"

        # Small cap: $300M - $2B
        assert classify_market_cap_tier(1_999_000_000) == "small"
        assert classify_market_cap_tier(2_000_000_000) == "mid"

        # Mid cap: $2B - $10B
        assert classify_market_cap_tier(9_999_000_000) == "mid"
        assert classify_market_cap_tier(10_000_000_000) == "large"

    def test_adv_threshold_by_tier(self):
        """ADV thresholds vary by tier."""
        from liquidity_scoring import get_adv_threshold_for_tier

        assert get_adv_threshold_for_tier("micro") == 750_000
        assert get_adv_threshold_for_tier("small") == 2_000_000
        assert get_adv_threshold_for_tier("mid") == 5_000_000
        assert get_adv_threshold_for_tier("large") == 10_000_000

    def test_score_components_sum(self, sample_market_data):
        """ADV + spread scores sum to total."""
        result = compute_liquidity_score("GOOD", sample_market_data)

        assert result["liquidity_score"] == result["adv_score"] + result["spread_score"]

    def test_penny_stock_caps_score(self, sample_market_data):
        """Penny stock penalty caps score at 10."""
        result = compute_liquidity_score("PENNY", sample_market_data)

        assert result["liquidity_score"] <= 10
        assert FLAG_PENNY_STOCK in result["risk_flags"]


# =============================================================================
# PARAMETER SNAPSHOT TESTS
# =============================================================================

class TestParameterSnapshots:
    """Tests for parameter snapshot functionality."""

    def test_risk_params_complete(self):
        """Risk params snapshot has all required fields."""
        params = get_risk_params()

        required_fields = [
            "version",
            "ADV_MINIMUM",
            "PRICE_MINIMUM",
            "MARKET_CAP_MINIMUM",
            "RUNWAY_MINIMUM_MONTHS",
        ]

        for field in required_fields:
            assert field in params, f"Missing field: {field}"

    def test_liq_params_complete(self):
        """Liquidity params snapshot has all required fields."""
        params = get_liq_params()

        required_fields = [
            "version",
            "TIER_MICRO_MAX",
            "TIER_SMALL_MAX",
            "TIER_MID_MAX",
            "ADV_THRESHOLDS",
            "SPREAD_LOW_BPS",
            "SPREAD_HIGH_BPS",
            "PENNY_STOCK_PRICE",
            "PENNY_STOCK_MAX_SCORE",
            "ADV_MAX_SCORE",
            "SPREAD_MAX_SCORE",
        ]

        for field in required_fields:
            assert field in params, f"Missing field: {field}"

    def test_hash_length(self):
        """Parameter hashes have correct length."""
        assert len(risk_params_hash()) == 16
        assert len(liq_params_hash()) == 16

    def test_hash_hex_format(self):
        """Parameter hashes are valid hex strings."""
        import re

        hex_pattern = re.compile(r'^[0-9a-f]{16}$')

        assert hex_pattern.match(risk_params_hash())
        assert hex_pattern.match(liq_params_hash())


# =============================================================================
# AUDIT TRAIL TESTS
# =============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_record_structure(self, sample_market_data):
        """Audit records have correct structure."""
        from liquidity_scoring import create_liquidity_audit_record

        score_result = compute_liquidity_score("GOOD", sample_market_data)
        audit = create_liquidity_audit_record("GOOD", score_result, "2024-01-15")

        required_fields = [
            "as_of_date",
            "ticker",
            "score_version",
            "thresholds_by_tier",
            "chosen_tier",
            "adv_usd",
            "spread_bps",
            "adv_score",
            "spread_score",
            "liquidity_score",
            "risk_flags",
        ]

        for field in required_fields:
            assert field in audit, f"Missing field: {field}"

    def test_audit_log_jsonl_format(self, sample_market_data):
        """Audit log uses JSONL format."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit_path = f.name

        try:
            score_all_tickers(
                ["GOOD", "LOW_LIQ"],
                sample_market_data,
                "2024-01-15",
                audit_path
            )

            with open(audit_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == 2

            for line in lines:
                # Each line should be valid JSON
                record = json.loads(line.strip())
                assert "ticker" in record
                assert "score_version" in record
        finally:
            Path(audit_path).unlink(missing_ok=True)

    def test_audit_keys_sorted(self, sample_market_data):
        """Audit records have sorted keys for determinism."""
        from liquidity_scoring import append_audit_log

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit_path = f.name

        try:
            record = {"z_field": 1, "a_field": 2, "m_field": 3}
            append_audit_log(audit_path, record)

            with open(audit_path, 'r') as f:
                line = f.read().strip()

            # Keys should appear in sorted order in the JSON string
            assert line.index("a_field") < line.index("m_field") < line.index("z_field")
        finally:
            Path(audit_path).unlink(missing_ok=True)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_missing_ticker_in_market_data(self):
        """Missing ticker returns fail-closed result."""
        result = apply_all_gates("MISSING", {}, {})

        assert result["passes"] is False
        assert result["adv_usd"] == 0.0

    def test_zero_market_cap(self):
        """Zero market cap is handled correctly."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(0) == "unknown"
        assert classify_market_cap_tier(-100) == "unknown"
        assert classify_market_cap_tier(None) == "unknown"

    def test_negative_values_handled(self):
        """Negative values don't cause crashes."""
        data = {
            "NEG": {
                "ticker": "NEG",
                "price": -10.0,
                "avg_volume": -1000,
                "market_cap": -1_000_000,
            }
        }

        # Should not raise exceptions
        result = compute_liquidity_score("NEG", data)
        assert result["liquidity_score"] == 0

        gate_result = apply_all_gates("NEG", data, {})
        assert gate_result["passes"] is False

    def test_very_large_values(self):
        """Very large values are handled correctly."""
        data = {
            "BIG": {
                "ticker": "BIG",
                "price": 1_000_000.0,
                "avg_volume": 1_000_000_000,
                "market_cap": 1_000_000_000_000,  # $1T
                "spread_bps": 10.0,  # Tight spread for max score
            }
        }

        result = compute_liquidity_score("BIG", data)
        assert result["liquidity_tier"] == "large"
        # Max ADV score (70) + max spread score (30) = 100
        assert result["liquidity_score"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
