"""
Unit tests for risk_gates.py

Tests the fail-closed risk gate module including:
- ADV calculation and ADV_UNKNOWN handling
- LOW_LIQUIDITY thresholding
- PENNY_STOCK gating
- MICRO_CAP gating
- CASH_RISK gating
- Mixed case scenarios
"""
import pytest
import json
import tempfile
from pathlib import Path

from risk_gates import (
    load_market_data,
    load_financial_data,
    calculate_adv,
    calculate_runway_months,
    apply_liquidity_gate,
    apply_financial_gate,
    apply_all_gates,
    get_parameters_snapshot,
    compute_parameters_hash,
    ADV_MINIMUM,
    PRICE_MINIMUM,
    MARKET_CAP_MINIMUM,
    RUNWAY_MINIMUM_MONTHS,
    FLAG_ADV_UNKNOWN,
    FLAG_LOW_LIQUIDITY,
    FLAG_PENNY_STOCK,
    FLAG_MICRO_CAP,
    FLAG_CASH_RISK,
    FLAG_RUNWAY_UNKNOWN,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "GOOD": {
            "ticker": "GOOD",
            "price": 50.0,
            "avg_volume": 1_000_000,  # ADV = $50M
            "market_cap": 5_000_000_000,  # $5B
        },
        "LOW_ADV": {
            "ticker": "LOW_ADV",
            "price": 50.0,
            "avg_volume": 5_000,  # ADV = $250K (below minimum)
            "market_cap": 1_000_000_000,
        },
        "PENNY": {
            "ticker": "PENNY",
            "price": 1.50,  # Below $2 threshold
            "avg_volume": 1_000_000,  # ADV = $1.5M
            "market_cap": 500_000_000,
        },
        "MICRO": {
            "ticker": "MICRO",
            "price": 5.0,
            "avg_volume": 500_000,  # ADV = $2.5M
            "market_cap": 30_000_000,  # Below $50M threshold
        },
        "NO_VOLUME": {
            "ticker": "NO_VOLUME",
            "price": 50.0,
            # No volume data - ADV unknown
            "market_cap": 1_000_000_000,
        },
        "NO_PRICE": {
            "ticker": "NO_PRICE",
            "avg_volume": 1_000_000,
            # No price data - ADV unknown
            "market_cap": 1_000_000_000,
        },
        "DIRECT_ADV": {
            "ticker": "DIRECT_ADV",
            "adv_usd_20d": 10_000_000,  # Direct ADV field
            "price": 100.0,
            "avg_volume": 50_000,  # Should be ignored
            "market_cap": 2_000_000_000,
        },
        "GOSS_COLLAPSE": {
            "ticker": "GOSS_COLLAPSE",
            "price": 3.0,  # After volume collapse
            "avg_volume": 100_000,  # ADV = $300K (below minimum)
            "market_cap": 100_000_000,
        },
    }


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "HEALTHY": {
            "ticker": "HEALTHY",
            "Cash": 500_000_000,
            "NetIncome": -50_000_000,  # Quarterly loss
            # Implied burn: 200M/year = 16.7M/month
            # Runway: 500M / 16.7M = ~30 months
        },
        "CASH_RISK_TICKER": {
            "ticker": "CASH_RISK_TICKER",
            "Cash": 20_000_000,
            "NetIncome": -40_000_000,  # Quarterly loss
            # Implied burn: 160M/year = 13.3M/month
            # Runway: 20M / 13.3M = ~1.5 months
        },
        "PROFITABLE": {
            "ticker": "PROFITABLE",
            "Cash": 100_000_000,
            "NetIncome": 50_000_000,  # Positive = no burn
            # Should get 999 months (no cash risk)
        },
        "DIRECT_RUNWAY": {
            "ticker": "DIRECT_RUNWAY",
            "runway_months": 24.0,  # Direct field
            "Cash": 100_000_000,
        },
        "NO_CASH": {
            "ticker": "NO_CASH",
            # No Cash field
            "NetIncome": -10_000_000,
        },
        "RD_ONLY": {
            "ticker": "RD_ONLY",
            "Cash": 100_000_000,
            "R&D": 25_000_000,  # Quarterly R&D as burn proxy
            # Implied burn: 100M/year = 8.3M/month
            # Runway: 100M / 8.3M = ~12 months
        },
    }


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestDataLoading:
    def test_load_market_data_list_format(self, tmp_path):
        """Load market data from list format."""
        data = [
            {"ticker": "AAPL", "price": 150.0},
            {"ticker": "MSFT", "price": 300.0},
        ]
        filepath = tmp_path / "market_data.json"
        filepath.write_text(json.dumps(data))

        result = load_market_data(str(filepath))

        assert "AAPL" in result
        assert result["AAPL"]["price"] == 150.0
        assert "MSFT" in result

    def test_load_financial_data_list_format(self, tmp_path):
        """Load financial data from list format."""
        data = [
            {"ticker": "AAPL", "Cash": 100_000_000},
            {"ticker": "MSFT", "Cash": 200_000_000},
        ]
        filepath = tmp_path / "financial_data.json"
        filepath.write_text(json.dumps(data))

        result = load_financial_data(str(filepath))

        assert "AAPL" in result
        assert result["AAPL"]["Cash"] == 100_000_000


# =============================================================================
# ADV CALCULATION TESTS
# =============================================================================

class TestADVCalculation:
    def test_adv_from_volume_and_price(self, sample_market_data):
        """Calculate ADV from avg_volume * price."""
        adv = calculate_adv("GOOD", sample_market_data)
        assert adv == 50_000_000  # 1M shares * $50

    def test_adv_direct_field_priority(self, sample_market_data):
        """Direct ADV field takes priority over computed."""
        adv = calculate_adv("DIRECT_ADV", sample_market_data)
        assert adv == 10_000_000  # Direct field, not 50K * 100

    def test_adv_unknown_no_volume(self, sample_market_data):
        """ADV = 0 when volume is missing (fail-closed)."""
        adv = calculate_adv("NO_VOLUME", sample_market_data)
        assert adv == 0.0

    def test_adv_unknown_no_price(self, sample_market_data):
        """ADV = 0 when price is missing (fail-closed)."""
        adv = calculate_adv("NO_PRICE", sample_market_data)
        assert adv == 0.0

    def test_adv_unknown_ticker_not_found(self, sample_market_data):
        """ADV = 0 when ticker not in data (fail-closed)."""
        adv = calculate_adv("NONEXISTENT", sample_market_data)
        assert adv == 0.0

    def test_goss_proof_volume_collapse(self, sample_market_data):
        """Test GOSS-like scenario with volume collapse."""
        adv = calculate_adv("GOSS_COLLAPSE", sample_market_data)
        assert adv == 300_000  # 100K * $3


# =============================================================================
# RUNWAY CALCULATION TESTS
# =============================================================================

class TestRunwayCalculation:
    def test_runway_from_net_income(self, sample_financial_data):
        """Calculate runway from negative net income."""
        runway = calculate_runway_months("HEALTHY", sample_financial_data)
        # 500M / (50M * 4 / 12) = 500M / 16.67M = ~30
        assert runway is not None
        assert 29 < runway < 31

    def test_runway_cash_risk(self, sample_financial_data):
        """Detect cash risk when runway is low."""
        runway = calculate_runway_months("CASH_RISK_TICKER", sample_financial_data)
        # 20M / (40M * 4 / 12) = 20M / 13.33M = ~1.5
        assert runway is not None
        assert runway < RUNWAY_MINIMUM_MONTHS

    def test_runway_profitable(self, sample_financial_data):
        """Profitable companies get high runway."""
        runway = calculate_runway_months("PROFITABLE", sample_financial_data)
        assert runway == 999.0

    def test_runway_direct_field(self, sample_financial_data):
        """Direct runway_months field is used."""
        runway = calculate_runway_months("DIRECT_RUNWAY", sample_financial_data)
        assert runway == 24.0

    def test_runway_unknown_no_cash(self, sample_financial_data):
        """Runway is None when Cash is missing."""
        runway = calculate_runway_months("NO_CASH", sample_financial_data)
        assert runway is None

    def test_runway_rd_fallback(self, sample_financial_data):
        """Use R&D as burn rate proxy when no NetIncome."""
        runway = calculate_runway_months("RD_ONLY", sample_financial_data)
        # 100M / (25M * 4 / 12) = 100M / 8.33M = ~12
        assert runway is not None
        assert 11 < runway < 13


# =============================================================================
# LIQUIDITY GATE TESTS
# =============================================================================

class TestLiquidityGate:
    def test_passes_all_checks(self, sample_market_data):
        """GOOD ticker passes all liquidity checks."""
        passes, reason = apply_liquidity_gate("GOOD", sample_market_data)
        assert passes is True
        assert reason is None

    def test_fails_adv_unknown(self, sample_market_data):
        """Fail-closed when ADV cannot be computed."""
        passes, reason = apply_liquidity_gate("NO_VOLUME", sample_market_data)
        assert passes is False
        assert reason == FLAG_ADV_UNKNOWN

    def test_fails_low_liquidity(self, sample_market_data):
        """Fail when ADV below threshold."""
        passes, reason = apply_liquidity_gate("LOW_ADV", sample_market_data)
        assert passes is False
        assert reason == FLAG_LOW_LIQUIDITY

    def test_fails_penny_stock(self, sample_market_data):
        """Fail when price below threshold."""
        passes, reason = apply_liquidity_gate("PENNY", sample_market_data)
        assert passes is False
        assert reason == FLAG_PENNY_STOCK

    def test_fails_micro_cap(self, sample_market_data):
        """Fail when market cap below threshold."""
        passes, reason = apply_liquidity_gate("MICRO", sample_market_data)
        assert passes is False
        assert reason == FLAG_MICRO_CAP

    def test_goss_collapse_fails(self, sample_market_data):
        """GOSS-proof: volume collapse triggers LOW_LIQUIDITY."""
        passes, reason = apply_liquidity_gate("GOSS_COLLAPSE", sample_market_data)
        assert passes is False
        assert reason == FLAG_LOW_LIQUIDITY


# =============================================================================
# FINANCIAL GATE TESTS
# =============================================================================

class TestFinancialGate:
    def test_passes_healthy(self, sample_financial_data):
        """Healthy company passes financial gate."""
        passes, reason = apply_financial_gate("HEALTHY", sample_financial_data)
        assert passes is True
        assert reason is None

    def test_fails_cash_risk(self, sample_financial_data):
        """Fail when runway below threshold."""
        passes, reason = apply_financial_gate("CASH_RISK_TICKER", sample_financial_data)
        assert passes is False
        assert reason == FLAG_CASH_RISK

    def test_passes_profitable(self, sample_financial_data):
        """Profitable company passes (no cash risk)."""
        passes, reason = apply_financial_gate("PROFITABLE", sample_financial_data)
        assert passes is True
        assert reason is None

    def test_passes_unknown_runway(self, sample_financial_data):
        """Missing financial data does NOT fail gate (informational only)."""
        passes, reason = apply_financial_gate("NO_CASH", sample_financial_data)
        # We pass the gate but will have RUNWAY_UNKNOWN flag
        assert passes is True
        assert reason is None


# =============================================================================
# COMBINED GATE TESTS
# =============================================================================

class TestApplyAllGates:
    def test_all_gates_pass(self, sample_market_data, sample_financial_data):
        """Ticker passing all gates."""
        # Create a combined test case
        market_data = {"GOOD": sample_market_data["GOOD"]}
        financial_data = {"GOOD": sample_financial_data["HEALTHY"]}
        financial_data["GOOD"]["ticker"] = "GOOD"

        result = apply_all_gates("GOOD", market_data, financial_data)

        assert result["passes"] is True
        assert len(result["rejection_reasons"]) == 0
        assert result["adv_usd"] == 50_000_000
        assert result["price"] == 50.0
        assert result["market_cap"] == 5_000_000_000

    def test_mixed_failures(self, sample_market_data, sample_financial_data):
        """Test case with both liquidity and financial failures."""
        # LOW_ADV ticker with cash risk financial data
        market_data = {"TEST": sample_market_data["LOW_ADV"]}
        market_data["TEST"]["ticker"] = "TEST"
        financial_data = {"TEST": sample_financial_data["CASH_RISK_TICKER"]}
        financial_data["TEST"]["ticker"] = "TEST"

        result = apply_all_gates("TEST", market_data, financial_data)

        assert result["passes"] is False
        assert FLAG_LOW_LIQUIDITY in result["risk_flags"]
        assert FLAG_CASH_RISK in result["risk_flags"]
        assert len(result["rejection_reasons"]) == 2

    def test_no_market_data(self, sample_financial_data):
        """Test with no market data (liquidity gates skipped)."""
        result = apply_all_gates("HEALTHY", None, sample_financial_data)

        # Passes because only financial gate applied
        assert result["passes"] is True
        assert result["adv_usd"] == 0.0

    def test_no_financial_data(self, sample_market_data):
        """Test with no financial data (financial gates skipped)."""
        result = apply_all_gates("GOOD", sample_market_data, None)

        # Passes because only liquidity gate applied
        assert result["passes"] is True
        assert result["runway_months"] is None

    def test_risk_flags_sorted(self, sample_market_data, sample_financial_data):
        """Risk flags are sorted for deterministic output."""
        market_data = {"TEST": sample_market_data["LOW_ADV"]}
        market_data["TEST"]["ticker"] = "TEST"
        financial_data = {"TEST": sample_financial_data["CASH_RISK_TICKER"]}
        financial_data["TEST"]["ticker"] = "TEST"

        result = apply_all_gates("TEST", market_data, financial_data)

        # Flags should be sorted alphabetically
        assert result["risk_flags"] == sorted(result["risk_flags"])


# =============================================================================
# PARAMETER SNAPSHOT TESTS
# =============================================================================

class TestParameterSnapshot:
    def test_snapshot_contains_all_params(self):
        """Snapshot contains all threshold parameters."""
        snapshot = get_parameters_snapshot()

        assert "ADV_MINIMUM" in snapshot
        assert "PRICE_MINIMUM" in snapshot
        assert "MARKET_CAP_MINIMUM" in snapshot
        assert "RUNWAY_MINIMUM_MONTHS" in snapshot
        assert "version" in snapshot

    def test_hash_is_deterministic(self):
        """Parameter hash is deterministic."""
        hash1 = compute_parameters_hash()
        hash2 = compute_parameters_hash()

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    def test_zero_cash(self):
        """Zero cash should give zero runway."""
        financial_data = {
            "ZERO": {
                "ticker": "ZERO",
                "Cash": 0,
                "NetIncome": -10_000_000,
            }
        }
        runway = calculate_runway_months("ZERO", financial_data)
        assert runway == 0.0

    def test_negative_market_cap_ignored(self):
        """Negative market cap treated as unknown."""
        market_data = {
            "NEG": {
                "ticker": "NEG",
                "price": 50.0,
                "avg_volume": 1_000_000,
                "market_cap": -1000,
            }
        }
        result = apply_all_gates("NEG", market_data, None)
        # Passes liquidity (ADV is good, price is good)
        # Market cap check skipped for negative values
        assert result["market_cap"] is None  # Negative filtered out

    def test_string_numeric_fields(self):
        """Handle numeric fields stored as strings."""
        market_data = {
            "STR": {
                "ticker": "STR",
                "price": "50.0",
                "avg_volume": "1000000",
                "market_cap": "5000000000",
            }
        }
        adv = calculate_adv("STR", market_data)
        assert adv == 50_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
