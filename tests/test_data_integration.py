"""
Data Integration Tests - End-to-End Pipeline with Sample Data

Tests the complete data flow from raw inputs through risk gates,
liquidity scoring, and institutional signal generation.

Uses realistic sample data to verify:
1. Data loading and field extraction
2. Risk gate application across tickers
3. Liquidity scoring with tiered thresholds
4. Institutional signal report generation
5. Deterministic output across multiple runs
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pytest

from risk_gates import (
    load_market_data,
    load_financial_data,
    apply_all_gates,
    calculate_adv,
    calculate_runway_months,
    FLAG_LOW_LIQUIDITY,
    FLAG_PENNY_STOCK,
    FLAG_MICRO_CAP,
    FLAG_CASH_RISK,
)

from liquidity_scoring import (
    compute_liquidity_score,
    score_all_tickers,
    classify_market_cap_tier,
)


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_market_data_file(tmp_path):
    """Create a sample market_data.json file."""
    data = [
        {
            "ticker": "GOOD",
            "price": 75.50,
            "avg_volume": 500000,
            "market_cap": 8_000_000_000,
            "spread_bps": 25.0,
        },
        {
            "ticker": "LOWLIQ",
            "price": 15.00,
            "avg_volume": 20000,
            "market_cap": 800_000_000,
        },
        {
            "ticker": "PENNY",
            "price": 1.25,
            "avg_volume": 1000000,
            "market_cap": 150_000_000,
        },
        {
            "ticker": "MICRO",
            "price": 8.00,
            "avg_volume": 100000,
            "market_cap": 40_000_000,
        },
        {
            "ticker": "LARGE",
            "price": 250.00,
            "avg_volume": 2000000,
            "market_cap": 50_000_000_000,
            "spread_bps": 10.0,
        },
        {
            "ticker": "MIDCAP",
            "price": 45.00,
            "avg_volume": 300000,
            "market_cap": 3_500_000_000,
            "spread_bps": 75.0,
        },
    ]

    filepath = tmp_path / "market_data.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)

    return filepath


@pytest.fixture
def sample_financial_data_file(tmp_path):
    """Create a sample financial_data.json file."""
    data = [
        {
            "ticker": "GOOD",
            "Cash": 500_000_000,
            "NetIncome": 50_000_000,  # Profitable
        },
        {
            "ticker": "LOWLIQ",
            "Cash": 100_000_000,
            "NetIncome": -20_000_000,  # 15 month runway
        },
        {
            "ticker": "PENNY",
            "Cash": 30_000_000,
            "NetIncome": -50_000_000,  # ~2 month runway
        },
        {
            "ticker": "MICRO",
            "Cash": 5_000_000,
            "NetIncome": -10_000_000,  # ~1.5 month runway
        },
        {
            "ticker": "LARGE",
            "Cash": 2_000_000_000,
            "NetIncome": 200_000_000,  # Profitable
        },
        {
            "ticker": "MIDCAP",
            "Cash": 300_000_000,
            "NetIncome": -100_000_000,  # 9 month runway
        },
    ]

    filepath = tmp_path / "financial_data.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)

    return filepath


@pytest.fixture
def sample_holdings_data():
    """Sample holdings snapshot data for institutional signals."""
    return {
        "GOOD": {
            "ticker": "GOOD",
            "managers": [
                {"name": "Baker Bros Advisors", "action": "NEW", "shares": 500000, "value_usd": 37_750_000},
                {"name": "OrbiMed Advisors", "action": "INCREASED", "shares": 300000, "value_usd": 22_650_000},
                {"name": "Perceptive Advisors", "action": "INCREASED", "shares": 200000, "value_usd": 15_100_000},
            ],
            "total_managers": 3,
            "net_buyers": 3,
        },
        "LOWLIQ": {
            "ticker": "LOWLIQ",
            "managers": [
                {"name": "Tang Capital", "action": "INCREASED", "shares": 100000, "value_usd": 1_500_000},
            ],
            "total_managers": 1,
            "net_buyers": 1,
        },
        "LARGE": {
            "ticker": "LARGE",
            "managers": [
                {"name": "Deerfield Management", "action": "NEW", "shares": 100000, "value_usd": 25_000_000},
                {"name": "Farallon Capital", "action": "INCREASED", "shares": 80000, "value_usd": 20_000_000},
                {"name": "Redmile Group", "action": "INCREASED", "shares": 60000, "value_usd": 15_000_000},
                {"name": "RTW Investments", "action": "NEW", "shares": 40000, "value_usd": 10_000_000},
            ],
            "total_managers": 4,
            "net_buyers": 4,
        },
    }


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_market_data_list_format(self, sample_market_data_file):
        """Load market data from list format JSON."""
        data = load_market_data(str(sample_market_data_file))

        assert len(data) == 6
        assert "GOOD" in data
        assert "LARGE" in data
        assert data["GOOD"]["price"] == 75.50

    def test_load_financial_data_list_format(self, sample_financial_data_file):
        """Load financial data from list format JSON."""
        data = load_financial_data(str(sample_financial_data_file))

        assert len(data) == 6
        assert "GOOD" in data
        assert data["GOOD"]["Cash"] == 500_000_000

    def test_load_market_data_dict_format(self, tmp_path):
        """Load market data from dict format JSON."""
        data = {
            "TICKER1": {"ticker": "TICKER1", "price": 50.0},
            "TICKER2": {"ticker": "TICKER2", "price": 100.0},
        }

        filepath = tmp_path / "market_dict.json"
        with open(filepath, 'w') as f:
            json.dump(data, f)

        loaded = load_market_data(str(filepath))
        assert len(loaded) == 2
        assert loaded["TICKER1"]["price"] == 50.0


# =============================================================================
# FULL PIPELINE INTEGRATION TESTS
# =============================================================================

class TestFullPipelineIntegration:
    """End-to-end pipeline tests with sample data."""

    def test_risk_gates_across_universe(
        self, sample_market_data_file, sample_financial_data_file
    ):
        """Apply risk gates to full sample universe."""
        market_data = load_market_data(str(sample_market_data_file))
        financial_data = load_financial_data(str(sample_financial_data_file))

        results = {}
        for ticker in market_data.keys():
            results[ticker] = apply_all_gates(ticker, market_data, financial_data)

        # GOOD should pass all gates
        assert results["GOOD"]["passes"] is True
        assert len(results["GOOD"]["rejection_reasons"]) == 0

        # LOWLIQ should fail liquidity gate
        assert results["LOWLIQ"]["passes"] is False
        assert FLAG_LOW_LIQUIDITY in results["LOWLIQ"]["rejection_reasons"]

        # PENNY should fail penny stock gate
        assert results["PENNY"]["passes"] is False
        assert FLAG_PENNY_STOCK in results["PENNY"]["rejection_reasons"]

        # MICRO should fail micro cap gate
        assert results["MICRO"]["passes"] is False
        assert FLAG_MICRO_CAP in results["MICRO"]["rejection_reasons"]

        # LARGE should pass all gates
        assert results["LARGE"]["passes"] is True

    def test_liquidity_scoring_across_universe(self, sample_market_data_file):
        """Score liquidity for full sample universe."""
        market_data = load_market_data(str(sample_market_data_file))
        tickers = list(market_data.keys())

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit_path = f.name

        try:
            results = score_all_tickers(
                tickers, market_data, "2024-01-15", audit_path
            )

            # Results should be sorted
            result_tickers = [r["ticker"] for r in results]
            assert result_tickers == sorted(result_tickers)

            # Find specific results
            results_by_ticker = {r["ticker"]: r for r in results}

            # LARGE should have highest score (good ADV + tight spread)
            assert results_by_ticker["LARGE"]["liquidity_score"] >= 80
            assert results_by_ticker["LARGE"]["liquidity_tier"] == "large"

            # GOOD should have high score
            assert results_by_ticker["GOOD"]["liquidity_score"] >= 50
            assert results_by_ticker["GOOD"]["liquidity_tier"] == "mid"

            # PENNY should be capped
            assert results_by_ticker["PENNY"]["liquidity_score"] <= 10

            # Audit log should have all entries
            audit_lines = Path(audit_path).read_text().strip().split('\n')
            assert len(audit_lines) == len(tickers)
        finally:
            Path(audit_path).unlink(missing_ok=True)

    def test_tier_distribution(self, sample_market_data_file):
        """Verify market cap tier distribution."""
        market_data = load_market_data(str(sample_market_data_file))

        tier_counts = {"micro": 0, "small": 0, "mid": 0, "large": 0, "unknown": 0}

        for ticker, record in market_data.items():
            mcap = record.get("market_cap")
            tier = classify_market_cap_tier(mcap)
            tier_counts[tier] += 1

        # Expected distribution based on sample data
        assert tier_counts["micro"] == 2  # PENNY, MICRO
        assert tier_counts["small"] == 1  # LOWLIQ
        assert tier_counts["mid"] == 2    # GOOD, MIDCAP
        assert tier_counts["large"] == 1  # LARGE

    def test_adv_calculation_consistency(self, sample_market_data_file):
        """Verify ADV calculations are consistent."""
        market_data = load_market_data(str(sample_market_data_file))

        # GOOD: 500000 shares * $75.50 = $37.75M
        adv_good = calculate_adv("GOOD", market_data)
        assert abs(adv_good - 37_750_000) < 1

        # LARGE: 2000000 shares * $250 = $500M
        adv_large = calculate_adv("LARGE", market_data)
        assert abs(adv_large - 500_000_000) < 1

    def test_runway_calculation_consistency(self, sample_financial_data_file):
        """Verify runway calculations are consistent."""
        financial_data = load_financial_data(str(sample_financial_data_file))

        # GOOD: Profitable, should return 999
        runway_good = calculate_runway_months("GOOD", financial_data)
        assert runway_good == 999.0

        # MIDCAP: $300M / ($100M * 4/12) = 9 months
        runway_midcap = calculate_runway_months("MIDCAP", financial_data)
        assert runway_midcap is not None
        assert 8.5 < runway_midcap < 9.5


# =============================================================================
# DETERMINISM VERIFICATION TESTS
# =============================================================================

class TestDeterminismVerification:
    """Tests that verify deterministic behavior across runs."""

    def test_multiple_runs_identical(
        self, sample_market_data_file, sample_financial_data_file
    ):
        """Multiple runs with same data produce identical results."""
        market_data = load_market_data(str(sample_market_data_file))
        financial_data = load_financial_data(str(sample_financial_data_file))

        results_list = []
        for _ in range(5):
            run_results = {}
            for ticker in sorted(market_data.keys()):
                run_results[ticker] = {
                    "gates": apply_all_gates(ticker, market_data, financial_data),
                    "liquidity": compute_liquidity_score(ticker, market_data),
                }
            results_list.append(run_results)

        # All runs should be identical
        for i in range(1, 5):
            assert results_list[0] == results_list[i], f"Run {i} differs from run 0"

    def test_audit_log_deterministic(self, sample_market_data_file):
        """Audit logs are deterministic across runs."""
        market_data = load_market_data(str(sample_market_data_file))
        tickers = list(market_data.keys())

        audit_contents = []

        for _ in range(3):
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                audit_path = f.name

            try:
                score_all_tickers(tickers, market_data, "2024-01-15", audit_path)
                audit_contents.append(Path(audit_path).read_text())
            finally:
                Path(audit_path).unlink(missing_ok=True)

        # All audit logs should be identical
        assert audit_contents[0] == audit_contents[1] == audit_contents[2]

    def test_json_output_deterministic(
        self, sample_market_data_file, sample_financial_data_file
    ):
        """JSON serialization is deterministic."""
        market_data = load_market_data(str(sample_market_data_file))
        financial_data = load_financial_data(str(sample_financial_data_file))

        outputs = []
        for _ in range(3):
            results = {}
            for ticker in market_data.keys():
                results[ticker] = apply_all_gates(ticker, market_data, financial_data)

            # Serialize to JSON
            output = json.dumps(results, sort_keys=True, separators=(',', ':'))
            outputs.append(output)

        assert outputs[0] == outputs[1] == outputs[2]


# =============================================================================
# SIGNAL FILTERING TESTS
# =============================================================================

class TestSignalFiltering:
    """Tests for signal filtering based on risk gates."""

    def test_passing_signals_count(
        self, sample_market_data_file, sample_financial_data_file
    ):
        """Count signals that pass all gates."""
        market_data = load_market_data(str(sample_market_data_file))
        financial_data = load_financial_data(str(sample_financial_data_file))

        passing = []
        killed = []

        for ticker in market_data.keys():
            result = apply_all_gates(ticker, market_data, financial_data)
            if result["passes"]:
                passing.append(ticker)
            else:
                killed.append((ticker, result["rejection_reasons"]))

        # Based on sample data:
        # GOOD, LARGE, MIDCAP should pass
        # LOWLIQ (low liquidity), PENNY (penny stock), MICRO (micro cap + cash risk) should be killed
        assert len(passing) == 3
        assert "GOOD" in passing
        assert "LARGE" in passing
        assert "MIDCAP" in passing

        assert len(killed) == 3

    def test_rejection_reasons_correct(
        self, sample_market_data_file, sample_financial_data_file
    ):
        """Verify rejection reasons are correct."""
        market_data = load_market_data(str(sample_market_data_file))
        financial_data = load_financial_data(str(sample_financial_data_file))

        # LOWLIQ: ADV = 20000 * 15 = $300K (below $2M threshold for small cap)
        result = apply_all_gates("LOWLIQ", market_data, financial_data)
        assert FLAG_LOW_LIQUIDITY in result["rejection_reasons"]

        # PENNY: Price $1.25 < $2.00
        result = apply_all_gates("PENNY", market_data, financial_data)
        assert FLAG_PENNY_STOCK in result["rejection_reasons"]

        # MICRO: Market cap $40M < $50M
        result = apply_all_gates("MICRO", market_data, financial_data)
        assert FLAG_MICRO_CAP in result["rejection_reasons"]


# =============================================================================
# EDGE CASE INTEGRATION TESTS
# =============================================================================

class TestEdgeCaseIntegration:
    """Integration tests for edge cases."""

    def test_empty_market_data(self):
        """Handle empty market data."""
        result = apply_all_gates("MISSING", {}, {})
        assert result["passes"] is False
        assert result["adv_usd"] == 0.0

    def test_partial_data(self, tmp_path):
        """Handle partial data (some fields missing)."""
        data = [
            {
                "ticker": "PARTIAL",
                "price": 50.0,
                # Missing avg_volume - ADV unknown
            }
        ]

        filepath = tmp_path / "partial.json"
        with open(filepath, 'w') as f:
            json.dump(data, f)

        market_data = load_market_data(str(filepath))
        result = apply_all_gates("PARTIAL", market_data, {})

        # Should fail due to ADV unknown
        assert result["passes"] is False

    def test_mixed_data_quality(self, tmp_path):
        """Handle mixed data quality in universe."""
        market_data = [
            {"ticker": "GOOD", "price": 50.0, "avg_volume": 1000000, "market_cap": 5_000_000_000},
            {"ticker": "BAD", "price": -10.0, "avg_volume": -1000, "market_cap": -100},
            {"ticker": "PARTIAL", "price": 30.0},  # Missing volume and mcap
        ]

        financial_data = [
            {"ticker": "GOOD", "Cash": 500_000_000, "NetIncome": 50_000_000},
            {"ticker": "BAD"},  # Missing fields
        ]

        market_path = tmp_path / "market.json"
        with open(market_path, 'w') as f:
            json.dump(market_data, f)

        fin_path = tmp_path / "financial.json"
        with open(fin_path, 'w') as f:
            json.dump(financial_data, f)

        market = load_market_data(str(market_path))
        financial = load_financial_data(str(fin_path))

        # GOOD should pass
        result = apply_all_gates("GOOD", market, financial)
        assert result["passes"] is True

        # BAD should fail (negative values)
        result = apply_all_gates("BAD", market, financial)
        assert result["passes"] is False

        # PARTIAL should fail (missing data)
        result = apply_all_gates("PARTIAL", market, financial)
        assert result["passes"] is False


# =============================================================================
# PERFORMANCE SANITY TESTS
# =============================================================================

class TestPerformanceSanity:
    """Basic performance sanity checks."""

    def test_large_universe_performance(self, tmp_path):
        """Verify reasonable performance with larger universe."""
        # Generate 500 tickers
        market_data = []
        for i in range(500):
            market_data.append({
                "ticker": f"TICK{i:04d}",
                "price": 50.0 + (i % 100),
                "avg_volume": 100000 + (i * 1000),
                "market_cap": 1_000_000_000 + (i * 10_000_000),
                "spread_bps": 50.0 + (i % 50),
            })

        filepath = tmp_path / "large_market.json"
        with open(filepath, 'w') as f:
            json.dump(market_data, f)

        loaded = load_market_data(str(filepath))
        tickers = list(loaded.keys())

        # Should complete in reasonable time
        import time
        start = time.time()

        results = score_all_tickers(tickers, loaded, "2024-01-15", None)

        elapsed = time.time() - start

        assert len(results) == 500
        assert elapsed < 5.0  # Should complete in under 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
