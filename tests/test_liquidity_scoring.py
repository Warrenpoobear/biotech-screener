"""
Unit tests for liquidity_scoring.py

Tests the tiered liquidity scoring module including:
- Market cap tier classification
- ADV score computation
- Spread score computation
- Penny stock penalty
- GOSS-proof volume collapse
- Audit logging
"""
import json
import tempfile
from pathlib import Path

from liquidity_scoring import (
    classify_market_cap_tier,
    get_adv_threshold_for_tier,
    extract_spread_bps,
    compute_adv_score,
    compute_spread_score,
    compute_liquidity_score,
    create_liquidity_audit_record,
    append_audit_log,
    score_all_tickers,
    get_parameters_snapshot,
    compute_parameters_hash,
    LIQUIDITY_SCORING_VERSION,
    ADV_THRESHOLDS,
    TIER_MICRO_MAX,
    TIER_SMALL_MAX,
    TIER_MID_MAX,
    FLAG_WIDE_SPREAD,
)
from risk_gates import FLAG_ADV_UNKNOWN, FLAG_LOW_LIQUIDITY, FLAG_PENNY_STOCK


# =============================================================================
# TIER CLASSIFICATION TESTS
# =============================================================================

class TestTierClassification:
    def test_micro_cap(self):
        """Classify micro cap correctly."""
        assert classify_market_cap_tier(100_000_000) == "micro"
        assert classify_market_cap_tier(299_000_000) == "micro"

    def test_small_cap(self):
        """Classify small cap correctly."""
        assert classify_market_cap_tier(300_000_000) == "small"
        assert classify_market_cap_tier(1_500_000_000) == "small"
        assert classify_market_cap_tier(1_999_000_000) == "small"

    def test_mid_cap(self):
        """Classify mid cap correctly."""
        assert classify_market_cap_tier(2_000_000_000) == "mid"
        assert classify_market_cap_tier(5_000_000_000) == "mid"
        assert classify_market_cap_tier(9_999_000_000) == "mid"

    def test_large_cap(self):
        """Classify large cap correctly."""
        assert classify_market_cap_tier(10_000_000_000) == "large"
        assert classify_market_cap_tier(100_000_000_000) == "large"

    def test_unknown_market_cap(self):
        """Handle unknown market cap."""
        assert classify_market_cap_tier(None) == "unknown"
        assert classify_market_cap_tier(0) == "unknown"
        assert classify_market_cap_tier(-1000) == "unknown"

    def test_threshold_lookup(self):
        """Get correct threshold for each tier."""
        assert get_adv_threshold_for_tier("micro") == 750_000
        assert get_adv_threshold_for_tier("small") == 2_000_000
        assert get_adv_threshold_for_tier("mid") == 5_000_000
        assert get_adv_threshold_for_tier("large") == 10_000_000
        # Unknown tier defaults to small
        assert get_adv_threshold_for_tier("unknown") == 2_000_000


# =============================================================================
# SPREAD EXTRACTION TESTS
# =============================================================================

class TestSpreadExtraction:
    def test_direct_spread_bps(self):
        """Extract direct spread_bps field."""
        record = {"spread_bps": 150.0}
        assert extract_spread_bps(record) == 150.0

    def test_compute_from_bid_ask(self):
        """Compute spread from bid/ask."""
        record = {"bid": 49.50, "ask": 50.50}
        spread = extract_spread_bps(record)
        # (50.50 - 49.50) / 50.00 * 10000 = 200 bps
        assert spread is not None
        assert 199 < spread < 201

    def test_missing_spread_data(self):
        """Return None when no spread data available."""
        record = {"price": 50.0}
        assert extract_spread_bps(record) is None

    def test_invalid_bid_ask(self):
        """Handle invalid bid/ask values."""
        record = {"bid": 0, "ask": 50.0}
        assert extract_spread_bps(record) is None

        record = {"bid": 50.0, "ask": 49.0}  # Ask < bid
        assert extract_spread_bps(record) is None


# =============================================================================
# ADV SCORE TESTS
# =============================================================================

class TestADVScore:
    def test_zero_adv(self):
        """Zero ADV = zero score."""
        assert compute_adv_score(0, 1_000_000) == 0

    def test_at_threshold(self):
        """ADV at threshold = 35 (half of max)."""
        threshold = 2_000_000
        score = compute_adv_score(threshold, threshold)
        assert score == 35

    def test_at_double_threshold(self):
        """ADV at 2x threshold = 70 (max)."""
        threshold = 2_000_000
        score = compute_adv_score(threshold * 2, threshold)
        assert score == 70

    def test_above_double_threshold(self):
        """ADV above 2x threshold capped at 70."""
        threshold = 2_000_000
        score = compute_adv_score(threshold * 10, threshold)
        assert score == 70

    def test_below_threshold(self):
        """ADV below threshold gives proportional score."""
        threshold = 2_000_000
        score = compute_adv_score(threshold / 2, threshold)
        # 1M / 4M = 0.25 * 70 = 17
        assert score == 17


# =============================================================================
# SPREAD SCORE TESTS
# =============================================================================

class TestSpreadScore:
    def test_tight_spread(self):
        """Spread <= 50bps = full score (30)."""
        assert compute_spread_score(50.0) == 30
        assert compute_spread_score(25.0) == 30
        assert compute_spread_score(0.0) == 30

    def test_wide_spread(self):
        """Spread >= 400bps = zero score."""
        assert compute_spread_score(400.0) == 0
        assert compute_spread_score(500.0) == 0

    def test_mid_spread(self):
        """Spread at midpoint = ~15."""
        # 225bps is midpoint between 50 and 400
        score = compute_spread_score(225.0)
        assert 14 <= score <= 16

    def test_missing_spread(self):
        """None spread = zero score (conservative)."""
        assert compute_spread_score(None) == 0


# =============================================================================
# FULL LIQUIDITY SCORE TESTS
# =============================================================================

class TestLiquidityScore:
    def test_good_liquidity_large_cap(self):
        """Large cap with good ADV gets high score."""
        market_data = {
            "GOOD": {
                "ticker": "GOOD",
                "price": 100.0,
                "avg_volume": 1_000_000,  # ADV = $100M
                "market_cap": 50_000_000_000,  # $50B = large
                "spread_bps": 25.0,  # Tight spread
            }
        }
        result = compute_liquidity_score("GOOD", market_data)

        assert result["liquidity_tier"] == "large"
        assert result["liquidity_score"] >= 80
        assert len(result["risk_flags"]) == 0

    def test_low_liquidity_small_cap(self):
        """Small cap with low ADV gets flagged."""
        market_data = {
            "LOW": {
                "ticker": "LOW",
                "price": 20.0,
                "avg_volume": 50_000,  # ADV = $1M (below $2M threshold)
                "market_cap": 500_000_000,  # $500M = small
            }
        }
        result = compute_liquidity_score("LOW", market_data)

        assert result["liquidity_tier"] == "small"
        assert FLAG_LOW_LIQUIDITY in result["risk_flags"]
        assert result["liquidity_score"] < 50

    def test_penny_stock_penalty(self):
        """Penny stock capped at score 10."""
        market_data = {
            "PENNY": {
                "ticker": "PENNY",
                "price": 1.50,  # Below $2
                "avg_volume": 10_000_000,  # Very high volume
                "market_cap": 100_000_000,  # Micro cap
            }
        }
        result = compute_liquidity_score("PENNY", market_data)

        assert FLAG_PENNY_STOCK in result["risk_flags"]
        assert result["liquidity_score"] <= 10

    def test_mid_cap_just_below_threshold(self):
        """Mid cap ticker just below ADV threshold."""
        market_data = {
            "MID": {
                "ticker": "MID",
                "price": 50.0,
                "avg_volume": 90_000,  # ADV = $4.5M (below $5M threshold)
                "market_cap": 5_000_000_000,  # $5B = mid
            }
        }
        result = compute_liquidity_score("MID", market_data)

        assert result["liquidity_tier"] == "mid"
        assert FLAG_LOW_LIQUIDITY in result["risk_flags"]

    def test_goss_proof_volume_collapse(self):
        """GOSS-like scenario with volume collapse."""
        market_data = {
            "GOSS": {
                "ticker": "GOSS",
                "price": 3.0,  # Crashed price
                "avg_volume": 100_000,  # ADV = $300K
                "market_cap": 100_000_000,  # $100M = micro
            }
        }
        result = compute_liquidity_score("GOSS", market_data)

        # $300K is below micro threshold of $750K
        assert FLAG_LOW_LIQUIDITY in result["risk_flags"]
        assert result["liquidity_score"] < 40

    def test_missing_market_data(self):
        """Handle ticker not in market data."""
        result = compute_liquidity_score("MISSING", {})

        assert FLAG_ADV_UNKNOWN in result["risk_flags"]
        assert result["liquidity_score"] == 0

    def test_wide_spread_flag(self):
        """Wide spread gets flagged."""
        market_data = {
            "WIDE": {
                "ticker": "WIDE",
                "price": 50.0,
                "avg_volume": 1_000_000,  # Good ADV
                "market_cap": 5_000_000_000,
                "spread_bps": 500.0,  # Very wide
            }
        }
        result = compute_liquidity_score("WIDE", market_data)

        assert FLAG_WIDE_SPREAD in result["risk_flags"]
        assert result["spread_score"] == 0


# =============================================================================
# AUDIT LOGGING TESTS
# =============================================================================

class TestAuditLogging:
    def test_create_audit_record(self):
        """Audit record has required fields."""
        score_result = {
            "ticker": "TEST",
            "liquidity_score": 75,
            "liquidity_tier": "mid",
            "adv_usd": 5_000_000,
            "spread_bps": 100.0,
            "adv_score": 50,
            "spread_score": 25,
            "risk_flags": [],
        }
        record = create_liquidity_audit_record(
            "TEST", score_result, "2024-01-15"
        )

        assert record["ticker"] == "TEST"
        assert record["as_of_date"] == "2024-01-15"
        assert record["score_version"] == LIQUIDITY_SCORING_VERSION
        assert "thresholds_by_tier" in record
        assert record["chosen_tier"] == "mid"

    def test_append_audit_log(self, tmp_path):
        """Append to JSONL audit log."""
        log_path = tmp_path / "audit.jsonl"

        record1 = {"ticker": "A", "score": 50}
        record2 = {"ticker": "B", "score": 60}

        append_audit_log(str(log_path), record1)
        append_audit_log(str(log_path), record2)

        lines = log_path.read_text().strip().split('\n')
        assert len(lines) == 2

        parsed1 = json.loads(lines[0])
        parsed2 = json.loads(lines[1])
        assert parsed1["ticker"] == "A"
        assert parsed2["ticker"] == "B"

    def test_score_all_with_audit(self, tmp_path):
        """Score all tickers with audit logging."""
        market_data = {
            "A": {"ticker": "A", "price": 50.0, "avg_volume": 1_000_000, "market_cap": 5_000_000_000},
            "B": {"ticker": "B", "price": 25.0, "avg_volume": 500_000, "market_cap": 1_000_000_000},
        }
        log_path = tmp_path / "audit.jsonl"

        results = score_all_tickers(
            ["B", "A"],  # Out of order - should be sorted
            market_data,
            "2024-01-15",
            str(log_path)
        )

        # Results should be sorted by ticker
        assert results[0]["ticker"] == "A"
        assert results[1]["ticker"] == "B"

        # Audit log should exist
        assert log_path.exists()
        lines = log_path.read_text().strip().split('\n')
        assert len(lines) == 2


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    def test_same_input_same_output(self):
        """Same inputs produce same outputs."""
        market_data = {
            "TEST": {
                "ticker": "TEST",
                "price": 50.0,
                "avg_volume": 1_000_000,
                "market_cap": 5_000_000_000,
                "spread_bps": 100.0,
            }
        }

        result1 = compute_liquidity_score("TEST", market_data)
        result2 = compute_liquidity_score("TEST", market_data)

        assert result1 == result2

    def test_parameters_hash_deterministic(self):
        """Parameters hash is deterministic."""
        hash1 = compute_parameters_hash()
        hash2 = compute_parameters_hash()

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_flags_sorted(self):
        """Risk flags are always sorted."""
        market_data = {
            "TEST": {
                "ticker": "TEST",
                "price": 1.50,  # Penny stock
                "avg_volume": 10_000,  # Low liquidity
                "market_cap": 500_000_000,
                "spread_bps": 500.0,  # Wide spread
            }
        }
        result = compute_liquidity_score("TEST", market_data)

        assert result["risk_flags"] == sorted(result["risk_flags"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
