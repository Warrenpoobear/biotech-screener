#!/usr/bin/env python3
"""
Tests for common/constants.py

Centralized configuration constants and helper functions for the biotech screener.
Tests cover:
- Market cap thresholds
- Liquidity thresholds
- Scoring defaults
- Helper functions (bucket_market_cap_mm, bucket_adv_usd, is_liquid)
"""

import pytest

from common.constants import (
    # Market cap thresholds (millions)
    MCAP_SMALL_THRESHOLD_MM,
    MCAP_MID_THRESHOLD_MM,
    MCAP_LARGE_THRESHOLD_MM,
    # Market cap thresholds (full USD)
    MCAP_SMALL_THRESHOLD,
    MCAP_MID_THRESHOLD,
    MCAP_LARGE_THRESHOLD,
    # Liquidity thresholds
    ADV_ILLIQUID_THRESHOLD,
    ADV_MINIMUM_THRESHOLD,
    ADV_LIQUID_THRESHOLD,
    LIQUIDITY_TIER_THRESHOLDS,
    # Scoring defaults
    DEFAULT_FALLBACK_SCORE,
    SCORE_MIN,
    SCORE_MAX,
    DEFAULT_CATALYST_DECAY_DAYS,
    # Data quality thresholds
    MIN_FINANCIAL_COVERAGE_PCT,
    MIN_CLINICAL_COVERAGE_PCT,
    MIN_MARKET_COVERAGE_PCT,
    MIN_CATALYST_COVERAGE_PCT,
    MAX_MISSING_DATE_PCT,
    MAX_ORPHAN_TICKER_PCT,
    # File size limits
    MAX_JSON_FILE_SIZE_BYTES,
    MAX_RECORDS_PER_BATCH,
    # API rate limits
    OPENFIGI_RATE_LIMIT_RPS,
    SEC_EDGAR_RATE_LIMIT_RPS,
    # Helper functions
    bucket_market_cap_mm,
    bucket_adv_usd,
    is_liquid,
)


class TestMarketCapThresholds:
    """Tests for market cap threshold constants."""

    def test_mm_thresholds_ordering(self):
        """Thresholds should be in ascending order."""
        assert MCAP_SMALL_THRESHOLD_MM < MCAP_MID_THRESHOLD_MM
        assert MCAP_MID_THRESHOLD_MM < MCAP_LARGE_THRESHOLD_MM

    def test_full_usd_thresholds_ordering(self):
        """Full USD thresholds should be in ascending order."""
        assert MCAP_SMALL_THRESHOLD < MCAP_MID_THRESHOLD
        assert MCAP_MID_THRESHOLD < MCAP_LARGE_THRESHOLD

    def test_mm_to_full_usd_conversion(self):
        """MM thresholds * 1M should equal full USD thresholds."""
        assert MCAP_SMALL_THRESHOLD_MM * 1_000_000 == MCAP_SMALL_THRESHOLD
        assert MCAP_MID_THRESHOLD_MM * 1_000_000 == MCAP_MID_THRESHOLD
        assert MCAP_LARGE_THRESHOLD_MM * 1_000_000 == MCAP_LARGE_THRESHOLD

    def test_small_threshold_value(self):
        """Small cap threshold should be $500M."""
        assert MCAP_SMALL_THRESHOLD_MM == 500
        assert MCAP_SMALL_THRESHOLD == 500_000_000

    def test_mid_threshold_value(self):
        """Mid cap threshold should be $2B."""
        assert MCAP_MID_THRESHOLD_MM == 2000
        assert MCAP_MID_THRESHOLD == 2_000_000_000

    def test_large_threshold_value(self):
        """Large cap threshold should be $5B."""
        assert MCAP_LARGE_THRESHOLD_MM == 5000
        assert MCAP_LARGE_THRESHOLD == 5_000_000_000


class TestLiquidityThresholds:
    """Tests for liquidity threshold constants."""

    def test_thresholds_ordering(self):
        """Thresholds should be in ascending order."""
        assert ADV_ILLIQUID_THRESHOLD < ADV_MINIMUM_THRESHOLD
        assert ADV_MINIMUM_THRESHOLD < ADV_LIQUID_THRESHOLD

    def test_illiquid_threshold_value(self):
        """Illiquid threshold should be $250K."""
        assert ADV_ILLIQUID_THRESHOLD == 250_000

    def test_minimum_threshold_value(self):
        """Minimum threshold should be $500K."""
        assert ADV_MINIMUM_THRESHOLD == 500_000

    def test_liquid_threshold_value(self):
        """Liquid threshold should be $2M."""
        assert ADV_LIQUID_THRESHOLD == 2_000_000

    def test_liquidity_tier_thresholds_keys(self):
        """Liquidity tier thresholds should have expected keys."""
        expected_keys = {"illiquid", "thin", "moderate", "liquid"}
        assert set(LIQUIDITY_TIER_THRESHOLDS.keys()) == expected_keys

    def test_liquidity_tier_thresholds_ordering(self):
        """Tier thresholds should be in ascending order."""
        assert LIQUIDITY_TIER_THRESHOLDS["illiquid"] < LIQUIDITY_TIER_THRESHOLDS["thin"]
        assert LIQUIDITY_TIER_THRESHOLDS["thin"] < LIQUIDITY_TIER_THRESHOLDS["moderate"]
        assert LIQUIDITY_TIER_THRESHOLDS["moderate"] < LIQUIDITY_TIER_THRESHOLDS["liquid"]


class TestScoringDefaults:
    """Tests for scoring default constants."""

    def test_fallback_score_in_range(self):
        """Fallback score should be within min/max range."""
        assert SCORE_MIN <= DEFAULT_FALLBACK_SCORE <= SCORE_MAX

    def test_score_range(self):
        """Score range should be 0-100."""
        assert SCORE_MIN == 0.0
        assert SCORE_MAX == 100.0

    def test_catalyst_decay_positive(self):
        """Catalyst decay should be positive."""
        assert DEFAULT_CATALYST_DECAY_DAYS > 0


class TestDataQualityThresholds:
    """Tests for data quality threshold constants."""

    def test_coverage_thresholds_in_range(self):
        """Coverage thresholds should be percentages (0-100)."""
        assert 0 <= MIN_FINANCIAL_COVERAGE_PCT <= 100
        assert 0 <= MIN_CLINICAL_COVERAGE_PCT <= 100
        assert 0 <= MIN_MARKET_COVERAGE_PCT <= 100
        assert 0 <= MIN_CATALYST_COVERAGE_PCT <= 100

    def test_max_missing_date_in_range(self):
        """Max missing date should be percentage (0-100)."""
        assert 0 <= MAX_MISSING_DATE_PCT <= 100

    def test_max_orphan_ticker_in_range(self):
        """Max orphan ticker should be percentage (0-100)."""
        assert 0 <= MAX_ORPHAN_TICKER_PCT <= 100


class TestFileSizeLimits:
    """Tests for file size limit constants."""

    def test_max_json_size_reasonable(self):
        """Max JSON file size should be 100 MB."""
        assert MAX_JSON_FILE_SIZE_BYTES == 100 * 1024 * 1024

    def test_max_records_reasonable(self):
        """Max records per batch should be 10000."""
        assert MAX_RECORDS_PER_BATCH == 10000


class TestAPIRateLimits:
    """Tests for API rate limit constants."""

    def test_openfigi_rate_limit_positive(self):
        """OpenFIGI rate limit should be positive."""
        assert OPENFIGI_RATE_LIMIT_RPS > 0

    def test_sec_edgar_rate_limit_positive(self):
        """SEC EDGAR rate limit should be positive."""
        assert SEC_EDGAR_RATE_LIMIT_RPS > 0


class TestBucketMarketCapMM:
    """Tests for bucket_market_cap_mm function."""

    def test_none_returns_unknown(self):
        """None market cap should return UNKNOWN."""
        assert bucket_market_cap_mm(None) == "UNKNOWN"

    def test_small_cap(self):
        """Market cap < $500M should return SMALL."""
        assert bucket_market_cap_mm(100) == "SMALL"
        assert bucket_market_cap_mm(499) == "SMALL"

    def test_mid_cap(self):
        """Market cap $500M - $2B should return MID."""
        assert bucket_market_cap_mm(500) == "MID"
        assert bucket_market_cap_mm(1000) == "MID"
        assert bucket_market_cap_mm(1999) == "MID"

    def test_large_cap(self):
        """Market cap $2B - $5B should return LARGE."""
        assert bucket_market_cap_mm(2000) == "LARGE"
        assert bucket_market_cap_mm(3500) == "LARGE"
        assert bucket_market_cap_mm(4999) == "LARGE"

    def test_mega_cap(self):
        """Market cap >= $5B should return MEGA."""
        assert bucket_market_cap_mm(5000) == "MEGA"
        assert bucket_market_cap_mm(10000) == "MEGA"
        assert bucket_market_cap_mm(100000) == "MEGA"

    def test_boundary_small_to_mid(self):
        """Test boundary between SMALL and MID."""
        assert bucket_market_cap_mm(499.99) == "SMALL"
        assert bucket_market_cap_mm(500.0) == "MID"

    def test_boundary_mid_to_large(self):
        """Test boundary between MID and LARGE."""
        assert bucket_market_cap_mm(1999.99) == "MID"
        assert bucket_market_cap_mm(2000.0) == "LARGE"

    def test_boundary_large_to_mega(self):
        """Test boundary between LARGE and MEGA."""
        assert bucket_market_cap_mm(4999.99) == "LARGE"
        assert bucket_market_cap_mm(5000.0) == "MEGA"

    def test_zero_market_cap(self):
        """Zero market cap should return SMALL."""
        assert bucket_market_cap_mm(0) == "SMALL"

    def test_negative_market_cap(self):
        """Negative market cap should return SMALL."""
        assert bucket_market_cap_mm(-100) == "SMALL"


class TestBucketAdvUsd:
    """Tests for bucket_adv_usd function."""

    def test_none_returns_unknown(self):
        """None ADV should return UNKNOWN."""
        assert bucket_adv_usd(None) == "UNKNOWN"

    def test_illiquid(self):
        """ADV < $250K should return ILLIQ."""
        assert bucket_adv_usd(100_000) == "ILLIQ"
        assert bucket_adv_usd(249_999) == "ILLIQ"

    def test_moderate(self):
        """ADV $250K - $2M should return MODERATE."""
        assert bucket_adv_usd(250_000) == "MODERATE"
        assert bucket_adv_usd(1_000_000) == "MODERATE"
        assert bucket_adv_usd(1_999_999) == "MODERATE"

    def test_liquid(self):
        """ADV >= $2M should return LIQUID."""
        assert bucket_adv_usd(2_000_000) == "LIQUID"
        assert bucket_adv_usd(5_000_000) == "LIQUID"
        assert bucket_adv_usd(10_000_000) == "LIQUID"

    def test_boundary_illiquid_to_moderate(self):
        """Test boundary between ILLIQ and MODERATE."""
        assert bucket_adv_usd(249_999) == "ILLIQ"
        assert bucket_adv_usd(250_000) == "MODERATE"

    def test_boundary_moderate_to_liquid(self):
        """Test boundary between MODERATE and LIQUID."""
        assert bucket_adv_usd(1_999_999) == "MODERATE"
        assert bucket_adv_usd(2_000_000) == "LIQUID"

    def test_zero_adv(self):
        """Zero ADV should return ILLIQ."""
        assert bucket_adv_usd(0) == "ILLIQ"

    def test_negative_adv(self):
        """Negative ADV should return ILLIQ."""
        assert bucket_adv_usd(-100_000) == "ILLIQ"


class TestIsLiquid:
    """Tests for is_liquid function."""

    def test_none_returns_false(self):
        """None ADV should return False."""
        assert is_liquid(None) is False

    def test_below_minimum_returns_false(self):
        """ADV below minimum should return False."""
        assert is_liquid(0) is False
        assert is_liquid(250_000) is False
        assert is_liquid(499_999) is False

    def test_at_minimum_returns_true(self):
        """ADV at minimum should return True."""
        assert is_liquid(500_000) is True

    def test_above_minimum_returns_true(self):
        """ADV above minimum should return True."""
        assert is_liquid(500_001) is True
        assert is_liquid(1_000_000) is True
        assert is_liquid(10_000_000) is True

    def test_negative_adv_returns_false(self):
        """Negative ADV should return False."""
        assert is_liquid(-500_000) is False
