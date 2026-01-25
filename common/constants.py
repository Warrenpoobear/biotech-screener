"""
common/constants.py - Centralized Configuration Constants

All magic numbers and thresholds used across the biotech screener pipeline.
Centralizing these makes maintenance easier and ensures consistency.

Usage:
    from common.constants import (
        MCAP_SMALL_THRESHOLD,
        MCAP_MID_THRESHOLD,
        ADV_ILLIQUID_THRESHOLD,
        ADV_MINIMUM_THRESHOLD,
    )
"""

# =============================================================================
# MARKET CAP THRESHOLDS (in millions USD)
# =============================================================================

# Market cap classification buckets
MCAP_SMALL_THRESHOLD_MM = 500       # < $500M = Small cap
MCAP_MID_THRESHOLD_MM = 2000        # $500M - $2B = Mid cap
MCAP_LARGE_THRESHOLD_MM = 5000      # $2B - $5B = Large cap
# >= $5B = Mega cap

# Market cap in full USD (for calculations that use raw values)
MCAP_SMALL_THRESHOLD = 500_000_000      # $500M
MCAP_MID_THRESHOLD = 2_000_000_000      # $2B
MCAP_LARGE_THRESHOLD = 5_000_000_000    # $5B


# =============================================================================
# LIQUIDITY THRESHOLDS (Average Daily Volume in USD)
# =============================================================================

# ADV classification buckets
ADV_ILLIQUID_THRESHOLD = 250_000        # < $250K = Illiquid
ADV_MINIMUM_THRESHOLD = 500_000         # $500K minimum for trading
ADV_LIQUID_THRESHOLD = 2_000_000        # > $2M = Highly liquid

# Liquidity tier thresholds
LIQUIDITY_TIER_THRESHOLDS = {
    "illiquid": 250_000,
    "thin": 500_000,
    "moderate": 2_000_000,
    "liquid": 5_000_000,
}


# =============================================================================
# SCORING DEFAULTS
# =============================================================================

# Default/fallback score when data is missing
DEFAULT_FALLBACK_SCORE = 50.0

# Score normalization range
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# Catalyst decay constant (days)
DEFAULT_CATALYST_DECAY_DAYS = 30.0


# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================

# Minimum coverage percentages for data validation
MIN_FINANCIAL_COVERAGE_PCT = 80.0
MIN_CLINICAL_COVERAGE_PCT = 50.0
MIN_MARKET_COVERAGE_PCT = 50.0
MIN_CATALYST_COVERAGE_PCT = 10.0

# Maximum missing date percentage for PIT validation
MAX_MISSING_DATE_PCT = 50.0

# Maximum orphan ticker percentage
MAX_ORPHAN_TICKER_PCT = 10.0


# =============================================================================
# FILE SIZE LIMITS
# =============================================================================

# Maximum JSON file size to load (100 MB)
MAX_JSON_FILE_SIZE_BYTES = 100 * 1024 * 1024

# Maximum records per batch for processing
MAX_RECORDS_PER_BATCH = 10000


# =============================================================================
# API RATE LIMITS
# =============================================================================

# OpenFIGI rate limit (requests per second)
OPENFIGI_RATE_LIMIT_RPS = 4

# SEC EDGAR rate limit
SEC_EDGAR_RATE_LIMIT_RPS = 10


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def bucket_market_cap_mm(mcap_mm: float | None) -> str:
    """Classify market cap (in millions) into bucket."""
    if mcap_mm is None:
        return "UNKNOWN"
    if mcap_mm < MCAP_SMALL_THRESHOLD_MM:
        return "SMALL"
    if mcap_mm < MCAP_MID_THRESHOLD_MM:
        return "MID"
    if mcap_mm < MCAP_LARGE_THRESHOLD_MM:
        return "LARGE"
    return "MEGA"


def bucket_adv_usd(adv_usd: float | None) -> str:
    """Classify ADV$ into liquidity bucket."""
    if adv_usd is None:
        return "UNKNOWN"
    if adv_usd < ADV_ILLIQUID_THRESHOLD:
        return "ILLIQ"
    if adv_usd < ADV_LIQUID_THRESHOLD:
        return "MODERATE"
    return "LIQUID"


def is_liquid(adv_usd: float | None) -> bool:
    """Check if ADV meets minimum liquidity threshold."""
    if adv_usd is None:
        return False
    return adv_usd >= ADV_MINIMUM_THRESHOLD
