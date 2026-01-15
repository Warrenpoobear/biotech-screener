"""
Liquidity Scoring Module - Adaptive Tiered Liquidity Assessment

Provides market-cap-tiered liquidity scoring with ADV and spread components.
Produces deterministic scores with audit logging for governance.

All functions are deterministic (no datetime.now(), no network calls).
"""
import json
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple

from risk_gates import (
    calculate_adv,
    _extract_price,
    _extract_market_cap,
    FLAG_ADV_UNKNOWN,
    FLAG_LOW_LIQUIDITY,
    FLAG_PENNY_STOCK,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

LIQUIDITY_SCORING_VERSION = "1.0.0"

# Market cap tier boundaries (USD)
TIER_MICRO_MAX = 300_000_000           # < $300M
TIER_SMALL_MAX = 2_000_000_000         # $300M - $2B
TIER_MID_MAX = 10_000_000_000          # $2B - $10B
# Large cap: >= $10B

# ADV thresholds by tier (USD)
ADV_THRESHOLDS = {
    "micro": 750_000,
    "small": 2_000_000,
    "mid": 5_000_000,
    "large": 10_000_000,
}

# Spread scoring thresholds (basis points)
SPREAD_LOW_BPS = 50      # Full score (30) at or below
SPREAD_HIGH_BPS = 400    # Zero score at or above

# Penny stock threshold
PENNY_STOCK_PRICE = 2.00
PENNY_STOCK_MAX_SCORE = 10

# Score component weights
ADV_MAX_SCORE = 70
SPREAD_MAX_SCORE = 30

# Risk flag for wide spread
FLAG_WIDE_SPREAD = "WIDE_SPREAD"


# =============================================================================
# TIER CLASSIFICATION
# =============================================================================

def classify_market_cap_tier(market_cap: Optional[float]) -> str:
    """
    Classify ticker into market cap tier.

    Args:
        market_cap: Market cap in USD, or None

    Returns:
        Tier name: "micro", "small", "mid", "large", or "unknown"
    """
    if market_cap is None or market_cap <= 0:
        return "unknown"

    if market_cap < TIER_MICRO_MAX:
        return "micro"
    elif market_cap < TIER_SMALL_MAX:
        return "small"
    elif market_cap < TIER_MID_MAX:
        return "mid"
    else:
        return "large"


def get_adv_threshold_for_tier(tier: str) -> float:
    """
    Get ADV threshold for a market cap tier.

    Args:
        tier: Tier name

    Returns:
        ADV threshold in USD
    """
    return ADV_THRESHOLDS.get(tier, ADV_THRESHOLDS["small"])


# =============================================================================
# SPREAD EXTRACTION
# =============================================================================

def extract_spread_bps(market_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract or compute spread in basis points.

    Strategy:
    1. Use direct spread_bps field if present
    2. Compute from bid/ask if available: (ask - bid) / mid * 10000
    3. Return None if cannot compute

    Args:
        market_record: Market data record

    Returns:
        Spread in basis points, or None
    """
    # Try direct spread field
    for field in ["spread_bps", "bid_ask_spread_bps", "spread"]:
        value = market_record.get(field)
        if value is not None:
            try:
                spread = float(value)
                if spread >= 0:
                    return spread
            except (ValueError, TypeError):
                continue

    # Try to compute from bid/ask
    bid = market_record.get("bid")
    ask = market_record.get("ask")

    if bid is not None and ask is not None:
        try:
            bid_f = float(bid)
            ask_f = float(ask)
            if bid_f > 0 and ask_f > bid_f:
                mid = (bid_f + ask_f) / 2
                spread_pct = (ask_f - bid_f) / mid
                return spread_pct * 10000  # Convert to basis points
        except (ValueError, TypeError):
            pass

    return None


# =============================================================================
# SCORE COMPUTATION
# =============================================================================

def compute_adv_score(adv_usd: float, tier_threshold: float) -> int:
    """
    Compute ADV component of liquidity score.

    Linear scaling:
    - 0 at ADV = 0
    - 70 at ADV = 2x tier threshold
    - Clipped to [0, 70]

    Args:
        adv_usd: Average dollar volume in USD
        tier_threshold: ADV threshold for the ticker's tier

    Returns:
        ADV score (0-70)
    """
    if adv_usd <= 0 or tier_threshold <= 0:
        return 0

    # 2x threshold gives full score
    target = tier_threshold * 2
    ratio = adv_usd / target

    # Linear scaling, capped at max
    score = int(ratio * ADV_MAX_SCORE)
    return min(max(score, 0), ADV_MAX_SCORE)


def compute_spread_score(spread_bps: Optional[float]) -> int:
    """
    Compute spread component of liquidity score.

    Linear scaling:
    - 30 at spread <= 50bps
    - 0 at spread >= 400bps
    - Linear interpolation between

    Args:
        spread_bps: Spread in basis points, or None

    Returns:
        Spread score (0-30)
    """
    if spread_bps is None:
        # No spread data - return 0 (conservative)
        return 0

    if spread_bps <= SPREAD_LOW_BPS:
        return SPREAD_MAX_SCORE

    if spread_bps >= SPREAD_HIGH_BPS:
        return 0

    # Linear interpolation
    range_bps = SPREAD_HIGH_BPS - SPREAD_LOW_BPS
    position = (spread_bps - SPREAD_LOW_BPS) / range_bps
    score = int(SPREAD_MAX_SCORE * (1 - position))

    return min(max(score, 0), SPREAD_MAX_SCORE)


def compute_liquidity_score(
    ticker: str,
    market_data: Dict[str, Dict[str, Any]],
    as_of_date: str = None
) -> Dict[str, Any]:
    """
    Compute comprehensive liquidity score for a ticker.

    Args:
        ticker: Stock ticker symbol
        market_data: Dict mapping ticker -> market data record
        as_of_date: Analysis date for audit trail (optional)

    Returns:
        Dict with:
        - ticker: str
        - liquidity_score: int (0-100)
        - liquidity_tier: str
        - adv_usd: float
        - spread_bps: Optional[float]
        - adv_score: int
        - spread_score: int
        - risk_flags: List[str]
        - score_version: str
    """
    result = {
        "ticker": ticker,
        "liquidity_score": 0,
        "liquidity_tier": "unknown",
        "adv_usd": 0.0,
        "spread_bps": None,
        "adv_score": 0,
        "spread_score": 0,
        "risk_flags": [],
        "score_version": LIQUIDITY_SCORING_VERSION,
    }

    record = market_data.get(ticker, {})
    if not record:
        result["risk_flags"].append(FLAG_ADV_UNKNOWN)
        return result

    # Extract metrics
    adv_usd = calculate_adv(ticker, market_data)
    market_cap = _extract_market_cap(record)
    price = _extract_price(record)
    spread_bps = extract_spread_bps(record)

    result["adv_usd"] = adv_usd
    result["spread_bps"] = spread_bps

    # Classify tier
    tier = classify_market_cap_tier(market_cap)
    result["liquidity_tier"] = tier

    # Handle ADV unknown
    if adv_usd <= 0:
        result["risk_flags"].append(FLAG_ADV_UNKNOWN)
        return result

    # Get tier-specific threshold
    tier_threshold = get_adv_threshold_for_tier(tier)

    # Check low liquidity
    if adv_usd < tier_threshold:
        result["risk_flags"].append(FLAG_LOW_LIQUIDITY)

    # Check wide spread
    if spread_bps is not None and spread_bps >= SPREAD_HIGH_BPS:
        result["risk_flags"].append(FLAG_WIDE_SPREAD)

    # Compute score components
    adv_score = compute_adv_score(adv_usd, tier_threshold)
    spread_score = compute_spread_score(spread_bps)

    result["adv_score"] = adv_score
    result["spread_score"] = spread_score

    # Total score
    total_score = adv_score + spread_score

    # Penny stock penalty
    if price is not None and price < PENNY_STOCK_PRICE:
        result["risk_flags"].append(FLAG_PENNY_STOCK)
        total_score = min(total_score, PENNY_STOCK_MAX_SCORE)

    result["liquidity_score"] = total_score

    # Sort flags for determinism
    result["risk_flags"].sort()

    return result


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def create_liquidity_audit_record(
    ticker: str,
    score_result: Dict[str, Any],
    as_of_date: str,
    thresholds_by_tier: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create an audit record for liquidity scoring.

    Args:
        ticker: Stock ticker symbol
        score_result: Result from compute_liquidity_score
        as_of_date: Analysis date
        thresholds_by_tier: Threshold configuration used

    Returns:
        Audit record dict with deterministic key ordering
    """
    if thresholds_by_tier is None:
        thresholds_by_tier = ADV_THRESHOLDS

    return {
        "as_of_date": as_of_date,
        "ticker": ticker,
        "score_version": LIQUIDITY_SCORING_VERSION,
        "thresholds_by_tier": dict(sorted(thresholds_by_tier.items())),
        "chosen_tier": score_result.get("liquidity_tier", "unknown"),
        "adv_usd": score_result.get("adv_usd", 0.0),
        "spread_bps": score_result.get("spread_bps"),
        "adv_score": score_result.get("adv_score", 0),
        "spread_score": score_result.get("spread_score", 0),
        "liquidity_score": score_result.get("liquidity_score", 0),
        "risk_flags": score_result.get("risk_flags", []),
    }


def append_audit_log(
    filepath: str,
    record: Dict[str, Any]
) -> None:
    """
    Append audit record to JSONL log file.

    Args:
        filepath: Path to audit log file
        record: Audit record to append
    """
    # Serialize with sorted keys for determinism
    line = json.dumps(record, sort_keys=True, separators=(',', ':'))

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def score_all_tickers(
    tickers: List[str],
    market_data: Dict[str, Dict[str, Any]],
    as_of_date: str,
    audit_log_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Score liquidity for all tickers with optional audit logging.

    Args:
        tickers: List of ticker symbols
        market_data: Dict mapping ticker -> market data record
        as_of_date: Analysis date
        audit_log_path: Path to JSONL audit log (optional)

    Returns:
        List of score results, sorted by ticker for determinism
    """
    results = []

    for ticker in sorted(tickers):  # Sort for determinism
        score_result = compute_liquidity_score(ticker, market_data, as_of_date)
        results.append(score_result)

        if audit_log_path:
            audit_record = create_liquidity_audit_record(
                ticker, score_result, as_of_date
            )
            append_audit_log(audit_log_path, audit_record)

    return results


# =============================================================================
# PARAMETER SNAPSHOT
# =============================================================================

def get_parameters_snapshot() -> Dict[str, Any]:
    """
    Get current parameter values for audit trail.

    Returns:
        Dict of all scoring parameters
    """
    return {
        "version": LIQUIDITY_SCORING_VERSION,
        "TIER_MICRO_MAX": TIER_MICRO_MAX,
        "TIER_SMALL_MAX": TIER_SMALL_MAX,
        "TIER_MID_MAX": TIER_MID_MAX,
        "ADV_THRESHOLDS": dict(sorted(ADV_THRESHOLDS.items())),
        "SPREAD_LOW_BPS": SPREAD_LOW_BPS,
        "SPREAD_HIGH_BPS": SPREAD_HIGH_BPS,
        "PENNY_STOCK_PRICE": PENNY_STOCK_PRICE,
        "PENNY_STOCK_MAX_SCORE": PENNY_STOCK_MAX_SCORE,
        "ADV_MAX_SCORE": ADV_MAX_SCORE,
        "SPREAD_MAX_SCORE": SPREAD_MAX_SCORE,
    }


def compute_parameters_hash() -> str:
    """
    Compute SHA256 hash of parameters for audit trail.

    Returns:
        First 16 characters of SHA256 hash
    """
    import hashlib
    params = get_parameters_snapshot()
    canonical = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
