"""
Risk Gates Module - Deterministic, Fail-Closed Risk Assessment

Provides liquidity, price, market cap, and cash runway gates for
institutional signal filtering. Implements fail-closed philosophy:
if a required metric cannot be computed, the gate rejects.

All functions are deterministic (no datetime.now(), no network calls).
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Risk Gate Thresholds
# =============================================================================

# Version tracking for audit trail
RISK_GATES_VERSION = "1.0.0"

# Liquidity thresholds
ADV_MINIMUM = 500_000           # $500K minimum average dollar volume
PRICE_MINIMUM = 2.00            # $2.00 penny stock threshold
MARKET_CAP_MINIMUM = 50_000_000 # $50M micro cap floor

# Financial thresholds
RUNWAY_MINIMUM_MONTHS = 6       # 6 months minimum cash runway

# Risk flag constants
FLAG_ADV_UNKNOWN = "ADV_UNKNOWN"
FLAG_LOW_LIQUIDITY = "LOW_LIQUIDITY"
FLAG_PENNY_STOCK = "PENNY_STOCK"
FLAG_MICRO_CAP = "MICRO_CAP"
FLAG_CASH_RISK = "CASH_RISK"
FLAG_PRICE_UNKNOWN = "PRICE_UNKNOWN"
FLAG_MARKET_CAP_UNKNOWN = "MARKET_CAP_UNKNOWN"
FLAG_RUNWAY_UNKNOWN = "RUNWAY_UNKNOWN"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load market data from JSON file and index by ticker.

    Args:
        filepath: Path to market_data.json (list of records)

    Returns:
        Dict mapping ticker -> market data record
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle list format (array of records)
    if isinstance(data, list):
        return {record.get("ticker", ""): record for record in data if record.get("ticker")}

    # Handle dict format (already keyed by ticker)
    return data


def load_financial_data(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load financial data from JSON file and index by ticker.

    Args:
        filepath: Path to financial_data.json (list of records)

    Returns:
        Dict mapping ticker -> financial data record
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle list format (array of records)
    if isinstance(data, list):
        return {record.get("ticker", ""): record for record in data if record.get("ticker")}

    # Handle dict format (already keyed by ticker)
    return data


# =============================================================================
# FIELD EXTRACTION HELPERS
# =============================================================================

def _extract_price(market_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract price from market data using priority order.

    Priority: price > current_price > close

    Returns:
        Price as float, or None if not available
    """
    price_fields = ["price", "current_price", "close"]

    for field in price_fields:
        value = market_record.get(field)
        if value is not None:
            try:
                price = float(value)
                if price > 0:
                    return price
            except (ValueError, TypeError):
                continue

    return None


def _extract_volume(market_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract average volume from market data using priority order.

    Priority: avg_volume_20d > avg_volume > volume_avg_30d > volume > avg_volume_90d

    Returns:
        Volume as float, or None if not available
    """
    volume_fields = [
        "avg_volume_20d",
        "avg_volume",
        "volume_avg_30d",
        "volume",
        "avg_volume_90d"
    ]

    for field in volume_fields:
        value = market_record.get(field)
        if value is not None:
            try:
                volume = float(value)
                if volume > 0:
                    return volume
            except (ValueError, TypeError):
                continue

    return None


def _extract_market_cap(market_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract market cap from market data using priority order.

    Priority: market_cap > marketCap > market_cap_usd

    Returns:
        Market cap as float, or None if not available
    """
    mcap_fields = ["market_cap", "marketCap", "market_cap_usd"]

    for field in mcap_fields:
        value = market_record.get(field)
        if value is not None:
            try:
                mcap = float(value)
                if mcap > 0:
                    return mcap
            except (ValueError, TypeError):
                continue

    return None


def _extract_direct_adv(market_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract direct ADV (average dollar volume) field if present.

    Priority: adv_usd_20d > adv_20d_usd > avg_volume_usd > avg_dollar_volume_20d > adv_usd

    Returns:
        ADV in USD as float, or None if no direct field available
    """
    adv_fields = [
        "adv_usd_20d",
        "adv_20d_usd",
        "avg_volume_usd",
        "avg_dollar_volume_20d",
        "adv_usd"
    ]

    for field in adv_fields:
        value = market_record.get(field)
        if value is not None:
            try:
                adv = float(value)
                if adv > 0:
                    return adv
            except (ValueError, TypeError):
                continue

    return None


# =============================================================================
# ADV CALCULATION
# =============================================================================

def calculate_adv(ticker: str, market_data: Dict[str, Dict[str, Any]]) -> float:
    """
    Calculate Average Dollar Volume (ADV) for a ticker.

    Strategy:
    1. Check for direct ADV fields first
    2. If not available, compute from avg_volume * price
    3. If cannot compute, return 0.0 (fail-closed)

    Args:
        ticker: Stock ticker symbol
        market_data: Dict mapping ticker -> market data record

    Returns:
        ADV in USD, or 0.0 if cannot be computed (fail-closed)
    """
    record = market_data.get(ticker)
    if not record:
        logger.debug(f"No market data for {ticker}, ADV = 0 (fail-closed)")
        return 0.0

    # Try direct ADV field first
    direct_adv = _extract_direct_adv(record)
    if direct_adv is not None:
        return direct_adv

    # Fallback: compute from volume * price
    volume = _extract_volume(record)
    price = _extract_price(record)

    if volume is not None and price is not None:
        return volume * price

    # Cannot compute - fail closed
    logger.debug(f"Cannot compute ADV for {ticker}, returning 0 (fail-closed)")
    return 0.0


# =============================================================================
# RUNWAY CALCULATION
# =============================================================================

def calculate_runway_months(
    ticker: str,
    financial_data: Dict[str, Dict[str, Any]]
) -> Optional[float]:
    """
    Calculate cash runway in months for a ticker.

    Strategy:
    1. Check for direct runway_months field
    2. Otherwise compute: Cash / (monthly burn rate)
       - Burn rate from R&D (annualized) or negative NetIncome
    3. If cannot compute, return None (fail-closed in gate)

    Args:
        ticker: Stock ticker symbol
        financial_data: Dict mapping ticker -> financial data record

    Returns:
        Runway in months, or None if cannot be computed
    """
    record = financial_data.get(ticker)
    if not record:
        return None

    # Check for direct runway field
    if "runway_months" in record:
        try:
            return float(record["runway_months"])
        except (ValueError, TypeError):
            pass

    # Get cash
    cash = record.get("Cash")
    if cash is None:
        return None
    try:
        cash = float(cash)
    except (ValueError, TypeError):
        return None

    if cash <= 0:
        return 0.0

    # Estimate monthly burn rate
    # Try R&D first (multiply by 4 to annualize if quarterly, then divide by 12)
    rd_expense = record.get("R&D")
    net_income = record.get("NetIncome")

    monthly_burn = None
    is_profitable = False

    # Check net income first - determine if profitable or burning
    if net_income is not None:
        try:
            ni = float(net_income)
            if ni >= 0:
                # Profitable company = no cash burn risk
                is_profitable = True
            else:
                # Negative net income = cash burn
                # Assume quarterly, annualize then monthly
                monthly_burn = abs(ni) * 4 / 12
        except (ValueError, TypeError):
            pass

    # If profitable, return high runway immediately (no cash risk)
    if is_profitable:
        return 999.0

    # If no burn from net income (missing data), try R&D as proxy
    if monthly_burn is None and rd_expense is not None:
        try:
            rd = float(rd_expense)
            if rd > 0:
                # Assume quarterly, annualize then monthly
                monthly_burn = rd * 4 / 12
        except (ValueError, TypeError):
            pass

    if monthly_burn is None or monthly_burn <= 0:
        # Cannot determine burn rate
        return None

    return cash / monthly_burn


# =============================================================================
# RISK GATES
# =============================================================================

def apply_liquidity_gate(
    ticker: str,
    market_data: Dict[str, Dict[str, Any]]
) -> Tuple[bool, Optional[str]]:
    """
    Apply liquidity-related risk gates.

    Checks (in order):
    1. ADV must be computable (not 0) - fail: ADV_UNKNOWN
    2. ADV >= ADV_MINIMUM - fail: LOW_LIQUIDITY
    3. Price >= PRICE_MINIMUM (if price known) - fail: PENNY_STOCK
    4. Market cap >= MARKET_CAP_MINIMUM (if known) - fail: MICRO_CAP

    Args:
        ticker: Stock ticker symbol
        market_data: Dict mapping ticker -> market data record

    Returns:
        Tuple of (passes: bool, rejection_reason: Optional[str])
    """
    record = market_data.get(ticker, {})

    # Check ADV
    adv = calculate_adv(ticker, market_data)
    if adv <= 0:
        return (False, FLAG_ADV_UNKNOWN)

    if adv < ADV_MINIMUM:
        return (False, FLAG_LOW_LIQUIDITY)

    # Check price (penny stock)
    price = _extract_price(record)
    if price is not None and price < PRICE_MINIMUM:
        return (False, FLAG_PENNY_STOCK)

    # Check market cap (micro cap)
    market_cap = _extract_market_cap(record)
    if market_cap is not None and market_cap < MARKET_CAP_MINIMUM:
        return (False, FLAG_MICRO_CAP)

    return (True, None)


def apply_financial_gate(
    ticker: str,
    financial_data: Dict[str, Dict[str, Any]]
) -> Tuple[bool, Optional[str]]:
    """
    Apply financial health risk gates.

    Checks:
    1. Runway >= RUNWAY_MINIMUM_MONTHS - fail: CASH_RISK

    Note: If runway cannot be computed, we do NOT fail here.
    The signal will carry a RUNWAY_UNKNOWN flag but won't be killed
    solely for missing financial data.

    Args:
        ticker: Stock ticker symbol
        financial_data: Dict mapping ticker -> financial data record

    Returns:
        Tuple of (passes: bool, rejection_reason: Optional[str])
    """
    runway = calculate_runway_months(ticker, financial_data)

    # If runway is unknown, we don't fail the gate but flag it
    # This allows us to not kill signals just because we lack financial data
    if runway is None:
        return (True, None)  # Pass but may have RUNWAY_UNKNOWN flag

    if runway < RUNWAY_MINIMUM_MONTHS:
        return (False, FLAG_CASH_RISK)

    return (True, None)


def apply_all_gates(
    ticker: str,
    market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    financial_data: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Apply all risk gates and return comprehensive results.

    Args:
        ticker: Stock ticker symbol
        market_data: Dict mapping ticker -> market data record (optional)
        financial_data: Dict mapping ticker -> financial data record (optional)

    Returns:
        Dict with:
        - passes: bool - True if all gates passed
        - risk_flags: List[str] - All applicable risk flags
        - adv_usd: float - Calculated ADV
        - price: Optional[float] - Price if available
        - market_cap: Optional[float] - Market cap if available
        - runway_months: Optional[float] - Runway if available
        - rejection_reasons: List[str] - Reasons for gate failures
    """
    result = {
        "passes": True,
        "risk_flags": [],
        "adv_usd": 0.0,
        "price": None,
        "market_cap": None,
        "runway_months": None,
        "rejection_reasons": []
    }

    # Apply liquidity gates if market data provided
    if market_data is not None:
        record = market_data.get(ticker, {})

        # Extract metrics
        result["adv_usd"] = calculate_adv(ticker, market_data)
        result["price"] = _extract_price(record)
        result["market_cap"] = _extract_market_cap(record)

        # Apply gate
        liq_passes, liq_reason = apply_liquidity_gate(ticker, market_data)
        if not liq_passes:
            result["passes"] = False
            if liq_reason:
                result["risk_flags"].append(liq_reason)
                result["rejection_reasons"].append(liq_reason)

        # Add informational flags
        if result["price"] is None:
            result["risk_flags"].append(FLAG_PRICE_UNKNOWN)
        if result["market_cap"] is None:
            result["risk_flags"].append(FLAG_MARKET_CAP_UNKNOWN)

    # Apply financial gates if financial data provided
    if financial_data is not None:
        runway = calculate_runway_months(ticker, financial_data)
        result["runway_months"] = runway

        if runway is None:
            result["risk_flags"].append(FLAG_RUNWAY_UNKNOWN)

        fin_passes, fin_reason = apply_financial_gate(ticker, financial_data)
        if not fin_passes:
            result["passes"] = False
            if fin_reason:
                result["risk_flags"].append(fin_reason)
                result["rejection_reasons"].append(fin_reason)

    # Sort flags for deterministic output
    result["risk_flags"].sort()
    result["rejection_reasons"].sort()

    return result


# =============================================================================
# PARAMETER SNAPSHOT
# =============================================================================

def get_parameters_snapshot() -> Dict[str, Any]:
    """
    Get current parameter values for audit trail.

    Returns:
        Dict of all threshold parameters
    """
    return {
        "version": RISK_GATES_VERSION,
        "ADV_MINIMUM": ADV_MINIMUM,
        "PRICE_MINIMUM": PRICE_MINIMUM,
        "MARKET_CAP_MINIMUM": MARKET_CAP_MINIMUM,
        "RUNWAY_MINIMUM_MONTHS": RUNWAY_MINIMUM_MONTHS,
    }


def compute_parameters_hash() -> str:
    """
    Compute SHA256 hash of parameters for audit trail.

    Returns:
        First 16 characters of SHA256 hash
    """
    import hashlib
    params = get_parameters_snapshot()
    # Canonical JSON (sorted keys, no spaces)
    canonical = json.dumps(params, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
