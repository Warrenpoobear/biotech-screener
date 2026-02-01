"""
Ticker validation for biotech universe.
Implements fail-loud validation to prevent data contamination.
"""
from typing import Tuple, List, Dict, TypedDict, Union
import re

# Known delisted/merged securities (as of Jan 2026)
# Extracted as module constant so validate_blacklist() can cross-reference market data.
DELISTED_TICKERS = [
    'ADRO',   # Delisted
    'AKE',    # Delisted
    'BGXX',   # Bad data (0% volatility)
    'CRGX',   # Delisted
    'THRD',   # Delisted
    'SGAFT',  # Delisted/Invalid
    'XTSLA',  # Invalid ticker
]


class ValidationStats(TypedDict):
    """Statistics from ticker validation."""
    total_input: int
    valid_count: int
    invalid_count: int
    pass_rate: float


class ValidationResult(TypedDict):
    """Result from validate_ticker_list."""
    valid: List[str]
    invalid: Dict[str, str]
    stats: ValidationStats


def is_valid_ticker(ticker: str, allow_internal: bool = True) -> Tuple[bool, str]:
    """
    Validate that a string is a legitimate ticker symbol.

    Args:
        ticker: String to validate as ticker
        allow_internal: If True, allow internal tickers like _XBI_BENCHMARK_

    Returns:
        (is_valid, reason) tuple

    Examples:
        >>> is_valid_ticker("MRNA")
        (True, "")
        >>> is_valid_ticker("THE CONTENT...")
        (False, "Contains non-ticker keywords")
        >>> is_valid_ticker("_XBI_BENCHMARK_")
        (True, "")
    """
    if not ticker:
        return False, "Empty string"

    # Remove whitespace
    ticker = ticker.strip()

    # Allow internal/benchmark tickers (start and end with underscore)
    if allow_internal and ticker.startswith('_') and ticker.endswith('_'):
        # Validate internal ticker format: _NAME_ (alphanumeric with underscores)
        inner = ticker[1:-1]
        if len(inner) >= 1 and all(c.isalnum() or c == '_' for c in inner):
            return True, ""

    # Length check - US tickers are typically 1-5 chars, max 6 for some cases
    if len(ticker) > 6:
        return False, f"Too long ({len(ticker)} chars, max 6)"

    if len(ticker) == 0:
        return False, "Empty after strip"

    # Character validation - only alphanumeric, dots, hyphens
    if not re.match(r'^[A-Z0-9][A-Z0-9.\-]*$', ticker.upper()):
        return False, "Contains invalid characters (only A-Z, 0-9, '.', '-' allowed)"

    # Blacklist - known contamination patterns
    contamination_keywords = [
        'THE', 'CONTENT', 'COPYRIGHT', 'BLACKROCK', 'HEREIN',
        'OWNED', 'LICENSED', 'HOLDINGS', 'SUBJECT', 'CHANGE',
        'TRADEMARK', 'ISHARES', 'PROSPECTUS', 'CAREFULLY',
        'NAREIT', 'FTSE', 'EPRA', 'INDEX', 'AFFILIATES',
        'RIGHTS', 'RESERVED', 'IBONDS', 'LICENSE'
    ]

    ticker_upper = ticker.upper()
    for keyword in contamination_keywords:
        if keyword in ticker_upper:
            return False, f"Contains contamination keyword '{keyword}'"

    # Blacklist - known invalid/placeholder tickers
    invalid_tickers = ['NAME', 'TICKER', 'SYMBOL', 'TEST', 'EXAMPLE', 'NULL', 'NONE', 'NA', 'N/A']
    if ticker_upper in invalid_tickers:
        return False, f"Placeholder ticker '{ticker}'"

    # Blacklist - known delisted/merged securities
    if ticker_upper in DELISTED_TICKERS:
        return False, f"Delisted/invalid ticker '{ticker}'"

    # Detect futures contract tickers (e.g., RTYH6, IXCH6, ESM5)
    # Pattern: letters followed by month code (F,G,H,J,K,M,N,Q,U,V,X,Z) and year digit
    if re.match(r'^[A-Z]+[FGHJKMNQUVXZ]\d$', ticker_upper):
        return False, f"Futures contract ticker '{ticker}'"

    # Additional sanity checks
    if ticker.startswith('.') or ticker.endswith('.'):
        return False, "Starts or ends with period"

    if ticker.startswith('-') or ticker.endswith('-'):
        return False, "Starts or ends with hyphen"

    # Check for excessive special characters
    special_char_count = ticker.count('.') + ticker.count('-')
    if special_char_count > 1:
        return False, f"Too many special characters ({special_char_count})"

    # Check for just a dash or special char
    if ticker in ['-', '.', '--', '..']:
        return False, "Just special characters"

    return True, ""


def validate_ticker_list(tickers: List[str]) -> ValidationResult:
    """
    Validate a list of tickers and return detailed results.

    Args:
        tickers: List of ticker strings to validate

    Returns:
        ValidationResult with:
        - valid: list of valid tickers
        - invalid: dict mapping invalid ticker to reason
        - stats: summary statistics
    """
    valid: List[str] = []
    invalid: Dict[str, str] = {}

    for ticker in tickers:
        is_valid, reason = is_valid_ticker(ticker)
        if is_valid:
            valid.append(ticker)
        else:
            invalid[ticker] = reason

    stats: ValidationStats = {
        'total_input': len(tickers),
        'valid_count': len(valid),
        'invalid_count': len(invalid),
        'pass_rate': len(valid) / len(tickers) if tickers else 0.0
    }

    return {
        'valid': valid,
        'invalid': invalid,
        'stats': stats
    }


def validate_blacklist(market_data: List[Dict]) -> List[str]:
    """
    Cross-reference DELISTED_TICKERS against market data.
    Returns warnings for any blacklisted ticker that has valid price/market_cap.
    """
    warnings = []
    mkt_by_ticker = {m.get('ticker', '').upper(): m for m in market_data if isinstance(m, dict)}
    for ticker in DELISTED_TICKERS:
        mkt = mkt_by_ticker.get(ticker)
        if mkt:
            price = mkt.get('price') or mkt.get('regularMarketPrice')
            mcap = mkt.get('market_cap') or mkt.get('marketCap')
            if price and price > 0 and mcap and mcap > 100_000_000:
                warnings.append(
                    f"BLACKLIST CONFLICT: {ticker} is in DELISTED_TICKERS "
                    f"but has price=${price:.2f}, market_cap=${mcap/1e9:.1f}B"
                )
    return warnings


def clean_ticker(ticker: str) -> str:
    """
    Clean a ticker string (strip whitespace, uppercase).

    Args:
        ticker: Raw ticker string

    Returns:
        Cleaned ticker string
    """
    if not ticker:
        return ""
    return ticker.strip().upper()


if __name__ == "__main__":
    # Quick test
    test_cases = [
        "MRNA",
        "VRTX",
        "BRK.B",
        "BF-B",
        "_XBI_BENCHMARK_",  # Internal ticker - should pass
        "",
        "TOOLONGX",
        "THE CONTENT",
        "COPYRIGHT",
        "BLACKROCK",
        "-",
        ".ABC",
        "©2023 BLACKROCK, INC",
    ]

    print("Ticker Validation Test")
    print("=" * 50)

    for ticker in test_cases:
        is_valid, reason = is_valid_ticker(ticker)
        status = "✓" if is_valid else "✗"
        reason_str = f" ({reason})" if reason else ""
        print(f"  {status} '{ticker}'{reason_str}")
