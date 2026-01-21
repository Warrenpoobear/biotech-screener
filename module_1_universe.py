"""
Module 1: Universe Management

Manages the investable universe with status gates.
Filters shell companies, delisted, acquired, etc.

Input: Raw ticker list with metadata
Output: Filtered universe with status classifications
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.types import StatusGate
from common.integration_contracts import (
    validate_module_1_output,
    is_validation_enabled,
)

RULESET_VERSION = "1.1.0"

# Minimum market cap for inclusion (in millions)
MIN_MARKET_CAP_MM = Decimal("50")

# Shell company indicators
SHELL_KEYWORDS = frozenset([
    "acquisition corp", "spac", "blank check", "shell company",
])


def _extract_market_cap_mm(record: Dict[str, Any]) -> Optional[Decimal]:
    """
    Extract market cap in millions from various data structures.

    Handles:
    - Direct field: market_cap_mm
    - Nested field: market_data.market_cap (in raw $, needs conversion)
    - Flat field: market_cap (in raw $, needs conversion)
    """
    # Try direct mm field first
    market_cap_mm = record.get("market_cap_mm")
    if market_cap_mm is not None:
        try:
            return Decimal(str(market_cap_mm))
        except (ValueError, TypeError, InvalidOperation):
            pass

    # Try nested market_data.market_cap (raw $)
    market_data = record.get("market_data", {})
    if isinstance(market_data, dict):
        raw_cap = market_data.get("market_cap")
        if raw_cap is not None:
            try:
                return Decimal(str(raw_cap)) / Decimal("1000000")
            except (ValueError, TypeError, InvalidOperation):
                pass

    # Try flat market_cap (raw $)
    raw_cap = record.get("market_cap")
    if raw_cap is not None:
        try:
            return Decimal(str(raw_cap)) / Decimal("1000000")
        except (ValueError, TypeError, InvalidOperation):
            pass

    return None


def _classify_status(record: Dict[str, Any]) -> tuple[StatusGate, Optional[str]]:
    """
    Classify security status.

    Returns:
        Tuple of (StatusGate, reason_detail) where reason_detail provides
        additional context for exclusions (e.g., which keyword matched).
    """
    # Check for explicit status
    status = record.get("status", "").lower()
    if status in ("delisted", "d"):
        return (StatusGate.EXCLUDED_DELISTED, "status=delisted")
    if status in ("acquired", "m&a"):
        return (StatusGate.EXCLUDED_ACQUIRED, "status=acquired")

    # Check for shell indicators in company name
    name = record.get("company_name", "")
    # Also check nested market_data.company_name
    if not name:
        market_data = record.get("market_data", {})
        if isinstance(market_data, dict):
            name = market_data.get("company_name", "")
    name_lower = name.lower() if name else ""

    for keyword in SHELL_KEYWORDS:
        if keyword in name_lower:
            return (StatusGate.EXCLUDED_SHELL, f"shell_keyword={keyword}")

    # Extract market cap (handles various data structures)
    market_cap_mm = _extract_market_cap_mm(record)

    # FAIL-LOUD: Missing market cap is an exclusion (data quality issue)
    if market_cap_mm is None:
        return (StatusGate.EXCLUDED_MISSING_DATA, "missing_market_cap")

    # Check minimum market cap threshold
    if market_cap_mm < MIN_MARKET_CAP_MM:
        return (StatusGate.EXCLUDED_SMALL_CAP, f"market_cap_mm={market_cap_mm:.1f}<{MIN_MARKET_CAP_MM}")

    return (StatusGate.ACTIVE, None)


def compute_module_1_universe(
    raw_records: List[Dict[str, Any]],
    as_of_date: str,
    universe_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute universe with status classifications.
    
    Args:
        raw_records: List of dicts with ticker, company_name, status, market_cap_mm, etc.
        as_of_date: Analysis date (YYYY-MM-DD)
        universe_tickers: Optional whitelist of tickers to include
    
    Returns:
        {
            "as_of_date": str,
            "active_securities": [{ticker, status, market_cap_mm, ...}],
            "excluded_securities": [{ticker, reason, ...}],
            "diagnostic_counts": {total, active, excluded_by_reason},
            "provenance": {...}
        }
    """
    active = []
    excluded = []
    reason_counts = {}

    for record in raw_records:
        ticker = record.get("ticker", "").upper()
        if not ticker:
            continue

        # Check universe whitelist
        if universe_tickers and ticker not in universe_tickers:
            continue

        status, reason_detail = _classify_status(record)

        if status == StatusGate.ACTIVE:
            # Extract market cap for output (already validated if we're here)
            market_cap_mm = _extract_market_cap_mm(record)

            # Get company name from various locations
            company_name = record.get("company_name")
            if not company_name:
                market_data = record.get("market_data", {})
                if isinstance(market_data, dict):
                    company_name = market_data.get("company_name")

            active.append({
                "ticker": ticker,
                "status": status.value,
                "market_cap_mm": str(market_cap_mm) if market_cap_mm else None,
                "company_name": company_name,
            })
        else:
            reason = status.value
            excluded.append({
                "ticker": ticker,
                "reason": reason,
                "reason_detail": reason_detail,
            })
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    output = {
        "as_of_date": as_of_date,
        "active_securities": sorted(active, key=lambda x: x["ticker"]),
        "excluded_securities": sorted(excluded, key=lambda x: x["ticker"]),
        "diagnostic_counts": {
            "total_input": len(raw_records),
            "active": len(active),
            "excluded": len(excluded),
            "excluded_by_reason": reason_counts,
        },
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": [r.get("ticker") for r in raw_records]},
            as_of_date,
        ),
    }

    # Output validation
    if is_validation_enabled():
        validate_module_1_output(output)

    return output
