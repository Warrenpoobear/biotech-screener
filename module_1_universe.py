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

RULESET_VERSION = "1.0.0"

# Minimum market cap for inclusion (in millions)
MIN_MARKET_CAP_MM = Decimal("50")

# Shell company indicators
SHELL_KEYWORDS = frozenset([
    "acquisition corp", "spac", "blank check", "shell company",
])


def _classify_status(record: Dict[str, Any]) -> StatusGate:
    """Classify security status."""
    # Check for explicit status
    status = record.get("status", "").lower()
    if status in ("delisted", "d"):
        return StatusGate.EXCLUDED_DELISTED
    if status in ("acquired", "m&a"):
        return StatusGate.EXCLUDED_ACQUIRED
    
    # Check for shell indicators
    name = record.get("company_name", "").lower()
    for keyword in SHELL_KEYWORDS:
        if keyword in name:
            return StatusGate.EXCLUDED_SHELL
    
    # Check minimum market cap
    market_cap = record.get("market_cap_mm")
    if market_cap is not None:
        try:
            if Decimal(str(market_cap)) < MIN_MARKET_CAP_MM:
                return StatusGate.EXCLUDED_SHELL
        except (ValueError, TypeError, InvalidOperation):
            pass  # Invalid market cap format, continue to ACTIVE
    
    return StatusGate.ACTIVE


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
        
        status = _classify_status(record)
        
        if status == StatusGate.ACTIVE:
            active.append({
                "ticker": ticker,
                "status": status.value,
                "market_cap_mm": str(record.get("market_cap_mm", "")) if record.get("market_cap_mm") else None,
                "company_name": record.get("company_name"),
            })
        else:
            reason = status.value
            excluded.append({
                "ticker": ticker,
                "reason": reason,
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
