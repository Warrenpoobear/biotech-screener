"""
Point-in-Time (PIT) enforcement utilities.

Ensures no look-ahead bias by filtering data to as_of_date - 1.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional


def compute_pit_cutoff(as_of_date: str, strict: bool = False) -> str:
    """
    Compute PIT cutoff date.

    Convention: source_date <= as_of_date - 1 (default)
    Strict mode: source_date < as_of_date - 1 (extra day buffer for intraday data)

    This ensures we only use data that would have been available
    before the trading day begins.

    Args:
        as_of_date: The analysis date
        strict: If True, add extra day buffer to prevent intraday data leakage
    """
    dt = date.fromisoformat(as_of_date)
    buffer_days = 2 if strict else 1
    cutoff = dt - timedelta(days=buffer_days)
    return cutoff.isoformat()


def is_pit_admissible(source_date: Optional[str], pit_cutoff: str) -> bool:
    """
    Check if source_date is PIT-admissible.
    
    Returns True if source_date <= pit_cutoff.
    Returns False if source_date is None or after cutoff.
    """
    if source_date is None:
        return False
    
    try:
        src = date.fromisoformat(source_date[:10])
        cutoff = date.fromisoformat(pit_cutoff)
        return src <= cutoff
    except (ValueError, TypeError):
        return False


def filter_pit_admissible(
    records: List[Dict[str, Any]],
    date_field: str,
    pit_cutoff: str,
) -> List[Dict[str, Any]]:
    """
    Filter records to only PIT-admissible ones.
    """
    return [
        r for r in records
        if is_pit_admissible(r.get(date_field), pit_cutoff)
    ]
