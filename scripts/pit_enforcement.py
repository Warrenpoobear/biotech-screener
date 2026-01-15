"""
common/pit_enforcement.py - Point-in-Time enforcement utilities.

Prevents lookahead bias by ensuring all data used in scoring
was actually available at the analysis date.

Key concepts:
- pit_cutoff: The date/time beyond which data is inadmissible
- source_date: When the data was published/filed
- Admissibility: source_date < pit_cutoff
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Union

# Default buffer for 13F filing delays (45 days after quarter end)
DEFAULT_13F_BUFFER_DAYS = 45

# Default buffer for trial registry updates
DEFAULT_REGISTRY_BUFFER_DAYS = 7


def compute_pit_cutoff(
    as_of_date: Union[str, date],
    buffer_days: int = 0,
) -> str:
    """
    Compute point-in-time cutoff from analysis date.
    
    Args:
        as_of_date: The "as of" date for the analysis (YYYY-MM-DD or date)
        buffer_days: Days to subtract for publication lag (default 0)
    
    Returns:
        ISO format date string for cutoff
    
    Example:
        as_of_date = "2024-12-15"
        buffer_days = 0
        → "2024-12-15" (data must be from before this date)
    """
    if isinstance(as_of_date, str):
        parsed = date.fromisoformat(as_of_date[:10])
    else:
        parsed = as_of_date
    
    cutoff = parsed - timedelta(days=buffer_days)
    return cutoff.isoformat()


def is_pit_admissible(
    source_date: Optional[Union[str, date]],
    pit_cutoff: Union[str, date],
) -> bool:
    """
    Check if data with given source date is admissible under PIT rules.
    
    Args:
        source_date: When the data was published/available
        pit_cutoff: The cutoff date (data must be strictly before this)
    
    Returns:
        True if source_date < pit_cutoff, False otherwise
    
    Note:
        Returns True if source_date is None (assumes admissible by default).
        This is intentional - missing dates should be flagged elsewhere,
        not treated as PIT violations.
    """
    if source_date is None:
        return True  # Assume admissible; flag missing date separately
    
    # Parse source date
    if isinstance(source_date, str):
        try:
            source = date.fromisoformat(source_date[:10])
        except ValueError:
            return True  # Can't parse; assume admissible
    else:
        source = source_date
    
    # Parse cutoff
    if isinstance(pit_cutoff, str):
        cutoff = date.fromisoformat(pit_cutoff[:10])
    else:
        cutoff = pit_cutoff
    
    return source < cutoff


def compute_13f_pit_cutoff(quarter_end: Union[str, date]) -> str:
    """
    Compute PIT cutoff for 13F data.
    
    13F filings are due 45 days after quarter end. To use 13F data
    for a given quarter, the analysis date must be after the filing deadline.
    
    Args:
        quarter_end: End of the quarter (e.g., "2024-09-30" for Q3)
    
    Returns:
        First date when Q3 13F data is admissible
    
    Example:
        quarter_end = "2024-09-30"
        → "2024-11-14" (45 days after Q3 end)
    """
    if isinstance(quarter_end, str):
        qe = date.fromisoformat(quarter_end[:10])
    else:
        qe = quarter_end
    
    filing_deadline = qe + timedelta(days=DEFAULT_13F_BUFFER_DAYS)
    return filing_deadline.isoformat()


def get_admissible_quarter(as_of_date: Union[str, date]) -> str:
    """
    Get the most recent quarter for which 13F data is admissible.
    
    Args:
        as_of_date: Analysis date
    
    Returns:
        Quarter string (e.g., "2024Q3")
    
    Example:
        as_of_date = "2024-12-15"
        → "2024Q3" (Q4 filings not yet due)
    """
    if isinstance(as_of_date, str):
        d = date.fromisoformat(as_of_date[:10])
    else:
        d = as_of_date
    
    # Work backwards to find most recent admissible quarter
    year = d.year
    month = d.month
    
    # Quarter end months: 3, 6, 9, 12
    # Filing deadlines: May 15, Aug 14, Nov 14, Feb 14
    
    # Check each quarter's filing deadline
    quarters = [
        (f"{year}Q4", date(year + 1, 2, 14)),  # Q4 due mid-Feb next year
        (f"{year}Q3", date(year, 11, 14)),
        (f"{year}Q2", date(year, 8, 14)),
        (f"{year}Q1", date(year, 5, 15)),
        (f"{year-1}Q4", date(year, 2, 14)),
    ]
    
    for quarter, deadline in quarters:
        if d >= deadline:
            return quarter
    
    # Fallback
    return f"{year-1}Q4"
