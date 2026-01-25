"""
PATCH 002: Deterministic Data Collection
=========================================

CRITICAL FIX: Multiple data collectors use datetime.now() and date.today(),
making data collection non-deterministic.

This patch provides:
1. Wrapper functions that enforce explicit as_of_date
2. Deterministic timestamp handling
3. Collection date validation

Files affected:
- collect_market_data.py:80 (end_date = datetime.now())
- collect_market_data.py:130 (collected_at = date.today())
- extend_universe_yfinance.py:130,213
- wake_robin_data_pipeline/collectors/time_series_collector.py:34

Usage:
    from patches.patch_002_deterministic_collection import (
        DeterministicMarketDataCollector,
        validate_collection_metadata,
    )
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DeterministicCollectionError(Exception):
    """Raised when collection would be non-deterministic."""
    pass


@dataclass(frozen=True)
class CollectionMetadata:
    """Immutable collection metadata."""
    as_of_date: str
    collection_date: str  # The date data was actually collected
    source: str
    record_count: int
    content_hash: str
    is_retrospective: bool  # True if collection_date != as_of_date


def validate_collection_date(
    as_of_date: str,
    max_staleness_days: int = 1
) -> Tuple[bool, str]:
    """
    Validate that we're collecting data for an appropriate date.

    Rules:
    1. Cannot collect data for future dates
    2. For current date, collection is allowed
    3. For past dates, flag as retrospective (backfill)

    Args:
        as_of_date: The date we want data for (YYYY-MM-DD)
        max_staleness_days: Maximum days in the past for non-retrospective

    Returns:
        (is_valid, message)
    """
    as_of = date.fromisoformat(as_of_date)
    today = date.today()

    if as_of > today:
        return False, f"Cannot collect data for future date: {as_of_date} (today is {today})"

    days_old = (today - as_of).days

    if days_old == 0:
        return True, "Current date collection"
    elif days_old <= max_staleness_days:
        return True, f"Recent date collection ({days_old} day(s) ago)"
    else:
        return True, f"RETROSPECTIVE collection ({days_old} days ago) - flag as backfill"


def enforce_explicit_as_of_date(func):
    """
    Decorator that enforces explicit as_of_date parameter.

    Raises DeterministicCollectionError if as_of_date is not provided.
    """
    def wrapper(*args, **kwargs):
        # Check for as_of_date in kwargs
        as_of_date = kwargs.get('as_of_date')

        # Check for as_of_date in args (assuming it's first positional arg)
        if as_of_date is None and len(args) > 0:
            # This is fragile - prefer kwargs
            as_of_date = args[0] if isinstance(args[0], str) and len(args[0]) == 10 else None

        if as_of_date is None:
            raise DeterministicCollectionError(
                f"Function {func.__name__} requires explicit as_of_date parameter. "
                "Using date.today() or datetime.now() violates determinism. "
                "Pass as_of_date='YYYY-MM-DD' explicitly."
            )

        # Validate the date
        is_valid, message = validate_collection_date(as_of_date)
        if not is_valid:
            raise DeterministicCollectionError(message)

        logger.debug(f"{func.__name__}: {message}")

        return func(*args, **kwargs)

    return wrapper


class DeterministicMarketDataCollector:
    """
    Wrapper for market data collection that enforces determinism.

    Key differences from collect_market_data.py:
    1. as_of_date is REQUIRED, not optional
    2. No datetime.now() calls - explicit dates only
    3. Timestamps use as_of_date, not collection time
    4. Retrospective collections are flagged

    Usage:
        collector = DeterministicMarketDataCollector()
        data = collector.collect(
            as_of_date='2026-01-15',
            universe_file=Path('production_data/universe.json'),
            output_file=Path('production_data/market_data.json'),
        )
    """

    def __init__(self, source: str = "yahoo_finance"):
        self.source = source

    def collect(
        self,
        as_of_date: str,  # REQUIRED - no default!
        universe_file: Path,
        output_file: Path,
        force_refresh: bool = False,
    ) -> CollectionMetadata:
        """
        Collect market data with deterministic timestamps.

        Args:
            as_of_date: The date to collect data for (REQUIRED)
            universe_file: Path to universe.json
            output_file: Path to save market_data.json
            force_refresh: If True, ignore cache

        Returns:
            CollectionMetadata with collection details

        Raises:
            DeterministicCollectionError: If as_of_date is not provided or invalid
        """
        # Validate as_of_date
        is_valid, message = validate_collection_date(as_of_date)
        if not is_valid:
            raise DeterministicCollectionError(message)

        collection_date = date.today().isoformat()
        is_retrospective = collection_date != as_of_date

        if is_retrospective:
            logger.warning(
                f"RETROSPECTIVE COLLECTION: as_of_date={as_of_date}, "
                f"collection_date={collection_date}. Data may differ from "
                "what was available on as_of_date."
            )

        # Load universe
        with open(universe_file) as f:
            universe = json.load(f)

        tickers = [
            s['ticker'] for s in universe
            if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_'
        ]

        # Collect data with deterministic timestamps
        all_data = []

        for ticker in tickers:
            data = self._get_ticker_data(ticker, as_of_date)
            if data:
                # Use as_of_date in metadata, NOT collection time
                data['_metadata'] = {
                    'as_of_date': as_of_date,
                    'collection_date': collection_date,
                    'source': self.source,
                    'is_retrospective': is_retrospective,
                }
                # CRITICAL: collected_at uses as_of_date, not now()
                data['collected_at'] = as_of_date
                all_data.append(data)

        # Compute content hash (for reproducibility checking)
        import hashlib
        content_hash = hashlib.sha256(
            json.dumps(all_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Save output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2, sort_keys=True)

        return CollectionMetadata(
            as_of_date=as_of_date,
            collection_date=collection_date,
            source=self.source,
            record_count=len(all_data),
            content_hash=content_hash,
            is_retrospective=is_retrospective,
        )

    def _get_ticker_data(
        self,
        ticker: str,
        as_of_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get market data for a single ticker.

        CRITICAL: Uses as_of_date as end date, NOT datetime.now()
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Use as_of_date as end date, NOT datetime.now()
            end_date = date.fromisoformat(as_of_date)
            start_date = end_date - timedelta(days=90)

            hist = stock.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat()  # end is exclusive
            )

            if hist.empty:
                return None

            # Calculate metrics using data up to as_of_date
            # NOT using current prices (which would be look-ahead)
            hist_to_date = hist[hist.index.date <= end_date]

            if hist_to_date.empty:
                return None

            current_price = float(hist_to_date['Close'].iloc[-1])
            avg_volume_90d = float(hist_to_date['Volume'].mean())

            # Volatility calculation - corrected (sqrt(252), not *252)
            returns = hist_to_date['Close'].pct_change()
            volatility_90d = float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else None

            info = stock.info

            return {
                "ticker": ticker,
                "price": current_price,
                "market_cap": info.get('marketCap'),
                "avg_volume": info.get('averageVolume'),
                "avg_volume_90d": avg_volume_90d,
                "volatility_90d": volatility_90d,
                # ... other fields
            }

        except Exception as e:
            logger.warning(f"Failed to get data for {ticker}: {e}")
            return None


def validate_collection_metadata(
    data_file: Path,
    expected_as_of_date: str,
) -> Tuple[bool, List[str]]:
    """
    Validate that collected data has proper metadata.

    Checks:
    1. All records have as_of_date
    2. as_of_date matches expected
    3. Retrospective collections are flagged

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not data_file.exists():
        return False, [f"File not found: {data_file}"]

    with open(data_file) as f:
        data = json.load(f)

    if not data:
        return False, ["Empty data file"]

    for i, record in enumerate(data):
        ticker = record.get('ticker', f'record_{i}')

        # Check collected_at
        collected_at = record.get('collected_at')
        if not collected_at:
            issues.append(f"{ticker}: Missing collected_at field")
        elif collected_at != expected_as_of_date:
            # This might be OK if retrospective, but should be flagged
            metadata = record.get('_metadata', {})
            if not metadata.get('is_retrospective'):
                issues.append(
                    f"{ticker}: collected_at={collected_at} != expected={expected_as_of_date}"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    print("=" * 70)
    print("PATCH 002: Deterministic Data Collection")
    print("=" * 70)
    print()
    print("This patch fixes non-deterministic data collection.")
    print()
    print("PROBLEM with original implementation:")
    print("  - Uses datetime.now() and date.today() for timestamps")
    print("  - Same script run on different days produces different data")
    print("  - Breaks reproducibility and audit trails")
    print()
    print("SOLUTION (this patch):")
    print("  - Requires explicit as_of_date parameter")
    print("  - Uses as_of_date for all timestamps, not collection time")
    print("  - Flags retrospective collections explicitly")
    print()
    print("Usage:")
    print("  from patches.patch_002_deterministic_collection import (")
    print("      DeterministicMarketDataCollector")
    print("  )")
    print("  collector = DeterministicMarketDataCollector()")
    print("  metadata = collector.collect(")
    print("      as_of_date='2026-01-15',  # REQUIRED!")
    print("      universe_file=Path('production_data/universe.json'),")
    print("      output_file=Path('production_data/market_data.json'),")
    print("  )")
    print()
