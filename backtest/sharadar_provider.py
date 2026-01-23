"""
Sharadar Returns Provider

Production-grade returns provider for Sharadar equity prices.
Handles:
- Survivorship bias (includes delisted securities)
- Corporate actions (adjusted prices)
- Decimal precision (6 decimal places)
- PIT-safe date lookups

Sharadar Schema (SEP table):
  ticker, date, open, high, low, close, volume, closeadj, closeunadj, lastupdated

Usage:
    # From local CSV export
    provider = SharadarReturnsProvider.from_csv("path/to/SEP.csv")
    
    # With API (requires QUANDL_API_KEY)
    provider = SharadarReturnsProvider.from_api(api_key="...")
"""
from __future__ import annotations

import csv
import json
import urllib.request
import urllib.parse
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Quantization for determinism
PRICE_QUANTIZE = Decimal("0.000001")  # 6 decimal places
RETURN_QUANTIZE = Decimal("0.000001")

# Date tolerance for price lookups
DATE_TOLERANCE_DAYS = 5

# Delisting policies
DELISTING_POLICY_CONSERVATIVE = "conservative"  # Return None if delisted (drops observation)
DELISTING_POLICY_PENALTY = "penalty"            # Return -1.0 if delisted (harsh)
DELISTING_POLICY_LAST_PRICE = "last_price"      # Use last available price (mild survivorship bias)


class SharadarReturnsProvider:
    """
    Returns provider using Sharadar equity prices.
    
    Supports both local CSV and API access.
    Handles delisted securities with configurable policy.
    
    Delisting Policies:
        - conservative: Return None if end price missing due to delisting (drops obs)
        - penalty: Return -1.0 if delisted before end_date (harsh, bankruptcy-like)
        - last_price: Use last available price (mild survivorship bias)
    """
    
    def __init__(
        self,
        prices: Dict[str, Dict[str, Decimal]],
        delisting_policy: str = DELISTING_POLICY_CONSERVATIVE,
    ):
        """
        Initialize with price data.
        
        Args:
            prices: Dict[ticker, Dict[date_str, adjusted_close]]
            delisting_policy: How to handle delisted securities
        """
        self._prices = prices
        self._tickers = set(prices.keys())
        self._delisting_policy = delisting_policy
        
        # Build date index for efficient lookups
        self._date_index: Dict[str, List[str]] = {}
        for ticker, ticker_prices in prices.items():
            dates = sorted(ticker_prices.keys())
            self._date_index[ticker] = dates
        
        # Track diagnostics
        self._diagnostics = {
            "delisting_policy": delisting_policy,
            "n_returns_calculated": 0,
            # Missing returns breakdown
            "n_missing_ticker_not_in_data": 0,
            "n_missing_start_price": 0,
            "n_missing_end_price": 0,
            "n_missing_due_to_delist": 0,
        }
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        ticker_filter: Optional[List[str]] = None,
        delisting_policy: str = DELISTING_POLICY_CONSERVATIVE,
    ) -> "SharadarReturnsProvider":
        """
        Load from Sharadar SEP CSV export.
        
        Expected columns: ticker, date, closeadj (or close)
        
        Args:
            filepath: Path to CSV file
            ticker_filter: Optional list of tickers to include
            delisting_policy: How to handle delisted securities
        """
        prices: Dict[str, Dict[str, Decimal]] = {}
        
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                ticker = row.get("ticker", "").upper()
                
                # Apply filter if provided
                if ticker_filter and ticker not in ticker_filter:
                    continue
                
                date_str = row.get("date", "")[:10]
                
                # Prefer adjusted close, fall back to close
                price_str = row.get("closeadj") or row.get("adj_close") or row.get("close")
                if not price_str:
                    continue
                
                try:
                    price = Decimal(price_str).quantize(PRICE_QUANTIZE, rounding=ROUND_HALF_UP)
                except (ValueError, TypeError, InvalidOperation):
                    continue
                
                if ticker not in prices:
                    prices[ticker] = {}
                
                prices[ticker][date_str] = price
        
        return cls(prices, delisting_policy=delisting_policy)
    
    @classmethod
    def from_api(
        cls,
        api_key: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        cache_dir: Optional[Path] = None,
    ) -> "SharadarReturnsProvider":
        """
        Load from Sharadar/Nasdaq Data Link API.
        
        Args:
            api_key: Nasdaq Data Link API key
            tickers: List of tickers to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cache_dir: Optional cache directory
        """
        prices: Dict[str, Dict[str, Decimal]] = {}
        
        # Check cache first
        if cache_dir:
            cache_file = cache_dir / f"sharadar_{start_date}_{end_date}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                # Convert to Decimal
                for ticker, ticker_prices in cached.items():
                    prices[ticker] = {
                        d: Decimal(p) for d, p in ticker_prices.items()
                    }
                return cls(prices)
        
        # Fetch from API
        base_url = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP"
        
        for ticker in tickers:
            params = {
                "ticker": ticker,
                "date.gte": start_date,
                "date.lte": end_date,
                "api_key": api_key,
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                
                # Parse response
                columns = data.get("datatable", {}).get("columns", [])
                rows = data.get("datatable", {}).get("data", [])
                
                # Find column indices
                col_idx = {c["name"]: i for i, c in enumerate(columns)}
                date_idx = col_idx.get("date")
                price_idx = col_idx.get("closeadj")
                
                if date_idx is None or price_idx is None:
                    continue
                
                prices[ticker] = {}
                for row in rows:
                    date_str = str(row[date_idx])[:10]
                    price = Decimal(str(row[price_idx])).quantize(
                        PRICE_QUANTIZE, rounding=ROUND_HALF_UP
                    )
                    prices[ticker][date_str] = price
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        
        # Cache results
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                t: {d: str(p) for d, p in tp.items()}
                for t, tp in prices.items()
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        
        return cls(prices)
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers."""
        return sorted(self._tickers)
    
    def get_date_range(self, ticker: str) -> Optional[Tuple[str, str]]:
        """Get date range for a ticker."""
        if ticker not in self._date_index:
            return None
        dates = self._date_index[ticker]
        return (dates[0], dates[-1]) if dates else None
    
    def _find_nearest_price(
        self,
        ticker: str,
        target_date: str,
        tolerance_days: int = DATE_TOLERANCE_DAYS,
    ) -> Optional[Tuple[str, Decimal]]:
        """
        Find price on or near target date.
        
        Returns (actual_date, price) or None if not found.
        """
        if ticker not in self._prices:
            return None
        
        ticker_prices = self._prices[ticker]
        
        # Check exact date first
        if target_date in ticker_prices:
            return (target_date, ticker_prices[target_date])
        
        # Search within tolerance
        target = date.fromisoformat(target_date)
        
        for offset in range(1, tolerance_days + 1):
            # Check before
            check_date = (target - timedelta(days=offset)).isoformat()
            if check_date in ticker_prices:
                return (check_date, ticker_prices[check_date])
            
            # Check after
            check_date = (target + timedelta(days=offset)).isoformat()
            if check_date in ticker_prices:
                return (check_date, ticker_prices[check_date])
        
        return None
    
    def get_forward_total_return(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Decimal]:
        """
        Calculate forward total return between two dates.
        
        Uses adjusted close prices (handles splits/dividends).
        Applies delisting policy when end price is missing.
        Tracks detailed diagnostics for missing returns.
        
        Args:
            ticker: Stock ticker
            start_date: Period start (YYYY-MM-DD)
            end_date: Period end (YYYY-MM-DD)
        
        Returns:
            Decimal return (e.g., 0.15 for 15%) or None if unavailable
        """
        # Check if ticker exists in data
        if ticker not in self._tickers:
            self._diagnostics["n_missing_ticker_not_in_data"] += 1
            return None
        
        start_result = self._find_nearest_price(ticker, start_date)
        end_result = self._find_nearest_price(ticker, end_date)
        
        if start_result is None:
            self._diagnostics["n_missing_start_price"] += 1
            return None
        
        start_price = start_result[1]
        if start_price == 0:
            self._diagnostics["n_missing_start_price"] += 1
            return None
        
        # Handle missing end price (potential delisting)
        if end_result is None:
            delisted_date = self.get_delisted_date(ticker)
            
            if delisted_date:
                self._diagnostics["n_missing_due_to_delist"] += 1
                
                if self._delisting_policy == DELISTING_POLICY_CONSERVATIVE:
                    # Drop the observation
                    return None
                elif self._delisting_policy == DELISTING_POLICY_PENALTY:
                    # Treat as total loss
                    return Decimal("-1.0")
                elif self._delisting_policy == DELISTING_POLICY_LAST_PRICE:
                    # Use last available price
                    last_price_result = self._find_nearest_price(ticker, delisted_date)
                    if last_price_result:
                        ret = (last_price_result[1] / start_price) - Decimal("1")
                        self._diagnostics["n_returns_calculated"] += 1
                        return ret.quantize(RETURN_QUANTIZE, rounding=ROUND_HALF_UP)
                    return None
            else:
                # End price missing but not clearly delisted
                self._diagnostics["n_missing_end_price"] += 1
            
            return None
        
        end_price = end_result[1]
        
        # Calculate return
        ret = (end_price / start_price) - Decimal("1")
        self._diagnostics["n_returns_calculated"] += 1
        return ret.quantize(RETURN_QUANTIZE, rounding=ROUND_HALF_UP)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics including delisting handling stats."""
        return dict(self._diagnostics)
    
    def __call__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Decimal]:
        """Make provider callable for backtest harness compatibility."""
        return self.get_forward_total_return(ticker, start_date, end_date)
    
    def get_price(self, ticker: str, date_str: str) -> Optional[Decimal]:
        """Get adjusted close price for a specific date."""
        result = self._find_nearest_price(ticker, date_str)
        return result[1] if result else None
    
    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker is available."""
        return ticker in self._tickers
    
    def get_delisted_date(
        self,
        ticker: str,
        as_of_date: Optional[date] = None
    ) -> Optional[str]:
        """
        Get last trading date for a ticker (if delisted).

        Args:
            ticker: Stock ticker symbol
            as_of_date: Reference date for delisting check. REQUIRED for backtests.
                        If None, returns last date without checking recency
                        (for inspection purposes only, not for backtest logic).

        Returns:
            Last trading date if ticker appears delisted, None if active.

        Note:
            For backtest usage, ALWAYS pass as_of_date to ensure reproducibility.
            Using date.today() would cause survivorship bias - a stock that
            delistted after the backtest period would be incorrectly excluded.
        """
        if ticker not in self._date_index:
            return None

        dates = self._date_index[ticker]
        if not dates:
            return None

        last_date = date.fromisoformat(dates[-1])

        # CRITICAL: For reproducibility, require explicit as_of_date for delisting logic
        # Using date.today() would cause different backtest results on different days
        if as_of_date is None:
            # Without as_of_date, we cannot determine if "delisted" - return None
            # This is safer than guessing based on wall-clock time
            return None

        # If last date is more than 10 trading days before as_of_date, consider delisted
        if (as_of_date - last_date).days > 10:
            return dates[-1]

        return None


class PolygonReturnsProvider:
    """
    Returns provider using Polygon.io API.
    
    Similar interface to SharadarReturnsProvider.
    Requires POLYGON_API_KEY.
    """
    
    def __init__(self, prices: Dict[str, Dict[str, Decimal]]):
        self._prices = prices
        self._tickers = set(prices.keys())
        self._date_index: Dict[str, List[str]] = {}
        for ticker, ticker_prices in prices.items():
            self._date_index[ticker] = sorted(ticker_prices.keys())
    
    @classmethod
    def from_api(
        cls,
        api_key: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        cache_dir: Optional[Path] = None,
    ) -> "PolygonReturnsProvider":
        """
        Load from Polygon.io API.
        
        Args:
            api_key: Polygon API key
            tickers: List of tickers to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cache_dir: Optional cache directory
        """
        prices: Dict[str, Dict[str, Decimal]] = {}
        
        # Check cache
        if cache_dir:
            cache_file = cache_dir / f"polygon_{start_date}_{end_date}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                for ticker, ticker_prices in cached.items():
                    prices[ticker] = {
                        d: Decimal(p) for d, p in ticker_prices.items()
                    }
                return cls(prices)
        
        # Fetch from API
        for ticker in tickers:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{start_date}/{end_date}?adjusted=true&apiKey={api_key}"
            )
            
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                
                results = data.get("results", [])
                prices[ticker] = {}
                
                for bar in results:
                    # Polygon returns timestamp in ms
                    ts = bar.get("t", 0) / 1000
                    bar_date = date.fromtimestamp(ts).isoformat()
                    close = Decimal(str(bar.get("c", 0))).quantize(
                        PRICE_QUANTIZE, rounding=ROUND_HALF_UP
                    )
                    prices[ticker][bar_date] = close
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        
        # Cache
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                t: {d: str(p) for d, p in tp.items()}
                for t, tp in prices.items()
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        
        return cls(prices)
    
    # Same interface methods as SharadarReturnsProvider
    def get_available_tickers(self) -> List[str]:
        return sorted(self._tickers)
    
    def get_forward_total_return(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[Decimal]:
        if ticker not in self._prices:
            return None
        
        ticker_prices = self._prices[ticker]
        
        # Find nearest dates
        start_price = ticker_prices.get(start_date)
        end_price = ticker_prices.get(end_date)
        
        if start_price is None or end_price is None or start_price == 0:
            return None
        
        ret = (end_price / start_price) - Decimal("1")
        return ret.quantize(RETURN_QUANTIZE, rounding=ROUND_HALF_UP)


# Factory function
def create_returns_provider(
    source: str,
    **kwargs,
) -> SharadarReturnsProvider | PolygonReturnsProvider:
    """
    Factory to create returns provider.
    
    Args:
        source: "sharadar_csv", "sharadar_api", "polygon_api"
        **kwargs: Provider-specific arguments
    
    Returns:
        Configured returns provider
    """
    if source == "sharadar_csv":
        return SharadarReturnsProvider.from_csv(
            filepath=kwargs["filepath"],
            ticker_filter=kwargs.get("ticker_filter"),
        )
    elif source == "sharadar_api":
        return SharadarReturnsProvider.from_api(
            api_key=kwargs["api_key"],
            tickers=kwargs["tickers"],
            start_date=kwargs["start_date"],
            end_date=kwargs["end_date"],
            cache_dir=kwargs.get("cache_dir"),
        )
    elif source == "polygon_api":
        return PolygonReturnsProvider.from_api(
            api_key=kwargs["api_key"],
            tickers=kwargs["tickers"],
            start_date=kwargs["start_date"],
            end_date=kwargs["end_date"],
            cache_dir=kwargs.get("cache_dir"),
        )
    else:
        raise ValueError(f"Unknown source: {source}")
