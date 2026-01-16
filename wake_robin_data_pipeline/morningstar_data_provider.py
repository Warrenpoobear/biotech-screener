"""
morningstar_data_provider.py - Morningstar Direct API integration for daily returns

Uses morningstar_data.direct.portfolio API with data_set_id="2" for daily returns.
Provides point-in-time safe price/return series with caching.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import hashlib
import math

# Morningstar Data SDK
try:
    import morningstar_data as md
    MORNINGSTAR_AVAILABLE = True
except ImportError:
    md = None
    MORNINGSTAR_AVAILABLE = False

# Fallback to yfinance if Morningstar unavailable
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

# Constants
CACHE_DIR = Path("cache/morningstar_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24
DECIMAL_PRECISION = Decimal("0.00000001")
TRADING_DAYS_PER_YEAR = 252

# Morningstar Data Set IDs
MSTAR_DATA_SET_SNAPSHOT = "1"
MSTAR_DATA_SET_RETURNS_DAILY = "2"


def _quantize(value: float) -> Decimal:
    """Quantize a float to standard decimal precision."""
    return Decimal(str(value)).quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)


def _date_to_str(d: date) -> str:
    """Convert date to ISO string."""
    return d.isoformat()


def _compute_cache_key(ticker: str, as_of: date, lookback_days: int, source: str = "mstar") -> str:
    """Compute a unique cache key for the data request."""
    key_data = f"{source}|{ticker}|{_date_to_str(as_of)}|{lookback_days}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _is_cache_valid(cache_path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    """Check if cache file is still valid."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() < (ttl_hours * 3600)


def _write_cache(cache_path: Path, data: Dict) -> None:
    """Write data to cache with integrity check."""
    cache_data = {
        "data": data,
        "cached_at": datetime.now().isoformat(),
        "integrity": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    }
    temp_path = cache_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
        temp_path.replace(cache_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Cache write failed: {e}")


def _read_cache(cache_path: Path) -> Optional[Dict]:
    """Read and validate cached data."""
    try:
        with open(cache_path) as f:
            cache_data = json.load(f)
        data = cache_data["data"]
        expected_hash = cache_data["integrity"]
        actual_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if expected_hash != actual_hash:
            print(f"WARNING: Cache integrity mismatch for {cache_path.name}")
            return None
        return data
    except Exception:
        return None


class MorningstarDataProvider:
    """
    Morningstar Direct data provider for daily returns.

    Uses data_set_id="2" (Returns Daily) from Morningstar Direct portfolio API.
    Falls back to yfinance if Morningstar is unavailable.
    """

    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS, prefer_morningstar: bool = True):
        """
        Initialize the Morningstar data provider.

        Args:
            cache_ttl_hours: Hours before cache expires
            prefer_morningstar: If True, use Morningstar as primary source
        """
        self.cache_ttl_hours = cache_ttl_hours
        self.prefer_morningstar = prefer_morningstar
        self._data_points_cache: Optional[Dict] = None

        if prefer_morningstar and not MORNINGSTAR_AVAILABLE:
            print("WARNING: morningstar_data not installed, falling back to yfinance")

        if not MORNINGSTAR_AVAILABLE and not YFINANCE_AVAILABLE:
            raise RuntimeError("Neither morningstar_data nor yfinance is installed")

    def get_available_data_sets(self) -> Optional[Any]:
        """
        Get all available Morningstar portfolio data sets.

        Returns:
            DataFrame with data_set_id and name columns, or None if unavailable
        """
        if not MORNINGSTAR_AVAILABLE:
            return None

        try:
            return md.direct.portfolio.get_data_sets()
        except Exception as e:
            print(f"ERROR: Failed to fetch Morningstar data sets: {e}")
            return None

    def get_daily_returns_data_points(self) -> Optional[Any]:
        """
        Get data points available in the Returns (Daily) data set.

        Uses data_set_id="2" which is the Returns (Daily) data set.

        Returns:
            DataFrame with data_point_id and name columns, or None if unavailable
        """
        if not MORNINGSTAR_AVAILABLE:
            return None

        try:
            return md.direct.portfolio.get_data_set_data_points(
                data_set_id=MSTAR_DATA_SET_RETURNS_DAILY
            )
        except Exception as e:
            print(f"ERROR: Failed to fetch daily returns data points: {e}")
            return None

    def _get_morningstar_returns(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int
    ) -> Optional[List[float]]:
        """
        Fetch daily returns from Morningstar Direct.

        Uses data_set_id="2" (Returns Daily) with data point PD003 (Total Ret 1 Day).

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date (only data up to this date)
            lookback_days: Number of calendar days to look back

        Returns:
            List of daily returns, or None if fetch failed
        """
        if not MORNINGSTAR_AVAILABLE:
            return None

        try:
            # Calculate date range
            start_date = as_of - timedelta(days=lookback_days)

            returns_df = None

            # Method 1: Use md.direct.get_returns with Frequency enum (preferred)
            if returns_df is None and hasattr(md.direct, 'get_returns'):
                # Try Frequency enum first (avoids deprecation warning)
                try:
                    returns_df = md.direct.get_returns(
                        investments=[ticker],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=as_of.strftime('%Y-%m-%d'),
                        freq=md.direct.Frequency.daily
                    )
                except Exception:
                    # Fall back to string 'daily' if enum doesn't work
                    try:
                        returns_df = md.direct.get_returns(
                            investments=[ticker],
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=as_of.strftime('%Y-%m-%d'),
                            freq='daily'
                        )
                    except Exception:
                        pass

            # Method 2: Use md.direct.get_investment_data with PD003 (Total Ret 1 Day Daily)
            # Note: data_points must be list of dicts, not strings
            if returns_df is None and hasattr(md.direct, 'get_investment_data'):
                try:
                    # PD003 = "Total Ret 1 Day (Daily)" from data_set_id=2
                    # data_points format: list of dicts with datapointId
                    returns_df = md.direct.get_investment_data(
                        investments=[ticker],
                        data_points=[{"datapointId": "PD003"}]
                    )
                except Exception as e2:
                    # Try with different dict key names
                    try:
                        returns_df = md.direct.get_investment_data(
                            investments=[ticker],
                            data_points=[{"id": "PD003"}]
                        )
                    except Exception:
                        pass

            # Method 3: Use md.direct.portfolio.get_data with data_set_id
            if returns_df is None and hasattr(md.direct.portfolio, 'get_data'):
                try:
                    returns_df = md.direct.portfolio.get_data(
                        investments=[ticker],
                        data_set_id=MSTAR_DATA_SET_RETURNS_DAILY,
                        data_points=["PD003"],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=as_of.strftime('%Y-%m-%d')
                    )
                except Exception as e3:
                    try:
                        # Try without date filters
                        returns_df = md.direct.portfolio.get_data(
                            investments=[ticker],
                            data_set_id=MSTAR_DATA_SET_RETURNS_DAILY,
                            data_points=["PD003"]
                        )
                    except Exception:
                        pass

            if returns_df is None or (hasattr(returns_df, 'empty') and returns_df.empty):
                return None

            # Extract returns, filtering for PIT compliance
            returns = []
            for idx, row in returns_df.iterrows():
                # Handle various index types
                if hasattr(idx, 'date'):
                    return_date = idx.date()
                elif hasattr(idx, 'to_pydatetime'):
                    return_date = idx.to_pydatetime().date()
                elif isinstance(idx, str):
                    try:
                        return_date = datetime.strptime(idx, '%Y-%m-%d').date()
                    except ValueError:
                        continue
                else:
                    continue

                if return_date <= as_of and return_date >= start_date:
                    # Try common column names for daily returns
                    ret_value = None
                    for col_name in ['Daily Return', 'DailyReturn', 'daily_return',
                                     'PD003', 'Total Ret 1 Day (Daily)',
                                     'Return', 'return', 'value', ticker]:
                        if col_name in row.index:
                            ret_value = row[col_name]
                            break
                    # Also check columns case-insensitively
                    if ret_value is None:
                        for col in row.index:
                            if 'return' in str(col).lower() and 'monthly' not in str(col).lower():
                                ret_value = row[col]
                                break
                    if ret_value is None and len(row) > 0:
                        ret_value = row.iloc[0]

                    if ret_value is not None:
                        try:
                            ret_float = float(ret_value)
                            if not math.isnan(ret_float):
                                # Convert percentage to decimal if needed (e.g., 1.5 -> 0.015)
                                if abs(ret_float) > 1:
                                    ret_float = ret_float / 100.0
                                returns.append(ret_float)
                        except (ValueError, TypeError):
                            continue

            return returns if returns else None

        except Exception as e:
            print(f"WARNING: Morningstar fetch failed for {ticker}: {e}")
            return None

    def _get_yfinance_returns(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int
    ) -> Optional[List[float]]:
        """
        Fetch daily returns from yfinance as fallback.

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date
            lookback_days: Number of calendar days to look back

        Returns:
            List of daily log returns, or None if fetch failed
        """
        if not YFINANCE_AVAILABLE:
            return None

        try:
            stock = yf.Ticker(ticker)
            today = date.today()

            if as_of < today:
                # Historical data: use start/end dates
                start_date = as_of - timedelta(days=lookback_days)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(as_of + timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    actions=False
                )
            else:
                # Current data: use period
                if lookback_days <= 30:
                    period = '1mo'
                elif lookback_days <= 90:
                    period = '3mo'
                elif lookback_days <= 180:
                    period = '6mo'
                elif lookback_days <= 365:
                    period = '1y'
                else:
                    period = '2y'
                hist = stock.history(period=period, auto_adjust=True, actions=False)

            if hist.empty:
                return None

            # Calculate log returns from prices
            prices = []
            for idx, row in hist.iterrows():
                if hasattr(idx, 'date'):
                    price_date = idx.date()
                else:
                    price_date = idx.to_pydatetime().date()

                if price_date <= as_of:
                    close = row['Close']
                    if not math.isnan(close):
                        prices.append(close)

            if len(prices) < 2:
                return None

            # Calculate log returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0 and prices[i] > 0:
                    returns.append(math.log(prices[i] / prices[i-1]))

            return returns if returns else None

        except Exception as e:
            print(f"WARNING: yfinance fetch failed for {ticker}: {e}")
            return None

    def get_daily_returns(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int = 365,
        use_cache: bool = True
    ) -> List[float]:
        """
        Get daily returns for a ticker with Morningstar as primary source.

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date (only returns up to this date)
            lookback_days: Number of calendar days to look back
            use_cache: Whether to use cached data

        Returns:
            List of daily returns (empty list if fetch failed)
        """
        source = "mstar" if self.prefer_morningstar and MORNINGSTAR_AVAILABLE else "yfinance"

        # Check cache first
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days, source)
            cache_path = CACHE_DIR / f"{cache_key}_returns.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return cached["returns"]

        # Try Morningstar first if preferred
        returns = None
        actual_source = None

        if self.prefer_morningstar and MORNINGSTAR_AVAILABLE:
            returns = self._get_morningstar_returns(ticker, as_of, lookback_days)
            if returns:
                actual_source = "morningstar"

        # Fall back to yfinance
        if returns is None and YFINANCE_AVAILABLE:
            returns = self._get_yfinance_returns(ticker, as_of, lookback_days)
            if returns:
                actual_source = "yfinance"

        if returns is None:
            return []

        # Cache the results
        if use_cache and returns:
            cache_data = {
                "ticker": ticker,
                "as_of": _date_to_str(as_of),
                "lookback_days": lookback_days,
                "source": actual_source,
                "num_returns": len(returns),
                "returns": returns,
            }
            _write_cache(cache_path, cache_data)

        return returns

    def get_prices(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int = 365,
        use_cache: bool = True
    ) -> List[Decimal]:
        """
        Get daily closing prices for a ticker.

        Note: Currently uses yfinance for prices. Morningstar integration
        can be extended to include price data if needed.

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date
            lookback_days: Number of calendar days to look back
            use_cache: Whether to use cached data

        Returns:
            List of Decimal prices (empty list if fetch failed)
        """
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days, "prices")
            cache_path = CACHE_DIR / f"{cache_key}_prices.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return [Decimal(str(p)) for p in cached["prices"]]

        if not YFINANCE_AVAILABLE:
            return []

        try:
            stock = yf.Ticker(ticker)
            today = date.today()

            if as_of < today:
                start_date = as_of - timedelta(days=lookback_days)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(as_of + timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    actions=False
                )
            else:
                if lookback_days <= 30:
                    period = '1mo'
                elif lookback_days <= 90:
                    period = '3mo'
                elif lookback_days <= 365:
                    period = '1y'
                else:
                    period = '2y'
                hist = stock.history(period=period, auto_adjust=True, actions=False)

            if hist.empty:
                return []

            prices = []
            for idx, row in hist.iterrows():
                if hasattr(idx, 'date'):
                    price_date = idx.date()
                else:
                    price_date = idx.to_pydatetime().date()

                if price_date <= as_of:
                    close = row['Close']
                    if not math.isnan(close):
                        prices.append(_quantize(close))

            if use_cache and prices:
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "num_prices": len(prices),
                    "prices": [str(p) for p in prices],
                }
                _write_cache(cache_path, cache_data)

            return prices

        except Exception as e:
            print(f"ERROR: Failed to fetch prices for {ticker}: {e}")
            return []

    def get_volumes(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int = 365,
        use_cache: bool = True
    ) -> List[int]:
        """
        Get daily trading volumes for a ticker.

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date
            lookback_days: Number of calendar days to look back
            use_cache: Whether to use cached data

        Returns:
            List of integer volumes (empty list if fetch failed)
        """
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days, "volumes")
            cache_path = CACHE_DIR / f"{cache_key}_volumes.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return cached["volumes"]

        if not YFINANCE_AVAILABLE:
            return []

        try:
            stock = yf.Ticker(ticker)
            today = date.today()

            if as_of < today:
                start_date = as_of - timedelta(days=lookback_days)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(as_of + timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    actions=False
                )
            else:
                if lookback_days <= 30:
                    period = '1mo'
                elif lookback_days <= 90:
                    period = '3mo'
                elif lookback_days <= 365:
                    period = '1y'
                else:
                    period = '2y'
                hist = stock.history(period=period, auto_adjust=True, actions=False)

            if hist.empty:
                return []

            volumes = []
            for idx, row in hist.iterrows():
                if hasattr(idx, 'date'):
                    volume_date = idx.date()
                else:
                    volume_date = idx.to_pydatetime().date()

                if volume_date <= as_of:
                    vol = row['Volume']
                    if not math.isnan(vol):
                        volumes.append(int(vol))

            if use_cache and volumes:
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "volumes": volumes,
                }
                _write_cache(cache_path, cache_data)

            return volumes

        except Exception as e:
            print(f"ERROR: Failed to fetch volumes for {ticker}: {e}")
            return []

    def get_ticker_data(
        self,
        ticker: str,
        as_of: date,
        lookback_days: int = 365
    ) -> Dict:
        """
        Get complete data package for a ticker.

        Args:
            ticker: Security ticker symbol
            as_of: Point-in-time date
            lookback_days: Number of calendar days to look back

        Returns:
            Dict with prices, returns, volumes, and metadata
        """
        prices = self.get_prices(ticker, as_of, lookback_days)
        returns = self.get_daily_returns(ticker, as_of, lookback_days)
        volumes = self.get_volumes(ticker, as_of, lookback_days)

        return {
            "ticker": ticker,
            "as_of": as_of,
            "prices": prices,
            "returns": returns,
            "volumes": volumes,
            "num_days": len(prices),
            "data_source": "morningstar" if (self.prefer_morningstar and MORNINGSTAR_AVAILABLE) else "yfinance"
        }


class BatchMorningstarProvider:
    """Batch data provider using Morningstar as primary source."""

    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS, prefer_morningstar: bool = True):
        self.provider = MorningstarDataProvider(cache_ttl_hours, prefer_morningstar)

    def get_batch_data(
        self,
        tickers: List[str],
        as_of: date,
        lookback_days: int = 365,
        include_xbi: bool = True
    ) -> Dict[str, Dict]:
        """
        Get data for multiple tickers efficiently.

        Args:
            tickers: List of ticker symbols
            as_of: Point-in-time date
            lookback_days: Number of calendar days to look back
            include_xbi: Whether to include XBI benchmark data

        Returns:
            Dict mapping ticker to data package
        """
        results = {}

        if include_xbi:
            print("Fetching XBI benchmark data...")
            xbi_data = self.provider.get_ticker_data("XBI", as_of, lookback_days)
            results["_xbi_"] = xbi_data

        for ticker in tickers:
            print(f"Fetching {ticker}...")
            try:
                data = self.provider.get_ticker_data(ticker, as_of, lookback_days)
                if data["num_days"] > 0:
                    results[ticker] = data
                else:
                    print(f"  WARNING: No data for {ticker}")
            except Exception as e:
                print(f"  ERROR: Failed to fetch {ticker}: {e}")
                continue

        return results


# Convenience functions for backward compatibility
def get_daily_returns(ticker: str, as_of: date, lookback_days: int = 365) -> List[float]:
    """Get daily returns using Morningstar as primary source."""
    provider = MorningstarDataProvider()
    return provider.get_daily_returns(ticker, as_of, lookback_days)


def get_prices(ticker: str, as_of: date, lookback_days: int) -> List[Decimal]:
    """Get prices (currently via yfinance)."""
    provider = MorningstarDataProvider()
    return provider.get_prices(ticker, as_of, lookback_days)


def get_morningstar_data_sets() -> Optional[Any]:
    """Get available Morningstar data sets."""
    if not MORNINGSTAR_AVAILABLE:
        print("morningstar_data not installed")
        return None
    return md.direct.portfolio.get_data_sets()


def get_morningstar_daily_returns_schema() -> Optional[Any]:
    """Get schema for Morningstar daily returns data set (id=2)."""
    if not MORNINGSTAR_AVAILABLE:
        print("morningstar_data not installed")
        return None
    return md.direct.portfolio.get_data_set_data_points(data_set_id=MSTAR_DATA_SET_RETURNS_DAILY)


def check_morningstar_availability() -> Dict[str, bool]:
    """Check availability of data sources."""
    return {
        "morningstar_available": MORNINGSTAR_AVAILABLE,
        "yfinance_available": YFINANCE_AVAILABLE,
        "primary_source": "morningstar" if MORNINGSTAR_AVAILABLE else "yfinance"
    }


def diagnose_morningstar_api() -> Dict[str, Any]:
    """
    Diagnose available Morningstar API methods and structure.

    Run this to discover what API methods are available for fetching data.
    """
    result = {
        "morningstar_available": MORNINGSTAR_AVAILABLE,
        "modules": {},
        "methods": {}
    }

    if not MORNINGSTAR_AVAILABLE:
        print("morningstar_data not installed")
        return result

    print("=== Morningstar API Structure ===\n")

    # Check top-level modules
    print("Top-level modules in md:")
    top_level = [x for x in dir(md) if not x.startswith('_')]
    print(f"  {top_level}")
    result["modules"]["top_level"] = top_level

    # Check md.direct
    if hasattr(md, 'direct'):
        print("\nModules in md.direct:")
        direct_modules = [x for x in dir(md.direct) if not x.startswith('_')]
        print(f"  {direct_modules}")
        result["modules"]["md.direct"] = direct_modules

        # Check md.direct.portfolio
        if hasattr(md.direct, 'portfolio'):
            print("\nMethods in md.direct.portfolio:")
            portfolio_methods = [x for x in dir(md.direct.portfolio) if not x.startswith('_') and callable(getattr(md.direct.portfolio, x, None))]
            print(f"  {portfolio_methods}")
            result["methods"]["md.direct.portfolio"] = portfolio_methods

        # Check md.direct.investments (if exists)
        if hasattr(md.direct, 'investments'):
            print("\nMethods in md.direct.investments:")
            inv_methods = [x for x in dir(md.direct.investments) if not x.startswith('_') and callable(getattr(md.direct.investments, x, None))]
            print(f"  {inv_methods}")
            result["methods"]["md.direct.investments"] = inv_methods

        # Check md.direct for data fetching methods
        print("\nCallable methods in md.direct:")
        direct_methods = [x for x in dir(md.direct) if not x.startswith('_') and callable(getattr(md.direct, x, None))]
        print(f"  {direct_methods}")
        result["methods"]["md.direct"] = direct_methods

    # Check md.time_series (if exists)
    if hasattr(md, 'time_series'):
        print("\nMethods in md.time_series:")
        ts_methods = [x for x in dir(md.time_series) if not x.startswith('_') and callable(getattr(md.time_series, x, None))]
        print(f"  {ts_methods}")
        result["methods"]["md.time_series"] = ts_methods

    # Try to get data sets
    print("\n=== Available Data Sets ===")
    try:
        data_sets = md.direct.portfolio.get_data_sets()
        print(data_sets)
        result["data_sets"] = data_sets.to_dict() if hasattr(data_sets, 'to_dict') else str(data_sets)
    except Exception as e:
        print(f"  Error: {e}")

    # Try to get daily returns data points
    print("\n=== Daily Returns Data Points (data_set_id=2) ===")
    try:
        data_points = md.direct.portfolio.get_data_set_data_points(data_set_id="2")
        print(data_points)
        result["daily_returns_data_points"] = data_points.to_dict() if hasattr(data_points, 'to_dict') else str(data_points)
    except Exception as e:
        print(f"  Error: {e}")

    return result


if __name__ == "__main__":
    """Test Morningstar data provider."""
    print("=== Morningstar Data Provider Test ===\n")

    # Check availability
    availability = check_morningstar_availability()
    print("Data Source Availability:")
    for key, value in availability.items():
        print(f"  {key}: {value}")

    # Test fetching data
    print("\n=== Testing Data Fetch ===")
    provider = MorningstarDataProvider()

    test_ticker = "VRTX"
    test_date = date(2024, 12, 31)

    print(f"\nFetching data for {test_ticker} as of {test_date}...")

    returns = provider.get_daily_returns(test_ticker, test_date, lookback_days=60)
    print(f"  Daily returns: {len(returns)} data points")
    if returns:
        print(f"    First 5: {returns[:5]}")
        print(f"    Mean: {sum(returns)/len(returns):.6f}")

    prices = provider.get_prices(test_ticker, test_date, lookback_days=60)
    print(f"  Prices: {len(prices)} data points")
    if prices:
        print(f"    Latest: ${prices[-1]:.2f}")

    # Show Morningstar data sets if available
    if MORNINGSTAR_AVAILABLE:
        print("\n=== Morningstar Data Sets ===")
        data_sets = get_morningstar_data_sets()
        if data_sets is not None:
            print(data_sets)

        print("\n=== Daily Returns Data Points (data_set_id=2) ===")
        data_points = get_morningstar_daily_returns_schema()
        if data_points is not None:
            print(data_points)

    print("\n=== Test Complete ===")
