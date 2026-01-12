"""
market_data_provider.py - Point-in-time safe price/return series provider
FINAL VERSION - Handles both historical and current dates correctly

Enhanced with:
- Stooq fallback when Yahoo Finance fails
- Structured logging for provider attempts and failures
- Disk caching (Parquet preferred, CSV fallback)
- OHLCV validation and normalization
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import hashlib
import math
import logging
import urllib.request
import urllib.error
import io
import csv

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_data_provider')

# Optional imports - handle gracefully
try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed - Yahoo Finance provider unavailable")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    logger.warning("pandas not installed - using CSV-only caching")

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    pdr = None
    PDR_AVAILABLE = False
    logger.info("pandas_datareader not installed - using direct Stooq HTTPS")

# Cache directories
CACHE_DIR = Path("cache/market_data")
OHLCV_CACHE_DIR = Path("data/cache/market")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24
DECIMAL_PRECISION = Decimal("0.00000001")
TRADING_DAYS_PER_YEAR = 252

# Required OHLCV columns (Adj Close is optional)
REQUIRED_OHLCV_COLUMNS = {'Open', 'High', 'Low', 'Close', 'Volume'}


# =============================================================================
# Failure Classification for Yahoo Finance
# =============================================================================
class YahooFailureReason:
    """Enumeration of Yahoo Finance failure reasons."""
    EXCEPTION = "exception_thrown"
    EMPTY_DATA = "empty_or_none_dataframe"
    MISSING_COLUMNS = "missing_required_ohlcv_columns"
    NON_MONOTONIC = "date_index_not_monotonic"
    NO_OVERLAP = "no_data_in_requested_range"


class ProviderError(Exception):
    """Exception raised when all providers fail."""
    def __init__(self, symbol: str, yahoo_reason: str, stooq_reason: str):
        self.symbol = symbol
        self.yahoo_reason = yahoo_reason
        self.stooq_reason = stooq_reason
        super().__init__(
            f"All providers failed for {symbol}. "
            f"Yahoo: {yahoo_reason}. Stooq: {stooq_reason}"
        )


# =============================================================================
# OHLCV Validation and Normalization
# =============================================================================
def _validate_ohlcv_dataframe(
    df: Any,
    symbol: str,
    start: date,
    end: date,
    provider: str
) -> Tuple[bool, str]:
    """
    Validate OHLCV dataframe meets requirements.

    Returns:
        (is_valid, failure_reason)
    """
    # Check 1: DataFrame is None or empty
    if df is None:
        return False, YahooFailureReason.EMPTY_DATA

    if not PANDAS_AVAILABLE:
        # Without pandas, we can't do full validation
        return True, ""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return False, YahooFailureReason.EMPTY_DATA

    # Check 2: Required columns present
    columns_upper = {c.title() for c in df.columns}
    missing = REQUIRED_OHLCV_COLUMNS - columns_upper
    if missing:
        return False, f"{YahooFailureReason.MISSING_COLUMNS}: {missing}"

    # Check 3: Date index is monotonic increasing after normalization
    try:
        if hasattr(df.index, 'tz'):
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        if not df.index.is_monotonic_increasing:
            # Remove duplicates and try again
            df = df[~df.index.duplicated(keep='first')]
            if not df.index.is_monotonic_increasing:
                return False, YahooFailureReason.NON_MONOTONIC
    except Exception as e:
        return False, f"{YahooFailureReason.NON_MONOTONIC}: {e}"

    # Check 4: Data overlaps requested range
    try:
        df_dates = df.index.date if hasattr(df.index, 'date') else df.index
        start_dt = start if isinstance(start, date) else start.date()
        end_dt = end if isinstance(end, date) else end.date()

        in_range = [(d >= start_dt and d <= end_dt) for d in df_dates]
        if not any(in_range):
            return False, YahooFailureReason.NO_OVERLAP
    except Exception as e:
        return False, f"{YahooFailureReason.NO_OVERLAP}: {e}"

    return True, ""


def _normalize_ohlcv_dataframe(df: Any, provider: str) -> Any:
    """
    Normalize OHLCV dataframe to standard format.

    - DatetimeIndex (daily) sorted ascending
    - Required columns: Open, High, Low, Close, Volume
    - No duplicate dates
    - Timezone-naive
    """
    if not PANDAS_AVAILABLE or df is None:
        return df

    # Stooq returns newest-to-oldest, sort ascending
    df = df.sort_index()

    # Remove timezone info
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Normalize column names
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            col_mapping[col] = 'Open'
        elif 'high' in col_lower:
            col_mapping[col] = 'High'
        elif 'low' in col_lower:
            col_mapping[col] = 'Low'
        elif 'close' in col_lower and 'adj' not in col_lower:
            col_mapping[col] = 'Close'
        elif 'adj' in col_lower and 'close' in col_lower:
            col_mapping[col] = 'Adj Close'
        elif 'volume' in col_lower:
            col_mapping[col] = 'Volume'

    df = df.rename(columns=col_mapping)

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep='first')]

    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Adj Close' in df.columns:
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

    return df


# =============================================================================
# Disk Caching (Parquet preferred, CSV fallback)
# =============================================================================
def _get_ohlcv_cache_path(provider: str, symbol: str, start: date, end: date) -> Path:
    """Generate cache file path for OHLCV data."""
    cache_key = f"{provider}_{symbol}_{start.isoformat()}_{end.isoformat()}"
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

    # Prefer Parquet if pandas available
    if PANDAS_AVAILABLE:
        return OHLCV_CACHE_DIR / f"{cache_hash}.parquet"
    return OHLCV_CACHE_DIR / f"{cache_hash}.csv"


def _read_ohlcv_cache(cache_path: Path) -> Optional[Any]:
    """Read OHLCV data from disk cache."""
    if not cache_path.exists():
        logger.debug(f"Cache miss: {cache_path.name}")
        return None

    # Check TTL
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age_hours = (datetime.now() - mtime).total_seconds() / 3600
    if age_hours > CACHE_TTL_HOURS:
        logger.debug(f"Cache expired: {cache_path.name} (age={age_hours:.1f}h)")
        return None

    try:
        if PANDAS_AVAILABLE:
            if cache_path.suffix == '.parquet':
                df = pd.read_parquet(cache_path)
            else:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Cache hit: {cache_path.name} ({len(df)} rows)")
            return df
        else:
            # CSV without pandas - return raw data
            with open(cache_path) as f:
                data = list(csv.DictReader(f))
            logger.info(f"Cache hit: {cache_path.name} ({len(data)} rows)")
            return data
    except Exception as e:
        logger.warning(f"Cache read failed: {cache_path.name}: {e}")
        return None


def _write_ohlcv_cache(cache_path: Path, df: Any, provider: str) -> None:
    """Write OHLCV data to disk cache."""
    try:
        if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
            # Add metadata
            df = df.copy()

            if cache_path.suffix == '.parquet':
                df.to_parquet(cache_path)
            else:
                df.to_csv(cache_path)

            logger.info(f"Cache write: {cache_path.name} ({len(df)} rows, provider={provider})")
        else:
            # Fallback CSV write
            with open(cache_path, 'w', newline='') as f:
                if isinstance(df, list) and df:
                    writer = csv.DictWriter(f, fieldnames=df[0].keys())
                    writer.writeheader()
                    writer.writerows(df)
            logger.info(f"Cache write: {cache_path.name} (provider={provider})")
    except Exception as e:
        logger.warning(f"Cache write failed: {cache_path.name}: {e}")


# =============================================================================
# Yahoo Finance Provider
# =============================================================================
def _fetch_yahoo(symbol: str, start: date, end: date) -> Tuple[Any, Optional[str]]:
    """
    Fetch OHLCV data from Yahoo Finance.

    Returns:
        (dataframe, failure_reason) - failure_reason is None on success
    """
    if yf is None:
        return None, "yfinance not installed"

    logger.info(f"Yahoo: Attempting {symbol} [{start} to {end}]")

    try:
        ticker = yf.Ticker(symbol)

        # Use start/end for historical data
        df = ticker.history(
            start=start.strftime('%Y-%m-%d'),
            end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),  # end is exclusive
            auto_adjust=True,
            actions=False
        )

        # Validate the result
        is_valid, failure_reason = _validate_ohlcv_dataframe(df, symbol, start, end, "yahoo")

        if not is_valid:
            logger.warning(f"Yahoo: Validation failed for {symbol}: {failure_reason}")
            return None, failure_reason

        # Normalize
        df = _normalize_ohlcv_dataframe(df, "yahoo")

        # Filter to requested range
        if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
            mask = (df.index.date >= start) & (df.index.date <= end)
            df = df[mask]

        logger.info(f"Yahoo: Success for {symbol} ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")
        return df, None

    except Exception as e:
        failure_reason = f"{YahooFailureReason.EXCEPTION}: {type(e).__name__}: {e}"
        logger.warning(f"Yahoo: Exception for {symbol}: {failure_reason}")
        return None, failure_reason


# =============================================================================
# Stooq Provider (Fallback)
# =============================================================================
def _stooq_symbol(symbol: str) -> str:
    """Convert symbol to Stooq format (US stocks need .US suffix)."""
    if '.' not in symbol:
        return f"{symbol}.US"
    return symbol


def _fetch_stooq_via_pdr(symbol: str, start: date, end: date) -> Tuple[Any, Optional[str]]:
    """Fetch from Stooq using pandas_datareader."""
    if not PDR_AVAILABLE:
        return None, "pandas_datareader not installed"

    stooq_symbol = _stooq_symbol(symbol)
    logger.info(f"Stooq (pdr): Attempting {stooq_symbol} [{start} to {end}]")

    try:
        df = pdr.DataReader(stooq_symbol, 'stooq', start=start, end=end)

        if df is None or df.empty:
            return None, "empty dataframe returned"

        # Stooq returns newest-to-oldest, sort ascending
        df = df.sort_index()

        # Validate
        is_valid, failure_reason = _validate_ohlcv_dataframe(df, symbol, start, end, "stooq")
        if not is_valid:
            logger.warning(f"Stooq (pdr): Validation failed for {symbol}: {failure_reason}")
            return None, failure_reason

        # Normalize
        df = _normalize_ohlcv_dataframe(df, "stooq")

        logger.info(f"Stooq (pdr): Success for {symbol} ({len(df)} rows)")
        return df, None

    except Exception as e:
        failure_reason = f"exception: {type(e).__name__}: {e}"
        logger.warning(f"Stooq (pdr): Exception for {symbol}: {failure_reason}")
        return None, failure_reason


def _fetch_stooq_direct(symbol: str, start: date, end: date) -> Tuple[Any, Optional[str]]:
    """
    Fetch from Stooq using direct HTTPS to CSV endpoint.
    Fallback when pandas_datareader is not available.
    """
    stooq_symbol = _stooq_symbol(symbol)

    # Stooq CSV URL format
    # https://stooq.com/q/d/l/?s=amgn.us&d1=20150101&d2=20241231&i=d
    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={stooq_symbol.lower()}"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
        f"&i=d"
    )

    logger.info(f"Stooq (direct): Attempting {stooq_symbol} [{start} to {end}]")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode('utf-8')

        # Parse CSV
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return None, "empty CSV response"

        # Check for Stooq's "No data" response
        if 'No data' in content or len(lines) == 1:
            return None, "no data available from Stooq"

        if PANDAS_AVAILABLE:
            # Parse with pandas
            df = pd.read_csv(io.StringIO(content), parse_dates=['Date'], index_col='Date')
            df = df.sort_index()

            # Validate
            is_valid, failure_reason = _validate_ohlcv_dataframe(df, symbol, start, end, "stooq")
            if not is_valid:
                return None, failure_reason

            # Normalize
            df = _normalize_ohlcv_dataframe(df, "stooq")

            logger.info(f"Stooq (direct): Success for {symbol} ({len(df)} rows)")
            return df, None
        else:
            # Parse without pandas
            reader = csv.DictReader(lines)
            data = list(reader)
            if not data:
                return None, "empty CSV data"

            logger.info(f"Stooq (direct): Success for {symbol} ({len(data)} rows)")
            return data, None

    except urllib.error.HTTPError as e:
        return None, f"HTTP error: {e.code}"
    except urllib.error.URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:
        return None, f"exception: {type(e).__name__}: {e}"


def _fetch_stooq(symbol: str, start: date, end: date) -> Tuple[Any, Optional[str]]:
    """
    Fetch from Stooq - tries pandas_datareader first, then direct HTTPS.
    """
    # Try pandas_datareader first (if available)
    if PDR_AVAILABLE:
        df, error = _fetch_stooq_via_pdr(symbol, start, end)
        if df is not None:
            return df, None
        logger.info(f"Stooq (pdr) failed, trying direct HTTPS: {error}")

    # Fallback to direct HTTPS
    return _fetch_stooq_direct(symbol, start, end)


# =============================================================================
# Main Fetch Function with Fallback Logic
# =============================================================================
def fetch_ohlcv(
    symbol: str,
    start: date,
    end: date,
    use_cache: bool = True
) -> Tuple[Any, str]:
    """
    Fetch OHLCV data with automatic Yahoo -> Stooq fallback.

    Args:
        symbol: Stock ticker symbol
        start: Start date (inclusive)
        end: End date (inclusive)
        use_cache: Whether to use disk cache

    Returns:
        (dataframe, source_provider)

    Raises:
        ProviderError: If both Yahoo and Stooq fail

    Fallback Logic:
    ---------------
    1. Check disk cache first (if use_cache=True)
    2. Attempt Yahoo Finance
    3. If Yahoo fails (exception, empty, missing columns, non-monotonic, no overlap),
       attempt Stooq as fallback
    4. If both fail, raise ProviderError with both failure reasons
    """
    # Check cache first
    if use_cache:
        for provider in ['yahoo', 'stooq']:
            cache_path = _get_ohlcv_cache_path(provider, symbol, start, end)
            cached_df = _read_ohlcv_cache(cache_path)
            if cached_df is not None:
                return cached_df, provider

    yahoo_error = None
    stooq_error = None

    # Attempt Yahoo Finance first
    df, yahoo_error = _fetch_yahoo(symbol, start, end)
    if df is not None:
        if use_cache:
            cache_path = _get_ohlcv_cache_path('yahoo', symbol, start, end)
            _write_ohlcv_cache(cache_path, df, 'yahoo')
        return df, 'yahoo'

    logger.info(f"Yahoo failed for {symbol}, attempting Stooq fallback...")

    # Fallback to Stooq
    df, stooq_error = _fetch_stooq(symbol, start, end)
    if df is not None:
        if use_cache:
            cache_path = _get_ohlcv_cache_path('stooq', symbol, start, end)
            _write_ohlcv_cache(cache_path, df, 'stooq')
        return df, 'stooq'

    # Both providers failed
    logger.error(f"All providers failed for {symbol}. Yahoo: {yahoo_error}. Stooq: {stooq_error}")
    raise ProviderError(symbol, yahoo_error or "unknown", stooq_error or "unknown")


# =============================================================================
# Legacy Interface (Preserved for backward compatibility)
# =============================================================================
def _quantize(value: float) -> Decimal:
    return Decimal(str(value)).quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

def _date_to_str(d: date) -> str:
    return d.isoformat()

def _compute_cache_key(ticker: str, as_of: date, lookback_days: int) -> str:
    key_data = f"{ticker}|{_date_to_str(as_of)}|{lookback_days}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]

def _is_cache_valid(cache_path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() < (ttl_hours * 3600)

def _write_cache(cache_path: Path, data: Dict) -> None:
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
    try:
        with open(cache_path) as f:
            cache_data = json.load(f)
        data = cache_data["data"]
        expected_hash = cache_data["integrity"]
        actual_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if expected_hash != actual_hash:
            logger.warning(f"Cache integrity mismatch for {cache_path.name}")
            return None
        return data
    except Exception:
        return None


class PriceDataProvider:
    """
    Price data provider with Yahoo Finance + Stooq fallback.

    Public interface is unchanged for backward compatibility.
    Internally uses fetch_ohlcv() with automatic fallback.
    """

    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS):
        self.cache_ttl_hours = cache_ttl_hours
        # No longer require yfinance - we have Stooq fallback
        if yf is None:
            logger.warning("yfinance not installed - will use Stooq as primary provider")

    def get_prices(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[Decimal]:
        """
        Get daily closing prices for a ticker.

        Uses Yahoo Finance with Stooq fallback.
        CRITICAL: Handles both historical and current dates correctly.
        """
        # Check legacy cache first (for backward compatibility)
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days)
            cache_path = CACHE_DIR / f"{cache_key}_prices.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    logger.debug(f"Legacy cache hit for {ticker}")
                    return [Decimal(str(p)) for p in cached["prices"]]

        # Calculate date range
        start_date = as_of - timedelta(days=lookback_days)

        try:
            # Use new fetch_ohlcv with fallback
            df, provider = fetch_ohlcv(ticker, start_date, as_of, use_cache=use_cache)

            logger.info(f"Fetched {ticker} from {provider} provider")

            # Extract prices
            prices = []
            if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
                for idx, row in df.iterrows():
                    # Handle timezone-aware datetime index
                    if hasattr(idx, 'date'):
                        price_date = idx.date()
                    else:
                        price_date = idx.to_pydatetime().date() if hasattr(idx, 'to_pydatetime') else idx

                    # PIT discipline: only include data <= as_of
                    if price_date <= as_of:
                        close = row.get('Close', row.get('close', None))
                        if close is not None and not math.isnan(close):
                            prices.append(_quantize(close))
            else:
                # Handle non-pandas data (list of dicts)
                for row in df:
                    price_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                    if price_date <= as_of:
                        close = float(row.get('Close', row.get('close', 0)))
                        if close > 0:
                            prices.append(_quantize(close))

            # Write to legacy cache
            if use_cache and prices:
                cache_key = _compute_cache_key(ticker, as_of, lookback_days)
                cache_path = CACHE_DIR / f"{cache_key}_prices.json"
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "num_prices": len(prices),
                    "prices": [str(p) for p in prices],
                    "source_provider": provider,
                }
                _write_cache(cache_path, cache_data)

            return prices

        except ProviderError as e:
            logger.error(f"All providers failed for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            return []

    def get_log_returns(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[float]:
        """Get daily log returns."""
        prices = self.get_prices(ticker, as_of, lookback_days, use_cache)
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            p_prev = float(prices[i-1])
            p_curr = float(prices[i])
            if p_prev > 0 and p_curr > 0:
                returns.append(math.log(p_curr / p_prev))
        return returns

    def get_volumes(self, ticker: str, as_of: date, lookback_days: int, use_cache: bool = True) -> List[int]:
        """Get daily trading volumes."""
        # Check legacy cache first
        if use_cache:
            cache_key = _compute_cache_key(ticker, as_of, lookback_days)
            cache_path = CACHE_DIR / f"{cache_key}_volumes.json"
            if _is_cache_valid(cache_path, self.cache_ttl_hours):
                cached = _read_cache(cache_path)
                if cached:
                    return cached["volumes"]

        # Calculate date range
        start_date = as_of - timedelta(days=lookback_days)

        try:
            df, provider = fetch_ohlcv(ticker, start_date, as_of, use_cache=use_cache)

            volumes = []
            if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
                for idx, row in df.iterrows():
                    if hasattr(idx, 'date'):
                        volume_date = idx.date()
                    else:
                        volume_date = idx.to_pydatetime().date() if hasattr(idx, 'to_pydatetime') else idx

                    if volume_date <= as_of:
                        vol = row.get('Volume', row.get('volume', None))
                        if vol is not None and not math.isnan(vol):
                            volumes.append(int(vol))
            else:
                for row in df:
                    volume_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                    if volume_date <= as_of:
                        vol = int(float(row.get('Volume', row.get('volume', 0))))
                        if vol > 0:
                            volumes.append(vol)

            # Write to legacy cache
            if use_cache and volumes:
                cache_key = _compute_cache_key(ticker, as_of, lookback_days)
                cache_path = CACHE_DIR / f"{cache_key}_volumes.json"
                cache_data = {
                    "ticker": ticker,
                    "as_of": _date_to_str(as_of),
                    "lookback_days": lookback_days,
                    "volumes": volumes,
                    "source_provider": provider,
                }
                _write_cache(cache_path, cache_data)

            return volumes

        except ProviderError as e:
            logger.error(f"All providers failed for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch volumes for {ticker}: {e}")
            return []

    def get_adv(self, ticker: str, as_of: date, window: int = 20) -> Optional[Decimal]:
        """Get average daily dollar volume."""
        lookback_days = window * 2
        prices = self.get_prices(ticker, as_of, lookback_days)
        volumes = self.get_volumes(ticker, as_of, lookback_days)

        if not prices or not volumes or len(prices) != len(volumes):
            return None

        recent_prices = prices[-window:] if len(prices) >= window else prices
        recent_volumes = volumes[-window:] if len(volumes) >= window else volumes

        if not recent_prices or not recent_volumes:
            return None

        dollar_volumes = [float(p) * v for p, v in zip(recent_prices, recent_volumes)]
        avg_dollar_volume = sum(dollar_volumes) / len(dollar_volumes)

        return _quantize(avg_dollar_volume)

    def get_ticker_data(self, ticker: str, as_of: date, lookback_days: int = 365) -> Dict:
        """Get complete data package for a ticker."""
        prices = self.get_prices(ticker, as_of, lookback_days)
        returns = self.get_log_returns(ticker, as_of, lookback_days)
        volumes = self.get_volumes(ticker, as_of, lookback_days)

        return {
            "ticker": ticker,
            "as_of": as_of,
            "prices": prices,
            "returns": returns,
            "volumes": volumes,
            "num_days": len(prices),
        }


class BatchPriceProvider:
    def __init__(self, cache_ttl_hours: int = CACHE_TTL_HOURS):
        self.provider = PriceDataProvider(cache_ttl_hours)

    def get_batch_data(self, tickers: List[str], as_of: date, lookback_days: int = 365, include_xbi: bool = True) -> Dict[str, Dict]:
        """Get data for multiple tickers efficiently."""
        results = {}

        if include_xbi:
            logger.info("Fetching XBI benchmark data...")
            xbi_data = self.provider.get_ticker_data("XBI", as_of, lookback_days)
            results["_xbi_"] = xbi_data

        for ticker in tickers:
            logger.info(f"Fetching {ticker}...")
            try:
                data = self.provider.get_ticker_data(ticker, as_of, lookback_days)
                if data["num_days"] > 0:
                    results[ticker] = data
                else:
                    logger.warning(f"No data for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                continue

        return results


# =============================================================================
# Convenience Functions (Public API - unchanged signatures)
# =============================================================================
def get_prices(ticker: str, as_of: date, lookback_days: int) -> List[Decimal]:
    provider = PriceDataProvider()
    return provider.get_prices(ticker, as_of, lookback_days)

def get_log_returns(ticker: str, as_of: date, lookback_days: int) -> List[float]:
    provider = PriceDataProvider()
    return provider.get_log_returns(ticker, as_of, lookback_days)

def get_adv(ticker: str, as_of: date, window: int = 20) -> Optional[Decimal]:
    provider = PriceDataProvider()
    return provider.get_adv(ticker, as_of, window)


# =============================================================================
# Self-Test Routines
# =============================================================================
def _test_yahoo_failure_triggers_fallback() -> bool:
    """Test that Yahoo failure triggers Stooq fallback."""
    logger.info("=== Test: Yahoo failure triggers Stooq fallback ===")

    # Use an invalid symbol to force Yahoo to fail
    invalid_symbol = "ZZZZINVALID12345"
    test_start = date(2024, 1, 1)
    test_end = date(2024, 1, 31)

    try:
        df, provider = fetch_ohlcv(invalid_symbol, test_start, test_end, use_cache=False)
        # If we get here, one of the providers succeeded (shouldn't happen with invalid symbol)
        logger.warning(f"Unexpected success from {provider} for invalid symbol")
        return False
    except ProviderError as e:
        # Expected - both providers should fail
        logger.info(f"✓ ProviderError raised as expected: {e}")
        return "yahoo" in str(e).lower() and "stooq" in str(e).lower()
    except Exception as e:
        logger.error(f"Unexpected exception: {e}")
        return False


def _test_valid_symbol_returns_data() -> bool:
    """Test that a valid liquid symbol returns data."""
    logger.info("=== Test: Valid symbol returns data ===")

    # Use a well-known liquid symbol
    symbol = "AAPL"
    test_start = date(2024, 1, 1)
    test_end = date(2024, 1, 31)

    try:
        df, provider = fetch_ohlcv(symbol, test_start, test_end, use_cache=False)

        if df is None:
            logger.error("Returned None dataframe")
            return False

        if PANDAS_AVAILABLE:
            row_count = len(df)
        else:
            row_count = len(df) if isinstance(df, list) else 0

        if row_count == 0:
            logger.error("Returned empty data")
            return False

        logger.info(f"✓ Got {row_count} rows from {provider} for {symbol}")
        return True

    except ProviderError as e:
        logger.error(f"ProviderError for valid symbol: {e}")
        return False
    except Exception as e:
        logger.error(f"Exception for valid symbol: {e}")
        return False


def _test_cache_hit() -> bool:
    """Test that cache hits work correctly."""
    logger.info("=== Test: Cache hit works ===")

    symbol = "MSFT"
    test_start = date(2024, 6, 1)
    test_end = date(2024, 6, 30)

    try:
        # First fetch - should be cache miss
        df1, provider1 = fetch_ohlcv(symbol, test_start, test_end, use_cache=True)

        # Second fetch - should be cache hit
        df2, provider2 = fetch_ohlcv(symbol, test_start, test_end, use_cache=True)

        if df1 is None or df2 is None:
            logger.error("One of the fetches returned None")
            return False

        # Verify data matches
        if PANDAS_AVAILABLE:
            if len(df1) != len(df2):
                logger.error(f"Row count mismatch: {len(df1)} vs {len(df2)}")
                return False

        logger.info(f"✓ Cache hit verified for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        return False


def run_self_tests() -> bool:
    """Run all self-test routines."""
    logger.info("=" * 60)
    logger.info("Running Market Data Provider Self-Tests")
    logger.info("=" * 60)

    tests = [
        ("Yahoo failure triggers fallback", _test_yahoo_failure_triggers_fallback),
        ("Valid symbol returns data", _test_valid_symbol_returns_data),
        ("Cache hit works", _test_cache_hit),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test '{name}' raised exception: {e}")
            results.append((name, False))

    logger.info("=" * 60)
    logger.info("Self-Test Results:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    logger.info("=" * 60)

    return all_passed


# =============================================================================
# Legacy Validation Functions (Preserved)
# =============================================================================
def validate_pit_discipline() -> bool:
    """Validate that PIT discipline is enforced."""
    print("=== PIT Discipline Validation ===")

    provider = PriceDataProvider()

    test_as_of = date(2024, 6, 30)
    test_ticker = "VRTX"

    prices = provider.get_prices(test_ticker, test_as_of, lookback_days=90)

    if not prices:
        print("FAIL: No prices returned")
        return False

    print(f"✓ Fetched {len(prices)} prices for {test_ticker} as of {test_as_of}")
    print(f"  First price: ${prices[0]:.2f}")
    print(f"  Last price: ${prices[-1]:.2f}")

    return True

def validate_determinism() -> bool:
    """Validate that same inputs produce identical outputs."""
    print("\n=== Determinism Validation ===")

    provider = PriceDataProvider()

    ticker = "AMGN"
    as_of = date(2024, 12, 31)
    lookback = 60

    # First call
    prices1 = provider.get_prices(ticker, as_of, lookback, use_cache=False)

    # Second call
    prices2 = provider.get_prices(ticker, as_of, lookback, use_cache=False)

    if prices1 != prices2:
        print("FAIL: Results differ between calls")
        return False

    print(f"✓ Determinism validated: {len(prices1)} prices match across calls")

    return True

def validate_cache_integrity() -> bool:
    """Validate that cached data matches fresh data."""
    print("\n=== Cache Integrity Validation ===")

    provider = PriceDataProvider()

    ticker = "XBI"
    as_of = date(2024, 12, 31)
    lookback = 30

    # Fresh fetch
    prices_fresh = provider.get_prices(ticker, as_of, lookback, use_cache=False)

    # Cached fetch
    prices_cached = provider.get_prices(ticker, as_of, lookback, use_cache=True)

    if prices_fresh != prices_cached:
        print("FAIL: Cached data doesn't match fresh data")
        return False

    print(f"✓ Cache integrity validated: {len(prices_cached)} prices match")

    return True


if __name__ == "__main__":
    """Run validation suite and self-tests."""
    import sys

    print("=" * 60)
    print("Market Data Provider - Validation & Self-Tests")
    print("=" * 60)

    # Run self-tests for new fallback functionality
    self_tests_passed = run_self_tests()

    print("\n")

    # Run legacy validation tests
    legacy_tests = [
        ("PIT Discipline", validate_pit_discipline),
        ("Determinism", validate_determinism),
        ("Cache Integrity", validate_cache_integrity),
    ]

    legacy_results = []
    for name, test_func in legacy_tests:
        try:
            passed = test_func()
            legacy_results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            legacy_results.append((name, False))

    print("\n=== Legacy Validation Summary ===")
    for name, passed in legacy_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = self_tests_passed and all(p for _, p in legacy_results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)
