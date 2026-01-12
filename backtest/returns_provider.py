"""
Returns Provider Module

Provides forward total returns for backtest metrics.
Supports CSV (MVP), plus diagnostic wrappers for null expectation and lag stress tests.
"""
from __future__ import annotations

import csv
import random
from abc import ABC, abstractmethod
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ReturnProvider = Callable[[str, str, str], Optional[str]]


class BaseReturnsProvider(ABC):
    """Abstract base for returns providers."""
    
    @abstractmethod
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        """Get total return as Decimal string, or None if unavailable."""
        pass
    
    def __call__(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        return self.get_forward_total_return(ticker, start_date, end_date)


class CSVReturnsProvider(BaseReturnsProvider):
    """
    Returns provider from local CSV with columns: date, ticker, adj_close
    Forward return = adj_close[end] / adj_close[start] - 1
    """
    
    def __init__(self, csv_path: str | Path, date_col: str = "date", ticker_col: str = "ticker", 
                 price_col: str = "adj_close", date_tolerance_days: int = 5):
        self.csv_path = Path(csv_path)
        self.date_col, self.ticker_col, self.price_col = date_col, ticker_col, price_col
        self.date_tolerance_days = date_tolerance_days
        self._prices: Dict[str, Dict[date, Decimal]] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Price file not found: {self.csv_path}")
        
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ticker = row[self.ticker_col].strip().upper()
                date_str = row[self.date_col].strip()
                price_str = row[self.price_col].strip()
                if not ticker or not date_str or not price_str:
                    continue
                try:
                    dt = date.fromisoformat(date_str[:10])
                    price = Decimal(price_str)
                    if ticker not in self._prices:
                        self._prices[ticker] = {}
                    self._prices[ticker][dt] = price
                except (ValueError, TypeError):
                    continue
    
    def _find_nearest_price(self, ticker: str, target_date: date) -> Optional[Tuple[date, Decimal]]:
        if ticker not in self._prices:
            return None
        prices = self._prices[ticker]
        if target_date in prices:
            return (target_date, prices[target_date])
        for delta in range(1, self.date_tolerance_days + 1):
            for check_date in [target_date + timedelta(days=delta), target_date - timedelta(days=delta)]:
                if check_date in prices:
                    return (check_date, prices[check_date])
        return None
    
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        ticker = ticker.upper()
        start_price = self._find_nearest_price(ticker, date.fromisoformat(start_date))
        end_price = self._find_nearest_price(ticker, date.fromisoformat(end_date))
        if start_price is None or end_price is None or start_price[1] == 0:
            return None
        total_return = (end_price[1] / start_price[1]) - Decimal("1")
        return str(total_return.quantize(Decimal("0.000001")))
    
    def get_available_tickers(self) -> List[str]:
        return sorted(self._prices.keys())


class NullReturnsProvider(BaseReturnsProvider):
    """Returns None for all tickers. For baseline testing."""
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        return None


class FixedReturnsProvider(BaseReturnsProvider):
    """Returns fixed value for all tickers. For testing."""
    def __init__(self, return_value: str = "0.05"):
        self.return_value = return_value
    
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        return self.return_value


class ShuffledReturnsProvider(BaseReturnsProvider):
    """
    Shuffles ticker→return mapping for null expectation test.
    IC should collapse to ~0 if model is not leaking.
    """
    
    def __init__(self, base_provider: BaseReturnsProvider, seed: int = 42):
        self.base_provider = base_provider
        self.seed = seed
        self._cache: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}
    
    def prepare_for_tickers(self, tickers: List[str], start_date: str, end_date: str) -> None:
        """Pre-build shuffled mapping for given tickers and date range."""
        cache_key = (start_date, end_date)
        returns = {t: self.base_provider.get_forward_total_return(t, start_date, end_date) for t in tickers}
        available = [(t, r) for t, r in returns.items() if r is not None]
        missing = [t for t, r in returns.items() if r is None]
        
        rng = random.Random(self.seed)
        return_values = [r for _, r in available]
        rng.shuffle(return_values)
        
        shuffled = {available[i][0]: return_values[i] for i in range(len(available))}
        for t in missing:
            shuffled[t] = None
        self._cache[cache_key] = shuffled
    
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        cache_key = (start_date, end_date)
        if cache_key not in self._cache:
            return self.base_provider.get_forward_total_return(ticker, start_date, end_date)
        return self._cache[cache_key].get(ticker)


class LaggedReturnsProvider(BaseReturnsProvider):
    """
    Adds lag to dates for staleness test.
    IC should degrade (not invert) if PIT enforcement is working.
    """
    
    def __init__(self, base_provider: BaseReturnsProvider, lag_days: int = 30):
        self.base_provider = base_provider
        self.lag_days = lag_days
    
    def get_forward_total_return(self, ticker: str, start_date: str, end_date: str) -> Optional[str]:
        start_dt = date.fromisoformat(start_date) + timedelta(days=self.lag_days)
        end_dt = date.fromisoformat(end_date) + timedelta(days=self.lag_days)
        return self.base_provider.get_forward_total_return(ticker, start_dt.isoformat(), end_dt.isoformat())


class PITADVProvider:
    """
    Point-in-time ADV$ (Average Daily Dollar Volume) provider.

    Computes rolling 20-day mean of (price * volume) for PIT-safe liquidity analysis.
    Requires minimum 10 days of data to compute ADV$.
    """

    def __init__(self, csv_path: str | Path, date_col: str = "date", ticker_col: str = "ticker",
                 price_col: str = "adj_close", volume_col: str = "volume",
                 lookback_days: int = 20, min_days: int = 10):
        self.csv_path = Path(csv_path)
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.lookback_days = lookback_days
        self.min_days = min_days

        # Data storage: ticker -> sorted list of (date, price, volume)
        self._data: Dict[str, List[Tuple[date, Decimal, int]]] = {}
        self._has_volume = False
        self._volume_coverage: Dict[str, int] = {}  # ticker -> count of days with volume
        self._load_data()

    def _load_data(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Price file not found: {self.csv_path}")

        # First pass: check if volume column exists
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = [h.lower().strip() for h in (reader.fieldnames or [])]
            self._has_volume = self.volume_col.lower() in headers

        if not self._has_volume:
            return  # No volume data available

        # Second pass: load data
        temp_data: Dict[str, List[Tuple[date, Decimal, int]]] = {}

        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ticker = row[self.ticker_col].strip().upper()
                date_str = row[self.date_col].strip()
                price_str = row.get(self.price_col, "").strip()
                volume_str = row.get(self.volume_col, "").strip()

                if not ticker or not date_str or not price_str:
                    continue

                try:
                    dt = date.fromisoformat(date_str[:10])
                    price = Decimal(price_str)

                    # Volume may be empty/missing
                    volume = None
                    if volume_str:
                        try:
                            volume = int(float(volume_str))
                        except (ValueError, TypeError):
                            pass

                    if ticker not in temp_data:
                        temp_data[ticker] = []

                    if volume is not None and volume > 0:
                        temp_data[ticker].append((dt, price, volume))
                        self._volume_coverage[ticker] = self._volume_coverage.get(ticker, 0) + 1

                except (ValueError, TypeError):
                    continue

        # Sort each ticker's data by date
        for ticker, data_list in temp_data.items():
            self._data[ticker] = sorted(data_list, key=lambda x: x[0])

    @property
    def has_volume(self) -> bool:
        """Check if volume column was found in CSV."""
        return self._has_volume

    def get_volume_coverage_stats(self) -> Dict:
        """Return volume coverage statistics."""
        if not self._has_volume:
            return {"has_volume_column": False, "tickers_with_volume": 0, "total_volume_days": 0}

        tickers_with_volume = len(self._volume_coverage)
        total_volume_days = sum(self._volume_coverage.values())

        return {
            "has_volume_column": True,
            "tickers_with_volume": tickers_with_volume,
            "total_volume_days": total_volume_days,
            "tickers": sorted(self._volume_coverage.keys()),
        }

    def get_adv(self, ticker: str, as_of: date) -> Optional[float]:
        """
        Compute ADV$ (average daily dollar volume) for a ticker as of a given date.

        Uses trailing lookback_days of (price * volume), requiring min_days of data.
        Returns None if insufficient data.
        """
        ticker = ticker.upper()

        if ticker not in self._data:
            return None

        data = self._data[ticker]

        # Filter to dates <= as_of and within lookback window
        cutoff = as_of - timedelta(days=self.lookback_days * 2)  # Allow for weekends
        relevant = [(dt, price, vol) for dt, price, vol in data
                    if cutoff <= dt <= as_of]

        # Take the most recent lookback_days entries
        relevant = relevant[-self.lookback_days:]

        if len(relevant) < self.min_days:
            return None

        # Compute mean of (price * volume)
        dollar_volumes = [float(price) * vol for _, price, vol in relevant]
        return sum(dollar_volumes) / len(dollar_volumes)

    def get_adv_for_universe(self, tickers: List[str], as_of: date) -> Dict[str, Optional[float]]:
        """Get ADV$ for all tickers in universe as of a given date."""
        return {t: self.get_adv(t, as_of) for t in tickers}

    def get_available_tickers(self) -> List[str]:
        """Return list of tickers with volume data."""
        return sorted(self._data.keys())


# Factory functions
def create_csv_provider(csv_path: str | Path, **kwargs) -> CSVReturnsProvider:
    return CSVReturnsProvider(csv_path, **kwargs)

def create_shuffled_provider(base: BaseReturnsProvider, seed: int = 42) -> ShuffledReturnsProvider:
    return ShuffledReturnsProvider(base, seed)

def create_lagged_provider(base: BaseReturnsProvider, lag_days: int = 30) -> LaggedReturnsProvider:
    return LaggedReturnsProvider(base, lag_days)

def create_pit_adv_provider(csv_path: str | Path, **kwargs) -> PITADVProvider:
    return PITADVProvider(csv_path, **kwargs)
