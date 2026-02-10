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

    def get_last_date(self) -> Optional[date]:
        """Return the latest date with data across all tickers."""
        last = None
        for prices in self._prices.values():
            if prices:
                ticker_max = max(prices.keys())
                if last is None or ticker_max > last:
                    last = ticker_max
        return last

    # Minimum bars required for a meaningful forward max-DD estimate
    MIN_BARS_FORWARD = 10

    def get_forward_max_drawdown(
        self, ticker: str, as_of_date: str, horizon_bars: int,
    ) -> Tuple[Optional[float], str]:
        """Max peak-to-trough drawdown in the forward window.

        Returns (max_dd, missing_reason).
        max_dd is negative (e.g. -0.20 = 20% drawdown), or 0.0 for monotonic up.
        None if data is missing/insufficient.
        missing_reason: "" | "no_price_series" | "series_too_short"
        """
        ticker = ticker.upper()
        prices = self._prices.get(ticker)
        if not prices:
            return (None, "no_price_series")

        as_of = date.fromisoformat(as_of_date)
        # Dates strictly after as_of_date, sorted ascending, take first horizon_bars
        forward_dates = sorted(d for d in prices if d > as_of)[:horizon_bars]

        if len(forward_dates) < self.MIN_BARS_FORWARD:
            return (None, "series_too_short")

        # Running peak, track min drawdown
        peak = float(prices[forward_dates[0]])
        max_dd = 0.0
        for d in forward_dates:
            p = float(prices[d])
            if p > peak:
                peak = p
            dd = (p / peak) - 1.0 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        return (round(max_dd, 6), "")

    def get_daily_returns(
        self, ticker: str, start: date, end: date,
    ) -> List[Tuple[date, float]]:
        """
        Get daily simple returns for a ticker between start and end (inclusive).

        Returns list of (date, return) sorted by date.
        Skips days where the prior close is missing or zero.
        """
        ticker = ticker.upper()
        prices = self._prices.get(ticker)
        if not prices:
            return []

        sorted_dates = sorted(d for d in prices if start <= d <= end)
        if len(sorted_dates) < 2:
            return []

        result: List[Tuple[date, float]] = []
        for i in range(1, len(sorted_dates)):
            prev = prices[sorted_dates[i - 1]]
            curr = prices[sorted_dates[i]]
            if prev and prev != 0:
                ret = float((curr / prev) - Decimal("1"))
                result.append((sorted_dates[i], ret))
        return result


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


# Factory functions
def create_csv_provider(csv_path: str | Path, **kwargs) -> CSVReturnsProvider:
    return CSVReturnsProvider(csv_path, **kwargs)

def create_shuffled_provider(base: BaseReturnsProvider, seed: int = 42) -> ShuffledReturnsProvider:
    return ShuffledReturnsProvider(base, seed)

def create_lagged_provider(base: BaseReturnsProvider, lag_days: int = 30) -> LaggedReturnsProvider:
    return LaggedReturnsProvider(base, lag_days)
