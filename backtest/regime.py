"""
Deterministic regime classifier from market price history.

Labels each date as BULL / BEAR / CHOP using dual moving-average crossover
and rolling volatility on a market ticker (default: XBI).

Algorithm:
  - ma_fast = 50-day SMA(close)
  - ma_slow = 200-day SMA(close)
  - vol = 20-day rolling std(daily_returns)
  - vol_pct = expanding percentile of vol (deterministic, no lookahead)
  - BULL if ma_fast > ma_slow and vol_pct < 60
  - BEAR if ma_fast < ma_slow and vol_pct > 40
  - else CHOP

Thresholds are constants at module level for transparency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Configurable constants ────────────────────────────────────────────
MA_FAST_WINDOW = 50
MA_SLOW_WINDOW = 200
VOL_WINDOW = 20
VOL_BULL_THRESHOLD = 60   # percentile; below this = low vol
VOL_BEAR_THRESHOLD = 40   # percentile; above this = high vol
WARMUP_DAYS = MA_SLOW_WINDOW  # need 200 days before first regime label

BULL = "BULL"
BEAR = "BEAR"
CHOP = "CHOP"


# ── Price loading ─────────────────────────────────────────────────────

def load_price_history(price_csv: Path) -> pd.DataFrame:
    """
    Load price history CSV. Expected columns: date, ticker, close.

    Returns DataFrame sorted by (ticker, date).
    """
    df = pd.read_csv(price_csv)
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price CSV missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ── Regime computation ────────────────────────────────────────────────

def compute_regime_series(
    price_df: pd.DataFrame,
    market_ticker: str = "XBI",
) -> pd.DataFrame:
    """
    Compute daily regime labels for the market ticker.

    Parameters
    ----------
    price_df : DataFrame with columns date, ticker, close
    market_ticker : ticker to use for regime classification

    Returns
    -------
    DataFrame with columns: date, regime
    Sorted by date. Dates before warmup period have regime=NaN.
    """
    mkt = price_df[price_df["ticker"] == market_ticker].copy()
    if len(mkt) == 0:
        available = sorted(price_df["ticker"].unique()[:10])
        raise ValueError(
            f"Market ticker '{market_ticker}' not found in price data. "
            f"Available tickers (sample): {available}"
        )

    mkt = mkt.sort_values("date").reset_index(drop=True)
    close = mkt["close"].astype(float)

    # Moving averages
    ma_fast = close.rolling(window=MA_FAST_WINDOW, min_periods=MA_FAST_WINDOW).mean()
    ma_slow = close.rolling(window=MA_SLOW_WINDOW, min_periods=MA_SLOW_WINDOW).mean()

    # Daily returns and rolling volatility
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW).std()

    # Expanding percentile of vol (no lookahead: uses only data up to current day)
    # Use min_periods equal to vol window so percentile is available as soon as
    # both MAs and vol are available.
    vol_pct = vol.expanding(min_periods=VOL_WINDOW).apply(
        _percentile_rank, raw=True,
    )

    # Classify
    regime = pd.Series(np.nan, index=mkt.index, dtype=object)
    for i in range(len(mkt)):
        if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]) or pd.isna(vol_pct.iloc[i]):
            continue
        fast_above = ma_fast.iloc[i] > ma_slow.iloc[i]
        vp = vol_pct.iloc[i]
        if fast_above and vp < VOL_BULL_THRESHOLD:
            regime.iloc[i] = BULL
        elif (not fast_above) and vp > VOL_BEAR_THRESHOLD:
            regime.iloc[i] = BEAR
        else:
            regime.iloc[i] = CHOP

    result = pd.DataFrame({
        "date": mkt["date"].values,
        "regime": regime.values,
    })
    return result


def _percentile_rank(arr: np.ndarray) -> float:
    """Percentile rank of the last element within the array (0-100)."""
    if len(arr) < 2:
        return np.nan
    val = arr[-1]
    if np.isnan(val):
        return np.nan
    count_below = np.sum(arr[:-1] < val)
    count_equal = np.sum(arr[:-1] == val)
    rank = (count_below + 0.5 * count_equal) / (len(arr) - 1) * 100.0
    return rank


# ── Map to rebalance dates ────────────────────────────────────────────

def assign_regime_to_rebalance_dates(
    regime_df: pd.DataFrame,
    rebalance_dates: List[str],
) -> Dict[str, str]:
    """
    Map each rebalance date to a regime using the most recent trading day <= that date.

    Parameters
    ----------
    regime_df : DataFrame with columns date, regime (from compute_regime_series)
    rebalance_dates : list of YYYY-MM-DD strings

    Returns
    -------
    Dict mapping rebalance_date -> regime label (BULL/BEAR/CHOP).
    Dates before warmup or without data get "CHOP" as fallback.
    """
    # Build sorted list of (date, regime) pairs with valid regime
    valid = regime_df.dropna(subset=["regime"]).sort_values("date")
    dates_arr = valid["date"].values
    regimes_arr = valid["regime"].values

    result: Dict[str, str] = {}
    for rd in sorted(rebalance_dates):
        # Binary search for last date <= rd
        idx = np.searchsorted(dates_arr, rd, side="right") - 1
        if idx >= 0:
            result[rd] = str(regimes_arr[idx])
        else:
            result[rd] = CHOP  # fallback before warmup
    return result
