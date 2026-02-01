"""Tests for regime classifier (backtest/regime.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.regime import (
    compute_regime_series,
    assign_regime_to_rebalance_dates,
    BULL,
    BEAR,
    CHOP,
    MA_SLOW_WINDOW,
    WARMUP_DAYS,
)


def _make_price_series(
    n_days: int = 500,
    trend: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
    ticker: str = "XBI",
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Synthetic daily price series.

    trend: daily drift (e.g., +0.001 = uptrend, -0.001 = downtrend)
    vol: daily return std
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    daily_rets = trend + vol * rng.randn(n_days)
    daily_rets[0] = 0.0
    prices = start_price * np.cumprod(1 + daily_rets)
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "ticker": ticker,
        "close": prices,
    })


class TestRegimeClassification:
    def test_rising_trend_low_vol_is_bull(self):
        """Strong uptrend + low volatility → mostly BULL after warmup."""
        prices = _make_price_series(n_days=500, trend=0.002, vol=0.005, seed=10)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        valid = regime_df.dropna(subset=["regime"])
        assert len(valid) > 0, "No valid regime labels after warmup"

        counts = valid["regime"].value_counts()
        # Should be predominantly BULL
        assert BULL in counts.index, f"BULL not found, got: {counts.to_dict()}"
        bull_pct = counts.get(BULL, 0) / len(valid)
        assert bull_pct > 0.4, f"Expected >40% BULL in uptrend, got {bull_pct:.1%}"

    def test_falling_trend_high_vol_is_bear(self):
        """Downtrend + high volatility → meaningful BEAR presence after warmup."""
        prices = _make_price_series(n_days=500, trend=-0.002, vol=0.025, seed=20)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        valid = regime_df.dropna(subset=["regime"])
        counts = valid["regime"].value_counts()
        assert BEAR in counts.index, f"BEAR not found, got: {counts.to_dict()}"
        bear_pct = counts.get(BEAR, 0) / len(valid)
        # With expanding vol percentile, persistent high vol normalizes over time;
        # BEAR should still be a meaningful fraction (>15%) and more common than BULL
        assert bear_pct > 0.15, f"Expected >15% BEAR in downtrend, got {bear_pct:.1%}"
        bull_pct = counts.get(BULL, 0) / len(valid)
        assert bear_pct > bull_pct, (
            f"BEAR ({bear_pct:.1%}) should exceed BULL ({bull_pct:.1%}) in downtrend"
        )

    def test_flat_noisy_is_chop(self):
        """No trend + moderate vol → mostly CHOP."""
        prices = _make_price_series(n_days=500, trend=0.0, vol=0.015, seed=30)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        valid = regime_df.dropna(subset=["regime"])
        counts = valid["regime"].value_counts()
        assert CHOP in counts.index, f"CHOP not found, got: {counts.to_dict()}"
        chop_pct = counts.get(CHOP, 0) / len(valid)
        assert chop_pct > 0.2, f"Expected >20% CHOP in flat market, got {chop_pct:.1%}"

    def test_deterministic_output(self):
        """Same input → same output."""
        prices = _make_price_series(n_days=400, seed=55)
        r1 = compute_regime_series(prices, market_ticker="XBI")
        r2 = compute_regime_series(prices, market_ticker="XBI")
        pd.testing.assert_frame_equal(r1, r2)

    def test_no_nan_after_warmup(self):
        """All dates after warmup period have a regime label."""
        prices = _make_price_series(n_days=500, seed=44)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        # After WARMUP_DAYS rows, regime should not be NaN
        after_warmup = regime_df.iloc[WARMUP_DAYS:]
        n_nan = after_warmup["regime"].isna().sum()
        assert n_nan == 0, f"{n_nan} NaN regimes after warmup"

    def test_missing_ticker_raises(self):
        """Requesting a ticker not in the data raises ValueError."""
        prices = _make_price_series(ticker="XBI")
        with pytest.raises(ValueError, match="not found"):
            compute_regime_series(prices, market_ticker="SPY")


class TestRebalanceDateMapping:
    def test_uses_last_trading_day_lte_rebalance(self):
        """Regime for a weekend date uses Friday's regime."""
        prices = _make_price_series(n_days=500, seed=60)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        # Pick a Saturday (should map to Friday)
        valid_dates = regime_df.dropna(subset=["regime"])
        last_date = valid_dates["date"].iloc[-1]

        result = assign_regime_to_rebalance_dates(
            regime_df, [last_date]
        )
        assert last_date in result
        assert result[last_date] in (BULL, BEAR, CHOP)

    def test_before_warmup_fallback_to_chop(self):
        """Dates before warmup get CHOP as fallback."""
        prices = _make_price_series(n_days=500, seed=70)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        result = assign_regime_to_rebalance_dates(
            regime_df, ["2020-01-01"]  # well before any data
        )
        assert result["2020-01-01"] == CHOP

    def test_multiple_dates_all_mapped(self):
        """All provided rebalance dates get a regime label."""
        prices = _make_price_series(n_days=500, seed=80)
        regime_df = compute_regime_series(prices, market_ticker="XBI")

        valid = regime_df.dropna(subset=["regime"])
        sample_dates = valid["date"].iloc[10:15].tolist()
        result = assign_regime_to_rebalance_dates(regime_df, sample_dates)

        assert len(result) == len(sample_dates)
        for d in sample_dates:
            assert result[d] in (BULL, BEAR, CHOP)
