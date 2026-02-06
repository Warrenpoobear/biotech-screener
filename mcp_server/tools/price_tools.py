"""Price, returns, volume, and volatility tools."""

import json
import logging
import math
from datetime import date, timedelta

from mcp_server.app import mcp

logger = logging.getLogger(__name__)

# Lazy-initialised provider singleton
_provider = None


def _get_provider():
    global _provider
    if _provider is None:
        from wake_robin_data_pipeline.morningstar_data_provider import (
            MorningstarDataProvider,
        )
        _provider = MorningstarDataProvider()
    return _provider


def _parse_date(as_of_date: str | None) -> date:
    """Parse an ISO date string, defaulting to today."""
    if as_of_date:
        return date.fromisoformat(as_of_date)
    return date.today()


@mcp.tool()
def get_daily_returns(
    ticker: str,
    as_of_date: str | None = None,
    lookback_days: int = 365,
) -> str:
    """Get PIT-safe daily returns for a ticker.

    Args:
        ticker: Equity ticker symbol (e.g. "VRTX").
        as_of_date: ISO date (YYYY-MM-DD). Defaults to today.
        lookback_days: Calendar days of history. Defaults to 365.

    Returns JSON with the return series and summary statistics.
    """
    try:
        provider = _get_provider()
        as_of = _parse_date(as_of_date)
        returns = provider.get_daily_returns(ticker, as_of, lookback_days)

        if not returns:
            return json.dumps({"ticker": ticker, "error": "No return data available"})

        mean_ret = sum(returns) / len(returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5

        return json.dumps({
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "lookback_days": lookback_days,
            "num_observations": len(returns),
            "mean_daily_return": round(mean_ret, 8),
            "std_daily_return": round(std_ret, 8),
            "min_return": round(min(returns), 8),
            "max_return": round(max(returns), 8),
            "cumulative_return": round(math.exp(sum(returns)) - 1, 6),
            "returns": [round(r, 8) for r in returns],
        })
    except Exception as e:
        logger.exception("get_daily_returns failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def get_prices(
    ticker: str,
    as_of_date: str | None = None,
    lookback_days: int = 365,
) -> str:
    """Get daily closing prices for a ticker.

    Args:
        ticker: Equity ticker symbol.
        as_of_date: ISO date (YYYY-MM-DD). Defaults to today.
        lookback_days: Calendar days of history.

    Returns JSON with the price series.
    """
    try:
        provider = _get_provider()
        as_of = _parse_date(as_of_date)
        prices = provider.get_prices(ticker, as_of, lookback_days)

        if not prices:
            return json.dumps({"ticker": ticker, "error": "No price data available"})

        float_prices = [float(p) for p in prices]
        return json.dumps({
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "lookback_days": lookback_days,
            "num_observations": len(float_prices),
            "latest_price": float_prices[-1],
            "high": max(float_prices),
            "low": min(float_prices),
            "prices": float_prices,
        })
    except Exception as e:
        logger.exception("get_prices failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def get_volumes(
    ticker: str,
    as_of_date: str | None = None,
    lookback_days: int = 365,
) -> str:
    """Get daily trading volumes for a ticker.

    Args:
        ticker: Equity ticker symbol.
        as_of_date: ISO date (YYYY-MM-DD). Defaults to today.
        lookback_days: Calendar days of history.

    Returns JSON with the volume series and average.
    """
    try:
        provider = _get_provider()
        as_of = _parse_date(as_of_date)
        volumes = provider.get_volumes(ticker, as_of, lookback_days)

        if not volumes:
            return json.dumps({"ticker": ticker, "error": "No volume data available"})

        return json.dumps({
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "lookback_days": lookback_days,
            "num_observations": len(volumes),
            "average_volume": int(sum(volumes) / len(volumes)),
            "max_volume": max(volumes),
            "min_volume": min(volumes),
            "latest_volume": volumes[-1],
            "volumes": volumes,
        })
    except Exception as e:
        logger.exception("get_volumes failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def compute_volatility(
    ticker: str,
    as_of_date: str | None = None,
    window_days: int = 60,
) -> str:
    """Compute annualised volatility from daily returns.

    Args:
        ticker: Equity ticker symbol.
        as_of_date: ISO date (YYYY-MM-DD). Defaults to today.
        window_days: Calendar days for the return window.

    Uses sqrt(252) annualisation.
    """
    try:
        provider = _get_provider()
        as_of = _parse_date(as_of_date)
        returns = provider.get_daily_returns(ticker, as_of, window_days)

        if len(returns) < 5:
            return json.dumps({"ticker": ticker, "error": "Insufficient return data"})

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = variance ** 0.5
        annual_vol = daily_vol * (252 ** 0.5)

        return json.dumps({
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "window_days": window_days,
            "num_observations": len(returns),
            "daily_volatility": round(daily_vol, 8),
            "annualized_volatility": round(annual_vol, 6),
            "annualized_volatility_pct": f"{annual_vol * 100:.2f}%",
        })
    except Exception as e:
        logger.exception("compute_volatility failed for %s", ticker)
        return json.dumps({"ticker": ticker, "error": str(e)})


@mcp.tool()
def get_batch_market_data(
    tickers: list[str],
    as_of_date: str | None = None,
) -> str:
    """Fetch summary market data for multiple tickers at once.

    Args:
        tickers: List of ticker symbols.
        as_of_date: ISO date (YYYY-MM-DD). Defaults to today.

    Returns JSON with per-ticker summaries (latest price, 30-day avg
    volume, annualised volatility).
    """
    try:
        from wake_robin_data_pipeline.morningstar_data_provider import (
            BatchMorningstarProvider,
        )

        as_of = _parse_date(as_of_date)
        batch = BatchMorningstarProvider()
        raw = batch.get_batch_data(tickers, as_of, lookback_days=90, include_xbi=False)

        summaries = {}
        for tk, data in raw.items():
            prices = data.get("prices", [])
            returns = data.get("returns", [])
            volumes = data.get("volumes", [])

            summary: dict = {"ticker": tk, "num_days": data.get("num_days", 0)}
            if prices:
                summary["latest_price"] = float(prices[-1])
            if volumes:
                summary["avg_volume_30d"] = int(
                    sum(volumes[-30:]) / min(len(volumes), 30)
                )
            if len(returns) >= 5:
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
                summary["annualized_vol"] = round((var ** 0.5) * (252 ** 0.5), 6)

            summaries[tk] = summary

        return json.dumps({
            "as_of": as_of.isoformat(),
            "num_tickers": len(summaries),
            "data": summaries,
        })
    except Exception as e:
        logger.exception("get_batch_market_data failed")
        return json.dumps({"error": str(e)})
