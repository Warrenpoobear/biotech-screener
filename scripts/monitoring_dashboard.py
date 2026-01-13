#!/usr/bin/env python3
"""
Wake Robin Biotech Screener - Monitoring Dashboard

Tracks ongoing portfolio performance, calculates risk metrics,
and generates alerts when thresholds are breached.

Usage:
    python scripts/monitoring_dashboard.py
    python scripts/monitoring_dashboard.py --portfolio portfolio.csv
    python scripts/monitoring_dashboard.py --html reports/dashboard.html
    python scripts/monitoring_dashboard.py --json reports/metrics.json

Author: Wake Robin Capital
Version: 1.0
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

# Try to import optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Using cached data only.")


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'benchmark_ticker': 'XBI',
    'risk_free_rate': 0.05,  # 5% annual
    'trading_days_per_year': 252,
    'alerts': {
        'drawdown_threshold': -0.15,  # -15% triggers alert
        'alpha_alert_months': 3,       # Alert if alpha < 0 for 3 months
        'volatility_threshold': 0.50,  # 50% annualized vol triggers alert
    },
    'cache_dir': 'data/cache/prices',
    'portfolio_file': 'data/portfolio/holdings.csv',
    'history_file': 'data/portfolio/performance_history.json',
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    # Returns
    portfolio_return_1d: Optional[float] = None
    portfolio_return_1w: Optional[float] = None
    portfolio_return_1m: Optional[float] = None
    portfolio_return_3m: Optional[float] = None
    portfolio_return_ytd: Optional[float] = None
    portfolio_return_1y: Optional[float] = None

    # Benchmark
    benchmark_return_1d: Optional[float] = None
    benchmark_return_1w: Optional[float] = None
    benchmark_return_1m: Optional[float] = None
    benchmark_return_3m: Optional[float] = None
    benchmark_return_ytd: Optional[float] = None
    benchmark_return_1y: Optional[float] = None


@dataclass
class AlphaMetrics:
    """Alpha generation metrics."""
    alpha_1d: Optional[float] = None
    alpha_1w: Optional[float] = None
    alpha_1m: Optional[float] = None
    alpha_3m: Optional[float] = None
    alpha_ytd: Optional[float] = None
    alpha_1y: Optional[float] = None

    # Rolling alpha
    alpha_rolling_3m: Optional[float] = None
    alpha_positive_months: int = 0
    alpha_negative_months: int = 0


@dataclass
class RiskMetrics:
    """Risk and volatility metrics."""
    volatility_30d: Optional[float] = None
    volatility_90d: Optional[float] = None
    volatility_1y: Optional[float] = None

    max_drawdown_ytd: Optional[float] = None
    max_drawdown_1y: Optional[float] = None
    current_drawdown: Optional[float] = None

    sharpe_ratio_ytd: Optional[float] = None
    sharpe_ratio_1y: Optional[float] = None

    sortino_ratio_ytd: Optional[float] = None
    beta_vs_xbi: Optional[float] = None
    correlation_vs_xbi: Optional[float] = None


@dataclass
class OperationalMetrics:
    """Operational health metrics."""
    universe_size: int = 0
    tickers_with_prices: int = 0
    tickers_screened: int = 0
    data_completeness: float = 0.0

    last_screen_date: Optional[str] = None
    last_rebalance_date: Optional[str] = None
    days_since_rebalance: int = 0

    positions_count: int = 0
    avg_position_size: float = 0.0
    max_position_size: float = 0.0


@dataclass
class Alerts:
    """Alert flags and messages."""
    drawdown_alert: bool = False
    alpha_alert: bool = False
    volatility_alert: bool = False
    data_quality_alert: bool = False
    rebalance_alert: bool = False

    alert_messages: List[str] = None

    def __post_init__(self):
        if self.alert_messages is None:
            self.alert_messages = []


@dataclass
class DashboardMetrics:
    """Complete dashboard metrics."""
    timestamp: str
    performance: PerformanceMetrics
    alpha: AlphaMetrics
    risk: RiskMetrics
    operational: OperationalMetrics
    alerts: Alerts


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_price_history(
    ticker: str,
    days: int = 365,
    use_cache: bool = True
) -> Optional[Dict[str, float]]:
    """
    Fetch price history for a ticker.

    Returns dict of {date_str: close_price}
    """
    cache_dir = Path(CONFIG['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_prices.json"

    # Check cache
    if use_cache and cache_file.exists():
        cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if cache_age < 86400:  # Less than 24 hours old
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Fetch from yfinance
    if not HAS_YFINANCE:
        return None

    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=f"{days}d")

        if hist.empty:
            return None

        prices = {}
        for date, row in hist.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            prices[date_str] = float(row['Close'])

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(prices, f)

        return prices

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def load_portfolio(portfolio_file: str = None) -> List[Dict]:
    """
    Load current portfolio holdings.

    Expected CSV format:
    Ticker,Shares,CostBasis,EntryDate
    """
    if portfolio_file is None:
        portfolio_file = CONFIG['portfolio_file']

    portfolio_path = Path(portfolio_file)

    if not portfolio_path.exists():
        # Return sample portfolio for demo
        return [
            {'ticker': 'CRSP', 'shares': 100, 'cost_basis': 50.00, 'entry_date': '2024-01-15'},
            {'ticker': 'AGIO', 'shares': 150, 'cost_basis': 30.00, 'entry_date': '2024-01-15'},
            {'ticker': 'INSM', 'shares': 200, 'cost_basis': 25.00, 'entry_date': '2024-01-15'},
            {'ticker': 'PHAT', 'shares': 100, 'cost_basis': 15.00, 'entry_date': '2024-01-15'},
            {'ticker': 'RCUS', 'shares': 150, 'cost_basis': 20.00, 'entry_date': '2024-01-15'},
        ]

    holdings = []
    with open(portfolio_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdings.append({
                'ticker': row.get('Ticker', row.get('ticker', '')),
                'shares': float(row.get('Shares', row.get('shares', 0))),
                'cost_basis': float(row.get('CostBasis', row.get('cost_basis', 0))),
                'entry_date': row.get('EntryDate', row.get('entry_date', '')),
            })

    return holdings


# =============================================================================
# Calculations
# =============================================================================

def calculate_returns(
    prices: Dict[str, float],
    periods: Dict[str, int]
) -> Dict[str, Optional[float]]:
    """Calculate returns over various periods."""
    if not prices:
        return {k: None for k in periods}

    sorted_dates = sorted(prices.keys(), reverse=True)
    if not sorted_dates:
        return {k: None for k in periods}

    latest_price = prices[sorted_dates[0]]
    returns = {}

    for period_name, days in periods.items():
        target_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Find closest date
        past_price = None
        for date in sorted_dates:
            if date <= target_date:
                past_price = prices[date]
                break

        if past_price and past_price > 0:
            returns[period_name] = (latest_price - past_price) / past_price
        else:
            returns[period_name] = None

    return returns


def calculate_volatility(prices: Dict[str, float], days: int = 30) -> Optional[float]:
    """Calculate annualized volatility."""
    if not prices or len(prices) < days:
        return None

    sorted_dates = sorted(prices.keys(), reverse=True)[:days]
    if len(sorted_dates) < 2:
        return None

    # Calculate daily returns
    daily_returns = []
    for i in range(len(sorted_dates) - 1):
        p1 = prices[sorted_dates[i]]
        p2 = prices[sorted_dates[i + 1]]
        if p2 > 0:
            daily_returns.append((p1 - p2) / p2)

    if not daily_returns:
        return None

    # Calculate standard deviation
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
    daily_vol = math.sqrt(variance)

    # Annualize
    annual_vol = daily_vol * math.sqrt(CONFIG['trading_days_per_year'])

    return annual_vol


def calculate_max_drawdown(prices: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate maximum drawdown and current drawdown.

    Returns (max_drawdown, current_drawdown)
    """
    if not prices or len(prices) < 2:
        return None, None

    sorted_dates = sorted(prices.keys())
    price_series = [prices[d] for d in sorted_dates]

    peak = price_series[0]
    max_dd = 0

    for price in price_series:
        if price > peak:
            peak = price
        dd = (price - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    # Current drawdown
    current_peak = max(price_series)
    current_price = price_series[-1]
    current_dd = (current_price - current_peak) / current_peak if current_peak > 0 else 0

    return max_dd, current_dd


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = None
) -> Optional[float]:
    """Calculate Sharpe ratio from returns series."""
    if not returns or len(returns) < 2:
        return None

    if risk_free_rate is None:
        risk_free_rate = CONFIG['risk_free_rate']

    # Daily risk-free rate
    daily_rf = risk_free_rate / CONFIG['trading_days_per_year']

    # Excess returns
    excess_returns = [r - daily_rf for r in returns]

    mean = sum(excess_returns) / len(excess_returns)
    variance = sum((r - mean) ** 2 for r in excess_returns) / len(excess_returns)
    std = math.sqrt(variance)

    if std == 0:
        return None

    # Annualize
    sharpe = (mean / std) * math.sqrt(CONFIG['trading_days_per_year'])

    return sharpe


def calculate_portfolio_value(
    holdings: List[Dict],
    prices: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Calculate portfolio value over time."""
    if not holdings or not prices:
        return {}

    # Get all dates
    all_dates = set()
    for ticker_prices in prices.values():
        if ticker_prices:
            all_dates.update(ticker_prices.keys())

    sorted_dates = sorted(all_dates)

    portfolio_values = {}
    for date in sorted_dates:
        total_value = 0
        for holding in holdings:
            ticker = holding['ticker']
            shares = holding['shares']

            ticker_prices = prices.get(ticker, {})
            if ticker_prices and date in ticker_prices:
                total_value += shares * ticker_prices[date]

        if total_value > 0:
            portfolio_values[date] = total_value

    return portfolio_values


# =============================================================================
# Dashboard Generation
# =============================================================================

def calculate_all_metrics(
    holdings: List[Dict],
    benchmark_ticker: str = None
) -> DashboardMetrics:
    """Calculate all dashboard metrics."""

    if benchmark_ticker is None:
        benchmark_ticker = CONFIG['benchmark_ticker']

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Initialize metrics
    performance = PerformanceMetrics()
    alpha = AlphaMetrics()
    risk = RiskMetrics()
    operational = OperationalMetrics()
    alerts = Alerts()

    # Fetch price data for all holdings
    ticker_prices = {}
    for holding in holdings:
        ticker = holding['ticker']
        prices = fetch_price_history(ticker, days=365)
        if prices:
            ticker_prices[ticker] = prices

    # Fetch benchmark
    benchmark_prices = fetch_price_history(benchmark_ticker, days=365)

    # Calculate portfolio values
    portfolio_values = calculate_portfolio_value(holdings, ticker_prices)

    # Return periods
    periods = {
        '1d': 1,
        '1w': 7,
        '1m': 30,
        '3m': 90,
        'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
        '1y': 365,
    }

    # Portfolio returns
    if portfolio_values:
        port_returns = calculate_returns(portfolio_values, periods)
        performance.portfolio_return_1d = port_returns.get('1d')
        performance.portfolio_return_1w = port_returns.get('1w')
        performance.portfolio_return_1m = port_returns.get('1m')
        performance.portfolio_return_3m = port_returns.get('3m')
        performance.portfolio_return_ytd = port_returns.get('ytd')
        performance.portfolio_return_1y = port_returns.get('1y')

    # Benchmark returns
    if benchmark_prices:
        bench_returns = calculate_returns(benchmark_prices, periods)
        performance.benchmark_return_1d = bench_returns.get('1d')
        performance.benchmark_return_1w = bench_returns.get('1w')
        performance.benchmark_return_1m = bench_returns.get('1m')
        performance.benchmark_return_3m = bench_returns.get('3m')
        performance.benchmark_return_ytd = bench_returns.get('ytd')
        performance.benchmark_return_1y = bench_returns.get('1y')

    # Alpha calculations
    if performance.portfolio_return_1d is not None and performance.benchmark_return_1d is not None:
        alpha.alpha_1d = performance.portfolio_return_1d - performance.benchmark_return_1d
    if performance.portfolio_return_1w is not None and performance.benchmark_return_1w is not None:
        alpha.alpha_1w = performance.portfolio_return_1w - performance.benchmark_return_1w
    if performance.portfolio_return_1m is not None and performance.benchmark_return_1m is not None:
        alpha.alpha_1m = performance.portfolio_return_1m - performance.benchmark_return_1m
    if performance.portfolio_return_3m is not None and performance.benchmark_return_3m is not None:
        alpha.alpha_3m = performance.portfolio_return_3m - performance.benchmark_return_3m
    if performance.portfolio_return_ytd is not None and performance.benchmark_return_ytd is not None:
        alpha.alpha_ytd = performance.portfolio_return_ytd - performance.benchmark_return_ytd
    if performance.portfolio_return_1y is not None and performance.benchmark_return_1y is not None:
        alpha.alpha_1y = performance.portfolio_return_1y - performance.benchmark_return_1y

    # Risk metrics
    if portfolio_values:
        risk.volatility_30d = calculate_volatility(portfolio_values, 30)
        risk.volatility_90d = calculate_volatility(portfolio_values, 90)
        risk.volatility_1y = calculate_volatility(portfolio_values, 365)

        max_dd, current_dd = calculate_max_drawdown(portfolio_values)
        risk.max_drawdown_ytd = max_dd
        risk.current_drawdown = current_dd

        # Sharpe ratio
        sorted_dates = sorted(portfolio_values.keys())
        daily_returns = []
        for i in range(1, len(sorted_dates)):
            p1 = portfolio_values[sorted_dates[i]]
            p0 = portfolio_values[sorted_dates[i-1]]
            if p0 > 0:
                daily_returns.append((p1 - p0) / p0)

        risk.sharpe_ratio_ytd = calculate_sharpe_ratio(daily_returns)

    # Operational metrics
    operational.universe_size = 316  # From screener
    operational.tickers_with_prices = len(ticker_prices)
    operational.positions_count = len(holdings)
    operational.data_completeness = len(ticker_prices) / len(holdings) if holdings else 0

    if holdings:
        total_value = sum(h['shares'] * h['cost_basis'] for h in holdings)
        if total_value > 0:
            position_sizes = [(h['shares'] * h['cost_basis']) / total_value for h in holdings]
            operational.avg_position_size = sum(position_sizes) / len(position_sizes)
            operational.max_position_size = max(position_sizes)

    # Alerts
    if risk.current_drawdown is not None:
        if risk.current_drawdown < CONFIG['alerts']['drawdown_threshold']:
            alerts.drawdown_alert = True
            alerts.alert_messages.append(
                f"DRAWDOWN ALERT: Current drawdown {risk.current_drawdown*100:.1f}% "
                f"exceeds threshold {CONFIG['alerts']['drawdown_threshold']*100:.0f}%"
            )

    if risk.volatility_30d is not None:
        if risk.volatility_30d > CONFIG['alerts']['volatility_threshold']:
            alerts.volatility_alert = True
            alerts.alert_messages.append(
                f"VOLATILITY ALERT: 30-day volatility {risk.volatility_30d*100:.1f}% "
                f"exceeds threshold {CONFIG['alerts']['volatility_threshold']*100:.0f}%"
            )

    if operational.data_completeness < 0.90:
        alerts.data_quality_alert = True
        alerts.alert_messages.append(
            f"DATA QUALITY ALERT: Only {operational.data_completeness*100:.0f}% of positions have price data"
        )

    return DashboardMetrics(
        timestamp=timestamp,
        performance=performance,
        alpha=alpha,
        risk=risk,
        operational=operational,
        alerts=alerts,
    )


# =============================================================================
# Output Formatters
# =============================================================================

def format_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format value as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:+.{decimals}f}%"


def generate_text_report(metrics: DashboardMetrics) -> str:
    """Generate text-based dashboard report."""
    lines = [
        "",
        "=" * 70,
        "WAKE ROBIN BIOTECH SCREENER - MONITORING DASHBOARD",
        "=" * 70,
        f"Generated: {metrics.timestamp}",
        "",
    ]

    # Alerts section (if any)
    if metrics.alerts.alert_messages:
        lines.extend([
            "!" * 70,
            "ALERTS",
            "!" * 70,
        ])
        for msg in metrics.alerts.alert_messages:
            lines.append(f"  >> {msg}")
        lines.append("")

    # Performance
    lines.extend([
        "-" * 70,
        "PERFORMANCE",
        "-" * 70,
        f"  {'Period':<12} {'Portfolio':<12} {'XBI':<12} {'Alpha':<12}",
        f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}",
    ])

    perf_data = [
        ('1 Day', metrics.performance.portfolio_return_1d, metrics.performance.benchmark_return_1d, metrics.alpha.alpha_1d),
        ('1 Week', metrics.performance.portfolio_return_1w, metrics.performance.benchmark_return_1w, metrics.alpha.alpha_1w),
        ('1 Month', metrics.performance.portfolio_return_1m, metrics.performance.benchmark_return_1m, metrics.alpha.alpha_1m),
        ('3 Months', metrics.performance.portfolio_return_3m, metrics.performance.benchmark_return_3m, metrics.alpha.alpha_3m),
        ('YTD', metrics.performance.portfolio_return_ytd, metrics.performance.benchmark_return_ytd, metrics.alpha.alpha_ytd),
        ('1 Year', metrics.performance.portfolio_return_1y, metrics.performance.benchmark_return_1y, metrics.alpha.alpha_1y),
    ]

    for period, port, bench, alpha in perf_data:
        lines.append(f"  {period:<12} {format_pct(port):<12} {format_pct(bench):<12} {format_pct(alpha):<12}")

    lines.append("")

    # Risk Metrics
    lines.extend([
        "-" * 70,
        "RISK METRICS",
        "-" * 70,
        f"  Volatility (30d):     {format_pct(metrics.risk.volatility_30d)}",
        f"  Volatility (90d):     {format_pct(metrics.risk.volatility_90d)}",
        f"  Volatility (1y):      {format_pct(metrics.risk.volatility_1y)}",
        "",
        f"  Max Drawdown (YTD):   {format_pct(metrics.risk.max_drawdown_ytd)}",
        f"  Current Drawdown:     {format_pct(metrics.risk.current_drawdown)}",
        "",
        f"  Sharpe Ratio (YTD):   {metrics.risk.sharpe_ratio_ytd:.2f}" if metrics.risk.sharpe_ratio_ytd else "  Sharpe Ratio (YTD):   N/A",
        "",
    ])

    # Operational
    lines.extend([
        "-" * 70,
        "OPERATIONAL METRICS",
        "-" * 70,
        f"  Universe Size:        {metrics.operational.universe_size}",
        f"  Positions:            {metrics.operational.positions_count}",
        f"  Data Completeness:    {format_pct(metrics.operational.data_completeness, 0)}",
        f"  Avg Position Size:    {format_pct(metrics.operational.avg_position_size)}",
        f"  Max Position Size:    {format_pct(metrics.operational.max_position_size)}",
        "",
    ])

    # Status summary
    status = "OK" if not any([
        metrics.alerts.drawdown_alert,
        metrics.alerts.alpha_alert,
        metrics.alerts.volatility_alert,
        metrics.alerts.data_quality_alert,
    ]) else "ALERT"

    status_color = "GREEN" if status == "OK" else "RED"

    lines.extend([
        "-" * 70,
        f"OVERALL STATUS: [{status}] ({status_color})",
        "-" * 70,
        "",
        "=" * 70,
        "END OF DASHBOARD",
        "=" * 70,
        "",
    ])

    return "\n".join(lines)


def generate_html_report(metrics: DashboardMetrics) -> str:
    """Generate HTML dashboard report."""

    def pct_cell(value: Optional[float], invert_color: bool = False) -> str:
        """Generate colored table cell for percentage value."""
        if value is None:
            return '<td class="na">N/A</td>'

        pct = value * 100
        if invert_color:
            color_class = "negative" if pct > 0 else "positive" if pct < 0 else "neutral"
        else:
            color_class = "positive" if pct > 0 else "negative" if pct < 0 else "neutral"

        return f'<td class="{color_class}">{pct:+.2f}%</td>'

    alert_section = ""
    if metrics.alerts.alert_messages:
        alert_items = "".join(f"<li>{msg}</li>" for msg in metrics.alerts.alert_messages)
        alert_section = f"""
        <div class="alert-box">
            <h3>⚠️ ALERTS</h3>
            <ul>{alert_items}</ul>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Wake Robin Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #00d4ff; margin-top: 30px; }}
        .timestamp {{ color: #888; font-size: 14px; }}
        .alert-box {{
            background: #ff4444;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .alert-box h3 {{ margin: 0 0 10px 0; }}
        .alert-box ul {{ margin: 0; padding-left: 20px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: right;
            border-bottom: 1px solid #0f3460;
        }}
        th {{
            background: #0f3460;
            color: #00d4ff;
            text-align: left;
        }}
        td:first-child {{ text-align: left; font-weight: 500; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        .neutral {{ color: #888; }}
        .na {{ color: #666; font-style: italic; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #00d4ff;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .status-ok {{ color: #00ff88; }}
        .status-alert {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Wake Robin Biotech Screener</h1>
        <p class="timestamp">Generated: {metrics.timestamp}</p>

        {alert_section}

        <h2>Performance Overview</h2>
        <table>
            <tr>
                <th>Period</th>
                <th>Portfolio</th>
                <th>XBI Benchmark</th>
                <th>Alpha</th>
            </tr>
            <tr>
                <td>1 Day</td>
                {pct_cell(metrics.performance.portfolio_return_1d)}
                {pct_cell(metrics.performance.benchmark_return_1d)}
                {pct_cell(metrics.alpha.alpha_1d)}
            </tr>
            <tr>
                <td>1 Week</td>
                {pct_cell(metrics.performance.portfolio_return_1w)}
                {pct_cell(metrics.performance.benchmark_return_1w)}
                {pct_cell(metrics.alpha.alpha_1w)}
            </tr>
            <tr>
                <td>1 Month</td>
                {pct_cell(metrics.performance.portfolio_return_1m)}
                {pct_cell(metrics.performance.benchmark_return_1m)}
                {pct_cell(metrics.alpha.alpha_1m)}
            </tr>
            <tr>
                <td>3 Months</td>
                {pct_cell(metrics.performance.portfolio_return_3m)}
                {pct_cell(metrics.performance.benchmark_return_3m)}
                {pct_cell(metrics.alpha.alpha_3m)}
            </tr>
            <tr>
                <td>YTD</td>
                {pct_cell(metrics.performance.portfolio_return_ytd)}
                {pct_cell(metrics.performance.benchmark_return_ytd)}
                {pct_cell(metrics.alpha.alpha_ytd)}
            </tr>
            <tr>
                <td>1 Year</td>
                {pct_cell(metrics.performance.portfolio_return_1y)}
                {pct_cell(metrics.performance.benchmark_return_1y)}
                {pct_cell(metrics.alpha.alpha_1y)}
            </tr>
        </table>

        <h2>Risk Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Volatility (30d)</h4>
                <div class="value">{format_pct(metrics.risk.volatility_30d)}</div>
            </div>
            <div class="metric-card">
                <h4>Max Drawdown (YTD)</h4>
                <div class="value negative">{format_pct(metrics.risk.max_drawdown_ytd)}</div>
            </div>
            <div class="metric-card">
                <h4>Current Drawdown</h4>
                <div class="value negative">{format_pct(metrics.risk.current_drawdown)}</div>
            </div>
            <div class="metric-card">
                <h4>Sharpe Ratio (YTD)</h4>
                <div class="value">{metrics.risk.sharpe_ratio_ytd:.2f if metrics.risk.sharpe_ratio_ytd else 'N/A'}</div>
            </div>
        </div>

        <h2>Operational Status</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Positions</h4>
                <div class="value">{metrics.operational.positions_count}</div>
            </div>
            <div class="metric-card">
                <h4>Data Completeness</h4>
                <div class="value">{format_pct(metrics.operational.data_completeness, 0)}</div>
            </div>
            <div class="metric-card">
                <h4>Avg Position Size</h4>
                <div class="value">{format_pct(metrics.operational.avg_position_size)}</div>
            </div>
            <div class="metric-card">
                <h4>Max Position Size</h4>
                <div class="value">{format_pct(metrics.operational.max_position_size)}</div>
            </div>
        </div>

        <h2>System Status</h2>
        <p class="{'status-ok' if not metrics.alerts.alert_messages else 'status-alert'}" style="font-size: 24px; font-weight: bold;">
            {'✅ ALL SYSTEMS NORMAL' if not metrics.alerts.alert_messages else '⚠️ ALERTS ACTIVE'}
        </p>
    </div>
</body>
</html>"""

    return html


def generate_json_output(metrics: DashboardMetrics) -> str:
    """Generate JSON output of all metrics."""
    return json.dumps(asdict(metrics), indent=2, default=str)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for monitoring dashboard."""
    parser = argparse.ArgumentParser(
        description='Wake Robin Monitoring Dashboard'
    )
    parser.add_argument(
        '--portfolio',
        type=str,
        help='Path to portfolio CSV file'
    )
    parser.add_argument(
        '--html',
        type=str,
        help='Output HTML report to file'
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Output JSON metrics to file'
    )
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Use cached data only, do not fetch new prices'
    )

    args = parser.parse_args()

    # Load portfolio
    holdings = load_portfolio(args.portfolio)
    print(f"Loaded {len(holdings)} positions")

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_all_metrics(holdings)

    # Generate text report (always shown)
    text_report = generate_text_report(metrics)
    print(text_report)

    # Save HTML if requested
    if args.html:
        html_report = generate_html_report(metrics)
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML report saved: {args.html}")

    # Save JSON if requested
    if args.json:
        json_output = generate_json_output(metrics)
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"JSON metrics saved: {args.json}")

    # Return exit code based on alerts
    if metrics.alerts.drawdown_alert or metrics.alerts.alpha_alert:
        return 1  # Critical alert
    return 0


if __name__ == '__main__':
    sys.exit(main())
