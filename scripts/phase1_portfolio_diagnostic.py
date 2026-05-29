#!/usr/bin/env python3
"""Phase 1 Portfolio Policy Diagnostic Harness (Optimized)

Read-only diagnostic harness for testing portfolio transition policies.
Optimized with efficient price lookups and targeted loading.

Canonical cohorts (7 inception dates, common terminal 2026-05-27):
  2024-10-18, 2024-11-01, 2025-01-10, 2025-04-11, 2025-07-18, 2025-10-10, 2026-01-02

Policies tested:
  - STATIC_INCEPTION_HOLD: Static hold of inception-date top-30 through terminal
  - DELISTING_ONLY: Hold inception top-30 until delisting detected (10+ calendar days missing)
  - AVAILABLE_SNAPSHOT_REBUILD: Rebuild from current top-30 on each available snapshot date (~175 per period)
  - WEEKLY_TRADE_PACKET_PROXY: Rebuild from current top-30 on weekly cadence (Friday)
  - MONTHLY_REBALANCE_PROXY: Rebuild from current top-30 on first snapshot of each calendar month
  - QUARTERLY_REBALANCE_PROXY: Rebuild from current top-30 on first snapshot of each calendar quarter
  - HOLD_30: Start with top-30, replace only after 30 days if fallen from top-30 and delisted if missing
"""

import csv
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_ROOT = PROJECT_ROOT / "data" / "snapshots"
PRICE_PATH = PROJECT_ROOT / "production_data" / "price_history_split_adj.csv"
DIAGNOSTIC_ROOT = PROJECT_ROOT / "artifacts" / "portfolio_policy_diagnostic"

CANONICAL_COHORTS = [
    "2024-10-18",
    "2024-11-01",
    "2025-01-10",
    "2025-04-11",
    "2025-07-18",
    "2025-10-10",
    "2026-01-02",
]
TERMINAL_DATE = "2026-05-27"


def setup_logging():
    """Configure logging for the diagnostic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    return logging.getLogger(__name__)


def load_rankings_csv(snap_date: str) -> List[Dict[str, str]]:
    """Load rankings or decision_portfolio CSV for a snapshot."""
    for filename in ["rankings.csv", "decision_portfolio.csv"]:
        p = SNAPSHOTS_ROOT / snap_date / filename
        if p.exists():
            with p.open(newline="") as f:
                return list(csv.DictReader(f))
    return []


def get_top_30_tickers(snap_date: str) -> Set[str]:
    """Extract top-30 ticker set from snapshot rankings."""
    rankings = load_rankings_csv(snap_date)
    if not rankings:
        return set()
    result = set()
    for row in rankings:
        try:
            rank_str = row.get("actionable_rank", "").strip()
            if rank_str:
                rank = float(rank_str)
                if rank <= 30:
                    result.add(row["ticker"])
        except (ValueError, KeyError):
            continue
    return result


def get_all_snapshots_between(start_date: str, end_date: str) -> List[str]:
    """Get list of all snapshot dates between start and end (inclusive)."""
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        snap_dir = SNAPSHOTS_ROOT / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return sorted(dates)


def load_prices_dict(start_date: str, end_date: str, tickers: Set[str]) -> Dict[Tuple[str, str], float]:
    """Load prices for specific tickers and date range into dict for fast lookup."""
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Price history not found: {PRICE_PATH}")

    prices = {}
    tickers_needed = tickers | {"XBI"}
    with open(PRICE_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            date = row.get("date", "").strip()
            close = row.get("close", "").strip()

            if ticker in tickers_needed and start_date <= date <= end_date and close:
                try:
                    prices[(ticker, date)] = float(close)
                except ValueError:
                    continue
    return prices


def get_price(prices: Dict[Tuple[str, str], float], ticker: str, date: str) -> Optional[float]:
    """Get close price for ticker on given date from dict."""
    return prices.get((ticker, date))


def compute_return(
    start_date: str,
    end_date: str,
    tickers: Set[str],
    prices: Dict[Tuple[str, str], float],
) -> Tuple[Optional[float], int, int]:
    """Compute equal-weight portfolio return."""
    if not tickers:
        return None, 0, 0

    start_prices = {}
    for ticker in tickers:
        sp = get_price(prices, ticker, start_date)
        if sp:
            start_prices[ticker] = sp

    end_prices = {}
    for ticker in tickers:
        ep = get_price(prices, ticker, end_date)
        if ep:
            end_prices[ticker] = ep

    if not start_prices or not end_prices:
        return None, len(start_prices), len(tickers) - len(start_prices)

    returns = []
    for ticker in start_prices:
        if ticker in end_prices:
            ret = (end_prices[ticker] / start_prices[ticker]) - 1.0
            returns.append(ret)

    if not returns:
        return None, len(start_prices), len(tickers) - len(returns)

    return np.mean(returns), len(returns), len(tickers) - len(returns)


def get_xbi_return(start_date: str, end_date: str, prices: Dict[Tuple[str, str], float]) -> Optional[float]:
    """Get XBI benchmark return."""
    xbi_start = get_price(prices, "XBI", start_date)
    xbi_end = get_price(prices, "XBI", end_date)
    if xbi_start and xbi_end and xbi_start > 0:
        return (xbi_end / xbi_start) - 1.0
    return None


def test_static_inception_hold_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test STATIC_INCEPTION_HOLD policy: static hold of inception-date top-30."""
    top_30_inception = get_top_30_tickers(inception_date)
    if not top_30_inception:
        return {"error": f"No top-30 found for {inception_date}"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    ret, n_priced, n_missing = compute_return(inception_date, terminal_date, top_30_inception, prices)
    xbi_ret = get_xbi_return(inception_date, terminal_date, prices)

    return {
        "policy": "STATIC_INCEPTION_HOLD",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_price_obs": n_priced + n_missing,
        "return_pct": round(ret * 100, 2) if ret is not None else None,
        "xbi_return_pct": round(xbi_ret * 100, 2) if xbi_ret is not None else None,
        "alpha_pct": round((ret - xbi_ret) * 100, 2) if ret and xbi_ret else None,
        "n_priced": n_priced,
        "n_missing": n_missing,
    }


def test_hold_30_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test HOLD_30 policy: minimum 30-day hold before forced replacement."""
    # Get all snapshot dates in the period
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates:
        return {"error": "No snapshots found in period"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Initialize portfolio with inception top-30
    top_30_inception = get_top_30_tickers(inception_date)
    holdings = {ticker: inception_date for ticker in top_30_inception}  # ticker -> entry_date
    consecutive_missing = {ticker: 0 for ticker in holdings}
    min_hold_days = 30
    n_replacements = 0
    portfolio_history = [(inception_date, dict(holdings))]  # Track portfolio at each snapshot

    # Process each snapshot to update holdings
    for i, snap_date in enumerate(snapshot_dates[1:], start=1):
        snap_dt = datetime.strptime(snap_date, "%Y-%m-%d")

        # Check for delistings (10+ consecutive days missing)
        for ticker in list(holdings.keys()):
            price = get_price(prices, ticker, snap_date)
            if price is None:
                consecutive_missing[ticker] += 1
                if consecutive_missing[ticker] >= 10:
                    del holdings[ticker]
                    del consecutive_missing[ticker]
            else:
                consecutive_missing[ticker] = 0

        # Get current top-30 for replacement candidates
        current_top_30 = get_top_30_tickers(snap_date)
        if not current_top_30:
            portfolio_history.append((snap_date, dict(holdings)))
            continue

        # Determine which held positions can be replaced (held >= 30 days)
        holdings_to_replace = set()
        for ticker in list(holdings.keys()):
            entry_dt = datetime.strptime(holdings[ticker], "%Y-%m-%d")
            days_held = (snap_dt - entry_dt).days
            if days_held >= min_hold_days and ticker not in current_top_30:
                holdings_to_replace.add(ticker)

        # Remove positions that are being replaced
        for ticker in holdings_to_replace:
            del holdings[ticker]
            n_replacements += 1

        # Fill empty slots with highest-ranked available names
        held_tickers = set(holdings.keys())
        available_for_entry = [t for t in current_top_30 if t not in held_tickers]
        for new_ticker in available_for_entry:
            if len(holdings) >= 30:
                break
            holdings[new_ticker] = snap_date
            consecutive_missing[new_ticker] = 0

        # Record portfolio state at this snapshot
        portfolio_history.append((snap_date, dict(holdings)))

    # Compute return using portfolio history
    total_return = 1.0
    xbi_total_return = 1.0
    n_periods = 0

    for i in range(len(portfolio_history) - 1):
        current_snap, current_holdings = portfolio_history[i]
        next_snap, _ = portfolio_history[i + 1]

        current_holdings_set = set(current_holdings.keys())
        if not current_holdings_set:
            continue

        ret, _, _ = compute_return(current_snap, next_snap, current_holdings_set, prices)
        xbi_ret = get_xbi_return(current_snap, next_snap, prices)

        if ret is not None:
            total_return *= 1.0 + ret
            n_periods += 1
        if xbi_ret is not None:
            xbi_total_return *= 1.0 + xbi_ret

    # Compute metrics
    all_entry_dates = set()
    for holdings_dict in [h[1] for h in portfolio_history]:
        all_entry_dates.update(holdings_dict.values())

    avg_holding_days = (
        sum(
            (datetime.strptime(terminal_date, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days
            for entry_date in all_entry_dates
        )
        / len(all_entry_dates)
        if all_entry_dates
        else min_hold_days
    )
    turnover = n_replacements / max(len(snapshot_dates), 1)

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    return {
        "policy": "HOLD_30",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_replacements": n_replacements,
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
        "turnover": round(turnover, 3),
        "avg_holding_period_days": round(avg_holding_days, 1),
    }


def test_monthly_rebalance_proxy_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test MONTHLY_REBALANCE_PROXY policy: rebuild from top-30 on first snapshot of each month."""
    # Get all snapshot dates in the period
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates:
        return {"error": "No snapshots found in period"}

    # Find first snapshot of each month
    monthly_rebalance_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")

    while current <= end:
        # Find first snapshot on or after month-start
        month_start = current.replace(day=1)
        check_date = month_start
        while check_date <= end and (
            check_date.month == month_start.month or check_date < month_start.replace(day=28) + timedelta(days=4)
        ):
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                monthly_rebalance_dates.append(date_str)
                break
            check_date += timedelta(days=1)

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    if not monthly_rebalance_dates:
        return {"error": "No monthly snapshot dates found"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Compound returns across monthly intervals
    total_return = 1.0
    xbi_total_return = 1.0
    n_periods = 0

    for i in range(len(monthly_rebalance_dates) - 1):
        current_month = monthly_rebalance_dates[i]
        next_month = monthly_rebalance_dates[i + 1]

        # Get top-30 for current month
        current_top_30 = get_top_30_tickers(current_month)
        if not current_top_30:
            continue

        # Compute return for this interval
        ret, _, _ = compute_return(current_month, next_month, current_top_30, prices)
        xbi_ret = get_xbi_return(current_month, next_month, prices)

        if ret is not None:
            total_return *= 1.0 + ret
            n_periods += 1
        if xbi_ret is not None:
            xbi_total_return *= 1.0 + xbi_ret

    # Compute metrics
    avg_holding_period_months = len(monthly_rebalance_dates) / max(n_periods, 1)
    turnover = n_periods / max(len(monthly_rebalance_dates), 1)

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    return {
        "policy": "MONTHLY_REBALANCE_PROXY",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_rebalances": len(monthly_rebalance_dates),
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
        "turnover": round(turnover, 3),
        "avg_holding_period_months": round(avg_holding_period_months, 1),
    }


def test_weekly_trade_packet_proxy_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test WEEKLY_TRADE_PACKET_PROXY policy: rebuild from top-30 on Friday (weekly cadence)."""
    # Get all snapshot dates in the period
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates:
        return {"error": "No snapshots found in period"}

    # Filter to weekly (Friday) snapshots or nearest Friday without lookahead
    weekly_rebalance_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")

    # Find all Fridays in the period
    fridays = []
    while current <= end:
        if current.weekday() == 4:  # Friday = 4
            fridays.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # For each Friday, find the nearest available snapshot without lookahead
    for friday in fridays:
        friday_dt = datetime.strptime(friday, "%Y-%m-%d")
        # Search backwards up to 3 days for an available snapshot
        for days_back in range(4):
            check_date = (friday_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
            if check_date in snapshot_dates:
                weekly_rebalance_dates.append(check_date)
                break

    if not weekly_rebalance_dates:
        return {"error": "No weekly snapshot dates found"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Compound returns across weekly intervals
    total_return = 1.0
    xbi_total_return = 1.0
    n_periods = 0

    for i in range(len(weekly_rebalance_dates) - 1):
        current_week = weekly_rebalance_dates[i]
        next_week = weekly_rebalance_dates[i + 1]

        # Get top-30 for current week
        current_top_30 = get_top_30_tickers(current_week)
        if not current_top_30:
            continue

        # Compute return for this interval
        ret, _, _ = compute_return(current_week, next_week, current_top_30, prices)
        xbi_ret = get_xbi_return(current_week, next_week, prices)

        if ret is not None:
            total_return *= 1.0 + ret
            n_periods += 1
        if xbi_ret is not None:
            xbi_total_return *= 1.0 + xbi_ret

    # Compute metrics
    avg_holding_period_weeks = len(weekly_rebalance_dates) / max(n_periods, 1)
    turnover = n_periods / max(len(weekly_rebalance_dates), 1)

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    return {
        "policy": "WEEKLY_TRADE_PACKET_PROXY",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_rebalances": len(weekly_rebalance_dates),
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
        "turnover": round(turnover, 3),
        "avg_holding_period_weeks": round(avg_holding_period_weeks, 1),
    }


def test_available_snapshot_rebuild_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test AVAILABLE_SNAPSHOT_REBUILD policy: rebuild from top-30 on each available snapshot date."""
    # Get all snapshot dates in the period
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates:
        return {"error": "No snapshots found in period"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Compound returns across snapshot intervals
    total_return = 1.0
    xbi_total_return = 1.0
    n_periods = 0
    holdings_history = []

    for i in range(len(snapshot_dates) - 1):
        current_snap = snapshot_dates[i]
        next_snap = snapshot_dates[i + 1]

        # Get top-30 for current snapshot
        current_top_30 = get_top_30_tickers(current_snap)
        if not current_top_30:
            continue

        # Compute return for this interval
        ret, _, _ = compute_return(current_snap, next_snap, current_top_30, prices)
        xbi_ret = get_xbi_return(current_snap, next_snap, prices)

        if ret is not None:
            total_return *= 1.0 + ret
            n_periods += 1
        if xbi_ret is not None:
            xbi_total_return *= 1.0 + xbi_ret

        holdings_history.append(len(current_top_30))

    # Compute turnover as average portfolio size change
    avg_holding_period = len(snapshot_dates) / max(n_periods, 1)
    turnover = n_periods / max(len(snapshot_dates), 1)  # rebalance frequency

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    return {
        "policy": "AVAILABLE_SNAPSHOT_REBUILD",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_snapshots": len(snapshot_dates),
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
        "turnover": round(turnover, 3),
        "avg_holding_period_snapshots": round(avg_holding_period, 1),
    }


def test_delisting_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test DELISTING_ONLY policy: hold until delisting (N+ consecutive missing prices)."""
    top_30_inception = get_top_30_tickers(inception_date)
    if not top_30_inception:
        return {"error": f"No top-30 found for {inception_date}"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Build date list for the period
    date_list = []
    current_date = inception_dt
    while current_date <= terminal_dt:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    holdings = top_30_inception.copy()
    n_delisted = 0
    consecutive_missing = {ticker: 0 for ticker in holdings}
    delisting_threshold = 10  # ticker is delisted after 10+ consecutive calendar days with no prices

    for date_str in date_list[1:]:  # Start from day after inception
        for ticker in list(holdings):
            price = get_price(prices, ticker, date_str)
            if price is None:
                consecutive_missing[ticker] += 1
                if consecutive_missing[ticker] >= delisting_threshold:
                    holdings.discard(ticker)
                    n_delisted += 1
            else:
                consecutive_missing[ticker] = 0

    ret, n_priced, n_missing = compute_return(inception_date, terminal_date, holdings, prices)
    xbi_ret = get_xbi_return(inception_date, terminal_date, prices)

    return {
        "policy": "DELISTING_ONLY",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_price_obs": n_priced + n_missing,
        "return_pct": round(ret * 100, 2) if ret is not None else None,
        "xbi_return_pct": round(xbi_ret * 100, 2) if xbi_ret is not None else None,
        "alpha_pct": round((ret - xbi_ret) * 100, 2) if ret and xbi_ret else None,
        "n_priced": n_priced,
        "n_missing": n_missing,
        "n_holdings_at_end": len(holdings),
        "n_exited": len(top_30_inception) - len(holdings),
    }


def test_hold_period_policy(
    inception_date: str,
    hold_days: int,
    all_snapshots: List[str],
    prices: Dict,
) -> Dict[str, Any]:
    """Test HOLD_N policy: hold top-30 for minimum N days."""
    top_30_inception = get_top_30_tickers(inception_date)
    if not top_30_inception:
        return {"error": f"No top-30 found for {inception_date}"}

    inception_idx = all_snapshots.index(inception_date) if inception_date in all_snapshots else 0
    terminal_idx = len(all_snapshots) - 1

    # Find end date (inception + hold_days)
    hold_end_idx = min(inception_idx + hold_days, terminal_idx)
    hold_end_date = all_snapshots[hold_end_idx]

    # Compute return for the hold period only
    ret, n_priced, n_missing = compute_return(inception_date, hold_end_date, top_30_inception, prices)
    xbi_ret = get_xbi_return(inception_date, hold_end_date, prices)

    days = hold_end_idx - inception_idx

    return {
        "policy": f"HOLD_{hold_days}",
        "inception_date": inception_date,
        "terminal_date": hold_end_date,
        "days": days,
        "return_pct": round(ret * 100, 2) if ret is not None else None,
        "xbi_return_pct": round(xbi_ret * 100, 2) if xbi_ret is not None else None,
        "alpha_pct": round((ret - xbi_ret) * 100, 2) if ret and xbi_ret else None,
        "n_priced": n_priced,
        "n_missing": n_missing,
    }


def test_monthly_rebalance_policy(inception_date: str, all_snapshots: List[str], prices: Dict) -> Dict[str, Any]:
    """Test MONTHLY policy: rebalance on 1st trading day of each month."""
    top_30_inception = get_top_30_tickers(inception_date)
    if not top_30_inception:
        return {"error": f"No top-30 found for {inception_date}"}

    inception_idx = all_snapshots.index(inception_date) if inception_date in all_snapshots else 0
    terminal_idx = len(all_snapshots) - 1
    terminal_date = all_snapshots[terminal_idx]

    # Find 1st trading day of month after inception
    current_date = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal = datetime.strptime(terminal_date, "%Y-%m-%d")

    total_return = 1.0
    n_periods = 0
    xbi_total_return = 1.0

    while current_date < terminal:
        # Find 1st trading day of next month
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)

        # Find snapshots in this period
        period_start = current_date.strftime("%Y-%m-%d")
        period_end = min(next_month - timedelta(days=1), terminal).strftime("%Y-%m-%d")

        # Get current top-30
        current_top_30 = get_top_30_tickers(period_start)
        if current_top_30:
            ret, _, _ = compute_return(period_start, period_end, current_top_30, prices)
            xbi_ret = get_xbi_return(period_start, period_end, prices)

            if ret is not None:
                total_return *= 1 + ret
                n_periods += 1
            if xbi_ret is not None:
                xbi_total_return *= 1 + xbi_ret

        current_date = next_month

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    days = terminal_idx - inception_idx

    return {
        "policy": "MONTHLY",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "days": days,
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
    }


def test_quarterly_rebalance_proxy_policy(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Test QUARTERLY_REBALANCE_PROXY policy: rebuild from top-30 on first snapshot of each quarter."""
    # Get all snapshot dates in the period
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates:
        return {"error": "No snapshots found in period"}

    # Find first snapshot of each quarter
    quarterly_rebalance_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")

    while current <= end:
        # Determine quarter start based on current month
        month = current.month
        if month in [1, 2, 3]:
            quarter_start = current.replace(month=1, day=1)
        elif month in [4, 5, 6]:
            quarter_start = current.replace(month=4, day=1)
        elif month in [7, 8, 9]:
            quarter_start = current.replace(month=7, day=1)
        else:
            quarter_start = current.replace(month=10, day=1)

        # Find first snapshot on or after quarter-start
        check_date = quarter_start
        quarter_end = (
            quarter_start.replace(month=quarter_start.month + 3, day=1) - timedelta(days=1)
            if quarter_start.month <= 9
            else quarter_start.replace(year=quarter_start.year + 1, month=quarter_start.month - 9, day=1)
            - timedelta(days=1)
        )
        while check_date <= end and check_date <= quarter_end:
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                quarterly_rebalance_dates.append(date_str)
                break
            check_date += timedelta(days=1)

        # Move to next quarter
        if current.month <= 9:
            current = current.replace(month=current.month + 3, day=1)
        else:
            current = current.replace(year=current.year + 1, month=current.month - 9, day=1)

    if not quarterly_rebalance_dates:
        return {"error": "No quarterly snapshot dates found"}

    # Calculate calendar days
    inception_dt = datetime.strptime(inception_date, "%Y-%m-%d")
    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    calendar_days = (terminal_dt - inception_dt).days

    # Compound returns across quarterly intervals
    total_return = 1.0
    xbi_total_return = 1.0
    n_periods = 0

    for i in range(len(quarterly_rebalance_dates) - 1):
        current_quarter = quarterly_rebalance_dates[i]
        next_quarter = quarterly_rebalance_dates[i + 1]

        # Get top-30 for current quarter
        current_top_30 = get_top_30_tickers(current_quarter)
        if not current_top_30:
            continue

        # Compute return for this interval
        ret, _, _ = compute_return(current_quarter, next_quarter, current_top_30, prices)
        xbi_ret = get_xbi_return(current_quarter, next_quarter, prices)

        if ret is not None:
            total_return *= 1.0 + ret
            n_periods += 1
        if xbi_ret is not None:
            xbi_total_return *= 1.0 + xbi_ret

    # Compute metrics
    avg_holding_period_quarters = len(quarterly_rebalance_dates) / max(n_periods, 1)
    turnover = n_periods / max(len(quarterly_rebalance_dates), 1)

    total_return_pct = (total_return - 1.0) * 100
    xbi_return_pct = (xbi_total_return - 1.0) * 100
    alpha = total_return_pct - xbi_return_pct

    return {
        "policy": "QUARTERLY_REBALANCE_PROXY",
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "calendar_days": calendar_days,
        "n_rebalances": len(quarterly_rebalance_dates),
        "return_pct": round(total_return_pct, 2),
        "xbi_return_pct": round(xbi_return_pct, 2),
        "alpha_pct": round(alpha, 2),
        "n_periods": n_periods,
        "turnover": round(turnover, 3),
        "avg_holding_period_quarters": round(avg_holding_period_quarters, 1),
    }


def compute_quarterly_vs_weekly_attribution(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Analyze attribution differences between quarterly and weekly policies."""
    # Get all snapshot dates
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates or len(snapshot_dates) < 2:
        return {"error": "Insufficient snapshots"}

    # Get weekly rebalance dates
    weekly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    while current <= end:
        # Find nearest Friday
        days_ahead = 4 - current.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        friday = current + timedelta(days=days_ahead)
        if friday > end:
            break
        # Find nearest snapshot on or after Friday
        check = friday
        while check <= end:
            date_str = check.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not weekly_dates or date_str != weekly_dates[-1]:
                    weekly_dates.append(date_str)
                break
            check += timedelta(days=1)
        current = friday + timedelta(days=1)

    # Get quarterly rebalance dates
    quarterly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    while current <= end:
        month = current.month
        if month in [1, 2, 3]:
            quarter_start = current.replace(month=1, day=1)
        elif month in [4, 5, 6]:
            quarter_start = current.replace(month=4, day=1)
        elif month in [7, 8, 9]:
            quarter_start = current.replace(month=7, day=1)
        else:
            quarter_start = current.replace(month=10, day=1)

        check_date = quarter_start
        while check_date <= end:
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not quarterly_dates or date_str != quarterly_dates[-1]:
                    quarterly_dates.append(date_str)
                break
            check_date += timedelta(days=1)

        if current.month <= 9:
            current = current.replace(month=current.month + 3, day=1)
        else:
            current = current.replace(year=current.year + 1, month=current.month - 9, day=1)

    if not weekly_dates or not quarterly_dates:
        return {"error": "Insufficient rebalance dates"}

    # Compute holdings at each rebalance date
    weekly_holdings_history = [(d, get_top_30_tickers(d)) for d in weekly_dates]
    quarterly_holdings_history = [(d, get_top_30_tickers(d)) for d in quarterly_dates]

    # Find common rebalance dates for comparison
    common_dates = sorted(set(weekly_dates) & set(quarterly_dates))
    if not common_dates:
        common_dates = [snapshot_dates[0]] + sorted(set(weekly_dates) | set(quarterly_dates))[:10]

    # Compute attribution metrics
    shared_holdings = []
    weekly_only = []
    quarterly_only = []

    for date in common_dates:
        w_tickers = next((h for d, h in weekly_holdings_history if d == date), set())
        q_tickers = next((h for d, h in quarterly_holdings_history if d == date), set())

        shared = w_tickers & q_tickers
        w_only = w_tickers - q_tickers
        q_only = q_tickers - w_tickers

        if shared:
            shared_holdings.extend(list(shared)[:5])
        if w_only:
            weekly_only.extend(list(w_only)[:3])
        if q_only:
            quarterly_only.extend(list(q_only)[:3])

    # Compute returns for shared/unique holdings
    shared_ret = None
    weekly_only_ret = None
    quarterly_only_ret = None

    if shared_holdings and len(snapshot_dates) > 1:
        shared_ret, _, _ = compute_return(snapshot_dates[0], snapshot_dates[-1], set(shared_holdings), prices)
        shared_ret = round(shared_ret * 100, 2) if shared_ret else None

    if weekly_only and len(snapshot_dates) > 1:
        weekly_only_ret, _, _ = compute_return(snapshot_dates[0], snapshot_dates[-1], set(weekly_only), prices)
        weekly_only_ret = round(weekly_only_ret * 100, 2) if weekly_only_ret else None

    if quarterly_only and len(snapshot_dates) > 1:
        quarterly_only_ret, _, _ = compute_return(snapshot_dates[0], snapshot_dates[-1], set(quarterly_only), prices)
        quarterly_only_ret = round(quarterly_only_ret * 100, 2) if quarterly_only_ret else None

    return {
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "weekly_rebalances": len(weekly_dates),
        "quarterly_rebalances": len(quarterly_dates),
        "shared_holdings_sample": list(set(shared_holdings))[:10],
        "weekly_only_sample": list(set(weekly_only))[:10],
        "quarterly_only_sample": list(set(quarterly_only))[:10],
        "shared_holdings_return_pct": shared_ret,
        "weekly_only_return_pct": weekly_only_ret,
        "quarterly_only_return_pct": quarterly_only_ret,
        "n_snapshots": len(snapshot_dates),
    }


def compute_timeline_attribution(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Compute entry/exit timeline attribution: holding duration & rotation effects."""
    # Get all snapshot dates
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates or len(snapshot_dates) < 2:
        return {"error": "Insufficient snapshots"}

    # Get rebalance dates for both policies
    weekly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    while current <= end_dt:
        days_ahead = 4 - current.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        friday = current + timedelta(days=days_ahead)
        if friday > end_dt:
            break
        check = friday
        while check <= end_dt:
            date_str = check.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not weekly_dates or date_str != weekly_dates[-1]:
                    weekly_dates.append(date_str)
                break
            check += timedelta(days=1)
        current = friday + timedelta(days=1)

    quarterly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    while current <= end_dt:
        month = current.month
        if month in [1, 2, 3]:
            quarter_start = current.replace(month=1, day=1)
        elif month in [4, 5, 6]:
            quarter_start = current.replace(month=4, day=1)
        elif month in [7, 8, 9]:
            quarter_start = current.replace(month=7, day=1)
        else:
            quarter_start = current.replace(month=10, day=1)

        check_date = quarter_start
        while check_date <= end_dt:
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not quarterly_dates or date_str != quarterly_dates[-1]:
                    quarterly_dates.append(date_str)
                break
            check_date += timedelta(days=1)

        if current.month <= 9:
            current = current.replace(month=current.month + 3, day=1)
        else:
            current = current.replace(year=current.year + 1, month=current.month - 9, day=1)

    if not weekly_dates or not quarterly_dates:
        return {"error": "Insufficient rebalance dates"}

    terminal_dt = datetime.strptime(terminal_date, "%Y-%m-%d")

    def build_holding_spells(rebalance_dates, get_holdings_fn):
        """Track holding spells with entry/exit boundaries."""
        ticker_spells = {}
        prev_holdings = set()
        prev_holdings_date = None

        for date in rebalance_dates:
            current_holdings = get_holdings_fn(date)

            # Detect exits (ticker in prev but not current)
            if prev_holdings_date is not None:
                exits = prev_holdings - current_holdings
                for ticker in exits:
                    if ticker in ticker_spells:
                        ticker_spells[ticker]["spells"][-1]["exit_date"] = prev_holdings_date

            # Detect entries (ticker in current but not prev)
            entries = current_holdings - prev_holdings
            for ticker in entries:
                if ticker not in ticker_spells:
                    ticker_spells[ticker] = {
                        "first_entry": date,
                        "spells": [],
                        "last_appearance": date,
                    }
                ticker_spells[ticker]["spells"].append({"entry_date": date, "exit_date": terminal_date})
                ticker_spells[ticker]["last_appearance"] = date

            # Ticker remained held: update last_appearance
            for ticker in current_holdings & prev_holdings:
                if ticker in ticker_spells:
                    ticker_spells[ticker]["last_appearance"] = date

            prev_holdings = current_holdings
            prev_holdings_date = date

        # Finalize spells: set exit to terminal if still held at end
        for ticker_data in ticker_spells.values():
            if ticker_data["spells"] and ticker_data["spells"][-1]["exit_date"] == terminal_date:
                pass  # Already terminal
            for spell in ticker_data["spells"]:
                if spell["exit_date"] == terminal_date:
                    spell["days"] = (terminal_dt - datetime.strptime(spell["entry_date"], "%Y-%m-%d")).days
                else:
                    spell["days"] = (
                        datetime.strptime(spell["exit_date"], "%Y-%m-%d")
                        - datetime.strptime(spell["entry_date"], "%Y-%m-%d")
                    ).days

        # Compute aggregates per ticker
        for ticker in ticker_spells:
            spells = ticker_spells[ticker]["spells"]
            ticker_spells[ticker]["total_exposure_days"] = sum(s["days"] for s in spells)
            ticker_spells[ticker]["num_spells"] = len(spells)
            ticker_spells[ticker]["final_spell_days"] = spells[-1]["days"] if spells else 0

        return ticker_spells

    # Build spell-based timelines
    ticker_timeline_weekly = build_holding_spells(weekly_dates, get_top_30_tickers)
    ticker_timeline_quarterly = build_holding_spells(quarterly_dates, get_top_30_tickers)

    # Compare timelines at spell level
    all_tickers = set(ticker_timeline_weekly.keys()) | set(ticker_timeline_quarterly.keys())
    timeline_spells = []

    for ticker in all_tickers:
        w_data = ticker_timeline_weekly.get(ticker, {})
        q_data = ticker_timeline_quarterly.get(ticker, {})

        w_first_entry = w_data.get("first_entry", terminal_date)
        w_total_days = w_data.get("total_exposure_days", 0)
        w_num_spells = w_data.get("num_spells", 0)
        w_spells = w_data.get("spells", [])

        q_first_entry = q_data.get("first_entry", terminal_date)
        q_total_days = q_data.get("total_exposure_days", 0)
        q_num_spells = q_data.get("num_spells", 0)
        q_spells = q_data.get("spells", [])

        # Determine exposure relationship
        if q_total_days > w_total_days:
            exposure_relationship = "quarterly_more_days"
        elif w_total_days > q_total_days:
            exposure_relationship = "weekly_more_days"
        else:
            exposure_relationship = "equal_days"

        # Determine entry timing
        if q_first_entry < w_first_entry and q_first_entry != terminal_date:
            entry_timing = "quarterly_earlier"
        elif w_first_entry < q_first_entry and w_first_entry != terminal_date:
            entry_timing = "weekly_earlier"
        else:
            entry_timing = "same_or_absent"

        timeline_spells.append(
            {
                "ticker": ticker,
                "quarterly_first_entry": q_first_entry,
                "quarterly_total_exposure_days": q_total_days,
                "quarterly_num_spells": q_num_spells,
                "quarterly_spells": q_spells,
                "weekly_first_entry": w_first_entry,
                "weekly_total_exposure_days": w_total_days,
                "weekly_num_spells": w_num_spells,
                "weekly_spells": w_spells,
                "exposure_difference_days": q_total_days - w_total_days,
                "exposure_relationship": exposure_relationship,
                "entry_timing": entry_timing,
            }
        )

    timeline_spells.sort(key=lambda x: x["exposure_difference_days"], reverse=True)
    more_days_quarterly = [t for t in timeline_spells if t["exposure_relationship"] == "quarterly_more_days"][:10]
    more_days_weekly = [t for t in timeline_spells if t["exposure_relationship"] == "weekly_more_days"][:10]

    return {
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "weekly_rebalances": len(weekly_dates),
        "quarterly_rebalances": len(quarterly_dates),
        "n_tickers_analyzed": len(all_tickers),
        "more_exposure_days_quarterly": more_days_quarterly,
        "more_exposure_days_weekly": more_days_weekly,
        "exposure_days_comparison": {
            "quarterly_avg_total_exposure_days": round(
                sum(t["total_exposure_days"] for t in ticker_timeline_quarterly.values())
                / max(len(ticker_timeline_quarterly), 1),
                1,
            ),
            "weekly_avg_total_exposure_days": round(
                sum(t["total_exposure_days"] for t in ticker_timeline_weekly.values())
                / max(len(ticker_timeline_weekly), 1),
                1,
            ),
        },
    }


def compute_exposure_weighted_attribution(inception_date: str, terminal_date: str, prices: Dict) -> Dict[str, Any]:
    """Compute exposure-weighted attribution: per-ticker contribution by policy."""
    # Get all snapshot dates
    snapshot_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end = datetime.strptime(terminal_date, "%Y-%m-%d")
    snapshots_root = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    while current <= end:
        snap_dir = snapshots_root / current.strftime("%Y-%m-%d")
        if snap_dir.exists():
            snapshot_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not snapshot_dates or len(snapshot_dates) < 2:
        return {"error": "Insufficient snapshots"}

    # Get weekly rebalance dates (Friday-based)
    weekly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    end_dt = datetime.strptime(terminal_date, "%Y-%m-%d")
    while current <= end_dt:
        days_ahead = 4 - current.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        friday = current + timedelta(days=days_ahead)
        if friday > end_dt:
            break
        check = friday
        while check <= end_dt:
            date_str = check.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not weekly_dates or date_str != weekly_dates[-1]:
                    weekly_dates.append(date_str)
                break
            check += timedelta(days=1)
        current = friday + timedelta(days=1)

    # Get quarterly rebalance dates
    quarterly_dates = []
    current = datetime.strptime(inception_date, "%Y-%m-%d")
    while current <= end_dt:
        month = current.month
        if month in [1, 2, 3]:
            quarter_start = current.replace(month=1, day=1)
        elif month in [4, 5, 6]:
            quarter_start = current.replace(month=4, day=1)
        elif month in [7, 8, 9]:
            quarter_start = current.replace(month=7, day=1)
        else:
            quarter_start = current.replace(month=10, day=1)

        check_date = quarter_start
        while check_date <= end_dt:
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in snapshot_dates:
                if not quarterly_dates or date_str != quarterly_dates[-1]:
                    quarterly_dates.append(date_str)
                break
            check_date += timedelta(days=1)

        if current.month <= 9:
            current = current.replace(month=current.month + 3, day=1)
        else:
            current = current.replace(year=current.year + 1, month=current.month - 9, day=1)

    if not weekly_dates or not quarterly_dates:
        return {"error": "Insufficient rebalance dates"}

    # Compute per-ticker contribution for each policy
    ticker_contribution_weekly = {}
    ticker_contribution_quarterly = {}

    # Weekly contributions
    for i in range(len(weekly_dates) - 1):
        start_date = weekly_dates[i]
        end_date = weekly_dates[i + 1]
        holdings = get_top_30_tickers(start_date)
        if not holdings:
            continue
        weight = 1.0 / len(holdings)
        for ticker in holdings:
            ret, _, _ = compute_return(start_date, end_date, {ticker}, prices)
            if ret is not None:
                contribution = weight * ret
                ticker_contribution_weekly[ticker] = ticker_contribution_weekly.get(ticker, 0.0) + contribution

    # Quarterly contributions
    for i in range(len(quarterly_dates) - 1):
        start_date = quarterly_dates[i]
        end_date = quarterly_dates[i + 1]
        holdings = get_top_30_tickers(start_date)
        if not holdings:
            continue
        weight = 1.0 / len(holdings)
        for ticker in holdings:
            ret, _, _ = compute_return(start_date, end_date, {ticker}, prices)
            if ret is not None:
                contribution = weight * ret
                ticker_contribution_quarterly[ticker] = ticker_contribution_quarterly.get(ticker, 0.0) + contribution

    # Compute total contributions
    total_weekly = sum(ticker_contribution_weekly.values())
    total_quarterly = sum(ticker_contribution_quarterly.values())
    gap = total_quarterly - total_weekly

    # Find top contributors and detractors
    all_tickers = set(ticker_contribution_weekly.keys()) | set(ticker_contribution_quarterly.keys())
    gap_by_ticker = []
    for ticker in all_tickers:
        w_contrib = ticker_contribution_weekly.get(ticker, 0.0)
        q_contrib = ticker_contribution_quarterly.get(ticker, 0.0)
        diff = q_contrib - w_contrib
        gap_by_ticker.append(
            {
                "ticker": ticker,
                "weekly_contribution": round(w_contrib * 100, 2),
                "quarterly_contribution": round(q_contrib * 100, 2),
                "gap": round(diff * 100, 2),
            }
        )

    gap_by_ticker.sort(key=lambda x: x["gap"], reverse=True)
    top_helpers = gap_by_ticker[:10]
    top_hurters = gap_by_ticker[-10:]

    return {
        "inception_date": inception_date,
        "terminal_date": terminal_date,
        "weekly_rebalances": len(weekly_dates),
        "quarterly_rebalances": len(quarterly_dates),
        "weekly_total_contribution_pct": round(total_weekly * 100, 2),
        "quarterly_total_contribution_pct": round(total_quarterly * 100, 2),
        "gap_pct": round(gap * 100, 2),
        "n_tickers_analyzed": len(all_tickers),
        "top_10_helping_quarterly": top_helpers,
        "top_10_hurting_quarterly": top_hurters,
    }


def main():
    """Run Phase 1 diagnostic."""
    log = setup_logging()
    log.info("=== Phase 1 Portfolio Policy Diagnostic ===")
    log.info(f"Canonical cohorts: {len(CANONICAL_COHORTS)}")
    log.info(f"Terminal: {TERMINAL_DATE}")

    DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)

    all_snapshots = get_all_snapshots_between(CANONICAL_COHORTS[0], TERMINAL_DATE)
    log.info(f"Found {len(all_snapshots)} snapshots")

    results = []
    for cohort in CANONICAL_COHORTS:
        log.info(f"\nCohort {cohort}...")

        # Load prices for this cohort
        top_30 = get_top_30_tickers(cohort)
        if not top_30:
            log.warning("  No top-30 found")
            continue

        log.info(f"  Loading prices for {len(top_30)} tickers...")
        prices = load_prices_dict(cohort, TERMINAL_DATE, top_30)
        log.info(f"  Loaded {len(prices)} price points")

        # Test policies using calendar dates, not snapshots
        static_inception = test_static_inception_hold_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in static_inception:
            results.append(static_inception)
            log.info(f"    STATIC_INCEPTION_HOLD: {static_inception['return_pct']}%")

        delisting = test_delisting_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in delisting:
            results.append(delisting)
            log.info(f"    DELISTING: {delisting['return_pct']}%")

        snapshot_rebuild = test_available_snapshot_rebuild_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in snapshot_rebuild:
            results.append(snapshot_rebuild)
            log.info(
                f"    AVAILABLE_SNAPSHOT_REBUILD: {snapshot_rebuild['return_pct']}% (turnover={snapshot_rebuild['turnover']}, snapshots={snapshot_rebuild['n_snapshots']})"
            )

        weekly_proxy = test_weekly_trade_packet_proxy_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in weekly_proxy:
            results.append(weekly_proxy)
            log.info(
                f"    WEEKLY_TRADE_PACKET_PROXY: {weekly_proxy['return_pct']}% (turnover={weekly_proxy['turnover']}, rebalances={weekly_proxy['n_rebalances']})"
            )

        monthly_proxy = test_monthly_rebalance_proxy_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in monthly_proxy:
            results.append(monthly_proxy)
            log.info(
                f"    MONTHLY_REBALANCE_PROXY: {monthly_proxy['return_pct']}% (turnover={monthly_proxy['turnover']}, rebalances={monthly_proxy['n_rebalances']})"
            )

        hold_30 = test_hold_30_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in hold_30:
            results.append(hold_30)
            log.info(
                f"    HOLD_30: {hold_30['return_pct']}% (turnover={hold_30['turnover']}, replacements={hold_30['n_replacements']})"
            )

        quarterly_proxy = test_quarterly_rebalance_proxy_policy(cohort, TERMINAL_DATE, prices)
        if "error" not in quarterly_proxy:
            results.append(quarterly_proxy)
            log.info(
                f"    QUARTERLY_REBALANCE_PROXY: {quarterly_proxy['return_pct']}% (turnover={quarterly_proxy['turnover']}, rebalances={quarterly_proxy['n_rebalances']})"
            )

    # Compute attribution: quarterly vs weekly
    log.info("\n=== Attribution Analysis: Quarterly vs Weekly ===")
    attribution_results = []
    for cohort in CANONICAL_COHORTS:
        top_30 = get_top_30_tickers(cohort)
        if not top_30:
            continue
        prices = load_prices_dict(cohort, TERMINAL_DATE, top_30)
        if not prices:
            continue
        attr = compute_quarterly_vs_weekly_attribution(cohort, TERMINAL_DATE, prices)
        if "error" not in attr:
            attribution_results.append(attr)
            log.info(
                f"  {cohort}: W={attr['weekly_rebalances']} Q={attr['quarterly_rebalances']} "
                f"(shared_ret={attr['shared_holdings_return_pct']}% "
                f"weekly_only={attr['weekly_only_return_pct']}% "
                f"quarterly_only={attr['quarterly_only_return_pct']}%)"
            )

    # Compute exposure-weighted attribution: quarterly vs weekly
    log.info("\n=== Exposure-Weighted Attribution Analysis ===")
    exposure_attr_results = []
    for cohort in CANONICAL_COHORTS:
        top_30 = get_top_30_tickers(cohort)
        if not top_30:
            continue
        prices = load_prices_dict(cohort, TERMINAL_DATE, top_30)
        if not prices:
            continue
        exp_attr = compute_exposure_weighted_attribution(cohort, TERMINAL_DATE, prices)
        if "error" not in exp_attr:
            exposure_attr_results.append(exp_attr)
            log.info(
                f"  {cohort}: W={exp_attr['weekly_total_contribution_pct']:.2f}% "
                f"Q={exp_attr['quarterly_total_contribution_pct']:.2f}% "
                f"Gap={exp_attr['gap_pct']:.2f}%"
            )

    # Compute timeline attribution: quarterly vs weekly
    log.info("\n=== Timeline Attribution Analysis ===")
    timeline_attr_results = []
    for cohort in CANONICAL_COHORTS:
        top_30 = get_top_30_tickers(cohort)
        if not top_30:
            continue
        prices = load_prices_dict(cohort, TERMINAL_DATE, top_30)
        if not prices:
            continue
        timeline_attr = compute_timeline_attribution(cohort, TERMINAL_DATE, prices)
        if "error" not in timeline_attr:
            timeline_attr_results.append(timeline_attr)
            q_avg_days = timeline_attr["exposure_days_comparison"]["quarterly_avg_total_exposure_days"]
            w_avg_days = timeline_attr["exposure_days_comparison"]["weekly_avg_total_exposure_days"]
            log.info(
                f"  {cohort}: Q_avg_exposure={q_avg_days}d W_avg_exposure={w_avg_days}d "
                f"(more_q={len(timeline_attr['more_exposure_days_quarterly'])} more_w={len(timeline_attr['more_exposure_days_weekly'])})"
            )

    # Write results
    output_file = DIAGNOSTIC_ROOT / "canonical_cohorts.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✓ Results: {output_file}")

    # Write attribution analysis
    attr_file = DIAGNOSTIC_ROOT / "quarterly_vs_weekly_attribution.json"
    with attr_file.open("w") as f:
        json.dump(attribution_results, f, indent=2)
    log.info(f"✓ Attribution (set-based): {attr_file}")

    # Write exposure-weighted attribution
    exp_attr_file = DIAGNOSTIC_ROOT / "quarterly_vs_weekly_exposure_attribution.json"
    with exp_attr_file.open("w") as f:
        json.dump(exposure_attr_results, f, indent=2)
    log.info(f"✓ Attribution (exposure-weighted): {exp_attr_file}")

    # Write timeline attribution
    timeline_attr_file = DIAGNOSTIC_ROOT / "quarterly_vs_weekly_timeline_attribution.json"
    with timeline_attr_file.open("w") as f:
        json.dump(timeline_attr_results, f, indent=2)
    log.info(f"✓ Attribution (timeline): {timeline_attr_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
