#!/usr/bin/env python3
"""Phase 1 Portfolio Policy Diagnostic Harness (Optimized)

Read-only diagnostic harness for testing portfolio transition policies.
Optimized with efficient price lookups and targeted loading.

Canonical cohorts (7 inception dates, common terminal 2026-05-27):
  2024-10-18, 2024-11-01, 2025-01-10, 2025-04-11, 2025-07-18, 2025-10-10, 2026-01-02

Policies tested:
  - STATIC_INCEPTION_HOLD: Static hold of inception-date top-30 through terminal
  - DELISTING_ONLY: Hold inception top-30 until delisting detected
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

    # Write results
    output_file = DIAGNOSTIC_ROOT / "canonical_cohorts.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✓ Results: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
