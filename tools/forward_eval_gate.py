#!/usr/bin/env python3
"""
forward_eval_gate.py — Rolling-window IC evaluation using PIT-frozen prices.

Computes Spearman IC between ranking signal and PIT-cached forward returns
over a rolling window of recent snapshots. Designed as a WARN-only gate
for run_daily_production.py.

Never FAIL — always returns PASS or WARN.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse IC computation from eval_forward_returns (import, not duplicate)
from scripts.eval_forward_returns import spearman_ic, _avg_ranks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("forward_eval_gate")


# ---------------------------------------------------------------------------
# PIT cache discovery
# ---------------------------------------------------------------------------

def _discover_pit_dates(
    cache_base: Path,
    before_date: str,
    horizon: int,
    lookback_n: int,
) -> List[str]:
    """Find up to lookback_n snapshot dates with PIT caches having horizon filled.

    Returns dates sorted descending (most recent first), all < before_date.
    """
    if not cache_base.exists():
        return []

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    candidates: List[str] = []

    for entry in sorted(cache_base.iterdir(), reverse=True):
        if not entry.is_dir() or not date_re.match(entry.name):
            continue
        if entry.name >= before_date:
            continue

        index_path = entry / "index.json"
        if not index_path.exists():
            continue

        try:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if horizon in index.get("horizons_filled", []):
            candidates.append(entry.name)
            if len(candidates) >= lookback_n:
                break

    return candidates


def _load_pit_prices(cache_dir: Path) -> List[Dict[str, str]]:
    """Load prices.csv from a PIT cache dir."""
    prices_path = cache_dir / "prices.csv"
    if not prices_path.exists():
        return []
    with open(prices_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_rankings_signal(snapshot_dir: Path) -> Dict[str, float]:
    """Load rankings.csv and return {ticker: negative_rank} signal.

    Lower actionable_rank = better → higher signal value (negated).
    """
    rankings_path = snapshot_dir / "rankings.csv"
    if not rankings_path.exists():
        return {}

    signal: Dict[str, float] = {}
    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            rank_str = (row.get("actionable_rank") or row.get("composite_rank") or "").strip()
            if not ticker or not rank_str:
                continue
            try:
                signal[ticker] = -float(rank_str)
            except (ValueError, TypeError):
                continue
    return signal


def _get_split_warning_tickers(cache_dir: Path, horizon: int) -> set:
    """Get tickers with split warnings for a given horizon."""
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        return set()
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()

    return {
        w["ticker"]
        for w in index.get("split_warnings", [])
        if w.get("horizon") == horizon
    }


# ---------------------------------------------------------------------------
# Core: evaluate_rolling_ic
# ---------------------------------------------------------------------------

def evaluate_rolling_ic(
    snapshot_dir: Path,
    price_cache_base: Path,
    current_date: str,
    *,
    horizon: int = 20,
    lookback_n: int = 10,
    min_dates: int = 3,
    top_k: int = 20,
    ic_warn_floor: float = 0.00,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Rolling-window IC from PIT price caches.

    Returns (status, detail, value, threshold).
    Status is always "PASS" or "WARN" — never "FAIL".
    """
    thresholds = {
        "ic_warn_floor": ic_warn_floor,
        "lookback_n": lookback_n,
        "min_dates": min_dates,
        "horizon": horizon,
    }

    # Discover dates with filled PIT caches
    pit_dates = _discover_pit_dates(price_cache_base, current_date, horizon, lookback_n)

    if not pit_dates:
        return (
            "PASS",
            "cold-start: no PIT price caches with filled horizons",
            {"n_evaluated": 0, "pit_dates": []},
            thresholds,
        )

    # Compute IC for each date
    ics: List[float] = []
    date_details: List[Dict[str, Any]] = []

    for snap_date in pit_dates:
        cache_dir = price_cache_base / snap_date
        snap_dir = snapshot_dir / snap_date

        # Load signal from rankings
        signal = _load_rankings_signal(snap_dir)
        if not signal:
            date_details.append({"date": snap_date, "ic": None, "reason": "no_rankings"})
            continue

        # Load PIT prices
        pit_rows = _load_pit_prices(cache_dir)
        if not pit_rows:
            date_details.append({"date": snap_date, "ic": None, "reason": "no_pit_prices"})
            continue

        # Exclude split-warning tickers
        split_tickers = _get_split_warning_tickers(cache_dir, horizon)

        # Compute forward returns from PIT prices
        close_col = f"h{horizon}_close"
        fwd_rets: Dict[str, float] = {}
        for row in pit_rows:
            ticker = (row.get("ticker") or "").strip().upper()
            if ticker in split_tickers:
                continue
            anchor_str = (row.get("anchor_close") or "").strip()
            fwd_str = (row.get(close_col) or "").strip()
            if not anchor_str or not fwd_str:
                continue
            try:
                anchor = float(anchor_str)
                fwd = float(fwd_str)
            except (ValueError, TypeError):
                continue
            if anchor <= 0:
                continue
            fwd_rets[ticker] = fwd / anchor - 1.0

        # Compute Spearman IC
        common_tickers = [t for t in signal if t in fwd_rets]
        if len(common_tickers) < 3:
            date_details.append({
                "date": snap_date, "ic": None,
                "reason": f"insufficient_overlap ({len(common_tickers)})",
            })
            continue

        signal_vals = [signal[t] for t in common_tickers]
        return_vals = [fwd_rets[t] for t in common_tickers]
        ic = spearman_ic(signal_vals, return_vals)

        if ic is not None:
            ics.append(ic)

        date_details.append({
            "date": snap_date,
            "ic": round(ic, 4) if ic is not None else None,
            "n_tickers": len(common_tickers),
            "n_split_excluded": len(split_tickers),
        })

    n_evaluated = len(ics)
    value: Dict[str, Any] = {
        "n_evaluated": n_evaluated,
        "n_pit_dates": len(pit_dates),
        "date_details": date_details,
    }

    if n_evaluated == 0:
        return (
            "PASS",
            "cold-start: insufficient PIT price caches",
            value,
            thresholds,
        )

    mean_ic = sum(ics) / len(ics)
    median_ic = sorted(ics)[len(ics) // 2]
    value["mean_ic"] = round(mean_ic, 4)
    value["median_ic"] = round(median_ic, 4)
    value["ics"] = [round(x, 4) for x in ics]

    if n_evaluated < min_dates:
        return (
            "WARN",
            f"insufficient dates: {n_evaluated} < {min_dates} "
            f"(mean_ic={mean_ic:.4f})",
            value,
            thresholds,
        )

    if mean_ic < ic_warn_floor:
        return (
            "WARN",
            f"mean_ic={mean_ic:.4f} < floor={ic_warn_floor:.4f} "
            f"(n={n_evaluated}, horizon={horizon}d)",
            value,
            thresholds,
        )

    return (
        "PASS",
        f"mean_ic={mean_ic:.4f}, median_ic={median_ic:.4f} "
        f"(n={n_evaluated}, horizon={horizon}d)",
        value,
        thresholds,
    )
