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
from datetime import date as _date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse IC computation from eval_forward_returns (import, not duplicate)
from scripts.eval_forward_returns import spearman_ic

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
    min_gap_days: int = 0,
) -> List[str]:
    """Find up to lookback_n snapshot dates with PIT caches having horizon filled.

    Returns dates sorted descending (most recent first), all < before_date.

    When ``min_gap_days > 0`` the dates are de-overlapped: walking from the most
    recent, a candidate is kept only if it is at least ``min_gap_days`` calendar
    days before the previously kept date. This prevents near-adjacent snapshots
    (whose forward-return windows overlap almost entirely) from being counted as
    independent observations — the flat mean over the window would otherwise
    double-count a single market episode. Callers pass a gap derived from the
    horizon (≈ horizon trading days) so the retained windows are non-overlapping.
    """
    if not cache_base.exists():
        return []

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    # Gather all filled candidates first (descending), then de-overlap greedily.
    filled: List[str] = []
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
            filled.append(entry.name)

    selected: List[str] = []
    last_kept: Optional[str] = None
    for d in filled:  # descending (most recent first)
        if min_gap_days > 0 and last_kept is not None:
            gap = (_date.fromisoformat(last_kept) - _date.fromisoformat(d)).days
            if gap < min_gap_days:
                continue
        selected.append(d)
        last_kept = d
        if len(selected) >= lookback_n:
            break

    return selected


def _load_pit_prices(cache_dir: Path) -> List[Dict[str, str]]:
    """Load prices.csv from a PIT cache dir."""
    prices_path = cache_dir / "prices.csv"
    if not prices_path.exists():
        return []
    with open(prices_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_truthy(val: Any) -> bool:
    """Parse a CSV cell as a boolean (handles True/1/yes and float-ish '1.0')."""
    s = str(val).strip().lower()
    return s in ("true", "1", "1.0", "yes", "t", "y")


def _load_rankings_signal(
    snapshot_dir: Path,
    scope: str = "universe",
    top_k: int = 20,
) -> Dict[str, float]:
    """Load rankings.csv and return {ticker: negative_rank} signal.

    Lower actionable_rank = better → higher signal value (negated).

    ``scope`` restricts which names enter the IC — the gate should measure the
    model we actually trade, not the full universe:

      - ``"universe"``  — every ranked row (legacy behaviour).
      - ``"eligible"``  — only rows with ``eligible`` truthy. Excludes names the
        decision engine gates out (distressed/ineligible), which otherwise
        pollute the IC: they are correctly not held, yet a full-universe rank IC
        penalises the model when such names bounce, and lets a distressed name
        that floated to a high composite rank dominate the score.
      - ``"portfolio"`` — only rows with ``target_weight_pct > 0``.
      - ``"top_k"``     — only rows with rank <= ``top_k``.

    If the column a scope needs is absent from rankings.csv, we fall back to
    ``"universe"`` (with a warning) rather than silently dropping every row —
    a malformed file must not quietly blank the gate.
    """
    rankings_path = snapshot_dir / "rankings.csv"
    if not rankings_path.exists():
        return {}

    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    effective_scope = scope
    if scope == "eligible" and "eligible" not in fieldnames:
        logger.warning("forward_eval: scope='eligible' but no 'eligible' column — falling back to universe")
        effective_scope = "universe"
    elif scope == "portfolio" and "target_weight_pct" not in fieldnames:
        logger.warning("forward_eval: scope='portfolio' but no 'target_weight_pct' column — falling back to universe")
        effective_scope = "universe"

    signal: Dict[str, float] = {}
    for row in rows:
        ticker = (row.get("ticker") or "").strip().upper()
        rank_str = (row.get("actionable_rank") or row.get("composite_rank") or "").strip()
        if not ticker or not rank_str:
            continue
        try:
            rank_val = float(rank_str)
        except (ValueError, TypeError):
            continue

        if effective_scope == "eligible":
            if not _is_truthy(row.get("eligible", "")):
                continue
        elif effective_scope == "portfolio":
            try:
                weight = float(row.get("target_weight_pct") or 0)
            except (ValueError, TypeError):
                weight = 0.0
            if weight <= 0:
                continue
        elif effective_scope == "top_k":
            if rank_val > top_k:
                continue

        signal[ticker] = -rank_val
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

    return {w["ticker"] for w in index.get("split_warnings", []) if w.get("horizon") == horizon}


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
    scope: str = "eligible",
    min_gap_days: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Rolling-window IC from PIT price caches.

    Returns (status, detail, value, threshold).
    Status is always "PASS" or "WARN" — never "FAIL".

    ``scope`` (default ``"eligible"``) restricts the IC to the tradeable set so
    the gate measures the model actually run, not full-universe monotonicity —
    see ``_load_rankings_signal``. ``min_gap_days`` de-overlaps the rolling
    window; when None it defaults to ≈ ``horizon`` trading days
    (``round(horizon * 7 / 5)`` calendar days) so retained forward-return
    windows do not overlap and one market episode is not counted many times.
    """
    gap_days = min_gap_days if min_gap_days is not None else int(round(horizon * 7 / 5))

    thresholds = {
        "ic_warn_floor": ic_warn_floor,
        "lookback_n": lookback_n,
        "min_dates": min_dates,
        "horizon": horizon,
        "scope": scope,
        "min_gap_days": gap_days,
    }

    # Discover dates with filled PIT caches (de-overlapped by gap_days)
    pit_dates = _discover_pit_dates(price_cache_base, current_date, horizon, lookback_n, min_gap_days=gap_days)

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

        # Load signal from rankings (scoped to the tradeable set)
        signal = _load_rankings_signal(snap_dir, scope=scope, top_k=top_k)
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
            date_details.append(
                {
                    "date": snap_date,
                    "ic": None,
                    "reason": f"insufficient_overlap ({len(common_tickers)})",
                }
            )
            continue

        signal_vals = [signal[t] for t in common_tickers]
        return_vals = [fwd_rets[t] for t in common_tickers]
        ic = spearman_ic(signal_vals, return_vals)

        if ic is not None:
            ics.append(ic)

        date_details.append(
            {
                "date": snap_date,
                "ic": round(ic, 4) if ic is not None else None,
                "n_tickers": len(common_tickers),
                "n_split_excluded": len(split_tickers),
            }
        )

    n_evaluated = len(ics)
    value: Dict[str, Any] = {
        "n_evaluated": n_evaluated,
        "n_pit_dates": len(pit_dates),
        "date_details": date_details,
        "scope": scope,
        "min_gap_days": gap_days,
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
            f"insufficient dates: {n_evaluated} < {min_dates} " f"(mean_ic={mean_ic:.4f})",
            value,
            thresholds,
        )

    if mean_ic < ic_warn_floor:
        return (
            "WARN",
            f"mean_ic={mean_ic:.4f} < floor={ic_warn_floor:.4f} " f"(n={n_evaluated}, horizon={horizon}d)",
            value,
            thresholds,
        )

    return (
        "PASS",
        f"mean_ic={mean_ic:.4f}, median_ic={median_ic:.4f} " f"(n={n_evaluated}, horizon={horizon}d)",
        value,
        thresholds,
    )
