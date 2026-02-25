#!/usr/bin/env python3
"""Portfolio-realistic backtest: turnover, costs, net returns, regime slices.

Builds equal-weight top-K (and optional bottom-K short) portfolios from
ranked snapshot data, computes forward returns from price_history.csv,
applies transaction-cost haircuts, and slices results by XBI regime.

Reuses infrastructure from ``eval_forward_returns.py`` (discovery, price
loading, turnover, net return) and ``backtest/regime.py`` (regime labels).

Outputs: portfolio_timeseries.csv + summary.json → ``--out-dir``
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_forward_returns import (
    compute_forward_return,
    compute_turnover,
    discover_snapshot_dates,
    load_price_series,
    load_rankings,
    net_return,
    resolve_trade_date,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HORIZON = 20
DEFAULT_TOP_K = 20
DEFAULT_COST_BPS = 30
DEFAULT_MIN_COVERAGE = 0.50
DEFAULT_ANCHOR_MODE = "next_trading_day"

# Supported signals and their CSV column + sort direction (True = higher is better)
SIGNAL_COLUMNS: Dict[str, Tuple[str, bool]] = {
    "catalyst": ("catalyst_decay_w", True),
    "clinical": ("clinical_score_z_tier", True),
    "alpha": ("alpha_cohort_raw", True),
    "composite": ("composite_score", True),
    "actionable_rank": ("actionable_rank", False),  # lower rank = better
    "alpha_incr": ("alpha_cohort_raw", True),  # alias for incremental alpha
}


def _safe_float(v: Any, default: float = float("nan")) -> float:
    """Parse a value to float, returning *default* on failure."""
    if v is None or v == "":
        return default
    try:
        f = float(v)
        return f if not math.isnan(f) else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def extract_signal(
    rankings: List[Dict[str, str]],
    signal: str,
) -> List[Tuple[str, float]]:
    """Extract (ticker, signal_value) pairs for eligible rows.

    Returns list sorted by signal value descending (best first) for
    higher-is-better signals, ascending for lower-is-better.
    """
    col, higher_better = SIGNAL_COLUMNS[signal]
    pairs: List[Tuple[str, float]] = []
    for r in rankings:
        if r.get("eligible") != "1":
            continue
        ticker = r.get("ticker", "")
        if not ticker:
            continue
        val = _safe_float(r.get(col, ""))
        if math.isnan(val):
            continue
        pairs.append((ticker, val))
    pairs.sort(key=lambda x: x[1], reverse=higher_better)
    return pairs


def select_top_k(
    signal_pairs: List[Tuple[str, float]],
    k: int,
) -> List[str]:
    """Select top-K tickers from signal-sorted pairs."""
    return [t for t, _ in signal_pairs[:k]]


def select_bottom_k(
    signal_pairs: List[Tuple[str, float]],
    k: int,
) -> List[str]:
    """Select bottom-K tickers from signal-sorted pairs (worst signal)."""
    return [t for t, _ in signal_pairs[-k:]] if len(signal_pairs) >= k else []


# ---------------------------------------------------------------------------
# Per-date evaluation
# ---------------------------------------------------------------------------

@dataclass
class DateRow:
    """One row in portfolio_timeseries.csv."""
    date: str
    trade_date: str
    regime: str = ""
    long_gross: Optional[float] = None
    long_net: Optional[float] = None
    long_turnover: float = 0.0
    long_n_held: int = 0
    short_gross: Optional[float] = None
    short_net: Optional[float] = None
    short_turnover: float = 0.0
    short_n_held: int = 0
    ls_gross: Optional[float] = None
    ls_net: Optional[float] = None
    n_signal: int = 0
    coverage: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


def _portfolio_gross(
    tickers: List[str],
    fwd_rets: Dict[str, float],
) -> Tuple[Optional[float], int]:
    """Equal-weight gross return for a set of tickers."""
    held = [t for t in tickers if t in fwd_rets]
    if not held:
        return None, 0
    return statistics.mean(fwd_rets[t] for t in held), len(held)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_signal_portfolio(
    snapshot_root: Path,
    price_csv: Path,
    signal: str,
    horizon: int = DEFAULT_HORIZON,
    top_k: int = DEFAULT_TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    anchor_mode: str = DEFAULT_ANCHOR_MODE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    long_short: bool = False,
    regime_price_csv: Optional[Path] = None,
) -> Tuple[List[DateRow], Dict[str, Any]]:
    """Run portfolio backtest for a single signal at one horizon.

    Returns (date_rows, summary_dict).
    """
    if signal not in SIGNAL_COLUMNS:
        raise ValueError(f"Unknown signal '{signal}'. Choose from: {list(SIGNAL_COLUMNS)}")

    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    prices = load_price_series(price_csv)

    # Build sorted trading dates
    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)

    # Regime labels (optional)
    regime_map: Dict[str, str] = {}
    if regime_price_csv is not None or price_csv.exists():
        try:
            import pandas as pd
            from backtest.regime import (
                assign_regime_to_rebalance_dates,
                compute_regime_series,
                load_price_history,
            )
            rpcsv = regime_price_csv or price_csv
            pdf = load_price_history(rpcsv)
            regime_df = compute_regime_series(pdf, market_ticker="XBI")
            regime_map = assign_regime_to_rebalance_dates(regime_df, snap_dates)
        except Exception:
            pass  # regime slicing is optional

    date_rows: List[DateRow] = []
    prev_long: List[str] = []
    prev_short: List[str] = []

    # Accumulators
    long_gross_list: List[float] = []
    long_net_list: List[float] = []
    long_turn_list: List[float] = []
    short_gross_list: List[float] = []
    short_net_list: List[float] = []
    short_turn_list: List[float] = []
    ls_gross_list: List[float] = []
    ls_net_list: List[float] = []
    n_evaluated = 0
    n_skipped = 0

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date
        trade_date = resolve_trade_date(sorted_dates, snap_date, anchor_mode)
        if trade_date is None:
            row = DateRow(date=snap_date, trade_date="", skipped=True,
                          skip_reason="ANCHOR_NOT_FOUND")
            date_rows.append(row)
            n_skipped += 1
            continue

        rankings = load_rankings(snap_dir)
        if not rankings:
            row = DateRow(date=snap_date, trade_date=trade_date, skipped=True,
                          skip_reason="EMPTY_RANKINGS")
            date_rows.append(row)
            n_skipped += 1
            continue

        # Extract signal
        signal_pairs = extract_signal(rankings, signal)
        if len(signal_pairs) < top_k:
            row = DateRow(date=snap_date, trade_date=trade_date, skipped=True,
                          skip_reason=f"INSUFFICIENT_SIGNAL ({len(signal_pairs)}<{top_k})",
                          n_signal=len(signal_pairs))
            date_rows.append(row)
            n_skipped += 1
            continue

        # Forward returns for all tickers with prices
        fwd_rets: Dict[str, float] = {}
        for ticker, _ in signal_pairs:
            if ticker not in prices:
                continue
            ret = compute_forward_return(
                prices[ticker], sorted_dates, trade_date, horizon,
            )
            if ret is not None:
                fwd_rets[ticker] = ret

        coverage = len(fwd_rets) / max(len(signal_pairs), 1)
        if coverage < min_coverage:
            row = DateRow(date=snap_date, trade_date=trade_date, skipped=True,
                          skip_reason=f"LOW_COVERAGE ({coverage:.1%})",
                          n_signal=len(signal_pairs), coverage=round(coverage, 4))
            date_rows.append(row)
            n_skipped += 1
            continue

        # Long portfolio
        long_tickers = select_top_k(signal_pairs, top_k)
        long_gross, long_n = _portfolio_gross(long_tickers, fwd_rets)
        long_turn = compute_turnover(prev_long, long_tickers)
        prev_long = long_tickers

        long_net_val = None
        if long_gross is not None:
            long_net_val = net_return(long_gross, long_turn, cost_bps)
            long_gross_list.append(long_gross)
            long_net_list.append(long_net_val)
            long_turn_list.append(long_turn)

        # Short portfolio (optional)
        short_gross_val = None
        short_net_val = None
        short_turn = 0.0
        short_n = 0
        ls_gross_val = None
        ls_net_val = None

        if long_short:
            short_tickers = select_bottom_k(signal_pairs, top_k)
            short_gross_val, short_n = _portfolio_gross(short_tickers, fwd_rets)
            short_turn = compute_turnover(prev_short, short_tickers)
            prev_short = short_tickers

            if short_gross_val is not None:
                # Short PnL = -1 * short_return (we profit when shorts fall)
                short_net_val = net_return(-short_gross_val, short_turn, cost_bps)
                short_gross_list.append(-short_gross_val)
                short_net_list.append(short_net_val)
                short_turn_list.append(short_turn)

            if long_gross is not None and short_gross_val is not None:
                ls_gross_val = long_gross - short_gross_val
                ls_net_val = (long_net_val or 0.0) + (short_net_val or 0.0)
                ls_gross_list.append(ls_gross_val)
                ls_net_list.append(ls_net_val)

        n_evaluated += 1
        regime = regime_map.get(snap_date, "")

        row = DateRow(
            date=snap_date,
            trade_date=trade_date,
            regime=regime,
            long_gross=_round_opt(long_gross, 6),
            long_net=_round_opt(long_net_val, 6),
            long_turnover=round(long_turn, 4),
            long_n_held=long_n,
            short_gross=_round_opt(short_gross_val, 6) if long_short else None,
            short_net=_round_opt(short_net_val, 6) if long_short else None,
            short_turnover=round(short_turn, 4) if long_short else 0.0,
            short_n_held=short_n if long_short else 0,
            ls_gross=_round_opt(ls_gross_val, 6) if long_short else None,
            ls_net=_round_opt(ls_net_val, 6) if long_short else None,
            n_signal=len(signal_pairs),
            coverage=round(coverage, 4),
        )
        date_rows.append(row)

    # Build summary
    summary = _build_summary(
        signal=signal,
        horizon=horizon,
        top_k=top_k,
        cost_bps=cost_bps,
        anchor_mode=anchor_mode,
        long_short=long_short,
        n_dates=len(snap_dates),
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
        long_gross_list=long_gross_list,
        long_net_list=long_net_list,
        long_turn_list=long_turn_list,
        short_gross_list=short_gross_list,
        short_net_list=short_net_list,
        short_turn_list=short_turn_list,
        ls_gross_list=ls_gross_list,
        ls_net_list=ls_net_list,
        date_rows=[r for r in date_rows if not r.skipped],
        regime_map=regime_map,
    )

    return date_rows, summary


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _safe_mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _safe_median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _safe_std(vals: List[float]) -> Optional[float]:
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _cumulative(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    cum = 1.0
    for r in returns:
        cum *= (1.0 + r)
    return cum - 1.0


def _round_opt(v: Optional[float], d: int) -> Optional[float]:
    return round(v, d) if v is not None else None


def _build_summary(
    *,
    signal: str,
    horizon: int,
    top_k: int,
    cost_bps: float,
    anchor_mode: str,
    long_short: bool,
    n_dates: int,
    n_evaluated: int,
    n_skipped: int,
    long_gross_list: List[float],
    long_net_list: List[float],
    long_turn_list: List[float],
    short_gross_list: List[float],
    short_net_list: List[float],
    short_turn_list: List[float],
    ls_gross_list: List[float],
    ls_net_list: List[float],
    date_rows: List[DateRow],
    regime_map: Dict[str, str],
) -> Dict[str, Any]:
    """Build summary dict from accumulated results."""
    summary: Dict[str, Any] = {
        "signal": signal,
        "horizon": horizon,
        "top_k": top_k,
        "cost_bps": cost_bps,
        "anchor_mode": anchor_mode,
        "is_long_short": long_short,
        "n_dates": n_dates,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "long": {
            "mean_gross": _round_opt(_safe_mean(long_gross_list), 6),
            "mean_net": _round_opt(_safe_mean(long_net_list), 6),
            "cumulative_gross": _round_opt(_cumulative(long_gross_list), 6),
            "cumulative_net": _round_opt(_cumulative(long_net_list), 6),
            "mean_turnover": _round_opt(_safe_mean(long_turn_list), 4),
            "median_turnover": _round_opt(_safe_median(long_turn_list), 4),
        },
    }

    if long_short:
        summary["short"] = {
            "mean_gross": _round_opt(_safe_mean(short_gross_list), 6),
            "mean_net": _round_opt(_safe_mean(short_net_list), 6),
            "cumulative_gross": _round_opt(_cumulative(short_gross_list), 6),
            "cumulative_net": _round_opt(_cumulative(short_net_list), 6),
            "mean_turnover": _round_opt(_safe_mean(short_turn_list), 4),
        }
        summary["long_short"] = {
            "mean_gross": _round_opt(_safe_mean(ls_gross_list), 6),
            "mean_net": _round_opt(_safe_mean(ls_net_list), 6),
            "cumulative_gross": _round_opt(_cumulative(ls_gross_list), 6),
            "cumulative_net": _round_opt(_cumulative(ls_net_list), 6),
        }

    # Regime slices
    if regime_map and date_rows:
        regime_slices: Dict[str, Dict[str, Any]] = {}
        for regime in sorted(set(regime_map.values())):
            regime_rows = [r for r in date_rows if r.regime == regime]
            if not regime_rows:
                continue
            rg = [r.long_gross for r in regime_rows if r.long_gross is not None]
            rn = [r.long_net for r in regime_rows if r.long_net is not None]
            rt = [r.long_turnover for r in regime_rows]
            regime_slices[regime] = {
                "n_dates": len(regime_rows),
                "mean_gross": _round_opt(_safe_mean(rg), 6),
                "mean_net": _round_opt(_safe_mean(rn), 6),
                "cumulative_gross": _round_opt(_cumulative(rg), 6),
                "cumulative_net": _round_opt(_cumulative(rn), 6),
                "mean_turnover": _round_opt(_safe_mean(rt), 4),
            }
        summary["regime_slices"] = regime_slices

    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_timeseries_csv(date_rows: List[DateRow], out_dir: Path) -> Path:
    """Write portfolio_timeseries.csv."""
    path = out_dir / "portfolio_timeseries.csv"
    fieldnames = [
        "date", "trade_date", "regime",
        "long_gross", "long_net", "long_turnover", "long_n_held",
        "short_gross", "short_net", "short_turnover", "short_n_held",
        "ls_gross", "ls_net",
        "n_signal", "coverage", "skipped", "skip_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in date_rows:
            writer.writerow(asdict(row))
    return path


def write_summary_json(summary: Dict[str, Any], out_dir: Path) -> Path:
    """Write summary.json."""
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Portfolio-realistic signal backtest with turnover and costs",
    )
    parser.add_argument(
        "--signal", required=True,
        choices=list(SIGNAL_COLUMNS.keys()),
        help="Signal column to rank by",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help=f"Forward-return horizon in trading days (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K,
        help=f"Portfolio size (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=DEFAULT_COST_BPS,
        help=f"One-way transaction cost in bps (default: {DEFAULT_COST_BPS})",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
        help="Root directory for snapshot dates",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--anchor-mode", default=DEFAULT_ANCHOR_MODE,
        choices=["exact", "next_trading_day", "prev_trading_day"],
        help=f"Trade date anchor mode (default: {DEFAULT_ANCHOR_MODE})",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
        help=f"Minimum price coverage fraction (default: {DEFAULT_MIN_COVERAGE})",
    )
    parser.add_argument(
        "--long-short", action="store_true",
        help="Include short leg (bottom-K)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "signal_portfolios",
        help="Output directory",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Signal: {args.signal}  Horizon: {args.horizon}d  K: {args.k}  "
          f"Cost: {args.cost_bps} bps  L/S: {args.long_short}")

    date_rows, summary = evaluate_signal_portfolio(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        signal=args.signal,
        horizon=args.horizon,
        top_k=args.k,
        cost_bps=args.cost_bps,
        date_from=args.start,
        date_to=args.end,
        anchor_mode=args.anchor_mode,
        min_coverage=args.min_coverage,
        long_short=args.long_short,
    )

    ts_path = write_timeseries_csv(date_rows, args.out_dir)
    sm_path = write_summary_json(summary, args.out_dir)

    print(f"\nResults: {summary['n_evaluated']} / {summary['n_dates']} dates evaluated")
    lg = summary["long"]
    print(f"Long:  mean_gross={_fmt_pct(lg['mean_gross'])}  "
          f"mean_net={_fmt_pct(lg['mean_net'])}  "
          f"cum_net={_fmt_pct(lg['cumulative_net'])}  "
          f"mean_turn={_fmt(lg['mean_turnover'])}")
    if args.long_short and "long_short" in summary:
        ls = summary["long_short"]
        print(f"L/S:   mean_gross={_fmt_pct(ls['mean_gross'])}  "
              f"mean_net={_fmt_pct(ls['mean_net'])}  "
              f"cum_net={_fmt_pct(ls['cumulative_net'])}")
    if "regime_slices" in summary:
        print("\nRegime slices:")
        for regime, sl in summary["regime_slices"].items():
            print(f"  {regime:12s}  n={sl['n_dates']:2d}  "
                  f"mean_net={_fmt_pct(sl['mean_net'])}  "
                  f"cum_net={_fmt_pct(sl['cumulative_net'])}")

    print(f"\nWrote: {ts_path}")
    print(f"Wrote: {sm_path}")
    return 0


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "\u2014"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.2%}" if v is not None else "\u2014"


if __name__ == "__main__":
    sys.exit(main())
