#!/usr/bin/env python3
"""Forward-return evaluation across snapshot dates.

Walk-forward evaluation: Spearman IC, top-K portfolio gross/net return,
turnover, and skip tracking.  Strict PIT enforcement via metadata.json.

Outputs: summary.json, summary.md, by_date.csv, skips.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS = [5, 20, 63]
DEFAULT_TOP_K = 20
DEFAULT_COST_BPS = 30
DEFAULT_MIN_PRICE_COVERAGE = 0.50


# ---------------------------------------------------------------------------
# Price loader  (lightweight, no pandas dependency)
# ---------------------------------------------------------------------------

def load_price_series(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Load price_history.csv → {ticker: {date_str: close}}."""
    prices: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            close_str = (row.get("close") or "").strip()
            date_str = (row.get("date") or "").strip()
            if not ticker or not close_str or not date_str:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            prices.setdefault(ticker, {})[date_str] = close
    return prices


def _trading_days_after(all_dates: List[str], start: str, n: int) -> Optional[str]:
    """Return the trading date *n* trading days after *start*, or None."""
    try:
        idx = all_dates.index(start)
    except ValueError:
        return None
    target = idx + n
    if target < len(all_dates):
        return all_dates[target]
    return None


def compute_forward_return(
    prices: Dict[str, float],
    sorted_dates: List[str],
    snap_date: str,
    horizon: int,
) -> Optional[float]:
    """Simple forward return = P(t+h)/P(t) - 1."""
    p0 = prices.get(snap_date)
    if p0 is None or p0 <= 0:
        return None
    end_date = _trading_days_after(sorted_dates, snap_date, horizon)
    if end_date is None:
        return None
    p1 = prices.get(end_date)
    if p1 is None or p1 <= 0:
        return None
    return p1 / p0 - 1.0


# ---------------------------------------------------------------------------
# Spearman IC (manual, no scipy)
# ---------------------------------------------------------------------------

def _avg_ranks(values: List[float]) -> List[float]:
    """Average-rank with tie handling (1-based)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman_ic(signal: List[float], returns: List[float]) -> Optional[float]:
    """Rank correlation (Pearson of ranks). Returns None if n < 3."""
    n = len(signal)
    if n < 3:
        return None
    rx = _avg_ranks(signal)
    ry = _avg_ranks(returns)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if sx == 0.0 or sy == 0.0:
        return None
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Portfolio return + turnover
# ---------------------------------------------------------------------------

def top_k_portfolio_return(
    tickers_ranked: List[str],
    fwd_rets: Dict[str, float],
    k: int,
) -> Tuple[Optional[float], int]:
    """Equal-weight top-K gross return. Returns (mean_return, n_held)."""
    held = [t for t in tickers_ranked[:k] if t in fwd_rets]
    if not held:
        return None, 0
    ret = statistics.mean(fwd_rets[t] for t in held)
    return ret, len(held)


def compute_turnover(prev_set: List[str], curr_set: List[str]) -> float:
    """Turnover = 0.5 * |symmetric difference| / max(len(prev), len(curr), 1)."""
    if not prev_set and not curr_set:
        return 0.0
    s_prev = set(prev_set)
    s_curr = set(curr_set)
    diff = len(s_prev ^ s_curr)
    denom = max(len(s_prev), len(s_curr), 1)
    return 0.5 * diff / denom


def net_return(gross: float, turnover: float, cost_bps: float) -> float:
    """Net return after transaction cost haircut."""
    return gross - turnover * cost_bps / 10_000


# ---------------------------------------------------------------------------
# Snapshot discovery + loading
# ---------------------------------------------------------------------------

def discover_snapshot_dates(
    snapshot_root: Path,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[str]:
    """Return sorted list of YYYY-MM-DD snapshot dates."""
    dates: List[str] = []
    if not snapshot_root.exists():
        return dates
    for d in sorted(snapshot_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Must be YYYY-MM-DD format
        if len(name) != 10 or name[4] != "-" or name[7] != "-":
            continue
        try:
            date.fromisoformat(name)
        except ValueError:
            continue
        if date_from and name < date_from:
            continue
        if date_to and name > date_to:
            continue
        dates.append(name)
    return dates


def load_rankings(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load rankings.csv from a snapshot directory."""
    csv_path = snapshot_dir / "rankings.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_positions(snapshot_dir: Path) -> List[Dict[str, str]]:
    """Load portfolio_positions.csv from a snapshot directory."""
    csv_path = snapshot_dir / "portfolio_positions.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pit_metadata(snapshot_dir: Path, snap_date: str) -> Tuple[bool, str]:
    """Validate PIT enforcement via metadata.json."""
    meta_path = snapshot_dir / "metadata.json"
    if not meta_path.exists():
        return False, "metadata.json missing"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"metadata.json unreadable: {e}"
    as_of = meta.get("as_of_date", "")
    if as_of != snap_date:
        return False, f"as_of_date={as_of} != {snap_date}"
    return True, ""


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@dataclass
class DateResult:
    date: str
    horizon: int
    ic: Optional[float] = None
    gross_return: Optional[float] = None
    net_return: Optional[float] = None
    turnover: Optional[float] = None
    n_signal: int = 0
    n_held: int = 0
    coverage: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class EvalSummary:
    horizons: List[int] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    cost_bps: float = DEFAULT_COST_BPS
    n_dates: int = 0
    n_evaluated: int = 0
    n_skipped: int = 0
    by_horizon: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    use_positions: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate(
    snapshot_root: Path,
    price_csv: Path,
    horizons: List[int],
    top_k: int = DEFAULT_TOP_K,
    cost_bps: float = DEFAULT_COST_BPS,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    pit_mode: str = "strict",
    use_positions: bool = False,
    min_price_coverage: float = DEFAULT_MIN_PRICE_COVERAGE,
) -> Tuple[EvalSummary, List[DateResult], List[Dict[str, str]]]:
    """Run the full walk-forward evaluation.

    Returns (summary, date_results, skips).
    """
    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    prices = load_price_series(price_csv)

    # Build sorted trading dates from all price data
    all_dates_set: set = set()
    for ticker_prices in prices.values():
        all_dates_set.update(ticker_prices.keys())
    sorted_dates = sorted(all_dates_set)

    date_results: List[DateResult] = []
    skips: List[Dict[str, str]] = []

    # Per-horizon accumulators
    horizon_ics: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_gross: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_net: Dict[int, List[float]] = {h: [] for h in horizons}
    horizon_turnover: Dict[int, List[float]] = {h: [] for h in horizons}
    prev_holdings: Dict[int, List[str]] = {h: [] for h in horizons}

    n_evaluated = 0

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date

        # PIT check
        if pit_mode == "strict":
            ok, reason = check_pit_metadata(snap_dir, snap_date)
            if not ok:
                for h in horizons:
                    dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                    skip_reason=f"PIT: {reason}")
                    date_results.append(dr)
                skips.append({"date": snap_date, "reason": f"PIT: {reason}"})
                continue

        # Load rankings or positions
        if use_positions:
            positions = load_positions(snap_dir)
            tickers_ranked = [r["ticker"] for r in positions
                              if r.get("ticker")]
        else:
            rankings = load_rankings(snap_dir)
            if not rankings:
                for h in horizons:
                    dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                    skip_reason="EMPTY_RANKINGS")
                    date_results.append(dr)
                skips.append({"date": snap_date, "reason": "EMPTY_RANKINGS"})
                continue
            # Sort by actionable_rank ascending
            try:
                rankings.sort(key=lambda r: int(r.get("actionable_rank", 9999)))
            except (ValueError, TypeError):
                pass
            tickers_ranked = [r["ticker"] for r in rankings if r.get("ticker")]

        date_evaluated = False
        for h in horizons:
            # Compute forward returns for all tickers with prices
            fwd_rets: Dict[str, float] = {}
            for ticker in tickers_ranked:
                if ticker not in prices:
                    continue
                ret = compute_forward_return(prices[ticker], sorted_dates, snap_date, h)
                if ret is not None:
                    fwd_rets[ticker] = ret

            coverage = len(fwd_rets) / max(len(tickers_ranked), 1)
            if coverage < min_price_coverage:
                dr = DateResult(date=snap_date, horizon=h, skipped=True,
                                skip_reason="LOW_COVERAGE",
                                coverage=round(coverage, 4))
                date_results.append(dr)
                skips.append({"date": snap_date, "horizon": str(h),
                              "reason": f"LOW_COVERAGE ({coverage:.1%})"})
                continue

            # Signal: negative actionable_rank (lower rank = better = higher signal)
            signal_tickers = [t for t in tickers_ranked if t in fwd_rets]
            signal_vals = [-float(i + 1) for i, t in enumerate(tickers_ranked)
                           if t in fwd_rets]
            return_vals = [fwd_rets[t] for t in signal_tickers]

            ic = spearman_ic(signal_vals, return_vals)

            # Top-K portfolio
            top_k_tickers = tickers_ranked[:top_k]
            gross, n_held = top_k_portfolio_return(top_k_tickers, fwd_rets, top_k)

            # Turnover
            turn = compute_turnover(prev_holdings[h], top_k_tickers)
            prev_holdings[h] = top_k_tickers

            net_ret = None
            if gross is not None:
                net_ret = net_return(gross, turn, cost_bps)

            dr = DateResult(
                date=snap_date, horizon=h, ic=_round_opt(ic, 4),
                gross_return=_round_opt(gross, 6),
                net_return=_round_opt(net_ret, 6),
                turnover=round(turn, 4),
                n_signal=len(signal_tickers), n_held=n_held,
                coverage=round(coverage, 4),
            )
            date_results.append(dr)
            date_evaluated = True

            if ic is not None:
                horizon_ics[h].append(ic)
            if gross is not None:
                horizon_gross[h].append(gross)
            if net_ret is not None:
                horizon_net[h].append(net_ret)
            horizon_turnover[h].append(turn)

        if date_evaluated:
            n_evaluated += 1

    # Build summary
    summary = EvalSummary(
        horizons=horizons, top_k=top_k, cost_bps=cost_bps,
        n_dates=len(snap_dates), n_evaluated=n_evaluated,
        n_skipped=len(snap_dates) - n_evaluated,
        use_positions=use_positions,
    )

    for h in horizons:
        ics = horizon_ics[h]
        gross_list = horizon_gross[h]
        net_list = horizon_net[h]
        turn_list = horizon_turnover[h]

        summary.by_horizon[h] = {
            "n_dates": len(ics),
            "mean_ic": _round_opt(_safe_mean(ics), 4),
            "median_ic": _round_opt(_safe_median(ics), 4),
            "std_ic": _round_opt(_safe_std(ics), 4),
            "mean_gross_return": _round_opt(_safe_mean(gross_list), 6),
            "mean_net_return": _round_opt(_safe_mean(net_list), 6),
            "cumulative_gross": _round_opt(_cumulative(gross_list), 6),
            "cumulative_net": _round_opt(_cumulative(net_list), 6),
            "mean_turnover": _round_opt(_safe_mean(turn_list), 4),
        }

    return summary, date_results, skips


def _round_opt(v: Optional[float], d: int) -> Optional[float]:
    return round(v, d) if v is not None else None


def _safe_mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _safe_median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _safe_std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _cumulative(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    cum = 1.0
    for r in returns:
        cum *= (1.0 + r)
    return cum - 1.0


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary_json(summary: EvalSummary, out_dir: Path) -> Path:
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)
    return path


def write_summary_md(summary: EvalSummary, out_dir: Path) -> Path:
    path = out_dir / "summary.md"
    lines = [
        "# Forward-Return Evaluation Summary",
        "",
        f"- **Dates evaluated**: {summary.n_evaluated} / {summary.n_dates}",
        f"- **Skipped**: {summary.n_skipped}",
        f"- **Top-K**: {summary.top_k}",
        f"- **Cost (bps)**: {summary.cost_bps}",
        f"- **Positions mode**: {summary.use_positions}",
        "",
        "## By Horizon",
        "",
        "| Horizon | N | Mean IC | Median IC | Std IC | Mean Gross | Mean Net | Cum Gross | Cum Net | Mean Turn |",
        "|---------|---|---------|-----------|--------|------------|----------|-----------|---------|-----------|",
    ]
    for h in summary.horizons:
        bh = summary.by_horizon.get(h, {})
        lines.append(
            f"| {h}d | {bh.get('n_dates', 0)} "
            f"| {_fmt(bh.get('mean_ic'))} "
            f"| {_fmt(bh.get('median_ic'))} "
            f"| {_fmt(bh.get('std_ic'))} "
            f"| {_fmt_pct(bh.get('mean_gross_return'))} "
            f"| {_fmt_pct(bh.get('mean_net_return'))} "
            f"| {_fmt_pct(bh.get('cumulative_gross'))} "
            f"| {_fmt_pct(bh.get('cumulative_net'))} "
            f"| {_fmt(bh.get('mean_turnover'))} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_by_date_csv(date_results: List[DateResult], out_dir: Path) -> Path:
    path = out_dir / "by_date.csv"
    fieldnames = [
        "date", "horizon", "ic", "gross_return", "net_return",
        "turnover", "n_signal", "n_held", "coverage", "skipped", "skip_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dr in date_results:
            writer.writerow(asdict(dr))
    return path


def write_skips_json(skips: List[Dict[str, str]], out_dir: Path) -> Path:
    path = out_dir / "skips.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skips, f, indent=2)
    return path


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.2%}" if v is not None else "—"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-return evaluation across snapshot dates",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots",
        help="Root dir containing YYYY-MM-DD snapshot dirs",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
        help="Path to price_history.csv",
    )
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--date-to", type=str, default=None)
    parser.add_argument(
        "--horizons", type=str, default="5,20,63",
        help="Comma-separated forward-return horizons in trading days",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--pit-mode", choices=["strict", "lenient"], default="strict",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "forward_eval",
    )
    parser.add_argument("--use-positions", action="store_true", default=False)
    parser.add_argument(
        "--min-price-coverage", type=float, default=DEFAULT_MIN_PRICE_COVERAGE,
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating snapshots in {args.snapshot_root}")
    print(f"  Horizons: {horizons}, Top-K: {args.top_k}, Cost: {args.cost_bps} bps")

    summary, date_results, skips = evaluate(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        top_k=args.top_k,
        cost_bps=args.cost_bps,
        date_from=args.date_from,
        date_to=args.date_to,
        pit_mode=args.pit_mode,
        use_positions=args.use_positions,
        min_price_coverage=args.min_price_coverage,
    )

    write_summary_json(summary, args.out_dir)
    write_summary_md(summary, args.out_dir)
    write_by_date_csv(date_results, args.out_dir)
    write_skips_json(skips, args.out_dir)

    print(f"\nResults → {args.out_dir}")
    print(f"  {summary.n_evaluated}/{summary.n_dates} dates evaluated, "
          f"{summary.n_skipped} skipped")
    for h in horizons:
        bh = summary.by_horizon.get(h, {})
        print(f"  {h}d: IC={_fmt(bh.get('mean_ic'))}, "
              f"Gross={_fmt_pct(bh.get('mean_gross_return'))}, "
              f"Net={_fmt_pct(bh.get('mean_net_return'))}, "
              f"Turn={_fmt(bh.get('mean_turnover'))}")


if __name__ == "__main__":
    main()
