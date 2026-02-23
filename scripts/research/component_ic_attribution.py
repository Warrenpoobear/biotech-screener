#!/usr/bin/env python3
"""Component-level IC attribution for PIT backtest.

RESEARCH ONLY — not for production use.
Largely superseded by --component-eval in eval_forward_returns.py,
but retained for standalone per-signal analysis with direction control.

Inputs:  snapshot-root, price-csv
Outputs: summary.json, summary.md, by_signal.csv, by_date_signal.csv

For each sub-signal (clinical, catalyst, coinvest, price overlay), computes
Spearman IC and long-short decile spread against forward returns.

Identifies which scoring component drives alpha (or anti-alpha).

Usage:
    python3 scripts/research/component_ic_attribution.py \
        --snapshot-root data/snapshots_pit \
        --price-csv production_data/price_history.csv \
        --horizons 5,20,63 \
        --anchor-mode next_trading_day \
        --date-from 2020-03-31 --date-to 2025-12-31 \
        --out-dir output/component_ic
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_forward_returns import (
    compute_forward_return,
    discover_snapshot_dates,
    load_price_series,
    load_rankings,
    resolve_trade_date,
    spearman_ic,
)

# ---------------------------------------------------------------------------
# Signal definitions: (column_name, direction, label, component)
#   direction: +1 means higher value = better expected return
#              -1 means lower value = better expected return
# ---------------------------------------------------------------------------

SIGNALS = [
    # Clinical
    ("clinical_score", +1, "clinical_score", "clinical"),
    ("clinical_alpha_z", +1, "clinical_alpha_z", "clinical"),
    ("clinical_score_z", +1, "clinical_score_z", "clinical"),

    # Catalyst
    ("catalyst_decay_w", +1, "catalyst_decay_w", "catalyst"),
    ("catalyst_days", -1, "catalyst_days (inv)", "catalyst"),

    # Coinvest
    ("sponsor_tier1_count", +1, "sponsor_tier1_count", "coinvest"),
    ("coinvest_score_z", +1, "coinvest_score_z", "coinvest"),

    # Price overlay / defensive
    ("de_alpha_60d", +1, "alpha_60d", "price"),
    ("de_drawdown", +1, "drawdown (less neg=better)", "price"),
    ("de_beta_xbi_60d", +1, "beta_xbi_60d", "price"),
    ("de_rsi_14d", +1, "rsi_14d", "price"),

    # Composite
    ("actionable_rank", -1, "actionable_rank (inv)", "composite"),
]


# ---------------------------------------------------------------------------
# Per-date, per-signal evaluation
# ---------------------------------------------------------------------------

@dataclass
class SignalDateResult:
    date: str
    trade_date: str
    signal_name: str
    component: str
    horizon: int
    ic: Optional[float] = None
    ls_spread: Optional[float] = None
    n_obs: int = 0
    top_decile_ret: Optional[float] = None
    bottom_decile_ret: Optional[float] = None


def _extract_signal(
    rankings: List[Dict[str, str]],
    column: str,
) -> Dict[str, float]:
    """Extract signal values from rankings rows → {ticker: float}."""
    result: Dict[str, float] = {}
    for r in rankings:
        t = r.get("ticker", "")
        v = r.get(column, "")
        if t and v and v not in ("", "nan", "None"):
            try:
                result[t] = float(v)
            except (ValueError, TypeError):
                pass
    return result


def _decile_spread_from_signal(
    signal: Dict[str, float],
    fwd_rets: Dict[str, float],
    direction: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute L/S decile spread from a signal, sorted by direction.

    Returns (spread, top_decile_ret, bottom_decile_ret).
    """
    # Tickers with both signal and return
    common = [t for t in signal if t in fwd_rets]
    if len(common) < 10:
        return None, None, None

    # Sort: direction=+1 → highest first (best), direction=-1 → lowest first
    common.sort(key=lambda t: signal[t], reverse=(direction == +1))

    d_size = max(1, len(common) // 10)
    top = common[:d_size]
    bottom = common[-d_size:]
    top_ret = statistics.mean(fwd_rets[t] for t in top)
    bottom_ret = statistics.mean(fwd_rets[t] for t in bottom)
    return top_ret - bottom_ret, top_ret, bottom_ret


def evaluate_signals(
    snapshot_root: Path,
    price_csv: Path,
    horizons: List[int],
    anchor_mode: str = "next_trading_day",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_price_coverage: float = 0.30,
) -> List[SignalDateResult]:
    """Evaluate all signals across all dates and horizons."""
    snap_dates = discover_snapshot_dates(snapshot_root, date_from, date_to)
    prices = load_price_series(price_csv)

    all_dates_set: set = set()
    for tp in prices.values():
        all_dates_set.update(tp.keys())
    sorted_dates = sorted(all_dates_set)

    results: List[SignalDateResult] = []

    for snap_date in snap_dates:
        snap_dir = snapshot_root / snap_date

        # Resolve trade date
        trade_date = resolve_trade_date(sorted_dates, snap_date, anchor_mode)
        if trade_date is None:
            continue

        # Load rankings
        rankings = load_rankings(snap_dir)
        if not rankings:
            continue

        # Extract all signals
        signal_maps: Dict[str, Dict[str, float]] = {}
        for col, direction, label, component in SIGNALS:
            signal_maps[label] = _extract_signal(rankings, col)

        for h in horizons:
            # Compute forward returns
            all_tickers = set()
            for sm in signal_maps.values():
                all_tickers.update(sm.keys())

            fwd_rets: Dict[str, float] = {}
            for ticker in all_tickers:
                if ticker not in prices:
                    continue
                ret = compute_forward_return(
                    prices[ticker], sorted_dates, trade_date, h
                )
                if ret is not None:
                    fwd_rets[ticker] = ret

            if not fwd_rets:
                continue

            for col, direction, label, component in SIGNALS:
                sig = signal_maps[label]
                # Tickers with both signal and return
                common = [t for t in sig if t in fwd_rets]
                n = len(common)

                ic = None
                if n >= 10:
                    sig_vals = [sig[t] * direction for t in common]
                    ret_vals = [fwd_rets[t] for t in common]
                    ic = spearman_ic(sig_vals, ret_vals)

                spread, top_ret, bot_ret = _decile_spread_from_signal(
                    sig, fwd_rets, direction
                )

                results.append(SignalDateResult(
                    date=snap_date,
                    trade_date=trade_date,
                    signal_name=label,
                    component=component,
                    horizon=h,
                    ic=round(ic, 4) if ic is not None else None,
                    ls_spread=round(spread, 6) if spread is not None else None,
                    n_obs=n,
                    top_decile_ret=round(top_ret, 6) if top_ret is not None else None,
                    bottom_decile_ret=round(bot_ret, 6) if bot_ret is not None else None,
                ))

    return results


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def aggregate(
    results: List[SignalDateResult],
    horizons: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Aggregate per-signal, per-horizon across dates.

    Returns {horizon: [row_dict, ...]}.
    """
    from collections import defaultdict

    # Group by (signal_name, horizon)
    grouped: Dict[Tuple[str, int], List[SignalDateResult]] = defaultdict(list)
    for r in results:
        grouped[(r.signal_name, r.horizon)].append(r)

    out: Dict[int, List[Dict[str, Any]]] = {h: [] for h in horizons}

    for (signal_name, h), date_results in sorted(grouped.items()):
        if h not in out:
            continue

        ics = [r.ic for r in date_results if r.ic is not None]
        spreads = [r.ls_spread for r in date_results if r.ls_spread is not None]
        n_obs_list = [r.n_obs for r in date_results]
        component = date_results[0].component if date_results else ""

        mean_ic = statistics.mean(ics) if ics else None
        median_ic = statistics.median(ics) if ics else None
        std_ic = statistics.stdev(ics) if len(ics) >= 2 else None
        ic_t = None
        if mean_ic is not None and std_ic and std_ic > 0 and len(ics) >= 5:
            ic_t = mean_ic / (std_ic / math.sqrt(len(ics)))

        mean_ls = statistics.mean(spreads) if spreads else None
        ls_t = None
        if mean_ls is not None and len(spreads) >= 5:
            ls_std = statistics.stdev(spreads) if len(spreads) >= 2 else None
            if ls_std and ls_std > 0:
                ls_t = mean_ls / (ls_std / math.sqrt(len(spreads)))

        out[h].append({
            "signal": signal_name,
            "component": component,
            "n_dates": len(ics),
            "mean_n_obs": round(statistics.mean(n_obs_list), 0) if n_obs_list else 0,
            "mean_ic": round(mean_ic, 4) if mean_ic is not None else None,
            "median_ic": round(median_ic, 4) if median_ic is not None else None,
            "std_ic": round(std_ic, 4) if std_ic is not None else None,
            "ic_t": round(ic_t, 3) if ic_t is not None else None,
            "mean_ls": round(mean_ls, 6) if mean_ls is not None else None,
            "ls_t": round(ls_t, 3) if ls_t is not None else None,
        })

    return out


def write_summary(
    agg: Dict[int, List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    """Write summary.json, summary.md, and by_signal.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    # CSV (all horizons)
    csv_path = out_dir / "by_signal.csv"
    fieldnames = [
        "horizon", "component", "signal", "n_dates", "mean_n_obs",
        "mean_ic", "median_ic", "std_ic", "ic_t",
        "mean_ls", "ls_t",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h, rows in sorted(agg.items()):
            for row in rows:
                writer.writerow({"horizon": h, **row})

    # Markdown
    md_path = out_dir / "summary.md"
    lines = [
        "# Component-Level IC Attribution",
        "",
    ]

    def _f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "\u2014"

    def _fp(v):
        return f"{v:.2%}" if v is not None else "\u2014"

    for h, rows in sorted(agg.items()):
        lines.append(f"## {h}d Horizon")
        lines.append("")
        lines.append(
            "| Component | Signal | N | Obs | Mean IC | Median IC | "
            "IC t | Mean L/S | L/S t |"
        )
        lines.append(
            "|-----------|--------|---|-----|---------|-----------|"
            "------|----------|-------|"
        )
        # Sort by absolute IC descending
        sorted_rows = sorted(
            rows,
            key=lambda r: abs(r["mean_ic"]) if r["mean_ic"] is not None else 0,
            reverse=True,
        )
        for row in sorted_rows:
            ic_str = _f(row["mean_ic"])
            # Highlight significant ICs
            if row["ic_t"] is not None and abs(row["ic_t"]) >= 2.0:
                ic_str = f"**{ic_str}**"
            lines.append(
                f"| {row['component']} "
                f"| {row['signal']} "
                f"| {row['n_dates']} "
                f"| {int(row['mean_n_obs'])} "
                f"| {ic_str} "
                f"| {_f(row['median_ic'])} "
                f"| {_f(row['ic_t'], 2)} "
                f"| {_fp(row['mean_ls'])} "
                f"| {_f(row['ls_t'], 2)} |"
            )
        lines.append("")

    # Interpretation guide
    lines.extend([
        "## Interpretation",
        "",
        "- **IC > 0**: higher signal value predicts higher forward returns "
        "(after applying direction correction).",
        "- **IC t > 2.0**: statistically significant at ~95% confidence "
        "(bold in table).",
        "- **Mean L/S**: top-decile minus bottom-decile equal-weight return "
        "(positive = signal sorts well at tails).",
        "- Signals with **negative IC and significant t-stat**: "
        "actively hurting the ranking.",
        "- Signals with **near-zero IC**: not contributing predictive power.",
        "",
        "### Signal Directions",
        "- clinical_score, clinical_alpha_z, clinical_score_z: "
        "higher = better clinical pipeline.",
        "- catalyst_decay_w: higher = stronger upcoming catalyst.",
        "- catalyst_days (inv): *inverted* — fewer days to catalyst = higher signal.",
        "- sponsor_tier1_count, coinvest_score_z: "
        "higher = more institutional support.",
        "- alpha_60d: higher = better recent momentum vs XBI.",
        "- drawdown: higher (less negative) = less beaten down.",
        "- beta_xbi_60d: higher = more market-sensitive.",
        "- rsi_14d: higher = more overbought.",
        "- actionable_rank (inv): *inverted* — lower rank = better.",
        "",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def write_detail_csv(
    results: List[SignalDateResult],
    out_dir: Path,
) -> None:
    """Write per-date, per-signal detail CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "by_date_signal.csv"
    fieldnames = [
        "date", "trade_date", "signal_name", "component", "horizon",
        "ic", "ls_spread", "n_obs",
        "top_decile_ret", "bottom_decile_ret",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "date": r.date,
                "trade_date": r.trade_date,
                "signal_name": r.signal_name,
                "component": r.component,
                "horizon": r.horizon,
                "ic": r.ic,
                "ls_spread": r.ls_spread,
                "n_obs": r.n_obs,
                "top_decile_ret": r.top_decile_ret,
                "bottom_decile_ret": r.bottom_decile_ret,
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component-level IC attribution",
    )
    parser.add_argument(
        "--snapshot-root", type=Path,
        default=PROJECT_ROOT / "data" / "snapshots_pit",
    )
    parser.add_argument(
        "--price-csv", type=Path,
        default=PROJECT_ROOT / "production_data" / "price_history.csv",
    )
    parser.add_argument("--horizons", default="5,20,63")
    parser.add_argument(
        "--anchor-mode", default="next_trading_day",
        choices=["exact", "next_trading_day", "prev_trading_day"],
    )
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "output" / "component_ic",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    print(f"Component IC attribution: {args.snapshot_root}")
    print(f"  Horizons: {horizons}, Anchor: {args.anchor_mode}")

    results = evaluate_signals(
        snapshot_root=args.snapshot_root,
        price_csv=args.price_csv,
        horizons=horizons,
        anchor_mode=args.anchor_mode,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    agg = aggregate(results, horizons)
    write_summary(agg, args.out_dir)
    write_detail_csv(results, args.out_dir)

    print(f"\nResults → {args.out_dir}")
    for h in horizons:
        rows = agg.get(h, [])
        print(f"\n  {h}d horizon:")
        sorted_rows = sorted(
            rows,
            key=lambda r: abs(r["mean_ic"]) if r["mean_ic"] is not None else 0,
            reverse=True,
        )
        for row in sorted_rows:
            ic = row["mean_ic"]
            ic_t = row["ic_t"]
            sig = "**" if ic_t is not None and abs(ic_t) >= 2.0 else "  "
            ic_str = f"{ic:+.4f}" if ic is not None else "    —"
            t_str = f"t={ic_t:+.2f}" if ic_t is not None else "t=   —"
            ls = row["mean_ls"]
            ls_str = f"L/S={ls:+.2%}" if ls is not None else "L/S=    —"
            print(
                f"    {sig}{row['component']:10s} {row['signal']:30s} "
                f"IC={ic_str} {t_str}  {ls_str}"
            )


if __name__ == "__main__":
    main()
