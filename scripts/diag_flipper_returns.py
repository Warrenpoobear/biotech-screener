#!/usr/bin/env python3
"""Flipper Return Attribution — diagnostic script.

For each bad→good catalyst flip recorded in catalyst_shadow_metrics.json,
compute forward returns at t+1d, t+5d, t+20d (trading days) and compare
against same-date universe of dev tickers.

Usage:
    python3 scripts/diag_flipper_returns.py \
        --snapshot-dir data/snapshots \
        --price-csv production_data/price_history.csv \
        --output-dir data/diag \
        [--no-fetch]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_FLIP_THRESHOLD = 10          # flips > this → regime (not organic)
HORIZONS = [1, 5, 20]              # forward return horizons (trading days)
PRICE_CACHE_NAME = "price_cache.csv"
PRICE_COLS = ["date", "ticker", "close"]


# ---------------------------------------------------------------------------
# Step 1 — Collect flip events
# ---------------------------------------------------------------------------
def collect_flip_events(
    snapshot_dir: Path,
) -> List[dict]:
    """Scan snapshot dirs for bad_to_good flips, enrich from rankings.csv."""
    events: List[dict] = []
    dates = sorted(
        d.name
        for d in snapshot_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )
    for date_str in dates:
        sm_path = snapshot_dir / date_str / "catalyst_shadow_metrics.json"
        if not sm_path.exists():
            continue
        with sm_path.open(encoding="utf-8") as f:
            sm = json.load(f)
        b2g = sm.get("bad_to_good_tickers", [])
        if not b2g:
            continue
        flip_count = sm.get("bad_to_good_count", len(b2g))
        cohort = "regime" if flip_count > REGIME_FLIP_THRESHOLD else "organic"

        # Load rankings for enrichment
        rank_path = snapshot_dir / date_str / "rankings.csv"
        rank_lookup: Dict[str, dict] = {}
        if rank_path.exists():
            with rank_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rank_lookup[row["ticker"]] = row

        for ticker in b2g:
            rk = rank_lookup.get(ticker, {})
            events.append(
                {
                    "flip_date": date_str,
                    "ticker": ticker,
                    "cohort": cohort,
                    "tier_dev": rk.get("tier_dev", ""),
                    "size_band": rk.get("size_band", ""),
                    "archetype": rk.get("archetype", ""),
                    "composite_rank": _safe_float(rk.get("composite_rank")),
                    "catalyst_mode": rk.get("catalyst_mode")
                    or rk.get("de_catalyst_mode", ""),
                }
            )
    return events


# ---------------------------------------------------------------------------
# Step 2 — Load / fetch prices
# ---------------------------------------------------------------------------
def load_prices(price_csv: Path, cache_csv: Optional[Path]) -> Dict[Tuple[str, str], float]:
    """Return (date_str, ticker) → close price dict from CSV(s)."""
    prices: Dict[Tuple[str, str], float] = {}
    for path in [price_csv, cache_csv]:
        if path is None or not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = row["date"][:10]  # normalize to YYYY-MM-DD
                t = row["ticker"]
                c = row.get("close")
                if c is None or c == "":
                    continue
                try:
                    val = float(c)
                except (ValueError, TypeError):
                    continue
                if val != val:  # NaN
                    continue
                prices[(d, t)] = val
    return prices


def _trading_dates_sorted(prices: Dict[Tuple[str, str], float]) -> List[str]:
    """Sorted unique dates from price dict."""
    return sorted({k[0] for k in prices})


def _forward_date(
    trading_dates: List[str], date_str: str, offset: int
) -> Optional[str]:
    """Return trading date at +offset business days from date_str, or None."""
    try:
        idx = trading_dates.index(date_str)
    except ValueError:
        return None
    target = idx + offset
    if 0 <= target < len(trading_dates):
        return trading_dates[target]
    return None


def identify_missing_tickers(
    events: List[dict],
    prices: Dict[Tuple[str, str], float],
    trading_dates: List[str],
    all_dev_tickers: set,
) -> set:
    """Return set of tickers that need price data for return calc."""
    needed: set = set()
    flip_dates = {e["flip_date"] for e in events}
    tickers_of_interest = {e["ticker"] for e in events} | all_dev_tickers
    for d in flip_dates:
        for h in HORIZONS:
            td = _forward_date(trading_dates, d, h)
            # If we can't even find the flip date in trading calendar,
            # we need prices for all relevant tickers around that period.
            for t in tickers_of_interest:
                if (d, t) not in prices:
                    needed.add(t)
                elif td and (td, t) not in prices:
                    needed.add(t)
    return needed


def fetch_missing_prices(
    tickers: set,
    cache_csv: Path,
) -> Dict[Tuple[str, str], float]:
    """Fetch prices via yfinance and cache to CSV. Returns new prices."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("  yfinance not installed — skipping fetch", file=sys.stderr)
        return {}

    if not tickers:
        return {}

    print(f"  Fetching prices for {len(tickers)} tickers via yfinance...",
          file=sys.stderr)

    new_prices: Dict[Tuple[str, str], float] = {}
    # Fetch in batches to avoid yfinance limits
    ticker_list = sorted(tickers)
    batch_size = 50
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers",
              file=sys.stderr)
        for ticker in batch:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="3mo", auto_adjust=False)
                for idx, row in hist.iterrows():
                    close = row.get("Close")
                    if close is None or close != close:
                        continue
                    d = idx.strftime("%Y-%m-%d")
                    new_prices[(d, ticker)] = float(close)
            except Exception as exc:
                print(f"    {ticker}: error {exc}", file=sys.stderr)

    # Persist to cache
    if new_prices:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        existing_keys: set = set()
        if cache_csv.exists():
            with cache_csv.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing_keys.add((row["date"], row["ticker"]))

        rows_to_write = [
            {"date": k[0], "ticker": k[1], "close": f"{v:.4f}"}
            for k, v in sorted(new_prices.items())
            if k not in existing_keys
        ]
        if rows_to_write:
            write_header = not cache_csv.exists()
            with cache_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PRICE_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows_to_write)
        print(f"  Cached {len(rows_to_write)} new price rows to {cache_csv}",
              file=sys.stderr)

    return new_prices


# ---------------------------------------------------------------------------
# Step 3+4 — Compute forward returns + control group
# ---------------------------------------------------------------------------
def compute_returns(
    events: List[dict],
    prices: Dict[Tuple[str, str], float],
    trading_dates: List[str],
    snapshot_dir: Path,
) -> List[dict]:
    """For each flip event, compute raw + excess returns at each horizon."""
    # Pre-build per-date dev ticker sets from rankings
    dev_tickers_by_date: Dict[str, List[str]] = {}
    for e in events:
        d = e["flip_date"]
        if d in dev_tickers_by_date:
            continue
        rank_path = snapshot_dir / d / "rankings.csv"
        if not rank_path.exists():
            dev_tickers_by_date[d] = []
            continue
        tickers = []
        with rank_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("archetype", "") == "drug_developer":
                    tickers.append(row["ticker"])
        dev_tickers_by_date[d] = tickers

    # Compute universe median returns per (date, horizon)
    universe_medians: Dict[Tuple[str, int], Optional[float]] = {}
    flip_dates = {e["flip_date"] for e in events}
    for d in flip_dates:
        dev_tk = dev_tickers_by_date.get(d, [])
        base_price_d = {t: prices.get((d, t)) for t in dev_tk}
        for h in HORIZONS:
            td = _forward_date(trading_dates, d, h)
            if td is None:
                universe_medians[(d, h)] = None
                continue
            rets = []
            for t in dev_tk:
                p0 = base_price_d.get(t)
                p1 = prices.get((td, t)) if td else None
                if p0 and p1 and p0 > 0:
                    rets.append(p1 / p0 - 1)
            universe_medians[(d, h)] = (
                statistics.median(rets) if len(rets) >= 5 else None
            )

    # Compute per-event returns
    results: List[dict] = []
    for e in events:
        d = e["flip_date"]
        t = e["ticker"]
        p0 = prices.get((d, t))
        row = {
            "flip_date": d,
            "ticker": t,
            "cohort": e["cohort"],
            "tier_dev": e["tier_dev"],
            "size_band": e["size_band"],
            "composite_rank": e["composite_rank"],
            "close_t": p0,
        }
        for h in HORIZONS:
            td = _forward_date(trading_dates, d, h)
            p1 = prices.get((td, t)) if td else None
            raw_ret = (p1 / p0 - 1) if (p0 and p1 and p0 > 0) else None
            univ_med = universe_medians.get((d, h))
            excess = (raw_ret - univ_med) if (raw_ret is not None and univ_med is not None) else None
            row[f"ret_{h}d"] = raw_ret
            row[f"universe_ret_{h}d"] = univ_med
            row[f"excess_{h}d"] = excess
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Step 5 — Statistics
# ---------------------------------------------------------------------------
def _cohort_stats(rows: List[dict], label: str) -> dict:
    """Compute summary stats for a set of flip events."""
    n = len(rows)
    stats: dict = {"cohort": label, "n_flips": n}
    for h in HORIZONS:
        rets = [r[f"ret_{h}d"] for r in rows if r[f"ret_{h}d"] is not None]
        excess = [r[f"excess_{h}d"] for r in rows if r[f"excess_{h}d"] is not None]
        stats[f"n_ret_{h}d"] = len(rets)
        stats[f"coverage_{h}d"] = f"{len(rets)/n*100:.0f}%" if n else "0%"
        if rets:
            stats[f"mean_ret_{h}d"] = statistics.mean(rets)
            stats[f"median_ret_{h}d"] = statistics.median(rets)
            stats[f"min_ret_{h}d"] = min(rets)
            stats[f"max_ret_{h}d"] = max(rets)
        else:
            for k in ["mean_ret", "median_ret", "min_ret", "max_ret"]:
                stats[f"{k}_{h}d"] = None
        if excess:
            stats[f"mean_excess_{h}d"] = statistics.mean(excess)
            stats[f"median_excess_{h}d"] = statistics.median(excess)
            hit = sum(1 for x in excess if x > 0)
            stats[f"hit_rate_{h}d"] = hit / len(excess)
        else:
            stats[f"mean_excess_{h}d"] = None
            stats[f"median_excess_{h}d"] = None
            stats[f"hit_rate_{h}d"] = None
    return stats


def _trimmed_stats(rows: List[dict], label: str, trim: int = 2) -> dict:
    """Stats after removing top/bottom N by excess_5d return (outlier check)."""
    rets5 = [(i, r.get("excess_5d")) for i, r in enumerate(rows)]
    rets5 = [(i, v) for i, v in rets5 if v is not None]
    if len(rets5) <= 2 * trim:
        return _cohort_stats(rows, label + " (trimmed=noop)")
    rets5.sort(key=lambda x: x[1])
    drop_idx = {x[0] for x in rets5[:trim]} | {x[0] for x in rets5[-trim:]}
    trimmed = [r for i, r in enumerate(rows) if i not in drop_idx]
    return _cohort_stats(trimmed, label + f" (trim-{trim})")


def compute_all_stats(results: List[dict]) -> List[dict]:
    """Compute stats for all / organic / regime cohorts + trimmed variants."""
    all_rows = results
    organic = [r for r in results if r["cohort"] == "organic"]
    regime = [r for r in results if r["cohort"] == "regime"]

    stats = [
        _cohort_stats(all_rows, "all"),
        _cohort_stats(organic, "organic"),
        _cohort_stats(regime, "regime"),
        _trimmed_stats(all_rows, "all"),
    ]

    # Per-tier stats (if sufficient N)
    tiers: Dict[str, List[dict]] = {}
    for r in results:
        t = r["tier_dev"] or "unknown"
        tiers.setdefault(t, []).append(r)
    for tier, rows in sorted(tiers.items()):
        if len(rows) >= 3:
            stats.append(_cohort_stats(rows, f"tier={tier}"))

    return stats


# ---------------------------------------------------------------------------
# Step 6 — Output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "flip_date", "ticker", "cohort", "tier_dev", "size_band", "composite_rank",
    "close_t",
    "ret_1d", "ret_5d", "ret_20d",
    "universe_ret_1d", "universe_ret_5d", "universe_ret_20d",
    "excess_1d", "excess_5d", "excess_20d",
]


def write_csv(results: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {}
            for col in CSV_COLUMNS:
                v = r.get(col)
                if v is None:
                    row[col] = ""
                elif isinstance(v, float):
                    row[col] = f"{v:.6f}" if "ret" in col or "excess" in col else f"{v:.2f}"
                else:
                    row[col] = v
            writer.writerow(row)


def write_report(
    results: List[dict], stats: List[dict], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = sorted({r["flip_date"] for r in results})
    lines: List[str] = []
    lines.append("# Flipper Return Attribution Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Flip dates: {dates[0]} to {dates[-1]} ({len(dates)} dates)")
    lines.append(f"Total flip events: {len(results)}")
    lines.append("")

    # Flip date summary
    lines.append("## Flip Date Summary")
    lines.append("| Date | Cohort | Flips |")
    lines.append("|------|--------|-------|")
    by_date: Dict[str, List[dict]] = {}
    for r in results:
        by_date.setdefault(r["flip_date"], []).append(r)
    for d in sorted(by_date):
        rows = by_date[d]
        lines.append(f"| {d} | {rows[0]['cohort']} | {len(rows)} |")
    lines.append("")

    # Cohort stats tables
    lines.append("## Cohort Summary")
    for s in stats:
        lines.append(f"### {s['cohort']} (N={s['n_flips']})")
        lines.append("| Horizon | Coverage | Mean Ret | Median Ret | Mean Excess | Median Excess | Hit Rate | Min | Max |")
        lines.append("|---------|----------|----------|------------|-------------|---------------|----------|-----|-----|")
        for h in HORIZONS:
            cov = s.get(f"coverage_{h}d", "0%")
            mean_r = _fmt_pct(s.get(f"mean_ret_{h}d"))
            med_r = _fmt_pct(s.get(f"median_ret_{h}d"))
            mean_e = _fmt_pct(s.get(f"mean_excess_{h}d"))
            med_e = _fmt_pct(s.get(f"median_excess_{h}d"))
            hit = _fmt_pct(s.get(f"hit_rate_{h}d"))
            mn = _fmt_pct(s.get(f"min_ret_{h}d"))
            mx = _fmt_pct(s.get(f"max_ret_{h}d"))
            lines.append(f"| t+{h}d | {cov} | {mean_r} | {med_r} | {mean_e} | {med_e} | {hit} | {mn} | {mx} |")
        lines.append("")

    # Top/bottom movers by excess_5d
    has_5d = [r for r in results if r.get("excess_5d") is not None]
    if has_5d:
        lines.append("## Top 5 Movers (excess return at t+5d)")
        has_5d_sorted = sorted(has_5d, key=lambda x: x["excess_5d"], reverse=True)
        lines.append("| Rank | Ticker | Date | Cohort | Tier | Excess 5d | Raw 5d |")
        lines.append("|------|--------|------|--------|------|-----------|--------|")
        for i, r in enumerate(has_5d_sorted[:5], 1):
            lines.append(
                f"| {i} | {r['ticker']} | {r['flip_date']} | {r['cohort']} "
                f"| {r['tier_dev']} | {_fmt_pct(r['excess_5d'])} | {_fmt_pct(r['ret_5d'])} |"
            )
        lines.append("")

        lines.append("## Bottom 5 Movers (excess return at t+5d)")
        lines.append("| Rank | Ticker | Date | Cohort | Tier | Excess 5d | Raw 5d |")
        lines.append("|------|--------|------|--------|------|-----------|--------|")
        for i, r in enumerate(has_5d_sorted[-5:], 1):
            lines.append(
                f"| {i} | {r['ticker']} | {r['flip_date']} | {r['cohort']} "
                f"| {r['tier_dev']} | {_fmt_pct(r['excess_5d'])} | {_fmt_pct(r['ret_5d'])} |"
            )
        lines.append("")

    # Coverage summary
    lines.append("## Coverage Summary")
    total = len(results)
    for h in HORIZONS:
        n_with = sum(1 for r in results if r.get(f"ret_{h}d") is not None)
        lines.append(f"- t+{h}d: {n_with}/{total} ({n_with/total*100:.0f}%)")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v*100:+.2f}%"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        val = float(v)
        return val if val == val else None
    except (ValueError, TypeError):
        return None


def _all_dev_tickers(snapshot_dir: Path, dates: List[str]) -> set:
    """Collect union of all drug_developer tickers across flip dates."""
    tickers: set = set()
    for d in dates:
        rank_path = snapshot_dir / d / "rankings.csv"
        if not rank_path.exists():
            continue
        with rank_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("archetype", "") == "drug_developer":
                    tickers.add(row["ticker"])
    return tickers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Flipper return attribution")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Path to snapshot directory",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=Path("production_data/price_history.csv"),
        help="Path to price_history.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diag"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip yfinance fetch; use existing price data only",
    )
    args = parser.parse_args()

    # Step 1 — Collect flip events
    print("Step 1: Collecting flip events...", file=sys.stderr)
    events = collect_flip_events(args.snapshot_dir)
    if not events:
        print("No flip events found. Exiting.", file=sys.stderr)
        sys.exit(0)

    cohort_counts = {}
    for e in events:
        cohort_counts[e["cohort"]] = cohort_counts.get(e["cohort"], 0) + 1
    print(f"  Found {len(events)} flip events across "
          f"{len({e['flip_date'] for e in events})} dates", file=sys.stderr)
    for c, n in sorted(cohort_counts.items()):
        print(f"    {c}: {n}", file=sys.stderr)

    # Step 2 — Load / fetch prices
    print("Step 2: Loading prices...", file=sys.stderr)
    cache_csv = args.output_dir / PRICE_CACHE_NAME
    prices = load_prices(args.price_csv, cache_csv)
    trading_dates = _trading_dates_sorted(prices)
    print(f"  Loaded {len(prices)} price points, "
          f"{len(trading_dates)} trading dates "
          f"({trading_dates[0] if trading_dates else '?'} to "
          f"{trading_dates[-1] if trading_dates else '?'})", file=sys.stderr)

    if not args.no_fetch:
        flip_dates = sorted({e["flip_date"] for e in events})
        dev_tickers = _all_dev_tickers(args.snapshot_dir, flip_dates)
        missing = identify_missing_tickers(events, prices, trading_dates, dev_tickers)
        if missing:
            print(f"  {len(missing)} tickers need price data", file=sys.stderr)
            new_prices = fetch_missing_prices(missing, cache_csv)
            prices.update(new_prices)
            trading_dates = _trading_dates_sorted(prices)
            print(f"  Now {len(prices)} price points, "
                  f"{len(trading_dates)} trading dates "
                  f"({trading_dates[0]} to {trading_dates[-1]})",
                  file=sys.stderr)
        else:
            print("  No missing tickers to fetch", file=sys.stderr)

    # Step 3+4 — Compute returns + control group
    print("Step 3: Computing forward returns...", file=sys.stderr)
    results = compute_returns(events, prices, trading_dates, args.snapshot_dir)

    # Coverage report
    for h in HORIZONS:
        n_with = sum(1 for r in results if r[f"ret_{h}d"] is not None)
        print(f"  t+{h}d coverage: {n_with}/{len(results)} "
              f"({n_with/len(results)*100:.0f}%)", file=sys.stderr)

    # Step 5 — Statistics
    print("Step 4: Computing statistics...", file=sys.stderr)
    stats = compute_all_stats(results)

    # Step 6 — Output
    flip_dates = sorted({r["flip_date"] for r in results})
    first = flip_dates[0].replace("-", "")
    last = flip_dates[-1].replace("-", "")
    stem = f"flipper_returns_{first}_{last}"

    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"

    write_csv(results, csv_path)
    write_report(results, stats, md_path)

    print(f"\nOutputs:", file=sys.stderr)
    print(f"  CSV:    {csv_path}", file=sys.stderr)
    print(f"  Report: {md_path}", file=sys.stderr)

    # Print quick summary to stdout
    s = stats[0]  # "all" cohort
    print(f"\n--- Quick Summary (all, N={s['n_flips']}) ---")
    for h in HORIZONS:
        mean_e = s.get(f"mean_excess_{h}d")
        hit = s.get(f"hit_rate_{h}d")
        cov = s.get(f"coverage_{h}d", "0%")
        print(f"  t+{h:2d}d: mean excess={_fmt_pct(mean_e):>8s}  "
              f"hit rate={_fmt_pct(hit):>8s}  coverage={cov}")


if __name__ == "__main__":
    main()
