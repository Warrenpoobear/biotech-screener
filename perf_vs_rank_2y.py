#!/usr/bin/env python3
"""
perf_vs_rank_2y.py — Validate composite ranking against realized 2-year returns.

Computes trailing 2-year total return per ticker from daily price history,
joins against composite_rank from the screen CSV, and reports:
  - Spearman rank correlation (composite_rank vs return_rank)
  - Pearson correlation (composite_score vs return)
  - Decile lift table (avg return by composite rank decile)

Usage:
    python perf_vs_rank_2y.py \
      --rank-csv production_data/screen_2026-02-01.csv \
      --prices production_data/price_history.csv \
      --as-of 2026-02-01 \
      --out-csv production_data/perf_vs_rank_2y.csv \
      --out-json production_data/perf_vs_rank_2y.summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_prices(path: Path) -> dict[str, list[tuple[date, float]]]:
    """Load long-format price CSV: date,ticker,close,..."""
    prices: dict[str, list[tuple[date, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip().upper()
            close_str = row.get("close") or row.get("adj_close") or ""
            if not close_str:
                continue
            try:
                close = float(close_str)
            except ValueError:
                continue
            if close <= 0:
                continue
            dt = parse_date(row["date"])
            prices[ticker].append((dt, close))
    # Sort each ticker's prices by date
    for ticker in prices:
        prices[ticker].sort()
    return dict(prices)


def get_price_on_or_before(series: list[tuple[date, float]], target: date, max_lookback: int = 10) -> float | None:
    """Binary-search for the last price on or before target, within max_lookback trading days."""
    if not series:
        return None
    lo, hi = 0, len(series) - 1
    result_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= target:
            result_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if result_idx < 0:
        return None
    dt, price = series[result_idx]
    if (target - dt).days > max_lookback:
        return None
    return price


def load_ranks(path: Path) -> list[dict]:
    """Load screen CSV with composite_rank and composite_score."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip().upper()
            try:
                rank = int(row["composite_rank"])
                score = float(row["composite_score"])
            except (ValueError, KeyError):
                continue
            rows.append({"ticker": ticker, "composite_rank": rank, "composite_score": score})
    return rows


def spearman_corr(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return float("nan")

    def rank_data(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_corr(rx, ry)


def pearson_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def main():
    parser = argparse.ArgumentParser(description="Validate composite rank vs 2-year realized returns")
    parser.add_argument("--rank-csv", type=Path, required=True, help="Screen CSV with composite_rank, composite_score")
    parser.add_argument("--prices", type=Path, required=True, help="Daily price history CSV (long format)")
    parser.add_argument("--as-of", type=str, required=True, help="End date for return calc (YYYY-MM-DD)")
    parser.add_argument("--lookback-years", type=int, default=2, help="Years of return lookback (default: 2)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV with per-ticker results")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON summary")
    args = parser.parse_args()

    end_date = parse_date(args.as_of)
    start_date = date(end_date.year - args.lookback_years, end_date.month, end_date.day)

    print(f"Return window: {start_date} → {end_date} ({args.lookback_years}y)")
    print(f"Loading prices from {args.prices}...")
    prices = load_prices(args.prices)
    print(f"  {len(prices)} tickers in price history")

    print(f"Loading ranks from {args.rank_csv}...")
    ranks = load_ranks(args.rank_csv)
    print(f"  {len(ranks)} ranked tickers")

    # Compute returns
    results = []
    missing_start = 0
    missing_end = 0
    for r in ranks:
        ticker = r["ticker"]
        series = prices.get(ticker)
        if not series:
            missing_start += 1
            missing_end += 1
            continue
        p_start = get_price_on_or_before(series, start_date, max_lookback=10)
        p_end = get_price_on_or_before(series, end_date, max_lookback=10)
        if p_start is None:
            missing_start += 1
            continue
        if p_end is None:
            missing_end += 1
            continue
        ret = (p_end - p_start) / p_start
        results.append({
            "ticker": ticker,
            "composite_rank": r["composite_rank"],
            "composite_score": r["composite_score"],
            "price_start": round(p_start, 4),
            "price_end": round(p_end, 4),
            "return_2y": round(ret, 4),
        })

    print(f"\nReturn coverage: {len(results)}/{len(ranks)} tickers")
    if missing_start:
        print(f"  Missing start price: {missing_start}")
    if missing_end:
        print(f"  Missing end price: {missing_end}")

    if len(results) < 10:
        print("ERROR: Too few tickers with complete data for meaningful analysis.")
        return 1

    # Rank by realized return (1 = highest return)
    results.sort(key=lambda r: r["return_2y"], reverse=True)
    for i, r in enumerate(results, 1):
        r["return_rank_2y"] = i
        r["rank_delta"] = r["composite_rank"] - r["return_rank_2y"]

    # Correlations
    comp_ranks = [r["composite_rank"] for r in results]
    ret_ranks = [r["return_rank_2y"] for r in results]
    comp_scores = [r["composite_score"] for r in results]
    returns = [r["return_2y"] for r in results]

    spearman = spearman_corr(comp_ranks, ret_ranks)
    pearson_score_ret = pearson_corr(comp_scores, returns)
    pearson_rank_ret = pearson_corr(comp_ranks, returns)

    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS ({len(results)} tickers)")
    print(f"{'='*60}")
    print(f"Spearman(composite_rank, return_rank):  {spearman:+.4f}")
    print(f"Pearson(composite_score, return_2y):     {pearson_score_ret:+.4f}")
    print(f"Pearson(composite_rank, return_2y):      {pearson_rank_ret:+.4f}")

    # Decile analysis
    n = len(results)
    decile_size = n // 10
    results_by_rank = sorted(results, key=lambda r: r["composite_rank"])

    print(f"\n{'='*60}")
    print(f"DECILE LIFT TABLE (by composite rank)")
    print(f"{'='*60}")
    print(f"{'Decile':<8} {'Ranks':<14} {'N':<5} {'Avg Return':<12} {'Median Ret':<12} {'Win Rate':<10}")
    print(f"{'-'*8} {'-'*14} {'-'*5} {'-'*12} {'-'*12} {'-'*10}")

    decile_stats = []
    for d in range(10):
        start_idx = d * decile_size
        end_idx = start_idx + decile_size if d < 9 else n
        chunk = results_by_rank[start_idx:end_idx]
        rets = [c["return_2y"] for c in chunk]
        avg_ret = sum(rets) / len(rets)
        sorted_rets = sorted(rets)
        median_ret = sorted_rets[len(sorted_rets) // 2]
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
        rank_lo = chunk[0]["composite_rank"]
        rank_hi = chunk[-1]["composite_rank"]

        decile_stats.append({
            "decile": d + 1,
            "rank_range": f"{rank_lo}-{rank_hi}",
            "n": len(chunk),
            "avg_return": round(avg_ret, 4),
            "median_return": round(median_ret, 4),
            "win_rate": round(win_rate, 4),
        })
        print(f"D{d+1:<7} {rank_lo:>3}-{rank_hi:<10} {len(chunk):<5} {avg_ret:>+10.2%}   {median_ret:>+10.2%}   {win_rate:>8.1%}")

    # Top-1 decile vs bottom-1 decile spread
    d1_avg = decile_stats[0]["avg_return"]
    d10_avg = decile_stats[-1]["avg_return"]
    spread = d1_avg - d10_avg
    print(f"\nD1-D10 spread: {spread:+.2%}")
    print(f"D1 avg return: {d1_avg:+.2%}")
    print(f"D10 avg return: {d10_avg:+.2%}")

    # Quintile summary
    print(f"\n{'='*60}")
    print(f"QUINTILE SUMMARY")
    print(f"{'='*60}")
    for q in range(5):
        d_lo = q * 2
        d_hi = d_lo + 1
        avg = (decile_stats[d_lo]["avg_return"] + decile_stats[d_hi]["avg_return"]) / 2
        print(f"Q{q+1}: {avg:+.2%}")

    # Summary object
    summary = {
        "as_of_date": args.as_of,
        "lookback_years": args.lookback_years,
        "return_window": f"{start_date} to {end_date}",
        "tickers_ranked": len(ranks),
        "tickers_with_returns": len(results),
        "missing_start_price": missing_start,
        "missing_end_price": missing_end,
        "correlations": {
            "spearman_rank_vs_return_rank": round(spearman, 4),
            "pearson_score_vs_return": round(pearson_score_ret, 4),
            "pearson_rank_vs_return": round(pearson_rank_ret, 4),
        },
        "decile_lift": decile_stats,
        "d1_d10_spread": round(spread, 4),
    }

    # Write outputs
    if args.out_csv:
        results_sorted = sorted(results, key=lambda r: r["composite_rank"])
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ticker", "composite_rank", "composite_score",
                "price_start", "price_end", "return_2y",
                "return_rank_2y", "rank_delta",
            ])
            writer.writeheader()
            writer.writerows(results_sorted)
        print(f"\nWrote per-ticker results to {args.out_csv}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
