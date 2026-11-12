#!/usr/bin/env python3
"""
Patch price field in market_data.json from price_history.csv.

When yfinance rate-limits during the nightly collect_market_data.py run,
market_data.json falls back to cached data and close_price in snapshots
goes stale. price_history.csv refreshes via a different endpoint and stays
current. This tool copies the latest price_history close into market_data.json.

It records the close's date in ``price_date`` (and ``price_source``) and never
touches ``collected_at``: that field is when collect_market_data.py last fetched
the row's full field set (avg_volume, beta, volatility_90d, short_percent, ...),
and the market_data_staleness gate reads it. Re-stamping it here made
months-old rows look fresh (2026-11-12).

Usage:
    python3 tools/patch_market_data_prices.py
    python3 tools/patch_market_data_prices.py --as-of-date 2026-06-24
    python3 tools/patch_market_data_prices.py --dry-run

Output: overwrites production_data/market_data.json (atomic rename).
"""

import argparse
import csv
import json
import os
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_DATA = REPO_ROOT / "production_data" / "market_data.json"
PRICE_HISTORY = REPO_ROOT / "production_data" / "price_history.csv"


def load_latest_prices(price_csv: Path, as_of_date: str) -> dict:
    """Return {ticker: close} for the most recent date <= as_of_date in price_history.csv."""
    prices_by_ticker = {}
    with open(price_csv) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")
            t = row.get("ticker", "").strip().upper()
            c = row.get("close", "")
            if d <= as_of_date and t and c:
                try:
                    prices_by_ticker[t] = (d, float(c))
                except ValueError:
                    pass
    return {t: (d, p) for t, (d, p) in prices_by_ticker.items()}


def main():
    parser = argparse.ArgumentParser(description="Patch market_data.json prices from price_history.csv")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Patching market_data.json prices from price_history.csv (as-of {args.as_of_date})")

    with open(MARKET_DATA) as f:
        records = json.load(f)

    old_date = records[0].get("collected_at", "unknown") if records else "unknown"
    print(f"  market_data.json current collected_at: {old_date}")

    prices = load_latest_prices(PRICE_HISTORY, args.as_of_date)
    print(f"  price_history.csv: {len(prices)} tickers with prices on/before {args.as_of_date}")

    patched, unchanged, missing = 0, 0, []
    for rec in records:
        ticker = rec.get("ticker", "").strip().upper()
        if ticker in prices:
            price_date, new_price = prices[ticker]
            old_price = rec.get("price")
            if not args.dry_run:
                # Never write collected_at here; see module docstring.
                rec["price_date"] = price_date
                rec["price_source"] = "price_history"
            if old_price != new_price:
                if not args.dry_run:
                    rec["price"] = new_price
                patched += 1
            else:
                unchanged += 1
        else:
            missing.append(ticker)

    print(f"  Patched: {patched}, Unchanged: {unchanged}, Missing from price_history: {len(missing)}")
    if missing:
        print(f"  Missing tickers: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")

    if args.dry_run:
        print("  [dry-run] No changes written.")
        return

    fd, tmp = tempfile.mkstemp(dir=MARKET_DATA.parent, prefix=".tmp_market_data_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        Path(tmp).replace(MARKET_DATA)
    except Exception:
        os.unlink(tmp)
        raise

    print(f"  Wrote {MARKET_DATA} (price + price_date updated; collected_at unchanged)")


if __name__ == "__main__":
    main()
