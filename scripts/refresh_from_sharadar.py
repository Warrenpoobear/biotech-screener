#!/usr/bin/env python3
"""Refresh price_history.csv and market_data.json price fields from Sharadar.

Replaces the yfinance-based refresh for environments where yfinance is unavailable.
Requires SHARADAR_API_KEY environment variable.

Usage:
    python3 scripts/refresh_from_sharadar.py [--as-of YYYY-MM-DD] [--lookback-years N]

Output:
    production_data/price_history.csv   -- appended/created with Sharadar SEP rows
    production_data/market_data.json    -- price + collected_at fields updated

CCFT: all writes carry a collected_at timestamp equal to the last Sharadar lastupdated
      date returned, never datetime.now() on its own.
"""
import argparse
import csv
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "production_data"
SHARADAR_BASE = "https://api.sharadar.com/v1.0/data"
BATCH_SIZE = 20  # tickers per request; keeps URL short and responses fast
RETRY_DELAYS = [2, 4, 8, 16]
CA_BUNDLE = "/root/.ccr/ca-bundle.crt"


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if Path(CA_BUNDLE).exists():
        ctx.load_verify_locations(CA_BUNDLE)
    return ctx


def _get_table(table: str, ctx: ssl.SSLContext, api_key: str, **filters) -> list:
    """Fetch rows from one Sharadar table with retry on transient errors."""
    params = {"format": "json", "api_key": api_key}
    params.update(filters)
    url = f"{SHARADAR_BASE}/{table}?" + urllib.parse.urlencode(params)
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            log.debug("Retrying after %ds (attempt %d)", delay, attempt)
            time.sleep(delay)
        try:
            resp = urllib.request.urlopen(url, context=ctx, timeout=60)
            body = json.loads(resp.read())
            rows = body.get("data", [])
            count = body.get("count", 0)
            if len(rows) < count:
                raise RuntimeError(
                    f"{table}: got {len(rows)} of {count} rows — narrow the date range"
                )
            return rows
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < len(RETRY_DELAYS):
                log.warning("HTTP %d from Sharadar; will retry", e.code)
                continue
            raise
    raise RuntimeError(f"All retries exhausted for {table}")


def _load_universe(data_dir: Path) -> list[str]:
    """Return tickers from market_data.json, plus benchmark tickers."""
    path = data_dir / "market_data.json"
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    universe = sorted(set(r["ticker"] for r in records if r.get("ticker")))
    # Add benchmarks used for relative momentum
    for bm in ("XBI", "IBB", "SPY"):
        if bm not in universe:
            universe.append(bm)
    return universe


def _fetch_sep(
    tickers: list[str], start_date: str, ctx: ssl.SSLContext, api_key: str
) -> list[dict]:
    """Fetch SEP (daily prices) in batches of BATCH_SIZE."""
    all_rows: list[dict] = []
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for idx, batch in enumerate(batches):
        ticker_str = ",".join(batch)
        log.info("SEP batch %d/%d: %s...", idx + 1, len(batches), ticker_str[:60])
        rows = _get_table(
            "SEP", ctx, api_key, ticker=ticker_str, **{"date.gte": start_date}
        )
        all_rows.extend(rows)
        if idx < len(batches) - 1:
            time.sleep(0.3)  # polite rate limiting
    return all_rows


def _fetch_daily(
    tickers: list[str], latest_date: str, ctx: ssl.SSLContext, api_key: str
) -> dict[str, dict]:
    """Fetch DAILY table (marketcap, ev) for the most recent date."""
    result: dict[str, dict] = {}
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        rows = _get_table(
            "DAILY",
            ctx,
            api_key,
            ticker=",".join(batch),
            **{"date.gte": latest_date},
        )
        for row in rows:
            t = row["ticker"]
            # keep the most recent row per ticker
            if t not in result or row["date"] > result[t]["date"]:
                result[t] = row
        time.sleep(0.3)
    return result


def _write_price_history(rows: list[dict], out_path: Path) -> str:
    """Write SEP rows to price_history.csv. Returns the latest date written."""
    # Sort for determinism
    rows_sorted = sorted(rows, key=lambda r: (r["date"], r["ticker"]))
    latest_date = rows_sorted[-1]["date"] if rows_sorted else ""
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["date", "ticker", "close", "open", "high", "low", "volume"])
        for r in rows_sorted:
            writer.writerow(
                [
                    r["date"],
                    r["ticker"],
                    r.get("closeadj") or r.get("close", ""),
                    r.get("open", ""),
                    r.get("high", ""),
                    r.get("low", ""),
                    r.get("volume", ""),
                ]
            )
    log.info("Wrote %d rows to %s (latest: %s)", len(rows_sorted), out_path, latest_date)
    return latest_date


def _update_market_data(
    daily_by_ticker: dict[str, dict],
    sep_rows: list[dict],
    data_dir: Path,
    collected_at: str,
) -> None:
    """Update price + market cap fields in market_data.json."""
    # Build latest price per ticker from SEP
    latest_price: dict[str, float] = {}
    for row in sep_rows:
        t = row["ticker"]
        if t not in latest_price or row["date"] > latest_price.get(t + "_date", ""):
            latest_price[t] = row.get("closeadj") or row.get("close")
            latest_price[t + "_date"] = row["date"]

    path = data_dir / "market_data.json"
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)

    updated = 0
    for rec in records:
        t = rec.get("ticker", "")
        if t in latest_price:
            rec["price"] = latest_price[t]
            rec["collected_at"] = collected_at
            updated += 1
        daily = daily_by_ticker.get(t)
        if daily:
            if daily.get("marketcap") is not None:
                rec["market_cap"] = daily["marketcap"]  # already in $M per Sharadar docs
            if daily.get("ev") is not None:
                rec["enterprise_value"] = daily["ev"]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(records, fh, indent=2)
        fh.write("\n")

    log.info("Updated %d/%d tickers in market_data.json", updated, len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh price data from Sharadar")
    parser.add_argument(
        "--as-of",
        default=str(date.today()),
        help="As-of date (YYYY-MM-DD); defaults to today",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=2,
        help="Years of price history to fetch (default 2)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to production_data directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate but do not write output files",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SHARADAR_API_KEY")
    if not api_key:
        raise SystemExit("SHARADAR_API_KEY environment variable is not set")

    as_of = date.fromisoformat(args.as_of)
    start_date = str(as_of - timedelta(days=args.lookback_years * 365))
    log.info("Sharadar refresh: as_of=%s, lookback start=%s", as_of, start_date)

    ctx = _ssl_ctx()
    tickers = _load_universe(args.data_dir)
    log.info("Universe: %d tickers", len(tickers))

    # Fetch SEP (price history)
    sep_rows = _fetch_sep(tickers, start_date, ctx, api_key)
    log.info("SEP: %d total rows fetched", len(sep_rows))

    # Collect latest date seen across all tickers
    latest_date = max((r["date"] for r in sep_rows), default=str(as_of))
    # Use lastupdated from rows as CCFT collected_at
    collected_at = max(
        (r.get("lastupdated", r["date"]) for r in sep_rows), default=str(as_of)
    )

    # Fetch DAILY for market cap / EV (only need recent rows)
    daily_start = str(as_of - timedelta(days=5))
    daily_by_ticker = _fetch_daily(tickers, daily_start, ctx, api_key)
    log.info("DAILY: %d tickers with market data", len(daily_by_ticker))

    if args.dry_run:
        log.info("Dry run — skipping writes. Latest date: %s", latest_date)
        return

    price_csv = args.data_dir / "price_history.csv"
    _write_price_history(sep_rows, price_csv)
    _update_market_data(daily_by_ticker, sep_rows, args.data_dir, collected_at)
    log.info("Done. collected_at=%s", collected_at)


if __name__ == "__main__":
    main()
