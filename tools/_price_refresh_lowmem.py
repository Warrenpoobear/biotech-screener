"""Memory-efficient price refresh that avoids the pd.concat MultiIndex explosion.

Writes directly to price_history.csv one ticker at a time — peak RAM is one
ticker's DataFrame rather than all 338 concatenated.

Usage:
    python tools/_price_refresh_lowmem.py --as-of-date 2026-06-04
"""

import argparse
import csv
import gc
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDNAMES = ["date", "ticker", "close", "open", "high", "low", "volume"]
BOOTSTRAP_START = "2020-01-01"
DELAY_SEC = 1.5
MAX_RETRIES = 3


def _flatten_yf_df(df, ticker: str) -> list[dict]:
    """Convert a yfinance history DataFrame (possibly MultiIndex columns) to flat dicts."""
    import pandas as pd

    if df is None or df.empty:
        return []

    # yfinance 0.2+ returns MultiIndex columns like ('Close', 'AAPL')
    # Flatten by taking the first level or by droplevel
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1) if df.columns.nlevels == 2 else df
        # After droplevel, duplicate column names may exist — keep first occurrence
        df = df.loc[:, ~df.columns.duplicated()]

    # Normalize column names to lowercase
    df.columns = [str(c).lower() for c in df.columns]

    rows = []
    for idx, row in df.iterrows():
        close = row.get("close")
        if close is None or (hasattr(close, "__float__") and close != close):
            continue
        try:
            close_f = float(close)
        except (TypeError, ValueError):
            continue
        if close_f != close_f:  # NaN check
            continue

        def _safe(col):
            v = row.get(col)
            if v is None:
                return ""
            try:
                fv = float(v)
                return "" if fv != fv else str(fv)
            except (TypeError, ValueError):
                return ""

        rows.append(
            {
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10],
                "ticker": ticker.upper(),
                "close": str(close_f),
                "open": _safe("open"),
                "high": _safe("high"),
                "low": _safe("low"),
                "volume": (
                    str(int(float(row["volume"])))
                    if row.get("volume") is not None and str(row.get("volume")) not in ("", "nan")
                    else ""
                ),
            }
        )
    return rows


def fetch_ticker(ticker: str, start: str, end: str) -> list[dict]:
    """Download one ticker from yfinance, return flat dicts. Never throws."""
    import yfinance as yf

    retry = 0
    delay = DELAY_SEC
    while retry < MAX_RETRIES:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            rows = _flatten_yf_df(data, ticker)
            del data
            gc.collect()
            return rows
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err or "Expecting value" in err:
                backoff = delay * (1.5**retry) * (0.5 + random.random())
                logger.warning("Rate-limit on %s, backoff %.1fs", ticker, backoff)
                time.sleep(backoff)
                retry += 1
            else:
                logger.error("Failed %s: %s", ticker, err[:120])
                return []
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "production_data" / "price_history.csv"),
    )
    parser.add_argument("--start", default=BOOTSTRAP_START)
    args = parser.parse_args()

    price_csv = Path(args.output)
    through_date = args.as_of_date

    # Load universe
    universe_path = REPO_ROOT / "production_data" / "universe.json"
    with open(universe_path) as f:
        universe = json.load(f)
    if isinstance(universe, list):
        tickers = [e.get("ticker", e) if isinstance(e, dict) else str(e) for e in universe]
    elif isinstance(universe, dict) and "tickers" in universe:
        tickers = universe["tickers"]
    else:
        tickers = []
    tickers = [t for t in tickers if t and not t.startswith("_")]
    if "XBI" not in tickers:
        tickers.append("XBI")
    logger.info("Universe: %d tickers", len(tickers))

    # Load existing rows (if any) into a dict keyed by (ticker, date)
    existing: dict[tuple, dict] = {}
    if price_csv.exists():
        with open(price_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("ticker", "").upper(), (row.get("date") or "")[:10])
                existing[key] = row
        logger.info("Loaded %d existing rows from %s", len(existing), price_csv)

    # Determine which tickers need fetching and from what date
    needs_fetch: dict[str, str] = {}
    for t in tickers:
        # Find latest existing date for this ticker
        latest = None
        for k in existing:
            if k[0] == t.upper() and (latest is None or k[1] > latest):
                latest = k[1]
        if latest is None or latest < through_date:
            needs_fetch[t] = latest if latest and latest >= args.start else args.start  # incremental from last date

    logger.info(
        "%d tickers need fetching, %d already current",
        len(needs_fetch),
        len(tickers) - len(needs_fetch),
    )

    # Download each ticker and merge into existing dict
    n_ok = n_fail = 0
    for i, (ticker, fetch_start) in enumerate(needs_fetch.items()):
        if i > 0:
            time.sleep(DELAY_SEC * (0.5 + random.random()))
        rows = fetch_ticker(ticker, fetch_start, through_date)
        if rows:
            for row in rows:
                key = (row["ticker"].upper(), row["date"][:10])
                existing[key] = row
            n_ok += 1
            logger.debug("✓ %s: %d rows", ticker, len(rows))
        else:
            n_fail += 1
            logger.warning("✗ %s: no data", ticker)
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d tickers", i + 1, len(needs_fetch))

    logger.info("Fetch complete: %d ok, %d failed", n_ok, n_fail)

    # Write atomically
    price_csv.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=price_csv.parent)
    try:
        with open(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            for row in sorted(existing.values(), key=lambda r: (r.get("date", ""), r.get("ticker", ""))):
                writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
        Path(tmp_path).replace(price_csv)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Written %d rows to %s", len(existing), price_csv)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
