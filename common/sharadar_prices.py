"""Sharadar as the production price source for price_history.csv and market caps.

Price convention: SPLIT-ADJUSTED, not dividend-adjusted (Sharadar SEP/SFP
``open/high/low/close/volume``). This is a deliberate choice, verified
2026-11-11 against the incumbent file:

  * the yfinance-built price_history.csv is NOT raw: yfinance split-adjusts a
    ticker's history as of its first fetch (DRMA 2022 = 446.40 in the file vs
    0.185 raw — its cumulative reverse splits), and only splits after that fetch
    appear raw. 36,621 rows since 2020 are split-adjusted that way;
  * scripts/repair_price_history_splits.py detects splits from the data only
    (single-day +300% / -75%), so a raw series would leak every smaller split
    (1:2 reverse = +100%, 2:1 forward = -50%) into momentum as a fake return;
  * a split-adjusted series is continuous, so the repair step finds nothing to
    undo, and the latest close equals the raw price anyway.
  ``closeunadj`` (raw) and ``closeadj`` (split+dividend) are NOT used.

Why Sharadar: on the 2026-01-02..2026-11-10 overlap ~15% of days in the
yfinance-built file differ from the official close by >0.5% (intraday values
captured at run time and never revised, e.g. ABVX 2026-03-20 123.65 vs 120.34),
because the yfinance refresh only ever appends new dates.

Daily model (``refresh_prices_from_sharadar``):
  * re-fetch a trailing window (default 10 calendar days) and UPSERT it, so
    Sharadar's official closes replace provisional values (including any
    yfinance row written for a day Sharadar had not yet published);
  * tickers with no history at all get a full backfill window;
  * tickers SEP does not cover (ETFs such as XBI/IBB) are fetched from SFP;
  * anything Sharadar still lacks is left for the yfinance fallback, which
    only appends dates after each ticker's last row.

DAILY ``marketcap``/``ev`` are in $ MILLIONS; ``update_market_caps_from_daily``
stores raw USD (see fix/market-cap-units, 2026-11-11).

All writes are atomic (common.atomic_io): production_data files are
hard-linked into staged PIT inputs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "ticker", "close", "open", "high", "low", "volume"]
SEP_FIELDS = ["ticker", "date", "open", "high", "low", "close", "volume"]
DAILY_FIELDS = ["ticker", "date", "marketcap", "ev"]
SHARADAR_DAILY_UNIT_USD = 1_000_000
DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_BACKFILL_START = "2020-01-01"


# --------------------------------------------------------------------------- rows
def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def to_bar(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A Sharadar SEP/SFP row as a split-adjusted price_history bar (see module doc)."""
    close = _f(row.get("close"))
    if close is None or close <= 0:
        return None

    def num(v: Any) -> Any:
        x = _f(v)
        return "" if x is None else round(x, 6)

    vol = _f(row.get("volume"))
    return {
        "date": str(row["date"])[:10],
        "ticker": str(row["ticker"]).upper(),
        "close": round(close, 6),
        "open": num(row.get("open")),
        "high": num(row.get("high")),
        "low": num(row.get("low")),
        "volume": "" if vol is None else int(round(vol)),
    }


# --------------------------------------------------------------------------- csv
def _read_prices(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not csv_path.exists():
        return list(PRICE_COLUMNS), []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or PRICE_COLUMNS), list(reader)


def last_dates(csv_path: Path) -> Dict[str, str]:
    """Latest date per ticker in price_history.csv."""
    out: Dict[str, str] = {}
    for r in _read_prices(csv_path)[1]:
        t, d = r.get("ticker", ""), r.get("date", "")
        if t and d > out.get(t, ""):
            out[t] = d
    return out


def upsert_price_history(csv_path: Path, bars: Iterable[Dict[str, Any]], tolerance: float = 1e-4) -> Dict[str, Any]:
    """Insert or replace (ticker, date) bars in price_history.csv, atomically.

    Existing rows keep their position; a row is rewritten only when its close
    differs by more than ``tolerance`` (relative). New rows are appended in
    (date, ticker) order. Returns counts plus the largest relative revision.
    """
    columns, rows = _read_prices(csv_path)
    for c in PRICE_COLUMNS:
        if c not in columns:
            columns.append(c)
    index = {(r.get("ticker", ""), r.get("date", "")): i for i, r in enumerate(rows)}
    incoming: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for b in bars:
        if b:
            incoming[(b["ticker"], b["date"])] = b

    revised, appended, max_rev, revised_tickers = 0, [], 0.0, set()
    for key, b in incoming.items():
        if key in index:
            old = rows[index[key]]
            old_close = _f(old.get("close"))
            rel = abs(old_close - b["close"]) / b["close"] if old_close else 1.0
            if rel > tolerance:
                rows[index[key]] = {**old, **{c: b[c] for c in PRICE_COLUMNS}}
                revised += 1
                revised_tickers.add(key[0])
                max_rev = max(max_rev, rel)
        else:
            appended.append(b)
    appended.sort(key=lambda b: (b["date"], b["ticker"]))
    rows.extend({c: b[c] for c in PRICE_COLUMNS} for b in appended)

    if revised or appended:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        atomic_write_text(csv_path, buf.getvalue())
    return {
        "n_revised": revised,
        "n_appended": len(appended),
        "max_rel_revision": round(max_rev, 6),
        "revised_tickers": sorted(revised_tickers),
        "appended_tickers": sorted({b["ticker"] for b in appended}),
    }


# --------------------------------------------------------------------------- fetch
def _iso(d: date) -> str:
    return d.isoformat()


def fetch_bars(client: Any, tickers: Sequence[str], start: str, end: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split-adjusted bars from SEP, then SFP for tickers SEP does not cover (ETFs).

    Returns (bars, tickers_with_no_data).
    """
    tickers = sorted({t.upper() for t in tickers if t and not t.startswith("_")})
    if not tickers:
        return [], []
    rows = client.fetch("SEP", tickers=tickers, date_range=(start, end), fields=SEP_FIELDS)
    got = {str(r["ticker"]).upper() for r in rows}
    missing = [t for t in tickers if t not in got]
    if missing:
        rows += client.fetch("SFP", tickers=missing, date_range=(start, end), fields=SEP_FIELDS)
        got = {str(r["ticker"]).upper() for r in rows}
    bars = [b for b in (to_bar(r) for r in rows) if b]
    return bars, [t for t in tickers if t not in got]


def refresh_prices_from_sharadar(
    csv_path: Path,
    through_date: str,
    tickers: Sequence[str],
    client: Any = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    backfill_start: str = DEFAULT_BACKFILL_START,
) -> Dict[str, Any]:
    """Upsert a trailing window (and full backfills for new tickers) from Sharadar.

    Never raises for data problems: returns ``status`` OK / PARTIAL / ERROR so the
    caller can fall back to yfinance for whatever is still missing.
    """
    stats: Dict[str, Any] = {"status": "ERROR", "source": "sharadar", "n_revised": 0, "n_appended": 0}
    try:
        if client is None:
            from common.sharadar_client import SharadarClient

            client = SharadarClient()
        end = date.fromisoformat(through_date)
        have = last_dates(csv_path)
        universe = sorted({t.upper() for t in tickers if t and not t.startswith("_")})
        existing = [t for t in universe if t in have]
        new = [t for t in universe if t not in have]
        window_start = _iso(end - timedelta(days=lookback_days))

        bars, no_data = fetch_bars(client, existing, window_start, through_date)
        if new:
            nb, nd = fetch_bars(client, new, backfill_start, through_date)
            bars += nb
            no_data += nd
        upsert = upsert_price_history(csv_path, bars)
        latest = max((b["date"] for b in bars), default=None)
        stats.update(upsert)
        stats.update(
            {
                "status": "OK" if not no_data else "PARTIAL",
                "n_tickers": len(universe),
                "n_backfilled": len(new) - len([t for t in new if t in no_data]),
                "no_data_tickers": sorted(no_data),
                "latest_date": latest,
                "has_through_date": latest == through_date,
            }
        )
    except Exception as exc:  # caller falls back to yfinance
        stats["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        logger.warning("Sharadar price refresh failed: %s", stats["detail"])
    return stats


def update_market_caps_from_daily(
    market_data_path: Path,
    through_date: str,
    client: Any = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """Set market_cap / enterprise_value (raw USD) from Sharadar DAILY, atomically."""
    stats: Dict[str, Any] = {"status": "ERROR", "n_updated": 0}
    try:
        if client is None:
            from common.sharadar_client import SharadarClient

            client = SharadarClient()
        records = json.loads(market_data_path.read_text(encoding="utf-8"))
        tickers = [r["ticker"] for r in records if r.get("ticker") and not r["ticker"].startswith("_")]
        start = _iso(date.fromisoformat(through_date) - timedelta(days=lookback_days))
        rows = client.fetch("DAILY", tickers=tickers, date_range=(start, through_date), fields=DAILY_FIELDS)
        latest: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            t = str(r["ticker"]).upper()
            if str(r["date"]) >= str(latest.get(t, {}).get("date", "")):
                latest[t] = r
        n = 0
        for rec in records:
            d = latest.get(str(rec.get("ticker", "")).upper())
            if not d:
                continue
            if _f(d.get("marketcap")) is not None:
                rec["market_cap"] = _f(d["marketcap"]) * SHARADAR_DAILY_UNIT_USD
            if _f(d.get("ev")) is not None:
                rec["enterprise_value"] = _f(d["ev"]) * SHARADAR_DAILY_UNIT_USD
            rec["market_cap_source"] = "sharadar_daily"
            rec["market_cap_date"] = str(d["date"])[:10]
            n += 1
        if n:
            atomic_write_text(market_data_path, json.dumps(records, indent=2))
        stats.update({"status": "OK", "n_updated": n, "n_records": len(records)})
    except Exception as exc:
        stats["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        logger.warning("Sharadar market-cap update failed: %s", stats["detail"])
    return stats
