"""Massive options-history provider — contracts, day/minute aggs, trades.

Thin wrapper around the ``massive`` Python client (REST) and boto3 (S3 flat
files).  Designed for historical research and backfills, NOT live ranking.

Environment variables:
    MASSIVE_API_KEY              — REST API key (Bearer token)
    MASSIVE_S3_ACCESS_KEY_ID     — S3 flat-file access key
    MASSIVE_S3_SECRET_ACCESS_KEY — S3 flat-file secret key
    MASSIVE_CACHE_DIR            — local cache root (default: data/caches/massive_options)
    MASSIVE_USE_FLAT_FILES       — "true" to prefer flat files for bulk history

S3 flat file structure (Options Developer, 4yr history):
    us_options_opra/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    us_options_opra/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    us_options_opra/trades_v1/YYYY/MM/YYYY-MM-DD.csv.gz

REST endpoints used:
    GET /v3/reference/options/contracts  — contract discovery
    GET /v3/snapshot/options/{ticker}    — chain snapshot
    GET /v2/aggs/ticker/{ticker}/range/… — OHLCV bars
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("massive_options")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"

# Flat-file prefixes
_PREFIX_DAY_AGGS = "us_options_opra/day_aggs_v1"
_PREFIX_MINUTE_AGGS = "us_options_opra/minute_aggs_v1"
_PREFIX_TRADES = "us_options_opra/trades_v1"

# Cache schema
SCHEMA_VERSION = "massive_options_cache.v1"


def _get_config() -> Dict[str, Any]:
    """Read configuration from environment variables."""
    return {
        "api_key": os.environ.get("MASSIVE_API_KEY", ""),
        "s3_access_key_id": os.environ.get("MASSIVE_S3_ACCESS_KEY_ID", ""),
        "s3_secret_access_key": os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY", ""),
        "cache_dir": os.environ.get(
            "MASSIVE_CACHE_DIR",
            str(_REPO_ROOT / "data" / "caches" / "massive_options"),
        ),
        "use_flat_files": os.environ.get("MASSIVE_USE_FLAT_FILES", "true").lower() == "true",
    }


def _cache_dir() -> Path:
    """Return the cache root, creating it if needed."""
    p = Path(_get_config()["cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# REST client (lazy singleton)
# ---------------------------------------------------------------------------

_rest_client = None


def _get_rest_client():
    """Lazy-init the Massive REST client."""
    global _rest_client
    if _rest_client is None:
        from massive import RESTClient

        cfg = _get_config()
        if not cfg["api_key"]:
            raise EnvironmentError("MASSIVE_API_KEY is not set")
        _rest_client = RESTClient(api_key=cfg["api_key"])
    return _rest_client


# ---------------------------------------------------------------------------
# S3 client (lazy singleton)
# ---------------------------------------------------------------------------

_s3_client = None


def _get_s3_client():
    """Lazy-init the boto3 S3 client for flat file access."""
    global _s3_client
    if _s3_client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        cfg = _get_config()
        if not cfg["s3_access_key_id"] or not cfg["s3_secret_access_key"]:
            raise EnvironmentError("MASSIVE_S3_ACCESS_KEY_ID and MASSIVE_S3_SECRET_ACCESS_KEY must be set")
        session = boto3.Session(
            aws_access_key_id=cfg["s3_access_key_id"],
            aws_secret_access_key=cfg["s3_secret_access_key"],
        )
        _s3_client = session.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_client


def reset_clients():
    """Reset cached clients (useful for testing)."""
    global _rest_client, _s3_client
    _rest_client = None
    _s3_client = None


# ---------------------------------------------------------------------------
# Contract discovery (REST)
# ---------------------------------------------------------------------------


def list_contracts(
    underlying_ticker: str,
    as_of: Optional[str] = None,
    expiration_from: Optional[str] = None,
    expiration_to: Optional[str] = None,
    expired: bool = False,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """List options contracts for an underlying ticker.

    Returns list of contract dicts with keys: ticker, underlying_ticker,
    contract_type, strike_price, expiration_date, exercise_style, etc.
    """
    client = _get_rest_client()
    kwargs: Dict[str, Any] = {
        "underlying_ticker": underlying_ticker.upper(),
        "limit": limit,
        "order": "asc",
    }
    if as_of:
        kwargs["as_of"] = as_of
    if expiration_from:
        kwargs["expiration_date_gte"] = expiration_from
    if expiration_to:
        kwargs["expiration_date_lte"] = expiration_to
    if expired:
        kwargs["expired"] = True

    results = []
    for contract in client.list_options_contracts(**kwargs):
        results.append(
            {
                "ticker": getattr(contract, "ticker", ""),
                "underlying_ticker": getattr(contract, "underlying_ticker", ""),
                "contract_type": getattr(contract, "contract_type", ""),
                "strike_price": getattr(contract, "strike_price", None),
                "expiration_date": getattr(contract, "expiration_date", ""),
                "exercise_style": getattr(contract, "exercise_style", ""),
                "shares_per_contract": getattr(contract, "shares_per_contract", 100),
                "primary_exchange": getattr(contract, "primary_exchange", ""),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Chain snapshot (REST)
# ---------------------------------------------------------------------------


def compute_bid_ask_spread_pct(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    """Compute bid-ask spread as a fraction of the midpoint: (ask - bid) / mid.

    Returns None if bid or ask is missing/non-positive, or if the midpoint
    is non-positive (would produce a nonsensical or division-by-zero result).
    Never fabricates a spread when the underlying quote data is absent --
    callers must treat None as "spread unknown", not "spread is zero/tight".
    """
    if bid is None or ask is None:
        return None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except (ValueError, TypeError):
        return None
    if bid_f < 0 or ask_f < 0:
        return None
    mid = (bid_f + ask_f) / 2.0
    if mid <= 0:
        return None
    return round((ask_f - bid_f) / mid, 4)


def fetch_chain_snapshot(
    underlying_ticker: str,
    expiration_date: Optional[str] = None,
    contract_type: Optional[str] = None,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    """Fetch current option chain snapshot for an underlying.

    Returns list of dicts with greeks, IV, open interest, day OHLCV, quotes.

    Quote fields (bid/ask/midpoint/spread_pct): the underlying massive/Polygon
    ``OptionContractSnapshot.last_quote`` object DOES expose ``bid``, ``ask``,
    and ``midpoint`` (confirmed against the installed ``massive`` package's
    ``LastQuoteOptionContractSnapshot`` model: fields are
    bid/bid_size/ask/ask_size/midpoint/last_updated/timeframe). Prior to this
    change, ``last_quote`` was fetched from the snapshot object but never
    read into the returned record -- bid/ask/spread were silently dropped.
    This function now extracts them and computes ``bid_ask_spread_pct`` via
    compute_bid_ask_spread_pct(). If the vendor did not populate last_quote
    for a given contract (e.g. no NBBO available), these fields are None --
    they are never fabricated or defaulted to a "tight" value.
    """
    client = _get_rest_client()
    query: Dict[str, Any] = {"limit": limit}
    if expiration_date:
        query["expiration_date"] = expiration_date
    if contract_type:
        query["contract_type"] = contract_type

    results = []
    try:
        chain_iter = client.list_snapshot_options_chain(underlying_ticker.upper(), params=query)
    except Exception as exc:
        logger.warning("fetch_chain_snapshot failed for %s: %s", underlying_ticker, exc)
        return []
    for snap in chain_iter:
        rec: Dict[str, Any] = {}
        details = getattr(snap, "details", None)
        if details:
            rec["ticker"] = getattr(details, "ticker", "")
            rec["contract_type"] = getattr(details, "contract_type", "")
            rec["strike_price"] = getattr(details, "strike_price", None)
            rec["expiration_date"] = getattr(details, "expiration_date", "")

        rec["implied_volatility"] = getattr(snap, "implied_volatility", None)
        rec["open_interest"] = getattr(snap, "open_interest", None)
        rec["break_even_price"] = getattr(snap, "break_even_price", None)

        greeks = getattr(snap, "greeks", None)
        if greeks:
            rec["delta"] = getattr(greeks, "delta", None)
            rec["gamma"] = getattr(greeks, "gamma", None)
            rec["theta"] = getattr(greeks, "theta", None)
            rec["vega"] = getattr(greeks, "vega", None)

        day = getattr(snap, "day", None)
        if day:
            rec["day_open"] = getattr(day, "open", None)
            rec["day_high"] = getattr(day, "high", None)
            rec["day_low"] = getattr(day, "low", None)
            rec["day_close"] = getattr(day, "close", None)
            rec["day_volume"] = getattr(day, "volume", None)

        last_quote = getattr(snap, "last_quote", None)
        bid = getattr(last_quote, "bid", None) if last_quote else None
        ask = getattr(last_quote, "ask", None) if last_quote else None
        rec["bid"] = bid
        rec["ask"] = ask
        rec["quote_midpoint"] = getattr(last_quote, "midpoint", None) if last_quote else None
        rec["bid_ask_spread_pct"] = compute_bid_ask_spread_pct(bid, ask)

        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Flat file download helpers
# ---------------------------------------------------------------------------


def _s3_key_for_date(prefix: str, dt: date) -> str:
    """Build S3 key for a given date and data type prefix."""
    return f"{prefix}/{dt.year}/{dt.month:02d}/{dt.strftime('%Y-%m-%d')}.csv.gz"


def _download_flat_file(s3_key: str, dest_path: Path) -> bool:
    """Download a flat file from S3 to local path. Returns True on success."""
    s3 = _get_s3_client()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(S3_BUCKET, s3_key, str(dest_path))
        return True
    except Exception as exc:
        logger.warning("Failed to download s3://%s/%s: %s", S3_BUCKET, s3_key, exc)
        return False


def _read_csv_gz(path: Path) -> List[Dict[str, str]]:
    """Read a gzip CSV into list of dicts."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Day aggregates
# ---------------------------------------------------------------------------


def download_day_aggs(
    dt: date,
    force: bool = False,
) -> Optional[Path]:
    """Download options day aggregates flat file for a date.

    Returns path to local .csv.gz file, or None on failure.
    Skips download if file already exists unless force=True.
    """
    cache = _cache_dir() / "day_aggs" / str(dt.year) / f"{dt.month:02d}"
    dest = cache / f"{dt.strftime('%Y-%m-%d')}.csv.gz"

    if dest.exists() and not force:
        logger.info("Day aggs %s already cached: %s", dt, dest)
        return dest

    s3_key = _s3_key_for_date(_PREFIX_DAY_AGGS, dt)
    if _download_flat_file(s3_key, dest):
        logger.info("Downloaded day aggs for %s → %s", dt, dest)
        return dest
    return None


def ingest_day_aggs(dt: date, force: bool = False) -> List[Dict[str, Any]]:
    """Download and parse day aggregates for a date.

    Flat file columns: ticker, volume, open, close, high, low, window_start, transactions
    window_start is nanoseconds UTC.

    Returns normalized records with schema:
        option_ticker, underlying_ticker, date, open, high, low, close,
        volume, transactions, source
    """
    path = download_day_aggs(dt, force=force)
    if path is None:
        return []

    raw = _read_csv_gz(path)
    normalized = []
    for row in raw:
        ticker = row.get("ticker", "")
        underlying = _extract_underlying(ticker)
        normalized.append(
            {
                "option_ticker": ticker,
                "underlying_ticker": underlying,
                "date": dt.strftime("%Y-%m-%d"),
                "open": _to_float(row.get("open")),
                "high": _to_float(row.get("high")),
                "low": _to_float(row.get("low")),
                "close": _to_float(row.get("close")),
                "volume": _to_float(row.get("volume")),
                "transactions": _to_int(row.get("transactions")),
                "source": "massive",
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Minute aggregates
# ---------------------------------------------------------------------------


def download_minute_aggs(
    dt: date,
    force: bool = False,
) -> Optional[Path]:
    """Download options minute aggregates flat file for a date.

    Returns path to local .csv.gz file, or None on failure.
    """
    cache = _cache_dir() / "minute_aggs" / str(dt.year) / f"{dt.month:02d}"
    dest = cache / f"{dt.strftime('%Y-%m-%d')}.csv.gz"

    if dest.exists() and not force:
        logger.info("Minute aggs %s already cached: %s", dt, dest)
        return dest

    s3_key = _s3_key_for_date(_PREFIX_MINUTE_AGGS, dt)
    if _download_flat_file(s3_key, dest):
        logger.info("Downloaded minute aggs for %s → %s", dt, dest)
        return dest
    return None


def ingest_minute_aggs(dt: date, force: bool = False) -> List[Dict[str, Any]]:
    """Download and parse minute aggregates for a date.

    Flat file columns: ticker, volume, open, close, high, low, window_start, transactions
    window_start is nanoseconds UTC.

    Returns normalized records with schema:
        option_ticker, underlying_ticker, timestamp_utc, timestamp_et,
        open, high, low, close, volume, transactions, source
    """
    path = download_minute_aggs(dt, force=force)
    if path is None:
        return []

    raw = _read_csv_gz(path)
    normalized = []
    for row in raw:
        ticker = row.get("ticker", "")
        underlying = _extract_underlying(ticker)
        # window_start is nanoseconds UTC
        ts_ns = _to_int(row.get("window_start"))
        ts_utc = _unix_ns_to_utc(ts_ns) if ts_ns else ""
        ts_et = _utc_to_et(ts_utc) if ts_utc else ""

        normalized.append(
            {
                "option_ticker": ticker,
                "underlying_ticker": underlying,
                "timestamp_utc": ts_utc,
                "timestamp_et": ts_et,
                "open": _to_float(row.get("open")),
                "high": _to_float(row.get("high")),
                "low": _to_float(row.get("low")),
                "close": _to_float(row.get("close")),
                "volume": _to_float(row.get("volume")),
                "transactions": _to_int(row.get("transactions")),
                "source": "massive",
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


def download_trades(
    dt: date,
    force: bool = False,
) -> Optional[Path]:
    """Download options trades flat file for a date.

    Returns path to local .csv.gz file, or None on failure.
    """
    cache = _cache_dir() / "trades" / str(dt.year) / f"{dt.month:02d}"
    dest = cache / f"{dt.strftime('%Y-%m-%d')}.csv.gz"

    if dest.exists() and not force:
        logger.info("Trades %s already cached: %s", dt, dest)
        return dest

    s3_key = _s3_key_for_date(_PREFIX_TRADES, dt)
    if _download_flat_file(s3_key, dest):
        logger.info("Downloaded trades for %s → %s", dt, dest)
        return dest
    return None


def ingest_trades(dt: date, force: bool = False) -> List[Dict[str, Any]]:
    """Download and parse trades for a date.

    Returns normalized records with raw trade fields plus:
        option_ticker, underlying_ticker, timestamp_utc, timestamp_et,
        price, size, exchange, conditions, source
    """
    path = download_trades(dt, force=force)
    if path is None:
        return []

    raw = _read_csv_gz(path)
    normalized = []
    for row in raw:
        ticker = row.get("ticker", "")
        underlying = _extract_underlying(ticker)
        # Participant/SIP timestamp in nanoseconds UTC
        ts_ns = _to_int(row.get("sip_timestamp") or row.get("participant_timestamp") or row.get("t"))
        ts_utc = _unix_ns_to_utc(ts_ns) if ts_ns else ""
        ts_et = _utc_to_et(ts_utc) if ts_utc else ""

        normalized.append(
            {
                "option_ticker": ticker,
                "underlying_ticker": underlying,
                "timestamp_utc": ts_utc,
                "timestamp_et": ts_et,
                "price": _to_float(row.get("price") or row.get("p")),
                "size": _to_int(row.get("size") or row.get("s")),
                "exchange": row.get("exchange", ""),
                "conditions": row.get("conditions", ""),
                "source": "massive",
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _extract_underlying(option_ticker: str) -> str:
    """Extract underlying ticker from option ticker.

    O:AAPL231215C00150000 → AAPL
    O:MRNA260320P00025000 → MRNA
    """
    if not option_ticker:
        return ""
    # Strip O: prefix
    t = option_ticker
    if t.startswith("O:"):
        t = t[2:]
    if not t:
        return ""
    # Underlying is letters before the first digit
    i = 0
    while i < len(t) and t[i].isalpha():
        i += 1
    return t[:i].upper() if i > 0 else ""


def _to_float(v: Any) -> Optional[float]:
    """Safe float conversion."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_int(v: Any) -> Optional[int]:
    """Safe int conversion."""
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _unix_ms_to_utc(ms: int) -> str:
    """Convert Unix milliseconds to UTC ISO string."""
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (OSError, ValueError):
        return ""


def _unix_ns_to_utc(ns: int) -> str:
    """Convert Unix nanoseconds to UTC ISO string."""
    try:
        dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (OSError, ValueError):
        return ""


def _utc_to_et(utc_str: str) -> str:
    """Convert UTC ISO string to US/Eastern.

    Returns ISO string in ET (no pytz dependency — uses fixed offset
    heuristic: EST=-5, EDT=-4).  For research use only.
    """
    if not utc_str:
        return ""
    try:
        from zoneinfo import ZoneInfo

        dt_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
        return dt_et.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except Exception:
        return utc_str  # fallback: return UTC unchanged


def write_index(cache_subdir: Path, dt: date, record_count: int, data_type: str) -> Path:
    """Write a cache index.json sidecar."""
    idx = {
        "schema": SCHEMA_VERSION,
        "data_type": data_type,
        "date": dt.strftime("%Y-%m-%d"),
        "record_count": record_count,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    idx_path = cache_subdir / "index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    return idx_path
