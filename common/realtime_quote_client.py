"""Provider-agnostic near-real-time intraday quote adapter.

Used by `tools/build_intraday_mover_watch.py` (Spec 063) to fetch last-trade,
previous-close, and day OHLC/volume for a watchlist of equities plus the
sector benchmark (XBI) on a polling cadence.

Design invariants (from Spec 063)
---------------------------------
- Alpaca Basic is the production primary (15-min delayed REST snapshots).
  No separate exchange license required.
- Polygon / Massive is an optional upgrade path when a paid consolidated
  feed is later acquired.
- `wake_robin_data_pipeline/market_data_provider.py` and yfinance are dev
  fallback only, gated by `BIOTECH_INTRADAY_DEV_FALLBACK=1`, and must never
  be the primary live source for alerting.
- When no credentials are present, the agent operates in a no-op /
  artifact-only dry-run mode. It must not crash and must not emit alerts.

Env vars
--------
APCA_API_KEY_ID                 Alpaca API key (primary production credential).
APCA_API_SECRET_KEY             Alpaca API secret (primary production credential).
APCA_API_DATA_URL               Override Alpaca data base URL (default:
                                https://data.alpaca.markets).
MASSIVE_API_KEY                 Paid-upgrade credential (Polygon/Massive).
                                Matches existing integration in
                                common/options_history_massive.py.
POLYGON_API_KEY                 Accepted alias for MASSIVE_API_KEY.
BIOTECH_INTRADAY_DEV_FALLBACK   "1" or "true" to allow yfinance-backed dev
                                fallback when no provider key is set. For
                                local smoke tests only.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol

logger = logging.getLogger("realtime_quote_client")

MarketStatus = Literal["pre", "open", "post", "closed", "unknown"]
QuoteSource = Literal["alpaca", "polygon", "massive", "dev_fallback", "none"]


@dataclass(frozen=True)
class QuoteRecord:
    ticker: str
    last: float
    prev_close: float
    open: float
    high: float
    low: float
    volume: int
    avg_volume_20d: Optional[int]
    quote_ts: str  # ISO8601 UTC
    market_status: MarketStatus
    source: QuoteSource


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    mode: Literal["live", "dry_run", "dev_fallback", "no_credentials"]
    detail: str


class RealtimeQuoteClient(Protocol):
    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]: ...

    def health(self) -> HealthStatus: ...


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------
def _resolve_alpaca_credentials() -> Optional[tuple]:
    """Return (key_id, secret_key) if both set, else None."""
    key_id = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    if key_id and secret:
        return (key_id, secret)
    return None


def _resolve_api_key() -> Optional[str]:
    """Return the configured Polygon/Massive key, or None."""
    return os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or None


def _realtime_tier_confirmed() -> bool:
    """Legacy gate from the pre-Alpaca spec. Retained for PolygonMassiveQuoteClient."""
    return os.environ.get("BIOTECH_INTRADAY_REALTIME_TIER", "").lower() in ("1", "true", "yes")


def _dev_fallback_enabled() -> bool:
    return os.environ.get("BIOTECH_INTRADAY_DEV_FALLBACK", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Null / no-op client (default when no credentials or tier not confirmed)
# ---------------------------------------------------------------------------
class NullQuoteClient:
    """No-op client. Returns empty dict; used when credentials missing.

    Phase 1 scaffolding default. The intraday builder inspects `.health()`
    and writes `status=NO_DATA` artifacts instead of crashing or emitting
    alerts.
    """

    def __init__(self, reason: str):
        self._reason = reason

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]:
        _ = list(tickers)  # force-consume for logging
        logger.info("NullQuoteClient active (%s); returning no quotes", self._reason)
        return {}

    def health(self) -> HealthStatus:
        return HealthStatus(ok=False, mode="no_credentials", detail=self._reason)


# ---------------------------------------------------------------------------
# Polygon/Massive REST client (production path)
# ---------------------------------------------------------------------------
class PolygonMassiveQuoteClient:
    """Production intraday quote client against the Polygon/Massive REST API.

    Uses the `massive` Python SDK that is already wired in this repo
    (see `common/options_history_massive.py`). The same RESTClient exposes
    equity snapshot endpoints.

    Phase 1 status: scaffolded and gated. It does NOT execute live HTTP
    calls unless:
      1. An API key is present (MASSIVE_API_KEY or POLYGON_API_KEY), AND
      2. BIOTECH_INTRADAY_REALTIME_TIER=1 is explicitly set.

    If either condition is missing, the client reports mode="dry_run" and
    returns an empty dict. This preserves read-only invariants until the
    account tier is confirmed.
    """

    DEFAULT_TIMEOUT_S = 10.0
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BACKOFF_S = 1.0

    def __init__(
        self,
        api_key: str,
        *,
        tier_confirmed: bool,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    ):
        self._api_key = api_key
        self._tier_confirmed = tier_confirmed
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._rest_client = None

    def _get_rest_client(self):
        if self._rest_client is None:
            try:
                from massive import RESTClient  # type: ignore
            except ImportError as exc:
                raise RuntimeError("`massive` Python SDK is required for PolygonMassiveQuoteClient") from exc
            self._rest_client = RESTClient(self._api_key)
        return self._rest_client

    def _fetch_snapshot(self, ticker: str) -> Optional[QuoteRecord]:
        """Fetch a single-ticker snapshot with retry. Returns None on failure."""
        client = self._get_rest_client()
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                snap = client.get_snapshot_ticker(  # type: ignore[attr-defined]
                    market_type="stocks",
                    ticker=ticker,
                )
                return _parse_snapshot(ticker, snap)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
        logger.warning("snapshot fetch failed for %s: %s", ticker, last_err)
        return None

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]:
        if not self._tier_confirmed:
            logger.info("PolygonMassiveQuoteClient: tier not confirmed; operating in dry-run")
            return {}
        out: Dict[str, QuoteRecord] = {}
        for t in tickers:
            rec = self._fetch_snapshot(t)
            if rec is not None:
                out[t] = rec
        return out

    def health(self) -> HealthStatus:
        if not self._tier_confirmed:
            return HealthStatus(
                ok=False,
                mode="dry_run",
                detail="API key present but BIOTECH_INTRADAY_REALTIME_TIER not set",
            )
        return HealthStatus(ok=True, mode="live", detail="polygon/massive live")


def _parse_snapshot(ticker: str, snap) -> Optional[QuoteRecord]:
    """Map a Polygon/Massive snapshot object to a QuoteRecord.

    Tolerant of both dict-style and attribute-style payloads (the `massive`
    SDK returns dataclass-like objects; older polygon clients return dicts).
    """
    if snap is None:
        return None

    def g(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    day = g(snap, "day") or {}
    prev_day = g(snap, "prev_day") or g(snap, "prevDay") or {}
    last_trade = g(snap, "last_trade") or g(snap, "lastTrade") or {}
    last_quote = g(snap, "last_quote") or g(snap, "lastQuote") or {}

    last_price = g(last_trade, "price") or g(last_trade, "p") or g(day, "close") or g(day, "c")
    prev_close = g(prev_day, "close") or g(prev_day, "c")
    day_open = g(day, "open") or g(day, "o")
    day_high = g(day, "high") or g(day, "h")
    day_low = g(day, "low") or g(day, "l")
    day_volume = g(day, "volume") or g(day, "v") or 0

    if last_price is None or prev_close is None:
        return None

    # Quote timestamp: prefer last_trade ts, else last_quote ts, else now
    ts_ns = g(last_trade, "sip_timestamp") or g(last_trade, "t") or g(last_quote, "sip_timestamp") or g(last_quote, "t")
    if ts_ns:
        # Massive SDK returns nanoseconds since epoch
        from datetime import datetime, timezone

        quote_ts = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()
    else:
        from datetime import datetime, timezone

        quote_ts = datetime.now(timezone.utc).isoformat()

    market_status: MarketStatus = g(snap, "market_status", "unknown") or "unknown"
    if market_status not in ("pre", "open", "post", "closed"):
        market_status = "unknown"

    return QuoteRecord(
        ticker=ticker,
        last=float(last_price),
        prev_close=float(prev_close),
        open=float(day_open) if day_open is not None else float(last_price),
        high=float(day_high) if day_high is not None else float(last_price),
        low=float(day_low) if day_low is not None else float(last_price),
        volume=int(day_volume),
        avg_volume_20d=None,  # populated by caller from a separate source
        quote_ts=quote_ts,
        market_status=market_status,
        source="massive",
    )


# ---------------------------------------------------------------------------
# Alpaca Basic — production primary (15-min delayed REST snapshots)
# ---------------------------------------------------------------------------
class AlpacaQuoteClient:
    """Alpaca Basic REST snapshot client.

    Alpaca's free Trading API plan includes 15-minute-delayed US equity
    snapshots via REST (`GET /v2/stocks/snapshots?symbols=...`), with no
    separate exchange license required. IEX-only real-time is available
    via websocket but is not used by this client in Phase 2.

    Account requirement: an Alpaca account at alpaca.markets. The Basic
    (free) plan is sufficient. Paper and live keys both work for data
    access, but paper keys are recommended since this client never places
    orders.

    Response mapping (per symbol):
        latestTrade.p   → last
        prevDailyBar.c  → prev_close
        dailyBar.{o,h,l,v} → open / high / low / volume
        latestTrade.t   → quote_ts
    """

    DEFAULT_BASE_URL = "https://data.alpaca.markets"
    DEFAULT_TIMEOUT_S = 10.0
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BACKOFF_S = 1.0
    SNAPSHOT_BATCH_SIZE = 100  # Alpaca caps a single snapshot call

    def __init__(
        self,
        api_key_id: str,
        secret_key: str,
        *,
        base_url: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        http_session=None,
    ):
        self._key_id = api_key_id
        self._secret = secret_key
        self._base_url = base_url or os.environ.get("APCA_API_DATA_URL", self.DEFAULT_BASE_URL)
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._session = http_session  # dependency injection for tests

    def _get_session(self):
        if self._session is not None:
            return self._session
        # lazy import
        import requests  # type: ignore[import-untyped]

        self._session = requests.Session()
        return self._session

    def _fetch_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Call Alpaca snapshots endpoint for up to SNAPSHOT_BATCH_SIZE symbols."""
        url = f"{self._base_url}/v2/stocks/snapshots"
        headers = {
            "APCA-API-KEY-ID": self._key_id,
            "APCA-API-SECRET-KEY": self._secret,
            "Accept": "application/json",
        }
        params = {"symbols": ",".join(symbols)}
        session = self._get_session()
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = session.get(url, headers=headers, params=params, timeout=self._timeout_s)
                if resp.status_code == 200:
                    return resp.json() or {}
                if resp.status_code == 429 and attempt < self._max_retries:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
                    continue
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
        logger.warning("Alpaca snapshot fetch failed: %s", last_err)
        return {}

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]:
        symbols = [t for t in tickers if t]
        out: Dict[str, QuoteRecord] = {}
        for i in range(0, len(symbols), self.SNAPSHOT_BATCH_SIZE):
            batch = symbols[i : i + self.SNAPSHOT_BATCH_SIZE]
            payload = self._fetch_batch(batch)
            for sym in batch:
                snap = payload.get(sym)
                rec = _parse_alpaca_snapshot(sym, snap)
                if rec is not None:
                    out[sym] = rec
        return out

    def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            mode="live",
            detail="alpaca basic (15-min delayed REST snapshots)",
        )


def _parse_alpaca_snapshot(ticker: str, snap: Optional[Dict[str, Any]]) -> Optional[QuoteRecord]:
    """Parse an Alpaca snapshot payload into a QuoteRecord.

    Snapshot shape (v2):
        {
          "latestTrade": {"t": iso8601, "p": float, "s": int, ...},
          "latestQuote": {"t": iso8601, "bp": float, "ap": float, ...},
          "minuteBar":   {"t": iso8601, "o": float, "h": float, "l": float, "c": float, "v": int},
          "dailyBar":    {"t": iso8601, "o": float, "h": float, "l": float, "c": float, "v": int},
          "prevDailyBar":{"t": iso8601, "o": float, "h": float, "l": float, "c": float, "v": int}
        }
    """
    if not snap:
        return None

    latest_trade = snap.get("latestTrade") or {}
    daily = snap.get("dailyBar") or {}
    prev_daily = snap.get("prevDailyBar") or {}

    last_price = latest_trade.get("p") or daily.get("c")
    prev_close = prev_daily.get("c")
    if last_price is None or prev_close is None:
        return None

    day_open = daily.get("o") or last_price
    day_high = daily.get("h") or last_price
    day_low = daily.get("l") or last_price
    day_volume = daily.get("v") or 0
    quote_ts = latest_trade.get("t") or daily.get("t")
    if not quote_ts:
        from datetime import datetime, timezone

        quote_ts = datetime.now(timezone.utc).isoformat()

    return QuoteRecord(
        ticker=ticker,
        last=float(last_price),
        prev_close=float(prev_close),
        open=float(day_open),
        high=float(day_high),
        low=float(day_low),
        volume=int(day_volume),
        avg_volume_20d=None,
        quote_ts=str(quote_ts),
        market_status="unknown",  # Alpaca snapshot does not carry market status
        source="alpaca",
    )


# ---------------------------------------------------------------------------
# Dev fallback (yfinance) — NEVER production primary
# ---------------------------------------------------------------------------
class DevFallbackQuoteClient:
    """yfinance-backed dev fallback. Gated by BIOTECH_INTRADAY_DEV_FALLBACK=1.

    Not suitable for production alerting. Use for local smoke tests only.
    """

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            logger.warning("yfinance not installed; dev fallback unavailable")
            return {}
        from datetime import datetime, timezone

        out: Dict[str, QuoteRecord] = {}
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                info = tk.fast_info  # minimal call
                last = float(getattr(info, "last_price", None) or 0.0)
                prev = float(getattr(info, "previous_close", None) or 0.0)
                if last <= 0 or prev <= 0:
                    continue
                day_open = float(getattr(info, "open", None) or last)
                day_high = float(getattr(info, "day_high", None) or last)
                day_low = float(getattr(info, "day_low", None) or last)
                vol = int(getattr(info, "last_volume", None) or 0)
                out[t] = QuoteRecord(
                    ticker=t,
                    last=last,
                    prev_close=prev,
                    open=day_open,
                    high=day_high,
                    low=day_low,
                    volume=vol,
                    avg_volume_20d=None,
                    quote_ts=datetime.now(timezone.utc).isoformat(),
                    market_status="unknown",
                    source="dev_fallback",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("yfinance fetch failed for %s: %s", t, exc)
        return out

    def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            mode="dev_fallback",
            detail="yfinance — not suitable for production alerting",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_quote_client() -> RealtimeQuoteClient:
    """Construct the appropriate quote client for the current environment.

    Decision tree (Spec 063 provider selection order):
      1. APCA_API_KEY_ID + APCA_API_SECRET_KEY present → AlpacaQuoteClient
         (production primary; 15-min delayed REST snapshots).
      2. else MASSIVE_API_KEY or POLYGON_API_KEY present →
         PolygonMassiveQuoteClient (paid-upgrade path; enters dry-run if
         BIOTECH_INTRADAY_REALTIME_TIER is not set).
      3. else if BIOTECH_INTRADAY_DEV_FALLBACK=1 → DevFallbackQuoteClient
         (yfinance, local smoke tests only).
      4. else → NullQuoteClient.
    """
    alpaca = _resolve_alpaca_credentials()
    if alpaca is not None:
        key_id, secret = alpaca
        return AlpacaQuoteClient(api_key_id=key_id, secret_key=secret)

    api_key = _resolve_api_key()
    if api_key:
        return PolygonMassiveQuoteClient(
            api_key=api_key,
            tier_confirmed=_realtime_tier_confirmed(),
        )
    if _dev_fallback_enabled():
        return DevFallbackQuoteClient()
    return NullQuoteClient(
        reason="no APCA_API_KEY_ID/APCA_API_SECRET_KEY, no MASSIVE_API_KEY/POLYGON_API_KEY, dev fallback not enabled"
    )


__all__ = [
    "QuoteRecord",
    "HealthStatus",
    "RealtimeQuoteClient",
    "NullQuoteClient",
    "AlpacaQuoteClient",
    "PolygonMassiveQuoteClient",
    "DevFallbackQuoteClient",
    "make_quote_client",
]
