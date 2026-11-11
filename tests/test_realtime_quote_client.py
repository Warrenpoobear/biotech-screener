"""Unit tests for common.realtime_quote_client (Spec 063)."""

from __future__ import annotations

import pytest

from common.realtime_quote_client import (
    AlpacaQuoteClient,
    DevFallbackQuoteClient,
    NullQuoteClient,
    PolygonMassiveQuoteClient,
    QuoteRecord,
    _parse_alpaca_snapshot,
    _parse_snapshot,
    _realtime_tier_confirmed,
    _resolve_alpaca_credentials,
    _resolve_api_key,
    make_quote_client,
)

ALPACA_ENV_VARS = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
)
POLYGON_ENV_VARS = ("MASSIVE_API_KEY", "POLYGON_API_KEY", "BIOTECH_INTRADAY_REALTIME_TIER")
FALLBACK_ENV_VARS = ("BIOTECH_INTRADAY_DEV_FALLBACK",)


def _clear_env(monkeypatch, *groups):
    for group in groups:
        for name in group:
            monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Env resolution
# ---------------------------------------------------------------------------
def test_resolve_alpaca_credentials_requires_both(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS)
    assert _resolve_alpaca_credentials() is None
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    assert _resolve_alpaca_credentials() is None  # still missing secret
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    assert _resolve_alpaca_credentials() == ("k", "s")


def test_resolve_alpaca_credentials_alpaca_prefix_alias(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k2")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s2")
    assert _resolve_alpaca_credentials() == ("k2", "s2")


def test_resolve_api_key_prefers_massive(monkeypatch):
    monkeypatch.setenv("MASSIVE_ENABLED", "1")  # Massive is opt-in since 2026-11-11
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-123")
    monkeypatch.setenv("POLYGON_API_KEY", "polygon-456")
    assert _resolve_api_key() == "massive-123"


def test_resolve_api_key_falls_back_to_polygon(monkeypatch):
    monkeypatch.setenv("MASSIVE_ENABLED", "1")  # Massive is opt-in since 2026-11-11
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("POLYGON_API_KEY", "polygon-456")
    assert _resolve_api_key() == "polygon-456"


def test_resolve_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert _resolve_api_key() is None


@pytest.mark.parametrize(
    "val,expected", [("1", True), ("true", True), ("yes", True), ("0", False), ("", False), ("no", False)]
)
def test_realtime_tier_confirmation(monkeypatch, val, expected):
    monkeypatch.setenv("BIOTECH_INTRADAY_REALTIME_TIER", val)
    assert _realtime_tier_confirmed() is expected


# ---------------------------------------------------------------------------
# Factory behavior
# ---------------------------------------------------------------------------
def test_factory_returns_null_when_no_credentials(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS, POLYGON_ENV_VARS, FALLBACK_ENV_VARS)
    client = make_quote_client()
    assert isinstance(client, NullQuoteClient)
    h = client.health()
    assert h.mode == "no_credentials"
    assert h.ok is False


def test_factory_prefers_alpaca_over_polygon(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS, POLYGON_ENV_VARS, FALLBACK_ENV_VARS)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("MASSIVE_API_KEY", "m")
    client = make_quote_client()
    assert isinstance(client, AlpacaQuoteClient)
    assert client.health().mode == "live"


def test_factory_without_massive_opt_in_does_not_use_polygon(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS, POLYGON_ENV_VARS, FALLBACK_ENV_VARS)
    monkeypatch.delenv("MASSIVE_ENABLED", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY", "abc")
    from common.realtime_quote_client import _resolve_api_key

    assert _resolve_api_key() is None


def test_factory_falls_back_to_polygon_when_no_alpaca(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS, POLYGON_ENV_VARS, FALLBACK_ENV_VARS)
    monkeypatch.setenv("MASSIVE_ENABLED", "1")  # Massive is opt-in since 2026-11-11
    monkeypatch.setenv("MASSIVE_API_KEY", "abc")
    client = make_quote_client()
    assert isinstance(client, PolygonMassiveQuoteClient)
    # Key present but tier not confirmed → dry_run
    assert client.health().mode == "dry_run"


def test_factory_returns_dev_fallback_when_enabled(monkeypatch):
    _clear_env(monkeypatch, ALPACA_ENV_VARS, POLYGON_ENV_VARS, FALLBACK_ENV_VARS)
    monkeypatch.setenv("BIOTECH_INTRADAY_DEV_FALLBACK", "1")
    client = make_quote_client()
    assert isinstance(client, DevFallbackQuoteClient)


# ---------------------------------------------------------------------------
# NullQuoteClient
# ---------------------------------------------------------------------------
def test_null_client_returns_empty_dict():
    c = NullQuoteClient(reason="test")
    assert c.get_quotes(["AAPL", "XBI"]) == {}


# ---------------------------------------------------------------------------
# PolygonMassiveQuoteClient: dry-run when tier not confirmed
# ---------------------------------------------------------------------------
def test_polygon_client_dry_run_returns_empty_without_tier():
    c = PolygonMassiveQuoteClient(api_key="abc", tier_confirmed=False)
    # Must not make any network calls, must return empty
    assert c.get_quotes(["SRPT", "XBI"]) == {}


def test_polygon_client_health_dry_run():
    c = PolygonMassiveQuoteClient(api_key="abc", tier_confirmed=False)
    h = c.health()
    assert h.ok is False
    assert h.mode == "dry_run"


def test_polygon_client_health_live():
    c = PolygonMassiveQuoteClient(api_key="abc", tier_confirmed=True)
    h = c.health()
    assert h.ok is True
    assert h.mode == "live"


# ---------------------------------------------------------------------------
# Snapshot parser — attribute and dict forms
# ---------------------------------------------------------------------------
class _AttrSnap:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_parse_snapshot_dict_shape():
    snap = {
        "day": {"open": 10.0, "high": 12.0, "low": 9.5, "close": 11.0, "volume": 120000},
        "prev_day": {"close": 10.0},
        "last_trade": {"price": 11.25, "sip_timestamp": 1_700_000_000_000_000_000},
    }
    rec = _parse_snapshot("SRPT", snap)
    assert rec is not None
    assert rec.ticker == "SRPT"
    assert rec.last == 11.25
    assert rec.prev_close == 10.0
    assert rec.open == 10.0
    assert rec.volume == 120000
    assert rec.source == "massive"


def test_parse_snapshot_attribute_shape():
    snap = _AttrSnap(
        day=_AttrSnap(open=50.0, high=52.0, low=49.0, close=51.0, volume=500000),
        prev_day=_AttrSnap(close=50.0),
        last_trade=_AttrSnap(price=51.5, sip_timestamp=1_700_000_000_000_000_000),
    )
    rec = _parse_snapshot("XBI", snap)
    assert rec is not None
    assert rec.last == 51.5
    assert rec.prev_close == 50.0


def test_parse_snapshot_returns_none_on_missing_fields():
    snap = {"day": {}, "prev_day": {}}
    assert _parse_snapshot("FOO", snap) is None


def test_parse_snapshot_none_input():
    assert _parse_snapshot("FOO", None) is None


def test_quote_record_is_frozen():
    rec = QuoteRecord(
        ticker="X",
        last=1.0,
        prev_close=1.0,
        open=1.0,
        high=1.0,
        low=1.0,
        volume=0,
        avg_volume_20d=None,
        quote_ts="2026-04-17T14:30:00Z",
        market_status="open",
        source="massive",
    )
    with pytest.raises(Exception):
        rec.last = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Alpaca snapshot parser — production shape
# ---------------------------------------------------------------------------
def _alpaca_snap(
    *,
    last_price=107.5,
    prev_close=100.0,
    day_open=100.5,
    day_high=108.0,
    day_low=100.0,
    day_volume=1_200_000,
    ts="2026-04-17T14:30:00Z",
):
    return {
        "latestTrade": {"t": ts, "p": last_price, "s": 100},
        "latestQuote": {"t": ts, "bp": last_price - 0.05, "ap": last_price + 0.05},
        "minuteBar": {"t": ts, "o": last_price, "h": last_price, "l": last_price, "c": last_price, "v": 1500},
        "dailyBar": {
            "t": "2026-04-17T04:00:00Z",
            "o": day_open,
            "h": day_high,
            "l": day_low,
            "c": last_price,
            "v": day_volume,
        },
        "prevDailyBar": {"t": "2026-04-16T04:00:00Z", "o": 99.0, "h": 101.0, "l": 98.5, "c": prev_close, "v": 900_000},
    }


def test_parse_alpaca_snapshot_happy_path():
    rec = _parse_alpaca_snapshot("SRPT", _alpaca_snap())
    assert rec is not None
    assert rec.ticker == "SRPT"
    assert rec.last == 107.5
    assert rec.prev_close == 100.0
    assert rec.open == 100.5
    assert rec.high == 108.0
    assert rec.low == 100.0
    assert rec.volume == 1_200_000
    assert rec.quote_ts == "2026-04-17T14:30:00Z"
    assert rec.source == "alpaca"
    assert rec.market_status == "unknown"  # Alpaca snapshot doesn't carry this


def test_parse_alpaca_snapshot_returns_none_when_missing_prev_close():
    snap = _alpaca_snap()
    del snap["prevDailyBar"]
    assert _parse_alpaca_snapshot("SRPT", snap) is None


def test_parse_alpaca_snapshot_returns_none_on_empty():
    assert _parse_alpaca_snapshot("SRPT", None) is None
    assert _parse_alpaca_snapshot("SRPT", {}) is None


def test_parse_alpaca_snapshot_tolerates_missing_daily_bar():
    snap = _alpaca_snap()
    # daily bar missing → open/high/low default to last_price, volume=0
    del snap["dailyBar"]
    rec = _parse_alpaca_snapshot("SRPT", snap)
    assert rec is not None
    assert rec.last == 107.5
    assert rec.open == 107.5
    assert rec.high == 107.5
    assert rec.low == 107.5
    assert rec.volume == 0


# ---------------------------------------------------------------------------
# AlpacaQuoteClient — with injected HTTP session
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeSession:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status = status_code
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return _FakeResponse(self._payload, status_code=self._status)


def test_alpaca_client_fetches_and_parses(monkeypatch):
    snap = _alpaca_snap()
    session = _FakeSession({"SRPT": snap, "XBI": _alpaca_snap(last_price=99.0, prev_close=100.0)})
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["SRPT", "XBI"])
    assert "SRPT" in quotes
    assert "XBI" in quotes
    assert quotes["SRPT"].last == 107.5
    assert quotes["XBI"].last == 99.0
    # One batch call made
    assert len(session.calls) == 1
    # Auth headers present
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "k"
    assert session.calls[0]["headers"]["APCA-API-SECRET-KEY"] == "s"


def test_alpaca_client_skips_missing_symbols():
    session = _FakeSession({"SRPT": _alpaca_snap()})
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["SRPT", "DOES_NOT_EXIST"])
    assert "SRPT" in quotes
    assert "DOES_NOT_EXIST" not in quotes


def test_alpaca_client_batches_over_100_symbols():
    # 150 tickers → 2 batches
    snap = _alpaca_snap()
    payload = {f"T{i}": snap for i in range(150)}
    session = _FakeSession(payload)
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes([f"T{i}" for i in range(150)])
    assert len(quotes) == 150
    assert len(session.calls) == 2


def test_alpaca_client_handles_http_error_gracefully():
    session = _FakeSession({}, status_code=500)
    client = AlpacaQuoteClient("k", "s", http_session=session, max_retries=0)
    quotes = client.get_quotes(["SRPT"])
    assert quotes == {}


def test_alpaca_client_health_is_live():
    client = AlpacaQuoteClient("k", "s")
    h = client.health()
    assert h.ok is True
    assert h.mode == "live"
    assert "alpaca" in h.detail.lower()


# ---------------------------------------------------------------------------
# Phase 1.5 — fixture-backed integration test against real Alpaca payload
# ---------------------------------------------------------------------------
# Fixture captured 2026-04-17 via GET /v2/stocks/snapshots?symbols=XBI,SRPT,ZZZZZ
# using the paper-trading account's live data entitlement. XBI + SRPT return
# full snapshots; ZZZZZ (intentional missing-symbol probe) is omitted from the
# response entirely — validating the client's silent-miss handling.
def _load_alpaca_fixture():
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "alpaca" / "snapshots_XBI_SRPT_ZZZZZ.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_alpaca_fixture_parses_xbi_benchmark():
    """Real Alpaca snapshot: XBI benchmark ETF, full payload."""
    payload = _load_alpaca_fixture()
    session = _FakeSession(payload)
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["XBI", "SRPT", "ZZZZZ"])

    assert "XBI" in quotes
    xbi = quotes["XBI"]
    # Values from the frozen response — these pin the parser against reality
    assert xbi.last == 138.6
    assert xbi.prev_close == 135.445
    assert xbi.open == 137.725
    assert xbi.high == 139.19
    assert xbi.low == 137.21
    assert xbi.volume == 756318
    assert xbi.source == "alpaca"
    # latestTrade.t is ISO8601 with nanosecond precision; preserved as-is
    assert xbi.quote_ts.startswith("2026-04-17T20:35:34")


def test_alpaca_fixture_parses_srpt_biotech():
    """Real Alpaca snapshot: SRPT biotech, full payload."""
    payload = _load_alpaca_fixture()
    session = _FakeSession(payload)
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["XBI", "SRPT", "ZZZZZ"])

    assert "SRPT" in quotes
    srpt = quotes["SRPT"]
    assert srpt.last == 21.55
    assert srpt.prev_close == 21.15
    assert srpt.open == 21.28
    assert srpt.high == 21.69
    assert srpt.low == 21.085
    assert srpt.volume == 95148
    assert srpt.source == "alpaca"


def test_alpaca_fixture_missing_symbol_silently_dropped():
    """ZZZZZ is not in Alpaca's response payload at all — client must skip it."""
    payload = _load_alpaca_fixture()
    assert "ZZZZZ" not in payload  # pin the real API behavior
    session = _FakeSession(payload)
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["XBI", "SRPT", "ZZZZZ"])
    assert "ZZZZZ" not in quotes
    # And the other two are still there
    assert set(quotes.keys()) == {"XBI", "SRPT"}


def test_alpaca_fixture_relative_move_vs_xbi_math():
    """Validate the downstream math against real numbers.

    Using the frozen fixture:
        XBI abs  = (138.60 / 135.445 - 1) * 100  ≈ +2.33%
        SRPT abs = (21.55  / 21.15   - 1) * 100  ≈ +1.89%
        SRPT rel vs XBI                          ≈ -0.44pp (underperformer)
    """
    payload = _load_alpaca_fixture()
    session = _FakeSession(payload)
    client = AlpacaQuoteClient("k", "s", http_session=session)
    quotes = client.get_quotes(["XBI", "SRPT"])

    xbi = quotes["XBI"]
    srpt = quotes["SRPT"]

    xbi_abs_pct = 100.0 * (xbi.last / xbi.prev_close - 1.0)
    srpt_abs_pct = 100.0 * (srpt.last / srpt.prev_close - 1.0)
    rel_vs_xbi = srpt_abs_pct - xbi_abs_pct

    assert xbi_abs_pct == pytest.approx(2.33, abs=0.01)
    assert srpt_abs_pct == pytest.approx(1.89, abs=0.01)
    assert rel_vs_xbi == pytest.approx(-0.44, abs=0.01)
