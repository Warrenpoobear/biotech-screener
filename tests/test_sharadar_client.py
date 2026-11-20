"""Tests for common/sharadar_client.py (direct api.sharadar.com client).

All HTTP is faked via the injectable ``fetch`` callable — no live calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from common.sharadar_client import API_KEY_ENV_VAR, BASE_URL, SharadarAuthError, SharadarClient, SharadarError

FAKE_KEY = "unit-test-key-not-real"  # pragma: allowlist secret


class FakeFetch:
    """Records calls and replays scripted (status, body) responses."""

    def __init__(self, responses: List[Tuple[int, Any]]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, params, headers, timeout):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        if not self._responses:
            raise AssertionError("unexpected extra request")
        status, body = self._responses.pop(0)
        if isinstance(body, Exception):
            raise body
        return status, json.dumps(body).encode("utf-8")


def _rows(n: int, start: int = 0) -> List[Dict[str, Any]]:
    return [{"ticker": "VKTX", "date": f"d{i:05d}"} for i in range(start, start + n)]


def _client(fetch: FakeFetch, **kw) -> SharadarClient:
    return SharadarClient(api_key=FAKE_KEY, fetch=fetch, sleep=lambda s: None, **kw)


class TestAuth:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(SharadarAuthError, match=API_KEY_ENV_VAR):
            SharadarClient(fetch=FakeFetch([]))

    def test_key_read_from_env(self, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV_VAR, FAKE_KEY)
        fetch = FakeFetch([(200, {"count": 0, "data": []})])
        SharadarClient(fetch=fetch).get_table("SEP", filters={"ticker": "VKTX"})
        assert fetch.calls[0]["headers"]["x-api-key"] == FAKE_KEY

    def test_key_sent_as_header_never_in_query(self):
        fetch = FakeFetch([(200, {"count": 0, "data": []})])
        _client(fetch).get_table("SEP", filters={"ticker": "VKTX"})
        call = fetch.calls[0]
        assert call["url"] == f"{BASE_URL}/data/SEP"
        assert FAKE_KEY not in json.dumps(call["params"])
        assert "api_key" not in call["params"]

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_raises_without_leaking_key(self, status):
        body = {"error": "Exceeds free tier", "description": "Please sign up at /subscribe."}
        fetch = FakeFetch([(status, body)])
        with pytest.raises(SharadarAuthError) as exc:
            _client(fetch).get_table("SEP", filters={"ticker": "VKTX"})
        assert "Exceeds free tier" in str(exc.value)
        assert FAKE_KEY not in str(exc.value)

    def test_repr_hides_key(self):
        assert FAKE_KEY not in repr(_client(FakeFetch([])))


class FakeServer:
    """Minimal Sharadar stand-in: filters by ticker / date.gte / date.lte and
    truncates at ``limit`` exactly like the live API (count == len(data))."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, params, headers, timeout):
        self.calls.append(dict(params))
        out = list(self.rows)
        if "ticker" in params:
            wanted = set(params["ticker"].split(","))
            out = [r for r in out if r["ticker"] in wanted]
        if "date.gte" in params:
            out = [r for r in out if r["date"] >= params["date.gte"]]
        if "date.lte" in params:
            out = [r for r in out if r["date"] <= params["date.lte"]]
        out = out[: int(params["limit"])]
        return 200, json.dumps({"count": len(out), "data": out}).encode("utf-8")


def _daily_rows(tickers, start_day=1, days=10):
    return [{"ticker": t, "date": f"2026-01-{d:02d}"} for t in tickers for d in range(start_day, start_day + days)]


class TestTruncation:
    """Offset paging is unstable server-side (2026-11-04 backfill: 107 rows
    duplicated AND a VKTX row skipped). Every request must fit in one page;
    a full page means 'truncated' and the query is split, never offset-paged."""

    def test_full_page_raises_truncated(self):
        from common.sharadar_client import SharadarTruncatedError

        fetch = FakeFetch([(200, {"count": 3, "data": _rows(3)})])
        with pytest.raises(SharadarTruncatedError):
            _client(fetch, page_limit=3).get_table("SEP", filters={"ticker": "VKTX"})

    def test_never_sends_offset(self):
        server = FakeServer(_daily_rows(["VKTX"], days=10))
        _client(server, page_limit=4).fetch("SEP", tickers=["VKTX"], date_range=("2026-01-01", "2026-01-10"))
        assert all("offset" not in c for c in server.calls)

    def test_short_page_single_request(self):
        fetch = FakeFetch([(200, {"count": 1, "data": _rows(1)})])
        assert len(_client(fetch, page_limit=10).get_table("SEP")) == 1
        assert len(fetch.calls) == 1

    def test_date_range_split_until_each_request_fits(self):
        data = _daily_rows(["VKTX", "AMGN"], days=10)  # 20 rows
        server = FakeServer(data)
        rows = _client(server, page_limit=4).fetch(
            "SEP", tickers=["VKTX", "AMGN"], date_range=("2026-01-01", "2026-01-10")
        )
        assert sorted((r["ticker"], r["date"]) for r in rows) == sorted((r["ticker"], r["date"]) for r in data)
        assert len(rows) == len(data)  # no duplicates, nothing skipped

    def test_single_day_overflow_splits_tickers(self):
        data = _daily_rows(["A", "B", "C", "D", "E"], days=1)  # 5 rows on one day
        server = FakeServer(data)
        rows = _client(server, page_limit=2).fetch(
            "SEP", tickers=["A", "B", "C", "D", "E"], date_range=("2026-01-01", "2026-01-01")
        )
        assert sorted(r["ticker"] for r in rows) == ["A", "B", "C", "D", "E"]

    def test_unsplittable_overflow_raises(self):
        from common.sharadar_client import SharadarTruncatedError

        server = FakeServer([{"ticker": "VKTX", "date": "2026-01-01", "n": i} for i in range(5)])
        with pytest.raises(SharadarTruncatedError):
            _client(server, page_limit=2).fetch("SF1", tickers=["VKTX"], date_range=("2026-01-01", "2026-01-01"))

    def test_no_date_range_overflow_splits_tickers(self):
        data = [{"ticker": t, "date": "2026-01-01"} for t in "ABCD"]
        server = FakeServer(data)
        rows = _client(server, page_limit=2).fetch("TICKERS", tickers=list("ABCD"))
        assert sorted(r["ticker"] for r in rows) == list("ABCD")

    def test_date_range_combined_with_other_filters(self):
        server = FakeServer(_daily_rows(["VKTX"], days=3))
        _client(server).fetch(
            "SEP", tickers=["VKTX"], filters={"lastupdated.gte": "2026-01-02"}, date_range=("2026-01-01", "2026-01-03")
        )
        c = server.calls[0]
        assert (c["lastupdated.gte"], c["date.gte"], c["date.lte"]) == ("2026-01-02", "2026-01-01", "2026-01-03")

    def test_whole_market_when_no_tickers(self):
        server = FakeServer(_daily_rows(["A", "B"], days=2))
        rows = _client(server, page_limit=3).fetch("SEP", date_range=("2026-01-01", "2026-01-02"))
        assert len(rows) == 4
        assert all("ticker" not in c for c in server.calls)


class TestFiltersAndFields:
    def test_filters_and_fields_passed_through(self):
        fetch = FakeFetch([(200, {"count": 0, "data": []})])
        _client(fetch).get_table(
            "SF1",
            filters={"ticker": "VKTX", "lastupdated.gte": "2026-10-01"},
            fields=["ticker", "dimension", "date"],
        )
        p = fetch.calls[0]["params"]
        assert p["ticker"] == "VKTX"
        assert p["lastupdated.gte"] == "2026-10-01"
        assert p["fields"] == "ticker,dimension,date"
        assert p["format"] == "json"

    def test_ticker_batching(self):
        tickers = [f"T{i}" for i in range(5)]
        fetch = FakeFetch(
            [
                (200, {"count": 1, "data": [{"ticker": "T0"}]}),
                (200, {"count": 1, "data": [{"ticker": "T2"}]}),
                (200, {"count": 1, "data": [{"ticker": "T4"}]}),
            ]
        )
        rows = _client(fetch).get_table_for_tickers("SEP", tickers, batch_size=2)
        assert [c["params"]["ticker"] for c in fetch.calls] == ["T0,T1", "T2,T3", "T4"]
        assert [r["ticker"] for r in rows] == ["T0", "T2", "T4"]

    def test_ticker_batches_respect_200_char_param_limit(self):
        """Live API rejects ticker params > 200 chars (HTTP 400, seen 2026-11-04)."""
        from common.sharadar_client import MAX_TICKER_PARAM_CHARS, MAX_TICKERS_PER_REQUEST

        tickers = [f"TK{i:03d}" for i in range(357)]  # 5 chars each, like the real universe
        fetch = FakeFetch([(200, {"count": 0, "data": []})] * 20)
        _client(fetch).get_table_for_tickers("SEP", tickers)  # default batch_size
        params = [c["params"]["ticker"] for c in fetch.calls]
        assert all(len(p) <= MAX_TICKER_PARAM_CHARS for p in params)
        # Live API also caps at 30 tickers/request (HTTP 400, seen 2026-11-04).
        assert MAX_TICKERS_PER_REQUEST == 30
        assert all(len(p.split(",")) <= MAX_TICKERS_PER_REQUEST for p in params)
        assert sorted(t for p in params for t in p.split(",")) == sorted(tickers)

    def test_oversized_batch_size_is_clamped(self):
        tickers = [f"T{i}" for i in range(31)]
        fetch = FakeFetch([(200, {"count": 0, "data": []})] * 2)
        _client(fetch).get_table_for_tickers("SEP", tickers, batch_size=500)
        assert [len(c["params"]["ticker"].split(",")) for c in fetch.calls] == [30, 1]

    def test_ticker_batching_dedupes_and_sorts(self):
        fetch = FakeFetch([(200, {"count": 0, "data": []})])
        _client(fetch).get_table_for_tickers("SEP", ["VKTX", "AMGN", "VKTX"], batch_size=10)
        assert fetch.calls[0]["params"]["ticker"] == "AMGN,VKTX"

    def test_empty_ticker_list_makes_no_request(self):
        fetch = FakeFetch([])
        assert _client(fetch).get_table_for_tickers("SEP", []) == []


class TestRetries:
    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_transient_status_retried(self, status):
        fetch = FakeFetch([(status, {"error": "x"}), (200, {"count": 1, "data": _rows(1)})])
        assert len(_client(fetch).get_table("SEP")) == 1
        assert len(fetch.calls) == 2

    def test_network_error_retried(self):
        fetch = FakeFetch([(0, OSError("conn reset")), (200, {"count": 0, "data": []})])
        assert _client(fetch).get_table("SEP") == []
        assert len(fetch.calls) == 2

    def test_gives_up_after_max_retries(self):
        fetch = FakeFetch([(503, {"error": "x"})] * 3)
        with pytest.raises(SharadarError, match="503"):
            _client(fetch, max_retries=2).get_table("SEP")
        assert len(fetch.calls) == 3

    def test_backoff_sleeps_between_attempts(self):
        slept: List[float] = []
        fetch = FakeFetch([(429, {}), (429, {}), (200, {"count": 0, "data": []})])
        SharadarClient(api_key=FAKE_KEY, fetch=fetch, sleep=slept.append, backoff_seconds=1.0).get_table("SEP")
        assert slept == [1.0, 2.0]

    def test_other_4xx_not_retried(self):
        fetch = FakeFetch([(400, {"error": "bad filter"})])
        with pytest.raises(SharadarError, match="bad filter"):
            _client(fetch).get_table("SEP")
        assert len(fetch.calls) == 1

    def test_malformed_body_raises(self):
        fetch = FakeFetch([(200, {"unexpected": True})])
        with pytest.raises(SharadarError, match="data"):
            _client(fetch).get_table("SEP")
