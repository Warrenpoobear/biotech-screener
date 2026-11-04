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


class TestPagination:
    """Sharadar reports count == len(data) even when a page is truncated at
    ``limit``, so a full page must be treated as 'maybe more'."""

    def test_full_page_triggers_next_offset(self):
        fetch = FakeFetch(
            [
                (200, {"count": 3, "data": _rows(3, 0)}),
                (200, {"count": 3, "data": _rows(3, 3)}),
                (200, {"count": 1, "data": _rows(1, 6)}),
            ]
        )
        rows = _client(fetch, page_limit=3).get_table("SEP", filters={"ticker": "VKTX"})
        assert len(rows) == 7
        assert [c["params"]["offset"] for c in fetch.calls] == [0, 3, 6]
        assert all(c["params"]["limit"] == 3 for c in fetch.calls)

    def test_exactly_full_last_page_then_empty(self):
        fetch = FakeFetch(
            [
                (200, {"count": 2, "data": _rows(2)}),
                (200, {"count": 0, "data": []}),
            ]
        )
        rows = _client(fetch, page_limit=2).get_table("SEP")
        assert len(rows) == 2
        assert len(fetch.calls) == 2

    def test_short_first_page_single_request(self):
        fetch = FakeFetch([(200, {"count": 1, "data": _rows(1)})])
        assert len(_client(fetch, page_limit=10).get_table("SEP")) == 1
        assert len(fetch.calls) == 1


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
