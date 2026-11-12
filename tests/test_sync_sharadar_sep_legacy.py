"""sec_13f/scripts/sync_sharadar_sep.py must use the direct Sharadar API.

The script predates the direct subscription and targeted Nasdaq Data Link
(data.nasdaq.com), which a sharadar.com key cannot use.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "sec_13f" / "scripts" / "sync_sharadar_sep.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("sync_sharadar_sep_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def fetch(self, table, *, tickers=None, filters=None, fields=None, date_range=None, **kw):
        self.calls.append(
            {
                "table": table,
                "tickers": tickers,
                "filters": dict(filters or {}),
                "fields": fields,
                "date_range": date_range,
            }
        )
        return list(self.rows)


def test_no_nasdaq_data_link_references(mod):
    text = SCRIPT.read_text(encoding="utf-8")
    assert "data.nasdaq.com" not in text
    assert "NASDAQ_DATA_LINK_API_KEY" not in text


def test_api_key_from_sharadar_env(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "API_KEY_FILE", tmp_path / "missing.txt")
    monkeypatch.setenv("SHARADAR_API_KEY", " k-123 ")  # pragma: allowlist secret
    assert mod.get_api_key() == "k-123"
    monkeypatch.delenv("SHARADAR_API_KEY")
    with pytest.raises(ValueError, match="SHARADAR_API_KEY"):
        mod.get_api_key()


def test_fetch_whole_market_window(mod):
    rows = [{"ticker": "VKTX", "date": "2026-11-03", "closeadj": 26.09}]
    client = _FakeClient(rows)
    out = mod.fetch_sep_data("ignored", date_gte="2026-11-01", date_lte="2026-11-03", client=client)
    assert out == rows
    assert client.calls == [
        {
            "table": "SEP",
            "tickers": None,
            "filters": {},
            "fields": ["ticker", "date", "closeadj"],
            "date_range": ("2026-11-01", "2026-11-03"),
        }
    ]


def test_open_ended_window_gets_explicit_bounds(mod):
    client = _FakeClient([])
    mod.fetch_sep_data("ignored", date_gte="2026-11-01", client=client)
    start, end = client.calls[0]["date_range"]
    assert start == "2026-11-01" and end >= "2026-11-01"
    mod.fetch_sep_data("ignored", client=client)
    assert client.calls[1]["date_range"][0] == mod.FULL_HISTORY_START


def test_fetch_ticker_subset(mod):
    client = _FakeClient([])
    mod.fetch_sep_data("ignored", date_gte="2026-11-01", tickers=["VKTX", "AMGN"], client=client)
    assert client.calls[0]["tickers"] == ["VKTX", "AMGN"]
