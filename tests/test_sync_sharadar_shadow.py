"""Tests for tools/sync_sharadar_shadow.py (Sharadar shadow cache sync).

Uses a fake client — no live HTTP. Covers backfill vs incremental planning,
PIT-preserving raw slices, curated upsert semantics, failure isolation,
determinism, and the guard that production code never reads the shadow cache.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from common.sharadar_client import SharadarAuthError, SharadarError
from tools import sync_sharadar_shadow as sync

REPO_ROOT = Path(__file__).resolve().parents[1]
T1 = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)


class FakeClient:
    """Serves rows per table; records (table, tickers, filters, date_range) calls."""

    def __init__(self, rows_by_table: Dict[str, List[Dict[str, Any]]], fail: Dict[str, Exception] = None):
        self.rows_by_table = rows_by_table
        self.fail = fail or {}
        self.calls: List[Dict[str, Any]] = []

    def fetch(self, table, *, tickers=None, filters=None, fields=None, date_range=None, **kw):
        tickers = sorted(set(tickers))
        self.calls.append(
            {"table": table, "tickers": tickers, "filters": dict(filters or {}), "date_range": date_range}
        )
        if table in self.fail:
            raise self.fail[table]
        return [dict(r) for r in self.rows_by_table.get(table, []) if r.get("ticker") in tickers]


def _sep(ticker, date, close, lastupdated):
    return {"ticker": ticker, "date": date, "close": close, "closeadj": close, "lastupdated": lastupdated}


def _run(tmp_path, client, tables=("SEP",), tickers=("VKTX", "AMGN"), now=T1, **kw):
    return sync.run_sync(
        client=client,
        cache_root=tmp_path,
        tables=list(tables),
        universe_tickers=list(tickers),
        now=now,
        backfill_start="2010-01-01",
        **kw,
    )


def _curated(tmp_path, table) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "curated" / f"{table}.csv.gz", dtype=str)


def _manifest(tmp_path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]


class TestUniverseLoader:
    def test_includes_all_statuses(self, tmp_path):
        u = tmp_path / "universe.json"
        u.write_text(
            json.dumps(
                [
                    {"ticker": "VKTX", "status": "active"},
                    {"ticker": "CELG", "status": "delisted"},
                    {"ticker": "vktx", "status": "active"},
                    {"name": "no ticker"},
                ]
            )
        )
        assert sync.load_universe_tickers(u) == ["CELG", "VKTX"]


class TestBackfillThenIncremental:
    def test_first_run_backfills_from_start_date(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        result = _run(tmp_path, client)
        assert result.exit_code == 0
        assert client.calls == [
            {
                "table": "SEP",
                "tickers": ["AMGN", "VKTX"],
                "filters": {},
                "date_range": ("2010-01-01", "2026-11-04"),
            }
        ]
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["SEP"]["tickers"] == ["AMGN", "VKTX"]
        assert state["SEP"]["watermark"] == "2026-11-03"

    def test_second_run_is_incremental_with_overlap(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        _run(tmp_path, client)
        client.calls.clear()
        _run(tmp_path, client, now=T2)
        overlap = sync.TABLE_SPECS["SEP"].overlap_days
        expected = (pd.Timestamp("2026-11-03") - pd.Timedelta(days=overlap)).strftime("%Y-%m-%d")
        assert client.calls == [
            {
                "table": "SEP",
                "tickers": ["AMGN", "VKTX"],
                "filters": {"lastupdated.gte": expected},
                "date_range": ("2010-01-01", "2026-11-05"),
            }
        ]

    def test_new_universe_ticker_gets_its_own_backfill(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        _run(tmp_path, client, tickers=("VKTX",))
        client.calls.clear()
        _run(tmp_path, client, tickers=("VKTX", "ORIC"), now=T2)
        by_tickers = {tuple(c["tickers"]): c for c in client.calls}
        assert by_tickers[("ORIC",)]["filters"] == {}
        assert by_tickers[("ORIC",)]["date_range"] == ("2010-01-01", "2026-11-05")
        assert "lastupdated.gte" in by_tickers[("VKTX",)]["filters"]
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["SEP"]["tickers"] == ["ORIC", "VKTX"]

    def test_date_incremental_table_uses_date_filter(self, tmp_path):
        rows = {"SF3A": [{"ticker": "VKTX", "date": "2026-09-30", "shrholders": "123"}]}
        client = FakeClient(rows)
        _run(tmp_path, client, tables=("SF3A",), tickers=("VKTX",))
        client.calls.clear()
        _run(tmp_path, client, tables=("SF3A",), tickers=("VKTX",), now=T2)
        overlap = sync.TABLE_SPECS["SF3A"].overlap_days
        expected = (pd.Timestamp("2026-09-30") - pd.Timedelta(days=overlap)).strftime("%Y-%m-%d")
        assert client.calls[0]["filters"] == {}
        assert client.calls[0]["date_range"] == (expected, "2026-11-05")

    def test_snapshot_table_has_no_filters(self, tmp_path):
        rows = {
            "TICKERS": [{"table": "stocks", "permaticker": "157194", "ticker": "VKTX", "lastupdated": "2026-11-01"}]
        }
        client = FakeClient(rows)
        _run(tmp_path, client, tables=("TICKERS",), tickers=("VKTX",))
        _run(tmp_path, client, tables=("TICKERS",), tickers=("VKTX",), now=T2)
        assert [(c["filters"], c["date_range"]) for c in client.calls] == [({}, None), ({}, None)]

    def test_benchmark_table_uses_benchmark_tickers_not_universe(self, tmp_path):
        client = FakeClient({"SFP": [_sep("XBI", "2026-11-03", 154.97, "2026-11-03")]})
        _run(tmp_path, client, tables=("SFP",), tickers=("VKTX",))
        assert client.calls[0]["tickers"] == sorted(sync.BENCHMARK_TICKERS)


class TestPitPreservation:
    def test_raw_slices_are_immutable_and_manifested(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        _run(tmp_path, client)
        raw1 = sorted((tmp_path / "raw" / "SEP").iterdir())
        first_bytes = raw1[0].read_bytes()

        client.rows_by_table["SEP"] = [_sep("VKTX", "2026-11-03", 26.50, "2026-11-04")]  # restated
        _run(tmp_path, client, now=T2)
        raw2 = sorted((tmp_path / "raw" / "SEP").iterdir())
        assert len(raw2) == 2
        assert raw2[0].read_bytes() == first_bytes

        m = _manifest(tmp_path)
        assert [e["fetched_at_utc"] for e in m] == ["2026-11-04T12:00:00Z", "2026-11-05T12:00:00Z"]
        assert [e["mode"] for e in m] == ["backfill", "incremental"]
        assert all(e["rows"] == 1 and len(e["sha256"]) == 64 for e in m)
        with gzip.open(raw2[0], "rt", encoding="utf-8") as f:
            first_raw_row = json.loads(f.readline())
        assert first_raw_row["close"] == 26.09

    def test_manifest_never_contains_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHARADAR_API_KEY", "sekrit-test-value")  # pragma: allowlist secret
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        _run(tmp_path, client)
        for p in tmp_path.rglob("*"):
            if p.is_file() and p.suffix != ".gz":
                assert "sekrit-test-value" not in p.read_text()

    def test_empty_fetch_writes_no_raw_file_but_is_manifested(self, tmp_path):
        client = FakeClient({"SEP": []})
        _run(tmp_path, client)
        assert not (tmp_path / "raw" / "SEP").exists() or not any((tmp_path / "raw" / "SEP").iterdir())
        assert _manifest(tmp_path)[0]["rows"] == 0


class TestCuratedUpsert:
    def test_restatement_replaces_by_key_newest_lastupdated_wins(self, tmp_path):
        client = FakeClient(
            {
                "SEP": [
                    _sep("VKTX", "2026-11-02", 26.83, "2026-11-02"),
                    _sep("VKTX", "2026-11-03", 26.09, "2026-11-03"),
                ]
            }
        )
        _run(tmp_path, client)
        client.rows_by_table["SEP"] = [_sep("VKTX", "2026-11-03", 26.50, "2026-11-04")]
        _run(tmp_path, client, now=T2)
        df = _curated(tmp_path, "SEP")
        assert len(df) == 2
        assert not df.duplicated(["ticker", "date"]).any()
        assert df.set_index("date").loc["2026-11-03", "close"] == "26.5"
        assert df.set_index("date").loc["2026-11-02", "close"] == "26.83"

    def test_stale_refetch_does_not_overwrite_newer(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.50, "2026-11-04")]})
        _run(tmp_path, client)
        client.rows_by_table["SEP"] = [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]
        _run(tmp_path, client, now=T2)
        assert _curated(tmp_path, "SEP").loc[0, "close"] == "26.5"

    def test_curated_is_sorted_and_deterministic(self, tmp_path):
        rows = {
            "SEP": [
                _sep("VKTX", "2026-11-03", 26.09, "2026-11-03"),
                _sep("AMGN", "2026-11-03", 461.26, "2026-11-03"),
                _sep("AMGN", "2026-11-02", 460.00, "2026-11-03"),
            ]
        }
        a, b = tmp_path / "a", tmp_path / "b"
        _run(a, FakeClient(rows))
        _run(b, FakeClient({"SEP": list(reversed(rows["SEP"]))}))
        ba = (a / "curated" / "SEP.csv.gz").read_bytes()
        assert ba == (b / "curated" / "SEP.csv.gz").read_bytes()
        df = _curated(a, "SEP")
        assert list(zip(df.ticker, df.date)) == [
            ("AMGN", "2026-11-02"),
            ("AMGN", "2026-11-03"),
            ("VKTX", "2026-11-03"),
        ]


class TestFailureHandling:
    def test_one_table_failing_does_not_block_others(self, tmp_path):
        client = FakeClient(
            {"DAILY": [{"ticker": "VKTX", "date": "2026-11-03", "marketcap": 3327.1, "lastupdated": "2026-11-03"}]},
            fail={"SEP": SharadarError("SEP: HTTP 503")},
        )
        result = _run(tmp_path, client, tables=("SEP", "DAILY"), tickers=("VKTX",))
        assert result.exit_code == 1
        assert result.tables["SEP"]["status"] == "error"
        assert result.tables["DAILY"]["status"] == "ok"
        state = json.loads((tmp_path / "state.json").read_text())
        assert "SEP" not in state and "DAILY" in state

    def test_auth_error_aborts_run(self, tmp_path):
        client = FakeClient({}, fail={"SEP": SharadarAuthError("SEP: HTTP 403: Exceeds free tier")})
        result = _run(tmp_path, client, tables=("SEP", "DAILY"), tickers=("VKTX",))
        assert result.exit_code == 2
        assert [c["table"] for c in client.calls] == ["SEP"]

    def test_status_file_written(self, tmp_path):
        client = FakeClient({"SEP": [_sep("VKTX", "2026-11-03", 26.09, "2026-11-03")]})
        _run(tmp_path, client, tickers=("VKTX",))
        status = json.loads((tmp_path / "status_latest.json").read_text())
        assert status["run_at_utc"] == "2026-11-04T12:00:00Z"
        assert status["tables"]["SEP"] == {"status": "ok", "rows": 1, "requests": 1}
        assert status["exit_code"] == 0

    def test_unknown_table_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="BOGUS"):
            _run(tmp_path, FakeClient({}), tables=("BOGUS",))


class TestShadowIsolation:
    """The Sharadar cache is shadow-only during the NO_MODEL_CHANGE window.
    Promoting it to a production input is a governed price-source change."""

    PRODUCTION_FILES = [
        "run_screen.py",
        "decision_engine.py",
        "selector_engine.py",
        "ranker_engine.py",
        "ranker_v2_pairwise.py",
        "tools/run_daily_production.py",
        *sorted(p.name for p in REPO_ROOT.glob("module_*.py")),
    ]

    @pytest.mark.parametrize("rel", PRODUCTION_FILES)
    def test_production_file_does_not_touch_sharadar_shadow(self, rel):
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.skip(f"{rel} not present")
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "sharadar_client" not in text
        assert "sync_sharadar_shadow" not in text
        assert "cache/sharadar" not in text
