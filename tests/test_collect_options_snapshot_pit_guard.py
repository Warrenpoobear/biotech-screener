"""PIT guard for tools/collect_options_snapshot.py.

tastytrade market metrics are current-state only, so the collector must refuse
to write a snapshot labelled with any date other than today (America/New_York).
"""

from datetime import date

import pytest

import tools.collect_options_snapshot as cos


@pytest.fixture
def no_fetch(monkeypatch, tmp_path):
    """Fail loudly if the guard lets a fetch through; redirect writes to tmp."""

    async def _boom(*a, **k):
        raise AssertionError("fetch must not run for a non-current as_of_date")

    monkeypatch.setattr(cos, "_fetch", _boom)
    monkeypatch.setattr(cos, "REPO_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("as_of", ["2026-10-07", "2026-11-10", "2026-11-12"])
def test_refuses_non_current_date(no_fetch, as_of):
    out = cos.run(as_of, 200, dry_run=False, verbose=False, today=date(2026, 11, 11))
    meta = out["metadata"]
    assert meta["skipped"] is True
    assert meta["skip_reason"] == "NON_CURRENT_AS_OF_DATE"
    assert meta["as_of_date"] == as_of
    assert meta["market_today"] == "2026-11-11"
    assert out["tickers"] == {}
    assert not list(no_fetch.rglob("options_snapshot_*.json"))


def test_current_date_proceeds(monkeypatch, tmp_path):
    (tmp_path / "production_data").mkdir()
    (tmp_path / "production_data" / "universe.json").write_text('["AAA"]')
    monkeypatch.setattr(cos, "REPO_ROOT", tmp_path)

    async def _fake_fetch(tickers, batch_size, verbose):
        return {}

    monkeypatch.setattr(cos, "_fetch", _fake_fetch)
    out = cos.run("2026-11-11", 200, dry_run=False, verbose=False, today=date(2026, 11, 11))
    assert not out["metadata"].get("skipped")
    assert (tmp_path / "production_data" / "options_snapshot_2026-11-11.json").exists()
    assert (tmp_path / "production_data" / "options_snapshot_latest.json").exists()


def test_market_today_uses_new_york():
    assert str(cos.MARKET_TZ) == "America/New_York"
