"""Massive API dropped 2026-11-11: off unless MASSIVE_ENABLED=1."""

import pytest

from common import options_history_massive as ohm


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    monkeypatch.setenv("POLYGON_API_KEY", "p")
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "s")
    monkeypatch.delenv("MASSIVE_ENABLED", raising=False)


def test_disabled_by_default_ignores_credentials(creds):
    assert not ohm.massive_enabled()
    cfg = ohm._get_config()
    assert cfg["api_key"] == "" and cfg["s3_access_key_id"] == "" and cfg["s3_secret_access_key"] == ""
    assert cfg["cache_dir"]  # local PIT cache still addressable


def test_enabled_opt_in(creds, monkeypatch):
    monkeypatch.setenv("MASSIVE_ENABLED", "1")
    assert ohm._get_config()["api_key"] == "k"


def test_realtime_client_gets_no_key_when_disabled(creds):
    from common.realtime_quote_client import _resolve_api_key

    assert _resolve_api_key() is None


def test_hedge_report_treats_massive_unavailable(creds):
    from tools.biotech_hedge_report import _massive_available

    assert _massive_available() is False


def test_chain_snapshot_refuses_without_network(creds):
    with pytest.raises(EnvironmentError):
        ohm.fetch_chain_snapshot("XBI")
