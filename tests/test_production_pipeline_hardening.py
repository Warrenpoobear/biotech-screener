"""Fail-closed production and CI hardening.

Covers promotion gates and the same-day rerun skip. Does not exercise scoring.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.run_daily_production as rdp
from scripts import webhook_receiver

REPO = Path(__file__).resolve().parent.parent


def test_input_hash_changes_when_trial_records_change(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    prices = tmp_path / "price_history.csv"
    prices.write_text("date,ticker,close\n2026-11-20,XBI,100\n", encoding="utf-8")
    trials = data / "trial_records.json"
    trials.write_text("[]", encoding="utf-8")
    (data / "coinvest_signals.json").write_text("{}", encoding="utf-8")

    before = rdp.compute_daily_input_hash(data, prices)
    trials.write_text('[{"nct_id":"NCT1","last_update_posted":"2026-11-18"}]', encoding="utf-8")
    after = rdp.compute_daily_input_hash(data, prices)

    assert before != after
    assert rdp.should_skip_completed_manifest({"input_hash": before}, before) is True
    assert rdp.should_skip_completed_manifest({"input_hash": before}, after) is False
    assert rdp.should_skip_completed_manifest({}, after) is False
    assert rdp.should_skip_completed_manifest(None, after) is False


def test_override_file_change_invalidates_skip(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    prices = tmp_path / "price_history.csv"
    prices.write_text("date,ticker,close\n", encoding="utf-8")
    overrides = data / "catalyst_overrides.json"
    overrides.write_text("{}", encoding="utf-8")
    first = rdp.compute_daily_input_hash(data, prices)
    overrides.write_text('{"ACAD": {"note": "fixed"}}', encoding="utf-8")
    second = rdp.compute_daily_input_hash(data, prices)
    assert first != second


def _price_csv(path: Path) -> None:
    path.write_text("date,ticker,close\n2026-02-19,XBI,100\n", encoding="utf-8")


def test_strict_exit_one_with_rankings_is_fail_and_not_promoted(tmp_path: Path):
    """--strict exit 1 stays FAIL even when rankings.csv was written."""
    price_csv = tmp_path / "prices.csv"
    _price_csv(price_csv)
    cache_dir = tmp_path / "ctgov"
    cache_dir.mkdir()
    (cache_dir / "trial_records_2026-02-19.json").write_text("[]", encoding="utf-8")
    data_dir = tmp_path / "production_data"
    data_dir.mkdir()
    (data_dir / "universe.json").write_text(json.dumps([{"ticker": "XBI"}]), encoding="utf-8")
    (data_dir / "market_data.json").write_text(
        json.dumps([{"ticker": "XBI", "price": 100, "market_cap": 1e9, "collected_at": "2026-02-19"}]),
        encoding="utf-8",
    )
    (data_dir / "trial_records.json").write_text(
        json.dumps([{"ticker": "XBI", "last_update_posted": "2026-02-19"}]),
        encoding="utf-8",
    )
    final_dir = tmp_path / "snapshots"
    final_dir.mkdir()

    def _screen(as_of_date, data_dir, snapshot_dir, price_csv, **kwargs):
        out = Path(snapshot_dir) / as_of_date
        out.mkdir(parents=True, exist_ok=True)
        (out / "rankings.csv").write_text("ticker\nXBI\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="health FAIL")

    with (
        patch("tools.run_daily_production.run_screen", side_effect=_screen),
        patch("tools.poll_ctgov_daily.poll_ctgov_daily", return_value={"error": "test"}),
        patch("tools.collect_options_snapshot.run", return_value={"metadata": {"skipped": True}}),
        patch("tools.enrich_options_features.run", return_value={}),
        patch("tools.options_shadow_analysis.run", return_value={"market_implied_xs": {}}),
        patch(
            "tools.run_daily_production.get_git_info",
            return_value={"branch": "main", "commit_sha": "test", "dirty": False},
        ),
    ):
        manifest = rdp.run_daily(
            "2026-02-19",
            data_dir=data_dir,
            price_csv=price_csv,
            final_snapshots_dir=final_dir,
            skip_price_refresh=True,
            skip_pit_warm=True,
            ctgov_cache_dir=cache_dir,
        )

    assert manifest["overall_status"] == "FAIL"
    health = [g for g in manifest["gates"] if g["name"] == "phase2_health"]
    assert len(health) == 1
    assert health[0]["status"] == "FAIL"
    assert "NOT promoted" in health[0]["detail"]
    assert not (final_dir / "2026-02-19").exists()


def test_daily_workflow_refreshes_trials_before_cache_and_screen():
    text = (REPO / ".github/workflows/phase2-daily-production.yml").read_text(encoding="utf-8")
    refresh = text.index("collect_ctgov_data.py")
    warm = text.index("Warm ctgov cache")
    screen = text.index("Run production daily")
    assert refresh < warm < screen


def test_tastytrade_pin_matches_session_api_and_workflows_pass_secrets():
    """13.2.3 Session(is_test=...) reads TT_SECRET/TT_REFRESH. Gate threshold stays 40%."""
    req = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "tastytrade==13.2.3" in req
    lock = (REPO / "requirements.lock").read_text(encoding="utf-8")
    assert "tastytrade==13.2.3" in lock

    prod = (REPO / ".github/workflows/phase2-daily-production.yml").read_text(encoding="utf-8")
    legacy = (REPO / ".github/workflows/phase2-daily.yml").read_text(encoding="utf-8")
    for text, marker in (
        (prod, "Run production daily"),
        (legacy, "Run Phase-2 screen"),
    ):
        step = text[text.index(marker) : text.index(marker) + 800]
        assert "TT_SECRET: ${{ secrets.TT_SECRET }}" in step
        assert "TT_REFRESH: ${{ secrets.TT_REFRESH }}" in step

    assert rdp.GateConfig().hard_options_coverage_fail_pct == 40.0


def test_replay_golden_download_failure_exits_nonzero():
    text = (REPO / ".github/workflows/replay-regression.yml").read_text(encoding="utf-8")
    start = text.index("Download golden bundle")
    end = text.index("Verify baseline")
    step = text[start:end]
    assert "exit 1" in step
    assert "exit 0" not in step


def test_webhook_rejects_when_secret_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(webhook_receiver.WEBHOOK_SECRET_ENV, raising=False)
    handler = webhook_receiver.WebhookHandler.__new__(webhook_receiver.WebhookHandler)
    handler.path = "/webhook/fda"
    handler.headers = {}
    sent: dict[str, int] = {}

    def _send_error(code, message=None, explain=None):
        sent["code"] = code

    handler.send_error = _send_error
    handler.do_POST()
    assert sent["code"] == 401
    assert webhook_receiver.webhook_authorized("anything") is False


def test_webhook_requires_matching_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(webhook_receiver.WEBHOOK_SECRET_ENV, "s3cret-value")
    assert webhook_receiver.webhook_authorized(None) is False
    assert webhook_receiver.webhook_authorized("nope") is False
    assert webhook_receiver.webhook_authorized("s3cret-value") is True


def test_webhook_default_bind_and_startup_refusal(monkeypatch: pytest.MonkeyPatch):
    assert webhook_receiver.DEFAULT_BIND_HOST == "127.0.0.1"
    monkeypatch.delenv(webhook_receiver.WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["webhook_receiver.py"])
    assert webhook_receiver.main() == 2
