"""Tests for tools/agent_heartbeat_checks.py.

Covers the production_qa heartbeat, which previously looked for the wrong
filename pattern and the wrong JSON fields — silently reporting STALE even
when production_qa_check wrote a healthy RED/YELLOW/GREEN report.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def hb_mod(tmp_path, monkeypatch):
    import tools.agent_heartbeat_checks as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SNAPSHOT_DIR", tmp_path / "data" / "snapshots")
    monkeypatch.setattr(mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(mod, "LOGS_DIR", tmp_path / "logs")
    (tmp_path / "data" / "snapshots").mkdir(parents=True)
    (tmp_path / "artifacts" / "production_qa").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    return mod


def _write_qa_report(tmp_path: Path, ds: str, verdict: str, fails: list[str]) -> None:
    """Write a report in the exact format production_qa_check.py emits."""
    report = {
        "schema": "production_qa.v1",
        "as_of_date": ds,
        "generated_at": "2026-04-15T00:00:00+00:00",
        "verdict": verdict,
        "n_checks": 9,
        "n_pass": 9 - len(fails),
        "n_fail": len(fails),
        "checks": [{"check": name, "status": "FAIL", "detail": ""} for name in fails]
        + [{"check": "other", "status": "PASS", "detail": ""}],
    }
    out = tmp_path / "artifacts" / "production_qa" / f"{ds}_report.json"
    out.write_text(json.dumps(report), encoding="utf-8")


def test_production_qa_finds_report_with_correct_filename(hb_mod, tmp_path):
    """Heartbeat must look for `{ds}_report.json` (the actual production_qa output)."""
    ds = "2026-04-15"
    _write_qa_report(tmp_path, ds, verdict="GREEN", fails=[])

    result = hb_mod.check_production_qa(date.fromisoformat(ds))
    assert result.status == "OK", f"Expected OK for GREEN verdict; got {result.status} ({result.detail})"


def test_production_qa_red_verdict_reports_fail(hb_mod, tmp_path):
    ds = "2026-04-15"
    _write_qa_report(tmp_path, ds, verdict="RED", fails=["sidecars", "schema", "tracebacks"])

    result = hb_mod.check_production_qa(date.fromisoformat(ds))
    assert result.status == "WARN"  # current code WARNs on any anomaly
    assert result.anomalies
    joined = " ".join(result.anomalies)
    assert "VERDICT_RED" in joined
    # Must name at least one failing check so ops knows what to look at
    assert "sidecars" in joined or "schema" in joined


def test_production_qa_yellow_verdict_reports_warn(hb_mod, tmp_path):
    ds = "2026-04-15"
    _write_qa_report(tmp_path, ds, verdict="YELLOW", fails=["tracebacks"])

    result = hb_mod.check_production_qa(date.fromisoformat(ds))
    assert result.status == "WARN"
    joined = " ".join(result.anomalies)
    assert "YELLOW" in joined


def test_production_qa_missing_report_is_stale(hb_mod):
    result = hb_mod.check_production_qa(date.fromisoformat("2026-04-15"))
    assert result.status == "STALE"


def test_production_qa_corrupt_report_is_flagged(hb_mod, tmp_path):
    ds = "2026-04-15"
    (tmp_path / "artifacts" / "production_qa" / f"{ds}_report.json").write_text("{ malformed", encoding="utf-8")
    result = hb_mod.check_production_qa(date.fromisoformat(ds))
    assert result.status == "WARN"
    joined = " ".join(result.anomalies)
    assert "CORRUPT" in joined


def test_news_digest_press_release_freshness_parses_classified_prefix(hb_mod, tmp_path, monkeypatch):
    """Press-release source freshness should parse classified/classified_YYYY-MM-DD.jsonl."""
    from datetime import datetime

    digest_dir = tmp_path / "artifacts" / "news_digest"
    digest_dir.mkdir(parents=True)
    (digest_dir / "biotech_news_digest_2026-05-07_evening.json").write_text("{}")
    classified_dir = tmp_path / "data" / "press_releases" / "classified"
    classified_dir.mkdir(parents=True)
    (classified_dir / "classified_2026-05-01.jsonl").write_text("{}\n")

    # Before 19:00 ET path checks yesterday's digest (present); isolates press-release staleness.
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 5, 8, 10, 0, 0)

    monkeypatch.setattr(hb_mod, "datetime", _FixedDatetime)

    result = hb_mod.check_herald_news_pipeline(date.fromisoformat("2026-05-08"))

    assert result.agent == "herald"
    assert result.status == "WARN"
    assert any("STALE_SOURCE" in a and "7d" in a for a in result.anomalies)


def test_shadow_monitor_performance_csv_fallback_uses_real_header(hb_mod, tmp_path):
    """CSV fallback should use first line as header even when file has many rows."""
    ds = "2026-05-08"
    monitor_dir = tmp_path / "artifacts" / "shadow_monitor"
    monitor_dir.mkdir(parents=True)
    (monitor_dir / f"{ds}_monitor.json").write_text(json.dumps({"attention": "LOW", "alerts": []}))
    policy_dir = tmp_path / "artifacts" / "policy_shadow" / "tier_weighted"
    policy_dir.mkdir(parents=True)
    (policy_dir / f"{ds}_comparison.json").write_text("{}")
    perf_dir = tmp_path / "artifacts" / "live_shadow"
    perf_dir.mkdir(parents=True)
    rows = ["date,pnl"]
    rows.extend([f"2026-05-0{i},0.01" for i in range(1, 4)])
    rows.extend([f"2026-05-0{i},-0.01" for i in range(4, 9)])
    (perf_dir / "performance.csv").write_text("\n".join(rows) + "\n")

    result = hb_mod.check_shadow_monitor(date.fromisoformat(ds))

    assert result.status == "WARN"
    assert any("DRAWDOWN_STREAK" in a for a in result.anomalies)


# ── Fleet receipt (Fix #3) ────────────────────────────────────


def _mk_result(hb_mod, agent, status, detail="", anomalies=None):
    return hb_mod.CheckResult(agent, status, detail, anomalies or [])


def test_derive_verdict_red_on_missing_snapshot(hb_mod):
    results = [_mk_result(hb_mod, "qa", "OK")]
    assert hb_mod._derive_verdict(results, {"missing_count": 0}, snapshot_ok=False) == "RED"


def test_derive_verdict_red_on_fail(hb_mod):
    results = [_mk_result(hb_mod, "qa", "FAIL", "pipeline crash")]
    assert hb_mod._derive_verdict(results, {"missing_count": 0}, snapshot_ok=True) == "RED"


def test_derive_verdict_red_on_coverage_gap(hb_mod):
    results = [_mk_result(hb_mod, "qa", "OK")]
    assert hb_mod._derive_verdict(results, {"missing_count": 3}, snapshot_ok=True) == "RED"


def test_terminal_unsupervised_agent_does_not_create_coverage_gap(hb_mod, tmp_path, monkeypatch):
    """ops_supervisor is the terminal layer and should not RED fleet coverage."""
    registry = {
        "agents": {
            "qa": {
                "status": "active",
                "cadence": "daily_after_production",
                "artifact_paths": ["agents/qa/memory/"],
                "supervised_by_orchestrator": True,
            },
            "ops_supervisor": {
                "status": "active",
                "cadence": "daily_after_production",
                "artifact_paths": ["artifacts/ops_supervisor/"],
                "supervised_by_orchestrator": False,
                "notes": "terminal interpretive layer",
            },
        }
    }
    (tmp_path / "agents" / "qa" / "memory").mkdir(parents=True)
    (tmp_path / "agents" / "qa" / "memory" / "2026-05-08.md").write_text("ok")
    reg_path = tmp_path / "agents" / "AGENT_REGISTRY.json"
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(hb_mod, "REGISTRY_PATH", reg_path)
    monkeypatch.setattr(hb_mod, "SPECIALIZED_CHECKS", {})

    results, counts = hb_mod.run_registry_checks(date.fromisoformat("2026-05-08"))

    assert counts["missing_count"] == 0
    assert any(r.agent == "ops_supervisor" and r.status == "SKIP" for r in results)
    assert hb_mod._derive_verdict(results, counts, snapshot_ok=True) == "GREEN"


def test_hermes_on_demand_agents_skip_heartbeat_freshness(hb_mod, tmp_path, monkeypatch):
    registry = {
        "agents": {
            "hermes-held-spec-ledger": {
                "status": "active",
                "cadence": "on_demand",
                "artifact_paths": ["artifacts/ops/held_spec_ledger/"],
                "supervised_by_orchestrator": True,
                "llm_policy": "none",
            },
            "qa": {
                "status": "active",
                "cadence": "daily_after_production",
                "artifact_paths": ["agents/qa/memory/"],
                "supervised_by_orchestrator": True,
            },
        }
    }
    (tmp_path / "agents" / "qa" / "memory").mkdir(parents=True)
    (tmp_path / "agents" / "qa" / "memory" / "2026-05-08.md").write_text("ok")
    reg_path = tmp_path / "agents" / "AGENT_REGISTRY.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(hb_mod, "REGISTRY_PATH", reg_path)
    monkeypatch.setattr(hb_mod, "SPECIALIZED_CHECKS", {"qa": lambda dt: hb_mod.CheckResult("qa", "OK")})

    results, counts = hb_mod.run_registry_checks(date.fromisoformat("2026-05-08"))

    hermes = next(r for r in results if r.agent == "hermes-held-spec-ledger")
    assert hermes.status == "SKIP"
    assert "run_job.py" in hermes.detail
    assert counts["stale_count"] == 0


def test_nonterminal_active_unsupervised_agent_still_counts_as_coverage_gap(hb_mod, tmp_path, monkeypatch):
    registry = {
        "agents": {
            "some_agent": {
                "status": "active",
                "cadence": "daily_after_production",
                "artifact_paths": ["artifacts/some_agent/"],
                "supervised_by_orchestrator": False,
            }
        }
    }
    reg_path = tmp_path / "agents" / "AGENT_REGISTRY.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(hb_mod, "REGISTRY_PATH", reg_path)
    monkeypatch.setattr(hb_mod, "SPECIALIZED_CHECKS", {})

    _results, counts = hb_mod.run_registry_checks(date.fromisoformat("2026-05-08"))

    assert counts["missing_count"] == 1


def test_cloud_environment_downgrades_artifact_staleness_to_skip(hb_mod, tmp_path, monkeypatch):
    registry = {
        "agents": {
            "artifact_agent": {
                "status": "active",
                "cadence": "daily_after_production",
                "artifact_paths": ["artifacts/artifact_agent/"],
                "supervised_by_orchestrator": True,
            }
        }
    }
    reg_path = tmp_path / "agents" / "AGENT_REGISTRY.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(hb_mod, "REGISTRY_PATH", reg_path)
    monkeypatch.setattr(hb_mod, "SPECIALIZED_CHECKS", {})
    monkeypatch.setattr(hb_mod, "is_cloud_agent_environment", lambda: True)

    results, counts = hb_mod.run_registry_checks(date.fromisoformat("2026-05-08"))

    assert counts["stale_count"] == 0
    result = results[0]
    assert result.status == "SKIP"
    assert "UNKNOWN_CLOUD_ENV" in result.detail
    assert "no artifacts at any declared path" in result.detail


def test_derive_verdict_yellow_on_warn_or_stale(hb_mod):
    r = [_mk_result(hb_mod, "a", "WARN"), _mk_result(hb_mod, "b", "OK")]
    assert hb_mod._derive_verdict(r, {"missing_count": 0}, snapshot_ok=True) == "YELLOW"
    r = [_mk_result(hb_mod, "a", "STALE"), _mk_result(hb_mod, "b", "OK")]
    assert hb_mod._derive_verdict(r, {"missing_count": 0}, snapshot_ok=True) == "YELLOW"


def test_derive_verdict_green_when_all_ok(hb_mod):
    r = [_mk_result(hb_mod, "a", "OK"), _mk_result(hb_mod, "b", "SKIP")]
    assert hb_mod._derive_verdict(r, {"missing_count": 0}, snapshot_ok=True) == "GREEN"


def test_write_fleet_receipt_creates_file_with_core_sections(hb_mod, tmp_path):
    ds = "2026-04-24"
    (tmp_path / "data" / "snapshots" / ds).mkdir(parents=True)
    (tmp_path / "data" / "snapshots" / ds / "rankings.csv").write_text("x\n")

    results = [
        _mk_result(hb_mod, "qa", "OK", "snapshot valid"),
        _mk_result(hb_mod, "ic_health_monitor", "FAIL", "attention=HIGH", ["SIGNAL_ALERT: foo"]),
        _mk_result(hb_mod, "sentinel", "STALE", "10d > 2d"),
        _mk_result(hb_mod, "bioshort_watch", "SKIP", "unsupervised: cosmetic"),
    ]
    counts = {
        "active_count": 27,
        "monitored_count": 26,
        "stale_count": 1,
        "missing_count": 1,
        "deprecated_count": 2,
    }
    out_path = hb_mod.write_fleet_receipt(results, counts, date.fromisoformat(ds))

    assert out_path.exists()
    assert out_path.name == f"{ds}_receipt.md"
    # Canonical location every consumer reads (cron_watchdog, ops_supervisor,
    # fleet_ops_status, telegram_command_handler) — must be artifacts/heartbeat/.
    assert out_path == hb_mod.ARTIFACTS_DIR / "heartbeat" / f"{ds}_receipt.md"
    # Historical agent-memory audit copy is also written.
    assert (hb_mod.REPO_ROOT / "agents" / "fleet_steward" / "memory" / f"{ds}_receipt.md").exists()
    text = out_path.read_text()
    assert "# Fleet Receipt" in text
    assert "Verdict: RED" in text
    assert "## Pipeline" in text
    assert "## Fleet (AGENT_REGISTRY.json)" in text
    assert "Active: 27" in text
    assert "Coverage gap" in text
    assert "## Agent Status" in text
    assert "### FAIL" in text
    assert "ic_health_monitor" in text
    assert "SIGNAL_ALERT: foo" in text
    assert "## Escalated to ops" in text


def test_write_fleet_receipt_green_verdict_minimal(hb_mod, tmp_path):
    ds = "2026-04-24"
    (tmp_path / "data" / "snapshots" / ds).mkdir(parents=True)
    (tmp_path / "data" / "snapshots" / ds / "rankings.csv").write_text("x\n")

    results = [_mk_result(hb_mod, "qa", "OK"), _mk_result(hb_mod, "ops", "OK")]
    counts = {
        "active_count": 2,
        "monitored_count": 2,
        "stale_count": 0,
        "missing_count": 0,
        "deprecated_count": 0,
    }
    out_path = hb_mod.write_fleet_receipt(results, counts, date.fromisoformat(ds))
    text = out_path.read_text()
    assert "Verdict: GREEN" in text
    assert "## Escalated to ops" not in text


def test_write_fleet_receipt_cloud_missing_snapshot_is_unknown_not_red(hb_mod, monkeypatch):
    ds = "2026-04-24"
    monkeypatch.setattr(hb_mod, "is_cloud_agent_environment", lambda: True)

    results = [_mk_result(hb_mod, "qa", "SKIP", "UNKNOWN_CLOUD_ENV: artifact unavailable")]
    counts = {
        "active_count": 1,
        "monitored_count": 1,
        "stale_count": 0,
        "missing_count": 0,
        "deprecated_count": 0,
    }
    out_path = hb_mod.write_fleet_receipt(results, counts, date.fromisoformat(ds))

    text = out_path.read_text()
    assert "Verdict: GREEN" in text
    assert "Today's snapshot (2026-04-24): UNKNOWN_CLOUD_ENV" in text


# ── check_ic_health: carried-alert muffle (P1 #1) ─────────────


def _write_dashboard(hb_mod, ds: str, signals: dict, attention: str = "MEDIUM") -> Path:
    """Write a minimal ic_dashboard.json for the given date + signals."""
    dash_dir = hb_mod.ARTIFACTS_DIR / "ic_dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)
    dash_path = dash_dir / f"{ds}_dashboard.json"
    dash_path.write_text(
        json.dumps({"attention": attention, "signals": signals}),
        encoding="utf-8",
    )
    return dash_path


class TestCheckIcHealthCarriedMuffle:
    """P1 #1 — carried-alert muffle for ic_health_monitor.

    Signals in IC_HEALTH_CARRIED_ALERTS before expires_after downgrade to WARN
    with [CARRIED] tag (no FAIL escalation). Any other signal at ALERT, OR a
    carried signal after expiry, must still trigger FAIL.
    """

    @pytest.fixture
    def carried_inst_delta(self, hb_mod, monkeypatch):
        monkeypatch.setitem(
            hb_mod.IC_HEALTH_CARRIED_ALERTS,
            "inst_delta_z",
            {
                "expires_after": "2026-05-15",
                "reason": "test fixture: expected cohort distortion",
            },
        )

    def test_carried_alert_pre_expiry_yields_warn(self, hb_mod, carried_inst_delta):
        ds = "2026-05-06"  # before inst_delta_z expires_after=2026-05-15
        _write_dashboard(
            hb_mod,
            ds,
            {"inst_delta_z": {"health": "ALERT", "ic": -0.1}},
        )
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "WARN"
        assert any("[CARRIED]" in a and "inst_delta_z" in a for a in result.anomalies)
        # Must not falsely report a clean ALERT signal
        assert not any(a == "SIGNAL_ALERT: inst_delta_z" for a in result.anomalies)

    def test_carried_alert_post_expiry_yields_fail(self, hb_mod, carried_inst_delta):
        ds = "2026-05-16"  # day after inst_delta_z expires_after
        _write_dashboard(
            hb_mod,
            ds,
            {"inst_delta_z": {"health": "ALERT", "ic": -0.1}},
        )
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "FAIL"
        assert "SIGNAL_ALERT: inst_delta_z" in result.anomalies
        # Must not be tagged [CARRIED] post-expiry
        assert not any("[CARRIED]" in a for a in result.anomalies)

    def test_inst_delta_alert_fails_without_muffle_entry(self, hb_mod):
        """Production dict is empty after muffle expiry — ALERT must FAIL."""
        ds = "2026-05-30"
        _write_dashboard(hb_mod, ds, {"inst_delta_z": {"health": "ALERT", "ic": -0.1}})
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "FAIL"
        assert "SIGNAL_ALERT: inst_delta_z" in result.anomalies

    def test_unmuffled_signal_alert_preserves_fail(self, hb_mod):
        """Any signal NOT in IC_HEALTH_CARRIED_ALERTS at ALERT must still FAIL."""
        ds = "2026-05-06"
        _write_dashboard(
            hb_mod,
            ds,
            {"coinvest_score_z": {"health": "ALERT", "ic": -0.05}},
        )
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "FAIL"
        assert "SIGNAL_ALERT: coinvest_score_z" in result.anomalies

    def test_mixed_carried_and_unmuffled_yields_fail(self, hb_mod, carried_inst_delta):
        """If both a carried alert and an unmuffled alert are present, the
        unmuffled one still drives FAIL — carried tag does not mask other
        ALERTs."""
        ds = "2026-05-06"
        _write_dashboard(
            hb_mod,
            ds,
            {
                "inst_delta_z": {"health": "ALERT", "ic": -0.1},
                "coinvest_score_z": {"health": "CRITICAL", "ic": -0.2},
            },
        )
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "FAIL"
        assert any("[CARRIED]" in a and "inst_delta_z" in a for a in result.anomalies)
        assert "SIGNAL_CRITICAL: coinvest_score_z" in result.anomalies

    def test_warn_signal_unchanged(self, hb_mod):
        """SIGNAL_WARN behavior is unchanged — non-ALERT path doesn't touch the muffle."""
        ds = "2026-05-06"
        _write_dashboard(
            hb_mod,
            ds,
            {"inst_delta_z": {"health": "WARN", "ic": -0.05}},
        )
        result = hb_mod.check_ic_health(date.fromisoformat(ds))
        assert result.status == "WARN"
        assert "SIGNAL_WARN: inst_delta_z" in result.anomalies


class TestNeedsLLMCarriedSuppression:
    """P1 #1 — needs_llm should suppress LLM escalation when ALL anomalies
    are [CARRIED]-tagged. Mixed (some [CARRIED], some unmuffled) still
    escalates so unmuffled anomalies get their narrative."""

    def test_all_carried_anomalies_skip_llm(self, hb_mod):
        result = hb_mod.CheckResult(
            "ic_health_monitor",
            "WARN",
            "attention=HIGH",
            ["[CARRIED] SIGNAL_ALERT: inst_delta_z (expected, expires 2026-05-15; ...)"],
        )
        assert result.needs_llm is False

    def test_mixed_carried_and_unmuffled_still_escalates(self, hb_mod):
        result = hb_mod.CheckResult(
            "ic_health_monitor",
            "FAIL",
            "attention=HIGH",
            [
                "[CARRIED] SIGNAL_ALERT: inst_delta_z (expected, expires 2026-05-15; ...)",
                "SIGNAL_WARN: score_rank_pct",
            ],
        )
        assert result.needs_llm is True

    def test_no_anomalies_skips_llm(self, hb_mod):
        result = hb_mod.CheckResult("ic_health_monitor", "WARN", "edge", [])
        assert result.needs_llm is False

    def test_ok_status_skips_llm(self, hb_mod):
        result = hb_mod.CheckResult(
            "ic_health_monitor",
            "OK",
            "attention=LOW",
            ["[CARRIED] SIGNAL_ALERT: inst_delta_z"],
        )
        assert result.needs_llm is False

    def test_unmuffled_warn_still_escalates(self, hb_mod):
        """Back-compat: pre-P1 behavior preserved for unmuffled WARN."""
        result = hb_mod.CheckResult("qa", "WARN", "1 issue", ["snapshot row count drift +5%"])
        assert result.needs_llm is True


def test_generic_freshness_uses_content_date_not_mtime(hb_mod, tmp_path):
    """Generic freshness must not treat fresh mtime as fresh content."""
    art = tmp_path / "agents" / "sentinel" / "memory"
    art.mkdir(parents=True)
    (art / "2026-03-01.md").write_text("old note")

    entry = {
        "cadence": "daily_after_production",
        "artifact_paths": ["agents/sentinel/memory/"],
    }
    result = hb_mod.check_generic_freshness("sentinel", entry, date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
    assert "2026-03-01" in result.detail
    assert any("STALE_SOURCE" in a for a in result.anomalies)


def test_check_ops_missing_digest_reports_latest_content_date(hb_mod, tmp_path):
    digest_dir = tmp_path / "artifacts" / "ops_digest"
    digest_dir.mkdir(parents=True)
    (digest_dir / "2026-03-20_digest.json").write_text('{"attention":"LOW"}')

    result = hb_mod.check_ops(date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
    assert "2026-03-20" in result.detail
    assert any("STALE_SOURCE" in a for a in result.anomalies)


def test_check_ops_ok_when_today_digest_present(hb_mod, tmp_path):
    ds = "2026-05-08"
    digest_dir = tmp_path / "artifacts" / "ops_digest"
    digest_dir.mkdir(parents=True)
    (digest_dir / f"{ds}_digest.json").write_text('{"attention":"LOW"}')

    result = hb_mod.check_ops(date.fromisoformat(ds))
    assert result.status == "OK"


def test_check_ctgov_poller_flags_missing_today_and_stale_cache(hb_mod, tmp_path):
    cache_dir = tmp_path / "cache" / "ctgov"
    cache_dir.mkdir(parents=True)
    (cache_dir / "trial_records_2026-03-01.json").write_text("{}")

    result = hb_mod.check_ctgov_poller(date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
    assert any("MISSING_CACHE_TODAY" in a for a in result.anomalies)
    assert any("STALE_SOURCE" in a for a in result.anomalies)


def test_check_ctgov_poller_ok_when_today_present(hb_mod, tmp_path):
    ds = "2026-05-08"
    cache_dir = tmp_path / "cache" / "ctgov"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"trial_records_{ds}.json").write_text("{}")
    diff_dir = tmp_path / "artifacts" / "ctgov_daily"
    diff_dir.mkdir(parents=True)
    (diff_dir / f"{ds}_diff.json").write_text("{}")

    result = hb_mod.check_ctgov_poller(date.fromisoformat(ds))
    assert result.status == "OK"


def test_stale_memory_uses_filename_dates(hb_mod, tmp_path):
    mem = tmp_path / "agents" / "ops" / "memory"
    mem.mkdir(parents=True)
    (mem / "2026-03-01.md").write_text("old")
    digest = tmp_path / "artifacts" / "ops_digest"
    digest.mkdir(parents=True)
    (digest / "2026-05-01_digest.json").write_text("{}")

    entry = {
        "artifact_paths": ["agents/ops/memory/", "artifacts/ops_digest/"],
    }
    anomaly = hb_mod.check_stale_memory("ops", entry, date.fromisoformat("2026-05-08"))
    assert anomaly is not None
    assert "STALE_MEMORY" in anomaly
    assert "2026-03-01" in anomaly
    assert "2026-05-01" in anomaly


def test_check_sentinel_warn_on_rollback_recommendation(hb_mod, tmp_path):
    ds = "2026-05-08"
    snap = tmp_path / "data" / "snapshots" / ds
    snap.mkdir(parents=True)
    (snap / "ruleset_health.json").write_text(
        json.dumps({"status": "WARN", "consecutive_warn_days": 3, "recommend_rollback": True})
    )
    (snap / "drift_report.json").write_text("{}")
    promos = tmp_path / "artifacts" / "promotions"
    promos.mkdir(parents=True)
    (promos / "2026-01-01_receipt.json").write_text("{}")

    result = hb_mod.check_sentinel(date.fromisoformat(ds))
    assert result.status == "FAIL"
    assert any("ROLLBACK_RECOMMENDED" in a for a in result.anomalies)


def test_ic_health_stale_includes_latest_dashboard_date(hb_mod, tmp_path):
    dash_dir = tmp_path / "artifacts" / "ic_dashboard"
    dash_dir.mkdir(parents=True)
    (dash_dir / "2026-03-15_dashboard.json").write_text("{}")

    result = hb_mod.check_ic_health(date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
    assert "2026-03-15" in result.detail


def test_production_qa_stale_includes_latest_report_date(hb_mod, tmp_path, monkeypatch):
    from datetime import datetime

    qa_dir = tmp_path / "artifacts" / "production_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "2026-03-10_report.json").write_text('{"verdict":"GREEN","checks":[]}')

    class _FixedDatetime(datetime):
        @classmethod
        def utcnow(cls):
            return datetime(2026, 5, 8, 22, 0, 0)

    monkeypatch.setattr(hb_mod, "datetime", _FixedDatetime)

    result = hb_mod.check_production_qa(date.fromisoformat("2026-05-07"))
    assert result.status == "STALE"
    assert "2026-03-10" in result.detail


# ── Specialized dated-artifact checks (fleet completion) ─────


def test_check_catalyst_delta_ok_when_today_present(hb_mod, tmp_path):
    ds = "2026-05-08"
    d = tmp_path / "artifacts" / "catalyst_delta"
    d.mkdir(parents=True)
    (d / f"{ds}_delta.json").write_text("{}")

    result = hb_mod.check_catalyst_delta(date.fromisoformat(ds))
    assert result.status == "OK"


def test_check_catalyst_delta_stale_when_missing(hb_mod, tmp_path):
    d = tmp_path / "artifacts" / "catalyst_delta"
    d.mkdir(parents=True)
    (d / "2026-03-01_delta.json").write_text("{}")

    result = hb_mod.check_catalyst_delta(date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
    assert "2026-03-01" in result.detail


def test_check_postmortem_ok_on_recent_capture(hb_mod, tmp_path):
    pm = tmp_path / "artifacts" / "postmortem" / "2026-05-07"
    pm.mkdir(parents=True)
    (pm / "MRNA_postmortem.json").write_text("{}")

    result = hb_mod.check_postmortem(date.fromisoformat("2026-05-08"))
    assert result.status == "OK"


def test_check_crt_resolution_watcher_ok(hb_mod, tmp_path):
    ds = "2026-05-08"
    out = tmp_path / "output" / "catalyst_ev"
    out.mkdir(parents=True)
    (out / "crt_options_join.json").write_text('{"n_resolutions": 1}')
    res = tmp_path / "data" / "snapshots" / "resolutions"
    res.mkdir(parents=True)
    (res / f"{ds}_MRNA.json").write_text("{}")

    result = hb_mod.check_crt_resolution_watcher(date.fromisoformat(ds))
    assert result.status == "OK"


def test_check_options_watch_stale(hb_mod, tmp_path):
    d = tmp_path / "artifacts" / "options_watch"
    d.mkdir(parents=True)
    (d / "2026-03-01_watch.json").write_text("{}")

    result = hb_mod.check_options_watch(date.fromisoformat("2026-05-08"))
    assert result.status == "STALE"
