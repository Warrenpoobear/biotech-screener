"""Tests for the read-only biotech-mcp server (tools/biotech_mcp_server.py).

Hermetic: each test points the server at a tmp repo fixture via REPO_ROOT,
so no real snapshot/artifact data is required.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "biotech_mcp_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("biotech_mcp_server", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    m = _load_module()
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    _build_fixture(tmp_path)
    return m


def _build_fixture(root: Path) -> None:
    snaps = root / "data" / "snapshots"
    for date in ("2026-06-19", "2026-06-20"):
        d = snaps / date
        d.mkdir(parents=True)
        (d / "snapshot_manifest.json").write_text(
            json.dumps({"snapshot_dir": date, "files": [{"name": "rankings.csv", "size_bytes": 10}]})
        )
        (d / "phase2_health.json").write_text(
            json.dumps({"status": "OK" if date == "2026-06-20" else "WARN", "reasons": [], "metrics": {}})
        )
        (d / "rankings.csv").write_text("ticker,actionable_rank,target_weight_pct\nCOGT,1,5.0\nDNTH,2,4.0\n")
        (d / "ees_gate_diagnostics.json").write_text(json.dumps({"as_of_date": date, "gate_mode": "advisory"}))

    art = root / "artifacts"
    (art / "readiness").mkdir(parents=True)
    (art / "gate_verdict_ledger.jsonl").write_text(
        json.dumps({"as_of_date": "2026-06-19", "overall_status": "WARN", "n_fail": 0})
        + "\n"
        + json.dumps({"as_of_date": "2026-06-20", "overall_status": "PASS", "n_fail": 0})
        + "\n"
    )
    (art / "readiness" / "forward_eval_ic_baseline.json").write_text(
        json.dumps(
            {
                "window_start": "2026-06-01",
                "path_c_status": "IC_UNOBSERVABLE",
                "observations": [{"date": f"2026-06-{d:02d}", "mean_ic": 0.0} for d in range(1, 21)],
            }
        )
    )

    cart = art / "scientific_cartography" / "2026-06-18"
    cart.mkdir(parents=True)
    (cart / "scientific_cartography_status.json").write_text(
        json.dumps({"as_of_date": "2026-06-18", "status": "ok", "governance": {"read_only_diagnostic": True}})
    )
    (cart / "landscape_feature_coverage_report.json").write_text(
        json.dumps({"as_of_date": "2026-06-18", "program_records": 5, "mean_white_space_score": 0.4})
    )
    (cart / "disease_map_summary.json").write_text(json.dumps({"diseases": 3, "mapped": 2}))
    (cart / "program_records.jsonl").write_text('{"id": "p1"}\n')

    semgrep = root / ".semgrep"
    semgrep.mkdir()
    (semgrep / "governance.yml").write_text(
        "rules:\n  - id: no-autopush-to-main\n    severity: ERROR\n  - id: langgraph-none-guard\n    severity: WARNING\n"
    )


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
def test_initialize(srv):
    resp = srv._handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["serverInfo"]["name"] == "biotech-mcp"


def test_tools_list_has_eleven(srv):
    resp = srv._handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert len(names) == 11
    assert "run_readonly_diagnostics" in names and "list_snapshots" in names


def test_unknown_method(srv):
    resp = srv._handle_message({"jsonrpc": "2.0", "id": 3, "method": "bogus/method"})
    assert resp["error"]["code"] == -32601


def test_notification_has_no_response(srv):
    assert srv._handle_message({"jsonrpc": "2.0", "method": "initialized"}) is None


def test_call_unknown_tool(srv):
    resp = srv._handle_message(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}
    )
    assert resp["error"]["code"] == -32602


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_list_snapshots(srv):
    out = srv._list_snapshots({})
    assert out["latest"] == "2026-06-20"
    assert out["count"] == 2
    assert out["snapshots"][0]["key_files"]["rankings.csv"] is True


def test_latest_manifest_defaults_to_newest(srv):
    out = srv._read_latest_snapshot_manifest({})
    assert out["date"] == "2026-06-20"
    assert out["content"]["snapshot_dir"] == "2026-06-20"


def test_manifest_explicit_date(srv):
    out = srv._read_latest_snapshot_manifest({"date": "2026-06-19"})
    assert out["date"] == "2026-06-19"


def test_gate_verdicts_tail_and_ees(srv):
    out = srv._read_gate_verdicts({"limit": 1, "date": "2026-06-20"})
    assert out["ledger"]["records"][-1]["overall_status"] == "PASS"
    assert out["ledger"]["records_total"] == 2
    assert out["ees_gate_diagnostics"]["content"]["gate_mode"] == "advisory"


def test_phase2_health(srv):
    assert srv._read_phase2_health({})["content"]["status"] == "OK"


def test_rankings_schema(srv):
    out = srv._read_rankings_schema({"sample_rows": 1})
    assert out["columns"] == ["ticker", "actionable_rank", "target_weight_pct"]
    assert out["row_count"] == 2
    assert out["sample"][0]["ticker"] == "COGT"


def test_event_ev_feature_coverage(srv):
    out = srv._read_event_ev_feature_coverage({})
    assert out["date"] == "2026-06-18"
    assert out["content"]["program_records"] == 5


def test_forward_eval_ic_ledger_trims(srv):
    out = srv._read_forward_eval_ic_ledger({"limit": 3})
    assert out["content"]["observations_total"] == 20
    assert len(out["content"]["observations"]) == 3
    assert out["content"]["path_c_status"] == "IC_UNOBSERVABLE"


def test_scientific_cartography_status(srv):
    out = srv._read_scientific_cartography_status({})
    assert out["content"]["governance"]["read_only_diagnostic"] is True


def test_list_disease_map_artifacts(srv):
    out = srv._list_disease_map_artifacts({})
    names = {f["name"] for f in out["files"]}
    assert "program_records.jsonl" in names
    assert out["disease_map_summary"]["diseases"] == 3


def test_semgrep_findings_reports_rules_not_results(srv):
    out = srv._read_semgrep_findings({})
    assert out["findings_persisted"] is False
    assert "no-autopush-to-main" in out["rules"]
    assert out["rule_count"] == 2


def test_run_readonly_diagnostics(srv):
    out = srv._run_readonly_diagnostics({})
    assert out["executes_scripts"] is False and out["mutates"] is False and out["network"] is False
    assert out["latest_snapshot"] == "2026-06-20"
    statuses = {c["check"]: c for c in out["checks"]}
    assert statuses["gate_verdicts"]["status"] == "PASS"
    assert statuses["phase2_health"]["status"] == "OK"


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["2026-06-20/../..", "../etc", "2026_06_20", "latest", "../../../etc/passwd"])
def test_date_traversal_rejected(srv, bad):
    with pytest.raises(ValueError):
        srv._read_phase2_health({"date": bad})


def test_missing_artifact_returns_exists_false(srv):
    # gate ledger absent in the fixture path? It exists; instead delete it and re-read.
    (srv.REPO_ROOT / "artifacts" / "gate_verdict_ledger.jsonl").unlink()
    out = srv._read_gate_verdicts({"limit": 5})
    assert out["ledger"]["exists"] is False
    assert out["ledger"]["records"] == []


def test_readonly_diagnostics_on_empty_repo(srv, tmp_path):
    srv.REPO_ROOT = tmp_path / "empty2"
    (tmp_path / "empty2").mkdir()
    out = srv._run_readonly_diagnostics({})
    assert out["latest_snapshot"] is None
    assert out["summary"]["artifacts_present"] == 0


def test_int_arg_bounds(srv):
    with pytest.raises(ValueError):
        srv._list_snapshots({"limit": 0})
    with pytest.raises(ValueError):
        srv._list_snapshots({"limit": 10_000})
