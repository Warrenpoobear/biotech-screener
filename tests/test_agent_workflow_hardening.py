"""Tests for agent workflow hardening checks."""

from pathlib import Path


def test_dependency_lock_detects_missing_locked_package(tmp_path: Path):
    from tools.check_agent_workflow import check_dependency_lock

    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock"
    requirements.write_text("requests==2.34.2\nlanggraph==1.2.6\n", encoding="utf-8")
    lock.write_text("requests==2.34.2 \\\n    --hash=sha256:abc\n", encoding="utf-8")

    result = check_dependency_lock(requirements, lock)

    assert result.ok is False
    assert "langgraph" in result.message


def test_dependency_lock_passes_when_all_pins_are_locked(tmp_path: Path):
    from tools.check_agent_workflow import check_dependency_lock

    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock"
    requirements.write_text("requests==2.34.2\nlanggraph==1.2.6\n", encoding="utf-8")
    lock.write_text(
        "requests==2.34.2 \\\n" "    --hash=sha256:abc\n" "langgraph==1.2.6 \\\n" "    --hash=sha256:def\n",
        encoding="utf-8",
    )

    result = check_dependency_lock(requirements, lock)

    assert result.ok is True


def test_wall_clock_check_blocks_review_paths(tmp_path: Path):
    from tools.check_agent_workflow import check_wall_clock_usage

    bad_file = tmp_path / "review.py"
    bad_file.write_text("from datetime import datetime\nstamp = datetime.now()\n", encoding="utf-8")

    result = check_wall_clock_usage([bad_file])

    assert result.ok is False
    assert "datetime.now" in result.message


def test_preflight_report_includes_agent_checklist(monkeypatch):
    from tools import agent_preflight

    monkeypatch.setattr(agent_preflight, "run_cmd", lambda cmd, capture=True: "" if "status" in cmd else "ok")

    report = agent_preflight.build_preflight_report(
        git_state={"state": "on main, clean", "head": "abc123", "message": "test"},
        snapshot={"description": "2026-06-18, QA PASS"},
        blocked=["none"],
        contradictions=[],
        quarantine=["none"],
        allowed="Proceed with Tier 1 workflow hardening",
        not_allowed=["No scoring changes"],
    )

    assert "session_preflight_checklist" in report
    assert "governance_tier_classified" in report["session_preflight_checklist"]
    assert report["shell_health"]["status"] in {"ok", "unknown"}
    assert report["codegraph_hint"]["status"] in {"present", "missing"}


def test_scheduled_review_defaults_to_no_auto_approval():
    from tools.run_scientific_cartography_scheduled_review import run_scheduled_review

    defaults = run_scheduled_review.__defaults__

    assert defaults is not None
    assert defaults[1] is False


def test_changed_file_classifier_reports_highest_tier():
    from tools.check_agent_workflow import classify_changed_files

    result = classify_changed_files(["docs/AGENT_WORKFLOW_HARDENING.md", "selector_engine.py"])

    assert result.ok is True
    assert "Tier 3" in result.message


def test_third_party_import_checker_detects_missing_dependency(tmp_path: Path):
    from tools.check_agent_workflow import check_third_party_imports

    module = tmp_path / "uses_vendor.py"
    module.write_text("import vendor_sdk\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    pyproject = tmp_path / "pyproject.toml"
    requirements.write_text("requests==2.34.2\n", encoding="utf-8")
    pyproject.write_text("dependencies = []\n", encoding="utf-8")

    result = check_third_party_imports([module], requirements, pyproject)

    assert result.ok is False
    assert "vendor_sdk" in result.message


def test_approval_language_scan_blocks_production_approval(tmp_path: Path):
    from tools.check_agent_workflow import check_approval_language

    module = tmp_path / "approval.py"
    module.write_text('state = {"production_deployment_approved": True}\n', encoding="utf-8")

    result = check_approval_language([module])

    assert result.ok is False
    assert "production_deployment_approved" in result.message


def test_artifact_schema_registry_requires_governance(tmp_path: Path):
    from tools.check_agent_workflow import check_artifact_schema_registry

    registry = tmp_path / "schemas.json"
    registry.write_text(
        """
{
  "artifacts": {
    "hermes_recursive_self_improvement_queue": {"required_fields": ["artifact_type", "schema_version", "governance"]},
    "scientific_cartography_langgraph_review_summary": {"required_fields": ["artifact_type", "schema_version"]},
    "scientific_cartography_langgraph_human_decision": {"required_fields": ["artifact_type", "schema_version", "governance"]},
    "scientific_cartography_lg3_scheduled_review_cron_execution": {"required_fields": ["artifact_type", "schema_version", "governance"]}
  }
}
""",
        encoding="utf-8",
    )

    result = check_artifact_schema_registry(registry)

    assert result.ok is False
    assert "governance" in result.message


def test_committed_artifact_schema_registry_passes():
    from tools.check_agent_workflow import REPO_ROOT, check_artifact_schema_registry

    result = check_artifact_schema_registry(REPO_ROOT / "docs" / "agent_artifact_schemas.json")

    assert result.ok is True


def test_stale_artifact_reference_detector(tmp_path: Path):
    from tools.check_agent_workflow import check_stale_artifact_references

    doc = tmp_path / "doc.md"
    doc.write_text("Read artifacts/scientific_cartography/2026-06-18/review.json\n", encoding="utf-8")

    result = check_stale_artifact_references([doc])

    assert result.ok is False
    assert "2026-06-18" in result.message


def test_network_marker_detector_flags_unmarked_live_call(tmp_path: Path):
    from tools.check_agent_workflow import check_network_tests_marked

    test_file = tmp_path / "test_live.py"
    test_file.write_text(
        "import requests\n\ndef test_live():\n    requests.get('https://example.com')\n", encoding="utf-8"
    )

    result = check_network_tests_marked([test_file])

    assert result.ok is False
    assert "test_live.py" in result.message
