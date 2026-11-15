"""The universe coverage refresh records every status change in the audit log.

On 2026-11-15 the refresh promoted SRZN from pending_coverage to active and left no
audit entry, because only tools/apply_ticker_change.py wrote to universe_audit_log.jsonl.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.refresh_eligible_biotech_universe import audit_lines, status_changes  # noqa: E402

TOOL = REPO / "tools" / "refresh_eligible_biotech_universe.py"


def test_status_changes_lists_only_changed_tickers():
    before = [
        {"ticker": "AAA", "status": "pending_coverage"},
        {"ticker": "BBB", "status": "active"},
        {"ticker": "CCC"},
    ]
    after = [
        {"ticker": "AAA", "status": "active"},
        {"ticker": "BBB", "status": "active", "coverage_refreshed_as_of": "2026-11-13"},
        {"ticker": "CCC", "status": "pending_data_collection", "status_reason": "coverage_pending:market_data"},
    ]
    changes = status_changes(before, after)
    assert [(c["ticker"], c["prior_status"], c["new_status"]) for c in changes] == [
        ("AAA", "pending_coverage", "active"),
        ("CCC", "active", "pending_data_collection"),
    ]


def test_audit_lines_shape():
    changes = [{"ticker": "AAA", "prior_status": "pending_coverage", "new_status": "active", "status_reason": ""}]
    (line,) = audit_lines(changes, "2026-11-13", "operator", "2026-11-15T16:00:00Z")
    entry = json.loads(line)
    assert entry == {
        "timestamp": "2026-11-15T16:00:00Z",
        "action": "coverage_refresh",
        "ticker": "AAA",
        "prior_status": "pending_coverage",
        "new_status": "active",
        "reason": "coverage refresh as of 2026-11-13",
        "operator": "operator",
    }


def _run(tmp_path: Path, apply: bool) -> tuple[dict, Path, Path]:
    universe = tmp_path / "universe.json"
    rows = [{"ticker": "ZZZQ", "name": "Zzzq Bio", "sector": "Biotechnology", "status": "active"}]
    universe.write_text(json.dumps(rows), encoding="utf-8")
    trials = tmp_path / "trial_records.json"
    trials.write_text("[]", encoding="utf-8")
    audit = tmp_path / "universe_audit_log.jsonl"
    audit.write_text('{"action": "earlier"}\n', encoding="utf-8")
    cmd = [sys.executable, str(TOOL), "--as-of-date", "2026-11-13", "--universe-path", str(universe)]
    cmd += ["--trial-records-path", str(trials)] + (["--apply"] if apply else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out), universe, audit


def test_apply_appends_one_audit_line_per_status_change(tmp_path):
    # A biotech row with no market data or trials is demoted to pending_data_collection.
    report, universe, audit = _run(tmp_path, apply=True)
    assert [c["ticker"] for c in report["status_changes"]] == ["ZZZQ"]
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"action": "earlier"}', "existing entries are kept"
    entry = json.loads(lines[1])
    assert entry["action"] == "coverage_refresh" and entry["ticker"] == "ZZZQ"
    assert entry["prior_status"] == "active" and entry["new_status"] == "pending_data_collection"
    assert len(lines) == 2 and report["audit_log_entries"] == 1


def test_dry_run_reports_changes_but_writes_nothing(tmp_path):
    report, universe, audit = _run(tmp_path, apply=False)
    assert report["dry_run"] is True
    assert [c["ticker"] for c in report["status_changes"]] == ["ZZZQ"]
    assert audit.read_text(encoding="utf-8") == '{"action": "earlier"}\n'
    assert json.loads(universe.read_text(encoding="utf-8"))[0]["status"] == "active"
