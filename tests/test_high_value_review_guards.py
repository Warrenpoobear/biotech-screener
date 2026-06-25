"""Guards from high-value review pass (#413–#417 tooling)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_forward_evidence_generated_at_is_deterministic():
    from tools.forward_evidence_package import build_package

    package = build_package(
        as_of_date="2026-06-24",
        snapshot_dir=REPO / "data" / "snapshots",
        ic_start_date="2026-04-03",
        horizons=[20],
    )
    assert package["generated_at"] == "2026-06-24T00:00:00Z"


def test_spec105_generated_at_is_deterministic(tmp_path):
    from tools.verify_expectation_coverage_spec105 import build_report

    rankings = tmp_path / "rankings.csv"
    headers = [
        "short_interest_pct",
        "close_price",
        "market_cap_mm",
        "priced_move_pct",
        "insider_net_buy_value_90d",
    ]
    import csv

    with open(rankings, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows([{h: "1.0" for h in headers} for _ in range(10)])

    report = build_report(as_of_date="2026-06-24", rankings_path=rankings)
    assert report["generated_at"] == "2026-06-24T00:00:00Z"


def test_data_auditor_has_repo_on_sys_path():
    import agents.data_auditor.run_audit as mod

    assert str(mod.REPO_ROOT) in sys.path


def test_path_c_close_outcome_date_is_deterministic():
    from tools.path_c_window_close_decision import decision_tree

    outcome = decision_tree(window_end="2026-06-03", as_of_date="2026-06-24")
    assert outcome["date"] == "2026-06-24T00:00:00Z"


def test_semgrepignore_excludes_event_binder_false_positive():
    ignore = (REPO / ".semgrepignore").read_text(encoding="utf-8")
    assert "tests/test_event_outcome_binder.py" in ignore
