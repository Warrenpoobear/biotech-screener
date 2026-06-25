"""Tests for non-fleet research sweep tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_research_host_battery_script_references_core_tools():
    script = REPO / "tools" / "run_research_host_battery.sh"
    text = script.read_text(encoding="utf-8")
    assert "checklist_v2_rerun.py" in text
    assert "measure_final_score_ic_spec100.py" in text
    assert "verify_expectation_coverage_spec105.py" in text
    assert "snapshots_pit_v2" in text


def test_verify_spec105_passes_on_fixture_rankings(tmp_path):
    rankings = tmp_path / "rankings.csv"
    headers = [
        "short_interest_pct",
        "close_price",
        "market_cap_mm",
        "priced_move_pct",
        "insider_net_buy_value_90d",
    ]
    rows = [{h: "1.0" for h in headers} for _ in range(10)]
    import csv

    with open(rankings, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    from tools.verify_expectation_coverage_spec105 import build_report

    report = build_report(as_of_date="2026-06-24", rankings_path=rankings)
    assert report["overall"] == "PASS"
    assert report["schema"] == "spec105_expectation_coverage.v1"


def test_sciart_sample_review_with_fixture(tmp_path, monkeypatch):
    trials_path = tmp_path / "trials.json"
    trials_path.write_text(
        json.dumps(
            [
                {
                    "nct_id": "NCT00000001",
                    "brief_title": "Breast cancer study",
                    "sponsor": "Test Co",
                    "conditions": ["Stage II Breast Cancer"],
                    "interventions": ["Drug A"],
                    "phases": ["Phase 2"],
                },
                {
                    "nct_id": "NCT00000002",
                    "brief_title": "Lymphoma study",
                    "sponsor": "Test Co",
                    "conditions": ["Diffuse Large B-Cell Lymphoma"],
                    "interventions": ["Drug B"],
                    "phases": ["Phase 3"],
                },
            ]
        ),
        encoding="utf-8",
    )

    from tools import sciart_normalization_sample_review as mod

    rows = mod.collect_condition_rows(trials_path, "2026-06-24")
    assert rows
    sample = mod.build_sample(rows)
    assert sample
    text = mod.render_markdown(sample, trials_path=trials_path, as_of="2026-06-24")
    assert "Normalization Sample Review" in text
    assert "manual_verdict" in text.lower() or "verdict" in text.lower()


def test_sciart_summarize_counts_verdicts(tmp_path):
    worksheet = tmp_path / "worksheet.md"
    worksheet.write_text(
        "\n".join(
            [
                "## Annotated sample",
                "",
                "| target | nct_id | raw_condition | normalized | mondo_id | tier | confidence | verdict |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| lymphoma | NCT1 | foo | lymphoma | MONDO:1 | exact | 0.95 | TRUE_POSITIVE |",
                "| lymphoma | NCT2 | bar | lymphoma | MONDO:1 | substring | 0.8 | FALSE_POSITIVE |",
                "| breast cancer | NCT3 | baz | breast cancer | MONDO:2 | exact | 0.95 | TRUE_POSITIVE |",
                "| breast cancer | NCT4 | qux | breast cancer | MONDO:2 | substring | 0.8 |  |",
            ]
        ),
        encoding="utf-8",
    )

    from tools import sciart_normalization_sample_review as mod

    parsed = mod.parse_worksheet_verdicts(worksheet)
    assert len(parsed) == 4
    summary = mod.summarize_verdicts(parsed, worksheet=worksheet)
    assert summary["labeled_rows"] == 3
    assert summary["pending_rows"] == 1
    assert summary["verdict_counts"]["TRUE_POSITIVE"] == 2
    assert summary["verdict_counts"]["FALSE_POSITIVE"] == 1
    assert summary["complete"] is False
    assert summary["r4_pass"] is False

