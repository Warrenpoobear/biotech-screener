from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from self_improvement_harness import (  # noqa: E402
    build_daily_model_diagnosis,
    build_weekly_remediation_queue,
    week_for_as_of_date,
    write_daily_model_diagnosis,
    write_weekly_remediation_queue,
    write_weekly_remediation_queue_from_dir,
)


def _write_rankings(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_daily_diagnosis_treats_missing_expectation_fields_as_plumbing_gap(tmp_path):
    snapshot = tmp_path / "snapshot"
    _write_rankings(
        snapshot / "rankings.csv",
        [
            {
                "ticker": "AAA",
                "eligible": "1",
                "actionable_rank": "1",
                "short_interest_pct": "12.5",
                "close_price": "8.25",
            },
            {
                "ticker": "BBB",
                "eligible": "1",
                "actionable_rank": "2",
                "short_interest_pct": "",
                "close_price": "11.00",
            },
        ],
        ["ticker", "eligible", "actionable_rank", "short_interest_pct", "close_price"],
    )
    manifest = {
        "effective_as_of_date": "2026-06-05",
        "overall_status": "WARN",
        "gates": [
            {
                "name": "risk_concentration",
                "status": "WARN",
                "detail": "catalyst<=7d=35% > 30% WARN",
                "value": {"catalyst_7d_wt": 0.35},
            },
            {
                "name": "forward_eval",
                "status": "PASS",
                "detail": "mean_ic=0.0410",
                "value": {"mean_ic": 0.041, "n_evaluated": 10},
            },
        ],
    }

    diagnosis = build_daily_model_diagnosis(snapshot, manifest, "2026-06-05")
    finding = next(item for item in diagnosis["findings"] if item["area"] == "Data coverage")

    assert finding["classification"] == "Plumbing/export gap"
    assert finding["proposal_type"] == "Plumbing fix"
    assert finding["governance_classification"] == "AUTO_SAFE_DIAGNOSTIC"
    assert "Do not invent insider signal" in finding["forbidden_changes"]
    assert diagnosis["data_coverage"]["fields"]["market_cap_mm"]["status"] == "MISSING_COLUMN"
    assert diagnosis["data_coverage"]["fields"]["priced_move_pct"]["status"] == "MISSING_COLUMN"
    assert "selector" in " ".join(diagnosis["do_not_change"]).lower()


def test_daily_diagnosis_surfaces_probabilistic_feature_feedback_gap(tmp_path):
    snapshot = tmp_path / "snapshot"
    _write_rankings(
        snapshot / "rankings.csv",
        [
            {
                "ticker": "AAA",
                "eligible": "1",
                "actionable_rank": "1",
                "confidence_overall": "0.90",
                "confidence_pos": "",
                "p_move_gt_implied": "0.75",
            },
            {
                "ticker": "BBB",
                "eligible": "1",
                "actionable_rank": "2",
                "confidence_overall": "",
                "confidence_pos": "1.20",
                "p_move_gt_implied": "0.99",
            },
        ],
        [
            "ticker",
            "eligible",
            "actionable_rank",
            "confidence_overall",
            "confidence_pos",
            "p_move_gt_implied",
        ],
    )
    manifest = {
        "effective_as_of_date": "2026-06-05",
        "overall_status": "WARN",
        "gates": [],
    }

    diagnosis = build_daily_model_diagnosis(snapshot, manifest, "2026-06-05")
    feedback = diagnosis["probabilistic_feature_feedback"]
    finding_ids = {item["id"] for item in diagnosis["findings"]}

    assert feedback["observed_fields"] == [
        "confidence_overall",
        "confidence_pos",
        "p_move_gt_implied",
    ]
    assert feedback["fields"]["confidence_pos"]["out_of_bounds_count"] == 1
    assert "probabilistic_feature_feedback_gap" in finding_ids
    assert "probabilistic_feature_contract_violation" in finding_ids


def test_daily_diagnosis_writes_markdown_and_machine_readable_json(tmp_path):
    snapshot = tmp_path / "snapshot"
    _write_rankings(
        snapshot / "rankings.csv",
        [
            {
                "ticker": "AAA",
                "eligible": "1",
                "actionable_rank": "1",
                "market_cap_mm": "125.0",
            }
        ],
        ["ticker", "eligible", "actionable_rank", "market_cap_mm"],
    )
    manifest = {
        "effective_as_of_date": "2026-06-05",
        "overall_status": "PASS",
        "gates": [
            {
                "name": "forward_eval",
                "status": "PASS",
                "detail": "cold start",
                "value": {},
            }
        ],
    }

    paths = write_daily_model_diagnosis(snapshot, manifest, "2026-06-05", tmp_path / "out")

    assert paths["json"].name == "DAILY_MODEL_DIAGNOSIS_2026_06_05.json"
    assert paths["markdown"].name == "DAILY_MODEL_DIAGNOSIS_2026_06_05.md"
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert payload["as_of_date"] == "2026-06-05"
    assert "## 1. Data coverage regressions" in markdown
    assert "## 8. Do-not-change list" in markdown
    assert "No ranker, selector, sizing, final_score, or alpha changes" in markdown


def test_weekly_remediation_queue_routes_by_governance_classification(tmp_path):
    diagnosis_a = build_daily_model_diagnosis(
        tmp_path,
        {
            "effective_as_of_date": "2026-06-01",
            "overall_status": "WARN",
            "gates": [
                {
                    "name": "forward_eval",
                    "status": "WARN",
                    "detail": "mean_ic=-0.0100 below floor",
                    "value": {"mean_ic": -0.01, "n_evaluated": 10},
                }
            ],
        },
        "2026-06-01",
    )
    diagnosis_b = {
        "as_of_date": "2026-06-02",
        "findings": [
            {
                "id": "manual_alpha_scope_creep",
                "title": "New insider buying signal requested",
                "area": "Forward eval",
                "severity": "HIGH",
                "classification": "New alpha signal",
                "proposal_type": "New alpha signal",
                "risk_level": "Highest",
                "governance_classification": "BLOCKED_BY_GOVERNANCE",
                "evidence": ["No PIT validation available"],
                "forbidden_changes": ["Do not add new alpha signal"],
                "verification": "Requires PIT validation and explicit approval.",
                "promotion_status": "Rejected scope creep",
            }
        ],
    }

    queue = build_weekly_remediation_queue([diagnosis_a, diagnosis_b], "2026-W23")

    classifications = {item["classification"] for item in queue["items"]}
    assert "RESEARCH_ONLY" in classifications
    assert "BLOCKED_BY_GOVERNANCE" in classifications

    paths = write_weekly_remediation_queue(queue, tmp_path / "out")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert paths["markdown"].name == "WEEKLY_REMEDIATION_QUEUE_2026_W23.md"
    assert "OPERATOR_APPROVAL_REQUIRED" in markdown
    assert "RESEARCH_ONLY" in markdown
    assert "BLOCKED_BY_GOVERNANCE" in markdown


def test_weekly_queue_can_be_written_from_daily_diagnosis_directory(tmp_path):
    diagnosis_dir = tmp_path / "diagnoses"
    snapshot = tmp_path / "snapshot"
    _write_rankings(
        snapshot / "rankings.csv",
        [{"ticker": "AAA", "eligible": "1", "actionable_rank": "1"}],
        ["ticker", "eligible", "actionable_rank"],
    )
    write_daily_model_diagnosis(
        snapshot,
        {
            "effective_as_of_date": "2026-06-05",
            "overall_status": "WARN",
            "gates": [],
        },
        "2026-06-05",
        diagnosis_dir,
    )

    week = week_for_as_of_date("2026-06-05")
    paths = write_weekly_remediation_queue_from_dir(diagnosis_dir, week, tmp_path / "weekly")

    assert week == "2026-W23"
    assert paths["json"].name == "WEEKLY_REMEDIATION_QUEUE_2026_W23.json"
    assert paths["markdown"].name == "WEEKLY_REMEDIATION_QUEUE_2026_W23.md"
