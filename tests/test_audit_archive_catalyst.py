"""Tests for scripts/audit_archive_catalyst.py."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_archive_catalyst import (
    audit_single_archive,
    compute_hypothetical_coverage,
    format_audit_json,
    format_audit_report_md,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal archive .tar.gz for testing
# ---------------------------------------------------------------------------

def _make_archive(
    tmp_dir: Path,
    date_str: str,
    rankings_rows: List[Dict[str, str]],
    decision_inputs: Dict[str, Dict[str, Any]] | None = None,
    trials: List[Dict[str, Any]] | None = None,
    pdufa: List[Dict[str, Any]] | None = None,
) -> Path:
    """Create a minimal .tar.gz archive for testing."""
    archive_root = tmp_dir / date_str
    inputs_dir = archive_root / "inputs"
    inputs_dir.mkdir(parents=True)

    # rankings.csv
    cols = ["ticker", "archetype", "eligible", "composite_score", "composite_rank"]
    csv_path = archive_root / "rankings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rankings_rows:
            writer.writerow({c: row.get(c, "") for c in cols})

    # decision_inputs.json
    if decision_inputs is not None:
        with open(inputs_dir / "decision_inputs.json", "w") as f:
            json.dump(decision_inputs, f)

    # trial_records.json
    if trials is not None:
        with open(inputs_dir / "trial_records.json", "w") as f:
            json.dump(trials, f)

    # pdufa_dates.json
    if pdufa is not None:
        with open(inputs_dir / "pdufa_dates.json", "w") as f:
            json.dump(pdufa, f)

    # Pack into tar.gz
    tar_path = tmp_dir / f"{date_str}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(archive_root), arcname=date_str)

    # Clean up uncompressed dir so audit reads from tar only
    shutil.rmtree(archive_root)
    return tar_path


def _dev_row(ticker: str) -> Dict[str, str]:
    return {
        "ticker": ticker,
        "archetype": "drug_developer",
        "eligible": "1",
        "composite_score": "50",
        "composite_rank": "100",
    }


def _comm_row(ticker: str) -> Dict[str, str]:
    return {
        "ticker": ticker,
        "archetype": "commercial_biotech",
        "eligible": "1",
        "composite_score": "40",
        "composite_rank": "200",
    }


# ---------------------------------------------------------------------------
# TestAuditSingleArchive
# ---------------------------------------------------------------------------

class TestAuditSingleArchive:

    def test_full_catalyst_coverage(self):
        """All eligible tickers have catalyst → 0% missing."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [_dev_row("AAA"), _dev_row("BBB"), _dev_row("CCC")]
            di = {
                "AAA": {"days_to_catalyst": 30, "in_optimal_window": True},
                "BBB": {"days_to_catalyst": 90, "in_optimal_window": False},
                "CCC": {"days_to_catalyst": 150, "in_optimal_window": False},
            }
            tar = _make_archive(tmp_path, "2025-06-30", rows, decision_inputs=di)
            result = audit_single_archive(tar)

        assert result["status"] == "ok"
        assert result["catalyst_missing_pct"] == 0.0
        assert result["catalyst_present"] == 3
        assert result["catalyst_missing"] == 0
        assert result["passes_dq_gate"] is True

    def test_no_catalyst_at_all(self):
        """No tickers have catalyst → 100% missing."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [_dev_row("AAA"), _dev_row("BBB")]
            di = {
                "AAA": {"alpha_60d": 0.05},  # has enrichment but no catalyst
                "BBB": {},
            }
            tar = _make_archive(tmp_path, "2025-06-30", rows, decision_inputs=di)
            result = audit_single_archive(tar)

        assert result["catalyst_missing_pct"] == 100.0
        assert result["catalyst_present"] == 0
        assert result["passes_dq_gate"] is False

    def test_partial_coverage(self):
        """2 of 5 tickers have catalyst → 60% missing."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [_dev_row(f"T{i}") for i in range(5)]
            di = {
                "T0": {"days_to_catalyst": 30},
                "T1": {"days_to_catalyst": 200},
                # T2, T3, T4 have no catalyst
            }
            tar = _make_archive(tmp_path, "2025-06-30", rows, decision_inputs=di)
            result = audit_single_archive(tar)

        assert result["catalyst_present"] == 2
        assert result["catalyst_missing"] == 3
        assert result["catalyst_missing_pct"] == 60.0
        assert result["passes_dq_gate"] is True  # 60 <= 80

    def test_dq_gate_boundary(self):
        """Exactly 80% missing → passes DQ gate (<=)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 10 tickers, 2 with catalyst → 80% missing
            rows = [_dev_row(f"T{i}") for i in range(10)]
            di = {
                "T0": {"days_to_catalyst": 30},
                "T1": {"days_to_catalyst": 90},
            }
            tar = _make_archive(tmp_path, "2025-06-30", rows, decision_inputs=di)
            result = audit_single_archive(tar, dq_threshold=80.0)

        assert result["catalyst_missing_pct"] == 80.0
        assert result["passes_dq_gate"] is True

    def test_excludes_non_dev(self):
        """Commercial tickers are excluded from eligible set."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [_dev_row("DEV1"), _comm_row("COMM1")]
            di = {"DEV1": {"days_to_catalyst": 30}}
            tar = _make_archive(tmp_path, "2025-06-30", rows, decision_inputs=di)
            result = audit_single_archive(tar)

        assert result["n_eligible"] == 1
        assert result["catalyst_present"] == 1
        assert result["catalyst_missing_pct"] == 0.0

    def test_skips_pre_backfill_archive(self):
        """Archive without eligible/archetype columns → skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive_root = tmp_path / "2024-01-31"
            inputs_dir = archive_root / "inputs"
            inputs_dir.mkdir(parents=True)

            # CSV without archetype/eligible columns
            csv_path = archive_root / "rankings.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ticker", "composite_score"])
                writer.writeheader()
                writer.writerow({"ticker": "AAA", "composite_score": "50"})

            tar_path = tmp_path / "2024-01-31.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(str(archive_root), arcname="2024-01-31")
            shutil.rmtree(archive_root)

            result = audit_single_archive(tar_path)

        assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# TestHypotheticalCoverage
# ---------------------------------------------------------------------------

class TestHypotheticalCoverage:

    def test_trial_within_180d(self):
        """Trial at 100d → counted at all windows (180, 270, 365)."""
        eligible = {"AAA"}
        trials = [{
            "ticker": "AAA", "phase": "PHASE3",
            "first_posted": "2025-01-01",
            "primary_completion_date": "2025-07-11",  # ~100d from 2025-04-01
        }]
        result = compute_hypothetical_coverage(
            trials, [], eligible, date(2025, 4, 1), [180, 270, 365]
        )
        assert result[180] == 1
        assert result[270] == 1
        assert result[365] == 1

    def test_trial_at_250d(self):
        """Trial at 250d → counted only at 270d and 365d, not 180d."""
        eligible = {"AAA"}
        trials = [{
            "ticker": "AAA", "phase": "PHASE2",
            "first_posted": "2024-06-01",
            "primary_completion_date": "2025-12-07",  # 250d from 2025-04-01
        }]
        result = compute_hypothetical_coverage(
            trials, [], eligible, date(2025, 4, 1), [180, 270, 365]
        )
        assert result[180] == 0
        assert result[270] == 1
        assert result[365] == 1

    def test_pdufa_no_cap(self):
        """PDUFA at 400d → counted at all windows (no cap on PDUFA)."""
        eligible = {"AAA"}
        pdufa = [{
            "ticker": "AAA",
            "pdufa_date": "2026-05-06",  # 400d from 2025-04-01
        }]
        result = compute_hypothetical_coverage(
            [], pdufa, eligible, date(2025, 4, 1), [180, 270, 365]
        )
        assert result[180] == 1
        assert result[270] == 1
        assert result[365] == 1

    def test_pit_filter_excludes_future_posted(self):
        """Trial with first_posted > as_of is excluded at all windows."""
        eligible = {"AAA"}
        trials = [{
            "ticker": "AAA", "phase": "PHASE3",
            "first_posted": "2025-06-01",  # AFTER as_of
            "primary_completion_date": "2025-07-01",
        }]
        result = compute_hypothetical_coverage(
            trials, [], eligible, date(2025, 4, 1), [180, 270, 365]
        )
        assert result[180] == 0
        assert result[270] == 0
        assert result[365] == 0

    def test_only_eligible_counted(self):
        """Trials for non-eligible tickers are ignored."""
        eligible = {"AAA"}
        trials = [{
            "ticker": "BBB", "phase": "PHASE3",
            "first_posted": "2025-01-01",
            "primary_completion_date": "2025-05-01",
        }]
        result = compute_hypothetical_coverage(
            trials, [], eligible, date(2025, 4, 1), [180]
        )
        assert result[180] == 0

    def test_past_trial_excluded(self):
        """Trial with PCD before as_of is excluded."""
        eligible = {"AAA"}
        trials = [{
            "ticker": "AAA", "phase": "PHASE3",
            "first_posted": "2024-01-01",
            "primary_completion_date": "2025-03-01",  # BEFORE as_of
        }]
        result = compute_hypothetical_coverage(
            trials, [], eligible, date(2025, 4, 1), [180, 365]
        )
        assert result[180] == 0
        assert result[365] == 0


# ---------------------------------------------------------------------------
# TestFormatReport
# ---------------------------------------------------------------------------

class TestFormatReport:

    def _sample_results(self) -> List[Dict[str, Any]]:
        return [
            {
                "date": "2025-06-30", "status": "ok",
                "n_eligible": 10, "catalyst_present": 5, "catalyst_missing": 5,
                "catalyst_missing_pct": 50.0, "passes_dq_gate": True,
                "source_breakdown": {"trial_pcd": 3, "pdufa": 2},
                "hypothetical": {
                    "180d": {"present": 5, "missing": 5, "missing_pct": 50.0, "passes_dq": True},
                    "270d": {"present": 7, "missing": 3, "missing_pct": 30.0, "passes_dq": True},
                    "365d": {"present": 9, "missing": 1, "missing_pct": 10.0, "passes_dq": True},
                },
            }
        ]

    def test_markdown_has_correct_columns(self):
        """Markdown report contains expected table headers."""
        results = self._sample_results()
        md = format_audit_report_md(results)
        assert "# Archive Catalyst Coverage Audit" in md
        assert "## Current Coverage" in md
        assert "## Hypothetical Coverage" in md
        assert "2025-06-30" in md
        assert "PASS" in md

    def test_json_has_all_keys(self):
        """JSON output has required top-level keys."""
        results = self._sample_results()
        j = format_audit_json(results, dq_threshold=80.0)
        assert j["dq_threshold"] == 80.0
        assert j["n_archives"] == 1
        assert j["n_ok"] == 1
        assert "current_gate" in j
        assert j["current_gate"]["pass"] == 1
        assert j["current_gate"]["fail"] == 0
        assert "archives" in j
        assert len(j["archives"]) == 1
