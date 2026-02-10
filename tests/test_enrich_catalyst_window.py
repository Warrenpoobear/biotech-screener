"""Tests for compute_catalyst() with trial_window_days parameter."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enrich_archive_inputs import (
    CATALYST_KEYS,
    TRIAL_PCD_WINDOW_DAYS,
    compute_catalyst,
    enrich_archive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trial(
    ticker: str,
    phase: str = "PHASE3",
    first_posted: str = "2024-01-01",
    pcd: str = "2025-07-01",
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "phase": phase,
        "first_posted": first_posted,
        "primary_completion_date": pcd,
    }


def _pdufa(ticker: str, pdufa_date: str) -> Dict[str, Any]:
    return {"ticker": ticker, "pdufa_date": pdufa_date}


AS_OF = date(2025, 4, 1)


# ---------------------------------------------------------------------------
# TestComputeCatalystWindow
# ---------------------------------------------------------------------------

class TestComputeCatalystWindow:

    def test_trial_100d_with_365d_window(self):
        """Trial 100d away with 365d window → captured, specific_days."""
        trials = [_trial("AAA", pcd="2025-07-10")]  # ~100d from 2025-04-01
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=365)
        assert result["days_to_catalyst"] == 100
        assert result["catalyst_source"] == "trial_pcd"

    def test_trial_250d_with_365d_window(self):
        """Trial 250d away with 365d window → captured."""
        trials = [_trial("AAA", pcd="2025-12-07")]  # 250d from 2025-04-01
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=365)
        assert result["days_to_catalyst"] == 250
        assert result["catalyst_source"] == "trial_pcd"

    def test_trial_250d_with_180d_window(self):
        """Trial 250d away with 180d window → NOT captured (outside window)."""
        trials = [_trial("AAA", pcd="2025-12-07")]  # 250d from 2025-04-01
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=180)
        assert result == {}  # No catalyst found

    def test_trial_400d_with_365d_window(self):
        """Trial 400d away with 365d window → NOT captured (exceeds 365)."""
        trials = [_trial("AAA", pcd="2026-05-06")]  # 400d from 2025-04-01
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=365)
        assert result == {}

    def test_trial_exactly_365d_boundary(self):
        """Trial at exactly 365d → captured (<=)."""
        # 2025-04-01 + 365d = 2026-04-01
        trials = [_trial("AAA", pcd="2026-04-01")]
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=365)
        assert result["days_to_catalyst"] == 365
        assert result["catalyst_source"] == "trial_pcd"

    def test_pit_filter_future_posted(self):
        """Trial first_posted > as_of → excluded regardless of window."""
        trials = [_trial("AAA", first_posted="2025-06-01", pcd="2025-09-01")]
        result = compute_catalyst("AAA", trials, [], AS_OF, trial_window_days=365)
        assert result == {}

    def test_pdufa_unaffected_by_window(self):
        """PDUFA dates are not constrained by trial_window_days."""
        pdufa = [_pdufa("AAA", "2026-06-01")]  # 426d from 2025-04-01
        result = compute_catalyst("AAA", [], pdufa, AS_OF, trial_window_days=180)
        assert result["days_to_catalyst"] == 426
        assert result["catalyst_source"] == "pdufa"

    def test_nearest_of_trial_and_pdufa(self):
        """When both trial and PDUFA present, nearest wins."""
        trials = [_trial("AAA", pcd="2025-08-01")]  # 122d
        pdufa = [_pdufa("AAA", "2025-06-01")]  # 61d
        result = compute_catalyst("AAA", trials, pdufa, AS_OF, trial_window_days=365)
        assert result["days_to_catalyst"] == 61
        assert result["catalyst_source"] == "pdufa"

    def test_default_window_is_365(self):
        """Default trial_window_days uses TRIAL_PCD_WINDOW_DAYS constant."""
        # Trial at 250d — would fail at 180 but pass at 365
        trials = [_trial("AAA", pcd="2025-12-07")]
        result = compute_catalyst("AAA", trials, [], AS_OF)
        assert result["days_to_catalyst"] == 250  # Captured with default 365d


class TestTrialWindowConstant:

    def test_constant_value(self):
        """TRIAL_PCD_WINDOW_DAYS == 365."""
        assert TRIAL_PCD_WINDOW_DAYS == 365


class TestCatalystOnlyMode:

    def test_catalyst_only_preserves_drawdown(self, tmp_path):
        """catalyst_only=True preserves existing defensive features."""
        import csv
        import json
        import shutil
        import tarfile

        date_str = "2025-06-30"
        archive_root = tmp_path / date_str
        inputs_dir = archive_root / "inputs"
        inputs_dir.mkdir(parents=True)

        # Write rankings.csv with one ticker
        csv_path = archive_root / "rankings.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker"])
            writer.writeheader()
            writer.writerow({"ticker": "AAA"})

        # Write existing decision_inputs.json with drawdown + old catalyst
        existing = {
            "AAA": {
                "drawdown": -0.25,
                "beta_xbi_60d": 1.1,
                "rsi_14d": 55.0,
                "alpha_60d": 0.03,
                "tier1_count": 2,
                "days_to_catalyst": 50,
                "in_optimal_window": True,
                "catalyst_source": "trial_pcd",
                "catalyst_event_type": "DATA_READOUT",
            }
        }
        with open(inputs_dir / "decision_inputs.json", "w") as f:
            json.dump(existing, f)

        # Write trial_records.json with a trial at 250d (outside old 180d window)
        trials = [{
            "ticker": "AAA", "phase": "PHASE3",
            "first_posted": "2024-01-01",
            "primary_completion_date": "2025-12-07",  # 160d from 2025-06-30
        }]
        with open(inputs_dir / "trial_records.json", "w") as f:
            json.dump(trials, f)
        with open(inputs_dir / "pdufa_dates.json", "w") as f:
            json.dump([], f)

        # Pack archive
        tar_path = tmp_path / f"{date_str}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)
        shutil.rmtree(archive_root)

        # Run catalyst_only enrichment (empty prices OK — not used)
        result = enrich_archive(tar_path, {}, catalyst_only=True, trial_window_days=365)
        assert result["status"] == "ok"

        # Extract and verify
        tmp_extract = tmp_path / "verify"
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_extract)
        with open(tmp_extract / date_str / "inputs" / "decision_inputs.json") as f:
            updated = json.load(f)

        aaa = updated["AAA"]
        # Defensive features preserved
        assert aaa["drawdown"] == -0.25
        assert aaa["beta_xbi_60d"] == 1.1
        assert aaa["rsi_14d"] == 55.0
        assert aaa["alpha_60d"] == 0.03
        assert aaa["tier1_count"] == 2
        # Catalyst updated (160d trial captured at 365d window)
        assert aaa["days_to_catalyst"] == 160
        assert aaa["catalyst_source"] == "trial_pcd"

    def test_catalyst_only_removes_stale_catalyst(self, tmp_path):
        """catalyst_only removes old catalyst when no trial matches new window."""
        import csv
        import json
        import shutil
        import tarfile

        date_str = "2025-06-30"
        archive_root = tmp_path / date_str
        inputs_dir = archive_root / "inputs"
        inputs_dir.mkdir(parents=True)

        csv_path = archive_root / "rankings.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker"])
            writer.writeheader()
            writer.writerow({"ticker": "BBB"})

        # Existing with catalyst from old enrichment
        existing = {
            "BBB": {
                "drawdown": -0.15,
                "days_to_catalyst": 50,
                "in_optimal_window": True,
                "catalyst_source": "trial_pcd",
                "catalyst_event_type": "DATA_READOUT",
            }
        }
        with open(inputs_dir / "decision_inputs.json", "w") as f:
            json.dump(existing, f)

        # No trials or PDUFA
        with open(inputs_dir / "trial_records.json", "w") as f:
            json.dump([], f)
        with open(inputs_dir / "pdufa_dates.json", "w") as f:
            json.dump([], f)

        tar_path = tmp_path / f"{date_str}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)
        shutil.rmtree(archive_root)

        result = enrich_archive(tar_path, {}, catalyst_only=True, trial_window_days=365)
        assert result["status"] == "ok"

        tmp_extract = tmp_path / "verify"
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp_extract)
        with open(tmp_extract / date_str / "inputs" / "decision_inputs.json") as f:
            updated = json.load(f)

        bbb = updated["BBB"]
        # Drawdown preserved
        assert bbb["drawdown"] == -0.15
        # Old catalyst keys removed (no matching trials)
        assert "days_to_catalyst" not in bbb
        assert "catalyst_source" not in bbb


class TestOutputTarPath:
    """Tests for output_tar_path: write to separate path, leave original intact."""

    def _make_archive(self, tmp_path, date_str="2025-06-30"):
        """Build a minimal archive and return (tar_path, original_hash)."""
        import csv
        import hashlib
        import json
        import shutil
        import tarfile

        archive_root = tmp_path / date_str
        inputs_dir = archive_root / "inputs"
        inputs_dir.mkdir(parents=True)

        csv_path = archive_root / "rankings.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticker"])
            writer.writeheader()
            writer.writerow({"ticker": "ZZZ"})

        existing = {"ZZZ": {"drawdown": -0.10}}
        with open(inputs_dir / "decision_inputs.json", "w") as f:
            json.dump(existing, f)

        with open(inputs_dir / "trial_records.json", "w") as f:
            json.dump([], f)
        with open(inputs_dir / "pdufa_dates.json", "w") as f:
            json.dump([], f)

        tar_path = tmp_path / f"{date_str}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(archive_root), arcname=date_str)
        shutil.rmtree(archive_root)

        original_hash = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        return tar_path, original_hash

    def test_enrich_output_tar_path_leaves_original(self, tmp_path):
        """enrich_archive with output_tar_path writes there, original unmodified."""
        import hashlib
        tar_path, original_hash = self._make_archive(tmp_path)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        out_tar = out_dir / tar_path.name

        result = enrich_archive(
            tar_path, {}, catalyst_only=True, output_tar_path=out_tar,
        )
        assert result["status"] == "ok"

        # Original unmodified
        assert hashlib.sha256(tar_path.read_bytes()).hexdigest() == original_hash
        # Output exists and differs
        assert out_tar.exists()
        assert out_tar.stat().st_size > 0

    def test_backfill_output_tar_path_leaves_original(self, tmp_path):
        """process_archive with output_tar_path writes there, original unmodified."""
        import hashlib
        import shutil

        from backfill_decision_columns import process_archive

        tar_path, original_hash = self._make_archive(tmp_path)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        out_tar = out_dir / tar_path.name
        # process_archive reads from tar_path, so copy first
        shutil.copy2(tar_path, out_tar)

        result = process_archive(out_tar, dry_run=False)
        assert result["status"] == "ok"

        # Original unmodified
        assert hashlib.sha256(tar_path.read_bytes()).hexdigest() == original_hash
        # Output was modified (backfill added columns)
        assert out_tar.exists()
        assert out_tar.stat().st_size > 0
