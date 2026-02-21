"""Tests for scripts/internal_consistency_scorecard.py."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.internal_consistency_scorecard import (
    SCHEMA_VERSION,
    CheckResult,
    Scorecard,
    check_duplicate_tickers,
    check_eligibility_tier_consistency,
    check_missingness,
    check_nan_hotspots,
    check_rank_invariants,
    check_required_columns,
    check_tie_density,
    run_scorecard,
    write_scorecard_json,
    write_scorecard_md,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_rows(
    n: int = 10,
    *,
    duplicates: bool = False,
    missing_col: str | None = None,
    missing_frac: float = 0.0,
    tie_rank: bool = False,
    bad_rank_sequence: bool = False,
    ineligible_with_tier: bool = False,
) -> List[Dict[str, str]]:
    """Generate synthetic rankings rows for testing."""
    rows = []
    for i in range(n):
        ticker = f"T{i}"
        if duplicates and i == n - 1:
            ticker = "T0"  # duplicate first ticker

        rank = str(i + 1)
        if tie_rank and i > 0:
            rank = "1"  # all tied at rank 1
        if bad_rank_sequence and i == n - 1:
            rank = str(n + 5)  # gap in sequence

        eligible = "1"
        tier_dev = "A"
        tier_any = "A"
        if ineligible_with_tier and i == 0:
            eligible = "0"
            tier_dev = "A"
            tier_any = ""

        row = {
            "ticker": ticker,
            "actionable_rank": rank,
            "eligible": eligible,
            "tier_dev": tier_dev,
            "tier_any": tier_any,
            "composite_rank": rank,
            "composite_score": f"{50.0 + i:.1f}",
            "archetype": "drug_developer",
            "score_rank_pct": f"{0.1 * i:.2f}",
            "clinical_optionality_pct_dev": f"{0.6 + 0.01 * i:.3f}",
            "alpha_cohort_pct": f"{0.5:.3f}",
            "de_drawdown": f"{-0.1 - 0.01 * i:.4f}",
            "de_rsi_14d": f"{50.0 + i:.1f}",
            "de_beta_xbi_60d": f"{0.8 + 0.01 * i:.3f}",
            "de_alpha_60d": f"{0.05:.4f}",
        }

        if missing_col and i < int(n * missing_frac):
            row[missing_col] = ""

        rows.append(row)

    return rows


def _write_snapshot(snap_dir: Path, rows: List[Dict[str, str]]) -> None:
    """Write rankings.csv into a snapshot directory."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(snap_dir / "rankings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Unit tests: duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateTickers:
    def test_no_duplicates(self):
        rows = _make_rows(5)
        result = check_duplicate_tickers(rows)
        assert result.status == "PASS"

    def test_finds_duplicates(self):
        rows = _make_rows(5, duplicates=True)
        result = check_duplicate_tickers(rows)
        assert result.status == "WARN"
        assert "T0" in result.detail


# ---------------------------------------------------------------------------
# Unit tests: missingness
# ---------------------------------------------------------------------------

class TestMissingness:
    def test_clean_data(self):
        rows = _make_rows(10)
        result, rates = check_missingness(rows, warn_threshold=0.05)
        assert result.status == "PASS"

    def test_high_missingness(self):
        rows = _make_rows(10, missing_col="de_drawdown", missing_frac=0.6)
        result, rates = check_missingness(rows, warn_threshold=0.05)
        assert result.status == "WARN"
        assert rates["de_drawdown"] >= 0.5

    def test_empty_rows(self):
        result, rates = check_missingness([], warn_threshold=0.05)
        assert result.status == "WARN"


# ---------------------------------------------------------------------------
# Unit tests: NaN hotspots
# ---------------------------------------------------------------------------

class TestNaNHotspots:
    def test_no_nans(self):
        rows = _make_rows(10)
        result = check_nan_hotspots(rows)
        assert result.status == "PASS"

    def test_detects_nan_strings(self):
        rows = _make_rows(10)
        # Inject NaN values in > 10% of rows
        for i in range(5):
            rows[i]["de_drawdown"] = "nan"
        result = check_nan_hotspots(rows)
        assert "de_drawdown" in str(result.detail)


# ---------------------------------------------------------------------------
# Unit tests: tie density
# ---------------------------------------------------------------------------

class TestTieDensity:
    def test_no_ties(self):
        rows = _make_rows(10)
        result = check_tie_density(rows, warn_threshold=0.02)
        assert result.status == "PASS"

    def test_excessive_ties(self):
        rows = _make_rows(10, tie_rank=True)
        result = check_tie_density(rows, warn_threshold=0.02)
        assert result.status == "WARN"
        assert "ties" in result.detail.lower()


# ---------------------------------------------------------------------------
# Unit tests: rank invariants
# ---------------------------------------------------------------------------

class TestRankInvariants:
    def test_contiguous_ranks(self):
        rows = _make_rows(10)
        result = check_rank_invariants(rows)
        assert result.status == "PASS"

    def test_gap_in_ranks(self):
        rows = _make_rows(10, bad_rank_sequence=True)
        result = check_rank_invariants(rows)
        assert result.status == "WARN"

    def test_no_eligible(self):
        rows = _make_rows(3)
        for r in rows:
            r["eligible"] = "0"
        result = check_rank_invariants(rows)
        assert result.status == "PASS"  # nothing to check


# ---------------------------------------------------------------------------
# Unit tests: eligibility/tier consistency
# ---------------------------------------------------------------------------

class TestEligibilityConsistency:
    def test_consistent(self):
        rows = _make_rows(10)
        result = check_eligibility_tier_consistency(rows)
        assert result.status == "PASS"

    def test_ineligible_with_tier(self):
        rows = _make_rows(10, ineligible_with_tier=True)
        result = check_eligibility_tier_consistency(rows)
        assert result.status == "WARN"


# ---------------------------------------------------------------------------
# Unit tests: required columns
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_all_present(self):
        rows = _make_rows(3)
        result = check_required_columns(rows)
        assert result.status == "PASS"

    def test_missing_column(self):
        rows = _make_rows(3)
        for r in rows:
            del r["archetype"]
        result = check_required_columns(rows)
        assert result.status == "WARN"
        assert "archetype" in result.detail

    def test_empty_rows(self):
        result = check_required_columns([])
        assert result.status == "WARN"


# ---------------------------------------------------------------------------
# Integration: run_scorecard
# ---------------------------------------------------------------------------

class TestRunScorecard:
    def test_clean_snapshot(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, _make_rows(20))

        sc = run_scorecard(snap_dir)
        assert sc.verdict == "PASS"
        assert sc.n_rows == 20
        assert sc.schema == SCHEMA_VERSION

    def test_problematic_snapshot(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        rows = _make_rows(10, duplicates=True, tie_rank=True)
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir)
        assert sc.verdict == "WARN"

    def test_missing_csv(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        snap_dir.mkdir(parents=True)

        sc = run_scorecard(snap_dir)
        assert sc.n_rows == 0


# ---------------------------------------------------------------------------
# Integration: output writers
# ---------------------------------------------------------------------------

class TestOutputWriters:
    def test_json_md_written(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, _make_rows(10))

        sc = run_scorecard(snap_dir)

        out_dir = tmp_dir / "output"
        out_dir.mkdir()

        write_scorecard_json(sc, out_dir)
        write_scorecard_md(sc, out_dir)

        json_path = out_dir / "scorecard_2025-06-01.json"
        md_path = out_dir / "scorecard_2025-06-01.md"

        assert json_path.exists()
        assert md_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["schema"] == SCHEMA_VERSION
        assert data["n_rows"] == 10

    def test_missingness_detail_in_json(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        rows = _make_rows(10, missing_col="de_drawdown", missing_frac=0.8)
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)

        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_json(sc, out_dir)

        with open(out_dir / "scorecard_2025-06-01.json") as f:
            data = json.load(f)
        # Should have missingness detail for de_drawdown
        assert "de_drawdown" in data.get("missingness_detail", {})
