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
    NA_FIELD_RULES,
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
            "alpha_cohort_pct": f"{0.5:.3f}",
            "de_drawdown": f"{-0.1 - 0.01 * i:.4f}",
            "de_rsi_14d": f"{50.0 + i:.1f}",
            "de_beta_xbi_60d": f"{0.8 + 0.01 * i:.3f}",
            "de_alpha_60d": f"{0.05:.4f}",
            # New columns for N/A-aware checks
            "has_commercial_quality": "0",
            "has_clinical_optionality_dev": "1",
            "clinical_optionality_pct_dev": f"{0.6 + 0.01 * i:.3f}",
            "clinical_score_z_tier": f"{0.1 * i:.4f}",
            "coinvest_score_z": f"{0.05 * i:.4f}",
            "inst_delta_z": f"{0.02 * i:.4f}",
            "catalyst_decay_w": f"{0.3:.4f}",
            "catalyst_mode": "specific_days",
            "catalyst_days": str(30 + i),
            "coinvest_recency_state": "fresh",
            "coinvest_filing_age_days": "45",
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
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert result.status == "PASS"

    def test_high_missingness(self):
        rows = _make_rows(10, missing_col="de_drawdown", missing_frac=0.6)
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert result.status == "WARN"
        assert total["de_drawdown"] >= 0.5

    def test_empty_rows(self):
        result, total, real = check_missingness([], warn_threshold=0.05)
        assert result.status == "WARN"

    def test_returns_three_tuple(self):
        rows = _make_rows(5)
        result = check_missingness(rows)
        assert len(result) == 3


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
# N/A-aware missingness tests
# ---------------------------------------------------------------------------

class TestNAAwareMissingness:
    def test_commercial_quality_pct_na_not_counted_as_real_miss(self):
        """drug_developer rows with empty commercial_quality_pct are expected N/A."""
        rows = _make_rows(10)
        for r in rows:
            r["archetype"] = "drug_developer"
            r["has_commercial_quality"] = "0"
            r["commercial_quality_pct"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["commercial_quality_pct"] == 1.0  # all missing
        assert real["commercial_quality_pct"] == 0.0    # all expected N/A

    def test_clinical_optionality_na_not_counted_as_real_miss(self):
        """commercial rows with empty clinical_optionality_pct_dev are expected N/A."""
        rows = _make_rows(10)
        for r in rows:
            r["archetype"] = "commercial_biotech"
            r["has_clinical_optionality_dev"] = "0"
            r["clinical_optionality_pct_dev"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["clinical_optionality_pct_dev"] == 1.0
        assert real["clinical_optionality_pct_dev"] == 0.0

    def test_actionable_rank_na_for_ineligible_not_counted(self):
        """Ineligible tickers with empty actionable_rank are expected N/A."""
        rows = _make_rows(10)
        for r in rows:
            r["eligible"] = "0"
            r["actionable_rank"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["actionable_rank"] == 1.0
        assert real["actionable_rank"] == 0.0

    def test_catalyst_days_na_for_no_upcoming_not_counted(self):
        """catalyst_mode=no_upcoming with empty catalyst_days is expected N/A."""
        rows = _make_rows(10)
        for r in rows:
            r["catalyst_mode"] = "no_upcoming"
            r["catalyst_days"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["catalyst_days"] == 1.0
        assert real["catalyst_days"] == 0.0

    def test_real_missingness_still_triggers_warn(self):
        """Real missingness (not N/A) should still trigger WARN."""
        rows = _make_rows(10)
        # de_drawdown has no N/A rule, so missing = real missing
        for r in rows:
            r["de_drawdown"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert result.status == "WARN"
        assert real["de_drawdown"] == 1.0

    def test_mixed_na_and_real_miss(self):
        """Mix of expected N/A and real missingness in same column."""
        rows = _make_rows(10)
        # 5 drug_developer (expected N/A for commercial_quality_pct)
        # 5 commercial_biotech (should have commercial_quality_pct)
        for i, r in enumerate(rows):
            if i < 5:
                r["archetype"] = "drug_developer"
                r["has_commercial_quality"] = "0"
                r["commercial_quality_pct"] = ""
            else:
                r["archetype"] = "commercial_biotech"
                r["has_commercial_quality"] = "1"
                r["commercial_quality_pct"] = ""  # real miss!
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["commercial_quality_pct"] == 1.0   # all 10 missing
        assert real["commercial_quality_pct"] == 0.5     # 5/10 real miss

    def test_coinvest_filing_age_days_na_when_no_recency(self):
        """Empty coinvest_recency_state → coinvest_filing_age_days is expected N/A."""
        rows = _make_rows(10)
        for r in rows:
            r["coinvest_recency_state"] = ""
            r["coinvest_filing_age_days"] = ""
        result, total, real = check_missingness(rows, warn_threshold=0.05)
        assert total["coinvest_filing_age_days"] == 1.0
        assert real["coinvest_filing_age_days"] == 0.0


# ---------------------------------------------------------------------------
# Breakdown tables tests
# ---------------------------------------------------------------------------

class TestBreakdownTables:
    def test_md_output_contains_archetype_breakdown(self, tmp_dir):
        rows = _make_rows(10)
        # Create some missingness > 5% to trigger breakdowns
        for i in range(8):
            rows[i]["de_drawdown"] = ""
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)
        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_md(sc, out_dir)

        md_path = out_dir / "scorecard_2025-06-01.md"
        content = md_path.read_text()
        assert "NaN Hotspots by Archetype" in content

    def test_md_output_contains_tier_breakdown(self, tmp_dir):
        rows = _make_rows(10)
        for i in range(8):
            rows[i]["de_drawdown"] = ""
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)
        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_md(sc, out_dir)

        md_path = out_dir / "scorecard_2025-06-01.md"
        content = md_path.read_text()
        assert "NaN Hotspots by Tier" in content

    def test_md_output_contains_catalyst_breakdown(self, tmp_dir):
        rows = _make_rows(10)
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir)
        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_md(sc, out_dir)

        md_path = out_dir / "scorecard_2025-06-01.md"
        content = md_path.read_text()
        assert "Catalyst Days Missingness by Mode" in content

    def test_md_missingness_table_has_real_vs_na(self, tmp_dir):
        """Markdown output should show Total Miss, Real Miss, Expected N/A columns."""
        rows = _make_rows(10)
        for r in rows:
            r["commercial_quality_pct"] = ""
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)
        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_md(sc, out_dir)

        content = (out_dir / "scorecard_2025-06-01.md").read_text()
        assert "Real Miss" in content
        assert "Expected N/A" in content


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

    def test_na_fields_dont_cause_warn(self, tmp_dir):
        """Snapshot where all missingness is expected N/A should PASS."""
        rows = _make_rows(10)
        for r in rows:
            # drug_developer → commercial_quality_pct is expected N/A
            r["commercial_quality_pct"] = ""
            r["has_commercial_quality"] = "0"
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)
        # missingness check should PASS since it's all expected N/A
        miss_check = next(c for c in sc.checks if c["name"] == "missingness")
        assert miss_check["status"] == "PASS"

    def test_real_missingness_detail_in_json(self, tmp_dir):
        snap_dir = tmp_dir / "2025-06-01"
        rows = _make_rows(10, missing_col="de_drawdown", missing_frac=0.8)
        _write_snapshot(snap_dir, rows)

        sc = run_scorecard(snap_dir, warn_missingness=0.05)
        data = sc.to_dict()
        assert "de_drawdown" in data.get("real_missingness_detail", {})


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

    def test_json_excludes_raw_rows(self, tmp_dir):
        """The rows field should not be serialized to JSON."""
        snap_dir = tmp_dir / "2025-06-01"
        _write_snapshot(snap_dir, _make_rows(5))
        sc = run_scorecard(snap_dir)

        out_dir = tmp_dir / "output"
        out_dir.mkdir()
        write_scorecard_json(sc, out_dir)

        with open(out_dir / "scorecard_2025-06-01.json") as f:
            data = json.load(f)
        assert "rows" not in data
