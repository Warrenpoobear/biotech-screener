"""
Integration test suite for run_screening_pipeline() using real production data
subsetted to 8 tickers + XBI.

Tests:
- Determinism: identical inputs produce byte-identical output.
- PIT safety: future-dated data is rejected (strict) or degraded.
- Output contracts: all required keys, schemas, and invariants hold.

Fixture data lives in tests/fixtures/mini_data/ (committed).
Rebuild with: python3 scripts/build_mini_fixtures.py
"""

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from catalyst_diagnostics import PITViolationError
from run_screen import run_screening_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINI_DATA = Path(__file__).resolve().parent.parent / "fixtures" / "mini_data"
AS_OF_DATE = "2026-01-15"

_PIPELINE_KWARGS = dict(
    as_of_date=AS_OF_DATE,
    pit_mode="degrade",
    enable_enhancements=False,
    enable_short_interest=False,
    enable_coinvest=False,
    no_clinical_filter=True,
)


def _serialize(result: dict) -> str:
    """Canonical JSON serialization for hashing."""
    return json.dumps(result, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_data_dir(tmp_path):
    """Copy MINI_DATA to an isolated tmp dir so side-effects don't leak."""
    dest = tmp_path / "data"
    shutil.copytree(MINI_DATA, dest)
    return dest


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """Run the pipeline once and share the result across TestOutputContracts.

    Uses module scope to avoid re-running the pipeline for each contract test.
    """
    dest = tmp_path_factory.mktemp("contract") / "data"
    shutil.copytree(MINI_DATA, dest)
    return run_screening_pipeline(data_dir=dest, **_PIPELINE_KWARGS)


# ---------------------------------------------------------------------------
# Class 1: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Pipeline must produce identical output for identical inputs."""

    def test_determinism_sha256(self, tmp_path):
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        shutil.copytree(MINI_DATA, dir_a)
        shutil.copytree(MINI_DATA, dir_b)

        result_a = run_screening_pipeline(data_dir=dir_a, **_PIPELINE_KWARGS)
        result_b = run_screening_pipeline(data_dir=dir_b, **_PIPELINE_KWARGS)

        hash_a = hashlib.sha256(_serialize(result_a).encode()).hexdigest()
        hash_b = hashlib.sha256(_serialize(result_b).encode()).hexdigest()

        assert hash_a == hash_b, (
            f"Non-deterministic output.\n  run 1: {hash_a}\n  run 2: {hash_b}"
        )


# ---------------------------------------------------------------------------
# Class 2: PIT Safety
# ---------------------------------------------------------------------------

def _inject_future_dates(data_dir: Path, future_date: str = "2026-02-01") -> None:
    """Patch trial records with future-dated metadata for PIT testing."""
    trial_path = data_dir / "trial_records.json"
    trials = json.loads(trial_path.read_text())

    # Priority-1: _metadata.data_date on first record
    trials[0]["_metadata"] = {"data_date": future_date}

    # Priority-2: last_update_posted on all records
    for rec in trials:
        rec["last_update_posted"] = future_date

    trial_path.write_text(json.dumps(trials, indent=2))


class TestPITSafety:
    """PIT gate must reject or degrade on future-dated trial data."""

    def test_pit_strict_rejects_future_data(self, fresh_data_dir):
        _inject_future_dates(fresh_data_dir)
        with pytest.raises(PITViolationError):
            run_screening_pipeline(
                data_dir=fresh_data_dir,
                **{**_PIPELINE_KWARGS, "pit_mode": "strict"},
            )

    def test_pit_degrade_completes(self, fresh_data_dir):
        _inject_future_dates(fresh_data_dir)
        result = run_screening_pipeline(
            data_dir=fresh_data_dir,
            **{**_PIPELINE_KWARGS, "pit_mode": "degrade"},
        )
        assert result is not None
        catalyst_mode = result["run_metadata"]["catalyst_mode"]
        assert catalyst_mode == "corporate_only_due_to_pit_violation", (
            f"Expected degraded catalyst_mode, got: {catalyst_mode!r}"
        )

    def test_pit_degrade_catalyst_gated(self, fresh_data_dir):
        _inject_future_dates(fresh_data_dir)
        result = run_screening_pipeline(
            data_dir=fresh_data_dir,
            **{**_PIPELINE_KWARGS, "pit_mode": "degrade"},
        )

        # Catalyst component should be gated across all tickers
        composite = result["module_5_composite"]
        gated = composite.get("gated_component_counts", {})
        assert gated.get("catalyst", 0) > 0, (
            "Expected catalyst to appear in gated_component_counts"
        )

        # Every ranked security should have catalyst effective weight of 0
        for entry in composite["ranked_securities"]:
            ew_catalyst = entry["effective_weights"].get("catalyst", "")
            assert Decimal(ew_catalyst) == Decimal("0"), (
                f"{entry['ticker']}: expected catalyst effective_weight 0, "
                f"got {ew_catalyst!r}"
            )


# ---------------------------------------------------------------------------
# Class 3: Output Contracts
# ---------------------------------------------------------------------------

class TestOutputContracts:
    """Validate the shape and invariants of pipeline output."""

    # --- Top-level structure ---

    def test_top_level_keys(self, pipeline_result):
        expected = {
            "run_metadata",
            "module_1_universe",
            "module_2_financial",
            "module_3_catalyst",
            "module_4_clinical",
            "module_5_composite",
            "summary",
            "clinical_exclusions",
            "clinical_exemptions",
            "company_archetypes",
        }
        assert expected.issubset(pipeline_result.keys()), (
            f"Missing keys: {expected - pipeline_result.keys()}"
        )

    def test_run_metadata_keys(self, pipeline_result):
        meta = pipeline_result["run_metadata"]
        expected = {
            "as_of_date",
            "version",
            "deterministic_timestamp",
            "input_hashes",
            "enhancements_enabled",
            "catalyst_window",
            "catalyst_decay",
            "catalyst_half_life_days",
            "enable_smart_money",
            "smart_money_source",
            "min_component_coverage",
            "min_component_coverage_mode",
            "defensive_multiplier_enabled",
            "defensive_config",
            "position_sizing_enabled",
            "pit_mode",
            "pit_violation_sources",
            "catalyst_mode",
        }
        assert expected.issubset(meta.keys()), (
            f"Missing run_metadata keys: {expected - meta.keys()}"
        )

    # --- Module 5 ranked securities ---

    def test_ranked_securities_schema(self, pipeline_result):
        ranked = pipeline_result["module_5_composite"]["ranked_securities"]
        assert len(ranked) > 0, "No ranked securities returned"

        required_fields = {
            "ticker": str,
            "composite_score": str,
            "composite_rank": int,
            "severity": str,
            "flags": list,
            "effective_weights": dict,
            "score_breakdown": dict,
            "schema_version": str,
        }

        for entry in ranked:
            for field, expected_type in required_fields.items():
                assert field in entry, (
                    f"Missing field '{field}' in ranked entry for "
                    f"{entry.get('ticker', '???')}"
                )
                assert isinstance(entry[field], expected_type), (
                    f"Field '{field}' for {entry.get('ticker', '???')}: "
                    f"expected {expected_type.__name__}, "
                    f"got {type(entry[field]).__name__}"
                )

    def test_ranked_securities_score_bounds(self, pipeline_result):
        ranked = pipeline_result["module_5_composite"]["ranked_securities"]
        for entry in ranked:
            score = Decimal(entry["composite_score"])
            assert Decimal("0") <= score <= Decimal("100"), (
                f"{entry['ticker']}: composite_score {score} out of [0, 100]"
            )

    # --- Summary ---

    def test_summary_keys(self, pipeline_result):
        summary = pipeline_result["summary"]
        expected = {
            "total_evaluated",
            "active_universe",
            "excluded",
            "clinical_activity_filter",
            "final_ranked",
            "catalyst_events",
            "severe_negatives",
        }
        assert expected.issubset(summary.keys()), (
            f"Missing summary keys: {expected - summary.keys()}"
        )

    # --- Diagnostic counts ---

    def test_module_outputs_have_diagnostic_counts(self, pipeline_result):
        modules_with_diagnostics = [
            "module_2_financial",
            "module_3_catalyst",
            "module_4_clinical",
            "module_5_composite",
        ]
        for mod_key in modules_with_diagnostics:
            mod = pipeline_result[mod_key]
            assert "diagnostic_counts" in mod, (
                f"{mod_key} missing 'diagnostic_counts'"
            )
