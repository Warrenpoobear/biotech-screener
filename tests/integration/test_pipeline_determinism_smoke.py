"""
Smoke tests: pipeline determinism and PIT (point-in-time) safety.

Tests:
1. Determinism — identical inputs produce byte-identical JSON output.
2. PIT strict — future-dated trial data raises PITViolationError.
3. PIT degrade — future-dated trial data completes with degraded catalyst_mode.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

from catalyst_diagnostics import PITViolationError
from run_screen import run_screening_pipeline


# ---------------------------------------------------------------------------
# Common pipeline kwargs (disable optional features not present in fixtures)
# ---------------------------------------------------------------------------
_PIPELINE_KWARGS = dict(
    as_of_date="2026-01-15",
    pit_mode="strict",
    enable_enhancements=False,
    enable_short_interest=False,
    enable_coinvest=False,
    no_clinical_filter=True,
    ctgov_cache_dir=False,  # Disable ctgov cache; use data_dir trial_records.json
)


def _serialize(result: dict) -> str:
    """Canonical JSON serialization for hashing."""
    return json.dumps(result, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_trial_records(data_dir: Path) -> None:
    """Patch trial records so they pass pipeline schema/adapter validation.

    Adds ``ticker`` (from ``sponsor_ticker``) and ``last_update_posted``
    if missing — both are required by the ctgov adapter.
    """
    trial_path = data_dir / "trial_records.json"
    trials = json.loads(trial_path.read_text())
    for rec in trials:
        if "ticker" not in rec and "sponsor_ticker" in rec:
            rec["ticker"] = rec["sponsor_ticker"]
        if "last_update_posted" not in rec:
            rec["last_update_posted"] = "2026-01-10"
    trial_path.write_text(json.dumps(trials, indent=2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_data_dir(sample_data_dir: Path) -> Path:
    """sample_data_dir with trial records patched to include ``ticker``."""
    _patch_trial_records(sample_data_dir)
    return sample_data_dir


@pytest.fixture
def pit_data_dir(sample_data_dir: Path) -> Path:
    """Copy sample_data_dir but inject future-dated trial metadata.

    Sets both ``_metadata.data_date`` (priority-1 path in staleness check)
    and ``last_update_posted`` on every record to 2026-02-01, which is
    *after* the as_of_date of 2026-01-15.
    """
    _patch_trial_records(sample_data_dir)

    pit_dir = sample_data_dir.parent / "pit_data"
    shutil.copytree(sample_data_dir, pit_dir)

    trial_path = pit_dir / "trial_records.json"
    trials = json.loads(trial_path.read_text())

    future_date = "2026-02-01"

    # Inject _metadata on the first record (priority-1 for staleness check)
    trials[0]["_metadata"] = {"data_date": future_date}

    # Also set last_update_posted on all records (priority-2 fallback)
    for record in trials:
        record["last_update_posted"] = future_date

    trial_path.write_text(json.dumps(trials, indent=2))
    return pit_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineDeterminism:
    """run_screening_pipeline must be deterministic for identical inputs."""

    def test_determinism_sha256(self, patched_data_dir: Path):
        # Each run gets its own copy of the data directory so that
        # side-effects (snapshot files, catalyst event JSONs written by
        # Module 3) don't cause input_hashes to diverge on the second run.
        dir_a = patched_data_dir.parent / "run_a"
        dir_b = patched_data_dir.parent / "run_b"
        shutil.copytree(patched_data_dir, dir_a)
        shutil.copytree(patched_data_dir, dir_b)

        result_a = run_screening_pipeline(data_dir=dir_a, **_PIPELINE_KWARGS)
        result_b = run_screening_pipeline(data_dir=dir_b, **_PIPELINE_KWARGS)

        hash_a = hashlib.sha256(_serialize(result_a).encode()).hexdigest()
        hash_b = hashlib.sha256(_serialize(result_b).encode()).hexdigest()

        assert hash_a == hash_b, (
            f"Non-deterministic output.\n  run 1: {hash_a}\n  run 2: {hash_b}"
        )


class TestPITSafety:
    """PIT gate must reject or degrade on future-dated trial data."""

    def test_pit_strict_rejects_future_data(self, pit_data_dir: Path):
        with pytest.raises(PITViolationError):
            run_screening_pipeline(
                data_dir=pit_data_dir,
                **{**_PIPELINE_KWARGS, "pit_mode": "strict"},
            )

    def test_pit_degrade_completes(self, pit_data_dir: Path):
        result = run_screening_pipeline(
            data_dir=pit_data_dir,
            **{**_PIPELINE_KWARGS, "pit_mode": "degrade"},
        )

        assert result is not None
        catalyst_mode = result["run_metadata"]["catalyst_mode"]
        assert catalyst_mode == "corporate_only_due_to_pit_violation", (
            f"Expected degraded catalyst_mode, got: {catalyst_mode!r}"
        )
