"""Draft Track B production-governance contracts.

These tests intentionally describe the fail-closed behavior requested for the
deferred production hardening track. They are test-only/spec guardrails and do
not modify production ranker, snapshot, or promotion behavior.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_pairwise_ranker_mode_fails_closed_when_v2_scoring_fails():
    """pairwise_minimal must not silently fall back to clinical_50."""
    src = _source("run_screen.py")

    assert "Ranker mode pairwise_minimal requested but model failed; fell back to clinical_50" not in src
    assert re.search(
        r"if\s+ranker_mode\s*==\s*[\"']pairwise_minimal[\"']\s+and\s+not\s+_rv2_ok:\s*raise\b",
        src,
        flags=re.DOTALL,
    ), "ranker_mode=pairwise_minimal must raise when v2 scoring/parsing fails"


def test_snapshot_save_failure_is_not_reported_as_success():
    """A failed snapshot write must produce a non-zero main() result."""
    src = _source("run_screen.py")

    assert re.search(
        r"snap_result\s*=\s*save_validation_snapshot\(.*?if\s+not\s+snap_result:\s*.*?return\s+[1-9]",
        src,
        flags=re.DOTALL,
    ), "run_screen.py main() must hard-fail when save_validation_snapshot returns None"


def test_strict_phase2_exit_one_does_not_promote_snapshot():
    """Daily production --strict exit code 1 must not be downgraded to WARN."""
    src = _source("tools/run_daily_production.py")

    assert "Phase-2 health FAIL (exit 1) — snapshot promoted with downstream artifacts" not in src
    assert re.search(
        r"if\s+screen_proc\.returncode\s*==\s*1:\s*.*?status\s*=\s*[\"']FAIL[\"']",
        src,
        flags=re.DOTALL,
    ), "run_daily_production.py must treat strict Phase-2 exit 1 as a non-promotable failure"


def test_snapshot_enrichment_sidecars_are_pit_bounded():
    """Earnings and AACT sidecar fallbacks must be bounded by as_of_date."""
    src = _source("run_screen.py")

    assert '_earnings_dir.glob("earnings_raw_*.json")' not in src
    assert '_aact_dir.glob("aact_deltas_*.json")' not in src
    assert "adcom_calendar_{as_of_date}.json" in src
