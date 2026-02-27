"""Manifest invariants: single active, no dup IDs, pinned constants match."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "production_data" / "decision_rulesets" / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Invariant 1: exactly one active entry
# ---------------------------------------------------------------------------

def test_manifest_exactly_one_active(manifest):
    actives = [r for r in manifest["rulesets"] if r["status"] == "active"]
    ids = [r["id"] for r in actives]
    assert len(actives) == 1, f"Expected 1 active, found {len(actives)}: {ids}"


# ---------------------------------------------------------------------------
# Invariant 2: no duplicate IDs with conflicting status
# ---------------------------------------------------------------------------

def test_manifest_ids_unique(manifest):
    counts = Counter(r["id"] for r in manifest["rulesets"])
    dups = {rid: n for rid, n in counts.items() if n > 1}
    assert not dups, f"Duplicate ruleset IDs: {dups}"


# ---------------------------------------------------------------------------
# Invariant 3: each (id, file) pair is unique
# ---------------------------------------------------------------------------

def test_manifest_id_file_pairs_unique(manifest):
    pairs = [(r["id"], r["file"]) for r in manifest["rulesets"]]
    counts = Counter(pairs)
    dups = {pair: n for pair, n in counts.items() if n > 1}
    assert not dups, f"Duplicate (id, file) pairs: {dups}"


# ---------------------------------------------------------------------------
# Invariant 4: active entry matches pinned constants in both scripts
# ---------------------------------------------------------------------------

def test_manifest_active_matches_pinned_constants(manifest):
    from run_screen import PHASE2_PINNED_RULESET_ID, PHASE2_DEFAULT_RULESET_PATH
    from run_phase2_snapshot_delta import PHASE2_PINNED_RULESET_ID as DELTA_PINNED_ID

    # Pinned IDs agree between modules
    assert PHASE2_PINNED_RULESET_ID == DELTA_PINNED_ID, (
        f"run_screen ({PHASE2_PINNED_RULESET_ID}) != "
        f"run_phase2_snapshot_delta ({DELTA_PINNED_ID})"
    )

    # Active manifest entry matches pinned ID and file
    actives = [r for r in manifest["rulesets"] if r["status"] == "active"]
    assert len(actives) == 1, "Need exactly 1 active to check pinned match"
    active = actives[0]

    assert active["id"] == PHASE2_PINNED_RULESET_ID, (
        f"Manifest active id '{active['id']}' != "
        f"PHASE2_PINNED_RULESET_ID '{PHASE2_PINNED_RULESET_ID}'"
    )
    assert active["file"] == PHASE2_DEFAULT_RULESET_PATH.name, (
        f"Manifest active file '{active['file']}' != "
        f"PHASE2_DEFAULT_RULESET_PATH.name '{PHASE2_DEFAULT_RULESET_PATH.name}'"
    )


# ---------------------------------------------------------------------------
# Invariant 5: pinned file exists and its computed ID matches
# ---------------------------------------------------------------------------

def test_pinned_ruleset_file_exists_and_id_matches():
    from run_screen import PHASE2_PINNED_RULESET_ID, PHASE2_DEFAULT_RULESET_PATH
    from decision_engine import DecisionRuleset

    assert PHASE2_DEFAULT_RULESET_PATH.exists(), (
        f"Pinned ruleset file missing: {PHASE2_DEFAULT_RULESET_PATH}"
    )

    rs = DecisionRuleset.from_json(str(PHASE2_DEFAULT_RULESET_PATH))
    assert rs.ruleset_id == PHASE2_PINNED_RULESET_ID, (
        f"File computes to '{rs.ruleset_id}', "
        f"pinned expects '{PHASE2_PINNED_RULESET_ID}'"
    )
