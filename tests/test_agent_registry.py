#!/usr/bin/env python3
"""
Lint for agents/AGENT_REGISTRY.json.

Bidirectional consistency:
- Every subdirectory of agents/ must appear in the registry exactly once.
- Every non-tombstone registry entry must correspond to an existing subdirectory.
- Every entry must carry all required fields with enum-valid values.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "agents"
REGISTRY_PATH = AGENTS_DIR / "AGENT_REGISTRY.json"

REQUIRED_FIELDS = {
    "role",
    "category",
    "cadence",
    "status",
    "artifact_paths",
    "authority_level",
    "supervised_by_orchestrator",
    "owner",
    "notes",
}


@pytest.fixture(scope="module")
def registry():
    with REGISTRY_PATH.open() as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="module")
def agent_dirs():
    return sorted(
        p.name for p in AGENTS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    )


def test_registry_file_exists():
    assert REGISTRY_PATH.exists(), f"Missing registry at {REGISTRY_PATH}"


def test_registry_has_enums(registry):
    assert "enums" in registry
    for key in ("category", "cadence", "status", "authority_level"):
        assert key in registry["enums"], f"Missing enum: {key}"


def test_every_directory_in_registry(registry, agent_dirs):
    registered = set(registry["agents"].keys())
    on_disk = set(agent_dirs)
    missing_from_registry = on_disk - registered
    assert (
        not missing_from_registry
    ), f"Agent directories missing from AGENT_REGISTRY.json: {sorted(missing_from_registry)}"


def test_every_registry_entry_has_directory(registry, agent_dirs):
    registered = set(registry["agents"].keys())
    on_disk = set(agent_dirs)
    dangling = []
    for name in sorted(registered - on_disk):
        entry = registry["agents"][name]
        is_tombstone = (
            entry.get("status") == "deprecated"
            and entry.get("requires_preflight") is False
            and entry.get("supervised_by_orchestrator") is False
            and entry.get("artifact_paths") == []
        )
        if not is_tombstone:
            dangling.append(name)
    assert not dangling, f"AGENT_REGISTRY.json references missing non-tombstone directories: {dangling}"


@pytest.mark.parametrize(
    "field",
    sorted(REQUIRED_FIELDS),
)
def test_every_entry_has_required_field(registry, field):
    missing = [name for name, entry in registry["agents"].items() if field not in entry]
    assert not missing, f"Agents missing required field '{field}': {missing}"


def test_enum_values_valid(registry):
    enums = registry["enums"]
    violations = []
    for name, entry in registry["agents"].items():
        for field in ("category", "cadence", "status", "authority_level"):
            value = entry.get(field)
            if value not in enums[field]:
                violations.append(f"{name}.{field}={value!r} not in {enums[field]}")
    assert not violations, "Enum violations:\n  " + "\n  ".join(violations)


def test_supervised_flag_is_boolean(registry):
    violations = [
        name
        for name, entry in registry["agents"].items()
        if not isinstance(entry.get("supervised_by_orchestrator"), bool)
    ]
    assert not violations, f"Non-boolean supervised_by_orchestrator: {violations}"


def test_artifact_paths_is_list(registry):
    violations = [
        name for name, entry in registry["agents"].items() if not isinstance(entry.get("artifact_paths"), list)
    ]
    assert not violations, f"artifact_paths must be a list: {violations}"


def test_deprecated_agents_declare_merged_into(registry):
    """Deprecated/suppressed agents must point at their canonical successor."""
    need_merge = {
        "bioshort_watch",
        "shadow_watch",
        "biotech_news_digest",
        "company_news_ingest",
        "policy_shadow_watch",
    }
    missing = [name for name in need_merge if not registry["agents"][name].get("merged_into")]
    assert not missing, f"Deprecated agents missing merged_into: {missing}"


def test_active_agents_declare_supervision(registry):
    """Active agents must declare supervised_by_orchestrator=true or explain why not.

    supervised_by_orchestrator is declarative: true means the orchestrator MUST
    check it. Setting false on an active agent is a deliberate opt-out and must
    be justified in the notes field.
    """
    violations = []
    for name, entry in registry["agents"].items():
        if entry.get("status") != "active":
            continue
        if entry.get("supervised_by_orchestrator") is True:
            continue
        note = entry.get("notes", "")
        if not note.strip():
            violations.append(f"{name}: active but supervised_by_orchestrator=false with empty notes")
    assert not violations, "Active agents opting out of orchestrator supervision must document why:\n  " + "\n  ".join(
        violations
    )
