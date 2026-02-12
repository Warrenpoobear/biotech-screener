"""Validate the parked levers registry schema and invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "production_data"
    / "experiments"
    / "parked_levers.json"
)

VALID_STATUSES = {"PARKED", "ACTIVE", "RETIRED"}
VALID_DECISIONS = {"PARK", "PROMOTE", "RETIRE"}
REQUIRED_LEVER_KEYS = {
    "name",
    "ruleset_fields",
    "status",
    "decision",
    "evaluated_at",
    "summary",
    "metrics",
    "artifact_paths",
    "revisit_conditions",
}
REQUIRED_METRIC_KEYS = {
    "sep_delta_60d_pp",
    "turnover_delta_pp",
    "a_tier_mean_delta_pp",
}


@pytest.fixture(scope="module")
def registry():
    assert REGISTRY_PATH.exists(), f"Registry not found: {REGISTRY_PATH}"
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    return data


class TestRegistrySchema:
    def test_top_level_keys(self, registry):
        assert "schema_version" in registry
        assert "levers" in registry
        assert isinstance(registry["levers"], list)
        assert len(registry["levers"]) > 0

    def test_schema_version(self, registry):
        assert registry["schema_version"] == 1

    def test_lever_required_keys(self, registry):
        for lever in registry["levers"]:
            missing = REQUIRED_LEVER_KEYS - set(lever.keys())
            assert not missing, (
                f"Lever '{lever.get('name', '?')}' missing keys: {missing}"
            )

    def test_lever_status_valid(self, registry):
        for lever in registry["levers"]:
            assert lever["status"] in VALID_STATUSES, (
                f"Lever '{lever['name']}' has invalid status '{lever['status']}'. "
                f"Must be one of {VALID_STATUSES}"
            )

    def test_lever_decision_valid(self, registry):
        for lever in registry["levers"]:
            assert lever["decision"] in VALID_DECISIONS, (
                f"Lever '{lever['name']}' has invalid decision '{lever['decision']}'. "
                f"Must be one of {VALID_DECISIONS}"
            )

    def test_evaluated_at_is_iso_date(self, registry):
        import re
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for lever in registry["levers"]:
            assert date_re.match(lever["evaluated_at"]), (
                f"Lever '{lever['name']}' evaluated_at '{lever['evaluated_at']}' "
                f"is not YYYY-MM-DD format"
            )

    def test_artifact_paths_non_empty(self, registry):
        for lever in registry["levers"]:
            assert isinstance(lever["artifact_paths"], list), (
                f"Lever '{lever['name']}' artifact_paths must be a list"
            )
            assert len(lever["artifact_paths"]) >= 1, (
                f"Lever '{lever['name']}' must have at least one artifact path"
            )

    def test_metrics_has_required_keys(self, registry):
        for lever in registry["levers"]:
            metrics = lever["metrics"]
            assert isinstance(metrics, dict), (
                f"Lever '{lever['name']}' metrics must be a dict"
            )
            missing = REQUIRED_METRIC_KEYS - set(metrics.keys())
            assert not missing, (
                f"Lever '{lever['name']}' metrics missing keys: {missing}"
            )

    def test_metric_values_are_numeric_or_null(self, registry):
        for lever in registry["levers"]:
            metrics = lever["metrics"]
            for key in REQUIRED_METRIC_KEYS:
                val = metrics[key]
                assert val is None or isinstance(val, (int, float)), (
                    f"Lever '{lever['name']}' metric '{key}' must be "
                    f"numeric or null, got {type(val).__name__}"
                )

    def test_revisit_conditions_non_empty(self, registry):
        for lever in registry["levers"]:
            assert isinstance(lever["revisit_conditions"], list), (
                f"Lever '{lever['name']}' revisit_conditions must be a list"
            )
            assert len(lever["revisit_conditions"]) >= 1, (
                f"Lever '{lever['name']}' must have at least one revisit condition"
            )

    def test_unique_names(self, registry):
        names = [l["name"] for l in registry["levers"]]
        assert len(names) == len(set(names)), (
            f"Duplicate lever names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_ruleset_fields_are_strings(self, registry):
        for lever in registry["levers"]:
            assert isinstance(lever["ruleset_fields"], list), (
                f"Lever '{lever['name']}' ruleset_fields must be a list"
            )
            for field in lever["ruleset_fields"]:
                assert isinstance(field, str), (
                    f"Lever '{lever['name']}' ruleset_fields entry must be "
                    f"a string, got {type(field).__name__}"
                )


class TestRegistryConsistency:
    def test_status_decision_alignment(self, registry):
        """PARKED→PARK, ACTIVE→PROMOTE, RETIRED→RETIRE."""
        expected = {
            "PARKED": "PARK",
            "ACTIVE": "PROMOTE",
            "RETIRED": "RETIRE",
        }
        for lever in registry["levers"]:
            status = lever["status"]
            decision = lever["decision"]
            assert decision == expected[status], (
                f"Lever '{lever['name']}': status={status} expects "
                f"decision={expected[status]}, got {decision}"
            )

    def test_parked_has_at_least_one_numeric_metric_or_notes(self, registry):
        """Parked levers must document why they were evaluated."""
        for lever in registry["levers"]:
            if lever["status"] != "PARKED":
                continue
            metrics = lever["metrics"]
            has_numeric = any(
                metrics[k] is not None for k in REQUIRED_METRIC_KEYS
            )
            has_notes = bool(metrics.get("notes", "").strip())
            assert has_numeric or has_notes, (
                f"Parked lever '{lever['name']}' must have at least one "
                f"numeric metric or a notes field explaining the evaluation"
            )
