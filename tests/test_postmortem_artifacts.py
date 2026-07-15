"""Tests for postmortem artifact schema and consumer contracts.

The postmortem agent (agents/postmortem/) produces structured JSON records
when catalysts resolve.  These records are consumed by:
- event_analyst agent (reads pre_event state)
- calibration_evidence builder (reads outcome returns)
- signal evidence harness (reads for promotion governance)

This test suite validates:
1. Schema structure matches postmortem.v1
2. Required fields are present and correctly typed
3. Pre-event state fields match what consumers expect
4. Outcome return fields are well-formed
5. Agent workspace files are complete
"""

import json
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTMORTEM_DIR = REPO_ROOT / "agents" / "postmortem"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "postmortem"

SCHEMA_VERSION = "postmortem.v1"

# Fields required by the schema defined in agents/postmortem/TOOLS.md
PRE_EVENT_REQUIRED_FIELDS = {
    "snapshot_date": str,
    "actionable_rank": int,
    "tier_dev": str,
    "catalyst_days": int,
    "catalyst_family": str,
    "is_hard_catalyst": bool,
    "in_shadow": bool,
    "in_trade_plan": bool,
    "readiness_verdict": str,
    "ruleset_id": str,
}

OUTCOME_REQUIRED_FIELDS = {
    "return_t1": (int, float, type(None)),
    "return_t3": (int, float, type(None)),
    "return_t5": (int, float, type(None)),
    "excess_vs_xbi_t1": (int, float, type(None)),
    "excess_vs_xbi_t3": (int, float, type(None)),
    "abs_gap": (int, float, type(None)),
}

TOP_LEVEL_REQUIRED_FIELDS = {
    "schema": str,
    "ticker": str,
    "event_date": str,
    "captured_at": str,
    "pre_event": dict,
    "outcome": dict,
}


# ---------------------------------------------------------------------------
# Fixture: example postmortem record matching TOOLS.md schema
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_postmortem():
    """A valid postmortem.v1 record."""
    return {
        "schema": SCHEMA_VERSION,
        "ticker": "CELC",
        "event_date": "2026-04-01",
        "captured_at": "2026-04-04T18:30:00Z",
        "pre_event": {
            "snapshot_date": "2026-03-31",
            "actionable_rank": 7,
            "tier_dev": "A",
            "catalyst_days": 1,
            "catalyst_family": "CLINICAL",
            "is_hard_catalyst": True,
            "in_shadow": True,
            "in_trade_plan": False,
            "readiness_verdict": "REVIEW",
            "ruleset_id": "9f1f4587",
        },
        "outcome": {
            "return_t1": 0.12,
            "return_t3": 0.08,
            "return_t5": 0.05,
            "excess_vs_xbi_t1": 0.11,
            "excess_vs_xbi_t3": 0.07,
            "abs_gap": 0.15,
        },
    }


def _validate_postmortem(record: dict) -> list:
    """Validate a postmortem record, returning a list of error strings."""
    errors = []

    # Top-level fields
    for field, expected_type in TOP_LEVEL_REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"Missing top-level field: {field}")
        elif not isinstance(record[field], expected_type):
            errors.append(f"Field {field}: expected {expected_type.__name__}, " f"got {type(record[field]).__name__}")

    # Schema version
    if record.get("schema") != SCHEMA_VERSION:
        errors.append(f"Schema mismatch: expected {SCHEMA_VERSION}, got {record.get('schema')}")

    # Ticker format
    ticker = record.get("ticker", "")
    if ticker and not ticker.isupper():
        errors.append(f"Ticker should be uppercase: {ticker}")

    # Event date format (YYYY-MM-DD)
    event_date = record.get("event_date", "")
    if event_date:
        try:
            date.fromisoformat(event_date)
        except ValueError:
            errors.append(f"Invalid event_date format: {event_date}")

    # Pre-event fields (None is acceptable for optional fields in older artifacts)
    pre_event = record.get("pre_event", {})
    for field, expected_type in PRE_EVENT_REQUIRED_FIELDS.items():
        if field not in pre_event:
            errors.append(f"Missing pre_event field: {field}")
        elif pre_event[field] is not None and not isinstance(pre_event[field], expected_type):
            errors.append(
                f"pre_event.{field}: expected {expected_type.__name__}, " f"got {type(pre_event[field]).__name__}"
            )

    # Ruleset ID must be non-empty
    if not pre_event.get("ruleset_id"):
        errors.append("pre_event.ruleset_id must be non-empty")

    # Tier must be valid
    valid_tiers = {"A", "B", "C", "D", "INELIGIBLE", "COMMERCIAL"}
    if pre_event.get("tier_dev") and pre_event["tier_dev"] not in valid_tiers:
        errors.append(f"Invalid tier_dev: {pre_event['tier_dev']}")

    # Readiness verdict must be valid
    valid_verdicts = {"READY", "REVIEW", "HOLD", "NOT_RUN"}
    if pre_event.get("readiness_verdict") and pre_event["readiness_verdict"] not in valid_verdicts:
        errors.append(f"Invalid readiness_verdict: {pre_event['readiness_verdict']}")

    # Outcome fields (abs_gap added later — tolerate absence in older artifacts)
    _outcome_optional = {"abs_gap"}
    outcome = record.get("outcome", {})
    for field, expected_types in OUTCOME_REQUIRED_FIELDS.items():
        if field not in outcome:
            if field not in _outcome_optional:
                errors.append(f"Missing outcome field: {field}")
        elif outcome[field] is not None and not isinstance(outcome[field], (int, float)):
            errors.append(f"outcome.{field}: expected numeric or None, " f"got {type(outcome[field]).__name__}")

    # Return magnitudes should be reasonable (sanity)
    for field in ("return_t1", "return_t3", "return_t5"):
        val = outcome.get(field)
        if val is not None and abs(val) > 5.0:
            errors.append(f"outcome.{field}={val} seems unreasonable (>500% return)")

    return errors


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestPostmortemSchemaValid:
    def test_sample_validates(self, sample_postmortem):
        errors = _validate_postmortem(sample_postmortem)
        assert errors == [], f"Validation errors: {errors}"

    def test_schema_version(self, sample_postmortem):
        assert sample_postmortem["schema"] == SCHEMA_VERSION

    def test_ticker_uppercase(self, sample_postmortem):
        assert sample_postmortem["ticker"].isupper()

    def test_event_date_iso(self, sample_postmortem):
        parsed = date.fromisoformat(sample_postmortem["event_date"])
        assert parsed.isoformat() == sample_postmortem["event_date"]

    def test_pre_event_all_fields(self, sample_postmortem):
        pre = sample_postmortem["pre_event"]
        for field in PRE_EVENT_REQUIRED_FIELDS:
            assert field in pre, f"Missing: {field}"

    def test_outcome_all_fields(self, sample_postmortem):
        outcome = sample_postmortem["outcome"]
        for field in OUTCOME_REQUIRED_FIELDS:
            assert field in outcome, f"Missing: {field}"


class TestPostmortemSchemaInvalid:
    def test_missing_ticker(self, sample_postmortem):
        del sample_postmortem["ticker"]
        errors = _validate_postmortem(sample_postmortem)
        assert any("ticker" in e for e in errors)

    def test_missing_pre_event(self, sample_postmortem):
        del sample_postmortem["pre_event"]
        errors = _validate_postmortem(sample_postmortem)
        assert any("pre_event" in e for e in errors)

    def test_missing_outcome(self, sample_postmortem):
        del sample_postmortem["outcome"]
        errors = _validate_postmortem(sample_postmortem)
        assert any("outcome" in e for e in errors)

    def test_wrong_schema_version(self, sample_postmortem):
        sample_postmortem["schema"] = "postmortem.v99"
        errors = _validate_postmortem(sample_postmortem)
        assert any("Schema mismatch" in e for e in errors)

    def test_missing_ruleset_id(self, sample_postmortem):
        sample_postmortem["pre_event"]["ruleset_id"] = ""
        errors = _validate_postmortem(sample_postmortem)
        assert any("ruleset_id" in e for e in errors)

    def test_invalid_tier(self, sample_postmortem):
        sample_postmortem["pre_event"]["tier_dev"] = "X"
        errors = _validate_postmortem(sample_postmortem)
        assert any("tier_dev" in e for e in errors)

    def test_invalid_readiness_verdict(self, sample_postmortem):
        sample_postmortem["pre_event"]["readiness_verdict"] = "MAYBE"
        errors = _validate_postmortem(sample_postmortem)
        assert any("readiness_verdict" in e for e in errors)

    def test_unreasonable_return(self, sample_postmortem):
        sample_postmortem["outcome"]["return_t1"] = 10.0  # 1000% return
        errors = _validate_postmortem(sample_postmortem)
        assert any("unreasonable" in e for e in errors)

    def test_wrong_type_actionable_rank(self, sample_postmortem):
        sample_postmortem["pre_event"]["actionable_rank"] = "7"  # string instead of int
        errors = _validate_postmortem(sample_postmortem)
        assert any("actionable_rank" in e for e in errors)


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------


class TestPostmortemSerialization:
    def test_json_round_trip(self, sample_postmortem):
        serialized = json.dumps(sample_postmortem, indent=2)
        deserialized = json.loads(serialized)
        assert deserialized == sample_postmortem

    def test_json_valid_utf8(self, sample_postmortem):
        serialized = json.dumps(sample_postmortem, ensure_ascii=False)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Agent workspace integrity
# ---------------------------------------------------------------------------


class TestPostmortemAgentWorkspace:
    def test_soul_md_exists(self):
        assert (POSTMORTEM_DIR / "SOUL.md").exists()

    def test_tools_md_exists(self):
        assert (POSTMORTEM_DIR / "TOOLS.md").exists()

    def test_heartbeat_md_exists(self):
        assert (POSTMORTEM_DIR / "HEARTBEAT.md").exists()

    def test_agents_md_exists(self):
        assert (POSTMORTEM_DIR / "AGENTS.md").exists()

    def test_soul_not_stub(self):
        """SOUL.md should be substantive (>10 lines), not a stub."""
        content = (POSTMORTEM_DIR / "SOUL.md").read_text()
        lines = [line for line in content.strip().splitlines() if line.strip()]
        assert len(lines) >= 10, f"SOUL.md only has {len(lines)} non-blank lines"

    def test_tools_has_schema(self):
        """TOOLS.md should document the postmortem JSON schema."""
        content = (POSTMORTEM_DIR / "TOOLS.md").read_text()
        assert "postmortem.v1" in content

    def test_write_scope(self):
        """SOUL.md must restrict writes to postmortem paths only."""
        content = (POSTMORTEM_DIR / "SOUL.md").read_text()
        assert "artifacts/postmortem/" in content

    def test_never_edit_scoring(self):
        """SOUL.md must explicitly forbid editing scoring logic."""
        content = (POSTMORTEM_DIR / "SOUL.md").read_text().lower()
        assert "never" in content and "scoring" in content


# ---------------------------------------------------------------------------
# Consumer contract: event_analyst expects these fields
# ---------------------------------------------------------------------------


class TestEventAnalystContract:
    def test_pre_event_has_rank(self, sample_postmortem):
        assert isinstance(sample_postmortem["pre_event"]["actionable_rank"], int)

    def test_pre_event_has_tier(self, sample_postmortem):
        assert sample_postmortem["pre_event"]["tier_dev"] in {"A", "B", "C", "D", "INELIGIBLE", "COMMERCIAL"}

    def test_pre_event_has_catalyst_family(self, sample_postmortem):
        assert isinstance(sample_postmortem["pre_event"]["catalyst_family"], str)
        assert len(sample_postmortem["pre_event"]["catalyst_family"]) > 0

    def test_has_event_date(self, sample_postmortem):
        assert len(sample_postmortem["event_date"]) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Consumer contract: calibration_evidence expects outcome returns
# ---------------------------------------------------------------------------


class TestCalibrationEvidenceContract:
    def test_outcome_has_t1_return(self, sample_postmortem):
        assert "return_t1" in sample_postmortem["outcome"]

    def test_outcome_has_t3_return(self, sample_postmortem):
        assert "return_t3" in sample_postmortem["outcome"]

    def test_outcome_has_excess_return(self, sample_postmortem):
        assert "excess_vs_xbi_t1" in sample_postmortem["outcome"]

    def test_outcome_has_abs_gap(self, sample_postmortem):
        assert "abs_gap" in sample_postmortem["outcome"]


# ---------------------------------------------------------------------------
# Consumer contract: signal_evidence expects ruleset provenance
# ---------------------------------------------------------------------------


class TestSignalEvidenceContract:
    def test_has_ruleset_id(self, sample_postmortem):
        assert len(sample_postmortem["pre_event"]["ruleset_id"]) == 8  # 8-char hex

    def test_has_schema_version(self, sample_postmortem):
        assert sample_postmortem["schema"].startswith("postmortem.")


# ---------------------------------------------------------------------------
# Existing artifact validation (if any postmortems exist)
# ---------------------------------------------------------------------------


class TestExistingArtifacts:
    def _iter_postmortem_jsons(self):
        if not ARTIFACTS_DIR.exists():
            return
        for date_dir in sorted(ARTIFACTS_DIR.iterdir()):
            if not date_dir.is_dir():
                continue
            for f in date_dir.glob("*.json"):
                yield f

    def test_all_existing_artifacts_valid(self):
        """If any postmortem artifacts exist, they should all pass validation."""
        count = 0
        for path in self._iter_postmortem_jsons():
            record = json.loads(path.read_text())
            errors = _validate_postmortem(record)
            assert errors == [], f"{path}: {errors}"
            count += 1
        # This test is a no-op if no artifacts exist yet — that's OK.
        # It becomes load-bearing once the first postmortem is written.
