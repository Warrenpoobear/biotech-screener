"""Tests for DecisionRuleset dataclass — parameterized threshold config."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import (
    DecisionRuleset,
    DEFAULT_RULESET,
    RULESET_ID,
    SIZING_WEIGHTS,
    VERSION,
    compute_decision_fields,
    compute_target_weights,
)


# =============================================================================
# Helper to build minimal recs
# =============================================================================

def _base_rec(**overrides):
    """Build a minimal rec dict with sane defaults."""
    rec = {
        "ticker": "TEST",
        "composite_score": 50.0,
        "severity": "NONE",
        "fundamental_red_flag": False,
        "confidence_overall": 0.65,
        "defensive_features": {
            "vol_60d": 0.80,
            "beta_xbi_60d": 1.20,
            "drawdown": -0.15,
            "rsi_14d": 50.0,
        },
        "smart_money_signal": {
            "tier1_holders": ["FundA", "FundB"],
            "holders_increasing": ["FundA"],
            "holders_decreasing": [],
            "overlap_count": 2,
        },
        "coinvest": {"tier1_count": 3},
        "catalyst_decay": {
            "days_to_catalyst": 45,
            "in_optimal_window": True,
        },
        "score_breakdown": {
            "enhancements": {
                "momentum": {"alpha_60d": 0.08},
            }
        },
    }
    # Apply deep overrides
    for key, val in overrides.items():
        if isinstance(val, dict) and key in rec and isinstance(rec[key], dict):
            rec[key].update(val)
        else:
            rec[key] = val
    return rec


# =============================================================================
# Tests: Defaults and backward compatibility
# =============================================================================

class TestDefaults:
    def test_default_values_match_v1(self):
        """All DEFAULT_RULESET values match the original v1.0.0 hardcodes."""
        rs = DEFAULT_RULESET
        assert rs.drawdown_gate == -0.40
        assert rs.vol_high_threshold == 1.20
        assert rs.beta_high_threshold == 1.80
        assert rs.drawdown_flag_threshold == -0.35
        assert rs.rsi_overbought == 70.0
        assert rs.confidence_low_threshold == 0.30
        assert rs.alpha_tailwind == 0.05
        assert rs.alpha_headwind == -0.05
        assert rs.tier_a_optionality_floor == 0.60
        assert rs.tier_b_optionality_floor == 0.30
        assert rs.catalyst_near_days == 90
        assert rs.sponsor_confirm_threshold == 2
        assert rs.drawdown_gate_mode == "hard"
        assert rs.drawdown_size_penalty == -1
        assert rs.drawdown_hard_floor == -0.75

    def test_sizing_weights_match_v1(self):
        """Default sizing_weights produce the v1 dict."""
        assert DEFAULT_RULESET.sizing_weights_dict == {
            "L": 1.0, "M": 0.6, "S": 0.3, "XS": 0.15,
        }

    def test_backward_compat_sizing_weights_alias(self):
        """Module-level SIZING_WEIGHTS matches DEFAULT_RULESET."""
        assert SIZING_WEIGHTS == DEFAULT_RULESET.sizing_weights_dict

    def test_backward_compat_ruleset_id_alias(self):
        """Module-level RULESET_ID matches DEFAULT_RULESET.ruleset_id."""
        assert RULESET_ID == DEFAULT_RULESET.ruleset_id

    def test_version_bumped(self):
        """VERSION is v1.3.0 after catalyst strength bands."""
        assert VERSION == "v1.3.0"


# =============================================================================
# Tests: Ruleset ID properties
# =============================================================================

class TestRulesetId:
    def test_id_is_8_hex_chars(self):
        assert len(DEFAULT_RULESET.ruleset_id) == 8
        int(DEFAULT_RULESET.ruleset_id, 16)  # validates hex

    def test_id_deterministic(self):
        """Same params → same ID across instances."""
        rs1 = DecisionRuleset()
        rs2 = DecisionRuleset()
        assert rs1.ruleset_id == rs2.ruleset_id

    def test_id_changes_with_param(self):
        """Changing any param produces a different ID."""
        base_id = DEFAULT_RULESET.ruleset_id
        altered = DecisionRuleset(drawdown_gate=-0.50)
        assert altered.ruleset_id != base_id

    def test_id_changes_with_catalyst_near_days(self):
        altered = DecisionRuleset(catalyst_near_days=120)
        assert altered.ruleset_id != DEFAULT_RULESET.ruleset_id

    def test_id_changes_with_sponsor_confirm(self):
        altered = DecisionRuleset(sponsor_confirm_threshold=3)
        assert altered.ruleset_id != DEFAULT_RULESET.ruleset_id

    def test_id_changes_with_sizing_weights(self):
        altered = DecisionRuleset(
            sizing_weights=(("L", 1.0), ("M", 0.5), ("S", 0.3), ("XS", 0.15))
        )
        assert altered.ruleset_id != DEFAULT_RULESET.ruleset_id


# =============================================================================
# Tests: JSON round-trip
# =============================================================================

class TestJsonRoundTrip:
    def test_to_json_creates_file(self, tmp_path):
        path = str(tmp_path / "test_ruleset.json")
        DEFAULT_RULESET.to_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            d = json.load(f)
        assert "drawdown_gate" in d
        assert "sizing_weights" in d
        assert isinstance(d["sizing_weights"], dict)

    def test_round_trip_preserves_equality(self, tmp_path):
        path = str(tmp_path / "rt.json")
        DEFAULT_RULESET.to_json(path)
        loaded = DecisionRuleset.from_json(path)
        assert loaded == DEFAULT_RULESET
        assert loaded.ruleset_id == DEFAULT_RULESET.ruleset_id

    def test_round_trip_custom_ruleset(self, tmp_path):
        custom = DecisionRuleset(
            drawdown_gate=-0.50,
            catalyst_near_days=120,
            sponsor_confirm_threshold=3,
            sizing_weights=(("L", 1.0), ("M", 0.5), ("S", 0.25), ("XS", 0.10)),
        )
        path = str(tmp_path / "custom.json")
        custom.to_json(path)
        loaded = DecisionRuleset.from_json(path)
        assert loaded == custom
        assert loaded.ruleset_id == custom.ruleset_id

    def test_from_json_production_file(self):
        """Can load the committed v1.json production file."""
        prod_path = Path(__file__).resolve().parent.parent / "production_data" / "decision_rulesets" / "v1.json"
        if not prod_path.exists():
            pytest.skip("v1.json not yet created")
        loaded = DecisionRuleset.from_json(str(prod_path))
        assert loaded == DEFAULT_RULESET


# =============================================================================
# Tests: Custom ruleset behavior
# =============================================================================

class TestCustomRulesetBehavior:
    def test_looser_drawdown_gate_keeps_borderline_eligible(self):
        """A ticker at -0.42 drawdown is ineligible with defaults but eligible with -0.50."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.42, "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        # Default: -0.40 gate → -0.42 is worse → ineligible
        default_result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert default_result["eligible"] == "0"
        assert "deep_drawdown" in default_result["ineligible_reasons"]

        # Looser: -0.50 gate → -0.42 is above → eligible
        loose = DecisionRuleset(drawdown_gate=-0.50)
        loose_result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=loose)
        assert loose_result["eligible"] == "1"

    def test_wider_catalyst_window_promotes_ticker(self):
        """A ticker at 200 days is 'far' with default mid=180 but 'mid' with mid=250."""
        rec = _base_rec(
            catalyst_decay={"days_to_catalyst": 200, "in_optimal_window": False},
        )
        # Default: mid_days=180 → 200 days is far → B (not A)
        default_result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert default_result["tier_dev"] == "B"
        assert "catalyst_far" in default_result["tier_reason"]

        # Wider: mid_days=250 → 200 days is mid → actionable → A
        wide = DecisionRuleset(catalyst_mid_days=250)
        wide_result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=wide)
        assert wide_result["tier_dev"] == "A"
        assert "catalyst_mid" in wide_result["tier_reason"]

    def test_higher_sponsor_threshold_changes_size_band(self):
        """With sponsor_confirm=3, tier1=2 no longer confirms → different band."""
        rec = _base_rec(
            coinvest={"tier1_count": 2},
            catalyst_decay={"days_to_catalyst": 45, "in_optimal_window": True},
        )
        # Default: threshold=2 → 2 >= 2 → confirmed → band gets boosted
        default_result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert "sponsor_confirmed" in default_result["size_reasons"]

        # Higher threshold: 3 → 2 < 3 → not confirmed
        strict = DecisionRuleset(sponsor_confirm_threshold=3)
        strict_result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=strict)
        assert "sponsor_confirmed" not in strict_result["size_reasons"]

    def test_custom_sizing_weights(self):
        """Custom sizing weights flow through to compute_target_weights."""
        custom = DecisionRuleset(
            sizing_weights=(("L", 2.0), ("M", 1.0), ("S", 0.5), ("XS", 0.1))
        )
        rows = [
            {"size_band": "L", "eligible": "1"},
            {"size_band": "M", "eligible": "1"},
        ]
        compute_target_weights(rows, ruleset=custom)
        # L=2.0, M=1.0 → total=3.0 → L=66.67%, M=33.33%
        assert abs(rows[0]["target_weight_pct"] - 66.67) < 0.01
        assert abs(rows[1]["target_weight_pct"] - 33.33) < 0.01

    def test_ruleset_id_in_output(self):
        """Output includes the active ruleset's ID, not the module-level constant."""
        custom = DecisionRuleset(drawdown_gate=-0.50)
        rec = _base_rec()
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=custom)
        assert result["decision_engine_ruleset_id"] == custom.ruleset_id
        assert result["decision_engine_ruleset_id"] != DEFAULT_RULESET.ruleset_id


# =============================================================================
# Tests: Frozen immutability
# =============================================================================

class TestFrozenImmutability:
    def test_cannot_mutate(self):
        with pytest.raises(AttributeError):
            DEFAULT_RULESET.drawdown_gate = -0.50  # type: ignore[misc]


# =============================================================================
# CI guardrails: prevent silent ruleset drift
# =============================================================================

class TestRulesetDriftGuardrails:
    """These tests fail loudly when defaults change without an intentional bump.

    If a default threshold changes, update both the expected_id here AND
    regenerate production_data/decision_rulesets/v1.json.
    """

    EXPECTED_DEFAULT_RULESET_ID = "fbf9d299"

    def test_default_ruleset_id_pinned(self):
        """DEFAULT_RULESET.ruleset_id must match the committed expected value.

        If this fails, a default threshold was changed. Update EXPECTED_DEFAULT_RULESET_ID
        and regenerate the production JSON: DEFAULT_RULESET.to_json('production_data/decision_rulesets/v1.json')
        """
        assert DEFAULT_RULESET.ruleset_id == self.EXPECTED_DEFAULT_RULESET_ID, (
            f"DEFAULT_RULESET.ruleset_id changed from {self.EXPECTED_DEFAULT_RULESET_ID} "
            f"to {DEFAULT_RULESET.ruleset_id}. If intentional, update "
            f"EXPECTED_DEFAULT_RULESET_ID and regenerate v1.json."
        )

    def test_production_json_matches_defaults(self):
        """production_data/decision_rulesets/v1.json must equal DEFAULT_RULESET."""
        prod_path = (
            Path(__file__).resolve().parent.parent
            / "production_data" / "decision_rulesets" / "v1.json"
        )
        assert prod_path.exists(), f"Production ruleset JSON not found: {prod_path}"
        loaded = DecisionRuleset.from_json(str(prod_path))
        assert loaded == DEFAULT_RULESET, (
            f"v1.json ruleset_id={loaded.ruleset_id} != DEFAULT_RULESET "
            f"ruleset_id={DEFAULT_RULESET.ruleset_id}. Regenerate v1.json."
        )

    def test_module_level_aliases_consistent(self):
        """RULESET_ID and SIZING_WEIGHTS aliases must match DEFAULT_RULESET."""
        assert RULESET_ID == DEFAULT_RULESET.ruleset_id
        assert SIZING_WEIGHTS == DEFAULT_RULESET.sizing_weights_dict

    # -- Manifest governance --------------------------------------------------

    MANIFEST_PATH = (
        Path(__file__).resolve().parent.parent
        / "production_data" / "decision_rulesets" / "manifest.json"
    )

    REQUIRED_ENTRY_KEYS = {
        "id", "file", "created_at", "engine_version",
        "description", "status", "requires_replay", "notes",
    }

    VALID_STATUSES = {"active", "candidate", "retired", "rejected"}

    def test_manifest_json_valid(self):
        """manifest.json exists, has valid structure, and exactly 1 active ruleset."""
        assert self.MANIFEST_PATH.exists(), (
            f"manifest.json not found at {self.MANIFEST_PATH}"
        )
        with open(self.MANIFEST_PATH) as f:
            data = json.load(f)

        assert "schema_version" in data, "Missing schema_version"
        assert data["schema_version"] == 1
        assert "rulesets" in data, "Missing rulesets array"
        assert isinstance(data["rulesets"], list)
        assert len(data["rulesets"]) >= 1, "rulesets array is empty"

        active_count = 0
        for entry in data["rulesets"]:
            missing = self.REQUIRED_ENTRY_KEYS - set(entry.keys())
            assert not missing, (
                f"Entry for {entry.get('file', '?')} missing keys: {missing}"
            )
            assert entry["status"] in self.VALID_STATUSES, (
                f"Invalid status {entry['status']!r} for {entry['file']}"
            )
            if entry["status"] == "active":
                active_count += 1

        assert active_count == 1, (
            f"Expected exactly 1 active ruleset, found {active_count}"
        )

    VALID_UPDATED_BY = {
        "bump_ruleset.py",
        "promote_ruleset.py",
        "manual_bootstrap",
    }

    # Accepts both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SSZ (ISO 8601 UTC)
    _ISO_DATE_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$"
    )

    def test_manifest_active_has_provenance(self):
        """Every active/candidate entry must have valid updated_by and updated_at.

        Retired/rejected entries are grandfathered and do not require these fields.
        """
        assert self.MANIFEST_PATH.exists(), "manifest.json not found"
        with open(self.MANIFEST_PATH) as f:
            data = json.load(f)

        provenance_required = {"active", "candidate"}
        for entry in data["rulesets"]:
            if entry["status"] not in provenance_required:
                continue
            label = f"{entry['id']} ({entry['file']})"

            # Must have both fields
            assert "updated_by" in entry, (
                f"Entry {label} status={entry['status']} "
                f"missing 'updated_by' provenance field"
            )
            assert "updated_at" in entry, (
                f"Entry {label} status={entry['status']} "
                f"missing 'updated_at' provenance field"
            )

            # updated_by must be a known source
            assert entry["updated_by"] in self.VALID_UPDATED_BY, (
                f"Entry {label}: updated_by={entry['updated_by']!r} not in "
                f"allowed set {self.VALID_UPDATED_BY}"
            )

            # updated_at must be ISO date or ISO datetime UTC
            assert self._ISO_DATE_RE.match(entry["updated_at"]), (
                f"Entry {label}: updated_at={entry['updated_at']!r} is not "
                f"valid ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)"
            )

    def test_manifest_ids_match_files(self):
        """Each manifest entry's id matches the recomputed ruleset_id from its file."""
        assert self.MANIFEST_PATH.exists(), "manifest.json not found"
        with open(self.MANIFEST_PATH) as f:
            data = json.load(f)

        rulesets_dir = self.MANIFEST_PATH.parent
        for entry in data["rulesets"]:
            filepath = rulesets_dir / entry["file"]
            assert filepath.exists(), (
                f"Manifest references {entry['file']} but file not found"
            )
            loaded = DecisionRuleset.from_json(str(filepath))
            assert loaded.ruleset_id == entry["id"], (
                f"{entry['file']}: manifest id={entry['id']} != "
                f"computed id={loaded.ruleset_id}"
            )


# =============================================================================
# Tests: Schema-stability (ruleset_id must not change with schema expansion)
# =============================================================================

class TestRulesetIdSchemaStability:
    """Ensure that from_json hashes file content, not expanded dataclass.

    Adding new optional fields to DecisionRuleset must NOT change the
    ruleset_id of existing production JSON files.  These tests pin known
    file-content hashes and verify round-trip + stability properties.
    """

    RULESETS_DIR = (
        Path(__file__).resolve().parent.parent
        / "production_data" / "decision_rulesets"
    )

    # Pinned file-content hashes for key rulesets
    PINNED_FILE_HASHES = {
        "v1.2.2_candidate.json": "bf6815e2",
        "v1.3.0_candidate.json": "f3454ef7",
        "v1.json": "fbf9d299",
        "v1.3.1_candidate.json": "898e5d0d",
        "v1.3.2_candidate.json": "96f655ee",
        "v1.3.3_missing_sort_only_candidate.json": "e1be5370",
        "v1.4.1_tier_first_candidate.json": "054bc5cc",
    }

    def test_pinned_file_hashes_stable(self):
        """Production file-content hashes must match pinned values."""
        for filename, expected_id in self.PINNED_FILE_HASHES.items():
            path = self.RULESETS_DIR / filename
            assert path.exists(), f"{filename} not found"
            rs = DecisionRuleset.from_json(str(path))
            assert rs.ruleset_id == expected_id, (
                f"{filename}: expected {expected_id}, got {rs.ruleset_id}. "
                f"If the file content changed intentionally, update PINNED_FILE_HASHES."
            )

    def test_from_json_uses_file_content_hash(self):
        """from_json stores _file_content_hash on the frozen instance."""
        path = self.RULESETS_DIR / "v1.2.2_candidate.json"
        rs = DecisionRuleset.from_json(str(path))
        # Must have the hidden attribute
        assert hasattr(rs, "_file_content_hash")
        assert rs._file_content_hash == rs.ruleset_id

    def test_programmatic_construction_uses_canonical_hash(self):
        """DecisionRuleset() (no file) hashes via _canonical_json."""
        rs = DecisionRuleset()
        assert not hasattr(rs, "_file_content_hash") or not rs._file_content_hash
        # Should still have a valid ruleset_id
        assert len(rs.ruleset_id) == 8

    def test_round_trip_preserves_id(self):
        """to_json → from_json round-trip preserves identity.

        Uses v1.json (full-schema file) for exact ID match, and verifies
        logical equality for older files that gain new default fields on write.
        """
        import tempfile
        # Full-schema file: round-trip preserves exact ID
        rs = DecisionRuleset.from_json(
            str(self.RULESETS_DIR / "v1.json")
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name
        try:
            rs.to_json(tmp_path)
            rs2 = DecisionRuleset.from_json(tmp_path)
            assert rs.ruleset_id == rs2.ruleset_id
            assert rs == rs2
        finally:
            os.unlink(tmp_path)

        # Older file (may lack new fields): logical equality preserved
        rs_old = DecisionRuleset.from_json(
            str(self.RULESETS_DIR / "v1.2.2_candidate.json")
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name
        try:
            rs_old.to_json(tmp_path)
            rs_old2 = DecisionRuleset.from_json(tmp_path)
            assert rs_old == rs_old2  # same field values
        finally:
            os.unlink(tmp_path)

    def test_file_missing_future_field_keeps_hash(self):
        """A file that lacks a field still hashes to its file content.

        Simulates future schema expansion: writes a file WITHOUT a known
        field, then loads it. The hash must be based on what's in the file,
        not the expanded defaults.
        """
        import tempfile
        # Write a minimal valid ruleset (only a few fields)
        minimal = {"tier_a_optionality_floor": 0.55, "tier_b_optionality_floor": 0.30}
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            json.dump(minimal, tmp, sort_keys=True)
            tmp_path = tmp.name
        try:
            rs = DecisionRuleset.from_json(tmp_path)
            # Hash should be based on the 2-key dict, NOT the full dataclass
            expected_hash = hashlib.sha256(
                json.dumps(minimal, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]
            assert rs.ruleset_id == expected_hash, (
                f"from_json hash {rs.ruleset_id} != raw file hash {expected_hash}. "
                f"ruleset_id may be using expanded dataclass fields."
            )
        finally:
            os.unlink(tmp_path)


# =============================================================================
# Tests: Soft drawdown mode
# =============================================================================

class TestSoftDrawdownMode:
    """Tests for drawdown_gate_mode='soft' behavior."""

    SOFT_RULESET = DecisionRuleset(drawdown_gate_mode="soft")

    def test_soft_mode_drawdown_breach_stays_eligible(self):
        """Soft mode + drawdown breach above hard_floor → eligible."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "vol_60d": 0.5,
                                "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75,
                                         ruleset=self.SOFT_RULESET)
        assert result["eligible"] == "1"
        assert "deep_drawdown" not in result["ineligible_reasons"]

    def test_soft_mode_drawdown_breach_reduces_size(self):
        """Soft mode + drawdown breach → size_band one step lower, drawdown_penalty in reasons."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "vol_60d": 0.5,
                                "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        # Hard mode baseline: ineligible → XS
        hard_result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert hard_result["eligible"] == "0"

        # Soft mode: eligible, but drawdown_penalty applied
        soft_result = compute_decision_fields(rec, "drug_developer", 0.75,
                                              ruleset=self.SOFT_RULESET)
        assert soft_result["eligible"] == "1"
        assert "drawdown_penalty" in soft_result["size_reasons"]

    def test_soft_mode_catastrophic_drawdown_still_ineligible(self):
        """Soft mode + catastrophic drawdown (< hard_floor) → ineligible."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.80, "vol_60d": 0.5,
                                "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75,
                                         ruleset=self.SOFT_RULESET)
        assert result["eligible"] == "0"
        assert "deep_drawdown" in result["ineligible_reasons"]

    def test_hard_mode_drawdown_breach_still_ineligible(self):
        """Hard mode + drawdown breach → ineligible (unchanged v1 behavior)."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "vol_60d": 0.5,
                                "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["eligible"] == "0"
        assert "deep_drawdown" in result["ineligible_reasons"]

    def test_drawdown_gate_mode_changes_ruleset_id(self):
        """Flipping drawdown_gate_mode changes ruleset_id."""
        assert self.SOFT_RULESET.ruleset_id != DEFAULT_RULESET.ruleset_id


# =============================================================================
# Regime-Aware Drawdown Gate (XBI-Relative) — AND logic tests
# =============================================================================

class TestDrawdownAndGate:
    """Tests for the AND gate: FAIL only if BOTH absolute AND relative thresholds breach."""

    def test_both_breach_ineligible(self):
        """Both absolute (-0.50 < -0.40) and relative (-0.25 < -0.15) breach → INELIGIBLE."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.25,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["eligible"] == "0"
        assert "deep_drawdown" in result["ineligible_reasons"]

    def test_only_abs_breach_eligible(self):
        """Absolute breaches (-0.50 < -0.40) but relative does NOT (-0.10 > -0.15) → ELIGIBLE."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.10,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["eligible"] == "1"
        assert "deep_drawdown" not in result.get("ineligible_reasons", "")

    def test_only_rel_breach_eligible(self):
        """Absolute does NOT breach (-0.30 > -0.40) but relative does (-0.25 < -0.15) → ELIGIBLE."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.30, "drawdown_rel_xbi": -0.25,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["eligible"] == "1"

    def test_missing_rel_fallback_ineligible(self):
        """Absolute breaches and relative is None → fallback to absolute-only → INELIGIBLE."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "vol_60d": 0.5,
                                "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["eligible"] == "0"
        assert "deep_drawdown" in result["ineligible_reasons"]

    def test_soft_mode_with_relative(self):
        """Soft mode + both breaches → sizing penalty, not ineligibility (above hard floor)."""
        rs = DecisionRuleset(drawdown_gate_mode="soft")
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.25,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "1"
        assert "deep_drawdown" not in result.get("ineligible_reasons", "")

    def test_rel_risk_flag(self):
        """Relative drawdown breach → deep_drawdown_rel_xbi in risk_flags."""
        rec = _base_rec(
            defensive_features={"drawdown": -0.30, "drawdown_rel_xbi": -0.25,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert "deep_drawdown_rel_xbi" in result["risk_flags"]


# =============================================================================
# Drawdown near-rel-gate rescue (dd_rel_margin_rescue)
# =============================================================================

class TestDdRelMarginRescue:
    """Tests for the near-rel-gate rescue: override drawdown exclusion when
    the relative margin is close to the gate (sector-driven drawdown)."""

    def _rs(self, **kw):
        defaults = dict(
            drawdown_gate_require_both=True,
            enable_dd_rel_margin_rescue=True,
            dd_rel_margin_rescue_threshold=-0.05,
        )
        defaults.update(kw)
        return DecisionRuleset(**defaults)

    def test_rescue_fires_when_rel_margin_above_threshold(self):
        """Both breach, but rel margin = -0.03 > threshold -0.05 → RESCUED, ELIGIBLE."""
        # abs: -0.50 < -0.40 → breach.  rel: -0.23 < -0.20 → breach, margin = -0.03
        rs = self._rs()
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.23,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "1"
        assert result["dd_rel_margin_rescued"] == "1"
        assert "deep_drawdown" not in result.get("ineligible_reasons", "")

    def test_rescue_does_not_fire_when_rel_margin_below_threshold(self):
        """Both breach, rel margin = -0.08 < threshold -0.05 → NOT rescued, INELIGIBLE."""
        # abs: -0.50 < -0.40 → breach.  rel: -0.28 < -0.20 → breach, margin = -0.08
        rs = self._rs()
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.28,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "0"
        assert result["dd_rel_margin_rescued"] == "0"
        assert "deep_drawdown" in result["ineligible_reasons"]

    def test_rescue_disabled_by_default(self):
        """Default ruleset has rescue disabled — both breach → INELIGIBLE."""
        rs = DecisionRuleset()  # enable_dd_rel_margin_rescue=False
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.23,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "0"
        assert result["dd_rel_margin_rescued"] == "0"

    def test_rescue_at_threshold_boundary_not_rescued(self):
        """Rel margin just below threshold → NOT rescued."""
        # rel: -0.2501, gate: -0.20, margin = -0.0501 < threshold -0.05 → NOT rescued
        rs = self._rs()
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.2501,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "0"
        assert result["dd_rel_margin_rescued"] == "0"

    def test_rescue_only_applies_to_require_both_mode(self):
        """Rescue disabled when require_both=False (OR mode)."""
        rs = self._rs(drawdown_gate_require_both=False)
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.23,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        # In OR mode, abs breach alone → ineligible (rescue doesn't apply)
        assert result["eligible"] == "0"
        assert result["dd_rel_margin_rescued"] == "0"

    def test_rescue_does_not_apply_when_only_abs_breaches(self):
        """Abs breaches but rel does NOT → already eligible, no rescue needed."""
        rs = self._rs()
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.10,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "1"
        assert result["dd_rel_margin_rescued"] == "0"  # Not rescued — already eligible

    def test_rescue_does_not_override_other_gates(self):
        """Even if rescue fires for drawdown, sev3 still blocks eligibility."""
        rs = self._rs()
        rec = _base_rec(
            severity="SEV3",
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.23,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "0"
        assert "sev3" in result["ineligible_reasons"]

    def test_rescued_ticker_gets_proper_tier(self):
        """Rescued ticker with high optionality + near catalyst → tier A."""
        rs = self._rs(
            tier_a_optionality_floor=0.60,
            catalyst_near_days=90,
        )
        rec = _base_rec(
            defensive_features={"drawdown": -0.50, "drawdown_rel_xbi": -0.23,
                                "vol_60d": 0.5, "beta_xbi_60d": 1.0, "rsi_14d": 40},
            catalyst_decay={"days_to_catalyst": 30, "in_optimal_window": True},
        )
        result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert result["eligible"] == "1"
        assert result["dd_rel_margin_rescued"] == "1"
        assert result["tier_dev"] == "A"

    def test_validation_rejects_positive_threshold(self):
        """dd_rel_margin_rescue_threshold must be <= 0."""
        with pytest.raises(ValueError, match="dd_rel_margin_rescue_threshold"):
            DecisionRuleset(dd_rel_margin_rescue_threshold=0.05)


# =============================================================================
# Intent guardrail: baseline-based (compares working tree vs origin/main)
# =============================================================================

_ROOT = Path(__file__).resolve().parents[1]


def _git_show(ref: str, relpath: str) -> str | None:
    """Retrieve file contents from a git ref. Returns None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{relpath}"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def _extract_expected_fingerprint(py_text: str) -> str | None:
    m = re.search(r'EXPECTED_FINGERPRINT\s*=\s*"([0-9a-f]{8,64})"', py_text)
    return m.group(1) if m else None


def _active_id_from_manifest_text(manifest_text: str) -> str | None:
    obj = json.loads(manifest_text)
    actives = [e["id"] for e in obj.get("rulesets", []) if e.get("status") == "active"]
    return actives[0] if len(actives) == 1 else None


def _candidate_ids_from_manifest_text(manifest_text: str) -> list[str]:
    obj = json.loads(manifest_text)
    return [e["id"] for e in obj.get("rulesets", []) if e.get("status") == "candidate"]


_CHANGELOG_HEADING_RE = re.compile(
    r"^##\s+\[v\d+\.\d+\.\d+\]\s+[0-9a-f]{8}\b",
)


def _changelog_has_final_entry(changelog_text: str, ruleset_id: str) -> bool:
    """Check that a non-DRAFT versioned heading references this ruleset ID.

    Matches our changelog format: ``## [vX.Y.Z] {id} — date — desc``
    Rejects DRAFT format: ``## [DRAFT] [vX.Y.Z] {id} — date — desc``
    Requires semantic version marker to prevent incidental ID mentions.
    """
    for line in changelog_text.splitlines():
        if not line.startswith("## "):
            continue
        if "[DRAFT]" in line:
            continue
        if ruleset_id not in line:
            continue
        if not _CHANGELOG_HEADING_RE.match(line):
            continue
        return True
    return False


class TestRulesetIntentGuardrail:
    """Baseline-based intent guard: detects fingerprint change vs origin/main.

    If the golden fingerprint changed on this branch, enforces:
      1. A candidate exists in the manifest OR active ruleset id changed.
      2. The "intent target" id (new active if changed, else first candidate)
         has a finalized (non-DRAFT) changelog entry.

    Skips gracefully when origin/main is not available (shallow clone, local
    dev without remote).
    """

    def test_fingerprint_change_requires_ruleset_intent_and_changelog(self):
        # Read baselines: try origin/main first, fall back to HEAD~1,
        # skip if neither available (e.g. initial commit or detached HEAD).
        prev_contract = None
        prev_manifest = None
        for ref in ("origin/main", "HEAD~1"):
            prev_contract = _git_show(
                ref, "tests/test_decision_engine_contract.py",
            )
            prev_manifest = _git_show(
                ref, "production_data/decision_rulesets/manifest.json",
            )
            if prev_contract and prev_manifest:
                break
        if not prev_contract or not prev_manifest:
            pytest.skip("No baseline available (origin/main or HEAD~1)")

        prev_fp = _extract_expected_fingerprint(prev_contract)
        assert prev_fp, "Could not extract EXPECTED_FINGERPRINT from baseline"

        # Read current branch fingerprint from the working-tree file
        cur_contract_path = _ROOT / "tests" / "test_decision_engine_contract.py"
        cur_fp = _extract_expected_fingerprint(
            cur_contract_path.read_text(encoding="utf-8"),
        )
        assert cur_fp, "Could not extract EXPECTED_FINGERPRINT from working tree"

        if cur_fp == prev_fp:
            return  # no drift — nothing to enforce

        # ── Fingerprint changed — enforce governance ────────────────────
        cur_manifest_path = _ROOT / "production_data" / "decision_rulesets" / "manifest.json"
        cur_manifest_text = cur_manifest_path.read_text(encoding="utf-8")

        cur_active = _active_id_from_manifest_text(cur_manifest_text)
        assert cur_active, "manifest must have exactly 1 active ruleset"

        prev_active = _active_id_from_manifest_text(prev_manifest)
        assert prev_active, "origin/main manifest must have exactly 1 active ruleset"

        cur_candidates = _candidate_ids_from_manifest_text(cur_manifest_text)
        active_changed = cur_active != prev_active
        candidate_exists = len(cur_candidates) >= 1

        assert candidate_exists or active_changed, (
            f"Golden fingerprint changed ({prev_fp} -> {cur_fp}), but no "
            f"candidate ruleset exists and active ruleset did not change. "
            f"Create a candidate via bump_ruleset.py or promote a candidate."
        )

        # Determine which ruleset id must have a finalized changelog entry
        target_id = cur_active if active_changed else cur_candidates[0]

        changelog_path = _ROOT / "RULESET_CHANGELOG.md"
        assert changelog_path.exists(), (
            f"Fingerprint changed ({prev_fp} -> {cur_fp}) but "
            f"RULESET_CHANGELOG.md not found"
        )
        changelog = changelog_path.read_text(encoding="utf-8")

        assert _changelog_has_final_entry(changelog, target_id), (
            f"Fingerprint changed ({prev_fp} -> {cur_fp}) but changelog "
            f"has no finalized (non-DRAFT) entry for ruleset {target_id}. "
            f"Finalize the DRAFT entry before landing."
        )


# =============================================================================
# Tests: Cost param validation
# =============================================================================

class TestCostParamValidation:
    """Tests for cost-aware sizing parameter validation and serialization."""

    def test_round_trip_custom_cost_buckets(self, tmp_path):
        """Custom buckets + floor_mult round-trip preserves equality and ruleset_id."""
        custom = DecisionRuleset(
            cost_haircut_buckets=((50, 1.0), (100, 0.85), (150, 0.70)),
            cost_haircut_floor_mult=0.50,
        )
        path = str(tmp_path / "cost_custom.json")
        custom.to_json(path)
        loaded = DecisionRuleset.from_json(path)
        assert loaded == custom
        assert loaded.ruleset_id == custom.ruleset_id
        assert loaded.cost_haircut_buckets == ((50, 1.0), (100, 0.85), (150, 0.70))
        assert loaded.cost_haircut_floor_mult == 0.50

    def test_invalid_bucket_ordering_rejected(self):
        """Descending bucket thresholds raise ValueError."""
        with pytest.raises(ValueError, match="strictly ascending"):
            DecisionRuleset(
                cost_haircut_buckets=((100, 1.0), (50, 0.85), (200, 0.70)),
            )

    def test_invalid_floor_mult_rejected(self):
        """floor_mult=0 and floor_mult>1 both raise ValueError."""
        with pytest.raises(ValueError, match="cost_haircut_floor_mult"):
            DecisionRuleset(cost_haircut_floor_mult=0.0)
        with pytest.raises(ValueError, match="cost_haircut_floor_mult"):
            DecisionRuleset(cost_haircut_floor_mult=1.5)

    def test_default_behavior_when_cost_fields_absent(self, tmp_path):
        """JSON missing cost fields → defaults applied, matches expected."""
        # Write a minimal JSON without cost fields
        minimal = {
            "drawdown_gate": -0.40,
            "vol_high_threshold": 1.20,
        }
        path = str(tmp_path / "no_cost.json")
        with open(path, "w") as f:
            json.dump(minimal, f)
        loaded = DecisionRuleset.from_json(path)
        assert loaded.cost_haircut_buckets == DEFAULT_RULESET.cost_haircut_buckets
        assert loaded.cost_haircut_floor_mult == DEFAULT_RULESET.cost_haircut_floor_mult
        assert loaded.cost_impact_cap_bps == DEFAULT_RULESET.cost_impact_cap_bps
        assert loaded.enable_cost_haircut == DEFAULT_RULESET.enable_cost_haircut

    def test_from_json_with_explicit_cost_buckets(self, tmp_path):
        """JSON with cost_haircut_buckets as list-of-lists → loaded as tuple-of-tuples."""
        data = {
            "cost_haircut_buckets": [[50, 1.0], [100, 0.85], [150, 0.70]],
            "cost_haircut_floor_mult": 0.60,
        }
        path = str(tmp_path / "explicit_cost.json")
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = DecisionRuleset.from_json(path)
        assert isinstance(loaded.cost_haircut_buckets, tuple)
        assert all(isinstance(pair, tuple) for pair in loaded.cost_haircut_buckets)
        assert loaded.cost_haircut_buckets == ((50, 1.0), (100, 0.85), (150, 0.70))
        assert loaded.cost_haircut_floor_mult == 0.60


# =============================================================================
# Tests: Catalyst tilt
# =============================================================================

class TestCatalystTilt:
    """Tests for catalyst strength sizing tilt (L3-only, opt-in)."""

    def test_disabled_weights_identical(self):
        """With enable_catalyst_tilt=False, weights are bit-for-bit identical to baseline."""
        rows_baseline = [
            {"size_band": "L", "eligible": "1", "catalyst_strength": "near"},
            {"size_band": "M", "eligible": "1", "catalyst_strength": "missing"},
        ]
        rows_tilt_off = [
            {"size_band": "L", "eligible": "1", "catalyst_strength": "near"},
            {"size_band": "M", "eligible": "1", "catalyst_strength": "missing"},
        ]
        compute_target_weights(rows_baseline)
        rs_off = DecisionRuleset(enable_catalyst_tilt=False)
        compute_target_weights(rows_tilt_off, ruleset=rs_off)
        assert rows_baseline[0]["target_weight_pct"] == rows_tilt_off[0]["target_weight_pct"]
        assert rows_baseline[1]["target_weight_pct"] == rows_tilt_off[1]["target_weight_pct"]

    def test_enabled_near_gets_higher_weight_than_far(self):
        """With tilt enabled, NEAR ticker gets higher weight than FAR for same band."""
        rs = DecisionRuleset(enable_catalyst_tilt=True)
        # Build two otherwise-identical rows differing only in catalyst tilt
        rec_near = _base_rec(
            catalyst_decay={"days_to_catalyst": 45, "in_optimal_window": True},
        )
        rec_far = _base_rec(
            catalyst_decay={"days_to_catalyst": 200, "in_optimal_window": False},
        )
        result_near = compute_decision_fields(rec_near, "drug_developer", 0.75, ruleset=rs)
        result_far = compute_decision_fields(rec_far, "drug_developer", 0.75, ruleset=rs)
        assert result_near["catalyst_tilt_mult"] > result_far["catalyst_tilt_mult"]
        assert result_near["catalyst_tilt_applied"] == "1"
        assert result_far["catalyst_tilt_applied"] == "1"

    def test_disabled_emits_neutral_tilt(self):
        """With tilt disabled, catalyst_tilt_mult=1.0, catalyst_tilt_applied='0'."""
        rec = _base_rec()
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["catalyst_tilt_mult"] == 1.0
        assert result["catalyst_tilt_applied"] == "0"

    def test_invalid_tilt_band_rejected(self):
        """Unknown band in catalyst_tilt_mults raises ValueError."""
        with pytest.raises(ValueError, match="unknown band"):
            DecisionRuleset(
                catalyst_tilt_mults=(("NEAR", 1.10), ("BOGUS", 0.90)),
            )

    def test_non_positive_tilt_mult_rejected(self):
        """Multiplier <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            DecisionRuleset(
                catalyst_tilt_mults=(("NEAR", 0.0), ("MID", 1.05)),
            )

    def test_round_trip_custom_tilt_mults(self, tmp_path):
        """Custom tilt mults round-trip preserves equality and ruleset_id."""
        custom = DecisionRuleset(
            enable_catalyst_tilt=True,
            catalyst_tilt_mults=(("NEAR", 1.20), ("MID", 1.10), ("FAR", 0.90), ("MISSING", 0.80)),
        )
        path = str(tmp_path / "tilt_custom.json")
        custom.to_json(path)
        loaded = DecisionRuleset.from_json(path)
        assert loaded == custom
        assert loaded.ruleset_id == custom.ruleset_id
        assert loaded.enable_catalyst_tilt is True
        assert loaded.catalyst_tilt_mults == (
            ("NEAR", 1.20), ("MID", 1.10), ("FAR", 0.90), ("MISSING", 0.80)
        )

    def test_tilt_weights_in_portfolio(self):
        """With tilt enabled, NEAR weight > FAR weight in portfolio normalization."""
        rs = DecisionRuleset(enable_catalyst_tilt=True)
        rows = [
            {"size_band": "L", "eligible": "1", "cost_mult": 1.0,
             "catalyst_tilt_mult": 1.10},  # NEAR
            {"size_band": "L", "eligible": "1", "cost_mult": 1.0,
             "catalyst_tilt_mult": 0.95},  # FAR
        ]
        compute_target_weights(rows, ruleset=rs)
        assert rows[0]["target_weight_pct"] > rows[1]["target_weight_pct"]


# =============================================================================
# Tests: Momentum state tilt
# =============================================================================

class TestMomStateTilt:
    """Tests for momentum state sizing tilt (L3-only, opt-in)."""

    def test_disabled_weights_identical(self):
        """With enable_mom_state_tilt=False, weights are bit-for-bit identical to baseline."""
        rows_baseline = [
            {"size_band": "L", "eligible": "1", "mom_state": "tailwind"},
            {"size_band": "M", "eligible": "1", "mom_state": "neutral"},
        ]
        rows_tilt_off = [
            {"size_band": "L", "eligible": "1", "mom_state": "tailwind"},
            {"size_band": "M", "eligible": "1", "mom_state": "neutral"},
        ]
        compute_target_weights(rows_baseline)
        rs_off = DecisionRuleset(enable_mom_state_tilt=False)
        compute_target_weights(rows_tilt_off, ruleset=rs_off)
        assert rows_baseline[0]["target_weight_pct"] == rows_tilt_off[0]["target_weight_pct"]
        assert rows_baseline[1]["target_weight_pct"] == rows_tilt_off[1]["target_weight_pct"]

    def test_enabled_produces_differential_weights(self):
        """With tilt enabled and neutral penalized, tailwind gets higher weight than neutral."""
        rs = DecisionRuleset(
            enable_mom_state_tilt=True,
            mom_state_tilt_mults=(
                ("tailwind", 1.0), ("neutral", 0.85), ("headwind", 1.0),
            ),
        )
        # Two otherwise-identical recs differing only in momentum
        rec_tailwind = _base_rec(
            catalyst_decay={"days_to_catalyst": 45, "in_optimal_window": True},
            score_breakdown={"enhancements": {"momentum": {"alpha_60d": 0.08}}},
        )
        rec_neutral = _base_rec(
            catalyst_decay={"days_to_catalyst": 45, "in_optimal_window": True},
            score_breakdown={"enhancements": {"momentum": {"alpha_60d": 0.02}}},
        )
        result_tw = compute_decision_fields(rec_tailwind, "drug_developer", 0.75, ruleset=rs)
        result_nu = compute_decision_fields(rec_neutral, "drug_developer", 0.75, ruleset=rs)
        assert result_tw["mom_state_tilt_mult"] > result_nu["mom_state_tilt_mult"]
        assert result_tw["mom_state_tilt_applied"] == "0"  # 1.0 is neutral
        assert result_nu["mom_state_tilt_applied"] == "1"  # 0.85 != 1.0

    def test_disabled_emits_neutral_tilt(self):
        """With tilt disabled, mom_state_tilt_mult=1.0, mom_state_tilt_applied='0'."""
        rec = _base_rec()
        result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert result["mom_state_tilt_mult"] == 1.0
        assert result["mom_state_tilt_applied"] == "0"

    def test_invalid_state_name_rejected(self):
        """Unknown state in mom_state_tilt_mults raises ValueError."""
        with pytest.raises(ValueError, match="unknown state"):
            DecisionRuleset(
                mom_state_tilt_mults=(("tailwind", 1.0), ("bogus", 0.90)),
            )

    def test_non_positive_tilt_mult_rejected(self):
        """Multiplier <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            DecisionRuleset(
                mom_state_tilt_mults=(("tailwind", 0.0), ("neutral", 1.0)),
            )

    def test_round_trip_custom_mults(self, tmp_path):
        """Custom mom tilt mults round-trip preserves equality and ruleset_id."""
        custom = DecisionRuleset(
            enable_mom_state_tilt=True,
            mom_state_tilt_mults=(
                ("tailwind", 1.10), ("neutral", 0.85), ("headwind", 0.90),
            ),
        )
        path = str(tmp_path / "mom_tilt_custom.json")
        custom.to_json(path)
        loaded = DecisionRuleset.from_json(path)
        assert loaded == custom
        assert loaded.ruleset_id == custom.ruleset_id
        assert loaded.enable_mom_state_tilt is True
        assert loaded.mom_state_tilt_mults == (
            ("tailwind", 1.10), ("neutral", 0.85), ("headwind", 0.90),
        )

    def test_tilt_weights_in_portfolio(self):
        """With tilt enabled, tailwind weight > neutral weight in portfolio normalization."""
        rows = [
            {"size_band": "L", "eligible": "1", "cost_mult": 1.0,
             "catalyst_tilt_mult": 1.0, "mom_state_tilt_mult": 1.0},
            {"size_band": "L", "eligible": "1", "cost_mult": 1.0,
             "catalyst_tilt_mult": 1.0, "mom_state_tilt_mult": 0.85},
        ]
        compute_target_weights(rows)
        assert rows[0]["target_weight_pct"] > rows[1]["target_weight_pct"]
