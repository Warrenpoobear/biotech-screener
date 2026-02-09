"""Tests for DecisionRuleset dataclass — parameterized threshold config."""
import json
import os
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
        """VERSION is v1.1.0 after parameterization."""
        assert VERSION == "v1.1.0"


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
        """A ticker at 100 days is 'far' with 90d window but 'near' with 120d."""
        rec = _base_rec(
            catalyst_decay={"days_to_catalyst": 100, "in_optimal_window": False},
        )
        # Default: 90d → 100 days is far → B (not A)
        default_result = compute_decision_fields(rec, "drug_developer", 0.75)
        assert default_result["tier_dev"] == "B"
        assert "catalyst_far" in default_result["tier_reason"]

        # Wider: 120d → 100 days is near → A
        wide = DecisionRuleset(catalyst_near_days=120)
        wide_result = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=wide)
        assert wide_result["tier_dev"] == "A"
        assert "catalyst_near" in wide_result["tier_reason"]

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

    EXPECTED_DEFAULT_RULESET_ID = "a70b515b"

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
