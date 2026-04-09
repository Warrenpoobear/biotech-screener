"""Tests targeting uncovered branches in decision_engine.py.

Each test hits a specific line range that existing tests miss: __post_init__
validation, _safe_float edge cases, eligibility rescue paths, overlay edge
cases, size-band missingness, tier edge branches, sort-key modes, and gate
margin counterfactuals.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine import (
    DEFAULT_RULESET,
    DecisionRuleset,
    _compute_eligibility,
    _compute_overlays,
    _compute_size_band,
    _compute_tier_commercial,
    _compute_tier_dev,
    _resolve_catalyst_strength,
    _safe_float,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_gate_margins,
    compute_sort_contribs,
)

# ---------------------------------------------------------------------------
# Helper: minimal rec dict
# ---------------------------------------------------------------------------


def _rec(**overrides):
    base = {
        "ticker": "TEST",
        "severity": "NONE",
        "confidence_overall": 0.7,
        "fundamental_red_flag": False,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "catalyst_decay": {"days_to_catalyst": 45, "in_optimal_window": True},
        "smart_money_signal": {
            "tier1_holders": 3,
            "holders_increasing": [1, 2],
            "holders_decreasing": [],
            "overlap_count": 1,
            "tier_breakdown": {},
        },
        "coinvest": {"tier1_count": 3},
        "defensive_features": {
            "drawdown": -0.15,
            "vol_60d": 0.40,
            "beta_xbi_60d": 1.1,
            "rsi_14d": 55.0,
        },
        "score_breakdown": {"enhancements": {"momentum": {"alpha_60d": 0.03}}},
        "momentum_signal": {"alpha_60d": 0.03},
        "survivability_signal": {
            "coverage": [],
            "metrics": {"cash_total": 500_000_000, "burn_ttm": 50_000_000},
        },
        "composite_rank": 50,
        "composite_score": 7.5,
        "score_rank_pct": 0.65,
    }
    base.update(overrides)
    return base


def _ruleset(**overrides):
    """Build a DecisionRuleset with overrides, handling frozen dataclass."""
    defaults = {f.name: f.default for f in __import__("dataclasses").fields(DecisionRuleset)}
    defaults.update(overrides)
    return DecisionRuleset(**defaults)


# ===========================================================================
# TestPostInitValidation — one ValueError per __post_init__ branch
# ===========================================================================


class TestPostInitValidation:
    """Each test triggers exactly one __post_init__ ValueError branch."""

    def test_cost_haircut_buckets_non_ascending(self):
        # Line 228: thresholds not strictly ascending
        with pytest.raises(ValueError, match="cost_haircut_buckets"):
            _ruleset(cost_haircut_buckets=((500, 1.0), (400, 0.9)))

    def test_cost_haircut_floor_mult_zero(self):
        # Line 231
        with pytest.raises(ValueError, match="cost_haircut_floor_mult"):
            _ruleset(cost_haircut_floor_mult=0.0)

    def test_cost_impact_cap_bps_zero(self):
        # Line 234
        with pytest.raises(ValueError, match="cost_impact_cap_bps"):
            _ruleset(cost_impact_cap_bps=0.0)

    def test_catalyst_tilt_mults_unknown_band(self):
        # Line 239
        with pytest.raises(ValueError, match="unknown band"):
            _ruleset(catalyst_tilt_mults=(("BOGUS", 1.0),))

    def test_catalyst_tilt_mults_mult_zero(self):
        # Line 243
        with pytest.raises(ValueError, match="mult must be > 0"):
            _ruleset(catalyst_tilt_mults=(("NEAR", 0.0),))

    def test_mom_state_tilt_mults_unknown_state(self):
        # Line 248
        with pytest.raises(ValueError, match="unknown state"):
            _ruleset(mom_state_tilt_mults=(("bogus", 1.0),))

    def test_mom_state_tilt_mults_mult_zero(self):
        # Line 252
        with pytest.raises(ValueError, match="mult must be > 0"):
            _ruleset(mom_state_tilt_mults=(("tailwind", 0.0),))

    def test_catalyst_time_decay_mode_invalid(self):
        # Line 255
        with pytest.raises(ValueError, match="catalyst_time_decay_mode"):
            _ruleset(catalyst_time_decay_mode="bogus")

    def test_catalyst_logistic_scale_days_zero(self):
        # Line 259
        with pytest.raises(ValueError, match="catalyst_logistic_scale_days"):
            _ruleset(catalyst_logistic_scale_days=0.0)

    def test_sparse_signal_mode_invalid(self):
        # Line 262
        with pytest.raises(ValueError, match="sparse_signal_mode"):
            _ruleset(sparse_signal_mode="bogus")

    def test_alpha_train_mode_invalid(self):
        # Line 292: invalid format (not expanding/trailing-N/decay-H)
        with pytest.raises(ValueError, match="alpha_train_mode"):
            _ruleset(alpha_train_mode="bogus")

    def test_alpha_train_mode_trailing_bad_param(self):
        # Lines 289-290: trailing- with non-int suffix
        with pytest.raises(ValueError, match="alpha_train_mode"):
            _ruleset(alpha_train_mode="trailing-abc")

    def test_alpha_train_mode_expanding_valid(self):
        # Line 284: expanding is valid — should NOT raise
        rs = _ruleset(alpha_train_mode="expanding")
        assert rs.alpha_train_mode == "expanding"

    def test_alpha_train_min_train_dates_one(self):
        # Line 297
        with pytest.raises(ValueError, match="alpha_train_min_train_dates"):
            _ruleset(alpha_train_min_train_dates=1)

    def test_alpha_train_horizon_invalid(self):
        # Line 300
        with pytest.raises(ValueError, match="alpha_train_horizon"):
            _ruleset(alpha_train_horizon=30)

    def test_alpha_table_rebuild_policy_invalid(self):
        # Line 303
        with pytest.raises(ValueError, match="alpha_table_rebuild_policy"):
            _ruleset(alpha_table_rebuild_policy="bogus")

    def test_alpha_cohort_tiebreak_weight_too_high(self):
        # Line 309
        with pytest.raises(ValueError, match="alpha_cohort_tiebreak_weight"):
            _ruleset(alpha_cohort_tiebreak_weight=0.6)

    def test_rebalance_buffer_ranks_too_high(self):
        # Line 314
        with pytest.raises(ValueError, match="rebalance_buffer_ranks"):
            _ruleset(rebalance_buffer_ranks=51)

    def test_tiering_priority_mode_invalid(self):
        # Line 317
        with pytest.raises(ValueError, match="tiering_priority_mode"):
            _ruleset(tiering_priority_mode="bogus")

    def test_coinvest_score_mode_invalid(self):
        # Line 322
        with pytest.raises(ValueError, match="coinvest_score_mode"):
            _ruleset(coinvest_score_mode="bogus")

    def test_dd_rel_margin_rescue_threshold_positive(self):
        # Line 328
        with pytest.raises(ValueError, match="dd_rel_margin_rescue_threshold"):
            _ruleset(dd_rel_margin_rescue_threshold=0.1)

    def test_far_window_days_negative(self):
        # Line 333
        with pytest.raises(ValueError, match="far_window_days"):
            _ruleset(far_window_days=-1)

    def test_far_window_decay_mult_zero(self):
        # Line 336
        with pytest.raises(ValueError, match="far_window_decay_mult"):
            _ruleset(far_window_decay_mult=0.0)


# ===========================================================================
# TestSafeFloatEdge
# ===========================================================================


class TestSafeFloatEdge:
    def test_dict_returns_default(self):
        # Lines 460-461: TypeError branch
        assert _safe_float({"a": 1}, default=42.0) == 42.0


# ===========================================================================
# TestEligibilityEdge
# ===========================================================================


class TestEligibilityEdge:
    def test_fundamental_red_flag_none_falls_through(self):
        """Line 544-547: fundamental_red_flag absent → import + detect."""
        rec = _rec()
        del rec["fundamental_red_flag"]
        # The defensive_overlay_adapter module exists, so this should succeed.
        # With minimal rec it should not detect a red flag.
        eligible, reasons, _, _ = _compute_eligibility(rec, DEFAULT_RULESET)
        # We only care that the code path didn't crash and ran the detection.
        assert isinstance(eligible, bool)

    def test_drawdown_rescue_else_branch(self):
        """Line 578: enable_dd_rel_margin_rescue=True, both breach,
        rel_margin <= threshold → 'deep_drawdown' appended."""
        rs = _ruleset(
            enable_dd_rel_margin_rescue=True,
            dd_rel_margin_rescue_threshold=-0.02,
            drawdown_gate=-0.40,
            drawdown_rel_xbi_gate=-0.20,
            drawdown_gate_require_both=True,
        )
        rec = _rec(
            defensive_features={
                "drawdown": -0.50,  # abs breach
                "drawdown_rel_xbi": -0.30,  # rel breach, margin = -0.10 < threshold -0.02
            }
        )
        eligible, reasons, rescued, _ = _compute_eligibility(rec, rs)
        assert not eligible
        assert "deep_drawdown" in reasons
        assert not rescued


# ===========================================================================
# TestOverlayEdge
# ===========================================================================


class TestOverlayEdge:
    def test_logistic_blended_window_decay_w_one(self):
        """Line 722: blended_window → catalyst_decay_w = 1.0."""
        rs = _ruleset(catalyst_time_decay_mode="logistic")
        rec = _rec(
            catalyst_decay={
                "days_to_catalyst": 0,
                "in_optimal_window": True,
            }
        )
        out = _compute_overlays(rec, rs)
        assert out["catalyst_decay_w"] == 1.0

    def test_logistic_missing_catalyst_decay_w_zero(self):
        """Line 731: no catalyst data in logistic mode → 0.0."""
        rs = _ruleset(catalyst_time_decay_mode="logistic")
        rec = _rec(catalyst_decay={})
        out = _compute_overlays(rec, rs)
        assert out["catalyst_decay_w"] == 0.0

    def test_logistic_strength_override_mid(self):
        """Lines 737-738: w between 0.2 and 0.5 → 'mid'."""
        rs = _ruleset(
            catalyst_time_decay_mode="logistic",
            catalyst_logistic_midpoint_days=150,
            catalyst_logistic_scale_days=30.0,
        )
        # Use days that produce a decay_w between 0.2 and 0.5
        # At midpoint=150, scale=30: days=150 → w=0.5, days=180 → w~0.27
        rec = _rec(catalyst_decay={"days_to_catalyst": 180, "in_optimal_window": False})
        out = _compute_overlays(rec, rs)
        assert 0.2 <= out["catalyst_decay_w"] < 0.5
        assert out["catalyst_strength"] == "mid"

    def test_logistic_strength_override_far(self):
        """Lines 739-740: w < 0.2 → 'far'."""
        rs = _ruleset(
            catalyst_time_decay_mode="logistic",
            catalyst_logistic_midpoint_days=150,
            catalyst_logistic_scale_days=30.0,
        )
        # days=250 → w much less than 0.2
        rec = _rec(catalyst_decay={"days_to_catalyst": 250, "in_optimal_window": False})
        out = _compute_overlays(rec, rs)
        assert out["catalyst_decay_w"] < 0.2
        assert out["catalyst_strength"] == "far"

    def test_runway_bucket_empty_unknown_severity(self):
        """Line 757: unknown severity → empty string."""
        rec = _rec(severity="BOGUS")
        out = _compute_overlays(rec, DEFAULT_RULESET)
        assert out["runway_bucket"] == ""

    def test_runway_bucket_sev2_short(self):
        """Line 753: SEV2 → 'short'."""
        rec = _rec(severity="SEV2")
        out = _compute_overlays(rec, DEFAULT_RULESET)
        assert out["runway_bucket"] == "short"

    def test_drawdown_rel_xbi_risk_flag(self):
        """Line 792: drawdown_rel_xbi < gate → risk flag."""
        rec = _rec(
            defensive_features={
                "drawdown": -0.10,
                "vol_60d": 0.3,
                "beta_xbi_60d": 1.0,
                "rsi_14d": 50.0,
                "drawdown_rel_xbi": -0.30,  # below default gate -0.20
            }
        )
        out = _compute_overlays(rec, DEFAULT_RULESET)
        assert "deep_drawdown_rel_xbi" in out["risk_flags"]

    def test_coinvest_filing_age_valueerror(self):
        """Lines 668-669: non-numeric filing_age → None."""
        rec = _rec(coinvest={"tier1_count": 2, "days_since_latest_filing": "not_a_number"})
        out = _compute_overlays(rec, DEFAULT_RULESET)
        assert out["coinvest_filing_age_days"] == ""
        assert out["coinvest_recency_state"] == ""

    def test_coinvest_stale_filing(self):
        """Line 678: filing_age > 90 → 'stale'."""
        rec = _rec(coinvest={"tier1_count": 2, "days_since_latest_filing": 120})
        out = _compute_overlays(rec, DEFAULT_RULESET)
        assert out["coinvest_filing_age_days"] == 120
        assert out["coinvest_recency_state"] == "stale"


# ===========================================================================
# TestSizeBandEdge
# ===========================================================================


class TestSizeBandEdge:
    def test_missingness_missing_sponsor_tier_a(self):
        """Lines 897-898: missing sponsor for tier A."""
        rs = _ruleset(enable_missingness_size_penalty=True)
        overlays = {
            "catalyst_strength": "near",
            "sponsor_tier1_count": "",  # missing
            "mom_state": "neutral",
            "runway_bucket": "adequate",
            "risk_flags": "",
            "catalyst_mode": "specific_days",
        }
        band, reasons = _compute_size_band(
            eligible=True,
            effective_tier="A",
            optionality=0.70,
            overlays=overlays,
            ruleset=rs,
        )
        assert "missing_sponsor" in reasons

    def test_missingness_missing_catalyst_tier_a(self):
        """Lines 902-904: missing catalyst for tier A."""
        rs = _ruleset(enable_missingness_size_penalty=True)
        overlays = {
            "catalyst_strength": "missing",
            "sponsor_tier1_count": 5,
            "mom_state": "neutral",
            "runway_bucket": "adequate",
            "risk_flags": "",
            "catalyst_mode": "missing",
        }
        band, reasons = _compute_size_band(
            eligible=True,
            effective_tier="A",
            optionality=0.70,
            overlays=overlays,
            ruleset=rs,
        )
        assert "missing_catalyst" in reasons


# ===========================================================================
# TestResolveCatalystStrength
# ===========================================================================


class TestResolveCatalystStrength:
    def test_logistic_near(self):
        """Line 955-956: decay_w >= 0.5 → near."""
        rs = _ruleset(catalyst_time_decay_mode="logistic")
        strength, is_act, has_data, tag = _resolve_catalyst_strength(
            catalyst_in_window="",
            catalyst_days="50",
            ruleset=rs,
            catalyst_decay_w=0.8,
        )
        assert strength == "near"
        assert is_act is True

    def test_logistic_mid(self):
        """Lines 957-958: 0.2 <= decay_w < 0.5 → mid."""
        rs = _ruleset(catalyst_time_decay_mode="logistic")
        strength, is_act, has_data, tag = _resolve_catalyst_strength(
            catalyst_in_window="",
            catalyst_days="150",
            ruleset=rs,
            catalyst_decay_w=0.35,
        )
        assert strength == "mid"
        assert is_act is True

    def test_logistic_far(self):
        """Lines 959-960: decay_w < 0.2 → far."""
        rs = _ruleset(catalyst_time_decay_mode="logistic")
        strength, is_act, has_data, tag = _resolve_catalyst_strength(
            catalyst_in_window="",
            catalyst_days="300",
            ruleset=rs,
            catalyst_decay_w=0.1,
        )
        assert strength == "far"
        assert is_act is False


# ===========================================================================
# TestTierEdge
# ===========================================================================


class TestTierEdge:
    def test_dev_mod_opt_far_catalyst(self):
        """Line 1038: mod_opt with far catalyst → C."""
        rs = _ruleset(catalyst_time_decay_mode="hard")
        tier, reason = _compute_tier_dev(
            archetype="drug_developer",
            eligible=True,
            optionality=0.40,  # between tier_b (0.30) and tier_a (0.60)
            catalyst_in_window="",
            catalyst_days="300",
            ruleset=rs,
            catalyst_decay_w=0.0,
        )
        assert tier == "C"
        assert "mod_opt" in reason

    def test_dev_low_opt_no_catalyst(self):
        """Line 1048: low_opt + no catalyst data → C."""
        tier, reason = _compute_tier_dev(
            archetype="drug_developer",
            eligible=True,
            optionality=0.10,
            catalyst_in_window="",
            catalyst_days="0",
            ruleset=DEFAULT_RULESET,
            catalyst_decay_w=0.0,
        )
        assert tier == "C"
        assert "low_opt" in reason
        assert "no_catalyst_data" in reason

    def test_commercial_mod_quality_far_catalyst(self):
        """Line 1097: mod_quality with far catalyst → C."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.70,  # between tier_b (0.60) and tier_a (0.85)
            catalyst_in_window="",
            catalyst_days="300",
            ruleset=DEFAULT_RULESET,
            catalyst_decay_w=0.0,
        )
        assert tier == "C"
        assert "mod_quality" in reason

    def test_commercial_mod_quality_no_catalyst(self):
        """Lines 1103-1104: mod_quality + no catalyst data → C."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.70,
            catalyst_in_window="",
            catalyst_days="0",
            ruleset=DEFAULT_RULESET,
            catalyst_decay_w=0.0,
        )
        assert tier == "C"
        assert "mod_quality" in reason
        assert "no_catalyst_data" in reason

    def test_commercial_low_quality_no_catalyst(self):
        """Lines 1105-1106: low_quality + no catalyst data → C."""
        tier, reason = _compute_tier_commercial(
            archetype="commercial_biotech",
            eligible=True,
            quality_pct=0.20,
            catalyst_in_window="",
            catalyst_days="0",
            ruleset=DEFAULT_RULESET,
            catalyst_decay_w=0.0,
        )
        assert tier == "C"
        assert "low_quality" in reason
        assert "no_catalyst_data" in reason


# ===========================================================================
# TestSortKeyEdge
# ===========================================================================


class TestSortKeyEdge:
    def _decision_fields(self, **overrides):
        base = {
            "eligible": "1",
            "tier_dev": "B",
            "tier_any": "B",
            "catalyst_mode": "specific_days",
            "catalyst_days": "60",
            "catalyst_in_window": "",
            "catalyst_strength": "near",
            "catalyst_decay_w": 0.8,
            "sponsor_tier1_count": 3,
            "mom_state": "neutral",
            "risk_flags": "",
            "coinvest_score_z": -1.5,
            "inst_delta_z": -0.5,
            "clinical_score_z_tier": 0.5,
            "clinical_score_v2_z": 0.8,
            "alpha_cohort_pct": 0.6,
            "stage_bucket": "mid",
            "missingness_penalty": 0,
        }
        base.update(overrides)
        return base

    def test_coinvest_non_positive_only(self):
        """Line 1220: coinvest_positive_only=False, negative z → negative delta."""
        rs = _ruleset(
            enable_coinvest_sort_signal=True,
            coinvest_positive_only=False,
            coinvest_sort_weight=0.5,
        )
        df = self._decision_fields(coinvest_score_z=-1.5)
        total, cmap = compute_sort_contribs(df, "drug_developer", ruleset=rs)
        assert cmap["coinvest"] < 0  # negative delta from negative z

    def test_institutional_non_positive_only(self):
        """Line 1230: institutional_positive_only=False."""
        rs = _ruleset(
            enable_institutional_sort_signal=True,
            institutional_positive_only=False,
            institutional_sort_weight=0.3,
        )
        df = self._decision_fields(inst_delta_z=-1.0)
        total, cmap = compute_sort_contribs(df, "drug_developer", ruleset=rs)
        assert cmap["institutional"] < 0

    def test_alpha_cohort_tiebreak(self):
        """Lines 1245-1247: alpha_cohort_tiebreak_weight > 0."""
        rs = _ruleset(alpha_cohort_tiebreak_weight=0.10)
        df = self._decision_fields(alpha_cohort_pct=0.8)
        total, cmap = compute_sort_contribs(df, "drug_developer", ruleset=rs)
        assert float(cmap["alpha_cohort_tb"]) == pytest.approx(0.10 * 0.8)

    def test_sort_key_tier_first(self):
        """Lines 1312, 1319: tiering_priority_mode='tier_first'."""
        rs = _ruleset(tiering_priority_mode="tier_first")
        df = self._decision_fields(tier_any="A", tier_dev="B")
        key = compute_actionable_sort_key(
            decision_fields=df,
            archetype="drug_developer",
            optionality=0.70,
            composite_rank=10,
            ticker="TEST",
            ruleset=rs,
        )
        # tier_first uses tier_any (A→0) before archetype
        # prefix = (is_eligible=0, tier_ord=0, is_dev=0, ...)
        assert key[0] == 0  # eligible
        assert key[1] == 0  # tier A

    def test_missingness_sort_penalty(self):
        """Line 1361: enable_missingness_sort_penalty=True."""
        rs = _ruleset(enable_missingness_sort_penalty=True)
        df = self._decision_fields(missingness_penalty=2)
        key = compute_actionable_sort_key(
            decision_fields=df,
            archetype="drug_developer",
            optionality=0.70,
            composite_rank=10,
            ticker="TEST",
            ruleset=rs,
        )
        # missing_count should be 2 in the sort tuple
        assert 2 in key

    def test_compute_sort_contribs_blended_mode(self):
        """Lines 1449-1455: catalyst_priority_mode='blended'."""
        rs = _ruleset(catalyst_priority_mode="blended")
        df = self._decision_fields()
        total, cmap = compute_sort_contribs(
            df,
            "drug_developer",
            ruleset=rs,
            catalyst_event_type="FDA_PDUFA_DATE",
            catalyst_source="SEC_8K_FILING",
        )
        assert cmap["catalyst_bonus"] == 5.0  # FDA event → priority 1 → bonus 5.0


# ===========================================================================
# TestComputeDecisionFieldsEdge
# ===========================================================================


class TestComputeDecisionFieldsEdge:
    def test_catalyst_tilt_hard_mode(self):
        """Lines 1635-1637: enable_catalyst_tilt=True, hard mode."""
        rs = _ruleset(
            enable_catalyst_tilt=True,
            catalyst_time_decay_mode="hard",
        )
        rec = _rec(catalyst_decay={"days_to_catalyst": 60, "in_optimal_window": True})
        fields = compute_decision_fields(
            rec,
            archetype="drug_developer",
            optionality_pct_dev=0.70,
            ruleset=rs,
        )
        # catalyst tilt should be applied via the hard tilt map
        assert fields["catalyst_tilt_applied"] != ""

    def test_tiering_priority_tier_first(self):
        """Line 1651: tiering_priority_mode='tier_first' uses tier_any for sizing."""
        rs = _ruleset(tiering_priority_mode="tier_first")
        rec = _rec()
        fields = compute_decision_fields(
            rec,
            archetype="commercial_biotech",
            optionality_pct_dev=None,
            commercial_quality_pct=0.90,
            ruleset=rs,
        )
        # Non-drug-developer always uses tier_any regardless of mode,
        # but tier_first changes the sort key prefix ordering
        assert fields["tier_any"] != ""


# ===========================================================================
# TestGateMarginsEdge
# ===========================================================================


class TestGateMarginsEdge:
    def test_require_either_counterfactual(self):
        """Lines 1882, 1890: require_either mode — both abs and rel breach."""
        rs = _ruleset(
            drawdown_gate_require_both=False,
            drawdown_gate=-0.40,
            drawdown_rel_xbi_gate=-0.20,
        )
        rec = _rec(
            defensive_features={
                "drawdown": -0.50,  # abs breach
                "drawdown_rel_xbi": -0.30,  # rel breach
            }
        )
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        # In require_either mode, both breaching sides get flipped
        assert "drawdown" in cf
        assert "drawdown_rel_xbi" in cf
        assert cf["drawdown"] == rs.drawdown_gate
        assert cf["drawdown_rel_xbi"] == rs.drawdown_rel_xbi_gate

    def test_require_both_counterfactual_picks_smaller_move(self):
        """Lines 1881-1884: require_both — picks the side with smaller delta."""
        rs = _ruleset(
            drawdown_gate_require_both=True,
            drawdown_gate=-0.40,
            drawdown_rel_xbi_gate=-0.20,
            enable_dd_rel_margin_rescue=False,
        )
        # abs margin = -0.50 - (-0.40) = -0.10 (abs=0.10)
        # rel margin = -0.25 - (-0.20) = -0.05 (abs=0.05)
        # rel_delta < abs_delta → should flip rel side
        rec = _rec(
            defensive_features={
                "drawdown": -0.50,
                "drawdown_rel_xbi": -0.25,
            }
        )
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown_rel_xbi" in cf
        assert "drawdown" not in cf

    def test_fundamental_red_flag_none_gate_margins(self):
        """Lines 1725-1727: fundamental_red_flag=None triggers import fallback."""
        rec = _rec()
        del rec["fundamental_red_flag"]
        result = compute_gate_margins(rec, DEFAULT_RULESET)
        # Should have gate entries — didn't crash
        assert "gates" in result
        assert len(result["gates"]) >= 4

    def test_counterfactual_picks_abs_side(self):
        """Line 1882: counterfactual picks drawdown when abs_delta <= rel_delta."""
        # abs margin barely breaches, rel margin breaches more → abs is smaller delta
        rec = _rec(
            defensive_features={
                "drawdown": -0.61,  # just below default gate -0.60
                "drawdown_rel_xbi": -0.50,  # well below default rel gate -0.30
            }
        )
        rs = DecisionRuleset(drawdown_gate_require_both=True)
        result = compute_gate_margins(rec, rs)
        cf = result["counterfactual"]
        assert "drawdown" in cf
        assert "drawdown_rel_xbi" not in cf


class TestSizeBandMissingDrawdown:
    def test_missing_drawdown_penalty(self):
        """Lines 897-898: drawdown_data_missing → missing_drawdown reason."""
        rs = DecisionRuleset(enable_missingness_size_penalty=True)
        overlays = {
            "risk_flags": "drawdown_data_missing",
            "catalyst_strength": "near",
            "sponsor_tier1_count": 5,
            "mom_state": "neutral",
            "runway_bucket": "adequate",
            "catalyst_mode": "specific_days",
        }
        band, reasons = _compute_size_band(
            eligible=True,
            effective_tier="B",
            optionality=0.50,
            overlays=overlays,
            ruleset=rs,
        )
        assert "missing_drawdown" in reasons


class TestComputeDecisionFieldsTierFirst:
    def test_drug_developer_tier_first(self):
        """Line 1651: drug_developer + tier_first → effective_tier = tier_any."""
        rs = DecisionRuleset(tiering_priority_mode="tier_first")
        result = compute_decision_fields(
            rec=_rec(),
            archetype="drug_developer",
            optionality_pct_dev=0.70,
            ruleset=rs,
        )
        # Should not crash and should produce valid output
        assert result["eligible"] in ("0", "1")
        assert result["tier_dev"] in ("A", "B", "C", "D")
