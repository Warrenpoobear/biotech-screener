"""
Decision Engine Contract + Golden Fixture Tests (CI-enforced).

Prevents silent model regressions: weight normalization bugs, flat scores,
missing fields, confidence gates misfiring, overrides not taking effect, etc.

Test categories:
    1. SCHEMA — required fields + types on every output
    2. INVARIANTS — hard asserts on dispersion, uniqueness, normalization, monotonicity
    3. GOLDEN FIXTURES — 10 synthetic tickers → expected exact outputs (deterministic)
    4. CONFIDENCE MONOTONICITY — lower confidence never boosts score / tier
    5. OVERRIDE EFFECTS — changing a ruleset param actually changes output
    6. FULL PIPELINE — cohort-level sort + weight invariants
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DECISION_COLUMNS,
    VERSION,
    DecisionRuleset,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_target_weights,
)

# =============================================================================
# FIXTURE BUILDER
# =============================================================================


def _rec(
    ticker: str = "TEST",
    severity: str = "NONE",
    confidence_overall: float = 0.72,
    fundamental_red_flag: bool = False,
    fundamental_red_flag_reasons: list | None = None,
    catalyst_days: int | None = None,
    catalyst_in_window: bool | None = None,
    vol_60d: float | None = 0.65,
    beta_xbi_60d: float | None = 1.1,
    drawdown: float | None = -0.15,
    rsi_14d: float | None = 48.0,
    alpha_60d: float | None = 0.02,
    tier1_count: int | None = 3,
    overlap_count: int | None = 2,
    holders_inc: list | None = None,
    holders_dec: list | None = None,
) -> Dict[str, Any]:
    """Build a fully-populated synthetic rec for decision engine testing."""
    rec: Dict[str, Any] = {
        "ticker": ticker,
        "severity": severity,
        "confidence_overall": confidence_overall,
        "fundamental_red_flag": fundamental_red_flag,
        "fundamental_red_flag_reasons": fundamental_red_flag_reasons or [],
        "flags": [],
        "attn_flags": [],
    }

    # Catalyst
    cd: Dict[str, Any] = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window
    rec["catalyst_decay"] = cd if cd else None

    # Smart money
    sms: Dict[str, Any] = {}
    if overlap_count is not None:
        sms["overlap_count"] = overlap_count
    if holders_inc is not None:
        sms["holders_increasing"] = holders_inc
    if holders_dec is not None:
        sms["holders_decreasing"] = holders_dec
    rec["smart_money_signal"] = sms if sms else {}

    # Coinvest
    rec["coinvest"] = {"tier1_count": tier1_count} if tier1_count is not None else {}

    # Defensive features
    df: Dict[str, Any] = {}
    if vol_60d is not None:
        df["vol_60d"] = vol_60d
    if beta_xbi_60d is not None:
        df["beta_xbi_60d"] = beta_xbi_60d
    if drawdown is not None:
        df["drawdown"] = drawdown
    if rsi_14d is not None:
        df["rsi_14d"] = rsi_14d
    rec["defensive_features"] = df

    # Momentum (via score_breakdown)
    if alpha_60d is not None:
        rec["score_breakdown"] = {"enhancements": {"momentum": {"alpha_60d": alpha_60d}}}
    else:
        rec["score_breakdown"] = {}
    rec["momentum_signal"] = {}

    return rec


# =============================================================================
# 10 GOLDEN FIXTURES — diverse cohort covering all tiers, archetypes, modes
# =============================================================================

# Fixture 1: Tier A — high opt + specific catalyst near
GOLDEN_1 = {
    "ticker": "GLD_TIERA",
    "rec": _rec("GLD_TIERA", catalyst_days=45, catalyst_in_window=True, alpha_60d=0.08, tier1_count=3),
    "archetype": "drug_developer",
    "optionality": 0.75,
    "expected": {
        "eligible": "1",
        "tier_dev": "A",
        "tier_reason": "high_opt+catalyst_near",
        "catalyst_mode": "specific_days",
        "catalyst_days": 45,
        "size_band": "L",  # tier_a_dev + sponsor + tailwind
        "mom_state": "tailwind",
    },
}

# Fixture 2: Tier A — high opt + catalyst mid (90 < 120 ≤ 180)
GOLDEN_2 = {
    "ticker": "GLD_TIERB_FAR",
    "rec": _rec("GLD_TIERB_FAR", catalyst_days=120, catalyst_in_window=False, alpha_60d=0.02, tier1_count=1),
    "archetype": "drug_developer",
    "optionality": 0.70,
    "expected": {
        "eligible": "1",
        "tier_dev": "A",
        "tier_reason": "high_opt+catalyst_mid",
        "catalyst_mode": "specific_days",
        "catalyst_strength": "mid",
        "catalyst_days": 120,
        "mom_state": "neutral",
    },
}

# Fixture 3: Tier B — blended window + high opt
GOLDEN_3 = {
    "ticker": "GLD_BLEND",
    "rec": _rec("GLD_BLEND", catalyst_days=0, catalyst_in_window=True, alpha_60d=0.02, tier1_count=2),
    "archetype": "drug_developer",
    "optionality": 0.65,
    "expected": {
        "eligible": "1",
        "tier_dev": "A",
        "tier_reason": "high_opt+catalyst_window",
        "catalyst_mode": "blended_window",
        "catalyst_days": 0,
        "catalyst_in_window": "1",
    },
}

# Fixture 4: Tier B — no catalyst data, high opt
GOLDEN_4 = {
    "ticker": "GLD_NOCAT",
    "rec": _rec("GLD_NOCAT"),  # no catalyst args → missing mode
    "archetype": "drug_developer",
    "optionality": 0.70,
    "expected": {
        "eligible": "1",
        "tier_dev": "B",
        "tier_reason": "high_opt+no_catalyst_data",
        "catalyst_mode": "missing",
        "catalyst_days": "",
        "catalyst_in_window": "",
    },
}

# Fixture 5: Tier C — low optionality
GOLDEN_5 = {
    "ticker": "GLD_LOWOPT",
    "rec": _rec("GLD_LOWOPT", catalyst_days=30, catalyst_in_window=True, alpha_60d=0.02, tier1_count=0),
    "archetype": "drug_developer",
    "optionality": 0.20,
    "expected": {
        "eligible": "1",
        "tier_dev": "C",
        "tier_reason": "low_opt",
        "catalyst_mode": "specific_days",
    },
}

# Fixture 6: Tier D — ineligible (SEV3)
GOLDEN_6 = {
    "ticker": "GLD_INEL",
    "rec": _rec(
        "GLD_INEL",
        severity="SEV3",
        fundamental_red_flag=True,
        fundamental_red_flag_reasons=["cash_runway_lt_6m"],
        catalyst_days=30,
        catalyst_in_window=True,
    ),
    "archetype": "drug_developer",
    "optionality": 0.80,
    "expected": {
        "eligible": "0",
        "tier_dev": "D",
        "tier_reason": "ineligible",
        "size_band": "XS",
        "runway_bucket": "critical",
    },
}

# Fixture 7: Tier D — ineligible (deep drawdown)
GOLDEN_7 = {
    "ticker": "GLD_DD",
    "rec": _rec("GLD_DD", drawdown=-0.45, catalyst_days=30, catalyst_in_window=True),
    "archetype": "drug_developer",
    "optionality": 0.80,
    "expected": {
        "eligible": "0",
        "tier_dev": "D",
        "size_band": "XS",
    },
}

# Fixture 8: Commercial archetype — blank tier
GOLDEN_8 = {
    "ticker": "GLD_COMM",
    "rec": _rec("GLD_COMM", catalyst_days=30, catalyst_in_window=True, alpha_60d=0.06),
    "archetype": "commercial_biotech",
    "optionality": 0.80,
    "expected": {
        "eligible": "1",
        "tier_dev": "",
        "tier_reason": "",
        "catalyst_mode": "specific_days",
    },
}

# Fixture 9: High risk flags — vol + beta + drawdown flag + overbought + headwind
GOLDEN_9 = {
    "ticker": "GLD_RISK",
    "rec": _rec(
        "GLD_RISK",
        vol_60d=1.50,
        beta_xbi_60d=2.0,
        drawdown=-0.38,
        rsi_14d=75.0,
        alpha_60d=-0.10,
        catalyst_days=30,
        catalyst_in_window=True,
        tier1_count=0,
    ),
    "archetype": "drug_developer",
    "optionality": 0.40,
    "expected": {
        "eligible": "1",
        "mom_state": "headwind",
    },
}

# Fixture 10: Blank-when-missing sponsorship
GOLDEN_10 = {
    "ticker": "GLD_NOSPON",
    "rec": _rec("GLD_NOSPON", tier1_count=None, overlap_count=None),
    "archetype": "drug_developer",
    "optionality": 0.50,
    "expected": {
        "eligible": "1",
        "sponsor_tier1_count": "",
        "sponsor_overlap_count": "",
        "sponsor_net_buying": "",
    },
}

ALL_GOLDEN = [GOLDEN_1, GOLDEN_2, GOLDEN_3, GOLDEN_4, GOLDEN_5, GOLDEN_6, GOLDEN_7, GOLDEN_8, GOLDEN_9, GOLDEN_10]


def _compute_golden(g: dict) -> Dict[str, Any]:
    """Run compute_decision_fields for a golden fixture."""
    return compute_decision_fields(
        g["rec"],
        archetype=g["archetype"],
        optionality_pct_dev=g["optionality"],
    )


# =============================================================================
# 1. SCHEMA CONTRACT
# =============================================================================


class TestSchemaContract:
    """Every output must contain all DECISION_COLUMNS with correct types."""

    @pytest.fixture(params=ALL_GOLDEN, ids=[g["ticker"] for g in ALL_GOLDEN])
    def golden_output(self, request):
        g = request.param
        return g, _compute_golden(g)

    def test_all_decision_columns_present(self, golden_output):
        g, fields = golden_output
        for col in DECISION_COLUMNS:
            assert col in fields, f"[{g['ticker']}] Missing column: {col}"

    def test_version_string(self, golden_output):
        _, fields = golden_output
        assert fields["decision_engine_version"] == VERSION

    def test_ruleset_id_format(self, golden_output):
        _, fields = golden_output
        rid = fields["decision_engine_ruleset_id"]
        assert isinstance(rid, str) and len(rid) == 8, f"ruleset_id must be 8-char hex, got: {rid!r}"

    def test_eligible_is_binary_string(self, golden_output):
        _, fields = golden_output
        assert fields["eligible"] in ("0", "1"), f"eligible must be '0' or '1', got: {fields['eligible']!r}"

    def test_tier_dev_valid(self, golden_output):
        _, fields = golden_output
        assert fields["tier_dev"] in (
            "A",
            "B",
            "C",
            "D",
            "",
        ), f"tier_dev must be A/B/C/D/'', got: {fields['tier_dev']!r}"

    def test_size_band_valid(self, golden_output):
        _, fields = golden_output
        assert fields["size_band"] in ("L", "M", "S", "XS"), f"size_band must be L/M/S/XS, got: {fields['size_band']!r}"

    def test_catalyst_mode_valid(self, golden_output):
        _, fields = golden_output
        assert fields["catalyst_mode"] in (
            "specific_days",
            "blended_window",
            "far_window",
            "no_upcoming",
            "missing",
        ), f"catalyst_mode invalid: {fields['catalyst_mode']!r}"

    def test_catalyst_strength_valid(self, golden_output):
        _, fields = golden_output
        assert fields["catalyst_strength"] in (
            "near",
            "mid",
            "far",
            "missing",
        ), f"catalyst_strength invalid: {fields['catalyst_strength']!r}"

    def test_mom_state_valid(self, golden_output):
        _, fields = golden_output
        assert fields["mom_state"] in (
            "tailwind",
            "neutral",
            "headwind",
        ), f"mom_state must be tailwind/neutral/headwind, got: {fields['mom_state']!r}"

    def test_runway_bucket_valid(self, golden_output):
        _, fields = golden_output
        assert fields["runway_bucket"] in (
            "critical",
            "short",
            "adequate",
            "",
        ), f"runway_bucket invalid: {fields['runway_bucket']!r}"

    def test_catalyst_days_type(self, golden_output):
        _, fields = golden_output
        cd = fields["catalyst_days"]
        assert isinstance(cd, (int, str)), f"catalyst_days must be int or '' (str), got {type(cd).__name__}: {cd!r}"
        if isinstance(cd, str):
            assert cd == "", f"catalyst_days str must be '', got: {cd!r}"

    def test_catalyst_in_window_type(self, golden_output):
        _, fields = golden_output
        ciw = fields["catalyst_in_window"]
        assert ciw in ("0", "1", ""), f"catalyst_in_window must be '0'/'1'/'', got: {ciw!r}"

    def test_sponsor_tier1_count_type(self, golden_output):
        _, fields = golden_output
        val = fields["sponsor_tier1_count"]
        assert isinstance(val, (int, str)), f"sponsor_tier1_count must be int or '' (str), got {type(val).__name__}"

    def test_risk_flags_format(self, golden_output):
        _, fields = golden_output
        rf = fields["risk_flags"]
        assert isinstance(rf, str), f"risk_flags must be str, got {type(rf).__name__}"
        if rf:
            valid = {
                "high_vol",
                "high_beta",
                "deep_drawdown",
                "overbought_rsi",
                "low_confidence",
                "drawdown_data_missing",
            }
            for flag in rf.split("|"):
                assert flag in valid, f"Unknown risk flag: {flag!r}"


# =============================================================================
# 2. INVARIANTS — cohort-level checks on the full fixture set
# =============================================================================


class TestInvariants:
    """Hard asserts on aggregate properties of the decision engine."""

    @pytest.fixture(scope="class")
    def cohort(self) -> List[Dict[str, Any]]:
        """Run all 10 golden fixtures through the engine."""
        return [_compute_golden(g) for g in ALL_GOLDEN]

    def test_tier_dispersion(self, cohort):
        """Tiers must not all be identical — engine produces meaningful differentiation."""
        tiers = {f["tier_dev"] for f in cohort}
        assert len(tiers) >= 3, (
            f"Only {len(tiers)} distinct tier(s) across 10 fixtures: {tiers}. "
            "Engine may be producing flat/degenerate output."
        )

    def test_size_band_dispersion(self, cohort):
        """Size bands must not all be identical."""
        bands = {f["size_band"] for f in cohort}
        assert len(bands) >= 2, f"Only {len(bands)} distinct band(s): {bands}. Sizing logic may be broken."

    def test_eligible_dispersion(self, cohort):
        """Both eligible and ineligible tickers must appear."""
        eligibles = {f["eligible"] for f in cohort}
        assert eligibles == {"0", "1"}, f"Expected both eligible and ineligible, got: {eligibles}"

    def test_catalyst_mode_dispersion(self, cohort):
        """At least 3 distinct catalyst modes should appear."""
        modes = {f["catalyst_mode"] for f in cohort}
        assert len(modes) >= 3, f"Only {len(modes)} catalyst mode(s): {modes}"

    def test_ineligible_implies_tier_d_for_dev(self, cohort):
        """Every ineligible drug_developer gets tier D."""
        for i, g in enumerate(ALL_GOLDEN):
            fields = cohort[i]
            if fields["eligible"] == "0" and g["archetype"] == "drug_developer":
                assert (
                    fields["tier_dev"] == "D"
                ), f"[{g['ticker']}] Ineligible dev should be D, got {fields['tier_dev']}"

    def test_ineligible_implies_xs_band(self, cohort):
        """Every ineligible ticker gets XS size band."""
        for i, g in enumerate(ALL_GOLDEN):
            fields = cohort[i]
            if fields["eligible"] == "0":
                assert (
                    fields["size_band"] == "XS"
                ), f"[{g['ticker']}] Ineligible should be XS, got {fields['size_band']}"

    def test_version_consistent(self, cohort):
        """All outputs carry the same version."""
        versions = {f["decision_engine_version"] for f in cohort}
        assert len(versions) == 1, f"Inconsistent versions: {versions}"

    def test_ruleset_id_consistent(self, cohort):
        """All outputs carry the same ruleset_id (default ruleset)."""
        ids = {f["decision_engine_ruleset_id"] for f in cohort}
        assert len(ids) == 1, f"Inconsistent ruleset IDs: {ids}"


# =============================================================================
# 3. GOLDEN FIXTURE EXACT-MATCH TESTS
# =============================================================================


class TestGoldenFixtures:
    """Each golden fixture must produce the exact expected key-value pairs."""

    @pytest.fixture(params=ALL_GOLDEN, ids=[g["ticker"] for g in ALL_GOLDEN])
    def golden_pair(self, request):
        g = request.param
        return g, _compute_golden(g)

    def test_expected_fields_match(self, golden_pair):
        g, fields = golden_pair
        for key, expected_val in g["expected"].items():
            actual = fields.get(key)
            assert actual == expected_val, f"[{g['ticker']}] {key}: expected {expected_val!r}, got {actual!r}"


# =============================================================================
# 4. CONFIDENCE MONOTONICITY
# =============================================================================


class TestConfidenceMonotonicity:
    """Lower confidence never boosts score / removes risk flags.

    Confidence is a risk flag (not a hard gate), so lowering confidence
    should only ADD the low_confidence flag — never remove existing flags
    or improve tier/sizing.
    """

    def _fields_at_confidence(self, conf: float) -> Dict[str, Any]:
        rec = _rec(
            "CONF_TEST",
            confidence_overall=conf,
            catalyst_days=45,
            catalyst_in_window=True,
            alpha_60d=0.02,
            tier1_count=1,
        )
        return compute_decision_fields(
            rec,
            archetype="drug_developer",
            optionality_pct_dev=0.65,
        )

    def test_low_confidence_adds_flag(self):
        """Confidence below threshold adds low_confidence flag."""
        high = self._fields_at_confidence(0.80)
        low = self._fields_at_confidence(0.10)
        assert "low_confidence" not in high["risk_flags"]
        assert "low_confidence" in low["risk_flags"]

    def test_tier_unchanged_by_confidence(self):
        """Tier is NOT a hard gate on confidence — same tier at any confidence."""
        high = self._fields_at_confidence(0.80)
        low = self._fields_at_confidence(0.10)
        assert (
            high["tier_dev"] == low["tier_dev"]
        ), f"Tier changed with confidence: {high['tier_dev']} → {low['tier_dev']}"

    def test_eligibility_unchanged_by_confidence(self):
        """Confidence alone does not gate eligibility."""
        high = self._fields_at_confidence(0.80)
        low = self._fields_at_confidence(0.05)
        assert high["eligible"] == low["eligible"] == "1"

    def test_low_confidence_never_improves_band(self):
        """Lower confidence never results in a LARGER size band."""
        band_order = {"XS": 0, "S": 1, "M": 2, "L": 3}
        high = self._fields_at_confidence(0.80)
        low = self._fields_at_confidence(0.10)
        assert (
            band_order[low["size_band"]] <= band_order[high["size_band"]]
        ), f"Lower confidence increased band: {high['size_band']} → {low['size_band']}"


# =============================================================================
# 5. OVERRIDE / RULESET EFFECT TESTS
# =============================================================================


class TestOverrideEffects:
    """Changing a ruleset parameter must actually change output."""

    def _base_rec(self) -> Dict[str, Any]:
        return _rec(
            "OVERRIDE_TEST", catalyst_days=45, catalyst_in_window=True, alpha_60d=0.02, tier1_count=1, drawdown=-0.15
        )

    def test_a_floor_changes_tier(self):
        """Raising a_floor from 0.55 to 0.90 should downgrade tier for 0.65 optionality."""
        rec = self._base_rec()
        rs_low = DecisionRuleset(tier_a_optionality_floor=0.55)
        rs_high = DecisionRuleset(tier_a_optionality_floor=0.90)

        f_low = compute_decision_fields(rec, "drug_developer", 0.65, ruleset=rs_low)
        f_high = compute_decision_fields(rec, "drug_developer", 0.65, ruleset=rs_high)

        assert f_low["tier_dev"] == "A", f"Expected A with floor=0.55, got {f_low['tier_dev']}"
        assert f_high["tier_dev"] != "A", f"Expected non-A with floor=0.90, got {f_high['tier_dev']}"

    def test_drawdown_gate_changes_eligibility(self):
        """Tightening drawdown gate from -0.40 to -0.10 should make -0.15 drawdown ineligible."""
        rec = self._base_rec()
        rs_loose = DecisionRuleset(drawdown_gate=-0.40)
        rs_tight = DecisionRuleset(drawdown_gate=-0.10)

        f_loose = compute_decision_fields(rec, "drug_developer", 0.65, ruleset=rs_loose)
        f_tight = compute_decision_fields(rec, "drug_developer", 0.65, ruleset=rs_tight)

        assert f_loose["eligible"] == "1"
        assert f_tight["eligible"] == "0", "Tighter drawdown gate should make -0.15 ineligible"

    def test_soft_drawdown_preserves_eligibility(self):
        """Soft drawdown mode: breach stays eligible but band is penalized."""
        rec = _rec("DD_SOFT", drawdown=-0.50, catalyst_days=45, catalyst_in_window=True)

        rs_hard = DecisionRuleset(drawdown_gate_mode="hard")
        rs_soft = DecisionRuleset(drawdown_gate_mode="soft")

        f_hard = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_hard)
        f_soft = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_soft)

        assert f_hard["eligible"] == "0"
        assert f_soft["eligible"] == "1"
        assert "drawdown_penalty" in f_soft["size_reasons"]

    def test_catalyst_mid_days_changes_tier(self):
        """Shortening catalyst_mid_days to 30 with days=45 should downgrade from A to B.

        When catalyst_mid_days=30, days=45 > 30 → strength=far → NOT actionable → B.
        Must use in_optimal_window=False so only the days path counts.
        """
        rec = _rec(
            "CAT_DAYS_TEST", catalyst_days=45, catalyst_in_window=False, alpha_60d=0.02, tier1_count=1, drawdown=-0.15
        )
        rs_mid180 = DecisionRuleset(catalyst_near_days=90, catalyst_mid_days=180)
        rs_mid30 = DecisionRuleset(catalyst_near_days=30, catalyst_mid_days=30)

        f_mid180 = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_mid180)
        f_mid30 = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_mid30)

        assert f_mid180["tier_dev"] == "A", "days=45 within mid=180 should be tier A"
        assert f_mid30["tier_dev"] == "B", f"days=45 outside mid=30 should be tier B (far), got {f_mid30['tier_dev']}"

    def test_sponsor_threshold_changes_sizing(self):
        """Raising sponsor_confirm_threshold above tier1_count should change sizing."""
        rec = _rec("SPON_TEST", catalyst_days=45, catalyst_in_window=True, tier1_count=2, alpha_60d=0.02)

        rs_low = DecisionRuleset(sponsor_confirm_threshold=2)
        rs_high = DecisionRuleset(sponsor_confirm_threshold=10)

        f_low = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_low)
        f_high = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs_high)

        assert "sponsor_confirmed" in f_low["size_reasons"]
        assert "sponsor_confirmed" not in f_high["size_reasons"]

    def test_ruleset_id_changes_with_params(self):
        """Different params produce different ruleset IDs."""
        rs1 = DecisionRuleset(tier_a_optionality_floor=0.55)
        rs2 = DecisionRuleset(tier_a_optionality_floor=0.60)
        assert rs1.ruleset_id != rs2.ruleset_id


# =============================================================================
# 6. FULL PIPELINE — sort + weight invariants on a diverse cohort
# =============================================================================


class TestFullPipeline:
    """Run all 10 golden fixtures through sort + weight assignment."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Compute decision fields, sort, and assign weights for the full cohort."""
        rows = []
        for g in ALL_GOLDEN:
            fields = _compute_golden(g)
            fields["ticker"] = g["ticker"]
            fields["archetype"] = g["archetype"]
            fields["optionality"] = g["optionality"]
            fields["composite_rank"] = ALL_GOLDEN.index(g) + 1
            rows.append(fields)

        # Sort by actionable key
        rows.sort(
            key=lambda r: compute_actionable_sort_key(
                decision_fields=r,
                archetype=r["archetype"],
                optionality=r["optionality"],
                composite_rank=r["composite_rank"],
                ticker=r["ticker"],
            )
        )

        # Assign actionable ranks
        rank = 0
        for r in rows:
            if r["eligible"] == "1":
                rank += 1
                r["actionable_rank"] = rank
            else:
                r["actionable_rank"] = ""

        # Assign weights to eligible subset
        eligible = [r for r in rows if r["eligible"] == "1"]
        compute_target_weights(eligible)

        return rows, eligible

    def test_eligible_sort_before_ineligible(self, pipeline_result):
        """Eligible tickers appear before ineligible ones."""
        rows, _ = pipeline_result
        saw_ineligible = False
        for r in rows:
            if r["eligible"] == "0":
                saw_ineligible = True
            elif saw_ineligible:
                pytest.fail(f"Eligible ticker {r['ticker']} found after ineligible ticker")

    def test_actionable_ranks_unique(self, pipeline_result):
        """Actionable ranks are unique among eligible tickers."""
        _, eligible = pipeline_result
        ranks = [r["actionable_rank"] for r in eligible]
        assert len(ranks) == len(set(ranks)), f"Duplicate actionable ranks: {ranks}"

    def test_actionable_ranks_sequential(self, pipeline_result):
        """Actionable ranks are 1-indexed and sequential."""
        _, eligible = pipeline_result
        ranks = [r["actionable_rank"] for r in eligible]
        expected = list(range(1, len(eligible) + 1))
        assert ranks == expected, f"Ranks not sequential: {ranks} (expected {expected})"

    def test_weights_sum_to_100(self, pipeline_result):
        """Eligible weights sum to ~100%."""
        _, eligible = pipeline_result
        total = sum(r.get("target_weight_pct", 0) for r in eligible)
        assert abs(total - 100.0) < 0.1, f"Weights sum to {total:.2f}%, expected ~100%"

    def test_weight_proportionality(self, pipeline_result):
        """Larger size bands always get >= weight than smaller bands."""
        _, eligible = pipeline_result
        band_order = {"XS": 0, "S": 1, "M": 2, "L": 3}
        # Group by band and check max of smaller <= min of larger
        by_band: Dict[str, list] = {}
        for r in eligible:
            by_band.setdefault(r["size_band"], []).append(r["target_weight_pct"])

        bands_present = sorted(by_band.keys(), key=lambda b: band_order[b])
        for i in range(len(bands_present) - 1):
            smaller = bands_present[i]
            larger = bands_present[i + 1]
            max_smaller = max(by_band[smaller])
            min_larger = min(by_band[larger])
            assert (
                max_smaller <= min_larger + 0.01
            ), f"{smaller} weight ({max_smaller:.2f}) > {larger} weight ({min_larger:.2f})"

    def test_dev_sorts_before_commercial(self, pipeline_result):
        """Eligible drug_developer tickers sort before eligible commercial tickers."""
        rows, _ = pipeline_result
        eligible_rows = [r for r in rows if r["eligible"] == "1"]
        saw_commercial = False
        for r in eligible_rows:
            if r["archetype"] != "drug_developer":
                saw_commercial = True
            elif saw_commercial:
                pytest.fail(f"Dev ticker {r['ticker']} sorted after commercial ticker")

    def test_tier_a_before_tier_b_in_dev(self, pipeline_result):
        """Within eligible dev tickers, A-tier sorts before B-tier."""
        rows, _ = pipeline_result
        dev_eligible = [r for r in rows if r["eligible"] == "1" and r["archetype"] == "drug_developer"]
        saw_b = False
        for r in dev_eligible:
            if r["tier_dev"] == "B":
                saw_b = True
            elif r["tier_dev"] == "A" and saw_b:
                pytest.fail(f"A-tier {r['ticker']} sorted after B-tier ticker")

    def test_ineligible_have_no_weight(self, pipeline_result):
        """Ineligible tickers should not have target_weight_pct."""
        rows, _ = pipeline_result
        for r in rows:
            if r["eligible"] == "0":
                assert "target_weight_pct" not in r or r.get("target_weight_pct") in (
                    "",
                    None,
                ), f"Ineligible {r['ticker']} has weight: {r.get('target_weight_pct')}"


# =============================================================================
# 7. DETERMINISM — same inputs always produce same outputs
# =============================================================================


class TestDeterminism:
    """Decision engine is a pure function: same inputs → same outputs."""

    def test_repeated_calls_identical(self):
        """Calling compute_decision_fields twice produces identical dicts."""
        rec = _rec("DET_TEST", catalyst_days=45, catalyst_in_window=True)
        f1 = compute_decision_fields(rec, "drug_developer", 0.65)
        f2 = compute_decision_fields(rec, "drug_developer", 0.65)
        assert f1 == f2

    def test_sort_order_stable(self):
        """Sorting the same cohort twice produces identical order."""
        keys1 = []
        keys2 = []
        for g in ALL_GOLDEN:
            fields = _compute_golden(g)
            idx = ALL_GOLDEN.index(g)
            key = compute_actionable_sort_key(
                fields,
                g["archetype"],
                g["optionality"],
                idx + 1,
                g["ticker"],
            )
            keys1.append((g["ticker"], key))

        for g in ALL_GOLDEN:
            fields = _compute_golden(g)
            idx = ALL_GOLDEN.index(g)
            key = compute_actionable_sort_key(
                fields,
                g["archetype"],
                g["optionality"],
                idx + 1,
                g["ticker"],
            )
            keys2.append((g["ticker"], key))

        order1 = [t for t, _ in sorted(keys1, key=lambda x: x[1])]
        order2 = [t for t, _ in sorted(keys2, key=lambda x: x[1])]
        assert order1 == order2

    def test_weight_normalization_stable(self):
        """compute_target_weights produces same output on repeated calls."""

        def _make_rows():
            return [
                {"size_band": "L", "ticker": "A"},
                {"size_band": "M", "ticker": "B"},
                {"size_band": "S", "ticker": "C"},
            ]

        r1 = compute_target_weights(_make_rows())
        r2 = compute_target_weights(_make_rows())
        for a, b in zip(r1, r2):
            assert a["target_weight_pct"] == b["target_weight_pct"]


# =============================================================================
# 8. DRAWDOWN DATA MISSING — risk flag behaviour
# =============================================================================


class TestDrawdownDataMissing:
    """drawdown_data_missing flag when defensive_features lacks drawdown."""

    def test_missing_drawdown_produces_flag(self):
        """When drawdown is None, drawdown_data_missing is in risk_flags."""
        rec = _rec("DD_MISS", drawdown=None)
        fields = compute_decision_fields(rec, "drug_developer", 0.65)
        assert "drawdown_data_missing" in fields["risk_flags"]

    def test_present_drawdown_no_missing_flag(self):
        """When drawdown is present, drawdown_data_missing is NOT in risk_flags."""
        rec = _rec("DD_PRESENT", drawdown=-0.15)
        fields = compute_decision_fields(rec, "drug_developer", 0.65)
        assert "drawdown_data_missing" not in fields["risk_flags"]

    def test_deep_drawdown_still_fires(self):
        """deep_drawdown flag still fires when value is below threshold."""
        rec = _rec("DD_DEEP", drawdown=-0.55)
        fields = compute_decision_fields(rec, "drug_developer", 0.65)
        assert "deep_drawdown" in fields["risk_flags"]
        assert "drawdown_data_missing" not in fields["risk_flags"]

    def test_missing_and_deep_mutually_exclusive(self):
        """drawdown_data_missing and deep_drawdown never co-occur."""
        # Missing
        f1 = compute_decision_fields(_rec("A", drawdown=None), "drug_developer", 0.5)
        flags1 = f1["risk_flags"].split("|") if f1["risk_flags"] else []
        assert not ("drawdown_data_missing" in flags1 and "deep_drawdown" in flags1)
        # Deep
        f2 = compute_decision_fields(_rec("B", drawdown=-0.55), "drug_developer", 0.5)
        flags2 = f2["risk_flags"].split("|") if f2["risk_flags"] else []
        assert not ("drawdown_data_missing" in flags2 and "deep_drawdown" in flags2)


# =============================================================================
# 9. CATALYST STRENGTH BANDS
# =============================================================================


class TestCatalystStrengthBands:
    """Tests for the catalyst_strength band computation and tier/sizing effects."""

    def test_near_within_near_days(self):
        """days=30 with default near_days=90 → strength=near."""
        rec = _rec("NEAR", catalyst_days=30, catalyst_in_window=True)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "near"

    def test_mid_between_near_and_mid_days(self):
        """days=150 with near_days=90, mid_days=180 → strength=mid."""
        rec = _rec("MID", catalyst_days=150, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "mid"

    def test_far_beyond_mid_days(self):
        """days=250 with mid_days=180 → strength=far."""
        rec = _rec("FAR", catalyst_days=250, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "far"

    def test_blended_window_always_near(self):
        """Blended window mode → strength=near regardless of days."""
        rec = _rec("BLEND", catalyst_days=0, catalyst_in_window=True)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "near"

    def test_missing_mode_is_missing_strength(self):
        """No catalyst data → strength=missing."""
        rec = _rec("MISS")  # no catalyst args
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "missing"

    def test_no_upcoming_is_missing_strength(self):
        """no_upcoming mode → strength=missing."""
        rec = _rec("NOUP", catalyst_days=0, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["catalyst_strength"] == "missing"

    def test_near_qualifies_for_tier_a(self):
        """near + high_opt → A."""
        rec = _rec("A_NEAR", catalyst_days=30, catalyst_in_window=True)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["tier_dev"] == "A"

    def test_mid_qualifies_for_tier_a(self):
        """mid + high_opt → A."""
        rec = _rec("A_MID", catalyst_days=150, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["tier_dev"] == "A"

    def test_far_excluded_from_tier_a(self):
        """far + high_opt → B (not A)."""
        rec = _rec("B_FAR", catalyst_days=250, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert f["tier_dev"] == "B"
        assert "catalyst_far" in f["tier_reason"]

    def test_far_gives_sizing_lift(self):
        """FAR strength → catalyst_far_lift in size_reasons."""
        rec = _rec("FAR_LIFT", catalyst_days=250, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75)
        assert "catalyst_far_lift" in f["size_reasons"]

    def test_boundary_near_exact_equals(self):
        """days == near_days → near (inclusive)."""
        rs = DecisionRuleset(catalyst_near_days=90, catalyst_mid_days=180)
        rec = _rec("BOUND_NEAR", catalyst_days=90, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert f["catalyst_strength"] == "near"

    def test_boundary_mid_exact_equals(self):
        """days == mid_days → mid (inclusive)."""
        rs = DecisionRuleset(catalyst_near_days=90, catalyst_mid_days=180)
        rec = _rec("BOUND_MID", catalyst_days=180, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert f["catalyst_strength"] == "mid"

    def test_custom_ruleset_boundaries(self):
        """Custom near=60, mid=120: days=90 → mid."""
        rs = DecisionRuleset(catalyst_near_days=60, catalyst_mid_days=120)
        rec = _rec("CUSTOM", catalyst_days=90, catalyst_in_window=False)
        f = compute_decision_fields(rec, "drug_developer", 0.75, ruleset=rs)
        assert f["catalyst_strength"] == "mid"
        # And days=130 → far
        rec2 = _rec("CUSTOM2", catalyst_days=130, catalyst_in_window=False)
        f2 = compute_decision_fields(rec2, "drug_developer", 0.75, ruleset=rs)
        assert f2["catalyst_strength"] == "far"


# =============================================================================
# 10. SNAPSHOT INPUT COLUMNS — lock de_* contract surface
# =============================================================================


class TestSnapshotInputContract:
    """Prevent silent addition, removal, or renaming of de_* input columns."""

    EXPECTED_DE_INPUT_COLUMNS = frozenset(
        {
            "de_catalyst_days",
            "de_catalyst_in_window",
            "de_catalyst_mode",
            "de_alpha_60d",
            "de_alpha_60d_source",
            "de_alpha_60d_missing_reason",
            "de_tier1_count",
            "de_beta_xbi_60d",
            "de_beta_xbi_60d_source",
            "de_beta_xbi_60d_missing_reason",
            "de_drawdown",
            "de_drawdown_missing_reason",
            "de_rsi_14d",
            "de_vol_60d",
            "de_drawdown_xbi",
            "de_drawdown_rel_xbi",
            # Sort contribution diagnostics (populated at sort time, not by DE)
            "de_sort_total_adj",
            "de_sort_contrib_clinical",
            "de_sort_contrib_coinvest",
            "de_sort_contrib_institutional",
            "de_sort_contrib_calendar_alpha",
            "de_sort_contrib_alpha_cohort_tb",
            "de_sort_contrib_catalyst_bonus",
            "de_sort_contrib_binary_quality",
            "de_sort_contrib_binary_institutional",
            "de_sort_contrib_clinical_quality_91_180",
            "de_sort_contrib_options_quality_91_180",
            "de_sort_contrib_binary_quality_now",
            "de_sort_contrib_binary_institutional_now",
            "de_sort_contrib_clinical_build_window",
            "de_sort_contrib_pcr_penalty_bw",
            "de_sort_contrib_oncology_crowding",
            "de_sort_contrib_options_verdict",
            "de_sort_contrib_event_ev",
        }
    )

    def test_de_input_columns_present(self):
        """de_* columns in SNAPSHOT_COLUMNS match the pinned set exactly."""
        from run_screen import SNAPSHOT_COLUMNS

        actual_de = frozenset(c for c in SNAPSHOT_COLUMNS if c.startswith("de_"))
        assert actual_de == self.EXPECTED_DE_INPUT_COLUMNS, (
            f"de_* column mismatch.\n"
            f"  Added:   {actual_de - self.EXPECTED_DE_INPUT_COLUMNS}\n"
            f"  Removed: {self.EXPECTED_DE_INPUT_COLUMNS - actual_de}"
        )

    def test_drawdown_missing_reason_values(self):
        """drawdown_missing_reason set to a valid enum value when drawdown is None."""
        from run_screen import VALID_DRAWDOWN_MISSING_REASONS

        rec = _rec("DD_ENUM", drawdown=None)
        compute_decision_fields(rec, "drug_developer", 0.65)
        # The engine doesn't write drawdown_missing_reason itself (that's
        # _hydrate_drawdown's job), but it propagates the de_drawdown and
        # de_drawdown_missing_reason inputs.  Verify the enum is importable
        # and contains the three documented values.
        assert VALID_DRAWDOWN_MISSING_REASONS == frozenset({"", "no_price_series", "series_too_short"})

    def test_drawdown_price_symbol_not_persisted(self):
        """de_drawdown_price_symbol must NOT appear in SNAPSHOT_COLUMNS.

        It is an intermediate diagnostic field only (zero impact on current
        universe — all 353 tickers are clean alphanumeric).
        """
        from run_screen import SNAPSHOT_COLUMNS

        assert "de_drawdown_price_symbol" not in SNAPSHOT_COLUMNS


# =============================================================================
# 10. GOLDEN OUTPUT FINGERPRINT — CI gate for logic changes
# =============================================================================

# Fields already pinned by separate tests (version string, ruleset ID).
_FINGERPRINT_EXCLUDED = {"decision_engine_version", "decision_engine_ruleset_id"}


def _compute_golden_output_fingerprint() -> str:
    """Hash all golden fixture outputs + full pipeline to detect logic changes.

    Returns SHA256[:12] covering:
      - Individual compute_decision_fields output for each golden fixture
      - Full sort + weight pipeline output (same cohort as TestFullPipeline)

    Excluded: decision_engine_version and decision_engine_ruleset_id (pinned
    separately by TestSchemaContract and TestRulesetDriftGuardrails).
    """
    hasher = hashlib.sha256()

    # Part 1: individual golden outputs (stable fixture-list order)
    all_fields = [_compute_golden(g) for g in ALL_GOLDEN]
    for fields in all_fields:
        for key in sorted(fields.keys()):
            if key in _FINGERPRINT_EXCLUDED:
                continue
            hasher.update(f"{key}={fields[key]}".encode())

    # Part 2: full pipeline (sort + weight)
    rows = []
    for i, g in enumerate(ALL_GOLDEN):
        fields = _compute_golden(g)
        fields["ticker"] = g["ticker"]
        fields["archetype"] = g["archetype"]
        fields["optionality"] = g["optionality"]
        fields["composite_rank"] = i + 1
        rows.append(fields)

    rows.sort(
        key=lambda r: compute_actionable_sort_key(
            decision_fields=r,
            archetype=r["archetype"],
            optionality=r["optionality"],
            composite_rank=r["composite_rank"],
            ticker=r["ticker"],
        )
    )

    rank = 0
    for r in rows:
        if r["eligible"] == "1":
            rank += 1
            r["actionable_rank"] = rank
        else:
            r["actionable_rank"] = ""

    eligible = [r for r in rows if r["eligible"] == "1"]
    compute_target_weights(eligible)

    for r in rows:
        for key in sorted(r.keys()):
            if key in _FINGERPRINT_EXCLUDED:
                continue
            hasher.update(f"{key}={r[key]}".encode())

    return hasher.hexdigest()[:12]


class TestGoldenOutputFingerprint:
    """CI gate: detects ANY logic change that alters golden fixture outputs.

    The ruleset_id only hashes parameters — it cannot detect code changes
    (e.g. >= → > in a tier boundary, sort-key tweak, weight formula change).
    This fingerprint hashes all output fields across all 10 golden fixtures
    plus the full sort+weight pipeline.

    If this test fails:
      1. Verify the logic change is intentional
      2. Update EXPECTED_FINGERPRINT below
      3. Add an entry to RULESET_CHANGELOG.md
      4. Bump VERSION in decision_engine.py if the change is material
    """

    EXPECTED_FINGERPRINT = "d9cd64b3a321"  # pragma: allowlist secret

    def test_golden_output_fingerprint_pinned(self):
        actual = _compute_golden_output_fingerprint()
        assert actual == self.EXPECTED_FINGERPRINT, (
            f"Golden output fingerprint changed: {self.EXPECTED_FINGERPRINT} → {actual}.\n"
            f"A logic change in decision_engine.py altered fixture outputs.\n"
            f"If intentional: update EXPECTED_FINGERPRINT, add RULESET_CHANGELOG.md entry, "
            f"bump VERSION if material."
        )


# =============================================================================
# PHASE-2 PINNED RULESET ID SYNC GUARDRAIL
# =============================================================================


class TestPhase2PinnedIdSync:
    """Verify run_screen.py and run_phase2_snapshot_delta.py pin the same ruleset ID.

    These two constants MUST stay in sync — a mismatch causes silent health-gate
    FAIL (ruleset_mismatch) on every screen run.
    """

    def test_pinned_ids_match(self):
        from run_phase2_snapshot_delta import PHASE2_PINNED_RULESET_ID as delta_id
        from run_screen import PHASE2_PINNED_RULESET_ID as screen_id

        assert screen_id == delta_id, (
            f"PHASE2_PINNED_RULESET_ID mismatch: "
            f"run_screen={screen_id}, run_phase2_snapshot_delta={delta_id}. "
            f"These MUST be identical."
        )


# =============================================================================
# WEIGHT NORMALIZATION EDGE CASES
# =============================================================================


class TestWeightNormalizationEdgeCases:
    """Edge cases for compute_target_weights — zero/near-zero/single-row."""

    def test_all_zero_multipliers(self):
        """All cost_mult=0 → every weight empty (degenerate case)."""
        rows = [
            {"size_band": "L", "ticker": "A", "cost_mult": 0.0, "catalyst_tilt_mult": 1.0, "mom_state_tilt_mult": 1.0},
            {"size_band": "M", "ticker": "B", "cost_mult": 0.0, "catalyst_tilt_mult": 1.0, "mom_state_tilt_mult": 1.0},
        ]
        result = compute_target_weights(rows)
        for r in result:
            assert r["target_weight_pct"] == ""

    def test_near_zero_total_treated_as_degenerate(self):
        """Tiny sizing weights that sum to near-zero → empty weights."""
        rs = DecisionRuleset(
            sizing_weights=(("L", 1e-15), ("M", 1e-15), ("S", 1e-15), ("XS", 1e-15)),
        )
        rows = [
            {"size_band": "L", "ticker": "A"},
            {"size_band": "M", "ticker": "B"},
        ]
        result = compute_target_weights(rows, ruleset=rs)
        for r in result:
            assert r["target_weight_pct"] == ""

    def test_single_eligible_row_gets_100_pct(self):
        """One eligible row → 100% weight."""
        rows = [{"size_band": "L", "ticker": "ONLY"}]
        result = compute_target_weights(rows)
        assert result[0]["target_weight_pct"] == 100.0

    def test_normal_weights_sum_to_100(self):
        """Normal case — weights sum to 100%."""
        rows = [
            {"size_band": "L", "ticker": "A"},
            {"size_band": "M", "ticker": "B"},
            {"size_band": "S", "ticker": "C"},
            {"size_band": "XS", "ticker": "D"},
        ]
        result = compute_target_weights(rows)
        total = sum(r["target_weight_pct"] for r in result)
        assert abs(total - 100.0) < 0.01

    def test_empty_rows_no_crash(self):
        """Empty rows list should return empty without IndexError."""
        result = compute_target_weights([])
        assert result == []
