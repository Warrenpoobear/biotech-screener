"""
Golden-record tests for decision_engine.compute_decision_fields().

Tests 7 scenarios using synthetic recs to verify:
  1. specific_days catalyst mode → Tier A with high optionality
  2. blended_window mode (days=0, in_window=True)
  3. no_upcoming mode (days=0, in_window=False)
  4. missing catalyst mode (days=None, in_window=None)
  5. High vol + deep drawdown → risk flags + size band demotion
  6. SEV3 → ineligible
  7. Non-drug_developer archetype → blank tier_dev
"""
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import compute_decision_fields


# =============================================================================
# HELPER: Build a minimal rec dict for testing
# =============================================================================

def _make_rec(
    severity="",
    confidence_overall=0.5,
    fundamental_red_flag=False,
    catalyst_days=None,
    catalyst_in_window=None,
    vol_60d=None,
    beta_xbi_60d=None,
    drawdown=None,
    rsi_14d=None,
    alpha_60d=None,
    tier1_count=None,
    tier1_holders=0,
    holders_inc=None,
    holders_dec=None,
):
    """Build a synthetic rec dict for decision engine testing."""
    rec = {
        "severity": severity,
        "confidence_overall": confidence_overall,
        "fundamental_red_flag": fundamental_red_flag,
        "attn_flags": [],
        "flags": [],
    }

    # Catalyst
    cd = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window
    rec["catalyst_decay"] = cd

    # Smart money signal
    sms = {"tier1_holders": tier1_holders}
    if holders_inc is not None:
        sms["holders_increasing"] = holders_inc
    if holders_dec is not None:
        sms["holders_decreasing"] = holders_dec
    rec["smart_money_signal"] = sms

    # Coinvest
    if tier1_count is not None:
        rec["coinvest"] = {"tier1_count": tier1_count}
    else:
        rec["coinvest"] = {}

    # Defensive features
    df = {}
    if vol_60d is not None:
        df["vol_60d"] = vol_60d
    if beta_xbi_60d is not None:
        df["beta_xbi_60d"] = beta_xbi_60d
    if drawdown is not None:
        df["drawdown"] = drawdown
    if rsi_14d is not None:
        df["rsi_14d"] = rsi_14d
    rec["defensive_features"] = df

    # Momentum
    if alpha_60d is not None:
        rec["score_breakdown"] = {
            "enhancements": {"momentum": {"alpha_60d": alpha_60d}}
        }
    else:
        rec["score_breakdown"] = {}
    rec["momentum_signal"] = {}

    return rec


# =============================================================================
# TEST 1: specific_days catalyst → Tier A
# =============================================================================

class TestSpecificDaysCatalyst:
    """Ticker with specific days_to_catalyst=45, in_window=True, high optionality."""

    def setup_method(self):
        self.rec = _make_rec(
            catalyst_days=45,
            catalyst_in_window=True,
            alpha_60d=0.08,
            tier1_count=3,
        )
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.75
        )

    def test_eligible(self):
        assert self.fields["eligible"] == "1"

    def test_catalyst_mode(self):
        assert self.fields["catalyst_mode"] == "specific_days"

    def test_catalyst_days(self):
        assert self.fields["catalyst_days"] == 45

    def test_catalyst_in_window(self):
        assert self.fields["catalyst_in_window"] == "1"

    def test_tier_a(self):
        assert self.fields["tier_dev"] == "A"

    def test_tier_reason(self):
        assert "high_opt" in self.fields["tier_reason"]
        assert "catalyst_near" in self.fields["tier_reason"]

    def test_mom_tailwind(self):
        assert self.fields["mom_state"] == "tailwind"

    def test_sponsor_tier1(self):
        assert self.fields["sponsor_tier1_count"] == 3


# =============================================================================
# TEST 2: blended_window catalyst mode
# =============================================================================

class TestBlendedWindowCatalyst:
    """days=0, in_window=True → blended_window mode."""

    def setup_method(self):
        self.rec = _make_rec(catalyst_days=0, catalyst_in_window=True)
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.65
        )

    def test_catalyst_mode(self):
        assert self.fields["catalyst_mode"] == "blended_window"

    def test_catalyst_days_zero(self):
        assert self.fields["catalyst_days"] == 0

    def test_tier_at_least_b(self):
        # High optionality + blended window → should get A or B
        assert self.fields["tier_dev"] in ("A", "B")


# =============================================================================
# TEST 3: no_upcoming catalyst mode
# =============================================================================

class TestNoUpcomingCatalyst:
    """days=0, in_window=False → no_upcoming mode."""

    def setup_method(self):
        self.rec = _make_rec(catalyst_days=0, catalyst_in_window=False)
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.50
        )

    def test_catalyst_mode(self):
        assert self.fields["catalyst_mode"] == "no_upcoming"

    def test_eligible(self):
        assert self.fields["eligible"] == "1"


# =============================================================================
# TEST 4: missing catalyst mode
# =============================================================================

class TestMissingCatalyst:
    """No catalyst data at all → missing mode."""

    def setup_method(self):
        self.rec = _make_rec()  # No catalyst_days, no in_window
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.70
        )

    def test_catalyst_mode(self):
        assert self.fields["catalyst_mode"] == "missing"

    def test_tier_b_without_catalyst(self):
        # High optionality but no catalyst data → B (not A)
        assert self.fields["tier_dev"] == "B"

    def test_tier_reason_no_catalyst(self):
        assert "no_catalyst_data" in self.fields["tier_reason"]


# =============================================================================
# TEST 5: High risk → flags + size demotion
# =============================================================================

class TestHighRiskFlags:
    """High vol + deep drawdown → risk flags and size band demotion."""

    def setup_method(self):
        self.rec = _make_rec(
            vol_60d=1.50,        # > 1.20 threshold
            beta_xbi_60d=2.0,   # > 1.80 threshold
            drawdown=-0.38,     # worse than -0.35 flag threshold
            rsi_14d=75,         # > 70 overbought
            alpha_60d=-0.10,    # headwind
        )
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.40
        )

    def test_eligible(self):
        # drawdown=-0.38 is above -0.40 gate, so still eligible
        assert self.fields["eligible"] == "1"

    def test_risk_flags_present(self):
        flags = self.fields["risk_flags"]
        assert "high_vol" in flags
        assert "high_beta" in flags
        assert "deep_drawdown" in flags
        assert "overbought_rsi" in flags

    def test_mom_headwind(self):
        assert self.fields["mom_state"] == "headwind"

    def test_size_band_demoted(self):
        # Multiple risk factors should push band down from default M
        assert self.fields["size_band"] in ("XS", "S")


# =============================================================================
# TEST 6: SEV3 → ineligible
# =============================================================================

class TestSev3Ineligible:
    """SEV3 severity triggers ineligibility gate."""

    def setup_method(self):
        self.rec = _make_rec(
            severity="SEV3",
            catalyst_days=30,
            catalyst_in_window=True,
        )
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.80
        )

    def test_ineligible(self):
        assert self.fields["eligible"] == "0"

    def test_ineligible_reasons(self):
        assert "sev3" in self.fields["ineligible_reasons"]

    def test_tier_d(self):
        assert self.fields["tier_dev"] == "D"

    def test_tier_reason_ineligible(self):
        assert self.fields["tier_reason"] == "ineligible"

    def test_size_xs(self):
        assert self.fields["size_band"] == "XS"


# =============================================================================
# TEST 7: Non-drug_developer → blank tier_dev
# =============================================================================

class TestNonDrugDeveloper:
    """Commercial archetype should have blank tier_dev."""

    def setup_method(self):
        self.rec = _make_rec(
            catalyst_days=30,
            catalyst_in_window=True,
            alpha_60d=0.06,
        )
        self.fields = compute_decision_fields(
            self.rec, archetype="commercial_biotech", optionality_pct_dev=0.80
        )

    def test_tier_dev_blank(self):
        assert self.fields["tier_dev"] == ""

    def test_tier_reason_blank(self):
        assert self.fields["tier_reason"] == ""

    def test_eligible(self):
        assert self.fields["eligible"] == "1"

    def test_catalyst_still_computed(self):
        assert self.fields["catalyst_mode"] == "specific_days"
        assert self.fields["catalyst_days"] == 30


# =============================================================================
# TEST: Deep drawdown gate (below -40%)
# =============================================================================

class TestDeepDrawdownGate:
    """Drawdown worse than -40% triggers eligibility gate."""

    def setup_method(self):
        self.rec = _make_rec(drawdown=-0.45)
        self.fields = compute_decision_fields(
            self.rec, archetype="drug_developer", optionality_pct_dev=0.80
        )

    def test_ineligible(self):
        assert self.fields["eligible"] == "0"

    def test_reason(self):
        assert "deep_drawdown" in self.fields["ineligible_reasons"]
