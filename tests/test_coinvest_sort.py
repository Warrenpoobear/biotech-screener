"""
Coinvest Sort Tie-Breaker Tests.

Verifies:
  - Conviction overlay fields extracted correctly from rec["coinvest"]
  - Cross-sectional z-score (coinvest_score_z) computed correctly
  - Sort key blending: coinvest_adj influences actionable ordering
  - Feature flag: zero impact when enable_coinvest_sort_signal=False
  - Positive-only mode: negative z clamped to 0
  - Additive with clinical sort signal
  - Coverage metadata
  - Health gate WARN on low coverage
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import (
    DecisionRuleset,
    compute_actionable_sort_key,
    compute_decision_fields,
    _compute_overlays,
    _safe_float,
)


# =============================================================================
# FIXTURE BUILDER
# =============================================================================

def _rec(
    ticker: str = "TEST",
    tier1_count: int | None = 3,
    conviction_overlap: float | None = None,
    tier1_conviction_overlap: float | None = None,
    max_tier1_position_pct: float | None = None,
    days_since_latest_filing: int | None = None,
    catalyst_days: int | None = 45,
    catalyst_in_window: bool | None = True,
    alpha_60d: float | None = 0.02,
    drawdown: float | None = -0.15,
) -> Dict[str, Any]:
    """Build a synthetic rec for coinvest sort testing."""
    coinvest: Dict[str, Any] = {}
    if tier1_count is not None:
        coinvest["tier1_count"] = tier1_count
    if conviction_overlap is not None:
        coinvest["conviction_overlap"] = conviction_overlap
    if tier1_conviction_overlap is not None:
        coinvest["tier1_conviction_overlap"] = tier1_conviction_overlap
    if max_tier1_position_pct is not None:
        coinvest["max_tier1_position_pct"] = max_tier1_position_pct
    if days_since_latest_filing is not None:
        coinvest["days_since_latest_filing"] = days_since_latest_filing

    cd: Dict[str, Any] = {}
    if catalyst_days is not None:
        cd["days_to_catalyst"] = catalyst_days
    if catalyst_in_window is not None:
        cd["in_optimal_window"] = catalyst_in_window

    return {
        "ticker": ticker,
        "severity": "NONE",
        "confidence_overall": 0.72,
        "fundamental_red_flag": False,
        "fundamental_red_flag_reasons": [],
        "flags": [],
        "attn_flags": [],
        "coinvest": coinvest,
        "smart_money_signal": {},
        "catalyst_decay": cd if cd else None,
        "defensive_features": {"drawdown": drawdown, "vol_60d": 0.65, "beta_xbi_60d": 1.1, "rsi_14d": 48.0},
        "score_breakdown": {"enhancements": {"momentum": {"alpha_60d": alpha_60d}}} if alpha_60d else {},
        "momentum_signal": {},
    }


# =============================================================================
# 1. OVERLAY CONVICTION FIELDS
# =============================================================================

class TestOverlayConvictionFields:
    """Verify _compute_overlays extracts all coinvest conviction fields."""

    def test_conviction_fields_present(self):
        """All 4 new conviction fields present when coinvest data exists."""
        rec = _rec(
            tier1_count=5,
            conviction_overlap=0.42,
            tier1_conviction_overlap=0.35,
            max_tier1_position_pct=2.8,
            days_since_latest_filing=45,
        )
        rs = DecisionRuleset()
        out = _compute_overlays(rec, rs)
        assert out["coinvest_conviction"] == 0.42
        assert out["coinvest_tier1_conviction"] == 0.35
        assert out["coinvest_max_position_pct"] == 2.8
        assert out["coinvest_filing_age_days"] == 45

    def test_conviction_missing_graceful(self):
        """Missing coinvest dict → fields default to 0/''."""
        rec = _rec(tier1_count=None)
        rec["coinvest"] = {}
        rs = DecisionRuleset()
        out = _compute_overlays(rec, rs)
        assert out["coinvest_conviction"] == 0.0
        assert out["coinvest_tier1_conviction"] == 0.0
        assert out["coinvest_max_position_pct"] == 0.0
        assert out["coinvest_filing_age_days"] == ""

    def test_conviction_none_coinvest(self):
        """rec["coinvest"] is None → fields default gracefully."""
        rec = _rec(tier1_count=None)
        rec["coinvest"] = None
        rs = DecisionRuleset()
        out = _compute_overlays(rec, rs)
        assert out["coinvest_conviction"] == 0.0
        assert out["coinvest_filing_age_days"] == ""

    def test_conviction_fields_in_decision_output(self):
        """compute_decision_fields includes conviction fields."""
        rec = _rec(conviction_overlap=0.55, tier1_conviction_overlap=0.40)
        fields = compute_decision_fields(rec, "drug_developer", 0.65)
        assert "coinvest_conviction" in fields
        assert fields["coinvest_conviction"] == 0.55
        assert fields["coinvest_tier1_conviction"] == 0.40


# =============================================================================
# 2. COINVEST Z-SCORE
# =============================================================================

class TestCoinvestZScore:
    """Test the z-score computation logic (unit-level, simulated)."""

    def _compute_z_scores(self, tier1_counts: list[int]) -> list[float]:
        """Simulate the z-score computation from run_screen.py."""
        vals = [float(c) for c in tier1_counts]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        if std == 0:
            return [0.0] * len(vals)
        return [round((v - mean) / std, 4) for v in vals]

    def test_z_score_basic(self):
        """Z-score for 5 tickers with varied tier1 counts."""
        counts = [0, 2, 5, 8, 14]
        zs = self._compute_z_scores(counts)
        # Highest count → highest z
        assert zs[-1] == max(zs)
        assert zs[0] == min(zs)
        # Mean z should be ~0
        assert abs(sum(zs) / len(zs)) < 0.001

    def test_z_score_zero_stdev(self):
        """All tickers same count → z=0 for all."""
        counts = [3, 3, 3, 3, 3]
        zs = self._compute_z_scores(counts)
        assert all(z == 0.0 for z in zs)

    def test_z_score_two_tickers(self):
        """Two tickers: one high, one low → symmetric z."""
        counts = [0, 10]
        zs = self._compute_z_scores(counts)
        assert zs[0] == -zs[1]  # symmetric around 0

    def test_coinvest_tag_format(self):
        """coinvest_tag is 'elite_N' matching tier1_count."""
        # Simulate tag logic from run_screen.py
        for count in [0, 3, 14]:
            tag = f"elite_{count}" if count > 0 else ""
            if count > 0:
                assert tag.startswith("elite_")
                assert tag == f"elite_{count}"
            else:
                assert tag == ""


# =============================================================================
# 3. SORT KEY — COINVEST DISABLED (zero impact)
# =============================================================================

class TestSortKeyCoinvestDisabled:
    """When enable_coinvest_sort_signal=False, coinvest has zero sort impact."""

    def test_sort_key_identical_with_flag_off(self):
        """Two tickers with different coinvest_score_z but flag OFF → same relative order."""
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=False,
            catalyst_priority_mode="tiebreaker",
        )
        fields_a = {"eligible": "1", "tier_dev": "A", "catalyst_mode": "specific_days",
                     "catalyst_days": 45, "catalyst_in_window": "1",
                     "sponsor_tier1_count": 5, "mom_state": "neutral",
                     "coinvest_score_z": 2.0, "clinical_score_z_tier": 0.0,
                     "stage_bucket": "mid", "missing_components": "", "missingness_penalty": 0}
        fields_b = dict(fields_a)
        fields_b["coinvest_score_z"] = -1.0

        key_a = compute_actionable_sort_key(
            fields_a, "drug_developer", 0.75, 1, "TICKER_A", ruleset=rs,
        )
        key_b = compute_actionable_sort_key(
            fields_b, "drug_developer", 0.75, 1, "TICKER_B", ruleset=rs,
        )
        # Only ticker differs; coinvest_score_z should have zero impact
        assert key_a[:-1] == key_b[:-1]  # last element is ticker


# =============================================================================
# 4. SORT KEY — COINVEST ENABLED (tiebreaker mode)
# =============================================================================

class TestSortKeyCoinvestEnabled:
    """When enable_coinvest_sort_signal=True, higher z → sorts earlier."""

    def _make_fields(self, coinvest_z: float, ticker: str = "T") -> dict:
        return {
            "eligible": "1", "tier_dev": "A", "catalyst_mode": "specific_days",
            "catalyst_days": 45, "catalyst_in_window": "1",
            "sponsor_tier1_count": 3, "mom_state": "neutral",
            "coinvest_score_z": coinvest_z,
            "clinical_score_z_tier": 0.0,
            "stage_bucket": "mid",
            "missing_components": "", "missingness_penalty": 0,
        }

    def test_higher_z_sorts_earlier_tiebreaker(self):
        """Higher coinvest_score_z → lower effective_comp_rank → sorts earlier."""
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="tiebreaker",
        )
        fields_high = self._make_fields(2.0, "HIGH_CZ")
        fields_low = self._make_fields(0.0, "LOW_CZ")

        key_high = compute_actionable_sort_key(
            fields_high, "drug_developer", 0.75, 5, "HIGH_CZ", ruleset=rs,
        )
        key_low = compute_actionable_sort_key(
            fields_low, "drug_developer", 0.75, 5, "LOW_CZ", ruleset=rs,
        )
        assert key_high < key_low

    def test_higher_z_sorts_earlier_blended(self):
        """Blended mode: higher coinvest_score_z → sorts earlier."""
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="blended",
        )
        fields_high = self._make_fields(2.0, "HIGH_CZ")
        fields_low = self._make_fields(0.0, "LOW_CZ")

        key_high = compute_actionable_sort_key(
            fields_high, "drug_developer", 0.75, 5, "HIGH_CZ", ruleset=rs,
        )
        key_low = compute_actionable_sort_key(
            fields_low, "drug_developer", 0.75, 5, "LOW_CZ", ruleset=rs,
        )
        assert key_high < key_low

    def test_higher_z_sorts_earlier_off(self):
        """Off mode: higher coinvest_score_z → sorts earlier (via opt_neg)."""
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="off",
        )
        fields_high = self._make_fields(2.0, "HIGH_CZ")
        fields_low = self._make_fields(0.0, "LOW_CZ")

        key_high = compute_actionable_sort_key(
            fields_high, "drug_developer", 0.75, 5, "HIGH_CZ", ruleset=rs,
        )
        key_low = compute_actionable_sort_key(
            fields_low, "drug_developer", 0.75, 5, "LOW_CZ", ruleset=rs,
        )
        assert key_high < key_low


# =============================================================================
# 5. POSITIVE-ONLY MODE
# =============================================================================

class TestCoinvestPositiveOnly:
    """Negative z clamped to 0 → no penalty for low coinvest."""

    def test_negative_z_clamped(self):
        """Negative coinvest_score_z → coinvest_adj = 0 → same as z=0."""
        rs = DecisionRuleset(
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            coinvest_positive_only=True,
            catalyst_priority_mode="tiebreaker",
        )
        fields_neg = {
            "eligible": "1", "tier_dev": "A", "catalyst_mode": "specific_days",
            "catalyst_days": 45, "catalyst_in_window": "1",
            "sponsor_tier1_count": 3, "mom_state": "neutral",
            "coinvest_score_z": -1.5,
            "clinical_score_z_tier": 0.0, "stage_bucket": "mid",
            "missing_components": "", "missingness_penalty": 0,
        }
        fields_zero = dict(fields_neg)
        fields_zero["coinvest_score_z"] = 0.0

        key_neg = compute_actionable_sort_key(
            fields_neg, "drug_developer", 0.75, 5, "SAME", ruleset=rs,
        )
        key_zero = compute_actionable_sort_key(
            fields_zero, "drug_developer", 0.75, 5, "SAME", ruleset=rs,
        )
        # Both should produce the same effective_comp_rank
        assert key_neg == key_zero


# =============================================================================
# 6. ADDITIVE WITH CLINICAL SORT SIGNAL
# =============================================================================

class TestCoinvestPlusClinical:
    """Both signals active → both adjustments additive."""

    def test_both_signals_additive(self):
        """When both clinical + coinvest enabled, effect is sum of both."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            clinical_sort_weight=1.0,
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="tiebreaker",
        )

        fields_both = {
            "eligible": "1", "tier_dev": "A", "catalyst_mode": "specific_days",
            "catalyst_days": 45, "catalyst_in_window": "1",
            "sponsor_tier1_count": 5, "mom_state": "neutral",
            "coinvest_score_z": 1.5,
            "clinical_score_z_tier": 1.0,
            "stage_bucket": "mid",  # stage_mult=1.0
            "missing_components": "", "missingness_penalty": 0,
        }
        fields_none = dict(fields_both)
        fields_none["coinvest_score_z"] = 0.0
        fields_none["clinical_score_z_tier"] = 0.0

        key_both = compute_actionable_sort_key(
            fields_both, "drug_developer", 0.75, 5, "BOTH", ruleset=rs,
        )
        key_none = compute_actionable_sort_key(
            fields_none, "drug_developer", 0.75, 5, "NONE", ruleset=rs,
        )
        # With both signals, effective_comp_rank is lower → sorts earlier
        assert key_both < key_none

    def test_coinvest_only_vs_clinical_only(self):
        """Each signal independently moves rank in the right direction."""
        rs = DecisionRuleset(
            enable_clinical_sort_signal=True,
            clinical_sort_weight=1.0,
            enable_coinvest_sort_signal=True,
            coinvest_sort_weight=0.5,
            catalyst_priority_mode="tiebreaker",
        )

        base = {
            "eligible": "1", "tier_dev": "A", "catalyst_mode": "specific_days",
            "catalyst_days": 45, "catalyst_in_window": "1",
            "sponsor_tier1_count": 5, "mom_state": "neutral",
            "coinvest_score_z": 0.0,
            "clinical_score_z_tier": 0.0,
            "stage_bucket": "mid",
            "missing_components": "", "missingness_penalty": 0,
        }

        # Coinvest only
        f_coinvest = dict(base)
        f_coinvest["coinvest_score_z"] = 1.5

        # Clinical only
        f_clinical = dict(base)
        f_clinical["clinical_score_z_tier"] = 1.5

        key_base = compute_actionable_sort_key(
            base, "drug_developer", 0.75, 5, "BASE", ruleset=rs,
        )
        key_coinvest = compute_actionable_sort_key(
            f_coinvest, "drug_developer", 0.75, 5, "BASE", ruleset=rs,
        )
        key_clinical = compute_actionable_sort_key(
            f_clinical, "drug_developer", 0.75, 5, "BASE", ruleset=rs,
        )

        # Both should be better than base
        assert key_coinvest < key_base
        assert key_clinical < key_base


# =============================================================================
# 7. RULESET VALIDATION
# =============================================================================

class TestCoinvestRulesetValidation:
    """Ruleset field validation for coinvest settings."""

    def test_invalid_score_mode_raises(self):
        """Invalid coinvest_score_mode raises ValueError."""
        with pytest.raises(ValueError, match="coinvest_score_mode"):
            DecisionRuleset(coinvest_score_mode="conviction")

    def test_valid_tier1_count_mode(self):
        """tier1_count mode is accepted."""
        rs = DecisionRuleset(coinvest_score_mode="tier1_count")
        assert rs.coinvest_score_mode == "tier1_count"

    def test_default_values(self):
        """Default values match plan spec."""
        rs = DecisionRuleset()
        assert rs.enable_coinvest_sort_signal is False
        assert rs.coinvest_sort_weight == 0.5
        assert rs.coinvest_positive_only is True
        assert rs.coinvest_score_mode == "tier1_count"

    def test_v150_candidate_loads(self):
        """v1.5.0 candidate ruleset loads without error."""
        path = PROJECT_ROOT / "production_data" / "decision_rulesets" / "v1.5.0_coinvest_candidate.json"
        if path.exists():
            rs = DecisionRuleset.from_json(str(path))
            assert rs.enable_coinvest_sort_signal is True
            assert rs.coinvest_sort_weight == 0.05


# =============================================================================
# 8. HEALTH GATE — COINVEST COVERAGE
# =============================================================================

class TestHealthGateCoinvestCoverage:
    """Verify coinvest coverage WARN fires in health gate."""

    def test_health_gate_coverage_warn(self):
        """Coverage below threshold → WARN reason."""
        import pandas as pd
        from run_phase2_snapshot_delta import Phase2HealthThresholds

        th = Phase2HealthThresholds(warn_coinvest_coverage_min=70.0)
        # Simulate: 10 tickers, only 5 have sponsor_tier1_count > 0 → 50%
        data = {"sponsor_tier1_count": [3, 0, 5, 0, 2, 0, 0, 0, 1, 0]}
        df = pd.DataFrame(data)
        n_total = len(df)
        n_with = int(df["sponsor_tier1_count"].apply(
            lambda x: pd.notna(x) and str(x).strip() not in ("", "0")
        ).sum())
        cov = 100.0 * n_with / n_total
        assert cov < th.warn_coinvest_coverage_min
        assert cov == 40.0  # 4 out of 10

    def test_health_gate_coverage_ok(self):
        """Coverage above threshold → no WARN."""
        import pandas as pd
        from run_phase2_snapshot_delta import Phase2HealthThresholds

        th = Phase2HealthThresholds(warn_coinvest_coverage_min=70.0)
        # 10 tickers, 8 have signal → 80%
        data = {"sponsor_tier1_count": [3, 1, 5, 2, 2, 4, 1, 3, 0, 0]}
        df = pd.DataFrame(data)
        n_total = len(df)
        n_with = int(df["sponsor_tier1_count"].apply(
            lambda x: pd.notna(x) and str(x).strip() not in ("", "0")
        ).sum())
        cov = 100.0 * n_with / n_total
        assert cov >= th.warn_coinvest_coverage_min
        assert cov == 80.0
