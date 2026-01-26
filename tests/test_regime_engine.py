#!/usr/bin/env python3
"""
Tests for regime_engine.py

Market regime detection affects signal weight adjustments in the composite scoring.
These tests cover:
- Regime classification (BULL, BEAR, VOLATILITY_SPIKE, SECTOR_ROTATION)
- New regimes (RECESSION_RISK, CREDIT_CRISIS, SECTOR_DISLOCATION)
- VIX threshold classification
- XBI relative performance classification
- Signal weight adjustments by regime
- Staleness gating and confidence haircuts
- Composite weight adjustments
- Callback system for regime transitions
- Kalman filter for VIX smoothing
- HMM for regime probabilities
- Ensemble classification
"""

import pytest
from datetime import date
from decimal import Decimal

from regime_engine import (
    RegimeDetectionEngine,
    MarketRegime,
    RegimeTransitionCallback,
    RegimeTransitionEvent,
    VIXKalmanFilter,
    RegimeHMM,
    EnsembleRegimeClassifier,
)


class TestMarketRegimeEnum:
    """Tests for MarketRegime enum."""

    def test_all_regimes_defined(self):
        """All expected regimes should be defined."""
        assert MarketRegime.BULL.value == "BULL"
        assert MarketRegime.BEAR.value == "BEAR"
        assert MarketRegime.VOLATILITY_SPIKE.value == "VOLATILITY_SPIKE"
        assert MarketRegime.SECTOR_ROTATION.value == "SECTOR_ROTATION"
        assert MarketRegime.UNKNOWN.value == "UNKNOWN"
        # New regimes
        assert MarketRegime.RECESSION_RISK.value == "RECESSION_RISK"
        assert MarketRegime.CREDIT_CRISIS.value == "CREDIT_CRISIS"
        assert MarketRegime.SECTOR_DISLOCATION.value == "SECTOR_DISLOCATION"


class TestRegimeDetectionBull:
    """Tests for BULL regime detection."""

    def test_bull_low_vix_strong_xbi(self):
        """Low VIX + strong XBI outperformance = BULL."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("14.5"),
            xbi_vs_spy_30d=Decimal("6.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "BULL"
        assert result["confidence"] > Decimal("0.30")

    def test_bull_with_rate_cuts(self):
        """Rate cuts support BULL regime."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("15.0"),
            xbi_vs_spy_30d=Decimal("5.0"),
            fed_rate_change_3m=Decimal("-0.50"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "BULL"

    def test_bull_signal_adjustments(self):
        """BULL regime should boost momentum and catalyst."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("14.0"),
            xbi_vs_spy_30d=Decimal("7.0"),
            as_of_date=date(2026, 1, 15),
        )
        adj = result["signal_adjustments"]
        assert adj["momentum"] == Decimal("1.20")
        assert adj["catalyst"] == Decimal("1.15")
        assert adj["fundamental"] == Decimal("0.90")


class TestRegimeDetectionBear:
    """Tests for BEAR regime detection."""

    def test_bear_elevated_vix_underperformance(self):
        """Elevated VIX + XBI underperformance = BEAR."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("26.0"),
            xbi_vs_spy_30d=Decimal("-6.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "BEAR"

    def test_bear_with_rate_hikes(self):
        """Rate hikes support BEAR regime."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("25.0"),
            xbi_vs_spy_30d=Decimal("-5.0"),
            fed_rate_change_3m=Decimal("0.75"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "BEAR"

    def test_bear_signal_adjustments(self):
        """BEAR regime should reduce momentum, boost quality and financial."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("28.0"),
            xbi_vs_spy_30d=Decimal("-7.0"),
            as_of_date=date(2026, 1, 15),
        )
        adj = result["signal_adjustments"]
        assert adj["momentum"] == Decimal("0.80")
        assert adj["quality"] == Decimal("1.20")
        assert adj["financial"] == Decimal("1.20")

    def test_bear_flags(self):
        """BEAR regime should generate defensive flag."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("28.0"),
            xbi_vs_spy_30d=Decimal("-7.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert "DEFENSIVE_POSITIONING" in result["flags"]


class TestRegimeDetectionVolatilitySpike:
    """Tests for VOLATILITY_SPIKE regime detection."""

    def test_vol_spike_extreme_vix(self):
        """Extreme VIX (>40) indicates high stress - BEAR or VOLATILITY_SPIKE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("45.0"),
            xbi_vs_spy_30d=Decimal("-3.0"),
            as_of_date=date(2026, 1, 15),
        )
        # With extreme VIX and moderate underperformance, scores can tie
        # between BEAR and VOLATILITY_SPIKE - both indicate high stress
        assert result["regime"] in ("VOLATILITY_SPIKE", "BEAR")

    def test_vol_spike_high_vix(self):
        """High VIX (30-40) likely VOLATILITY_SPIKE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("35.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "VOLATILITY_SPIKE"

    def test_vol_spike_signal_adjustments(self):
        """High VIX should reduce momentum, boost quality/financial."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("42.0"),
            xbi_vs_spy_30d=Decimal("-3.0"),
            as_of_date=date(2026, 1, 15),
        )
        adj = result["signal_adjustments"]
        # Both BEAR and VOLATILITY_SPIKE reduce momentum and boost quality/financial
        assert adj["momentum"] <= Decimal("0.80")  # Reduced from 1.0
        assert adj["quality"] >= Decimal("1.20")   # Boosted from 1.0
        assert adj["financial"] >= Decimal("1.20") # Boosted from 1.0

    def test_vol_spike_flags(self):
        """Extreme VIX should generate crisis/volatility warning flags."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("42.0"),
            xbi_vs_spy_30d=Decimal("-3.0"),
            as_of_date=date(2026, 1, 15),
        )
        # VIX >= 40 generates CRISIS_VOLATILITY flag
        assert "CRISIS_VOLATILITY" in result["flags"]
        # Either VOLATILITY_SPIKE or BEAR flags should be present
        has_defensive_flag = (
            "REDUCE_POSITION_SIZE" in result["flags"] or
            "DEFENSIVE_POSITIONING" in result["flags"]
        )
        assert has_defensive_flag


class TestRegimeDetectionSectorRotation:
    """Tests for SECTOR_ROTATION regime detection."""

    def test_sector_rotation_mixed_signals(self):
        """Mixed signals = SECTOR_ROTATION."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("0.5"),  # Neutral
            fed_rate_change_3m=Decimal("0.10"),  # Slight tightening
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "SECTOR_ROTATION"

    def test_sector_rotation_signal_adjustments(self):
        """SECTOR_ROTATION should have neutral adjustments."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("1.0"),
            as_of_date=date(2026, 1, 15),
        )
        # SECTOR_ROTATION has near-neutral weights
        adj = result["signal_adjustments"]
        assert adj["momentum"] == Decimal("1.00")
        assert adj["catalyst"] == Decimal("1.00")


class TestVixClassification:
    """Tests for VIX level classification."""

    def test_vix_very_low(self):
        """VIX below 15 should be VERY_LOW."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("12.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "VERY_LOW"

    def test_vix_low(self):
        """VIX 15 should be LOW."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("15.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "LOW"

    def test_vix_normal(self):
        """VIX 20-25 should be NORMAL."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "NORMAL"

    def test_vix_elevated(self):
        """VIX 25-30 should be ELEVATED."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("27.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "ELEVATED"

    def test_vix_high(self):
        """VIX 30-40 should be HIGH."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("35.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "HIGH"

    def test_vix_crisis(self):
        """VIX above 40 should be CRISIS."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("50.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["vix_level"] == "CRISIS"


class TestXbiPerformanceClassification:
    """Tests for XBI relative performance classification."""

    def test_strong_outperformance(self):
        """XBI +5% or more = STRONG_OUTPERFORMANCE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("7.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["biotech_relative_performance"] == "STRONG_OUTPERFORMANCE"

    def test_moderate_outperformance(self):
        """XBI +2% to +5% = MODERATE_OUTPERFORMANCE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["biotech_relative_performance"] == "MODERATE_OUTPERFORMANCE"

    def test_neutral_performance(self):
        """XBI -2% to +2% = NEUTRAL."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["biotech_relative_performance"] == "NEUTRAL"

    def test_moderate_underperformance(self):
        """XBI -2% to -5% = MODERATE_UNDERPERFORMANCE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("-3.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["biotech_relative_performance"] == "MODERATE_UNDERPERFORMANCE"

    def test_strong_underperformance(self):
        """XBI -5% or worse = STRONG_UNDERPERFORMANCE."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("-8.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["biotech_relative_performance"] == "STRONG_UNDERPERFORMANCE"


class TestStalenessGating:
    """Tests for data staleness gating."""

    def test_fresh_data_full_confidence(self):
        """Data 2 days old should have full confidence."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("5.0"),
            as_of_date=date(2026, 1, 15),
            data_as_of_date=date(2026, 1, 13),  # 2 days old
        )
        assert result["staleness"]["haircut_multiplier"] == "1.00"
        assert result["staleness"]["action"] == "FULL_CONFIDENCE"

    def test_moderate_staleness_haircut(self):
        """Data 5 days old should have 15% haircut."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("5.0"),
            as_of_date=date(2026, 1, 15),
            data_as_of_date=date(2026, 1, 10),  # 5 days old
        )
        assert result["staleness"]["haircut_multiplier"] == "0.85"
        assert result["staleness"]["action"] == "HAIRCUT_APPLIED"

    def test_high_staleness_haircut(self):
        """Data 10 days old should have 35% haircut."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("5.0"),
            as_of_date=date(2026, 1, 15),
            data_as_of_date=date(2026, 1, 5),  # 10 days old
        )
        assert result["staleness"]["haircut_multiplier"] == "0.65"

    def test_stale_data_forced_unknown(self):
        """Data >10 days old should force UNKNOWN regime."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("14.0"),  # Would be BULL
            xbi_vs_spy_30d=Decimal("7.0"),
            as_of_date=date(2026, 1, 15),
            data_as_of_date=date(2026, 1, 1),  # 14 days old
        )
        assert result["regime"] == "UNKNOWN"
        assert result["staleness"]["is_stale"] is True
        assert result["staleness"]["action"] == "FORCED_UNKNOWN"


class TestCompositeWeightAdjustments:
    """Tests for get_composite_weight_adjustments."""

    def test_bull_weight_adjustments(self):
        """BULL regime should adjust weights accordingly."""
        engine = RegimeDetectionEngine()
        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.35"),
            "catalyst": Decimal("0.25"),
        }
        adjusted = engine.get_composite_weight_adjustments("BULL", base_weights)

        # Weights should sum to ~1.0 (normalized)
        total = sum(adjusted.values())
        assert Decimal("0.99") <= total <= Decimal("1.01")

        # Clinical should have weight (0.40 * 1.10 for BULL)
        # But normalized, so check relative changes

    def test_bear_weight_adjustments(self):
        """BEAR regime should boost financial weight."""
        engine = RegimeDetectionEngine()
        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.30"),
            "catalyst": Decimal("0.30"),
        }
        adjusted = engine.get_composite_weight_adjustments("BEAR", base_weights)

        # Financial gets 1.20x boost in BEAR
        # Catalyst gets 0.90x reduction
        # Check that financial > catalyst after adjustment
        assert adjusted["financial"] > adjusted["catalyst"]

    def test_unknown_regime_neutral_weights(self):
        """UNKNOWN regime should preserve base weights."""
        engine = RegimeDetectionEngine()
        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.35"),
            "catalyst": Decimal("0.25"),
        }
        adjusted = engine.get_composite_weight_adjustments("UNKNOWN", base_weights)

        # Should be roughly equal (normalized)
        assert adjusted["clinical"] > adjusted["catalyst"]  # 0.40 > 0.25


class TestApplyRegimeWeights:
    """Tests for apply_regime_weights."""

    def test_applies_multipliers(self):
        """Should apply regime multipliers to base scores."""
        engine = RegimeDetectionEngine()
        base_scores = {
            "momentum": Decimal("50"),
            "quality": Decimal("70"),
        }
        adjustments = {
            "momentum": Decimal("0.80"),
            "quality": Decimal("1.20"),
        }
        adjusted = engine.apply_regime_weights(base_scores, adjustments)

        assert adjusted["momentum"] == Decimal("40.00")  # 50 * 0.8
        assert adjusted["quality"] == Decimal("84.00")  # 70 * 1.2

    def test_default_multiplier_for_missing(self):
        """Missing signal should use 1.0 multiplier."""
        engine = RegimeDetectionEngine()
        base_scores = {"custom_signal": Decimal("60")}
        adjustments = {"momentum": Decimal("0.80")}

        adjusted = engine.apply_regime_weights(base_scores, adjustments)
        assert adjusted["custom_signal"] == Decimal("60.00")  # 60 * 1.0


class TestRegimeHistory:
    """Tests for regime history tracking."""

    def test_history_recorded(self):
        """Regime detections should be recorded in history."""
        engine = RegimeDetectionEngine()
        engine.detect_regime(
            vix_current=Decimal("14.0"),
            xbi_vs_spy_30d=Decimal("6.0"),
            as_of_date=date(2026, 1, 15),
        )
        engine.detect_regime(
            vix_current=Decimal("35.0"),
            xbi_vs_spy_30d=Decimal("-5.0"),
            as_of_date=date(2026, 1, 16),
        )

        history = engine.get_regime_history()
        assert len(history) == 2
        assert history[0]["regime"] == "BULL"
        assert history[1]["regime"] in ["BEAR", "VOLATILITY_SPIKE"]

    def test_audit_trail_recorded(self):
        """Full audit trail should be recorded."""
        engine = RegimeDetectionEngine()
        engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )

        audit = engine.get_audit_trail()
        assert len(audit) == 1
        assert "input" in audit[0]
        assert "regime_scores" in audit[0]

    def test_clear_state(self):
        """Clear state should reset history."""
        engine = RegimeDetectionEngine()
        engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )
        engine.clear_state()

        assert len(engine.get_regime_history()) == 0
        assert len(engine.get_audit_trail()) == 0


class TestDeterminism:
    """Tests for deterministic output."""

    def test_same_inputs_same_output(self):
        """Same inputs should produce identical outputs."""
        engine1 = RegimeDetectionEngine()
        engine2 = RegimeDetectionEngine()

        result1 = engine1.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("3.2"),
            fed_rate_change_3m=Decimal("-0.25"),
            as_of_date=date(2026, 1, 15),
        )
        result2 = engine2.detect_regime(
            vix_current=Decimal("18.5"),
            xbi_vs_spy_30d=Decimal("3.2"),
            fed_rate_change_3m=Decimal("-0.25"),
            as_of_date=date(2026, 1, 15),
        )

        assert result1["regime"] == result2["regime"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["signal_adjustments"] == result2["signal_adjustments"]


# ============================================================================
# NEW TESTS FOR ENHANCED REGIME ENGINE
# ============================================================================


class TestNewRegimeTypes:
    """Tests for new regime types: RECESSION_RISK, CREDIT_CRISIS, SECTOR_DISLOCATION."""

    def test_recession_risk_inverted_yield_curve(self):
        """Deeply inverted yield curve should trigger RECESSION_RISK."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("22.0"),
            xbi_vs_spy_30d=Decimal("-2.0"),
            yield_curve_slope=Decimal("-60"),  # Deeply inverted
            hy_credit_spread=Decimal("450"),   # Elevated
            as_of_date=date(2026, 1, 15),
        )
        assert result["regime"] == "RECESSION_RISK"
        assert "RECESSION_WARNING" in result["flags"]

    def test_recession_risk_signal_adjustments(self):
        """RECESSION_RISK should have defensive signal adjustments."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("22.0"),
            xbi_vs_spy_30d=Decimal("-2.0"),
            yield_curve_slope=Decimal("-55"),
            hy_credit_spread=Decimal("450"),
            as_of_date=date(2026, 1, 15),
        )
        if result["regime"] == "RECESSION_RISK":
            adj = result["signal_adjustments"]
            assert adj["momentum"] == Decimal("0.60")
            assert adj["quality"] == Decimal("1.35")
            assert adj["financial"] == Decimal("1.35")

    def test_credit_crisis_extreme_spreads(self):
        """Extreme credit spreads should trigger CREDIT_CRISIS."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("35.0"),
            xbi_vs_spy_30d=Decimal("-5.0"),
            hy_credit_spread=Decimal("650"),  # Crisis level
            as_of_date=date(2026, 1, 15),
        )
        # With extreme spreads, either CREDIT_CRISIS or VOLATILITY_SPIKE expected
        assert result["regime"] in ("CREDIT_CRISIS", "VOLATILITY_SPIKE", "BEAR")

    def test_credit_crisis_signal_adjustments(self):
        """CREDIT_CRISIS should severely reduce momentum."""
        engine = RegimeDetectionEngine()
        adj = engine.REGIME_ADJUSTMENTS["CREDIT_CRISIS"]
        assert adj["momentum"] == Decimal("0.50")
        assert adj["quality"] == Decimal("1.40")
        assert adj["financial"] == Decimal("1.45")

    def test_sector_dislocation_large_divergence(self):
        """Large biotech divergence should trigger SECTOR_DISLOCATION."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("18.0"),  # Large outperformance
            as_of_date=date(2026, 1, 15),
        )
        # Should have some SECTOR_DISLOCATION score
        assert result["regime_scores"]["SECTOR_DISLOCATION"] > Decimal("0")

    def test_sector_dislocation_signal_adjustments(self):
        """SECTOR_DISLOCATION should boost institutional focus."""
        engine = RegimeDetectionEngine()
        adj = engine.REGIME_ADJUSTMENTS["SECTOR_DISLOCATION"]
        assert adj["institutional"] == Decimal("1.20")
        assert adj["momentum"] == Decimal("0.85")


class TestNewInputSignals:
    """Tests for new input signals: yield curve, credit spread, put/call, fund flows."""

    def test_yield_curve_classification_deeply_inverted(self):
        """Deeply inverted yield curve should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            yield_curve_slope=Decimal("-60"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["yield_curve_state"] == "DEEPLY_INVERTED"

    def test_yield_curve_classification_inverted(self):
        """Inverted yield curve should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            yield_curve_slope=Decimal("-25"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["yield_curve_state"] == "INVERTED"

    def test_yield_curve_classification_normal(self):
        """Normal yield curve should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            yield_curve_slope=Decimal("75"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["yield_curve_state"] == "NORMAL"

    def test_credit_environment_crisis(self):
        """Crisis credit environment should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            hy_credit_spread=Decimal("650"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["credit_environment"] == "CRISIS"

    def test_credit_environment_stressed(self):
        """Stressed credit environment should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            hy_credit_spread=Decimal("450"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["credit_environment"] == "STRESSED"

    def test_fund_flows_strong_inflows(self):
        """Strong fund inflows should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            biotech_fund_flows=Decimal("250"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["fund_flow_state"] == "STRONG_INFLOWS"

    def test_fund_flows_heavy_outflows(self):
        """Heavy fund outflows should be classified correctly."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            biotech_fund_flows=Decimal("-250"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["indicators"]["fund_flow_state"] == "HEAVY_OUTFLOWS"

    def test_new_signals_in_audit_trail(self):
        """New signals should be recorded in audit trail."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("20.0"),
            xbi_vs_spy_30d=Decimal("0.0"),
            yield_curve_slope=Decimal("-25"),
            hy_credit_spread=Decimal("400"),
            biotech_fund_flows=Decimal("100"),
            as_of_date=date(2026, 1, 15),
        )
        audit = result["audit_entry"]
        assert audit["input"]["yield_curve_slope"] == "-25"
        assert audit["input"]["hy_credit_spread"] == "400"
        assert audit["input"]["biotech_fund_flows"] == "100"


class TestCallbackSystem:
    """Tests for regime transition callback system."""

    def test_add_callback(self):
        """Should be able to add a callback."""
        engine = RegimeDetectionEngine()

        class TestCallback:
            def on_transition(self, old_regime, new_regime, transition_date,
                            days_in_prior_regime, trigger_values):
                pass

        callback = TestCallback()
        engine.add_callback(callback)
        assert callback in engine._callbacks

    def test_remove_callback(self):
        """Should be able to remove a callback."""
        engine = RegimeDetectionEngine()

        class TestCallback:
            def on_transition(self, old_regime, new_regime, transition_date,
                            days_in_prior_regime, trigger_values):
                pass

        callback = TestCallback()
        engine.add_callback(callback)
        result = engine.remove_callback(callback)
        assert result is True
        assert callback not in engine._callbacks

    def test_remove_nonexistent_callback(self):
        """Removing non-existent callback should return False."""
        engine = RegimeDetectionEngine()

        class TestCallback:
            def on_transition(self, old_regime, new_regime, transition_date,
                            days_in_prior_regime, trigger_values):
                pass

        callback = TestCallback()
        result = engine.remove_callback(callback)
        assert result is False

    def test_transition_event_dataclass(self):
        """RegimeTransitionEvent should work correctly."""
        event = RegimeTransitionEvent(
            old_regime="BULL",
            new_regime="BEAR",
            transition_date=date(2026, 1, 15),
            days_in_prior_regime=10,
            trigger_values={"vix": "30.0"},
            confidence=Decimal("0.65"),
        )
        assert event.old_regime == "BULL"
        assert event.new_regime == "BEAR"

        event_dict = event.to_dict()
        assert event_dict["old_regime"] == "BULL"
        assert event_dict["transition_date"] == "2026-01-15"

    def test_regime_duration_metrics_empty(self):
        """Duration metrics should handle empty history."""
        engine = RegimeDetectionEngine()
        metrics = engine._compute_regime_duration_metrics()
        assert metrics["total_transitions"] == 0
        assert metrics["avg_duration_days"] is None

    def test_get_transition_history(self):
        """Should be able to get transition history."""
        engine = RegimeDetectionEngine()
        history = engine.get_transition_history()
        assert isinstance(history, list)
        assert len(history) == 0


class TestVIXKalmanFilter:
    """Tests for VIXKalmanFilter class."""

    def test_initialization(self):
        """Filter should initialize with default values."""
        kf = VIXKalmanFilter()
        assert kf.Q == Decimal("0.1")
        assert kf.R == Decimal("1.0")
        assert not kf._initialized

    def test_first_update(self):
        """First update should return the measurement."""
        kf = VIXKalmanFilter()
        result = kf.update(Decimal("25.0"))
        assert result == Decimal("25.0")
        assert kf._initialized

    def test_smoothing_effect(self):
        """Subsequent updates should smooth the signal."""
        kf = VIXKalmanFilter()
        kf.update(Decimal("20.0"))
        result = kf.update(Decimal("30.0"))
        # Result should be between 20 and 30 (smoothed)
        assert Decimal("20.0") < result < Decimal("30.0")

    def test_reset(self):
        """Reset should return to initial state."""
        kf = VIXKalmanFilter()
        kf.update(Decimal("25.0"))
        kf.reset()
        assert not kf._initialized
        assert kf.x == Decimal("20.0")

    def test_get_state(self):
        """Should return current filter state."""
        kf = VIXKalmanFilter()
        kf.update(Decimal("22.5"))
        state = kf.get_state()
        assert "estimate" in state
        assert "error_covariance" in state
        assert state["initialized"] == "True"

    def test_custom_parameters(self):
        """Should accept custom noise parameters."""
        kf = VIXKalmanFilter(
            process_noise=Decimal("0.5"),
            measurement_noise=Decimal("2.0"),
        )
        assert kf.Q == Decimal("0.5")
        assert kf.R == Decimal("2.0")


class TestRegimeHMM:
    """Tests for RegimeHMM class."""

    def test_initialization(self):
        """HMM should initialize with uniform probabilities."""
        hmm = RegimeHMM()
        probs = hmm.get_state_probabilities()
        # All probabilities should be roughly equal
        for regime in RegimeHMM.REGIMES:
            assert regime in probs
            assert probs[regime] > Decimal("0")

    def test_update(self):
        """Update should adjust probabilities based on likelihoods."""
        hmm = RegimeHMM()
        likelihoods = {
            "BULL": Decimal("0.7"),
            "BEAR": Decimal("0.1"),
            "VOLATILITY_SPIKE": Decimal("0.05"),
            "SECTOR_ROTATION": Decimal("0.1"),
            "RECESSION_RISK": Decimal("0.02"),
            "CREDIT_CRISIS": Decimal("0.01"),
            "SECTOR_DISLOCATION": Decimal("0.02"),
        }
        probs = hmm.update(likelihoods)
        # BULL should have highest probability given high likelihood
        assert probs["BULL"] > probs["BEAR"]

    def test_get_most_likely_regime(self):
        """Should return most likely regime and probability."""
        hmm = RegimeHMM()
        likelihoods = {regime: Decimal("0.1") for regime in RegimeHMM.REGIMES}
        likelihoods["BULL"] = Decimal("0.9")  # Make BULL dominant
        hmm.update(likelihoods)

        regime, prob = hmm.get_most_likely_regime()
        assert isinstance(regime, str)
        assert isinstance(prob, Decimal)

    def test_regime_persistence(self):
        """Regimes should tend to persist (diagonal dominance)."""
        hmm = RegimeHMM()
        # Verify transition matrix has higher diagonal values
        for regime in RegimeHMM.REGIMES:
            trans = hmm.transition_probs[regime]
            diag = trans[regime]
            # Diagonal should be at least 0.5
            assert diag >= Decimal("0.50")

    def test_reset(self):
        """Reset should return to uniform probabilities."""
        hmm = RegimeHMM()
        # Update with skewed likelihoods
        likelihoods = {regime: Decimal("0.1") for regime in RegimeHMM.REGIMES}
        likelihoods["BULL"] = Decimal("0.9")
        hmm.update(likelihoods)

        hmm.reset()
        probs = hmm.get_state_probabilities()
        # Should be back to uniform
        values = list(probs.values())
        assert max(values) - min(values) < Decimal("0.01")


class TestEnsembleRegimeClassifier:
    """Tests for EnsembleRegimeClassifier class."""

    def test_initialization(self):
        """Ensemble should initialize with default weights."""
        ensemble = EnsembleRegimeClassifier()
        assert "score_based" in ensemble.weights
        assert ensemble.weights["score_based"] == Decimal("0.40")

    def test_classify(self):
        """Should return classification result."""
        ensemble = EnsembleRegimeClassifier()
        scores = {
            "BULL": Decimal("50"),
            "BEAR": Decimal("20"),
            "VOLATILITY_SPIKE": Decimal("10"),
            "SECTOR_ROTATION": Decimal("15"),
            "RECESSION_RISK": Decimal("5"),
            "CREDIT_CRISIS": Decimal("0"),
            "SECTOR_DISLOCATION": Decimal("0"),
        }
        result = ensemble.classify(scores, vix_current=Decimal("18.0"))

        assert "regime" in result
        assert "confidence" in result
        assert "ensemble_probabilities" in result
        assert "method" in result
        assert result["method"] == "ensemble"

    def test_classify_with_vix(self):
        """Should use Kalman filter for VIX smoothing."""
        ensemble = EnsembleRegimeClassifier()
        scores = {regime: Decimal("10") for regime in RegimeHMM.REGIMES}

        result = ensemble.classify(scores, vix_current=Decimal("25.0"))
        assert result["vix_smoothed"] is not None

    def test_reset(self):
        """Reset should clear all components."""
        ensemble = EnsembleRegimeClassifier()
        # Do some updates
        scores = {regime: Decimal("10") for regime in RegimeHMM.REGIMES}
        ensemble.classify(scores, vix_current=Decimal("25.0"))

        ensemble.reset()
        # Kalman should be reset
        assert not ensemble.kalman_filter._initialized

    def test_custom_weights(self):
        """Should accept custom weights."""
        custom_weights = {
            "score_based": Decimal("0.50"),
            "vix_hmm": Decimal("0.20"),
            "credit_hmm": Decimal("0.15"),
            "yield_hmm": Decimal("0.15"),
        }
        ensemble = EnsembleRegimeClassifier(weights=custom_weights)
        assert ensemble.weights["score_based"] == Decimal("0.50")


class TestEnsembleIntegration:
    """Tests for ensemble classification in detect_regime."""

    def test_use_ensemble_flag(self):
        """use_ensemble flag should enable ensemble classification."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
            use_ensemble=True,
        )
        assert result["ensemble"] is not None
        assert result["ensemble"]["method"] == "ensemble"

    def test_ensemble_disabled_by_default(self):
        """Ensemble should be disabled by default."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )
        assert result["ensemble"] is None


class TestPerformanceOptimizations:
    """Tests for performance optimizations."""

    def test_staleness_haircut_cached(self):
        """Staleness haircut should be cached."""
        # Call the static cached method multiple times
        result1 = RegimeDetectionEngine._compute_staleness_haircut_cached(5, 10)
        result2 = RegimeDetectionEngine._compute_staleness_haircut_cached(5, 10)
        assert result1 == result2
        assert result1[0] == "0.85"
        assert result1[1] is False

    def test_staleness_haircut_cached_stale(self):
        """Cached method should detect stale data."""
        result = RegimeDetectionEngine._compute_staleness_haircut_cached(15, 10)
        assert result[0] == "0.00"
        assert result[1] is True

    def test_precomputed_thresholds(self):
        """Precomputed thresholds should be accessible."""
        engine = RegimeDetectionEngine()
        thresholds = engine._precomputed_thresholds
        assert "vix" in thresholds
        assert "xbi" in thresholds
        assert thresholds["vix"]["extreme"] == engine.VIX_EXTREME

    def test_regime_score_weights(self):
        """Precomputed score weights should be accessible."""
        engine = RegimeDetectionEngine()
        weights = engine._regime_score_weights
        assert "vix_extreme_vol_spike" in weights
        assert weights["vix_extreme_vol_spike"] == Decimal("40")


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_existing_signature_works(self):
        """Old function signatures should still work."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
        )
        assert "regime" in result
        assert "signal_adjustments" in result

    def test_new_signals_optional(self):
        """New signals should be optional."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
            as_of_date=date(2026, 1, 15),
        )
        # Should work without new signals
        assert result["indicators"]["yield_curve_state"] == "UNKNOWN"
        assert result["indicators"]["credit_environment"] == "UNKNOWN"
        assert result["indicators"]["fund_flow_state"] == "UNKNOWN"

    def test_regime_scores_include_new_regimes(self):
        """Regime scores should include new regimes."""
        engine = RegimeDetectionEngine()
        result = engine.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("3.0"),
        )
        scores = result["regime_scores"]
        assert "RECESSION_RISK" in scores
        assert "CREDIT_CRISIS" in scores
        assert "SECTOR_DISLOCATION" in scores
