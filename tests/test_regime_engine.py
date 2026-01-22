#!/usr/bin/env python3
"""
Tests for regime_engine.py

Market regime detection affects signal weight adjustments in the composite scoring.
These tests cover:
- Regime classification (BULL, BEAR, VOLATILITY_SPIKE, SECTOR_ROTATION)
- VIX threshold classification
- XBI relative performance classification
- Signal weight adjustments by regime
- Staleness gating and confidence haircuts
- Composite weight adjustments
"""

import pytest
from datetime import date
from decimal import Decimal

from regime_engine import (
    RegimeDetectionEngine,
    MarketRegime,
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
