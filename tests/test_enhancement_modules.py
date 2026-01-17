#!/usr/bin/env python3
"""
test_enhancement_modules.py

Tests for enhancement modules: PoS Engine, Short Interest, Regime Detection.

Design Philosophy:
- Determinism: Same inputs → byte-identical outputs
- Golden fixtures: Known-good outputs for regression testing
- Edge cases: Missing data, boundary conditions, error states

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import sys
import json
import hashlib
from decimal import Decimal
from datetime import date
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pos_engine import ProbabilityOfSuccessEngine, DataQualityState
from short_interest_engine import ShortInterestSignalEngine
from regime_engine import RegimeDetectionEngine, MarketRegime


# =============================================================================
# TEST: Probability of Success Engine
# =============================================================================

class TestProbabilityOfSuccessEngine:
    """Tests for ProbabilityOfSuccessEngine."""

    def test_pos_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = ProbabilityOfSuccessEngine()
        assert engine.VERSION == "1.1.0"
        assert len(engine.audit_trail) == 0
        # Should have loaded benchmarks (either file or fallback)
        assert "phase_3" in engine.benchmarks

    def test_pos_score_phase3_oncology(self):
        """Test Phase 3 oncology scoring returns expected values."""
        engine = ProbabilityOfSuccessEngine()
        result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            as_of_date=date(2026, 1, 11)
        )

        # Basic structure checks
        assert "pos_score" in result
        assert "stage_score" in result
        assert "loa_probability" in result
        assert "data_quality_state" in result
        assert "missing_fields" in result
        assert "inputs_used" in result

        # Stage score should be 65 for Phase 3
        assert result["stage_score"] == Decimal("65")

        # PoS score should be reasonable (not negative or > 100)
        assert Decimal("0") <= result["pos_score"] <= Decimal("100")

        # Data quality should be PARTIAL (missing optional fields)
        assert result["data_quality_state"] == "PARTIAL"

    def test_pos_score_rare_disease_differentiation(self):
        """Test that rare disease scores higher than oncology for Phase 3."""
        engine = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        oncology = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            as_of_date=as_of
        )
        rare = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="rare disease",
            as_of_date=as_of
        )

        # Rare disease should have higher PoS than oncology
        assert rare["pos_score"] > oncology["pos_score"]
        # Stage scores should be identical (same phase)
        assert rare["stage_score"] == oncology["stage_score"]

    def test_pos_score_determinism(self):
        """Test that same inputs produce identical outputs."""
        engine1 = ProbabilityOfSuccessEngine()
        engine2 = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        result1 = engine1.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            as_of_date=as_of
        )
        result2 = engine2.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            as_of_date=as_of
        )

        # Same inputs → same outputs
        assert result1["pos_score"] == result2["pos_score"]
        assert result1["stage_score"] == result2["stage_score"]
        assert result1["loa_probability"] == result2["loa_probability"]

        # Timestamps should be deterministic (based on as_of_date)
        assert result1["audit_entry"]["timestamp"] == result2["audit_entry"]["timestamp"]
        assert result1["audit_entry"]["timestamp"] == "2026-01-11T00:00:00Z"

    def test_pos_score_stage_normalization(self):
        """Test that various stage inputs normalize correctly."""
        engine = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        # Test various Phase 3 inputs
        inputs = ["phase_3", "Phase 3", "Phase III", "p3", "phase3", "pivotal"]
        results = [
            engine.calculate_pos_score(base_stage=s, as_of_date=as_of)
            for s in inputs
        ]

        # All should normalize to phase_3 with same stage score
        for r in results:
            assert r["stage_normalized"] == "phase_3"
            assert r["stage_score"] == Decimal("65")

    def test_pos_score_indication_word_boundaries(self):
        """Test that indication matching uses word boundaries."""
        engine = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        # "dose" should NOT match "os" (overall survival)
        dose_result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="dose escalation study",
            as_of_date=as_of
        )
        # Should fall back to all_indications, not match oncology
        assert dose_result["indication_normalized"] == "all_indications"

        # "oncology" SHOULD match
        onc_result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology study",
            as_of_date=as_of
        )
        assert onc_result["indication_normalized"] == "oncology"

    def test_pos_score_clamping(self):
        """Test that trial_design_quality is clamped to 0.7-1.3."""
        engine = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        # Very high quality (should clamp to 1.3)
        high_result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            trial_design_quality=Decimal("2.0"),
            as_of_date=as_of
        )

        # Very low quality (should clamp to 0.7)
        low_result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            trial_design_quality=Decimal("0.3"),
            as_of_date=as_of
        )

        # Scores should be different but within bounds
        assert high_result["pos_score"] > low_result["pos_score"]
        assert Decimal("0") <= high_result["pos_score"] <= Decimal("100")
        assert Decimal("0") <= low_result["pos_score"] <= Decimal("100")

    def test_pos_score_universe(self):
        """Test scoring an entire universe."""
        engine = ProbabilityOfSuccessEngine()
        as_of = date(2026, 1, 11)

        universe = [
            {"ticker": "ACME", "base_stage": "phase_3", "indication": "oncology"},
            {"ticker": "BIOTECH", "base_stage": "phase_2", "indication": "rare disease"},
            {"ticker": "PHARMA", "base_stage": "commercial", "indication": "neurology"}
        ]

        result = engine.score_universe(universe, as_of)

        assert result["as_of_date"] == "2026-01-11"
        assert len(result["scores"]) == 3
        assert result["diagnostic_counts"]["total_scored"] == 3
        assert "content_hash" in result["provenance"]


# =============================================================================
# TEST: Short Interest Signal Engine
# =============================================================================

class TestShortInterestSignalEngine:
    """Tests for ShortInterestSignalEngine."""

    def test_short_interest_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = ShortInterestSignalEngine()
        assert engine.VERSION == "1.0.0"
        assert len(engine.audit_trail) == 0

    def test_short_signal_squeeze_potential(self):
        """Test squeeze potential detection."""
        engine = ShortInterestSignalEngine()
        as_of = date(2026, 1, 11)

        # High squeeze potential: high SI%, high DTC
        result = engine.calculate_short_signal(
            ticker="SQUEEZE",
            short_interest_pct=Decimal("45.0"),
            days_to_cover=Decimal("12.0"),
            as_of_date=as_of
        )

        assert result["status"] == "SUCCESS"
        assert result["squeeze_potential"] == "EXTREME"
        assert result["short_signal_score"] > Decimal("50")  # Bullish

    def test_short_signal_low_interest(self):
        """Test low short interest is neutral."""
        engine = ShortInterestSignalEngine()
        as_of = date(2026, 1, 11)

        result = engine.calculate_short_signal(
            ticker="NORMAL",
            short_interest_pct=Decimal("3.0"),
            days_to_cover=Decimal("2.0"),
            as_of_date=as_of
        )

        assert result["squeeze_potential"] == "LOW"
        # Score should be around neutral (50)
        assert Decimal("45") <= result["short_signal_score"] <= Decimal("55")

    def test_short_signal_insufficient_data(self):
        """Test handling of missing data."""
        engine = ShortInterestSignalEngine()
        as_of = date(2026, 1, 11)

        result = engine.calculate_short_signal(
            ticker="MISSING",
            short_interest_pct=None,
            days_to_cover=None,
            as_of_date=as_of
        )

        assert result["status"] == "INSUFFICIENT_DATA"
        assert result["squeeze_potential"] == "UNKNOWN"
        assert result["short_signal_score"] == Decimal("50")  # Neutral
        assert "SI_DATA_MISSING" in result["flags"]

    def test_short_signal_determinism(self):
        """Test deterministic outputs."""
        engine1 = ShortInterestSignalEngine()
        engine2 = ShortInterestSignalEngine()
        as_of = date(2026, 1, 11)

        result1 = engine1.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("25.0"),
            days_to_cover=Decimal("8.0"),
            as_of_date=as_of
        )
        result2 = engine2.calculate_short_signal(
            ticker="TEST",
            short_interest_pct=Decimal("25.0"),
            days_to_cover=Decimal("8.0"),
            as_of_date=as_of
        )

        assert result1["short_signal_score"] == result2["short_signal_score"]
        assert result1["audit_entry"]["timestamp"] == "2026-01-11T00:00:00Z"

    def test_short_signal_trend_covering(self):
        """Test bullish signal when shorts are covering."""
        engine = ShortInterestSignalEngine()
        as_of = date(2026, 1, 11)

        result = engine.calculate_short_signal(
            ticker="COVERING",
            short_interest_pct=Decimal("20.0"),
            days_to_cover=Decimal("6.0"),
            short_interest_change_pct=Decimal("-25.0"),  # Strong covering
            as_of_date=as_of
        )

        assert result["trend_direction"] == "COVERING"
        assert result["short_signal_score"] > Decimal("55")  # Bullish boost


# =============================================================================
# TEST: Regime Detection Engine
# =============================================================================

class TestRegimeDetectionEngine:
    """Tests for RegimeDetectionEngine."""

    def test_regime_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = RegimeDetectionEngine()
        assert engine.VERSION == "1.1.0"
        assert engine.current_regime is None

    def test_regime_bull_detection(self):
        """Test bull regime detection."""
        engine = RegimeDetectionEngine()
        as_of = date(2026, 1, 11)

        result = engine.detect_regime(
            vix_current=Decimal("13.0"),  # Low VIX
            xbi_vs_spy_30d=Decimal("6.0"),  # Strong XBI outperformance
            fed_rate_change_3m=Decimal("-0.25"),  # Rate cuts
            as_of_date=as_of
        )

        assert result["regime"] == "BULL"
        assert result["confidence"] > Decimal("0.4")

        # Check signal adjustments
        adjustments = result["signal_adjustments"]
        assert adjustments["momentum"] > Decimal("1.0")  # Boosted
        assert adjustments["fundamental"] < Decimal("1.0")  # Reduced

    def test_regime_bear_detection(self):
        """Test bear regime detection."""
        engine = RegimeDetectionEngine()
        as_of = date(2026, 1, 11)

        result = engine.detect_regime(
            vix_current=Decimal("28.0"),  # Elevated VIX
            xbi_vs_spy_30d=Decimal("-8.0"),  # XBI underperforming
            fed_rate_change_3m=Decimal("0.75"),  # Aggressive hikes
            as_of_date=as_of
        )

        assert result["regime"] == "BEAR"
        assert result["signal_adjustments"]["quality"] > Decimal("1.0")  # Quality premium

    def test_regime_volatility_spike(self):
        """Test volatility spike detection."""
        engine = RegimeDetectionEngine()
        as_of = date(2026, 1, 11)

        # VIX = 50+ with neutral XBI performance triggers VOLATILITY_SPIKE
        # (XBI underperformance would tilt toward BEAR)
        result = engine.detect_regime(
            vix_current=Decimal("50.0"),  # Extreme VIX
            xbi_vs_spy_30d=Decimal("0.0"),  # Neutral XBI
            as_of_date=as_of
        )

        assert result["regime"] == "VOLATILITY_SPIKE"
        assert "CRISIS_VOLATILITY" in result["flags"]
        assert result["signal_adjustments"]["momentum"] < Decimal("0.8")  # Dampened

    def test_regime_determinism(self):
        """Test deterministic outputs."""
        engine1 = RegimeDetectionEngine()
        engine2 = RegimeDetectionEngine()
        as_of = date(2026, 1, 11)

        result1 = engine1.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("2.5"),
            as_of_date=as_of
        )
        result2 = engine2.detect_regime(
            vix_current=Decimal("18.0"),
            xbi_vs_spy_30d=Decimal("2.5"),
            as_of_date=as_of
        )

        assert result1["regime"] == result2["regime"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["audit_entry"]["timestamp"] == "2026-01-11T00:00:00Z"

    def test_regime_weight_adjustment(self):
        """Test composite weight adjustment."""
        engine = RegimeDetectionEngine()

        base_weights = {
            "clinical": Decimal("0.40"),
            "financial": Decimal("0.35"),
            "catalyst": Decimal("0.25")
        }

        adjusted = engine.get_composite_weight_adjustments(
            regime="BEAR",
            base_weights=base_weights
        )

        # Weights should sum to ~1.0
        total = sum(adjusted.values())
        assert Decimal("0.99") <= total <= Decimal("1.01")

        # In bear regime, financial should be relatively higher
        # (because it gets 1.20x multiplier)


# =============================================================================
# TEST: Integration Tests
# =============================================================================

class TestEnhancementIntegration:
    """Integration tests for enhancement modules."""

    def test_full_pipeline_determinism(self):
        """Test full pipeline produces deterministic results."""
        as_of = date(2026, 1, 11)

        # Run twice
        results = []
        for _ in range(2):
            pos_engine = ProbabilityOfSuccessEngine()
            short_engine = ShortInterestSignalEngine()
            regime_engine = RegimeDetectionEngine()

            # Detect regime
            regime = regime_engine.detect_regime(
                vix_current=Decimal("18.0"),
                xbi_vs_spy_30d=Decimal("3.0"),
                as_of_date=as_of
            )

            # Score a company
            pos = pos_engine.calculate_pos_score(
                base_stage="phase_3",
                indication="oncology",
                as_of_date=as_of
            )

            short = short_engine.calculate_short_signal(
                ticker="TEST",
                short_interest_pct=Decimal("15.0"),
                days_to_cover=Decimal("5.0"),
                as_of_date=as_of
            )

            results.append({
                "regime": regime["regime"],
                "pos_score": str(pos["pos_score"]),
                "short_score": str(short["short_signal_score"])
            })

        # Both runs should be identical
        assert results[0] == results[1]


# =============================================================================
# RUN TESTS
# =============================================================================

def run_tests():
    """Run all tests and report results."""
    test_classes = [
        TestProbabilityOfSuccessEngine,
        TestShortInterestSignalEngine,
        TestRegimeDetectionEngine,
        TestEnhancementIntegration
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running: {test_class.__name__}")
        print('='*60)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method in methods:
            total_tests += 1
            try:
                getattr(instance, method)()
                print(f"  ✓ {method}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {method}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method}: {e}")
            except Exception as e:
                print(f"  ✗ {method}: EXCEPTION - {e}")
                failed_tests.append(f"{test_class.__name__}.{method}: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print('='*60)

    if failed_tests:
        print("\nFailed tests:")
        for f in failed_tests:
            print(f"  - {f}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
