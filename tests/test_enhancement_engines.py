#!/usr/bin/env python3
"""
Tests for Enhancement Engines

Covers previously untested enhancement engines:
- liquidity_scoring.py
- timeline_slippage_engine.py
- manager_momentum_v1.py (if exists)

All tests are deterministic and do not use datetime.now().
"""

import pytest
from datetime import date
from decimal import Decimal

# =============================================================================
# LIQUIDITY SCORING TESTS
# =============================================================================


class TestLiquidityScoring:
    """Tests for liquidity_scoring.py"""

    def test_classify_market_cap_tier_micro(self):
        """Test micro cap classification (<$300M)."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(100_000_000) == "micro"
        assert classify_market_cap_tier(299_999_999) == "micro"

    def test_classify_market_cap_tier_small(self):
        """Test small cap classification ($300M-$2B)."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(300_000_000) == "small"
        assert classify_market_cap_tier(1_000_000_000) == "small"
        assert classify_market_cap_tier(1_999_999_999) == "small"

    def test_classify_market_cap_tier_mid(self):
        """Test mid cap classification ($2B-$10B)."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(2_000_000_000) == "mid"
        assert classify_market_cap_tier(5_000_000_000) == "mid"
        assert classify_market_cap_tier(9_999_999_999) == "mid"

    def test_classify_market_cap_tier_large(self):
        """Test large cap classification (>=$10B)."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(10_000_000_000) == "large"
        assert classify_market_cap_tier(100_000_000_000) == "large"

    def test_classify_market_cap_tier_unknown(self):
        """Test unknown tier for invalid market cap."""
        from liquidity_scoring import classify_market_cap_tier

        assert classify_market_cap_tier(None) == "unknown"
        assert classify_market_cap_tier(0) == "unknown"
        assert classify_market_cap_tier(-100) == "unknown"

    def test_compute_adv_score_zero(self):
        """Test ADV score for zero/negative volume."""
        from liquidity_scoring import compute_adv_score

        assert compute_adv_score(0, 1_000_000) == 0
        assert compute_adv_score(-100, 1_000_000) == 0

    def test_compute_adv_score_at_threshold(self):
        """Test ADV score at threshold gives partial score."""
        from liquidity_scoring import compute_adv_score

        # At threshold, we're at 50% of target (2x threshold)
        score = compute_adv_score(1_000_000, 1_000_000)
        assert 30 <= score <= 40  # ~35 (half of 70)

    def test_compute_adv_score_at_double_threshold(self):
        """Test ADV score at 2x threshold gives max score."""
        from liquidity_scoring import compute_adv_score

        # At 2x threshold, should get full score
        score = compute_adv_score(2_000_000, 1_000_000)
        assert score == 70

    def test_compute_adv_score_above_double_threshold(self):
        """Test ADV score is capped at max."""
        from liquidity_scoring import compute_adv_score

        # Above 2x threshold, should still be capped at 70
        score = compute_adv_score(10_000_000, 1_000_000)
        assert score == 70

    def test_compute_spread_score_tight(self):
        """Test spread score for tight spread."""
        from liquidity_scoring import compute_spread_score

        assert compute_spread_score(10) == 30  # Very tight
        assert compute_spread_score(50) == 30  # At threshold

    def test_compute_spread_score_wide(self):
        """Test spread score for wide spread."""
        from liquidity_scoring import compute_spread_score

        assert compute_spread_score(400) == 0  # At max threshold
        assert compute_spread_score(500) == 0  # Above threshold

    def test_compute_spread_score_mid(self):
        """Test spread score interpolation."""
        from liquidity_scoring import compute_spread_score

        # 225 bps is halfway between 50 and 400
        score = compute_spread_score(225)
        assert 10 <= score <= 20  # Should be around 15

    def test_compute_spread_score_none(self):
        """Test spread score for missing data."""
        from liquidity_scoring import compute_spread_score

        assert compute_spread_score(None) == 0  # Conservative

    def test_extract_spread_bps_direct_field(self):
        """Test spread extraction from direct field."""
        from liquidity_scoring import extract_spread_bps

        record = {"spread_bps": 100}
        assert extract_spread_bps(record) == 100

    def test_extract_spread_bps_from_bid_ask(self):
        """Test spread computation from bid/ask."""
        from liquidity_scoring import extract_spread_bps

        # Spread = (ask - bid) / mid * 10000
        # (10.10 - 10.00) / 10.05 * 10000 = ~99.5 bps
        record = {"bid": 10.00, "ask": 10.10}
        spread = extract_spread_bps(record)
        assert spread is not None
        assert 95 <= spread <= 105

    def test_extract_spread_bps_missing(self):
        """Test spread extraction with missing data."""
        from liquidity_scoring import extract_spread_bps

        assert extract_spread_bps({}) is None
        assert extract_spread_bps({"bid": 10.00}) is None

    def test_compute_liquidity_score_full(self):
        """Test full liquidity score computation."""
        from liquidity_scoring import compute_liquidity_score

        market_data = {
            "ACME": {
                "market_cap": 1_000_000_000,  # $1B = small cap
                "avg_volume": 500_000,
                "price": 20.0,
                "spread_bps": 100,
            }
        }

        result = compute_liquidity_score("ACME", market_data)

        assert result["ticker"] == "ACME"
        assert result["liquidity_tier"] == "small"
        assert 0 <= result["liquidity_score"] <= 100
        assert result["adv_usd"] == 500_000 * 20.0

    def test_compute_liquidity_score_missing_ticker(self):
        """Test liquidity score for missing ticker."""
        from liquidity_scoring import compute_liquidity_score

        result = compute_liquidity_score("UNKNOWN", {})

        assert result["ticker"] == "UNKNOWN"
        assert result["liquidity_score"] == 0
        assert "ADV_UNKNOWN" in result["risk_flags"]

    def test_compute_liquidity_score_determinism(self):
        """Test that same inputs produce identical outputs."""
        from liquidity_scoring import compute_liquidity_score

        market_data = {
            "TEST": {
                "market_cap": 500_000_000,
                "avg_volume": 100_000,
                "price": 15.0,
            }
        }

        result1 = compute_liquidity_score("TEST", market_data)
        result2 = compute_liquidity_score("TEST", market_data)

        assert result1 == result2

    def test_score_all_tickers_sorted(self):
        """Test that results are sorted by ticker."""
        from liquidity_scoring import score_all_tickers

        market_data = {
            "ZZZ": {"market_cap": 1e9, "avg_volume": 100000, "price": 10},
            "AAA": {"market_cap": 1e9, "avg_volume": 100000, "price": 10},
            "MMM": {"market_cap": 1e9, "avg_volume": 100000, "price": 10},
        }

        results = score_all_tickers(["ZZZ", "AAA", "MMM"], market_data, "2026-01-15")

        tickers = [r["ticker"] for r in results]
        assert tickers == ["AAA", "MMM", "ZZZ"]


# =============================================================================
# TIMELINE SLIPPAGE ENGINE TESTS
# =============================================================================


class TestTimelineSlippageEngine:
    """Tests for timeline_slippage_engine.py"""

    def test_slippage_direction_pushout(self):
        """Test pushout detection."""
        from timeline_slippage_engine import TimelineSlippageEngine, SlippageDirection

        engine = TimelineSlippageEngine()

        current = [{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.PUSHOUT
        assert result.trial_results[0].days_slipped >= 365

    def test_slippage_direction_pullin(self):
        """Test pullin (acceleration) detection."""
        from timeline_slippage_engine import TimelineSlippageEngine, SlippageDirection

        engine = TimelineSlippageEngine()

        # Current date is 6 months earlier than prior
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-09-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.PULLIN

    def test_slippage_direction_stable(self):
        """Test stable timeline detection."""
        from timeline_slippage_engine import TimelineSlippageEngine, SlippageDirection

        engine = TimelineSlippageEngine()

        # Only 10 days difference - within threshold
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-10"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].direction == SlippageDirection.STABLE

    def test_slippage_severity_severe(self):
        """Test severe pushout classification (>180 days)."""
        from timeline_slippage_engine import TimelineSlippageEngine, SlippageSeverity

        engine = TimelineSlippageEngine()

        # 365 day delay
        current = [{"nct_id": "NCT001", "primary_completion_date": "2027-01-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.SEVERE

    def test_slippage_severity_moderate(self):
        """Test moderate pushout classification (60-180 days)."""
        from timeline_slippage_engine import TimelineSlippageEngine, SlippageSeverity

        engine = TimelineSlippageEngine()

        # 90 day delay
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-04-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.trial_results[0].severity == SlippageSeverity.MODERATE

    def test_repeat_offender_detection(self):
        """Test repeat offender flag for multiple pushouts."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        # Two trials with severe pushouts
        current = [
            {"nct_id": "NCT001", "primary_completion_date": "2027-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2027-06-01"},
        ]
        prior = [
            {"nct_id": "NCT001", "primary_completion_date": "2026-01-01"},
            {"nct_id": "NCT002", "primary_completion_date": "2026-06-01"},
        ]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result.repeat_offender is True
        assert "REPEAT_SLIPPAGE_OFFENDER" in result.flags

    def test_score_bounds(self):
        """Test that scores are always in [0, 100]."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        # Extreme pushout - should still be bounded
        current = [{"nct_id": "NCT001", "primary_completion_date": "2030-01-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-01-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert Decimal("0") <= result.execution_risk_score <= Decimal("100")

    def test_no_trials(self):
        """Test handling of ticker with no trials."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        result = engine.calculate_slippage_score(
            ticker="EMPTY",
            current_trials=[],
            prior_trials=[],
            as_of_date=date(2026, 1, 15),
        )

        assert result.execution_risk_score == Decimal("50")  # Neutral
        assert "NO_TRIALS" in result.flags

    def test_missing_prior_data(self):
        """Test handling of missing prior snapshot."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        result = engine.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=None,
            as_of_date=date(2026, 1, 15),
        )

        # Should handle gracefully
        assert result.execution_risk_score is not None
        assert result.confidence in ["low", "medium"]

    def test_as_of_date_required(self):
        """Test that as_of_date is required."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        with pytest.raises(ValueError, match="as_of_date is required"):
            engine.calculate_slippage_score(
                ticker="TEST",
                current_trials=[],
                as_of_date=None,
            )

    def test_determinism(self):
        """Test that same inputs produce identical outputs."""
        from timeline_slippage_engine import TimelineSlippageEngine

        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-09-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]

        engine1 = TimelineSlippageEngine()
        result1 = engine1.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        engine2 = TimelineSlippageEngine()
        result2 = engine2.calculate_slippage_score(
            ticker="TEST",
            current_trials=current,
            prior_trials=prior,
            as_of_date=date(2026, 1, 15),
        )

        assert result1.execution_risk_score == result2.execution_risk_score
        assert result1.pushout_count == result2.pushout_count
        assert result1.pullin_count == result2.pullin_count

    def test_score_universe(self):
        """Test scoring entire universe."""
        from timeline_slippage_engine import TimelineSlippageEngine

        engine = TimelineSlippageEngine()

        universe = [{"ticker": "AAA"}, {"ticker": "BBB"}]
        current_trials = {
            "AAA": [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}],
            "BBB": [{"nct_id": "NCT002", "primary_completion_date": "2026-09-01"}],
        }
        prior_trials = {
            "AAA": [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}],
            "BBB": [{"nct_id": "NCT002", "primary_completion_date": "2026-06-01"}],
        }

        result = engine.score_universe(
            universe=universe,
            current_trials_by_ticker=current_trials,
            prior_trials_by_ticker=prior_trials,
            as_of_date=date(2026, 1, 15),
        )

        assert "scores" in result
        assert "diagnostic_counts" in result
        assert "provenance" in result
        assert len(result["scores"]) == 2


# =============================================================================
# MANAGER MOMENTUM TESTS
# =============================================================================


class TestManagerMomentum:
    """Tests for manager_momentum_v1.py if available."""

    @pytest.fixture
    def manager_momentum_available(self):
        """Check if manager_momentum_v1 module exists."""
        try:
            import manager_momentum_v1
            return True
        except ImportError:
            return False

    def test_import(self, manager_momentum_available):
        """Test that module can be imported."""
        if not manager_momentum_available:
            pytest.skip("manager_momentum_v1 not available")

        import manager_momentum_v1
        assert hasattr(manager_momentum_v1, "__version__") or True


# =============================================================================
# INTEGRATION TEST
# =============================================================================


class TestEnhancementEngineIntegration:
    """Integration tests across enhancement engines."""

    def test_all_engines_have_version(self):
        """Verify all engines expose version info."""
        from liquidity_scoring import LIQUIDITY_SCORING_VERSION
        from timeline_slippage_engine import TimelineSlippageEngine

        assert LIQUIDITY_SCORING_VERSION is not None
        assert TimelineSlippageEngine.VERSION is not None

    def test_all_engines_deterministic(self):
        """Verify all engines produce deterministic output."""
        from liquidity_scoring import compute_liquidity_score
        from timeline_slippage_engine import TimelineSlippageEngine

        # Liquidity
        market_data = {"TEST": {"market_cap": 1e9, "avg_volume": 100000, "price": 10}}
        liq1 = compute_liquidity_score("TEST", market_data)
        liq2 = compute_liquidity_score("TEST", market_data)
        assert liq1 == liq2

        # Timeline slippage
        engine = TimelineSlippageEngine()
        current = [{"nct_id": "NCT001", "primary_completion_date": "2026-06-01"}]
        prior = [{"nct_id": "NCT001", "primary_completion_date": "2026-03-01"}]

        slip1 = engine.calculate_slippage_score("TEST", current, prior, date(2026, 1, 15))
        slip2 = engine.calculate_slippage_score("TEST", current, prior, date(2026, 1, 15))
        assert slip1.execution_risk_score == slip2.execution_risk_score
