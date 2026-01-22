#!/usr/bin/env python3
"""
Tests for common/score_utils.py

Score utilities provide standardized operations for financial score manipulation.
These tests cover:
- Type conversion (to_decimal)
- Score clamping (clamp_score, clamp_weight)
- Score normalization (normalize_to_range, rank_normalize)
- Score aggregation (weighted_average, hybrid_aggregate)
- Penalty application (apply_penalty, apply_multiplier)
"""

import pytest
from decimal import Decimal

from common.score_utils import (
    # Type conversion
    to_decimal,
    to_float_safe,
    # Score clamping
    clamp_score,
    clamp_weight,
    clamp_in_place,
    clamp_all_scores,
    # Score normalization
    normalize_to_range,
    rank_normalize,
    # Score aggregation
    weighted_average,
    hybrid_aggregate,
    # Penalty application
    apply_penalty,
    apply_multiplier,
    # Validation
    validate_score_bounds,
    # Constants
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_SCORE,
    SCORE_PRECISION,
)


class TestToDecimal:
    """Tests for to_decimal function."""

    def test_decimal_passthrough(self):
        """Decimal input should pass through."""
        result = to_decimal(Decimal("5.5"))
        assert result == Decimal("5.5")

    def test_int_conversion(self):
        """Int should be converted via string."""
        result = to_decimal(5)
        assert result == Decimal("5")

    def test_float_conversion(self):
        """Float should be converted via string."""
        result = to_decimal(5.5)
        assert result == Decimal("5.5")

    def test_string_conversion(self):
        """String number should be converted."""
        result = to_decimal("5.5")
        assert result == Decimal("5.5")

    def test_string_with_whitespace(self):
        """String with whitespace should be trimmed."""
        result = to_decimal("  5.5  ")
        assert result == Decimal("5.5")

    def test_empty_string(self):
        """Empty string should return default."""
        result = to_decimal("", default=Decimal("0"))
        assert result == Decimal("0")

    def test_none_returns_default(self):
        """None should return default."""
        result = to_decimal(None, default=Decimal("0"))
        assert result == Decimal("0")

    def test_bool_returns_default(self):
        """Bool should return default (prevent True -> 1)."""
        result = to_decimal(True, default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_invalid_string(self):
        """Invalid string should return default."""
        result = to_decimal("abc", default=Decimal("0"))
        assert result == Decimal("0")


class TestToFloatSafe:
    """Tests for to_float_safe function."""

    def test_float_passthrough(self):
        """Float input should pass through."""
        result = to_float_safe(5.5)
        assert result == 5.5

    def test_decimal_conversion(self):
        """Decimal should be converted."""
        result = to_float_safe(Decimal("5.5"))
        assert result == 5.5

    def test_none_returns_default(self):
        """None should return default."""
        result = to_float_safe(None, default=0.0)
        assert result == 0.0


class TestClampScore:
    """Tests for clamp_score function."""

    def test_within_range(self):
        """Score within range should be unchanged."""
        result = clamp_score(Decimal("50"))
        assert result == Decimal("50.00")

    def test_above_max(self):
        """Score above max should be clamped."""
        result = clamp_score(Decimal("150"))
        assert result == Decimal("100.00")

    def test_below_min(self):
        """Score below min should be clamped."""
        result = clamp_score(Decimal("-10"))
        assert result == Decimal("0.00")

    def test_exactly_at_min(self):
        """Score at min should be unchanged."""
        result = clamp_score(Decimal("0"))
        assert result == Decimal("0.00")

    def test_exactly_at_max(self):
        """Score at max should be unchanged."""
        result = clamp_score(Decimal("100"))
        assert result == Decimal("100.00")

    def test_custom_range(self):
        """Custom range should be respected."""
        result = clamp_score(Decimal("150"), min_val=50, max_val=200)
        assert result == Decimal("150.00")

    def test_none_returns_default(self):
        """None should return default."""
        result = clamp_score(None, default=Decimal("50"))
        assert result == Decimal("50")

    def test_int_input(self):
        """Int input should be converted."""
        result = clamp_score(75)
        assert result == Decimal("75.00")

    def test_float_input(self):
        """Float input should be converted."""
        result = clamp_score(75.5)
        assert result == Decimal("75.50")

    def test_string_input(self):
        """String input should be converted."""
        result = clamp_score("75")
        assert result == Decimal("75.00")

    def test_precision(self):
        """Result should have standard precision."""
        result = clamp_score(Decimal("75.12345"))
        assert result == Decimal("75.12")


class TestClampWeight:
    """Tests for clamp_weight function."""

    def test_within_range(self):
        """Weight within range should be unchanged."""
        result = clamp_weight(Decimal("0.5"))
        assert result == Decimal("0.5000")

    def test_above_max(self):
        """Weight above max should be clamped."""
        result = clamp_weight(Decimal("1.5"))
        assert result == Decimal("1.0000")

    def test_below_min(self):
        """Weight below min should be clamped."""
        result = clamp_weight(Decimal("-0.5"))
        assert result == Decimal("0.0000")

    def test_none_returns_zero(self):
        """None should return 0."""
        result = clamp_weight(None)
        assert result == Decimal("0.0000")


class TestClampInPlace:
    """Tests for clamp_in_place function."""

    def test_modifies_dict(self):
        """Should modify dict in place."""
        scores = {"clinical": Decimal("150")}
        clamp_in_place(scores, "clinical")
        assert scores["clinical"] == Decimal("100.00")

    def test_skips_missing_key(self):
        """Should skip if key not present."""
        scores = {"clinical": Decimal("50")}
        clamp_in_place(scores, "financial")  # Not present
        assert "financial" not in scores

    def test_skips_none_value(self):
        """Should skip None values."""
        scores = {"clinical": None}
        clamp_in_place(scores, "clinical")
        assert scores["clinical"] is None


class TestNormalizeToRange:
    """Tests for normalize_to_range function."""

    def test_midpoint(self):
        """Midpoint of input should map to midpoint of output."""
        result = normalize_to_range(500, input_min=0, input_max=1000)
        assert result == Decimal("50.00")

    def test_minimum(self):
        """Input minimum should map to output minimum."""
        result = normalize_to_range(0, input_min=0, input_max=1000)
        assert result == Decimal("0.00")

    def test_maximum(self):
        """Input maximum should map to output maximum."""
        result = normalize_to_range(1000, input_min=0, input_max=1000)
        assert result == Decimal("100.00")

    def test_fractional_input(self):
        """Fractional input range should work."""
        result = normalize_to_range(0.75, input_min=0, input_max=1)
        assert result == Decimal("75.00")

    def test_custom_output_range(self):
        """Custom output range should be respected."""
        result = normalize_to_range(
            500, input_min=0, input_max=1000,
            output_min=Decimal("20"), output_max=Decimal("80")
        )
        assert result == Decimal("50.00")  # Midpoint of 20-80 is 50

    def test_below_input_min(self):
        """Value below input min should clamp to output min."""
        result = normalize_to_range(-100, input_min=0, input_max=1000)
        assert result == Decimal("0.00")

    def test_above_input_max(self):
        """Value above input max should clamp to output max."""
        result = normalize_to_range(1500, input_min=0, input_max=1000)
        assert result == Decimal("100.00")

    def test_zero_input_range(self):
        """Zero input range should return output minimum."""
        result = normalize_to_range(500, input_min=500, input_max=500)
        assert result == Decimal("0.00")

    def test_none_returns_default(self):
        """None should return default."""
        result = normalize_to_range(None, input_min=0, input_max=100, default=Decimal("50"))
        assert result == Decimal("50")


class TestRankNormalize:
    """Tests for rank_normalize function."""

    def test_basic_ranking(self):
        """Basic ranking should work."""
        values = [10, 20, 30]
        result = rank_normalize(values)
        assert result[0] == Decimal("0.00")   # Lowest
        assert result[2] == Decimal("100.00")  # Highest

    def test_handles_ties(self):
        """Ties should get average rank."""
        values = [10, 20, 20, 30]
        result = rank_normalize(values)
        # The two 20s should have same rank
        assert result[1] == result[2]

    def test_preserves_none(self):
        """None values should be preserved."""
        values = [10, None, 30]
        result = rank_normalize(values)
        assert result[1] is None
        assert result[0] is not None
        assert result[2] is not None

    def test_all_none(self):
        """All None should return all None."""
        values = [None, None, None]
        result = rank_normalize(values)
        assert result == [None, None, None]

    def test_single_value(self):
        """Single value should return midpoint."""
        values = [50]
        result = rank_normalize(values)
        assert result[0] == Decimal("50.00")

    def test_custom_output_range(self):
        """Custom output range should be respected."""
        values = [10, 20, 30]
        result = rank_normalize(
            values,
            output_min=Decimal("20"),
            output_max=Decimal("80"),
        )
        assert result[0] == Decimal("20.00")
        assert result[2] == Decimal("80.00")


class TestWeightedAverage:
    """Tests for weighted_average function."""

    def test_equal_weights(self):
        """Equal weights should give simple average."""
        scores = [Decimal("60"), Decimal("80")]
        weights = [Decimal("0.5"), Decimal("0.5")]
        result, total_weight = weighted_average(scores, weights)
        assert result == Decimal("70.00")
        assert total_weight == Decimal("1.0")

    def test_unequal_weights(self):
        """Unequal weights should be respected."""
        scores = [Decimal("100"), Decimal("0")]
        weights = [Decimal("0.75"), Decimal("0.25")]
        result, total_weight = weighted_average(scores, weights)
        assert result == Decimal("75.00")

    def test_skip_none(self):
        """With skip_none=True, None values should be skipped."""
        scores = [Decimal("100"), None, Decimal("50")]
        weights = [Decimal("0.4"), Decimal("0.3"), Decimal("0.3")]
        result, total_weight = weighted_average(scores, weights, skip_none=True)
        # Only first and third scores count
        # (100 * 0.4 + 50 * 0.3) / (0.4 + 0.3) = 55 / 0.7 = 78.57
        assert result is not None
        assert total_weight == Decimal("0.7")

    def test_no_skip_none(self):
        """With skip_none=False, any None should return None."""
        scores = [Decimal("100"), None, Decimal("50")]
        weights = [Decimal("0.4"), Decimal("0.3"), Decimal("0.3")]
        result, total_weight = weighted_average(scores, weights, skip_none=False)
        assert result is None
        assert total_weight == Decimal("0")

    def test_all_none(self):
        """All None scores should return None."""
        scores = [None, None]
        weights = [Decimal("0.5"), Decimal("0.5")]
        result, total_weight = weighted_average(scores, weights)
        assert result is None

    def test_zero_total_weight(self):
        """Zero total weight should return None."""
        scores = [Decimal("50"), Decimal("50")]
        weights = [Decimal("0"), Decimal("0")]
        result, total_weight = weighted_average(scores, weights)
        assert result is None

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise."""
        scores = [Decimal("50"), Decimal("50")]
        weights = [Decimal("0.5")]
        with pytest.raises(ValueError) as exc_info:
            weighted_average(scores, weights)
        assert "mismatch" in str(exc_info.value).lower()


class TestHybridAggregate:
    """Tests for hybrid_aggregate function."""

    def test_basic_hybrid(self):
        """Hybrid should combine weighted sum and minimum."""
        scores = {
            "clinical": Decimal("80"),
            "financial": Decimal("60"),
            "catalyst": Decimal("40"),
        }
        weights = {
            "clinical": Decimal("0.5"),
            "financial": Decimal("0.3"),
            "catalyst": Decimal("0.2"),
        }
        critical = ["clinical", "financial"]
        alpha = Decimal("0.85")

        result = hybrid_aggregate(scores, weights, critical, alpha)

        # Weighted sum = 80*0.5 + 60*0.3 + 40*0.2 = 40 + 18 + 8 = 66
        # Min critical = min(80, 60) = 60
        # Hybrid = 0.85 * 66 + 0.15 * 60 = 56.1 + 9 = 65.1
        assert result == Decimal("65.10")

    def test_no_critical_scores(self):
        """No critical scores should use weighted average only."""
        scores = {"clinical": Decimal("80")}
        weights = {"clinical": Decimal("1.0")}
        critical = ["nonexistent"]

        result = hybrid_aggregate(scores, weights, critical)
        assert result == Decimal("80.00")

    def test_missing_score(self):
        """Missing scores should be skipped."""
        scores = {"clinical": Decimal("80")}
        weights = {
            "clinical": Decimal("0.5"),
            "financial": Decimal("0.5"),  # Missing from scores
        }
        critical = ["clinical"]

        result = hybrid_aggregate(scores, weights, critical)
        assert result is not None


class TestApplyPenalty:
    """Tests for apply_penalty function."""

    def test_ten_percent_penalty(self):
        """10% penalty should reduce score by 10%."""
        result = apply_penalty(Decimal("100"), 0.10)
        assert result == Decimal("90.00")

    def test_fifty_percent_penalty(self):
        """50% penalty should halve the score."""
        result = apply_penalty(Decimal("100"), 0.50)
        assert result == Decimal("50.00")

    def test_full_penalty(self):
        """100% penalty should reduce to floor."""
        result = apply_penalty(Decimal("100"), 1.0)
        assert result == Decimal("0.00")

    def test_over_100_percent_penalty(self):
        """Over 100% penalty should reduce to floor."""
        result = apply_penalty(Decimal("100"), 1.5)
        assert result == Decimal("0.00")

    def test_custom_floor(self):
        """Custom floor should be respected."""
        result = apply_penalty(Decimal("100"), 1.0, floor=Decimal("10"))
        assert result == Decimal("10.00")

    def test_none_returns_none(self):
        """None score should return None."""
        result = apply_penalty(None, 0.10)
        assert result is None


class TestApplyMultiplier:
    """Tests for apply_multiplier function."""

    def test_double_multiplier(self):
        """2x multiplier should double score (capped)."""
        result = apply_multiplier(Decimal("30"), 2.0)
        assert result == Decimal("60.00")

    def test_multiplier_with_cap(self):
        """Multiplier should be capped at max."""
        result = apply_multiplier(Decimal("80"), 2.0)
        assert result == Decimal("100.00")  # Capped at 100

    def test_custom_cap(self):
        """Custom cap should be respected."""
        result = apply_multiplier(Decimal("80"), 2.0, cap=Decimal("150"))
        assert result == Decimal("150.00")

    def test_none_returns_none(self):
        """None score should return None."""
        result = apply_multiplier(None, 2.0)
        assert result is None


class TestClampAllScores:
    """Tests for clamp_all_scores function."""

    def test_clamps_all_records(self):
        """All records should be clamped."""
        records = [
            {"ticker": "A", "score": 150},
            {"ticker": "B", "score": -10},
            {"ticker": "C", "score": 50},
        ]
        result = clamp_all_scores(records, ["score"])
        assert result[0]["score"] == Decimal("100.00")
        assert result[1]["score"] == Decimal("0.00")
        assert result[2]["score"] == Decimal("50.00")

    def test_skips_missing_keys(self):
        """Missing keys should be skipped."""
        records = [
            {"ticker": "A", "score": 50},
            {"ticker": "B"},  # No score
        ]
        result = clamp_all_scores(records, ["score"])
        assert "score" not in result[1]


class TestValidateScoreBounds:
    """Tests for validate_score_bounds function."""

    def test_within_bounds(self):
        """Score within bounds should pass."""
        is_valid, error = validate_score_bounds(Decimal("50"))
        assert is_valid is True
        assert error is None

    def test_below_bounds(self):
        """Score below bounds should fail."""
        is_valid, error = validate_score_bounds(Decimal("-10"))
        assert is_valid is False
        assert "below minimum" in error

    def test_above_bounds(self):
        """Score above bounds should fail."""
        is_valid, error = validate_score_bounds(Decimal("150"))
        assert is_valid is False
        assert "above maximum" in error

    def test_none_is_valid(self):
        """None score should be valid."""
        is_valid, error = validate_score_bounds(None)
        assert is_valid is True

    def test_custom_score_name(self):
        """Custom score name should appear in error."""
        is_valid, error = validate_score_bounds(
            Decimal("-10"),
            score_name="clinical_score"
        )
        assert "clinical_score" in error


class TestConstants:
    """Tests for module constants."""

    def test_default_bounds(self):
        """Default bounds should be 0-100."""
        assert DEFAULT_MIN_SCORE == Decimal("0")
        assert DEFAULT_MAX_SCORE == Decimal("100")

    def test_precision(self):
        """Score precision should be 2 decimal places."""
        assert SCORE_PRECISION == Decimal("0.01")
