#!/usr/bin/env python3
"""
Tests for common/null_safety.py

Null safety helpers are critical for preventing common bugs in financial calculations.
These tests cover:
- Null checks (is_present, is_missing, is_zero_or_none)
- Safe access (safe_get, safe_get_nested, safe_get_index)
- Coalesce/default functions
- Safe arithmetic (safe_divide, safe_multiply, safe_sum)
- Safe conversions (safe_int, safe_float, safe_decimal)
- Conditional execution (if_present, map_if_present)
- Collection safety (ensure_list, filter_none)
"""

import pytest
from decimal import Decimal

from common.null_safety import (
    # Null checks
    is_present,
    is_missing,
    is_zero_or_none,
    is_empty_or_none,
    # Safe access
    safe_get,
    safe_get_nested,
    safe_get_index,
    safe_get_first,
    safe_get_last,
    # Coalesce
    coalesce,
    default_if_none,
    default_if_empty,
    # Safe arithmetic
    safe_divide,
    safe_multiply,
    safe_sum,
    # Safe conversion
    safe_int,
    safe_float,
    safe_decimal,
    safe_str,
    # Conditional execution
    if_present,
    map_if_present,
    # Collection safety
    ensure_list,
    ensure_dict,
    filter_none,
    count_none,
    count_present,
)


class TestIsPresent:
    """Tests for is_present function."""

    def test_none_is_not_present(self):
        """None should not be present."""
        assert is_present(None) is False

    def test_zero_is_present(self):
        """Zero (int) should be present - critical for financial data."""
        assert is_present(0) is True

    def test_zero_float_is_present(self):
        """Zero (float) should be present."""
        assert is_present(0.0) is True

    def test_zero_decimal_is_present(self):
        """Zero (Decimal) should be present - critical for financial data."""
        assert is_present(Decimal("0")) is True

    def test_empty_string_is_present(self):
        """Empty string is present (not None)."""
        assert is_present("") is True

    def test_empty_list_is_present(self):
        """Empty list is present (not None)."""
        assert is_present([]) is True

    def test_empty_dict_is_present(self):
        """Empty dict is present (not None)."""
        assert is_present({}) is True

    def test_value_is_present(self):
        """Non-None value should be present."""
        assert is_present(1) is True
        assert is_present("hello") is True
        assert is_present([1, 2, 3]) is True


class TestIsMissing:
    """Tests for is_missing function."""

    def test_none_is_missing(self):
        """None should be missing."""
        assert is_missing(None) is True

    def test_zero_is_not_missing(self):
        """Zero should not be missing."""
        assert is_missing(0) is False

    def test_value_is_not_missing(self):
        """Non-None value should not be missing."""
        assert is_missing(1) is False
        assert is_missing("") is False


class TestIsZeroOrNone:
    """Tests for is_zero_or_none function."""

    def test_none_is_zero_or_none(self):
        """None should return True."""
        assert is_zero_or_none(None) is True

    def test_zero_int_is_zero_or_none(self):
        """Zero int should return True."""
        assert is_zero_or_none(0) is True

    def test_zero_float_is_zero_or_none(self):
        """Zero float should return True."""
        assert is_zero_or_none(0.0) is True

    def test_zero_decimal_is_zero_or_none(self):
        """Zero Decimal should return True."""
        assert is_zero_or_none(Decimal("0")) is True
        assert is_zero_or_none(Decimal("0.00")) is True

    def test_zero_string_is_zero_or_none(self):
        """String '0' should return True."""
        assert is_zero_or_none("0") is True
        assert is_zero_or_none("0.0") is True

    def test_non_zero_is_not_zero_or_none(self):
        """Non-zero values should return False."""
        assert is_zero_or_none(1) is False
        assert is_zero_or_none(-1) is False
        assert is_zero_or_none(Decimal("0.01")) is False

    def test_non_numeric_string_is_not_zero_or_none(self):
        """Non-numeric string should return False."""
        assert is_zero_or_none("abc") is False


class TestIsEmptyOrNone:
    """Tests for is_empty_or_none function."""

    def test_none_is_empty_or_none(self):
        """None should return True."""
        assert is_empty_or_none(None) is True

    def test_empty_string_is_empty_or_none(self):
        """Empty string should return True."""
        assert is_empty_or_none("") is True

    def test_empty_list_is_empty_or_none(self):
        """Empty list should return True."""
        assert is_empty_or_none([]) is True

    def test_empty_dict_is_empty_or_none(self):
        """Empty dict should return True."""
        assert is_empty_or_none({}) is True

    def test_empty_set_is_empty_or_none(self):
        """Empty set should return True."""
        assert is_empty_or_none(set()) is True

    def test_non_empty_string_is_not_empty_or_none(self):
        """Non-empty string should return False."""
        assert is_empty_or_none("hello") is False

    def test_non_empty_list_is_not_empty_or_none(self):
        """Non-empty list should return False."""
        assert is_empty_or_none([1]) is False

    def test_zero_is_not_empty_or_none(self):
        """Zero is not considered empty."""
        assert is_empty_or_none(0) is False


class TestSafeGet:
    """Tests for safe_get function."""

    def test_key_exists(self):
        """Existing key should return value."""
        assert safe_get({"a": 1}, "a") == 1

    def test_key_missing_no_default(self):
        """Missing key should return None by default."""
        assert safe_get({"a": 1}, "b") is None

    def test_key_missing_with_default(self):
        """Missing key should return specified default."""
        assert safe_get({"a": 1}, "b", default=0) == 0

    def test_none_mapping_returns_default(self):
        """None mapping should return default."""
        assert safe_get(None, "a", default=0) == 0

    def test_none_mapping_no_default(self):
        """None mapping with no default should return None."""
        assert safe_get(None, "a") is None

    def test_value_is_zero(self):
        """Should correctly return zero value."""
        assert safe_get({"a": 0}, "a", default=999) == 0

    def test_value_is_none(self):
        """Should correctly return None if that's the stored value."""
        assert safe_get({"a": None}, "a", default=999) is None


class TestSafeGetNested:
    """Tests for safe_get_nested function."""

    def test_single_level(self):
        """Single level access should work."""
        assert safe_get_nested({"a": 1}, ["a"]) == 1

    def test_two_levels(self):
        """Two level access should work."""
        assert safe_get_nested({"a": {"b": 2}}, ["a", "b"]) == 2

    def test_three_levels(self):
        """Three level access should work."""
        assert safe_get_nested({"a": {"b": {"c": 3}}}, ["a", "b", "c"]) == 3

    def test_missing_intermediate_key(self):
        """Missing intermediate key should return default."""
        assert safe_get_nested({"a": {"b": 1}}, ["a", "x", "c"], default=0) == 0

    def test_missing_final_key(self):
        """Missing final key should return default."""
        assert safe_get_nested({"a": {"b": 1}}, ["a", "c"], default=0) == 0

    def test_none_data_returns_default(self):
        """None data should return default."""
        assert safe_get_nested(None, ["a", "b"], default=0) == 0

    def test_non_dict_intermediate(self):
        """Non-dict intermediate should return default."""
        assert safe_get_nested({"a": "string"}, ["a", "b"], default=0) == 0

    def test_empty_keys(self):
        """Empty keys should return the data itself."""
        data = {"a": 1}
        assert safe_get_nested(data, []) == data


class TestSafeGetIndex:
    """Tests for safe_get_index function."""

    def test_valid_index(self):
        """Valid index should return element."""
        assert safe_get_index([1, 2, 3], 0) == 1
        assert safe_get_index([1, 2, 3], 2) == 3

    def test_negative_index(self):
        """Negative index should work."""
        assert safe_get_index([1, 2, 3], -1) == 3

    def test_out_of_bounds(self):
        """Out of bounds should return default."""
        assert safe_get_index([1, 2, 3], 10, default=0) == 0

    def test_none_sequence(self):
        """None sequence should return default."""
        assert safe_get_index(None, 0, default=0) == 0

    def test_empty_sequence(self):
        """Empty sequence should return default."""
        assert safe_get_index([], 0, default=0) == 0


class TestSafeGetFirst:
    """Tests for safe_get_first function."""

    def test_non_empty_list(self):
        """Non-empty list should return first element."""
        assert safe_get_first([1, 2, 3]) == 1

    def test_empty_list(self):
        """Empty list should return default."""
        assert safe_get_first([], default=0) == 0

    def test_none_list(self):
        """None should return default."""
        assert safe_get_first(None, default=0) == 0


class TestSafeGetLast:
    """Tests for safe_get_last function."""

    def test_non_empty_list(self):
        """Non-empty list should return last element."""
        assert safe_get_last([1, 2, 3]) == 3

    def test_empty_list(self):
        """Empty list should return default."""
        assert safe_get_last([], default=0) == 0


class TestCoalesce:
    """Tests for coalesce function."""

    def test_first_non_none(self):
        """Should return first non-None value."""
        assert coalesce(None, None, 1, 2) == 1

    def test_all_none(self):
        """All None should return None."""
        assert coalesce(None, None, None) is None

    def test_first_value_non_none(self):
        """First value non-None should return it."""
        assert coalesce(1, 2, 3) == 1

    def test_zero_is_valid(self):
        """Zero should be returned (not treated as None)."""
        assert coalesce(None, 0, 1) == 0

    def test_empty_string_is_valid(self):
        """Empty string should be returned."""
        assert coalesce(None, "", "hello") == ""


class TestDefaultIfNone:
    """Tests for default_if_none function."""

    def test_value_not_none(self):
        """Non-None value should be returned."""
        assert default_if_none(5, 0) == 5

    def test_value_none(self):
        """None should return default."""
        assert default_if_none(None, 0) == 0

    def test_zero_not_replaced(self):
        """Zero should not be replaced."""
        assert default_if_none(0, 999) == 0


class TestDefaultIfEmpty:
    """Tests for default_if_empty function."""

    def test_non_empty_string(self):
        """Non-empty string should be returned."""
        assert default_if_empty("hello", "default") == "hello"

    def test_empty_string(self):
        """Empty string should return default."""
        assert default_if_empty("", "default") == "default"

    def test_none(self):
        """None should return default."""
        assert default_if_empty(None, "default") == "default"

    def test_non_empty_list(self):
        """Non-empty list should be returned."""
        assert default_if_empty([1, 2], []) == [1, 2]


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_basic_division(self):
        """Basic division should work."""
        result = safe_divide(Decimal("10"), Decimal("2"))
        assert result == Decimal("5")

    def test_division_by_zero(self):
        """Division by zero should return default."""
        assert safe_divide(Decimal("10"), Decimal("0")) is None
        assert safe_divide(Decimal("10"), Decimal("0"), default=Decimal("0")) == Decimal("0")

    def test_division_by_none(self):
        """Division by None should return default."""
        assert safe_divide(Decimal("10"), None) is None

    def test_none_numerator(self):
        """None numerator should return default."""
        assert safe_divide(None, Decimal("2")) is None

    def test_int_inputs(self):
        """Integer inputs should be converted."""
        result = safe_divide(10, 2)
        assert result == Decimal("5")

    def test_float_inputs(self):
        """Float inputs should be converted."""
        result = safe_divide(10.0, 2.0)
        assert result == Decimal("5")

    def test_near_zero_denominator(self):
        """Near-zero denominator (< epsilon) should return default."""
        result = safe_divide(Decimal("10"), Decimal("0.0000001"))
        assert result is None

    def test_small_but_valid_denominator(self):
        """Small but valid denominator should work."""
        result = safe_divide(Decimal("10"), Decimal("0.001"))
        assert result == Decimal("10000")


class TestSafeMultiply:
    """Tests for safe_multiply function."""

    def test_basic_multiply(self):
        """Basic multiplication should work."""
        result = safe_multiply(Decimal("2"), Decimal("3"), Decimal("4"))
        assert result == Decimal("24")

    def test_any_none_returns_default(self):
        """Any None value should return default."""
        assert safe_multiply(Decimal("2"), None, Decimal("4")) is None
        assert safe_multiply(None, Decimal("3")) is None

    def test_empty_returns_default(self):
        """No arguments should return default."""
        assert safe_multiply() is None

    def test_single_value(self):
        """Single value should return that value."""
        assert safe_multiply(Decimal("5")) == Decimal("5")

    def test_with_default(self):
        """Should use provided default."""
        assert safe_multiply(Decimal("2"), None, default=Decimal("0")) == Decimal("0")


class TestSafeSum:
    """Tests for safe_sum function."""

    def test_basic_sum(self):
        """Basic sum should work."""
        result = safe_sum([Decimal("1"), Decimal("2"), Decimal("3")])
        assert result == Decimal("6")

    def test_skip_none(self):
        """With skip_none=True, None values should be skipped."""
        result = safe_sum([Decimal("1"), None, Decimal("3")], skip_none=True)
        assert result == Decimal("4")

    def test_no_skip_none(self):
        """With skip_none=False, any None should return default."""
        result = safe_sum([Decimal("1"), None, Decimal("3")], skip_none=False)
        assert result == Decimal("0")  # Default

    def test_empty_list(self):
        """Empty list should return default."""
        result = safe_sum([])
        assert result == Decimal("0")

    def test_all_none(self):
        """All None should return 0 with skip_none=True."""
        result = safe_sum([None, None])
        assert result == Decimal("0")

    def test_mixed_types(self):
        """Mixed numeric types should work."""
        result = safe_sum([1, 2.0, Decimal("3")])
        assert result == Decimal("6")


class TestSafeInt:
    """Tests for safe_int function."""

    def test_int_input(self):
        """Int input should return int."""
        assert safe_int(5) == 5

    def test_float_input(self):
        """Float input should be truncated."""
        assert safe_int(5.9) == 5

    def test_decimal_input(self):
        """Decimal input should be converted."""
        assert safe_int(Decimal("5.5")) == 5

    def test_string_input(self):
        """String number should be converted."""
        assert safe_int("5") == 5

    def test_none_input(self):
        """None should return default."""
        assert safe_int(None, default=0) == 0

    def test_invalid_string(self):
        """Invalid string should return default."""
        assert safe_int("abc", default=0) == 0


class TestSafeFloat:
    """Tests for safe_float function."""

    def test_float_input(self):
        """Float input should return float."""
        assert safe_float(5.5) == 5.5

    def test_int_input(self):
        """Int input should be converted."""
        assert safe_float(5) == 5.0

    def test_string_input(self):
        """String number should be converted."""
        assert safe_float("5.5") == 5.5

    def test_none_input(self):
        """None should return default."""
        assert safe_float(None, default=0.0) == 0.0


class TestSafeDecimal:
    """Tests for safe_decimal function."""

    def test_decimal_input(self):
        """Decimal input should return Decimal."""
        assert safe_decimal(Decimal("5.5")) == Decimal("5.5")

    def test_int_input(self):
        """Int input should be converted."""
        assert safe_decimal(5) == Decimal("5")

    def test_float_input(self):
        """Float input should be converted via string."""
        assert safe_decimal(5.5) == Decimal("5.5")

    def test_string_input(self):
        """String number should be converted."""
        assert safe_decimal("5.5") == Decimal("5.5")

    def test_none_input(self):
        """None should return default."""
        assert safe_decimal(None, default=Decimal("0")) == Decimal("0")

    def test_bool_input_returns_default(self):
        """Bool should return default to prevent True->1 conversion."""
        assert safe_decimal(True, default=Decimal("-1")) == Decimal("-1")

    def test_invalid_string(self):
        """Invalid string should return default."""
        assert safe_decimal("abc", default=Decimal("0")) == Decimal("0")


class TestSafeStr:
    """Tests for safe_str function."""

    def test_string_input(self):
        """String input should return string."""
        assert safe_str("hello") == "hello"

    def test_int_input(self):
        """Int input should be converted."""
        assert safe_str(5) == "5"

    def test_none_input(self):
        """None should return default."""
        assert safe_str(None, default="N/A") == "N/A"


class TestIfPresent:
    """Tests for if_present function."""

    def test_value_present(self):
        """Non-None value should have function applied."""
        result = if_present(5, lambda x: x * 2)
        assert result == 10

    def test_value_none(self):
        """None should return default."""
        result = if_present(None, lambda x: x * 2, default=0)
        assert result == 0

    def test_function_returning_none(self):
        """Function can return None."""
        result = if_present(5, lambda x: None)
        assert result is None


class TestMapIfPresent:
    """Tests for map_if_present function."""

    def test_no_none_values(self):
        """List without None should have function applied to all."""
        result = map_if_present([1, 2, 3], lambda x: x * 2)
        assert result == [2, 4, 6]

    def test_with_none_values(self):
        """None values should be preserved."""
        result = map_if_present([1, None, 3], lambda x: x * 2)
        assert result == [2, None, 6]

    def test_all_none(self):
        """All None should return all None."""
        result = map_if_present([None, None], lambda x: x * 2)
        assert result == [None, None]


class TestEnsureList:
    """Tests for ensure_list function."""

    def test_none_returns_empty_list(self):
        """None should return empty list."""
        assert ensure_list(None) == []

    def test_list_returns_same_list(self):
        """List should return same list."""
        assert ensure_list([1, 2]) == [1, 2]

    def test_single_value_wrapped(self):
        """Single value should be wrapped in list."""
        assert ensure_list(5) == [5]
        assert ensure_list("hello") == ["hello"]


class TestEnsureDict:
    """Tests for ensure_dict function."""

    def test_none_returns_empty_dict(self):
        """None should return empty dict."""
        assert ensure_dict(None) == {}

    def test_dict_returns_same_dict(self):
        """Dict should return same dict."""
        assert ensure_dict({"a": 1}) == {"a": 1}


class TestFilterNone:
    """Tests for filter_none function."""

    def test_removes_none_values(self):
        """None values should be removed."""
        assert filter_none([1, None, 2, None, 3]) == [1, 2, 3]

    def test_no_none_values(self):
        """List without None should be unchanged."""
        assert filter_none([1, 2, 3]) == [1, 2, 3]

    def test_all_none(self):
        """All None should return empty list."""
        assert filter_none([None, None]) == []

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert filter_none([]) == []

    def test_preserves_zero(self):
        """Zero should not be filtered."""
        assert filter_none([0, None, 1]) == [0, 1]


class TestCountNone:
    """Tests for count_none function."""

    def test_counts_none(self):
        """Should correctly count None values."""
        assert count_none([1, None, 2, None, 3]) == 2

    def test_no_none(self):
        """No None should return 0."""
        assert count_none([1, 2, 3]) == 0

    def test_all_none(self):
        """All None should return length."""
        assert count_none([None, None, None]) == 3


class TestCountPresent:
    """Tests for count_present function."""

    def test_counts_present(self):
        """Should correctly count non-None values."""
        assert count_present([1, None, 2, None, 3]) == 3

    def test_all_present(self):
        """All present should return length."""
        assert count_present([1, 2, 3]) == 3

    def test_none_present(self):
        """All None should return 0."""
        assert count_present([None, None]) == 0


class TestFinancialEdgeCases:
    """Tests for financial calculation edge cases using null safety."""

    def test_runway_with_zero_burn(self):
        """Runway calculation with zero burn should return None, not infinity."""
        cash = Decimal("500000000")
        burn = Decimal("0")
        runway = safe_divide(cash, burn)
        assert runway is None  # Not infinity

    def test_runway_with_none_cash(self):
        """Runway calculation with missing cash should return None."""
        cash = None
        burn = Decimal("50000000")
        runway = safe_divide(cash, burn)
        assert runway is None

    def test_dilution_with_zero_shares(self):
        """Dilution calculation with zero shares should handle gracefully."""
        new_shares = Decimal("10000000")
        old_shares = Decimal("0")
        dilution = safe_divide(new_shares - old_shares, old_shares)
        assert dilution is None

    def test_market_cap_components(self):
        """Market cap with any None component should handle gracefully."""
        price = Decimal("45.50")
        shares = None
        market_cap = safe_multiply(price, shares)
        assert market_cap is None

    def test_pe_ratio_with_negative_earnings(self):
        """P/E ratio calculation should work with negative earnings."""
        price = Decimal("45.50")
        eps = Decimal("-2.50")
        pe = safe_divide(price, eps)
        assert pe == Decimal("-18.2")  # Valid negative P/E

    def test_weighted_average_with_partial_data(self):
        """Weighted average should skip None values."""
        scores = [Decimal("80"), None, Decimal("60")]
        weights = [Decimal("0.4"), Decimal("0.3"), Decimal("0.3")]

        # Filter out None scores and corresponding weights
        valid_pairs = [(s, w) for s, w in zip(scores, weights) if s is not None]
        weighted_sum = safe_sum([s * w for s, w in valid_pairs])
        total_weight = safe_sum([w for _, w in valid_pairs])

        result = safe_divide(weighted_sum, total_weight)
        assert result == Decimal("50") / Decimal("0.7")  # ~71.43
