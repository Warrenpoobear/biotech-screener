"""Tests for ticker validation."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from validators.ticker_validator import is_valid_ticker, validate_ticker_list


class TestIsValidTicker:
    """Tests for is_valid_ticker function."""

    def test_valid_standard_tickers(self):
        """Test that standard valid tickers pass."""
        valid_cases = [
            "MRNA",
            "BMRN",
            "GILD",
            "BIIB",
            "VRTX",
            "ALNY",
            "A",      # Single letter
            "GOOGL",  # 5 letters
        ]

        for ticker in valid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert is_valid, f"'{ticker}' should be valid but got: {reason}"

    def test_valid_special_format_tickers(self):
        """Test that special format tickers pass."""
        valid_cases = [
            "BRK.B",  # Class B shares with dot
            "BF-B",   # Hyphenated
            "BRK.A",  # Class A shares
        ]

        for ticker in valid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert is_valid, f"'{ticker}' should be valid but got: {reason}"

    def test_invalid_empty_strings(self):
        """Test that empty/whitespace strings fail."""
        invalid_cases = [
            ("", "Empty string"),
            ("   ", "Empty after strip"),
            (None, "Empty string"),
        ]

        for ticker, expected_reason in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid"

    def test_invalid_too_long(self):
        """Test that long strings fail."""
        invalid_cases = [
            "TOOLONG",   # 7 chars
            "VERYLONGTICKER",
            "ABC123XYZ",
        ]

        for ticker in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid (too long)"
            assert "Too long" in reason

    def test_invalid_contamination_keywords(self):
        """Test that contamination keywords fail.

        Note: We use shorter keywords (<=6 chars) that pass the length check
        but fail the contamination keyword check. Longer contamination strings
        like 'BLACKROCK' or 'COPYRIGHT' are correctly rejected for being too long.
        """
        # Short keywords that pass length check but fail contamination check
        invalid_cases = [
            "FTSE",    # 4 chars - index provider
            "EPRA",    # 4 chars - index provider
            "INDEX",   # 5 chars - generic index term
            "OWNED",   # 5 chars - legal boilerplate
            "THE",     # 3 chars - boilerplate
        ]

        for ticker in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid (contamination)"
            assert "contamination keyword" in reason.lower(), f"'{ticker}' should fail with contamination keyword, got: {reason}"

    def test_invalid_special_char_positions(self):
        """Test that misplaced special characters fail."""
        invalid_cases = [
            (".ABC", "Starts with period"),
            ("ABC.", "Ends with period"),
            ("-ABC", "Starts with hyphen"),
            ("ABC-", "Ends with hyphen"),
        ]

        for ticker, expected_reason in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid"

    def test_invalid_excessive_special_chars(self):
        """Test that excessive special characters fail."""
        invalid_cases = [
            "A..B",
            "A--B",
            "A.-B",
        ]

        for ticker in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid (too many special chars)"

    def test_invalid_just_special_chars(self):
        """Test that just special characters fail."""
        invalid_cases = ["-", ".", "--", ".."]

        for ticker in invalid_cases:
            is_valid, reason = is_valid_ticker(ticker)
            assert not is_valid, f"'{ticker}' should be invalid"

    def test_blackrock_disclaimer(self):
        """Test that BlackRock disclaimer text fails."""
        disclaimer = "©2023 BLACKROCK, INC OR ITS AFFILIATES. ALL RIGHTS RESERVED."
        is_valid, reason = is_valid_ticker(disclaimer)
        assert not is_valid
        assert "Too long" in reason or "contamination" in reason.lower()


class TestValidateTickerList:
    """Tests for validate_ticker_list function."""

    def test_mixed_list(self):
        """Test validation of mixed valid/invalid tickers."""
        tickers = ["MRNA", "INVALID!", "GILD", "COPYRIGHT", "VRTX"]

        result = validate_ticker_list(tickers)

        assert len(result['valid']) == 3
        assert "MRNA" in result['valid']
        assert "GILD" in result['valid']
        assert "VRTX" in result['valid']

        assert len(result['invalid']) == 2
        assert "INVALID!" in result['invalid']
        assert "COPYRIGHT" in result['invalid']

    def test_stats(self):
        """Test that statistics are calculated correctly."""
        tickers = ["MRNA", "INVALID!", "GILD", "COPYRIGHT", "VRTX"]

        result = validate_ticker_list(tickers)

        assert result['stats']['total_input'] == 5
        assert result['stats']['valid_count'] == 3
        assert result['stats']['invalid_count'] == 2
        assert result['stats']['pass_rate'] == 0.6

    def test_empty_list(self):
        """Test validation of empty list."""
        result = validate_ticker_list([])

        assert len(result['valid']) == 0
        assert len(result['invalid']) == 0
        assert result['stats']['total_input'] == 0
        assert result['stats']['pass_rate'] == 0.0

    def test_all_valid(self):
        """Test validation of all valid tickers."""
        tickers = ["MRNA", "GILD", "VRTX", "BIIB"]

        result = validate_ticker_list(tickers)

        assert len(result['valid']) == 4
        assert len(result['invalid']) == 0
        assert result['stats']['pass_rate'] == 1.0

    def test_all_invalid(self):
        """Test validation of all invalid tickers."""
        tickers = ["COPYRIGHT", "BLACKROCK", "THE CONTENT"]

        result = validate_ticker_list(tickers)

        assert len(result['valid']) == 0
        assert len(result['invalid']) == 3
        assert result['stats']['pass_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
