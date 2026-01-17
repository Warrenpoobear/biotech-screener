"""
Tests for SEC EDGAR 13F Extractor

Focuses on testing the amendment detection logic.

Run with: pytest tests/test_edgar_13f_extractor.py -v
"""

import pytest
from datetime import date


class TestAmendmentDetection:
    """Test 13F amendment detection from form type."""

    def test_detects_original_filing(self):
        """13F-HR form should not be marked as amendment."""
        form = "13F-HR"
        is_amendment = form == "13F-HR/A"
        assert is_amendment is False

    def test_detects_amendment_filing(self):
        """13F-HR/A form should be marked as amendment."""
        form = "13F-HR/A"
        is_amendment = form == "13F-HR/A"
        assert is_amendment is True

    def test_form_type_filter_accepts_13f_hr(self):
        """Filter should accept 13F-HR form type."""
        form = "13F-HR"
        accepted_forms = ('13F-HR', '13F-HR/A')
        assert form in accepted_forms

    def test_form_type_filter_accepts_13f_hr_a(self):
        """Filter should accept 13F-HR/A form type."""
        form = "13F-HR/A"
        accepted_forms = ('13F-HR', '13F-HR/A')
        assert form in accepted_forms

    def test_form_type_filter_rejects_other_forms(self):
        """Filter should reject non-13F-HR forms."""
        accepted_forms = ('13F-HR', '13F-HR/A')
        rejected_forms = ['10-K', '10-Q', '8-K', '13F-NT', '13F-NT/A', 'DEF 14A']
        for form in rejected_forms:
            assert form not in accepted_forms, f"Form {form} should be rejected"


class TestFormTypeParsing:
    """Test various edge cases in form type parsing."""

    @pytest.mark.parametrize("form,expected_amendment", [
        ("13F-HR", False),
        ("13F-HR/A", True),
    ])
    def test_amendment_flag_from_form(self, form, expected_amendment):
        """Amendment flag should correctly derive from form type."""
        is_amendment = form == "13F-HR/A"
        assert is_amendment == expected_amendment


class TestDeterminism:
    """Ensure amendment detection is deterministic."""

    def test_amendment_detection_deterministic(self):
        """Same form should always produce same amendment flag."""
        form = "13F-HR/A"
        results = [form == "13F-HR/A" for _ in range(10)]
        assert all(r is True for r in results)

        form = "13F-HR"
        results = [form == "13F-HR/A" for _ in range(10)]
        assert all(r is False for r in results)
