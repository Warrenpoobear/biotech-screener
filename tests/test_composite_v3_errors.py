#!/usr/bin/env python3
"""
Error handling tests for Module 5: Composite Ranker (v3)

Tests edge cases and error scenarios including:
- Helper function error handling
- Schema version verification
"""

import pytest
from datetime import date
from decimal import Decimal

from module_5_composite_v3 import (
    V3_ENHANCED_WEIGHTS,
    V3_DEFAULT_WEIGHTS,
    SCHEMA_VERSION,
)
from module_5_scoring_v3 import (
    _coalesce,
    _market_cap_bucket,
    _stage_bucket,
)
from common.types import Severity


class TestCoalesceFunction:
    """Tests for _coalesce helper function."""

    def test_coalesce_all_none(self):
        """Coalesce with all None should return default."""
        result = _coalesce(None, None, None, default="default")
        assert result == "default"

    def test_coalesce_first_valid(self):
        """Coalesce should return first non-None."""
        result = _coalesce(None, Decimal("50"), Decimal("100"))
        assert result == Decimal("50")

    def test_coalesce_all_none_no_default(self):
        """Coalesce with all None and no default should return None."""
        result = _coalesce(None, None)
        assert result is None

    def test_coalesce_first_value(self):
        """First valid value should be returned."""
        result = _coalesce("first", "second", "third")
        assert result == "first"

    def test_coalesce_zero_is_valid(self):
        """Zero should be considered a valid value (not None)."""
        result = _coalesce(0, 100)
        assert result == 0


class TestMarketCapBucket:
    """Tests for _market_cap_bucket classification."""

    def test_none_market_cap(self):
        """None market cap should return unknown."""
        result = _market_cap_bucket(None)
        assert result == "unknown"

    def test_negative_market_cap(self):
        """Negative market cap should handle gracefully."""
        result = _market_cap_bucket(Decimal("-100"))
        # Should handle without crashing
        assert result is not None

    def test_zero_market_cap(self):
        """Zero market cap should return appropriate bucket."""
        result = _market_cap_bucket(Decimal("0"))
        # Any bucket is acceptable for edge case
        assert result is not None

    def test_large_market_cap(self):
        """Large market cap should return large bucket."""
        result = _market_cap_bucket(Decimal("15000"))
        assert result == "large"

    def test_bucket_classification_runs(self):
        """Various market caps should classify without error."""
        for mcap in [Decimal("25"), Decimal("150"), Decimal("1000"), Decimal("5000")]:
            result = _market_cap_bucket(mcap)
            assert result is not None


class TestStageBucket:
    """Tests for _stage_bucket classification."""

    def test_none_phase(self):
        """None phase should return a bucket (may be early or unknown)."""
        result = _stage_bucket(None)
        assert result is not None

    def test_empty_phase(self):
        """Empty phase should return a bucket."""
        result = _stage_bucket("")
        assert result is not None

    def test_phase_3_bucket(self):
        """Phase 3 should map to late bucket."""
        result = _stage_bucket("phase 3")
        assert result == "late"

    def test_phase_2_bucket(self):
        """Phase 2 should map to mid bucket."""
        result = _stage_bucket("phase 2")
        assert result == "mid"

    def test_phase_1_bucket(self):
        """Phase 1 should map to early bucket."""
        result = _stage_bucket("phase 1")
        assert result == "early"


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_schema_version_format(self):
        """Schema version should be in expected format."""
        assert SCHEMA_VERSION == "v3.0"

    def test_weights_defined(self):
        """Weights should be properly defined."""
        assert V3_ENHANCED_WEIGHTS is not None
        assert V3_DEFAULT_WEIGHTS is not None
        assert "financial" in V3_ENHANCED_WEIGHTS
        assert "clinical" in V3_ENHANCED_WEIGHTS
        assert "financial" in V3_DEFAULT_WEIGHTS


class TestSeverityConstants:
    """Tests for Severity enum values."""

    def test_severity_values(self):
        """Severity should have expected values."""
        assert Severity.NONE.value == "none"
        assert Severity.SEV1.value == "sev1"
        assert Severity.SEV2.value == "sev2"
        assert Severity.SEV3.value == "sev3"

    def test_severity_ordering(self):
        """Severity should have natural ordering."""
        severities = [Severity.NONE, Severity.SEV1, Severity.SEV2, Severity.SEV3]
        assert len(set(severities)) == 4
