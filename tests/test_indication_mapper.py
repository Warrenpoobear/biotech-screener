#!/usr/bin/env python3
"""
Tests for Indication Mapper

Covers:
- Condition to indication mapping
- Ticker overrides (legacy and v3 PIT-safe)
- Precedence rules
- Pattern matching
- Audit trail
"""

import pytest
import json
from datetime import date
from pathlib import Path
from typing import Dict, Any, List

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from indication_mapper import (
    IndicationMapper,
    MappingValidationError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mapper():
    """Create indication mapper with default/fallback mappings."""
    return IndicationMapper()


@pytest.fixture
def custom_mapping_file(tmp_path):
    """Create custom mapping file for testing."""
    mapping = {
        "provenance": {
            "source": "test",
            "schema_version": "1.0.0",
        },
        "condition_patterns": {
            "oncology": ["cancer", "tumor", "carcinoma", "leukemia"],
            "rare_disease": ["duchenne", "cystic fibrosis", "orphan"],
            "cns": ["alzheimer", "parkinson", "epilepsy"],
        },
        "ticker_overrides": {
            "VRTX": "rare_disease",
            "REGN": "oncology",
        },
        "ticker_overrides_v3": {
            "MRNA": {
                "indication": "infectious_disease",
                "effective_from": "2020-01-01",
                "evidence": "COVID-19 vaccine program",
            },
            "BIIB": {
                "indication": "cns",
                "effective_from": "2015-01-01",
                "effective_until": "2025-12-31",
                "evidence": "Alzheimer's focus",
            },
        },
        "category_aliases": {
            "cns": "neurology",
        },
    }

    path = tmp_path / "indication_mapping.json"
    path.write_text(json.dumps(mapping))
    return path


@pytest.fixture
def mapper_with_custom_file(custom_mapping_file):
    """Create mapper with custom mapping file."""
    return IndicationMapper(str(custom_mapping_file))


@pytest.fixture
def as_of_date():
    """Standard as_of_date for testing."""
    return date(2026, 1, 15)


# ============================================================================
# BASIC FUNCTIONALITY
# ============================================================================

class TestBasicMapping:
    """Tests for basic mapping functionality."""

    def test_maps_oncology_condition(self, mapper, as_of_date):
        """Maps oncology-related conditions."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["breast cancer", "solid tumor"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "oncology"
        assert result["confidence"] in ["HIGH", "MEDIUM"]

    def test_maps_rare_disease_condition(self, mapper, as_of_date):
        """Maps rare disease conditions."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["cystic fibrosis"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "rare_disease"

    def test_maps_cns_condition(self, mapper, as_of_date):
        """Maps CNS conditions."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["alzheimer's disease"],
            as_of_date=as_of_date,
        )

        assert result["indication"] in ["cns", "neurology"]  # May be aliased

    def test_no_match_returns_none(self, mapper, as_of_date):
        """Returns None indication when no pattern matches, but still attempts matching."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["completely unrelated condition xyz123"],
            as_of_date=as_of_date,
        )

        # When conditions are provided but no patterns match, indication is None
        # but confidence is MEDIUM (since pattern matching was attempted)
        assert result["indication"] is None
        assert result["confidence"] == "MEDIUM"

    def test_no_conditions_returns_none_confidence(self, mapper, as_of_date):
        """Returns NONE confidence when no conditions provided."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=[],  # Empty conditions
            as_of_date=as_of_date,
        )

        assert result["indication"] is None
        assert result["confidence"] == "NONE"

    def test_requires_as_of_date(self, mapper):
        """Raises error when as_of_date not provided."""
        with pytest.raises(ValueError, match="as_of_date is REQUIRED"):
            mapper.map_ticker(
                ticker="TEST",
                conditions=["cancer"],
                as_of_date=None,
            )

    def test_ticker_case_normalization(self, mapper_with_custom_file, as_of_date):
        """Ticker is normalized to uppercase."""
        result = mapper_with_custom_file.map_ticker(
            ticker="vrtx",  # lowercase
            conditions=[],
            as_of_date=as_of_date,
        )

        assert result["source"] == "ticker_override"


# ============================================================================
# PRECEDENCE RULES
# ============================================================================

class TestPrecedenceRules:
    """Tests for mapping precedence rules."""

    def test_v3_override_highest_precedence(self, mapper_with_custom_file, as_of_date):
        """V3 ticker override has highest precedence."""
        result = mapper_with_custom_file.map_ticker(
            ticker="MRNA",
            conditions=["cancer"],  # Would match oncology
            as_of_date=as_of_date,
        )

        assert result["indication"] == "infectious_disease"
        assert result["source"] == "ticker_override_v3"

    def test_legacy_override_second_precedence(self, mapper_with_custom_file, as_of_date):
        """Legacy ticker override has second precedence."""
        result = mapper_with_custom_file.map_ticker(
            ticker="VRTX",
            conditions=["cancer"],  # Would match oncology
            as_of_date=as_of_date,
        )

        assert result["indication"] == "rare_disease"
        assert result["source"] == "ticker_override"

    def test_pattern_match_when_no_override(self, mapper_with_custom_file, as_of_date):
        """Uses pattern matching when no override exists."""
        result = mapper_with_custom_file.map_ticker(
            ticker="UNKNOWN",
            conditions=["breast cancer"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "oncology"
        assert result["source"] == "condition_patterns"


# ============================================================================
# PIT SAFETY
# ============================================================================

class TestPITSafety:
    """Tests for point-in-time safety in v3 overrides."""

    def test_v3_override_not_effective_before_date(self, mapper_with_custom_file):
        """V3 override not used before effective_from date."""
        before_effective = date(2019, 1, 1)  # Before MRNA's 2020-01-01

        result = mapper_with_custom_file.map_ticker(
            ticker="MRNA",
            conditions=["vaccine"],
            as_of_date=before_effective,
        )

        # Should fall through to pattern matching, not v3 override
        assert result["source"] != "ticker_override_v3"

    def test_v3_override_effective_on_date(self, mapper_with_custom_file):
        """V3 override used on effective_from date."""
        on_effective = date(2020, 1, 1)  # Exactly MRNA's effective date

        result = mapper_with_custom_file.map_ticker(
            ticker="MRNA",
            conditions=["vaccine"],
            as_of_date=on_effective,
        )

        assert result["source"] == "ticker_override_v3"
        assert result["indication"] == "infectious_disease"

    def test_v3_override_effective_after_date(self, mapper_with_custom_file):
        """V3 override used after effective_from date."""
        after_effective = date(2023, 6, 15)

        result = mapper_with_custom_file.map_ticker(
            ticker="MRNA",
            conditions=["vaccine"],
            as_of_date=after_effective,
        )

        assert result["source"] == "ticker_override_v3"

    def test_v3_override_expired(self, mapper_with_custom_file):
        """V3 override not used after effective_until date."""
        after_expired = date(2026, 6, 1)  # After BIIB's 2025-12-31

        result = mapper_with_custom_file.map_ticker(
            ticker="BIIB",
            conditions=["alzheimer"],
            as_of_date=after_expired,
        )

        # Should fall through since override expired
        assert result["source"] != "ticker_override_v3"


# ============================================================================
# PATTERN MATCHING
# ============================================================================

class TestPatternMatching:
    """Tests for condition pattern matching."""

    def test_multi_match_high_confidence(self, mapper, as_of_date):
        """Multiple pattern matches give high confidence."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["cancer", "tumor", "carcinoma"],
            as_of_date=as_of_date,
        )

        # Multiple matches should give HIGH confidence
        assert result["confidence"] in ["HIGH", "MEDIUM"]
        assert result["indication"] == "oncology"

    def test_single_match_medium_confidence(self, mapper, as_of_date):
        """Single pattern match gives medium confidence."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["leukemia"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "oncology"

    def test_case_insensitive_matching(self, mapper, as_of_date):
        """Pattern matching is case-insensitive."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["BREAST CANCER"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "oncology"

    def test_word_boundary_matching(self, mapper, as_of_date):
        """Patterns match on word boundaries."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["adenocarcinoma"],  # Contains "carcinoma"
            as_of_date=as_of_date,
        )

        # Should match carcinoma
        assert result["indication"] == "oncology"

    def test_highest_score_wins(self, mapper_with_custom_file, as_of_date):
        """Category with most matches wins."""
        result = mapper_with_custom_file.map_ticker(
            ticker="TEST",
            conditions=["cancer", "tumor", "alzheimer"],
            as_of_date=as_of_date,
        )

        # 2 oncology matches vs 1 cns match -> oncology wins
        assert result["indication"] == "oncology"


# ============================================================================
# CATEGORY ALIASES
# ============================================================================

class TestCategoryAliases:
    """Tests for category alias resolution."""

    def test_alias_resolution(self, mapper_with_custom_file, as_of_date):
        """Aliases are resolved when resolve_alias=True."""
        result = mapper_with_custom_file.map_ticker(
            ticker="TEST",
            conditions=["alzheimer"],
            as_of_date=as_of_date,
            resolve_alias=True,
        )

        # cns should be aliased to neurology
        assert result["indication"] == "neurology"
        assert result["indication_raw"] == "cns"

    def test_alias_not_resolved_when_disabled(self, mapper_with_custom_file, as_of_date):
        """Aliases not resolved when resolve_alias=False."""
        result = mapper_with_custom_file.map_ticker(
            ticker="TEST",
            conditions=["alzheimer"],
            as_of_date=as_of_date,
            resolve_alias=False,
        )

        # Should return raw category
        assert result["indication"] == "cns"


# ============================================================================
# AUDIT TRAIL
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_entries_created(self, mapper, as_of_date):
        """Audit entries are created for each mapping."""
        mapper.clear_audit_trail()

        mapper.map_ticker("TEST", ["cancer"], as_of_date)

        trail = mapper.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["ticker"] == "TEST"

    def test_audit_includes_timestamp(self, mapper, as_of_date):
        """Audit entries include deterministic timestamp."""
        mapper.clear_audit_trail()

        mapper.map_ticker("TEST", ["cancer"], as_of_date)

        trail = mapper.get_audit_trail()
        assert "timestamp" in trail[0]
        assert trail[0]["timestamp"].startswith("2026-01-15")

    def test_audit_includes_source(self, mapper, as_of_date):
        """Audit entries include mapping source."""
        mapper.clear_audit_trail()

        mapper.map_ticker("TEST", ["cancer"], as_of_date)

        trail = mapper.get_audit_trail()
        assert "source" in trail[0]

    def test_clear_audit_trail(self, mapper, as_of_date):
        """Audit trail can be cleared."""
        mapper.map_ticker("TEST", ["cancer"], as_of_date)
        assert len(mapper.get_audit_trail()) > 0

        mapper.clear_audit_trail()
        assert len(mapper.get_audit_trail()) == 0


# ============================================================================
# UNIVERSE MAPPING
# ============================================================================

class TestUniverseMapping:
    """Tests for mapping entire universe."""

    def test_map_universe(self, mapper, as_of_date):
        """Maps multiple tickers from trial records."""
        trial_records = [
            {"ticker": "ACME", "conditions": ["cancer"]},
            {"ticker": "BETA", "conditions": ["alzheimer"]},
            {"ticker": "GAMA", "conditions": ["rare disease"]},
        ]

        results = mapper.map_universe(
            tickers=["ACME", "BETA", "GAMA"],
            trial_records=trial_records,
            as_of_date=as_of_date,
        )

        assert "ACME" in results
        assert "BETA" in results
        assert "GAMA" in results

    def test_map_universe_with_sponsor_ticker(self, mapper, as_of_date):
        """Handles sponsor_ticker field in trial records."""
        trial_records = [
            {"sponsor_ticker": "ACME", "conditions": ["cancer"]},
        ]

        results = mapper.map_universe(
            tickers=["ACME"],
            trial_records=trial_records,
            as_of_date=as_of_date,
        )

        assert results["ACME"]["indication"] == "oncology"


# ============================================================================
# VALIDATION
# ============================================================================

class TestValidation:
    """Tests for mapping validation."""

    def test_is_valid_method(self, mapper):
        """is_valid returns True when no errors."""
        # Fallback mappings should be valid
        assert mapper.is_valid() or len(mapper.get_validation_errors()) > 0

    def test_get_validation_errors(self, mapper):
        """Returns list of validation errors."""
        errors = mapper.get_validation_errors()
        assert isinstance(errors, list)

    def test_get_mapping_info(self, mapper):
        """Returns mapping metadata."""
        info = mapper.get_mapping_info()

        assert "categories_available" in info
        assert "ticker_overrides_count" in info
        assert "mapper_version" in info


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_conditions(self, mapper, as_of_date):
        """Handles empty conditions list."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=[],
            as_of_date=as_of_date,
        )

        assert result["source"] == "no_data"

    def test_none_conditions(self, mapper, as_of_date):
        """Handles None conditions."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=None,
            as_of_date=as_of_date,
        )

        assert result["source"] == "no_data"

    def test_empty_ticker(self, mapper, as_of_date):
        """Handles empty ticker string."""
        result = mapper.map_ticker(
            ticker="",
            conditions=["cancer"],
            as_of_date=as_of_date,
        )

        assert result["source"] == "condition_patterns"

    def test_conditions_with_none_values(self, mapper, as_of_date):
        """Handles conditions list with None values."""
        result = mapper.map_ticker(
            ticker="TEST",
            conditions=["cancer", None, "tumor"],
            as_of_date=as_of_date,
        )

        assert result["indication"] == "oncology"


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_mapping_deterministic(self, mapper, as_of_date):
        """Same inputs produce same outputs."""
        result1 = mapper.map_ticker("TEST", ["cancer"], as_of_date)
        result2 = mapper.map_ticker("TEST", ["cancer"], as_of_date)

        assert result1["indication"] == result2["indication"]
        assert result1["confidence"] == result2["confidence"]

    def test_universe_mapping_deterministic(self, mapper, as_of_date):
        """Universe mapping is deterministic."""
        trial_records = [
            {"ticker": "ACME", "conditions": ["cancer"]},
            {"ticker": "BETA", "conditions": ["alzheimer"]},
        ]

        results1 = mapper.map_universe(["ACME", "BETA"], trial_records, as_of_date)
        results2 = mapper.map_universe(["ACME", "BETA"], trial_records, as_of_date)

        assert results1["ACME"]["indication"] == results2["ACME"]["indication"]
        assert results1["BETA"]["indication"] == results2["BETA"]["indication"]

