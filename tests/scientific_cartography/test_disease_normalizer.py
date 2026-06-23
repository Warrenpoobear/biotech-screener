"""Tests for disease normalizer."""

import tempfile
from pathlib import Path

import pytest

from scientific_cartography.normalize.disease_normalizer import DiseaseNormalizer


class TestDiseaseNormalizerBasic:
    """Test disease normalizer basic functionality."""

    def test_normalizer_initializes(self):
        """Normalizer should initialize without error."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")
        assert normalizer.as_of_date == "2026-06-16"

    def test_unmapped_disease_stays_unmapped(self):
        """Unmapped disease should return with low confidence."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")
        result = normalizer.normalize("totally_unknown_disease_xyz")

        assert result.raw_name == "totally_unknown_disease_xyz"
        assert result.normalized_name == "totally_unknown_disease_xyz"
        assert result.confidence == 0.0
        assert result.source == "unmapped"

    def test_short_synonyms_do_not_substring_match_unrelated_diseases(self):
        """Short synonyms like AD should not map Prader-Willi to atopic dermatitis."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")

        result = normalizer.normalize("Prader-Willi Syndrome")

        assert result.raw_name == "Prader-Willi Syndrome"
        assert result.normalized_name == "Prader-Willi Syndrome"
        assert result.mondo_id is None
        assert result.confidence == 0.0
        assert result.source == "unmapped"

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")

        result1 = normalizer.normalize("unknown_disease")
        result2 = normalizer.normalize("UNKNOWN_DISEASE")
        result3 = normalizer.normalize("Unknown_Disease")

        # All should map to same cached result
        assert result1.disease_id == result2.disease_id == result3.disease_id

    def test_whitespace_normalized(self):
        """Whitespace should be normalized."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")

        result1 = normalizer.normalize("disease_name")
        result2 = normalizer.normalize("  disease_name  ")

        # Should hit cache
        assert result1.disease_id == result2.disease_id

    def test_bulk_normalize(self):
        """Bulk normalize should return list in order."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")
        diseases = ["unknown_1", "unknown_2", "unknown_3"]

        results = normalizer.bulk_normalize(diseases)

        assert len(results) == 3
        assert results[0].raw_name == "unknown_1"
        assert results[1].raw_name == "unknown_2"
        assert results[2].raw_name == "unknown_3"


class TestDiseaseNormalizerManualOverrides:
    """Test manual override functionality."""

    @pytest.fixture
    def manual_overrides_csv(self):
        """Create temporary manual overrides CSV."""
        csv_content = """raw_name,normalized_name,mondo_id,therapeutic_area,confidence,notes
Atopic Dermatitis,Atopic Dermatitis,MONDO:0004980,Dermatology,1.0,Primary indication
AD,Atopic Dermatitis,MONDO:0004980,Dermatology,0.95,Common abbreviation
eczema,Atopic Dermatitis,MONDO:0004980,Dermatology,0.80,Context-dependent synonym
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            return Path(f.name)

    def test_manual_override_wins_priority(self, manual_overrides_csv):
        """Manual overrides should have highest priority."""
        normalizer = DiseaseNormalizer(
            manual_overrides_csv=manual_overrides_csv,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("Atopic Dermatitis")

        assert result.raw_name == "Atopic Dermatitis"
        assert result.normalized_name == "Atopic Dermatitis"
        assert result.mondo_id == "MONDO:0004980"
        assert result.therapeutic_area == "Dermatology"
        assert result.confidence == 1.0
        assert result.source == "manual_override"

    def test_manual_override_case_insensitive(self, manual_overrides_csv):
        """Manual overrides should work case-insensitive."""
        normalizer = DiseaseNormalizer(
            manual_overrides_csv=manual_overrides_csv,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("atopic dermatitis")

        assert result.mondo_id == "MONDO:0004980"
        assert result.source == "manual_override"

    def test_manual_override_abbreviation(self, manual_overrides_csv):
        """Manual overrides should handle abbreviations."""
        normalizer = DiseaseNormalizer(
            manual_overrides_csv=manual_overrides_csv,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("AD")

        assert result.normalized_name == "Atopic Dermatitis"
        assert result.confidence == 0.95

    def test_manual_override_synonym(self, manual_overrides_csv):
        """Manual overrides should handle synonyms."""
        normalizer = DiseaseNormalizer(
            manual_overrides_csv=manual_overrides_csv,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("eczema")

        assert result.normalized_name == "Atopic Dermatitis"
        assert result.confidence == 0.80

    def test_nonexistent_override_file_ignored(self):
        """Nonexistent override file should be gracefully ignored."""
        normalizer = DiseaseNormalizer(
            manual_overrides_csv=Path("/nonexistent/path.csv"),
            as_of_date="2026-06-16",
        )

        # Should still work, just without overrides
        result = normalizer.normalize("unknown_disease")
        assert result.confidence == 0.0


class TestDiseaseNormalizerMONDOCache:
    """Test MONDO cache functionality."""

    @pytest.fixture
    def mondo_cache(self):
        """Create mock MONDO cache."""
        return {
            "atopic dermatitis": {
                "id": "MONDO:0004980",
                "name": "Atopic Dermatitis",
                "synonyms": ["AD", "eczema", "atopic eczema"],
                "therapeutic_area": "Dermatology",
            },
            "psoriasis": {
                "id": "MONDO:0005148",
                "name": "Psoriasis",
                "synonyms": ["psoriatic disease"],
                "therapeutic_area": "Dermatology",
            },
        }

    def test_mondo_exact_match(self, mondo_cache):
        """Exact MONDO match should use MONDO record."""
        normalizer = DiseaseNormalizer(
            mondo_cache=mondo_cache,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("atopic dermatitis")

        assert result.normalized_name == "Atopic Dermatitis"
        assert result.mondo_id == "MONDO:0004980"
        assert result.source == "mondo"
        assert result.confidence == 0.95

    def test_mondo_synonym_match(self, mondo_cache):
        """Synonym in MONDO should match."""
        normalizer = DiseaseNormalizer(
            mondo_cache=mondo_cache,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("eczema")

        assert result.normalized_name == "Atopic Dermatitis"
        assert result.mondo_id == "MONDO:0004980"
        assert result.source == "mondo_synonym"
        assert result.confidence == 0.90

    def test_mondo_case_insensitive(self, mondo_cache):
        """MONDO matching should be case-insensitive."""
        normalizer = DiseaseNormalizer(
            mondo_cache=mondo_cache,
            as_of_date="2026-06-16",
        )

        result = normalizer.normalize("ATOPIC DERMATITIS")

        assert result.mondo_id == "MONDO:0004980"


class TestDiseaseNormalizerCaching:
    """Test normalizer caching behavior."""

    def test_repeated_normalization_uses_cache(self):
        """Repeated normalizations should return cached result."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")

        result1 = normalizer.normalize("unknown_disease")
        result2 = normalizer.normalize("unknown_disease")

        # Should be the exact same object
        assert result1 is result2

    def test_cache_respects_confidence(self):
        """Cached results should preserve confidence."""
        normalizer = DiseaseNormalizer(as_of_date="2026-06-16")

        result = normalizer.normalize("unknown_disease")
        assert result.confidence == 0.0

        result2 = normalizer.normalize("unknown_disease")
        assert result2.confidence == 0.0


class TestDiseaseNormalizerMetadata:
    """Test metadata preservation."""

    def test_disease_record_serialization(self):
        """DiseaseRecord should serialize to dict."""
        from scientific_cartography.schemas.disease_schema import DiseaseRecord

        record = DiseaseRecord(
            disease_id="TEST_001",
            raw_name="Test Disease",
            normalized_name="Test Disease",
            mondo_id="MONDO:0000001",
            therapeutic_area="Test Area",
            confidence=0.95,
            source="manual_override",
            as_of_date="2026-06-16",
            source_refs=["ref1"],
        )

        d = record.to_dict()

        assert d["disease_id"] == "TEST_001"
        assert d["raw_name"] == "Test Disease"
        assert d["mondo_id"] == "MONDO:0000001"
        assert d["as_of_date"] == "2026-06-16"

    def test_disease_record_deserialization(self):
        """DiseaseRecord should deserialize from dict."""
        from scientific_cartography.schemas.disease_schema import DiseaseRecord

        data = {
            "disease_id": "TEST_001",
            "raw_name": "Test Disease",
            "normalized_name": "Test Disease",
            "mondo_id": "MONDO:0000001",
            "therapeutic_area": "Test Area",
            "confidence": 0.95,
            "source": "manual_override",
            "as_of_date": "2026-06-16",
            "source_refs": ["ref1"],
        }

        record = DiseaseRecord.from_dict(data)

        assert record.disease_id == "TEST_001"
        assert record.normalized_name == "Test Disease"
        assert record.as_of_date == "2026-06-16"
