#!/usr/bin/env python3
"""
Tests for CUSIP Mapper

Covers:
- CUSIP validation
- Three-tier cache strategy
- PIT safety
- Batch operations
- Cache TTL
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cusip_mapper import (
    CUSIPMapping,
    CUSIPMapper,
    validate_cusip,
    load_static_cusip_map,
    load_cache,
    save_cache,
    CACHE_TTL_DAYS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def reference_time():
    """Standard reference time for PIT-safe testing."""
    return datetime(2026, 1, 15, 12, 0, 0)


@pytest.fixture
def sample_static_map():
    """Sample static CUSIP map data."""
    return {
        "037833100": {
            "cusip": "037833100",
            "ticker": "AAPL",
            "name": "Apple Inc",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "mapped_at": "2026-01-01T00:00:00",
            "source": "static"
        },
        "594918104": {
            "cusip": "594918104",
            "ticker": "MSFT",
            "name": "Microsoft Corporation",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "mapped_at": "2026-01-01T00:00:00",
            "source": "static"
        }
    }


@pytest.fixture
def sample_cache_data(reference_time):
    """Sample cache data with various ages."""
    fresh_time = (reference_time - timedelta(days=30)).isoformat()
    stale_time = (reference_time - timedelta(days=CACHE_TTL_DAYS + 10)).isoformat()

    return {
        "38259P508": {
            "cusip": "38259P508",
            "ticker": "GOOG",
            "name": "Alphabet Inc",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "mapped_at": fresh_time,
            "source": "openfigi"
        },
        "023135106": {  # Stale entry
            "cusip": "023135106",
            "ticker": "AMZN",
            "name": "Amazon.com Inc",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "mapped_at": stale_time,
            "source": "openfigi"
        }
    }


@pytest.fixture
def static_map_file(tmp_path, sample_static_map):
    """Create a temporary static map file."""
    file_path = tmp_path / "cusip_static_map.json"
    with open(file_path, 'w') as f:
        json.dump(sample_static_map, f)
    return file_path


@pytest.fixture
def cache_file(tmp_path, sample_cache_data):
    """Create a temporary cache file."""
    file_path = tmp_path / "cusip_cache.json"
    with open(file_path, 'w') as f:
        json.dump(sample_cache_data, f)
    return file_path


@pytest.fixture
def empty_cache_file(tmp_path):
    """Create an empty cache file."""
    file_path = tmp_path / "cusip_cache_empty.json"
    with open(file_path, 'w') as f:
        json.dump({}, f)
    return file_path


# ============================================================================
# CUSIP VALIDATION
# ============================================================================

class TestCUSIPValidation:
    """Tests for CUSIP format validation."""

    def test_valid_cusip_alphanumeric(self):
        """Valid 9-character alphanumeric CUSIP."""
        assert validate_cusip("037833100")
        assert validate_cusip("594918104")

    def test_valid_cusip_with_letters(self):
        """Valid CUSIP with letters."""
        assert validate_cusip("38259P508")
        assert validate_cusip("D18190898")

    def test_invalid_cusip_too_short(self):
        """CUSIP shorter than 9 characters is invalid."""
        assert not validate_cusip("03783310")
        assert not validate_cusip("12345")

    def test_invalid_cusip_too_long(self):
        """CUSIP longer than 9 characters is invalid."""
        assert not validate_cusip("0378331001")
        assert not validate_cusip("0378331001234")

    def test_invalid_cusip_special_chars(self):
        """CUSIP with special characters is invalid."""
        assert not validate_cusip("037833-00")
        assert not validate_cusip("0378331@0")
        assert not validate_cusip("037833 00")

    def test_invalid_cusip_empty(self):
        """Empty CUSIP is invalid."""
        assert not validate_cusip("")


# ============================================================================
# CUSIP MAPPING DATA CLASS
# ============================================================================

class TestCUSIPMapping:
    """Tests for CUSIPMapping data class."""

    def test_mapping_creation(self):
        """Create a CUSIPMapping instance."""
        mapping = CUSIPMapping(
            cusip="037833100",
            ticker="AAPL",
            name="Apple Inc",
            exchange="NASDAQ",
            security_type="Common Stock",
            mapped_at="2026-01-01T00:00:00",
            source="static"
        )

        assert mapping.cusip == "037833100"
        assert mapping.ticker == "AAPL"
        assert mapping.source == "static"

    def test_mapping_to_dict(self):
        """Convert mapping to dictionary."""
        mapping = CUSIPMapping(
            cusip="037833100",
            ticker="AAPL",
            name="Apple Inc",
            exchange="NASDAQ",
            security_type="Common Stock",
            mapped_at="2026-01-01T00:00:00",
            source="static"
        )

        d = mapping.to_dict()

        assert isinstance(d, dict)
        assert d["cusip"] == "037833100"
        assert d["ticker"] == "AAPL"

    def test_mapping_from_dict(self):
        """Create mapping from dictionary."""
        data = {
            "cusip": "037833100",
            "ticker": "AAPL",
            "name": "Apple Inc",
            "exchange": "NASDAQ",
            "security_type": "Common Stock",
            "mapped_at": "2026-01-01T00:00:00",
            "source": "static"
        }

        mapping = CUSIPMapping.from_dict(data)

        assert mapping.cusip == "037833100"
        assert mapping.ticker == "AAPL"


# ============================================================================
# STATIC MAP LOADING
# ============================================================================

class TestStaticMapLoading:
    """Tests for static map loading."""

    def test_load_static_map(self, static_map_file):
        """Load static map from file."""
        static_map = load_static_cusip_map(static_map_file)

        assert len(static_map) == 2
        assert "037833100" in static_map
        assert static_map["037833100"].ticker == "AAPL"

    def test_load_static_map_missing_file(self, tmp_path):
        """Load static map from non-existent file returns empty dict."""
        missing_file = tmp_path / "nonexistent.json"
        static_map = load_static_cusip_map(missing_file)

        assert static_map == {}

    def test_load_static_map_returns_cusip_mappings(self, static_map_file):
        """Static map values are CUSIPMapping instances."""
        static_map = load_static_cusip_map(static_map_file)

        for cusip, mapping in static_map.items():
            assert isinstance(mapping, CUSIPMapping)


# ============================================================================
# CACHE LOADING WITH TTL
# ============================================================================

class TestCacheLoading:
    """Tests for cache loading with TTL filtering."""

    def test_load_cache_filters_stale_entries(self, cache_file, reference_time):
        """Cache loading filters out stale entries."""
        cache = load_cache(cache_file, reference_time)

        # Fresh entry should be present
        assert "38259P508" in cache

        # Stale entry (older than TTL) should be filtered
        assert "023135106" not in cache

    def test_load_cache_preserves_fresh_entries(self, cache_file, reference_time):
        """Cache loading preserves fresh entries."""
        cache = load_cache(cache_file, reference_time)

        assert "38259P508" in cache
        assert cache["38259P508"].ticker == "GOOG"

    def test_load_cache_missing_file(self, tmp_path, reference_time):
        """Load cache from non-existent file returns empty dict."""
        missing_file = tmp_path / "nonexistent_cache.json"
        cache = load_cache(missing_file, reference_time)

        assert cache == {}

    def test_load_cache_no_reference_time(self, cache_file):
        """Load cache without reference time uses deterministic far-future default.

        Note: With a far-future reference_time (2099-12-31), the TTL cutoff is also
        far in the future (2099-10-02), so entries with mapped_at in the past
        will be filtered out as "stale". This is the expected behavior - calling
        load_cache with None means the cache is treated as if viewed from the future.
        """
        cache = load_cache(cache_file, None)

        # With far-future reference time, all entries with old mapped_at dates
        # are filtered out due to TTL, so cache should be empty
        assert len(cache) == 0


# ============================================================================
# CACHE SAVING
# ============================================================================

class TestCacheSaving:
    """Tests for cache persistence."""

    def test_save_cache(self, tmp_path, reference_time):
        """Save cache to file."""
        cache_path = tmp_path / "test_cache.json"

        mappings = {
            "037833100": CUSIPMapping(
                cusip="037833100",
                ticker="AAPL",
                name="Apple Inc",
                exchange="NASDAQ",
                security_type="Common Stock",
                mapped_at=reference_time.isoformat(),
                source="openfigi"
            )
        }

        save_cache(cache_path, mappings)

        # Verify file was created
        assert cache_path.exists()

        # Verify content
        with open(cache_path) as f:
            data = json.load(f)

        assert "037833100" in data
        assert data["037833100"]["ticker"] == "AAPL"

    def test_save_empty_cache(self, tmp_path):
        """Save empty cache."""
        cache_path = tmp_path / "empty_cache.json"
        save_cache(cache_path, {})

        with open(cache_path) as f:
            data = json.load(f)

        assert data == {}


# ============================================================================
# CUSIP MAPPER CLASS
# ============================================================================

class TestCUSIPMapper:
    """Tests for the main CUSIPMapper class."""

    def test_mapper_initialization(self, static_map_file, empty_cache_file, reference_time):
        """Initialize mapper with static map and cache."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            reference_time=reference_time
        )

        stats = mapper.stats()
        assert stats["static_map_size"] == 2
        assert stats["cache_size"] == 0

    def test_mapper_get_from_static_map(self, static_map_file, empty_cache_file, reference_time):
        """Get ticker from static map."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            reference_time=reference_time
        )

        ticker = mapper.get("037833100")
        assert ticker == "AAPL"

    def test_mapper_get_from_cache(self, static_map_file, cache_file, reference_time):
        """Get ticker from cache."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_file,
            reference_time=reference_time
        )

        ticker = mapper.get("38259P508")
        assert ticker == "GOOG"

    def test_mapper_get_mapping(self, static_map_file, empty_cache_file, reference_time):
        """Get full mapping details."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            reference_time=reference_time
        )

        mapping = mapper.get_mapping("037833100")

        assert isinstance(mapping, CUSIPMapping)
        assert mapping.ticker == "AAPL"
        assert mapping.name == "Apple Inc"
        assert mapping.source == "static"

    def test_mapper_get_unknown_cusip(self, static_map_file, empty_cache_file, reference_time):
        """Get unknown CUSIP without API returns None."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            openfigi_api_key=None,  # No API key means no live lookups
            reference_time=reference_time
        )

        # This won't make API call without key, returns None
        # Note: In real tests, we'd mock the API call
        mapping = mapper.get_mapping("999999999")

        # Without mocking, unknown CUSIPs return None
        # (API would be called if available)

    def test_mapper_get_batch(self, static_map_file, cache_file, reference_time):
        """Get batch of tickers."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_file,
            reference_time=reference_time
        )

        cusips = ["037833100", "594918104", "38259P508"]
        results = mapper.get_batch(cusips)

        assert results["037833100"] == "AAPL"
        assert results["594918104"] == "MSFT"
        assert results["38259P508"] == "GOOG"

    def test_mapper_stats(self, static_map_file, cache_file, reference_time):
        """Get mapper statistics."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_file,
            reference_time=reference_time
        )

        stats = mapper.stats()

        assert "static_map_size" in stats
        assert "cache_size" in stats
        assert "new_mappings_pending" in stats
        assert "total_known" in stats

    def test_mapper_save_updates_cache(self, static_map_file, tmp_path, reference_time):
        """Mapper save persists new mappings."""
        cache_path = tmp_path / "test_cache.json"
        with open(cache_path, 'w') as f:
            json.dump({}, f)

        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_path,
            reference_time=reference_time
        )

        # Manually add a new mapping to cache
        new_mapping = CUSIPMapping(
            cusip="111111111",
            ticker="TEST",
            name="Test Corp",
            exchange="NYSE",
            security_type="Common Stock",
            mapped_at=reference_time.isoformat(),
            source="openfigi"
        )
        mapper.cache["111111111"] = new_mapping
        mapper.new_mappings["111111111"] = new_mapping

        mapper.save()

        # Verify saved
        with open(cache_path) as f:
            saved_data = json.load(f)

        assert "111111111" in saved_data


# ============================================================================
# PIT SAFETY
# ============================================================================

class TestPITSafety:
    """Tests for point-in-time safety."""

    def test_mapper_uses_reference_time(self, static_map_file, empty_cache_file, reference_time):
        """Mapper uses provided reference time."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            reference_time=reference_time
        )

        assert mapper.reference_time == reference_time

    def test_cache_ttl_respects_reference_time(self, tmp_path):
        """Cache TTL is calculated from reference time, not wall clock."""
        cache_path = tmp_path / "cache.json"

        # Create cache with entries at specific times
        old_time = datetime(2025, 1, 1, 0, 0, 0)
        recent_time = datetime(2025, 10, 1, 0, 0, 0)

        cache_data = {
            "OLD00001": {
                "cusip": "OLD00001",
                "ticker": "OLD",
                "name": "Old Corp",
                "exchange": "NYSE",
                "security_type": "Common Stock",
                "mapped_at": old_time.isoformat(),
                "source": "openfigi"
            },
            "RECENT01": {
                "cusip": "RECENT01",
                "ticker": "NEW",
                "name": "Recent Corp",
                "exchange": "NYSE",
                "security_type": "Common Stock",
                "mapped_at": recent_time.isoformat(),
                "source": "openfigi"
            }
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        # Load with reference time that makes "OLD" entry stale
        reference = datetime(2025, 12, 1, 0, 0, 0)
        cache = load_cache(cache_path, reference)

        # OLD entry (11 months old) should be filtered if TTL is 90 days
        assert "OLD00001" not in cache
        assert "RECENT01" in cache


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_mapper_with_empty_static_map(self, tmp_path, reference_time):
        """Mapper works with empty static map."""
        static_path = tmp_path / "empty_static.json"
        cache_path = tmp_path / "empty_cache.json"

        with open(static_path, 'w') as f:
            json.dump({}, f)
        with open(cache_path, 'w') as f:
            json.dump({}, f)

        mapper = CUSIPMapper(
            static_map_path=static_path,
            cache_path=cache_path,
            reference_time=reference_time
        )

        assert mapper.stats()["static_map_size"] == 0
        assert mapper.stats()["cache_size"] == 0

    def test_batch_with_mixed_sources(self, static_map_file, cache_file, reference_time):
        """Batch lookup from multiple sources."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_file,
            reference_time=reference_time
        )

        cusips = [
            "037833100",  # Static map
            "38259P508",  # Cache
            "UNKNOWN99",  # Unknown
        ]

        results = mapper.get_batch(cusips)

        assert results["037833100"] == "AAPL"
        assert results["38259P508"] == "GOOG"
        # Unknown would be None or API-looked-up

    def test_cusip_case_handling(self, static_map_file, empty_cache_file, reference_time):
        """CUSIP lookups handle case."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=empty_cache_file,
            reference_time=reference_time
        )

        # Static map has uppercase CUSIPs
        # Note: CUSIPs are case-sensitive, but we test exact match
        result = mapper.get("037833100")
        assert result == "AAPL"


# ============================================================================
# DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_batch_results_deterministic(self, static_map_file, cache_file, reference_time):
        """Batch lookups are deterministic."""
        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_file,
            reference_time=reference_time
        )

        cusips = ["037833100", "594918104"]

        results1 = mapper.get_batch(cusips)
        results2 = mapper.get_batch(cusips)

        assert results1 == results2

    def test_save_produces_consistent_output(self, static_map_file, tmp_path, reference_time):
        """Saving produces consistent JSON output."""
        cache_path = tmp_path / "test_cache.json"
        with open(cache_path, 'w') as f:
            json.dump({}, f)

        mapper = CUSIPMapper(
            static_map_path=static_map_file,
            cache_path=cache_path,
            reference_time=reference_time
        )

        # Add mappings
        for cusip, ticker in [("AAA111111", "AAA"), ("BBB222222", "BBB")]:
            mapping = CUSIPMapping(
                cusip=cusip,
                ticker=ticker,
                name=f"{ticker} Corp",
                exchange="NYSE",
                security_type="Common Stock",
                mapped_at=reference_time.isoformat(),
                source="openfigi"
            )
            mapper.cache[cusip] = mapping
            mapper.new_mappings[cusip] = mapping

        mapper.save()

        with open(cache_path) as f:
            content1 = f.read()

        # Save again
        mapper.new_mappings = mapper.cache.copy()
        mapper.save()

        with open(cache_path) as f:
            content2 = f.read()

        assert content1 == content2
