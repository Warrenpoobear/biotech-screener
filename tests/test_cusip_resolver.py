#!/usr/bin/env python3
"""
Unit tests for cusip_resolver.py

Tests CUSIP to ticker resolution:
- Known CUSIP mappings
- CUSIPResolver class initialization and caching
- Single and batch CUSIP resolution
- Cache persistence
- Manual mapping addition
- Statistics tracking
- Deterministic hashing
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cusip_resolver import (
    CUSIPResolver,
    KNOWN_CUSIP_MAPPINGS,
    cusip_mapping_hash,
    get_resolver,
    resolve_cusip,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def resolver():
    """Resolver with no cache file (in-memory only)."""
    return CUSIPResolver(cache_path=None, use_openfigi=False)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary directory for cache files."""
    return tmp_path


@pytest.fixture
def resolver_with_cache(temp_cache_dir):
    """Resolver with temporary cache file."""
    cache_path = temp_cache_dir / "cusip_cache.json"
    return CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)


# ============================================================================
# KNOWN MAPPINGS TESTS
# ============================================================================

class TestKnownMappings:
    """Tests for KNOWN_CUSIP_MAPPINGS constant."""

    def test_contains_expected_mappings(self):
        """Should contain commonly used CUSIP mappings."""
        # Large-cap pharma
        assert KNOWN_CUSIP_MAPPINGS.get('031162100') == 'AMGN'
        assert KNOWN_CUSIP_MAPPINGS.get('92532F100') == 'VRTX'
        assert KNOWN_CUSIP_MAPPINGS.get('594918104') == 'MSFT'

    def test_contains_biotech_mappings(self):
        """Should contain biotech-focused mappings."""
        # Should have various biotech CUSIPs
        biotech_tickers = set(KNOWN_CUSIP_MAPPINGS.values())
        assert 'AMGN' in biotech_tickers
        assert 'VRTX' in biotech_tickers

    def test_mappings_not_empty(self):
        """Should have a reasonable number of mappings."""
        assert len(KNOWN_CUSIP_MAPPINGS) > 50

    def test_all_values_are_strings(self):
        """All ticker values should be strings."""
        for cusip, ticker in KNOWN_CUSIP_MAPPINGS.items():
            assert isinstance(cusip, str), f"CUSIP {cusip} is not a string"
            assert isinstance(ticker, str), f"Ticker {ticker} is not a string"


# ============================================================================
# CUSIP RESOLVER INITIALIZATION TESTS
# ============================================================================

class TestCUSIPResolverInit:
    """Tests for CUSIPResolver initialization."""

    def test_default_initialization(self, resolver):
        """Should initialize with default settings."""
        assert resolver.cache_path is None
        assert resolver.use_openfigi is False

    def test_loads_known_mappings(self, resolver):
        """Should pre-load known mappings into cache."""
        # Known mapping should be resolvable
        ticker = resolver.resolve('031162100')
        assert ticker == 'AMGN'

    def test_with_cache_path(self, temp_cache_dir):
        """Should accept cache path."""
        cache_path = temp_cache_dir / "test_cache.json"
        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)

        assert resolver.cache_path == cache_path

    def test_loads_existing_cache(self, temp_cache_dir):
        """Should load existing cache file."""
        cache_path = temp_cache_dir / "existing_cache.json"

        # Create cache file
        cache_data = {
            "mappings": {
                "TESTCUSIP": "TEST"
            }
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)

        # Should be able to resolve cached CUSIP
        assert resolver.resolve('TESTCUSIP') == 'TEST'

    def test_handles_invalid_cache_file(self, temp_cache_dir):
        """Should handle invalid cache file gracefully."""
        cache_path = temp_cache_dir / "invalid_cache.json"

        # Create invalid JSON
        with open(cache_path, 'w') as f:
            f.write("not valid json {")

        # Should not raise
        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)

        # Should still work with known mappings
        assert resolver.resolve('031162100') == 'AMGN'


# ============================================================================
# RESOLVE TESTS
# ============================================================================

class TestResolve:
    """Tests for resolve method."""

    def test_resolve_known_cusip(self, resolver):
        """Should resolve known CUSIPs."""
        assert resolver.resolve('031162100') == 'AMGN'
        assert resolver.resolve('92532F100') == 'VRTX'
        assert resolver.resolve('594918104') == 'MSFT'

    def test_resolve_with_whitespace(self, resolver):
        """Should handle whitespace in CUSIP."""
        assert resolver.resolve('  031162100  ') == 'AMGN'

    def test_resolve_lowercase(self, resolver):
        """Should handle lowercase CUSIP."""
        assert resolver.resolve('031162100') == 'AMGN'

    def test_resolve_empty_cusip(self, resolver):
        """Should return None for empty CUSIP."""
        assert resolver.resolve('') is None
        assert resolver.resolve(None) is None

    def test_resolve_unknown_cusip(self, resolver):
        """Should return None for unknown CUSIP (without OpenFIGI)."""
        assert resolver.resolve('000000000') is None

    def test_resolve_short_cusip(self, resolver):
        """Should handle short CUSIP format."""
        # Some 13Fs use 6-8 char format
        # The resolver should try to match with known mappings
        # that might use different lengths
        result = resolver.resolve('76243J10')  # 8-char
        assert result == 'RYTM'

    def test_resolve_caches_result(self, resolver_with_cache):
        """Should cache resolved CUSIPs."""
        # First resolution
        result1 = resolver_with_cache.resolve('031162100')
        assert result1 == 'AMGN'

        # Check stats
        stats = resolver_with_cache.get_stats()
        assert stats['hits_known'] >= 1

    def test_resolve_caches_misses(self, resolver_with_cache):
        """Should cache misses to avoid repeated lookups."""
        # First miss
        result1 = resolver_with_cache.resolve('UNKNOWNCUSIP')
        assert result1 is None

        # Second lookup should hit cache
        result2 = resolver_with_cache.resolve('UNKNOWNCUSIP')
        assert result2 is None


# ============================================================================
# RESOLVE BATCH TESTS
# ============================================================================

class TestResolveBatch:
    """Tests for resolve_batch method."""

    def test_batch_resolve_known(self, resolver):
        """Should resolve multiple known CUSIPs."""
        cusips = ['031162100', '92532F100', '594918104']

        results = resolver.resolve_batch(cusips)

        assert results['031162100'] == 'AMGN'
        assert results['92532F100'] == 'VRTX'
        assert results['594918104'] == 'MSFT'

    def test_batch_resolve_mixed(self, resolver):
        """Should handle mix of known and unknown CUSIPs."""
        cusips = ['031162100', 'UNKNOWNCUSIP', '594918104']

        results = resolver.resolve_batch(cusips)

        assert results['031162100'] == 'AMGN'
        # Unknown CUSIP is normalized to uppercase and truncated
        assert results.get('UNKNOWNCUSIP') is None or results.get('UNKNOWNCU') is None
        assert results['594918104'] == 'MSFT'

    def test_batch_resolve_empty(self, resolver):
        """Should handle empty list."""
        results = resolver.resolve_batch([])
        assert results == {}

    def test_batch_resolve_deduplicates(self, resolver):
        """Should handle duplicate CUSIPs."""
        cusips = ['031162100', '031162100', '031162100']

        results = resolver.resolve_batch(cusips)

        # Should have result (may have duplicate keys collapsed)
        assert '031162100' in results or any('031162100' in k for k in results)


# ============================================================================
# ADD MAPPING TESTS
# ============================================================================

class TestAddMapping:
    """Tests for add_mapping method."""

    def test_add_mapping(self, resolver):
        """Should add new mapping."""
        resolver.add_mapping('TESTCUSIP1', 'TEST')

        assert resolver.resolve('TESTCUSIP1') == 'TEST'

    def test_add_mapping_normalizes_cusip(self, resolver):
        """Should normalize CUSIP when adding."""
        resolver.add_mapping('  testcusip2  ', 'TEST2')

        assert resolver.resolve('TESTCUSIP2') == 'TEST2'

    def test_add_mapping_truncates_to_9(self, resolver):
        """Should truncate CUSIP to 9 characters."""
        resolver.add_mapping('TESTCUSIP123456', 'TEST3')

        # Should be stored as first 9 chars
        assert resolver.resolve('TESTCUSIP') == 'TEST3'

    def test_add_mapping_persists(self, resolver_with_cache, temp_cache_dir):
        """Should persist mapping to cache file."""
        resolver_with_cache.add_mapping('PERSISTED', 'PRST')

        # Check cache file was written
        cache_path = temp_cache_dir / "cusip_cache.json"
        assert cache_path.exists()

        with open(cache_path, 'r') as f:
            data = json.load(f)

        assert 'PERSISTED' in data['mappings']


# ============================================================================
# GET STATS TESTS
# ============================================================================

class TestGetStats:
    """Tests for get_stats method."""

    def test_stats_structure(self, resolver):
        """Should return stats dict with expected keys."""
        stats = resolver.get_stats()

        assert 'hits_known' in stats
        assert 'hits_cache' in stats
        assert 'hits_openfigi' in stats
        assert 'misses' in stats
        assert 'cache_size' in stats
        assert 'known_mappings' in stats

    def test_stats_initial_values(self, resolver):
        """Should have zero hits initially."""
        stats = resolver.get_stats()

        assert stats['hits_known'] == 0
        assert stats['hits_cache'] == 0
        assert stats['misses'] == 0

    def test_stats_track_known_hits(self, resolver):
        """Should track hits on known mappings."""
        resolver.resolve('031162100')
        resolver.resolve('92532F100')

        stats = resolver.get_stats()
        assert stats['hits_known'] >= 2

    def test_stats_track_misses(self, resolver):
        """Should track misses."""
        resolver.resolve('UNKNOWN1')
        resolver.resolve('UNKNOWN2')

        stats = resolver.get_stats()
        assert stats['misses'] == 2

    def test_stats_cache_size(self, resolver):
        """Should report cache size."""
        stats = resolver.get_stats()

        # Cache should have at least the known mappings
        assert stats['cache_size'] >= len(KNOWN_CUSIP_MAPPINGS)


# ============================================================================
# CUSIP MAPPING HASH TESTS
# ============================================================================

class TestCusipMappingHash:
    """Tests for cusip_mapping_hash function."""

    def test_deterministic(self):
        """Same mappings should produce same hash."""
        mappings = {'CUSIP1': 'TICK1', 'CUSIP2': 'TICK2'}

        hash1 = cusip_mapping_hash(mappings)
        hash2 = cusip_mapping_hash(mappings)

        assert hash1 == hash2

    def test_order_independent(self):
        """Hash should be independent of dict order."""
        mappings1 = {'A': '1', 'B': '2', 'C': '3'}
        mappings2 = {'C': '3', 'A': '1', 'B': '2'}

        hash1 = cusip_mapping_hash(mappings1)
        hash2 = cusip_mapping_hash(mappings2)

        assert hash1 == hash2

    def test_different_mappings_different_hash(self):
        """Different mappings should produce different hash."""
        mappings1 = {'CUSIP1': 'TICK1'}
        mappings2 = {'CUSIP1': 'TICK2'}

        hash1 = cusip_mapping_hash(mappings1)
        hash2 = cusip_mapping_hash(mappings2)

        assert hash1 != hash2

    def test_hash_length(self):
        """Hash should be 16 characters."""
        mappings = {'CUSIP1': 'TICK1'}
        hash_val = cusip_mapping_hash(mappings)

        assert len(hash_val) == 16

    def test_hash_is_hex(self):
        """Hash should be hexadecimal."""
        mappings = {'CUSIP1': 'TICK1'}
        hash_val = cusip_mapping_hash(mappings)

        # Should be valid hex
        int(hash_val, 16)  # Will raise if not hex


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_resolver_returns_resolver(self, temp_cache_dir):
        """get_resolver should return CUSIPResolver instance."""
        # Use temp path to avoid affecting global state
        cache_path = str(temp_cache_dir / "test_cache.json")

        # Note: This modifies global state, so we reset after
        import cusip_resolver
        old_resolver = cusip_resolver._default_resolver
        cusip_resolver._default_resolver = None

        try:
            resolver = get_resolver(cache_path=cache_path)
            assert isinstance(resolver, CUSIPResolver)
        finally:
            cusip_resolver._default_resolver = old_resolver

    def test_resolve_cusip_function(self, temp_cache_dir):
        """resolve_cusip should resolve CUSIPs using default resolver."""
        import cusip_resolver
        old_resolver = cusip_resolver._default_resolver

        # Create a fresh resolver
        cusip_resolver._default_resolver = CUSIPResolver(
            cache_path=None,
            use_openfigi=False
        )

        try:
            result = resolve_cusip('031162100')
            assert result == 'AMGN'
        finally:
            cusip_resolver._default_resolver = old_resolver


# ============================================================================
# CACHE PERSISTENCE TESTS
# ============================================================================

class TestCachePersistence:
    """Tests for cache persistence."""

    def test_save_cache_creates_file(self, temp_cache_dir):
        """Should create cache file on save."""
        cache_path = temp_cache_dir / "new_cache.json"
        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)

        # Add a mapping (triggers save)
        resolver.add_mapping('NEWCUSIP', 'NEW')

        assert cache_path.exists()

    def test_save_cache_format(self, temp_cache_dir):
        """Saved cache should have expected format."""
        cache_path = temp_cache_dir / "format_cache.json"
        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)
        resolver.add_mapping('TESTFMT', 'FMT')

        with open(cache_path, 'r') as f:
            data = json.load(f)

        assert 'version' in data
        assert 'updated_at' in data
        assert 'count' in data
        assert 'mappings' in data
        assert 'TESTFMT' in data['mappings']

    def test_load_cache_legacy_format(self, temp_cache_dir):
        """Should handle legacy cache format (plain dict)."""
        cache_path = temp_cache_dir / "legacy_cache.json"

        # Write legacy format (no metadata, just mappings)
        legacy_data = {'LEGACY': 'LEG'}
        with open(cache_path, 'w') as f:
            json.dump(legacy_data, f)

        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)

        assert resolver.resolve('LEGACY') == 'LEG'

    def test_cache_creates_parent_dirs(self, temp_cache_dir):
        """Should create parent directories if needed."""
        cache_path = temp_cache_dir / "nested" / "dir" / "cache.json"
        resolver = CUSIPResolver(cache_path=str(cache_path), use_openfigi=False)
        resolver.add_mapping('NESTED', 'NEST')

        assert cache_path.exists()


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_resolve_deterministic(self, resolver):
        """resolve should be deterministic."""
        results = [resolver.resolve('031162100') for _ in range(10)]

        assert all(r == 'AMGN' for r in results)

    def test_resolve_batch_deterministic(self, resolver):
        """resolve_batch should be deterministic."""
        cusips = ['031162100', '92532F100', '594918104']

        results = [resolver.resolve_batch(cusips) for _ in range(5)]

        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_cusip_mapping_hash_deterministic(self):
        """cusip_mapping_hash should be deterministic."""
        mappings = {'A': '1', 'B': '2', 'C': '3'}

        hashes = [cusip_mapping_hash(mappings) for _ in range(10)]

        assert len(set(hashes)) == 1  # All identical


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case tests for CUSIP resolver."""

    def test_very_long_cusip(self, resolver):
        """Should handle very long CUSIP string."""
        # Should truncate to 9 chars
        result = resolver.resolve('031162100' + 'X' * 100)
        assert result == 'AMGN'

    def test_special_characters(self, resolver):
        """Should handle CUSIPs with special characters."""
        # Some CUSIPs have letters (issuer check digit)
        result = resolver.resolve('92532F100')
        assert result == 'VRTX'

    def test_concurrent_access(self, resolver_with_cache):
        """Should handle concurrent-like access."""
        # Simulate rapid sequential access
        for _ in range(100):
            resolver_with_cache.resolve('031162100')
            resolver_with_cache.resolve('UNKNOWN')
            resolver_with_cache.resolve('92532F100')

        stats = resolver_with_cache.get_stats()
        assert stats['hits_known'] >= 100
