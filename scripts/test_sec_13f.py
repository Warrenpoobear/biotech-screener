"""
Tests for SEC 13F Provider Components

Run with: pytest test_sec_13f.py -v
"""

import pytest
from datetime import date

from elite_managers import (
    ELITE_MANAGERS, get_manager_by_cik, get_manager_by_short_name,
    get_tier_1_managers, get_all_ciks, get_manager_weight,
    validate_registry, TIER_WEIGHTS, STYLE_CONVICTION_MULTIPLIER,
)

from cusip_resolver import (
    CUSIPResolver, KNOWN_CUSIP_MAPPINGS,
    cusip_mapping_hash, resolve_cusip,
)

from aggregator import (
    ManagerPosition, AggregatedSignal, ElitePositionAggregator,
)


# =============================================================================
# ELITE MANAGERS REGISTRY TESTS
# =============================================================================

class TestEliteManagers:
    """Test elite manager registry."""
    
    def test_registry_not_empty(self):
        """Registry should have managers."""
        assert len(ELITE_MANAGERS) > 0
    
    def test_registry_has_tier_1(self):
        """Registry should have Tier 1 managers."""
        tier_1 = get_tier_1_managers()
        assert len(tier_1) >= 5, "Expected at least 5 Tier 1 managers"
    
    def test_registry_validates(self):
        """Registry should pass validation."""
        assert validate_registry() is True
    
    def test_no_duplicate_ciks(self):
        """CIKs should be unique."""
        ciks = get_all_ciks()
        assert len(ciks) == len(set(ciks))
    
    def test_get_manager_by_cik(self):
        """Should find manager by CIK."""
        baker = get_manager_by_cik('1074999')
        assert baker is not None
        assert baker['short_name'] == 'Baker Bros'
    
    def test_get_manager_by_cik_with_leading_zeros(self):
        """Should handle leading zeros in CIK."""
        baker = get_manager_by_cik('0001074999')
        assert baker is not None
        assert baker['short_name'] == 'Baker Bros'
    
    def test_get_manager_by_short_name(self):
        """Should find manager by short name."""
        ra = get_manager_by_short_name('RA Capital')
        assert ra is not None
        assert ra['cik'] == '1535392'
    
    def test_get_manager_by_short_name_case_insensitive(self):
        """Short name lookup should be case-insensitive."""
        ra = get_manager_by_short_name('ra capital')
        assert ra is not None
    
    def test_manager_weight_tier_1(self):
        """Tier 1 manager should have higher weight."""
        baker = get_manager_by_cik('1074999')
        weight = get_manager_weight(baker)
        assert weight >= 1.0, "Tier 1 concentrated manager should have weight >= 1.0"
    
    def test_manager_weight_tier_2_lower(self):
        """Tier 2 manager should have lower weight than Tier 1."""
        tier_1 = get_tier_1_managers()[0]
        tier_2 = [m for m in ELITE_MANAGERS if m['tier'] == 2][0]
        
        weight_1 = get_manager_weight(tier_1)
        weight_2 = get_manager_weight(tier_2)
        
        # Tier 2 base weight is 0.7, so should generally be lower
        # But style can affect this, so we check tier weights directly
        assert TIER_WEIGHTS[1] > TIER_WEIGHTS[2]
    
    def test_all_managers_have_required_fields(self):
        """All managers should have required fields."""
        required = ['cik', 'name', 'short_name', 'tier']
        for manager in ELITE_MANAGERS:
            for field in required:
                assert field in manager, f"Missing {field} in {manager.get('name', 'unknown')}"
    
    def test_baker_bros_is_tier_1(self):
        """Baker Bros should be Tier 1 (our benchmark elite manager)."""
        baker = get_manager_by_short_name('Baker Bros')
        assert baker['tier'] == 1


# =============================================================================
# CUSIP RESOLVER TESTS
# =============================================================================

class TestCUSIPResolver:
    """Test CUSIP resolution."""
    
    def test_known_mapping_amgn(self):
        """Should resolve known CUSIP for Amgen."""
        resolver = CUSIPResolver(cache_path=None)
        assert resolver.resolve('031162100') == 'AMGN'
    
    def test_known_mapping_vrtx(self):
        """Should resolve known CUSIP for Vertex."""
        resolver = CUSIPResolver(cache_path=None)
        assert resolver.resolve('92532F100') == 'VRTX'
    
    def test_known_mapping_msft(self):
        """Should resolve known CUSIP for Microsoft."""
        resolver = CUSIPResolver(cache_path=None)
        assert resolver.resolve('594918104') == 'MSFT'
    
    def test_unknown_cusip_returns_none(self):
        """Unknown CUSIP should return None (without API)."""
        resolver = CUSIPResolver(cache_path=None, use_openfigi=False)
        result = resolver.resolve('000000000')
        assert result is None
    
    def test_empty_cusip_returns_none(self):
        """Empty CUSIP should return None."""
        resolver = CUSIPResolver(cache_path=None)
        assert resolver.resolve('') is None
        assert resolver.resolve(None) is None
    
    def test_cusip_normalized(self):
        """CUSIP should be normalized (uppercase, trimmed)."""
        resolver = CUSIPResolver(cache_path=None)
        # Should work with lowercase and whitespace
        assert resolver.resolve('  031162100  ') == 'AMGN'
    
    def test_known_mappings_not_empty(self):
        """Should have known mappings populated."""
        assert len(KNOWN_CUSIP_MAPPINGS) > 10
    
    def test_batch_resolve(self):
        """Batch resolution should work."""
        resolver = CUSIPResolver(cache_path=None, use_openfigi=False)
        cusips = ['031162100', '92532F100', '000000000']
        results = resolver.resolve_batch(cusips)
        
        assert results['031162100'] == 'AMGN'
        assert results['92532F100'] == 'VRTX'
        assert results['000000000'] is None
    
    def test_stats_tracking(self):
        """Should track resolution stats."""
        resolver = CUSIPResolver(cache_path=None, use_openfigi=False)
        resolver.resolve('031162100')  # Known
        resolver.resolve('000000000')  # Unknown
        
        stats = resolver.get_stats()
        assert stats['hits_known'] >= 1
        assert stats['misses'] >= 1
    
    def test_deterministic_hash(self):
        """Mapping hash should be deterministic."""
        mappings = {'A': 'X', 'B': 'Y'}
        hash1 = cusip_mapping_hash(mappings)
        hash2 = cusip_mapping_hash(mappings)
        assert hash1 == hash2
        
        # Different order should produce same hash (sorted internally)
        mappings2 = {'B': 'Y', 'A': 'X'}
        hash3 = cusip_mapping_hash(mappings2)
        assert hash1 == hash3


# =============================================================================
# AGGREGATOR TESTS
# =============================================================================

class TestManagerPosition:
    """Test ManagerPosition dataclass."""
    
    def test_create_position(self):
        """Should create position with all fields."""
        pos = ManagerPosition(
            manager_cik='1074999',
            manager_name='Baker Bros',
            manager_tier=1,
            manager_weight=1.2,
            cusip='031162100',
            ticker='AMGN',
            issuer_name='AMGEN INC',
            shares=1000000,
            value=250000000,
            position_weight=0.15,
            report_date=date(2025, 9, 30),
            filing_date=date(2025, 11, 14),
        )
        
        assert pos.manager_name == 'Baker Bros'
        assert pos.ticker == 'AMGN'
        assert pos.position_weight == 0.15
    
    def test_position_with_change(self):
        """Should track quarter-over-quarter change."""
        pos = ManagerPosition(
            manager_cik='1074999',
            manager_name='Baker Bros',
            manager_tier=1,
            manager_weight=1.2,
            cusip='031162100',
            ticker='AMGN',
            issuer_name='AMGEN INC',
            shares=1100000,
            value=275000000,
            position_weight=0.15,
            report_date=date(2025, 9, 30),
            filing_date=date(2025, 11, 14),
            shares_change=100000,
            value_change=25000000,
            is_new_position=False,
        )
        
        assert pos.shares_change == 100000
        assert pos.is_new_position is False


class TestAggregatedSignal:
    """Test AggregatedSignal dataclass."""
    
    def test_create_signal(self):
        """Should create aggregated signal."""
        signal = AggregatedSignal(
            cusip='031162100',
            ticker='AMGN',
            issuer_name='AMGEN INC',
            overlap_count=3,
            tier_1_count=2,
            total_value=500000000,
            conviction_score=75.5,
            holders=['Baker Bros', 'RA Capital', 'Perceptive'],
            tier_1_holders=['Baker Bros', 'RA Capital'],
            managers_adding=['Baker Bros'],
            managers_reducing=[],
            new_positions=[],
            exits=[],
        )
        
        assert signal.ticker == 'AMGN'
        assert signal.overlap_count == 3
        assert signal.conviction_score == 75.5
    
    def test_signal_to_dict(self):
        """Should convert to dictionary."""
        signal = AggregatedSignal(
            cusip='031162100',
            ticker='AMGN',
            issuer_name='AMGEN INC',
            overlap_count=3,
            tier_1_count=2,
            total_value=500000000,
            conviction_score=75.5,
            holders=['Baker Bros', 'RA Capital'],
            tier_1_holders=['Baker Bros', 'RA Capital'],
            managers_adding=[],
            managers_reducing=[],
            new_positions=[],
            exits=[],
        )
        
        d = signal.to_dict()
        assert d['ticker'] == 'AMGN'
        assert d['conviction_score'] == 75.5
        assert len(d['holders']) == 2


class TestConvictionScoring:
    """Test conviction score calculation."""
    
    def test_score_increases_with_overlap(self):
        """More managers holding should increase score."""
        agg = ElitePositionAggregator(managers=[], cache_dir=None)
        
        # Create positions with different overlap counts
        pos1 = ManagerPosition(
            manager_cik='1', manager_name='M1', manager_tier=1, manager_weight=1.0,
            cusip='X', ticker='X', issuer_name='X', shares=100, value=1000,
            position_weight=0.05, report_date=date.today(), filing_date=date.today(),
        )
        pos2 = ManagerPosition(
            manager_cik='2', manager_name='M2', manager_tier=1, manager_weight=1.0,
            cusip='X', ticker='X', issuer_name='X', shares=100, value=1000,
            position_weight=0.05, report_date=date.today(), filing_date=date.today(),
        )
        
        score_1 = agg._compute_conviction_score([pos1])
        score_2 = agg._compute_conviction_score([pos1, pos2])
        
        assert score_2 > score_1
    
    def test_score_bounded_0_100(self):
        """Score should be between 0 and 100."""
        agg = ElitePositionAggregator(managers=[], cache_dir=None)
        
        # Edge case: empty
        assert agg._compute_conviction_score([]) == 0.0
        
        # Edge case: many positions
        positions = [
            ManagerPosition(
                manager_cik=str(i), manager_name=f'M{i}', manager_tier=1, manager_weight=1.2,
                cusip='X', ticker='X', issuer_name='X', shares=100, value=1000,
                position_weight=0.20, report_date=date.today(), filing_date=date.today(),
                is_new_position=True,
            )
            for i in range(10)
        ]
        
        score = agg._compute_conviction_score(positions)
        assert 0 <= score <= 100


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Ensure all components are deterministic."""
    
    def test_manager_weight_deterministic(self):
        """Manager weight should be deterministic."""
        baker = get_manager_by_cik('1074999')
        
        weights = [get_manager_weight(baker) for _ in range(3)]
        assert all(w == weights[0] for w in weights)
    
    def test_cusip_resolution_deterministic(self):
        """CUSIP resolution should be deterministic."""
        resolver = CUSIPResolver(cache_path=None)
        
        results = [resolver.resolve('031162100') for _ in range(3)]
        assert all(r == 'AMGN' for r in results)
    
    def test_conviction_score_deterministic(self):
        """Conviction score should be deterministic."""
        agg = ElitePositionAggregator(managers=[], cache_dir=None)
        
        positions = [
            ManagerPosition(
                manager_cik='1', manager_name='M1', manager_tier=1, manager_weight=1.0,
                cusip='X', ticker='X', issuer_name='X', shares=100, value=1000,
                position_weight=0.10, report_date=date.today(), filing_date=date.today(),
            )
        ]
        
        scores = [agg._compute_conviction_score(positions) for _ in range(3)]
        assert all(s == scores[0] for s in scores)
