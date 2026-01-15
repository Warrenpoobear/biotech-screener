"""
Tests for Module 5 Co-Invest Tie-Breaker Integration

Tests:
1. PIT guard: same-day filings excluded
2. Deterministic ordering: tie-breaker stable across runs
3. Missing CUSIP behavior: doesn't crash; flags computed correctly

Run with: pytest test_module_5_coinvest.py -v
"""

import pytest
from datetime import date
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# MOCK TYPES (minimal versions for testing)
# =============================================================================

@dataclass
class MockManagerPosition:
    """Mock of ManagerPosition from aggregator.py"""
    manager_cik: str
    manager_name: str
    manager_tier: int
    manager_weight: float
    cusip: str
    ticker: Optional[str]
    issuer_name: str
    shares: int
    value: int
    position_weight: float
    report_date: date
    filing_date: date
    shares_change: Optional[int] = None
    value_change: Optional[int] = None
    is_new_position: bool = False
    is_exit: bool = False


@dataclass
class MockAggregatedSignal:
    """Mock of AggregatedSignal from aggregator.py"""
    cusip: str
    ticker: Optional[str]
    issuer_name: str
    overlap_count: int
    tier_1_count: int
    total_value: int
    conviction_score: float
    holders: list
    tier_1_holders: list
    managers_adding: list
    managers_reducing: list
    new_positions: list
    exits: list
    positions: list = field(default_factory=list)
    as_of_date: Optional[date] = None


# =============================================================================
# HELPER FUNCTIONS UNDER TEST (copied from module_5_composite_v2.py)
# =============================================================================

def _quarter_from_date(d: date) -> str:
    """Convert date to quarter string (e.g., '2025Q3')."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _enrich_with_coinvest(
    ticker: str,
    coinvest_signals: dict,
    as_of_date: date,
) -> dict:
    """
    Look up co-invest signal for a ticker and return overlay fields.
    PIT Rule: Only include filings where filing_date < as_of_date.
    """
    signal = coinvest_signals.get(ticker)
    
    if not signal:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_quarter": None,
            "coinvest_published_at_max": None,
            "coinvest_usable": False,
            "coinvest_flags": ["no_signal"],
        }
    
    # PIT filter: only count positions from filings before as_of_date
    pit_positions = [
        p for p in signal.positions
        if p.filing_date < as_of_date
    ]
    
    if not pit_positions:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_quarter": _quarter_from_date(signal.positions[0].report_date) if signal.positions else None,
            "coinvest_published_at_max": None,
            "coinvest_usable": False,
            "coinvest_flags": ["filings_not_yet_public"],
        }
    
    # Compute PIT-safe metrics
    holders = sorted(set(p.manager_name for p in pit_positions))
    max_filing_date = max(p.filing_date for p in pit_positions)
    report_quarter = _quarter_from_date(pit_positions[0].report_date)
    
    flags = []
    if len(pit_positions) < len(signal.positions):
        flags.append("partial_manager_coverage")
    if signal.ticker is None:
        flags.append("cusip_unmapped")
    
    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": report_quarter,
        "coinvest_published_at_max": max_filing_date.isoformat(),
        "coinvest_usable": True,
        "coinvest_flags": sorted(flags),
    }


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_positions():
    """Create sample manager positions for testing."""
    return [
        MockManagerPosition(
            manager_cik="1263508",
            manager_name="Baker Bros",
            manager_tier=1,
            manager_weight=1.2,
            cusip="76243J105",
            ticker="RYTM",
            issuer_name="RHYTHM PHARMACEUTICALS INC",
            shares=1000000,
            value=50000000,
            position_weight=0.05,
            report_date=date(2025, 9, 30),
            filing_date=date(2025, 11, 14),  # Filed Nov 14
        ),
        MockManagerPosition(
            manager_cik="1346824",
            manager_name="RA Capital",
            manager_tier=1,
            manager_weight=1.0,
            cusip="76243J105",
            ticker="RYTM",
            issuer_name="RHYTHM PHARMACEUTICALS INC",
            shares=800000,
            value=40000000,
            position_weight=0.04,
            report_date=date(2025, 9, 30),
            filing_date=date(2025, 11, 15),  # Filed Nov 15
        ),
        MockManagerPosition(
            manager_cik="1224962",
            manager_name="Perceptive",
            manager_tier=1,
            manager_weight=0.9,
            cusip="76243J105",
            ticker="RYTM",
            issuer_name="RHYTHM PHARMACEUTICALS INC",
            shares=600000,
            value=30000000,
            position_weight=0.03,
            report_date=date(2025, 9, 30),
            filing_date=date(2025, 11, 20),  # Filed Nov 20 (later)
        ),
    ]


@pytest.fixture
def sample_signal(sample_positions):
    """Create a sample aggregated signal."""
    return MockAggregatedSignal(
        cusip="76243J105",
        ticker="RYTM",
        issuer_name="RHYTHM PHARMACEUTICALS INC",
        overlap_count=3,
        tier_1_count=3,
        total_value=120000000,
        conviction_score=77.7,
        holders=["Baker Bros", "RA Capital", "Perceptive"],
        tier_1_holders=["Baker Bros", "RA Capital", "Perceptive"],
        managers_adding=[],
        managers_reducing=[],
        new_positions=[],
        exits=[],
        positions=sample_positions,
        as_of_date=date(2025, 11, 25),
    )


@pytest.fixture
def unmapped_signal():
    """Create a signal with unmapped CUSIP."""
    pos = MockManagerPosition(
        manager_cik="1263508",
        manager_name="Baker Bros",
        manager_tier=1,
        manager_weight=1.2,
        cusip="99999X100",
        ticker=None,  # Unmapped
        issuer_name="UNKNOWN CORP",
        shares=100000,
        value=5000000,
        position_weight=0.01,
        report_date=date(2025, 9, 30),
        filing_date=date(2025, 11, 14),
    )
    return MockAggregatedSignal(
        cusip="99999X100",
        ticker=None,  # Unmapped
        issuer_name="UNKNOWN CORP",
        overlap_count=1,
        tier_1_count=1,
        total_value=5000000,
        conviction_score=20.0,
        holders=["Baker Bros"],
        tier_1_holders=["Baker Bros"],
        managers_adding=[],
        managers_reducing=[],
        new_positions=[],
        exits=[],
        positions=[pos],
    )


# =============================================================================
# TEST 1: PIT GUARD - Same-day filings excluded
# =============================================================================

class TestPITGuard:
    """Test point-in-time safety for 13F filings."""
    
    def test_filing_before_asof_included(self, sample_signal):
        """Filings before as_of_date should be included."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 25 - all three filings (Nov 14, 15, 20) should be included
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 25))
        
        assert result["coinvest_overlap_count"] == 3
        assert result["coinvest_usable"] is True
        assert len(result["coinvest_holders"]) == 3
    
    def test_filing_on_asof_excluded(self, sample_signal):
        """Filings ON as_of_date should be excluded (< not <=)."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 15 - only Nov 14 filing should be included
        # Nov 15 filing is NOT included (filing_date < as_of_date, not <=)
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 15))
        
        assert result["coinvest_overlap_count"] == 1
        assert result["coinvest_holders"] == ["Baker Bros"]
        assert "partial_manager_coverage" in result["coinvest_flags"]
    
    def test_all_filings_after_asof_excluded(self, sample_signal):
        """If all filings are after as_of_date, overlap should be 0."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 10 - no filings yet
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 10))
        
        assert result["coinvest_overlap_count"] == 0
        assert result["coinvest_usable"] is False
        assert "filings_not_yet_public" in result["coinvest_flags"]
    
    def test_partial_coverage_flagged(self, sample_signal):
        """Partial manager coverage should be flagged."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 16 - only Baker Bros and RA Capital (not Perceptive)
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 16))
        
        assert result["coinvest_overlap_count"] == 2
        assert "partial_manager_coverage" in result["coinvest_flags"]


# =============================================================================
# TEST 2: DETERMINISTIC ORDERING
# =============================================================================

class TestDeterministicOrdering:
    """Test that tie-breaker produces stable, deterministic results."""
    
    def test_sort_key_deterministic(self):
        """Sort key should produce same order across runs."""
        # Simulate records with same composite score
        records = [
            {"ticker": "ZYME", "composite_score": Decimal("65.00"), "coinvest": {"coinvest_overlap_count": 3}},
            {"ticker": "ABVX", "composite_score": Decimal("65.00"), "coinvest": {"coinvest_overlap_count": 3}},
            {"ticker": "RYTM", "composite_score": Decimal("65.00"), "coinvest": {"coinvest_overlap_count": 4}},
            {"ticker": "PEPG", "composite_score": Decimal("65.00"), "coinvest": {"coinvest_overlap_count": 4}},
        ]
        
        def sort_key(x):
            coinvest_count = x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0
            return (-x["composite_score"], -coinvest_count, x["ticker"])
        
        # Sort multiple times
        for _ in range(5):
            sorted_records = sorted(records, key=sort_key)
            tickers = [r["ticker"] for r in sorted_records]
            
            # Expected order: highest overlap first, then alphabetical
            # PEPG and RYTM have overlap 4, ABVX and ZYME have overlap 3
            # Within same overlap, alphabetical: PEPG < RYTM, ABVX < ZYME
            assert tickers == ["PEPG", "RYTM", "ABVX", "ZYME"]
    
    def test_no_coinvest_sorted_last(self):
        """Records without co-invest signals should sort after those with signals."""
        records = [
            {"ticker": "AAA", "composite_score": Decimal("70.00"), "coinvest": None},
            {"ticker": "BBB", "composite_score": Decimal("70.00"), "coinvest": {"coinvest_overlap_count": 2}},
            {"ticker": "CCC", "composite_score": Decimal("70.00"), "coinvest": {"coinvest_overlap_count": 0}},
        ]
        
        def sort_key(x):
            coinvest_count = x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0
            return (-x["composite_score"], -coinvest_count, x["ticker"])
        
        sorted_records = sorted(records, key=sort_key)
        tickers = [r["ticker"] for r in sorted_records]
        
        # BBB (overlap 2) first, then AAA and CCC (overlap 0) alphabetically
        assert tickers == ["BBB", "AAA", "CCC"]


# =============================================================================
# TEST 3: MISSING CUSIP BEHAVIOR
# =============================================================================

class TestMissingCUSIP:
    """Test behavior when CUSIP mapping is missing."""
    
    def test_no_signal_flags_correctly(self):
        """Ticker with no signal should have 'no_signal' flag."""
        signals = {}  # Empty
        
        result = _enrich_with_coinvest("UNKNOWN", signals, date(2025, 11, 25))
        
        assert result["coinvest_overlap_count"] == 0
        assert result["coinvest_usable"] is False
        assert "no_signal" in result["coinvest_flags"]
    
    def test_unmapped_cusip_flagged(self, unmapped_signal):
        """Signal with unmapped CUSIP should be flagged."""
        # Use CUSIP as key since ticker is None
        signals = {"99999X100": unmapped_signal}
        
        result = _enrich_with_coinvest("99999X100", signals, date(2025, 11, 25))
        
        assert result["coinvest_overlap_count"] == 1
        assert result["coinvest_usable"] is True
        assert "cusip_unmapped" in result["coinvest_flags"]
    
    def test_does_not_crash_on_empty_positions(self):
        """Should handle signal with empty positions gracefully."""
        empty_signal = MockAggregatedSignal(
            cusip="EMPTY",
            ticker="EMPTY",
            issuer_name="EMPTY CORP",
            overlap_count=0,
            tier_1_count=0,
            total_value=0,
            conviction_score=0,
            holders=[],
            tier_1_holders=[],
            managers_adding=[],
            managers_reducing=[],
            new_positions=[],
            exits=[],
            positions=[],  # Empty
        )
        
        signals = {"EMPTY": empty_signal}
        
        # Should not crash
        result = _enrich_with_coinvest("EMPTY", signals, date(2025, 11, 25))
        
        assert result["coinvest_overlap_count"] == 0
        assert result["coinvest_usable"] is False


# =============================================================================
# TEST 4: QUARTER COMPUTATION
# =============================================================================

class TestQuarterComputation:
    """Test quarter string computation."""
    
    def test_q1(self):
        assert _quarter_from_date(date(2025, 1, 15)) == "2025Q1"
        assert _quarter_from_date(date(2025, 3, 31)) == "2025Q1"
    
    def test_q2(self):
        assert _quarter_from_date(date(2025, 4, 1)) == "2025Q2"
        assert _quarter_from_date(date(2025, 6, 30)) == "2025Q2"
    
    def test_q3(self):
        assert _quarter_from_date(date(2025, 7, 1)) == "2025Q3"
        assert _quarter_from_date(date(2025, 9, 30)) == "2025Q3"
    
    def test_q4(self):
        assert _quarter_from_date(date(2025, 10, 1)) == "2025Q4"
        assert _quarter_from_date(date(2025, 12, 31)) == "2025Q4"


# =============================================================================
# TEST 5: PUBLISHED_AT_MAX COMPUTATION
# =============================================================================

class TestPublishedAtMax:
    """Test that published_at_max reflects latest PIT-usable filing."""
    
    def test_max_filing_date_computed(self, sample_signal):
        """Should return the max filing date among PIT-usable filings."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 25 - all filings included, max is Nov 20
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 25))
        
        assert result["coinvest_published_at_max"] == "2025-11-20"
    
    def test_max_filing_date_partial(self, sample_signal):
        """Max filing date should only consider PIT-usable filings."""
        signals = {"RYTM": sample_signal}
        
        # As of Nov 16 - only Baker Bros (Nov 14) and RA Capital (Nov 15) included
        result = _enrich_with_coinvest("RYTM", signals, date(2025, 11, 16))
        
        assert result["coinvest_published_at_max"] == "2025-11-15"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
