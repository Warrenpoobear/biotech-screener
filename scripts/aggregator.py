"""
Elite Position Aggregator for Wake Robin Biotech Alpha System

This module aggregates 13F holdings across elite biotech managers to generate
co-investment signals.

Core signals:
1. Overlap count: How many elite managers hold this position?
2. Weighted conviction: Sum of (manager_weight × position_weight)
3. Net change: Did managers add or reduce this quarter?
4. New position flags: Which managers initiated new positions?

Usage:
    from wake_robin.providers.sec_13f.aggregator import ElitePositionAggregator
    
    agg = ElitePositionAggregator()
    signals = agg.compute_signals(as_of_date='2025-09-30')
    
    for ticker, signal in signals.items():
        print(f"{ticker}: {signal['overlap_count']} managers, score={signal['conviction_score']:.2f}")
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from collections import defaultdict
import json
import hashlib

from elite_managers import (
    ELITE_MANAGERS, get_manager_by_cik, get_manager_weight,
    get_tier_1_managers, get_all_ciks,
)
from edgar_13f import SEC13FFetcher, Filing13F, Holding, get_manager_holdings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ManagerPosition:
    """A single manager's position in a security."""
    manager_cik: str
    manager_name: str
    manager_tier: int
    manager_weight: float
    
    cusip: str
    ticker: Optional[str]
    issuer_name: str
    
    shares: int
    value: int
    position_weight: float  # % of manager's portfolio
    
    report_date: date
    filing_date: date
    
    # Quarter-over-quarter change (if prior data available)
    shares_change: Optional[int] = None
    value_change: Optional[int] = None
    is_new_position: bool = False
    is_exit: bool = False


@dataclass 
class AggregatedSignal:
    """Aggregated signal for a single security across all elite managers."""
    cusip: str
    ticker: Optional[str]
    issuer_name: str
    
    # Core signals
    overlap_count: int  # How many managers hold this
    tier_1_count: int   # How many Tier 1 managers hold this
    total_value: int    # Sum of all manager positions
    
    # Conviction score (0-100)
    conviction_score: float
    
    # Manager list
    holders: list[str]  # Manager short names
    tier_1_holders: list[str]
    
    # Change signals
    managers_adding: list[str]  # Managers who increased position
    managers_reducing: list[str]  # Managers who decreased
    new_positions: list[str]  # Managers with new positions this quarter
    exits: list[str]  # Managers who exited
    
    # Position details
    positions: list[ManagerPosition] = field(default_factory=list)
    
    # Metadata
    as_of_date: Optional[date] = None
    
    def to_dict(self) -> dict:
        return {
            'cusip': self.cusip,
            'ticker': self.ticker,
            'issuer_name': self.issuer_name,
            'overlap_count': self.overlap_count,
            'tier_1_count': self.tier_1_count,
            'total_value': self.total_value,
            'conviction_score': self.conviction_score,
            'holders': self.holders,
            'tier_1_holders': self.tier_1_holders,
            'managers_adding': self.managers_adding,
            'managers_reducing': self.managers_reducing,
            'new_positions': self.new_positions,
            'exits': self.exits,
        }


# =============================================================================
# AGGREGATOR
# =============================================================================

class ElitePositionAggregator:
    """
    Aggregates 13F holdings across elite managers.
    
    Point-in-time safety:
    - Uses filing_date (not report_date) as knowledge date
    - For backtesting, only includes filings filed before as_of_date
    """
    
    def __init__(
        self,
        managers: Optional[list[dict]] = None,
        cache_dir: str = 'data/13f_cache',
        min_position_value: int = 100_000,  # Ignore tiny positions
    ):
        """
        Initialize aggregator.
        
        Args:
            managers: List of manager dicts (from elite_managers). Default: all.
            cache_dir: Directory for caching 13F data
            min_position_value: Minimum position value to include
        """
        self.managers = managers or ELITE_MANAGERS
        self.cache_dir = cache_dir
        self.min_position_value = min_position_value
        
        self.fetcher = SEC13FFetcher(cache_dir=cache_dir)
        
        # Cache for holdings
        self._holdings_cache: dict[str, list[Holding]] = {}
        self._filing_cache: dict[str, Filing13F] = {}
    
    def fetch_all_holdings(
        self,
        quarters_back: int = 1,
        as_of_date: Optional[date] = None,
    ) -> dict[str, list[ManagerPosition]]:
        """
        Fetch holdings for all elite managers.
        
        Args:
            quarters_back: Which quarter (1 = most recent, 2 = prior, etc.)
            as_of_date: For point-in-time, only use filings filed by this date
            
        Returns:
            Dict mapping manager CIK to list of ManagerPosition
        """
        all_positions = {}
        
        for manager in self.managers:
            cik = manager['cik']
            short_name = manager['short_name']
            
            print(f"  Fetching {short_name} (CIK {cik})...")
            
            try:
                # Get filings
                filings = self.fetcher.get_recent_filings(cik, count=quarters_back + 1)
                
                if not filings:
                    print(f"    No filings found")
                    continue
                
                # Filter by as_of_date if specified
                if as_of_date:
                    filings = [f for f in filings if f.filing_date <= as_of_date]
                
                if not filings:
                    print(f"    No filings before {as_of_date}")
                    continue
                
                # Get target quarter's filing
                filing = filings[0] if quarters_back == 1 else (
                    filings[quarters_back - 1] if len(filings) >= quarters_back else filings[-1]
                )
                
                # Get prior quarter for change detection
                prior_filing = filings[1] if len(filings) > 1 else None
                
                # Parse holdings
                holdings = self.fetcher.parse_holdings(filing)
                prior_holdings = (
                    self.fetcher.parse_holdings(prior_filing) if prior_filing else []
                )
                
                # Build prior holdings lookup
                prior_by_cusip = {h.cusip: h for h in prior_holdings}
                
                # Calculate portfolio totals
                total_value = sum(h.value for h in holdings)
                
                # Convert to ManagerPosition
                manager_weight = get_manager_weight(manager)
                positions = []
                
                for h in holdings:
                    if h.value < self.min_position_value:
                        continue
                    
                    position_weight = h.value / total_value if total_value else 0
                    
                    # Check for changes vs prior quarter
                    prior = prior_by_cusip.get(h.cusip)
                    shares_change = h.shares - prior.shares if prior else None
                    value_change = h.value - prior.value if prior else None
                    is_new = prior is None
                    
                    pos = ManagerPosition(
                        manager_cik=cik,
                        manager_name=short_name,
                        manager_tier=manager['tier'],
                        manager_weight=manager_weight,
                        cusip=h.cusip,
                        ticker=h.ticker,
                        issuer_name=h.issuer_name,
                        shares=h.shares,
                        value=h.value,
                        position_weight=position_weight,
                        report_date=filing.report_date,
                        filing_date=filing.filing_date,
                        shares_change=shares_change,
                        value_change=value_change,
                        is_new_position=is_new,
                    )
                    positions.append(pos)
                
                # Check for exits (in prior but not current)
                current_cusips = {h.cusip for h in holdings}
                for cusip, prior_h in prior_by_cusip.items():
                    if cusip not in current_cusips:
                        # This is an exit
                        pos = ManagerPosition(
                            manager_cik=cik,
                            manager_name=short_name,
                            manager_tier=manager['tier'],
                            manager_weight=manager_weight,
                            cusip=cusip,
                            ticker=prior_h.ticker,
                            issuer_name=prior_h.issuer_name,
                            shares=0,
                            value=0,
                            position_weight=0,
                            report_date=filing.report_date,
                            filing_date=filing.filing_date,
                            shares_change=-prior_h.shares,
                            value_change=-prior_h.value,
                            is_exit=True,
                        )
                        positions.append(pos)
                
                all_positions[cik] = positions
                print(f"    {len(holdings)} holdings, {len(positions)} positions tracked")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        return all_positions
    
    def compute_signals(
        self,
        quarters_back: int = 1,
        as_of_date: Optional[date] = None,
        min_overlap: int = 1,
    ) -> dict[str, AggregatedSignal]:
        """
        Compute aggregated signals across all managers.
        
        Args:
            quarters_back: Which quarter to analyze
            as_of_date: Point-in-time filter
            min_overlap: Minimum number of managers holding
            
        Returns:
            Dict mapping ticker (or CUSIP if no ticker) to AggregatedSignal
        """
        print(f"Fetching holdings (quarters_back={quarters_back})...")
        all_positions = self.fetch_all_holdings(quarters_back, as_of_date)
        
        # Group by security (using CUSIP as key)
        by_cusip: dict[str, list[ManagerPosition]] = defaultdict(list)
        
        for cik, positions in all_positions.items():
            for pos in positions:
                by_cusip[pos.cusip].append(pos)
        
        print(f"Aggregating {len(by_cusip)} unique securities...")
        
        # Compute signals
        signals = {}
        
        for cusip, positions in by_cusip.items():
            # Skip exits-only (no current holders)
            current_holders = [p for p in positions if not p.is_exit]
            if len(current_holders) < min_overlap:
                continue
            
            # Determine ticker and name from most valuable position
            top_pos = max(current_holders, key=lambda p: p.value)
            ticker = top_pos.ticker or cusip
            issuer_name = top_pos.issuer_name
            
            # Count overlaps
            overlap_count = len(current_holders)
            tier_1_holders = [p for p in current_holders if p.manager_tier == 1]
            tier_1_count = len(tier_1_holders)
            
            # Total value across managers
            total_value = sum(p.value for p in current_holders)
            
            # Compute conviction score
            conviction_score = self._compute_conviction_score(current_holders)
            
            # Track changes
            managers_adding = [
                p.manager_name for p in current_holders
                if p.shares_change and p.shares_change > 0
            ]
            managers_reducing = [
                p.manager_name for p in current_holders
                if p.shares_change and p.shares_change < 0
            ]
            new_positions = [
                p.manager_name for p in current_holders
                if p.is_new_position
            ]
            exits = [
                p.manager_name for p in positions
                if p.is_exit
            ]
            
            signal = AggregatedSignal(
                cusip=cusip,
                ticker=ticker if ticker != cusip else None,
                issuer_name=issuer_name,
                overlap_count=overlap_count,
                tier_1_count=tier_1_count,
                total_value=total_value,
                conviction_score=conviction_score,
                holders=[p.manager_name for p in current_holders],
                tier_1_holders=[p.manager_name for p in tier_1_holders],
                managers_adding=managers_adding,
                managers_reducing=managers_reducing,
                new_positions=new_positions,
                exits=exits,
                positions=current_holders,
                as_of_date=as_of_date,
            )
            
            # Key by ticker if available, else CUSIP
            key = ticker if ticker and ticker != cusip else cusip
            signals[key] = signal
        
        return signals
    
    def _compute_conviction_score(self, positions: list[ManagerPosition]) -> float:
        """
        Compute conviction score (0-100) for a security.
        
        Factors:
        - Number of holders (more = higher)
        - Tier of holders (Tier 1 = higher weight)
        - Position size in each portfolio (larger = higher conviction)
        - Recent additions (new positions boost score)
        """
        if not positions:
            return 0.0
        
        score = 0.0
        
        # Base score from overlap (max 40 points)
        overlap = len(positions)
        overlap_score = min(40, overlap * 8)  # 8 points per holder, max 40
        
        # Tier weighting (max 30 points)
        tier_score = 0
        for p in positions:
            tier_score += p.manager_weight * 10  # Up to ~12 per Tier 1 manager
        tier_score = min(30, tier_score)
        
        # Position concentration (max 20 points)
        # Average position weight across holders
        avg_weight = sum(p.position_weight for p in positions) / len(positions)
        concentration_score = min(20, avg_weight * 200)  # 10% avg position = 20 points
        
        # Momentum bonus (max 10 points)
        # New positions and additions
        new_count = sum(1 for p in positions if p.is_new_position)
        adding_count = sum(1 for p in positions if p.shares_change and p.shares_change > 0)
        momentum_score = min(10, (new_count * 3) + (adding_count * 2))
        
        score = overlap_score + tier_score + concentration_score + momentum_score
        
        return min(100.0, score)
    
    def get_top_signals(
        self,
        n: int = 20,
        min_overlap: int = 2,
        tier_1_only: bool = False,
        **kwargs
    ) -> list[AggregatedSignal]:
        """
        Get top N securities by conviction score.
        
        Args:
            n: Number of results
            min_overlap: Minimum manager overlap
            tier_1_only: Only include securities held by Tier 1 managers
            **kwargs: Passed to compute_signals()
        """
        signals = self.compute_signals(min_overlap=min_overlap, **kwargs)
        
        # Filter
        filtered = signals.values()
        if tier_1_only:
            filtered = [s for s in filtered if s.tier_1_count > 0]
        
        # Sort by conviction score
        ranked = sorted(filtered, key=lambda s: -s.conviction_score)
        
        return ranked[:n]


# =============================================================================
# SCORING INTERFACE (for integration with Module 3/6)
# =============================================================================

def get_elite_conviction_score(
    ticker: str,
    as_of_date: Optional[date] = None,
    cache_dir: str = 'data/13f_cache',
) -> dict:
    """
    Get elite manager conviction signal for a single ticker.
    
    Returns dict with:
        - conviction_score: 0-100
        - overlap_count: number of elite managers holding
        - tier_1_count: number of Tier 1 managers holding
        - holders: list of manager names
        - signal_strength: 'strong', 'moderate', 'weak', or 'none'
    """
    # This would ideally use a pre-computed cache
    # For now, this is a stub that would integrate with the aggregator
    
    return {
        'ticker': ticker,
        'conviction_score': 0,
        'overlap_count': 0,
        'tier_1_count': 0,
        'holders': [],
        'signal_strength': 'none',
        'note': 'Run aggregator to populate signals',
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("Elite Position Aggregator")
    print("=" * 60)
    
    # Use Tier 1 only for faster testing
    tier_1_managers = get_tier_1_managers()
    print(f"Analyzing {len(tier_1_managers)} Tier 1 managers...")
    print()
    
    agg = ElitePositionAggregator(managers=tier_1_managers)
    
    top_signals = agg.get_top_signals(n=30, min_overlap=2)
    
    print()
    print("TOP CONVICTION SIGNALS (held by 2+ Tier 1 managers)")
    print("-" * 80)
    print(f"{'Ticker':<10} {'Name':<30} {'Score':>6} {'Overlap':>8} {'T1':>4} {'Holders'}")
    print("-" * 80)
    
    for signal in top_signals:
        from cusip_resolver import resolve_cusip
        ticker = signal.ticker or resolve_cusip(signal.cusip) or signal.cusip[:8]
        name = signal.issuer_name[:28]
        holders = ', '.join(signal.holders[:3])
        if len(signal.holders) > 3:
            holders += f' +{len(signal.holders) - 3}'
        
        print(f"{ticker:<10} {name:<30} {signal.conviction_score:>6.1f} {signal.overlap_count:>8} {signal.tier_1_count:>4} {holders}")
