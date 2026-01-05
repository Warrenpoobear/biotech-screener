#!/usr/bin/env python3
"""
Quick test of data collectors on a single ticker.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collectors import yahoo_collector, sec_collector, trials_collector

def test_collectors():
    """Test all three collectors with a well-known ticker."""
    
    ticker = "VRTX"
    company = "Vertex Pharmaceuticals"
    
    print(f"\n🧪 Testing data collectors with {ticker}...")
    print("="*60)
    
    # Test Yahoo Finance
    print(f"\n1. Testing Yahoo Finance collector...")
    yahoo_data = yahoo_collector.collect_yahoo_data(ticker, force_refresh=True)
    
    if yahoo_data.get('success'):
        price = yahoo_data['price']['current']
        mcap = yahoo_data['market_cap']['value'] / 1e9
        print(f"   ✓ Price: ${price:.2f}")
        print(f"   ✓ Market Cap: ${mcap:.2f}B")
    else:
        print(f"   ✗ Error: {yahoo_data.get('error')}")
        return False
    
    # Test SEC EDGAR
    print(f"\n2. Testing SEC EDGAR collector...")
    sec_data = sec_collector.collect_sec_data(ticker, force_refresh=True)
    
    if sec_data.get('success'):
        cash = sec_data['financials'].get('cash')
        coverage = sec_data['coverage']['pct_complete']
        print(f"   ✓ Cash: ${cash/1e6:.0f}M" if cash else "   ✓ No cash data")
        print(f"   ✓ Coverage: {coverage:.1f}%")
    else:
        print(f"   ✗ Error: {sec_data.get('error')}")
        return False
    
    # Test ClinicalTrials.gov
    print(f"\n3. Testing ClinicalTrials.gov collector...")
    trials_data = trials_collector.collect_trials_data(ticker, company, force_refresh=True)
    
    if trials_data.get('success'):
        total = trials_data['summary']['total_trials']
        active = trials_data['summary']['active_trials']
        lead_stage = trials_data['summary']['lead_stage']
        print(f"   ✓ Total trials: {total}")
        print(f"   ✓ Active trials: {active}")
        print(f"   ✓ Lead stage: {lead_stage}")
    else:
        print(f"   ✗ Error: {trials_data.get('error')}")
        return False
    
    print("\n" + "="*60)
    print("✅ All collectors working correctly!")
    print("\nReady to run full pipeline with: python collect_universe_data.py\n")
    
    return True

if __name__ == "__main__":
    success = test_collectors()
    sys.exit(0 if success else 1)
