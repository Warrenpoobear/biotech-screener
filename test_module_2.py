#!/usr/bin/env python3
"""
test_module_2.py - Unit Tests for Module 2 Financial Scoring

Run: python test_module_2.py
"""

from module_2_financial import calculate_cash_runway, calculate_dilution_risk, score_liquidity, score_financial_health, run_module_2


def test_cash_runway():
    print("\n" + "="*60)
    print("TEST 1: Cash Runway")
    print("="*60)
    
    # Good runway (30 months)
    fin = {'Cash': 500e6, 'NetIncome': -50e6}
    runway, burn, score = calculate_cash_runway(fin, {})
    assert runway == 30.0 and score == 100.0
    print(f"✅ 30-month runway → Score {score}")
    
    # Short runway (3 months)
    fin = {'Cash': 50e6, 'NetIncome': -50e6}
    runway, burn, score = calculate_cash_runway(fin, {})
    assert runway == 3.0 and score == 10.0
    print(f"✅ 3-month runway → Score {score}")
    
    print("✅ Cash runway tests passed!")


def test_dilution_risk():
    print("\n" + "="*60)
    print("TEST 2: Dilution Risk")
    print("="*60)
    
    # Strong (40% cash/mcap)
    fin = {'Cash': 400e6}
    mkt = {'market_cap': 1000e6}
    ratio, score = calculate_dilution_risk(fin, mkt, None)
    assert ratio == 0.40 and score == 100.0
    print(f"✅ 40% cash/mcap → Score {score}")
    
    # High risk (3% cash/mcap)
    fin = {'Cash': 30e6}
    ratio, score = calculate_dilution_risk(fin, mkt, None)
    assert ratio == 0.03 and score == 10.0
    print(f"✅ 3% cash/mcap → Score {score}")
    
    print("✅ Dilution risk tests passed!")


def test_liquidity():
    print("\n" + "="*60)
    print("TEST 3: Liquidity")
    print("="*60)
    
    # Large cap + high volume
    mkt = {'market_cap': 15e9, 'avg_volume': 2e6, 'price': 50.0}
    score = score_liquidity(mkt)
    assert score == 100.0
    print(f"✅ $15B cap, $100M volume → Score {score}")
    
    # Micro cap + illiquid
    mkt = {'market_cap': 150e6, 'avg_volume': 50000, 'price': 5.0}
    score = score_liquidity(mkt)
    assert score < 40.0
    print(f"✅ $150M cap, $250K volume → Score {score:.1f}")
    
    print("✅ Liquidity tests passed!")


def test_composite():
    print("\n" + "="*60)
    print("TEST 4: Composite Scoring")
    print("="*60)
    
    # Strong company
    fin = {'Cash': 1000e6, 'NetIncome': -100e6}
    mkt = {'market_cap': 3000e6, 'avg_volume': 1e6, 'price': 30.0}
    result = score_financial_health("STRONG", fin, mkt)
    assert result['financial_normalized'] > 80.0
    print(f"✅ Strong company → Score {result['financial_normalized']:.1f}")
    
    # Weak company
    fin = {'Cash': 30e6, 'NetIncome': -30e6}
    mkt = {'market_cap': 200e6, 'avg_volume': 50000, 'price': 5.0}
    result = score_financial_health("WEAK", fin, mkt)
    assert result['financial_normalized'] < 40.0
    print(f"✅ Weak company → Score {result['financial_normalized']:.1f}")
    
    print("✅ Composite scoring tests passed!")


def test_batch():
    print("\n" + "="*60)
    print("TEST 5: Batch Scoring")
    print("="*60)
    
    universe = ['A', 'B', 'C']
    financial_data = [
        {'ticker': 'A', 'Cash': 500e6, 'NetIncome': -100e6},
        {'ticker': 'B', 'Cash': 100e6, 'NetIncome': -50e6},
        {'ticker': 'C', 'Cash': 0}
    ]
    market_data = [
        {'ticker': 'A', 'market_cap': 2e9, 'avg_volume': 500000, 'price': 20.0},
        {'ticker': 'B', 'market_cap': 400e6, 'avg_volume': 100000, 'price': 10.0},
        {'ticker': 'C', 'market_cap': 500e6, 'avg_volume': 200000, 'price': 15.0}
    ]
    
    results = run_module_2(universe, financial_data, market_data)
    assert len(results) == 3
    scores = [r['financial_normalized'] for r in results]
    assert len(set(scores)) > 1  # Different scores
    print(f"✅ Scored {len(results)} tickers: {scores}")
    
    print("✅ Batch scoring tests passed!")


def main():
    print("\n" + "="*80)
    print("MODULE 2 FINANCIAL SCORING - TEST SUITE")
    print("="*80)
    
    try:
        test_cash_runway()
        test_dilution_risk()
        test_liquidity()
        test_composite()
        test_batch()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ Module 2 ready for integration!\n")
        return 0
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
