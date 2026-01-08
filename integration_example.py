#!/usr/bin/env python3
"""
integration_example.py - Complete CFO Extraction → Module 2 Integration

Demonstrates the full pipeline from SEC filings to burn acceleration scoring.
"""

import json
from pathlib import Path
from datetime import date

# Prefer ENHANCED Module 2 if available; fall back to standard.
ENHANCED_M2 = False
try:
    from module_2_financial_ENHANCED import run_module_2  # type: ignore
    ENHANCED_M2 = True
    print("✅ Using ENHANCED Module 2 (module_2_financial_ENHANCED.py)")
except Exception:
    from module_2_financial import run_module_2  # type: ignore
    print("ℹ️  Using standard Module 2 (module_2_financial.py) — burn acceleration fields may be unavailable")

# Simulate CFO extractor output (in production, this comes from cfo_extractor.py)
SAMPLE_CFO_DATA = [
    {
        "ticker": "CVAC",
        "fiscal_period": "Q3",
        "CFO_ytd_current": -285000000.0,  # 9 months YTD
        "CFO_ytd_prev": -190000000.0,     # 6 months YTD (Q2)
        "CFO_ytd_q3": -285000000.0,
        "CFO_fy_annual": None,
        "filing_date": "2024-11-07",
        "period_end_date": "2024-09-30"
    },
    {
        "ticker": "RYTM",
        "fiscal_period": "Q2",
        "CFO_ytd_current": -85000000.0,   # 6 months YTD
        "CFO_ytd_prev": -45000000.0,      # 3 months (Q1)
        "CFO_ytd_q3": None,
        "CFO_fy_annual": None,
        "filing_date": "2024-08-12",
        "period_end_date": "2024-06-30"
    },
    {
        "ticker": "IMMP",
        "fiscal_period": "Q2",
        "CFO_ytd_current": -220000000.0,  # 6 months YTD
        "CFO_ytd_prev": -100000000.0,     # 3 months (Q1)
        "CFO_ytd_q3": None,
        "CFO_fy_annual": None,
        "filing_date": "2024-08-05",
        "period_end_date": "2024-06-30"
    }
]

# Existing financial data (Cash, NetIncome, R&D)
SAMPLE_FINANCIAL_DATA = [
    {
        "ticker": "CVAC",
        "Cash": 500000000.0,
        "NetIncome": -100000000.0,
        "R&D": 80000000.0
    },
    {
        "ticker": "RYTM",
        "Cash": 200000000.0,
        "NetIncome": -50000000.0,
        "R&D": 40000000.0
    },
    {
        "ticker": "IMMP",
        "Cash": 1000000000.0,
        "NetIncome": -150000000.0,
        "R&D": 120000000.0
    }
]

# Market data
SAMPLE_MARKET_DATA = [
    {"ticker": "CVAC", "market_cap": 2000000000.0, "avg_volume": 500000, "price": 20.0},
    {"ticker": "RYTM", "market_cap": 800000000.0, "avg_volume": 200000, "price": 15.0},
    {"ticker": "IMMP", "market_cap": 5000000000.0, "avg_volume": 1000000, "price": 50.0}
]


def merge_cfo_with_financial_data(financial_data, cfo_data):
    """
    Merge CFO data into existing financial data.
    
    Args:
        financial_data: Base financial data (Cash, NetIncome, etc.)
        cfo_data: CFO data from extractor
    
    Returns:
        Enhanced financial data with CFO fields
    """
    # Create lookup
    cfo_lookup = {rec['ticker']: rec for rec in cfo_data}
    
    # Merge
    enhanced_data = []
    for fin_rec in financial_data:
        ticker = fin_rec['ticker']
        merged = fin_rec.copy()
        
        # Add CFO fields if available
        if ticker in cfo_lookup:
            cfo_rec = cfo_lookup[ticker]
            merged.update({
                'fiscal_period': cfo_rec['fiscal_period'],
                'CFO_ytd_current': cfo_rec['CFO_ytd_current'],
                'CFO_ytd_prev': cfo_rec.get('CFO_ytd_prev'),
                'CFO_fy_annual': cfo_rec.get('CFO_fy_annual'),
                'CFO_ytd_q3': cfo_rec.get('CFO_ytd_q3'),
            })
        
        enhanced_data.append(merged)
    
    return enhanced_data


def demonstrate_module_2_without_cfo():
    """Show Module 2 scoring WITHOUT CFO data (baseline)"""
    print("\n" + "="*80)
    print("BASELINE: Module 2 WITHOUT CFO Data")
    print("="*80)
    
    universe = ["CVAC", "RYTM", "IMMP"]
    
    results = run_module_2(universe, SAMPLE_FINANCIAL_DATA, SAMPLE_MARKET_DATA)
    
    print("\nResults (No Burn Acceleration):")
    print("-" * 80)
    for r in results:
        print(f"{r['ticker']:6s} | Score: {r['financial_normalized']:5.1f} | "
              f"Runway: {r['runway_months']:5.1f}mo | "
              f"Dilution: {r['dilution_score']:5.1f}")
    
    return results


def demonstrate_module_2_with_cfo():
    """Show Module 2 scoring WITH CFO data (enhanced)"""
    print("\n" + "="*80)
    print("ENHANCED: Module 2 WITH CFO Data & Burn Acceleration")
    print("="*80)

    # Merge CFO data
    enhanced_financial_data = merge_cfo_with_financial_data(
        SAMPLE_FINANCIAL_DATA,
        SAMPLE_CFO_DATA
    )
    
    universe = ["CVAC", "RYTM", "IMMP"]
    as_of_date = "2024-12-01"

    # Call signature depends on whether enhanced module is loaded
    if ENHANCED_M2:
        results = run_module_2(
            universe,
            enhanced_financial_data,
            SAMPLE_MARKET_DATA,
            catalyst_summaries=None,  # Optional
            as_of_date=as_of_date
        )
    else:
        results = run_module_2(universe, enhanced_financial_data, SAMPLE_MARKET_DATA)
    
    print("\nResults (WITH Burn Acceleration):")
    print("-" * 80)
    print(f"{'Ticker':<6s} | {'Score':>5s} | {'Runway':>8s} | {'Dilution':>8s} | "
          f"{'Burn Accel':>11s} | {'CFO Flag':>10s}")
    print("-" * 80)
    
    for r in results:
        burn_accel = r.get('burn_acceleration', 1.0)
        cfo_flag = r.get('cfo_quality_flag', 'N/A')
        print(f"{r['ticker']:6s} | {r['financial_normalized']:5.1f} | "
              f"{r['runway_months']:5.1f}mo | {r['dilution_score']:8.1f} | "
              f"{burn_accel:11.2f}x | {cfo_flag:>10s}")
    
    # Detailed breakdown for one ticker
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN: CVAC")
    print("="*80)
    
    cvac = next(r for r in results if r['ticker'] == 'CVAC')
    
    print(f"\nCash Runway:")
    print(f"  Cash: ${cvac['cash_to_mcap'] * SAMPLE_MARKET_DATA[0]['market_cap']:,.0f}")
    print(f"  Monthly Burn: ${cvac['monthly_burn']:,.0f}")
    print(f"  Runway: {cvac['runway_months']:.1f} months")
    
    # Only show burn acceleration if available
    if 'burn_acceleration' in cvac:
        print(f"\nBurn Acceleration (NEW!):")
        print(f"  Recent Monthly Burn: ${cvac['burn_recent_m']:,.0f}")
        print(f"  4Q Avg Monthly Burn: ${cvac['burn_4q_avg_m']:,.0f}")
        print(f"  Acceleration: {cvac['burn_acceleration']:.2f}x")
        
        if cvac['burn_acceleration'] >= 1.3:
            print(f"  ⚠️  ACCELERATING BURN DETECTED!")
            print(f"  → Dilution penalty amplified by {1.5 if cvac['burn_acceleration'] < 1.6 else 1.75}x")
        
        print(f"\nModule 2 automatically converted YTD → Quarterly:")
        # Show the math
        cfo_data = next(c for c in SAMPLE_CFO_DATA if c['ticker'] == 'CVAC')
        q3_cfo = cfo_data['CFO_ytd_current'] - cfo_data['CFO_ytd_prev']
        print(f"  Q3 YTD CFO: ${cfo_data['CFO_ytd_current']:,.0f} (9 months)")
        print(f"  Q2 YTD CFO: ${cfo_data['CFO_ytd_prev']:,.0f} (6 months)")
        print(f"  Q3 Quarterly CFO: ${cfo_data['CFO_ytd_current']:,.0f} - ${cfo_data['CFO_ytd_prev']:,.0f}")
        print(f"                   = ${q3_cfo:,.0f} (3 months)")
    else:
        print(f"\n⚠️  Burn acceleration not available (using standard Module 2)")
        print(f"💡 To enable: use module_2_financial_ENHANCED.py")
    
    return results


def compare_scores():
    """Compare scores with and without CFO data"""
    print("\n" + "="*80)
    print("SCORE COMPARISON: Before vs After CFO Integration")
    print("="*80)
    
    # Run both
    baseline = demonstrate_module_2_without_cfo()
    enhanced = demonstrate_module_2_with_cfo()
    
    # Compare
    print("\n" + "="*80)
    print("IMPACT ANALYSIS")
    print("="*80)
    
    # Check if enhanced fields available
    has_burn_accel = 'burn_acceleration' in enhanced[0] if enhanced else False
    
    if has_burn_accel:
        print(f"\n{'Ticker':<6s} | {'Before':>7s} | {'After':>7s} | {'Change':>7s} | {'Burn Accel':>11s}")
        print("-" * 60)
        
        for base_r, enh_r in zip(baseline, enhanced):
            ticker = base_r['ticker']
            before = base_r['financial_normalized']
            after = enh_r['financial_normalized']
            change = after - before
            burn_accel = enh_r['burn_acceleration']
            
            indicator = "📉" if change < -5 else "📈" if change > 5 else "➡️"
            
            print(f"{ticker:6s} | {before:7.1f} | {after:7.1f} | {change:+7.1f} {indicator} | {burn_accel:11.2f}x")
        
        print("\n💡 Key Insights:")
        print("  - Tickers with burn acceleration >1.3x get dilution penalties")
        print("  - This catches 'silent killers' where runway looks OK but burn is ramping")
        print("  - Scores now differentiate based on actual CFO trends, not just snapshots")
    else:
        print("\n⚠️  Enhanced features not available in current Module 2")
        print("💡 To see burn acceleration impact:")
        print("   1. Copy module_2_financial_ENHANCED.py to your directory")
        print("   2. Rename it to module_2_financial.py")
        print("   3. Re-run this demo")
        print("\n📊 Basic comparison (without burn acceleration):")
        print(f"\n{'Ticker':<6s} | {'Before':>7s} | {'After':>7s} | {'Change':>7s}")
        print("-" * 40)
        
        for base_r, enh_r in zip(baseline, enhanced):
            ticker = base_r['ticker']
            before = base_r['financial_normalized']
            after = enh_r['financial_normalized']
            change = after - before
            
            indicator = "📉" if change < -5 else "📈" if change > 5 else "➡️"
            
            print(f"{ticker:6s} | {before:7.1f} | {after:7.1f} | {change:+7.1f} {indicator}")


def main():
    """Run complete integration demonstration"""
    print("\n" + "="*80)
    print("CFO EXTRACTOR → MODULE 2 INTEGRATION DEMO")
    print("="*80)
    print("\nThis demonstrates the complete pipeline:")
    print("  1. CFO data extracted from SEC filings (simulated)")
    print("  2. Merged with existing financial data")
    print("  3. Module 2 converts YTD → quarterly automatically")
    print("  4. Burn acceleration detected and scored")
    
    # Run comparison
    compare_scores()
    
    print("\n" + "="*80)
    print("✅ INTEGRATION DEMO COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run: python test_cfo_extractor.py (validate extractor)")
    print("  2. Extract real CFO data from SEC filings")
    print("  3. Merge with your production financial_data.json")
    print("  4. Run full screening pipeline with burn acceleration!")
    print("="*80)


if __name__ == "__main__":
    main()
