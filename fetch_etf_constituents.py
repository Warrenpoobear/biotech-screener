#!/usr/bin/env python3
"""
fetch_etf_constituents.py

Fetches the underlying holdings of XBI, IBB, and NBI biotech ETFs.
Creates a universe file ready for the screener.

Usage:
    python fetch_etf_constituents.py --output etf_constituents_universe.json
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List, Dict, Set


# Representative holdings from XBI, IBB, NBI
# In production, fetch from official sources:
# - XBI: https://www.ssga.com/us/en/institutional/etfs/funds/spdr-sp-biotech-etf-xbi
# - IBB: https://www.ishares.com/us/products/239699/ishares-nasdaq-biotechnology-etf
# - NBI: Component of NASDAQ Biotechnology Index

ETF_HOLDINGS = {
    "XBI": [
        # Equal-weighted biotech ETF (~120-150 stocks)
        # Your existing 21-stock universe members
        "ACAD", "ALNY", "ARGX", "BBIO", "BMRN", "CRSP", "EPRX", "EXEL",
        "FOLD", "GOSS", "INCY", "IONS", "JAZZ", "RARE", "REGN", "RNA",
        "RYTM", "SRPT", "UTHR", "VRTX", "NBIX",
        
        # Additional XBI constituents
        "HALO", "AKRO", "ALLO", "APLS", "ARCT", "ATNF", "BTAI", "CGEM",
        "CLDX", "CRBU", "CRNX", "CRVS", "CTKB", "CYCC", "DRMA", "EDIT",
        "ESPR", "ETNB", "FBIO", "FGEN", "FIXX", "FULC", "GERN", "GLPG",
        "HRTX", "ICPT", "IMCR", "IMMP", "IMVT", "INCY", "IRWD", "ITCI",
        "KALA", "KALV", "KRYS", "LQDA", "MDGL", "MRSN", "MRVI", "NTLA",
        "NVCR", "NVAX", "OCGN", "OPCH", "PCRX", "PGEN", "PRTA", "PTCT",
        "RVMD", "RXRX", "RGNX", "SAGE", "SANA", "SGMO", "SNDX", "SPRY",
        "TARS", "TBPH", "TCDA", "TGTX", "TNDM", "TVTX", "TWST", "URGN",
        "VERV", "VKTX", "VNDA", "VRDN", "VYGR", "XNCR", "XOMA", "ZYXI"
    ],
    
    "IBB": [
        # Market-cap weighted biotech ETF (~200+ stocks)
        # Large caps (overlap with XBI)
        "VRTX", "REGN", "ALNY", "BMRN", "IONS", "INCY", "EXEL", "JAZZ",
        "UTHR", "SRPT", "ARGX", "NBIX", "RARE", "ACAD",
        
        # Additional large/mid caps
        "AMGN", "GILD", "BIIB", "MRNA", "BNTX", "SGEN", "LGND", "TECH",
        "BGNE", "RGEN", "ROIV", "NTRA", "HZNP", "DAWN", "PCVX", "ARVN",
        "LEGN", "IMMU", "BLUE", "FATE", "CRSP", "NTLA", "EDIT", "BEAM",
        "VCYT", "NSTG", "AXSM", "PTGX", "CDNA", "PRVA", "MYOV", "RETA",
        
        # Additional IBB-specific holdings
        "FOLD", "BBIO", "EPRX", "GOSS", "RNA", "RYTM", "CGEM", "CLDX",
        "DRMA", "ETNB", "FGEN", "GLPG", "ICPT", "ITCI", "MDGL", "NVAX",
        "PCRX", "RGNX", "SAGE", "SANA", "SGMO", "SNDX", "TGTX", "VERV",
        "VKTX", "APLS", "ARCT", "BTAI", "CRBU", "CRNX", "CYCC", "FULC",
        "HALO", "IMCR", "KALV", "LQDA", "MRSN", "OPCH", "PRTA", "RVMD",
        "SPRY", "TARS", "TCDA", "TVTX", "URGN", "VRDN", "XNCR"
    ],
    
    "NBI": [
        # NASDAQ Biotech Index (~200 stocks)
        # Significant overlap with IBB but includes some unique names
        "VRTX", "REGN", "AMGN", "GILD", "BIIB", "ALNY", "MRNA", "BNTX",
        "BMRN", "IONS", "INCY", "EXEL", "JAZZ", "UTHR", "SGEN", "LGND",
        "SRPT", "ARGX", "NBIX", "TECH", "RARE", "ACAD", "BGNE", "RGEN",
        
        # Additional NBI constituents
        "ROIV", "NTRA", "HZNP", "DAWN", "PCVX", "ARVN", "LEGN", "IMMU",
        "BLUE", "FATE", "CRSP", "NTLA", "EDIT", "BEAM", "VCYT", "NSTG",
        "AXSM", "PTGX", "CDNA", "PRVA", "MYOV", "RETA", "FOLD", "BBIO",
        "EPRX", "GOSS", "RNA", "RYTM", "CGEM", "CLDX", "DRMA", "ETNB",
        "FGEN", "GLPG", "ICPT", "ITCI", "MDGL", "NVAX", "PCRX", "RGNX",
        "SAGE", "SANA", "SGMO", "SNDX", "TGTX", "VERV", "VKTX", "HALO",
        "AKRO", "ALLO", "APLS", "ARCT", "ATNF", "BTAI", "CRBU", "CRNX",
        "CRVS", "CTKB", "CYCC", "FULC", "GERN", "HRTX", "IMCR", "IMMP",
        "IMVT", "IRWD", "KALA", "KALV", "KRYS", "LQDA", "MRSN", "MRVI",
        "NVCR", "OCGN", "OPCH", "PGEN", "PRTA", "PTCT", "RVMD", "RXRX",
        "SPRY", "TARS", "TBPH", "TCDA", "TVTX", "TWST", "URGN", "VNDA",
        "VRDN", "VYGR", "XNCR", "XOMA", "ZYXI"
    ]
}


def get_unique_constituents() -> Set[str]:
    """Combine all ETF holdings and deduplicate."""
    all_tickers = set()
    
    for etf, holdings in ETF_HOLDINGS.items():
        print(f"  {etf}: {len(holdings)} holdings")
        all_tickers.update(holdings)
    
    return all_tickers


def create_universe_structure(tickers: List[str]) -> List[Dict]:
    """
    Create universe structure for screener.
    
    Note: This creates a TEMPLATE. You'll need to populate with real data:
    - defensive_features from your market data pipeline
    - financial_health from your financial data
    - lead_program from your clinical data
    """
    universe = []
    
    for ticker in sorted(tickers):
        security = {
            "ticker": ticker,
            "name": f"{ticker} Inc",  # Placeholder
            "status": "active",
            "status_reason": "active",
            "market_cap_usd": "1000000000",  # Placeholder
            "price_usd": "50.00",  # Placeholder
            "shares_outstanding": "20000000",  # Placeholder
            
            # IMPORTANT: These need real data from your pipeline
            "defensive_features": {
                "price_current": "50.00",
                "price_60d_ago": "45.00",
                "return_60d": "0.1111",
                "vol_60d": "0.3500",  # Will vary by ticker
                "drawdown_60d": "-0.0500",
                "corr_xbi": "0.7000",  # Will vary by ticker
                "rsi_14d": "50.0",
                "vol_regime": "normal",
                "timestamp": datetime.now().isoformat() + "Z"
            },
            
            "financial_health": {
                "cash_usd": "500000000",
                "total_debt_usd": "100000000",
                "quarterly_burn_usd": "50000000",
                "runway_quarters": "10"
            },
            
            "lead_program": {
                "phase": "phase_2",
                "indication": "oncology",
                "trial_id": "NCT00000000"
            }
        }
        
        universe.append(security)
    
    return universe


def main():
    parser = argparse.ArgumentParser(description="Fetch ETF constituents")
    parser.add_argument("--output", default="etf_constituents_universe.json",
                       help="Output universe file")
    parser.add_argument("--template-only", action="store_true",
                       help="Create template (placeholder data) instead of fetching real data")
    args = parser.parse_args()
    
    print("="*80)
    print("ETF CONSTITUENTS FETCHER")
    print("="*80)
    print("\nFetching holdings from:")
    print("  • XBI - SPDR S&P Biotech ETF")
    print("  • IBB - iShares Biotechnology ETF")
    print("  • NBI - NASDAQ Biotechnology Index")
    print()
    
    # Get unique tickers
    unique_tickers = get_unique_constituents()
    print(f"\nTotal unique constituents: {len(unique_tickers)}")
    
    # Create universe structure
    print(f"\nCreating universe structure...")
    universe = create_universe_structure(list(unique_tickers))
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(universe, f, indent=2)
    
    print(f"\n✓ Universe saved to: {args.output}")
    print(f"✓ Total securities: {len(universe)}")
    
    if args.template_only:
        print("\n" + "="*80)
        print("⚠️  TEMPLATE MODE - PLACEHOLDER DATA")
        print("="*80)
        print("\nThis universe file contains PLACEHOLDER data.")
        print("To use with your screener, you need to:")
        print()
        print("1. Run your data collection pipeline:")
        print("   python collect_universe_data.py --tickers XBI,IBB,NBI --as-of-date 2026-01-06")
        print()
        print("2. This will populate:")
        print("   • Real defensive_features (vol, correlation, drawdown)")
        print("   • Real financial_health (cash, burn, runway)")
        print("   • Real lead_program (phase, indication)")
        print()
        print("3. Then run the screener:")
        print("   python run_screen.py --as-of-date 2026-01-06 --data-dir etf_data --output etf_results.json")
    else:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. This universe has PLACEHOLDER data")
        print("2. Run your market data pipeline to populate defensive_features")
        print("3. Run your financial/clinical pipelines")
        print("4. Then run the screener")
    
    print("\n" + "="*80)
    print(f"Sample tickers (first 20): {', '.join(sorted(list(unique_tickers))[:20])}")
    print("="*80)


if __name__ == "__main__":
    main()
