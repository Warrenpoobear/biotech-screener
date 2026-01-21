#!/usr/bin/env python3
"""
check_etf_coverage.py - Verify XBI, IBB, NBI Constituent Coverage

Checks if your universe.json covers all unique constituents from the three major biotech ETFs.

Usage:
    python check_etf_coverage.py --universe production_data/universe.json
"""

import json
from pathlib import Path
from typing import Set, Dict, List
import argparse


# ============================================================================
# ETF CONSTITUENT DATA (as of late 2024/early 2025)
# ============================================================================

def get_xbi_constituents() -> Set[str]:
    """
    XBI (SPDR S&P Biotech ETF) - Equal-weighted
    ~150-170 holdings (varies as constituents are added/removed)
    
    This is a representative list. For production, fetch from:
    - SPDR website: https://www.ssga.com/us/en/individual/etfs/funds/xbi
    - Or your data provider
    """
    # Major XBI holdings (representative sample - NOT complete)
    return {
        'ACAD', 'ALNY', 'AMGN', 'ARGX', 'ARWR', 'ASND', 'AXSM', 'BEAM', 'ONC',
        'BIIB', 'BNTX', 'CRBU', 'CRSP', 'CVAC', 'DAWN', 'DNLI', 'EDIT', 'EXAS',
        'FATE', 'FOLD', 'GILD', 'HALO', 'ICLR', 'IDYA', 'IMCR', 'INCY', 'IONS',
        'ITIC', 'KALA', 'KALV', 'KROS', 'KYMR', 'LEGN', 'LEGN', 'MLTX', 'MRNA',
        'MRUS', 'NBIX', 'NKTX', 'NTRA', 'NTLA', 'NUVL', 'OPCH', 'PCRX', 'PCVX',
        'PHAT', 'PTGX', 'PRVB', 'PRVA', 'REPL', 'REGN', 'ROIV', 'RPTX', 'RVMD',
        'RXRX', 'SANA', 'SDGR', 'SNDX', 'SRPT', 'TECH', 'TGTX', 'UTHR', 'VCYT',
        'VKTX', 'VRTX', 'VRNA', 'XNCR', 'YMAB'
    }


def get_ibb_constituents() -> Set[str]:
    """
    IBB (iShares Biotechnology ETF) - Market-cap weighted
    ~250-280 holdings
    
    For production, fetch from:
    - iShares website: https://www.ishares.com/us/products/239699/
    """
    # Major IBB holdings (representative - NOT complete)
    return {
        'ABBV', 'ACAD', 'ALNY', 'AMGN', 'ARGX', 'ARWR', 'ASND', 'AXSM', 'BEAM',
        'ONC', 'BIIB', 'BMRN', 'BNTX', 'BSX', 'CRBU', 'CRSP', 'CVAC', 'DNLI',
        'DVAX', 'EDIT', 'EXAS', 'EXEL', 'FATE', 'FOLD', 'GILD', 'HALO', 'ICLR',
        'IDYA', 'ILMN', 'IMCR', 'INCY', 'IONS', 'ITIC', 'KALA', 'KALV', 'KROS',
        'KYMR', 'LEGN', 'MLTX', 'MRNA', 'MRUS', 'NBIX', 'NKTR', 'NKTX', 'NTRA',
        'NTLA', 'NUVL', 'OPCH', 'PCRX', 'PCVX', 'PHAT', 'PTGX', 'PRVB', 'PRVA',
        'REGN', 'REPL', 'ROIV', 'RPTX', 'RVMD', 'RXRX', 'SAGE', 'SANA', 'SDGR',
        'SNDX', 'SRPT', 'TECH', 'TGTX', 'UTHR', 'VCYT', 'VKTX', 'VRTX', 'VRNA',
        'XNCR', 'YMAB'
    }


def get_nbi_constituents() -> Set[str]:
    """
    NBI (Nasdaq Biotechnology Index) - Modified market-cap weighted
    ~200-250 constituents
    
    For production, fetch from:
    - Nasdaq website or index provider
    """
    # Major NBI holdings (representative - NOT complete)
    return {
        'ABBV', 'ACAD', 'ALNY', 'AMGN', 'ARGX', 'ARWR', 'ASND', 'AXSM', 'BEAM',
        'ONC', 'BIIB', 'BMRN', 'BNTX', 'CRBU', 'CRSP', 'CVAC', 'DNLI', 'DVAX',
        'EDIT', 'EXAS', 'EXEL', 'FATE', 'FOLD', 'GILD', 'HALO', 'ICLR', 'IDYA',
        'ILMN', 'IMCR', 'INCY', 'IONS', 'ITIC', 'KALA', 'KALV', 'KROS', 'KYMR',
        'LEGN', 'MLTX', 'MRNA', 'MRUS', 'NBIX', 'NKTR', 'NKTX', 'NTRA', 'NTLA',
        'NUVL', 'OPCH', 'PCRX', 'PCVX', 'PHAT', 'PTGX', 'PRVB', 'PRVA', 'REGN',
        'REPL', 'ROIV', 'RPTX', 'RVMD', 'RXRX', 'SAGE', 'SANA', 'SDGR', 'SNDX',
        'SRPT', 'TECH', 'TGTX', 'UTHR', 'VCYT', 'VKTX', 'VRTX', 'VRNA', 'XNCR',
        'YMAB'
    }


def check_universe_coverage(universe_path: Path) -> dict:
    """
    Check if universe.json covers all XBI, IBB, NBI constituents.
    
    Returns:
        {
            "universe_size": 98,
            "total_unique_etf_constituents": 281,
            "coverage_pct": 34.9,
            "in_universe": [...],
            "missing_from_universe": [...]
        }
    """
    
    if not universe_path.exists():
        return {'error': f'Universe file not found: {universe_path}'}
    
    # Load universe
    with open(universe_path) as f:
        universe_data = json.load(f)
    
    # Extract tickers from universe
    universe_tickers = set()
    for security in universe_data:
        ticker = security.get('ticker')
        if ticker and ticker != '_XBI_BENCHMARK_':
            universe_tickers.add(ticker)
    
    # Get ETF constituents
    xbi = get_xbi_constituents()
    ibb = get_ibb_constituents()
    nbi = get_nbi_constituents()
    
    # Combine all unique constituents
    all_etf_constituents = xbi | ibb | nbi
    
    # Calculate coverage
    in_universe = all_etf_constituents & universe_tickers
    missing = all_etf_constituents - universe_tickers
    
    coverage_pct = (len(in_universe) / len(all_etf_constituents)) * 100 if all_etf_constituents else 0
    
    return {
        'universe_size': len(universe_tickers),
        'total_unique_etf_constituents': len(all_etf_constituents),
        'coverage_pct': coverage_pct,
        'xbi_constituents': len(xbi),
        'ibb_constituents': len(ibb),
        'nbi_constituents': len(nbi),
        'in_universe_count': len(in_universe),
        'missing_count': len(missing),
        'in_universe': sorted(in_universe),
        'missing_from_universe': sorted(missing),
        'in_universe_but_not_etf': sorted(universe_tickers - all_etf_constituents)
    }


def generate_report(universe_path: Path) -> None:
    """Generate detailed coverage report"""
    
    print("="*80)
    print("ETF COVERAGE ANALYSIS: XBI, IBB, NBI")
    print("="*80)
    
    result = check_universe_coverage(universe_path)
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print(f"\n📊 COVERAGE SUMMARY")
    print("-"*80)
    print(f"Universe size: {result['universe_size']} tickers")
    print(f"Unique ETF constituents: {result['total_unique_etf_constituents']} tickers")
    print(f"  • XBI constituents: {result['xbi_constituents']}")
    print(f"  • IBB constituents: {result['ibb_constituents']}")
    print(f"  • NBI constituents: {result['nbi_constituents']}")
    print(f"\nCoverage: {result['in_universe_count']}/{result['total_unique_etf_constituents']} ({result['coverage_pct']:.1f}%)")
    print(f"Missing: {result['missing_count']} tickers")
    
    if result['missing_count'] > 0:
        print(f"\n❌ MISSING FROM UNIVERSE ({result['missing_count']} tickers):")
        print("-"*80)
        for i, ticker in enumerate(result['missing_from_universe'], 1):
            print(f"  {i:3d}. {ticker}")
    
    if result['in_universe_but_not_etf']:
        print(f"\n📝 IN UNIVERSE BUT NOT IN ETFs ({len(result['in_universe_but_not_etf'])} tickers):")
        print("-"*80)
        for i, ticker in enumerate(result['in_universe_but_not_etf'], 1):
            print(f"  {i:3d}. {ticker}")
    
    print(f"\n{'='*80}")
    
    # Recommendation
    if result['coverage_pct'] < 90:
        print(f"\n⚠️  WARNING: Coverage is {result['coverage_pct']:.1f}%")
        print(f"   You're missing {result['missing_count']} constituents from XBI/IBB/NBI")
        print(f"   Consider expanding your universe to include all ETF constituents.")
    elif result['coverage_pct'] < 100:
        print(f"\n✅ Good coverage: {result['coverage_pct']:.1f}%")
        print(f"   Missing {result['missing_count']} constituents (likely recent additions or small caps)")
    else:
        print(f"\n✅ COMPLETE COVERAGE: 100%")
        print(f"   All XBI/IBB/NBI constituents are in your universe")


def download_current_constituents() -> dict:
    """
    Download current ETF constituents from data provider.
    
    NOTE: This is a placeholder. In production, you would:
    1. Use SPDR/iShares APIs or data provider (Bloomberg, FactSet)
    2. Or scrape from ETF provider websites
    3. Or use a financial data API (Alpha Vantage, Polygon, etc.)
    """
    print("\n📥 DOWNLOADING CURRENT ETF CONSTITUENTS")
    print("-"*80)
    print("⚠️  Using representative data (not live)")
    print("   For production, fetch from:")
    print("   • XBI: https://www.ssga.com/us/en/individual/etfs/funds/xbi")
    print("   • IBB: https://www.ishares.com/us/products/239699/")
    print("   • NBI: Nasdaq data provider")
    print()
    
    return {
        'xbi': list(get_xbi_constituents()),
        'ibb': list(get_ibb_constituents()),
        'nbi': list(get_nbi_constituents())
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check if universe covers all XBI, IBB, NBI constituents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check coverage
  python check_etf_coverage.py --universe production_data/universe.json
  
  # Download current constituents (placeholder)
  python check_etf_coverage.py --download
  
Note:
  This script uses representative ETF constituent lists.
  For production, fetch current holdings from ETF providers.
        """
    )
    
    parser.add_argument(
        '--universe',
        type=Path,
        default=Path('production_data/universe.json'),
        help='Path to universe.json'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download current ETF constituents (placeholder)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Save report to JSON file'
    )
    
    args = parser.parse_args()
    
    if args.download:
        constituents = download_current_constituents()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(constituents, f, indent=2, sort_keys=True)
                f.write('\n')
            print(f"✅ Saved to: {args.output}")
        return 0
    
    # Generate report
    generate_report(args.universe)
    
    # Save to file if requested
    if args.output:
        result = check_universe_coverage(args.universe)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
        print(f"\n✅ Detailed report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
