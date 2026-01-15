"""
build_static_cusip_map.py

Helper script to populate static CUSIP map from various sources.

Sources supported:
1. Yahoo Finance ticker lookup
2. CSV import (ticker,cusip,name)
3. SEC EDGAR company search
4. Manual entry via CLI

Author: Wake Robin Capital Management
Date: 2026-01-09
"""

import json
import csv
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import urllib.request
import urllib.parse

# ==============================================================================
# YAHOO FINANCE CUSIP LOOKUP (WEB SCRAPING)
# ==============================================================================

def lookup_cusip_yahoo(ticker: str) -> Optional[Dict]:
    """
    Look up CUSIP from Yahoo Finance.
    
    WARNING: Web scraping is fragile. Use as fallback only.
    For production, use Bloomberg or paid data service.
    
    Returns:
        {'cusip': str, 'name': str} or None
    """
    # Yahoo Finance doesn't have a clean API for CUSIP lookup
    # This is a placeholder - implement based on current Yahoo Finance structure
    
    print(f"  {ticker}: Yahoo Finance lookup not implemented")
    print(f"    Manual lookup: https://finance.yahoo.com/quote/{ticker}")
    return None


# ==============================================================================
# SEC EDGAR COMPANY SEARCH
# ==============================================================================

def lookup_cusip_sec_edgar(ticker: str) -> Optional[Dict]:
    """
    Look up CUSIP from SEC EDGAR company search.
    
    This requires parsing SEC's company search results.
    For production, use SEC's Entity Management System or CIK lookup.
    
    Returns:
        {'cusip': str, 'name': str, 'cik': str} or None
    """
    # SEC EDGAR API is complex for CUSIP lookup
    # Best approach: Get CIK first, then extract CUSIP from recent 10-K/10-Q
    
    print(f"  {ticker}: SEC EDGAR lookup not implemented")
    print(f"    Manual lookup: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}")
    return None


# ==============================================================================
# CSV IMPORT
# ==============================================================================

def import_cusips_from_csv(
    csv_path: Path,
    output_path: Path,
    merge_with_existing: bool = True
) -> None:
    """
    Import CUSIPs from CSV file.
    
    CSV format:
        ticker,cusip,name
        AAPL,037833100,Apple Inc
        NVAX,670002401,Novavax Inc
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to output JSON (cusip_static_map.json)
        merge_with_existing: If True, merge with existing map
    """
    # Load existing map if merging
    if merge_with_existing and output_path.exists():
        with open(output_path) as f:
            static_map = json.load(f)
        
        # Remove metadata entries
        static_map = {k: v for k, v in static_map.items() if not k.startswith('_')}
    else:
        static_map = {}
    
    # Import from CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cusip = row['cusip'].upper().strip()
            ticker = row['ticker'].upper().strip()
            name = row.get('name', '').strip()
            
            # Validate CUSIP format
            if len(cusip) != 9 or not cusip.isalnum():
                print(f"  Warning: Invalid CUSIP format: {cusip} ({ticker})")
                continue
            
            static_map[cusip] = {
                'cusip': cusip,
                'ticker': ticker,
                'name': name,
                'exchange': row.get('exchange', 'NASDAQ'),
                'security_type': row.get('security_type', 'Common Stock'),
                'mapped_at': datetime.now().isoformat(),
                'source': 'static'
            }
            
            print(f"  Added: {cusip} → {ticker} ({name})")
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(static_map, f, indent=2)
    
    print(f"\nImported {len(static_map)} mappings to {output_path}")


# ==============================================================================
# MANUAL ENTRY CLI
# ==============================================================================

def add_cusip_manual(
    static_map_path: Path,
    ticker: str,
    cusip: str,
    name: str,
    exchange: str = 'NASDAQ',
    security_type: str = 'Common Stock'
) -> None:
    """
    Manually add CUSIP mapping to static map.
    
    Args:
        static_map_path: Path to cusip_static_map.json
        ticker: Stock ticker (e.g., 'NVAX')
        cusip: 9-character CUSIP
        name: Company name
        exchange: Exchange (default: NASDAQ)
        security_type: Security type (default: Common Stock)
    """
    # Validate inputs
    cusip = cusip.upper().strip()
    ticker = ticker.upper().strip()
    
    if len(cusip) != 9 or not cusip.isalnum():
        raise ValueError(f"Invalid CUSIP format: {cusip}")
    
    # Load existing map
    if static_map_path.exists():
        with open(static_map_path) as f:
            static_map = json.load(f)
        
        # Remove metadata entries
        static_map = {k: v for k, v in static_map.items() if not k.startswith('_')}
    else:
        static_map = {}
    
    # Add entry
    static_map[cusip] = {
        'cusip': cusip,
        'ticker': ticker,
        'name': name,
        'exchange': exchange,
        'security_type': security_type,
        'mapped_at': datetime.now().isoformat(),
        'source': 'static'
    }
    
    # Write back
    with open(static_map_path, 'w') as f:
        json.dump(static_map, f, indent=2)
    
    print(f"Added: {cusip} → {ticker} ({name})")
    print(f"Saved to: {static_map_path}")


# ==============================================================================
# BATCH PROCESSING FROM UNIVERSE
# ==============================================================================

def generate_cusip_template_from_universe(
    universe_path: Path,
    output_csv_path: Path
) -> None:
    """
    Generate CSV template from universe.json for manual CUSIP entry.
    
    Creates CSV with tickers that you can fill in CUSIPs manually.
    
    Args:
        universe_path: Path to universe.json
        output_csv_path: Path to output CSV template
    """
    with open(universe_path) as f:
        universe = json.load(f)
    
    # Extract tickers
    tickers = []
    for security in universe:
        ticker = security.get('ticker')
        if ticker and ticker != '_XBI_BENCHMARK_':
            tickers.append({
                'ticker': ticker,
                'name': security.get('name', ''),
                'cusip': '',  # To be filled manually
                'exchange': security.get('exchange', 'NASDAQ'),
                'security_type': 'Common Stock'
            })
    
    # Write CSV
    with open(output_csv_path, 'w', newline='') as f:
        fieldnames = ['ticker', 'cusip', 'name', 'exchange', 'security_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(tickers)
    
    print(f"Generated template with {len(tickers)} tickers")
    print(f"Saved to: {output_csv_path}")
    print("\nNext steps:")
    print("1. Open CSV in Excel/Google Sheets")
    print("2. Fill in CUSIP column for each ticker")
    print("3. Import with: python build_static_cusip_map.py import-csv <csv_file>")


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_static_map(static_map_path: Path) -> None:
    """
    Validate static CUSIP map for errors.
    
    Checks:
    - CUSIP format (9 alphanumeric)
    - Duplicate CUSIPs
    - Duplicate tickers
    - Missing required fields
    """
    with open(static_map_path) as f:
        static_map = json.load(f)
    
    # Remove metadata
    static_map = {k: v for k, v in static_map.items() if not k.startswith('_')}
    
    print(f"Validating {len(static_map)} entries...")
    
    errors = []
    warnings = []
    
    # Track duplicates
    seen_tickers = {}
    
    for cusip, mapping in static_map.items():
        # Check CUSIP format
        if len(cusip) != 9 or not cusip.isalnum():
            errors.append(f"Invalid CUSIP format: {cusip}")
        
        # Check required fields
        if not mapping.get('ticker'):
            errors.append(f"Missing ticker for CUSIP {cusip}")
        
        if not mapping.get('name'):
            warnings.append(f"Missing name for CUSIP {cusip}")
        
        # Check duplicate tickers
        ticker = mapping.get('ticker')
        if ticker:
            if ticker in seen_tickers:
                warnings.append(f"Duplicate ticker {ticker}: {cusip} and {seen_tickers[ticker]}")
            else:
                seen_tickers[ticker] = cusip
    
    # Report
    print(f"\nValidation Results:")
    print(f"  Entries: {len(static_map)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if not errors and not warnings:
        print("\n✅ All checks passed!")


# ==============================================================================
# STATISTICS
# ==============================================================================

def show_statistics(static_map_path: Path, universe_path: Path) -> None:
    """
    Show coverage statistics for static map vs universe.
    """
    # Load static map
    with open(static_map_path) as f:
        static_map = json.load(f)
    
    static_map = {k: v for k, v in static_map.items() if not k.startswith('_')}
    
    # Load universe
    with open(universe_path) as f:
        universe = json.load(f)
    
    universe_tickers = {
        s['ticker'] for s in universe
        if s.get('ticker') != '_XBI_BENCHMARK_'
    }
    
    # Count coverage
    mapped_tickers = {m['ticker'] for m in static_map.values()}
    covered = mapped_tickers & universe_tickers
    missing = universe_tickers - mapped_tickers
    
    print(f"\nStatic Map Coverage Statistics")
    print(f"=" * 50)
    print(f"Universe tickers:     {len(universe_tickers)}")
    print(f"Static map entries:   {len(static_map)}")
    print(f"Coverage:             {len(covered)} / {len(universe_tickers)} ({len(covered)/len(universe_tickers)*100:.1f}%)")
    print(f"Missing:              {len(missing)}")
    
    if missing and len(missing) <= 20:
        print(f"\nMissing tickers:")
        for ticker in sorted(missing):
            print(f"  - {ticker}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build and manage static CUSIP map"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Import CSV
    import_parser = subparsers.add_parser('import-csv', help='Import CUSIPs from CSV')
    import_parser.add_argument('csv_file', type=Path, help='CSV file with ticker,cusip,name')
    import_parser.add_argument('--output', type=Path, required=True, help='Output JSON path')
    import_parser.add_argument('--merge', action='store_true', help='Merge with existing map')
    
    # Add manual entry
    add_parser = subparsers.add_parser('add', help='Manually add CUSIP mapping')
    add_parser.add_argument('--static-map', type=Path, required=True)
    add_parser.add_argument('--ticker', type=str, required=True)
    add_parser.add_argument('--cusip', type=str, required=True)
    add_parser.add_argument('--name', type=str, required=True)
    add_parser.add_argument('--exchange', type=str, default='NASDAQ')
    add_parser.add_argument('--security-type', type=str, default='Common Stock')
    
    # Generate template
    template_parser = subparsers.add_parser('generate-template', help='Generate CSV template from universe')
    template_parser.add_argument('--universe', type=Path, required=True)
    template_parser.add_argument('--output', type=Path, required=True)
    
    # Validate
    validate_parser = subparsers.add_parser('validate', help='Validate static map')
    validate_parser.add_argument('--static-map', type=Path, required=True)
    
    # Statistics
    stats_parser = subparsers.add_parser('stats', help='Show coverage statistics')
    stats_parser.add_argument('--static-map', type=Path, required=True)
    stats_parser.add_argument('--universe', type=Path, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'import-csv':
        import_cusips_from_csv(
            csv_path=args.csv_file,
            output_path=args.output,
            merge_with_existing=args.merge
        )
    
    elif args.command == 'add':
        add_cusip_manual(
            static_map_path=args.static_map,
            ticker=args.ticker,
            cusip=args.cusip,
            name=args.name,
            exchange=args.exchange,
            security_type=args.security_type
        )
    
    elif args.command == 'generate-template':
        generate_cusip_template_from_universe(
            universe_path=args.universe,
            output_csv_path=args.output
        )
    
    elif args.command == 'validate':
        validate_static_map(args.static_map)
    
    elif args.command == 'stats':
        show_statistics(
            static_map_path=args.static_map,
            universe_path=args.universe
        )
    
    else:
        parser.print_help()
