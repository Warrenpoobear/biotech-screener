"""
Build Historical Returns Database for Wake Robin
Fetches and caches returns for entire biotech universe

Usage:
    python build_returns_database.py --start-date 2020-01-01 --universe universe.csv

This creates a returns database in data/returns/ with:
- Monthly returns for all tickers
- Benchmark returns (XBI, IBB)
- Complete provenance tracking
- Point-in-time discipline
"""

import os
import sys
import argparse
import csv
import json
from datetime import datetime
from typing import List, Dict
import time

# Add parent directory to path to import morningstar_returns
sys.path.insert(0, os.path.dirname(__file__))

try:
    import morningstar_returns as mr
except ImportError:
    print("❌ Could not import morningstar_returns module")
    print("   Make sure morningstar_returns.py is in the same directory")
    sys.exit(1)

try:
    import morningstar_data as md
except ImportError:
    print("❌ morningstar-data package not installed")
    print("   Install with: pip install morningstar-data --break-system-packages")
    sys.exit(1)


class ReturnsDatabaseBuilder:
    """Builds and manages historical returns database."""
    
    def __init__(
        self,
        output_dir: str = 'data/returns',
        cache_dir: str = 'data/returns/cache'
    ):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Known benchmarks
        self.benchmarks = {
            'XBI': 'FEUSA04AER',  # SPDR S&P Biotech ETF
            'IBB': None,  # Will search for this
        }
    
    def find_sec_ids(self, tickers: List[str]) -> Dict[str, str]:
        """
        Find Morningstar SecIds for list of tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            {ticker: sec_id} mapping
        """
        print(f"\nSearching for {len(tickers)} tickers...")
        mapping = {}
        failed = []
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tickers)}")
            
            try:
                results = md.direct.investments(keyword=ticker, count=5)
                
                if not results.empty:
                    # Look for exact match
                    exact = results[results['Ticker'].str.upper() == ticker.upper()]
                    
                    if not exact.empty:
                        sec_id = exact.iloc[0]['SecId']
                        mapping[ticker] = sec_id
                    else:
                        failed.append(ticker)
                else:
                    failed.append(ticker)
                    
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  ⚠️  Error searching {ticker}: {e}")
                failed.append(ticker)
        
        print(f"\n✅ Found {len(mapping)}/{len(tickers)} tickers")
        if failed:
            print(f"❌ Failed: {', '.join(failed[:10])}")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more")
        
        return mapping
    
    def fetch_batch_returns(
        self,
        sec_ids: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 20
    ) -> List[Dict]:
        """
        Fetch returns in batches to avoid timeouts.
        
        Args:
            sec_ids: List of security IDs
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            batch_size: Securities per batch
            
        Returns:
            List of returns data dicts
        """
        all_data = []
        num_batches = (len(sec_ids) + batch_size - 1) // batch_size
        
        print(f"\nFetching returns in {num_batches} batches...")
        
        for i in range(0, len(sec_ids), batch_size):
            batch = sec_ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  Batch {batch_num}/{num_batches}: {len(batch)} securities")
            
            try:
                data = mr.fetch_returns(
                    sec_ids=batch,
                    start_date=start_date,
                    end_date=end_date,
                    frequency='monthly',
                    benchmark_sec_id=self.benchmarks['XBI']
                )
                
                all_data.append(data)
                
                # Cache this batch
                filename = f"batch_{batch_num}_{datetime.now().strftime('%Y%m%d')}.json"
                cache_path = mr.save_returns_cache(
                    data,
                    output_dir=self.cache_dir,
                    filename=filename
                )
                
                print(f"    ✅ Cached to: {os.path.basename(cache_path)}")
                
                # Rate limiting
                time.sleep(1)
                
            except mr.MorningstarReturnsError as e:
                print(f"    ❌ Error: {e}")
                continue
        
        return all_data
    
    def build_database(
        self,
        ticker_universe: List[str],
        start_date: str,
        end_date: str = None
    ) -> str:
        """
        Build complete returns database for universe.
        
        Args:
            ticker_universe: List of tickers
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD (default: today)
            
        Returns:
            Path to database file
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("="*70)
        print("WAKE ROBIN RETURNS DATABASE BUILDER")
        print("="*70)
        print(f"Universe: {len(ticker_universe)} tickers")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Output: {self.output_dir}")
        
        # Step 1: Find SecIds
        print("\n" + "="*70)
        print("STEP 1: FIND SECURITY IDS")
        print("="*70)
        
        ticker_to_sec = self.find_sec_ids(ticker_universe)
        
        if not ticker_to_sec:
            print("❌ No securities found. Cannot continue.")
            return None
        
        # Step 2: Fetch benchmark returns
        print("\n" + "="*70)
        print("STEP 2: FETCH BENCHMARK RETURNS")
        print("="*70)
        
        benchmark_data = {}
        for name, sec_id in self.benchmarks.items():
            if sec_id:
                print(f"\nFetching {name}...")
                try:
                    data = mr.fetch_returns(
                        sec_ids=[sec_id],
                        start_date=start_date,
                        end_date=end_date,
                        frequency='monthly'
                    )
                    benchmark_data[name] = data
                    print(f"  ✅ Retrieved {data['provenance']['num_observations']} observations")
                except mr.MorningstarReturnsError as e:
                    print(f"  ❌ Error: {e}")
        
        # Step 3: Fetch universe returns
        print("\n" + "="*70)
        print("STEP 3: FETCH UNIVERSE RETURNS")
        print("="*70)
        
        sec_ids = list(ticker_to_sec.values())
        returns_data = self.fetch_batch_returns(
            sec_ids=sec_ids,
            start_date=start_date,
            end_date=end_date,
            batch_size=20
        )
        
        # Step 4: Build consolidated database
        print("\n" + "="*70)
        print("STEP 4: BUILD CONSOLIDATED DATABASE")
        print("="*70)
        
        database = {
            'metadata': {
                'created': datetime.now().isoformat() + 'Z',
                'start_date': start_date,
                'end_date': end_date,
                'universe_size': len(ticker_universe),
                'securities_found': len(ticker_to_sec),
                'benchmarks': list(benchmark_data.keys())
            },
            'ticker_to_sec_id': ticker_to_sec,
            'benchmarks': benchmark_data,
            'returns': returns_data
        }
        
        # Save database
        db_filename = f"returns_db_{start_date}_{end_date}.json"
        db_path = mr.save_returns_cache(
            database,
            output_dir=self.output_dir,
            filename=db_filename,
            atomic=True
        )
        
        print(f"\n✅ Database saved to: {db_path}")
        
        # Print summary
        total_observations = sum(d['provenance']['num_observations'] for d in returns_data)
        print(f"\nSummary:")
        print(f"  Tickers: {len(ticker_to_sec)}")
        print(f"  Observations: {total_observations:,}")
        print(f"  Benchmarks: {len(benchmark_data)}")
        print(f"  Cache files: {len(returns_data)}")
        
        return db_path


def load_universe_from_csv(filepath: str, ticker_column: str = 'ticker') -> List[str]:
    """
    Load ticker universe from CSV file.
    
    Args:
        filepath: Path to CSV file
        ticker_column: Name of ticker column
        
    Returns:
        List of ticker symbols
    """
    tickers = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get(ticker_column)
            if ticker:
                tickers.append(ticker.strip().upper())
    
    return list(set(tickers))  # Remove duplicates


def main():
    parser = argparse.ArgumentParser(
        description='Build historical returns database for Wake Robin'
    )
    parser.add_argument(
        '--universe',
        required=True,
        help='Path to universe CSV file'
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--ticker-column',
        default='ticker',
        help='Name of ticker column in CSV'
    )
    parser.add_argument(
        '--output-dir',
        default='data/returns',
        help='Output directory for database'
    )
    
    args = parser.parse_args()
    
    # Check Morningstar availability
    available, msg = mr.check_availability()
    if not available:
        print(f"❌ {msg}")
        sys.exit(1)
    
    print(f"✅ {msg}")
    
    # Load universe
    print(f"\nLoading universe from: {args.universe}")
    tickers = load_universe_from_csv(args.universe, args.ticker_column)
    print(f"✅ Loaded {len(tickers)} tickers")
    
    # Build database
    builder = ReturnsDatabaseBuilder(output_dir=args.output_dir)
    db_path = builder.build_database(
        ticker_universe=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if db_path:
        print("\n" + "="*70)
        print("DATABASE BUILD COMPLETE")
        print("="*70)
        print(f"\nDatabase: {db_path}")
        print("\nNext steps:")
        print("1. Use this database for backtesting validation")
        print("2. Run Phase 2: Basic validation framework")
        print("3. Test if screening signals predict returns")
    else:
        print("\n❌ Database build failed")
        sys.exit(1)


if __name__ == '__main__':
    # If run without arguments, show example usage
    if len(sys.argv) == 1:
        print("Wake Robin Returns Database Builder")
        print("\nExample usage:")
        print("  python build_returns_database.py \\")
        print("    --universe data/universe.csv \\")
        print("    --start-date 2020-01-01 \\")
        print("    --ticker-column ticker")
        print("\nFor full help:")
        print("  python build_returns_database.py --help")
    else:
        main()
