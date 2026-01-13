"""
Wake Robin Backtesting Validation Framework
Phase 2: Basic signal validation

Tests if screening signals predict forward returns.

Usage:
    python validate_signals.py \\
        --database data/returns/returns_db_2020-01-01_2024-12-31.json \\
        --screen-date 2023-01-01 \\
        --forward-months 6
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

try:
    import morningstar_returns as mr
except ImportError:
    print("❌ Could not import morningstar_returns")
    sys.exit(1)


class BacktestValidator:
    """Validates screening signals against forward returns."""
    
    def __init__(self, returns_database_path: str):
        """
        Initialize validator with returns database.
        
        Args:
            returns_database_path: Path to returns database JSON
        """
        self.db_path = returns_database_path
        self.database = None
        self.returns_by_ticker = {}
        self.load_database()
    
    def load_database(self):
        """Load and index returns database."""
        print(f"Loading returns database: {self.db_path}")
        
        self.database = mr.load_returns_cache(self.db_path)
        
        print(f"✅ Database loaded")
        print(f"   Universe size: {self.database['metadata']['securities_found']}")
        print(f"   Date range: {self.database['metadata']['start_date']} to {self.database['metadata']['end_date']}")
        
        # Build index: ticker -> [(date, return)]
        self._build_returns_index()
    
    def _build_returns_index(self):
        """Build fast lookup index for returns by ticker and date."""
        print("\nBuilding returns index...")
        
        ticker_to_sec = self.database['ticker_to_sec_id']
        sec_to_ticker = {v: k for k, v in ticker_to_sec.items()}
        
        # Index all returns data
        for batch_data in self.database['returns']:
            for record in batch_data['absolute']:
                sec_id = record['sec_id']
                ticker = sec_to_ticker.get(sec_id, sec_id)
                
                if ticker not in self.returns_by_ticker:
                    self.returns_by_ticker[ticker] = {}
                
                date = record['date']
                return_pct = Decimal(record['return_pct'])
                
                self.returns_by_ticker[ticker][date] = return_pct
        
        print(f"✅ Indexed {len(self.returns_by_ticker)} tickers")
    
    def get_forward_return(
        self,
        ticker: str,
        start_date: str,
        months: int
    ) -> Tuple[Decimal, int]:
        """
        Calculate forward return from start_date for N months.
        
        Args:
            ticker: Ticker symbol
            start_date: Starting date (YYYY-MM-DD)
            months: Number of months forward
            
        Returns:
            (cumulative_return, num_periods) where:
            - cumulative_return is compounded return
            - num_periods is actual months of data found
        """
        if ticker not in self.returns_by_ticker:
            return Decimal('0'), 0
        
        ticker_returns = self.returns_by_ticker[ticker]
        
        # Get all dates after start_date
        available_dates = sorted([
            d for d in ticker_returns.keys()
            if d > start_date
        ])
        
        if not available_dates:
            return Decimal('0'), 0
        
        # Take first N months
        forward_dates = available_dates[:months]
        
        # Compound returns: (1+r1) * (1+r2) * ... - 1
        cumulative = Decimal('1')
        for date in forward_dates:
            monthly_return = ticker_returns[date]
            cumulative *= (Decimal('1') + monthly_return / Decimal('100'))
        
        cumulative_return = (cumulative - Decimal('1')) * Decimal('100')
        
        return cumulative_return, len(forward_dates)
    
    def validate_ranked_list(
        self,
        ranked_tickers: List[str],
        screen_date: str,
        forward_months: int = 6
    ) -> Dict:
        """
        Validate a ranked list of tickers against forward returns.
        
        Args:
            ranked_tickers: List of tickers (highest rank first)
            screen_date: Date of screening (YYYY-MM-DD)
            forward_months: Months to look forward
            
        Returns:
            Validation results dict with hit rate, alpha, etc.
        """
        print(f"\nValidating ranked list of {len(ranked_tickers)} tickers")
        print(f"Screen date: {screen_date}")
        print(f"Forward period: {forward_months} months")
        
        results = []
        
        for rank, ticker in enumerate(ranked_tickers, 1):
            forward_return, periods = self.get_forward_return(
                ticker=ticker,
                start_date=screen_date,
                months=forward_months
            )
            
            results.append({
                'rank': rank,
                'ticker': ticker,
                'forward_return_pct': float(forward_return),
                'periods': periods,
                'profitable': forward_return > 0
            })
        
        # Calculate metrics
        valid_results = [r for r in results if r['periods'] == forward_months]
        
        if not valid_results:
            print("❌ No valid forward returns found")
            return None
        
        returns = [r['forward_return_pct'] for r in valid_results]
        profitable = [r for r in valid_results if r['profitable']]
        
        hit_rate = len(profitable) / len(valid_results) * 100
        avg_return = sum(returns) / len(returns)
        median_return = sorted(returns)[len(returns) // 2]
        
        # Top quintile vs bottom quintile
        n_quintile = max(1, len(valid_results) // 5)
        top_quintile = valid_results[:n_quintile]
        bottom_quintile = valid_results[-n_quintile:]
        
        top_avg = sum(r['forward_return_pct'] for r in top_quintile) / len(top_quintile)
        bottom_avg = sum(r['forward_return_pct'] for r in bottom_quintile) / len(bottom_quintile)
        spread = top_avg - bottom_avg
        
        validation = {
            'screen_date': screen_date,
            'forward_months': forward_months,
            'universe_size': len(ranked_tickers),
            'valid_tickers': len(valid_results),
            'hit_rate_pct': round(hit_rate, 1),
            'avg_return_pct': round(avg_return, 2),
            'median_return_pct': round(median_return, 2),
            'top_quintile_return_pct': round(top_avg, 2),
            'bottom_quintile_return_pct': round(bottom_avg, 2),
            'quintile_spread_pct': round(spread, 2),
            'results': valid_results
        }
        
        return validation
    
    def print_validation_report(self, validation: Dict):
        """Print formatted validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        print(f"\nScreen Date: {validation['screen_date']}")
        print(f"Forward Period: {validation['forward_months']} months")
        print(f"Universe: {validation['universe_size']} tickers")
        print(f"Valid Returns: {validation['valid_tickers']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Hit Rate:           {validation['hit_rate_pct']}%")
        print(f"  Average Return:     {validation['avg_return_pct']:+.2f}%")
        print(f"  Median Return:      {validation['median_return_pct']:+.2f}%")
        
        print(f"\nQuintile Analysis:")
        print(f"  Top Quintile:       {validation['top_quintile_return_pct']:+.2f}%")
        print(f"  Bottom Quintile:    {validation['bottom_quintile_return_pct']:+.2f}%")
        print(f"  Spread:             {validation['quintile_spread_pct']:+.2f}%")
        
        # Interpretation
        print(f"\nInterpretation:")
        if validation['hit_rate_pct'] >= 65:
            print("  ✅ STRONG: Hit rate >65% indicates strong signal")
        elif validation['hit_rate_pct'] >= 55:
            print("  ✓  GOOD: Hit rate >55% indicates useful signal")
        else:
            print("  ⚠️  WEAK: Hit rate <55% indicates weak signal")
        
        if validation['quintile_spread_pct'] >= 10:
            print("  ✅ STRONG: >10% spread indicates strong ranking")
        elif validation['quintile_spread_pct'] >= 5:
            print("  ✓  GOOD: >5% spread indicates useful ranking")
        else:
            print("  ⚠️  WEAK: <5% spread indicates weak ranking")
        
        # Top performers
        print(f"\nTop 10 Performers:")
        for result in validation['results'][:10]:
            print(f"  {result['rank']:2d}. {result['ticker']:6s}  {result['forward_return_pct']:+7.2f}%")


def load_ranked_list(filepath: str) -> List[str]:
    """
    Load ranked ticker list from file.
    
    Supports:
    - CSV with 'ticker' column
    - JSON with list of tickers
    - Plain text (one ticker per line)
    """
    _, ext = os.path.splitext(filepath)
    
    if ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif 'tickers' in data:
                return data['tickers']
    
    elif ext == '.csv':
        import csv
        tickers = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'ticker' in row:
                    tickers.append(row['ticker'].strip().upper())
        return tickers
    
    else:
        # Plain text
        with open(filepath, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='Validate screening signals against forward returns'
    )
    parser.add_argument(
        '--database',
        required=True,
        help='Path to returns database'
    )
    parser.add_argument(
        '--ranked-list',
        required=True,
        help='Path to ranked ticker list (CSV, JSON, or text)'
    )
    parser.add_argument(
        '--screen-date',
        required=True,
        help='Screen date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--forward-months',
        type=int,
        default=6,
        help='Forward return period in months (default: 6)'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load validator
    validator = BacktestValidator(args.database)
    
    # Load ranked list
    print(f"\nLoading ranked list: {args.ranked_list}")
    ranked_tickers = load_ranked_list(args.ranked_list)
    print(f"✅ Loaded {len(ranked_tickers)} tickers")
    
    # Validate
    validation = validator.validate_ranked_list(
        ranked_tickers=ranked_tickers,
        screen_date=args.screen_date,
        forward_months=args.forward_months
    )
    
    if validation:
        validator.print_validation_report(validation)
        
        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(validation, f, indent=2)
            print(f"\n✅ Results saved to: {args.output}")
    else:
        print("\n❌ Validation failed")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Wake Robin Signal Validation")
        print("\nExample usage:")
        print("  python validate_signals.py \\")
        print("    --database data/returns/returns_db_2020-01-01_2024-12-31.json \\")
        print("    --ranked-list screen_output_2023-01-01.csv \\")
        print("    --screen-date 2023-01-01 \\")
        print("    --forward-months 6")
        print("\nFor full help:")
        print("  python validate_signals.py --help")
    else:
        main()
