"""
Extract historical screening data for weight optimization.

Scans checkpoint files, extracts component scores, and calculates forward returns
from price data to create training dataset for weight optimization.
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import sys

try:
    import numpy as np
except ImportError:
    print("WARNING: numpy not installed. Some features may not work.")
    np = None


class HistoricalDataExtractor:
    """Extract historical screening data for optimization."""

    def __init__(self, checkpoint_dir='checkpoints', price_file=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.price_file = price_file
        self.prices = None

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def load_checkpoint_files(self):
        """
        Scan checkpoint directory for module_5 results.

        Returns: List of (date, filepath) tuples sorted by date
        """
        checkpoints = []

        # Look for module_5_*.json files
        for filepath in self.checkpoint_dir.glob('module_5_*.json'):
            # Extract date from filename: module_5_2024-01-15.json
            try:
                date_str = filepath.stem.replace('module_5_', '')
                date = datetime.strptime(date_str, '%Y-%m-%d')
                checkpoints.append((date, filepath))
            except ValueError:
                print(f"Warning: Could not parse date from {filepath.name}, skipping")
                continue

        checkpoints.sort(key=lambda x: x[0])

        print(f"Found {len(checkpoints)} checkpoint files")
        if checkpoints:
            print(f"  Date range: {checkpoints[0][0].date()} to {checkpoints[-1][0].date()}")

        return checkpoints

    def extract_component_scores(self, checkpoint_data, date):
        """
        Extract component scores from checkpoint JSON.

        Handles different checkpoint formats (v2, v3, with/without enhancements).

        Returns: Dict[ticker] -> {clinical, financial, catalyst, pos, momentum, valuation}
        """
        scores = {}

        # Try different possible structures
        if 'ranked_securities' in checkpoint_data:
            securities = checkpoint_data['ranked_securities']
        elif 'results' in checkpoint_data:
            securities = checkpoint_data['results']
        else:
            print(f"Warning: Unknown checkpoint structure for {date}, skipping")
            return scores

        for security in securities:
            ticker = security.get('ticker')
            if not ticker:
                continue

            # Extract scores - try different field names
            score_data = {}

            # Method 1: Direct score_components field
            if 'score_components' in security:
                components = security['score_components']
                score_data = {
                    'clinical': float(components.get('clinical', 0)),
                    'financial': float(components.get('financial', 0)),
                    'catalyst': float(components.get('catalyst', 0)),
                    'pos': float(components.get('pos', 0)),
                    'momentum': float(components.get('momentum', 0)),
                    'valuation': float(components.get('valuation', 0))
                }

            # Method 2: Individual score fields
            elif 'clinical_score' in security:
                score_data = {
                    'clinical': float(security.get('clinical_score', 0)),
                    'financial': float(security.get('financial_score', 0)),
                    'catalyst': float(security.get('catalyst_score', 0)),
                    'pos': float(security.get('pos_score', 0)),
                    'momentum': float(security.get('momentum_score', 0)),
                    'valuation': float(security.get('valuation_score', 0))
                }

            # Method 3: From module scores
            elif 'module_2_score' in security:
                # Module 2 = financial, Module 4 = clinical
                score_data = {
                    'clinical': float(security.get('module_4_score', 0)),
                    'financial': float(security.get('module_2_score', 0)),
                    'catalyst': float(security.get('module_3_score', 0)),
                    'pos': float(security.get('pos_score', 0)),
                    'momentum': float(security.get('momentum_score', 0)),
                    'valuation': float(security.get('valuation_score', 0))
                }

            if score_data:
                scores[ticker] = score_data

        return scores

    def load_price_data(self):
        """
        Load historical price data.

        Supports multiple formats:
        - CSV with columns: date, ticker, close
        - JSON with nested structure
        - Yahoo Finance cache files

        Returns: Dict[ticker][date] -> price
        """
        if self.prices is not None:
            return self.prices

        prices = defaultdict(dict)

        if self.price_file:
            # Load from specified file
            price_path = Path(self.price_file)

            if price_path.suffix == '.csv':
                prices = self._load_csv_prices(price_path)
            elif price_path.suffix == '.json':
                prices = self._load_json_prices(price_path)
        else:
            # Try to find price data in common locations
            search_paths = [
                'production_data/price_history.csv',
                'production_data/yahoo_cache.json',
                'data/prices.csv',
                'backtest/price_data.csv'
            ]

            for path in search_paths:
                if Path(path).exists():
                    print(f"Found price data: {path}")
                    if path.endswith('.csv'):
                        prices = self._load_csv_prices(Path(path))
                    else:
                        prices = self._load_json_prices(Path(path))
                    break

        if not prices:
            print("WARNING: No price data found. Forward returns will be unavailable.")
            print("  Provide price file with --price-file option")

        self.prices = prices
        return prices

    def _load_csv_prices(self, filepath):
        """Load prices from CSV file."""
        prices = defaultdict(dict)

        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get('ticker') or row.get('symbol')
                    date_str = row.get('date')
                    close = row.get('close') or row.get('adj_close')

                    if ticker and date_str and close:
                        try:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            prices[ticker][date] = float(close)
                        except (ValueError, TypeError):
                            continue

            print(f"Loaded prices for {len(prices)} tickers")
        except Exception as e:
            print(f"Error loading CSV prices: {e}")

        return prices

    def _load_json_prices(self, filepath):
        """Load prices from JSON file (e.g., Yahoo cache)."""
        prices = defaultdict(dict)

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Handle different JSON structures
            for ticker, ticker_data in data.items():
                if isinstance(ticker_data, dict):
                    # Format 1: {ticker: {date: price}}
                    for date_str, price in ticker_data.items():
                        try:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            prices[ticker][date] = float(price)
                        except (ValueError, TypeError):
                            continue
                elif isinstance(ticker_data, list):
                    # Format 2: {ticker: [{date: ..., close: ...}]}
                    for entry in ticker_data:
                        date_str = entry.get('date')
                        close = entry.get('close')
                        if date_str and close:
                            try:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                                prices[ticker][date] = float(close)
                            except (ValueError, TypeError):
                                continue

            print(f"Loaded prices for {len(prices)} tickers")
        except Exception as e:
            print(f"Error loading JSON prices: {e}")

        return prices

    def calculate_forward_return(self, ticker, start_date, horizon_days=28):
        """
        Calculate forward return for a ticker.

        Parameters:
        - ticker: Stock symbol
        - start_date: Start date (datetime)
        - horizon_days: Forward period (default: 28 days = 4 weeks)

        Returns: Forward return as decimal (e.g., 0.15 = 15%) or None if unavailable
        """
        if not self.prices or ticker not in self.prices:
            return None

        ticker_prices = self.prices[ticker]

        # Find price on or near start_date (allow ±3 days)
        start_price = None
        for offset in range(-3, 4):
            check_date = start_date + timedelta(days=offset)
            if check_date in ticker_prices:
                start_price = ticker_prices[check_date]
                break

        if start_price is None:
            return None

        # Find price on or near end_date (allow ±3 days)
        end_date = start_date + timedelta(days=horizon_days)
        end_price = None
        for offset in range(-3, 4):
            check_date = end_date + timedelta(days=offset)
            if check_date in ticker_prices:
                end_price = ticker_prices[check_date]
                break

        if end_price is None or start_price <= 0:
            return None

        # Calculate return
        return (end_price - start_price) / start_price

    def extract_training_data(self, horizon_days=28, min_observations=100):
        """
        Extract complete training dataset from checkpoints.

        Parameters:
        - horizon_days: Forward return period (default: 28 = 4 weeks)
        - min_observations: Minimum observations required (default: 100)

        Returns: List of dicts with training data
        """
        print("\n" + "="*60)
        print("EXTRACTING HISTORICAL TRAINING DATA")
        print("="*60)

        # Load checkpoints
        checkpoints = self.load_checkpoint_files()

        if not checkpoints:
            print("ERROR: No checkpoint files found")
            return []

        # Load prices
        self.load_price_data()

        # Extract data
        training_data = []
        stats = {
            'checkpoints_processed': 0,
            'tickers_with_scores': 0,
            'forward_returns_calculated': 0,
            'missing_prices': 0
        }

        for date, filepath in checkpoints:
            try:
                with open(filepath) as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            # Extract scores
            scores = self.extract_component_scores(checkpoint_data, date)

            if not scores:
                continue

            stats['checkpoints_processed'] += 1

            # Calculate forward returns
            for ticker, score_data in scores.items():
                stats['tickers_with_scores'] += 1

                # Calculate forward return
                fwd_return = self.calculate_forward_return(ticker, date, horizon_days)

                if fwd_return is None:
                    stats['missing_prices'] += 1
                    continue

                stats['forward_returns_calculated'] += 1

                # Add to training data
                training_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'clinical': score_data['clinical'],
                    'financial': score_data['financial'],
                    'catalyst': score_data['catalyst'],
                    'pos': score_data['pos'],
                    'momentum': score_data['momentum'],
                    'valuation': score_data['valuation'],
                    'fwd_return': fwd_return
                })

        # Print statistics
        print(f"\nExtraction Statistics:")
        print(f"  Checkpoints processed: {stats['checkpoints_processed']}")
        print(f"  Tickers with scores: {stats['tickers_with_scores']}")
        print(f"  Forward returns calculated: {stats['forward_returns_calculated']}")
        print(f"  Missing prices: {stats['missing_prices']}")
        print(f"  Total observations: {len(training_data)}")

        # Check minimum threshold
        if len(training_data) < min_observations:
            print(f"\n⚠️  WARNING: Only {len(training_data)} observations found")
            print(f"   Minimum recommended: {min_observations}")
            print("   Results may not be reliable with limited data")
        else:
            print(f"\n✓ Sufficient data: {len(training_data)} observations")

        return training_data

    def save_training_data(self, training_data, output_file='optimization/optimization_data/training_dataset.csv'):
        """Save training data to CSV."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not training_data:
            print("ERROR: No training data to save")
            return

        fieldnames = ['date', 'ticker', 'clinical', 'financial', 'catalyst',
                      'pos', 'momentum', 'valuation', 'fwd_return']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_data)

        print(f"\n✓ Saved {len(training_data)} observations to {output_file}")

        # Print sample statistics
        if np:
            returns = [d['fwd_return'] for d in training_data]
            print(f"\nForward return statistics:")
            print(f"  Mean: {np.mean(returns):.4f} ({np.mean(returns)*100:.2f}%)")
            print(f"  Std: {np.std(returns):.4f} ({np.std(returns)*100:.2f}%)")
            print(f"  Min: {np.min(returns):.4f} ({np.min(returns)*100:.2f}%)")
            print(f"  Max: {np.max(returns):.4f} ({np.max(returns)*100:.2f}%)")

        # Print date range
        dates = [d['date'] for d in training_data]
        print(f"\nDate range:")
        print(f"  First: {min(dates)}")
        print(f"  Last: {max(dates)}")

        # Print ticker coverage
        unique_tickers = len(set(d['ticker'] for d in training_data))
        print(f"\nTicker coverage:")
        print(f"  Unique tickers: {unique_tickers}")
        print(f"  Avg observations per ticker: {len(training_data) / unique_tickers:.1f}")


def main():
    """Main extraction workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract historical screening data for weight optimization'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints',
        help='Directory containing module_5_*.json checkpoint files'
    )
    parser.add_argument(
        '--price-file',
        help='Price data file (CSV or JSON). If not provided, will search common locations.'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=28,
        help='Forward return period in days (default: 28 = 4 weeks)'
    )
    parser.add_argument(
        '--output',
        default='optimization/optimization_data/training_dataset.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--min-observations',
        type=int,
        default=100,
        help='Minimum observations required (default: 100)'
    )

    args = parser.parse_args()

    # Create extractor
    extractor = HistoricalDataExtractor(
        checkpoint_dir=args.checkpoint_dir,
        price_file=args.price_file
    )

    # Extract data
    training_data = extractor.extract_training_data(
        horizon_days=args.horizon,
        min_observations=args.min_observations
    )

    if not training_data:
        print("\n❌ No training data extracted. Check:")
        print("  1. Checkpoint files exist in", args.checkpoint_dir)
        print("  2. Price data is available")
        print("  3. Dates overlap between checkpoints and prices")
        sys.exit(1)

    # Save
    extractor.save_training_data(training_data, args.output)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print("\nNext step: Run optimization")
    print("  python -m optimization.optimize_weights_scipy")


if __name__ == '__main__':
    main()
