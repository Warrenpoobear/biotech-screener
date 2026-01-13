#!/usr/bin/env python3
"""
Historical Snapshot Reconstructor

Combines SEC EDGAR financials and ClinicalTrials.gov data
to reconstruct point-in-time snapshots for backtesting.

Usage:
    # Reconstruct a single date
    python reconstruct_snapshot.py --date 2023-01-15 --tickers-file universe.txt

    # Reconstruct multiple dates
    python reconstruct_snapshot.py --dates 2022-01-15,2023-01-15,2024-01-15
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from historical_fetchers.sec_edgar import get_historical_financials, fetch_batch as fetch_financials
from historical_fetchers.clinicaltrials_gov import get_historical_clinical, fetch_batch as fetch_clinical


def load_universe(tickers_file: str = None) -> List[str]:
    """Load universe of tickers to reconstruct."""
    if tickers_file:
        with open(tickers_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Handle both plain text files and CSVs
            lines = f.readlines()
            tickers = []
            ticker_col = None  # Will auto-detect

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # If CSV, extract ticker
                if ',' in line:
                    parts = [p.strip().strip('"') for p in line.split(',')]

                    # First line: detect which column has tickers
                    if i == 0 and ticker_col is None:
                        # Check header for 'ticker' column
                        lower_parts = [p.lower() for p in parts]
                        if 'ticker' in lower_parts:
                            ticker_col = lower_parts.index('ticker')
                        elif 'symbol' in lower_parts:
                            ticker_col = lower_parts.index('symbol')
                        else:
                            # Default to column 1 (Rank,Ticker,...) or 0 if only 1 col
                            ticker_col = 1 if len(parts) > 1 else 0
                        continue  # Skip header row

                    if ticker_col is None:
                        ticker_col = 0  # Default

                    if len(parts) > ticker_col:
                        ticker = parts[ticker_col]
                        # Validate ticker: 1-6 uppercase letters
                        if (ticker and ticker.upper() != 'TICKER' and
                            len(ticker) <= 6 and ticker.isalpha() and
                            ticker.isupper()):
                            tickers.append(ticker)
                else:
                    # Plain text file - one ticker per line
                    ticker = line.strip()
                    if (ticker and not ticker.startswith('#') and
                        len(ticker) <= 6 and ticker.isalpha() and
                        ticker.isupper()):
                        tickers.append(ticker)
            return tickers

    # Try to load from existing snapshot
    snapshot_dir = Path("data/snapshots")
    latest = sorted(snapshot_dir.iterdir())[-1] if snapshot_dir.exists() else None

    if latest:
        universe_file = latest / "universe.json"
        if universe_file.exists():
            with open(universe_file, 'r') as f:
                data = json.load(f)
                return data.get('tickers', [])

    # Fall back to a small test set
    return ['VRTX', 'REGN', 'ALNY', 'CRSP', 'MRNA']


def reconstruct_snapshot(as_of_date: str, tickers: List[str],
                        output_dir: str = None) -> Dict:
    """
    Reconstruct a point-in-time snapshot by fetching historical data.

    Args:
        as_of_date: Target date (YYYY-MM-DD)
        tickers: List of tickers to include
        output_dir: Directory to save snapshot (default: data/snapshots/{date})

    Returns:
        Complete snapshot dictionary
    """
    print(f"\n{'='*60}")
    print(f"Reconstructing snapshot for {as_of_date}")
    print(f"Tickers: {len(tickers)}")
    print(f"{'='*60}\n")

    # Create output directory
    if output_dir is None:
        output_dir = Path(f"data/snapshots/{as_of_date}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch financials from SEC EDGAR
    print("Step 1: Fetching financials from SEC EDGAR...")
    financials = fetch_financials(tickers, as_of_date, delay=0.15)
    print(f"  Retrieved: {len([f for f in financials.values() if f.get('cash')])} with cash data\n")

    # Step 2: Fetch clinical data from ClinicalTrials.gov
    print("Step 2: Fetching clinical data from ClinicalTrials.gov...")
    clinical = fetch_clinical(tickers, as_of_date, delay=0.5)  # Fetch all tickers
    print(f"  Retrieved: {len([c for c in clinical.values() if c.get('lead_stage')])} with stage data\n")

    # Step 3: Combine into snapshot
    print("Step 3: Building snapshot...")

    snapshot = {
        'metadata': {
            'as_of_date': as_of_date,
            'reconstructed_at': datetime.now().isoformat(),
            'ticker_count': len(tickers),
            'source_type': 'historical_reconstruction',
            'data_sources': ['SEC_EDGAR', 'ClinicalTrials.gov']
        },
        'universe': {
            'count': len(tickers),
            'tickers': tickers,
            'securities': {t: {'ticker': t, 'status': 'active'} for t in tickers}
        },
        'financials': {
            'count': len(financials),
            'tickers': financials
        },
        'clinical': {
            'count': len(clinical),
            'tickers': clinical
        },
        'catalysts': {
            'count': 0,
            'tickers': {},
            'note': 'Historical catalysts not yet implemented'
        },
        'composite_scores': {
            'count': 0,
            'tickers': {},
            'note': 'Scores must be computed from raw data'
        }
    }

    # Step 4: Save snapshot files
    print("Step 4: Saving snapshot...")

    # Save individual component files
    for component in ['universe', 'financials', 'clinical', 'catalysts']:
        filepath = output_dir / f"{component}.json"
        with open(filepath, 'w') as f:
            json.dump({'as_of_date': as_of_date, **snapshot[component]}, f, indent=2)

    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(snapshot['metadata'], f, indent=2)

    # Save complete snapshot
    with open(output_dir / "snapshot_complete.json", 'w') as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSnapshot saved to: {output_dir}")
    print(f"  Universe: {snapshot['universe']['count']} tickers")
    print(f"  Financials: {len([f for f in financials.values() if not f.get('error')])} valid")
    print(f"  Clinical: {len([c for c in clinical.values() if not c.get('error')])} valid")

    return snapshot


def generate_rankings_from_historical(snapshot_date: str) -> None:
    """
    Generate rankings from a reconstructed historical snapshot.

    Uses a simplified scoring model since we have limited data.
    """
    snapshot_dir = Path(f"data/snapshots/{snapshot_date}")

    # Load data
    with open(snapshot_dir / "financials.json", 'r') as f:
        financials = json.load(f).get('tickers', {})

    with open(snapshot_dir / "clinical.json", 'r') as f:
        clinical = json.load(f).get('tickers', {})

    with open(snapshot_dir / "universe.json", 'r') as f:
        tickers = json.load(f).get('tickers', [])

    # Score each ticker
    scored = []
    for ticker in tickers:
        fin = financials.get(ticker, {})
        clin = clinical.get(ticker, {})

        # Financial score (lower is better - penalty style)
        cash = fin.get('cash', 0) or 0
        debt = fin.get('debt', 0) or 0
        runway = fin.get('runway_months', 0) or 0

        # First try to use existing financial_normalized from snapshot collector
        existing_fin = fin.get('financial_normalized')
        if existing_fin is not None:
            try:
                financial_score = float(existing_fin)
            except (ValueError, TypeError):
                financial_score = 30  # Neutral fallback
        # Otherwise derive from cash (historical fetcher data)
        elif cash > 1e9:  # > $1B
            financial_score = 10
        elif cash > 500e6:  # > $500M
            financial_score = 20
        elif cash > 100e6:  # > $100M
            financial_score = 30
        elif cash > 0:
            financial_score = 40
        else:
            financial_score = 30  # Neutral - don't penalize missing data

        # Clinical score based on stage
        # NOTE: Unknown stage gets NEUTRAL score (30) to avoid penalizing
        # companies without clinical data coverage
        stage_bucket = clin.get('stage_bucket')

        # First try to use existing clinical_score from snapshot collector
        existing_score = clin.get('clinical_score')
        if existing_score is not None:
            try:
                clinical_score = float(existing_score)
            except (ValueError, TypeError):
                clinical_score = 30  # Neutral fallback
        # Otherwise derive from stage_bucket (historical fetcher data)
        elif stage_bucket == 'commercial':
            clinical_score = 10
        elif stage_bucket == 'late':
            clinical_score = 20
        elif stage_bucket == 'mid':
            clinical_score = 30
        elif stage_bucket == 'early':
            clinical_score = 40
        else:
            clinical_score = 30  # Neutral - don't penalize missing data

        # Composite (simple average for now)
        composite_score = (financial_score + clinical_score) / 2

        scored.append({
            'ticker': ticker,
            'composite_score': composite_score,
            'financial_score': financial_score,
            'clinical_score': clinical_score,
            'stage_bucket': stage_bucket,
            'cash': cash,
            'runway_months': runway
        })

    # Sort ascending (lower score = better)
    scored.sort(key=lambda x: (x['composite_score'], x['ticker']))

    # Assign ranks
    for i, rec in enumerate(scored):
        rec['composite_rank'] = i + 1

    # Write CSV
    import csv
    output_csv = snapshot_dir / "rankings.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Clinical_Score',
                         'Financial_Score', 'Stage_Bucket', 'Cash', 'Runway_Months'])
        for rec in scored:
            writer.writerow([
                rec['composite_rank'],
                rec['ticker'],
                rec['composite_score'],
                rec['clinical_score'],
                rec['financial_score'],
                rec['stage_bucket'],
                rec.get('cash', ''),
                rec.get('runway_months', '')
            ])

    print(f"\nGenerated rankings: {output_csv}")
    print(f"  Top 5: {[r['ticker'] for r in scored[:5]]}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct historical snapshots for backtesting"
    )
    parser.add_argument('--date', type=str, help='Single date to reconstruct (YYYY-MM-DD)')
    parser.add_argument('--dates', type=str,
                        help='Comma-separated dates to reconstruct')
    parser.add_argument('--tickers-file', type=str,
                        help='File with tickers (one per line)')
    parser.add_argument('--tickers', type=str,
                        help='Comma-separated list of tickers')
    parser.add_argument('--generate-rankings', action='store_true',
                        help='Also generate rankings from snapshot')

    args = parser.parse_args()

    # Collect dates
    dates = []
    if args.date:
        dates = [args.date]
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(',')]
    else:
        parser.error("Specify --date or --dates")

    # Load tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.tickers_file:
        tickers = load_universe(args.tickers_file)
    else:
        tickers = load_universe()

    print(f"Will reconstruct {len(dates)} snapshots for {len(tickers)} tickers")

    # Reconstruct each date
    for target_date in dates:
        snapshot = reconstruct_snapshot(target_date, tickers)

        if args.generate_rankings:
            generate_rankings_from_historical(target_date)


if __name__ == "__main__":
    main()
