#!/usr/bin/env python3
"""
Point-in-Time (PIT) Snapshot Collector

Captures fundamental data at a specific point in time for proper backtesting.
Eliminates look-ahead bias by preserving historical state.

Usage:
    # Capture current state
    python snapshot_collector.py --date 2024-01-15

    # Capture from existing results file
    python snapshot_collector.py --date 2024-01-15 --from-results results_FINAL_COMPLETE.json

    # List available snapshots
    python snapshot_collector.py --list
"""

import argparse
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
from decimal import Decimal


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def get_snapshot_dir(as_of_date: str) -> Path:
    """Get the directory path for a snapshot date."""
    return Path("data/snapshots") / as_of_date


def capture_from_results(results_file: str, as_of_date: str) -> Dict[str, Any]:
    """
    Extract snapshot data from an existing screening results file.

    This is useful for creating snapshots from historical screening runs.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    snapshot = {
        'metadata': {
            'as_of_date': as_of_date,
            'captured_at': datetime.now().isoformat(),
            'source': results_file,
            'source_type': 'results_file'
        },
        'universe': extract_universe(results),
        'financials': extract_financials(results),
        'clinical': extract_clinical(results),
        'catalysts': extract_catalysts(results),
        'composite_scores': extract_composite_scores(results)
    }

    return snapshot


def extract_universe(results: Dict) -> Dict[str, Any]:
    """Extract universe data from results."""
    module_1 = results.get('module_1_universe', {})
    securities = module_1.get('active_securities', [])

    return {
        'count': len(securities),
        'tickers': [s['ticker'] for s in securities if s.get('ticker')],
        'securities': {
            s['ticker']: {
                'company_name': s.get('company_name'),
                'market_cap_mm': s.get('market_cap_mm'),
                'status': s.get('status', 'active')
            }
            for s in securities if s.get('ticker')
        }
    }


def extract_financials(results: Dict) -> Dict[str, Any]:
    """Extract financial data from results."""
    module_2 = results.get('module_2_financial', {})
    # Try different possible keys
    records = module_2.get('financial_records', []) or module_2.get('scores', [])

    financials = {}
    for rec in records:
        ticker = rec.get('ticker')
        if not ticker:
            continue
        financials[ticker] = {
            'cash': rec.get('cash'),
            'debt': rec.get('debt'),
            'quarterly_burn': rec.get('quarterly_burn'),
            'runway_months': rec.get('runway_months'),
            'financial_score': rec.get('financial_score') or rec.get('score'),
            'financial_normalized': rec.get('financial_normalized') or rec.get('normalized'),
            'severity': rec.get('severity')
        }

    # If no records found, extract from composite scores
    if not financials:
        composite = results.get('module_5_composite', {}).get('ranked_securities', [])
        for rec in composite:
            ticker = rec.get('ticker')
            if ticker:
                financials[ticker] = {
                    'financial_normalized': rec.get('financial_normalized'),
                    'financial_raw': rec.get('financial_raw'),
                    'severity': rec.get('severity')
                }

    return {
        'count': len(financials),
        'tickers': financials
    }


def extract_clinical(results: Dict) -> Dict[str, Any]:
    """Extract clinical development data from results."""
    module_4 = results.get('module_4_clinical', {})
    records = module_4.get('clinical_records', []) or module_4.get('scores', [])

    clinical = {}
    for rec in records:
        ticker = rec.get('ticker')
        if not ticker:
            continue
        clinical[ticker] = {
            'lead_stage': rec.get('lead_stage'),
            'stage_bucket': rec.get('stage_bucket'),
            'pipeline_count': rec.get('pipeline_count'),
            'clinical_score': rec.get('clinical_score') or rec.get('score'),
            'clinical_normalized': rec.get('clinical_normalized') or rec.get('normalized')
        }

    # If no records found, extract from composite scores
    if not clinical:
        composite = results.get('module_5_composite', {}).get('ranked_securities', [])
        for rec in composite:
            ticker = rec.get('ticker')
            if ticker:
                clinical[ticker] = {
                    'stage_bucket': rec.get('stage_bucket'),
                    'clinical_dev_normalized': rec.get('clinical_dev_normalized'),
                    'clinical_dev_raw': rec.get('clinical_dev_raw')
                }

    return {
        'count': len(clinical),
        'tickers': clinical
    }


def extract_catalysts(results: Dict) -> Dict[str, Any]:
    """Extract catalyst data from results."""
    module_3 = results.get('module_3_catalyst', {})
    records = module_3.get('catalyst_records', []) or module_3.get('summaries', [])

    catalysts = {}
    for rec in records:
        ticker = rec.get('ticker')
        if not ticker:
            continue
        catalysts[ticker] = {
            'has_catalyst': rec.get('has_catalyst', False),
            'catalyst_type': rec.get('catalyst_type'),
            'catalyst_date': rec.get('catalyst_date'),
            'days_to_catalyst': rec.get('days_to_catalyst'),
            'catalyst_score': rec.get('catalyst_score') or rec.get('score'),
            'catalyst_normalized': rec.get('catalyst_normalized') or rec.get('normalized')
        }

    # If no records found, extract from composite scores
    if not catalysts:
        composite = results.get('module_5_composite', {}).get('ranked_securities', [])
        for rec in composite:
            ticker = rec.get('ticker')
            if ticker:
                catalysts[ticker] = {
                    'catalyst_normalized': rec.get('catalyst_normalized'),
                    'catalyst_raw': rec.get('catalyst_raw')
                }

    return {
        'count': len(catalysts),
        'tickers': catalysts
    }


def extract_composite_scores(results: Dict) -> Dict[str, Any]:
    """Extract final composite scores and rankings."""
    module_5 = results.get('module_5_composite', {})
    ranked = module_5.get('ranked_securities', [])

    scores = {}
    for rec in ranked:
        ticker = rec.get('ticker')
        if not ticker:
            continue
        scores[ticker] = {
            'composite_score': rec.get('composite_score'),
            'composite_rank': rec.get('composite_rank'),
            'clinical_dev_normalized': rec.get('clinical_dev_normalized'),
            'financial_normalized': rec.get('financial_normalized'),
            'catalyst_normalized': rec.get('catalyst_normalized'),
            'stage_bucket': rec.get('stage_bucket')
        }

    return {
        'count': len(scores),
        'weights_used': module_5.get('weights_used', {}),
        'tickers': scores
    }


def save_snapshot(snapshot: Dict[str, Any], as_of_date: str) -> Path:
    """Save snapshot to disk."""
    snapshot_dir = get_snapshot_dir(as_of_date)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save individual component files
    for component in ['universe', 'financials', 'clinical', 'catalysts', 'composite_scores']:
        if component in snapshot:
            filepath = snapshot_dir / f"{component}.json"
            with open(filepath, 'w') as f:
                json.dump(
                    {'as_of_date': as_of_date, **snapshot[component]},
                    f, indent=2, cls=DecimalEncoder
                )

    # Save metadata
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(snapshot['metadata'], f, indent=2)

    # Save complete snapshot
    complete_path = snapshot_dir / "snapshot_complete.json"
    with open(complete_path, 'w') as f:
        json.dump(snapshot, f, indent=2, cls=DecimalEncoder)

    return snapshot_dir


def load_snapshot(as_of_date: str) -> Optional[Dict[str, Any]]:
    """Load a snapshot from disk."""
    snapshot_dir = get_snapshot_dir(as_of_date)
    complete_path = snapshot_dir / "snapshot_complete.json"

    if not complete_path.exists():
        return None

    with open(complete_path, 'r') as f:
        return json.load(f)


def list_snapshots() -> List[str]:
    """List all available snapshot dates."""
    snapshots_dir = Path("data/snapshots")
    if not snapshots_dir.exists():
        return []

    dates = []
    for d in snapshots_dir.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            dates.append(d.name)

    return sorted(dates)


def generate_rankings_from_snapshot(as_of_date: str, output_csv: str) -> None:
    """
    Generate a rankings CSV from a snapshot.

    Uses the composite_scores from the snapshot, re-sorted with
    the CORRECT sort order (ascending = lower score is better).
    """
    snapshot = load_snapshot(as_of_date)
    if not snapshot:
        raise FileNotFoundError(f"No snapshot found for {as_of_date}")

    scores = snapshot.get('composite_scores', {}).get('tickers', {})

    # Convert to list and sort ASCENDING (lower score = better = rank 1)
    ranked_list = [
        {'ticker': ticker, **data}
        for ticker, data in scores.items()
    ]
    ranked_list.sort(key=lambda x: (float(x.get('composite_score', 999)), x['ticker']))

    # Assign ranks
    for i, rec in enumerate(ranked_list):
        rec['composite_rank'] = i + 1

    # Write CSV
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Clinical_Score',
                         'Financial_Score', 'Catalyst_Score', 'Stage_Bucket'])
        for rec in ranked_list:
            writer.writerow([
                rec['composite_rank'],
                rec['ticker'],
                rec.get('composite_score', ''),
                rec.get('clinical_dev_normalized', ''),
                rec.get('financial_normalized', ''),
                rec.get('catalyst_normalized', ''),
                rec.get('stage_bucket', '')
            ])

    print(f"Generated rankings: {output_csv}")
    print(f"  Tickers: {len(ranked_list)}")
    print(f"  Top 5: {[r['ticker'] for r in ranked_list[:5]]}")


def main():
    parser = argparse.ArgumentParser(
        description="Point-in-Time Snapshot Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture snapshot from existing results file
  python snapshot_collector.py --date 2024-01-15 --from-results results_FINAL_COMPLETE.json

  # List available snapshots
  python snapshot_collector.py --list

  # Generate rankings from a snapshot
  python snapshot_collector.py --date 2024-01-15 --generate-rankings
        """
    )

    parser.add_argument('--date', type=str, help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--from-results', type=str, help='Extract from results JSON file')
    parser.add_argument('--list', action='store_true', help='List available snapshots')
    parser.add_argument('--generate-rankings', action='store_true',
                        help='Generate rankings CSV from snapshot')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    if args.list:
        snapshots = list_snapshots()
        if snapshots:
            print("Available snapshots:")
            for s in snapshots:
                print(f"  {s}")
        else:
            print("No snapshots found.")
        return

    if not args.date:
        parser.error("--date is required (except with --list)")

    if args.generate_rankings:
        output = args.output or f"data/snapshots/{args.date}/rankings.csv"
        generate_rankings_from_snapshot(args.date, output)
        return

    if args.from_results:
        print(f"Capturing snapshot for {args.date} from {args.from_results}...")
        snapshot = capture_from_results(args.from_results, args.date)
        snapshot_dir = save_snapshot(snapshot, args.date)
        print(f"Snapshot saved to: {snapshot_dir}")
        print(f"  Universe: {snapshot['universe']['count']} tickers")
        print(f"  Financials: {snapshot['financials']['count']} records")
        print(f"  Clinical: {snapshot['clinical']['count']} records")
        print(f"  Catalysts: {snapshot['catalysts']['count']} records")
        print(f"  Composite: {snapshot['composite_scores']['count']} scores")
        return

    parser.error("Specify --from-results or --generate-rankings")


if __name__ == "__main__":
    main()
