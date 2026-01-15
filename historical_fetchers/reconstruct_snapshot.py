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

# Support momentum signal integration
USE_MOMENTUM = True  # Set to True to include 3-month momentum in rankings
MOMENTUM_WEIGHT = 0.15  # Default weight (overridden by regime detection)


def get_momentum_weight(negative_pct: float) -> float:
    """
    Adjust momentum weight based on market breadth (regime detection).

    Momentum is most useful when winners are scarce (high NEGATIVE% breadth).
    In broad rallies, even small momentum weights can dilute fundamentals.

    Validated thresholds from 7-quarter backtest:
    - 60%+: Strong trend-following (scarce winners)
    - 35-60%: Contrarian with high weight (mixed tape)
    - 25-35%: Minimal momentum (broad market)
    - <25%: Near-zero momentum (strong rally)

    Args:
        negative_pct: Percentage of stocks with negative momentum (0-100)

    Returns:
        Weight for momentum in composite score (0.03 to 0.25)
    """
    if negative_pct >= 60:
        # Scarce winners - strong trends, momentum works great
        # Example: 2023-Q4 (69% negative) -> +54% spread
        return 0.20
    elif negative_pct >= 35:
        # Mixed tape - CONTRARIAN mode needs HIGH weight to override fundamentals
        # Example: 2024-Q2 (38% negative) -> contrarian signal
        return 0.25
    elif negative_pct >= 25:
        # Broad market - minimal momentum, let fundamentals lead
        return 0.05
    else:
        # Strong rally - near-zero momentum weight
        # Example: 2024-Q1 (17% negative) -> fundamentals dominate
        return 0.03


def get_regime_adjusted_momentum_score(
    momentum_pct: float,
    negative_pct: float,
    clip_at: float = 50.0
) -> int:
    """
    Calculate regime-adaptive momentum score with trend/reversal switching.

    - Scarce winners (60%+ negative): Trend-following (favor recent winners)
    - Mixed tape (25-60% negative): Contrarian (fade momentum, buy beaten-down)
    - Broad rally (<25% negative): Slight contrarian

    Args:
        momentum_pct: 3-month return percentage
        negative_pct: Percentage of universe with negative momentum
        clip_at: Maximum absolute momentum to consider (prevents chasing)

    Returns:
        Momentum score (10-40, lower is better)
    """
    # Clip extreme momentum to prevent chasing
    clipped_momentum = max(-clip_at, min(clip_at, momentum_pct))

    # Determine regime and apply appropriate strategy
    if negative_pct >= 60:
        # TREND-FOLLOWING MODE (scarce winners)
        # Strong positive momentum = good (low score)
        effective_momentum = clipped_momentum

    elif negative_pct >= 25:
        # CONTRARIAN MODE (mixed tape)
        # Fade momentum - recent winners may consolidate, losers may bounce
        # Invert the signal: negative momentum becomes positive signal
        effective_momentum = -clipped_momentum * 0.7

    else:
        # WEAK CONTRARIAN MODE (broad rally)
        # Slight fade, mostly neutral
        effective_momentum = -clipped_momentum * 0.4

    # Convert to score (lower is better)
    if effective_momentum >= 20:
        return 10  # Best
    elif effective_momentum >= 5:
        return 20
    elif effective_momentum >= -5:
        return 30  # Neutral
    else:
        return 40  # Worst

# Support both gated and non-gated clinical scoring
USE_GATED_CLINICAL = True  # Set to True to use confidence-weighted scoring

if USE_GATED_CLINICAL:
    try:
        from historical_fetchers.clinicaltrials_gov_gated import (
            get_historical_clinical, fetch_batch as fetch_clinical
        )
        print("Using GATED clinical scoring (confidence-weighted)")
    except ImportError:
        from historical_fetchers.clinicaltrials_gov import (
            get_historical_clinical, fetch_batch as fetch_clinical
        )
        print("Warning: Gated module not found, using standard clinical scoring")
else:
    from historical_fetchers.clinicaltrials_gov import (
        get_historical_clinical, fetch_batch as fetch_clinical
    )

# Import momentum signal module if enabled
if USE_MOMENTUM:
    try:
        from historical_fetchers.momentum_signal import get_momentum_batch
        print("Momentum signal ENABLED (15% weight in composite)")
    except ImportError:
        USE_MOMENTUM = False
        print("Warning: Momentum module not found, disabling momentum signal")


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
    Optionally includes 3-month momentum signal if USE_MOMENTUM is enabled.
    """
    snapshot_dir = Path(f"data/snapshots/{snapshot_date}")

    # Load data
    with open(snapshot_dir / "financials.json", 'r') as f:
        financials = json.load(f).get('tickers', {})

    with open(snapshot_dir / "clinical.json", 'r') as f:
        clinical = json.load(f).get('tickers', {})

    with open(snapshot_dir / "universe.json", 'r') as f:
        tickers = json.load(f).get('tickers', [])

    # Fetch momentum data if enabled
    momentum_data = {}
    negative_pct = 50  # Default for regime detection
    adaptive_momentum_weight = MOMENTUM_WEIGHT  # Default

    if USE_MOMENTUM:
        print(f"\nStep 5: Calculating 3-month momentum (as of {snapshot_date})...")
        try:
            momentum_data = get_momentum_batch(tickers, snapshot_date, delay=0.1)

            # Calculate regime (negative percentage) for adaptive weighting
            if momentum_data:
                negative_count = sum(1 for t, m in momentum_data.items()
                                     if m.get('momentum_bucket') == 'NEGATIVE')
                total_with_momentum = len(momentum_data)
                negative_pct = (negative_count / total_with_momentum * 100) if total_with_momentum > 0 else 50
                adaptive_momentum_weight = get_momentum_weight(negative_pct)

                # Log regime detection
                if negative_pct >= 60:
                    regime = 'TREND'
                    mode = 'trend-following (favor winners)'
                elif negative_pct >= 25:
                    regime = 'MIXED'
                    mode = 'CONTRARIAN (fade momentum, buy beaten-down)'
                else:
                    regime = 'BROAD_RALLY'
                    mode = 'weak contrarian (fundamentals lead)'

                print(f"\n  Regime Detection:")
                print(f"    Negative %: {negative_pct:.1f}%")
                print(f"    Regime: {regime}")
                print(f"    Mode: {mode}")
                print(f"    Momentum Weight: {adaptive_momentum_weight:.0%}")

        except Exception as e:
            print(f"  Warning: Momentum fetch failed: {e}")
            momentum_data = {}

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

        # Momentum score (if enabled) - with regime-adaptive scoring
        momentum_score = 30  # Neutral default
        momentum_pct = 0
        momentum_bucket = 'NEUTRAL'
        if USE_MOMENTUM and ticker in momentum_data:
            mom = momentum_data[ticker]
            momentum_pct = mom.get('momentum_pct', 0)
            momentum_bucket = mom.get('momentum_bucket', 'NEUTRAL')

            # Use regime-adjusted momentum score (trend vs contrarian)
            momentum_score = get_regime_adjusted_momentum_score(
                momentum_pct=momentum_pct,
                negative_pct=negative_pct,
                clip_at=50.0
            )

        # Composite score calculation with regime-adaptive momentum weight
        if USE_MOMENTUM and momentum_data:
            # Use pre-calculated regime-adaptive weight
            base_weight = (1 - adaptive_momentum_weight) / 2
            composite_score = (
                financial_score * base_weight +
                clinical_score * base_weight +
                momentum_score * adaptive_momentum_weight
            )
        else:
            # Without momentum: financial(50%) + clinical(50%)
            composite_score = (financial_score + clinical_score) / 2

        scored.append({
            'ticker': ticker,
            'composite_score': composite_score,
            'financial_score': financial_score,
            'clinical_score': clinical_score,
            'momentum_score': momentum_score,
            'momentum_pct': momentum_pct,
            'momentum_bucket': momentum_bucket,
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
        if USE_MOMENTUM and momentum_data:
            writer.writerow(['Rank', 'Ticker', 'Composite_Score', 'Clinical_Score',
                             'Financial_Score', 'Momentum_Score', 'Momentum_Pct',
                             'Momentum_Bucket', 'Stage_Bucket', 'Cash', 'Runway_Months'])
            for rec in scored:
                writer.writerow([
                    rec['composite_rank'],
                    rec['ticker'],
                    f"{rec['composite_score']:.2f}",
                    rec['clinical_score'],
                    rec['financial_score'],
                    rec['momentum_score'],
                    f"{rec['momentum_pct']:.1f}",
                    rec['momentum_bucket'],
                    rec['stage_bucket'],
                    rec.get('cash', ''),
                    rec.get('runway_months', '')
                ])
        else:
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
