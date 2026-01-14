#!/usr/bin/env python3
"""
Generate institutional dossier for top biotech candidates.

Usage:
    python scripts/generate_dossier.py --ticker KMDA --date 2026-01-14
    python scripts/generate_dossier.py --top-n 5 --date 2026-01-14
    python scripts/generate_dossier.py --all-ranked --date 2026-01-14 --results results.json

Examples:
    # Generate dossier for current #1 pick
    python scripts/generate_dossier.py --ticker KMDA --date 2026-01-14

    # Generate dossiers for top 5 holdings
    python scripts/generate_dossier.py --top-n 5 --date 2026-01-14 --results results_fixed.json

    # Generate dossiers for all ranked securities (batch)
    python scripts/generate_dossier.py --all-ranked --date 2026-01-14

    # Custom output location
    python scripts/generate_dossier.py --ticker KMDA --date 2026-01-14 \\
      --output-dir ./ic_materials/2026-01/ --format md
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dossier.generator import DossierGenerator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate biotech investment dossiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker",
        help="Single ticker to analyze (e.g., KMDA)",
    )
    group.add_argument(
        "--top-n",
        type=int,
        help="Generate dossiers for top N from rankings",
    )
    group.add_argument(
        "--all-ranked",
        action="store_true",
        help="Generate dossiers for all ranked securities",
    )

    # Required configuration
    parser.add_argument(
        "--date",
        required=True,
        help="Snapshot date (YYYY-MM-DD)",
    )

    # Optional configuration
    parser.add_argument(
        "--results",
        default="results_fixed.json",
        help="Path to screening results file (default: results_fixed.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dossiers",
        help="Output directory for dossiers (default: reports/dossiers)",
    )
    parser.add_argument(
        "--data-dir",
        default="./production_data",
        help="Data directory (default: ./production_data)",
    )
    parser.add_argument(
        "--format",
        choices=["md", "pdf", "both"],
        default="md",
        help="Output format (default: md)",
    )

    # Data source options
    parser.add_argument(
        "--skip-sec",
        action="store_true",
        help="Skip SEC filing fetch",
    )
    parser.add_argument(
        "--skip-trials",
        action="store_true",
        help="Skip clinical trials fetch",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data if available",
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate results file exists if using rankings
    results_path = Path(args.results)
    if (args.top_n or args.all_ranked) and not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run the screening first or specify --results path")
        sys.exit(1)

    # Initialize generator
    logger.info("Initializing dossier generator...")
    generator = DossierGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_cache=args.use_cache,
    )

    # Determine tickers to process
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.top_n:
        tickers = generator.get_top_n_tickers(str(results_path), args.top_n)
        logger.info(f"Found top {args.top_n} tickers: {', '.join(tickers)}")
    else:  # all_ranked
        tickers = generator.get_all_ranked_tickers(str(results_path))
        logger.info(f"Found {len(tickers)} ranked tickers")

    if not tickers:
        logger.error("No tickers to process")
        sys.exit(1)

    # Generate dossiers
    print(f"\n{'='*60}")
    print(f"BIOTECH DOSSIER GENERATOR")
    print(f"{'='*60}")
    print(f"Date: {args.date}")
    print(f"Tickers: {len(tickers)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    generated = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Generating dossier for {ticker}...")

        try:
            report_path = generator.generate(
                ticker=ticker,
                as_of_date=args.date,
                results_path=str(results_path) if results_path.exists() else None,
                output_format=args.format,
                skip_sec=args.skip_sec,
                skip_trials=args.skip_trials,
            )
            generated.append((ticker, report_path))
            print(f"    -> Saved: {report_path}")

        except Exception as e:
            logger.error(f"Error generating {ticker}: {e}")
            failed.append((ticker, str(e)))
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"DOSSIER GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated: {len(generated)}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {args.output_dir}")

    if generated:
        print(f"\nGenerated files:")
        for ticker, path in generated[:10]:
            print(f"  - {ticker}: {path}")
        if len(generated) > 10:
            print(f"  ... and {len(generated) - 10} more")

    if failed:
        print(f"\nFailed tickers:")
        for ticker, error in failed:
            print(f"  - {ticker}: {error}")

    print(f"{'='*60}")

    # Return appropriate exit code
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
