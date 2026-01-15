#!/usr/bin/env python3
"""
automate_pipeline.py - Complete Automated Pipeline

Downloads SEC filings → Extracts CFO → Prepares for Module 2

Usage:
    python automate_pipeline.py --config universe.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import date
import logging

# Import our tools
from sec_filing_downloader import SECDownloader
from cfo_extractor import extract_cfo_batch, prepare_for_module_2, save_cfo_records

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_universe(config_path: Path) -> list:
    """Load universe tickers from config file"""
    with open(config_path) as f:
        data = json.load(f)
    
    # Support multiple formats
    if isinstance(data, list):
        # Simple list of tickers
        if isinstance(data[0], str):
            return [t.strip().upper() for t in data]
        # List of dicts with 'ticker' field
        elif isinstance(data[0], dict):
            return [t['ticker'].strip().upper() for t in data if 'ticker' in t]
    elif isinstance(data, dict):
        # Dict with 'tickers' key
        if 'tickers' in data:
            return [t.strip().upper() for t in data['tickers']]
        # Dict with 'active_securities' key (from Module 1)
        elif 'active_securities' in data:
            return [s['ticker'].strip().upper() for s in data['active_securities']]
    
    raise ValueError(f"Unsupported universe format in {config_path}")


def run_pipeline(
    tickers: list,
    output_dir: Path = Path("production_data"),
    filings_count: int = 8,
    user_agent: str = None,
    skip_download: bool = False
):
    """
    Run complete automated pipeline.
    
    Args:
        tickers: List of ticker symbols
        output_dir: Output directory for all files
        filings_count: Number of filings per ticker
        user_agent: SEC User-Agent (required)
        skip_download: Skip download step (use existing filings)
    """
    logger.info(f"Starting automated pipeline for {len(tickers)} tickers")
    
    # Create directories
    filings_dir = output_dir / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    
    as_of_date = date.today()
    
    # Step 1: Download SEC filings
    if not skip_download:
        logger.info("="*80)
        logger.info("STEP 1: Downloading SEC Filings")
        logger.info("="*80)
        
        if not user_agent:
            logger.error("User-Agent is required for SEC downloads!")
            logger.error("Use: --user-agent 'YourName/1.0 (your@email.com)'")
            return False
        
        downloader = SECDownloader(filings_dir, user_agent)
        
        download_results = downloader.download_batch(
            tickers,
            form_types=["10-Q", "10-K"],
            count=filings_count
        )
        
        # Save download log
        downloader.save_download_log(output_dir / "download_log.json")
        
        total_files = sum(len(v) for v in download_results.values())
        logger.info(f"Downloaded {total_files} files")
        
        if total_files == 0:
            logger.error("No files downloaded - check tickers and User-Agent")
            return False
    else:
        logger.info("Skipping download step (using existing filings)")
    
    # Step 2: Extract CFO data
    logger.info("="*80)
    logger.info("STEP 2: Extracting CFO Data")
    logger.info("="*80)
    
    # Build ticker -> filings mapping
    ticker_filings = {}
    for ticker_dir in filings_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name.upper()
            if ticker in tickers:
                filing_files = list(ticker_dir.glob("*.txt"))
                if filing_files:
                    ticker_filings[ticker] = filing_files
    
    logger.info(f"Found filings for {len(ticker_filings)} tickers")
    
    if not ticker_filings:
        logger.error("No filings found to extract!")
        return False
    
    # Extract CFO
    cfo_records = extract_cfo_batch(ticker_filings, as_of_date)
    
    # Save raw CFO records
    cfo_output = output_dir / f"cfo_data_{as_of_date.isoformat()}.json"
    save_cfo_records(cfo_records, cfo_output)
    
    # Step 3: Prepare for Module 2
    logger.info("="*80)
    logger.info("STEP 3: Preparing Module 2 Data")
    logger.info("="*80)
    
    module_2_data = prepare_for_module_2(cfo_records)
    
    # Save Module 2 format
    module_2_output = output_dir / f"financial_data_cfo_{as_of_date.isoformat()}.json"
    with open(module_2_output, 'w') as f:
        json.dump(module_2_data, f, indent=2)
    
    logger.info(f"Prepared {len(module_2_data)} ticker records for Module 2")
    
    # Summary
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Tickers processed: {len(ticker_filings)}")
    logger.info(f"CFO records extracted: {sum(len(r) for r in cfo_records.values())}")
    logger.info(f"Module 2 records ready: {len(module_2_data)}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - Raw CFO data: {cfo_output}")
    logger.info(f"  - Module 2 data: {module_2_output}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Merge with your existing financial_data.json")
    logger.info("  2. Run Module 2 with burn acceleration")
    logger.info("  3. Review burn_acceleration field in output")
    logger.info("="*80)
    
    return True


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Automated SEC filing → CFO extraction pipeline'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to universe config JSON file (list of tickers)'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated tickers (alternative to --config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='production_data',
        help='Output directory (default: production_data)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=8,
        help='Filings per ticker (default: 8)'
    )
    parser.add_argument(
        '--user-agent',
        type=str,
        required=True,
        help='User-Agent for SEC (REQUIRED): "YourName/1.0 (your@email.com)"'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step (use existing filings)'
    )
    
    args = parser.parse_args()
    
    # Get tickers
    if args.config:
        tickers = load_universe(Path(args.config))
        logger.info(f"Loaded {len(tickers)} tickers from {args.config}")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        logger.info(f"Using {len(tickers)} tickers from command line")
    else:
        parser.error("Must provide either --config or --tickers")
        return 1
    
    # Run pipeline
    success = run_pipeline(
        tickers,
        output_dir=Path(args.output_dir),
        filings_count=args.count,
        user_agent=args.user_agent,
        skip_download=args.skip_download
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
