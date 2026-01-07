#!/usr/bin/env python3
"""
module_3_catalyst.py - Main Module 3 Catalyst Detection Orchestrator

Entry point for CT.gov catalyst detection pipeline.
"""

from pathlib import Path
from datetime import date
from typing import Optional
import json
import time
import logging

from ctgov_adapter import process_trial_records_batch, AdapterConfig, CanonicalTrialRecord
from state_management import StateStore, StateSnapshot
from event_detector import EventDetector, EventDetectorConfig, SimpleMarketCalendar, MarketCalendar
from catalyst_summary import CatalystAggregator, TickerCatalystSummary, CatalystOutputWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODULE 3 CONFIGURATION
# ============================================================================

class Module3Config:
    """Configuration for Module 3 catalyst detection"""
    
    def __init__(self):
        # Adapter config
        self.adapter_config = AdapterConfig()
        
        # Event detector config
        self.event_detector_config = EventDetectorConfig()
        
        # Aggregator config
        self.decay_constant = 30.0
        
        # Output config
        self.module_version = "3A.1.1"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Module3Config':
        """Create config from dict"""
        config = cls()
        
        # Update event detector settings
        if 'noise_band_days' in config_dict:
            config.event_detector_config.noise_band_days = config_dict['noise_band_days']
        if 'recency_threshold_days' in config_dict:
            config.event_detector_config.recency_threshold_days = config_dict['recency_threshold_days']
        if 'decay_constant' in config_dict:
            config.decay_constant = config_dict['decay_constant']
        if 'confidence_scores' in config_dict:
            config.event_detector_config.confidence_scores = config_dict['confidence_scores']
        
        return config


# ============================================================================
# MAIN MODULE 3 FUNCTION
# ============================================================================

def compute_module_3_catalyst(
    trial_records_path: Path,
    state_dir: Path,
    active_tickers: set[str],
    as_of_date: date,
    market_calendar: Optional[MarketCalendar] = None,
    config: Optional[Module3Config] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Main Module 3 Catalyst Detection Entry Point
    
    Args:
        trial_records_path: Path to trial_records.json
        state_dir: Path to ctgov_state directory
        active_tickers: Set of tickers from Module 1
        as_of_date: Point-in-time date for analysis
        market_calendar: Optional market calendar (uses simple if not provided)
        config: Optional configuration
        output_dir: Optional output directory (defaults to state_dir parent)
    
    Returns:
        {
            "summaries": {ticker: TickerCatalystSummary},
            "diagnostic_counts": {...},
            "as_of_date": "2024-01-15"
        }
    """
    start_time = time.time()
    
    # Initialize
    if config is None:
        config = Module3Config()
    
    if market_calendar is None:
        market_calendar = SimpleMarketCalendar()
    
    if output_dir is None:
        output_dir = state_dir.parent
    
    logger.info(f"Starting Module 3 catalyst detection for {as_of_date}")
    logger.info(f"Active tickers: {len(active_tickers)}")
    
    # Initialize components
    state_store = StateStore(state_dir)
    event_detector = EventDetector(config.event_detector_config)
    aggregator = CatalystAggregator(market_calendar, config.decay_constant)
    
    # Load trial records
    logger.info(f"Loading trial records from {trial_records_path}")
    with open(trial_records_path) as f:
        trial_records = json.load(f)
    
    # Filter to active tickers
    trial_records = [r for r in trial_records if r.get('ticker') in active_tickers]
    logger.info(f"Processing {len(trial_records)} trials for active tickers")
    
    # Convert to canonical format
    logger.info("Converting to canonical format...")
    canonical_records, adapter_stats = process_trial_records_batch(
        trial_records,
        as_of_date,
        config.adapter_config
    )
    logger.info(f"Converted {len(canonical_records)} records successfully")
    
    # Load prior state
    prior_snapshot = state_store.get_most_recent_snapshot()
    prior_snapshot_date = prior_snapshot.snapshot_date if prior_snapshot else None
    
    if prior_snapshot:
        logger.info(f"Loaded prior snapshot from {prior_snapshot_date}")
        logger.info(f"Prior snapshot has {prior_snapshot.key_count} records")
    else:
        logger.info("No prior snapshot found - this is initial run")
    
    # Create current snapshot
    current_snapshot = StateSnapshot(
        snapshot_date=as_of_date,
        records=canonical_records
    )
    
    # Detect events by comparing states
    logger.info("Detecting catalyst events...")
    events_by_ticker = {}
    total_events = 0
    
    for current_record in canonical_records:
        ticker = current_record.ticker
        nct_id = current_record.nct_id
        
        # Get prior record for this trial
        prior_record = None
        if prior_snapshot:
            prior_record = prior_snapshot.get_record(ticker, nct_id)
        
        # Detect events
        events = event_detector.detect_events(
            current_record,
            prior_record,
            as_of_date
        )
        
        if events:
            if ticker not in events_by_ticker:
                events_by_ticker[ticker] = []
            events_by_ticker[ticker].extend(events)
            total_events += len(events)
    
    logger.info(f"Detected {total_events} events across {len(events_by_ticker)} tickers")
    
    # Aggregate events into ticker summaries
    logger.info("Aggregating events into ticker summaries...")
    summaries = []
    summaries_dict = {}
    
    for ticker in active_tickers:
        ticker_events = events_by_ticker.get(ticker, [])
        summary = aggregator.aggregate(ticker, ticker_events, as_of_date)
        summaries.append(summary)
        summaries_dict[ticker] = summary
    
    # Compute diagnostics
    severe_negatives = sum(1 for s in summaries if s.severe_negative_flag)
    tickers_with_events = len([s for s in summaries if s.events])
    
    diagnostic_counts = {
        'events_detected': total_events,
        'severe_negatives': severe_negatives,
        'tickers_with_events': tickers_with_events,
        'tickers_analyzed': len(active_tickers)
    }
    
    logger.info(f"Diagnostics: {diagnostic_counts}")
    
    # Save current snapshot
    logger.info("Saving current snapshot...")
    state_store.save_snapshot(current_snapshot)
    
    # Write outputs
    if output_dir:
        catalyst_events_path = output_dir / f"catalyst_events_{as_of_date.isoformat()}.json"
        logger.info(f"Writing catalyst events to {catalyst_events_path}")
        CatalystOutputWriter.write_catalyst_events(
            summaries,
            as_of_date,
            str(catalyst_events_path),
            prior_snapshot_date,
            config.module_version
        )
        
        # Write run log
        execution_time = time.time() - start_time
        run_log_path = output_dir / f"run_log_{as_of_date.isoformat()}.json"
        CatalystOutputWriter.write_run_log(
            str(run_log_path),
            execution_time,
            config={
                'noise_band_days': config.event_detector_config.noise_band_days,
                'recency_threshold_days': config.event_detector_config.recency_threshold_days,
                'decay_constant': config.decay_constant
            }
        )
    
    execution_time = time.time() - start_time
    logger.info(f"Module 3 completed in {execution_time:.2f} seconds")
    
    return {
        "summaries": summaries_dict,
        "diagnostic_counts": diagnostic_counts,
        "as_of_date": as_of_date.isoformat()
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for Module 3"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Module 3: CT.gov Catalyst Detection'
    )
    parser.add_argument(
        '--trial-records',
        type=str,
        default='production_data/trial_records.json',
        help='Path to trial_records.json'
    )
    parser.add_argument(
        '--state-dir',
        type=str,
        default='production_data/ctgov_state',
        help='Path to state directory'
    )
    parser.add_argument(
        '--as-of-date',
        type=str,
        required=True,
        help='Point-in-time date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--universe',
        type=str,
        default='production_data/universe.json',
        help='Path to universe.json (for active tickers)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='production_data',
        help='Output directory for catalyst events'
    )
    
    args = parser.parse_args()
    
    # Parse date
    as_of_date = date.fromisoformat(args.as_of_date)
    
    # Load universe for active tickers
    with open(args.universe) as f:
        universe = json.load(f)
    active_tickers = {s['ticker'] for s in universe if s.get('ticker') != '_XBI_BENCHMARK_'}
    
    # Run Module 3
    result = compute_module_3_catalyst(
        trial_records_path=Path(args.trial_records),
        state_dir=Path(args.state_dir),
        active_tickers=active_tickers,
        as_of_date=as_of_date,
        output_dir=Path(args.output_dir)
    )
    
    # Print summary
    diag = result['diagnostic_counts']
    print()
    print("="*80)
    print("MODULE 3 CATALYST DETECTION COMPLETE")
    print("="*80)
    print(f"Events detected: {diag['events_detected']}")
    print(f"Tickers with events: {diag['tickers_with_events']}/{diag['tickers_analyzed']}")
    print(f"Severe negatives: {diag['severe_negatives']}")
    print("="*80)


if __name__ == "__main__":
    main()
