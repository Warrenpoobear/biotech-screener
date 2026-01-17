#!/usr/bin/env python3
"""
module_3_catalyst.py - Module 3 Catalyst Detection Orchestrator vNext

Entry point for CT.gov catalyst detection pipeline.

Determinism Guarantees:
- All event generation uses explicit sorting
- No datetime.now() or wall-clock defaults
- Stable JSON serialization with sorted keys
- Event dedup via deterministic event_id hash

PIT Safety:
- as_of_date is REQUIRED (no defaults)
- Events after as_of_date are excluded from scoring
- All timestamps from source data, never fabricated

Version: See SCHEMA_VERSION and SCORE_VERSION in module_3_schema.py
"""

from pathlib import Path
from datetime import date
from typing import Optional, Dict, List, Set, Tuple, Any, Union
import json
import hashlib
import time
import logging
import warnings

# Type alias for flexible date input
DateLike = Union[str, date]

from ctgov_adapter import process_trial_records_batch, AdapterConfig, CanonicalTrialRecord
from state_management import StateStore, StateSnapshot
from event_detector import (
    EventDetector,
    EventDetectorConfig,
    SimpleMarketCalendar,
    MarketCalendar,
    CatalystEvent,
    EventType as LegacyEventType,
)
from catalyst_summary import CatalystAggregator, TickerCatalystSummary, CatalystOutputWriter
from module_3_schema import (
    SCHEMA_VERSION,
    SCORE_VERSION,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    CatalystEventV2,
    TickerCatalystSummaryV2,
    DiagnosticCounts,
    EVENT_SEVERITY_MAP,
    EVENT_DEFAULT_CONFIDENCE,
    canonical_json_dumps,
    validate_summary_schema,
)
from module_3_scoring import (
    calculate_ticker_catalyst_score,
    score_catalyst_events,
    compute_proximity_score,
)
from common.integration_contracts import (
    validate_module_3_output,
    is_validation_enabled,
)

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
        self.module_version = "3A.2.0"  # vNext

        # Schema versions (from module_3_schema)
        self.schema_version = SCHEMA_VERSION
        self.score_version = SCORE_VERSION

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
# EVENT CONVERSION (LEGACY -> V2)
# ============================================================================

def convert_legacy_event_to_v2(
    ticker: str,
    legacy_event: CatalystEvent,
) -> CatalystEventV2:
    """
    Convert legacy CatalystEvent to CatalystEventV2.

    Maps old event types to new taxonomy and adds required fields.
    """
    # Map legacy event type to new EventType
    legacy_type = legacy_event.event_type.value if hasattr(legacy_event.event_type, 'value') else str(legacy_event.event_type)

    try:
        event_type = EventType(legacy_type)
    except ValueError:
        event_type = EventType.UNKNOWN

    # Get severity from mapping
    severity = EVENT_SEVERITY_MAP.get(event_type, EventSeverity.NEUTRAL)

    # Get confidence from mapping or use legacy value
    if hasattr(legacy_event, 'confidence') and isinstance(legacy_event.confidence, float):
        # Map float confidence to ConfidenceLevel
        if legacy_event.confidence >= 0.85:
            confidence = ConfidenceLevel.HIGH
        elif legacy_event.confidence >= 0.6:
            confidence = ConfidenceLevel.MED
        else:
            confidence = ConfidenceLevel.LOW
    else:
        confidence = EVENT_DEFAULT_CONFIDENCE.get(event_type, ConfidenceLevel.MED)

    # Extract field changed info
    fields_changed = legacy_event.fields_changed or {}
    field_changed = list(fields_changed.keys())[0] if fields_changed else ""

    # Extract prior/new values
    prior_value = None
    new_value = None
    if fields_changed and field_changed:
        values = fields_changed.get(field_changed, [])
        if isinstance(values, list) and len(values) >= 2:
            prior_value = str(values[0]) if values[0] is not None else None
            new_value = str(values[1]) if values[1] is not None else None

    # Determine event date
    event_date = None
    if legacy_event.actual_date:
        event_date = legacy_event.actual_date.isoformat()
    elif legacy_event.disclosed_at:
        event_date = legacy_event.disclosed_at.isoformat()

    return CatalystEventV2(
        ticker=ticker,
        nct_id=legacy_event.nct_id,
        event_type=event_type,
        event_severity=severity,
        event_date=event_date,
        field_changed=field_changed,
        prior_value=prior_value,
        new_value=new_value,
        source=legacy_event.source,
        confidence=confidence,
        disclosed_at=legacy_event.disclosed_at.isoformat() if legacy_event.disclosed_at else "",
    )


# ============================================================================
# DEDUPLICATION
# ============================================================================

def dedup_events(events: List[CatalystEventV2]) -> Tuple[List[CatalystEventV2], int]:
    """
    Deduplicate events by event_id.

    Returns:
        (deduped_events, count_removed)
    """
    seen_ids: Set[str] = set()
    deduped = []

    for event in events:
        if event.event_id not in seen_ids:
            seen_ids.add(event.event_id)
            deduped.append(event)

    return (deduped, len(events) - len(deduped))


# ============================================================================
# DETERMINISTIC SORTING
# ============================================================================

def sort_canonical_records(records: List[CanonicalTrialRecord]) -> List[CanonicalTrialRecord]:
    """
    Sort canonical records deterministically by (ticker, nct_id).
    """
    return sorted(records, key=lambda r: (r.ticker, r.nct_id))


def sort_events_v2(events: List[CatalystEventV2]) -> List[CatalystEventV2]:
    """
    Sort events deterministically.

    Sort by (event_date, event_type, nct_id, field_changed, prior_value, new_value)
    """
    return sorted(events, key=lambda e: e.sort_key())


# ============================================================================
# MAIN MODULE 3 FUNCTION (CANONICAL ENTRYPOINT)
# ============================================================================

def compute_module_3_catalyst(
    trial_records_path: Path,
    state_dir: Path,
    active_tickers: Set[str],
    as_of_date: DateLike,
    market_calendar: Optional[MarketCalendar] = None,
    config: Optional[Module3Config] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main Module 3 Catalyst Detection Entry Point (vNext)

    CANONICAL API - Use this function for all new code.

    Args:
        trial_records_path: Path to trial_records.json
        state_dir: Path to ctgov_state directory
        active_tickers: Set of tickers from Module 1
        as_of_date: Point-in-time date for analysis (REQUIRED, accepts date or ISO string)
        market_calendar: Optional market calendar (uses simple if not provided)
        config: Optional configuration
        output_dir: Optional output directory (defaults to state_dir parent)

    Returns:
        {
            "summaries": {ticker: TickerCatalystSummaryV2},
            "summaries_legacy": {ticker: TickerCatalystSummary},  # DEPRECATED
            "diagnostic_counts": DiagnosticCounts,
            "as_of_date": "2024-01-15",
            "schema_version": "m3catalyst_vnext_YYYYMMDD",
            "score_version": "m3score_vnext_YYYYMMDD",
        }
    """
    start_time = time.time()

    # Validate as_of_date is provided (PIT safety)
    if as_of_date is None:
        raise ValueError("as_of_date is REQUIRED for PIT safety. Do not use defaults.")

    # Normalize date input - accept both date object and ISO string
    if isinstance(as_of_date, str):
        as_of_date = date.fromisoformat(as_of_date)
    elif not isinstance(as_of_date, date):
        raise ValueError(f"as_of_date must be date or ISO string, got {type(as_of_date)}")

    # Initialize
    if config is None:
        config = Module3Config()

    if market_calendar is None:
        market_calendar = SimpleMarketCalendar()

    if output_dir is None:
        output_dir = state_dir.parent

    logger.info(f"Starting Module 3 catalyst detection for {as_of_date}")
    logger.info(f"Schema version: {config.schema_version}")
    logger.info(f"Score version: {config.score_version}")
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

    # DETERMINISTIC: Sort canonical records
    canonical_records = sort_canonical_records(canonical_records)

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
    events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
    events_by_ticker_legacy: Dict[str, List[CatalystEvent]] = {}
    prior_events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
    total_events = 0
    total_deduped = 0

    for current_record in canonical_records:
        ticker = current_record.ticker
        nct_id = current_record.nct_id

        # Get prior record for this trial
        prior_record = None
        if prior_snapshot:
            prior_record = prior_snapshot.get_record(ticker, nct_id)

        # Detect events (legacy format)
        legacy_events = event_detector.detect_events(
            current_record,
            prior_record,
            as_of_date
        )

        if legacy_events:
            if ticker not in events_by_ticker_legacy:
                events_by_ticker_legacy[ticker] = []
            events_by_ticker_legacy[ticker].extend(legacy_events)

            # Convert to V2 format
            if ticker not in events_by_ticker_v2:
                events_by_ticker_v2[ticker] = []

            for le in legacy_events:
                v2_event = convert_legacy_event_to_v2(ticker, le)
                events_by_ticker_v2[ticker].append(v2_event)

            total_events += len(legacy_events)

    # DEDUP: Remove duplicate events by event_id
    for ticker in events_by_ticker_v2:
        events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
        total_deduped += deduped

    # DETERMINISTIC: Sort events per ticker
    for ticker in events_by_ticker_v2:
        events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])

    logger.info(f"Detected {total_events} events across {len(events_by_ticker_v2)} tickers")
    logger.info(f"Deduped {total_deduped} duplicate events")

    # Build prior events for delta scoring (from prior snapshot)
    if prior_snapshot:
        for ticker in active_tickers:
            prior_records = prior_snapshot.get_records_for_ticker(ticker) if hasattr(prior_snapshot, 'get_records_for_ticker') else []
            if prior_records:
                prior_events_by_ticker_v2[ticker] = []
                # Note: Prior events would need to be reconstructed from prior snapshot
                # For now, we use the detected events from delta comparison

    # Load historical proximity scores from state store
    historical_proximities = state_store.get_historical_proximities(4) if hasattr(state_store, 'get_historical_proximities') else {}

    # Score events using new scoring system
    logger.info("Scoring events with vNext scorer...")
    summaries_v2, diagnostics = score_catalyst_events(
        events_by_ticker_v2,
        list(active_tickers),
        as_of_date,
        prior_events_by_ticker=prior_events_by_ticker_v2 if prior_events_by_ticker_v2 else None,
        historical_proximities_by_ticker=historical_proximities if historical_proximities else None,
    )

    # Update diagnostics with dedup count
    diagnostics.events_deduped = total_deduped

    # Also generate legacy summaries for backwards compatibility
    logger.info("Generating legacy summaries for backwards compatibility...")
    summaries_legacy: Dict[str, TickerCatalystSummary] = {}

    for ticker in active_tickers:
        ticker_events_legacy = events_by_ticker_legacy.get(ticker, [])
        summary = aggregator.aggregate(ticker, ticker_events_legacy, as_of_date)
        summaries_legacy[ticker] = summary

    # Compute legacy diagnostic counts (for backwards compat)
    severe_negatives = sum(1 for s in summaries_legacy.values() if s.severe_negative_flag)
    tickers_with_events = len([s for s in summaries_legacy.values() if s.events])

    diagnostic_counts_legacy = {
        'events_detected': total_events,
        'events_deduped': total_deduped,
        'severe_negatives': severe_negatives,
        'tickers_with_events': tickers_with_events,
        'tickers_analyzed': len(active_tickers)
    }

    logger.info(f"Diagnostics: {diagnostics.to_dict()}")

    # Save current snapshot
    logger.info("Saving current snapshot...")
    state_store.save_snapshot(current_snapshot)

    # Write outputs
    if output_dir:
        # Write vNext format
        vnext_path = output_dir / f"catalyst_events_vnext_{as_of_date.isoformat()}.json"
        logger.info(f"Writing vNext catalyst events to {vnext_path}")
        write_vnext_output(
            summaries_v2,
            diagnostics,
            as_of_date,
            prior_snapshot_date,
            config,
            vnext_path,
        )

        # Write legacy format for backwards compat
        legacy_path = output_dir / f"catalyst_events_{as_of_date.isoformat()}.json"
        logger.info(f"Writing legacy catalyst events to {legacy_path}")
        CatalystOutputWriter.write_catalyst_events(
            list(summaries_legacy.values()),
            as_of_date,
            str(legacy_path),
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
                'decay_constant': config.decay_constant,
                'schema_version': config.schema_version,
                'score_version': config.score_version,
            }
        )

    execution_time = time.time() - start_time
    logger.info(f"Module 3 completed in {execution_time:.2f} seconds")

    # Emit deprecation warning for legacy format
    warnings.warn(
        "Module 3: 'summaries_legacy' and 'diagnostic_counts_legacy' are deprecated "
        "and will be removed in v2.0. Use 'summaries' with TickerCatalystSummaryV2 objects instead.",
        DeprecationWarning,
        stacklevel=2
    )

    output = {
        "summaries": summaries_v2,
        "summaries_legacy": summaries_legacy,  # DEPRECATED - use summaries instead
        "diagnostic_counts": diagnostics.to_dict(),
        "diagnostic_counts_legacy": diagnostic_counts_legacy,  # DEPRECATED
        "as_of_date": as_of_date.isoformat(),
        "schema_version": config.schema_version,
        "score_version": config.score_version,
    }

    # Output validation
    if is_validation_enabled():
        validate_module_3_output(output)

    return output


# ============================================================================
# OUTPUT WRITER (vNext)
# ============================================================================

def write_vnext_output(
    summaries: Dict[str, TickerCatalystSummaryV2],
    diagnostics: DiagnosticCounts,
    as_of_date: date,
    prior_snapshot_date: Optional[date],
    config: Module3Config,
    output_path: Path,
) -> str:
    """
    Write vNext catalyst events output with deterministic JSON.

    Returns:
        SHA256 hash of output for verification
    """
    # Sort tickers for determinism
    sorted_tickers = sorted(summaries.keys())

    # Build output
    output = {
        "_schema": {
            "schema_version": config.schema_version,
            "score_version": config.score_version,
        },
        "run_metadata": {
            "as_of_date": as_of_date.isoformat(),
            "prior_snapshot_date": prior_snapshot_date.isoformat() if prior_snapshot_date else None,
            "module_version": config.module_version,
        },
        "diagnostics": diagnostics.to_dict(),
        "summaries": {
            ticker: summaries[ticker].to_dict()
            for ticker in sorted_tickers
        },
    }

    # Write with canonical JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_json = canonical_json_dumps(output)
    output_path.write_text(canonical_json, encoding='utf-8')

    # Compute hash for verification
    file_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    logger.info(f"Wrote vNext catalyst events: {output_path} (hash: {file_hash[:16]}...)")

    return file_hash


# ============================================================================
# INTEGRATION HOOKS
# ============================================================================

def get_integration_record(summary: TickerCatalystSummaryV2) -> Dict[str, Any]:
    """
    Get merge-friendly record for institutional layer integration.

    Returns a flat dict that can be joined on ticker.
    """
    return {
        "ticker": summary.ticker,
        "score_override": str(summary.score_override),
        "score_blended": str(summary.score_blended),
        "severe_negative_flag": summary.severe_negative_flag,
        "next_catalyst_date": summary.next_catalyst_date,
        "catalyst_window_bucket": summary.catalyst_window_bucket.value,
        "catalyst_confidence": summary.catalyst_confidence.value,
        "top_3_events": summary.top_3_events,
    }


def get_all_integration_records(
    summaries: Dict[str, TickerCatalystSummaryV2],
) -> Dict[str, Dict[str, Any]]:
    """
    Get integration records for all tickers.

    Returns:
        {ticker: integration_record}
    """
    return {
        ticker: get_integration_record(summary)
        for ticker, summary in sorted(summaries.items())
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for Module 3"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Module 3: CT.gov Catalyst Detection (vNext)'
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
        help='Point-in-time date (YYYY-MM-DD) - REQUIRED'
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

    # Parse date (no defaults - PIT safety)
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
    print("MODULE 3 CATALYST DETECTION COMPLETE (vNext)")
    print("="*80)
    print(f"Schema version: {result['schema_version']}")
    print(f"Score version: {result['score_version']}")
    print(f"Events detected: {diag['events_detected_total']}")
    print(f"Events deduped: {diag['events_deduped']}")
    print(f"Tickers with events: {diag['tickers_with_events']}/{diag['tickers_analyzed']}")
    print(f"Severe negatives: {diag['tickers_with_severe_negative']}")
    print("="*80)


if __name__ == "__main__":
    main()
