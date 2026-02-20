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
from datetime import date, timedelta
from decimal import Decimal
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
    detect_activity_proxy_from_lookback,
    compute_activity_proxy_score,
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
)
from module_3_scoring import (
    score_catalyst_events,
)
from catalyst_diagnostics import (
    compute_delta_diagnostics,
    check_trial_records_staleness,
    detect_calendar_catalysts,
    detect_readout_window_catalysts,
    summarize_calendar_catalysts,
    DeltaDiagnostics,
    StalenessResult,
    CalendarCatalyst,
    EventRuleID,
)
from time_decay_scoring import (
    TimeDecayConfig,
    compute_time_decay_score,
    score_all_tickers_with_time_decay,
    DEFAULT_TIME_DECAY_WINDOWS,
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

        # Time-decay scoring config
        self.enable_time_decay = True
        self.time_decay_config = TimeDecayConfig()

        # Corporate catalyst config
        self.enable_corporate_catalysts = True
        self.corporate_catalysts_file = "corporate_catalysts.json"

        # FDA ADCOM config
        # Modes: "cache_only" (default) = load from cache if present, no network
        #        "live"       = fetch from web, write cache, merge
        #        "off"        = skip entirely
        self.enable_fda_adcom: str = "cache_only"
        self.fda_adcom_cache_dir = Path("cache/fda")

        # SEC 8-K catalyst config (same modes as above)
        self.enable_sec_8k_catalysts: str = "cache_only"
        self.sec_8k_cache_dir = Path("cache/sec/8k_catalysts")

        # SEC multi-form catalyst config (10-Q, 10-K, 6-K — same modes)
        self.enable_sec_multi_form: str = "off"

        # FDA regulatory notices config (approvals, CRLs, etc. from Federal Register)
        self.enable_fda_regulatory: str = "off"

        # CTGov cache override: None=auto-detect from repo, False=disable
        self.ctgov_cache_dir: Optional[Any] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Module3Config':
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

        # Time-decay settings
        if 'enable_time_decay' in config_dict:
            config.enable_time_decay = config_dict['enable_time_decay']
        if 'time_decay_config' in config_dict:
            config.time_decay_config = TimeDecayConfig.from_dict(config_dict['time_decay_config'])

        # Corporate catalyst settings
        if 'enable_corporate_catalysts' in config_dict:
            config.enable_corporate_catalysts = config_dict['enable_corporate_catalysts']
        if 'corporate_catalysts_file' in config_dict:
            config.corporate_catalysts_file = config_dict['corporate_catalysts_file']

        # FDA ADCOM settings (accept bool for backwards compat: True→"live", False→"off")
        if 'enable_fda_adcom' in config_dict:
            val = config_dict['enable_fda_adcom']
            if val is True:
                config.enable_fda_adcom = "live"
            elif val is False:
                config.enable_fda_adcom = "off"
            else:
                config.enable_fda_adcom = str(val)
        if 'fda_adcom_cache_dir' in config_dict:
            config.fda_adcom_cache_dir = Path(config_dict['fda_adcom_cache_dir'])

        # SEC 8-K settings (same bool compat)
        if 'enable_sec_8k_catalysts' in config_dict:
            val = config_dict['enable_sec_8k_catalysts']
            if val is True:
                config.enable_sec_8k_catalysts = "live"
            elif val is False:
                config.enable_sec_8k_catalysts = "off"
            else:
                config.enable_sec_8k_catalysts = str(val)
        if 'sec_8k_cache_dir' in config_dict:
            config.sec_8k_cache_dir = Path(config_dict['sec_8k_cache_dir'])

        # SEC multi-form settings (same bool compat)
        if 'enable_sec_multi_form' in config_dict:
            val = config_dict['enable_sec_multi_form']
            if val is True:
                config.enable_sec_multi_form = "live"
            elif val is False:
                config.enable_sec_multi_form = "off"
            else:
                config.enable_sec_multi_form = str(val)

        # FDA regulatory settings (same bool compat)
        if 'enable_fda_regulatory' in config_dict:
            val = config_dict['enable_fda_regulatory']
            if val is True:
                config.enable_fda_regulatory = "live"
            elif val is False:
                config.enable_fda_regulatory = "off"
            else:
                config.enable_fda_regulatory = str(val)

        return config


# ============================================================================
# EVENT CONVERSION (LEGACY -> V2)
# ============================================================================

def convert_calendar_catalyst_to_v2(
    calendar_catalyst: CalendarCatalyst,
    as_of_date: Optional[date] = None,
) -> CatalystEventV2:
    """
    Convert CalendarCatalyst (forward-looking) to CatalystEventV2.

    Maps calendar event types to appropriate EventType enum values.
    Calendar catalysts represent upcoming events detected from trial date fields.

    PIT safety: disclosed_at is clipped to as_of_date when the event date is
    in the future (the event is "known" from the current snapshot, not from
    the future date itself).
    """
    # Map calendar event types to EventType enum
    CALENDAR_EVENT_TYPE_MAP = {
        'UPCOMING_PCD': EventType.CT_PRIMARY_COMPLETION,
        'UPCOMING_SCD': EventType.CT_STUDY_COMPLETION,
        'RESULTS_DUE': EventType.CT_RESULTS_POSTED,
        'RESULTS_RECENT': EventType.CT_RESULTS_POSTED,
        'READOUT_WINDOW': EventType.DATA_READOUT,
    }

    event_type = CALENDAR_EVENT_TYPE_MAP.get(
        calendar_catalyst.event_type,
        EventType.CT_PRIMARY_COMPLETION  # Default fallback
    )

    # Get severity from mapping
    severity = EVENT_SEVERITY_MAP.get(event_type, EventSeverity.POSITIVE)

    # Map confidence float to ConfidenceLevel
    if calendar_catalyst.confidence >= 0.85:
        confidence = ConfidenceLevel.HIGH
    elif calendar_catalyst.confidence >= 0.65:
        confidence = ConfidenceLevel.MED
    else:
        confidence = ConfidenceLevel.LOW

    du = calendar_catalyst.days_until
    if du is None:
        rel = None
    else:
        du_i = int(du)
        rel = f"{abs(du_i)}d_ago" if du_i < 0 else f"{du_i}d_ahead"

    # Determine date range and precision for readout window events
    event_date_end = None
    date_precision = "DAY"
    tags = ()
    disclosed_at = calendar_catalyst.target_date.isoformat()

    if calendar_catalyst.event_type == 'READOUT_WINDOW':
        evidence_fields = calendar_catalyst.evidence.fields
        event_date_val = evidence_fields.get('window_start', calendar_catalyst.target_date.isoformat())
        event_date_end = evidence_fields.get('window_end')
        date_precision = "RANGE"
        tags = ("readout_window",)
        # PIT safety: disclosed_at = PCD (when it became known), not as_of_date
        disclosed_at = evidence_fields.get(
            'primary_completion_date', calendar_catalyst.target_date.isoformat()
        )
    else:
        event_date_val = calendar_catalyst.target_date.isoformat()

    # PIT safety: clip disclosed_at to as_of_date — a future event date is
    # "known" from the current snapshot, not from the future date itself.
    if as_of_date is not None:
        as_of_str = as_of_date.isoformat()
        if disclosed_at > as_of_str:
            disclosed_at = as_of_str
            tags = tuple(sorted(set(tags) | {"disclosed_at_clipped"}))

    return CatalystEventV2(
        ticker=calendar_catalyst.ticker,
        nct_id=calendar_catalyst.nct_id,
        event_type=event_type,
        event_severity=severity,
        event_date=event_date_val,
        field_changed=f"calendar_{calendar_catalyst.event_type.lower()}",
        prior_value=None,
        new_value=rel,
        source="CTGOV_CALENDAR",
        confidence=confidence,
        disclosed_at=disclosed_at,
        event_date_end=event_date_end,
        date_precision=date_precision,
        tags=tags,
    )


# ============================================================================
# CORPORATE CATALYST LOADING AND CONVERSION
# ============================================================================

def load_corporate_catalysts(
    data_dir: Path,
    filename: str = "corporate_catalysts.json",
    as_of_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """
    Load corporate catalyst events from JSON file.

    Args:
        data_dir: Directory containing the corporate catalysts file
        filename: Name of the corporate catalysts file
        as_of_date: Filter events to only include those on or after this date

    Returns:
        List of corporate catalyst event dicts
    """
    filepath = data_dir / filename
    if not filepath.exists():
        logger.debug(f"Corporate catalysts file not found: {filepath}")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = data.get('events', [])

        # Filter events by as_of_date if provided
        if as_of_date:
            filtered_events = []
            for event in events:
                event_date_str = event.get('event_date')
                if event_date_str:
                    try:
                        event_date = date.fromisoformat(event_date_str)
                        # Only include future events (on or after as_of_date)
                        if event_date >= as_of_date:
                            filtered_events.append(event)
                    except ValueError:
                        continue
            return filtered_events

        return events

    except Exception as e:
        logger.warning(f"Error loading corporate catalysts: {e}")
        return []


def convert_corporate_catalyst_to_v2(
    event: Dict[str, Any],
    as_of_date: date,
) -> Optional[CatalystEventV2]:
    """
    Convert corporate catalyst event dict to CatalystEventV2.

    Maps corporate event types (FDA_PDUFA_DATE, EARNINGS_RELEASE, etc.)
    to CatalystEventV2 format for scoring integration.
    """
    ticker = event.get('ticker')
    event_type_str = event.get('event_type')
    event_date_str = event.get('event_date')

    if not ticker or not event_type_str or not event_date_str:
        return None

    # Map event type string to EventType enum
    try:
        event_type = EventType(event_type_str)
    except ValueError:
        logger.debug(f"Unknown corporate event type: {event_type_str}")
        return None

    # Get severity from mapping
    severity = EVENT_SEVERITY_MAP.get(event_type, EventSeverity.POSITIVE)

    # Map confidence string to ConfidenceLevel
    confidence_str = event.get('confidence', 'MED').upper()
    if confidence_str == 'HIGH':
        confidence = ConfidenceLevel.HIGH
    elif confidence_str == 'LOW':
        confidence = ConfidenceLevel.LOW
    else:
        confidence = ConfidenceLevel.MED

    # Calculate days until event
    try:
        event_date = date.fromisoformat(event_date_str)
        days_until = (event_date - as_of_date).days
    except ValueError:
        days_until = 0

    # Build event name/description
    event_name = event.get('event_name', event_type_str)
    drug_name = event.get('drug_name', '')
    indication = event.get('indication', '')

    # Create descriptive new_value
    if drug_name:
        new_value = f"{days_until}d_ahead: {drug_name}"
        if indication:
            new_value += f" ({indication})"
    else:
        new_value = f"{days_until}d_ahead"

    # Source mapping
    source_map = {
        'FDA_PDUFA_DATE': 'FDA_CALENDAR',
        'FDA_ADCOM': 'FDA_CALENDAR',
        'FDA_SUBMISSION': 'FDA_CALENDAR',
        'FDA_APPROVAL': 'FDA_CALENDAR',
        'FDA_CRL': 'FDA_CALENDAR',
        'FDA_DESIGNATION': 'FDA_CALENDAR',
        'EARNINGS_RELEASE': 'CORPORATE_CALENDAR',
        'CONFERENCE_PRESENTATION': 'CORPORATE_CALENDAR',
        'INVESTOR_DAY': 'CORPORATE_CALENDAR',
        'DATA_READOUT': 'COMPANY_GUIDANCE',
        'DATA_PRESENTATION': 'COMPANY_GUIDANCE',
        'DATA_PUBLICATION': 'COMPANY_GUIDANCE',
        'PARTNERSHIP': 'CORPORATE_EVENT',
        'MA_ACTIVITY': 'CORPORATE_EVENT',
        'LICENSING_DEAL': 'CORPORATE_EVENT',
    }
    source = source_map.get(event_type_str, 'CORPORATE_CALENDAR')

    return CatalystEventV2(
        ticker=ticker,
        nct_id=f"CORP_{event_type_str}_{ticker}_{event_date_str}",  # Synthetic ID for corporate events
        event_type=event_type,
        event_severity=severity,
        event_date=event_date_str,
        field_changed=f"corporate_{event_type_str.lower()}",
        prior_value=None,
        new_value=new_value,
        source=source,
        confidence=confidence,
        disclosed_at=as_of_date.isoformat(),  # Use as_of_date since we don't have disclosure date
    )


def summarize_corporate_catalysts(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize corporate catalyst events for logging.
    """
    if not events:
        return {
            'total_events': 0,
            'tickers_with_events': 0,
            'by_type': {},
        }

    tickers = set()
    by_type = {}

    for event in events:
        ticker = event.get('ticker')
        event_type = event.get('event_type', 'UNKNOWN')

        if ticker:
            tickers.add(ticker)

        by_type[event_type] = by_type.get(event_type, 0) + 1

    return {
        'total_events': len(events),
        'tickers_with_events': len(tickers),
        'by_type': by_type,
    }


def convert_fda_adcom_to_v2(
    event: Dict[str, Any],
    as_of_date: date,
) -> Optional[CatalystEventV2]:
    """
    Convert FDA ADCOM event dict to CatalystEventV2.

    ADCOM dates are official → HIGH confidence.
    """
    ticker = event.get('ticker')
    event_date_str = event.get('event_date')
    if not ticker or not event_date_str:
        return None

    drug_name = event.get('drug_name', '')
    event_name = event.get('event_name', f"FDA_ADCOM: {drug_name}")

    try:
        event_date = date.fromisoformat(event_date_str)
        days_until = (event_date - as_of_date).days
    except ValueError:
        days_until = 0

    new_value = f"{days_until}d_ahead: {event_name}" if drug_name else f"{days_until}d_ahead"

    return CatalystEventV2(
        ticker=ticker,
        nct_id=f"FDA_ADCOM_{ticker}_{event_date_str}",
        event_type=EventType.FDA_ADCOM,
        event_severity=EVENT_SEVERITY_MAP.get(EventType.FDA_ADCOM, EventSeverity.CRITICAL_POSITIVE),
        event_date=event_date_str,
        field_changed="fda_adcom",
        prior_value=None,
        new_value=new_value,
        source="FDA_ADCOM_CALENDAR",
        confidence=ConfidenceLevel.HIGH,
        disclosed_at=event.get('disclosed_at', as_of_date.isoformat()),
        tags=tuple(event.get('tags', [])),
    )


def convert_sec_8k_to_v2(
    event: Dict[str, Any],
    as_of_date: date,
) -> Optional[CatalystEventV2]:
    """
    Convert SEC 8-K timing event dict to CatalystEventV2.

    Preserves date_precision, event_date_end, and uses filing date as disclosed_at.
    """
    ticker = event.get('ticker')
    event_type_str = event.get('event_type')
    event_date_str = event.get('event_date')
    if not ticker or not event_type_str or not event_date_str:
        return None

    try:
        event_type = EventType(event_type_str)
    except ValueError:
        logger.warning(
            "SEC 8-K: unknown event_type '%s' for %s, dropping event",
            event_type_str, ticker,
        )
        return None

    severity = EVENT_SEVERITY_MAP.get(event_type, EventSeverity.NEUTRAL)

    # Map confidence string to ConfidenceLevel
    conf_str = event.get('confidence', 'MED').upper()
    confidence = {'HIGH': ConfidenceLevel.HIGH, 'LOW': ConfidenceLevel.LOW}.get(
        conf_str, ConfidenceLevel.MED
    )

    try:
        event_date = date.fromisoformat(event_date_str)
        days_until = (event_date - as_of_date).days
    except ValueError:
        days_until = 0

    event_name = event.get('event_name', event_type_str)
    new_value = f"{days_until}d_ahead: {event_name}"

    return CatalystEventV2(
        ticker=ticker,
        nct_id=f"SEC8K_{event_type_str}_{ticker}_{event_date_str}",
        event_type=event_type,
        event_severity=severity,
        event_date=event_date_str,
        field_changed=f"sec_8k_{event_type_str.lower()}",
        prior_value=None,
        new_value=new_value,
        source=event.get("source", "SEC_8K_FILING"),
        confidence=confidence,
        disclosed_at=event.get('disclosed_at', as_of_date.isoformat()),
        event_date_end=event.get('event_date_end'),
        date_precision=event.get('date_precision', 'DAY'),
        tags=tuple(event.get('tags', [])),
    )


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


def fuzzy_dedup_events(
    events: List[CatalystEventV2],
    date_tolerance_days: int = 14,
) -> Tuple[List[CatalystEventV2], int]:
    """
    Fuzzy deduplication: merge events with same (ticker, event_type) and
    event_date within ±date_tolerance_days.

    When merging, keeps the higher-confidence event and unions tags.

    Returns:
        (deduped_events, count_merged)
    """
    if not events:
        return (events, 0)

    # Confidence ordering for comparison
    CONF_RANK = {ConfidenceLevel.HIGH: 3, ConfidenceLevel.MED: 2, ConfidenceLevel.LOW: 1}

    # Group by (ticker, event_type)
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[CatalystEventV2]] = defaultdict(list)
    for event in events:
        key = (event.ticker, event.event_type.value)
        groups[key].append(event)

    result = []
    total_merged = 0

    for key, group in groups.items():
        # Sort by event_date within group
        group.sort(key=lambda e: e.event_date or "9999-99-99")

        merged_group: List[CatalystEventV2] = []
        for event in group:
            if not event.event_date:
                merged_group.append(event)
                continue

            # Check if this event should merge with the last in merged_group
            if merged_group and merged_group[-1].event_date:
                try:
                    last_date = date.fromisoformat(merged_group[-1].event_date)
                    curr_date = date.fromisoformat(event.event_date)
                    delta = abs((curr_date - last_date).days)
                except ValueError:
                    delta = date_tolerance_days + 1  # Don't merge on parse failure

                if delta <= date_tolerance_days:
                    # Merge: keep higher confidence, union tags
                    existing = merged_group[-1]
                    existing_rank = CONF_RANK.get(existing.confidence, 0)
                    new_rank = CONF_RANK.get(event.confidence, 0)

                    if new_rank > existing_rank:
                        winner = event
                        loser = existing
                    else:
                        winner = existing
                        loser = event

                    # Union tags
                    merged_tags = tuple(sorted(set(winner.tags) | set(loser.tags)))

                    # Replace last event with merged version
                    merged_group[-1] = CatalystEventV2(
                        ticker=winner.ticker,
                        nct_id=winner.nct_id,
                        event_type=winner.event_type,
                        event_severity=winner.event_severity,
                        event_date=winner.event_date,
                        field_changed=winner.field_changed,
                        prior_value=winner.prior_value,
                        new_value=winner.new_value,
                        source=winner.source,
                        confidence=winner.confidence,
                        disclosed_at=winner.disclosed_at,
                        event_date_end=winner.event_date_end,
                        date_precision=winner.date_precision,
                        tags=merged_tags,
                    )
                    total_merged += 1
                    continue

            merged_group.append(event)

        result.extend(merged_group)

    return (result, total_merged)


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
    pit_mode: str = "strict",
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

    # Handle empty active_tickers gracefully
    if not active_tickers or len(active_tickers) == 0:
        logger.warning("Module 3: Empty active_tickers provided - returning empty results")
        return {
            "summaries": {},
            "summaries_legacy": {},
            "diagnostic_counts": {
                "events_detected_total": 0,
                "events_deduped": 0,
                "tickers_with_events": 0,
                "tickers_analyzed": 0,
                "tickers_with_severe_negative": 0,
            },
            "diagnostic_counts_legacy": {
                "events_detected": 0,
                "events_deduped": 0,
                "severe_negatives": 0,
                "tickers_with_events": 0,
                "tickers_analyzed": 0,
            },
            "as_of_date": as_of_date.isoformat(),
            "schema_version": SCHEMA_VERSION,
            "score_version": SCORE_VERSION,
        }

    # Initialize
    if config is None:
        config = Module3Config()

    if market_calendar is None:
        market_calendar = SimpleMarketCalendar()

    # Derive data_dir from trial_records_path for loading supplementary data
    data_dir = trial_records_path.parent

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

    # Load trial records with explicit error handling and schema validation
    logger.info(f"Loading trial records from {trial_records_path}")
    try:
        with open(trial_records_path) as f:
            trial_records_raw = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Trial records file not found: {trial_records_path}. "
            f"Ensure the file exists and path is correct."
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in trial records file {trial_records_path}: {e}. "
            f"File may be corrupted or malformed."
        )
    except PermissionError:
        raise PermissionError(
            f"Permission denied reading trial records: {trial_records_path}. "
            f"Check file permissions."
        )

    # Schema validation: fail fast on malformed data
    if not isinstance(trial_records_raw, list):
        raise ValueError(
            f"Trial records must be a JSON array, got {type(trial_records_raw).__name__}. "
            f"File: {trial_records_path}"
        )
    schema_errors = []
    for i, rec in enumerate(trial_records_raw):
        if not isinstance(rec, dict):
            schema_errors.append(f"Record {i}: expected dict, got {type(rec).__name__}")
            continue
        if "nct_id" not in rec:
            schema_errors.append(f"Record {i}: missing required field 'nct_id'")
        elif not isinstance(rec["nct_id"], str):
            schema_errors.append(f"Record {i}: 'nct_id' must be str, got {type(rec['nct_id']).__name__}")
        if "ticker" not in rec:
            schema_errors.append(f"Record {i}: missing required field 'ticker'")
        elif not isinstance(rec["ticker"], str):
            schema_errors.append(f"Record {i}: 'ticker' must be str, got {type(rec['ticker']).__name__}")
        if "status" not in rec and "overall_status" not in rec:
            schema_errors.append(f"Record {i}: missing required field 'status' (or 'overall_status')")
    if schema_errors:
        error_sample = schema_errors[:10]
        msg = (
            f"Trial records schema validation failed ({len(schema_errors)} errors). "
            f"First errors: {error_sample}"
        )
        if pit_mode == "strict":
            raise ValueError(msg)
        else:
            logger.error(f"SCHEMA_VALIDATION_DEGRADE: {msg}")
            logger.error(
                f"Continuing with {len(trial_records_raw)} records despite schema errors. "
                f"Fix trial_records.json to remove this warning."
            )
    else:
        logger.info(f"Trial records schema validated: {len(trial_records_raw)} records OK")

    # =========================================================================
    # STALENESS GATING: Check if trial_records is stale
    # =========================================================================
    staleness_result = check_trial_records_staleness(trial_records_raw, as_of_date)
    if staleness_result.is_stale:
        logger.warning(f"STALENESS WARNING: {staleness_result.recommendation}")
        logger.warning(f"  trial_records_date={staleness_result.trial_records_date}, "
                      f"as_of_date={as_of_date}, age={staleness_result.age_days} days")
    else:
        logger.info(f"Staleness check: {staleness_result.confidence_level} "
                   f"(age={staleness_result.age_days} days)")

    # =========================================================================
    # PIT VIOLATION GATE: Reject or degrade when data is from the future
    # =========================================================================
    # Direct date comparison — never key off confidence labels.
    pit_violation = (
        staleness_result.trial_records_date is not None
        and staleness_result.trial_records_date > as_of_date
    )
    pit_violation_info = None

    if pit_violation:
        if pit_mode == "strict":
            from catalyst_diagnostics import PITViolationError
            raise PITViolationError(
                source="trial_records",
                source_date=staleness_result.trial_records_date,
                as_of_date=as_of_date,
            )
        else:  # degrade
            logger.error(
                f"PIT_VIOLATION: trial_records dated {staleness_result.trial_records_date} "
                f"is after as_of_date {as_of_date}. "
                f"Skipping ALL trial_records-derived logic (diff + calendar + activity proxy). "
                f"Only corporate catalysts will be used."
            )
            pit_violation_info = {
                "source": "trial_records",
                "source_date": staleness_result.trial_records_date.isoformat(),
                "as_of_date": as_of_date.isoformat(),
                "age_days": staleness_result.age_days,
                "action": "corporate_only_fallback",
            }

    # =========================================================================
    # PIT VIOLATION DEGRADE: Skip ALL trial_records-derived logic
    # =========================================================================
    # When trial_records is from the future, ALL trial_records-derived signals
    # are contaminated: diff-based detection, calendar inference from trial dates,
    # and activity proxy. Only independently-sourced corporate catalysts survive.
    if pit_violation:
        logger.warning(
            "PIT_VIOLATION degrade: skipping ALL trial_records-derived logic "
            "(filter, canonical conversion, snapshots, diff detection, "
            "calendar inference, activity proxy). Only corporate catalysts will be used."
        )

    # Initialize variables that downstream code always expects
    canonical_records = []
    adapter_stats = {}
    current_snapshot = StateSnapshot(snapshot_date=as_of_date, records=[])
    prior_snapshot = None
    prior_snapshot_date = None
    delta_diagnostics = DeltaDiagnostics(prior_snapshot_missing=True)
    calendar_catalysts = []
    calendar_summary = {'total_catalysts': 0, 'tickers_with_catalysts': 0, 'by_window': {}}
    activity_proxy_by_ticker: Dict[str, Dict[str, Any]] = {}
    total_activity_120d = 0
    total_activity_30d = 0
    events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
    events_by_ticker_legacy: Dict[str, List[CatalystEvent]] = {}
    prior_events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
    total_events = 0
    total_deduped = 0
    historical_proximities = {}

    if not pit_violation:
        # Filter to active tickers
        trial_records = [r for r in trial_records_raw if r.get('ticker') in active_tickers]
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

        # Load prior state - must be strictly before as_of_date for valid delta comparison
        prior_snapshot = state_store.get_prior_snapshot(as_of_date)
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
    
        # CRITICAL: Verify snapshot dates for correctness
        current_snapshot_date = current_snapshot.snapshot_date
        logger.info(f"Snapshot dates: current={current_snapshot_date}, prior={prior_snapshot_date or 'None'}, as_of={as_of_date}")
    
        # Assertion 1: Current snapshot date must equal as_of_date
        if current_snapshot_date != as_of_date:
            raise ValueError(
                f"Snapshot date mismatch: current_snapshot_date={current_snapshot_date} != as_of_date={as_of_date}"
            )
    
        # Assertion 2: Prior snapshot date must be before current (if prior exists)
        if prior_snapshot_date is not None and prior_snapshot_date >= current_snapshot_date:
            raise ValueError(
                f"Snapshot date ordering violation: prior={prior_snapshot_date} >= current={current_snapshot_date}. "
                f"Prior snapshot should be strictly before current. Check state directory for stale data."
            )
    
        # =========================================================================
        # DELTA DIAGNOSTICS: Analyze what changed between snapshots
        # =========================================================================
        delta_diagnostics = compute_delta_diagnostics(current_snapshot, prior_snapshot)
        delta_diagnostics.log_summary()
    
        # =========================================================================
        # DETAILED SAMPLE COMPARISON: Check individual trials for changes
        # =========================================================================
        logger.info("=" * 60)
        logger.info("CATALYST DETECTION DIAGNOSTICS - SAMPLE COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Current snapshot: {current_snapshot.key_count} trials")
        logger.info(f"Prior snapshot: {prior_snapshot.key_count if prior_snapshot else 0} trials")
    
        if prior_snapshot and current_snapshot.records:
            # Build key sets for comparison using NCT ID ONLY (stable identifier)
            # Ticker association can change between CT.gov queries, but NCT ID is globally unique
            current_nct_ids = {r.nct_id for r in current_snapshot.records}
            prior_nct_ids = {r.nct_id for r in prior_snapshot.records}
            common_nct_ids = current_nct_ids & prior_nct_ids
            added_nct_ids = current_nct_ids - prior_nct_ids
            removed_nct_ids = prior_nct_ids - current_nct_ids
    
            # Also track (ticker, nct_id) tuples for detailed comparison
            current_keys = {(r.ticker, r.nct_id) for r in current_snapshot.records}
            prior_keys = {(r.ticker, r.nct_id) for r in prior_snapshot.records}
            common_keys = current_keys & prior_keys
            added_keys = current_keys - prior_keys
            removed_keys = prior_keys - current_keys
    
            logger.info(f"NCT ID overlap: {len(common_nct_ids)} common, {len(added_nct_ids)} added, {len(removed_nct_ids)} removed")
            logger.info(f"(ticker, nct_id) overlap: {len(common_keys)} common, {len(added_keys)} added, {len(removed_keys)} removed")
    
            # Calculate churn rate using NCT IDs (stable keys)
            total_current = len(current_nct_ids) if current_nct_ids else 1
            nct_churn_rate = (len(added_nct_ids) + len(removed_nct_ids)) / (2 * total_current)
    
            # Also calculate ticker association churn for diagnostics
            ticker_churn_rate = (len(added_keys) + len(removed_keys)) / (2 * len(current_keys)) if current_keys else 0
    
            if nct_churn_rate > 0.10:
                logger.warning(f"⚠️  HIGH NCT ID CHURN: {len(added_nct_ids)} added, {len(removed_nct_ids)} removed ({nct_churn_rate*100:.1f}% churn)")
                logger.warning("   This suggests actual trial population changed significantly.")
    
            if ticker_churn_rate > 0.30 and nct_churn_rate < 0.10:
                logger.info(f"ℹ️  Ticker association churn: {ticker_churn_rate*100:.1f}% (NCT churn only {nct_churn_rate*100:.1f}%)")
                logger.info("   Ticker-NCT associations changed but trials are stable - this is expected with CT.gov text search.")
    
            # HIGH CHURN GATE: Only trigger on NCT ID churn (actual trial changes)
            # Ticker association churn is expected and should not trigger fresh baseline
            if nct_churn_rate > 0.30:
                logger.error(f"CATALYST CHURN GATE TRIGGERED: {nct_churn_rate*100:.1f}% NCT churn exceeds 30% threshold")
                logger.error("   Forcing fresh baseline mode - all events will be detected as initial ingest")
                logger.error("   This prevents false negatives from dataset regeneration")
                # Force fresh baseline by clearing prior snapshot reference
                prior_snapshot = None
                prior_snapshot_date = None
                logger.warning("   Prior snapshot cleared - running in fresh baseline mode")
    
            # Sample up to 10 COMMON trials and check field changes
            # Only run if prior_snapshot is still valid (not cleared by churn gate)
            if prior_snapshot is not None:
                sample_keys = sorted(list(common_keys))[:10]
                changes_found = 0
    
                for ticker, nct_id in sample_keys:
                    current_record = current_snapshot.get_record(ticker, nct_id)
                    prior_record = prior_snapshot.get_record(ticker, nct_id)
                    if not current_record or not prior_record:
                        continue
    
                    changes = []
    
                    if current_record.overall_status != prior_record.overall_status:
                        changes.append(f"status: {prior_record.overall_status.name} → {current_record.overall_status.name}")
    
                    if current_record.primary_completion_date != prior_record.primary_completion_date:
                        changes.append(f"pcd: {prior_record.primary_completion_date} → {current_record.primary_completion_date}")
    
                    if current_record.last_update_posted != prior_record.last_update_posted:
                        changes.append(f"updated: {prior_record.last_update_posted} → {current_record.last_update_posted}")
    
                    if changes:
                        changes_found += 1
                        logger.info(f"  {nct_id}: {', '.join(changes)}")
    
                if changes_found == 0 and len(common_keys) > 0:
                    logger.warning("⚠️  NO FIELD CHANGES in common records - snapshots may have same underlying data")
                    logger.info(f"   Current snapshot date: {current_snapshot.snapshot_date}")
                    logger.info(f"   Prior snapshot date: {prior_snapshot.snapshot_date}")
    
        # Check update recency distribution
        cutoff_date = as_of_date - timedelta(days=7)
        recent_updates = sum(
            1 for r in current_snapshot.records
            if r.last_update_posted and r.last_update_posted >= cutoff_date
        )
    
        logger.info(f"Trials updated in past 7 days: {recent_updates} / {current_snapshot.key_count}")
    
        if recent_updates < 50:  # Expect ~1% weekly update rate for active trials
            logger.warning(f"⚠️  Very few recent updates ({recent_updates}) - trial_records.json may be stale")
            logger.warning(f"   This explains why 0 diff-based events are detected")
    
        logger.info("=" * 60)
    
        # =========================================================================
        # CALENDAR-BASED CATALYSTS: Forward-looking events from trial dates
        # =========================================================================
        logger.info("Detecting calendar-based catalysts...")
        calendar_catalysts = detect_calendar_catalysts(current_snapshot, as_of_date)

        # Readout window inference: PCD passed + no results → imminent readout
        readout_catalysts = detect_readout_window_catalysts(current_snapshot, as_of_date)
        calendar_catalysts.extend(readout_catalysts)
        if readout_catalysts:
            logger.info(f"Readout window catalysts: {len(readout_catalysts)} inferred events")

        calendar_summary = summarize_calendar_catalysts(calendar_catalysts)
        logger.info(f"Calendar catalysts: {calendar_summary['total_catalysts']} events "
                   f"across {calendar_summary['tickers_with_catalysts']} tickers")
        if calendar_summary['by_window']:
            logger.info(f"  By window: {calendar_summary['by_window']}")
    
        # =========================================================================
        # ACTIVITY PROXY: Historical data workaround based on last_update_posted
        # =========================================================================
        # This supplements diff-based detection when historical snapshots are limited.
        # It identifies trials with recent updates (even without knowing what changed).
        logger.info("Computing activity proxy scores (historical data workaround)...")
        activity_proxy_by_ticker: Dict[str, Dict[str, Any]] = {}
    
        # Group trials by ticker for activity proxy computation
        trials_by_ticker: Dict[str, List[CanonicalTrialRecord]] = {}
        for record in canonical_records:
            ticker = record.ticker
            if ticker not in trials_by_ticker:
                trials_by_ticker[ticker] = []
            trials_by_ticker[ticker].append(record)
    
        # Compute activity proxy scores per ticker
        total_activity_120d = 0
        total_activity_30d = 0
        for ticker in active_tickers:
            ticker_trials = trials_by_ticker.get(ticker, [])
            if ticker_trials:
                proxy_result = compute_activity_proxy_score(
                    ticker_trials,
                    as_of_date,
                    lookback_days=120,
                )
                activity_proxy_by_ticker[ticker] = proxy_result
                total_activity_120d += proxy_result['activity_count_120d']
                total_activity_30d += proxy_result['activity_count_30d']
    
        logger.info(f"Activity proxy: {total_activity_120d} trials updated in 120d, "
                   f"{total_activity_30d} in 30d across {len(activity_proxy_by_ticker)} tickers")
    
        # Detect events by comparing states
        logger.info("Detecting diff-based catalyst events...")
        events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
        events_by_ticker_legacy: Dict[str, List[CatalystEvent]] = {}
        prior_events_by_ticker_v2: Dict[str, List[CatalystEventV2]] = {}
        total_events = 0
        total_deduped = 0
    
        # Use cached NCT ID lookup for prior snapshot (stable key regardless of ticker association)
        # This ensures we find prior records even if ticker-NCT association changed between runs
        # The records_by_nct_id property is lazily computed and cached on first access
    
        for current_record in canonical_records:
            ticker = current_record.ticker
            nct_id = current_record.nct_id
    
            # Get prior record for this trial using NCT ID only (stable lookup via cached dict)
            prior_record = prior_snapshot.get_record_by_nct_id(nct_id) if prior_snapshot else None
    
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
    
        # =========================================================================
        # ZERO DIFF-BASED EVENTS DIAGNOSTIC: Explain why no diff events detected
        # =========================================================================
        # Note: Calendar-based events will still provide coverage for upcoming catalysts
        if total_events == 0:
            logger.info("=" * 70)
            logger.info("ZERO DIFF-BASED EVENTS - DIAGNOSTIC SUMMARY")
            logger.info("=" * 70)
    
            # Note calendar coverage as fallback
            if calendar_summary['total_catalysts'] > 0:
                logger.info(f"CALENDAR FALLBACK ACTIVE: {calendar_summary['total_catalysts']} calendar events "
                           f"across {calendar_summary['tickers_with_catalysts']} tickers will provide coverage")
    
            # Check data freshness
            if staleness_result.is_stale:
                logger.warning(f"DATA STALENESS: trial_records is {staleness_result.age_days} days old")
                logger.warning(f"  trial_records date: {staleness_result.trial_records_date}")
                logger.warning(f"  as_of_date:         {as_of_date}")
                logger.warning(f"  Recommendation:     {staleness_result.recommendation}")
    
            # Check delta diagnostics
            if delta_diagnostics.no_changes_detected:
                logger.info(f"SNAPSHOT COMPARISON: No field changes between snapshots")
                logger.info(f"  Current records: {delta_diagnostics.total_current_records}")
                logger.info(f"  Prior records:   {delta_diagnostics.total_prior_records}")
            elif delta_diagnostics.prior_snapshot_missing:
                logger.info(f"INITIAL RUN: No prior snapshot - diff detection disabled")
                logger.info(f"  Calendar-based events will provide catalyst coverage")
            else:
                logger.info(f"Delta shows changes but no events generated:")
                logger.info(f"  Records changed: {delta_diagnostics.records_changed_count}")
    
            logger.info("=" * 70)
    
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
    
        # =========================================================================
        # MERGE CALENDAR CATALYSTS INTO SCORING PIPELINE
        # =========================================================================
        # Convert calendar catalysts to CatalystEventV2 and merge with diff-based events
        # This ensures calendar-based forward-looking events are included in scoring
        logger.info("Merging calendar catalysts into scoring pipeline...")
        calendar_events_added = 0
        calendar_tickers_added = 0
    
        for calendar_catalyst in calendar_catalysts:
            ticker = calendar_catalyst.ticker
            v2_event = convert_calendar_catalyst_to_v2(calendar_catalyst, as_of_date=as_of_date)
    
            if ticker not in events_by_ticker_v2:
                events_by_ticker_v2[ticker] = []
                calendar_tickers_added += 1
    
            events_by_ticker_v2[ticker].append(v2_event)
            calendar_events_added += 1
    
        # Re-dedup and re-sort after merging calendar events
        if calendar_events_added > 0:
            calendar_deduped = 0
            for ticker in events_by_ticker_v2:
                events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                calendar_deduped += deduped
                events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])
    
            total_deduped += calendar_deduped
            total_events += calendar_events_added
    
            logger.info(f"Merged {calendar_events_added} calendar events across "
                       f"{calendar_summary['tickers_with_catalysts']} tickers "
                       f"(new tickers: {calendar_tickers_added}, deduped: {calendar_deduped})")

    # =========================================================================
    # MERGE CORPORATE CATALYSTS INTO SCORING PIPELINE
    # =========================================================================
    # Load and convert corporate catalyst events (PDUFA dates, earnings, conferences, etc.)
    corporate_catalysts = []
    corporate_summary = {'total_events': 0, 'tickers_with_events': 0, 'by_type': {}}

    if config.enable_corporate_catalysts:
        corporate_catalysts = load_corporate_catalysts(
            data_dir=data_dir,
            filename=config.corporate_catalysts_file,
            as_of_date=as_of_date,
        )
        corporate_summary = summarize_corporate_catalysts(corporate_catalysts)

        if corporate_catalysts:
            logger.info(f"Merging corporate catalysts into scoring pipeline...")
            corporate_events_added = 0
            corporate_tickers_added = 0

            for corp_event in corporate_catalysts:
                ticker = corp_event.get('ticker')
                if not ticker or ticker not in active_tickers:
                    continue

                v2_event = convert_corporate_catalyst_to_v2(corp_event, as_of_date)
                if v2_event is None:
                    continue

                if ticker not in events_by_ticker_v2:
                    events_by_ticker_v2[ticker] = []
                    corporate_tickers_added += 1

                events_by_ticker_v2[ticker].append(v2_event)
                corporate_events_added += 1

            # Re-dedup and re-sort after merging corporate events
            if corporate_events_added > 0:
                corporate_deduped = 0
                for ticker in events_by_ticker_v2:
                    events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                    corporate_deduped += deduped
                    events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])

                total_deduped += corporate_deduped
                total_events += corporate_events_added

                logger.info(f"Merged {corporate_events_added} corporate events across "
                           f"{corporate_summary['tickers_with_events']} tickers "
                           f"(new tickers: {corporate_tickers_added}, deduped: {corporate_deduped})")
                logger.info(f"  By type: {corporate_summary['by_type']}")
        else:
            logger.debug("No corporate catalysts to merge (file not found or empty)")

    # =========================================================================
    # MERGE FDA ADCOM EVENTS INTO SCORING PIPELINE
    # =========================================================================
    _adcom_mode = config.enable_fda_adcom
    if _adcom_mode == "off":
        logger.debug("FDA ADCOM source off (set enable_fda_adcom='cache_only' or 'live')")
    else:
        adcom_events = []
        adcom_cache_path = Path(config.fda_adcom_cache_dir) / f"adcom_calendar_{as_of_date.isoformat()}.json"

        if _adcom_mode == "cache_only":
            # Load from cache only — no network.
            # Fallback: if primary cache dir doesn't have the file,
            # check wake_robin_data_pipeline/cache/fda/ (where enrich writes).
            _resolved_adcom_path = adcom_cache_path
            if not adcom_cache_path.exists():
                _fallback_adcom_dir = data_dir.parent / "wake_robin_data_pipeline" / "cache" / "fda"
                _fallback_adcom_path = _fallback_adcom_dir / adcom_cache_path.name
                if _fallback_adcom_path.exists():
                    logger.warning(
                        f"FDA ADCOM: cache not in {adcom_cache_path.parent}, "
                        f"using fallback {_fallback_adcom_path}"
                    )
                    _resolved_adcom_path = _fallback_adcom_path

            if _resolved_adcom_path.exists():
                try:
                    with open(_resolved_adcom_path, 'r', encoding='utf-8') as f:
                        adcom_events = json.load(f)
                    logger.info(f"FDA ADCOM: loaded {len(adcom_events)} events from cache")
                except Exception as e:
                    logger.warning(f"FDA ADCOM cache read error: {e}")
            else:
                logger.debug(f"FDA ADCOM: no cache for {as_of_date} (run warm_caches.py to populate)")
        elif _adcom_mode == "live":
            try:
                from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
                    collect_fda_adcom_events,
                    build_product_ticker_map,
                )
                product_map = build_product_ticker_map(data_dir)
                adcom_events = collect_fda_adcom_events(
                    drug_to_ticker=product_map,
                    as_of_date=as_of_date,
                    cache_dir=config.fda_adcom_cache_dir,
                    universe_tickers=active_tickers,
                )
            except ImportError:
                logger.debug("FDA ADCOM collector not available")
            except Exception as e:
                logger.warning(f"FDA ADCOM live collection error: {e}")

        if adcom_events:
            adcom_added = 0
            adcom_tickers_added = 0
            adcom_touched: Set[str] = set()
            for adcom_event in adcom_events:
                ticker = adcom_event.get('ticker')
                if not ticker or ticker not in active_tickers:
                    continue
                v2_event = convert_fda_adcom_to_v2(adcom_event, as_of_date)
                if v2_event is None:
                    continue
                if ticker not in events_by_ticker_v2:
                    events_by_ticker_v2[ticker] = []
                    adcom_tickers_added += 1
                events_by_ticker_v2[ticker].append(v2_event)
                adcom_touched.add(ticker)
                adcom_added += 1

            if adcom_added > 0:
                adcom_deduped = 0
                for ticker in sorted(adcom_touched):
                    events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                    adcom_deduped += deduped
                    events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])
                total_deduped += adcom_deduped
                total_events += adcom_added
                logger.info(f"Merged {adcom_added} FDA ADCOM events "
                           f"(new tickers: {adcom_tickers_added}, deduped: {adcom_deduped})")

    # =========================================================================
    # MERGE SEC 8-K CATALYST EVENTS INTO SCORING PIPELINE
    # =========================================================================
    _sec_mode = config.enable_sec_8k_catalysts
    if _sec_mode == "off":
        logger.debug("SEC 8-K source off (set enable_sec_8k_catalysts='cache_only' or 'live')")
    else:
        sec_8k_events = []
        # Use versioned cache path so pattern changes auto-invalidate
        try:
            from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                _versioned_cache_path as _sec_versioned_path,
            )
            sec_cache_path = _sec_versioned_path(Path(config.sec_8k_cache_dir), as_of_date)
        except ImportError:
            # Fallback if collector not available (cache_only mode without collector installed)
            sec_cache_path = Path(config.sec_8k_cache_dir) / f"8k_catalysts_{as_of_date.isoformat()}.json"

        if _sec_mode == "cache_only":
            # Fallback: if primary cache dir doesn't have the versioned file,
            # check wake_robin_data_pipeline/cache/ (where enrich_archive_inputs writes).
            _FALLBACK_8K_DIR = data_dir.parent / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts"
            _resolved_8k_path = sec_cache_path
            if not sec_cache_path.exists() and _FALLBACK_8K_DIR.exists():
                try:
                    fallback_path = _sec_versioned_path(_FALLBACK_8K_DIR, as_of_date)
                except NameError:
                    fallback_path = _FALLBACK_8K_DIR / sec_cache_path.name
                if fallback_path.exists():
                    logger.warning(
                        f"SEC 8-K: versioned cache not in {sec_cache_path.parent}, "
                        f"using fallback {fallback_path}"
                    )
                    _resolved_8k_path = fallback_path

            if _resolved_8k_path.exists():
                try:
                    with open(_resolved_8k_path, 'r', encoding='utf-8') as f:
                        sec_8k_events = json.load(f)
                    logger.info(f"SEC 8-K: loaded {len(sec_8k_events)} events from cache")
                except Exception as e:
                    logger.warning(f"SEC 8-K cache read error: {e}")
            else:
                logger.warning(
                    f"SEC 8-K: no versioned cache for {as_of_date} "
                    f"(expected {sec_cache_path.name} in {sec_cache_path.parent} "
                    f"or {_FALLBACK_8K_DIR}; run enrich_archive_inputs.py --sec-8k-mode=live)"
                )
        elif _sec_mode == "live":
            try:
                from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                    collect_8k_timing_events,
                )
                universe_path = data_dir / "universe.json"
                if universe_path.exists():
                    with open(universe_path, 'r', encoding='utf-8') as f:
                        universe_data = json.load(f)
                    universe_entries = universe_data if isinstance(universe_data, list) else universe_data.get('tickers', [])
                else:
                    universe_entries = []

                sec_8k_events = collect_8k_timing_events(
                    universe=universe_entries,
                    as_of_date=as_of_date,
                    cache_dir=config.sec_8k_cache_dir,
                )
            except ImportError:
                logger.debug("SEC 8-K catalyst collector not available")
            except Exception as e:
                logger.warning(f"SEC 8-K live collection error: {e}")

        if sec_8k_events:
            sec_added = 0
            sec_tickers_added = 0
            sec_touched: Set[str] = set()
            for sec_event in sec_8k_events:
                ticker = sec_event.get('ticker')
                if not ticker or ticker not in active_tickers:
                    continue
                v2_event = convert_sec_8k_to_v2(sec_event, as_of_date)
                if v2_event is None:
                    continue
                if ticker not in events_by_ticker_v2:
                    events_by_ticker_v2[ticker] = []
                    sec_tickers_added += 1
                events_by_ticker_v2[ticker].append(v2_event)
                sec_touched.add(ticker)
                sec_added += 1

            if sec_added > 0:
                sec_deduped = 0
                for ticker in sorted(sec_touched):
                    events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                    sec_deduped += deduped
                    events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])
                total_deduped += sec_deduped
                total_events += sec_added
                logger.info(f"Merged {sec_added} SEC 8-K catalyst events "
                           f"(new tickers: {sec_tickers_added}, deduped: {sec_deduped})")

    # =========================================================================
    # MERGE SEC MULTI-FORM (10-Q, 10-K, 6-K) CATALYST EVENTS
    # =========================================================================
    _multi_form_mode = config.enable_sec_multi_form
    if _multi_form_mode == "off":
        logger.debug("SEC multi-form source off (set enable_sec_multi_form='cache_only' or 'live')")
    else:
        mf_events = []

        # Use versioned cache path (separate from 8-K)
        try:
            from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                _multi_form_cache_path as _mf_cache_path,
            )
            mf_cache = _mf_cache_path(Path(config.sec_8k_cache_dir), as_of_date)
        except ImportError:
            mf_cache = Path(config.sec_8k_cache_dir) / f"sec_filings_{as_of_date.isoformat()}.json"

        if _multi_form_mode == "cache_only":
            _resolved_mf_path = mf_cache
            if not mf_cache.exists():
                _fallback_mf_dir = data_dir.parent / "wake_robin_data_pipeline" / "cache" / "sec" / "8k_catalysts"
                try:
                    fallback_mf = _mf_cache_path(_fallback_mf_dir, as_of_date)
                except NameError:
                    fallback_mf = _fallback_mf_dir / mf_cache.name
                if fallback_mf.exists():
                    logger.warning(
                        f"SEC multi-form: cache not in {mf_cache.parent}, "
                        f"using fallback {fallback_mf}"
                    )
                    _resolved_mf_path = fallback_mf

            if _resolved_mf_path.exists():
                try:
                    with open(_resolved_mf_path, 'r', encoding='utf-8') as f:
                        mf_events = json.load(f)
                    logger.info(f"SEC multi-form: loaded {len(mf_events)} events from cache")
                except Exception as e:
                    logger.warning(f"SEC multi-form cache read error: {e}")
            else:
                logger.warning(
                    f"SEC multi-form: no cache for {as_of_date} "
                    f"(run enrich_archive_inputs.py --sec-multi-form-mode=live)"
                )

        elif _multi_form_mode == "live":
            try:
                from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
                    collect_sec_filing_events,
                )
                universe_path = data_dir / "universe.json"
                if universe_path.exists():
                    with open(universe_path, 'r', encoding='utf-8') as f:
                        universe_data = json.load(f)
                    universe_entries = universe_data if isinstance(universe_data, list) else universe_data.get('tickers', [])
                else:
                    universe_entries = []

                mf_events = collect_sec_filing_events(
                    universe=universe_entries,
                    as_of_date=as_of_date,
                    cache_dir=config.sec_8k_cache_dir,
                )
            except ImportError:
                logger.debug("SEC multi-form collector not available")
            except Exception as e:
                logger.warning(f"SEC multi-form live collection error: {e}")

        if mf_events:
            # Hard gate: only MED/HIGH confidence + precise dates (no HALF_YEAR/YEAR)
            _MF_ALLOWED_CONF = {ConfidenceLevel.MED, ConfidenceLevel.HIGH}
            _MF_ALLOWED_PREC = {"DAY", "WEEK", "MONTH", "QUARTER"}
            mf_added = 0
            mf_gated = 0
            mf_tickers_added = 0
            mf_touched: Set[str] = set()
            for mf_event in mf_events:
                ticker = mf_event.get('ticker')
                if not ticker or ticker not in active_tickers:
                    continue
                v2_event = convert_sec_8k_to_v2(mf_event, as_of_date)
                if v2_event is None:
                    continue
                # Gate: reject LOW confidence or imprecise dates
                if v2_event.confidence not in _MF_ALLOWED_CONF or v2_event.date_precision not in _MF_ALLOWED_PREC:
                    mf_gated += 1
                    continue
                if ticker not in events_by_ticker_v2:
                    events_by_ticker_v2[ticker] = []
                    mf_tickers_added += 1
                events_by_ticker_v2[ticker].append(v2_event)
                mf_touched.add(ticker)
                mf_added += 1

            if mf_added > 0:
                mf_deduped = 0
                for ticker in sorted(mf_touched):
                    events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                    mf_deduped += deduped
                    events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])
                total_deduped += mf_deduped
                total_events += mf_added
                logger.info(f"Merged {mf_added} SEC multi-form catalyst events "
                           f"(new tickers: {mf_tickers_added}, deduped: {mf_deduped}, "
                           f"gated: {mf_gated})")

    # =========================================================================
    # MERGE FDA REGULATORY NOTICES INTO SCORING PIPELINE
    # =========================================================================
    _fda_reg_mode = config.enable_fda_regulatory
    if _fda_reg_mode == "off":
        logger.debug("FDA regulatory source off (set enable_fda_regulatory='cache_only' or 'live')")
    else:
        fda_reg_events = []
        fda_reg_cache_path = Path(config.fda_adcom_cache_dir) / f"fda_regulatory_{as_of_date.isoformat()}.json"

        if _fda_reg_mode == "cache_only":
            _resolved_reg_path = fda_reg_cache_path
            if not fda_reg_cache_path.exists():
                _fallback_reg_dir = data_dir.parent / "wake_robin_data_pipeline" / "cache" / "fda"
                _fallback_reg_path = _fallback_reg_dir / fda_reg_cache_path.name
                if _fallback_reg_path.exists():
                    logger.warning(
                        f"FDA regulatory: cache not in {fda_reg_cache_path.parent}, "
                        f"using fallback {_fallback_reg_path}"
                    )
                    _resolved_reg_path = _fallback_reg_path

            if _resolved_reg_path.exists():
                try:
                    with open(_resolved_reg_path, 'r', encoding='utf-8') as f:
                        fda_reg_events = json.load(f)
                    logger.info(f"FDA regulatory: loaded {len(fda_reg_events)} events from cache")
                except Exception as e:
                    logger.warning(f"FDA regulatory cache read error: {e}")
            else:
                logger.debug(f"FDA regulatory: no cache for {as_of_date}")

        elif _fda_reg_mode == "live":
            try:
                from wake_robin_data_pipeline.collectors.fda_adcom_collector import (
                    collect_fda_regulatory_notices,
                    build_product_ticker_map,
                )
                product_map = build_product_ticker_map(data_dir)
                fda_reg_events = collect_fda_regulatory_notices(
                    product_ticker_map=product_map,
                    as_of_date=as_of_date,
                    cache_dir=config.fda_adcom_cache_dir,
                )
            except ImportError:
                logger.debug("FDA regulatory collector not available")
            except Exception as e:
                logger.warning(f"FDA regulatory live collection error: {e}")

        if fda_reg_events:
            reg_added = 0
            reg_tickers_added = 0
            reg_touched: Set[str] = set()
            for reg_event in fda_reg_events:
                ticker = reg_event.get('ticker')
                if not ticker or ticker not in active_tickers:
                    continue
                v2_event = convert_fda_adcom_to_v2(reg_event, as_of_date)
                if v2_event is None:
                    continue
                if ticker not in events_by_ticker_v2:
                    events_by_ticker_v2[ticker] = []
                    reg_tickers_added += 1
                events_by_ticker_v2[ticker].append(v2_event)
                reg_touched.add(ticker)
                reg_added += 1

            if reg_added > 0:
                reg_deduped = 0
                for ticker in sorted(reg_touched):
                    events_by_ticker_v2[ticker], deduped = dedup_events(events_by_ticker_v2[ticker])
                    reg_deduped += deduped
                    events_by_ticker_v2[ticker] = sort_events_v2(events_by_ticker_v2[ticker])
                total_deduped += reg_deduped
                total_events += reg_added
                logger.info(f"Merged {reg_added} FDA regulatory events "
                           f"(new tickers: {reg_tickers_added}, deduped: {reg_deduped})")

    # Update total event count for combined diff + calendar + corporate events
    combined_tickers_with_events = len([t for t in events_by_ticker_v2 if events_by_ticker_v2[t]])
    logger.info(f"Total events for scoring: {total_events} across {combined_tickers_with_events} tickers")

    # =========================================================================
    # FUZZY DEDUP: Cross-source deduplication before scoring
    # =========================================================================
    total_fuzzy_merged = 0
    for ticker in events_by_ticker_v2:
        events_by_ticker_v2[ticker], merged = fuzzy_dedup_events(events_by_ticker_v2[ticker])
        total_fuzzy_merged += merged
    if total_fuzzy_merged > 0:
        logger.info(f"Fuzzy dedup merged {total_fuzzy_merged} near-duplicate events")

    # Compute catalyst source mix from finalized events (before scoring)
    _src_counts: Dict[str, int] = {}
    _conf_counts: Dict[str, int] = {}
    _prec_counts: Dict[str, int] = {}
    _tickers_with_events: Set[str] = set()
    _total_events = 0
    for _ticker, _evts in events_by_ticker_v2.items():
        for _ev in _evts:
            _total_events += 1
            _tickers_with_events.add(_ticker)
            _src_counts[_ev.source] = _src_counts.get(_ev.source, 0) + 1
            _conf_counts[_ev.confidence.value] = _conf_counts.get(_ev.confidence.value, 0) + 1
            _prec_counts[getattr(_ev, 'date_precision', 'UNKNOWN')] = _prec_counts.get(
                getattr(_ev, 'date_precision', 'UNKNOWN'), 0) + 1
    catalyst_source_mix = {
        "total_events": _total_events,
        "unique_tickers_with_events": len(_tickers_with_events),
        "by_source": dict(sorted(_src_counts.items())),
        "by_confidence": dict(sorted(_conf_counts.items())),
        "by_date_precision": dict(sorted(_prec_counts.items())),
    }

    # =========================================================================
    # BUILD EVENT LEDGER for PIT-safe nearest-catalyst query + traceability
    # =========================================================================
    _event_ledger = None
    try:
        from event_ledger import build_event_ledger, LedgerConfig
        if config.ctgov_cache_dir is False:
            _ctgov_dir = False  # sentinel: disable ctgov cache in LedgerConfig
        elif config.ctgov_cache_dir:
            _ctgov_dir = Path(config.ctgov_cache_dir)
        else:
            _auto = Path(__file__).parent / "cache" / "ctgov"
            _ctgov_dir = _auto if _auto.exists() else None
        _ledger_config = LedgerConfig(
            ctgov_cache_dir=_ctgov_dir,
            sec_cache_dir=Path(config.sec_8k_cache_dir),
            fda_cache_dir=Path(config.fda_adcom_cache_dir),
            data_dir=data_dir,
            strict_ctgov=(pit_mode == "strict"),
        )
        _event_ledger = build_event_ledger(as_of_date, _ledger_config)
        logger.info(f"Event ledger built: {len(_event_ledger)} entries")
    except FileNotFoundError:
        if pit_mode == "strict":
            raise
        logger.warning("Event ledger build failed (missing cache) — nearest-catalyst will use legacy path")
    except Exception as e:
        logger.warning(f"Event ledger build failed: {e} — nearest-catalyst will use legacy path")

    # Score events using new scoring system
    logger.info("Scoring events with vNext scorer...")
    summaries_v2, diagnostics = score_catalyst_events(
        events_by_ticker_v2,
        list(active_tickers),
        as_of_date,
        prior_events_by_ticker=prior_events_by_ticker_v2 if prior_events_by_ticker_v2 else None,
        historical_proximities_by_ticker=historical_proximities if historical_proximities else None,
        event_ledger=_event_ledger,
    )

    # Update diagnostics with dedup count
    diagnostics.events_deduped = total_deduped
    diagnostics.events_fuzzy_merged = total_fuzzy_merged

    # =========================================================================
    # MERGE ACTIVITY PROXY DATA INTO SUMMARIES
    # =========================================================================
    for ticker in summaries_v2:
        if ticker in activity_proxy_by_ticker:
            proxy_data = activity_proxy_by_ticker[ticker]
            summary = summaries_v2[ticker]
            summary.activity_proxy_score = Decimal(str(proxy_data['activity_proxy_score']))
            summary.activity_proxy_count_120d = proxy_data['activity_count_120d']
            summary.activity_proxy_count_30d = proxy_data['activity_count_30d']

    # =========================================================================
    # TIME-DECAY SCORING: Multi-window analysis
    # =========================================================================
    time_decay_diagnostics = {}
    if config.enable_time_decay:
        logger.info("Computing time-decay scores across multiple windows...")
        logger.info(f"  Windows: {[w.name for w in config.time_decay_config.windows]}")

        # Score all tickers with time-decay
        time_decay_results, time_decay_diagnostics = score_all_tickers_with_time_decay(
            events_by_ticker_v2,
            list(active_tickers),
            as_of_date,
            config.time_decay_config,
        )

        # Merge time-decay results into summaries
        for ticker, td_result in time_decay_results.items():
            if ticker in summaries_v2:
                summary = summaries_v2[ticker]
                summary.time_decay_score = td_result.final_score
                summary.time_decay_contributing_window = td_result.contributing_window
                summary.time_decay_cluster_detected = td_result.cluster_detected
                summary.time_decay_cluster_bonus_applied = td_result.cluster_bonus_applied
                summary.time_decay_windows_with_events = td_result.windows_with_events
                summary.time_decay_window_scores = {
                    ws.window_name: str(ws.weighted_score)
                    for ws in td_result.window_scores
                }

        logger.info(f"Time-decay scoring complete: {time_decay_diagnostics.get('tickers_with_cluster', 0)} "
                   f"tickers with sustained catalyst clusters")
    else:
        logger.info("Time-decay scoring disabled")

    # =========================================================================
    # PIT VIOLATION DEGRADE: Downgrade scores and confidence
    # =========================================================================
    if pit_violation:
        pit_downgrade_factor = Decimal("0.50")
        for ticker, summary in summaries_v2.items():
            summary.score_override *= pit_downgrade_factor
            summary.score_blended *= pit_downgrade_factor
            summary.catalyst_confidence = ConfidenceLevel.LOW
        logger.warning(
            "PIT_VIOLATION degrade: catalyst scores halved and confidence "
            "downgraded to LOW (corporate-only mode)"
        )

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
            staleness_result=staleness_result,
            delta_diagnostics=delta_diagnostics,
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

    # Compute activity proxy summary
    activity_proxy_summary = {
        "total_activity_120d": total_activity_120d,
        "total_activity_30d": total_activity_30d,
        "tickers_with_activity": len([t for t, p in activity_proxy_by_ticker.items() if p['activity_count_120d'] > 0]),
    }

    output = {
        "summaries": summaries_v2,
        "summaries_legacy": summaries_legacy,  # DEPRECATED - use summaries instead
        "diagnostic_counts": diagnostics.to_dict(),
        "diagnostic_counts_legacy": diagnostic_counts_legacy,  # DEPRECATED
        "as_of_date": as_of_date.isoformat(),
        "schema_version": config.schema_version,
        "score_version": config.score_version,
        # New in v2.1: Enhanced diagnostics
        "delta_diagnostics": delta_diagnostics.to_dict(),
        "staleness": staleness_result.to_dict(),
        "calendar_catalysts": calendar_summary,
        # New in v2.4: Corporate catalysts (PDUFA, earnings, conferences, etc.)
        "corporate_catalysts_enabled": config.enable_corporate_catalysts,
        "corporate_catalysts": corporate_summary,
        # New in v2.2: Time-decay scoring
        "time_decay_enabled": config.enable_time_decay,
        "time_decay_diagnostics": time_decay_diagnostics,
        # New in v2.3: Activity proxy (historical data workaround)
        "activity_proxy_summary": activity_proxy_summary,
        # New: PIT violation tracking
        "pit_violation": pit_violation_info,
        "catalyst_mode": "corporate_only_due_to_pit_violation" if pit_violation else "full",
        # New: source mix for ablation diagnostics
        "catalyst_source_mix": catalyst_source_mix,
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
    staleness_result: Optional['StalenessResult'] = None,
    delta_diagnostics: Optional['DeltaDiagnostics'] = None,
) -> str:
    """
    Write vNext catalyst events output with deterministic JSON.

    Returns:
        SHA256 hash of output for verification
    """
    # Sort tickers for determinism
    sorted_tickers = sorted(summaries.keys())

    # Build data freshness metadata
    data_freshness = {}
    if staleness_result:
        data_freshness = {
            "is_stale": staleness_result.is_stale,
            "age_days": staleness_result.age_days,
            "trial_records_date": staleness_result.trial_records_date.isoformat() if staleness_result.trial_records_date else None,
            "confidence_level": staleness_result.confidence_level,
            "recommendation": staleness_result.recommendation,
        }

    # Build delta summary
    delta_summary = {}
    if delta_diagnostics:
        delta_summary = {
            "records_changed_count": delta_diagnostics.records_changed_count,
            "records_added_count": delta_diagnostics.records_added_count,
            "records_removed_count": delta_diagnostics.records_removed_count,
            "no_changes_detected": delta_diagnostics.no_changes_detected,
            "prior_snapshot_missing": delta_diagnostics.prior_snapshot_missing,
            "fields_changed_histogram": delta_diagnostics.fields_changed_histogram,
        }

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
            "data_freshness": data_freshness,
            "delta_summary": delta_summary,
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

def main() -> None:
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
