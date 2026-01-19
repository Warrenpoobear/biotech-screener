#!/usr/bin/env python3
"""
catalyst_diagnostics.py - Delta Diagnostics and Staleness Detection for Module 3

Provides:
1. Delta diagnostics: records_changed_count, fields_changed_histogram, sample diffs
2. Staleness gating: trial_records age validation and confidence degradation
3. Explainability helpers for event rule tracing
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import json
import logging
from pathlib import Path

from ctgov_adapter import CanonicalTrialRecord, CTGovStatus
from state_management import StateSnapshot

logger = logging.getLogger(__name__)


# ============================================================================
# DELTA DIAGNOSTICS
# ============================================================================

@dataclass
class FieldDiff:
    """Single field difference between two records"""
    ticker: str
    nct_id: str
    field_name: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'nct_id': self.nct_id,
            'field': self.field_name,
            'old': str(self.old_value) if self.old_value is not None else None,
            'new': str(self.new_value) if self.new_value is not None else None,
        }


@dataclass
class DeltaDiagnostics:
    """Comprehensive delta analysis between snapshots"""
    records_changed_count: int = 0
    records_added_count: int = 0
    records_removed_count: int = 0
    total_current_records: int = 0
    total_prior_records: int = 0

    # Field-level histogram
    fields_changed_histogram: Dict[str, int] = field(default_factory=dict)

    # Sample diffs (top 5)
    sample_diffs: List[FieldDiff] = field(default_factory=list)

    # Status-specific counters
    status_changes: Dict[str, int] = field(default_factory=dict)
    unknown_status_count: int = 0

    # Summary flags
    no_changes_detected: bool = False
    prior_snapshot_missing: bool = False

    def to_dict(self) -> dict:
        return {
            'records_changed_count': self.records_changed_count,
            'records_added_count': self.records_added_count,
            'records_removed_count': self.records_removed_count,
            'total_current_records': self.total_current_records,
            'total_prior_records': self.total_prior_records,
            'fields_changed_histogram': dict(sorted(
                self.fields_changed_histogram.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'sample_diffs': [d.to_dict() for d in self.sample_diffs[:5]],
            'status_changes': dict(sorted(
                self.status_changes.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'unknown_status_count': self.unknown_status_count,
            'no_changes_detected': self.no_changes_detected,
            'prior_snapshot_missing': self.prior_snapshot_missing,
        }

    def log_summary(self):
        """Log diagnostic summary"""
        if self.prior_snapshot_missing:
            logger.info("Delta diagnostics: No prior snapshot (initial run)")
            return

        if self.no_changes_detected:
            logger.warning(
                "Delta diagnostics: NO SNAPSHOT DIFFS DETECTED. "
                "Either no CT.gov changes between snapshots OR input data is stale."
            )
            logger.warning(
                f"  Current records: {self.total_current_records}, "
                f"Prior records: {self.total_prior_records}"
            )
            return

        logger.info(f"Delta diagnostics: {self.records_changed_count} records changed")
        logger.info(f"  Added: {self.records_added_count}, Removed: {self.records_removed_count}")

        if self.fields_changed_histogram:
            top_fields = list(self.fields_changed_histogram.items())[:5]
            logger.info(f"  Top changed fields: {dict(top_fields)}")

        if self.sample_diffs:
            logger.info("  Sample diffs:")
            for diff in self.sample_diffs[:3]:
                logger.info(f"    {diff.ticker}/{diff.nct_id}: {diff.field_name} "
                           f"'{diff.old_value}' → '{diff.new_value}'")

        if self.unknown_status_count > 0:
            logger.warning(f"  Unknown statuses mapped: {self.unknown_status_count}")


def compute_delta_diagnostics(
    current_snapshot: StateSnapshot,
    prior_snapshot: Optional[StateSnapshot],
) -> DeltaDiagnostics:
    """
    Compute comprehensive delta diagnostics between snapshots.

    Args:
        current_snapshot: Current trial state snapshot
        prior_snapshot: Prior snapshot (None if initial run)

    Returns:
        DeltaDiagnostics with full analysis
    """
    diag = DeltaDiagnostics()
    diag.total_current_records = current_snapshot.key_count

    if prior_snapshot is None:
        diag.prior_snapshot_missing = True
        return diag

    diag.total_prior_records = prior_snapshot.key_count

    # Build lookup maps
    current_by_key = {(r.ticker, r.nct_id): r for r in current_snapshot.records}
    prior_by_key = {(r.ticker, r.nct_id): r for r in prior_snapshot.records}

    current_keys = set(current_by_key.keys())
    prior_keys = set(prior_by_key.keys())

    # Added and removed
    added_keys = current_keys - prior_keys
    removed_keys = prior_keys - current_keys
    common_keys = current_keys & prior_keys

    diag.records_added_count = len(added_keys)
    diag.records_removed_count = len(removed_keys)

    # Analyze common records for changes
    fields_to_compare = [
        'overall_status',
        'primary_completion_date',
        'primary_completion_type',
        'completion_date',
        'completion_type',
        'last_update_posted',
        'results_first_posted',
    ]

    field_changes = Counter()
    status_transitions = Counter()
    all_diffs: List[FieldDiff] = []
    changed_record_keys = set()

    for key in sorted(common_keys):  # Sorted for determinism
        current = current_by_key[key]
        prior = prior_by_key[key]

        for field_name in fields_to_compare:
            current_val = getattr(current, field_name)
            prior_val = getattr(prior, field_name)

            if current_val != prior_val:
                changed_record_keys.add(key)
                field_changes[field_name] += 1

                # Track status transitions
                if field_name == 'overall_status':
                    transition = f"{prior_val.name}→{current_val.name}"
                    status_transitions[transition] += 1
                    if current_val == CTGovStatus.UNKNOWN:
                        diag.unknown_status_count += 1

                # Collect sample diff
                if len(all_diffs) < 20:  # Collect more than we show
                    all_diffs.append(FieldDiff(
                        ticker=current.ticker,
                        nct_id=current.nct_id,
                        field_name=field_name,
                        old_value=prior_val.name if isinstance(prior_val, CTGovStatus) else prior_val,
                        new_value=current_val.name if isinstance(current_val, CTGovStatus) else current_val,
                    ))

    diag.records_changed_count = len(changed_record_keys)
    diag.fields_changed_histogram = dict(field_changes)
    diag.status_changes = dict(status_transitions)
    diag.sample_diffs = all_diffs[:5]  # Keep top 5

    # Set no-changes flag
    diag.no_changes_detected = (
        diag.records_changed_count == 0 and
        diag.records_added_count == 0 and
        diag.records_removed_count == 0
    )

    return diag


# ============================================================================
# STALENESS GATING
# ============================================================================

@dataclass
class StalenessResult:
    """Result of staleness check"""
    is_stale: bool
    age_days: int
    trial_records_date: Optional[date]
    as_of_date: date
    confidence_level: str  # 'HIGH', 'MEDIUM', 'LOW', 'DEGRADED'
    recommendation: str

    def to_dict(self) -> dict:
        return {
            'is_stale': self.is_stale,
            'age_days': self.age_days,
            'trial_records_date': self.trial_records_date.isoformat() if self.trial_records_date else None,
            'as_of_date': self.as_of_date.isoformat(),
            'confidence_level': self.confidence_level,
            'recommendation': self.recommendation,
        }


def check_trial_records_staleness(
    trial_records: List[Dict[str, Any]],
    as_of_date: date,
    stale_threshold_days: int = 5,
    degraded_threshold_days: int = 3,
) -> StalenessResult:
    """
    Check if trial_records.json is stale relative to as_of_date.

    Infers data freshness from:
    1. Explicit 'provenance.as_of_date' or 'metadata.data_date' fields
    2. Max 'last_update_posted' across all records

    Args:
        trial_records: Raw trial records list
        as_of_date: Analysis date
        stale_threshold_days: Days after which data is considered stale (default 5)
        degraded_threshold_days: Days after which confidence is degraded (default 3)

    Returns:
        StalenessResult with staleness assessment
    """
    # Try to extract data date from various sources
    data_date = None

    # Check for explicit provenance
    if trial_records and isinstance(trial_records[0], dict):
        # Some formats wrap records with metadata
        first = trial_records[0]
        if '_metadata' in first:
            meta_date = first['_metadata'].get('data_date') or first['_metadata'].get('as_of_date')
            if meta_date:
                try:
                    data_date = date.fromisoformat(str(meta_date)[:10])
                except ValueError:
                    pass

    # Fall back to max last_update_posted
    if data_date is None:
        max_update = None
        for record in trial_records:
            lup = record.get('last_update_posted')
            if lup:
                try:
                    record_date = date.fromisoformat(str(lup)[:10])
                    if max_update is None or record_date > max_update:
                        max_update = record_date
                except ValueError:
                    continue
        data_date = max_update

    # Compute age
    if data_date is None:
        # Cannot determine - assume current but flag
        return StalenessResult(
            is_stale=False,
            age_days=0,
            trial_records_date=None,
            as_of_date=as_of_date,
            confidence_level='MEDIUM',
            recommendation='Cannot determine trial_records freshness. Add provenance metadata.',
        )

    age_days = (as_of_date - data_date).days

    # Validate: data should not be from the future
    if age_days < 0:
        return StalenessResult(
            is_stale=True,
            age_days=age_days,
            trial_records_date=data_date,
            as_of_date=as_of_date,
            confidence_level='DEGRADED',
            recommendation=f'CRITICAL: trial_records dated {data_date} is AFTER as_of_date {as_of_date}. '
                          f'This indicates lookahead bias or incorrect as_of_date.',
        )

    # Assess staleness
    if age_days > stale_threshold_days:
        return StalenessResult(
            is_stale=True,
            age_days=age_days,
            trial_records_date=data_date,
            as_of_date=as_of_date,
            confidence_level='LOW',
            recommendation=f'Data is {age_days} days old (threshold: {stale_threshold_days}). '
                          f'Catalyst detection may miss recent events. Refresh trial_records.json.',
        )
    elif age_days > degraded_threshold_days:
        return StalenessResult(
            is_stale=False,
            age_days=age_days,
            trial_records_date=data_date,
            as_of_date=as_of_date,
            confidence_level='MEDIUM',
            recommendation=f'Data is {age_days} days old. Results are usable but may miss recent events.',
        )
    else:
        return StalenessResult(
            is_stale=False,
            age_days=age_days,
            trial_records_date=data_date,
            as_of_date=as_of_date,
            confidence_level='HIGH',
            recommendation='Data is fresh.',
        )


# ============================================================================
# EVENT RULE REGISTRY
# ============================================================================

class EventRuleID:
    """Event rule identifiers for explainability"""
    # Diff-based events
    M3_DIFF_STATUS_SEVERE_NEG = "M3_DIFF_STATUS_SEVERE_NEG"
    M3_DIFF_STATUS_DOWNGRADE = "M3_DIFF_STATUS_DOWNGRADE"
    M3_DIFF_STATUS_UPGRADE = "M3_DIFF_STATUS_UPGRADE"
    M3_DIFF_DATE_PUSH = "M3_DIFF_DATE_PUSH"
    M3_DIFF_DATE_PULL = "M3_DIFF_DATE_PULL"
    M3_DIFF_DATE_CONFIRMED = "M3_DIFF_DATE_CONFIRMED"
    M3_DIFF_RESULTS_POSTED = "M3_DIFF_RESULTS_POSTED"

    # Calendar-based events
    M3_CAL_PCD_30D = "M3_CAL_PCD_30D"
    M3_CAL_PCD_60D = "M3_CAL_PCD_60D"
    M3_CAL_PCD_90D = "M3_CAL_PCD_90D"
    M3_CAL_SCD_30D = "M3_CAL_SCD_30D"
    M3_CAL_SCD_60D = "M3_CAL_SCD_60D"
    M3_CAL_SCD_90D = "M3_CAL_SCD_90D"
    M3_CAL_RESULTS_DUE = "M3_CAL_RESULTS_DUE"


@dataclass
class EventEvidence:
    """Evidence supporting an event detection"""
    rule_id: str
    fields: Dict[str, Any]
    confidence: float
    confidence_reason: str

    def to_dict(self) -> dict:
        return {
            'rule_id': self.rule_id,
            'fields': self.fields,
            'confidence': self.confidence,
            'confidence_reason': self.confidence_reason,
        }


# ============================================================================
# CALENDAR-BASED CATALYST DETECTION
# ============================================================================

@dataclass
class CalendarCatalyst:
    """Forward-looking catalyst from trial calendar dates"""
    ticker: str
    nct_id: str
    event_type: str  # 'UPCOMING_PCD', 'UPCOMING_SCD', 'RESULTS_DUE'
    target_date: date
    days_until: int
    window: str  # '30D', '60D', '90D'
    confidence: float
    rule_id: str
    evidence: EventEvidence

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'nct_id': self.nct_id,
            'event_type': self.event_type,
            'target_date': self.target_date.isoformat(),
            'days_until': self.days_until,
            'window': self.window,
            'confidence': self.confidence,
            'rule_id': self.rule_id,
            'evidence': self.evidence.to_dict(),
        }


def detect_calendar_catalysts(
    current_snapshot: StateSnapshot,
    as_of_date: date,
    windows: Tuple[int, ...] = (30, 60, 90),
) -> List[CalendarCatalyst]:
    """
    Detect forward-looking calendar-based catalysts.

    These are deterministic from CT.gov date fields and provide consistent
    signal even when no snapshot changes occurred.

    Args:
        current_snapshot: Current trial state
        as_of_date: Analysis date
        windows: Days-ahead windows to check (default 30, 60, 90)

    Returns:
        List of calendar catalysts
    """
    catalysts = []

    for record in current_snapshot.records:
        # Skip terminal trials
        if record.overall_status.is_terminal_negative:
            continue

        # Primary Completion Date upcoming
        if record.primary_completion_date:
            days_until = (record.primary_completion_date - as_of_date).days

            if 0 < days_until <= max(windows):
                # Determine window
                if days_until <= 30:
                    window = '30D'
                    rule_id = EventRuleID.M3_CAL_PCD_30D
                    confidence = 0.85
                elif days_until <= 60:
                    window = '60D'
                    rule_id = EventRuleID.M3_CAL_PCD_60D
                    confidence = 0.75
                else:
                    window = '90D'
                    rule_id = EventRuleID.M3_CAL_PCD_90D
                    confidence = 0.65

                # Boost confidence if date is ACTUAL
                if record.primary_completion_type and record.primary_completion_type.value == 'ACTUAL':
                    confidence = min(0.95, confidence + 0.10)
                    confidence_reason = "Date confirmed as ACTUAL"
                elif record.primary_completion_type and record.primary_completion_type.value == 'ESTIMATED':
                    confidence_reason = "Date is ESTIMATED, may shift"
                else:
                    confidence_reason = "Date type unknown"

                catalysts.append(CalendarCatalyst(
                    ticker=record.ticker,
                    nct_id=record.nct_id,
                    event_type='UPCOMING_PCD',
                    target_date=record.primary_completion_date,
                    days_until=days_until,
                    window=window,
                    confidence=confidence,
                    rule_id=rule_id,
                    evidence=EventEvidence(
                        rule_id=rule_id,
                        fields={
                            'primary_completion_date': record.primary_completion_date.isoformat(),
                            'primary_completion_type': record.primary_completion_type.value if record.primary_completion_type else None,
                            'days_until': days_until,
                        },
                        confidence=confidence,
                        confidence_reason=confidence_reason,
                    ),
                ))

        # Study Completion Date upcoming
        if record.completion_date and record.completion_date != record.primary_completion_date:
            days_until = (record.completion_date - as_of_date).days

            if 0 < days_until <= max(windows):
                if days_until <= 30:
                    window = '30D'
                    rule_id = EventRuleID.M3_CAL_SCD_30D
                    confidence = 0.75
                elif days_until <= 60:
                    window = '60D'
                    rule_id = EventRuleID.M3_CAL_SCD_60D
                    confidence = 0.65
                else:
                    window = '90D'
                    rule_id = EventRuleID.M3_CAL_SCD_90D
                    confidence = 0.55

                if record.completion_type and record.completion_type.value == 'ACTUAL':
                    confidence = min(0.90, confidence + 0.10)
                    confidence_reason = "Date confirmed as ACTUAL"
                else:
                    confidence_reason = "Date is estimated"

                catalysts.append(CalendarCatalyst(
                    ticker=record.ticker,
                    nct_id=record.nct_id,
                    event_type='UPCOMING_SCD',
                    target_date=record.completion_date,
                    days_until=days_until,
                    window=window,
                    confidence=confidence,
                    rule_id=rule_id,
                    evidence=EventEvidence(
                        rule_id=rule_id,
                        fields={
                            'completion_date': record.completion_date.isoformat(),
                            'completion_type': record.completion_type.value if record.completion_type else None,
                            'days_until': days_until,
                        },
                        confidence=confidence,
                        confidence_reason=confidence_reason,
                    ),
                ))

    # Sort by days_until for deterministic output
    catalysts.sort(key=lambda c: (c.days_until, c.ticker, c.nct_id))

    return catalysts


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def summarize_calendar_catalysts(
    catalysts: List[CalendarCatalyst],
) -> Dict[str, Any]:
    """Summarize calendar catalysts by ticker and window"""
    by_ticker: Dict[str, List[CalendarCatalyst]] = {}
    by_window = Counter()
    by_type = Counter()

    for cat in catalysts:
        if cat.ticker not in by_ticker:
            by_ticker[cat.ticker] = []
        by_ticker[cat.ticker].append(cat)
        by_window[cat.window] += 1
        by_type[cat.event_type] += 1

    return {
        'total_catalysts': len(catalysts),
        'tickers_with_catalysts': len(by_ticker),
        'by_window': dict(by_window),
        'by_type': dict(by_type),
        'catalysts_by_ticker': {
            ticker: [c.to_dict() for c in cats]
            for ticker, cats in sorted(by_ticker.items())
        },
    }


if __name__ == "__main__":
    print("Catalyst Diagnostics module loaded successfully")
    print("Components:")
    print("  - compute_delta_diagnostics(): Snapshot diff analysis")
    print("  - check_trial_records_staleness(): Data freshness validation")
    print("  - detect_calendar_catalysts(): Forward-looking catalyst detection")
