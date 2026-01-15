#!/usr/bin/env python3
"""
Test catalyst module v2 with live production data.
"""

import json
from datetime import date
from pathlib import Path
from decimal import Decimal
from collections import defaultdict

from module_3_schema_v2 import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    SourceReliability,
    DateSpecificity,
    TickerCatalystSummaryV2
)
from module_3_scoring_v2 import (
    score_catalyst_events,
    compute_proximity_score,
    compute_negative_catalyst_score,
    detect_deltas
)


def load_trial_records(path: Path) -> list:
    """Load trial records from JSON file."""
    with open(path) as f:
        return json.load(f)


def trial_to_v2_event(trial: dict, as_of_date: date) -> CatalystEventV2:
    """Convert a trial record to CatalystEventV2 format."""

    # Determine event type and severity based on trial status
    status = trial.get("status", "")
    phase = trial.get("phase", "")

    # Map status to v2 EventType enum
    if status == "COMPLETED":
        if trial.get("results_first_posted"):
            event_type = EventType.CT_RESULTS_POSTED
            event_severity = EventSeverity.CRITICAL_POSITIVE
        else:
            event_type = EventType.CT_STUDY_COMPLETION
            event_severity = EventSeverity.POSITIVE
    elif status == "ACTIVE_NOT_RECRUITING":
        event_type = EventType.CT_ENROLLMENT_COMPLETE
        event_severity = EventSeverity.POSITIVE
    elif status == "RECRUITING":
        event_type = EventType.CT_ENROLLMENT_STARTED
        event_severity = EventSeverity.POSITIVE
    elif status == "WITHDRAWN":
        event_type = EventType.CT_TRIAL_WITHDRAWN
        event_severity = EventSeverity.SEVERE_NEGATIVE
    elif status == "TERMINATED":
        event_type = EventType.CT_TRIAL_TERMINATED
        event_severity = EventSeverity.SEVERE_NEGATIVE
    elif status == "SUSPENDED":
        event_type = EventType.CT_TRIAL_SUSPENDED
        event_severity = EventSeverity.SEVERE_NEGATIVE
    else:
        event_type = EventType.UNKNOWN
        event_severity = EventSeverity.NEUTRAL

    # Get event date (primary completion date if available)
    event_date_str = trial.get("primary_completion_date") or trial.get("completion_date")
    if not event_date_str:
        event_date_str = None

    # Source date from collected_at
    source_date_str = trial.get("collected_at", as_of_date.isoformat())

    # Disclosed at from last_update_posted
    disclosed_at = trial.get("last_update_posted", source_date_str)

    # Determine source reliability
    if trial.get("results_first_posted"):
        source_reliability = SourceReliability.OFFICIAL
    elif status == "COMPLETED":
        source_reliability = SourceReliability.COMPANY
    else:
        source_reliability = SourceReliability.OFFICIAL  # CT.gov is official

    # Determine date specificity
    if event_date_str and len(event_date_str) == 10:  # YYYY-MM-DD
        date_specificity = DateSpecificity.EXACT
    else:
        date_specificity = DateSpecificity.UNKNOWN

    # Confidence level based on phase and source
    if "PHASE3" in phase:
        confidence = ConfidenceLevel.HIGH
    elif "PHASE2" in phase:
        confidence = ConfidenceLevel.MED
    else:
        confidence = ConfidenceLevel.LOW

    return CatalystEventV2(
        ticker=trial.get("ticker", ""),
        nct_id=trial.get("nct_id", ""),
        event_type=event_type,
        event_severity=event_severity,
        event_date=event_date_str,
        field_changed="status",
        prior_value=None,
        new_value=trial.get("status"),
        source="ctgov",
        confidence=confidence,
        disclosed_at=disclosed_at,
        source_date=source_date_str,
        pit_date_field_used="last_update_posted",
        source_reliability=source_reliability,
        date_specificity=date_specificity,
        corroboration_count=1 if trial.get("results_first_posted") else 0,
    )


def main():
    print("=" * 70)
    print("CATALYST MODULE v2 - LIVE DATA TEST")
    print("=" * 70)
    print()

    as_of_date = date(2026, 1, 11)
    print(f"As-of date: {as_of_date}")
    print()

    # Load trial records
    trial_path = Path("production_data/trial_records.json")
    if not trial_path.exists():
        print(f"ERROR: {trial_path} not found")
        return

    trials = load_trial_records(trial_path)
    print(f"Loaded {len(trials)} trial records")
    print()

    # Convert to v2 events
    events = []
    errors = 0
    for trial in trials:
        try:
            event = trial_to_v2_event(trial, as_of_date)
            events.append(event)
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only show first 3 errors
                import traceback
                print(f"  Error converting trial {trial.get('nct_id')}: {e}")
                traceback.print_exc()

    if errors > 3:
        print(f"  ... and {errors - 3} more conversion errors")
    if errors > 0:
        print()

    print(f"Converted {len(events)} trials to v2 events")
    print()

    # Organize events by ticker
    events_by_ticker = defaultdict(list)
    for event in events:
        if event.ticker:
            events_by_ticker[event.ticker].append(event)

    # Get unique tickers
    tickers = sorted(events_by_ticker.keys())
    print(f"Unique tickers: {len(tickers)}")
    print()

    # Run v2 scoring
    print("Running v2 scoring...")
    summaries, diagnostics = score_catalyst_events(
        events_by_ticker=dict(events_by_ticker),
        active_tickers=tickers,
        as_of_date=as_of_date,
    )
    print(f"Generated {len(summaries)} summaries")
    print()

    # Show diagnostics
    print("-" * 70)
    print("DIAGNOSTICS")
    print("-" * 70)
    print(f"  Tickers analyzed: {diagnostics.tickers_analyzed}")
    print(f"  Tickers with events: {diagnostics.tickers_with_events}")
    print(f"  Events detected total: {diagnostics.events_detected_total}")
    print(f"  Events deduped: {diagnostics.events_deduped}")
    print(f"  Upcoming events: {diagnostics.n_events_upcoming}")
    print(f"  High confidence: {diagnostics.n_high_conf}")
    print(f"  Tickers with severe negative: {diagnostics.tickers_with_severe_negative}")
    print()

    # Show top 10 by proximity score
    print("-" * 70)
    print("TOP 10 BY PROXIMITY SCORE")
    print("-" * 70)
    sorted_by_proximity = sorted(
        summaries.values(),
        key=lambda s: s.catalyst_proximity_score,
        reverse=True
    )[:10]

    print(f"{'Ticker':<8} {'Prox':<8} {'Cert':<8} {'Neg':<8} {'Events':<8} {'Window':<10}")
    print("-" * 70)
    for s in sorted_by_proximity:
        window = s.catalyst_window_days if s.catalyst_window_days else "-"
        print(f"{s.ticker:<8} {float(s.catalyst_proximity_score):<8.3f} "
              f"{float(s.avg_certainty_score):<8.3f} {float(s.negative_catalyst_score):<8.3f} "
              f"{s.n_events_upcoming:<8} {str(window):<10}")

    print()

    # Show negative catalysts
    print("-" * 70)
    print("TICKERS WITH NEGATIVE CATALYSTS")
    print("-" * 70)
    negative_tickers = [s for s in summaries.values() if s.n_negative_events > 0]
    print(f"Found {len(negative_tickers)} tickers with negative catalysts")
    print()

    for s in sorted(negative_tickers, key=lambda x: x.negative_catalyst_score, reverse=True)[:10]:
        print(f"  {s.ticker}: neg_score={float(s.negative_catalyst_score):.3f}, "
              f"n_negative={s.n_negative_events}, severe={s.severe_negative_flag}")

    print()

    # Show sample event IDs for stability verification
    print("-" * 70)
    print("SAMPLE EVENT IDs (FOR STABILITY VERIFICATION)")
    print("-" * 70)
    for event in events[:5]:
        print(f"  {event.ticker}/{event.nct_id}: {event.event_id[:16]}...")

    print()

    # Show diagnostics for top ticker
    if sorted_by_proximity:
        top = sorted_by_proximity[0]
        print("-" * 70)
        print(f"DIAGNOSTICS FOR TOP TICKER: {top.ticker}")
        print("-" * 70)
        print(f"  catalyst_proximity_score: {top.catalyst_proximity_score}")
        print(f"  score_override: {top.score_override}")
        print(f"  score_blended: {top.score_blended}")
        print(f"  negative_catalyst_score: {top.negative_catalyst_score}")
        print(f"  avg_certainty_score: {top.avg_certainty_score}")
        print(f"  n_events_upcoming: {top.n_events_upcoming}")
        print(f"  n_high_confidence: {top.n_high_confidence}")
        print(f"  n_negative_events: {top.n_negative_events}")
        print(f"  coverage_state: {top.coverage_state}")
        print(f"  next_catalyst_date: {top.next_catalyst_date}")
        print(f"  catalyst_window_bucket: {top.catalyst_window_bucket}")
        print()

        # Show events for this ticker
        ticker_events = events_by_ticker.get(top.ticker, [])
        print(f"  Events ({len(ticker_events)}):")
        for e in ticker_events[:5]:
            cert = e.compute_certainty_score(as_of_date)
            print(f"    - {e.event_type.value}: {e.event_date} (certainty={cert})")

    print()
    print("=" * 70)
    print("LIVE DATA TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
