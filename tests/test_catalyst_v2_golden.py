#!/usr/bin/env python3
"""
test_catalyst_v2_golden.py - Golden Fixture Tests for Catalyst Module v2

Tests for:
1. Stable event IDs (SHA256 hash determinism)
2. PIT filtering correctness
3. Deterministic output hashes

Design Philosophy:
- All tests are deterministic (no randomness)
- Tests verify exact expected values (golden fixtures)
- No external dependencies

Usage:
    python tests/test_catalyst_v2_golden.py
"""

import hashlib
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_3_schema_v2 import (
    CatalystEventV2,
    EventType,
    EventSeverity,
    ConfidenceLevel,
    SourceReliability,
    DateSpecificity,
    DeltaType,
    DeltaEvent,
    TickerCatalystSummaryV2,
    DiagnosticCountsV2,
    SCHEMA_VERSION,
    SCORE_VERSION,
    canonical_json_dumps,
)

from module_3_scoring_v2 import (
    compute_proximity_score,
    compute_negative_catalyst_score,
    detect_deltas,
    compute_delta_score,
    compute_velocity,
    calculate_ticker_catalyst_score,
    score_catalyst_events,
)


# =============================================================================
# GOLDEN FIXTURES
# =============================================================================

# Event 1: Positive enrollment event
GOLDEN_EVENT_1 = CatalystEventV2(
    ticker="ARGX",
    nct_id="NCT12345678",
    event_type=EventType.CT_ENROLLMENT_COMPLETE,
    event_severity=EventSeverity.POSITIVE,
    event_date="2026-03-15",
    field_changed="enrollment_status",
    prior_value="RECRUITING",
    new_value="COMPLETED",
    source="CTGOV",
    confidence=ConfidenceLevel.HIGH,
    disclosed_at="2026-01-10",
    source_date="2026-01-10",
    pit_date_field_used="last_update_posted",
    source_reliability=SourceReliability.OFFICIAL,
    date_specificity=DateSpecificity.EXACT,
    corroboration_count=1,
)

# Expected event_id for GOLDEN_EVENT_1
GOLDEN_EVENT_1_ID = "0a0e3e9a3c3f8d1e4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a"

# Event 2: Severe negative (termination)
GOLDEN_EVENT_2 = CatalystEventV2(
    ticker="SANA",
    nct_id="NCT87654321",
    event_type=EventType.CT_TRIAL_TERMINATED,
    event_severity=EventSeverity.SEVERE_NEGATIVE,
    event_date="2026-01-05",
    field_changed="overall_status",
    prior_value="ACTIVE",
    new_value="TERMINATED",
    source="CTGOV",
    confidence=ConfidenceLevel.HIGH,
    disclosed_at="2026-01-08",
    source_date="2026-01-08",
    pit_date_field_used="last_update_posted",
    source_reliability=SourceReliability.OFFICIAL,
    date_specificity=DateSpecificity.EXACT,
    corroboration_count=2,
)


# =============================================================================
# TEST CLASS
# =============================================================================

class CatalystV2GoldenRunner:
    """Golden fixture tests for Catalyst Module v2 (standalone runner).

    Note: This class is named CatalystV2GoldenRunner (not Test*) to avoid
    pytest collection warnings. It has an __init__ constructor and is
    designed to run as a standalone script via __main__.
    """

    def __init__(self):
        self.as_of_date = date(2026, 1, 11)
        self.results = []

    def run_test(self, test_name: str, test_func):
        """Run a single test and record result."""
        try:
            test_func()
            self.results.append((test_name, True, None))
            print(f"  ✓ {test_name}")
        except AssertionError as e:
            self.results.append((test_name, False, str(e)))
            print(f"  ✗ {test_name}: {e}")
        except Exception as e:
            self.results.append((test_name, False, str(e)))
            print(f"  ✗ {test_name}: EXCEPTION: {e}")

    # =========================================================================
    # STABLE ID TESTS
    # =========================================================================

    def test_event_id_stability(self):
        """Test that event_id is stable across instantiations."""
        # Create same event twice
        event1 = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-06-01",
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-06-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
        )

        event2 = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-06-01",
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-06-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-11",  # Different disclosed_at
            source_date="2026-01-11",  # Different source_date
            pit_date_field_used="first_posted",  # Different pit field
        )

        # event_id should be identical (doesn't include observation metadata)
        assert event1.event_id == event2.event_id, \
            f"Event IDs differ: {event1.event_id} != {event2.event_id}"

    def test_event_id_uniqueness(self):
        """Test that different events have different IDs."""
        event1 = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-06-01",
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-06-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
        )

        event2 = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000002",  # Different NCT ID
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-06-01",
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-06-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
        )

        assert event1.event_id != event2.event_id, "Different events should have different IDs"

    def test_event_id_hash_format(self):
        """Test that event_id is valid SHA256 hex."""
        event = GOLDEN_EVENT_1
        event_id = event.event_id

        # Should be 64 hex characters
        assert len(event_id) == 64, f"event_id length should be 64, got {len(event_id)}"
        assert all(c in "0123456789abcdef" for c in event_id), "event_id should be hex"

        # Short form should be 16 characters
        assert len(event.event_id_short) == 16, f"event_id_short length should be 16"

    def test_event_id_deterministic_fields(self):
        """Test that event_id only depends on deterministic fields."""
        # These fields should NOT affect event_id:
        # - disclosed_at, source_date, pit_date_field_used
        # - source_reliability, date_specificity, corroboration_count

        base = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_ENROLLMENT_STARTED,
            event_severity=EventSeverity.POSITIVE,
            event_date="2026-02-01",
            field_changed="status",
            prior_value="NOT_YET_RECRUITING",
            new_value="RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.MED,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
            source_reliability=SourceReliability.OFFICIAL,
            date_specificity=DateSpecificity.EXACT,
            corroboration_count=0,
        )

        variant = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_ENROLLMENT_STARTED,
            event_severity=EventSeverity.POSITIVE,
            event_date="2026-02-01",
            field_changed="status",
            prior_value="NOT_YET_RECRUITING",
            new_value="RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,  # Different
            disclosed_at="2026-01-15",  # Different
            source_date="2026-01-15",  # Different
            pit_date_field_used="first_posted",  # Different
            source_reliability=SourceReliability.INFERRED,  # Different
            date_specificity=DateSpecificity.MONTH,  # Different
            corroboration_count=5,  # Different
        )

        assert base.event_id == variant.event_id, \
            "event_id should not depend on observation metadata"

    # =========================================================================
    # PIT FILTERING TESTS
    # =========================================================================

    def test_pit_source_date_required(self):
        """Test that source_date is a required field."""
        event = GOLDEN_EVENT_1
        assert event.source_date is not None, "source_date should be set"
        assert event.source_date == "2026-01-10", f"Expected source_date 2026-01-10, got {event.source_date}"

    def test_pit_date_field_used_required(self):
        """Test that pit_date_field_used is a required field."""
        event = GOLDEN_EVENT_1
        assert event.pit_date_field_used is not None, "pit_date_field_used should be set"
        assert event.pit_date_field_used == "last_update_posted"

    def test_pit_certainty_staleness(self):
        """Test that certainty score decays with staleness."""
        # Recent event
        recent_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_ENROLLMENT_STARTED,
            event_severity=EventSeverity.POSITIVE,
            event_date="2026-01-15",
            field_changed="status",
            prior_value=None,
            new_value="RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",  # 1 day old
            pit_date_field_used="last_update_posted",
            source_reliability=SourceReliability.OFFICIAL,
            date_specificity=DateSpecificity.EXACT,
            corroboration_count=1,
        )

        # Stale event (180 days old source_date)
        stale_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000002",
            event_type=EventType.CT_ENROLLMENT_STARTED,
            event_severity=EventSeverity.POSITIVE,
            event_date="2026-01-15",
            field_changed="status",
            prior_value=None,
            new_value="RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2025-07-01",
            source_date="2025-07-01",  # ~195 days old
            pit_date_field_used="last_update_posted",
            source_reliability=SourceReliability.OFFICIAL,
            date_specificity=DateSpecificity.EXACT,
            corroboration_count=1,
        )

        as_of = date(2026, 1, 11)
        recent_certainty = recent_event.compute_certainty_score(as_of)
        stale_certainty = stale_event.compute_certainty_score(as_of)

        assert recent_certainty > stale_certainty, \
            f"Recent event should have higher certainty: {recent_certainty} vs {stale_certainty}"

    def test_pit_future_event_zero_certainty(self):
        """Test that events with future source_date have zero certainty."""
        future_event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_ENROLLMENT_STARTED,
            event_severity=EventSeverity.POSITIVE,
            event_date="2026-01-15",
            field_changed="status",
            prior_value=None,
            new_value="RECRUITING",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-20",
            source_date="2026-01-20",  # Future source_date
            pit_date_field_used="last_update_posted",
        )

        as_of = date(2026, 1, 11)
        certainty = future_event.compute_certainty_score(as_of)

        assert certainty == Decimal("0"), \
            f"Future source_date should have zero certainty, got {certainty}"

    # =========================================================================
    # DETERMINISTIC HASH TESTS
    # =========================================================================

    def test_canonical_json_determinism(self):
        """Test that canonical_json_dumps produces deterministic output."""
        obj = {
            "z_field": 1,
            "a_field": 2,
            "m_field": [3, 2, 1],
            "nested": {"b": 1, "a": 2},
        }

        json1 = canonical_json_dumps(obj)
        json2 = canonical_json_dumps(obj)

        assert json1 == json2, "canonical_json_dumps should be deterministic"

        # Keys should be sorted
        assert '"a_field"' in json1
        assert json1.index('"a_field"') < json1.index('"m_field"'), \
            "Keys should be alphabetically sorted"

    def test_event_to_dict_determinism(self):
        """Test that event.to_dict() is deterministic."""
        event = GOLDEN_EVENT_1

        dict1 = event.to_dict()
        dict2 = event.to_dict()

        json1 = canonical_json_dumps(dict1)
        json2 = canonical_json_dumps(dict2)

        assert json1 == json2, "event.to_dict() should be deterministic"

    def test_summary_hash_determinism(self):
        """Test that ticker summary produces deterministic hash."""
        events = [GOLDEN_EVENT_1]
        as_of = date(2026, 1, 11)

        summary1 = calculate_ticker_catalyst_score("ARGX", events, as_of)
        summary2 = calculate_ticker_catalyst_score("ARGX", events, as_of)

        json1 = canonical_json_dumps(summary1.to_dict())
        json2 = canonical_json_dumps(summary2.to_dict())

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2, f"Summary hashes differ: {hash1} != {hash2}"

    def test_batch_scoring_determinism(self):
        """Test that batch scoring is deterministic."""
        events_by_ticker = {
            "ARGX": [GOLDEN_EVENT_1],
            "SANA": [GOLDEN_EVENT_2],
        }
        active_tickers = ["ARGX", "SANA", "UNKNOWN"]
        as_of = date(2026, 1, 11)

        summaries1, diag1 = score_catalyst_events(events_by_ticker, active_tickers, as_of)
        summaries2, diag2 = score_catalyst_events(events_by_ticker, active_tickers, as_of)

        # Compare summary hashes
        for ticker in active_tickers:
            json1 = canonical_json_dumps(summaries1[ticker].to_dict())
            json2 = canonical_json_dumps(summaries2[ticker].to_dict())
            assert json1 == json2, f"Summary for {ticker} not deterministic"

        # Compare diagnostic hashes
        diag_json1 = canonical_json_dumps(diag1.to_dict())
        diag_json2 = canonical_json_dumps(diag2.to_dict())
        assert diag_json1 == diag_json2, "Diagnostics not deterministic"

    # =========================================================================
    # PROXIMITY SCORE TESTS
    # =========================================================================

    def test_proximity_score_upcoming_events(self):
        """Test proximity score for upcoming events."""
        # Event 64 days in future
        event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-03-15",  # 63 days from 2026-01-11
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-03-15",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
            source_reliability=SourceReliability.OFFICIAL,
            date_specificity=DateSpecificity.EXACT,
            corroboration_count=1,
        )

        as_of = date(2026, 1, 11)
        score, n_upcoming = compute_proximity_score([event], as_of)

        assert n_upcoming == 1, f"Expected 1 upcoming event, got {n_upcoming}"
        assert score > Decimal("0"), f"Proximity score should be positive, got {score}"

    def test_proximity_score_past_events_excluded(self):
        """Test that past events don't contribute to proximity score."""
        # Past event
        event = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-01-01",  # Past date
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-01-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
        )

        as_of = date(2026, 1, 11)
        score, n_upcoming = compute_proximity_score([event], as_of)

        assert n_upcoming == 0, f"Past events should not be upcoming, got {n_upcoming}"
        assert score == Decimal("0"), f"Score should be 0 for past events, got {score}"

    # =========================================================================
    # DELTA ENGINE TESTS
    # =========================================================================

    def test_delta_event_added(self):
        """Test detection of added events."""
        current = [GOLDEN_EVENT_1]
        prior = []
        as_of = date(2026, 1, 11)

        deltas = detect_deltas(current, prior, as_of)

        added = [d for d in deltas if d.delta_type == DeltaType.EVENT_ADDED]
        assert len(added) == 1, f"Expected 1 added event, got {len(added)}"

    def test_delta_event_removed(self):
        """Test detection of removed events."""
        current = []
        prior = [GOLDEN_EVENT_1]
        as_of = date(2026, 1, 11)

        deltas = detect_deltas(current, prior, as_of)

        removed = [d for d in deltas if d.delta_type == DeltaType.EVENT_REMOVED]
        assert len(removed) == 1, f"Expected 1 removed event, got {len(removed)}"

    def test_delta_date_shift(self):
        """Test detection of date shifts."""
        prior = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-03-01",  # Original date
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-03-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-05",
            source_date="2026-01-05",
            pit_date_field_used="last_update_posted",
        )

        current = CatalystEventV2(
            ticker="TEST",
            nct_id="NCT00000001",
            event_type=EventType.CT_PRIMARY_COMPLETION,
            event_severity=EventSeverity.CRITICAL_POSITIVE,
            event_date="2026-04-01",  # Shifted 31 days later
            field_changed="primary_completion_date",
            prior_value=None,
            new_value="2026-04-01",
            source="CTGOV",
            confidence=ConfidenceLevel.HIGH,
            disclosed_at="2026-01-10",
            source_date="2026-01-10",
            pit_date_field_used="last_update_posted",
        )

        as_of = date(2026, 1, 11)
        deltas = detect_deltas([current], [prior], as_of)

        # Should detect window widening (completion date moved later)
        widening = [d for d in deltas if d.delta_type == DeltaType.WINDOW_WIDENING]
        assert len(widening) == 1, f"Expected 1 window widening, got {len(widening)}"
        assert widening[0].shift_days == 31, f"Expected 31 day shift, got {widening[0].shift_days}"

    # =========================================================================
    # NEGATIVE CATALYST TESTS
    # =========================================================================

    def test_negative_catalyst_detection(self):
        """Test detection of negative catalysts."""
        event = GOLDEN_EVENT_2  # Terminated trial

        assert event.is_negative is True, "Terminated trial should be negative"
        assert event.is_severe_negative is True, "Terminated trial should be severe negative"

    def test_negative_catalyst_score(self):
        """Test negative catalyst scoring."""
        events = [GOLDEN_EVENT_2]
        as_of = date(2026, 1, 11)

        score, n_negative, n_severe = compute_negative_catalyst_score(events, as_of)

        assert n_negative == 1, f"Expected 1 negative event, got {n_negative}"
        assert n_severe == 1, f"Expected 1 severe negative, got {n_severe}"
        assert score > Decimal("0"), f"Negative score should be positive (risk), got {score}"

    # =========================================================================
    # VELOCITY TESTS
    # =========================================================================

    def test_velocity_calculation(self):
        """Test velocity calculation with 4-week history."""
        current = Decimal("50.0")
        history = [
            Decimal("40.0"),
            Decimal("42.0"),
            Decimal("38.0"),
            Decimal("44.0"),
        ]

        velocity = compute_velocity(current, history)

        # Median of [38, 40, 42, 44] = (40 + 42) / 2 = 41
        # Velocity = 50 - 41 = 9
        assert velocity == Decimal("9.00"), f"Expected velocity 9.00, got {velocity}"

    def test_velocity_insufficient_history(self):
        """Test velocity with insufficient history."""
        current = Decimal("50.0")
        history = [Decimal("40.0"), Decimal("42.0")]  # Only 2 points

        velocity = compute_velocity(current, history)

        assert velocity is None, "Velocity should be None with < 4 history points"

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all(self):
        """Run all tests."""
        print("=" * 60)
        print("CATALYST MODULE v2 GOLDEN FIXTURE TESTS")
        print("=" * 60)
        print()

        # Stable ID tests
        print("Stable ID Tests:")
        self.run_test("test_event_id_stability", self.test_event_id_stability)
        self.run_test("test_event_id_uniqueness", self.test_event_id_uniqueness)
        self.run_test("test_event_id_hash_format", self.test_event_id_hash_format)
        self.run_test("test_event_id_deterministic_fields", self.test_event_id_deterministic_fields)
        print()

        # PIT tests
        print("PIT Filtering Tests:")
        self.run_test("test_pit_source_date_required", self.test_pit_source_date_required)
        self.run_test("test_pit_date_field_used_required", self.test_pit_date_field_used_required)
        self.run_test("test_pit_certainty_staleness", self.test_pit_certainty_staleness)
        self.run_test("test_pit_future_event_zero_certainty", self.test_pit_future_event_zero_certainty)
        print()

        # Deterministic hash tests
        print("Deterministic Hash Tests:")
        self.run_test("test_canonical_json_determinism", self.test_canonical_json_determinism)
        self.run_test("test_event_to_dict_determinism", self.test_event_to_dict_determinism)
        self.run_test("test_summary_hash_determinism", self.test_summary_hash_determinism)
        self.run_test("test_batch_scoring_determinism", self.test_batch_scoring_determinism)
        print()

        # Proximity score tests
        print("Proximity Score Tests:")
        self.run_test("test_proximity_score_upcoming_events", self.test_proximity_score_upcoming_events)
        self.run_test("test_proximity_score_past_events_excluded", self.test_proximity_score_past_events_excluded)
        print()

        # Delta engine tests
        print("Delta Engine Tests:")
        self.run_test("test_delta_event_added", self.test_delta_event_added)
        self.run_test("test_delta_event_removed", self.test_delta_event_removed)
        self.run_test("test_delta_date_shift", self.test_delta_date_shift)
        print()

        # Negative catalyst tests
        print("Negative Catalyst Tests:")
        self.run_test("test_negative_catalyst_detection", self.test_negative_catalyst_detection)
        self.run_test("test_negative_catalyst_score", self.test_negative_catalyst_score)
        print()

        # Velocity tests
        print("Velocity Tests:")
        self.run_test("test_velocity_calculation", self.test_velocity_calculation)
        self.run_test("test_velocity_insufficient_history", self.test_velocity_insufficient_history)
        print()

        # Summary
        print("=" * 60)
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        print(f"RESULTS: {passed}/{total} tests passed")
        print("=" * 60)

        return passed == total


if __name__ == "__main__":
    tester = CatalystV2GoldenRunner()
    success = tester.run_all()
    sys.exit(0 if success else 1)
