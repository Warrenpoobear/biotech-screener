"""Tests for CT.gov date-precision plumbing (#535 / Spec 114, phase 1 of 2).

Classification: CATALYST_DATE_PRECISION_PROVENANCE / NO_MODEL_CHANGE

CT.gov reports many completion dates as month-only and ESTIMATED.
ctgov_client._parse_date() snaps "2026-08" to "2026-08-01" and returns a bare
string, so downstream code cannot distinguish an exact date from a month
placeholder. 62% of DAY-precision catalyst dates land on the 1st of a month and
20% on a weekend.

This carries the real granularity from the trial record through
CanonicalTrialRecord to CalendarCatalyst. It deliberately does NOT change
routing: module_3_catalyst still stamps date_precision="DAY", because switching
that moves catalyst_decay_w -> catalyst_tilt_mult -> target_weight_pct, and
sizing is frozen under the DEM NO_MODEL_CHANGE window. The switch is a separate
one-line change gated on a bitwise final_score comparison.

The contract these tests lock: absence of precision must mean "unknown", never
"DAY". Defaulting to the strongest claim is the defect being fixed.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from catalyst_diagnostics import CalendarCatalyst, EventEvidence
from ctgov_adapter import CanonicalTrialRecord, CTGovStatus


def _record(precision=None) -> CanonicalTrialRecord:
    return CanonicalTrialRecord(
        ticker="NRIX",
        nct_id="NCT06669754",
        overall_status=CTGovStatus.RECRUITING,
        last_update_posted=date(2026, 7, 1),
        primary_completion_date=date(2026, 8, 1),
        primary_completion_type=None,
        completion_date=None,
        completion_type=None,
        results_first_posted=None,
        primary_completion_precision=precision,
    )


class TestCanonicalRecordPrecision:
    def test_defaults_to_none_not_day(self):
        """Absence must mean unknown. Defaulting to DAY is the bug."""
        assert _record().primary_completion_precision is None

    def test_round_trips_month(self):
        rec = _record("MONTH")
        assert CanonicalTrialRecord.from_dict(rec.to_dict()).primary_completion_precision == "MONTH"

    def test_round_trips_none(self):
        rec = _record(None)
        assert CanonicalTrialRecord.from_dict(rec.to_dict()).primary_completion_precision is None

    def test_to_dict_exposes_the_field(self):
        assert _record("MONTH").to_dict()["primary_completion_precision"] == "MONTH"

    def test_from_dict_tolerates_legacy_payload(self):
        """Caches written before #538 have no precision key — must not KeyError."""
        payload = _record("MONTH").to_dict()
        del payload["primary_completion_precision"]
        assert CanonicalTrialRecord.from_dict(payload).primary_completion_precision is None

    def test_precision_does_not_alter_the_date(self):
        """Plumbing only — the snapped date itself is untouched at this stage."""
        assert _record("MONTH").primary_completion_date == date(2026, 8, 1)


class TestCalendarCatalystPrecision:
    def _catalyst(self, precision=None) -> CalendarCatalyst:
        return CalendarCatalyst(
            ticker="NRIX",
            nct_id="NCT06669754",
            event_type="UPCOMING_PCD",
            target_date=date(2026, 8, 1),
            days_until=5,
            window="30D",
            confidence=0.5,
            rule_id="R1",
            evidence=EventEvidence(rule_id="R1", fields={}, confidence=0.5, confidence_reason="test"),
            date_precision=precision,
        )

    def test_defaults_to_none_not_day(self):
        assert self._catalyst().date_precision is None

    def test_carries_month(self):
        assert self._catalyst("MONTH").date_precision == "MONTH"

    def test_to_dict_exposes_precision(self):
        assert self._catalyst("MONTH").to_dict()["date_precision"] == "MONTH"

    def test_to_dict_none_is_serialized_as_none(self):
        assert self._catalyst().to_dict()["date_precision"] is None
