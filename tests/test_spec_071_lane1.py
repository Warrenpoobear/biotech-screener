#!/usr/bin/env python3
"""Spec 071 Lane 1 — CT.gov catalyst status hard-reject.

The producer-level filter in `catalyst_diagnostics.detect_calendar_catalysts`
and `detect_readout_window_catalysts` must drop any trial whose `overall_status` is
catalyst-ineligible (WITHDRAWN / TERMINATED / SUSPENDED / APPROVED_FOR_MARKETING
/ NO_LONGER_AVAILABLE) before it can drive a `CalendarCatalyst`.

Lane 1 is a defect fix; it does not introduce the broader Lane 2 classifier.
"""
from __future__ import annotations

from datetime import date

import pytest

from catalyst_diagnostics import detect_calendar_catalysts, detect_readout_window_catalysts
from ctgov_adapter import CanonicalTrialRecord, CompletionType, CTGovStatus
from state_management import StateSnapshot

# ---------------------------------------------------------------------------
# Status property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        CTGovStatus.WITHDRAWN,
        CTGovStatus.TERMINATED,
        CTGovStatus.SUSPENDED,
        CTGovStatus.APPROVED_FOR_MARKETING,
        CTGovStatus.NO_LONGER_AVAILABLE,
    ],
)
def test_catalyst_ineligible_statuses(status: CTGovStatus) -> None:
    assert status.is_catalyst_ineligible is True


@pytest.mark.parametrize(
    "status",
    [
        CTGovStatus.RECRUITING,
        CTGovStatus.ACTIVE_NOT_RECRUITING,
        CTGovStatus.ENROLLING_BY_INVITATION,
        CTGovStatus.NOT_YET_RECRUITING,
        CTGovStatus.AVAILABLE,
        CTGovStatus.COMPLETED,
        CTGovStatus.UNKNOWN,
        CTGovStatus.WITHHELD,
    ],
)
def test_catalyst_eligible_statuses(status: CTGovStatus) -> None:
    assert status.is_catalyst_ineligible is False


def test_terminal_negative_unchanged() -> None:
    """is_terminal_negative is consumed by other call sites; do not widen."""
    expected = {CTGovStatus.WITHDRAWN, CTGovStatus.TERMINATED, CTGovStatus.SUSPENDED}
    actual = {s for s in CTGovStatus if s.is_terminal_negative}
    assert actual == expected


# ---------------------------------------------------------------------------
# Producer integration tests (spec 071 Lane 1 acceptance)
# ---------------------------------------------------------------------------


def _trial(
    ticker: str,
    nct_id: str,
    status: CTGovStatus,
    pcd: date,
) -> CanonicalTrialRecord:
    return CanonicalTrialRecord(
        ticker=ticker,
        nct_id=nct_id,
        overall_status=status,
        last_update_posted=date(2026, 4, 1),
        primary_completion_date=pcd,
        primary_completion_type=CompletionType.ESTIMATED,
        completion_date=None,
        completion_type=None,
        results_first_posted=None,
    )


@pytest.fixture
def as_of() -> date:
    return date(2026, 4, 29)


@pytest.mark.parametrize(
    "rejected_status",
    [
        CTGovStatus.WITHDRAWN,
        CTGovStatus.TERMINATED,
        CTGovStatus.SUSPENDED,
        CTGovStatus.APPROVED_FOR_MARKETING,
        CTGovStatus.NO_LONGER_AVAILABLE,
    ],
)
def test_calendar_producer_drops_ineligible(rejected_status: CTGovStatus, as_of: date) -> None:
    """detect_calendar_catalysts must not emit calendar events for ineligible statuses."""
    rejected = _trial("XXXX", "NCT99999991", rejected_status, pcd=date(2026, 5, 30))
    eligible = _trial("XXXX", "NCT99999992", CTGovStatus.RECRUITING, pcd=date(2026, 5, 30))
    snap = StateSnapshot(snapshot_date=as_of, records=[rejected, eligible])

    out = detect_calendar_catalysts(snap, as_of_date=as_of)
    nct_ids = {c.nct_id for c in out}

    assert "NCT99999991" not in nct_ids
    assert "NCT99999992" in nct_ids


@pytest.mark.parametrize(
    "rejected_status",
    [
        CTGovStatus.WITHDRAWN,
        CTGovStatus.TERMINATED,
        CTGovStatus.SUSPENDED,
        CTGovStatus.APPROVED_FOR_MARKETING,
        CTGovStatus.NO_LONGER_AVAILABLE,
    ],
)
def test_readout_window_producer_drops_ineligible(rejected_status: CTGovStatus, as_of: date) -> None:
    """detect_readout_window_catalysts must not emit readout events for ineligible statuses."""
    # PCD must be in the past for readout-window inference.
    past_pcd = date(2026, 3, 15)
    rejected = _trial("YYYY", "NCT99999993", rejected_status, pcd=past_pcd)
    eligible = _trial("YYYY", "NCT99999994", CTGovStatus.ACTIVE_NOT_RECRUITING, pcd=past_pcd)
    snap = StateSnapshot(snapshot_date=as_of, records=[rejected, eligible])

    out = detect_readout_window_catalysts(snap, as_of_date=as_of)
    nct_ids = {c.nct_id for c in out}

    assert "NCT99999993" not in nct_ids
    assert "NCT99999994" in nct_ids
