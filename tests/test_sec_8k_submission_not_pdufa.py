"""Submissions are not PDUFA dates; gaps stay inside one sentence (2026-11-11).

Real 8-K sentences that produced HIGH/DAY "PDUFA" events in production.
"""

from datetime import date

from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import _extract_timing_events

FILING = "2026-10-29"
AS_OF = date(2026, 11, 11)


def _pdufa(events):
    return [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]


def test_planned_nda_submission_is_not_a_pdufa():
    text = (
        "Stoke expects an NDA submission in the second half of 2027. Mr. Smith continued, "
        "the company will host a call on October 29, 2026."
    )
    events = _extract_timing_events(text, "STOK", FILING, AS_OF)
    assert _pdufa(events) == []
    # and the press-release date in the next sentence is not attached to it
    assert not any(e["event_date"] == "2026-10-29" for e in events)


def test_bla_submission_quarter_is_fda_submission():
    text = "Dyne plans a BLA submission for U.S. Accelerated Approval in Q3 2027."
    events = _extract_timing_events(text, "DYN", FILING, AS_OF)
    assert _pdufa(events) == []
    subs = [e for e in events if e["event_type"] == "FDA_SUBMISSION"]
    assert subs and subs[0]["date_precision"] == "QUARTER"


def test_real_pdufa_still_extracted():
    text = "The FDA granted priority review with a PDUFA target action date of January 21, 2027."
    events = _extract_timing_events(text, "DYN", FILING, AS_OF)
    assert [e["event_date"] for e in _pdufa(events)] == ["2027-01-21"]


def test_designation_gap_does_not_cross_sentences():
    text = (
        "FDA granted Breakthrough Therapy Designation for zanidatamab. "
        "Separately, the company anticipates topline data in Q1 2027."
    )
    events = _extract_timing_events(text, "ZYME", FILING, AS_OF)
    assert _pdufa(events) == []
