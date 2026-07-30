"""Tests for the look-ahead guard in EV validation matching.

Classification: EV_VALIDATION_PIT_INTEGRITY / NO_MODEL_CHANGE

Issue #541. match_predictions_to_resolutions() filters predictions with

    if pred_date >= catalyst_date:
        continue  # Prediction must be before the event

The intent is right, but `catalyst_date` is month-snapped (#535) and is
future-dated in 14 resolution records, so it is not the event date. A prediction
made *after* the real outcome passes that check. ELVN: catalyst_date 2026-07-01,
real event 2024-09-28, prediction 2026-05-04 — admitted, 583 days of hindsight.

All 25 affected records are HIT, so the contamination flatters calibration.

These guards compare against evidence of the *actual* event instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from tools.build_ev_validation import is_future_resolution_hit, is_lookahead_pair, source_event_date


class TestSourceEventDate:
    """The only trustworthy event date available is the one embedded in the
    source URL — catalyst_date and resolution_date are both derived from the
    snapped value."""

    def test_extracts_globenewswire_date(self):
        rec = {
            "source_id": (
                "https://www.globenewswire.com/news-release/2026/03/28/3264202/"
                "34323/en/NEJM-Publishes-Positive-Phase-3-VALOR-Results.html"
            )
        }
        assert source_event_date(rec) == "2026-03-28"

    def test_extracts_older_date(self):
        assert source_event_date({"source_id": "https://x.com/news/2024/09/28/foo.html"}) == "2024-09-28"

    def test_absent_source_returns_none(self):
        assert source_event_date({}) is None
        assert source_event_date({"source_id": ""}) is None

    def test_non_url_source_returns_none(self):
        assert source_event_date({"source_id": "PRESS_RELEASE manual entry"}) is None

    def test_does_not_confuse_a_bare_date(self):
        """Must require the /YYYY/MM/DD/ path shape, not any digits."""
        assert source_event_date({"source_id": "https://x.com/20260328/foo"}) is None

    def test_rejects_impossible_date(self):
        assert source_event_date({"source_id": "https://x.com/2026/13/45/foo"}) is None


class TestLookaheadPair:
    def test_the_real_elvn_case_is_lookahead(self):
        """The exact pair the current guard admits."""
        rec = {"source_id": "https://x.com/2024/09/28/elvn.html", "catalyst_date": "2026-07-01"}
        assert is_lookahead_pair(rec, prediction_date="2026-05-04") is True

    def test_prediction_before_source_event_is_clean(self):
        rec = {"source_id": "https://x.com/2026/03/28/imvt.html", "catalyst_date": "2026-08-01"}
        assert is_lookahead_pair(rec, prediction_date="2026-01-15") is False

    def test_same_day_is_not_lookahead(self):
        """Same-day is boundary, not hindsight — do not over-exclude."""
        rec = {"source_id": "https://x.com/2026/03/28/x.html"}
        assert is_lookahead_pair(rec, prediction_date="2026-03-28") is False

    def test_unknown_source_date_cannot_be_judged(self):
        """No source date means no evidence of look-ahead. Must NOT exclude —
        that would silently drop most of the ledger (only 27 of 250 records
        carry a parseable source date)."""
        assert is_lookahead_pair({"source_id": ""}, prediction_date="2026-05-04") is False

    def test_missing_prediction_date_is_not_lookahead(self):
        rec = {"source_id": "https://x.com/2026/03/28/x.html"}
        assert is_lookahead_pair(rec, prediction_date="") is False
        assert is_lookahead_pair(rec, prediction_date=None) is False


class TestFutureResolutionHit:
    """14 records carry a resolution_date in the future yet outcome=HIT."""

    def test_future_hit_is_rejected(self):
        rec = {"resolution_date": "2026-09-01", "outcome": "HIT"}
        assert is_future_resolution_hit(rec, today="2026-07-29") is True

    def test_past_hit_is_fine(self):
        rec = {"resolution_date": "2026-07-01", "outcome": "HIT"}
        assert is_future_resolution_hit(rec, today="2026-07-29") is False

    def test_today_is_not_future(self):
        rec = {"resolution_date": "2026-07-29", "outcome": "HIT"}
        assert is_future_resolution_hit(rec, today="2026-07-29") is False

    def test_future_miss_is_not_flagged_by_this_guard(self):
        """Scoped deliberately: a future MISS is a different anomaly and is not
        what this guard is for."""
        rec = {"resolution_date": "2026-09-01", "outcome": "MISS"}
        assert is_future_resolution_hit(rec, today="2026-07-29") is False

    def test_unparseable_dates_are_not_flagged(self):
        assert is_future_resolution_hit({"resolution_date": "", "outcome": "HIT"}, today="2026-07-29") is False
        assert is_future_resolution_hit({"resolution_date": "nonsense", "outcome": "HIT"}, today="2026-07-29") is False
