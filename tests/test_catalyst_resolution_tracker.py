"""Tests for Catalyst Resolution Tracker (Spec 042) — Phase 1 schemas + detection."""

import csv
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.catalyst_resolution_tracker import (
    ResolutionRecord,
    build_watchlist,
    check_8k_for_resolution,
    classify_outcome,
    compute_record_hash,
    get_prediction_snapshot,
)

# --- Schema tests ---


class TestResolutionRecordSchema:
    def test_valid_record_roundtrips(self):
        r = ResolutionRecord(
            ticker="PVLA",
            catalyst_date="2026-04-15",
            catalyst_type="PHASE_3_READOUT",
            catalyst_description="Phase 3 topline readout",
            resolution_date="2026-04-15",
            outcome="HIT",
            outcome_detail="Primary endpoint met",
            source_type="SEC_8K",
            source_id="8K_2026-04-15_PVLA",
            prediction_snapshot_date="2026-04-01",
            prediction_dem_rank=14,
            price_t_minus_1=12.50,
            price_t_0=18.75,
            price_t_plus_5=None,
            days_from_expected=0,
            as_of_date="2026-04-16",
        )
        d = r.to_dict()
        assert d["ticker"] == "PVLA"
        assert d["outcome"] == "HIT"
        assert d["schema_version"] == "1.0.0"

    def test_invalid_outcome_raises(self):
        try:
            ResolutionRecord(
                ticker="TEST",
                catalyst_date="2026-04-01",
                catalyst_type="PDUFA_ACTION",
                resolution_date="2026-04-01",
                outcome="INVALID",
                source_type="SEC_8K",
                source_id="test",
                as_of_date="2026-04-02",
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_catalyst_type_raises(self):
        try:
            ResolutionRecord(
                ticker="TEST",
                catalyst_date="2026-04-01",
                catalyst_type="MADE_UP_TYPE",
                resolution_date="2026-04-01",
                outcome="HIT",
                source_type="SEC_8K",
                source_id="test",
                as_of_date="2026-04-02",
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_source_type_raises(self):
        try:
            ResolutionRecord(
                ticker="TEST",
                catalyst_date="2026-04-01",
                catalyst_type="PDUFA_ACTION",
                resolution_date="2026-04-01",
                outcome="HIT",
                source_type="TWITTER",
                source_id="test",
                as_of_date="2026-04-02",
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestRecordHash:
    def test_hash_is_deterministic(self):
        r = ResolutionRecord(
            ticker="BIIB",
            catalyst_date="2026-05-24",
            catalyst_type="PDUFA_ACTION",
            resolution_date="2026-05-24",
            outcome="HIT",
            source_type="FDA_ACTION",
            source_id="fda_2026-05-24_BIIB",
            as_of_date="2026-05-25",
        )
        h1 = compute_record_hash(r)
        h2 = compute_record_hash(r)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_different_outcomes_different_hash(self):
        base = dict(
            ticker="TEST",
            catalyst_date="2026-04-01",
            catalyst_type="PDUFA_ACTION",
            resolution_date="2026-04-01",
            source_type="SEC_8K",
            source_id="test",
            as_of_date="2026-04-02",
        )
        r1 = ResolutionRecord(outcome="HIT", **base)
        r2 = ResolutionRecord(outcome="MISS", **base)
        assert compute_record_hash(r1) != compute_record_hash(r2)


# --- Watchlist tests ---


class TestBuildWatchlist:
    def _make_event(self, ticker, catalyst_date_str, catalyst_type="PHASE_3_READOUT"):
        return {
            "ticker": ticker,
            "catalyst_date": catalyst_date_str,
            "catalyst_type": catalyst_type,
        }

    def test_includes_events_in_window(self):
        events = [
            self._make_event("PVLA", "2026-04-10"),  # 5 days ago from as_of
            self._make_event("BIIB", "2026-04-20"),  # 5 days in future
        ]
        wl = build_watchlist(events, date(2026, 4, 15), existing_resolutions=set())
        tickers = {w["ticker"] for w in wl}
        assert "PVLA" in tickers
        assert "BIIB" in tickers

    def test_excludes_events_outside_window(self):
        events = [
            self._make_event("OLD", "2025-01-01"),  # way too old
            self._make_event("FAR", "2027-01-01"),  # way too far
        ]
        wl = build_watchlist(events, date(2026, 4, 15), existing_resolutions=set())
        assert len(wl) == 0

    def test_excludes_already_resolved(self):
        events = [self._make_event("PVLA", "2026-04-10")]
        resolved = {("PVLA", "2026-04-10")}
        wl = build_watchlist(events, date(2026, 4, 15), existing_resolutions=resolved)
        assert len(wl) == 0

    def test_window_boundaries(self):
        # T-30 (just inside), T-31 (just outside), T+7 (just inside), T+8 (just outside)
        as_of = date(2026, 4, 15)
        events = [
            self._make_event("A", "2026-03-16"),  # -30d, inside
            self._make_event("B", "2026-03-15"),  # -31d, outside
            self._make_event("C", "2026-04-22"),  # +7d, inside
            self._make_event("D", "2026-04-23"),  # +8d, outside
        ]
        wl = build_watchlist(events, as_of, existing_resolutions=set())
        tickers = {w["ticker"] for w in wl}
        assert "A" in tickers
        assert "B" not in tickers
        assert "C" in tickers
        assert "D" not in tickers


# --- Outcome classification tests ---


class TestClassifyOutcome:
    def test_pdufa_approved(self):
        assert classify_outcome("PDUFA_ACTION", fda_action="APPROVED") == "HIT"

    def test_pdufa_crl(self):
        assert classify_outcome("PDUFA_ACTION", fda_action="CRL") == "MISS"

    def test_pdufa_unknown_action(self):
        assert classify_outcome("PDUFA_ACTION", fda_action=None) == "NEEDS_REVIEW"

    def test_phase3_positive_headline(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Company announces positive topline results from Phase 3 trial",
            )
            == "HIT"
        )

    def test_phase3_met_endpoint(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Trial met primary endpoint with statistical significance",
            )
            == "HIT"
        )

    def test_phase3_failed(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Phase 3 trial did not meet primary endpoint",
            )
            == "MISS"
        )

    def test_phase3_discontinued(self):
        assert (
            classify_outcome(
                "PHASE_2_READOUT",
                headline="Company announces discontinuation of Phase 2 program",
            )
            == "MISS"
        )

    def test_ambiguous_headline_now_informational(self):
        # "business update" and "financial results" are informational keywords
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Company provides business update and financial results",
            )
            == "INFORMATIONAL"
        )

    def test_ctgov_completed(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                ctgov_status_from="ACTIVE_NOT_RECRUITING",
                ctgov_status_to="COMPLETED",
            )
            == "NEEDS_REVIEW"
        )
        # CT.gov completion alone doesn't tell us HIT/MISS — need headline

    def test_ctgov_terminated(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                ctgov_status_from="RECRUITING",
                ctgov_status_to="TERMINATED",
            )
            == "MISS"
        )

    def test_ctgov_withdrawn(self):
        assert (
            classify_outcome(
                "PHASE_2_READOUT",
                ctgov_status_from="NOT_YET_RECRUITING",
                ctgov_status_to="WITHDRAWN",
            )
            == "MISS"
        )


class TestClassifyInformational:
    def test_data_expected_is_informational(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="8-K: data expected in the second quarter of 2026",
            )
            == "INFORMATIONAL"
        )

    def test_enrollment_continuing(self):
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Phase 3 LUCIDITY trial underway. Enrollment continuing",
            )
            == "INFORMATIONAL"
        )

    def test_bla_submission_anticipated(self):
        assert (
            classify_outcome(
                "PDUFA_ACTION",
                headline="BLA submission anticipated in 1H 2026",
            )
            == "INFORMATIONAL"
        )

    def test_corporate_update(self):
        assert (
            classify_outcome(
                "CORPORATE_UPDATE",
                headline="Company provides business update and financial results",
            )
            == "INFORMATIONAL"
        )

    def test_regulatory_options_discussion(self):
        assert (
            classify_outcome(
                "PHASE_2_READOUT",
                headline="Type C meeting to discuss regulatory options to accelerate the development program",
            )
            == "INFORMATIONAL"
        )


class TestClassifySafetyMiss:
    def test_serious_adverse_event_is_needs_review(self):
        # Bug fix (issue #514): bare SAE mention is not sufficient evidence
        # of a binary MISS — route to NEEDS_REVIEW for human adjudication.
        assert (
            classify_outcome(
                "PHASE_2_READOUT",
                headline="Company reports serious adverse event in Phase 2 trial",
            )
            == "NEEDS_REVIEW"
        )

    def test_safety_signal_headline_is_needs_review(self):
        # Bug fix (issue #514): bare safety signal headline is not a MISS.
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="Safety signal observed in ongoing trial",
            )
            == "NEEDS_REVIEW"
        )

    def test_clinical_hold_remains_miss(self):
        # Clinical hold is actionable and definitive — remains MISS.
        assert (
            classify_outcome(
                "PHASE_3_READOUT",
                headline="FDA places clinical hold on Phase 3 program",
            )
            == "MISS"
        )


class TestCheck8kSkipsSafetySignal:
    def _make_8k_event(
        self, ticker, event_date, event_type="DATA_READOUT", confidence="HIGH", name="Positive topline data"
    ):
        return {
            "ticker": ticker,
            "event_date": event_date,
            "disclosed_at": event_date,
            "event_type": event_type,
            "confidence": confidence,
            "event_name": name,
        }

    def test_safety_signal_low_confidence_skipped(self):
        # Bug fix (issue #514): low-confidence SAFETY_SIGNAL 8-Ks must not
        # be treated as resolution signals.
        events = [
            self._make_8k_event(
                "CATX", "2026-03-16", event_type="SAFETY_SIGNAL", confidence="MED", name="8-K: serious adverse event"
            ),
        ]
        result = check_8k_for_resolution("CATX", events, date(2026, 3, 16), date(2026, 3, 31))
        assert result is None

    def test_safety_signal_high_confidence_accepted(self):
        # HIGH-confidence SAFETY_SIGNAL (e.g. DSMB halt) should still resolve.
        events = [
            self._make_8k_event(
                "XENE",
                "2026-03-09",
                event_type="SAFETY_SIGNAL",
                confidence="HIGH",
                name="DSMB recommended halt due to safety",
            ),
        ]
        result = check_8k_for_resolution("XENE", events, date(2026, 3, 9), date(2026, 3, 31))
        assert result is not None

    def test_non_safety_signal_unaffected(self):
        # Normal DATA_READOUT events should still be returned as before.
        events = [
            self._make_8k_event(
                "PVLA",
                "2026-03-31",
                event_type="DATA_READOUT",
                confidence="HIGH",
                name="Phase 3 SELVA met primary endpoint",
            ),
        ]
        result = check_8k_for_resolution("PVLA", events, date(2026, 3, 31), date(2026, 4, 15))
        assert result is not None
        assert "SELVA" in result["headline"]


class TestSafetySignalTypeMap:
    def test_safety_signal_not_in_watchlist_as_corporate_update(self):
        # Bug fix (issue #514): SAFETY_SIGNAL events must not appear in the
        # watchlist calendar as CORPORATE_UPDATE.
        events = [
            {
                "ticker": "CATX",
                "event_date": "2026-03-16",
                "catalyst_date": "2026-03-16",
                "catalyst_type": "SAFETY_SIGNAL",
                "description": "8-K: serious adverse event",
            }
        ]
        wl = build_watchlist(events, date(2026, 3, 31), existing_resolutions=set())
        corporate_update_entries = [
            w for w in wl if w.get("catalyst_type") == "CORPORATE_UPDATE" and w["ticker"] == "CATX"
        ]
        assert corporate_update_entries == []


# --- get_prediction_snapshot tests (Spec 073) ---


def _write_snap(tmp_path: Path, snap_date: str, rows: list[dict]) -> None:
    snap_dir = tmp_path / snap_date
    snap_dir.mkdir(parents=True)
    csv_path = snap_dir / "rankings.csv"
    fieldnames = ["ticker", "actionable_rank", "tier_any", "composite_score"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


class TestGetPredictionSnapshot:
    def test_exact_match_returns_exact_type(self, tmp_path):
        _write_snap(
            tmp_path,
            "2026-03-28",
            [{"ticker": "XENE", "actionable_rank": "12", "tier_any": "T1", "composite_score": "0.55"}],
        )
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "OK"
        assert result["match_type"] == "exact"
        assert result["snapshot_date"] == "2026-03-28"
        assert float(result["composite_score"]) == pytest.approx(0.55)

    def test_fallback_when_ticker_absent_from_immediate_prior(self, tmp_path):
        # 2026-03-29: ticker missing; 2026-03-27: ticker present
        _write_snap(
            tmp_path,
            "2026-03-29",
            [{"ticker": "OTHER", "actionable_rank": "1", "tier_any": "T1", "composite_score": "0.10"}],
        )
        _write_snap(
            tmp_path,
            "2026-03-27",
            [{"ticker": "XENE", "actionable_rank": "15", "tier_any": "T2", "composite_score": "0.42"}],
        )
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "OK"
        assert result["match_type"] == "fallback"
        assert result["snapshot_date"] == "2026-03-27"

    def test_missing_snapshot_when_no_dirs(self, tmp_path):
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "MISSING_SNAPSHOT"

    def test_ticker_not_in_snapshot_within_window(self, tmp_path):
        _write_snap(
            tmp_path,
            "2026-03-29",
            [{"ticker": "OTHER", "actionable_rank": "1", "tier_any": "T1", "composite_score": "0.10"}],
        )
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "TICKER_NOT_IN_SNAPSHOT"

    def test_snapshot_on_same_day_excluded(self, tmp_path):
        # Snapshot dated 2026-03-31 must not be used for catalyst_date 2026-03-31
        _write_snap(
            tmp_path,
            "2026-03-31",
            [{"ticker": "XENE", "actionable_rank": "5", "tier_any": "T1", "composite_score": "0.70"}],
        )
        _write_snap(
            tmp_path,
            "2026-03-29",
            [{"ticker": "XENE", "actionable_rank": "5", "tier_any": "T1", "composite_score": "0.65"}],
        )
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "OK"
        assert result["snapshot_date"] == "2026-03-29"

    def test_beyond_lookback_window_returns_missing(self, tmp_path):
        # Snapshot is 20 days before catalyst_date (> 14-day default window)
        _write_snap(
            tmp_path,
            "2026-03-10",
            [{"ticker": "XENE", "actionable_rank": "8", "tier_any": "T1", "composite_score": "0.30"}],
        )
        result = get_prediction_snapshot("XENE", date(2026, 3, 31), tmp_path)
        assert result["status"] == "MISSING_SNAPSHOT"

    def test_prediction_match_type_in_resolution_record(self):
        r = ResolutionRecord(
            ticker="XENE",
            catalyst_date="2026-03-31",
            catalyst_type="PHASE_3_READOUT",
            resolution_date="2026-03-31",
            outcome="HIT",
            source_type="PRESS_RELEASE",
            source_id="test",
            as_of_date="2026-04-01",
            prediction_match_type="fallback",
        )
        d = r.to_dict()
        assert d["prediction_match_type"] == "fallback"

    def test_prediction_match_type_defaults_to_none(self):
        r = ResolutionRecord(
            ticker="XENE",
            catalyst_date="2026-03-31",
            catalyst_type="PHASE_3_READOUT",
            resolution_date="2026-03-31",
            outcome="HIT",
            source_type="PRESS_RELEASE",
            source_id="test",
            as_of_date="2026-04-01",
        )
        assert r.prediction_match_type is None


# ---------------------------------------------------------------------------
# Tests for _bind_event_ev_p_hit (spec_077)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from tools.catalyst_resolution_tracker import _bind_event_ev_p_hit


def _make_ev_artifact(tmp_dir: Path, asof: str, events: list) -> Path:
    """Write a minimal event_ev_full artifact and return its path."""
    fp = tmp_dir / f"{asof}_event_ev_full.json"
    fp.write_text(json.dumps({"as_of_date": asof, "n_events": len(events), "events": events}))
    return fp


def _ev_event(ticker: str, expected_date: str, node_id: str = "abc123", p_hit: float = 0.62) -> dict:
    return {
        "node": {"node_id": node_id, "ticker": ticker, "expected_date": expected_date},
        "outcome": {"p_hit": p_hit, "p_miss": 0.25, "p_mixed": 0.13, "confidence": 0.70},
    }


class TestBindEventEvPHit:
    def test_date_fallback_exact_day(self, tmp_path):
        """Match when catalyst_date == expected_date (0-day distance)."""
        _make_ev_artifact(tmp_path, "2026-04-14", [_ev_event("BEAM", "2026-04-15", p_hit=0.58)])
        result = _bind_event_ev_p_hit("BEAM", "2026-04-15", "2026-04-14", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "ticker_date_7d"
        assert result["event_ev_p_hit"] == pytest.approx(0.58)
        assert result["event_ev_p_miss"] == pytest.approx(0.25)
        assert result["event_ev_confidence"] == pytest.approx(0.70)
        assert result["event_ev_asof_date"] == "2026-04-14"
        assert result["event_ev_node_id"] == "abc123"

    def test_date_fallback_within_window(self, tmp_path):
        """Match when |expected_date - catalyst_date| <= 7 days."""
        _make_ev_artifact(tmp_path, "2026-04-10", [_ev_event("RVMD", "2026-04-16", p_hit=0.72)])
        result = _bind_event_ev_p_hit("RVMD", "2026-04-13", "2026-04-10", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "ticker_date_7d"
        assert result["event_ev_p_hit"] == pytest.approx(0.72)

    def test_ambiguous_leaves_null(self, tmp_path):
        """Two events for same ticker within window → ambiguous → all None."""
        _make_ev_artifact(
            tmp_path,
            "2026-04-10",
            [
                _ev_event("DNLI", "2026-04-13", node_id="aaa", p_hit=0.50),
                _ev_event("DNLI", "2026-04-15", node_id="bbb", p_hit=0.60),
            ],
        )
        result = _bind_event_ev_p_hit("DNLI", "2026-04-14", "2026-04-10", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "ambiguous"
        assert result["event_ev_p_hit"] is None

    def test_no_match_outside_window(self, tmp_path):
        """expected_date > 7 days from catalyst_date → no_match."""
        _make_ev_artifact(tmp_path, "2026-04-10", [_ev_event("CATX", "2026-05-15", p_hit=0.55)])
        result = _bind_event_ev_p_hit("CATX", "2026-04-13", "2026-04-10", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "no_match"
        assert result["event_ev_p_hit"] is None

    def test_no_snap_date_leaves_null(self, tmp_path):
        """No prediction_snapshot_date → no_snap, all fields None."""
        result = _bind_event_ev_p_hit("BEAM", "2026-04-15", None, ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "no_snap"
        assert result["event_ev_p_hit"] is None

    def test_missing_ev_artifact_dir(self, tmp_path):
        """EV artifacts dir does not exist → no_ev_artifact."""
        result = _bind_event_ev_p_hit("BEAM", "2026-04-15", "2026-04-14", ev_artifacts_dir=tmp_path / "nonexistent")
        assert result["event_ev_match_type"] == "no_ev_artifact"
        assert result["event_ev_p_hit"] is None

    def test_no_artifact_before_snap_date(self, tmp_path):
        """EV artifact exists but is after snap date → not used → no_ev_artifact."""
        _make_ev_artifact(tmp_path, "2026-04-20", [_ev_event("BEAM", "2026-04-21")])
        result = _bind_event_ev_p_hit("BEAM", "2026-04-21", "2026-04-14", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "no_ev_artifact"

    def test_wrong_ticker_no_match(self, tmp_path):
        """EV has event for different ticker → no match."""
        _make_ev_artifact(tmp_path, "2026-04-10", [_ev_event("RVMD", "2026-04-13")])
        result = _bind_event_ev_p_hit("CATX", "2026-04-13", "2026-04-10", ev_artifacts_dir=tmp_path)
        assert result["event_ev_match_type"] == "no_match"

    def test_resolution_record_carries_ev_fields(self):
        """ResolutionRecord to_dict includes all seven event_ev_* fields."""
        r = ResolutionRecord(
            ticker="BEAM",
            catalyst_date="2026-04-15",
            catalyst_type="PHASE_3_READOUT",
            outcome="HIT",
            source_type="SEC_8K",
            source_id="test",
            as_of_date="2026-04-16",
            event_ev_node_id="abc123",
            event_ev_p_hit=0.58,
            event_ev_p_miss=0.25,
            event_ev_p_mixed=0.17,
            event_ev_confidence=0.70,
            event_ev_asof_date="2026-04-14",
            event_ev_match_type="ticker_date_7d",
        )
        d = r.to_dict()
        assert d["event_ev_node_id"] == "abc123"
        assert d["event_ev_p_hit"] == pytest.approx(0.58)
        assert d["event_ev_p_miss"] == pytest.approx(0.25)
        assert d["event_ev_p_mixed"] == pytest.approx(0.17)
        assert d["event_ev_confidence"] == pytest.approx(0.70)
        assert d["event_ev_asof_date"] == "2026-04-14"
        assert d["event_ev_match_type"] == "ticker_date_7d"

    def test_resolution_record_ev_fields_default_none(self):
        """New fields default to None when not supplied."""
        r = ResolutionRecord(
            ticker="BEAM",
            catalyst_date="2026-04-15",
            catalyst_type="PHASE_3_READOUT",
            outcome="HIT",
            source_type="SEC_8K",
            source_id="test",
            as_of_date="2026-04-16",
        )
        d = r.to_dict()
        for field in (
            "event_ev_node_id",
            "event_ev_p_hit",
            "event_ev_p_miss",
            "event_ev_p_mixed",
            "event_ev_confidence",
            "event_ev_asof_date",
            "event_ev_match_type",
        ):
            assert d[field] is None, f"{field} should default to None"

    def test_exact_node_id_match(self, tmp_path):
        """node_id exact match takes priority; skips date check; returns 'exact_node'."""
        _make_ev_artifact(
            tmp_path,
            "2026-04-14",
            [_ev_event("BIIB", "2026-05-24", node_id="deadbeef0000", p_hit=0.81)],
        )
        # Pass node_id directly — expected_date is 40d away, well outside ±7d fallback
        result = _bind_event_ev_p_hit(
            "BIIB", "2026-04-10", "2026-04-14", node_id="deadbeef0000", ev_artifacts_dir=tmp_path
        )
        assert result["event_ev_match_type"] == "exact_node"
        assert result["event_ev_node_id"] == "deadbeef0000"
        assert result["event_ev_p_hit"] == pytest.approx(0.81)
        assert result["event_ev_asof_date"] == "2026-04-14"

    def test_exact_node_id_not_found_falls_through_to_date(self, tmp_path):
        """Unknown node_id falls through to ticker+date matching."""
        _make_ev_artifact(tmp_path, "2026-04-14", [_ev_event("RVMD", "2026-04-15", node_id="known", p_hit=0.70)])
        result = _bind_event_ev_p_hit(
            "RVMD", "2026-04-15", "2026-04-14", node_id="unknown_id", ev_artifacts_dir=tmp_path
        )
        # node_id miss → fallback to date; date matches (0d)
        assert result["event_ev_match_type"] == "ticker_date_7d"
        assert result["event_ev_p_hit"] == pytest.approx(0.70)

    def test_smoke_real_ev_artifact(self):
        """Smoke: bind BIIB against the real 2026-05-05 EV artifact using exact node_id."""
        from pathlib import Path as _Path

        import pytest as _pytest

        ev_dir = _Path(__file__).resolve().parent.parent / "artifacts" / "event_ev"
        if not ev_dir.exists():
            _pytest.skip("EV artifact dir not present")

        # BIIB node_id verified 2026-05-05: expected_date 2026-05-24, p_hit 0.8111
        result = _bind_event_ev_p_hit(
            "BIIB", "2026-05-24", "2026-05-05", node_id="4ae53493f7ee", ev_artifacts_dir=ev_dir
        )
        assert result["event_ev_match_type"] == "exact_node", result
        assert result["event_ev_node_id"] == "4ae53493f7ee"
        assert result["event_ev_p_hit"] is not None
        assert 0.0 < result["event_ev_p_hit"] <= 1.0
        assert result["event_ev_asof_date"] == "2026-05-05"
