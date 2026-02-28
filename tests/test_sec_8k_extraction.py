"""Golden fixture tests for SEC 8-K timing/downside extraction and diagnostics."""
from __future__ import annotations

from datetime import date

import pytest

from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
    _extract_timing_events,
    _extract_downside_events,
    _strip_boilerplate,
    reset_extraction_diagnostics,
    get_extraction_diagnostics,
)

AS_OF = date(2026, 2, 28)
FILING = "2026-01-15"


# ===================================================================
# Golden extraction tests
# ===================================================================


class TestGoldenTimingExtraction:
    """Golden fixture tests for _extract_timing_events."""

    def setup_method(self):
        reset_extraction_diagnostics()

    def test_explicit_day_date(self):
        text = (
            "The FDA has assigned a PDUFA date of May 15, 2026 for "
            "the Company's NDA submission for its Phase 3 "
            "pivotal trial of XYZ-123."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "FDA_PDUFA_DATE"
        assert ev["event_date"] == "2026-05-15"
        assert ev["date_precision"] == "DAY"
        assert ev["confidence"] == "HIGH"
        assert ev["source"] == "SEC_8K_FILING"

    def test_quarter_precision(self):
        text = (
            "The Company expects topline data in Q3 2026 from its "
            "pivotal Phase 2 trial evaluating ABC-456 in patients with "
            "advanced solid tumors."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "DATA_READOUT"
        assert ev["date_precision"] == "QUARTER"
        assert ev["event_date"] == "2026-07-01"
        assert ev["event_date_end"] == "2026-09-30"
        assert ev["confidence"] == "MED"

    def test_half_year_h2(self):
        text = (
            "The Company expects topline results in the second half of "
            "2027 from its global registration-enabling study."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "DATA_READOUT"
        assert ev["date_precision"] == "HALF_YEAR"
        assert ev["event_date"] == "2027-07-01"
        assert ev["event_date_end"] == "2027-12-31"
        assert ev["confidence"] == "LOW"

    def test_year_end(self):
        text = (
            "Clinical data expected by year-end 2026 from the ongoing "
            "dose-expansion cohort in non-small cell lung cancer."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["date_precision"] == "HALF_YEAR"
        # year-end → second half
        assert ev["event_date"] == "2026-07-01"
        assert ev["event_date_end"] == "2026-12-31"

    def test_pdufa_day(self):
        text = (
            "The FDA has assigned a PDUFA date of March 15, 2026 for "
            "our NDA for drug candidate DEF-789."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "FDA_PDUFA_DATE"
        assert ev["date_precision"] == "DAY"
        assert ev["event_date"] == "2026-03-15"

    def test_adcom_day(self):
        text = (
            "An Advisory Committee meeting on July 17, 2026 has been "
            "scheduled by the FDA to discuss the Company's BLA."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "FDA_ADCOM"
        assert ev["date_precision"] == "DAY"
        assert ev["event_date"] == "2026-07-17"

    def test_negative_control_earnings(self):
        text = (
            "The Company expects revenue of $500M in Q3 2026 and "
            "adjusted EBITDA of $200M, reflecting strong commercial "
            "execution across its portfolio of marketed products."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) == 0

    def test_multiple_events_one_filing(self):
        text = (
            "The FDA has assigned a PDUFA date of March 15, 2026 for "
            "our lead product candidate. Additionally, the Company "
            "expects topline data in Q2 2026 from its Phase 3 trial "
            "of XYZ-789 in renal cell carcinoma."
        )
        events = _extract_timing_events(text, "ACME", FILING, AS_OF)
        assert len(events) >= 2
        types = {e["event_type"] for e in events}
        assert "FDA_PDUFA_DATE" in types
        assert "DATA_READOUT" in types


class TestGoldenDownsideExtraction:
    """Golden fixture tests for _extract_downside_events."""

    def test_downside_crl(self):
        text = (
            "On January 10, 2026, the Company received a Complete "
            "Response Letter from the FDA regarding its NDA for "
            "drug candidate GHI-321."
        )
        events = _extract_downside_events(text, "ACME", FILING)
        assert len(events) >= 1
        ev = events[0]
        assert ev["event_type"] == "FDA_CRL"
        assert "downside" in ev["tags"]

    def test_boilerplate_rejection(self):
        text = (
            "Forward-Looking Statements: This press release contains "
            "forward-looking statements. Risks include serious adverse "
            "events that could impact clinical development programs."
        )
        events = _extract_downside_events(text, "ACME", FILING)
        assert len(events) == 0


# ===================================================================
# Diagnostic counter tests
# ===================================================================


class TestDiagnosticCounters:
    """Tests for extraction diagnostic counters."""

    def setup_method(self):
        reset_extraction_diagnostics()

    def test_diag_counters_year_guard(self):
        text = (
            "The Company expects topline data in Q1 2035 from its "
            "Phase 3 study in advanced melanoma."
        )
        _extract_timing_events(text, "ACME", FILING, AS_OF)
        diag = get_extraction_diagnostics()
        assert diag["counters"]["year_guard_dropped"] > 0

    def test_diag_counters_staleness(self):
        # Q2 2025: year 2025 passes year guard (min=2025) but
        # end 2025-06-30 < staleness cutoff 2025-08-31
        text = (
            "The Company expects topline data in Q2 2025 from its "
            "Phase 2 trial evaluating novel therapy."
        )
        _extract_timing_events(text, "ACME", FILING, AS_OF)
        diag = get_extraction_diagnostics()
        assert diag["counters"]["staleness_dropped"] > 0

    def test_diag_reset_clears(self):
        text = (
            "The Company expects topline data in Q1 2035 from its "
            "Phase 3 study."
        )
        _extract_timing_events(text, "ACME", FILING, AS_OF)
        diag = get_extraction_diagnostics()
        assert diag["counters"]["year_guard_dropped"] > 0

        reset_extraction_diagnostics()
        diag = get_extraction_diagnostics()
        assert all(v == 0 for v in diag["counters"].values())
        assert len(diag["dropped_examples"]) == 0
