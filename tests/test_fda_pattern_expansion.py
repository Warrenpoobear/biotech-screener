#!/usr/bin/env python3
"""
Unit tests for expanded FDA patterns in SEC 8-K catalyst collector.

Tests _extract_timing_events and _extract_downside_events directly against
new FDA-specific regex patterns added for PDUFA/ADCOM capture uplift.
"""

from datetime import date

import pytest

from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
    PATTERN_VERSION,
    _extract_timing_events,
    _extract_downside_events,
)

# Common test fixtures
AS_OF = date(2026, 2, 13)
FILING_DATE = "2026-01-15"


class TestFDATimingPatterns:
    """Tests for new FDA timing patterns in TIMING_PATTERNS."""

    def test_fda_action_date_day(self):
        """'FDA action date of March 15, 2026' → FDA_PDUFA_DATE, DAY."""
        text = "The Company announced that the FDA action date of March 15, 2026 has been confirmed."
        events = _extract_timing_events(text, "AAAA", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "DAY"
        assert fda[0]["event_date"] == "2026-03-15"
        assert fda[0]["confidence"] == "HIGH"

    def test_fda_target_action_date(self):
        """'FDA target action date set for June 30, 2026' → FDA_PDUFA_DATE, DAY."""
        text = "The FDA target action date set for June 30, 2026 under the PDUFA."
        events = _extract_timing_events(text, "BBBB", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["event_date"] == "2026-06-30"

    def test_nda_accepted_pdufa(self):
        """'NDA accepted...PDUFA date of April 3, 2026' → FDA_PDUFA_DATE, DAY."""
        text = "The NDA was accepted for review with a PDUFA date of April 3, 2026."
        events = _extract_timing_events(text, "CCCC", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "DAY"
        assert fda[0]["event_date"] == "2026-04-03"

    def test_bla_pdufa_date(self):
        """'BLA...PDUFA date is August 22, 2026' → FDA_PDUFA_DATE, DAY."""
        text = "Following BLA acceptance, the PDUFA date is August 22, 2026."
        events = _extract_timing_events(text, "DDDD", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["event_date"] == "2026-08-22"

    def test_pdufa_full_name(self):
        """'Prescription Drug User Fee Act date of March 15, 2026' → FDA_PDUFA_DATE, DAY."""
        text = "The Prescription Drug User Fee Act date of March 15, 2026 is the target."
        events = _extract_timing_events(text, "EEEE", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "DAY"
        assert fda[0]["event_date"] == "2026-03-15"

    def test_regulatory_decision_quarter(self):
        """'regulatory decision expected in Q2 2026' → FDA_PDUFA_DATE, QUARTER."""
        text = "A regulatory decision is expected in Q2 2026 for the lead compound."
        events = _extract_timing_events(text, "FFFF", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "QUARTER"
        assert fda[0]["event_date"] == "2026-04-01"
        assert fda[0]["event_date_end"] == "2026-06-30"

    def test_fda_approval_quarter(self):
        """'FDA approval expected in Q2 2026' → FDA_PDUFA_DATE, QUARTER."""
        text = "FDA approval is expected in Q2 2026 based on the review timeline."
        events = _extract_timing_events(text, "GGGG", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "QUARTER"
        assert fda[0]["confidence"] == "MED"

    def test_fda_approval_half_year(self):
        """'approval anticipated second half of 2026' → FDA_PDUFA_DATE, HALF_YEAR."""
        text = "FDA approval is anticipated in the second half of 2026."
        events = _extract_timing_events(text, "HHHH", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "HALF_YEAR"
        assert fda[0]["event_date"] == "2026-07-01"
        assert fda[0]["event_date_end"] == "2026-12-31"
        assert fda[0]["confidence"] == "LOW"

    def test_advisory_committee_day(self):
        """'Advisory Committee meeting on July 17, 2026' → FDA_ADCOM, DAY."""
        text = "The FDA Advisory Committee meeting on July 17, 2026 will review the application."
        events = _extract_timing_events(text, "IIII", FILING_DATE, AS_OF)
        adcom = [e for e in events if e["event_type"] == "FDA_ADCOM"]
        assert len(adcom) >= 1
        assert adcom[0]["date_precision"] == "DAY"
        assert adcom[0]["event_date"] == "2026-07-17"

    def test_adcom_quarter(self):
        """'ADCOM meeting expected in Q3 2026' → FDA_ADCOM, QUARTER."""
        text = "An ADCOM meeting is expected in Q3 2026 to discuss the filing."
        events = _extract_timing_events(text, "JJJJ", FILING_DATE, AS_OF)
        adcom = [e for e in events if e["event_type"] == "FDA_ADCOM"]
        assert len(adcom) >= 1
        assert adcom[0]["date_precision"] == "QUARTER"
        assert adcom[0]["event_date"] == "2026-07-01"
        assert adcom[0]["event_date_end"] == "2026-09-30"

    def test_no_false_positive_boilerplate(self):
        """FDA mentions in forward-looking disclaimers NOT matched."""
        text = (
            "Some preamble text about the company. "
            "Forward-looking statements: This press release contains forward-looking "
            "statements. There can be no assurance that FDA approval will be obtained. "
            "FDA action date risks include regulatory decision expected in Q1 2027."
        )
        events = _extract_timing_events(text, "KKKK", FILING_DATE, AS_OF)
        # _extract_timing_events does not strip boilerplate (that's done in the
        # calling code flow), but a boilerplate section should still not produce
        # false positives if the text is pre-stripped. Verify the full text
        # produces events (since boilerplate stripping is upstream).
        # Here we test that the _strip_boilerplate path works:
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
            _strip_boilerplate,
        )
        clean = _strip_boilerplate(text)
        events_clean = _extract_timing_events(clean, "KKKK", FILING_DATE, AS_OF)
        fda = [e for e in events_clean if e["event_type"] in ("FDA_PDUFA_DATE", "FDA_ADCOM")]
        assert len(fda) == 0, f"Should not match FDA patterns in boilerplate, got: {fda}"

    def test_existing_pdufa_pattern_still_works(self):
        """Regression: original PDUFA pattern 'PDUFA date of ...' still matches."""
        text = "The PDUFA date of September 10, 2026 has been assigned."
        events = _extract_timing_events(text, "LLLL", FILING_DATE, AS_OF)
        fda = [e for e in events if e["event_type"] == "FDA_PDUFA_DATE"]
        assert len(fda) >= 1
        assert fda[0]["date_precision"] == "DAY"
        assert fda[0]["event_date"] == "2026-09-10"


class TestFDADownsidePatterns:
    """Tests for new FDA downside patterns in DOWNSIDE_PATTERNS."""

    def test_refuse_to_file_downside(self):
        """'refused to file' → FDA_RTF, HIGH confidence."""
        text = "The FDA refused to file the Company's NDA submission for drug X."
        events = _extract_downside_events(text, "MMMM", FILING_DATE)
        rtf = [e for e in events if e["event_type"] == "FDA_RTF"]
        assert len(rtf) == 1
        assert rtf[0]["confidence"] == "HIGH"
        assert "sec_8k" in rtf[0]["tags"]
        assert "downside" in rtf[0]["tags"]

    def test_refuse_to_file_capitalized(self):
        """'Refuse to File' (capitalized) → FDA_RTF."""
        text = "The Company received a Refuse to File letter from the FDA."
        events = _extract_downside_events(text, "NNNN", FILING_DATE)
        rtf = [e for e in events if e["event_type"] == "FDA_RTF"]
        assert len(rtf) == 1

    def test_fda_warning_letter(self):
        """'FDA warning letter' → FDA_WARNING_LETTER, MED confidence."""
        text = "The Company disclosed receiving an FDA warning letter regarding manufacturing."
        events = _extract_downside_events(text, "OOOO", FILING_DATE)
        wl = [e for e in events if e["event_type"] == "FDA_WARNING_LETTER"]
        assert len(wl) == 1
        assert wl[0]["confidence"] == "MED"

    def test_warning_letter_in_boilerplate_filtered(self):
        """FDA warning letter after boilerplate marker is stripped."""
        text = (
            "Some real content here. "
            "Forward-looking statements: risks include FDA warning letter "
            "and other regulatory actions."
        )
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
            _strip_boilerplate,
        )
        clean = _strip_boilerplate(text)
        events = _extract_downside_events(clean, "PPPP", FILING_DATE)
        wl = [e for e in events if e["event_type"] == "FDA_WARNING_LETTER"]
        assert len(wl) == 0


class TestPatternVersion:
    """Verify PATTERN_VERSION changed from pre-expansion value."""

    def test_pattern_version_changes(self):
        """PATTERN_VERSION differs from pre-expansion value (221244b7)."""
        OLD_PATTERN_VERSION = "221244b7"
        assert PATTERN_VERSION != OLD_PATTERN_VERSION, (
            f"PATTERN_VERSION should have changed after adding new patterns, "
            f"still {PATTERN_VERSION}"
        )

    def test_pattern_version_is_hex_string(self):
        """PATTERN_VERSION is an 8-char hex string."""
        assert len(PATTERN_VERSION) == 8
        int(PATTERN_VERSION, 16)  # Should not raise


class TestCacheFallback:
    """Tests for cache discoverability: primary dir → fallback dir."""

    def test_8k_cache_found_in_primary(self, tmp_path):
        """8-K collection uses primary cache dir when file exists there."""
        import json
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
            _versioned_cache_path,
            collect_8k_timing_events,
        )

        cache_dir = tmp_path / "primary"
        cache_dir.mkdir()

        as_of = date(2026, 2, 7)
        cached = [{"ticker": "TEST", "event_type": "DATA_READOUT",
                    "event_date": "2026-06-01"}]
        cache_path = _versioned_cache_path(cache_dir, as_of)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        result = collect_8k_timing_events(
            universe=[{"ticker": "TEST"}],
            as_of_date=as_of,
            cache_dir=cache_dir,
        )
        assert len(result) == 1
        assert result[0]["ticker"] == "TEST"

    def test_multi_form_cache_found_in_primary(self, tmp_path):
        """Multi-form collection uses primary cache dir when file exists there."""
        import json
        from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import (
            _multi_form_cache_path,
            collect_sec_filing_events,
        )

        cache_dir = tmp_path / "primary"
        cache_dir.mkdir()

        as_of = date(2026, 2, 7)
        cached = [{"ticker": "FOLD", "event_type": "FDA_PDUFA_DATE",
                    "event_date": "2026-09-01", "filing_form": "10-Q",
                    "source": "SEC_10Q_FILING"}]
        cache_path = _multi_form_cache_path(cache_dir, as_of)
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        result = collect_sec_filing_events(
            universe=[{"ticker": "FOLD"}],
            as_of_date=as_of,
            cache_dir=cache_dir,
        )
        assert len(result) == 1
        assert result[0]["ticker"] == "FOLD"
        assert result[0]["filing_form"] == "10-Q"
