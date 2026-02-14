"""Tests for scripts/audit_catalyst_coverage.py — compute_coverage_metrics()."""

import sys
from pathlib import Path

# Allow importing from the scripts package without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_catalyst_coverage import (
    FDA_EVENT_TYPES,
    KNOWN_EVENT_TYPES,
    KNOWN_SOURCES,
    compute_coverage_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_universe(n):
    """Return a set of N synthetic uppercase tickers: T001 .. T{n:03d}."""
    return {f"T{i:03d}" for i in range(1, n + 1)}


def _make_vnext_event(event_type, source, event_date="2026-06-01"):
    return {"event_type": event_type, "source": source, "event_date": event_date}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_universe_returns_error():
    """Empty universe short-circuits to an error dict."""
    result = compute_coverage_metrics(universe_tickers=set())
    assert result["universe_size"] == 0
    assert result["error"] == "empty_universe"
    # Should contain ONLY these two keys — no metrics to compute.
    assert set(result.keys()) == {"universe_size", "error"}


def test_basic_coverage_math():
    """10-ticker universe, 5 with events -> 50% coverage."""
    universe = _make_universe(10)
    # Give 5 tickers one vNext event each
    vnext = {}
    for i in range(1, 6):
        ticker = f"T{i:03d}"
        vnext[ticker] = {
            "events": [
                _make_vnext_event("DATA_READOUT", "CTGOV_CALENDAR"),
            ],
        }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    assert result["universe_size"] == 10
    assert result["tickers_with_any_catalyst"] == 5
    assert result["pct_with_any_catalyst"] == 50.0
    assert result["total_events"] == 5
    # All events have a date -> dated coverage also 50%
    assert result["tickers_with_dated_catalyst"] == 5
    assert result["pct_with_dated_catalyst"] == 50.0


def test_fda_event_type_counted():
    """Events with an FDA event_type appear in tickers_with_fda / pct_with_fda."""
    universe = _make_universe(4)
    vnext = {
        "T001": {"events": [_make_vnext_event("FDA_PDUFA_DATE", "FDA_CALENDAR")]},
        "T002": {"events": [_make_vnext_event("FDA_ADCOM", "FDA_ADCOM_CALENDAR")]},
    }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    assert result["tickers_with_fda"] == 2
    assert result["pct_with_fda"] == 50.0
    # Sanity: every FDA_EVENT_TYPE constant is a subset of KNOWN_EVENT_TYPES
    assert FDA_EVENT_TYPES <= KNOWN_EVENT_TYPES


def test_ctgov_source_counted():
    """Events with source=CTGOV_CALENDAR appear in tickers_with_ctgov."""
    universe = _make_universe(5)
    vnext = {
        "T001": {"events": [_make_vnext_event("DATA_READOUT", "CTGOV_CALENDAR")]},
        "T003": {"events": [_make_vnext_event("CT_PRIMARY_COMPLETION", "CTGOV_CALENDAR")]},
    }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    assert result["tickers_with_ctgov"] == 2
    assert result["pct_with_ctgov"] == 40.0


def test_unknown_type_detection():
    """An event_type not in KNOWN_EVENT_TYPES is flagged as unknown."""
    universe = _make_universe(3)
    vnext = {
        "T001": {"events": [_make_vnext_event("BOGUS", "FDA_CALENDAR")]},
        "T002": {"events": [_make_vnext_event("DATA_READOUT", "CTGOV_CALENDAR")]},
    }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    unk = result["unknown_events"]
    assert unk["unknown_type_count"] == 1
    assert unk["blank_type_count"] == 0
    # The example should reference T001 / BOGUS
    examples = unk["unknown_type_examples"]
    assert len(examples) == 1
    assert examples[0]["ticker"] == "T001"
    assert examples[0]["event_type"] == "BOGUS"


def test_blank_source_detection():
    """An event with source='' is counted as both blank and unknown source."""
    universe = _make_universe(2)
    vnext = {
        "T001": {"events": [_make_vnext_event("DATA_READOUT", "")]},
    }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    unk = result["unknown_events"]
    assert unk["blank_source_count"] == 1
    # Blank is a subset of unknown, so unknown_source_count >= blank_source_count
    assert unk["unknown_source_count"] >= 1
    # The by_source dict should use the "(blank)" placeholder for empty source
    assert "(blank)" in result["by_source"]
    assert result["by_source"]["(blank)"]["events"] == 1


def test_by_source_breakdown():
    """Verify by_source dict has correct event counts per source."""
    universe = _make_universe(5)
    vnext = {
        "T001": {
            "events": [
                _make_vnext_event("DATA_READOUT", "CTGOV_CALENDAR"),
                _make_vnext_event("FDA_PDUFA_DATE", "FDA_CALENDAR"),
            ],
        },
        "T002": {
            "events": [
                _make_vnext_event("DATA_READOUT", "CTGOV_CALENDAR"),
            ],
        },
        "T003": {
            "events": [
                _make_vnext_event("FDA_ADCOM", "FDA_ADCOM_CALENDAR"),
            ],
        },
    }

    result = compute_coverage_metrics(
        universe_tickers=universe,
        vnext_summaries=vnext,
    )

    by_src = result["by_source"]
    assert by_src["CTGOV_CALENDAR"]["events"] == 2
    assert by_src["CTGOV_CALENDAR"]["tickers"] == 2
    assert by_src["FDA_CALENDAR"]["events"] == 1
    assert by_src["FDA_CALENDAR"]["tickers"] == 1
    assert by_src["FDA_ADCOM_CALENDAR"]["events"] == 1
    assert by_src["FDA_ADCOM_CALENDAR"]["tickers"] == 1
    assert result["total_events"] == 4


def test_pdufa_and_corp_entries():
    """PDUFA and corporate catalyst entries are counted correctly."""
    universe = _make_universe(6)

    pdufa = [
        {"ticker": "T001", "drug": "DrugA"},
        {"symbol": "T002", "drug": "DrugB"},           # uses 'symbol' key
        {"ticker": "T999", "drug": "OutOfUniverse"},    # not in universe
    ]
    corp = [
        {"ticker": "T003", "event_type": "DATA_READOUT", "event_date": "2026-07-01"},
        {"ticker": "T004", "event_type": "TRIAL_ONGOING", "catalyst_date": "2026-08-15"},
        {"ticker": "T005", "event_type": "", "event_date": ""},  # blank type, no date
    ]

    result = compute_coverage_metrics(
        universe_tickers=universe,
        pdufa_entries=pdufa,
        corp_entries=corp,
    )

    # PDUFA: T001 and T002 in universe, T999 excluded
    assert sorted(result["source_ticker_lists"]["PDUFA"]) == ["T001", "T002"]
    # Corporate: T003, T004, T005 all in universe
    assert sorted(result["source_ticker_lists"]["CORPORATE"]) == ["T003", "T004", "T005"]

    # PDUFA entries always produce FDA_PDUFA_DATE + FDA_CALENDAR, always dated
    assert result["tickers_with_fda"] >= 2  # T001, T002 from PDUFA

    # T003 and T004 have dates (event_date / catalyst_date); T005 has neither
    assert result["tickers_with_dated_catalyst"] >= 4  # T001, T002 (PDUFA), T003, T004

    # Total tickers with any catalyst = 5 (T001..T005)
    assert result["tickers_with_any_catalyst"] == 5
    assert result["universe_size"] == 6

    # Corporate source should appear in by_source
    assert "CORPORATE_CALENDAR" in result["by_source"]
    assert result["by_source"]["CORPORATE_CALENDAR"]["events"] == 3

    # FDA_CALENDAR from PDUFA entries
    assert "FDA_CALENDAR" in result["by_source"]
    assert result["by_source"]["FDA_CALENDAR"]["events"] == 2


# ---------------------------------------------------------------------------
# CTGOV diagnostic fields (regression)
# ---------------------------------------------------------------------------

def test_ctgov_present_flag():
    """ctgov_present is True when CTGOV events exist, False otherwise."""
    universe = _make_universe(3)
    vnext_with = {
        "T001": {"events": [_make_vnext_event("CT_PRIMARY_COMPLETION", "CTGOV_CALENDAR")]},
    }
    vnext_without = {
        "T001": {"events": [_make_vnext_event("DATA_READOUT", "SEC_8K_FILING")]},
    }

    result_with = compute_coverage_metrics(universe_tickers=universe, vnext_summaries=vnext_with)
    result_without = compute_coverage_metrics(universe_tickers=universe, vnext_summaries=vnext_without)

    assert result_with["ctgov_present"] is True
    assert result_without["ctgov_present"] is False


def test_ctgov_by_type_breakdown():
    """ctgov_by_type has counts per CT_* event type."""
    universe = _make_universe(5)
    vnext = {
        "T001": {"events": [
            _make_vnext_event("CT_PRIMARY_COMPLETION", "CTGOV_CALENDAR"),
            _make_vnext_event("CT_STUDY_COMPLETION", "CTGOV_CALENDAR"),
        ]},
        "T002": {"events": [
            _make_vnext_event("CT_PRIMARY_COMPLETION", "CTGOV_CALENDAR"),
        ]},
    }

    result = compute_coverage_metrics(universe_tickers=universe, vnext_summaries=vnext)

    assert result["ctgov_by_type"]["CT_PRIMARY_COMPLETION"] == 2
    assert result["ctgov_by_type"]["CT_STUDY_COMPLETION"] == 1


def test_amended_form_source_is_unknown():
    """SEC_6K/A_FILING should be flagged as unknown source (regression)."""
    universe = _make_universe(2)
    multi_form = [
        {"ticker": "T001", "event_type": "DATA_READOUT",
         "source": "SEC_6K/A_FILING", "event_date": "2026-03-15"},
    ]

    result = compute_coverage_metrics(
        universe_tickers=universe,
        sec_multi_form_events=multi_form,
    )

    unk = result["unknown_events"]
    assert unk["unknown_source_count"] >= 1
    assert any(
        ex["source"] == "SEC_6K/A_FILING" for ex in unk["unknown_source_examples"]
    )


def test_multi_form_uplift_new_tickers():
    """Multi-form uplift counts tickers not already in 8-K."""
    universe = _make_universe(5)
    sec_8k = [
        {"ticker": "T001", "event_type": "DATA_READOUT", "event_date": "2026-05-01"},
    ]
    multi_form = [
        {"ticker": "T001", "event_type": "DATA_READOUT", "source": "SEC_10Q_FILING",
         "event_date": "2026-06-01"},
        {"ticker": "T002", "event_type": "DATA_READOUT", "source": "SEC_6K_FILING",
         "event_date": "2026-07-01"},
    ]

    result = compute_coverage_metrics(
        universe_tickers=universe,
        sec_8k_events=sec_8k,
        sec_multi_form_events=multi_form,
    )

    uplift = result["multi_form_uplift"]
    assert uplift["tickers_new_count"] == 1
    assert "T002" in uplift["tickers_new"]
    assert "T001" not in uplift["tickers_new"]
