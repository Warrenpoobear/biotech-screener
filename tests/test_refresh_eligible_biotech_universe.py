"""Tests for refreshing eligible biotech universe coverage status."""

from tools.refresh_eligible_biotech_universe import refresh_universe


def _row(ticker: str, **overrides):
    row = {
        "ticker": ticker,
        "name": f"{ticker} Therapeutics, Inc.",
        "sector": "Biotechnology",
        "status": "active",
        "market_data": {
            "price": 10.0,
            "market_cap": 500_000_000,
            "company_name": f"{ticker} Therapeutics, Inc.",
            "industry": "Biotechnology",
        },
        "financial_data": {"cash": 10_000_000},
    }
    row.update(overrides)
    return row


def test_refresh_keeps_fully_covered_biotech_active():
    universe = [_row("COVR")]
    trials = [{"ticker": "COVR", "interventions": ["COVR-101"]}]

    refreshed, report = refresh_universe(universe, trials, "2026-06-19")

    assert refreshed[0]["status"] == "active"
    assert report["marked_pending_count"] == 0
    assert report["promoted_active_count"] == 0


def test_refresh_marks_uncovered_biotech_pending():
    universe = [_row("MISS", name="MISS", market_data={"company_name": "Healthcare", "industry": "Biotechnology"})]
    trials = []

    refreshed, report = refresh_universe(universe, trials, "2026-06-19")

    assert refreshed[0]["status"] == "pending_data_collection"
    assert (
        refreshed[0]["status_reason"]
        == "coverage_pending:company_name,market_data,clinical_trials,scientific_cartography"
    )
    assert refreshed[0]["coverage_status"]["company_name"] == "pending"
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "pending"
    assert refreshed[0]["coverage_refreshed_as_of"] == "2026-06-19"
    assert report["marked_pending_tickers"] == ["MISS"]
    assert report["pending_collection_tickers"] == ["MISS"]
    assert report["pending_coverage_tickers"] == []


def test_refresh_promotes_pending_ticker_when_coverage_is_complete():
    universe = [_row("DONE", status="pending_data_collection", status_reason="coverage_pending:clinical_trials")]
    trials = [{"ticker": "DONE", "interventions": ["DONE-101"]}]

    refreshed, report = refresh_universe(universe, trials, "2026-06-19")

    assert refreshed[0]["status"] == "active"
    assert "status_reason" not in refreshed[0]
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "covered"
    assert report["promoted_active_tickers"] == ["DONE"]


def test_refresh_marks_trial_without_intervention_pending_for_cartography():
    universe = [_row("ILMN")]
    trials = [{"ticker": "ILMN", "interventions": []}]

    refreshed, _ = refresh_universe(universe, trials, "2026-06-19")

    assert refreshed[0]["status"] == "pending_data_collection"
    assert refreshed[0]["coverage_status"]["clinical_trials"] == "covered"
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "pending"


def test_refresh_finalize_marks_unavailable_coverage():
    universe = [_row("NTRI", status="pending_data_collection")]
    trials = []

    refreshed, report = refresh_universe(universe, trials, "2026-06-19", finalize_collection=True)

    assert refreshed[0]["status"] == "pending_coverage"
    assert refreshed[0]["status_reason"] == "coverage_unavailable:clinical_trials,scientific_cartography"
    assert refreshed[0]["coverage_status"]["clinical_trials"] == "unavailable"
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "unavailable"
    assert report["pending_collection_tickers"] == []
    assert report["pending_coverage_tickers"] == ["NTRI"]
