"""Tests for targeted pending biotech data fetch/merge."""

from tools.fetch_pending_biotech_data import fetch_pending_data


def _pending_row(ticker: str):
    return {
        "ticker": ticker,
        "name": ticker,
        "sector": "Biotechnology",
        "status": "pending_data_collection",
        "market_data": {"industry": "Biotechnology"},
    }


def test_fetch_pending_data_promotes_fully_covered_row():
    universe = [_pending_row("DONE")]
    trials = []

    def market_fetcher(ticker, as_of_date):
        assert ticker == "DONE"
        return {
            "price": 12.0,
            "market_cap": 600_000_000,
            "company_name": "Done Therapeutics, Inc.",
            "industry": "Biotechnology",
            "collected_at": as_of_date,
        }

    def financial_fetcher(row, ticker, as_of_date):
        return {"ticker": ticker, "cash": 10, "assets": 20, "collected_at": as_of_date}

    def trial_fetcher(ticker):
        return [
            {
                "ticker": ticker,
                "nct_id": "NCT00000001",
                "conditions": ["Disease"],
                "interventions": ["DONE-101"],
            }
        ]

    refreshed, merged_trials, report = fetch_pending_data(
        universe,
        trials,
        "2026-06-19",
        market_fetcher=market_fetcher,
        financial_fetcher=financial_fetcher,
        trial_fetcher=trial_fetcher,
        fallback_trial_fetcher=lambda row, ticker: [],
        sleep_seconds=0,
    )

    assert refreshed[0]["status"] == "active"
    assert refreshed[0]["name"] == "Done Therapeutics, Inc."
    assert refreshed[0]["financial_data"]["cash"] == 10
    assert len(merged_trials) == 1
    assert report["trial_records_added"] == 1
    assert report["refresh_report"]["promoted_active_tickers"] == ["DONE"]


def test_fetch_pending_data_keeps_row_pending_without_trials():
    universe = [_pending_row("MISS")]
    trials = []

    refreshed, merged_trials, report = fetch_pending_data(
        universe,
        trials,
        "2026-06-19",
        market_fetcher=lambda ticker, as_of_date: {
            "price": 12.0,
            "market_cap": 600_000_000,
            "company_name": "Miss Therapeutics, Inc.",
            "industry": "Biotechnology",
        },
        financial_fetcher=lambda row, ticker, as_of_date: {"ticker": ticker, "cash": 1},
        trial_fetcher=lambda ticker: [],
        fallback_trial_fetcher=lambda row, ticker: [],
        sleep_seconds=0,
    )

    assert refreshed[0]["status"] == "pending_coverage"
    assert refreshed[0]["coverage_status"]["clinical_trials"] == "unavailable"
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "unavailable"
    assert merged_trials == []
    assert report["trial_success"] == []


def test_fetch_pending_data_uses_company_name_trial_fallback():
    universe = [_pending_row("FALL")]
    universe[0]["company_name"] = "Fallback Therapeutics, Inc."
    trials = []

    refreshed, merged_trials, report = fetch_pending_data(
        universe,
        trials,
        "2026-06-19",
        market_fetcher=lambda ticker, as_of_date: {
            "price": 12.0,
            "market_cap": 600_000_000,
            "company_name": "Fallback Therapeutics, Inc.",
            "industry": "Biotechnology",
        },
        financial_fetcher=lambda row, ticker, as_of_date: {"ticker": ticker, "cash": 1},
        trial_fetcher=lambda ticker: [],
        fallback_trial_fetcher=lambda row, ticker: [
            {
                "ticker": ticker,
                "nct_id": "NCT00000002",
                "conditions": ["Disease"],
                "interventions": ["FALL-101"],
            }
        ],
        sleep_seconds=0,
    )

    assert refreshed[0]["status"] == "active"
    assert len(merged_trials) == 1
    assert report["trial_success"] == ["FALL"]


def test_universe_write_never_emits_nan():
    """2026-11-16: yfinance returned NaN as KALA's last close; the file then held `"price": NaN`,
    which orjson (run_screen's loader) rejects, so the daily screen could not start."""
    import math

    import orjson

    from tools.fetch_pending_biotech_data import dumps_universe_json

    rows = [{"ticker": "KALA", "market_data": {"price": float("nan"), "market_cap": 2684364, "x": [math.inf]}}]
    text = dumps_universe_json(rows)
    assert "NaN" not in text and "Infinity" not in text
    loaded = orjson.loads(text)
    assert loaded[0]["market_data"]["price"] is None
    assert loaded[0]["market_data"]["market_cap"] == 2684364
    assert loaded[0]["market_data"]["x"] == [None]


def _no_trial_fetchers():
    return dict(
        market_fetcher=lambda ticker, as_of_date: {
            "price": 12.0,
            "market_cap": 600_000_000,
            "company_name": "Royalty Co plc",
            "industry": "Biotechnology",
        },
        financial_fetcher=lambda row, ticker, as_of_date: {"ticker": ticker, "cash": 1},
        trial_fetcher=lambda ticker: [],
        fallback_trial_fetcher=lambda row, ticker: [],
        sleep_seconds=0,
    )


def test_fetch_pending_data_honours_coverage_exemptions():
    """An operator-exempted name must not be demoted to pending_coverage by the fetch step."""
    refreshed, _, report = fetch_pending_data(
        [_pending_row("ROY")],
        [],
        "2026-11-17",
        coverage_exemptions={"ROY": ["clinical_trials", "scientific_cartography"]},
        **_no_trial_fetchers(),
    )
    assert refreshed[0]["status"] == "active"
    assert refreshed[0]["coverage_status"]["clinical_trials"] == "exempt"
    assert refreshed[0]["coverage_status"]["scientific_cartography"] == "exempt"


def test_fetch_pending_data_loads_exemptions_file_by_default(tmp_path, monkeypatch):
    import json

    import tools.refresh_eligible_biotech_universe as refresh_mod

    path = tmp_path / "exemptions.json"
    path.write_text(
        json.dumps({"exemptions": [{"ticker": "ROY", "fields": ["clinical_trials", "scientific_cartography"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(refresh_mod, "DEFAULT_EXEMPTIONS_PATH", path)
    refreshed, _, _ = fetch_pending_data([_pending_row("ROY")], [], "2026-11-17", **_no_trial_fetchers())
    assert refreshed[0]["status"] == "active"
