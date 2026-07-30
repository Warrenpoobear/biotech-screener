#!/usr/bin/env python3
"""
Tests for CT.gov collector coverage and determinism guards.

Three defects found by the 2026-07-30 re-warm, all of which would have
silently corrupted production_data/trial_records.json:

1. Sponsor-mapping gaps — 16 in-universe tickers holding 1,003 trials returned
   zero because they were absent from TICKER_TO_SPONSORS (or, for GLPG, because
   the registered sponsor was renamed to Lakefront Biotherapeutics NV).
2. max_results truncation was order-dependent, so the 10 tickers sitting at the
   cap got a different arbitrary subset every run. That violates the
   "identical inputs -> byte-identical outputs" rule and makes a golden replay
   baseline unreproducible for those names.
3. Nothing noticed. A ticker going from 134 trials to 0 wrote out clean.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_ctgov_data import TICKER_TO_SPONSORS, _fetch_all_trials, detect_coverage_regressions

# Tickers that silently lost every trial in the 2026-07-30 re-warm.
RESTORED_TICKERS = [
    "ADPT",
    "BRKR",
    "CDXS",
    "CPRX",
    "CSTL",
    "DFTX",
    "FLGT",
    "FTRE",
    "GLPG",
    "LGND",
    "MDXG",
    "MYGN",
    "PSNL",
    "QTRX",
    "RXRX",
    "VTRS",
]


class TestSponsorMappingCoverage:
    @pytest.mark.parametrize("ticker", RESTORED_TICKERS)
    def test_ticker_has_sponsor_mapping(self, ticker):
        """Without a mapping these fall back to query.term, which returns 0."""
        assert ticker in TICKER_TO_SPONSORS, f"{ticker} would fall back to term search"
        assert TICKER_TO_SPONSORS[ticker], f"{ticker} has an empty sponsor list"

    def test_glpg_uses_renamed_sponsor(self):
        """Galapagos NV is registered on CT.gov as Lakefront Biotherapeutics NV.

        The old string returns 0 studies, which is how 134 trials vanished.
        """
        sponsors = TICKER_TO_SPONSORS["GLPG"]
        assert any("Lakefront" in s for s in sponsors)

    def test_no_empty_sponsor_lists_anywhere(self):
        empty = [t for t, s in TICKER_TO_SPONSORS.items() if not s]
        assert not empty, f"tickers with empty sponsor lists: {empty}"


class TestDeterministicTruncation:
    """The cap must select the same trials regardless of API page ordering."""

    @staticmethod
    def _study(nct: str) -> dict:
        return {
            "protocolSection": {
                "identificationModule": {"nctId": nct, "briefTitle": "T"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE1"], "studyType": "INTERVENTIONAL"},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "S"}},
                "conditionsModule": {"conditions": []},
                "armsInterventionsModule": {"interventions": []},
            }
        }

    def _run(self, order):
        studies = [self._study(n) for n in order]
        with patch("collect_ctgov_data._fetch_trials_page", return_value=(studies, None, True)):
            return _fetch_all_trials({"query.spons": "X"}, "TEST", max_results=3)

    def test_truncation_is_order_independent(self):
        forward = self._run(["NCT005", "NCT001", "NCT004", "NCT002", "NCT003"])
        reverse = self._run(["NCT003", "NCT002", "NCT004", "NCT001", "NCT005"])
        assert [t["nct_id"] for t in forward] == [t["nct_id"] for t in reverse]

    def test_truncation_keeps_lowest_nct_ids(self):
        got = self._run(["NCT005", "NCT001", "NCT004", "NCT002", "NCT003"])
        assert [t["nct_id"] for t in got] == ["NCT001", "NCT002", "NCT003"]

    def test_under_cap_is_still_sorted(self):
        got = self._run(["NCT003", "NCT001", "NCT002"])
        assert [t["nct_id"] for t in got] == ["NCT001", "NCT002", "NCT003"]


class TestCoverageRegressionGuard:
    @staticmethod
    def _recs(pairs):
        return [{"ticker": t, "nct_id": f"NCT{i}"} for t, n in pairs for i in range(n)]

    def test_detects_ticker_dropping_to_zero(self):
        prev = self._recs([("GLPG", 134), ("ABVX", 23)])
        curr = self._recs([("ABVX", 23)])
        assert detect_coverage_regressions(prev, curr) == {"GLPG": 134}

    def test_no_regression_when_counts_hold(self):
        prev = self._recs([("ABVX", 23)])
        curr = self._recs([("ABVX", 23)])
        assert detect_coverage_regressions(prev, curr) == {}

    def test_partial_drop_is_not_a_regression(self):
        """Only total loss is the tripwire; normal churn must not block a refresh."""
        prev = self._recs([("ABVX", 23)])
        curr = self._recs([("ABVX", 20)])
        assert detect_coverage_regressions(prev, curr) == {}

    def test_new_ticker_is_not_a_regression(self):
        prev = self._recs([("ABVX", 23)])
        curr = self._recs([("ABVX", 23), ("AARD", 6)])
        assert detect_coverage_regressions(prev, curr) == {}

    def test_reports_every_dropped_ticker(self):
        prev = self._recs([("GLPG", 134), ("VTRS", 699), ("ABVX", 23)])
        curr = self._recs([("ABVX", 23)])
        assert detect_coverage_regressions(prev, curr) == {"GLPG": 134, "VTRS": 699}

    def test_empty_previous_is_not_a_regression(self):
        assert detect_coverage_regressions([], self._recs([("ABVX", 23)])) == {}
