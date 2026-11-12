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
    # 2026-11-12: term-search fallback returned 0 for Shattuck Labs (7 trials,
    # lead sponsor "Shattuck Labs, Inc."); the coverage guard blocked the refresh.
    "STTK",
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

    def test_truncation_keeps_newest_then_highest_nct_ids(self):
        """No first_posted dates here, so the NCT-ID tiebreak decides: highest
        three kept, returned in nct_id order."""
        got = self._run(["NCT005", "NCT001", "NCT004", "NCT002", "NCT003"])
        assert [t["nct_id"] for t in got] == ["NCT003", "NCT004", "NCT005"]

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


class TestPaginatedCapIsDeterministic:
    """Models the real failure: >cap trials across pages, unstable order.

    The old loop stopped fetching at the cap, so WHICH trials came back
    depended on CT.gov's page order (~5.6k trial IDs rotated per collection).
    Now the request asks for a server-side date sort, fetches one tie-break
    page past the cap, and keeps the newest first-posted trials.
    """

    @staticmethod
    def _study(nct: str, posted: str) -> dict:
        st = TestDeterministicTruncation._study(nct)
        st["protocolSection"]["statusModule"]["studyFirstPostDateStruct"] = {"date": posted}
        return st

    def _run(self, seed, captured=None):
        import random

        # 300 trials, server-sorted by first-posted desc; ties within a date
        # come back in a different order on every run. 30 per date, so the
        # 4th date straddles the 100-trial cap / page boundary.
        dates = [f"2026-{m:02d}-01" for m in range(10, 0, -1)]  # 10 dates x 30 trials
        by_date = [[(f"NCT{d[5:7]}{i:03d}", d) for i in range(30)] for d in dates]
        rng = random.Random(seed)
        ordered = []
        for block in by_date:
            block = block[:]
            rng.shuffle(block)
            ordered.extend(block)
        pages = [ordered[i : i + 100] for i in range(0, len(ordered), 100)]

        def fake_page(base_url, params, max_retries=3):
            if captured is not None:
                captured.append(dict(params))
            idx = int(params.get("pageToken", 0))
            studies = [self._study(n, d) for n, d in pages[idx]]
            nxt = str(idx + 1) if idx + 1 < len(pages) else None
            return studies, nxt, True

        with patch("collect_ctgov_data._fetch_trials_page", side_effect=fake_page):
            return _fetch_all_trials({"query.spons": "X"}, "TEST", max_results=100)

    def test_same_subset_regardless_of_tie_order(self):
        a = [t["nct_id"] for t in self._run(seed=1)]
        b = [t["nct_id"] for t in self._run(seed=2)]
        assert a == b
        assert len(a) == 100

    def test_keeps_newest_first_posted(self):
        got = self._run(seed=3)
        posted = sorted({t["first_posted"] for t in got}, reverse=True)
        # 100 newest = the 3 newest dates (90) + 10 of the 4th
        assert posted == ["2026-10-01", "2026-09-01", "2026-08-01", "2026-07-01"]

    def test_requests_server_side_sort(self):
        captured = []
        self._run(seed=4, captured=captured)
        assert captured and all(p.get("sort") == "StudyFirstPostDate:desc" for p in captured)
