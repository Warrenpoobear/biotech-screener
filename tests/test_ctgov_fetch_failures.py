#!/usr/bin/env python3
"""A failed CT.gov request must never look like "this sponsor has no trials".

2026-11-12: two full collections in a row aborted on the coverage-regression
guard because different tickers came back with 0 trials each run (STTK, then
ACHV/AVTX/IMMX/MNPR). _fetch_trials_page returned ([], None, False) for any
non-200/non-429 response, and _fetch_all_trials treated that empty page as the
end of the results. So a transient server error was indistinguishable from a
company with no trials, and a mapped ticker could be silently truncated.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import collect_ctgov_data as cc  # noqa: E402


def _resp(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    return r


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("collect_ctgov_data.time.sleep"):
        yield


class TestPageRetries:
    def test_server_error_is_retried(self):
        ok = _resp(200, {"studies": [{"x": 1}], "nextPageToken": None})
        with patch("collect_ctgov_data.requests.get", side_effect=[_resp(503), ok]) as g:
            studies, nxt, success = cc._fetch_trials_page("u", {})
        assert success is True
        assert studies == [{"x": 1}]
        assert g.call_count == 2

    def test_forbidden_is_retried(self):
        ok = _resp(200, {"studies": [], "nextPageToken": None})
        with patch("collect_ctgov_data.requests.get", side_effect=[_resp(403), _resp(403), ok]) as g:
            _, _, success = cc._fetch_trials_page("u", {})
        assert success is True
        assert g.call_count == 3

    def test_exhausted_retries_report_failure(self):
        with patch("collect_ctgov_data.requests.get", return_value=_resp(500)):
            studies, nxt, success = cc._fetch_trials_page("u", {})
        assert success is False and studies == []


class TestFetchAllRaisesOnFailure:
    def test_failed_first_page_raises(self):
        with patch("collect_ctgov_data._fetch_trials_page", return_value=([], None, False)):
            with pytest.raises(cc.CtgovFetchError):
                cc._fetch_all_trials({"query.term": "ACHV"}, "ACHV")

    def test_failed_later_page_raises_instead_of_truncating(self):
        study = {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}
        pages = [([study], "1", True), ([], None, False)]
        with patch("collect_ctgov_data._fetch_trials_page", side_effect=pages):
            with pytest.raises(cc.CtgovFetchError):
                cc._fetch_all_trials({"query.spons": "X"}, "TEST")

    def test_genuine_zero_results_is_not_an_error(self):
        with patch("collect_ctgov_data._fetch_trials_page", return_value=([], None, True)):
            assert cc._fetch_all_trials({"query.spons": "X"}, "TEST") == []


class TestCollectorHandlesFailedTickers:
    @staticmethod
    def _setup(tmp_path, previous):
        uni = tmp_path / "universe.json"
        uni.write_text(json.dumps([{"ticker": "AAA"}, {"ticker": "BBB"}]), encoding="utf-8")
        out = tmp_path / "trial_records.json"
        out.write_text(json.dumps(previous), encoding="utf-8")
        return uni, out

    def test_failed_ticker_is_retried_once(self, tmp_path):
        uni, out = self._setup(tmp_path, [{"ticker": "AAA", "nct_id": "NCT_OLD"}])
        calls = {"AAA": 0}

        def fake(ticker, *a, **k):
            if ticker == "AAA":
                calls["AAA"] += 1
                if calls["AAA"] == 1:
                    raise cc.CtgovFetchError("503")
                return [{"ticker": "AAA", "nct_id": "NCT_NEW"}]
            return [{"ticker": "BBB", "nct_id": "NCT_B"}]

        with patch("collect_ctgov_data.get_trials_for_ticker", side_effect=fake):
            cc.collect_all_trials(uni, out)
        got = json.loads(out.read_text(encoding="utf-8"))
        assert calls["AAA"] == 2
        assert {r["nct_id"] for r in got} == {"NCT_NEW", "NCT_B"}

    def test_still_failing_ticker_carries_forward_previous_records(self, tmp_path, capsys):
        prev = [{"ticker": "AAA", "nct_id": "NCT_OLD1"}, {"ticker": "AAA", "nct_id": "NCT_OLD2"}]
        uni, out = self._setup(tmp_path, prev)

        def fake(ticker, *a, **k):
            if ticker == "AAA":
                raise cc.CtgovFetchError("500")
            return [{"ticker": "BBB", "nct_id": "NCT_B"}]

        with patch("collect_ctgov_data.get_trials_for_ticker", side_effect=fake):
            cc.collect_all_trials(uni, out)  # must not SystemExit on the coverage guard
        got = json.loads(out.read_text(encoding="utf-8"))
        assert {r["nct_id"] for r in got} == {"NCT_OLD1", "NCT_OLD2", "NCT_B"}
        assert "FETCH FAILED" in capsys.readouterr().out

    def test_real_zero_still_trips_the_coverage_guard(self, tmp_path):
        uni, out = self._setup(tmp_path, [{"ticker": "AAA", "nct_id": "NCT_OLD"}])

        def fake(ticker, *a, **k):
            return [] if ticker == "AAA" else [{"ticker": "BBB", "nct_id": "NCT_B"}]

        with patch("collect_ctgov_data.get_trials_for_ticker", side_effect=fake):
            with pytest.raises(SystemExit):
                cc.collect_all_trials(uni, out)
